#!/usr/bin/env python3
"""
Convert the Chaos "Isometric Bathroom" V-Ray sample scene to a Strelka scene.

    /Applications/Blender.app/Contents/MacOS/Blender -b \
        ~/Isometric_Bathroom_Scene/Iso_Bathroom.blend --factory-startup \
        -P tools/iso_bathroom/vray2strelka.py -- --out scenes/iso_bathroom

Why this exists rather than `scripts/blend2strelka.py`:

The .blend was authored with the V-Ray for Blender addon, which is not installed
(and does not run on Apple silicon).  Without it every V-Ray node loads as
`NodeUndefined`, so the *sockets* are dead but the plugin parameters survive as
IDProperties on the node (`node['BRDFVRayMtl']`, `node['BRDFSSS2Complex']`, ...).
This script reads those IDProperties and rebuilds each material as a native
Principled BSDF, which is the only thing the glTF exporter knows how to write.

Anything glTF cannot carry (sheen, subsurface, iridescence, the orthographic
camera's Strelka-side hints) is injected into the .gltf JSON afterwards as an
extension, so the file stays a description of the scene rather than of the
exporter's limits.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore
import numpy as np  # type: ignore
from mathutils import Matrix, Vector  # type: ignore


# ---------------------------------------------------------------------------
# V-Ray plugin defaults.
#
# IDProperties only store parameters that differ from the plugin default, so a
# missing key is not "unset" -- it is the default, and reading it as 0 silently
# turns every unwritten reflection black.  These are the BRDFVRayMtl /
# BRDFSSS2Complex defaults from the V-Ray plugin metadata.
# ---------------------------------------------------------------------------

VRAYMTL_DEFAULTS = {
    "diffuse": (0.5, 0.5, 0.5),
    "reflect": (1.0, 1.0, 1.0),
    "reflect_glossiness": 1.0,
    "fresnel": 1,
    "fresnel_ior": 1.6,
    "fresnel_ior_lock": 1,
    "metalness": 0.0,
    "refract": (0.0, 0.0, 0.0),
    "refract_ior": 1.6,
    "refract_glossiness": 1.0,
    "refract_thin_walled": 0,
    "fog_color_colortex": (1.0, 1.0, 1.0),
    "fog_mult": 1.0,
    "sheen_color": (0.0, 0.0, 0.0),
    "sheen_glossiness": 0.8,
    "coat_color": (1.0, 1.0, 1.0),
    "coat_amount": 0.0,
    "coat_glossiness": 1.0,
    "coat_ior": 1.6,
    "anisotropy": 0.0,
    "opacity_color": (1.0, 1.0, 1.0),
    "self_illumination": (0.0, 0.0, 0.0),
    "thin_film_on": 0,
    "thin_film_thickness": 0.1,
    "thin_film_thickness_min": 100.0,
    "thin_film_thickness_max": 1000.0,
    "thin_film_ior": 1.47,
    "bump_amount": 1.0,
    "bump_type": 0,
    "translucency": 0,
    "translucency_color": (1.0, 1.0, 1.0),
}

SSS2_DEFAULTS = {
    "diffuse_color": (0.5, 0.5, 0.5),
    "diffuse_amount": 1.0,
    "overall_color": (1.0, 1.0, 1.0),
    "sub_surface_color": (0.5, 0.5, 0.5),
    "scatter_radius": (0.92, 0.52, 0.175),
    "scatter_radius_mult": 1.0,
    "scale": 1.0,
    "phase_function": 0.0,
    "ior": 1.5,
    "specular_amount": 1.0,
    "specular_glossiness": 0.6,
    "specular_color": (1.0, 1.0, 1.0),
}

# Materials whose diffuse is black and whose Fresnel IOR was pushed this high are
# V-Ray's idiom for a conductor -- the scene sets no `metalness` anywhere.
METAL_IOR_THRESHOLD = 5.0

SHADER_PLUGINS = ("BRDFVRayMtl", "BRDFSSS2Complex", "BRDFLight")


def lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def safe_name(name):
    """Filename-safe material name.

    Strelka's glTF loader hands `image.uri` straight to the file opener without
    percent-decoding it, so a texture written for "Material #449" arrives as
    `Material%20%23449_ramp.png` and fails to load.  Keep the bytes ASCII and
    the problem never arises.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def diffuse_to_single_scattering_albedo(a):
    """Diffuse albedo -> single-scattering albedo.

    A DCC picks a subsurface colour the way it picks a diffuse one: 0.5 means
    "reflects about half". The random walk wants the probability that one
    extinction event scatters rather than absorbs, and the two are far apart --
    a diffuse 0.5 is a single-scattering 0.91, because the light that comes back
    out has scattered several times and each one paid the albedo again.

    Feeding the diffuse value straight in is what turned the rubber duck into a
    dark blob: 0.47 cubed is 0.1.

    Inverts Van de Hulst's approximation for a semi-infinite isotropically
    scattering medium,

        A(alpha) = (1 - s)(1 - 0.139 s) / (1 + 1.17 s),   s = sqrt(1 - alpha)

    by bisection. Semi-infinite and isotropic are both approximations here --
    these objects are neither -- but they are the ones every renderer that offers
    an albedo-coloured SSS makes.
    """

    def diffuse_albedo(alpha):
        s = math.sqrt(max(1.0 - alpha, 0.0))
        return (1.0 - s) * (1.0 - 0.139 * s) / (1.0 + 1.17 * s)

    a = min(max(float(a), 0.0), 0.999)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if diffuse_albedo(mid) < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gloss_to_rough(g):
    """V-Ray authors glossiness; Principled and glTF want roughness."""
    return max(0.0, min(1.0, 1.0 - float(g)))


def idprops_to_dict(group):
    """Deep-copy an IDPropertyGroup into plain Python.

    Everything must be copied out *before* the node tree is cleared: removing a
    node frees its IDProperties, and reads after that silently return defaults.
    That is not a hypothetical -- it is what made the first version of this
    script export white glass and no sheen at all.
    """
    out = {}
    try:
        keys = list(group.keys())
    except Exception:
        return out
    for k in keys:
        v = group[k]
        if hasattr(v, "keys"):
            out[k] = idprops_to_dict(v)
        elif hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
            # A V-Ray list is usually a colour or a vector, and sometimes a list
            # of groups: an include/exclude list holds one per object it names.
            # Recursing into those rather than forcing them through float() is
            # what lets a scene with a light that excludes something be read at
            # all -- the kids' bedroom has one, and it stopped the conversion
            # dead at the first light.
            out[k] = [idprops_to_dict(x) if hasattr(x, "keys") else x for x in v]
            if all(isinstance(x, (int, float)) for x in out[k]):
                out[k] = [float(x) for x in out[k]]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Texture handling
# ---------------------------------------------------------------------------


class TextureBaker:
    """
    Loads V-Ray Bitmap files and bakes the node effects glTF cannot express.

    V-Ray Color Correction and glossiness-to-roughness inversion are node-graph
    operations, and the glTF exporter only walks a small set of nodes before it
    gives up and writes the socket's flat default instead.  Baking them into an
    image keeps the look and keeps the file readable by any glTF importer.
    """

    def __init__(self, assets_dir, out_dir):
        self.assets_dir = Path(assets_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.baked = []
        self.missing = []

    def resolve(self, vray_path):
        """`//Assets\\Wood_Diff.png` -> a real file, matched by basename."""
        if not vray_path:
            return None
        base = str(vray_path).replace("\\", "/").split("/")[-1]
        cand = self.assets_dir / base
        if cand.is_file():
            return cand
        stem = Path(base).stem.lower()
        for f in sorted(self.assets_dir.iterdir()):
            if f.is_file() and f.stem.lower() == stem:
                return f
        self.missing.append(str(vray_path))
        return None

    def load(self, path, colorspace="sRGB"):
        key = (str(path), colorspace)
        if key in self.cache:
            return self.cache[key]
        img = bpy.data.images.load(str(path), check_existing=False)
        try:
            img.colorspace_settings.name = colorspace
        except Exception:
            pass
        self.cache[key] = img
        return img

    def _pixels(self, path, colorspace=None):
        img = bpy.data.images.load(str(path), check_existing=True)
        if colorspace is not None:
            # Before reading: `pixels` is decoded through whatever the image is
            # tagged with, and a mask tagged sRGB comes back curved.
            try:
                img.colorspace_settings.name = colorspace
            except Exception:
                pass
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        return buf.reshape(h, w, 4), w, h

    def _write(self, name, arr, w, h, colorspace):
        out = self.out_dir / name
        img = bpy.data.images.new(name, width=w, height=h, alpha=False, float_buffer=False)
        img.pixels.foreach_set(arr.reshape(-1))
        img.filepath_raw = str(out)
        img.file_format = "PNG"
        img.save()
        try:
            img.colorspace_settings.name = colorspace
        except Exception:
            pass
        self.baked.append(out.name)
        return img

    def mean_color(self, path):
        """Average linear colour of an image.

        The subsurface extension carries a single scatter albedo and no texture
        slot, so a textured translucent material (the rubber duck, the marble
        worktop) would otherwise walk with a flat grey albedo and come out grey.
        Tinting by the texture mean keeps the object the colour it is; the
        variation across the surface is what is lost.
        """
        px, _, _ = self._pixels(path)
        return [float(px[..., c].mean()) for c in range(3)]

    def bake_roughness(self, gloss_path, offset, name):
        """Glossiness map + V-Ray colour offset -> a linear roughness map."""
        px, w, h = self._pixels(gloss_path)
        g = np.clip(px[..., :3] + float(offset), 0.0, 1.0)
        out = np.ones((h, w, 4), dtype=np.float32)
        out[..., :3] = 1.0 - g
        return self._write(name, out, w, h, "Non-Color")

    def bake_color_correction(self, src_path, cc, name):
        """
        Apply a V-Ray ColorCorrection to an sRGB texture.

        Only the parameters this scene actually sets are implemented:
        brightness / contrast, the "advanced" lightness curve
        (adv_contrast / adv_brightness / adv_offset) and the hue tint.
        """
        px, w, h = self._pixels(src_path)
        c = px[..., :3].copy()

        if int(cc.get("lightness_mode", 0)) == 1:
            contrast = float(cc.get("adv_contrast", 1.0))
            bright = float(cc.get("adv_brightness", 1.0))
            offset = float(cc.get("adv_offset", 0.0))
            c = np.clip((c - 0.5) * contrast + 0.5, 0.0, 4.0)
            c = c * bright + offset
        else:
            b = float(cc.get("brightness", 0.0))
            k = float(cc.get("contrast", 1.0))
            c = np.clip((c - 0.5) * k + 0.5 + b, 0.0, 4.0)

        tint = cc.get("hue_tint")
        strength = float(cc.get("tint_strength", 0.0))
        if tint is not None and strength > 0.0:
            t = np.array([float(x) for x in tint], dtype=np.float32)
            t = t / max(float(t.max()), 1e-4)
            c = c * (1.0 - strength) + c * t * strength

        c = np.clip(c, 0.0, 1.0)
        out = np.ones((h, w, 4), dtype=np.float32)
        out[..., :3] = c
        return self._write(name, out, w, h, "sRGB")

    def bake_mix(self, mask_path, color1, color2, name):
        """V-Ray TexMix -> a texture, with the two colours folded in.

        `mix(color1, color2, mask)`: colour1 where the mask is black, colour2
        where it is white. That is the blue blanket with green stars and the
        green one with blue stars -- the same star mask, the two colours the
        other way round -- and the plaster on the walls.

        The mask is read as data and not as a picture. It is a blend factor, and
        decoding it through the sRGB curve would bend every value between the two
        colours while leaving the two colours themselves right, which is the kind
        of error that looks like a texture that is merely a bit off.
        """
        px, w, h = self._pixels(mask_path, colorspace="Non-Color")
        m = px[..., :3].mean(axis=2, keepdims=True)
        a = np.array(color1, dtype=np.float32).reshape(1, 1, 3)
        b = np.array(color2, dtype=np.float32).reshape(1, 1, 3)
        out = np.ones((h, w, 4), dtype=np.float32)
        out[..., :3] = np.clip(a + (b - a) * m, 0.0, 1.0)
        return self._write(name, out, w, h, "sRGB")

    def bake_gradient_ramp(self, name, color_a, color_b, size=512):
        """
        V-Ray Gradient Ramp with no stored stops -> the plugin default, a black
        to white V-ramp.  It drives the Sansevieria's diffuse amount through a
        50x tiled, 90-degree-rotated UV, which is what makes the leaf stripes.

        The two colours it mixes between are baked in here rather than left to a
        Mix node, because glTF has exactly one base-colour slot: the exporter
        resolves a node graph it cannot represent by taking the upstream texture
        and dropping everything downstream of it. That put a black-to-white ramp
        into the leaf's base colour and threw the plant's green away -- the
        Sansevieria rendered as pale grey stripes instead of banded green, which
        is the whole of what the ramp exists to draw.

        Written as linear, like every other bake here: _write hands the values to
        Blender's PNG writer, which encodes them.
        """
        v = np.linspace(0.0, 1.0, size, dtype=np.float32).reshape(-1, 1)
        ramp = np.tile(v, (1, size))[..., None]
        a = np.array(color_a, dtype=np.float32).reshape(1, 1, 3)
        b = np.array(color_b, dtype=np.float32).reshape(1, 1, 3)
        out = np.ones((size, size, 4), dtype=np.float32)
        out[..., :3] = a + (b - a) * ramp
        return self._write(name, out, size, size, "sRGB")

    def bake_noise_normal(self, name, size=256, amount=1.0):
        """
        V-Ray TexNoiseMax driving a bump socket.  Baked to a tiling normal map
        because glTF has no procedural textures; the water surfaces are the only
        users and they want a very small perturbation.
        """
        rng = np.random.default_rng(7)
        n = rng.random((size, size)).astype(np.float32)
        # Cheap value-noise smoothing: three box blurs approximate a gaussian.
        for _ in range(3):
            n = (n + np.roll(n, 1, 0) + np.roll(n, -1, 0) + np.roll(n, 1, 1) + np.roll(n, -1, 1)) / 5.0
        gx = (np.roll(n, -1, 1) - np.roll(n, 1, 1)) * amount
        gy = (np.roll(n, -1, 0) - np.roll(n, 1, 0)) * amount
        nz = np.ones_like(n)
        ln = np.sqrt(gx * gx + gy * gy + nz * nz)
        out = np.ones((size, size, 4), dtype=np.float32)
        out[..., 0] = (-gx / ln) * 0.5 + 0.5
        out[..., 1] = (-gy / ln) * 0.5 + 0.5
        out[..., 2] = (nz / ln) * 0.5 + 0.5
        return self._write(name, out, size, size, "Non-Color")


# ---------------------------------------------------------------------------
# Material conversion
# ---------------------------------------------------------------------------


class MaterialConverter:
    def __init__(self, baker, opts):
        self.baker = baker
        self.opts = opts
        self.report = []
        # V-Ray Light Mtl name -> its colour multiplier, so the quads carrying it
        # can be lifted out into analytic lights.
        self.light_materials = {}

    # -- snapshot ---------------------------------------------------------

    def _snap_node(self, node):
        d = {"name": node.name, "plugins": {}, "inputs": {}}
        for k in node.keys():
            if k in ("unique_id", "object_ptr"):
                continue
            v = node[k]
            d["plugins"][k] = idprops_to_dict(v) if hasattr(v, "keys") else v
        for s in node.inputs:
            if s.is_linked:
                d["inputs"][s.name] = self._snap_node(s.links[0].from_node)
        return d

    def snapshot(self, mat):
        """Copy the whole V-Ray graph to plain Python before touching the tree."""
        if mat.node_tree is None:
            return None
        shader = kind = None
        for n in mat.node_tree.nodes:
            for k in SHADER_PLUGINS:
                if k in n.keys():
                    shader, kind = n, k
        if shader is None:
            return None
        snap = {
            "kind": kind,
            "params": idprops_to_dict(shader[kind]),
            "inputs": {},
            "uvw": None,
        }
        for s in shader.inputs:
            if s.is_linked:
                snap["inputs"][s.name] = self._snap_node(s.links[0].from_node)
        for n in mat.node_tree.nodes:
            if "UVWGenMayaPlace2dTexture" in n.keys():
                snap["uvw"] = idprops_to_dict(n["UVWGenMayaPlace2dTexture"])
        return snap

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _get(params, key, defaults):
        if key in params:
            v = params[key]
            return tuple(v) if isinstance(v, list) else v
        d = defaults.get(key)
        return d

    @staticmethod
    def _uvw_of(snap):
        u = snap.get("uvw")
        if not u:
            return None
        return {
            "repeat_u": float(u.get("repeat_u", 1.0)),
            "repeat_v": float(u.get("repeat_v", 1.0)),
            "offset_u": float(u.get("offset_u", 0.0)),
            "offset_v": float(u.get("offset_v", 0.0)),
            "rotate_uv": float(u.get("rotate_uv", 0.0)),
        }

    def _bitmap_of(self, snapnode):
        """The image behind a V-Ray Bitmap node, however this file stores it.

        Two conventions, and one scene each. The bathroom puts the path on the
        plugin as `BitmapBuffer.file`. The kids' bedroom leaves that key off
        entirely and hangs the image on a Blender texture datablock named by the
        node's `texture_name` -- a `.VRayFakeTexture@...` whose `image.filepath`
        is the real thing. Reading only the first convention brought one image
        out of forty-one across that scene, so nearly every texture in it was
        silently a flat colour.
        """
        buf = snapnode["plugins"].get("BitmapBuffer") if snapnode else None
        if not isinstance(buf, dict):
            return None, {}
        raw = buf.get("file")
        if not raw:
            name = snapnode["plugins"].get("texture_name")
            tex = bpy.data.textures.get(name) if name else None
            img = getattr(tex, "image", None)
            raw = getattr(img, "filepath", None)
        return self.baker.resolve(raw), buf

    def _new_tree(self, mat):
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.location = (600, 0)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (200, 0)
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return nt, bsdf

    @staticmethod
    def _set(bsdf, name, value):
        """Principled socket names moved between Blender versions; skip misses."""
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = value
            return True
        return False

    def _tex_node(self, nt, image, x, y, uvw=None):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.location = (x, y)
        tex.extension = "REPEAT"
        if uvw and (uvw["repeat_u"] != 1.0 or uvw["repeat_v"] != 1.0
                    or uvw["offset_u"] or uvw["offset_v"] or uvw["rotate_uv"]):
            mapping = nt.nodes.new("ShaderNodeMapping")
            mapping.location = (x - 250, y)
            mapping.inputs["Scale"].default_value = (uvw["repeat_u"], uvw["repeat_v"], 1.0)
            mapping.inputs["Location"].default_value = (uvw["offset_u"], uvw["offset_v"], 0.0)
            mapping.inputs["Rotation"].default_value = (0.0, 0.0, uvw["rotate_uv"])
            uvmap = nt.nodes.new("ShaderNodeUVMap")
            uvmap.location = (x - 450, y)
            nt.links.new(uvmap.outputs["UV"], mapping.inputs["Vector"])
            nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        return tex

    def _mix_image(self, mat, snapnode, approx):
        """A TexMix, or a MultiSubTex that lost its textures on the way here."""
        if snapnode is None:
            return None
        plugins = snapnode["plugins"]

        if "TexMulti" in plugins:
            # V-Ray picks one of N textures by object ID. The list of N does not
            # survive into the .blend -- all five slots are empty and nothing is
            # linked -- so there is no colour here to carry, only the fact that
            # one is missing. The pencils it drives come out at the plugin's
            # default grey, and inventing colours from their object names would
            # be a guess in the shape of a conversion.
            approx.append("MultiSubTex carries no textures in the .blend; "
                          "the material keeps its flat diffuse")
            return None

        mix = plugins.get("TexMix")
        if not isinstance(mix, dict):
            return None
        mask_node = snapnode["inputs"].get("Mix Map")
        path, _ = self._bitmap_of(mask_node)
        if path is None:
            approx.append("TexMix with no mask texture -> flat diffuse")
            return None
        c1 = mix.get("color1", [0.0, 0.0, 0.0])
        c2 = mix.get("color2", [1.0, 1.0, 1.0])
        approx.append("TexMix baked to a texture")
        return self.baker.bake_mix(path, c1, c2, f"{safe_name(mat.name)}_mix.png")

    def _image_for(self, mat, snapnode, srgb=True):
        """Follow a Bitmap / ColorCorrection chain and return a Blender image."""
        if snapnode is None:
            return None
        plugins = snapnode["plugins"]
        if "BitmapBuffer" in plugins:
            path, _ = self._bitmap_of(snapnode)
            return self.baker.load(path, "sRGB" if srgb else "Non-Color") if path else None
        if "ColorCorrection" in plugins:
            up = next(iter(snapnode["inputs"].values()), None)
            path, _ = self._bitmap_of(up)
            if path is None:
                return None
            return self.baker.bake_color_correction(
                path, plugins["ColorCorrection"], f"{safe_name(mat.name)}_cc.png")
        return None

    # -- BRDFVRayMtl ------------------------------------------------------

    def convert_vraymtl(self, mat, snap):
        p = snap["params"]
        d = VRAYMTL_DEFAULTS
        g = lambda k: self._get(p, k, d)

        diffuse = g("diffuse")
        reflect = g("reflect")
        refract = g("refract")
        fresnel_on = int(g("fresnel"))
        fresnel_ior = float(g("fresnel_ior"))
        gloss = float(g("reflect_glossiness"))
        extra = {}

        nt, bsdf = self._new_tree(mat)
        uvw = self._uvw_of(snap)

        # -- conductor detection.  The scene never sets `metalness`; metals are
        # authored as black diffuse + coloured reflection + a Fresnel IOR pushed
        # far past any dielectric (16), or with Fresnel switched off entirely.
        transmissive = max(refract) > 0.005
        is_metal = (
            max(diffuse) < 0.02
            and not transmissive
            and (fresnel_ior >= METAL_IOR_THRESHOLD or fresnel_on == 0)
        )

        if is_metal:
            self._set(bsdf, "Base Color", (*reflect, 1.0))
            self._set(bsdf, "Metallic", 1.0)
            self._set(bsdf, "Roughness", gloss_to_rough(gloss))
            extra["metal"] = True
        else:
            self._set(bsdf, "Base Color", (*diffuse, 1.0))
            self._set(bsdf, "Metallic", 0.0)
            self._set(bsdf, "Roughness", gloss_to_rough(gloss))
            self._set(bsdf, "IOR", fresnel_ior)
            # V-Ray's reflection colour dims the specular lobe; glTF carries that
            # as KHR_materials_specular.  Blender writes specularFactor =
            # 2 x "Specular IOR Level", and glTF's neutral specularFactor is 1.0,
            # so a white V-Ray reflection has to land on 0.5 here -- setting 1.0
            # doubles F0 on every dielectric in the scene.
            self._set(bsdf, "Specular IOR Level", min(1.0, 0.5 * lum(reflect)))

        # -- transmission
        if transmissive:
            # glTF tints transmitted light by baseColor (standard_pbr.h:356 uses
            # si.albedo as the transmission throughput), while V-Ray keeps the
            # two separate and clear glass is authored as black diffuse + white
            # refraction.  Carrying that diffuse across renders every glass panel
            # solid black, so a pure refractor takes its base colour from the
            # refraction colour instead.
            if max(diffuse) < 0.02:
                peak = max(refract)
                self._set(bsdf, "Base Color", (*[c / peak for c in refract], 1.0))
                self._set(bsdf, "Transmission Weight", min(1.0, peak))
            else:
                self._set(bsdf, "Transmission Weight", min(1.0, lum(refract)))
            self._set(bsdf, "IOR", float(g("refract_ior")))
            rg = float(g("refract_glossiness"))
            if rg < 1.0:
                # One roughness drives both lobes in glTF.  Take the rougher.
                self._set(bsdf, "Roughness", max(gloss_to_rough(gloss), gloss_to_rough(rg)))
            fog = list(g("fog_color_colortex"))
            fog_mult = float(g("fog_mult"))
            thin = int(g("refract_thin_walled")) != 0
            # A coloured refraction that the fog colour does not already carry.
            if max(refract) - min(refract) > 0.02:
                fog = [a * b for a, b in zip(fog, refract)]
            if thin:
                extra["thinWalled"] = True
            else:
                # Stated, not implied. A loader cannot tell "solid glass" from
                # "thin-walled" by the absence of an extension, so every solid
                # refractor says so explicitly and the bubbles say the opposite.
                extra["volume"] = {"thicknessFactor": 1.0}
            if not thin and (max(fog) - min(fog) > 0.005 or fog_mult != 1.0):
                # Blender only emits KHR_materials_volume from a "glTF Material
                # Output" node group with a Thickness socket, which is more
                # fragile than writing the extension ourselves after the export.
                # V-Ray's fog depth is in centimetres and its multiplier scales
                # that depth -- a *larger* fog_mult is more transparent, not less.
                # glTF attenuates as attenuationColor^(d / attenuationDistance),
                # so the two agree at
                #
                #     attenuationDistance = fog_mult x (one centimetre)
                #
                # and with this scene at one Blender unit to the metre that is
                # fog_mult x 0.01. Nothing here is fitted: --fog-scale is one
                # centimetre expressed in scene units, and only changes if the
                # scene does.
                #
                # This had the multiplier on the wrong side, and the bath water
                # hid it: at fog_mult 10 the old `1.0 / fog_mult` and the correct
                # `0.01 * fog_mult` are both 0.1, so the one material anyone looks
                # at came out right while the shower glass, at fog_mult 1, was a
                # hundred times too weak and showed no green at all.
                extra["volume"] = {
                    "thicknessFactor": 1.0,
                    "attenuationColor": fog,
                    "attenuationDistance": self.opts.fog_scale * max(fog_mult, 1e-4),
                }

        # -- coat
        coat = float(g("coat_amount"))
        if coat > 0.0:
            self._set(bsdf, "Coat Weight", coat)
            self._set(bsdf, "Coat Roughness", gloss_to_rough(float(g("coat_glossiness"))))
            self._set(bsdf, "Coat IOR", float(g("coat_ior")))
            extra["coat"] = {"weight": coat, "ior": float(g("coat_ior"))}

        # -- sheen (glTF: KHR_materials_sheen, injected post-export)
        sheen = g("sheen_color")
        if max(sheen) > 0.001:
            sw = min(1.0, lum(sheen))
            sr = gloss_to_rough(float(g("sheen_glossiness")))
            self._set(bsdf, "Sheen Weight", sw)
            self._set(bsdf, "Sheen Roughness", sr)
            self._set(bsdf, "Sheen Tint", (*sheen, 1.0))
            extra["sheen"] = {
                "sheenColorFactor": [s * sw for s in sheen],
                "sheenRoughnessFactor": sr,
            }

        # -- thin film (glTF: KHR_materials_iridescence)
        if int(g("thin_film_on")):
            tmin = float(g("thin_film_thickness_min"))
            tmax = float(g("thin_film_thickness_max"))
            t = float(g("thin_film_thickness"))
            # V-Ray gives a thickness range and a blend between its ends;
            # KHR_materials_iridescence gives a range and expects a texture to
            # pick within it. With no texture the spec says to use the maximum,
            # so the blended film is written to both bounds -- a range with no
            # texture to sample it is just a way of saying the wrong thickness.
            thickness = tmin + (tmax - tmin) * t
            extra["iridescence"] = {
                "iridescenceFactor": 1.0,
                "iridescenceIor": float(g("thin_film_ior")),
                "iridescenceThicknessMinimum": thickness,
                "iridescenceThicknessMaximum": thickness,
            }
            self._set(bsdf, "Thin Film Thickness", tmin + (tmax - tmin) * t)
            self._set(bsdf, "Thin Film IOR", float(g("thin_film_ior")))

        self._wire_textures(mat, snap, nt, bsdf, uvw, extra)
        return extra

    def _wire_textures(self, mat, snap, nt, bsdf, uvw, extra):
        inputs = snap["inputs"]
        p = snap["params"]

        diffuse_in = inputs.get("Diffuse Color")
        img = self._image_for(mat, diffuse_in, srgb=True)
        if img is None:
            img = self._mix_image(mat, diffuse_in, extra.setdefault("approx", []))
        if img is not None:
            tex = self._tex_node(nt, img, -300, 300, uvw)
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

        gl = inputs.get("Reflection Glossiness")
        if gl is not None and "BitmapBuffer" in gl["plugins"]:
            path, meta = self._bitmap_of(gl)
            if path is not None:
                off = meta.get("color_offset", [0.0, 0.0, 0.0])
                off = float(off[0]) if isinstance(off, list) else float(off)
                img = self.baker.bake_roughness(path, off, f"{safe_name(mat.name)}_rough.png")
                tex = self._tex_node(nt, img, -300, 0, uvw)
                nt.links.new(tex.outputs["Color"], bsdf.inputs["Roughness"])

        bump = inputs.get("Bump Map")
        if bump is not None:
            bump_type = int(self._get(p, "bump_type", VRAYMTL_DEFAULTS))
            bump_amount = float(self._get(p, "bump_amount", VRAYMTL_DEFAULTS))
            img = None
            if "BitmapBuffer" in bump["plugins"]:
                path, _ = self._bitmap_of(bump)
                if path is not None:
                    img = self.baker.load(path, "Non-Color")
            elif "TexNoiseMax" in bump["plugins"]:
                img = self.baker.bake_noise_normal(
                    f"{safe_name(mat.name)}_noise_nrm.png", amount=self.opts.noise_bump_gain)
                extra.setdefault("approx", []).append("procedural noise bump baked to a normal map")
            if img is not None:
                tex = self._tex_node(nt, img, -300, -300, uvw)
                nm = nt.nodes.new("ShaderNodeNormalMap")
                nm.location = (-40, -300)
                # bump_type 1 = "Normal map (tangent)".  V-Ray's amount there is
                # a multiplier on data already in [-1,1], not a strength.
                nm.inputs["Strength"].default_value = (
                    1.0 if bump_type == 1 else min(2.0, bump_amount))
                nt.links.new(tex.outputs["Color"], nm.inputs["Color"])
                nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

    # -- BRDFSSS2Complex --------------------------------------------------

    def convert_sss(self, mat, snap):
        p = snap["params"]
        d = SSS2_DEFAULTS
        g = lambda k: self._get(p, k, d)

        overall = g("overall_color")
        diffuse = g("diffuse_color")
        # Overall colour is the general colour of the whole shader in Fast SSS2:
        # it multiplies the sub-surface colour as well as the diffuse one. Only
        # the diffuse got it here, which leaves the two ends of the material in
        # different hues. On the Sansevieria that is the whole plant: overall is
        # (0.32, 0.52, 0.09), so the diffuse end came out deep green and the
        # scattering end stayed the raw (0.44, 0.50, 0.25) grey-yellow, and the
        # gradient ramp between them drew the leaves pale instead of banded.
        subsurf = tuple(o * s for o, s in zip(overall, g("sub_surface_color")))
        radius = g("scatter_radius")
        mult = float(g("scatter_radius_mult"))
        scale = float(g("scale"))
        spec_amt = float(g("specular_amount"))
        spec_gloss = float(g("specular_glossiness"))

        base = tuple(o * dd for o, dd in zip(overall, diffuse))

        # V-Ray only stores what differs from the default, so an unset
        # sub-surface colour reads as the plugin's 0.5 grey -- which is a
        # perfectly plausible value and turns a yellow duck into a grey one.
        # Absent means "use the surface colour", not "use grey".
        scatter = tuple(subsurf) if "sub_surface_color" in p else base

        # The albedo the scatter colour is being derived from. The renderer scales
        # the walk's albedo by how far the surface it entered through departs from
        # this, which is how a textured translucent material carries its texture
        # inside -- the marble worktop's veining lives only on the boundary.
        #
        # Where a texture drives the diffuse colour, the reference is that
        # texture's mean, so the ratio is 1 on average and the variation is what
        # survives.
        reference = base
        diffuse_tex = self._bitmap_of(snap["inputs"].get("Diffuse Color"))[0] if snap["inputs"].get(
            "Diffuse Color") else None
        if diffuse_tex is not None:
            mean = self.baker.mean_color(diffuse_tex)
            scatter = tuple(c * m for c, m in zip(scatter, mean))
            reference = tuple(max(c * m, 1e-4) for c, m in zip(base, mean))
        # The renderer's scatterColor is the single-scattering albedo; V-Ray's
        # colour is a diffuse one. Converting here rather than in the shader keeps
        # the extension's meaning unambiguous and the fit in the tool that knows
        # which DCC it came from.
        scatter = tuple(diffuse_to_single_scattering_albedo(c) for c in scatter)
        # V-Ray authors the scatter radius in centimetres and this scene is
        # nowhere near metres -- the whole room is 0.5 units across -- so the raw
        # number means nothing here.  --sss-scale converts it, and getting it an
        # order of magnitude too large makes every translucent object *vanish*
        # rather than glow: a mean free path longer than the object is a medium
        # light passes straight through.
        radius_scene = [r * mult * scale * self.opts.sss_scale for r in radius]

        nt, bsdf = self._new_tree(mat)
        uvw = self._uvw_of(snap)
        inputs = snap["inputs"]

        self._set(bsdf, "Base Color", (*base, 1.0))
        self._set(bsdf, "Metallic", 0.0)
        self._set(bsdf, "Roughness", gloss_to_rough(spec_gloss))
        self._set(bsdf, "IOR", float(g("ior")))
        self._set(bsdf, "Specular IOR Level", min(1.0, 0.5 * spec_amt))
        self._set(bsdf, "Subsurface Weight", 1.0)
        self._set(bsdf, "Subsurface Radius", tuple(radius_scene))
        self._set(bsdf, "Subsurface Scale", 1.0)
        self._set(bsdf, "Subsurface Anisotropy", float(g("phase_function")))

        approx = []
        img = self._image_for(mat, inputs.get("Diffuse Color"), srgb=True)
        if img is not None:
            tex = self._tex_node(nt, img, -300, 300, uvw)
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

        fo = inputs.get("Sub-surface Color")
        if fo is not None and "TexFalloff" in fo["plugins"]:
            approx.append("TexFalloff on subsurface colour -> flat facing colour")

        if inputs.get("Diffuse Amount") is not None:
            approx.append("gradient-ramp diffuse mask baked into the base colour")
            ramp = self.baker.bake_gradient_ramp(f"{safe_name(mat.name)}_ramp.png",
                                                 subsurf, base)
            tex = self._tex_node(nt, ramp, -300, 100, uvw)
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

        return {
            "subsurface": {
                "subsurfaceFactor": 1.0,
                # The renderer reads this as the *single-scattering* albedo, not
                # the diffuse albedo V-Ray shows in its colour picker. The two are
                # related by an inversion that needs a fit; taking V-Ray's value
                # directly is the approximation, and it errs toward too little
                # multiple scattering rather than toward glowing.
                "scatterColor": list(scatter),
                "scatterReference": list(reference),
                "scatterRadius": radius_scene,
                "anisotropy": float(g("phase_function")),
                "ior": float(g("ior")),
            },
            # A renderer with no SSS should still not draw these as opaque matte;
            # diffuse transmission is the closest core-glTF fallback.
            "diffuseTransmission": {
                "diffuseTransmissionFactor": self.opts.sss_fallback_dt,
                "diffuseTransmissionColorFactor": list(scatter),
            },
            "approx": approx,
        }

    # -- BRDFLight --------------------------------------------------------

    def convert_light_mtl(self, mat, snap):
        p = snap["params"]
        mult = float(p.get("colorMultiplier", 1.0))
        color = tuple(p.get("color", [1.0, 1.0, 1.0]))
        strength = mult * self.opts.mesh_light_gain
        self.light_materials[mat.name] = mult
        nt, bsdf = self._new_tree(mat)
        self._set(bsdf, "Base Color", (0.0, 0.0, 0.0, 1.0))
        self._set(bsdf, "Emission Color", (*color, 1.0))
        self._set(bsdf, "Emission Strength", strength)
        return {"emissiveStrength": strength}

    # -- entry point ------------------------------------------------------

    def convert(self, mat):
        snap = self.snapshot(mat)
        if snap is None:
            return None  # already a native Blender material
        kind = snap["kind"]
        if kind == "BRDFVRayMtl":
            extra = self.convert_vraymtl(mat, snap)
        elif kind == "BRDFSSS2Complex":
            extra = self.convert_sss(mat, snap)
        else:
            extra = self.convert_light_mtl(mat, snap)
        self.report.append({"material": mat.name, "kind": kind, "extra": extra})
        return extra


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------


def light_to_sidecar(obj, vray, opts):
    """
    V-Ray light -> Strelka UniformLightDesc JSON.

    The coordinate conversion is the one from scripts/blend2strelka.py: a -90
    degree X rotation for Blender Z-up -> glTF Y-up, and no normal flip, because
    Blender lights and Strelka rect lights both emit along local -Z.
    """
    axis_conv = Matrix.Rotation(-math.pi / 2, 4, "X")
    conv = axis_conv @ obj.matrix_world
    loc = conv.translation
    euler = conv.to_3x3().normalized().to_euler("XYZ")

    lt = int(vray.get("light_type", 9))
    # 10 = LightDome, handled as the environment rather than as a light.
    if lt == 10:
        return None

    if lt == 6:  # LightSphere
        sph = vray.get("LightSphere", {})
        return {
            "name": obj.name,
            "type": "sphere",
            "unit": "radiance",
            "intensity": float(sph.get("intensity", 1.0)) * opts.rect_light_gain,
            "color": vray_light_color(sph),
            "position": [loc.x, loc.y, loc.z],
            "orientation": [math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z)],
            # A sphere emits the same in every direction, so the object's scale
            # is the only thing that can change its size, and only uniformly.
            "radius": float(sph.get("radius", 0.1)) * max(abs(v) for v in obj.matrix_world.to_scale()),
            "enabled": bool(sph.get("enabled", 1)),
            "visibleToCamera": not bool(sph.get("invisible", 0)),
        }

    if lt == 7:  # LightSpot
        spot = vray.get("LightSpot", {})
        # V-Ray gives the full cone and the width of the soft edge *inside* it;
        # glTF, and Strelka with it, give the two half-angles. So the outer half
        # angle is half the cone and the inner one is that less the penumbra --
        # not half the penumbra, which would put the falloff outside the cone.
        cone = float(spot.get("coneAngle", math.pi / 4.0))
        penumbra = max(0.0, float(spot.get("penumbraAngle", 0.0)))
        outer = 0.5 * cone
        inner = max(0.0, outer - 0.5 * penumbra)
        return {
            "name": obj.name,
            "type": "spot",
            # V-Ray's spot `units` 3 is radiant power in watts; anything else is
            # the same unitless multiplier the rect lights use. Only the units it
            # states are converted, because a guess here is a light that is wrong
            # by a factor nobody can see the size of.
            "unit": "power" if int(spot.get("units", 0)) == 3 else "radiance",
            "intensity": float(spot.get("intensity", 1.0)) * opts.rect_light_gain,
            "color": vray_light_color(spot),
            "position": [loc.x, loc.y, loc.z],
            "orientation": [math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z)],
            "innerConeAngle": math.degrees(inner),
            "outerConeAngle": math.degrees(outer),
            "radius": float(spot.get("shadowRadius", 0.0)),
            "enabled": bool(spot.get("enabled", 1)),
            "visibleToCamera": not bool(spot.get("invisible", 0)),
        }

    if lt != 9:  # 9 = LightRectangle
        return None

    rect = vray.get("LightRectangle", {})
    intensity = float(rect.get("intensity", 1.0))
    color = vray_light_color(rect)
    enabled = bool(rect.get("enabled", 1))
    # V-Ray's "invisible": the light still lights the scene and still shows up in
    # reflections, it just is not in frame.
    visible = not bool(rect.get("invisible", 0))

    lamp = obj.data
    width = lamp.size
    height = lamp.size_y if lamp.shape == "RECTANGLE" else lamp.size
    sx, sy, _ = obj.matrix_world.to_scale()
    width *= abs(sx)
    height *= abs(sy)

    return {
        "name": obj.name,
        "type": "rect",
        # V-Ray's default light units are a radiance multiplier, which is what
        # Strelka's "radiance" unit means -- no area or pi conversion.
        "unit": "radiance",
        "intensity": intensity * opts.rect_light_gain,
        "color": color,
        "position": [loc.x, loc.y, loc.z],
        "orientation": [math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z)],
        "width": width,
        "height": height,
        "enabled": enabled,
        "visibleToCamera": visible,
    }


def light_shape_summary(l):
    """The one dimension a light of this type is actually described by."""
    t = l.get("type", "rect")
    if t == "rect":
        return f"{l['width']:.3f}x{l['height']:.3f}"
    if t in ("sphere", "disc"):
        return f"r={l.get('radius', 0.0):.4f}"
    if t == "spot":
        return f"cone {l.get('innerConeAngle', 0.0):.0f}..{l.get('outerConeAngle', 0.0):.0f} deg"
    return t


def blackbody_rgb(kelvin):
    """Colour of a blackbody at `kelvin`, as linear sRGB normalised to peak 1.

    V-Ray lights can state their colour as a temperature instead of a triple --
    the bedroom's desk and table lamps are 3800 K and its corridor spot 4800 K --
    and the sidecar carries a colour, so the conversion happens here.

    Planckian locus -> CIE XYZ by the Kim et al. cubic fit for 1667-25000 K, then
    the Rec.709 matrix. Fitted rather than integrated because the input is a
    number an artist typed, not a spectrum, and the fit is under a MacAdam step
    over the range a light bulb occupies.

    Normalised to a peak of 1 rather than to luminance: intensity is authored
    separately, and dividing by luminance here would make a warm light brighter
    than a neutral one at the same number.
    """
    t = max(1667.0, min(25000.0, float(kelvin)))
    if t <= 4000.0:
        x = (-0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910)
    else:
        x = (-3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390)
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    if y <= 1e-6:
        return [1.0, 1.0, 1.0]
    Y = 1.0
    X = x * Y / y
    Z = (1.0 - x - y) * Y / y
    r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    rgb = [max(0.0, c) for c in (r, g, b)]
    peak = max(rgb)
    return [c / peak for c in rgb] if peak > 1e-6 else [1.0, 1.0, 1.0]


def vray_light_color(params):
    """A V-Ray light's colour, whichever way it was authored.

    `color_mode` 1 means the colour comes from `temperature` and the stored
    triple is whatever the UI last showed, so reading the triple regardless is
    how two warm lamps came out white.
    """
    if int(params.get("color_mode", 0)) == 1 and "temperature" in params:
        return blackbody_rgb(params["temperature"])
    c = params.get("color_colortex", [1.0, 1.0, 1.0])
    return [float(x) for x in c] if hasattr(c, "__len__") else [1.0, 1.0, 1.0]


def light_mtl_plane_to_rect(obj, strength, opts):
    """A quad with a V-Ray Light Mtl -> an analytic rect light.

    Left as emissive geometry it is a light Strelka cannot sample: mesh emitters
    are picked up only when a BSDF ray happens to hit them, so a 0.2 x 0.2 quad is
    a noise source and not much else. As an analytic light it gets next-event
    estimation, and it can be marked invisible to the camera -- which is what the
    reference render shows, a light doing its job from outside the frame.

    Any planar mesh is converted, not only a single quad: the plane in this scene
    is subdivided into four, and a rect light is described by its extent rather
    than by its tessellation.
    """
    me = obj.data
    if len(me.vertices) < 3:
        return None

    xs = [v.co.x for v in me.vertices]
    ys = [v.co.y for v in me.vertices]
    zs = [v.co.z for v in me.vertices]
    extents = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
    ordered = sorted(extents, reverse=True)
    if ordered[0] <= 0.0 or ordered[2] > ordered[0] * 1e-3:
        return None  # not planar: not a rect light

    axis_conv = Matrix.Rotation(-math.pi / 2, 4, "X")
    conv = axis_conv @ obj.matrix_world
    loc = conv.translation

    # Orientation from the mesh's own normal, not from the object's rotation.
    #
    # A Blender light emits along its local -Z, and reading the object transform
    # is how you find that. A *quad carrying a light material* has no such
    # convention: this one sits at rotation (0,0,0) with its face in the XZ plane,
    # so its normal is +Y and the object transform says nothing about where it
    # points. Taken as a light transform it emitted away from the room, and the
    # window it is meant to be the sky behind rendered black.
    normal = Vector((0.0, 0.0, 1.0))
    if me.polygons:
        acc = Vector((0.0, 0.0, 0.0))
        for poly in me.polygons:
            acc = acc + poly.normal
        if acc.length > 1e-6:
            normal = (obj.matrix_world.to_3x3() @ acc).normalized()
    world_normal = (axis_conv.to_3x3() @ normal).normalized()
    euler = world_normal.to_track_quat("-Z", "Y").to_euler("XYZ")

    dims = ordered
    sx, sy, sz = obj.matrix_world.to_scale()
    scale = max(abs(sx), abs(sy), abs(sz))

    return {
        "name": obj.name,
        "type": "rect",
        "unit": "radiance",
        "intensity": strength * opts.mesh_light_gain,
        "color": [1.0, 1.0, 1.0],
        "position": [loc.x, loc.y, loc.z],
        "orientation": [math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z)],
        "width": dims[0] * scale,
        "height": dims[1] * scale,
        "enabled": True,
        # Visible, unlike V-Ray's "invisible" rect lights. This quad is the fake
        # sky behind the window -- embedded in the wall's thickness, so the only
        # place the camera can see it is through the window opening, which is
        # exactly where the reference render has a blown-out white pane and this
        # export had a view of the pink backdrop.
        "visibleToCamera": True,
    }


def _boundary_loops(bm):
    """Boundary edges of `bm`, grouped into connected loops."""
    from collections import defaultdict

    boundary = [e for e in bm.edges if len(e.link_faces) == 1]
    at_vert = defaultdict(list)
    for e in boundary:
        at_vert[e.verts[0]].append(e)
        at_vert[e.verts[1]].append(e)
    seen, loops = set(), []
    for e in boundary:
        if e in seen:
            continue
        stack, group = [e], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            group.append(x)
            for v in x.verts:
                stack.extend(y for y in at_vert[v] if y not in seen)
        loops.append(group)
    return loops


def _loop_flatness(loop):
    """How far a boundary loop departs from a plane, as a fraction of its size.

    Principal axes of the loop's vertices: the smallest spread is the thickness
    of the slab they lie in and the largest is how big the loop is, so the ratio
    is scale free and does not care which way the loop is turned. Zero is a
    perfect plane.
    """
    pts = np.array(sorted({tuple(v.co) for e in loop for v in e.verts}), dtype=np.float64)
    if len(pts) < 3:
        return 1.0
    spread = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
    return float(spread[-1] / max(spread[0], 1e-12))


def _loop_span(loop):
    """Largest extent of a boundary loop, in object space."""
    pts = np.array([tuple(v.co) for e in loop for v in e.verts], dtype=np.float64)
    return float(np.max(pts.max(axis=0) - pts.min(axis=0)))


def close_open_transmissive():
    """Close the boundary loops of refracting meshes, where that is sound.

    A path entering a dielectric pushes it onto the renderer's IOR stack and pops
    it on the way out. An open mesh has no way out: the ray leaves through the
    hole without crossing a surface, so the entry never comes off and every
    segment the path travels afterwards -- anywhere in the room -- is attenuated
    as though it were still inside. There is no event for the renderer to hang a
    recovery on, so it has to be repaired here.

    Per loop, not per mesh, because the two failures this scene contains want
    opposite things. The bath water is a dish: one boundary loop, flat, and
    spanning it is exactly the water's surface. The foam is a cluster of open
    shells and the brush is open-ended strips; there is no surface that closing
    them would mean, and both a fill and a bridge invent one -- a fill drew a cap
    across the tub through the duck, a bridge drew a bright streak through the
    foam. Rendered both, and they are unmistakable.

    So: a flat loop is capped, and anything else is left exactly as it was and
    reported. Half-repaired geometry is worse than open geometry, because it
    still unbalances the stack and now also has faces the asset never had.

    Returns (object, faces, edges before, edges after, note) per mesh touched.
    """
    import bmesh  # local: the rest of this tool does not need it

    # A loop this far out of plane is not a hole, it is a torn shell.
    kFlatEnough = 1e-3
    # And a loop this large is not a hole either: it is a feature of the shape,
    # and spanning it does not close a volume, it adds a sheet across one.
    #
    # Measured on this scene, as a fraction of the object it belongs to: the
    # brush's fibre ends run 0.013 to 0.014, the foam's two loops 0.29 and 0.41,
    # the shower water 0.56 and the bath water 1.00. Two clusters an order of
    # magnitude apart, so the threshold sits in the gap rather than on a case.
    #
    # The bath water is the case that needs this. Its one loop is flat, and
    # capping it is arithmetically the best result in the scene -- the water's
    # red-to-green goes 0.776 to 0.823 against the reference's 0.837 -- and the
    # render is plainly wrong: a pale sheet across the tub with the duck half
    # under it. The mesh is a 2 mm dish, not a tub full of water, and its loop
    # spans the whole object. What makes the bath look full today is the very
    # defect this function exists to remove: an unbalanced stack applies the
    # water's absorption to everything the ray meets afterwards, including the
    # inside of the tub. The asset has no water volume, and no repair here can
    # invent one.
    kHoleFraction = 0.1

    report = []
    for obj in list(bpy.data.objects):
        if obj.type != "MESH" or not obj.data.polygons:
            continue
        if not any(sl.material is not None and is_transmissive_material(sl.material)
                   for sl in obj.material_slots):
            continue

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        loops = _boundary_loops(bm)
        if not loops:
            bm.free()
            continue

        before = sum(len(l) for l in loops)
        span = max(obj.dimensions) or 1.0
        flat = [l for l in loops
                if _loop_flatness(l) <= kFlatEnough and _loop_span(l) <= kHoleFraction * span]
        capped = 0
        for loop in flat:
            filled = bmesh.ops.holes_fill(bm, edges=list(loop), sides=0)
            faces = filled.get("faces", [])
            if faces:
                # The fill takes its winding from whichever face it started at,
                # and a dish's rim can be walked either way. Point them the same
                # way as the shell they close, or the surface is inside out and
                # every ray that meets it refracts the wrong way.
                bmesh.ops.recalc_face_normals(bm, faces=faces)
                capped += 1

        after = sum(1 for e in bm.edges if len(e.link_faces) == 1)
        if capped == 0:
            bm.free()
            report.append((obj.name, len(obj.data.polygons), before, after,
                           f"left open, {len(loops)} loop(s), none is a hole"))
            continue

        note = f"capped {capped} of {len(loops)} loop(s)"
        if after:
            note += f", {after} edges left open"
        tris = len(bm.faces)
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
        report.append((obj.name, tris, before, after, note))
    return report


def is_transmissive_material(mat):
    """Does this material refract, as the exported glTF will see it?

    Read off the Principled node the conversion has already built, rather than
    off the V-Ray plugin, so it answers the question the renderer will ask.
    """
    nt = getattr(mat, "node_tree", None)
    if nt is None:
        return False
    for n in nt.nodes:
        if n.type != "BSDF_PRINCIPLED":
            continue
        sock = n.inputs.get("Transmission Weight")
        if sock is not None and not sock.is_linked and sock.default_value > 0.0:
            return True
    return False


def rebuild_proxy_rug(obj, opts):
    """Replace a V-Ray proxy's preview mesh with a coiled braid.

    `Rug_Round.vrmesh` is a proxy: the .blend carries only the viewport preview
    that V-Ray draws in its place -- 9,990 triangles, three unshared vertices
    each, a decimated cloud. Blender cannot read the Chaos format, so the real
    geometry is simply not in the file and no export setting recovers it.

    What is recoverable is the *look*, which is a rope coiled from the centre
    outward. A curve is the natural way to author that: an Archimedean spiral
    with a round bevel is a rope, and converting it to a mesh at export keeps the
    renderer out of it entirely. Curve primitives in the tracer would be the
    bigger feature -- GEOMETRY_MASK_CURVE is reserved and nothing builds them --
    and they are what towel fuzz and hair would need, but a woven rug is
    geometry, not strands.
    """
    radius = max(obj.dimensions.x, obj.dimensions.y) * 0.5
    if radius <= 0.0:
        return None
    height = obj.dimensions.z
    matrix = obj.matrix_world.copy()
    material = obj.material_slots[0].material if obj.material_slots else None

    cord = height * 0.5 if height > 0.0 else radius * 0.04
    turns = max(3, int(radius / max(cord * 2.0, 1e-6)))
    steps_per_turn = 64

    curve = bpy.data.curves.new(obj.name + "_Braid", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = cord
    curve.bevel_resolution = opts.rug_bevel_resolution
    curve.use_fill_caps = True
    spline = curve.splines.new("POLY")
    total = turns * steps_per_turn
    spline.points.add(total - 1)
    for i in range(total):
        t = i / (total - 1)
        theta = t * turns * 2.0 * math.pi
        # Held off the very centre: a spiral through r = 0 self-intersects once
        # the bevel is applied, and the reference has a small closed eye there too.
        r = cord * 1.2 + (radius - cord * 1.6) * t
        # The rope rides up and down a little, which is what reads as a weave
        # rather than as a flat disc of concentric circles.
        z = math.sin(theta * 2.0) * cord * 0.25
        spline.points[i].co = (r * math.cos(theta), r * math.sin(theta), z, 1.0)

    braid = bpy.data.objects.new(obj.name + "_Braid", curve)
    bpy.context.collection.objects.link(braid)
    braid.matrix_world = matrix
    if material is not None:
        braid.data.materials.append(material)
    return braid


def collect_fog_volumes(world, opts):
    """V-Ray EnvironmentFog nodes -> {gizmo object name: medium description}.

    The fog lives in the world node tree and names its gizmo through a "V-Ray
    Object Select" node, so the volume and the mesh that bounds it are found in
    two different places. Both gizmos in this scene are hidden helper objects;
    the export turns them back into renderable geometry carrying a medium
    material.
    """
    volumes = {}
    if world is None or world.node_tree is None:
        return volumes
    for node in world.node_tree.nodes:
        if "EnvironmentFog" not in node.keys():
            continue
        fog = idprops_to_dict(node["EnvironmentFog"])
        gizmo = None
        for sock in node.inputs:
            if not sock.is_linked:
                continue
            up = sock.links[0].from_node
            if "objectName" in up.keys():
                gizmo = str(up["objectName"])
        if gizmo is None:
            continue
        # V-Ray's transparency is what the medium lets through over its own
        # extent; density scales the extinction. Only the product matters to a
        # homogeneous medium, and the scatter colour is what survives one event.
        volumes[gizmo] = {
            "density": float(fog.get("density_tex", 1.0)) * opts.fog_density_scale,
            "scatterColor": [float(c) for c in fog.get("transparency_colortex", [1.0, 1.0, 1.0])],
            "emissionColor": [float(c) * opts.fog_emission_scale
                              for c in fog.get("emission_colortex", [0.0, 0.0, 0.0])],
            "anisotropy": 0.0,
        }
    return volumes


def dome_to_environment(obj, baker, opts):
    """VRayDomeLight -> Strelka environment block.

    A dome is not always an HDRI. `use_dome_tex` off means a plain sky of one
    colour, and the colour can be a temperature -- which is what the kids'
    bedroom has: 6500 K at intensity 0.2, and no texture. Requiring a bitmap
    dropped the environment entirely and the room rendered black but for its
    lamps, so the texture is now what is optional and the dome itself is not.

    A textureless environment reaches the renderer as the miss colour, where a
    cosine-weighted BSDF sample is already the ideal way to integrate a constant
    sky; see the note in MetalRender.
    """
    nt = getattr(obj.data, "node_tree", None)
    dome = {}
    tex_path = None
    if nt is not None:
        for n in nt.nodes:
            if "LightDome" in n.keys():
                dome = idprops_to_dict(n["LightDome"])
            if "BitmapBuffer" in n.keys():
                p = baker.resolve(idprops_to_dict(n["BitmapBuffer"]).get("file"))
                if p is not None:
                    tex_path = str(p)
    if not dome:
        return None

    intensity = float(dome.get("intensity", 1.0))
    # Deliberately not gated on `use_dome_tex`. The bathroom's dome has it off
    # and is unmistakably lit by its HDRI, so the flag means something other than
    # "read the bitmap node" -- and gating on it took that scene's environment
    # away. A linked bitmap that resolves is the texture; anything else is a
    # coloured sky. The kids' bedroom lands in the second case because the file
    # its dome names is not in its Assets folder at all.

    env = {
        "intensity": intensity * opts.env_gain,
        "color": vray_light_color(dome),
        "rotation": math.radians(opts.env_rotation),
    }
    if tex_path is not None:
        env["texture"] = tex_path
        # A texture carries its own colour; the dome's tint would multiply it,
        # and V-Ray's UI leaves a stale one behind whenever the mode is a
        # temperature.
        env["color"] = [1.0, 1.0, 1.0]
    return env


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def add_perspective_proxy(cam_obj, opts):
    """
    Strelka renders the orthographic camera directly now, so this is a fallback
    rather than the main path: a long-lens perspective camera on the same axis,
    pulled back to `--proxy-distance` with the FOV narrowed so the framed extent
    still matches ortho_scale.  At 20 units back for a 0.9-unit frame the
    residual convergence is well under a pixel, which makes it a way to A/B the
    orthographic path against a perspective one that frames the same thing.
    """
    cam = cam_obj.data
    scene = bpy.context.scene
    ortho_scale = cam.ortho_scale
    d = opts.proxy_distance

    proxy_data = bpy.data.cameras.new("Render_Camera_Persp")
    proxy_data.type = "PERSP"
    proxy_data.sensor_fit = "AUTO"
    proxy_data.angle = 2.0 * math.atan((ortho_scale * 0.5) / d)
    proxy_data.clip_start = max(1e-3, d - 10.0)
    proxy_data.clip_end = d + 10.0

    proxy = bpy.data.objects.new("Render_Camera_Persp", proxy_data)
    bpy.context.collection.objects.link(proxy)
    proxy.matrix_world = cam_obj.matrix_world.copy()
    # Blender cameras look down local -Z; step back along +Z.
    back = cam_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
    proxy.location = cam_obj.matrix_world.translation + back.normalized() * d

    return proxy, {
        "orthographic": True,
        "orthoScale": ortho_scale,
        "xmag": ortho_scale * 0.5,
        "ymag": ortho_scale * 0.5 * (scene.render.resolution_y / scene.render.resolution_x),
        "proxyDistance": d,
        "proxyFovDegrees": math.degrees(proxy_data.angle),
    }


# ---------------------------------------------------------------------------
# glTF post-processing
# ---------------------------------------------------------------------------

EXT_MAP = {
    "medium": "STRELKA_materials_medium",
    "volume": "KHR_materials_volume",
    "sheen": "KHR_materials_sheen",
    "iridescence": "KHR_materials_iridescence",
    "diffuseTransmission": "KHR_materials_diffuse_transmission",
    "subsurface": "STRELKA_materials_subsurface",
}


def patch_gltf(gltf_path, per_material, camera_info, report_path, baker):
    """
    Inject what the Blender exporter drops.

    Blender writes KHR_materials_ior / _transmission / _volume / _specular /
    _anisotropy / _emissive_strength / _clearcoat, but not sheen, iridescence or
    anything subsurface -- so those are added here, keyed by material name.
    """
    with open(gltf_path) as f:
        doc = json.load(f)

    used = set()
    by_name = {m.get("name"): m for m in doc.get("materials", [])}
    patched = []
    unmatched = []

    for entry in per_material:
        name = entry["material"]
        extra = entry.get("extra") or {}
        mat = by_name.get(name)
        if mat is None:
            if extra:
                unmatched.append(name)
            continue
        exts = mat.setdefault("extensions", {})
        for key, ext_name in EXT_MAP.items():
            if key in extra:
                exts[ext_name] = extra[key]
                used.add(ext_name)
                patched.append(f"{name}: {ext_name}")
        if extra.get("thinWalled"):
            exts.setdefault("KHR_materials_volume", {})["thicknessFactor"] = 0.0
            used.add("KHR_materials_volume")
        # KHR_materials_clearcoat has no IOR field; Blender wrote the rest of the
        # extension, so the IOR is merged into what is already there rather than
        # replacing it.
        coat = extra.get("coat")
        if coat and "KHR_materials_clearcoat" in exts:
            exts["KHR_materials_clearcoat"]["clearcoatIor"] = coat["ior"]
            patched.append(f"{name}: clearcoatIor")

    for cam in doc.get("cameras", []):
        if cam.get("type") == "orthographic":
            cam.setdefault("extras", {}).update(camera_info)

    if used:
        doc["extensionsUsed"] = sorted(set(doc.get("extensionsUsed", [])) | used)

    with open(gltf_path, "w") as f:
        json.dump(doc, f)

    summary = {
        "patchedMaterials": patched,
        "unmatchedMaterials": unmatched,
        "extensionsUsed": doc.get("extensionsUsed", []),
        "images": len(doc.get("images", [])),
        "bakedTextures": baker.baked,
        "missingTextures": baker.missing,
        "camera": camera_info,
    }
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scenes/iso_bathroom")
    ap.add_argument("--name", default="iso_bathroom")
    ap.add_argument("--assets", default=None, help="Assets dir (default: <blend>/Assets)")
    ap.add_argument("--sss-scale", type=float, default=0.002,
                    help="V-Ray scatter radius (cm) -> scene units. The scene is "
                         "authored at roughly 1 unit = 5 m (the room is 0.5 across "
                         "and a bathroom is ~2.5 m), so a centimetre is ~0.002.")
    ap.add_argument("--sss-fallback-dt", type=float, default=0.4,
                    help="diffuse-transmission factor written as the no-SSS fallback")
    ap.add_argument("--fog-scale", type=float, default=0.01,
                    help="one centimetre in scene units; V-Ray's fog depth is in "
                         "centimetres. attenuationDistance = fog-scale * fog_mult")
    ap.add_argument("--rect-light-gain", type=float, default=1.0)
    ap.add_argument("--mesh-light-gain", type=float, default=1.0)
    ap.add_argument("--env-gain", type=float, default=1.0)
    ap.add_argument("--env-rotation", type=float, default=0.0)
    # Exposure, as a plain multiplier. Written with fstop and shutter pinned at 1
    # so the photographic equation collapses to iso/100 and the number in the
    # sidecar is the number that multiplies the image.
    #
    # 1.5 is fitted, and to a measurement rather than to a look: mean brightness
    # over the blue floor, the outer wall and the backdrop against the reference
    # comes out at 0.89, 1.07 and 1.02.
    ap.add_argument("--exposure", type=float, default=1.5,
                    help="exposure multiplier written to the light sidecar")
    ap.add_argument("--noise-bump-gain", type=float, default=3.0)
    # V-Ray's fog density is in units this conversion cannot recover: the value
    # is 10 and the gizmo is a quarter of a scene unit across, which taken at face
    # value is an optical depth of 2.4 and renders the bathtub as a white blob.
    # These two are fits to the reference render, and are the only numbers in this
    # script that are.
    ap.add_argument("--fog-density-scale", type=float, default=0.12,
                    help="multiplier on V-Ray's fog density (a fit, see the note)")
    ap.add_argument("--no-fog-volumes", dest="fog_volumes", action="store_false",
                    help="skip the EnvironmentFog gizmos. The bathtub one costs "
                         "more than it buys today -- see the note in main()")
    # On by default now that the repair decides per loop rather than per scene:
    # a flat rim is capped and a torn shell is left alone and reported. The
    # per-scene version had to be opt-in because it could only do both or
    # neither, and this asset wants one of each. See docs/open-defects.md.
    ap.add_argument("--no-close-transmissive", dest="close_transmissive",
                    action="store_false",
                    help="leave refracting meshes open; an unclosed one unbalances "
                         "the IOR stack and tints everything the path meets after it")
    ap.add_argument("--rebuild-rug", action="store_true",
                    help="replace the Rug_Round proxy preview with a generated "
                         "coiled braid (the .vrmesh it stands in for is unreadable)")
    ap.add_argument("--rug-bevel-resolution", type=int, default=3)
    ap.add_argument("--fog-emission-scale", type=float, default=1.0,
                    help="multiplier on V-Ray's fog emission colour")
    ap.add_argument("--proxy-distance", type=float, default=20.0)
    ap.add_argument("--drop-ortho-camera", action="store_true",
                    help="export only the long-lens perspective proxy")
    ap.add_argument("--no-subdiv", action="store_true",
                    help="strip Subdivision modifiers (faster, coarser)")
    ap.add_argument("--format", default="GLTF_SEPARATE", choices=["GLTF_SEPARATE", "GLB"])
    return ap.parse_args(argv)


def main():
    opts = parse_args()
    blend_path = Path(bpy.data.filepath)
    if not blend_path.name:
        print("ERROR: run with `blender -b <file.blend> -P ...`")
        sys.exit(1)

    assets = Path(opts.assets) if opts.assets else blend_path.parent / "Assets"
    out_dir = Path(opts.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== vray2strelka ===")
    print(f"  blend : {blend_path}")
    print(f"  assets: {assets}")
    print(f"  out   : {out_dir}")

    baker = TextureBaker(assets, out_dir / "baked")
    conv = MaterialConverter(baker, opts)

    n_conv = 0
    for mat in bpy.data.materials:
        try:
            if conv.convert(mat) is not None:
                n_conv += 1
        except Exception as e:  # one bad material must not kill the export
            print(f"  !! {mat.name}: {type(e).__name__}: {e}")
    print(f"  materials converted: {n_conv} / {len(bpy.data.materials)}")
    print(f"  textures baked     : {baker.baked}")
    if baker.missing:
        print(f"  textures MISSING   : {baker.missing}")

    scene_world = bpy.context.scene.world
    lights = []
    environment = None
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        vray = obj.data.get("vray")
        if vray is None:
            continue
        vray = idprops_to_dict(vray)
        if int(vray.get("light_type", 9)) == 10:
            environment = dome_to_environment(obj, baker, opts)
        else:
            d = light_to_sidecar(obj, vray, opts)
            if d:
                lights.append(d)
    for l in lights:
        print(f"  light : {l['name']:24s} {l['type']:6s} L={l['intensity']:.2f} "
              f"{light_shape_summary(l)}")
    if environment:
        where = (Path(environment["texture"]).name if "texture" in environment
                 else "colour " + " ".join(f"{c:.2f}" for c in environment["color"]))
        print(f"  env   : {where}  x{environment['intensity']:.3f}")

    # A V-Ray proxy carries no geometry Blender can read; rebuild the rug as a
    # coiled rope if asked.
    if opts.rebuild_rug:
        for obj in list(bpy.data.objects):
            if obj.type != "MESH" or "Rug_Round" not in obj.name:
                continue
            braid = rebuild_proxy_rug(obj, opts)
            if braid is None:
                continue
            print(f"  rug   : {obj.name} preview replaced with a coiled braid")
            bpy.data.objects.remove(obj, do_unlink=True)

    # V-Ray EnvironmentFog gizmos become renderable geometry carrying a medium
    # material. They are hidden helpers in the .blend precisely because V-Ray
    # never draws them; here the boundary *is* the description of the volume, so
    # it has to reach the renderer.
    # The bathtub volume is currently a net loss and the reason is a defect, not a
    # modelling choice.
    #
    # Measured on the bath water's red-to-green ratio, against the reference's
    # 0.871: with the gizmo present 0.975, with its density cut fourfold 0.971,
    # with the density at essentially zero and no emission 0.971, and with the
    # gizmo gone 0.734. Four orders of magnitude of density move it by 0.004 and
    # removing the boundary moves it by 0.24 -- so it is the crossing that costs
    # the colour, not the medium.
    #
    # One cause was found and fixed: the crossing returned before the block that
    # applies absorption over the segment just travelled, so a ray leaving the
    # water through the gizmo lost the water's cyan. That was worth 0.011. The
    # rest is unexplained, and until it is, this switch is the honest lever.
    fog_volumes = collect_fog_volumes(scene_world, opts) if opts.fog_volumes else {}
    fog_materials = {}
    for gizmo_name, medium in fog_volumes.items():
        obj = bpy.data.objects.get(gizmo_name)
        if obj is None or obj.type != "MESH":
            print(f"  fog   : gizmo '{gizmo_name}' not found, volume dropped")
            continue
        # Hidden three different ways in the .blend, because V-Ray never draws a
        # gizmo: render visibility, viewport visibility, and the view-layer
        # hide flag. The export filters on visible_get(), so all three matter.
        obj.hide_render = False
        obj.hide_viewport = False
        try:
            obj.hide_set(False)
        except Exception:
            pass
        for coll in obj.users_collection:
            coll.hide_render = False
            coll.hide_viewport = False
        mat = bpy.data.materials.new(f"{gizmo_name}_Medium")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        # Fully transmissive so any viewer without the extension sees through the
        # gizmo rather than through a solid box.
        bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.0
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        fog_materials[mat.name] = medium
        conv.report.append({"material": mat.name, "kind": "EnvironmentFog",
                            "extra": {"medium": medium}})
        print(f"  fog   : {gizmo_name} density={medium['density']:.2f} "
              f"emission={medium['emissionColor'][0]:.3f}")

    # V-Ray Light Mtl quads become analytic lights, and stop being geometry.
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        names = [sl.material.name for sl in obj.material_slots if sl.material]
        if not names or names[0] not in conv.light_materials:
            continue
        desc = light_mtl_plane_to_rect(obj, conv.light_materials[names[0]], opts)
        if desc is None:
            print(f"  light : {obj.name} kept as emissive geometry (not a quad)")
            continue
        lights.append(desc)
        bpy.data.objects.remove(obj, do_unlink=True)
        print(f"  light : {desc['name']:24s} {desc['type']:6s} L={desc['intensity']:.2f} "
              f"{light_shape_summary(desc)} (from a light material)")

    scene = bpy.context.scene
    cam_obj = scene.camera
    camera_info = {}
    if cam_obj and cam_obj.data.type == "ORTHO":
        proxy, camera_info = add_perspective_proxy(cam_obj, opts)
        scene.camera = proxy
        print(f"  camera: ORTHO scale={cam_obj.data.ortho_scale:.3f} -> perspective proxy "
              f"fov={camera_info['proxyFovDegrees']:.3f} deg at d={opts.proxy_distance}")
        if opts.drop_ortho_camera:
            bpy.data.objects.remove(cam_obj, do_unlink=True)
            print("  camera: orthographic camera dropped from the export")

    if opts.close_transmissive:
        for name, faces, before, after, note in close_open_transmissive():
            print(f"  mesh  : {name} ({faces} faces) {before} boundary edges -> {after}"
                  f"  [{note}]")

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if opts.no_subdiv:
            for m in list(obj.modifiers):
                if m.type == "SUBSURF":
                    obj.modifiers.remove(m)
        if not any(m.type == "TRIANGULATE" for m in obj.modifiers):
            m = obj.modifiers.new("_Triangulate", "TRIANGULATE")
            m.quad_method = "BEAUTY"
            m.ngon_method = "BEAUTY"


    ext = ".gltf" if opts.format == "GLTF_SEPARATE" else ".glb"
    gltf_path = out_dir / f"{opts.name}{ext}"
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(gltf_path),
        export_format=opts.format,
        use_visible=True,
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_vertex_color="NONE",
        export_cameras=True,
        export_lights=False,
        export_apply=True,
        export_image_format="AUTO",
        export_materials="EXPORT",
        export_yup=True,
    )
    print(f"  wrote {gltf_path}")

    sidecar = {"lights": lights}
    if environment:
        sidecar["environment"] = environment
    sidecar["exposure"] = {
        "iso": opts.exposure * 100.0,
        "fstop": 1.0,
        "shutter": 1.0,
        "cm2_factor": 1.0,
    }
    sidecar_path = out_dir / f"{opts.name}_light.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=4)
    print(f"  wrote {sidecar_path}")

    if opts.format == "GLTF_SEPARATE":
        summary = patch_gltf(gltf_path, conv.report, camera_info,
                             out_dir / f"{opts.name}_conversion.json", baker)
        print(f"  patched {len(summary['patchedMaterials'])} material extensions; "
              f"{summary['images']} images; extensions: {summary['extensionsUsed']}")
        if summary["unmatchedMaterials"]:
            print(f"  !! material names not found in glTF: {summary['unmatchedMaterials']}")

    with open(out_dir / f"{opts.name}_materials.json", "w") as f:
        json.dump(conv.report, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
