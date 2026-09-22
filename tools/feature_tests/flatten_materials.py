#!/usr/bin/env python3
"""
Rewrite shader graphs the glTF exporter cannot express into ones it can.

Imported by export_scene.py; not run on its own.

The glTF exporter understands one shape: a Principled BSDF wired straight to the
material output. Anything else -- and production foliage is always something else
-- exports as a flat opaque material with, at best, the base colour texture. The
pine forest's needle clusters came through as solid green cards, 7 800 of them,
and the render was unrecognisable.

Two idioms are handled here, both being the same thing written two ways:

    Output <- Add(Mix(mask, Principled, Transparent), Translucent)      pine_cover
    Output <- Mix(alpha, Transparent, Add(Principled, Mix(..)))         fir_twig

Note the operand order differs, and so does the polarity; the reasoning is at the
rewiring itself. Once the graph is a plain Principled BSDF the exporter merges the
base colour and the alpha into one RGBA texture on its own, which is the only
place glTF has to put alpha.

The Translucent BSDF these materials add is carried too, as
KHR_materials_diffuse_transmission -- light through a leaf is a large part of how
foliage reads, and it is emphatically not KHR_materials_transmission, which is
specular and would make needles look like glass. The exporter has no idea about
either, so the weight and tint are collected here and injected into the glTF
afterwards by export_scene.py.
"""

import bpy
import math
import numpy as np
import os
import zlib


def _surface(mat):
    out = next((n for n in mat.node_tree.nodes
                if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if out is None or not out.inputs["Surface"].links:
        return None
    return out.inputs["Surface"].links[0].from_node, out


def _find_principled(node, depth=0):
    """The Principled BSDF under a tree of Mix/Add shader nodes."""
    if node is None or depth > 8:
        return None
    if node.type == "BSDF_PRINCIPLED":
        return node
    if node.type in {"MIX_SHADER", "ADD_SHADER"}:
        for i in node.inputs:
            if i.type != "SHADER" or not i.links:
                continue
            found = _find_principled(i.links[0].from_node, depth + 1)
            if found is not None:
                return found
    return None


def _shadow_transparent_principled(node):
    """Principled branch of Mix(Is Shadow Ray, Principled, Transparent)."""
    if node is None or node.type != "MIX_SHADER":
        return None
    factor = node.inputs[0]
    if not factor.links or factor.links[0].from_node.type != "LIGHT_PATH" or \
            factor.links[0].from_socket.name != "Is Shadow Ray":
        return None
    shaders = [i for i in node.inputs if i.type == "SHADER"]
    if len(shaders) != 2 or not all(i.links for i in shaders):
        return None
    return shaders[0].links[0].from_node if shaders[0].links[0].from_node.type == "BSDF_PRINCIPLED" and \
        shaders[1].links[0].from_node.type == "BSDF_TRANSPARENT" else None


def _alpha_source(node, depth=0):
    """(socket driving the mix, transparent_is_first) for the Transparent branch.

    Returns None when the graph has no Transparent BSDF, which is the common
    case and not a problem -- an opaque material needs no rewriting.
    """
    if node is None or depth > 8:
        return None
    if node.type == "MIX_SHADER":
        shader_inputs = [i for i in node.inputs if i.type == "SHADER"]
        for slot, i in enumerate(shader_inputs):
            if i.links and i.links[0].from_node.type == "BSDF_TRANSPARENT":
                fac = node.inputs["Fac"] if "Fac" in node.inputs else node.inputs[0]
                if not fac.links:
                    return None
                return fac.links[0], slot == 0
    if node.type in {"MIX_SHADER", "ADD_SHADER"}:
        for i in node.inputs:
            if i.type != "SHADER" or not i.links:
                continue
            found = _alpha_source(i.links[0].from_node, depth + 1)
            if found is not None:
                return found
    return None


def _image_of(link):
    """The image behind a link, if it is a texture read and nothing else."""
    n = link.from_node
    if n.type == "TEX_IMAGE" and n.image is not None:
        return n.image, n
    return None, None


def _find_translucent(node, depth=0):
    """The Translucent BSDF under a tree of Mix/Add shader nodes, if any."""
    if node is None or depth > 8:
        return None
    if node.type == "BSDF_TRANSLUCENT":
        return node
    if node.type in {"MIX_SHADER", "ADD_SHADER"}:
        for i in node.inputs:
            if i.type != "SHADER" or not i.links:
                continue
            found = _find_translucent(i.links[0].from_node, depth + 1)
            if found is not None:
                return found
    return None


def _mean_colour(socket, depth=0):
    """The average colour a socket produces, following one texture back if need be.

    A constant is returned as-is. A texture is averaged over its pixels, which is
    a real approximation and a defensible one here: light diffusing through a
    leaf comes out low-frequency, so the transmitted tint carries almost none of
    the reflected texture's detail. glTF has a diffuseTransmissionColorTexture
    for the exact answer; this renderer samples only the factor.
    """
    if depth > 6:
        return None
    if not socket.links:
        v = getattr(socket, "default_value", None)
        if v is None:
            return None
        if hasattr(v, "__len__"):
            return [float(v[0]), float(v[1]), float(v[2])]
        # A scalar socket -- a Fac or a Value -- read as grey.
        return [float(v)] * 3
    node = socket.links[0].from_node
    if node.type == "TEX_IMAGE" and node.image is not None:
        w, h = node.image.size
        if w == 0 or h == 0:
            return None
        buf = np.empty(w * h * 4, dtype=np.float32)
        node.image.pixels.foreach_get(buf)
        px = buf.reshape(-1, 4)
        # Weighted by alpha where there is one: the transparent margin of a
        # foliage card is black, and averaging it in makes every leaf too dark.
        alpha = px[:, 3]
        if alpha.max() > 0.0 and alpha.min() < 1.0:
            weight = alpha.sum()
            return [float((px[:, c] * alpha).sum() / weight) for c in range(3)]
        return [float(px[:, c].mean()) for c in range(3)]
    # A plain RGB constant: the colour is on the node's output, not on any input,
    # so the name loop below never reaches it. The monster's skin is authored
    # this way -- a purple RGB node behind a Mix -- and without this the exporter
    # dropped the colour and wrote white, which lost both the diffuse tint and
    # the subsurface hue it is derived from.
    if node.type == "RGB":
        value = node.outputs[0].default_value
        return [float(value[0]), float(value[1]), float(value[2])]
    # The generic Mix node (ShaderNodeMix) carries duplicate A/B sockets for
    # every data type, so inputs["A"] resolves to the unused float one and the
    # name loop misses the colour entirely. Blend the two RGBA sockets by the
    # clamped factor, which for the monster's factor of 3 collapses to input B.
    if node.type == "MIX" and getattr(node, "data_type", "") == "RGBA":
        rgba_inputs = [i for i in node.inputs if i.type == "RGBA"]
        if len(rgba_inputs) >= 2:
            a = _mean_colour(rgba_inputs[0], depth + 1)
            b = _mean_colour(rgba_inputs[1], depth + 1)
            if a is None or b is None:
                return a if a is not None else b
            fac_socket = next((i for i in node.inputs
                               if i.name == "Factor" and i.type == "VALUE"), None)
            fac = float(fac_socket.default_value) if fac_socket is not None else 0.5
            if getattr(node, "clamp_factor", True):
                fac = min(max(fac, 0.0), 1.0)
            return [a[c] * (1.0 - fac) + b[c] * fac for c in range(3)]
    # Colour inputs only, and named ones first. Wandering into a Vector or a Fac
    # returns a number that is not a colour, and it is the sort of wrong answer
    # that renders rather than raises.
    for name in ("Color", "Base Color", "Color1", "A"):
        if name in node.inputs:
            got = _mean_colour(node.inputs[name], depth + 1)
            if got is not None:
                return got
    for i in node.inputs:
        if i.type == "RGBA":
            got = _mean_colour(i, depth + 1)
            if got is not None:
                return got
    return None


def _reaches_texture(socket, depth=0):
    """Whether a socket's chain ends at an image the exporter can carry itself."""
    if depth > 6 or not socket.links:
        return False
    node = socket.links[0].from_node
    if node.type == "TEX_IMAGE" and node.image is not None:
        return True
    return any(_reaches_texture(i, depth + 1) for i in node.inputs)


def _source_images(socket, depth=0, seen=None):
    """Image nodes contributing to a socket, for spotting layered base colour."""
    if depth > 12 or not socket.links:
        return set()
    seen = set() if seen is None else seen
    node = socket.links[0].from_node
    if node in seen:
        return set()
    seen.add(node)
    if node.type == "TEX_IMAGE" and node.image is not None:
        return {node.image}
    out = set()
    for inp in node.inputs:
        out.update(_source_images(inp, depth + 1, seen))
    return out


def _walk_node_trees(tree, seen=None):
    """The tree and its groups, once each."""
    seen = set() if seen is None else seen
    if tree in seen:
        return
    seen.add(tree)
    yield tree
    for node in tree.nodes:
        if node.type == "GROUP" and node.node_tree is not None:
            yield from _walk_node_trees(node.node_tree, seen)


def _uv_image_bake_safe(mat, socket):
    """Whether a socket is an image function of UV alone.

    Such a graph can be evaluated on one 0..1 quad and reattached to the
    original meshes without touching their UVs. Object/generated/attribute
    coordinates cannot: their value depends on the destination geometry.
    """
    if not _source_images(socket):
        return False
    spatial = {"ATTRIBUTE", "VERTEX_COLOR", "NEW_GEOMETRY", "OBJECT_INFO",
               "PARTICLE_INFO", "HAIR_INFO"}
    mapping_periodic = True
    for tree in _walk_node_trees(mat.node_tree):
        for node in tree.nodes:
            if node.type in spatial and any(out.links for out in node.outputs):
                return False
            if node.type == "TEX_COORD":
                if any(out.links and out.name != "UV" for out in node.outputs):
                    return False
            if node.type == "MAPPING":
                for name, expected in {"Location": (0.0, 0.0, 0.0),
                                       "Rotation": (0.0, 0.0, 0.0),
                                       "Scale": (1.0, 1.0, 1.0)}.items():
                    inp = node.inputs.get(name)
                    if inp is None or inp.links or any(
                            abs(float(a) - b) > 1e-6 for a, b in zip(inp.default_value, expected)):
                        mapping_periodic = False
            if node.type == "GROUP" and node.node_tree is not None and \
                    node.node_tree.name == "MAX_Texture_Mapping":
                defaults = {
                    "Angle": 0.0, "Location": (0.0, 0.0, 0.0),
                    "Scale": (1.0, 1.0, 1.0), "CropU": 0.0, "CropV": 0.0,
                    "CropWidth": 1.0, "CropHeight": 1.0,
                }
                for name, expected in defaults.items():
                    inp = node.inputs.get(name)
                    if inp is None or inp.links:
                        mapping_periodic = False
                        break
                    value = inp.default_value
                    if hasattr(value, "__len__"):
                        if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(value, expected)):
                            mapping_periodic = False
                            break
                    elif abs(float(value) - float(expected)) > 1e-6:
                        mapping_periodic = False
                        break
    if mapping_periodic:
        return True
    users = [ob for ob in bpy.data.objects if ob.type == "MESH" and
             any(slot.material == mat for slot in ob.material_slots)]
    return bool(users) and all(
        ob.data.uv_layers and all(0.0 <= p.uv.x <= 1.0 and 0.0 <= p.uv.y <= 1.0
                                  for p in next((u for u in ob.data.uv_layers if u.active_render),
                                                ob.data.uv_layers[0]).data)
        for ob in users)


def _mean_scalar(socket, depth=0):
    """The average value a scalar socket produces.

    Colour chains first, then scalar ones, then the socket's own value. That last
    fallback is not a guess dressed up: Blender keeps default_value on a linked
    socket, and it is what the author set before wiring something in, so for the
    river's Transmission Weight -- driven by noise, defaulting to 1.0 -- it is
    exactly right. Where it is not, it is at least the value the material would
    have had with the graph deleted, which is what the exporter was going to do
    anyway.
    """
    colour = _mean_colour(socket, depth)
    if colour is not None:
        return sum(colour) / 3.0
    if socket.links and depth < 6:
        node = socket.links[0].from_node
        for i in node.inputs:
            if i.type != "VALUE":
                continue
            got = _mean_scalar(i, depth + 1)
            if got is not None:
                return got
    value = getattr(socket, "default_value", None)
    if value is None or hasattr(value, "__len__"):
        return None
    return float(value)


# Inputs the exporter reads as a plain value and silently drops when a node
# drives them. Base Color is handled separately because a texture behind it is
# carried fine and only a procedural chain is not.
_SCALAR_INPUTS = ("Metallic", "Roughness", "IOR", "Transmission Weight", "Transmission",
                  "Specular IOR Level", "Anisotropic")


def _reduce_inputs(principled):
    """Collapse node-driven inputs to their average, where they would be lost.

    The exporter carries a constant or a texture and nothing in between. The
    river's water is a Principled whose Base Color and Transmission Weight are
    both driven by procedural noise, so it exported as a smooth *opaque white*
    surface -- which rendered as a bright sandbank where the reference has dark
    water, and looked like missing geometry rather than a dropped input.

    An average is a real approximation and is stated as one. It is the same
    trade as the translucency tint above, and the alternative is baking every
    such input to a texture, which is a much larger piece of work for inputs
    that are usually near-constant anyway.
    """
    changed = []
    base = principled.inputs.get("Base Color")
    if base is not None and base.links and not _reaches_texture(base):
        colour = _mean_colour(base)
        if colour is not None:
            for link in list(base.links):
                principled.id_data.links.remove(link)
            base.default_value = (colour[0], colour[1], colour[2], 1.0)
            changed.append("base colour")

    for name in _SCALAR_INPUTS:
        socket = principled.inputs.get(name)
        if socket is None or not socket.links or _reaches_texture(socket):
            continue
        value = _mean_scalar(socket)
        if value is None:
            continue
        for link in list(socket.links):
            principled.id_data.links.remove(link)
        socket.default_value = value
        changed.append("%s=%.3f" % (name, value))
    return changed


def _flatten_emission_multiplier(principled):
    """Move an RGB-multiply group into Principled's scalar emission strength."""
    color = principled.inputs.get("Emission Color") or principled.inputs.get("Emission")
    strength = principled.inputs.get("Emission Strength")
    if color is None or strength is None or not color.links or strength.links:
        return None
    group = color.links[0].from_node
    if group.type != "GROUP" or group.node_tree is None:
        return None
    source = group.inputs.get("Color")
    value = group.inputs.get("Value")
    multiplies = [n for n in group.node_tree.nodes if n.type == "MATH" and n.operation == "MULTIPLY"]
    if source is None or not source.links or value is None or value.links or len(multiplies) < 3:
        return None
    gain = float(value.default_value)
    if gain <= 0.0:
        return None
    upstream = source.links[0].from_socket
    for link in list(color.links):
        principled.id_data.links.remove(link)
    principled.id_data.links.new(upstream, color)
    strength.default_value = float(strength.default_value) * gain
    return gain


def _apply_rgb_curve(node, rgb):
    """A colour through an RGB Curves node: the combined curve, then per-channel."""
    mapping = node.mapping
    try:
        mapping.initialize()
    except (RuntimeError, AttributeError):
        pass
    out = []
    for c in range(3):
        v = float(rgb[c])
        v = mapping.evaluate(mapping.curves[3], v)
        v = mapping.evaluate(mapping.curves[c], v)
        out.append(v)
    return out


def _adjusted_mean(socket, depth=0):
    """Mean colour at a socket, applying the colour grades _mean_colour ignores.

    _mean_colour walks straight through an RGB Curves node to the texture behind
    it, which is what the exporter carries. This is the other half: the same
    walk with the grade applied, so the difference between the two is exactly the
    shift the exporter drops.
    """
    if depth > 6 or not socket.links:
        return _mean_colour(socket, depth)
    node = socket.links[0].from_node
    if node.type == "CURVE_RGB":
        inp = _adjusted_mean(node.inputs["Color"], depth + 1)
        if inp is None:
            return None
        return _apply_rgb_curve(node, inp)
    return _mean_colour(socket, depth)


def _base_color_tint(principled):
    """A multiplicative tint for a textured base colour whose grade is dropped.

    The pyjama fabric is a grey texture pushed to purple by an RGB Curves node.
    glTF cannot carry the curve, so the exporter keeps the grey texture and the
    hood comes out white. Rather than bake a whole new texture, the average shift
    the grade applies is folded into baseColorFactor -- which glTF multiplies
    onto the texture -- so the weave stays and the colour returns. Exact only for
    a per-channel scaling, an approximation for a curved one, and stated as such.
    """
    base = principled.inputs.get("Base Color")
    if base is None or not base.links or not _reaches_texture(base):
        return None
    # A bare texture is carried whole by the exporter -- nothing to recover.
    if base.links[0].from_node.type == "TEX_IMAGE":
        return None
    in_mean = _mean_colour(base)
    out_mean = _adjusted_mean(base)
    if in_mean is None or out_mean is None:
        return None
    tint = []
    for c in range(3):
        denom = in_mean[c] if in_mean[c] > 1e-4 else 1e-4
        tint.append(min(max(out_mean[c] / denom, 0.0), 1.0))
    if all(abs(t - 1.0) < 0.02 for t in tint):
        return None
    return tint


def _volume_absorption(mat):
    """(attenuation colour, attenuation distance) from a Volume Absorption node.

    glTF has KHR_materials_volume for exactly this and the exporter writes it
    only from setups it recognises, which a Volume Absorption wired straight to
    the output is not. Blender's density is an absorption coefficient, so the
    distance light travels before the colour is reached is its reciprocal.
    """
    out = next((n for n in mat.node_tree.nodes
                if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if out is None:
        return None
    socket = out.inputs.get("Volume")
    if socket is None or not socket.links:
        return None
    node = socket.links[0].from_node
    if node.type not in {"VOLUME_ABSORPTION", "PRINCIPLED_VOLUME"}:
        return None
    colour = node.inputs["Color"].default_value
    density = float(node.inputs["Density"].default_value)
    if density <= 0.0:
        return None
    return [float(colour[0]), float(colour[1]), float(colour[2])], 1.0 / density


def _diffuse_to_single_scattering_albedo(a):
    """Diffuse albedo -> single-scattering albedo (Van de Hulst inversion).

    The same fit build_features.py and tools/iso_bathroom/vray2strelka.py use for
    STRELKA_materials_subsurface: a DCC subsurface colour is a diffuse albedo,
    and the random walk wants the probability that one extinction event
    scatters. Feeding the diffuse value straight in darkens every multi-scatter
    path by albedo^n.
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


def _cycles_radius_scale(a):
    """Cycles' searchlight remap of a subsurface radius, per channel.

    Christensen-Burley's random-walk fit has two halves: the diffuse albedo
    becomes a single-scattering albedo, and the radius is stretched by
    s = 1.9 - A + 3.5 (A - 0.8)^2 before it is the mean free path Cycles walks.
    Only the first half was applied here, so every channel scattered over a
    shorter distance than Cycles gives it -- and s varies per channel, 2.25 for
    the monster's red against 1.10 for its blue, so the error is chromatic. Red
    stopped reaching through a thin lobe, the hue collapsed toward the surface
    albedo, and a magenta that Cycles renders at saturation 0.9 came out a pale
    salmon at 0.49.
    """
    a = min(max(float(a), 0.0), 1.0)
    return 1.9 - a + 3.5 * (a - 0.8) ** 2


def _subsurface(principled):
    """STRELKA_materials_subsurface payload from a Principled BSDF, or None.

    Cycles' Principled v2 carries subsurface as a weight, a per-channel radius
    scaled by a scalar, and Base Color as the surface albedo -- there is no
    separate Subsurface Color input since 4.0. The extension wants the
    single-scattering albedo, so Base Color is inverted through Van de Hulst per
    channel and kept unchanged as scatterReference, which the loader divides the
    textured base colour by so a textured skin still scatters the right amount.
    Radius is the mean free path in world units, which is Blender's Subsurface
    Radius times Subsurface Scale.
    """
    weight_socket = principled.inputs.get("Subsurface Weight")
    if weight_socket is None:
        return None
    weight = _mean_scalar(weight_socket)
    if weight is None or weight <= 0.0:
        return None

    radius = [0.01, 0.01, 0.01]
    radius_socket = principled.inputs.get("Subsurface Radius")
    if radius_socket is not None:
        value = getattr(radius_socket, "default_value", None)
        if value is not None and hasattr(value, "__len__"):
            radius = [float(value[0]), float(value[1]), float(value[2])]
    scale_socket = principled.inputs.get("Subsurface Scale")
    scale = _mean_scalar(scale_socket) if scale_socket is not None else 1.0
    if scale is None:
        scale = 1.0
    radius = [r * scale for r in radius]
    if max(radius) <= 0.0:
        return None

    anisotropy = 0.0
    aniso_socket = principled.inputs.get("Subsurface Anisotropy")
    if aniso_socket is not None:
        got = _mean_scalar(aniso_socket)
        if got is not None:
            anisotropy = got

    # scatterReference is the diffuse albedo the single-scattering colour is
    # derived from. Prefer the averaged Base Color, but a monster's skin runs its
    # colour through a procedural Mix chain _mean_colour cannot walk, and there
    # the socket keeps the constant the author last set behind the graph -- a far
    # better reference than a flat grey, which would leave the loader scaling the
    # skin texture against the wrong albedo and shift the scatter off-hue.
    base_socket = principled.inputs.get("Base Color")
    base = _mean_colour(base_socket) if base_socket is not None else None
    if base is None and base_socket is not None:
        cached = getattr(base_socket, "default_value", None)
        if cached is not None and hasattr(cached, "__len__"):
            base = [float(cached[0]), float(cached[1]), float(cached[2])]
    if base is None:
        base = [0.8, 0.8, 0.8]
    base = [min(max(c, 0.0), 1.0) for c in base]
    scatter = [_diffuse_to_single_scattering_albedo(c) for c in base]
    # The other half of the remap the single-scattering albedo above comes from.
    radius = [radius[c] * _cycles_radius_scale(base[c]) for c in range(3)]
    return {
        "weight": float(weight),
        "radius": radius,
        "anisotropy": float(anisotropy),
        "reference": base,
        "scatter": scatter,
    }


def _invert_image(img, out_dir):
    """1 - value, written to disk next to the export.

    Cached per source image, so a mask shared by several materials is inverted
    once rather than once per use.
    """
    name = "%s_inv" % os.path.splitext(img.name)[0]
    existing = bpy.data.images.get(name)
    if existing is not None:
        return existing

    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    px = buf.reshape(-1, 4)
    px[:, :3] = 1.0 - px[:, :3]

    # Byte buffer, and the colour space set before the pixels are written. Both
    # matter, and neither fails loudly. A float buffer saved to PNG puts the
    # values through a transfer function on the way out (0.75 arrives as 0.52),
    # and setting the colour space afterwards drops the write entirely -- the
    # file comes out uniformly black, which reads as a mask covering nothing
    # rather than as a save that did not happen. Measured, all four ways.
    out = bpy.data.images.new(name, width=w, height=h, alpha=False)
    out.colorspace_settings.name = img.colorspace_settings.name
    out.pixels.foreach_set(px.ravel())
    out.update()
    out.filepath_raw = os.path.join(out_dir, name + ".png")
    out.file_format = "PNG"
    out.save()
    return out


def _normal_is_plain_map(socket):
    """Whether a Normal input is already a tangent-space normal map, which glTF
    carries as-is and nothing here needs to touch."""
    if not socket.links:
        return True
    node = socket.links[0].from_node
    if node.type != "NORMAL_MAP" or node.space != "TANGENT":
        return False
    src = node.inputs["Color"]
    return bool(src.links) and src.links[0].from_node.type == "TEX_IMAGE"


def _connect_baked_map(tree, principled, target, kind):
    normal = kind == "normal"
    socket = principled.inputs["Normal" if normal else "Roughness" if kind == "roughness" else
                               "Alpha" if kind == "alpha" else
                               "Emission Color" if kind == "emission" else "Base Color"]
    source = target.outputs["Color"]
    if normal:
        nmap = tree.nodes.new("ShaderNodeNormalMap")
        nmap.space = "TANGENT"
        tree.links.new(source, nmap.inputs["Color"])
        source = nmap.outputs["Normal"]
    for link in list(socket.links):
        tree.links.remove(link)
    tree.links.new(source, socket)


def _bake_uv_image_map(mat, principled, out_dir, kind):
    """Evaluate a UV-only texture graph on a quad, preserving every mesh UV.

    This is deliberately before the object bake: colour correction, layering
    and scalar math are image operations, so packing and resampling the scene's
    UV islands only loses detail and creates seams.
    """
    normal = kind == "normal"
    socket = principled.inputs["Normal" if normal else "Roughness" if kind == "roughness" else
                               "Alpha" if kind == "alpha" else
                               "Emission Color" if kind == "emission" else "Base Color"]
    users = [ob for ob in bpy.data.objects if ob.type == "MESH" and
             any(slot.material == mat for slot in ob.material_slots)]
    # A previous object bake may already have replaced TEXCOORD_0 with an atlas.
    # A source-UV image attached after that would be sampled through the atlas UV,
    # which produced the sofa's displaced normal/roughness detail. Keep every
    # later map in the same atlas space instead.
    if any(any(uv.active_render and uv.name.startswith("STRELKA_BAKE_")
               for uv in ob.data.uv_layers) for ob in users):
        return None
    if not _uv_image_bake_safe(mat, socket):
        return None

    images = _source_images(socket)
    width = max((int(image.size[0]) for image in images), default=0)
    height = max((int(image.size[1]) for image in images), default=0)
    if width <= 0 or height <= 0:
        return None

    suffix = "normal" if normal else "roughness" if kind == "roughness" else "alpha" if kind == "alpha" else \
             "emission" if kind == "emission" else "basecolor"
    name = "%s_uv_%s" % (mat.name.replace(" ", "_"), suffix)
    path = os.path.join(out_dir, name + ".png")
    source_mtime = max(os.path.getmtime(bpy.data.filepath), os.path.getmtime(__file__))
    for image in images:
        source_path = bpy.path.abspath(image.filepath)
        if source_path and os.path.exists(source_path):
            source_mtime = max(source_mtime, os.path.getmtime(source_path))
    non_color = normal or kind in {"roughness", "alpha"}
    if os.path.exists(path) and os.path.getmtime(path) >= source_mtime:
        image = bpy.data.images.load(path, check_existing=True)
        if tuple(image.size) == (width, height):
            image.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
            target = mat.node_tree.nodes.new("ShaderNodeTexImage")
            target.image = image
            _connect_baked_map(mat.node_tree, principled, target, kind)
            return name
        bpy.data.images.remove(image)

    tree = mat.node_tree
    target = tree.nodes.new("ShaderNodeTexImage")
    image = bpy.data.images.new(name, width=width, height=height, alpha=False)
    image.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    target.image = image
    for node in tree.nodes:
        node.select = False
    target.select = True
    tree.nodes.active = target

    mesh = bpy.data.meshes.new(name + "_quad")
    mesh.from_pydata([(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
                      (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)], [], [(0, 1, 2, 3)])
    uv_names = {"UVMap"}
    for node_tree in _walk_node_trees(tree):
        uv_names.update(node.uv_map for node in node_tree.nodes
                        if node.type == "UVMAP" and node.uv_map)
    coords = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for uv_name in uv_names:
        layer = mesh.uv_layers.new(name=uv_name)
        for loop in mesh.loops:
            layer.data[loop.index].uv = coords[loop.vertex_index]
        layer.active_render = True
        mesh.uv_layers.active = layer
    quad = bpy.data.objects.new(name + "_quad", mesh)
    bpy.context.scene.collection.objects.link(quad)
    mesh.materials.append(mat)

    output = next((node for node in tree.nodes
                   if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
    original_surface = output.inputs["Surface"].links[0].from_socket if output and \
        output.inputs["Surface"].links else None
    probe = None
    if not normal:
        probe = tree.nodes.new("ShaderNodeEmission")
        if socket.links:
            tree.links.new(socket.links[0].from_socket, probe.inputs["Color"])
        else:
            value = socket.default_value
            probe.inputs["Color"].default_value = value if hasattr(value, "__len__") else \
                (value, value, value, 1.0)
        tree.links.new(probe.outputs["Emission"], output.inputs["Surface"])

    scene, view = bpy.context.scene, bpy.context.view_layer
    engine, samples = scene.render.engine, scene.cycles.samples
    selected = list(bpy.context.selected_objects)
    active = view.objects.active
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.bake.margin = 0
    if normal:
        scene.render.bake.normal_space = "TANGENT"
    for ob in selected:
        ob.select_set(False)
    quad.select_set(True)
    view.objects.active = quad
    target.select = True
    tree.nodes.active = target
    try:
        bpy.ops.object.bake(type="NORMAL" if normal else "EMIT")
    except Exception:
        tree.nodes.remove(target)
        bpy.data.images.remove(image)
        return None
    finally:
        if probe is not None and original_surface is not None:
            tree.links.new(original_surface, output.inputs["Surface"])
        if probe is not None:
            tree.nodes.remove(probe)
        bpy.data.objects.remove(quad, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        scene.render.engine, scene.cycles.samples = engine, samples
        for ob in selected:
            ob.select_set(True)
        view.objects.active = active

    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    _connect_baked_map(tree, principled, target, kind)
    target.select = False
    tree.nodes.active = None
    return name


def _prepare_bake_uv(mat, users):
    """Pack tiled source UVs for a baked texture, preserving the source UV."""
    signatures = {tuple(slot.material.name if slot.material else "" for slot in ob.material_slots)
                  for ob in users}
    if len(signatures) != 1:
        return None
    signature = "\0".join(next(iter(signatures))).encode("utf-8")
    suffix = "%08x" % (zlib.crc32(signature) & 0xffffffff)
    name = "STRELKA_BAKE_" + suffix
    source_key = "strelka_source_uv_" + suffix
    if all(ob.data.uv_layers.get(name) and source_key in ob.data for ob in users):
        for ob in users:
            ob.data.uv_layers.get(name).active_render = True
            ob.data.uv_layers.active = ob.data.uv_layers.get(name)
        return users[0].data[source_key]
    source_layers = [next((u for u in ob.data.uv_layers if u.active_render),
                          ob.data.uv_layers[0]) for ob in users]
    source_names = [u.name for u in source_layers]
    if len(set(source_names)) != 1:
        return None
    if all(all(0.0 <= p.uv.x <= 1.0 and 0.0 <= p.uv.y <= 1.0 for p in u.data)
           for u in source_layers):
        return None

    for ob, source_name in zip(users, source_names):
        ob.data[source_key] = source_name
        packed = ob.data.uv_layers.get(name) or ob.data.uv_layers.new(name=name, do_init=True)
        packed.active_render = True
        ob.data.uv_layers.active = packed

    for ob in bpy.data.objects:
        ob.select_set(False)
    for ob in users:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = users[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=False, margin=0.002)
    bpy.ops.object.mode_set(mode="OBJECT")
    return source_names[0]


def _pin_source_uv(tree, uv_name, seen=None):
    """Make nested Texture Coordinate nodes keep using the pre-bake UV set."""
    seen = set() if seen is None else seen
    if tree in seen:
        return []
    seen.add(tree)
    changes = []
    for node in list(tree.nodes):
        if node.type == "GROUP" and node.node_tree is not None:
            changes.extend(_pin_source_uv(node.node_tree, uv_name, seen))
        if (node.type == "TEX_IMAGE" and node.image is not None and
                "_baked_" not in node.image.name and not node.inputs["Vector"].links):
            uv = tree.nodes.new("ShaderNodeUVMap")
            uv.uv_map = uv_name
            tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
            changes.append((tree, uv, None, [node.inputs["Vector"]]))
        if node.type != "TEX_COORD":
            continue
        links = list(node.outputs["UV"].links)
        if not links:
            continue
        destinations = [link.to_socket for link in links]
        uv = tree.nodes.new("ShaderNodeUVMap")
        uv.uv_map = uv_name
        for destination in destinations:
            tree.links.new(uv.outputs["UV"], destination)
        changes.append((tree, uv, node, destinations))
    return changes


def _unpin_source_uv(changes):
    for tree, uv, texcoord, destinations in changes:
        if texcoord is not None:
            for destination in destinations:
                tree.links.new(texcoord.outputs["UV"], destination)
        tree.nodes.remove(uv)


def _isolate_bake_material(mat, users):
    """Make every baked face use mat; Blender otherwise targets every slot."""
    state = []
    seen = set()
    for ob in users:
        key = ob.data.as_pointer()
        if key in seen:
            continue
        seen.add(key)
        materials = list(ob.data.materials)
        indices = [poly.material_index for poly in ob.data.polygons]
        ob.data.materials.clear()
        ob.data.materials.append(mat)
        for poly in ob.data.polygons:
            poly.material_index = 0
        state.append((ob.data, materials, indices))
    return state


def _restore_bake_materials(state):
    for mesh, materials, indices in state:
        mesh.materials.clear()
        for mat in materials:
            mesh.materials.append(mat)
        for poly, index in zip(mesh.polygons, indices):
            poly.material_index = index


def _ray_switch_principled(mat):
    """Principled used for camera/diffuse rays inside a renderer-conversion group."""
    groups = [n for n in mat.node_tree.nodes if n.type == "GROUP" and n.node_tree is not None]
    group = next((n for n in groups if "direct" in n.node_tree.name.lower()), None)
    if group is None:
        group = next((n for n in groups if "gi" in n.node_tree.name.lower()), None)
    if group is None:
        group = next((g for g in groups
                      if any(n.type == "BSDF_PRINCIPLED" for n in g.node_tree.nodes)), None)
    if group is None:
        return None
    return next((n for n in group.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def _bake_material_output(mat, out_dir, kind, size=2048):
    """Bake a ray-switched material whose Principled node lives inside a group."""
    users = [ob for ob in bpy.data.objects
             if ob.type == "MESH" and ob.data.uv_layers and
             any(s.material == mat for s in ob.material_slots)]
    if not users:
        return None
    source_uv = _prepare_bake_uv(mat, users)
    name = "%s_baked_%s" % (mat.name.replace(" ", "_"), kind)
    path = os.path.join(out_dir, name + ".png")
    normal = kind == "normal"
    roughness = kind == "roughness"
    tree = mat.node_tree
    target = tree.nodes.new("ShaderNodeTexImage")
    if os.path.exists(path) and os.path.getmtime(path) >= max(os.path.getmtime(bpy.data.filepath),
                                                               os.path.getmtime(__file__)):
        target.image = bpy.data.images.load(path, check_existing=True)
        target.image.colorspace_settings.name = "Non-Color" if normal or roughness else "sRGB"
        if source_uv:
            _pin_source_uv(tree, source_uv)
        return target

    img = bpy.data.images.new(name, width=size, height=size, alpha=False, float_buffer=False)
    img.colorspace_settings.name = "Non-Color" if normal or roughness else "sRGB"
    target.image = img
    for node in tree.nodes:
        node.select = False
    target.select = True
    tree.nodes.active = target
    scene, view = bpy.context.scene, bpy.context.view_layer
    engine, samples = scene.render.engine, scene.cycles.samples
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.margin = 8
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    for ob in bpy.data.objects:
        ob.select_set(False)
    for ob in users:
        ob.select_set(True)
        ob.active_material_index = next(i for i, slot in enumerate(ob.material_slots) if slot.material == mat)
    view.objects.active = users[0]
    pinned = _pin_source_uv(tree, source_uv) if source_uv else []
    target.select = True
    tree.nodes.active = target
    material_state = _isolate_bake_material(mat, users)
    try:
        bpy.ops.object.bake(type="NORMAL" if normal else "ROUGHNESS" if roughness else "DIFFUSE")
    except Exception:
        _unpin_source_uv(pinned)
        tree.nodes.remove(target)
        bpy.data.images.remove(img)
        scene.render.engine, scene.cycles.samples = engine, samples
        return None
    finally:
        _restore_bake_materials(material_state)
    # Keep source images explicitly bound to the authored UV set. The packed UV
    # remains active for baked maps and is what glTF exports as TEXCOORD_0.
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    scene.render.engine, scene.cycles.samples = engine, samples
    target.select = False
    tree.nodes.active = None
    return target


def _bake_principled_map(mat, principled, out_dir, kind, size=2048):
    """Bake a Principled input that glTF cannot express into one texture.

    The floor's relief is a Bump node fed a Brick texture -- the parquet's plank
    seams -- mixed with the laminate photo. glTF has neither a bump node nor a
    procedural brick, and Blender's exporter resolves that by writing the *height
    source* into normalTexture: the laminate colour jpg, bound as a tangent-space
    normal map. Decoded as 2*rgb-1 a warm brown becomes a normal tilted into the
    surface, which is why the floor rendered near black with no parquet on it at
    all, the two complaints having one cause.

    Baking states the same relief in the one form glTF does carry. The seams
    arrive because they are in the bake, not because the format learned about
    bricks.
    """
    users = [ob for ob in bpy.data.objects
             if ob.type == "MESH" and any(s.material == mat for s in ob.material_slots)]
    users = [ob for ob in users if ob.data.uv_layers]
    if not users:
        return None

    unique_meshes = {ob.data.as_pointer(): ob.data for ob in users}
    if (size == 2048 and len(unique_meshes) >= 5 and
            sum(len(mesh.polygons) for mesh in unique_meshes.values()) >= 250000):
        size = 4096

    source_uv = _prepare_bake_uv(mat, users)

    normal = kind == "normal"
    emission = kind == "emission"
    roughness = kind == "roughness"
    alpha = kind == "alpha"
    name = "%s_baked_%s" % (mat.name.replace(" ", "_"),
                             "normal" if normal else "roughness" if roughness else
                             "emission" if emission else "alpha" if alpha else "basecolor")
    path = os.path.join(out_dir, name + ".png")
    source_mtime = max(os.path.getmtime(bpy.data.filepath), os.path.getmtime(__file__))
    if os.path.exists(path) and os.path.getmtime(path) >= source_mtime:
        img = bpy.data.images.load(path, check_existing=True)
        if img.size[0] >= size and img.size[1] >= size:
            img.colorspace_settings.name = "Non-Color" if normal or roughness or alpha else "sRGB"
            target = mat.node_tree.nodes.new("ShaderNodeTexImage")
            target.image = img
            if source_uv:
                _pin_source_uv(mat.node_tree, source_uv)
            _connect_baked_map(mat.node_tree, principled, target, kind)
            return name
        bpy.data.images.remove(img)

    img = bpy.data.images.new(name, width=size, height=size, alpha=False, float_buffer=False)
    img.colorspace_settings.name = "Non-Color" if normal or roughness or alpha else "sRGB"

    tree = mat.node_tree
    target = tree.nodes.new("ShaderNodeTexImage")
    target.image = img
    for node in tree.nodes:
        node.select = False
    target.select = True
    tree.nodes.active = target

    scene = bpy.context.scene
    engine, view = scene.render.engine, bpy.context.view_layer
    scene.render.engine = "CYCLES"
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.margin = 8
    socket_bake = roughness or alpha or (not normal and not emission)
    output = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    original_surface = output.inputs["Surface"].links[0].from_socket if output and output.inputs["Surface"].links else None
    probe = None
    if socket_bake and output is not None:
        probe = tree.nodes.new("ShaderNodeEmission")
        source_socket = principled.inputs["Roughness" if roughness else "Alpha" if alpha else "Base Color"]
        if source_socket.links:
            tree.links.new(source_socket.links[0].from_socket, probe.inputs["Color"])
        else:
            value = source_socket.default_value
            probe.inputs["Color"].default_value = value if hasattr(value, "__len__") else (value, value, value, 1.0)
        tree.links.new(probe.outputs["Emission"], output.inputs["Surface"])
    scene.cycles.bake_type = "NORMAL" if normal else "EMIT"
    if normal:
        scene.render.bake.normal_space = "TANGENT"
    elif not emission and not socket_bake:
        scene.render.bake.use_pass_direct = False
        scene.render.bake.use_pass_indirect = False
        scene.render.bake.use_pass_color = True
    # Neither a tangent normal nor the diffuse colour pass contains lighting, so
    # one sample gets the exact result. Inheriting the scene's 1024 only burns CPU.
    samples = scene.cycles.samples
    scene.cycles.samples = 1

    for ob in bpy.data.objects:
        ob.select_set(False)
    for ob in users:
        ob.select_set(True)
        ob.active_material_index = next(i for i, slot in enumerate(ob.material_slots)
                                        if slot.material == mat)
        # The bake writes through the active layer, while the exporter numbers by
        # position and Strelka reads TEXCOORD_0; promote_render_uv() moves the
        # render layer there, so the bake has to agree with that one.
        uvs = ob.data.uv_layers
        chosen = next((l for l in uvs if l.active_render), uvs[0])
        uvs.active = chosen
    view.objects.active = users[0]

    strength = principled.inputs.get("Emission Strength") if emission else None
    authored_strength = float(strength.default_value) if strength is not None else 1.0
    if strength is not None:
        strength.default_value = 1.0
    pinned = _pin_source_uv(tree, source_uv) if source_uv else []
    target.select = True
    tree.nodes.active = target
    material_state = _isolate_bake_material(mat, users)
    try:
        bpy.ops.object.bake(type="NORMAL" if normal else "EMIT")
    except Exception as exc:
        _unpin_source_uv(pinned)
        if strength is not None:
            strength.default_value = authored_strength
        tree.nodes.remove(target)
        bpy.data.images.remove(img)
        if probe is not None:
            if original_surface is not None:
                tree.links.new(original_surface, output.inputs["Surface"])
            tree.nodes.remove(probe)
        scene.render.engine, scene.cycles.samples = engine, samples
        return "bake failed: %s" % exc
    finally:
        _restore_bake_materials(material_state)

    # Keep the explicit authored-UV bindings; only the baked target uses the
    # packed active UV set exported as TEXCOORD_0.
    if strength is not None:
        strength.default_value = authored_strength
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    scene.render.engine, scene.cycles.samples = engine, samples

    if probe is not None:
        if original_surface is not None:
            tree.links.new(original_surface, output.inputs["Surface"])
        tree.nodes.remove(probe)

    _connect_baked_map(tree, principled, target, kind)
    # Objects commonly have several materials. Leaving this image active lets a
    # later bake for a neighbouring slot overwrite it (base colour became the
    # characteristic [0.5, 0.5, 1] normal-map blue in the interior scene).
    target.select = False
    tree.nodes.active = None
    return name


def _bake_normal_map(mat, principled, out_dir, size=2048):
    return _bake_uv_image_map(mat, principled, out_dir, "normal") or \
        _bake_principled_map(mat, principled, out_dir, "normal", size)


def _bake_roughness_map(mat, principled, out_dir, size=2048):
    roughness = principled.inputs.get("Roughness")
    if roughness is None or not roughness.links or roughness.links[0].from_node.type == "TEX_IMAGE":
        return None
    return _bake_uv_image_map(mat, principled, out_dir, "roughness") or \
        _bake_principled_map(mat, principled, out_dir, "roughness", size)


def _bake_transformed_base_color(mat, principled, out_dir, size=2048):
    base = principled.inputs.get("Base Color")
    if base is None or not base.links or base.links[0].from_node.type == "TEX_IMAGE":
        return None
    return _bake_uv_image_map(mat, principled, out_dir, "basecolor") or \
        _bake_principled_map(mat, principled, out_dir, "basecolor", size)


def _bake_alpha_map(mat, principled, out_dir, size=2048):
    alpha = principled.inputs.get("Alpha")
    if alpha is None or not alpha.links or alpha.links[0].from_node.type == "TEX_IMAGE":
        return None
    return _bake_uv_image_map(mat, principled, out_dir, "alpha") or \
        _bake_principled_map(mat, principled, out_dir, "alpha", size)


def _bake_emission_color(mat, principled, out_dir, size=2048):
    return _bake_uv_image_map(mat, principled, out_dir, "emission") or \
        _bake_principled_map(mat, principled, out_dir, "emission", size)


def _gradient_value(kind, p):
    """Blender's Gradient Texture, per type, over an (N, 3) array of points."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    if kind == "LINEAR":
        return x
    if kind == "QUADRATIC":
        r = np.maximum(x, 0.0)
        return r * r
    if kind == "EASING":
        r = np.clip(x, 0.0, 1.0)
        return r * r * (3.0 - 2.0 * r)
    if kind == "DIAGONAL":
        return (x + y) * 0.5
    if kind == "RADIAL":
        return np.arctan2(y, x) / (2.0 * math.pi) + 0.5
    r = np.maximum(1.0 - np.sqrt(x * x + y * y + z * z), 0.0)
    if kind == "SPHERICAL":
        return r
    if kind == "QUADRATIC_SPHERE":
        return r * r
    return None


def _object_space_samples(mat, per_triangle=4096):
    """Points on the surfaces that wear this material, in object space, with the
    area weight of the triangle each came from.

    A texture driving an emission has no single value; it has an average over the
    emitter, and the average is over *area*, since that is what the radiance is
    integrated against. Sampling the mesh gets that for a shape of any kind,
    where a bounding box would only get it for a rectangle.
    """
    # Seeded, because an exporter that writes a different number each run turns
    # every later comparison into a question about which run it came from.
    rng = np.random.default_rng(0x5721EA)
    pts = []
    wts = []
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        slots = [i for i, s in enumerate(ob.material_slots) if s.material == mat]
        if not slots:
            continue
        me = ob.data
        me.calc_loop_triangles()
        verts = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
        for tri in me.loop_triangles:
            if tri.material_index not in slots:
                continue
            a, b, c = verts[list(tri.vertices)]
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            if area <= 0.0:
                continue
            u = rng.random(per_triangle)
            v = rng.random(per_triangle)
            fold = u + v > 1.0
            u[fold], v[fold] = 1.0 - u[fold], 1.0 - v[fold]
            pts.append(a + np.outer(u, b - a) + np.outer(v, c - a))
            wts.append(np.full(per_triangle, area / per_triangle))
    if not pts:
        return None, None
    return np.concatenate(pts), np.concatenate(wts)


def _apply_mapping(node, p):
    """A Mapping node in Texture mode, which is the inverse of the point one:
    the coordinate is un-located, un-rotated and un-scaled before the texture
    reads it. Any other mode returns None rather than a wrong transform."""
    if node.vector_type != "TEXTURE":
        return None
    if any(node.inputs[k].links for k in ("Location", "Rotation", "Scale")):
        return None
    loc = np.array(node.inputs["Location"].default_value[:], dtype=np.float64)
    rot = np.array(node.inputs["Rotation"].default_value[:], dtype=np.float64)
    scale = np.array(node.inputs["Scale"].default_value[:], dtype=np.float64)
    q = p - loc
    if np.any(np.abs(rot) > 1e-9):
        cx, cy, cz = np.cos(rot)
        sx, sy, sz = np.sin(rot)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
        q = q @ (rz @ ry @ rx)
    safe = np.where(np.abs(scale) < 1e-9, 1.0, scale)
    return q / safe


def _procedural_mean(socket, mat, depth=0):
    """The area average of a scalar socket driven by a procedural texture.

    Only the chain this scene actually uses is walked -- a Gradient Texture
    behind an optional Mapping, optionally scaled by Math -- and anything else
    returns None so the caller reports it instead of averaging the wrong thing.
    """
    if not socket.links or depth > 4:
        return None
    node = socket.links[0].from_node
    if node.type == "MATH":
        # One operand constant, the other the texture: the mean is linear in it
        # for multiply and add, which are the two that appear on a strength.
        a, b = node.inputs[0], node.inputs[1]
        if node.operation not in {"MULTIPLY", "ADD"}:
            return None
        for driven, other in ((a, b), (b, a)):
            if driven.links and not other.links:
                inner = _procedural_mean(driven, mat, depth + 1)
                if inner is None:
                    return None
                k = float(other.default_value)
                return inner * k if node.operation == "MULTIPLY" else inner + k
        return None
    if node.type != "TEX_GRADIENT":
        return None
    vec = node.inputs["Vector"]
    pts, wts = _object_space_samples(mat)
    if pts is None:
        return None
    if vec.links:
        src = vec.links[0].from_node
        if src.type == "MAPPING":
            pts = _apply_mapping(src, pts)
            if pts is None:
                return None
            up = src.inputs["Vector"]
            if up.links and not (up.links[0].from_node.type == "TEX_COORD"
                                 and up.links[0].from_socket.name == "Object"):
                return None
        elif not (src.type == "TEX_COORD" and vec.links[0].from_socket.name == "Object"):
            return None
    vals = _gradient_value(node.gradient_type, pts)
    if vals is None:
        return None
    return float(np.average(vals, weights=wts))


def _resolve_emission_strength(mat, emission):
    """Collapse a textured Emission Strength to the average it emits.

    The softbox panels are lit by a quadratic-sphere gradient times 10: bright in
    the middle, dark at the corners, averaging 4.01 over the panel. Blender's
    glTF exporter reads the socket as if the gradient were 1 and writes 10, so
    the three panels -- 62% of this scene's light, measured by rendering it
    without them -- come across 2.5x too bright, which is most of why the
    interior sat above the reference while its walls sat below.
    """
    if "Strength" not in emission.inputs:
        return None
    sock = emission.inputs["Strength"]
    if not sock.links:
        return None
    mean = _procedural_mean(sock, mat)
    if mean is None:
        return None
    tree = mat.node_tree
    for link in list(sock.links):
        tree.links.remove(link)
    sock.default_value = mean
    return mean


def flatten(out_dir):
    """Rewrite every material that needs it.

    Returns the report and extension payloads the exporter cannot derive on its
    own, keyed by material name.
    """
    report = []
    translucency = {}
    volumes = {}
    subsurface = {}
    tints = {}
    shadow_transparent = set()
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        found = _surface(mat)
        if found is None:
            continue
        surface, out_node = found

        volume = _volume_absorption(mat)
        if volume is not None:
            volumes[mat.name] = volume

        if surface.type == "BSDF_PRINCIPLED":
            # Already the shape the exporter wants -- but its inputs may still be
            # driven by nodes it cannot read.
            emission_gain = _flatten_emission_multiplier(surface)
            baked_emission = _bake_emission_color(mat, surface, out_dir) if emission_gain is not None else None
            baked_base = _bake_transformed_base_color(mat, surface, out_dir)
            baked_alpha = _bake_alpha_map(mat, surface, out_dir)
            baked_roughness = _bake_roughness_map(mat, surface, out_dir)
            sss = _subsurface(surface)
            if sss is not None:
                subsurface[mat.name] = sss
            tint = _base_color_tint(surface)
            if tint is not None:
                tints[mat.name] = tint
            baked = None
            if not _normal_is_plain_map(surface.inputs["Normal"]):
                baked = _bake_normal_map(mat, surface, out_dir)
                if baked is not None:
                    report.append((mat.name, "normal baked -> %s" % baked))
            if baked_base is not None:
                report.append((mat.name, "transformed base colour baked -> %s" % baked_base))
            if baked_alpha is not None:
                report.append((mat.name, "alpha baked -> %s" % baked_alpha))
            if baked_roughness is not None:
                report.append((mat.name, "roughness baked -> %s" % baked_roughness))
            if emission_gain is not None:
                report.append((mat.name, "emission multiplier %.3f moved to strength" % emission_gain))
            if baked_emission is not None:
                report.append((mat.name, "emission colour baked -> %s" % baked_emission))
            reduced = _reduce_inputs(surface)
            if reduced or volume is not None or sss is not None:
                note = "inputs reduced: %s" % ", ".join(reduced) if reduced else "volume only"
                if volume is not None:
                    note += "; volume %s over %.2f" % (
                        "".join("%.2f " % c for c in volume[0]).strip(), volume[1])
                if sss is not None:
                    note += "; subsurface %.2f radius %s" % (
                        sss["weight"], "".join("%.3f " % r for r in sss["radius"]).strip())
                report.append((mat.name, note))
            continue

        if surface.type == "EMISSION":
            # A pure emitter is already what glTF wants; only its strength can be
            # driven by something the exporter reads as a single number.
            mean = _resolve_emission_strength(mat, surface)
            report.append((mat.name, "emission strength averaged to %.3f" % mean if mean is not None
                           else "emission strength left as authored"))
            continue

        shadow_principled = _shadow_transparent_principled(surface)
        principled = shadow_principled or _find_principled(surface)
        if principled is None:
            nested = _ray_switch_principled(mat)
            if nested is not None:
                baked_base = _bake_material_output(mat, out_dir, "basecolor")
                baked_roughness = _bake_material_output(mat, out_dir, "roughness")
                baked_normal = _bake_material_output(mat, out_dir, "normal")
                proxy = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
                proxy.inputs["Metallic"].default_value = nested.inputs["Metallic"].default_value
                proxy.inputs["Roughness"].default_value = _mean_scalar(nested.inputs["Roughness"])
                proxy.inputs["IOR"].default_value = nested.inputs["IOR"].default_value
                proxy.inputs["Specular IOR Level"].default_value = nested.inputs["Specular IOR Level"].default_value
                if baked_base is not None:
                    mat.node_tree.links.new(baked_base.outputs["Color"], proxy.inputs["Base Color"])
                if baked_roughness is not None:
                    mat.node_tree.links.new(baked_roughness.outputs["Color"], proxy.inputs["Roughness"])
                if baked_normal is not None:
                    nmap = mat.node_tree.nodes.new("ShaderNodeNormalMap")
                    mat.node_tree.links.new(baked_normal.outputs["Color"], nmap.inputs["Color"])
                    mat.node_tree.links.new(nmap.outputs["Normal"], proxy.inputs["Normal"])
                mat.node_tree.links.new(proxy.outputs["BSDF"], out_node.inputs["Surface"])
                report.append((mat.name, "grouped material baked"))
                continue
            report.append((mat.name, "no Principled BSDF under %s" % surface.type))
            continue

        nt = mat.node_tree

        baked_base = _bake_transformed_base_color(mat, principled, out_dir)
        baked_alpha = _bake_alpha_map(mat, principled, out_dir)
        baked_roughness = _bake_roughness_map(mat, principled, out_dir)
        sss = _subsurface(principled)
        if sss is not None:
            subsurface[mat.name] = sss
        tint = _base_color_tint(principled)
        if tint is not None:
            tints[mat.name] = tint

        # Foliage translucency, before the graph is rewritten and the node is
        # orphaned. Blender writes it as a Translucent BSDF added alongside the
        # Principled; glTF has KHR_materials_diffuse_transmission, and the
        # exporter knows nothing about either.
        translucent = _find_translucent(surface)
        if translucent is not None:
            colour = _mean_colour(translucent.inputs["Color"]) or [1.0, 1.0, 1.0]
            # 0.5, because the graph adds the two lobes at full strength and the
            # glTF model splits one lobe between them. Half each is the
            # energy-conserving reading of "reflect and transmit equally", which
            # is what the author expressed; carrying the addition across
            # literally would make a canopy brighter than the sky behind it.
            # A judgement call, and the only one available without a weight in
            # the graph to read.
            translucency[mat.name] = (0.5, colour)

        alpha = None if shadow_principled is not None else _alpha_source(surface)
        note = "flattened"

        if shadow_principled is not None:
            shadow_transparent.add(mat.name)
            note = "shadow rays transparent"

        if alpha is not None:
            link, transparent_first = alpha
            # Mix Shader outputs (1 - Fac) * slot0 + Fac * slot1. A Transparent
            # in slot 0 therefore means opacity *is* the factor; in slot 1 it
            # means opacity is one minus it.
            #
            # fir_twig settles which way round that is without any guessing: its
            # Principled node already drives Alpha from the same texture that
            # feeds the mix, and its Transparent sits in slot 0. So slot 0 means
            # no inversion.
            #
            # Backwards, this does not look like an inverted mask. It looks like
            # the canopy is the wrong density, and the error arrives as light on
            # the ground far from anything to do with foliage.
            src = link.from_socket
            note = "alpha from the %s mix branch" % ("first" if transparent_first else "second")
            if not transparent_first:
                # Inverted in the image rather than with a Math node, because the
                # exporter does not evaluate the graph feeding Alpha -- it looks
                # for a texture and takes it whole. A SUBTRACT node in between is
                # silently skipped, and the mask exports at exactly the polarity
                # it was meant not to have.
                mask_img, _ = _image_of(link)
                if mask_img is None:
                    report.append((mat.name, "alpha needs inverting but is not a plain texture"))
                    continue
                inverted = _invert_image(mask_img, out_dir)
                tex = nt.nodes.new("ShaderNodeTexImage")
                tex.image = inverted
                for old in list(link.from_node.inputs):
                    if old.name == "Vector" and old.links:
                        nt.links.new(old.links[0].from_socket, tex.inputs["Vector"])
                src = tex.outputs["Color"]
                note += ", inverted into %s" % inverted.name
            nt.links.new(src, principled.inputs["Alpha"])
        else:
            note = "flattened, no transparency found"

        nt.links.new(principled.outputs["BSDF"], out_node.inputs["Surface"])
        reduced = _reduce_inputs(principled)
        if reduced:
            note += ", reduced " + ", ".join(reduced)
        if mat.name in translucency:
            note += ", translucency %.2f %s" % (
                translucency[mat.name][0],
                "".join("%.2f " % c for c in translucency[mat.name][1]).strip())
        if mat.name in subsurface:
            note += ", subsurface %.2f radius %s" % (
                subsurface[mat.name]["weight"],
                "".join("%.3f " % r for r in subsurface[mat.name]["radius"]).strip())
        if mat.name in tints:
            note += ", tint %s" % "".join("%.2f " % t for t in tints[mat.name]).strip()
        if baked_base is not None:
            note += ", transformed base colour baked -> %s" % baked_base
        if baked_alpha is not None:
            note += ", alpha baked -> %s" % baked_alpha
        if baked_roughness is not None:
            note += ", roughness baked -> %s" % baked_roughness
        report.append((mat.name, note))
    return report, translucency, volumes, subsurface, tints, shadow_transparent
