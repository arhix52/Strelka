#!/usr/bin/env python3
"""Express a glTF scene's materials as MaterialX OpenPBR.

    tools/gltf_to_mtlx.py scenes/iso_bathroom/iso_bathroom.gltf

Writes <stem>.mtlx beside the scene, which the loader picks up automatically
(gltfloader.cpp::loadMaterialXFromSidecar). Materials are matched by name, so no
<look> is needed: unlike the Open Chess Set, a glTF exported from a DCC names its
materials properly.

This mirrors openpbr_from_gltf.h, and that header is the authority -- it is what
the renderer uses for `render/material/model = openpbr`. Two implementations of
one mapping is a real cost, and it is paid for a reason: the renderer converts in
memory and produces no artefact, while a .mtlx is a file a person can read, diff
and hand to another renderer.

Which is also how the two are kept honest. Render the scene both ways --

    StrelkaCLI -c scene.toml                        # reads the .mtlx
    StrelkaCLI -c scene.toml  (material_model=openpbr, .mtlx moved aside)

-- and the images must agree. They come from the same BSDF fed by two
independent transcriptions of the same mapping, so a divergence is one of them
being wrong.

Only what glTF can state is emitted. Everything OpenPBR adds -- coat darkening,
fuzz as a microflake layer, dispersion, a subsurface radius per channel -- stays
at its specification default, because the glTF has no opinion about it. Editing
those afterwards is the point of having the file at all.
"""

import json
import math
import os
import sys


def ext(mat, name, key, fallback):
    e = mat.get("extensions", {}).get(name)
    if not isinstance(e, dict):
        return fallback
    v = e.get(key, fallback)
    return v


def tex_index(d):
    return d.get("index") if isinstance(d, dict) else None


def convert(mat, images, textures):
    """One glTF material -> (inputs, texture slots). See openpbr_from_gltf.h."""
    pbr = mat.get("pbrMetallicRoughness", {})
    base = pbr.get("baseColorFactor", [1, 1, 1, 1])
    inputs = {}
    maps = {}

    inputs["base_color"] = ("color3", base[:3])
    inputs["base_metalness"] = ("float", pbr.get("metallicFactor", 1.0))
    inputs["specular_roughness"] = ("float", pbr.get("roughnessFactor", 1.0))
    inputs["specular_ior"] = ("float", ext(mat, "KHR_materials_ior", "ior", 1.5))

    # The factor of two: gltfloader.cpp stores specularFactor halved, and OpenPBR
    # means the unhalved thing. Here we read the glTF directly, so no halving has
    # happened and none is undone -- the value goes across as written.
    inputs["specular_weight"] = ("float",
                                 min(1.0, ext(mat, "KHR_materials_specular", "specularFactor", 1.0)))
    sc = ext(mat, "KHR_materials_specular", "specularColorFactor", None)
    if sc:
        inputs["specular_color"] = ("color3", sc)

    tr = ext(mat, "KHR_materials_transmission", "transmissionFactor", 0.0)
    if tr:
        inputs["transmission_weight"] = ("float", tr)
        d = ext(mat, "KHR_materials_volume", "attenuationDistance", 0.0)
        if d and math.isfinite(d):
            inputs["transmission_depth"] = ("float", d)
            inputs["transmission_color"] = (
                "color3", ext(mat, "KHR_materials_volume", "attenuationColor", [1, 1, 1]))
        # Thin-walled only when KHR_materials_volume is actually there and says
        # so. Absence of the extension reads as *solid*, deliberately against a
        # plain reading of the spec -- gltfloader.cpp carries the same rule and
        # the note about the Blender export quirk behind it. Defaulting
        # thicknessFactor to 0 instead made every untouched glass thin-walled,
        # which is what put 4.3% between this file and the renderer's own
        # mapping the first time the two were compared.
        vol = mat.get("extensions", {}).get("KHR_materials_volume")
        if isinstance(vol, dict) and vol.get("thicknessFactor", 0.0) <= 0.0:
            inputs["geometry_thin_walled"] = ("boolean", True)

    cc = ext(mat, "KHR_materials_clearcoat", "clearcoatFactor", 0.0)
    if cc:
        inputs["coat_weight"] = ("float", cc)
        inputs["coat_roughness"] = ("float", ext(mat, "KHR_materials_clearcoat",
                                                 "clearcoatRoughnessFactor", 0.0))
        # Not in the extension; Blender writes it and the bathroom ceramics use it.
        inputs["coat_ior"] = ("float", ext(mat, "KHR_materials_clearcoat", "clearcoatIor", 1.5))

    sheen = ext(mat, "KHR_materials_sheen", "sheenColorFactor", None)
    if sheen:
        w = max(sheen)
        if w > 0.0:
            inputs["fuzz_weight"] = ("float", min(1.0, w))
            inputs["fuzz_color"] = ("color3", [c / w for c in sheen])
            inputs["fuzz_roughness"] = ("float",
                                        ext(mat, "KHR_materials_sheen", "sheenRoughnessFactor", 0.0))

    ir = ext(mat, "KHR_materials_iridescence", "iridescenceFactor", 0.0)
    if ir:
        inputs["thin_film_weight"] = ("float", ir)
        inputs["thin_film_ior"] = ("float", ext(mat, "KHR_materials_iridescence",
                                                "iridescenceIor", 1.3))
        inputs["thin_film_thickness"] = ("float", ext(mat, "KHR_materials_iridescence",
                                                      "iridescenceThicknessMaximum", 400.0))

    em = mat.get("emissiveFactor", [0, 0, 0])
    strength = ext(mat, "KHR_materials_emissive_strength", "emissiveStrength", 1.0)
    lum = (0.2126 * em[0] + 0.7152 * em[1] + 0.0722 * em[2]) * strength
    if lum > 0.0:
        inputs["emission_luminance"] = ("float", lum)
        inputs["emission_color"] = ("color3", [c * strength / lum for c in em])

    # STRELKA_materials_subsurface, the custom extension the V-Ray conversion
    # writes. Not glTF, but it is what nine of this scene's materials carry, and
    # openpbr_from_gltf.h maps it -- leaving it out here is a divergence, not a
    # simplification.
    ss = ext(mat, "STRELKA_materials_subsurface", "subsurfaceFactor", 0.0)
    if ss:
        inputs["subsurface_weight"] = ("float", min(1.0, ss))
        sr = ext(mat, "STRELKA_materials_subsurface", "scatterRadius", None)
        if sr:
            m = max(sr)
            if m > 0.0:
                inputs["subsurface_radius"] = ("float", m)
                inputs["subsurface_radius_scale"] = ("color3", [c / m for c in sr])
        scol = ext(mat, "STRELKA_materials_subsurface", "scatterColor", None)
        if scol:
            inputs["subsurface_color"] = ("color3", scol)
        g = ext(mat, "STRELKA_materials_subsurface", "anisotropy", 0.0)
        if g:
            inputs["subsurface_scatter_anisotropy"] = ("float", g)

    # KHR_materials_diffuse_transmission. openpbr_from_gltf.h reads its colour as
    # the subsurface tint, so leaving it out here is another divergence: nine of
    # this scene's materials carry it.
    dtc = ext(mat, "KHR_materials_diffuse_transmission", "diffuseTransmissionColorFactor", None)
    if dtc and "subsurface_color" not in inputs:
        inputs["subsurface_color"] = ("color3", dtc)

    # Opacity. Not a lobe -- the BSDF does not consume it -- but the block is
    # incomplete without it and the C++ mapping carries it.
    alpha = base[3] if len(base) > 3 else 1.0
    if alpha < 1.0:
        inputs["geometry_opacity"] = ("float", alpha)

    an = ext(mat, "KHR_materials_anisotropy", "anisotropyStrength", 0.0)
    if an:
        inputs["specular_roughness_anisotropy"] = ("float", abs(an))

    # Maps. glTF packs roughness and metalness into one image; OpenPBR names each
    # input separately, so one file lands in two slots and each shader reads the
    # channel it wants. Recorded here as-is rather than split, which would mean
    # writing new images.
    for slot, src in (("base_color", tex_index(pbr.get("baseColorTexture"))),
                      ("specular_roughness", tex_index(pbr.get("metallicRoughnessTexture"))),
                      ("base_metalness", tex_index(pbr.get("metallicRoughnessTexture"))),
                      ("geometry_normal", tex_index(mat.get("normalTexture"))),
                      ("emission_color", tex_index(mat.get("emissiveTexture")))):
        if src is None:
            continue
        img = textures[src].get("source")
        if img is None:
            continue
        uri = images[img].get("uri")
        if uri:
            maps[slot] = uri
    return inputs, maps


def fmt(kind, value):
    if kind == "boolean":
        return "true" if value else "false"
    if kind == "color3":
        return ", ".join(f"{float(c):.6g}" for c in value[:3])
    return f"{float(value):.6g}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = sys.argv[1]
    doc = json.load(open(src))
    images = doc.get("images", [])
    textures = doc.get("textures", [])
    mats = doc.get("materials", [])

    out = [f'<?xml version="1.0"?>',
           f'<materialx version="1.39" colorspace="lin_rec709">',
           f'  <!-- Generated by tools/gltf_to_mtlx.py from {os.path.basename(src)}.',
           f'       Mirrors openpbr_from_gltf.h; see that header for what each',
           f'       mapping is exact about and what it approximates. -->']

    textured = 0
    for m in mats:
        name = m.get("name")
        if not name:
            continue
        inputs, maps = convert(m, images, textures)
        if maps:
            textured += 1
        # Only the *internal* node names are sanitised. The surfacematerial keeps
        # the glTF name exactly, because that is what the loader matches on: an
        # exporter that wrote "Material #385" has to be findable under it, and
        # replacing the space here silently unbound four of this scene's
        # sixty-seven materials.
        safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
        if maps:
            out.append(f'  <nodegraph name="NG_{safe}">')
            for slot, uri in sorted(maps.items()):
                t = "color3" if slot in ("base_color", "emission_color") else \
                    ("vector3" if slot == "geometry_normal" else "float")
                node = f"img_{safe}_{slot}"
                cs = ' colorspace="srgb_texture"' if t == "color3" else ""
                out.append(f'    <image name="{node}" type="{t}">')
                out.append(f'      <input name="file" type="filename" value="{uri}"{cs} />')
                out.append(f'    </image>')
                if slot == "geometry_normal":
                    out.append(f'    <normalmap name="nm_{safe}" type="vector3">')
                    out.append(f'      <input name="in" type="vector3" nodename="{node}" />')
                    out.append(f'    </normalmap>')
                    node = f"nm_{safe}"
                out.append(f'    <output name="{slot}_output" type="{t}" nodename="{node}" />')
            out.append(f'  </nodegraph>')

        out.append(f'  <surfacematerial name="{name}" type="material">')
        out.append(f'    <input name="surfaceshader" type="surfaceshader" nodename="S_{safe}" />')
        out.append(f'  </surfacematerial>')
        out.append(f'  <open_pbr_surface name="S_{safe}" type="surfaceshader">')
        for k, (t, v) in sorted(inputs.items()):
            if k in maps:
                continue
            out.append(f'    <input name="{k}" type="{t}" value="{fmt(t, v)}" />')
        for slot in sorted(maps):
            t = "color3" if slot in ("base_color", "emission_color") else \
                ("vector3" if slot == "geometry_normal" else "float")
            out.append(f'    <input name="{slot}" type="{t}" nodegraph="NG_{safe}" '
                       f'output="{slot}_output" />')
        out.append(f'  </open_pbr_surface>')
    out.append('</materialx>')

    dst = os.path.splitext(src)[0] + ".mtlx"
    open(dst, "w").write("\n".join(out) + "\n")
    print(f"{len(mats)} material(s), {textured} with maps -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
