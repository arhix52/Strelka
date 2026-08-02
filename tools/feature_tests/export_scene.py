#!/usr/bin/env python3
"""
Export a production .blend for Strelka.

    blender -b <file.blend> -P tools/feature_tests/export_scene.py -- --out DIR [--frame N]

Writes <DIR>/<name>.gltf (+ .bin + textures) and <DIR>/<name>_light.json.

Two things glTF cannot carry that Strelka needs, both handled by sidecars in the
same spirit as the existing light JSON:

  - Area lights. KHR_lights_punctual has point, spot and directional and nothing
    else, so Blender's area lamps -- which is what most interiors are lit by --
    would simply vanish. They go in the sidecar, and because the sidecar takes
    precedence in the loader, the punctual lights are written there too rather
    than split across two mechanisms.
  - Curves. Not written yet; see the note at the bottom.

Modifier evaluation is left to the glTF exporter's export_apply, which uses the
render-resolution depsgraph, so Subdivision comes through at its render level and
Geometry Nodes are realised.
"""

import bpy
import json
import math
import os
import sys

from mathutils import Vector


def gltf_pos(v):
    """Blender is Z-up, glTF is Y-up; Strelka sees the exported world."""
    return [v.x, v.z, -v.y]


def collect_lights(depsgraph):
    """Every light in the evaluated scene, in the sidecar's schema.

    Intensities are radiometric here, which is what the sidecar means by
    default: area lights convert watts to radiance as P/(A*pi), and point and
    spot convert to W/sr as P/(4*pi) -- the same conversion the glTF path does
    after dividing out 683 lm/W.
    """
    out = []
    for ob in depsgraph.objects:
        if ob.type != "LIGHT":
            continue
        L = ob.data
        m = ob.matrix_world
        pos = gltf_pos(m.translation)
        # -Z is the emission axis for every Blender lamp that has one.
        d = (m.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        entry = {
            "name": ob.name,
            "position": pos,
            "color": list(L.color),
            "direction": [d.x, d.z, -d.y],
        }
        if L.type == "AREA":
            if L.shape in {"SQUARE", "RECTANGLE"}:
                w = L.size
                h = L.size_y if L.shape == "RECTANGLE" else L.size
                entry.update(type="rect", width=w, height=h,
                             intensity=L.energy / (max(w * h, 1e-9) * math.pi))
            else:  # DISK / ELLIPSE
                r = L.size * 0.5
                area = math.pi * r * r
                entry.update(type="disc", radius=r,
                             intensity=L.energy / (max(area, 1e-9) * math.pi))
        elif L.type == "POINT":
            entry.update(type="point", radius=L.shadow_soft_size,
                         intensity=L.energy / (4.0 * math.pi))
        elif L.type == "SPOT":
            entry.update(type="spot", radius=L.shadow_soft_size,
                         intensity=L.energy / (4.0 * math.pi),
                         outerConeAngle=L.spot_size * 0.5,
                         innerConeAngle=L.spot_size * 0.5 * (1.0 - L.spot_blend))
        elif L.type == "SUN":
            entry.update(type="distant", halfAngle=L.angle * 0.5, intensity=L.energy)
        else:
            continue
        out.append(entry)
    return out


def export_gltf(path):
    """Filter kwargs against the operator's RNA so this survives version churn."""
    wanted = dict(
        filepath=path,
        export_format="GLTF_SEPARATE",
        use_selection=False,
        use_visible=True,
        export_apply=True,          # realise modifiers at render settings
        export_yup=True,
        export_cameras=True,
        export_lights=True,         # punctual only; area lamps go in the sidecar
        export_normals=True,
        export_tangents=True,
        export_texcoords=True,
        export_attributes=True,     # COLOR_0
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_extras=False,
        export_skins=True,
        export_animations=False,
    )
    props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    kwargs = {k: v for k, v in wanted.items() if k in props}
    dropped = sorted(set(wanted) - set(kwargs))
    if dropped:
        print("  [warn] exporter ignored: %s" % ", ".join(dropped))
    bpy.ops.export_scene.gltf(**kwargs)


def curve_objects(depsgraph):
    """Curve and hair-curves objects, which glTF has no representation for.

    Reported rather than written: Metal does support curve primitives
    (AccelerationStructureCurveGeometryDescriptor, with linear/B-spline/
    Catmull-Rom/Bezier bases and a motion variant), and oka::Curve already
    exists, so these want their own binary sidecar once that path is built.
    """
    return [ob.name for ob in depsgraph.objects if ob.type in {"CURVES", "CURVE"}]


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    out = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv else os.path.abspath(".")
    os.makedirs(out, exist_ok=True)
    if "--frame" in argv:
        bpy.context.scene.frame_set(int(argv[argv.index("--frame") + 1]))

    name = os.path.splitext(os.path.basename(bpy.data.filepath))[0] or "scene"
    depsgraph = bpy.context.evaluated_depsgraph_get()

    lights = collect_lights(depsgraph)
    with open(os.path.join(out, name + "_light.json"), "w") as f:
        json.dump({"lights": lights}, f, indent=2)
    print("lights -> sidecar: %d (%s)"
          % (len(lights), ", ".join(sorted({l["type"] for l in lights})) or "none"))

    curves = curve_objects(depsgraph)
    if curves:
        print("[gap] %d curve objects have no glTF representation and are NOT exported:"
              % len(curves))
        for c in curves[:10]:
            print("        %s" % c)
        if len(curves) > 10:
            print("        ... and %d more" % (len(curves) - 10))

    gltf = os.path.join(out, name + ".gltf")
    print("exporting %s ..." % gltf)
    export_gltf(gltf)

    with open(gltf) as f:
        doc = json.load(f)
    attrs = set()
    tris = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            attrs |= set(prim["attributes"])
            acc = prim.get("indices")
            if acc is not None:
                tris += doc["accessors"][acc]["count"] // 3
    print("exported: %d meshes, %.2f M triangles, %d materials, %d images"
          % (len(doc.get("meshes", [])), tris / 1e6,
             len(doc.get("materials", [])), len(doc.get("images", []))))
    print("  extensions: %s" % ", ".join(sorted(doc.get("extensionsUsed", []))) or "(none)")
    print("  attributes: %s" % ", ".join(sorted(attrs)))


main()
