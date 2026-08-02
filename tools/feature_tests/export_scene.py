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

from mathutils import Matrix, Vector


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
        # The sidecar takes euler angles in degrees, not a direction vector --
        # writing a "direction" key gets silently ignored and every light ends up
        # pointing wherever the default orientation happens to face. The angles
        # are in the glTF frame, so the Blender rotation is taken through the
        # same Z-up to Y-up change of basis the positions are.
        yup = Matrix.Rotation(math.radians(-90.0), 4, "X")
        eul = (yup @ m).to_euler("XYZ")
        entry = {
            "name": ob.name,
            "position": pos,
            "color": list(L.color),
            "orientation": [math.degrees(eul.x), math.degrees(eul.y), math.degrees(eul.z)],
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
            # Blender's sun strength is irradiance in W/m^2, not radiance. The
            # sidecar defaults to radiance, and taking 5 W/m^2 as radiance over a
            # 0.0046 rad cone under-lights the scene by four orders of magnitude
            # -- which reads as a black frame, not as a units mistake.
            entry.update(type="distant", halfAngle=L.angle * 0.5, intensity=L.energy,
                         unit="irradiance")
        else:
            continue
        out.append(entry)
    return out


def merge_scenes():
    """Fold every other Blender scene into the active one.

    A .blend with several scenes exports several glTF scenes, and a renderer that
    honours defaultScene then draws one of them. The pine forest keeps its camera
    in main_scene and its entire forest in trees, so the export looked successful
    and arrived without a single tree. Collections and objects can belong to more
    than one scene, so linking is enough -- nothing is copied.
    """
    target = bpy.context.scene
    linked_objects = 0
    linked_collections = 0
    for sc in bpy.data.scenes:
        if sc is target or sc.name.startswith("__"):
            continue
        for coll in list(sc.collection.children):
            if coll.name not in target.collection.children:
                target.collection.children.link(coll)
                linked_collections += 1
        for ob in list(sc.collection.objects):
            if ob.name not in target.collection.objects:
                target.collection.objects.link(ob)
                linked_objects += 1
    if linked_objects or linked_collections:
        bpy.context.view_layer.update()
        print("merged %d other scene(s): +%d collections, +%d objects"
              % (len(bpy.data.scenes) - 1, linked_collections, linked_objects))
    return linked_objects + linked_collections


def export_gltf(path):
    """Filter kwargs against the operator's RNA so this survives version churn."""
    wanted = dict(
        filepath=path,
        export_format="GLTF_SEPARATE",
        use_selection=False,
        use_visible=True,
        use_active_scene=True,   # a multi-scene file loses everything but scene 0
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
    merge_scenes()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    lights = collect_lights(depsgraph)
    sidecar = {"lights": lights}
    # An environment baked earlier by bake_env.py, if it is sitting next to us.
    # Absolute, because the loader joins it onto the resource search path and an
    # absolute path replaces rather than appends.
    env = os.path.join(out, name + "_env.exr")
    if os.path.exists(env):
        sidecar["environment"] = {"texture": env, "intensity": 1.0}
        print("environment -> %s" % env)
    else:
        print("[gap] no baked environment; run bake_env.py for the world")
    with open(os.path.join(out, name + "_light.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
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
