#!/usr/bin/env python3
"""
Inventory the rendering features a .blend actually uses.

    blender -b <file.blend> -P tools/feature_tests/inspect_blend.py

Prints one flat report. The point is to answer "what would a renderer need to
support to draw this?" from the file itself rather than from assumptions about
what a scene of that name probably contains.

Everything here reads the *authored* data, not an evaluated depsgraph: on a
300 MB scene evaluating every modifier costs minutes and can exhaust memory, and
the presence of a Subdivision or Particle modifier is the fact we want anyway.
Triangle counts are therefore pre-modifier and marked as such.
"""

import bpy
from collections import Counter


def hdr(title):
    print("\n=== %s ===" % title)


def counted(title, counter, limit=None):
    hdr(title)
    if not counter:
        print("  (none)")
        return
    items = counter.most_common(limit)
    width = max(len(str(k)) for k, _ in items)
    for k, v in items:
        print("  %-*s %d" % (width, k, v))
    if limit and len(counter) > limit:
        print("  ... and %d more kinds" % (len(counter) - limit))


def main():
    print("FILE: %s" % bpy.data.filepath)
    print("Blender %s" % bpy.app.version_string)

    # -- objects, geometry -------------------------------------------------
    obj_types = Counter(o.type for o in bpy.data.objects)
    counted("objects by type", obj_types)

    tris = 0
    verts = 0
    for me in bpy.data.meshes:
        verts += len(me.vertices)
        for p in me.polygons:
            tris += max(0, len(p.vertices) - 2)
    print("\nmesh data: %d meshes, %.2f M verts, %.2f M tris (BEFORE modifiers)"
          % (len(bpy.data.meshes), verts / 1e6, tris / 1e6))

    # -- what would change that count --------------------------------------
    mods = Counter()
    subsurf_max = 0
    for o in bpy.data.objects:
        for m in getattr(o, "modifiers", []):
            mods[m.type] += 1
            if m.type == "SUBSURF":
                subsurf_max = max(subsurf_max, getattr(m, "render_levels", 0))
    counted("modifiers", mods)
    if subsurf_max:
        print("  max subdivision render level: %d" % subsurf_max)

    # -- hair / particles ---------------------------------------------------
    psys = Counter()
    hair_total = 0
    for o in bpy.data.objects:
        for ps in getattr(o, "particle_systems", []):
            s = ps.settings
            psys["%s/%s" % (s.type, s.render_type)] += 1
            if s.type == "HAIR":
                hair_total += s.count
    counted("particle systems (type/render_type)", psys)
    if hair_total:
        print("  total hair strands authored: %d" % hair_total)

    curves = [o for o in bpy.data.objects if o.type in {"CURVES", "CURVE"}]
    if curves:
        print("  curve/hair-curves objects: %d" % len(curves))

    # -- instancing ---------------------------------------------------------
    inst = Counter(o.instance_type for o in bpy.data.objects if o.instance_type != "NONE")
    counted("object instancing", inst)
    linked = Counter(l.filepath for l in bpy.data.libraries)
    counted("linked libraries", linked, 10)

    # -- materials ----------------------------------------------------------
    node_types = Counter()
    principled_driven = Counter()
    alpha_modes = Counter()
    displacement_used = 0

    # Inputs whose presence means a shading feature the renderer must have.
    watch = {
        "Subsurface Weight": "subsurface",
        "Sheen Weight": "sheen",
        "Coat Weight": "coat/clearcoat",
        "Transmission Weight": "transmission",
        "Anisotropic": "anisotropy",
        "Emission Strength": "emission",
        "Alpha": "opacity",
        "Thin Film Thickness": "thin film",
        "Specular Tint": "specular tint",
        "Normal": "normal map",
    }

    for mat in bpy.data.materials:
        if not mat.node_tree:
            continue
        alpha_modes[getattr(mat, "blend_method", "n/a")] += 1
        for n in mat.node_tree.nodes:
            node_types[n.type] += 1
            if n.type == "OUTPUT_MATERIAL":
                d = n.inputs.get("Displacement")
                if d is not None and d.links:
                    displacement_used += 1
            if n.type == "BSDF_PRINCIPLED":
                for name, label in watch.items():
                    s = n.inputs.get(name)
                    if s is None:
                        continue
                    if s.links:
                        principled_driven[label + " (textured)"] += 1
                    elif hasattr(s, "default_value"):
                        v = s.default_value
                        try:
                            v = float(v)
                        except TypeError:
                            v = max(v[:3]) if len(v) >= 3 else 0.0
                        # Non-zero means authored; Alpha and Normal are inverted
                        # or structural so they are only counted when linked.
                        if name not in ("Alpha", "Normal") and v > 1e-6:
                            principled_driven[label] += 1

    print("\nmaterials: %d" % len(bpy.data.materials))
    counted("shader node types", node_types, 30)
    counted("Principled features in use", principled_driven)
    counted("material blend_method", alpha_modes)
    if displacement_used:
        print("\n  materials driving the Displacement output: %d" % displacement_used)

    # -- textures -----------------------------------------------------------
    img_src = Counter()
    img_cs = Counter()
    big = 0
    total_px = 0
    for im in bpy.data.images:
        img_src[im.source] += 1
        img_cs[im.colorspace_settings.name] += 1
        w, h = im.size
        total_px += w * h
        if max(w, h) >= 4096:
            big += 1
    print("\nimages: %d, %.1f Mpx total, %d at 4K or larger" % (len(bpy.data.images), total_px / 1e6, big))
    counted("image source", img_src)
    counted("image colorspace", img_cs)

    # -- lights and world ---------------------------------------------------
    lights = Counter(l.type for l in bpy.data.lights)
    counted("lights by type", lights)
    for w in bpy.data.worlds:
        kinds = Counter(n.type for n in w.node_tree.nodes) if w.node_tree else Counter()
        print("  world '%s': %s" % (w.name, dict(kinds) or "no nodes"))

    # -- volumes, uv sets, colours, shape keys ------------------------------
    vol_nodes = sum(node_types[k] for k in ("VOLUME_SCATTER", "VOLUME_ABSORPTION", "PRINCIPLED_VOLUME"))
    print("\nvolume shader nodes: %d, VOLUME objects: %d" % (vol_nodes, obj_types.get("VOLUME", 0)))

    uv_max = max((len(m.uv_layers) for m in bpy.data.meshes), default=0)
    col_max = max((len(m.color_attributes) for m in bpy.data.meshes), default=0)
    shape_keyed = sum(1 for m in bpy.data.meshes if m.shape_keys)
    print("max UV layers on a mesh: %d, max colour attributes: %d, meshes with shape keys: %d"
          % (uv_max, col_max, shape_keyed))

    # -- camera -------------------------------------------------------------
    for cam in bpy.data.cameras:
        if cam.dof.use_dof:
            print("camera '%s': DOF on, f/%.2f" % (cam.name, cam.dof.aperture_fstop))

    sc = bpy.context.scene
    print("\nrender engine: %s, resolution %dx%d, samples %s"
          % (sc.render.engine, sc.render.resolution_x, sc.render.resolution_y,
             getattr(sc.cycles, "samples", "n/a")))


main()
