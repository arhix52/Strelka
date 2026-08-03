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

import array

import bpy
import json
import math
import os
import sys

# Alongside this file, which is not on the path when Blender runs a script by
# absolute path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
            # The sidecar's "halfAngle" is an angular *diameter in degrees*
            # despite the name -- light_json.h halves it and converts on read,
            # and writes it back the same way. Blender's sun.angle is exactly
            # that quantity, in radians.
            #
            # Writing radians here made the loader see 4e-5 rad instead of
            # 0.0046. At that angle the pdf the shader computes stops matching
            # the radiance the host baked, and the sun came out 73 times too
            # bright -- which looked like a blown-out riverbed, not like a unit
            # mistake.
            entry.update(type="distant", halfAngle=math.degrees(L.angle), intensity=L.energy,
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


def force_render_geometry():
    """Make the viewport evaluate what the render would.

    The instance placements are read from the render depsgraph, but the *meshes*
    come from the glTF exporter, which uses the viewport one. In a scene that
    branches on Is Viewport that mismatch puts render-accurate transforms on
    proxy geometry: the pine forest came back with thin bare sticks leaning
    across the frame where the reference has fir trees, which looks like a
    transform bug and is not one.

    Cutting the Is Viewport links and pinning what they fed to False collapses
    the distinction, so both halves of the export see the same scene.
    """
    cut = 0
    for group in bpy.data.node_groups:
        for node in list(group.nodes):
            if node.type != "IS_VIEWPORT":
                continue
            for output in node.outputs:
                for link in list(output.links):
                    socket = link.to_socket
                    group.links.remove(link)
                    try:
                        socket.default_value = False
                    except (TypeError, ValueError, AttributeError):
                        pass
                    cut += 1
    if cut:
        bpy.context.view_layer.update()
        print("forced render geometry: %d Is Viewport links cut" % cut)
    return cut


def collect_instances(depsgraph):
    """Every geometry-nodes / particle instance, as (source object, world matrix).

    These are the scene. The glTF exporter walks real objects, and anything a
    Geometry Nodes tree instanced is not one -- so the pine forest exported its
    50 tree *variants*, sitting in tidy rows off to the side where they were
    authored, and not one of the 38 737 trees actually placed on the terrain. The
    render came back with the ground, the ruins and the hand-placed rocks, and no
    forest, which reads as a renderer fault and is not one.

    Realising them into meshes would be the obvious fix and the wrong one: it
    turns 50 shared meshes into 38 737 copies. They are written as glTF nodes
    pointing at the meshes that already exist instead, which is what glTF
    instancing is, and what the renderer wants -- one BLAS per distinct object,
    one TLAS instance per placement.
    """
    # Decomposed on the spot into flat float arrays rather than kept as Matrix
    # objects. At 2.2 M placements the objects alone are some 450 MB of Python,
    # which is memory the glTF exporter needs a few seconds later -- on a 17 GB
    # machine that is the difference between exporting and being killed.
    out = {}
    for it in depsgraph.object_instances:
        if not it.is_instance:
            continue
        # The *evaluated* object is what has geometry; the original may not be a
        # mesh at all. Every tall tree in the pine forest is a Curve object that
        # evaluates to one, so filtering on the original's type dropped all five
        # fir variants -- half a million placements -- and left a forest of
        # ground litter with nothing standing in it.
        if it.object.type != "MESH":
            continue
        # Keyed on the mesh data, not the object.
        #
        # The depsgraph flattens nested instancing, and every branch of a
        # scattered tree reports the *tree* as its original -- so keying on the
        # object drew a whole 25 m fir at each of its own branches, 806 of them
        # stacked within 14 m of the camera. That is what the trunks leaning
        # across the frame were. The mesh data names the geometry the instance
        # actually holds, and the exporter writes glTF meshes under those same
        # names, so the two line up.
        data = it.object.data
        src = it.object.original
        name = data.name if data is not None else None
        if name is None and src is not None:
            name = src.name
        if name is None:
            continue

        entry = out.get(name)
        if entry is None:
            entry = (array.array("f"), array.array("f"), array.array("f"))
            out[name] = entry
        # Conjugated into the glTF frame here -- see gltf_matrix() -- then split,
        # because EXT_mesh_gpu_instancing stores TRS and not a matrix.
        t, q, sc = (_YUP @ it.matrix_world @ _YUP_INV).decompose()
        entry[0].extend((t.x, t.y, t.z))
        # glTF orders a quaternion xyzw; Blender's is wxyz.
        entry[1].extend((q.x, q.y, q.z, q.w))
        entry[2].extend((sc.x, sc.y, sc.z))
    return out


def collect_atmosphere(depsgraph):
    """A Volume Scatter object, as the sidecar's global atmosphere.

    The pine forest describes its haze the usual way: one box the size of the
    set, containing the camera, with a Volume Scatter on the material output.
    Strelka offers a slab rather than bounded media (see fog.h), so the box
    becomes "everything below its ceiling", which for a box centred on the scene
    is the same region wherever the camera can go.

    Two approximations, both stated: the box's horizontal extent is dropped, and
    Blender's per-channel scattering coefficient is split into a scalar
    extinction and a colour, so a coloured medium picks up a little absorption
    Cycles would not have. At the densities haze is authored with -- 0.004 here
    -- neither is visible.
    """
    # bpy.data.objects rather than the depsgraph: a fog box is routinely hidden
    # in the viewport, which is the depsgraph this runs against, and it is the
    # object's parameters that are wanted rather than its evaluated geometry.
    best = None
    for ob in bpy.data.objects:
        if ob.type != "MESH" or ob.data is None or ob.hide_render:
            continue
        for mat in ob.data.materials:
            if mat is None or not mat.use_nodes:
                continue
            out = next((n for n in mat.node_tree.nodes
                        if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
            if out is None:
                continue
            socket = out.inputs.get("Volume")
            if socket is None or not socket.links:
                continue
            node = socket.links[0].from_node
            if node.type not in {"VOLUME_SCATTER", "PRINCIPLED_VOLUME"}:
                continue
            density = float(node.inputs["Density"].default_value)
            if density <= 0.0:
                continue
            colour = node.inputs["Color"].default_value
            g = float(node.inputs["Anisotropy"].default_value) if "Anisotropy" in node.inputs else 0.0
            top = max((ob.matrix_world @ Vector(corner)).z for corner in ob.bound_box)
            # Cycles' Volume Scatter has no absorption: its extinction is
            # density * colour and every extinction event scatters. Strelka
            # carries a scalar extinction and a colour albedo, so the colour goes
            # into the extinction and the albedo comes out as 1 -- otherwise a
            # grey 0.8 medium is read as 20% absorbing and the haze comes out
            # dimmer than the scene was authored with.
            tint = [float(colour[0]), float(colour[1]), float(colour[2])]
            mean_tint = max(sum(tint) / 3.0, 1e-6)
            entry = {
                "name": ob.name,
                "color": [c / mean_tint for c in tint],
                "density": density * mean_tint,
                "anisotropy": g,
                "height": top,   # Blender z is glTF y
            }
            # The largest one wins if a scene has several: the atmosphere is the
            # one that contains the camera, and a small volume elsewhere is
            # something else that a slab cannot represent anyway.
            if best is None or top > best["height"]:
                best = entry
    return best


def rendered_object_names():
    """The objects Cycles actually draws, as opposed to those merely present.

    A scatter rig keeps its sources in the file -- proxies, trunk variants, the
    row of tree types off to the side -- and excludes them from the view layer.
    They are not in the render and they should not be in the export: on the pine
    forest they are 174 of 450 mesh nodes and, because they are the full-detail
    originals, most of the acceleration structure. 9.79 GB of it, against a
    device that will not allocate more than 9.53.
    """
    names = set()

    class _Capture(bpy.types.RenderEngine):
        bl_idname = "STRELKA_RENDERED_CAPTURE"
        bl_label = "Strelka rendered-object capture"
        bl_use_preview = False

        def render(self, depsgraph):
            for it in depsgraph.object_instances:
                if not it.is_instance and it.object is not None:
                    names.add(it.object.original.name)

    sc = bpy.context.scene
    saved = (sc.render.engine, sc.render.resolution_x, sc.render.resolution_y)
    bpy.utils.register_class(_Capture)
    try:
        sc.render.engine = _Capture.bl_idname
        sc.render.resolution_x = 4
        sc.render.resolution_y = 4
        bpy.ops.render.render()
    finally:
        (sc.render.engine, sc.render.resolution_x, sc.render.resolution_y) = saved
        bpy.utils.unregister_class(_Capture)
    return names


def strip_unrendered_nodes(gltf_path, rendered):
    """Drop mesh nodes for objects the render does not draw.

    Their meshes stay in the file if something else references them -- the
    instancing nodes do, which is the whole point -- so this removes placements,
    not geometry.
    """
    with open(gltf_path) as f:
        doc = json.load(f)

    nodes = doc.get("nodes", [])
    def keep_node(n):
        if "mesh" not in n:
            return True
        name = n.get("name", "")
        if name.startswith("instances_mesh"):
            return True
        return name in rendered

    keep = [i for i, n in enumerate(nodes) if keep_node(n)]
    if len(keep) == len(nodes):
        return 0

    remap = {old: new for new, old in enumerate(keep)}
    doc["nodes"] = [nodes[i] for i in keep]
    for n in doc["nodes"]:
        if "children" in n:
            n["children"] = [remap[c] for c in n["children"] if c in remap]
            if not n["children"]:
                del n["children"]
    for scene in doc.get("scenes", []):
        scene["nodes"] = [remap[r] for r in scene.get("nodes", []) if r in remap]
    for skin in doc.get("skins", []):
        if "skeleton" in skin and skin["skeleton"] in remap:
            skin["skeleton"] = remap[skin["skeleton"]]
        if "joints" in skin:
            skin["joints"] = [remap[j] for j in skin["joints"] if j in remap]

    removed = len(nodes) - len(keep)
    with open(gltf_path, "w") as f:
        json.dump(doc, f)
    print("removed %d nodes for objects the render does not draw" % removed)
    return removed


def render_depsgraph_instances():
    """The same, but from the render depsgraph rather than the viewport one.

    Production scatter setups branch on the Is Viewport node -- the pine forest
    has fifty of them, each feeding a Switch -- so the viewport gets cheap proxy
    stand-ins and the render gets the real trees. Reading the viewport depsgraph
    therefore exports a forest of faceted blobs that is placed exactly right and
    shaped nothing like the reference, which is a confusing thing to debug from
    the image alone.

    There is no API for a render depsgraph outside a render, so a throwaway
    render engine is registered and asked to render four pixels; what it is
    handed is the real thing. The frame is discarded.
    """
    captured = {}

    class _Capture(bpy.types.RenderEngine):
        bl_idname = "STRELKA_INSTANCE_CAPTURE"
        bl_label = "Strelka instance capture"
        bl_use_preview = False

        def render(self, depsgraph):
            captured.update(collect_instances(depsgraph))

    sc = bpy.context.scene
    saved = (sc.render.engine, sc.render.resolution_x, sc.render.resolution_y,
             sc.render.resolution_percentage)
    bpy.utils.register_class(_Capture)
    try:
        sc.render.engine = _Capture.bl_idname
        sc.render.resolution_x = 4
        sc.render.resolution_y = 4
        sc.render.resolution_percentage = 100
        bpy.ops.render.render()
    finally:
        (sc.render.engine, sc.render.resolution_x, sc.render.resolution_y,
         sc.render.resolution_percentage) = saved
        bpy.utils.unregister_class(_Capture)
    return captured


# Blender is Z-up, glTF is Y-up, and the exporter has already converted the mesh
# data -- a 20 m pine measures 20 along glTF y. So an instance placement is the
# Blender matrix conjugated into that frame, not merely rotated by it.
_YUP = Matrix.Rotation(math.radians(-90.0), 4, "X")
_YUP_INV = _YUP.inverted()


def gltf_matrix(m):
    """Blender world matrix -> glTF node matrix, column-major as the spec stores it."""
    g = _YUP @ m @ _YUP.inverted()
    return [g[r][c] for c in range(4) for r in range(4)]


def write_instances(gltf_path, instances):
    """Write the placements as EXT_mesh_gpu_instancing: one node per mesh, TRS in
    the binary buffer.

    A node apiece was the obvious first version and it does not scale. The pine
    forest's 2 238 410 placements came to 763 MB of JSON -- a 16-float matrix
    written out as text, roughly 340 bytes a placement -- and 33 seconds of the
    103-second load was tinygltf reading it. The same data as three float
    accessors is about 60 bytes and lands in the .bin, where it is a memcpy.

    EXT_mesh_gpu_instancing rather than a private sidecar because it is the
    standard answer to exactly this and other tools can read it.

    A source can be missing when it lives in a collection excluded from the view
    layer -- common, because that is how the scattered originals are kept out of
    the render. Counted and named rather than passed over: an instance with no
    mesh to point at is geometry that will be missing from the render, and that
    has to be visible in the log rather than found later in the image.
    """
    with open(gltf_path) as f:
        doc = json.load(f)

    nodes = doc.setdefault("nodes", [])
    # By mesh name first, since that is what an instance names; by node name as
    # a fallback, for a source whose mesh the exporter renamed.
    mesh_of_name = {}
    for index, mesh in enumerate(doc.get("meshes", [])):
        if mesh.get("name"):
            mesh_of_name.setdefault(mesh["name"], index)
    for n in nodes:
        if "mesh" in n and n.get("name"):
            mesh_of_name.setdefault(n["name"], n["mesh"])

    by_mesh = {}
    missing = {}
    for src_name, (translation, rotation, scale) in instances.items():
        mesh = mesh_of_name.get(src_name)
        if mesh is None:
            missing[src_name] = missing.get(src_name, 0) + len(translation) // 3
            continue
        by_mesh.setdefault(mesh, []).append((translation, rotation, scale))

    if not by_mesh:
        _report_missing(missing)
        return 0

    buffers = doc.setdefault("buffers", [])
    if not buffers or "uri" not in buffers[0]:
        raise RuntimeError("expected a separate .bin buffer to append to")
    bin_path = os.path.join(os.path.dirname(gltf_path), buffers[0]["uri"])
    views = doc.setdefault("bufferViews", [])
    accessors = doc.setdefault("accessors", [])
    scene = doc["scenes"][doc.get("scene", 0)]
    roots = scene.setdefault("nodes", [])

    def add_accessor(payload, count, kind, offset):
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload)})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5126,
                          "count": count, "type": kind})
        return len(accessors) - 1

    added = 0
    with open(bin_path, "ab") as blob:
        offset = os.path.getsize(bin_path)
        for mesh, groups in sorted(by_mesh.items()):
            # Several sources can share one glTF mesh; their placements merge.
            translation = array.array("f")
            rotation = array.array("f")
            scale = array.array("f")
            for t, r, sc in groups:
                translation.extend(t)
                rotation.extend(r)
                scale.extend(sc)
            count = len(translation) // 3

            # Four-byte alignment, which is all a float accessor asks for.
            pad = (-offset) % 4
            if pad:
                blob.write(b"\x00" * pad)
                offset += pad

            attributes = {}
            for payload, kind, name in ((translation, "VEC3", "TRANSLATION"),
                                        (rotation, "VEC4", "ROTATION"),
                                        (scale, "VEC3", "SCALE")):
                raw = payload.tobytes()
                attributes[name] = add_accessor(raw, count, kind, offset)
                blob.write(raw)
                offset += len(raw)

            roots.append(len(nodes))
            nodes.append({
                "name": "instances_mesh%d" % mesh,
                "mesh": mesh,
                "extensions": {"EXT_mesh_gpu_instancing": {"attributes": attributes}},
            })
            added += count

    buffers[0]["byteLength"] = os.path.getsize(bin_path)
    used = set(doc.get("extensionsUsed", []))
    used.add("EXT_mesh_gpu_instancing")
    doc["extensionsUsed"] = sorted(used)

    with open(gltf_path, "w") as f:
        json.dump(doc, f)

    print("instances -> %d placements over %d meshes, %d nodes, %.1f MB of binary"
          % (added, len(by_mesh), len(by_mesh), added * 40 / 1e6))
    _report_missing(missing)
    return added


def strip_instance_nodes(gltf_path):
    """Remove nodes written by an earlier per-placement export.

    They are recognised by name -- "<source>_instN" -- because that is what the
    first version wrote, and by carrying EXT_mesh_gpu_instancing, which is what
    this one writes. Removing nodes renumbers the rest, so every reference has to
    be remapped rather than the entries simply dropped.
    """
    with open(gltf_path) as f:
        doc = json.load(f)

    nodes = doc.get("nodes", [])
    def is_placement(n):
        name = n.get("name", "")
        return ("_inst" in name and "matrix" in n) or \
               "EXT_mesh_gpu_instancing" in n.get("extensions", {})

    keep = [i for i, n in enumerate(nodes) if not is_placement(n)]
    if len(keep) == len(nodes):
        return 0

    remap = {old: new for new, old in enumerate(keep)}
    doc["nodes"] = [nodes[i] for i in keep]
    for n in doc["nodes"]:
        if "children" in n:
            n["children"] = [remap[c] for c in n["children"] if c in remap]
            if not n["children"]:
                del n["children"]
    for scene in doc.get("scenes", []):
        scene["nodes"] = [remap[r] for r in scene.get("nodes", []) if r in remap]
    for skin in doc.get("skins", []):
        if "skeleton" in skin and skin["skeleton"] in remap:
            skin["skeleton"] = remap[skin["skeleton"]]
        if "joints" in skin:
            skin["joints"] = [remap[j] for j in skin["joints"] if j in remap]

    removed = len(nodes) - len(keep)
    with open(gltf_path, "w") as f:
        json.dump(doc, f)
    print("removed %d placement nodes from a previous export" % removed)
    return removed


def _report_missing(missing):
    if not missing:
        return
    total = sum(missing.values())
    print("[gap] %d instances dropped: their source object was not exported "
          "(hidden, or in a collection excluded from the view layer)" % total)
    for name, count in sorted(missing.items(), key=lambda kv: -kv[1])[:10]:
        print("        %-40s x%d" % (name, count))


def write_material_extensions(gltf_path, translucency, volumes):
    """Add the KHR material extensions the exporter has no mapping for.

    KHR_materials_diffuse_transmission from a Translucent BSDF: without it the
    pine canopy is roughly ten times darker than the reference in a backlit shot,
    which looks like a shadowing problem and is a missing lobe.

    KHR_materials_volume from a Volume Absorption wired to the output: the river
    is a transmissive surface over an absorbing medium, and without the medium it
    is clear glass over sand.
    """
    with open(gltf_path) as f:
        doc = json.load(f)

    used = set(doc.get("extensionsUsed", []))
    counts = {"diffuse_transmission": 0, "volume": 0}
    for mat in doc.get("materials", []):
        name = mat.get("name")
        entry = translucency.get(name)
        if entry is not None:
            factor, colour = entry
            mat.setdefault("extensions", {})["KHR_materials_diffuse_transmission"] = {
                "diffuseTransmissionFactor": factor,
                "diffuseTransmissionColorFactor": colour,
            }
            used.add("KHR_materials_diffuse_transmission")
            counts["diffuse_transmission"] += 1

        volume = volumes.get(name)
        if volume is not None:
            colour, distance = volume
            mat.setdefault("extensions", {})["KHR_materials_volume"] = {
                "thicknessFactor": 1.0,
                "attenuationColor": colour,
                "attenuationDistance": distance,
            }
            used.add("KHR_materials_volume")
            counts["volume"] += 1

    doc["extensionsUsed"] = sorted(used)
    with open(gltf_path, "w") as f:
        json.dump(doc, f)
    print("material extensions -> diffuse transmission %d, volume %d"
          % (counts["diffuse_transmission"], counts["volume"]))
    return counts


def curve_objects(depsgraph):
    """Curve and hair-curves objects, which glTF has no representation for.

    Reported rather than written: Metal does support curve primitives
    (AccelerationStructureCurveGeometryDescriptor, with linear/B-spline/
    Catmull-Rom/Bezier bases and a motion variant), and oka::Curve already
    exists, so these want their own binary sidecar once that path is built.
    """
    # Only hair curves. A legacy Curve object is converted to a mesh on export
    # and comes through fine -- reporting those as missing sent the search after
    # trees that were in the file all along.
    return [ob.name for ob in depsgraph.objects if ob.type == "CURVES"]


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    out = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv else os.path.abspath(".")
    os.makedirs(out, exist_ok=True)
    if "--frame" in argv:
        bpy.context.scene.frame_set(int(argv[argv.index("--frame") + 1]))

    name = os.path.splitext(os.path.basename(bpy.data.filepath))[0] or "scene"
    merge_scenes()
    force_render_geometry()
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
        # What the camera sees, when bake_env.py found the world branching on
        # Light Path and baked both sides. Without it the backdrop in frame is
        # the lighting environment, which here is a procedural sky at strength
        # 0.7 standing in for an 8k HDRI at 0.2.
        backdrop = os.path.join(out, name + "_env_camera.exr")
        if os.path.exists(backdrop):
            sidecar["environment"]["backgroundTexture"] = backdrop
            sidecar["environment"]["backgroundIntensity"] = 1.0
            print("environment backdrop -> %s" % backdrop)
    else:
        print("[gap] no baked environment; run bake_env.py for the world")
    atmosphere = collect_atmosphere(depsgraph)
    if atmosphere is not None:
        sidecar["atmosphere"] = {k: v for k, v in atmosphere.items() if k != "name"}
        print("atmosphere <- %s: density %.4g, anisotropy %.2f, below y=%.1f"
              % (atmosphere["name"], atmosphere["density"], atmosphere["anisotropy"],
                 atmosphere["height"]))

    with open(os.path.join(out, name + "_light.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    print("lights -> sidecar: %d (%s)"
          % (len(lights), ", ".join(sorted({l["type"] for l in lights})) or "none"))

    instances = render_depsgraph_instances()
    placements = sum(len(t) // 3 for t, _, _ in instances.values())
    print("instances: %d placements over %d sources, from the render depsgraph"
          % (placements, len(instances)))

    curves = curve_objects(depsgraph)
    if curves:
        print("[gap] %d curve objects have no glTF representation and are NOT exported:"
              % len(curves))
        for c in curves[:10]:
            print("        %s" % c)
        if len(curves) > 10:
            print("        ... and %d more" % (len(curves) - 10))

    # Before the export, because it rewrites the graphs the exporter reads.
    import flatten_materials
    flat, translucency, volumes = flatten_materials.flatten(out)
    if flat:
        print("materials rewritten for export: %d" % len(flat))
        for mat_name, note in flat:
            print("        %-28s %s" % (mat_name, note))

    gltf = os.path.join(out, name + ".gltf")

    # --instances-only rewrites the placements of an export that already exists,
    # leaving its geometry, materials and textures alone. The glTF exporter needs
    # several gigabytes on a scene this size and there is no reason to pay that
    # again to change how the placements are stored.
    if "--instances-only" in argv:
        print("rewriting placements in %s (geometry left as it is)" % gltf)
        strip_instance_nodes(gltf)
        strip_unrendered_nodes(gltf, rendered_object_names())
        if instances:
            write_instances(gltf, instances)
        return

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

    # After the summary, because these rewrite the file the summary was read from.
    strip_unrendered_nodes(gltf, rendered_object_names())
    if translucency or volumes:
        write_material_extensions(gltf, translucency, volumes)
    if instances:
        write_instances(gltf, instances)


# Guarded, because these are imported as a module by env_check.py -- without it
# the import runs a full export as a side effect, which is slow, confusing, and
# writes files the caller did not ask for.
if __name__ == "__main__":
    main()
