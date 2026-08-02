#!/usr/bin/env python3
"""
Render a world-space normal pass in Cycles, to compare against Strelka's
DebugMode::eNormal.

    blender -b <file.blend> -P tools/feature_tests/normals_ref.py -- --out FILE
            [--width 1280] [--height 720] [--merge]

Shading is replaced wholesale by a material override that emits the normal, so
the output depends on geometry and nothing else -- no lights, no textures, no
BSDF. That is the point: it separates "is the geometry there, and facing the
right way" from every question about light transport, which is otherwise the
first thing to go wrong and the last thing to be ruled out.

The two renderers do not agree on axes -- Blender is Z-up, Strelka sees the
Y-up glTF -- so the comparison script converts rather than this one.
"""

import bpy
import math
import os
import sys


def normal_override():
    mat = bpy.data.materials.new("__normal_override")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    # (N + 1) / 2, the same encoding the debug view writes.
    mul = nt.nodes.new("ShaderNodeVectorMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = (0.5, 0.5, 0.5)
    add = nt.nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    nt.links.new(geo.outputs["Normal"], mul.inputs[0])
    nt.links.new(mul.outputs["Vector"], add.inputs[0])
    nt.links.new(add.outputs["Vector"], emi.inputs["Color"])
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])

    # Cycles refuses to render when the emissive triangle count exceeds its light
    # sampling limit, and overriding every material with an emitter puts a 50 M
    # triangle scene far past it. Nothing here needs the emission to be
    # *sampled* -- it only has to be visible to camera rays -- so turn light
    # sampling off for this material.
    for owner in (getattr(mat, "cycles", None), mat):
        if owner is not None and hasattr(owner, "emission_sampling"):
            try:
                owner.emission_sampling = "NONE"
                break
            except (TypeError, ValueError):
                pass
        if owner is not None and hasattr(owner, "sample_as_light"):
            try:
                owner.sample_as_light = False
                break
            except (TypeError, ValueError):
                pass
    return mat


def merge_scenes():
    target = bpy.context.scene
    for sc in bpy.data.scenes:
        if sc is target or sc.name.startswith("__"):
            continue
        for coll in list(sc.collection.children):
            if coll.name not in target.collection.children:
                target.collection.children.link(coll)
        for ob in list(sc.collection.objects):
            if ob.name not in target.collection.objects:
                target.collection.objects.link(ob)
    bpy.context.view_layer.update()


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    out = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv else "/tmp/normals.exr"
    width = int(argv[argv.index("--width") + 1]) if "--width" in argv else 1280
    height = int(argv[argv.index("--height") + 1]) if "--height" in argv else 720
    if "--merge" in argv:
        merge_scenes()

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"

    # Metal GPU where it exists. This pass is geometry-only, so the CPU/GPU
    # difference is not a correctness question, and a 50 M triangle forest is
    # minutes on the CPU against tens of seconds on the GPU -- long enough that
    # the check stops being run.
    #
    # --cpu because on unified memory the GPU buffers come out of the same pool
    # as the scene: the pine forest is SIGKILLed during scene build, before any
    # time limit can apply, and there is no fallback to detect from inside.
    if "--cpu" in argv:
        sc.cycles.device = "CPU"
        _pick_device = False
    else:
        _pick_device = True
    try:
        if not _pick_device:
            raise KeyError
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        found = False
        for dev in prefs.devices:
            dev.use = True
            found = found or dev.type == "METAL"
        sc.cycles.device = "GPU" if found else "CPU"
    except (KeyError, AttributeError, TypeError):
        sc.cycles.device = "CPU"

    # A hard ceiling, so a scene that turns out to be heavier than expected
    # returns a noisier image instead of never returning. Normals are constant
    # per direction, so the only thing extra samples buy is edge antialiasing.
    sc.cycles.time_limit = 60.0
    sc.cycles.samples = 8
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    for b in ("max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces"):
        setattr(sc.cycles, b, 0)
    sc.cycles.blur_glossy = 0.0
    sc.cycles.sample_clamp_indirect = 0.0
    sc.cycles.sample_clamp_direct = 0.0

    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.exr_codec = "ZIP"

    # Black world: a background of anything else is indistinguishable from a
    # surface whose encoded normal happens to be that colour.
    w = bpy.data.worlds.new("__black")
    w.use_nodes = False
    w.color = (0.0, 0.0, 0.0)
    if w.node_tree is not None:
        for n in w.node_tree.nodes:
            if n.type == "BACKGROUND":
                n.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                n.inputs["Strength"].default_value = 0.0
    sc.world = w

    bpy.context.view_layer.material_override = normal_override()

    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print("NORMALS %s %dx%d camera=%s device=%s" %
          (out, width, height, sc.camera.name if sc.camera else None, sc.cycles.device))


main()
