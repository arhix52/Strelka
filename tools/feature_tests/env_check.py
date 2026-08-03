#!/usr/bin/env python3
"""
The same world and the same camera, with every object deleted.

    blender -b <file.blend> -P tools/feature_tests/env_check.py -- --out DIR
            [--width 640] [--height 480]

Renders the Cycles reference, bakes the environment, and exports a glTF holding
the camera and a single triangle far behind it, so that comparing the two images
measures the environment pipeline and nothing else.

The pine forest is uniformly about half as bright as its reference, sky included,
and the sky is a direct texture lookup -- there is no light transport in it to be
wrong. Measuring that on the full scene means measuring it through backlit
foliage, and the patch of frame that looks like sky is full of needles. This
removes the question.
"""

import bpy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def strip_scene():
    """Delete everything but the camera. The world is a scene property and stays."""
    keep = bpy.context.scene.camera
    for ob in list(bpy.data.objects):
        if ob is keep:
            continue
        bpy.data.objects.remove(ob, do_unlink=True)
    bpy.context.view_layer.update()
    return keep


def add_backstop(camera):
    """A single triangle behind the camera.

    Only so the acceleration structure has something in it: a renderer handed an
    empty scene is a separate question from the one being asked here, and not the
    one worth answering by accident.
    """
    mesh = bpy.data.meshes.new("__backstop")
    mesh.from_pydata([(-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.0, 0.1, 0.0)], [], [(0, 1, 2)])
    mesh.update()
    ob = bpy.data.objects.new("__backstop", mesh)
    bpy.context.scene.collection.objects.link(ob)
    # A metre behind the camera, along its own +Z, which is backwards in Blender.
    from mathutils import Vector
    back = camera.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
    ob.location = camera.matrix_world.translation + back * 1.0
    return ob


def render_reference(path, width, height):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 64
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    for b in ("max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces"):
        setattr(sc.cycles, b, 0)
    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    # The scene's own sequencer and compositor are switched off. A production
    # .blend routinely has both enabled, and Blender runs the render through them
    # before it is written: on this file a constant world of (0.2, 0.5, 0.9) came
    # out as (0.197, 0.578, 1.428), which is a 59% lift in blue applied to every
    # reference rendered from this scene. What Strelka is being compared against
    # has to be the render, not a graded version of it.
    sc.render.use_sequencer = False
    sc.render.use_compositing = False
    sc.view_settings.view_transform = "Raw"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "32"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    def opt(name, default, cast=str):
        return cast(argv[argv.index(name) + 1]) if name in argv else default

    out = os.path.abspath(opt("--out", "/tmp/env_check"))
    width = opt("--width", 640, int)
    height = opt("--height", 480, int)
    os.makedirs(out, exist_ok=True)

    import export_scene
    import bake_env

    export_scene.merge_scenes()
    name = os.path.splitext(os.path.basename(bpy.data.filepath))[0] or "scene"

    branch = opt("--branch", "camera")

    world = bpy.context.scene.world
    camera = strip_scene()
    add_backstop(camera)

    # Resolved before the reference is rendered, not after, when the lighting
    # branch is the one under test: Cycles shows a camera ray the camera branch,
    # so measuring the other one means making it the only one there is.
    if branch != "camera":
        bake_env.resolve_light_path(world, branch)

    render_reference(os.path.join(out, "reference.exr"), width, height)

    bake_env.resolve_light_path(world, branch)
    sc = bake_env.make_bake_scene(world, 4096, 2048)
    raw = os.path.join(out, "env_raw.exr")
    sc.render.filepath = raw
    bpy.ops.render.render(write_still=True, scene=sc.name)
    src, w, h = bake_env.load_rgb(raw)
    dst = bake_env.resample_to_strelka(src, bake_env.direction_field(w, h), 2048, 1024)
    os.remove(raw)
    w, h = 2048, 1024
    import numpy as np
    img = bpy.data.images.new("env", width=w, height=h, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = dst[::-1]
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = os.path.join(out, "env.exr")
    img.file_format = "OPEN_EXR"
    img.save()

    with open(os.path.join(out, "scene_light.json"), "w") as f:
        json.dump({"environment": {"texture": os.path.join(out, "env.exr"), "intensity": 1.0}}, f,
                  indent=2)
    export_scene.export_gltf(os.path.join(out, "scene.gltf"))

    print("ENVCHECK %s  world='%s'  camera='%s'  %dx%d"
          % (out, world.name, camera.name, width, height))
    print("  branch=%s" % branch)


# Guarded so this can be imported for its helpers without running a whole bake.
if __name__ == "__main__":
    main()
