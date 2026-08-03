#!/usr/bin/env python3
"""
Bake a Blender world into an equirectangular HDRI Strelka can use.

    blender -b <file.blend> -P tools/feature_tests/bake_env.py -- --out DIR [--width 2048]

Writes <DIR>/<name>_env.exr and prints the `environment` block to put in the
light sidecar.

Why a render and not a direct evaluation: a world is a shader graph, and there is
no API to evaluate one per direction. Rendering an equirectangular panorama with
nothing in the scene samples exactly the graph, at whatever resolution is asked
for.

Why a resample afterwards: Cycles' panoramic camera and Strelka's environment
lookup do not agree on which pixel is which direction, and guessing costs a full
render to find out. Both mappings are written down below and the image is
resampled through them, so the result is correct by construction rather than by
a rotation someone tuned by eye.

One thing this cannot do: a world that branches on Light Path -- as the pine
forest's does, showing a sharp HDRI to camera rays and a sky texture to
everything else -- is two environments, not one. The bake captures the camera
branch, because that is what a camera ray sees. Lighting will differ from Cycles
by whatever the other branch contributed.
"""

import bpy
import math
import os
import sys

import numpy as np


def resolve_light_path(world, branch):
    """Collapse Light Path branches in a world to one side of the mix.

    A production world is routinely two environments: this one shows an 8k HDRI
    to camera rays at strength 0.2 and a procedural sky to everything else at
    0.7. Rendering a panorama through a camera captures only the first, so a bake
    that ignores this lights the scene with the backdrop -- here roughly three
    and a half times too bright, and wrong in colour as well as level.

    Blender's Mix Shader takes its first input at factor 0, so "Is Camera Ray"
    feeding the factor means slot 1 is the camera branch and slot 0 is everything
    else.
    """
    if world.node_tree is None:
        return 0
    nt = world.node_tree
    rewired = 0
    for node in list(nt.nodes):
        if node.type != "MIX_SHADER":
            continue
        fac = node.inputs["Fac"] if "Fac" in node.inputs else node.inputs[0]
        if not fac.links or fac.links[0].from_node.type != "LIGHT_PATH":
            continue
        if fac.links[0].from_socket.name != "Is Camera Ray":
            continue
        shaders = [i for i in node.inputs if i.type == "SHADER"]
        keep = shaders[1] if branch == "camera" else shaders[0]
        if not keep.links:
            continue
        source = keep.links[0].from_socket
        for link in list(node.outputs[0].links):
            nt.links.new(source, link.to_socket)
        rewired += 1
    return rewired


def make_bake_scene(src_world, width, height):
    sc = bpy.data.scenes.new("__env_bake")
    sc.world = src_world

    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 16          # a world is analytic; this is anti-aliasing only
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    for b in ("max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces"):
        setattr(sc.cycles, b, 0)

    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    # Raw, not Standard. Blender writes an EXR through the view transform when
    # the render is saved as a render, which is what bpy.ops.render.render does,
    # and Standard is an sRGB display transform -- so a bake meant to carry
    # radiance would carry display values instead.
    sc.view_settings.view_transform = "Raw"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.exr_codec = "ZIP"

    cam_data = bpy.data.cameras.new("__env_cam")
    cam_data.type = "PANO"
    # The property moved out of the cycles sub-struct in 4.x.
    for owner in (cam_data, getattr(cam_data, "cycles", None)):
        if owner is not None and hasattr(owner, "panorama_type"):
            try:
                owner.panorama_type = "EQUIRECTANGULAR"
                break
            except (TypeError, ValueError):
                pass
    cam = bpy.data.objects.new("__env_cam", cam_data)
    sc.collection.objects.link(cam)
    # Identity rotation, so Cycles' generated ray direction *is* the world
    # direction and the mapping below has no camera matrix in it.
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.location = (0.0, 0.0, 0.0)
    sc.camera = cam
    return sc


def load_rgb(path):
    im = bpy.data.images.load(path)
    w, h = im.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    im.pixels.foreach_get(buf)
    bpy.data.images.remove(im)
    return buf.reshape(h, w, 4)[:, :, :3], w, h  # bottom-up rows, as Blender stores them


def resample_to_strelka(src, w, h):
    """Rebuild the panorama in Strelka's convention.

    Strelka (env_light_metal.h):
        u = (atan2(d.x, d.z) + pi) / 2pi,  v = acos(d.y) / pi
    with d in the glTF frame (Y up) and v = 0 at the top row.

    Cycles (equirectangular_to_direction), camera space, which the identity
    camera rotation makes world space:
        phi = pi * (1 - 2u),  theta = pi * (v - 0.5)
        d = (cos(theta)cos(phi), cos(theta)sin(phi), sin(theta))
    with v = 0 at the bottom row, which is also how Blender stores pixels.
    """
    j, i = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    u_s = (i + 0.5) / w
    v_s = (j + 0.5) / h

    phi = u_s * 2.0 * math.pi - math.pi
    theta = v_s * math.pi
    # glTF-frame direction for this output pixel
    gx = np.sin(theta) * np.sin(phi)
    gy = np.cos(theta)
    gz = np.sin(theta) * np.cos(phi)
    # glTF -> Blender: gltf(x, y, z) == blender(bx, bz, -by)
    bx, by, bz = gx, -gz, gy

    theta_c = np.arcsin(np.clip(bz, -1.0, 1.0))
    phi_c = np.arctan2(by, bx)
    v_c = theta_c / math.pi + 0.5
    u_c = (1.0 - phi_c / math.pi) * 0.5

    sx = np.clip((u_c * w).astype(np.int32), 0, w - 1)
    sy = np.clip((v_c * h).astype(np.int32), 0, h - 1)
    return src[sy, sx]


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    out = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv else os.path.abspath(".")
    width = int(argv[argv.index("--width") + 1]) if "--width" in argv else 2048
    height = width // 2
    os.makedirs(out, exist_ok=True)

    world = bpy.context.scene.world
    if world is None:
        print("ENVBAKE none: scene has no world")
        return
    branch = argv[argv.index("--branch") + 1] if "--branch" in argv else "light"
    rewired = resolve_light_path(world, branch)
    if rewired:
        print("world Light Path resolved to the %s branch (%d mix nodes)" % (branch, rewired))

    name = os.path.splitext(os.path.basename(bpy.data.filepath))[0] or "scene"

    sc = make_bake_scene(world, width, height)
    raw = os.path.join(out, name + "_env_raw.exr")
    sc.render.filepath = raw
    bpy.ops.render.render(write_still=True, scene=sc.name)

    src, w, h = load_rgb(raw)
    dst = resample_to_strelka(src, w, h)

    suffix = "_env.exr" if branch == "light" else "_env_camera.exr"
    final = os.path.join(out, name + suffix)
    img = bpy.data.images.new(name + suffix, width=w, height=h, float_buffer=True)
    rgba = np.ones((h, w, 4), dtype=np.float32)
    # Strelka reads row 0 as v = 0; Blender writes pixels bottom-up, so flip.
    rgba[:, :, :3] = dst[::-1]
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = final
    img.file_format = "OPEN_EXR"
    img.save()
    if "--keep-raw" not in argv:
        os.remove(raw)

    print("ENVBAKE %s  %dx%d  world='%s'  mean=%.4f max=%.3f"
          % (final, w, h, world.name, float(dst.mean()), float(dst.max())))


main()
