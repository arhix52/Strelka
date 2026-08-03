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

import tempfile

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
    # Upright, not identity. Cycles maps an equirectangular panorama in *camera*
    # space, so its poles are the camera's own +/-Y -- with an identity rotation
    # that is the world's +/-Y, which is a horizontal axis in a Z-up scene. The
    # resulting panorama is a perfectly good image of the sphere and a terrible
    # grid to resample from: its dense rows run through the horizon and its poles
    # sit on it, ninety degrees from where the destination's are, so a scatter
    # between the two leaves a tenth of the output with no sample at all.
    #
    # Ninety degrees about X puts the camera's up on the world's, and the two
    # grids then agree on where the poles are.
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
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


def direction_field(width, height):
    """Render a panorama whose pixels hold the direction Cycles used for them.

    Rather than assume Cycles' equirectangular convention and hope. The assumed
    one was wrong, and wrong in a way that survived every check: a round trip
    through my own formula reproduced itself perfectly, because both halves of
    the test shared the mistake. What it could not survive was a render with the
    geometry deleted, where the reference showed a horizon and Strelka showed the
    zenith.

    The world is replaced by one that outputs its own incoming direction, so each
    pixel of the result *is* the answer for that pixel. Nothing is assumed and
    nothing can drift when Blender changes its mind.
    """
    world = bpy.data.worlds.new("__dir_probe")
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    # Incoming points back along the ray, so negate; then map [-1,1] to [0,1]
    # because the encoding has to survive a render.
    neg = nt.nodes.new("ShaderNodeVectorMath")
    neg.operation = "MULTIPLY"
    neg.inputs[1].default_value = (-0.5, -0.5, -0.5)
    add = nt.nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(geo.outputs["Incoming"], neg.inputs[0])
    nt.links.new(neg.outputs["Vector"], add.inputs[0])
    nt.links.new(add.outputs["Vector"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    sc = make_bake_scene(world, width, height)
    sc.cycles.samples = 1
    # No pixel filter: a filtered edge averages two directions into one that is
    # neither, and this pass is a coordinate, not an image.
    sc.render.filter_size = 0.01
    path = os.path.join(tempfile.gettempdir(), "__strelka_dir_probe.exr")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True, scene=sc.name)
    field, w, h = load_rgb(path)
    os.remove(path)
    bpy.data.scenes.remove(sc)
    bpy.data.worlds.remove(world)
    return field * 2.0 - 1.0   # back to [-1, 1], bottom-up like everything here


def resample_to_strelka(src, dirs, out_w, out_h):
    """Rebuild the panorama in Strelka's convention, using measured directions.

    Strelka (env_light_metal.h):
        u = (atan2(d.x, d.z) + pi) / 2pi,  v = acos(d.y) / pi
    with d in the glTF frame (Y up) and v = 0 at the top row.

    Forward-scattered rather than inverse-sampled: every source pixel knows its
    own direction, so it knows where it belongs, and no inverse of Cycles'
    mapping has to be written down -- which is the assumption that was wrong.

    The source is rendered at twice the output resolution so that each
    destination pixel collects several samples and the scatter is an average
    rather than a lottery. At equal resolutions it is neither: rounding leaves
    gaps that have to be filled from neighbours, and filling them lifts the dark
    bands until the sky has no contrast left.
    """
    norm = np.linalg.norm(dirs, axis=2, keepdims=True)
    d = dirs / np.maximum(norm, 1e-9)
    bx, by, bz = d[:, :, 0], d[:, :, 1], d[:, :, 2]
    # Blender (Z up) -> glTF (Y up), the same change of basis as gltf_pos().
    gx, gy, gz = bx, bz, -by

    u = (np.arctan2(gx, gz) + math.pi) / (2.0 * math.pi)
    v = np.arccos(np.clip(gy, -1.0, 1.0)) / math.pi

    tx = np.clip((u * out_w).astype(np.int32), 0, out_w - 1).ravel()
    ty = np.clip((v * out_h).astype(np.int32), 0, out_h - 1).ravel()

    accum = np.zeros((out_h, out_w, 3), dtype=np.float64)
    count = np.zeros((out_h, out_w), dtype=np.int32)
    np.add.at(accum, (ty, tx), src.reshape(-1, 3))
    np.add.at(count, (ty, tx), 1)

    hit = count > 0
    dst = np.zeros((out_h, out_w, 3), dtype=np.float32)
    dst[hit] = (accum[hit] / count[hit][:, None]).astype(np.float32)

    missing = int((~hit).sum())
    for _ in range(6):
        if hit.all():
            break
        for shift, axis in ((1, 1), (-1, 1), (1, 0), (-1, 0)):
            near_hit = np.roll(hit, shift, axis=axis)
            near_val = np.roll(dst, shift, axis=axis)
            fill = near_hit & ~hit
            dst[fill] = near_val[fill]
            hit |= fill
    if missing:
        print("  resample: %d of %d output texels had no source sample (%.3f%%)"
              % (missing, out_w * out_h, 100.0 * missing / (out_w * out_h)))
    return dst


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

    # Twice the output resolution: the resample scatters source pixels into
    # destination texels, and at 1:1 that leaves gaps.
    sc = make_bake_scene(world, width * 2, height * 2)
    raw = os.path.join(out, name + "_env_raw.exr")
    sc.render.filepath = raw
    bpy.ops.render.render(write_still=True, scene=sc.name)

    src, w, h = load_rgb(raw)
    dst = resample_to_strelka(src, direction_field(w, h), width, height)
    w, h = width, height

    suffix = "_env.exr" if branch == "light" else "_env_camera.exr"
    final = os.path.join(out, name + suffix)
    img = bpy.data.images.new(name + suffix, width=w, height=h, float_buffer=True)
    # Linear, set before the pixels are written. Without it the image is sRGB and
    # Blender encodes on save, so the file holds display values where the
    # renderer expects radiance: midtones lift, contrast flattens, and the sky
    # comes out about a quarter too bright in a way that looks like a lighting
    # difference rather than a colour-management one.
    img.colorspace_settings.name = "Non-Color"
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


# Guarded, because these are imported as a module by env_check.py -- without it
# the import runs a full export as a side effect, which is slow, confusing, and
# writes files the caller did not ask for.
if __name__ == "__main__":
    main()
