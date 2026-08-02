#!/usr/bin/env python3
"""Render validation cornell_box with Cycles to match Strelka sidecar lighting.

Usage:
  /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/compare_cornell_cycles.py
"""

import bpy
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GLB = os.path.join(REPO, "scenes/validation/cornell_box/cornell_box.glb")
OUT_DIR = "/tmp/cornell_compare"
OUT_EXR = os.path.join(OUT_DIR, "cycles.exr")
OUT_PNG = os.path.join(OUT_DIR, "cycles.png")

WIDTH, HEIGHT = 512, 384
SAMPLES = 256
MAX_DEPTH = 8

# Sidecar: rect radiance=400, 0.5x0.5, at (0, 1.98, 0), orientation (-90,0,0).
# Lambertian: P = L * A * pi  (same conversion as tools/feature_tests/build_features.py)
LIGHT_RADIANCE = 400.0
LIGHT_W = 0.5
LIGHT_H = 0.5
LIGHT_POWER = LIGHT_RADIANCE * (LIGHT_W * LIGHT_H) * math.pi
# glTF Y-up → Blender Z-up: (x,y,z)_gltf -> (x,-z,y)_blender
LIGHT_POS_GLTF = (0.0, 1.98, 0.0)
LIGHT_POS_BLENDER = (LIGHT_POS_GLTF[0], -LIGHT_POS_GLTF[2], LIGHT_POS_GLTF[1])
# orientation euler XYZ degrees in Strelka sidecar; -90 X = emit downward in Y-up.
# After Y-up→Z-up, downward is -Z in Blender, which AREA lights do by default
# when rotation is identity after converting the light transform carefully.
# Simpler: place light and point it at the floor.
CAM_POS_GLTF = (0.0, 1.0, 3.5)
CAM_TGT_GLTF = (0.0, 1.0, 0.0)
CAM_POS = (CAM_POS_GLTF[0], -CAM_POS_GLTF[2], CAM_POS_GLTF[1])
CAM_TGT = (CAM_TGT_GLTF[0], -CAM_TGT_GLTF[2], CAM_TGT_GLTF[1])
CAM_FOV_DEG = 45.0


def gltf_to_blender(p):
    return (p[0], -p[2], p[1])


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_cycles(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "METAL"
        scene.cycles.device = "GPU"
        prefs.get_devices()
    except Exception:
        scene.cycles.device = "CPU"

    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.max_bounces = MAX_DEPTH
    scene.cycles.diffuse_bounces = MAX_DEPTH
    scene.cycles.glossy_bounces = MAX_DEPTH
    scene.cycles.transmission_bounces = MAX_DEPTH
    scene.cycles.transparent_max_bounces = MAX_DEPTH
    scene.cycles.seed = 0

    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    # Linear out for comparison
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"

    world = bpy.data.worlds.new("World")
    world.use_nodes = False
    world.color = (0.0, 0.0, 0.0)
    scene.world = world


def import_glb():
    bpy.ops.import_scene.gltf(filepath=GLB)
    # Importer already converts Y-up → Z-up.


def make_diffuse_materials_lambertian():
    """Kill default Principled specular so we compare near-Lambertian paths."""
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if not bsdf:
            continue
        for name in ("Specular IOR Level", "Specular"):
            if name in bsdf.inputs:
                bsdf.inputs[name].default_value = 0.0
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 1.0


def add_rect_light():
    # Remove any lights that came with the glTF (there shouldn't be any).
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    data = bpy.data.lights.new("CeilingRect", type="AREA")
    data.shape = "RECTANGLE"
    data.size = LIGHT_W
    data.size_y = LIGHT_H
    data.energy = LIGHT_POWER
    data.color = (1.0, 1.0, 1.0)
    obj = bpy.data.objects.new("CeilingRect", data)
    bpy.context.collection.objects.link(obj)
    # glTF import already placed geometry in Blender space; light sidecar is
    # authored in the same glTF/Y-up frame Strelka uses, so convert.
    obj.location = LIGHT_POS_BLENDER
    # Area lights emit along local -Z. Point -Z toward the floor (world -Z).
    obj.rotation_euler = (0.0, 0.0, 0.0)
    return obj


def setup_camera():
    # Prefer the imported camera if present; otherwise create one.
    cam = next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
    if cam is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)

    cam.location = CAM_POS
    from mathutils import Vector
    direction = Vector(CAM_TGT) - Vector(CAM_POS)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    cam.data.sensor_fit = "VERTICAL"
    cam.data.lens_unit = "FOV"
    cam.data.angle = math.radians(CAM_FOV_DEG)
    bpy.context.scene.camera = cam
    return cam


def render_exr_and_png():
    os.makedirs(OUT_DIR, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = OUT_EXR
    bpy.ops.render.render(write_still=True)

    # Also write a display PNG (sRGB-ish) for quick viewing.
    img = bpy.data.images.load(OUT_EXR)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    # Save through the render result path using a temporary view transform.
    scene.view_settings.view_transform = "Standard"
    img.filepath_raw = OUT_PNG
    img.file_format = "PNG"
    # Apply a simple tonemap-ish save: Blender's save uses scene view settings
    # only via render; for loaded EXR use pixels clamped.
    import numpy as np
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    rgb = buf.reshape(h, w, 4)[:, :, :3]
    # Exposure match: leave linear→sRGB encode with a mild gain so the image
    # is visible; quantitative compare stays on EXR.
    x = np.clip(rgb * 0.15, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055)
    img.pixels = np.concatenate([srgb, np.ones((h, w, 1), dtype=np.float32)], axis=2).reshape(-1)
    img.save()
    bpy.data.images.remove(img)


def main():
    print("Blender", bpy.app.version_string)
    print("GLB", GLB)
    if not os.path.exists(GLB):
        print("Missing glb", file=sys.stderr)
        sys.exit(1)

    reset()
    scene = bpy.context.scene
    setup_cycles(scene)
    import_glb()
    make_diffuse_materials_lambertian()
    add_rect_light()
    setup_camera()

    print("Light power W =", LIGHT_POWER, "at", LIGHT_POS_BLENDER)
    print("Camera at", CAM_POS, "->", CAM_TGT)
    render_exr_and_png()
    print("Wrote", OUT_EXR)
    print("Wrote", OUT_PNG)


if __name__ == "__main__":
    main()
