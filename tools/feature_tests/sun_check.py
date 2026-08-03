#!/usr/bin/env python3
"""
One sun, one grey plane, no environment. Cycles reference plus a Strelka export.

    blender -b -P tools/feature_tests/sun_check.py -- --out DIR

Exists because the feature harness reaches Strelka's distant lights through
glTF's KHR_lights_punctual, and a production scene reaches them through the light
sidecar with unit="irradiance" -- a different path, with its own conversion from
irradiance to the radiance the shader wants, and until now never measured against
anything.

The scene is chosen so the answer is arithmetic rather than a render: a lambertian
plane of albedo a facing a sun of irradiance E at incidence theta has radiance
a * E * cos(theta) / pi, which is printed alongside what each renderer produced.
An error in the cone solid angle or in the sampling pdf shows up as a ratio, and
the two half-angles present tell a constant offset apart from one that scales
with the cone.
"""

import bpy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALBEDO = 0.5
# name, irradiance (W/m^2), sun angular diameter (deg), tilt from vertical (deg).
# The angle sweep is the useful part: a constant offset and one that scales with
# the cone look identical at a single angle.
CASES = [("sun_%s" % ("%g" % d).replace(".", "p"), 5.0, d, 0.0)
         for d in (0.1, 0.526, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0)]
CASES.append(("sun_tilted", 5.0, 0.526, 60.0))


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build(irradiance, angle_deg, tilt_deg):
    mesh = bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0, 0, 0))
    plane = bpy.context.active_object
    mat = bpy.data.materials.new("grey")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (ALBEDO, ALBEDO, ALBEDO, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0
    # Nothing but the diffuse lobe, so the arithmetic in the docstring is the
    # whole answer rather than most of it.
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.0
    plane.data.materials.append(mat)

    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = irradiance
    sun_data.angle = math.radians(angle_deg)
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(tilt_deg), 0.0, 0.0)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 20.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = cam

    # Black world: only the sun lights anything, so the measurement has one term.
    world = bpy.data.worlds.new("black")
    world.use_nodes = True
    for n in world.node_tree.nodes:
        if n.type == "BACKGROUND":
            n.inputs["Color"].default_value = (0, 0, 0, 1)
            n.inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world


def render_reference(path):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 64
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    # One bounce: the direct term is what is being measured, and a 200 m plane
    # bouncing light back on itself would add a second term to compare against.
    for b in ("max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces"):
        setattr(sc.cycles, b, 0)
    sc.render.resolution_x = 128
    sc.render.resolution_y = 128
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
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
    out = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv else "/tmp/sun_check"
    os.makedirs(out, exist_ok=True)

    import export_scene

    for name, irradiance, angle_deg, tilt_deg in CASES:
        clear()
        build(irradiance, angle_deg, tilt_deg)
        case_dir = os.path.join(out, name)
        os.makedirs(case_dir, exist_ok=True)
        render_reference(os.path.join(case_dir, "reference.exr"))

        depsgraph = bpy.context.evaluated_depsgraph_get()
        lights = export_scene.collect_lights(depsgraph)
        with open(os.path.join(case_dir, "scene_light.json"), "w") as f:
            json.dump({"lights": lights}, f, indent=2)
        export_scene.export_gltf(os.path.join(case_dir, "scene.gltf"))

        expected = ALBEDO * irradiance * math.cos(math.radians(tilt_deg)) / math.pi
        with open(os.path.join(case_dir, "expected.txt"), "w") as f:
            f.write("%.6f\n" % expected)
        print("CASE %-14s E=%.1f angle=%.3f deg tilt=%.0f deg -> expected radiance %.5f"
              % (name, irradiance, angle_deg, tilt_deg, expected))


main()
