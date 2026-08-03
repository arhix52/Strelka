#!/usr/bin/env python3
"""
A box of fog, a low sun, a grey floor. Cycles reference plus a Strelka export.

    blender -b -P tools/feature_tests/fog_check.py -- --out DIR

Atmospheric scattering is the one feature where the pine forest and its reference
still look plainly different -- the reference has a golden haze around the sun
and Strelka's is dark -- and the forest is far too heavy to iterate on. This is
the same physics with nothing else in it: single scattering off a homogeneous
medium, strongly forward (g = 0.8), lit by a sun close to the horizon so that the
phase function is doing visible work.

Two cases, because they fail differently and in opposite directions. Looking
*into* the sun measures forward in-scattering, which is what the glow is; looking
away measures what is left over. A medium whose phase function is inverted is too
dim in the first and far too bright in the second, and either one alone could be
read as a brightness mistake.
"""

import bpy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DENSITY = 0.02          # thicker than the forest's 0.004, so the effect is not noise
ANISOTROPY = 0.8
SUN_IRRADIANCE = 5.0
SUN_ELEVATION = 5.0     # degrees above the horizon
FOG_HALF = 600.0        # large enough that the slab approximation is not the thing being measured
FOG_TOP = 20.0


def build(look_into_sun):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Floor, mid grey, so the frame is not only fog.
    bpy.ops.mesh.primitive_plane_add(size=400.0, location=(0, 0, 0))
    floor = bpy.context.active_object
    mat = bpy.data.materials.new("floor")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.18, 0.18, 0.18, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
    floor.data.materials.append(mat)

    # The medium. A box centred on the origin, tall enough to hold the camera and
    # the light path to the sun -- Strelka reads it as everything below its top.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    fog = bpy.context.active_object
    fog.scale = (FOG_HALF * 2, FOG_HALF * 2, FOG_TOP)
    fog.name = "fog"
    fmat = bpy.data.materials.new("fog_material")
    fmat.use_nodes = True
    nt = fmat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    scatter = nt.nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Density"].default_value = DENSITY
    scatter.inputs["Anisotropy"].default_value = ANISOTROPY
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(scatter.outputs["Volume"], out.inputs["Volume"])
    fog.data.materials.append(fmat)

    # A low sun along -Y, so "into the sun" is a camera looking that way.
    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = SUN_IRRADIANCE
    sun_data.angle = math.radians(0.526)
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    # Default sun points down -Z; tilt it to sit just above the horizon on +Y.
    sun.rotation_euler = (math.radians(90.0 - SUN_ELEVATION), 0.0, 0.0)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 35.0
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 2.0)
    # Level, looking along -Y (into the sun) or +Y (away from it).
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0 if look_into_sun else math.radians(180.0))
    bpy.context.scene.camera = cam

    world = bpy.data.worlds.new("black")
    world.use_nodes = True
    for n in world.node_tree.nodes:
        if n.type == "BACKGROUND":
            n.inputs["Color"].default_value = (0, 0, 0, 1)
            n.inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world


def render_reference(path, width, height):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 512
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    # Two bounces, which is what Strelka's max_depth = 2 allows: camera into the
    # medium, medium onto the floor, floor to the sun. Cutting Cycles to one
    # forbids that last path while Strelka still takes it, and the difference --
    # a factor of twenty in the backward case -- looks exactly like a broken
    # phase function.
    sc.cycles.max_bounces = 2
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 0
    sc.cycles.transmission_bounces = 0
    sc.cycles.volume_bounces = 2
    sc.cycles.sample_clamp_direct = 0.0
    sc.cycles.sample_clamp_indirect = 0.0
    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
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
    out = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv else "/tmp/fog_check"
    width = int(argv[argv.index("--width") + 1]) if "--width" in argv else 320
    height = int(argv[argv.index("--height") + 1]) if "--height" in argv else 240
    os.makedirs(out, exist_ok=True)

    import export_scene

    # A Blender camera with rotation (90, 0, 0) looks along +Y, and the sun set
    # up above shines toward +Y -- so that camera has the sun behind it. Named
    # for where the sun is, not for which way the camera was turned, because
    # getting that backwards is how a phase function ends up inverted and
    # measured as correct.
    for name, into in (("into_sun", False), ("away", True)):
        build(into)
        case = os.path.join(out, name)
        os.makedirs(case, exist_ok=True)
        render_reference(os.path.join(case, "reference.exr"), width, height)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        sidecar = {"lights": export_scene.collect_lights(depsgraph)}
        atmosphere = export_scene.collect_atmosphere(depsgraph)
        if atmosphere is not None:
            sidecar["atmosphere"] = {k: v for k, v in atmosphere.items() if k != "name"}
        with open(os.path.join(case, "scene_light.json"), "w") as f:
            json.dump(sidecar, f, indent=2)

        # The fog box itself must not be exported as geometry: Strelka reads the
        # medium from the sidecar, and a black box around the camera would be a
        # rather effective way of hiding the result.
        bpy.data.objects.remove(bpy.data.objects["fog"], do_unlink=True)
        export_scene.export_gltf(os.path.join(case, "scene.gltf"))

        print("FOGCHECK %s  density=%.4g anisotropy=%.2f sun=%.1f deg  atmosphere=%s"
              % (case, DENSITY, ANISOTROPY, SUN_ELEVATION, sidecar.get("atmosphere")))


if __name__ == "__main__":
    main()
