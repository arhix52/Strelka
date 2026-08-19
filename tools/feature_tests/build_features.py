#!/usr/bin/env python3
"""
Feature-test scene generator for Strelka vs Cycles comparison.

Run headless:
    blender -b -P tools/feature_tests/build_features.py -- --out scenes/feature_tests

For each feature it writes into <out>/<NN_name>/:
    <name>.gltf + .bin + textures   glTF export (Y-up)
    <name>_light.json               Strelka light sidecar (takes precedence over KHR)
    <name>.toml                     StrelkaCLI config
    <name>_cycles.exr               Cycles reference, LINEAR, no view transform

Both sides render to linear EXR so the comparison never goes through a tone
curve. Strelka's tonemap is set to "none" and its photometric exposure is
pinned to exactly 1.0 (see EXPOSURE_* below).
"""

import bpy
import json
import math
import os
import struct
import sys
import zlib

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RES = 512
CYCLES_SAMPLES = 256
MAX_DEPTH = 8
STRELKA_SPP = 512

# Per-scene reference sample overrides. A groom is by far the noisiest subject in
# the ladder: at 256 samples the 28_hair reference carries rel 0.074 of its own
# variance on the hair sphere, measured against the same scene at 2048 -- which
# was most of what that row used to report as disagreement with Strelka. A row
# cannot measure a lobe through a reference noisier than the effect. Converging
# this one costs about a minute.
SCENE_SAMPLES = {"28_hair": 2048}

# The same argument applies to our side of the comparison, and it is cheap here:
# Strelka renders this scene in seconds. At 512 spp its own variance is rel 0.019
# on the hair half, which is half of what the row then reports as disagreement --
# the mean ratio does not move between 512 and 2048, only the noise does.
STRELKA_SCENE_SPP = {"28_hair": 2048}

# Strelka exposure: exposureValue = cm2_factor * iso / (shutter * fstop^2) / 100
# with cm2_factor pinned to 1.0 in headless mode (HeadlessApp.cpp).
# iso=100, shutter=1, fstop=1 -> exactly 1.0, i.e. no exposure scaling at all.
EXPOSURE_ISO = 100.0
EXPOSURE_FSTOP = 1.0
EXPOSURE_SHUTTER = 1.0

# Camera, in BLENDER coordinates (Z-up). Converted to glTF Y-up when written
# to the .toml, because Strelka sees the post-export world.
CAM_LOC = (0.0, -4.6, 1.15)
CAM_TARGET = (0.0, 0.0, 0.75)
CAM_FOV_DEG = 45.0
# A feature scene may override framing when the default two-subject stage makes
# the feature too small to inspect. 28_hair is deliberately a close-up: the old
# bald control duplicated 00_calibration, occupied half the frame, and diluted
# the groom's error below the row's threshold.
SCENE_CAMERAS = {
    "28_hair": ((0.0, -3.0, 1.05), (0.0, 0.0, 0.70), 45.0),
}
# Vertical ortho extent that matches the perspective framing at CAM_LOC:
# distance * 2 * tan(FOV/2) ≈ 4.617 * 2 * tan(22.5°) ≈ 3.82.
CAM_ORTHO_SCALE = 3.82

# Key light: a rect area light above the stage, in BLENDER coordinates.
KEY_SIZE = 1.6           # square, metres
KEY_POS = (0.0, -0.6, 3.2)
# Chosen so that the 0.18 grey sphere in scene 00 lands near 0.2 linear with
# exposure pinned to 1.0. Anything much brighter clips the contact sheets and
# hides exactly the differences this harness exists to show.
KEY_POWER_W = 150.0      # Blender area-light power in watts

# A Blender area light of power P over area A emits, on the Lambertian
# assumption Cycles uses, a radiance of P / (A * pi). Strelka's rect light
# takes radiance directly, so deriving both sides from one physical number is
# the only way the two renderers can agree without a fudge factor. If scene 00
# shows a constant ratio, that ratio is the bug -- fix it, don't paper over it.
KEY_RADIANCE = KEY_POWER_W / ((KEY_SIZE * KEY_SIZE) * math.pi)


# ---------------------------------------------------------------------------
# Minimal PNG writer
#
# Textures are written byte-exact rather than through bpy.data.images.save(),
# because that path applies a colour-space transform on the way out and the
# whole point of the sRGB scene is to know precisely which bytes are on disk.
# ---------------------------------------------------------------------------

def write_png(path, width, height, rows):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(blob)


def tex_srgb_probe(path, size=256):
    """16-step grey ramp on top, saturated colour patches below.

    The grey ramp is the actual probe: byte 128 must decode to linear 0.216,
    not 0.502. If Strelka skips the sRGB transfer function the ramp comes out
    visibly washed out and the error grows toward the dark end.
    """
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            step = min(15, x * 16 // size)
            if y < size // 2:
                v = step * 17  # 0, 17, ... 255
                row += bytes((v, v, v, 255))
            else:
                patches = [
                    (220, 30, 30), (30, 200, 60), (40, 70, 230), (240, 220, 40),
                    (230, 120, 20), (150, 40, 200), (30, 200, 200), (245, 245, 245),
                ]
                row += bytes(patches[step % 8] + (255,))
        rows.append(row)
    write_png(path, size, size, rows)


def tex_alpha_dots(path, size=256):
    """Opaque disc grid on a fully transparent background."""
    rows = []
    cell = size // 4
    for y in range(size):
        row = bytearray()
        for x in range(size):
            cx, cy = (x % cell) - cell / 2, (y % cell) - cell / 2
            inside = (cx * cx + cy * cy) < (cell * 0.33) ** 2
            row += bytes((235, 60, 40, 255) if inside else (235, 60, 40, 0))
        rows.append(row)
    write_png(path, size, size, rows)


def tex_normal_bumps(path, size=256):
    """Tangent-space normal map: a grid of round bumps."""
    rows = []
    cell = size // 8
    for y in range(size):
        row = bytearray()
        for x in range(size):
            cx, cy = (x % cell) - cell / 2, (y % cell) - cell / 2
            r = cell * 0.42
            d2 = cx * cx + cy * cy
            if d2 < r * r:
                nx, ny = cx / r, cy / r
                nz = math.sqrt(max(0.0, 1.0 - nx * nx - ny * ny))
            else:
                nx, ny, nz = 0.0, 0.0, 1.0
            row += bytes((
                int((nx * 0.5 + 0.5) * 255),
                int((ny * 0.5 + 0.5) * 255),
                int((nz * 0.5 + 0.5) * 255),
                255,
            ))
        rows.append(row)
    write_png(path, size, size, rows)


# ---------------------------------------------------------------------------
# Principled BSDF socket names moved around in 4.x. Resolve by trying aliases
# rather than pinning a Blender version.
# ---------------------------------------------------------------------------

ALIASES = {
    "base_color":        ["Base Color"],
    "metallic":          ["Metallic"],
    "roughness":         ["Roughness"],
    "ior":               ["IOR"],
    "alpha":             ["Alpha"],
    "normal":            ["Normal"],
    "transmission":      ["Transmission Weight", "Transmission"],
    "emission_color":    ["Emission Color", "Emission"],
    "emission_strength": ["Emission Strength"],
    "specular":          ["Specular IOR Level", "Specular"],
    "specular_tint":     ["Specular Tint"],
    "anisotropic":       ["Anisotropic"],
    "coat":              ["Coat Weight", "Clearcoat"],
    "coat_roughness":    ["Coat Roughness", "Clearcoat Roughness"],
    "coat_ior":          ["Coat IOR"],
    "sheen":             ["Sheen Weight", "Sheen"],
    "sheen_roughness":   ["Sheen Roughness"],
    "sheen_tint":        ["Sheen Tint"],
    "film_thickness":    ["Thin Film Thickness"],
    "film_ior":          ["Thin Film IOR"],
    "thin_wall":         ["Thin Wall"],
    "subsurface":        ["Subsurface Weight"],
    "subsurface_radius": ["Subsurface Radius"],
    "subsurface_scale":  ["Subsurface Scale"],
    "subsurface_anisotropy": ["Subsurface Anisotropy"],
}

MISSING_SOCKETS = set()


def sock(bsdf, key):
    for name in ALIASES[key]:
        if name in bsdf.inputs:
            return bsdf.inputs[name]
    MISSING_SOCKETS.add(key)
    return None


def set_val(bsdf, key, value):
    s = sock(bsdf, key)
    if s is not None:
        s.default_value = value


def new_material(name, **kwargs):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    for key, value in kwargs.items():
        set_val(bsdf, key, value)
    return mat


def link_texture(mat, key, image_path, non_color=False, normal_map=False):
    tree = mat.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    img = bpy.data.images.load(image_path, check_existing=True)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"

    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.location = (-600, 200)

    if normal_map:
        nm = tree.nodes.new("ShaderNodeNormalMap")
        nm.location = (-300, 200)
        tree.links.new(tex.outputs["Color"], nm.inputs["Color"])
        target = sock(bsdf, "normal")
        if target:
            tree.links.new(nm.outputs["Normal"], target)
        return tex

    target = sock(bsdf, key)
    out = tex.outputs["Alpha"] if key == "alpha" else tex.outputs["Color"]
    if target:
        tree.links.new(out, target)
    return tex


def set_alpha_clip(mat, cutoff=0.5):
    """Produce glTF alphaMode = MASK.

    Since 4.2 the exporter ignores the EEVEE blend_method and infers alphaMode
    from the node graph itself (search_node_tree.detect_alpha_clip). Setting
    material.blend_method = 'CLIP' therefore silently yields BLEND. The pattern
    it actually recognises is a comparison feeding the Alpha socket, so insert
    one between whatever drives Alpha and the BSDF.

    A constant Alpha strictly between 0 and 1 yields BLEND on its own, which is
    what scene 08 relies on -- no node surgery needed there.
    """
    tree = mat.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    alpha_in = sock(bsdf, "alpha")
    if alpha_in is None or not alpha_in.links:
        raise RuntimeError("%s: Alpha must be linked before clipping" % mat.name)

    upstream = alpha_in.links[0].from_socket
    tree.links.remove(alpha_in.links[0])

    cmp_node = tree.nodes.new("ShaderNodeMath")
    cmp_node.operation = "GREATER_THAN"
    cmp_node.location = (-300, -150)
    cmp_node.inputs[1].default_value = cutoff
    tree.links.new(upstream, cmp_node.inputs[0])
    tree.links.new(cmp_node.outputs["Value"], alpha_in)


GLTF_SETTINGS_GROUP = "glTF Material Output"


def add_volume(mat, attenuation_color, attenuation_distance, thickness):
    """Produce glTF KHR_materials_volume.

    The exporter reads attenuation from a Volume Absorption node on the Volume
    output, but bails out entirely unless the material also instantiates a node
    group literally named "glTF Material Output" carrying a Thickness socket
    (see exp/material/extensions/volume.py). Attenuation distance is written as
    1/density, so density is derived rather than set.
    """
    group = bpy.data.node_groups.get(GLTF_SETTINGS_GROUP)
    if group is None:
        group = bpy.data.node_groups.new(GLTF_SETTINGS_GROUP, "ShaderNodeTree")
        group.interface.new_socket("Occlusion", socket_type="NodeSocketFloat")
        group.interface.new_socket("Thickness", socket_type="NodeSocketFloat")
        group.interface.new_socket("Dispersion", socket_type="NodeSocketFloat")
        group.nodes.new("NodeGroupOutput")
        group.nodes.new("NodeGroupInput").location = (-200, 0)

    tree = mat.node_tree
    settings = tree.nodes.new("ShaderNodeGroup")
    settings.node_tree = group
    settings.name = GLTF_SETTINGS_GROUP
    settings.location = (300, -300)
    settings.inputs["Thickness"].default_value = thickness

    out = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL")
    absorb = tree.nodes.new("ShaderNodeVolumeAbsorption")
    absorb.location = (0, -350)
    absorb.inputs["Color"].default_value = attenuation_color
    absorb.inputs["Density"].default_value = 1.0 / attenuation_distance
    tree.links.new(absorb.outputs["Volume"], out.inputs["Volume"])


# ---------------------------------------------------------------------------
# Scene scaffolding
# ---------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False       # determinism over speed

    # Cycles ships two noise-reduction hacks on by default, and both change the
    # image rather than just its variance. Left alone they make the reference
    # something Strelka cannot match by being correct:
    #   blur_glossy (Filter Glossy) = 1.0 widens glossy lobes after a diffuse
    #     bounce, which smears and dims caustics;
    #   sample_clamp_indirect = 10.0 truncates indirect samples, and a caustic is
    #     exactly the high-energy indirect spike that gets truncated.
    # Turning them off costs noise in the reference -- which the harness measures
    # separately as a floor -- and buys a reference that means what it says.
    scene.cycles.blur_glossy = 0.0
    scene.cycles.sample_clamp_indirect = 0.0
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.max_bounces = MAX_DEPTH
    scene.cycles.diffuse_bounces = MAX_DEPTH
    scene.cycles.glossy_bounces = MAX_DEPTH
    scene.cycles.transmission_bounces = MAX_DEPTH
    scene.cycles.transparent_max_bounces = MAX_DEPTH
    # Blender defaults this one to 0, and 0 does not mean "no volumes" -- it means
    # single scattering. Left unset, 18_bounded_volume compared Strelka's multiply
    # scattered medium against a reference that scatters once, which is a
    # difference that grows with the albedo: at 0.05 the two agreed to 1%, at 1.0
    # the reference was 2.4x darker. That is the whole of what the row called a
    # renderer defect.
    scene.cycles.volume_bounces = MAX_DEPTH
    scene.cycles.seed = 0

    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    # Linear, untransformed output. AgX or Filmic here would make every
    # comparison meaningless.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"

    # Black world: Strelka has no environment unless the sidecar names one, so
    # Cycles must not have one either.
    #
    # `use_nodes = False` does NOT do this on Blender 5.2 -- it reads back True and
    # Cycles keeps shading through the node tree, whose default Background node is
    # 0.05 grey. That silently added a*0.05 of outgoing radiance to every surface
    # in every reference render, which is a purely additive term with no structure,
    # so it read as a uniform ~5% "Strelka is dim" offset across the whole ladder
    # rather than as a harness bug. Drive the node tree explicitly instead.
    world = bpy.data.worlds.new("World")
    world.use_nodes = False
    world.color = (0.0, 0.0, 0.0)
    if world.node_tree is not None:
        for node in world.node_tree.nodes:
            if node.type == "BACKGROUND":
                node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                node.inputs["Strength"].default_value = 0.0
    scene.world = world

    # Assert it rather than trust it. The same class of trap already bit this file
    # once via set_sun_strength(), and a silently-emitting world invalidates every
    # reference image at once.
    if world.node_tree is not None:
        for node in world.node_tree.nodes:
            if node.type == "BACKGROUND":
                c = node.inputs["Color"].default_value
                s = node.inputs["Strength"].default_value
                if s != 0.0 and (c[0] or c[1] or c[2]):
                    raise RuntimeError(
                        "world Background still emits (colour %s, strength %s) -- "
                        "every reference render would be contaminated" % (list(c), s))
    return scene


def camera_for_scene(name):
    return SCENE_CAMERAS.get(name, (CAM_LOC, CAM_TARGET, CAM_FOV_DEG))


def add_camera(name=None):
    location, target, fov = camera_for_scene(name)
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(fov)
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location

    direction = (
        target[0] - location[0],
        target[1] - location[1],
        target[2] - location[2],
    )
    from mathutils import Vector
    cam.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def add_key_light():
    light_data = bpy.data.lights.new("Key", type="AREA")
    light_data.shape = "SQUARE"
    light_data.size = KEY_SIZE
    light_data.energy = KEY_POWER_W
    light_data.color = (1.0, 1.0, 1.0)
    light = bpy.data.objects.new("Key", light_data)
    bpy.context.collection.objects.link(light)
    light.location = KEY_POS
    light.rotation_euler = (0.0, 0.0, 0.0)   # area lights emit along -Z
    return light


def add_stage():
    """Neutral floor + back wall so metal and glass have something to sample."""
    grey = new_material("stage_grey", base_color=(0.35, 0.35, 0.35, 1.0),
                        roughness=0.85, metallic=0.0)

    bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "Floor"
    floor.data.materials.append(grey)

    bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0.0, 3.2, 0.0),
                                     rotation=(math.radians(90), 0.0, 0.0))
    wall = bpy.context.object
    wall.name = "BackWall"
    wall.data.materials.append(grey)
    return floor, wall


def sphere(name, location, radius=0.42):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32,
                                         radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.shade_smooth()
    return obj


def quad(name, location, size=1.0, rotation=(math.radians(90), 0.0, 0.0)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    return obj


def row_positions(count, spacing=0.95, z=0.55):
    span = (count - 1) * spacing
    return [(-span / 2 + i * spacing, 0.0, z) for i in range(count)]


# ---------------------------------------------------------------------------
# Feature scenes
# ---------------------------------------------------------------------------

def set_sun_strength(light, watts_per_m2):
    """Set a sun's strength so Cycles and the glTF exporter agree on it.

    They do not read the same property. With Cycles active the exporter takes
    the sun's intensity from the lamp's Emission node Strength and ignores
    light.energy entirely (blender/exp/lights.py __gather_intensity: the
    emission-node branch is skipped only when the engine is BLENDER_EEVEE).
    Cycles itself multiplies both. So setting energy alone renders at the value
    you asked for and exports 1.0 * 683 lux -- silently a different sun on each
    side.

    Driving the node and pinning energy to 1.0 makes the product equal the node
    value, which is exactly what gets exported. Point and spot lamps are not
    affected: their branch falls back to light.energy (with a warning about a
    missing LightFalloff node), which is why they round-trip correctly.
    """
    tree = getattr(light, "node_tree", None)
    emission = None
    if tree is not None:
        emission = next((n for n in tree.nodes if n.type == "EMISSION"), None)
    if emission is not None:
        light.energy = 1.0
        emission.inputs["Strength"].default_value = watts_per_m2
    else:
        light.energy = watts_per_m2


def s00_calibration(tex):
    add_stage()
    obj = sphere("Grey", (0.0, 0.0, 0.6), radius=0.6)
    obj.data.materials.append(
        new_material("grey018", base_color=(0.18, 0.18, 0.18, 1.0),
                     roughness=1.0, metallic=0.0, specular=0.0))


def s01_srgb_texture(tex):
    add_stage()
    obj = quad("Probe", (0.0, 0.6, 0.9), size=2.2)
    mat = new_material("srgb_probe", roughness=1.0, metallic=0.0, specular=0.0)
    link_texture(mat, "base_color", tex["srgb"])
    obj.data.materials.append(mat)


def s02_basecolor(tex):
    add_stage()
    colors = [(0.8, 0.1, 0.1), (0.1, 0.7, 0.2), (0.1, 0.2, 0.8),
              (0.8, 0.75, 0.1), (0.75, 0.75, 0.75)]
    for i, (pos, c) in enumerate(zip(row_positions(len(colors)), colors)):
        obj = sphere("BC%d" % i, pos)
        obj.data.materials.append(
            new_material("bc%d" % i, base_color=c + (1.0,),
                         roughness=1.0, metallic=0.0, specular=0.0))


def s03_roughness(tex):
    add_stage()
    n = 7
    for i, pos in enumerate(row_positions(n)):
        r = i / (n - 1)
        obj = sphere("R%d" % i, pos)
        obj.data.materials.append(
            new_material("rough%d" % i, base_color=(0.25, 0.25, 0.28, 1.0),
                         roughness=r, metallic=0.0))


def s04_metal(tex):
    add_stage()
    n = 7
    for i, pos in enumerate(row_positions(n)):
        r = i / (n - 1)
        obj = sphere("M%d" % i, pos)
        obj.data.materials.append(
            new_material("metal%d" % i, base_color=(0.95, 0.78, 0.35, 1.0),
                         roughness=max(0.02, r), metallic=1.0))


def s05_anisotropy(tex):
    add_stage()
    for i, (pos, a) in enumerate(zip(row_positions(3, spacing=1.15), (0.0, 0.5, 0.9))):
        obj = sphere("A%d" % i, pos, radius=0.48)
        obj.data.materials.append(
            new_material("aniso%d" % i, base_color=(0.9, 0.9, 0.92, 1.0),
                         roughness=0.25, metallic=1.0, anisotropic=a))


def s06_normalmap(tex):
    add_stage()
    for i, (pos, scale) in enumerate(zip(row_positions(2, spacing=1.6, z=0.9), (0.0, 1.0))):
        obj = quad("N%d" % i, pos, size=1.35)
        mat = new_material("nmap%d" % i, base_color=(0.6, 0.6, 0.62, 1.0),
                           roughness=0.35, metallic=0.0)
        if scale > 0.0:
            link_texture(mat, "normal", tex["normal"], non_color=True, normal_map=True)
        obj.data.materials.append(mat)


def s07_alpha_clip(tex):
    add_stage()
    obj = quad("Clip", (0.0, 0.4, 0.9), size=2.0)
    mat = new_material("alpha_clip", roughness=0.6, metallic=0.0)
    link_texture(mat, "base_color", tex["alpha"])
    link_texture(mat, "alpha", tex["alpha"])
    set_alpha_clip(mat, cutoff=0.5)
    obj.data.materials.append(mat)


def s08_alpha_blend(tex):
    add_stage()
    sphere("Behind", (0.0, 1.2, 0.6)).data.materials.append(
        new_material("behind", base_color=(0.85, 0.2, 0.15, 1.0), roughness=0.9))
    obj = quad("Blend", (0.0, -0.2, 0.9), size=2.0)
    # A constant Alpha of 0.45 is enough: the exporter reads the value, not the
    # EEVEE blend mode, and anything strictly between 0 and 1 becomes BLEND.
    mat = new_material("alpha_blend", base_color=(0.3, 0.6, 0.9, 1.0),
                       roughness=0.4, metallic=0.0, alpha=0.45)
    obj.data.materials.append(mat)


def s09_glass_ior(tex):
    add_stage()
    for i, (pos, ior) in enumerate(zip(row_positions(3, spacing=1.15), (1.1, 1.5, 2.0))):
        obj = sphere("G%d" % i, pos, radius=0.48)
        obj.data.materials.append(
            new_material("glass%d" % i, base_color=(1.0, 1.0, 1.0, 1.0),
                         roughness=0.0, metallic=0.0, transmission=1.0, ior=ior))


def s10_glass_absorption(tex):
    """Colored glass, exported as KHR_materials_volume.

    Strelka has no volume absorption at all today, so this is expected to
    differ; it exists to measure by how much, and to stop differing once
    absorption lands."""
    add_stage()
    for i, (pos, dist) in enumerate(zip(row_positions(3, spacing=1.15), (0.15, 0.5, 2.0))):
        obj = sphere("V%d" % i, pos, radius=0.48)
        mat = new_material("absorb%d" % i, base_color=(1.0, 1.0, 1.0, 1.0),
                           roughness=0.0, metallic=0.0, transmission=1.0, ior=1.5)
        add_volume(mat, attenuation_color=(0.15, 0.55, 0.35, 1.0),
                   attenuation_distance=dist, thickness=0.96)
        obj.data.materials.append(mat)


def s11_emission(tex):
    add_stage()
    for i, (pos, strength) in enumerate(zip(row_positions(3, spacing=1.15, z=0.7),
                                            (1.0, 5.0, 20.0))):
        obj = quad("E%d" % i, pos, size=0.7)
        obj.data.materials.append(
            new_material("emit%d" % i, base_color=(0.02, 0.02, 0.02, 1.0),
                         roughness=1.0,
                         emission_color=(1.0, 0.85, 0.55, 1.0),
                         emission_strength=strength))


def s12_lights_punctual(tex):
    """The only scene with no light sidecar: it exercises KHR_lights_punctual,
    which the loader falls back to when no *_light.json exists."""
    add_stage()
    obj = sphere("Probe", (0.0, 0.0, 0.6), radius=0.6)
    obj.data.materials.append(
        new_material("probe", base_color=(0.6, 0.6, 0.6, 1.0),
                     roughness=0.7, metallic=0.0))

    pt = bpy.data.lights.new("Point", type="POINT")
    pt.energy = 500.0
    pt.shadow_soft_size = 0.0
    pt.color = (1.0, 0.4, 0.3)
    o = bpy.data.objects.new("Point", pt)
    bpy.context.collection.objects.link(o)
    o.location = (-1.9, -1.6, 1.9)

    sp = bpy.data.lights.new("Spot", type="SPOT")
    sp.energy = 900.0
    sp.shadow_soft_size = 0.0
    sp.spot_size = math.radians(45.0)
    sp.spot_blend = 0.25
    sp.color = (0.35, 0.55, 1.0)
    o = bpy.data.objects.new("Spot", sp)
    bpy.context.collection.objects.link(o)
    o.location = (1.9, -1.6, 2.4)
    from mathutils import Vector
    o.rotation_euler = Vector((-1.9, 1.6, -2.4)).to_track_quat("-Z", "Y").to_euler()

    sun = bpy.data.lights.new("Sun", type="SUN")
    set_sun_strength(sun, 1.5)
    sun.angle = math.radians(0.53)
    o = bpy.data.objects.new("Sun", sun)
    bpy.context.collection.objects.link(o)
    o.location = (0.0, -2.0, 4.0)
    o.rotation_euler = (math.radians(35.0), 0.0, math.radians(20.0))


def s13_uv2_vcol(tex):
    add_stage()
    bpy.ops.mesh.primitive_cube_add(size=1.1, location=(-0.75, 0.0, 0.7))
    cube = bpy.context.object
    cube.name = "VCol"
    me = cube.data
    me.uv_layers.new(name="UVMap2")
    attr = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
    for i, d in enumerate(attr.data):
        d.color = ((i % 3) / 2.0, ((i // 3) % 3) / 2.0, ((i // 9) % 3) / 2.0, 1.0)

    mat = new_material("vcol", roughness=0.9, metallic=0.0)
    tree = mat.node_tree
    node = tree.nodes.new("ShaderNodeVertexColor")
    node.layer_name = "Col"
    tree.links.new(node.outputs["Color"], sock(tree.nodes["Principled BSDF"], "base_color"))
    cube.data.materials.append(mat)

    obj = quad("UV2", (0.85, 0.4, 0.8), size=1.2)
    mat2 = new_material("uv2", roughness=1.0, metallic=0.0)
    link_texture(mat2, "base_color", tex["srgb"])
    obj.data.materials.append(mat2)


def s14_sheen(tex):
    """Sheen roughness ramp on a dark base.

    Read this row differently from the others. Cycles' Principled uses Zeltner et
    al.'s microflake sheen; Strelka implements the Charlie distribution with
    Ashikhmin visibility, because that is what KHR_materials_sheen is specified
    against. The two are different models of the same phenomenon, so this row
    measures how far apart they are, not whether one of them is wrong -- which
    makes it a regression guard on our side and not an agreement check.

    Dark base and full sheen weight on purpose: with a lit base underneath, most
    of what the comparison sees is the base agreeing with itself.
    """
    add_stage()
    n = 7
    for i, pos in enumerate(row_positions(n)):
        r = i / (n - 1)
        obj = sphere("S%d" % i, pos)
        obj.data.materials.append(
            new_material("sheen%d" % i, base_color=(0.04, 0.04, 0.05, 1.0),
                         roughness=0.6, metallic=0.0, specular=0.1,
                         sheen=1.0, sheen_roughness=max(0.05, r),
                         sheen_tint=(1.0, 1.0, 1.0, 1.0)))


def s15_clearcoat(tex):
    """Coat IOR ramp at full coat weight.

    Cycles' Coat is a layered dielectric with its own IOR that darkens what is
    under it, which is the same model Strelka implements, so unlike the sheen row
    this one is a real agreement check.

    The ramp starts at IOR 1.0, where the coat's F0 is exactly zero and the layer
    has to vanish. That end is the control: if the first sphere differs, the
    disagreement is in the layering rather than in the Fresnel.
    """
    add_stage()
    n = 7
    for i, pos in enumerate(row_positions(n)):
        ior = 1.0 + 1.2 * (i / (n - 1))
        obj = sphere("C%d" % i, pos)
        obj.data.materials.append(
            new_material("coat%d" % i, base_color=(0.55, 0.18, 0.16, 1.0),
                         roughness=0.35, metallic=0.0,
                         coat=1.0, coat_roughness=0.08, coat_ior=ior))


def s16_iridescence(tex):
    """Thin-film thickness ramp on a smooth dielectric.

    Isolated on purpose: no coat, no transmission, so what varies across the row
    is the film and nothing else. Blender's Principled has carried Thin Film
    Thickness and IOR since 4.2, and Cycles evaluates the same Airy summation
    Strelka does, so this is a real agreement check rather than a comparison of
    two different models.
    """
    add_stage()
    n = 7
    for i, pos in enumerate(row_positions(n)):
        thickness = 200.0 + 600.0 * (i / (n - 1))
        obj = sphere("F%d" % i, pos)
        obj.data.materials.append(
            new_material("film%d" % i, base_color=(0.02, 0.02, 0.02, 1.0),
                         roughness=0.08, metallic=0.0,
                         film_thickness=thickness, film_ior=1.4))


def s17_coated_glass(tex):
    """Transmission and a clearcoat on the same material.

    This configuration, and only this configuration, hid a double-count for as
    long as both features have existed: the separate specular lobe's *selection
    weight* is zeroed for a transmissive material -- the transmission lobe runs
    its own Fresnel -- while its BRDF was still summed into f_total. With no
    other reflection lobe selectable the specular term is simply never evaluated,
    which is why plain glass never showed it. Add a coat and the coat lobe gets
    selected, the specular term rides along, and it is divided by a pdf that does
    not include it.

    It shipped as soap bubbles that glowed instead of being transparent.
    """
    add_stage()
    n = 5
    for i, pos in enumerate(row_positions(n)):
        coat = i / (n - 1)
        obj = sphere("G%d" % i, pos)
        obj.data.materials.append(
            new_material("coatglass%d" % i, base_color=(1.0, 1.0, 1.0, 1.0),
                         roughness=0.05, metallic=0.0, ior=1.5,
                         transmission=1.0, coat=coat, coat_roughness=0.03))


def s18_bounded_volume(tex):
    """A scattering medium bounded by a box.

    Cycles' Principled Volume splits its Density into scattering and absorption
    by Color -- so Density is the extinction and Color is the single-scattering
    albedo, which is exactly what STRELKA_materials_medium carries and why the
    two can be compared at all without a fit.

    No emission: Cycles adds volume emission with its own coefficient and Strelka
    adds it per free-flight event, and the two conventions do not line up. What
    this row does test is free flight, the scattering albedo, the phase function
    and the boundary crossing -- the bulk of the feature.
    """
    add_stage()
    bpy.ops.mesh.primitive_cube_add(size=1.3, location=(0.0, 0.0, 0.75))
    obj = bpy.context.active_object
    obj.name = "FogBox"

    mat = bpy.data.materials.new("medium")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    vol = tree.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.75, 0.82, 0.95, 1.0)  # single-scattering albedo
    vol.inputs["Density"].default_value = 2.5                     # extinction, per unit
    vol.inputs["Anisotropy"].default_value = 0.0
    vol.inputs["Emission Strength"].default_value = 0.0
    tree.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    obj.data.materials.append(mat)


def s19_env_and_light(tex):
    """A textured environment and an area light, lighting the same surfaces.

    The rung the ladder was missing, and the reason docs/open-defects.md entry 6
    could not be settled. RIS and plain next-event estimation agree exactly on
    every other row here, and they must: resampling among candidates is a no-op
    when they all come from one light, and evidently faithful with three punctual
    ones. What no other row has is *two kinds* of light at once, which is where
    `connectToLight` splits its draw -- half the samples to the environment, half
    to the analytic lights -- and where the two estimators can differ.

    The sky is a Nishita model with the sun disc off. Off because a disc is a
    near-delta source in an environment map: it converges slowly on both sides
    and would make this rung measure variance rather than bias. What is left is
    smooth and strongly non-uniform -- an order of magnitude between the bright
    side and the dark one -- which is exactly the distribution environment
    importance sampling exists for, and the thing a uniform sky cannot test.

    Three roughnesses, because the split matters differently to a diffuse lobe
    (which sees both lights as area) and a near-specular one (which sees the sky
    as texture and the rect light as a highlight).
    """
    world = bpy.data.worlds.new("SkyWorld")
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    # The physical sky, whatever this Blender calls it. 5.x renamed Nishita to
    # MULTIPLE_SCATTERING and there is no alias, so pick from what the enum
    # actually offers rather than from what the manual said last year.
    types = sky.bl_rna.properties["sky_type"].enum_items.keys()
    sky.sky_type = next(t for t in ("MULTIPLE_SCATTERING", "NISHITA", "HOSEK_WILKIE") if t in types)
    sky.sun_elevation = math.radians(28.0)
    sky.sun_rotation = math.radians(135.0)
    sky.sun_disc = False
    sky.altitude = 0.0
    sky.air_density = 1.0
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = world

    add_stage()
    for pos, rough in zip(row_positions(3), (0.05, 0.3, 1.0)):
        obj = sphere("s_%.2f" % rough, pos)
        obj.data.materials.append(
            new_material("env_rough_%.2f" % rough, base_color=(0.5, 0.5, 0.5, 1.0),
                         roughness=rough, metallic=0.0))


def s20_mirror_and_floor(tex):
    """A large mirror facing a rough textured floor.

    Not a shading test -- every lobe in it is already covered by rows 03 and 04 --
    but a *denoiser* test, and the one docs/open-defects.md entry 7 says it needs.
    That entry measures `render.guide_primary_hit` on the bathroom, where the
    mirrors are small, and finds it worth about a fifth at low sample counts. The
    walk it replaces exists for the opposite case: a primary hit that *is* a
    mirror, where taking the guides at the surface hands the denoiser a
    featureless black albedo instead of the world being reflected.

    So this is that case, deliberately: the mirror fills most of the frame, and
    what it reflects is a rough floor with structure in it. Comparing the two
    guide sources here against a converged Strelka render is what says whether
    the walk should stay the default. The Cycles reference comes for free and
    keeps the row honest about the shading underneath.
    """
    grey = new_material("floor_rough", base_color=(0.45, 0.42, 0.38, 1.0),
                        roughness=0.75, metallic=0.0)
    bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "Floor"
    floor.data.materials.append(grey)
    link_texture(grey, "base_color", tex["srgb"])

    # The mirror stands where the back wall would be and is the whole background.
    mirror = new_material("mirror", base_color=(0.95, 0.95, 0.95, 1.0),
                          roughness=0.02, metallic=1.0)
    bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0.0, 2.6, 1.6),
                                     rotation=(math.radians(90), 0.0, 0.0))
    wall = bpy.context.object
    wall.name = "Mirror"
    wall.data.materials.append(mirror)

    # Something for it to reflect that is not the floor.
    for pos, rough in zip(row_positions(3), (0.15, 0.5, 0.9)):
        obj = sphere("s_%.2f" % rough, pos)
        obj.data.materials.append(
            new_material("ball_%.2f" % rough, base_color=(0.6, 0.25, 0.2, 1.0),
                         roughness=rough, metallic=0.0))


def s21_specular_color(tex):
    """KHR_materials_specular colour ramp on a grey dielectric.

    Specular IOR Level is pinned at 0.5 -- Blender's glTF exporter multiplies the
    tint by (level / 0.5), so level 1.0 would bake a factor of two into
    specularColorFactor and the row would measure that encoding rather than the
    tint. With level 0.5 the exported colour is the tint itself.
    """
    add_stage()
    tints = [
        (1.0, 1.0, 1.0),
        (1.0, 0.55, 0.35),
        (1.0, 0.2, 0.15),
        (0.95, 0.85, 0.2),
        (0.25, 0.85, 0.35),
        (0.2, 0.45, 1.0),
        (0.75, 0.3, 0.95),
    ]
    for i, (pos, tint) in enumerate(zip(row_positions(len(tints)), tints)):
        obj = sphere("Sp%d" % i, pos)
        obj.data.materials.append(
            new_material("spec%d" % i, base_color=(0.18, 0.18, 0.2, 1.0),
                         roughness=0.18, metallic=0.0,
                         specular=0.5, specular_tint=tint + (1.0,)))


def s22_thin_walled(tex):
    """Thin-walled glass: smooth + roughness ramp, vs one solid control.

    A low-frequency striped card sits behind the row so frosted sheets have
    structure to smear; the full-wall sRGB probe was too fine and made every
    column look worse than the blur itself. The ramp stops at 0.45: above that
    Cycles' multiscatter GGX and our single-scatter diverge on energy. The
    solid sphere at IOR 1.5 is the control -- same material without the wall
    flag -- so a framing or exposure shift cannot hide a missing patch.
    """
    add_stage()
    # Soft vertical bands behind the spheres. High-frequency texture on the
    # whole back wall made relative error measure registration noise instead of
    # the lobe.
    colours = [
        (0.75, 0.15, 0.12),
        (0.85, 0.85, 0.80),
        (0.12, 0.45, 0.75),
        (0.85, 0.85, 0.80),
        (0.15, 0.65, 0.25),
    ]
    span = 3.6
    strip_w = span / len(colours)
    for i, col in enumerate(colours):
        x = -span / 2 + strip_w * (i + 0.5)
        strip = quad("Stripe%d" % i, (x, 2.4, 0.9), size=1.0)
        strip.scale = (strip_w * 0.98, 1.6, 1.0)
        strip.data.materials.append(
            new_material("stripe%d" % i, base_color=col + (1.0,),
                         roughness=1.0, metallic=0.0, specular=0.0))

    specs = [
        ("thin0", 0.0, True),
        ("thin1", 0.15, True),
        ("thin2", 0.3, True),
        ("thin3", 0.45, True),
        ("solid1", 0.0, False),
    ]
    for (name, rough, thin), pos in zip(specs, row_positions(len(specs), spacing=1.05)):
        obj = sphere(name, pos, radius=0.42)
        obj.data.materials.append(
            new_material(name, base_color=(1.0, 1.0, 1.0, 1.0),
                         roughness=rough, metallic=0.0,
                         transmission=1.0, ior=1.5, thin_wall=thin))


def s23_diffuse_transmission(tex):
    """Diffuse transmission weight ramp, backlit.

    Principled in Blender 5.2 has no Diffuse Transmission socket, so Cycles sees
    a Mix of Principled and Translucent BSDF. The exporter cannot write
    KHR_materials_diffuse_transmission from that graph; the patcher does, with
    the same weights and colours. A bright panel behind the cards is what makes
    the lobe visible -- front lighting alone looks like a darker diffuse.
    """
    add_stage()
    # Backlight panel, behind the row.
    panel = quad("Backlight", (0.0, 1.35, 0.85), size=2.4)
    panel.data.materials.append(
        new_material("backlight", base_color=(0.02, 0.02, 0.02, 1.0),
                     roughness=1.0,
                     emission_color=(1.0, 0.95, 0.85, 1.0),
                     emission_strength=12.0))

    n = 5
    for i, pos in enumerate(row_positions(n, spacing=0.95, z=0.85)):
        weight = i / (n - 1)
        # Cards face the camera (and the backlight behind them).
        bpy.ops.mesh.primitive_plane_add(
            size=0.7, location=(pos[0], -0.15, pos[2]),
            rotation=(math.radians(90), 0.0, 0.0))
        obj = bpy.context.object
        obj.name = "DT%d" % i
        colour = (0.55, 0.75, 0.35)
        mat = bpy.data.materials.new("dtrans%d" % i)
        mat.use_nodes = True
        tree = mat.node_tree
        tree.nodes.clear()
        out = tree.nodes.new("ShaderNodeOutputMaterial")
        mix = tree.nodes.new("ShaderNodeMixShader")
        princ = tree.nodes.new("ShaderNodeBsdfPrincipled")
        transl = tree.nodes.new("ShaderNodeBsdfTranslucent")
        princ.inputs["Base Color"].default_value = colour + (1.0,)
        princ.inputs["Roughness"].default_value = 1.0
        princ.inputs["Metallic"].default_value = 0.0
        if "Specular IOR Level" in princ.inputs:
            princ.inputs["Specular IOR Level"].default_value = 0.0
        transl.inputs["Color"].default_value = colour + (1.0,)
        mix.inputs["Fac"].default_value = weight
        tree.links.new(princ.outputs["BSDF"], mix.inputs[1])
        tree.links.new(transl.outputs["BSDF"], mix.inputs[2])
        tree.links.new(mix.outputs["Shader"], out.inputs["Surface"])
        obj.data.materials.append(mat)


def s24_orthographic(tex):
    """Calibration twin under an orthographic camera.

    Same 0.18 grey sphere and key light as 00 -- only the projection changes. If
    this row drifts while 00 stays at ratio ≈1, the bug is in orthographic
    extents or the ray generation, not in light units.
    """
    cam = bpy.context.scene.camera
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = CAM_ORTHO_SCALE
    add_stage()
    obj = sphere("Grey", (0.0, 0.0, 0.6), radius=0.6)
    obj.data.materials.append(
        new_material("grey018", base_color=(0.18, 0.18, 0.18, 1.0),
                     roughness=1.0, metallic=0.0, specular=0.0))


def diffuse_to_single_scattering_albedo(a):
    """Diffuse albedo -> single-scattering albedo (Van de Hulst inversion).

    Same fit tools/iso_bathroom/vray2strelka.py uses for STRELKA_materials_subsurface:
    a DCC subsurface colour is a diffuse albedo, and the random walk wants the
    probability that one extinction event scatters. Feeding the diffuse value
    straight in darkens every multi-scatter path by albedo^n.
    """

    def diffuse_albedo(alpha):
        s = math.sqrt(max(1.0 - alpha, 0.0))
        return (1.0 - s) * (1.0 - 0.139 * s) / (1.0 + 1.17 * s)

    a = min(max(float(a), 0.0), 0.999)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if diffuse_albedo(mid) < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Mean free path in metres for the SSS row. Kept small against a ~0.5 m sphere so
# the walk is visible at the silhouette without turning the whole ball into a
# uniform glow; matching Cycles' Subsurface Radius * Scale.
SSS_RADIUS = (0.05, 0.025, 0.015)
SSS_SCALE = 1.0


def s25_subsurface(tex):
    """Subsurface random-walk albedo ramp, with the diffuse→σ_s recipe applied.

    Cycles authors a diffuse subsurface colour; Strelka's extension carries the
    single-scattering albedo. The patcher inverts Van de Hulst so both sides are
    asked the same physical question. Radius is identical on both sides.
    """
    add_stage()
    # Saturated and grey albedos: colour shift through the medium is half the
    # feature, and a grey control says whether the weight itself matches.
    albedos = [
        (0.55, 0.22, 0.18),
        (0.75, 0.45, 0.28),
        (0.85, 0.75, 0.55),
        (0.55, 0.55, 0.55),
        (0.35, 0.55, 0.75),
    ]
    for i, (pos, albedo) in enumerate(zip(row_positions(len(albedos)), albedos)):
        obj = sphere("SSS%d" % i, pos, radius=0.48)
        mat = new_material(
            "sss%d" % i,
            base_color=albedo + (1.0,),
            roughness=1.0,
            metallic=0.0,
            specular=0.0,
            subsurface=1.0,
            subsurface_radius=SSS_RADIUS,
            subsurface_scale=SSS_SCALE,
            subsurface_anisotropy=0.0,
        )
        obj.data.materials.append(mat)


# Focus distance from the shared camera to the stage centre. The DOF row puts
# its middle sphere there so the in-focus plane is a known, measurable place.
DOF_FOCUS_DISTANCE = math.sqrt(
    (CAM_TARGET[0] - CAM_LOC[0]) ** 2
    + (CAM_TARGET[1] - CAM_LOC[1]) ** 2
    + (CAM_TARGET[2] - CAM_LOC[2]) ** 2
)
# Vertical FOV 45° on a 24 mm sensor → the lens length Strelka's thin-lens
# radius formula wants. Blender's FOV-mode camera still carries this as cam.lens
# once the angle is set; the sidecar writes it so both sides share one number.
DOF_FOCAL_LENGTH_MM = 24.0 / (2.0 * math.tan(math.radians(CAM_FOV_DEG) * 0.5))
DOF_FSTOP = 2.0


def s26_dof(tex):
    """Thin-lens depth of field against three grey spheres at different depths.

    Same material on every sphere so the comparison is pure defocus: the middle
    one sits on the focus plane, the near and far ones measure the blur.
    """
    add_stage()
    grey = new_material("dof_grey", base_color=(0.55, 0.55, 0.55, 1.0),
                        roughness=0.65, metallic=0.0)
    # Depths along the camera look direction (Blender Y). The camera sits at
    # Y=-4.6 looking at Y=0; near/mid/far are spaced so the blur is obvious at
    # f/2 without the near sphere leaving the frame.
    for i, (name, y) in enumerate([("near", -1.1), ("mid", 0.0), ("far", 1.4)]):
        obj = sphere("Dof%s" % name.capitalize(), (0.0, y, 0.75), radius=0.42)
        obj.data.materials.append(grey)

    cam = bpy.context.scene.camera
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = DOF_FOCUS_DISTANCE
    cam.data.dof.aperture_fstop = DOF_FSTOP
    cam.data.dof.aperture_blades = 0


IES_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                         "philips_cdm_r111_35w_24deg.ies")


def install_ies(path):
    """Put the row's photometric file in place and report its peak candela.

    Prefers the real Philips CDM-R111 reflector in assets/ over the synthetic
    curve below, for the reason spelled out in assets/README.md: a synthetic
    cos^4 falloff renders as a slightly vignetted point light, and so does every
    way of mis-reading it. A 24-degree beam does not.
    """
    if os.path.exists(IES_ASSET):
        with open(IES_ASSET, "rb") as src, open(path, "wb") as dst:
            dst.write(src.read())
        peak = 0.0
        with open(IES_ASSET, "r", errors="replace") as f:
            for token in f.read().split():
                try:
                    peak = max(peak, float(token))
                except ValueError:
                    pass
        return peak
    return write_synthetic_ies(path)


def write_synthetic_ies(path):
    """A batwing luminaire with an asymmetric azimuth, for the IES row.

    Deliberately not a hotspot on the axis. The row used to carry a pure
    cos^4 curve, which is a perfectly good photometric file and a terrible test:
    a cos^4 point light and a bare point light differ by a smooth falloff, so
    the render looked like an ordinary lamp and any of the ways the table could
    have been misread -- dropped, transposed, folded wrongly, extrapolated --
    would still have looked like an ordinary lamp.

    Two features fix that, and each one fails visibly if it is not honoured:

      * Vertically it is a batwing: a dip at nadir and a peak near 40 degrees,
        which paints a bright *ring* on the floor with a dark centre. Nothing a
        point light or a spot cone can produce.
      * Azimuthally it is elongated -- brightest across one axis, dimmest across
        the other -- tabulated for a single quadrant, which is the LM-63
        symmetry that has to be *mirrored* into the other three rather than
        repeated. Getting that wrong rotates the pattern by 90 degrees, which is
        obvious in the image and was not obvious in the numbers.

    The elongation also puts light on the back wall, so the row stops being a
    picture of a floor.
    """
    vertical = [0.0, 10.0, 20.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    # One quadrant; LM-63 mirrors it about both vertical planes.
    horizontal = [0.0, 22.5, 45.0, 67.5, 90.0]

    def vertical_shape(deg):
        # Batwing: a lobe centred off-axis, over a small pedestal so the centre
        # is dark rather than black.
        lobe = math.exp(-(((deg - 40.0) / 15.0) ** 2))
        pedestal = 0.12 * math.cos(math.radians(deg)) ** 2
        return max(0.0, lobe + pedestal)

    def azimuth_shape(deg):
        # 1.0 across the 0-180 axis, 0.3 across 90-270.
        return 0.3 + 0.7 * math.cos(math.radians(deg)) ** 2

    peak = 1000.0
    columns = []
    for h in horizontal:
        columns.append([peak * vertical_shape(v) * azimuth_shape(h) for v in vertical])

    with open(path, "w") as f:
        f.write("IESNA:LM-63-2002\n")
        f.write("[TEST] STRELKA-FEATURE-27\n")
        f.write("[MANUFAC] Synthetic\n")
        f.write("[LUMINAIRE] Batwing, quadrant symmetry\n")
        f.write("TILT=NONE\n")
        # lamps lumens multiplier nV nH phototype units w l h
        f.write("1 -1 1.0 %d %d 1 1 0 0 0\n" % (len(vertical), len(horizontal)))
        f.write("1.0 1.0 10\n")
        f.write(" ".join("%.1f" % v for v in vertical) + "\n")
        f.write(" ".join("%.1f" % h for h in horizontal) + "\n")
        # LM-63 order: every vertical angle of the first horizontal plane, then
        # the next plane, and so on.
        for column in columns:
            f.write(" ".join("%.3f" % c for c in column) + "\n")
    return max(max(c) for c in columns)


def s27_ies(tex):
    """IES point light painting a photometric hotspot on the stage floor.

    No rect key: the IES light is the whole of the lighting, so a mismatch in
    the angular distribution cannot hide under a second source. Cycles gets the
    same .ies through a TexIES node; Strelka gets it through the light sidecar.
    """
    add_stage()
    probe = sphere("Probe", (0.0, 0.0, 0.55), radius=0.45)
    probe.data.materials.append(
        new_material("ies_probe", base_color=(0.55, 0.55, 0.55, 1.0),
                     roughness=0.7, metallic=0.0))

    # Point lamp for Cycles. Cycles renders the *product* of the lamp's energy
    # and the Emission strength its node graph produces, so energy has to be
    # pinned even though the IES table is what shapes the light: a new lamp
    # defaults to 10 W, and leaving it there scales the reference by ten while
    # every other number in the scene looks right. That is the same trap the sun
    # strength note below the table describes, and it cost a plausible-looking
    # fitted constant in the renderer before it was found.
    light_data = bpy.data.lights.new("IESKey", type="POINT")
    light_data.use_nodes = True
    light_data.energy = 1.0
    light_data.shadow_soft_size = 0.0
    light = bpy.data.objects.new("IESKey", light_data)
    bpy.context.collection.objects.link(light)
    # Above the stage, aimed down. Point lights have no orientation in Blender,
    # but TexIES uses the local −Z of the object; rotate so −Z points at the
    # floor (same convention as Strelka's photometric axis).
    light.location = (0.0, 0.0, 3.2)
    light.rotation_euler = (0.0, 0.0, 0.0)

    nt = light_data.node_tree
    nt.nodes.clear()
    ies = nt.nodes.new("ShaderNodeTexIES")
    ies.mode = "EXTERNAL"
    # filepath filled in by the main loop once the scene directory exists
    emission = nt.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputLight")
    nt.links.new(ies.outputs[0], emission.inputs["Strength"])
    nt.links.new(emission.outputs[0], out.inputs[0])
    light["strelka_ies_node"] = ies.name  # stash for the exporter step
    return light


HAIR_COLOR = (0.42, 0.22, 0.10, 1.0)
HAIR_ROUGHNESS = 0.35
HAIR_RADIAL = 0.35


def new_hair_material(name, color=HAIR_COLOR, roughness=HAIR_ROUGHNESS, radial=HAIR_RADIAL):
    """Principled Hair (Chiang, Direct Coloring) for the Cycles reference.

    The glTF export still sees a Principled BSDF stand-in so the pigment colour
    and roughness reach the file; STRELKA_materials_hair is patched on afterwards
    and is what flips Strelka onto the Chiang lobe.
    """
    # Stand-in the exporter understands: pigment + roughness + IOR 1.55.
    mat = new_material(name, base_color=color, roughness=roughness, metallic=0.0,
                       ior=1.55, specular=0.5)
    # Cycles reference: replace the tree with Principled Hair.
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    hair = tree.nodes.new("ShaderNodeBsdfHairPrincipled")
    hair.model = "CHIANG"
    hair.parametrization = "COLOR"
    hair.inputs["Color"].default_value = color
    hair.inputs["Roughness"].default_value = roughness
    hair.inputs["Radial Roughness"].default_value = radial
    hair.inputs["IOR"].default_value = 1.55
    hair.inputs["Coat"].default_value = 0.0
    tree.links.new(hair.outputs[0], out.inputs[0])
    return mat


def s28_hair(tex):
    """A close-up Chiang groom on a grey scalp.

    Geometry is a particle system exported to the curve sidecar; shading on our
    side is STRELKA_materials_hair. This row used to spend half its pixels on a
    bald control, but 00_calibration already owns exposure and framing; here that
    control only diluted a strand error and made individual curves hard to see.
    Strand count is kept modest so the row measures the BSDF, not variance.
    """
    add_stage()
    scalp = new_material("scalp", base_color=(0.35, 0.32, 0.30, 1.0),
                         roughness=0.85, metallic=0.0, specular=0.0)
    hair_mat = new_hair_material("hair0")

    fur = sphere("Fur", (0.0, 0.0, 0.7), radius=0.48)
    fur.data.materials.clear()
    # The scalp is a grey dielectric, the same one the bald control wears. It used
    # to carry hair_mat, which put a Chiang lobe on a triangle sphere -- a lobe
    # parameterised on a cylinder's tangent frame and azimuth, evaluated on
    # geometry that has neither, so each renderer invented a tangent and the row
    # measured that invention rather than the strands. It was worth more than half
    # the scene's disagreement: the scalp disc alone read 0.809 of the reference,
    # and dropping it took the hair half from rel 0.098 to 0.046 with the mean
    # ratio landing on 1.006.
    fur.data.materials.append(scalp)
    fur.data.materials.append(hair_mat)

    mod = fur.modifiers.new("Hair", type="PARTICLE_SYSTEM")
    psys = fur.particle_systems[0]
    s = psys.settings
    s.type = "HAIR"
    s.use_advanced_hair = True
    s.count = 1200
    s.hair_length = 0.28
    s.hair_step = 5
    s.display_step = 3
    s.render_step = 3
    s.root_radius = 0.004
    s.tip_radius = 0.0015
    s.radius_scale = 1.0
    s.shape = 0.0
    s.use_close_tip = True
    s.child_type = "NONE"
    s.material = 2  # 1-based slot -> hair0, which is slot 2 now that scalp is 1
    # Cycles needs the particle system rendered as path.
    s.use_rotations = False

    # Compare the same primitive on both sides. Cycles defaults to camera-facing
    # ribbons subdivided twice; Strelka receives linear round curves from the
    # sidecar, so leaving either default made the row a comparison of different
    # geometry even when the aggregate means happened to be close.
    bpy.context.scene.cycles_curves.shape = "THICK"
    bpy.context.scene.cycles_curves.subdivisions = 0
    return fur


def export_feature_hair(scene_dir, name):
    """Write <name>_curves.bin from every HAIR particle system in the scene."""
    # Import beside the iso_bathroom writer so the format stays one module.
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(repo, "tools", "iso_bathroom"))
    from curve_sidecar import (  # noqa: E402
        CurveSet, write_curve_sidecar, BASIS_LINEAR, hair_strand_radii)
    from mathutils import Matrix

    axis_conv = Matrix.Rotation(-math.pi / 2, 4, "X")
    # Push display to render resolution, then evaluate once.
    touched = []
    for obj in bpy.data.objects:
        for psys in obj.particle_systems:
            st = psys.settings
            if st.type != "HAIR":
                continue
            touched.append((st, st.display_step))
            st.display_step = st.render_step
    if not touched:
        return 0, 0
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    sets = []
    for obj in bpy.data.objects:
        if not obj.particle_systems or not obj.visible_get():
            continue
        ev = obj.evaluated_get(dg)
        for psys in ev.particle_systems:
            st = psys.settings
            if st.type != "HAIR":
                continue
            slots = [m.name if m else "" for m in obj.data.materials]
            slot = int(st.material) - 1
            material = slots[slot] if 0 <= slot < len(slots) else (slots[0] if slots else "")
            n_strands = len(psys.particles) + len(psys.child_particles)
            if n_strands == 0:
                continue
            n_points = (1 << st.display_step) + 1
            radii = hair_strand_radii(st, n_points)
            co = psys.co_hair
            strands = []
            for i in range(n_strands):
                pts = []
                for k in range(n_points):
                    c = axis_conv @ co(ev, particle_no=i, step=k)
                    pts.append((c.x, c.y, c.z, radii[k]))
                if pts[0][:3] == pts[-1][:3]:
                    continue
                strands.append(pts)
            if strands:
                sets.append(CurveSet(material, strands, BASIS_LINEAR))

    for st, display_step in touched:
        st.display_step = display_step

    path = os.path.join(scene_dir, name + "_curves.bin")
    n_strands, n_points = write_curve_sidecar(path, sets)
    print("  curves -> %s (%d strands, %d points)"
          % (os.path.relpath(path, os.path.dirname(scene_dir)), n_strands, n_points))
    return n_strands, n_points


# KHR_materials_clearcoat has no IOR field and Blender writes no sheen extension
# at all, so both scenes are patched after the export. Keyed by scene name; the
# function is handed the parsed glTF document and mutates it in place.
def patch_sheen(doc):
    for mat in doc.get("materials", []):
        if not mat.get("name", "").startswith("sheen"):
            continue
        i = int(mat["name"][len("sheen"):])
        roughness = max(0.05, i / 6.0)
        mat.setdefault("extensions", {})["KHR_materials_sheen"] = {
            "sheenColorFactor": [1.0, 1.0, 1.0],
            "sheenRoughnessFactor": roughness,
        }
    used = set(doc.get("extensionsUsed", []))
    used.add("KHR_materials_sheen")
    doc["extensionsUsed"] = sorted(used)


def patch_clearcoat_ior(doc):
    for mat in doc.get("materials", []):
        if not mat.get("name", "").startswith("coat"):
            continue
        i = int(mat["name"][len("coat"):])
        ext = mat.setdefault("extensions", {}).setdefault("KHR_materials_clearcoat", {})
        # Blender writes the weight and roughness; only the IOR is ours to add.
        ext.setdefault("clearcoatFactor", 1.0)
        ext["clearcoatIor"] = 1.0 + 1.2 * (i / 6.0)
    used = set(doc.get("extensionsUsed", []))
    used.add("KHR_materials_clearcoat")
    doc["extensionsUsed"] = sorted(used)


def patch_iridescence(doc):
    for mat in doc.get("materials", []):
        if not mat.get("name", "").startswith("film"):
            continue
        i = int(mat["name"][len("film"):])
        thickness = 200.0 + 600.0 * (i / 6.0)
        mat.setdefault("extensions", {})["KHR_materials_iridescence"] = {
            "iridescenceFactor": 1.0,
            "iridescenceIor": 1.4,
            # Both bounds the same: with no thickness texture the spec says to use
            # the maximum, and a range nothing samples is a way of writing the
            # wrong number.
            "iridescenceThicknessMinimum": thickness,
            "iridescenceThicknessMaximum": thickness,
        }
    used = set(doc.get("extensionsUsed", []))
    used.add("KHR_materials_iridescence")
    doc["extensionsUsed"] = sorted(used)


def patch_bounded_volume(doc):
    # Blender exports no volume at all, so the box arrives as an ordinary opaque
    # cube and the medium is written here. The numbers mirror the Principled
    # Volume node the Cycles side renders.
    for mat in doc.get("materials", []):
        if mat.get("name") != "medium":
            continue
        mat.setdefault("extensions", {})["STRELKA_materials_medium"] = {
            "density": 2.5,
            "scatterColor": [0.75, 0.82, 0.95],
            "emissionColor": [0.0, 0.0, 0.0],
            "anisotropy": 0.0,
        }
        # Transmissive to any viewer that does not know the extension, rather
        # than a solid white box.
        mat.setdefault("extensions", {})["KHR_materials_transmission"] = {
            "transmissionFactor": 1.0
        }
    used = set(doc.get("extensionsUsed", []))
    used.update(("STRELKA_materials_medium", "KHR_materials_transmission"))
    doc["extensionsUsed"] = sorted(used)


def patch_thin_walled(doc):
    # Strelka treats transmission as solid unless KHR_materials_volume states a
    # thickness of zero -- Blender's Thin Wall flag is not exported. Materials
    # named thin* get the wall; solid* keep the default (no volume extension).
    for mat in doc.get("materials", []):
        name = mat.get("name", "")
        if not name.startswith("thin"):
            continue
        mat.setdefault("extensions", {})["KHR_materials_volume"] = {
            "thicknessFactor": 0.0,
        }
        mat.setdefault("extensions", {})["KHR_materials_transmission"] = {
            "transmissionFactor": 1.0,
        }
    used = set(doc.get("extensionsUsed", []))
    used.update(("KHR_materials_volume", "KHR_materials_transmission"))
    doc["extensionsUsed"] = sorted(used)


def patch_specular_color(doc):
    # White tint is the glTF default, so the exporter drops the extension on
    # spec0. Write it back so every sphere in the ramp carries the same fields.
    tints = [
        [1.0, 1.0, 1.0],
        [1.0, 0.55, 0.35],
        [1.0, 0.2, 0.15],
        [0.95, 0.85, 0.2],
        [0.25, 0.85, 0.35],
        [0.2, 0.45, 1.0],
        [0.75, 0.3, 0.95],
    ]
    for mat in doc.get("materials", []):
        name = mat.get("name", "")
        if not name.startswith("spec"):
            continue
        i = int(name[len("spec"):])
        mat.setdefault("extensions", {})["KHR_materials_specular"] = {
            "specularFactor": 1.0,
            "specularColorFactor": tints[i],
        }
    used = set(doc.get("extensionsUsed", []))
    used.add("KHR_materials_specular")
    doc["extensionsUsed"] = sorted(used)


def patch_diffuse_transmission(doc):
    # Mix(Principled, Translucent) does not survive the exporter as
    # KHR_materials_diffuse_transmission. Rebuild the extension from the material
    # name, which encodes the mix weight; colour matches the Translucent node.
    colour = [0.55, 0.75, 0.35]
    for mat in doc.get("materials", []):
        name = mat.get("name", "")
        if not name.startswith("dtrans"):
            continue
        i = int(name[len("dtrans"):])
        weight = i / 4.0
        # Replace whatever the exporter wrote for the Mix with a diffuse base the
        # extension can ride on.
        mat["pbrMetallicRoughness"] = {
            "baseColorFactor": colour + [1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        }
        mat.setdefault("extensions", {})["KHR_materials_diffuse_transmission"] = {
            "diffuseTransmissionFactor": weight,
            "diffuseTransmissionColorFactor": colour,
        }
        # No leftover specular from a default Principled export.
        mat.setdefault("extensions", {})["KHR_materials_specular"] = {
            "specularFactor": 0.0,
            "specularColorFactor": [1.0, 1.0, 1.0],
        }
    used = set(doc.get("extensionsUsed", []))
    used.update(("KHR_materials_diffuse_transmission", "KHR_materials_specular"))
    doc["extensionsUsed"] = sorted(used)


def patch_subsurface(doc):
    # Cycles authors Base Color as a diffuse subsurface colour; the extension
    # carries the single-scattering albedo (see gltfloader.cpp). Invert Van de
    # Hulst per channel -- the same fit tools/iso_bathroom/vray2strelka.py uses
    # -- and keep the radius Cycles used. Cycles' own BaseColor→medium mapping is
    # not published as this function, so the row is a regression guard on our
    # side until that mapping is measured the way bake_env.py measured the sky.
    radius = list(SSS_RADIUS)
    for mat in doc.get("materials", []):
        name = mat.get("name", "")
        if not name.startswith("sss"):
            continue
        base = mat.get("pbrMetallicRoughness", {}).get("baseColorFactor", [0.5, 0.5, 0.5, 1.0])
        scatter = [diffuse_to_single_scattering_albedo(c) for c in base[:3]]
        mat.setdefault("extensions", {})["STRELKA_materials_subsurface"] = {
            "subsurfaceFactor": 1.0,
            "scatterColor": scatter,
            "scatterRadius": radius,
            "anisotropy": 0.0,
            "scatterReference": list(base[:3]),
        }
    used = set(doc.get("extensionsUsed", []))
    used.add("STRELKA_materials_subsurface")
    doc["extensionsUsed"] = sorted(used)


def patch_hair(doc):
    # The strands' material has to exist in the glTF even though no triangle wears
    # it. The exporter drops materials nothing references, and the curve sidecar
    # binds by name against what it finds, so with the scalp shaded grey the
    # strands fell through to material 0 -- they rendered as stage grey behind one
    # warning in the log, and the row still produced a plausible-looking number.
    mats = doc.setdefault("materials", [])
    if not any(m.get("name", "").startswith("hair") for m in mats):
        mats.append({"name": "hair0", "doubleSided": True})

    # Principled Hair does not survive the glTF exporter as anything useful.
    # Rewrite hair* materials to the pigment + roughness the Chiang lobe reads,
    # and mark them with STRELKA_materials_hair so the loader picks MATERIAL_TYPE_HAIR.
    for mat in doc.get("materials", []):
        name = mat.get("name", "")
        if not name.startswith("hair"):
            continue
        mat["pbrMetallicRoughness"] = {
            "baseColorFactor": list(HAIR_COLOR),
            "metallicFactor": 0.0,
            "roughnessFactor": HAIR_ROUGHNESS,
        }
        mat.setdefault("extensions", {})["KHR_materials_ior"] = {"ior": 1.55}
        mat["extensions"]["STRELKA_materials_hair"] = {
            "radialRoughness": HAIR_RADIAL,
            "coat": 0.0,
        }
    used = set(doc.get("extensionsUsed", []))
    used.update(("KHR_materials_ior", "STRELKA_materials_hair"))
    doc["extensionsUsed"] = sorted(used)


GLTF_PATCHERS = {
    "14_sheen": patch_sheen,
    "15_clearcoat": patch_clearcoat_ior,
    "16_iridescence": patch_iridescence,
    "18_bounded_volume": patch_bounded_volume,
    "21_specular_color": patch_specular_color,
    "22_thin_walled": patch_thin_walled,
    "23_diffuse_transmission": patch_diffuse_transmission,
    "25_subsurface": patch_subsurface,
    "28_hair": patch_hair,
}


SCENES = [
    ("00_calibration",      s00_calibration,   True),
    ("01_srgb_texture",     s01_srgb_texture,  True),
    ("02_basecolor",        s02_basecolor,     True),
    ("03_roughness",        s03_roughness,     True),
    ("04_metal",            s04_metal,         True),
    ("05_anisotropy",       s05_anisotropy,    True),
    ("06_normalmap",        s06_normalmap,     True),
    ("07_alpha_clip",       s07_alpha_clip,    True),
    ("08_alpha_blend",      s08_alpha_blend,   True),
    ("09_glass_ior",        s09_glass_ior,     True),
    ("10_glass_absorption", s10_glass_absorption, True),
    ("11_emission",         s11_emission,      True),
    ("12_lights_punctual",  s12_lights_punctual, False),   # no sidecar on purpose
    ("13_uv2_vcol",         s13_uv2_vcol,      True),
    ("14_sheen",            s14_sheen,         True),
    ("15_clearcoat",        s15_clearcoat,     True),
    ("16_iridescence",      s16_iridescence,   True),
    ("17_coated_glass",     s17_coated_glass,  True),
    ("18_bounded_volume",   s18_bounded_volume, True),
    ("19_env_and_light",    s19_env_and_light, True),
    ("20_mirror_and_floor", s20_mirror_and_floor, True),
    ("21_specular_color",   s21_specular_color, True),
    ("22_thin_walled",      s22_thin_walled,   True),
    ("23_diffuse_transmission", s23_diffuse_transmission, True),
    ("24_orthographic",     s24_orthographic,  True),
    ("25_subsurface",       s25_subsurface,    True),
    ("26_dof",              s26_dof,           True),
    ("27_ies",              s27_ies,           "ies"),
    ("28_hair",             s28_hair,          True),
]

# Scenes whose world is not black and therefore has to reach Strelka as an
# environment map. The world is baked to an equirectangular EXR in Strelka's own
# convention by bake_env.py, which measures Cycles' mapping rather than assuming
# it -- see the note there about the assumption that survived every check except
# a render.
ENV_BAKE = {"19_env_and_light"}
ENV_BAKE_WIDTH = 1024
ORTHO_SCENES = {"24_orthographic"}
CAMERA_JSON_SCENES = {"26_dof"}
# Sidecar is not the default rect key: the IES row owns its own point light.
HAIR_SCENES = {"28_hair"}
IES_SCENES = {"27_ies"}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def blender_to_gltf(p):
    """Blender is Z-up, glTF is Y-up. Strelka sees the exported world, so every
    coordinate written into the sidecar and the .toml has to be converted."""
    return [p[0], p[2], -p[1]]


def export_gltf(filepath, export_lights):
    """Filter kwargs against the operator's actual RNA so this survives the
    exporter's parameter churn across Blender versions."""
    wanted = dict(
        filepath=filepath,
        export_format="GLTF_SEPARATE",
        use_selection=False,
        export_apply=True,
        export_yup=True,
        export_cameras=True,
        export_lights=export_lights,
        export_normals=True,
        export_tangents=True,
        export_texcoords=True,
        export_attributes=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_extras=False,
    )
    props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    kwargs = {k: v for k, v in wanted.items() if k in props}
    dropped = sorted(set(wanted) - set(kwargs))
    if dropped:
        print("  [warn] exporter ignored unknown params: %s" % ", ".join(dropped))
    bpy.ops.export_scene.gltf(**kwargs)


def write_light_json(path, environment=None, lights=None):
    data = {
        "lights": lights if lights is not None else [
            {
                "type": "rect",
                "position": blender_to_gltf(KEY_POS),
                "orientation": [-90.0, 0.0, 0.0],
                "color": [1.0, 1.0, 1.0],
                "intensity": KEY_RADIANCE,
                "width": KEY_SIZE,
                "height": KEY_SIZE,
            }
        ]
    }
    if environment is not None:
        data["environment"] = environment
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def write_camera_json(path, focus_distance, fstop, focal_length_mm):
    """DOF and sensor extras glTF cannot carry. Matched to the exported camera
    by name -- the builder always names it Camera.
    """
    data = {
        "cameras": [
            {
                "name": "Camera",
                "focal_length_mm": focal_length_mm,
                "sensor_width": 36.0,
                "sensor_height": 24.0,
                "sensor_fit": "VERTICAL",
                "dof": {
                    "enabled": True,
                    "focus_distance": focus_distance,
                    "fstop": fstop,
                    "blades": 0,
                    "blade_rotation": 0.0,
                    "anamorphic_ratio": 1.0,
                },
            }
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def bake_world_env(scene_dir, name):
    """Bake the current scene's world to an equirectangular EXR Strelka can read.

    Imported from bake_env rather than reimplemented: the mapping between
    Cycles' panorama and Strelka's lookup is the part that is easy to get wrong
    and hard to notice, and that module gets it by measuring a direction field
    instead of writing the inverse down.
    """
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bake_env import make_bake_scene, load_rgb, direction_field, resample_to_strelka

    world = bpy.context.scene.world
    width, height = ENV_BAKE_WIDTH, ENV_BAKE_WIDTH // 2
    sc = make_bake_scene(world, width * 2, height * 2)
    raw = os.path.join(scene_dir, name + "_env_raw.exr")
    sc.render.filepath = raw
    bpy.ops.render.render(write_still=True, scene=sc.name)
    src, w, h = load_rgb(raw)
    dst = resample_to_strelka(src, direction_field(w, h), width, height)
    os.remove(raw)
    bpy.data.scenes.remove(sc)

    final = os.path.join(scene_dir, name + "_env.exr")
    img = bpy.data.images.new(name + "_env", width=width, height=height, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = dst[::-1]   # Strelka reads row 0 as v = 0; Blender is bottom-up
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = final
    img.file_format = "OPEN_EXR"
    img.save()
    bpy.data.images.remove(img)
    print("  env    -> %s  mean=%.4f max=%.3f"
          % (os.path.basename(final), float(dst.mean()), float(dst.max())))
    return {"texture": name + "_env.exr", "intensity": 1.0, "color": [1.0, 1.0, 1.0]}


def write_toml(path, name, gltf_rel, out_rel, orthographic=False):
    location, target, fov = camera_for_scene(name)
    cam = blender_to_gltf(location)
    tgt = blender_to_gltf(target)
    camera_block = (
        "[camera]\n"
        "index = 0\n"
        "position = [%.6f, %.6f, %.6f]\n"
        "target = [%.6f, %.6f, %.6f]\n"
        % (cam[0], cam[1], cam[2], tgt[0], tgt[1], tgt[2])
    )
    # Orthographic extents come from the glTF camera; writing a perspective fov
    # here would only set an unused field, and omitting it keeps the config
    # honest about what frames the image.
    if not orthographic:
        camera_block += "fov = %.4f\n" % fov
    camera_block += "\n"
    with open(path, "w") as f:
        f.write(
            "[scene]\n"
            'path = "%s"\n\n'
            "[output]\n"
            'path = "%s"\n'
            "width = %d\n"
            "height = %d\n\n"
            "[render]\n"
            'integrator = "pt"\n'
            "spp = %d\n"
            "spp_per_launch = 1\n"
            "max_depth = %d\n"
            'sampler = "sobol"\n'
            # The reference is Cycles, whose Volume Absorption node works out to
            # sigma_t = (1 - C)/d. The glTF spec says -ln(C)/d, which absorbs a
            # great deal more -- at C = 0.5 it is 0.69/d against 0.5/d. Matching
            # the reference means naming the convention rather than inheriting a
            # default.
            'volume_model = "cycles"\n\n'
            "%s"
            "[tonemap]\n"
            # Linear out on both sides: the comparison must not go through a
            # tone curve, or every difference gets squashed in the highlights.
            'type = "none"\n'
            "gamma = 0.0\n"
            "exposure_iso = %.4f\n"
            "exposure_fstop = %.4f\n"
            "exposure_shutter = %.4f\n"
            % (gltf_rel, out_rel, RES, RES, STRELKA_SCENE_SPP.get(name, STRELKA_SPP),
               MAX_DEPTH, camera_block,
               EXPOSURE_ISO, EXPOSURE_FSTOP, EXPOSURE_SHUTTER)
        )


def verify_export(gltf_path):
    """Read the exported glTF back and report what actually landed.

    The exporter decides a lot from the node graph, and it changes between
    releases. Reporting the real contents here is what makes the difference
    between "Strelka got it wrong" and "it was never in the file".
    """
    with open(gltf_path) as f:
        doc = json.load(f)

    # Report what some material or mesh actually carries, not what extensionsUsed
    # claims. The patchers append to that list unconditionally, so a scene whose
    # material vanished from the export -- the glTF exporter drops materials no
    # triangle references -- still declared its extension and read as a pass while
    # the geometry silently fell back to material 0.
    declared = set(doc.get("extensionsUsed", []))

    def carried_by(node):
        found = set()
        if isinstance(node, dict):
            found |= set(node.get("extensions", {}) or {})
            for key, value in node.items():
                if key not in ("extensionsUsed", "extensionsRequired"):
                    found |= carried_by(value)
        elif isinstance(node, list):
            for item in node:
                found |= carried_by(item)
        return found

    carried = carried_by(doc)
    exts = sorted(carried)
    orphaned = sorted(declared - carried)

    attrs = set()
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            attrs |= set(prim["attributes"])
    modes = sorted({m.get("alphaMode", "OPAQUE") for m in doc.get("materials", [])})

    print("  exts   : %s" % (", ".join(exts) if exts else "(none)"))
    if orphaned:
        print("  WARNING: declared but on no material: %s" % ", ".join(orphaned))
    print("  attrs  : %s" % ", ".join(sorted(attrs)))
    print("  alpha  : %s" % ", ".join(modes))
    return {"extensions": exts, "attributes": sorted(attrs), "alphaModes": modes}


def render_cycles(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    out_root = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv \
        else os.path.abspath("scenes/feature_tests")
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    skip_render = "--no-render" in argv

    print("Blender %s" % (bpy.app.version_string,))
    print("Output: %s" % out_root)

    tex_dir = os.path.join(out_root, "_textures")
    os.makedirs(tex_dir, exist_ok=True)
    tex = {
        "srgb": os.path.join(tex_dir, "srgb_probe.png"),
        "alpha": os.path.join(tex_dir, "alpha_dots.png"),
        "normal": os.path.join(tex_dir, "normal_bumps.png"),
    }
    tex_srgb_probe(tex["srgb"])
    tex_alpha_dots(tex["alpha"])
    tex_normal_bumps(tex["normal"])
    print("Textures written to %s" % tex_dir)

    manifest = {}

    for name, builder, use_sidecar in SCENES:
        if only and only not in name:
            continue
        print("\n=== %s ===" % name)
        scene_dir = os.path.join(out_root, name)
        os.makedirs(scene_dir, exist_ok=True)

        reset_scene()
        bpy.context.scene.cycles.samples = SCENE_SAMPLES.get(name, CYCLES_SAMPLES)
        add_camera(name)
        # IES owns its light; the default rect key would drown the photometric
        # hotspot and turn the row into a second copy of 00_calibration.
        if use_sidecar is True:
            add_key_light()
        builder(tex)

        if name in IES_SCENES:
            ies_path = os.path.join(scene_dir, name + ".ies")
            peak = install_ies(ies_path)
            # Point the Cycles TexIES node at the file we just wrote.
            for obj in bpy.data.objects:
                if obj.type != "LIGHT" or "strelka_ies_node" not in obj:
                    continue
                node = obj.data.node_tree.nodes.get(obj["strelka_ies_node"])
                if node is not None:
                    node.filepath = ies_path
            print("  ies    -> %s (peak %.0f cd)" % (os.path.basename(ies_path), peak))

        gltf_path = os.path.join(scene_dir, name + ".gltf")
        export_gltf(gltf_path, export_lights=not use_sidecar)
        patcher = GLTF_PATCHERS.get(name)
        if patcher is not None:
            with open(gltf_path) as f:
                doc = json.load(f)
            patcher(doc)
            with open(gltf_path, "w") as f:
                json.dump(doc, f)
            print("  patch  : %s" % patcher.__name__)
        print("  glTF   -> %s" % os.path.relpath(gltf_path, out_root))
        manifest[name] = verify_export(gltf_path)

        if name in HAIR_SCENES:
            export_feature_hair(scene_dir, name)

        environment = None
        if name in ENV_BAKE:
            environment = bake_world_env(scene_dir, name)

        if use_sidecar is True:
            write_light_json(os.path.join(scene_dir, name + "_light.json"), environment)
            print("  lights -> sidecar (radiance %.3f%s)"
                  % (KEY_RADIANCE, " + environment" if environment else ""))
        elif name in IES_SCENES:
            # Point light above the stage. Orientation −90° X puts local −Z
            # (photometric axis) along world −Y in the Y-up export, matching
            # the Cycles lamp whose −Z points at the floor.
            write_light_json(
                os.path.join(scene_dir, name + "_light.json"),
                lights=[{
                    "type": "point",
                    "name": "IESKey",
                    "position": blender_to_gltf((0.0, 0.0, 3.2)),
                    "orientation": [-90.0, 0.0, 0.0],
                    "color": [1.0, 1.0, 1.0],
                    "intensity": 1.0,
                    "unit": "intensity",
                    "ies": name + ".ies",
                }],
            )
            print("  lights -> sidecar (IES point, intensity 1.0)")
        else:
            print("  lights -> KHR_lights_punctual (no sidecar)")

        if name in CAMERA_JSON_SCENES:
            write_camera_json(
                os.path.join(scene_dir, name + "_camera.json"),
                focus_distance=DOF_FOCUS_DISTANCE,
                fstop=DOF_FSTOP,
                focal_length_mm=DOF_FOCAL_LENGTH_MM,
            )
            print("  camera -> sidecar (dof f/%.1f focus %.3f m, %.2f mm)"
                  % (DOF_FSTOP, DOF_FOCUS_DISTANCE, DOF_FOCAL_LENGTH_MM))

        write_toml(
            os.path.join(scene_dir, name + ".toml"),
            name,
            gltf_rel=os.path.join(scene_dir, name + ".gltf"),
            out_rel=os.path.join(scene_dir, name + "_strelka.exr"),
            orthographic=name in ORTHO_SCENES,
        )

        if not skip_render:
            exr = os.path.join(scene_dir, name + "_cycles.exr")
            render_cycles(exr)
            print("  cycles -> %s" % os.path.relpath(exr, out_root))

    manifest_path = os.path.join(out_root, "export_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"blender": bpy.app.version_string, "scenes": manifest}, f, indent=2)
    print("\nDone. Export manifest: %s" % manifest_path)

    if MISSING_SOCKETS:
        print("[warn] Principled sockets not found in this Blender: %s"
              % ", ".join(sorted(MISSING_SOCKETS)))
        print("       Those features were silently left at default -- add the")
        print("       correct name to ALIASES before trusting those scenes.")

    expected = {
        "05_anisotropy": "KHR_materials_anisotropy",
        "09_glass_ior": "KHR_materials_ior",
        "10_glass_absorption": "KHR_materials_volume",
        "11_emission": "KHR_materials_emissive_strength",
        "12_lights_punctual": "KHR_lights_punctual",
        "21_specular_color": "KHR_materials_specular",
        "22_thin_walled": "KHR_materials_volume",
        "23_diffuse_transmission": "KHR_materials_diffuse_transmission",
        "25_subsurface": "STRELKA_materials_subsurface",
        "28_hair": "STRELKA_materials_hair",
    }
    for scene, ext in expected.items():
        info = manifest.get(scene)
        if info and ext not in info["extensions"]:
            print("[warn] %s did not export %s -- that scene tests nothing yet."
                  % (scene, ext))
    info = manifest.get("07_alpha_clip")
    if info and "MASK" not in info["alphaModes"]:
        print("[warn] 07_alpha_clip exported %s, not MASK."
              % ", ".join(info["alphaModes"]))


if __name__ == "__main__":
    main()
