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
    "anisotropic":       ["Anisotropic"],
    "coat":              ["Coat Weight", "Clearcoat"],
    "coat_roughness":    ["Coat Roughness", "Clearcoat Roughness"],
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


def add_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(CAM_FOV_DEG)
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = CAM_LOC

    direction = (
        CAM_TARGET[0] - CAM_LOC[0],
        CAM_TARGET[1] - CAM_LOC[1],
        CAM_TARGET[2] - CAM_LOC[2],
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
]


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


def write_light_json(path):
    data = {
        "lights": [
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
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def write_toml(path, name, gltf_rel, out_rel):
    cam = blender_to_gltf(CAM_LOC)
    tgt = blender_to_gltf(CAM_TARGET)
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
            "[camera]\n"
            "index = 0\n"
            "position = [%.6f, %.6f, %.6f]\n"
            "target = [%.6f, %.6f, %.6f]\n"
            "fov = %.4f\n\n"
            "[tonemap]\n"
            # Linear out on both sides: the comparison must not go through a
            # tone curve, or every difference gets squashed in the highlights.
            'type = "none"\n'
            "gamma = 0.0\n"
            "exposure_iso = %.4f\n"
            "exposure_fstop = %.4f\n"
            "exposure_shutter = %.4f\n"
            % (gltf_rel, out_rel, RES, RES, STRELKA_SPP, MAX_DEPTH,
               cam[0], cam[1], cam[2], tgt[0], tgt[1], tgt[2], CAM_FOV_DEG,
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

    exts = sorted(doc.get("extensionsUsed", []))
    attrs = set()
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            attrs |= set(prim["attributes"])
    modes = sorted({m.get("alphaMode", "OPAQUE") for m in doc.get("materials", [])})

    print("  exts   : %s" % (", ".join(exts) if exts else "(none)"))
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
        add_camera()
        if use_sidecar:
            add_key_light()
        builder(tex)

        gltf_path = os.path.join(scene_dir, name + ".gltf")
        export_gltf(gltf_path, export_lights=not use_sidecar)
        print("  glTF   -> %s" % os.path.relpath(gltf_path, out_root))
        manifest[name] = verify_export(gltf_path)

        if use_sidecar:
            write_light_json(os.path.join(scene_dir, name + "_light.json"))
            print("  lights -> sidecar (radiance %.3f)" % KEY_RADIANCE)
        else:
            print("  lights -> KHR_lights_punctual (no sidecar)")

        write_toml(
            os.path.join(scene_dir, name + ".toml"),
            name,
            gltf_rel=os.path.join(scene_dir, name + ".gltf"),
            out_rel=os.path.join(scene_dir, name + "_strelka.exr"),
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
