#!/usr/bin/env python3
"""
Does one light stand in another light's way?

    blender -b -P tools/feature_tests/light_occlusion_probe.py -- --out /tmp/light_occl

Strelka puts analytic lights in the same acceleration structure as the geometry
and lets them stop a shadow ray (RAY_MASK_SHADOW carries GEOMETRY_MASK_LIGHT on
both backends), so a small emitter hung under a big one casts a shadow. That
costs a quarter of the PC samples in kids_room -- every shadow ray runs
__intersection__light -- so before paying it, this asks the reference whether it
is right.

The scene is the smallest thing that can answer: a grey floor, a big rect light
high above it, and a small rect light directly between the two. If lights
occlude, the floor carries the small light's silhouette; if they do not, the
patch under it is as bright as its surroundings.

Writes the same four artefacts build_features.py does, so StrelkaCLI renders it
with one -c, and prints the floor-patch ratio from the Cycles render.
"""

import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_features as bf  # noqa: E402

NAME = "light_occlusion"

# The blocker: small, dim and low. Dim because the question is what it does to
# the light behind it, not what it emits -- at a tenth of the key's radiance its
# own contribution cannot be mistaken for the effect.
BLOCK_SIZE = 0.7
BLOCK_POS = (0.0, -0.6, 1.7)
BLOCK_POWER_W = 5.0


def build():
    bf.add_stage()

    key = bf.add_key_light()

    block_data = bpy.data.lights.new("Blocker", type="AREA")
    block_data.shape = "SQUARE"
    block_data.size = BLOCK_SIZE
    block_data.energy = BLOCK_POWER_W
    block_data.color = (1.0, 1.0, 1.0)
    block = bpy.data.objects.new("Blocker", block_data)
    bpy.context.collection.objects.link(block)
    block.location = BLOCK_POS
    block.rotation_euler = (0.0, 0.0, 0.0)
    return key, block


def sidecar_lights():
    return [
        {
            "type": "rect",
            "position": bf.blender_to_gltf(bf.KEY_POS),
            "orientation": [-90.0, 0.0, 0.0],
            "color": [1.0, 1.0, 1.0],
            "intensity": bf.KEY_RADIANCE,
            "width": bf.KEY_SIZE,
            "height": bf.KEY_SIZE,
        },
        {
            "type": "rect",
            "position": bf.blender_to_gltf(BLOCK_POS),
            "orientation": [-90.0, 0.0, 0.0],
            "color": [1.0, 1.0, 1.0],
            "intensity": BLOCK_POWER_W / ((BLOCK_SIZE * BLOCK_SIZE) * math.pi),
            "width": BLOCK_SIZE,
            "height": BLOCK_SIZE,
        },
    ]


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    out_root = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv \
        else os.path.abspath("/tmp/light_occl")
    scene_dir = os.path.join(out_root, NAME)
    os.makedirs(scene_dir, exist_ok=True)

    bf.reset_scene()
    bf.add_camera(NAME)
    build()

    gltf_path = os.path.join(scene_dir, NAME + ".gltf")
    bf.export_gltf(gltf_path, export_lights=False)
    bf.write_light_json(os.path.join(scene_dir, NAME + "_light.json"),
                        lights=sidecar_lights())
    bf.write_toml(os.path.join(scene_dir, NAME + ".toml"), NAME,
                  NAME + ".gltf", NAME + "_strelka.exr")

    ref = os.path.join(scene_dir, NAME + "_cycles.exr")
    bf.render_cycles(ref)
    print("reference -> %s" % ref)


main()
