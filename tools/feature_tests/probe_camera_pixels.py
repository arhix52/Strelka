#!/usr/bin/env python3
"""Print the Blender object/material hit by image pixels."""

import argparse
import sys

import bpy
from mathutils import Vector


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("pixels", nargs="+", help="x,y in top-left image coordinates")
    args = parser.parse_args(argv)

    scene = bpy.context.scene
    camera = bpy.data.objects[args.camera]
    scene.camera = camera
    # view_frame order: top-right, bottom-right, bottom-left, top-left.
    tr, br, bl, tl = camera.data.view_frame(scene=scene)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation

    for text in args.pixels:
        x, y = (float(part) for part in text.split(",", 1))
        u = (x + 0.5) / args.size
        v = 1.0 - (y + 0.5) / args.size
        bottom = bl.lerp(br, u)
        top = tl.lerp(tr, u)
        target = camera.matrix_world @ bottom.lerp(top, v)
        hit, location, _normal, face, obj, _matrix = scene.ray_cast(
            depsgraph, origin, (target - origin).normalized()
        )
        material = ""
        if hit and face >= 0:
            evaluated = obj.evaluated_get(depsgraph)
            slot = evaluated.data.polygons[face].material_index
            if slot < len(obj.material_slots) and obj.material_slots[slot].material:
                material = obj.material_slots[slot].material.name
        print(f"{text}\t{obj.name if hit else '-'}\t{material or '-'}\t{tuple(round(v, 5) for v in location) if hit else '-'}")


if __name__ == "__main__":
    main()
