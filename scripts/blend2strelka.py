#!/usr/bin/env python3
"""
Export a .blend file to Strelka-compatible glTF Binary (.glb) + light JSON.

Usage (headless):
    blender --background <file.blend> --python blend2strelka.py -- [--output DIR] [--env-map PATH]

Usage (from open Blender):
    Run as script inside Blender with the .blend already open.

What it does:
    1. Triangulates all meshes (non-destructive via modifier).
    2. Extracts all lights → <name>_light.json  (Strelka format).
    3. Extracts environment / world HDRI if present.
    4. Exports scene as binary .glb (textures embedded).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Euler, Matrix  # type: ignore


# ---------------------------------------------------------------------------
# Argument parsing (arguments after "--" on the command-line)
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    # Blender passes everything after "--" to the script
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Export .blend → Strelka .glb + _light.json")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: same directory as .blend file)",
    )
    parser.add_argument(
        "--env-map",
        type=str,
        default=None,
        help="Path to an HDR/EXR environment map to reference in the light JSON",
    )
    parser.add_argument(
        "--env-intensity",
        type=float,
        default=1.0,
        help="Environment light intensity multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--env-rotation",
        type=float,
        default=0.0,
        help="Environment Y-axis rotation in degrees (default: 0)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Base name for output files (default: .blend filename stem)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Light extraction — convert Blender lights to Strelka JSON
# ---------------------------------------------------------------------------

def blender_light_to_strelka(obj):
    """
    Convert a Blender light object to a Strelka UniformLightDesc dict.
    Strelka supports: RECT(0), DISC(1), SPHERE(2), DISTANT(3), DOME(4).

    Two coordinate transformations are applied:

    1. Z-up → Y-up:  The glTF exporter (export_yup=True) converts geometry
       from Blender's Z-up to glTF's Y-up via a -90° X rotation.  We must
       apply the same conversion to light transforms so they live in the
       same coordinate space as the exported geometry.

    2. Light normal flip:  Blender lights emit from local -Z, while
       Strelka's light quad normal points along local +Z.  An additional
       180° X rotation corrects for this.
    """
    light = obj.data

    # -90° X rotation: Blender Z-up → glTF/Strelka Y-up
    axis_conv = Matrix.Rotation(-math.pi / 2, 4, 'X')
    converted = axis_conv @ obj.matrix_world

    # Position in Y-up space
    loc = converted.translation

    # Rotation in Y-up space.  Both Blender and Strelka emit along local -Z
    # (confirmed: rect light shader uses -cross(e1,e2), distant light uses -Z),
    # so NO additional normal flip is needed.
    rot_strelka = converted.to_3x3().normalized()

    # GLM's quat(vec3) uses the same R_z * R_y * R_x order as Blender "XYZ"
    euler = rot_strelka.to_euler("XYZ")
    orientation = [math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z)]

    base = {
        "position": [loc.x, loc.y, loc.z],
        "orientation": orientation,
        "color": [light.color.r, light.color.g, light.color.b],
        "intensity": light.energy,
        "enabled": obj.visible_get() if hasattr(obj, "visible_get") else True,
        "name": obj.name,
    }

    if light.type == "AREA":
        # Blender area lights use Power (W).
        base["unit"] = "power"
        if light.shape in ("RECTANGLE", "SQUARE"):
            base["width"] = light.size
            base["height"] = light.size_y if light.shape == "RECTANGLE" else light.size
            base["type"] = "rect"
        elif light.shape in ("DISK", "ELLIPSE"):
            base["radius"] = 0.5 * light.size
            base["type"] = "disc"
        else:
            base["width"] = light.size
            base["height"] = light.size
            base["type"] = "rect"

    elif light.type == "POINT":
        base["type"] = "point"
        base["unit"] = "power"
        base["radius"] = light.shadow_soft_size

    elif light.type == "SUN":
        # Distant / directional light. halfAngle is full angular diameter in degrees.
        base["type"] = "distant"
        base["unit"] = "irradiance"
        base["halfAngle"] = math.degrees(light.angle) if hasattr(light, "angle") else 0.53

    elif light.type == "SPOT":
        base["type"] = "spot"
        base["unit"] = "power"
        base["radius"] = light.shadow_soft_size
        # Blender spot_size is the outer cone full angle; blend softens the penumbra.
        outer = light.spot_size * 0.5
        blend = getattr(light, "spot_blend", 0.15)
        base["outerConeAngle"] = math.degrees(outer)
        base["innerConeAngle"] = math.degrees(outer * (1.0 - blend))

    else:
        base["width"] = 1.0
        base["height"] = 1.0
        base["type"] = "rect"
        base["unit"] = "power"

    return base


def extract_world_hdri(world):
    """
    Try to find the HDRI texture path from the World shader nodes.
    Returns the filepath string or None.
    """
    if world is None:
        return None
    # Blender < 6.0 has use_nodes; Blender >= 6.0 always uses nodes
    if hasattr(world, "use_nodes") and not world.use_nodes:
        return None

    for node in world.node_tree.nodes:
        if node.type == "TEX_ENVIRONMENT":
            img = node.image
            if img is not None and img.filepath:
                path = bpy.path.abspath(img.filepath)
                if os.path.isfile(path):
                    return path
    return None


# ---------------------------------------------------------------------------
# Mesh preparation
# ---------------------------------------------------------------------------

def triangulate_all():
    """Add a triangulate modifier to every mesh object (non-destructive)."""
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            # Skip if already has a triangulate modifier
            if any(m.type == "TRIANGULATE" for m in obj.modifiers):
                continue
            mod = obj.modifiers.new(name="_Triangulate", type="TRIANGULATE")
            mod.quad_method = "BEAUTY"
            mod.ngon_method = "BEAUTY"


def remove_triangulate_modifiers():
    """Clean up temporary triangulate modifiers."""
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            for mod in list(obj.modifiers):
                if mod.name == "_Triangulate":
                    obj.modifiers.remove(mod)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    blend_path = bpy.data.filepath
    if not blend_path:
        print("ERROR: No .blend file is open. Run with: blender --background <file.blend> --python ...")
        sys.exit(1)

    blend_stem = Path(blend_path).stem
    name = args.name or blend_stem

    out_dir = Path(args.output) if args.output else Path(blend_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    glb_path = out_dir / f"{name}.glb"
    light_json_path = out_dir / f"{name}_light.json"

    print(f"=== blend2strelka ===")
    print(f"  Input:   {blend_path}")
    print(f"  Output:  {glb_path}")
    print(f"  Lights:  {light_json_path}")
    print()

    # ---- Extract lights before export (glTF export ignores Blender lights) ----
    lights = []
    light_objects = [obj for obj in bpy.data.objects if obj.type == "LIGHT"]
    for obj in light_objects:
        desc = blender_light_to_strelka(obj)
        print(f"  Light: {obj.name:20s}  type={desc['type']:8s}  intensity={desc['intensity']:.1f}")
        lights.append(desc)

    # ---- Build light JSON ----
    light_data = {"lights": lights}

    # Environment map: explicit arg > world HDRI > nothing
    env_map_path = args.env_map
    if env_map_path is None:
        env_map_path = extract_world_hdri(bpy.context.scene.world)

    if env_map_path:
        light_data["environment"] = {
            "texture": env_map_path,
            "intensity": args.env_intensity,
            "color": [1.0, 1.0, 1.0],
            "rotation": math.radians(args.env_rotation),
        }
        print(f"  Env map: {env_map_path}")

    with open(light_json_path, "w") as f:
        json.dump(light_data, f, indent=4)
    print(f"  Wrote {light_json_path}")

    # ---- Triangulate meshes ----
    triangulate_all()

    # ---- Export glTF Binary (.glb) ----
    # Deselect light objects so they don't bloat the export
    # (glTF doesn't carry Blender lights in a useful way for Strelka)
    bpy.ops.object.select_all(action="DESELECT")

    export_kwargs = dict(
        filepath=str(glb_path),
        export_format="GLB",                     # binary
        export_texcoords=True,
        export_normals=True,
        export_vertex_color="NONE",              # Strelka doesn't use vertex colors
        export_cameras=True,
        export_lights=False,                     # lights are in the JSON
        export_apply=True,                       # apply modifiers (triangulate)
        export_image_format="AUTO",              # keep original texture formats
        export_materials="EXPORT",
    )

    # Blender 4.x moved the operator; try both paths
    try:
        bpy.ops.export_scene.gltf(**export_kwargs)
    except Exception as e:
        print(f"  Export error: {e}")
        sys.exit(1)

    print(f"  Wrote {glb_path}  ({glb_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ---- Clean up ----
    remove_triangulate_modifiers()

    print()
    print("Done! To render in Strelka, pass the .glb path to the renderer.")
    print(f"  Lights will be loaded automatically from {light_json_path.name}")


if __name__ == "__main__":
    main()
