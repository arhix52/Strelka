#!/usr/bin/env python3
"""
Generate test scenes with different materials to validate BDPT / VCM.

Scenes:
  1. cornell_box     — Classic Cornell Box (diffuse walls, two blocks)
  2. glass_caustics  — Glass sphere on diffuse floor, small area light
  3. metal_sphere    — Mirror conductor sphere on diffuse floor
  4. mixed_materials — Diffuse, metal, glass, rough-metal spheres in a row
  5. small_light     — Tiny bright light, large room (hard for PT)
  6. indirect_cove   — Hidden cove light, room lit indirectly (BDPT/VCM strength)
  7. caustic_pool    — Refractive caustic through glass onto a floor (VCM strength)

Correctness is checked by rendering PT/BDPT/VCM and confirming they agree in
energy (PT is the unbiased reference); see scripts/verify_bdpt_vcm.py.

Each scene outputs:
  scenes/bdpt_tests/<name>/<name>.glb
  scenes/bdpt_tests/<name>/<name>_light.json

No external dependencies — uses only Python stdlib.
"""

import json
import math
import os
import struct

# ─── Geometry helpers ────────────────────────────────────────────────────────


def make_plane(half_extent=5.0, y=0.0):
    """Ground plane at given Y, facing up (+Y normal)."""
    h = half_extent
    positions = [
        -h, y, -h,
         h, y, -h,
         h, y,  h,
        -h, y,  h,
    ]
    normals = [0.0, 1.0, 0.0] * 4
    uvs = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    indices = [0, 1, 2, 0, 2, 3]
    return positions, normals, uvs, indices


def make_quad(corners, normal):
    """Quad from 4 corner positions (CCW winding) and a uniform normal."""
    positions = []
    for c in corners:
        positions.extend(c)
    normals = list(normal) * 4
    uvs = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    indices = [0, 1, 2, 0, 2, 3]
    return positions, normals, uvs, indices


def make_box(center, half_extents):
    """Axis-aligned box from center and half-extents. Returns (pos, nrm, uv, idx)."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    all_pos, all_nrm, all_uv, all_idx = [], [], [], []

    faces = [
        # (corners CCW from outside, normal)
        # Front (+Z)
        ([[cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
          [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz]], [0, 0, 1]),
        # Back (-Z)
        ([[cx+hx, cy-hy, cz-hz], [cx-hx, cy-hy, cz-hz],
          [cx-hx, cy+hy, cz-hz], [cx+hx, cy+hy, cz-hz]], [0, 0, -1]),
        # Right (+X)
        ([[cx+hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz-hz],
          [cx+hx, cy+hy, cz-hz], [cx+hx, cy+hy, cz+hz]], [1, 0, 0]),
        # Left (-X)
        ([[cx-hx, cy-hy, cz-hz], [cx-hx, cy-hy, cz+hz],
          [cx-hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz-hz]], [-1, 0, 0]),
        # Top (+Y)
        ([[cx-hx, cy+hy, cz+hz], [cx+hx, cy+hy, cz+hz],
          [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz]], [0, 1, 0]),
        # Bottom (-Y)
        ([[cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
          [cx+hx, cy-hy, cz+hz], [cx-hx, cy-hy, cz+hz]], [0, -1, 0]),
    ]

    base = 0
    for corners, normal in faces:
        p, n, u, i = make_quad(corners, normal)
        all_pos.extend(p)
        all_nrm.extend(n)
        all_uv.extend(u)
        all_idx.extend([x + base for x in i])
        base += 4

    return all_pos, all_nrm, all_uv, all_idx


def make_uv_sphere(radius=1.0, rings=32, sectors=32):
    """UV sphere centered at origin."""
    positions, normals, uvs, indices = [], [], [], []

    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            positions.extend([x * radius, y * radius, z * radius])
            normals.extend([x, y, z])
            uvs.extend([s / sectors, r / rings])

    for r in range(rings):
        for s in range(sectors):
            a = r * (sectors + 1) + s
            b = a + (sectors + 1)
            indices.extend([a, b, a + 1])
            indices.extend([a + 1, b, b + 1])

    return positions, normals, uvs, indices


# ─── Cornell Box geometry ────────────────────────────────────────────────────


def make_cornell_box():
    """
    Classic Cornell Box: 5 walls (no front wall so camera can see in),
    two blocks inside.
    Box is 2x2x2 centered at (0,1,0), so floor at Y=0, ceiling at Y=2.
    """
    meshes = []  # list of (name, material_idx, pos, nrm, uv, idx, translation)

    # Floor (white)
    p, n, u, i = make_quad(
        [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], [0, 1, 0])
    meshes.append(("Floor", 0, p, n, u, i, None))

    # Ceiling (white)
    p, n, u, i = make_quad(
        [[-1, 2, 1], [1, 2, 1], [1, 2, -1], [-1, 2, -1]], [0, -1, 0])
    meshes.append(("Ceiling", 0, p, n, u, i, None))

    # Back wall (white)
    p, n, u, i = make_quad(
        [[-1, 0, -1], [1, 0, -1], [1, 2, -1], [-1, 2, -1]], [0, 0, 1])
    meshes.append(("BackWall", 0, p, n, u, i, None))

    # Left wall (red)
    p, n, u, i = make_quad(
        [[-1, 0, 1], [-1, 0, -1], [-1, 2, -1], [-1, 2, 1]], [1, 0, 0])
    meshes.append(("LeftWall", 1, p, n, u, i, None))

    # Right wall (green)
    p, n, u, i = make_quad(
        [[1, 0, -1], [1, 0, 1], [1, 2, 1], [1, 2, -1]], [-1, 0, 0])
    meshes.append(("RightWall", 2, p, n, u, i, None))

    # Tall block (white) — rotated ~17 degrees, approximated as axis-aligned
    p, n, u, i = make_box([-0.35, 0.6, -0.35], [0.3, 0.6, 0.3])
    meshes.append(("TallBlock", 0, p, n, u, i, None))

    # Short block (white)
    p, n, u, i = make_box([0.35, 0.3, 0.3], [0.3, 0.3, 0.3])
    meshes.append(("ShortBlock", 0, p, n, u, i, None))

    materials = [
        # 0: White diffuse
        {
            "name": "WhiteDiffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.73, 0.73, 0.73, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "alphaMode": "OPAQUE",
        },
        # 1: Red diffuse
        {
            "name": "RedDiffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.65, 0.05, 0.05, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "alphaMode": "OPAQUE",
        },
        # 2: Green diffuse
        {
            "name": "GreenDiffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.12, 0.45, 0.15, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "alphaMode": "OPAQUE",
        },
    ]

    lights = {
        "lights": [
            {
                "type": "rect",
                "position": [0.0, 1.98, 0.0],
                "orientation": [-90.0, 0.0, 0.0],
                "color": [1.0, 1.0, 1.0],
                "intensity": 400.0,
                "width": 0.5,
                "height": 0.5,
            }
        ]
    }

    camera_pos = [0.0, 1.0, 3.5]
    camera_target = [0.0, 1.0, 0.0]

    return meshes, materials, lights, camera_pos, camera_target


def make_glass_caustics():
    """Glass sphere on diffuse floor with small bright light — tests SDS caustic paths."""
    meshes = []

    # Floor
    p, n, u, i = make_plane(5.0, 0.0)
    meshes.append(("Floor", 0, p, n, u, i, None))

    # Back wall
    p, n, u, i = make_quad(
        [[-5, 0, -3], [5, 0, -3], [5, 5, -3], [-5, 5, -3]], [0, 0, 1])
    meshes.append(("BackWall", 0, p, n, u, i, None))

    # Glass sphere
    p, n, u, i = make_uv_sphere(1.0, 32, 32)
    meshes.append(("GlassSphere", 1, p, n, u, i, [0.0, 1.0, 0.0]))

    materials = [
        # 0: White diffuse floor
        {
            "name": "WhiteDiffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
            "alphaMode": "OPAQUE",
        },
        # 1: Glass (dielectric, BLEND triggers transmission)
        {
            "name": "Glass",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 0.5],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.0,
            },
            "alphaMode": "BLEND",
        },
    ]

    lights = {
        "lights": [
            {
                "type": "rect",
                "position": [2.0, 4.0, 2.0],
                "orientation": [-90.0, 0.0, 0.0],
                "color": [1.0, 0.95, 0.9],
                "intensity": 2000.0,
                "width": 1.0,
                "height": 1.0,
            }
        ]
    }

    camera_pos = [0.0, 1.5, 5.0]
    camera_target = [0.0, 1.5, 0.0]

    return meshes, materials, lights, camera_pos, camera_target


def make_metal_sphere():
    """Mirror conductor sphere on diffuse floor — tests specular reflection paths."""
    meshes = []

    # Floor
    p, n, u, i = make_plane(5.0, 0.0)
    meshes.append(("Floor", 0, p, n, u, i, None))

    # Back wall
    p, n, u, i = make_quad(
        [[-5, 0, -3], [5, 0, -3], [5, 5, -3], [-5, 5, -3]], [0, 0, 1])
    meshes.append(("BackWall", 1, p, n, u, i, None))

    # Mirror sphere
    p, n, u, i = make_uv_sphere(1.0, 32, 32)
    meshes.append(("MirrorSphere", 2, p, n, u, i, [0.0, 1.0, 0.0]))

    materials = [
        # 0: Light gray diffuse floor
        {
            "name": "FloorDiffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.7, 0.7, 0.7, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            },
            "alphaMode": "OPAQUE",
        },
        # 1: Blue-ish diffuse wall
        {
            "name": "WallDiffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.5, 0.5, 0.7, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
            "alphaMode": "OPAQUE",
        },
        # 2: Mirror (gold conductor — high metallic, zero roughness)
        {
            "name": "GoldMirror",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 0.85, 0.57, 1.0],
                "metallicFactor": 1.0,
                "roughnessFactor": 0.0,
            },
            "alphaMode": "OPAQUE",
        },
    ]

    lights = {
        "lights": [
            {
                "type": "rect",
                "position": [-2.0, 4.0, 1.0],
                "orientation": [-90.0, 0.0, 0.0],
                "color": [1.0, 1.0, 1.0],
                "intensity": 1500.0,
                "width": 1.5,
                "height": 1.5,
            }
        ]
    }

    camera_pos = [0.0, 1.5, 5.0]
    camera_target = [0.0, 1.5, 0.0]

    return meshes, materials, lights, camera_pos, camera_target


def make_mixed_materials():
    """Four spheres in a row: diffuse, rough metal, polished metal, glass."""
    meshes = []

    # Floor
    p, n, u, i = make_plane(6.0, 0.0)
    meshes.append(("Floor", 0, p, n, u, i, None))

    # Back wall
    p, n, u, i = make_quad(
        [[-6, 0, -3], [6, 0, -3], [6, 4, -3], [-6, 4, -3]], [0, 0, 1])
    meshes.append(("BackWall", 0, p, n, u, i, None))

    # Sphere 1: Diffuse red
    p, n, u, i = make_uv_sphere(0.8, 32, 32)
    meshes.append(("DiffuseRed", 1, p, n, u, i, [-3.0, 0.8, 0.0]))

    # Sphere 2: Rough metal (copper)
    p, n, u, i = make_uv_sphere(0.8, 32, 32)
    meshes.append(("RoughCopper", 2, p, n, u, i, [-1.0, 0.8, 0.0]))

    # Sphere 3: Polished metal (silver)
    p, n, u, i = make_uv_sphere(0.8, 32, 32)
    meshes.append(("PolishedSilver", 3, p, n, u, i, [1.0, 0.8, 0.0]))

    # Sphere 4: Glass
    p, n, u, i = make_uv_sphere(0.8, 32, 32)
    meshes.append(("GlassSphere", 4, p, n, u, i, [3.0, 0.8, 0.0]))

    materials = [
        # 0: White floor/wall
        {
            "name": "WhiteDiffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
            "alphaMode": "OPAQUE",
        },
        # 1: Diffuse red
        {
            "name": "DiffuseRed",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.1, 0.1, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "alphaMode": "OPAQUE",
        },
        # 2: Rough copper
        {
            "name": "RoughCopper",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.95, 0.64, 0.54, 1.0],
                "metallicFactor": 1.0,
                "roughnessFactor": 0.4,
            },
            "alphaMode": "OPAQUE",
        },
        # 3: Polished silver mirror
        {
            "name": "PolishedSilver",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.97, 0.96, 0.91, 1.0],
                "metallicFactor": 1.0,
                "roughnessFactor": 0.02,
            },
            "alphaMode": "OPAQUE",
        },
        # 4: Clear glass
        {
            "name": "ClearGlass",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 0.5],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.0,
            },
            "alphaMode": "BLEND",
        },
    ]

    lights = {
        "lights": [
            {
                "type": "rect",
                "position": [0.0, 5.0, 2.0],
                "orientation": [-90.0, 0.0, 0.0],
                "color": [1.0, 1.0, 1.0],
                "intensity": 2000.0,
                "width": 3.0,
                "height": 1.5,
            }
        ]
    }

    camera_pos = [0.0, 1.2, 7.0]
    camera_target = [0.0, 1.2, 0.0]

    return meshes, materials, lights, camera_pos, camera_target


def make_small_light():
    """
    Enclosed room with a tiny bright light.
    PT needs many samples; BDPT converges much faster by tracing from the light.
    """
    meshes = []

    # Floor
    p, n, u, i = make_quad(
        [[-3, 0, -3], [3, 0, -3], [3, 0, 3], [-3, 0, 3]], [0, 1, 0])
    meshes.append(("Floor", 0, p, n, u, i, None))

    # Ceiling
    p, n, u, i = make_quad(
        [[-3, 4, 3], [3, 4, 3], [3, 4, -3], [-3, 4, -3]], [0, -1, 0])
    meshes.append(("Ceiling", 0, p, n, u, i, None))

    # Back wall
    p, n, u, i = make_quad(
        [[-3, 0, -3], [3, 0, -3], [3, 4, -3], [-3, 4, -3]], [0, 0, 1])
    meshes.append(("BackWall", 0, p, n, u, i, None))

    # Left wall
    p, n, u, i = make_quad(
        [[-3, 0, 3], [-3, 0, -3], [-3, 4, -3], [-3, 4, 3]], [1, 0, 0])
    meshes.append(("LeftWall", 1, p, n, u, i, None))

    # Right wall
    p, n, u, i = make_quad(
        [[3, 0, -3], [3, 0, 3], [3, 4, 3], [3, 4, -3]], [-1, 0, 0])
    meshes.append(("RightWall", 2, p, n, u, i, None))

    # Glass sphere — creates interesting caustics under small light
    p, n, u, i = make_uv_sphere(0.8, 32, 32)
    meshes.append(("GlassSphere", 3, p, n, u, i, [0.0, 0.8, 0.0]))

    materials = [
        # 0: White diffuse
        {
            "name": "WhiteDiffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "alphaMode": "OPAQUE",
        },
        # 1: Warm diffuse (left wall)
        {
            "name": "WarmDiffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.6, 0.3, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "alphaMode": "OPAQUE",
        },
        # 2: Cool diffuse (right wall)
        {
            "name": "CoolDiffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.3, 0.5, 0.8, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "alphaMode": "OPAQUE",
        },
        # 3: Clear glass
        {
            "name": "Glass",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 0.5],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.0,
            },
            "alphaMode": "BLEND",
        },
    ]

    # Very small, very bright light — hard for PT to find
    lights = {
        "lights": [
            {
                "type": "rect",
                "position": [0.0, 3.98, 0.0],
                "orientation": [-90.0, 0.0, 0.0],
                "color": [1.0, 0.95, 0.85],
                "intensity": 5000.0,
                "width": 0.3,
                "height": 0.3,
            }
        ]
    }

    camera_pos = [0.0, 2.0, 5.5]
    camera_target = [0.0, 2.0, 0.0]

    return meshes, materials, lights, camera_pos, camera_target


def make_indirect_cove():
    """
    Indirect (cove) lighting: the area light is tucked into a recess near the
    ceiling behind a horizontal lip, so the camera never sees it directly and
    almost every visible surface is lit ONLY by light that has bounced off the
    ceiling first (>=2 bounces). Pure-diffuse, so PT, BDPT and VCM MUST converge
    to the *same* image -- a strict energy/correctness test -- while PT is very
    noisy and BDPT/VCM converge far faster.
    """
    meshes = []

    # Closed box: x in [-1,1], z in [-1,1], y in [0,2], open front (+z) for camera.
    p, n, u, i = make_quad([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], [0, 1, 0])
    meshes.append(("Floor", 0, p, n, u, i, None))
    p, n, u, i = make_quad([[-1, 2, 1], [1, 2, 1], [1, 2, -1], [-1, 2, -1]], [0, -1, 0])
    meshes.append(("Ceiling", 0, p, n, u, i, None))
    p, n, u, i = make_quad([[-1, 0, -1], [1, 0, -1], [1, 2, -1], [-1, 2, -1]], [0, 0, 1])
    meshes.append(("BackWall", 0, p, n, u, i, None))
    p, n, u, i = make_quad([[-1, 0, 1], [-1, 0, -1], [-1, 2, -1], [-1, 2, 1]], [1, 0, 0])
    meshes.append(("LeftWall", 1, p, n, u, i, None))
    p, n, u, i = make_quad([[1, 0, -1], [1, 0, 1], [1, 2, 1], [1, 2, -1]], [-1, 0, 0])
    meshes.append(("RightWall", 2, p, n, u, i, None))

    # Two diffuse blocks in the lower room (receive only indirect light).
    p, n, u, i = make_box([-0.4, 0.5, -0.3], [0.28, 0.5, 0.28])
    meshes.append(("TallBlock", 0, p, n, u, i, None))
    p, n, u, i = make_box([0.45, 0.25, 0.35], [0.28, 0.25, 0.28])
    meshes.append(("ShortBlock", 0, p, n, u, i, None))

    # Horizontal "lip": full-width shelf at y=1.6 covering the back third of the
    # box (z in [-1,-0.55]). The light sits above it facing UP; the lip hides the
    # light from the camera and blocks the direct downward path, so the room is
    # lit by the ceiling bounce.
    p, n, u, i = make_quad(
        [[-1, 1.6, -0.55], [1, 1.6, -0.55], [1, 1.6, -1], [-1, 1.6, -1]], [0, 1, 0])
    meshes.append(("CoveLip", 0, p, n, u, i, None))
    # Underside of the lip (faces down so it is opaque from below too).
    p, n, u, i = make_quad(
        [[-1, 1.6, -1], [1, 1.6, -1], [1, 1.6, -0.55], [-1, 1.6, -0.55]], [0, -1, 0])
    meshes.append(("CoveLipBottom", 0, p, n, u, i, None))

    materials = [
        {"name": "WhiteDiffuse",
         "pbrMetallicRoughness": {"baseColorFactor": [0.75, 0.75, 0.75, 1.0],
                                  "metallicFactor": 0.0, "roughnessFactor": 1.0},
         "alphaMode": "OPAQUE"},
        {"name": "RedDiffuse",
         "pbrMetallicRoughness": {"baseColorFactor": [0.63, 0.06, 0.06, 1.0],
                                  "metallicFactor": 0.0, "roughnessFactor": 1.0},
         "alphaMode": "OPAQUE"},
        {"name": "GreenDiffuse",
         "pbrMetallicRoughness": {"baseColorFactor": [0.12, 0.45, 0.15, 1.0],
                                  "metallicFactor": 0.0, "roughnessFactor": 1.0},
         "alphaMode": "OPAQUE"},
    ]

    # Light in the cove (y=1.75, above the lip at y=1.6), facing UP (+Y) toward
    # the ceiling. orientation +90 deg about X flips the default down-facing rect
    # to face up.
    lights = {
        "lights": [
            {
                "type": "rect",
                "position": [0.0, 1.75, -0.78],
                "orientation": [90.0, 0.0, 0.0],
                "color": [1.0, 0.98, 0.95],
                "intensity": 1200.0,
                "width": 1.4,
                "height": 0.3,
            }
        ]
    }

    camera_pos = [0.0, 1.0, 3.6]
    camera_target = [0.0, 0.9, 0.0]
    return meshes, materials, lights, camera_pos, camera_target


def make_caustic_pool():
    """
    Direct view of a refractive caustic. A clear glass sphere sits just above a
    diffuse floor in an enclosed room, lit from directly overhead. Light focused
    THROUGH the glass forms a bright caustic on the floor -- an L-S-D-E path that
    PT/BDPT reach only by chance (very noisy) while VCM resolves it cleanly via
    photon merging. The room is closed so the frame is fully lit (no black void),
    giving a fair energy comparison; VCM should show markedly lower caustic noise.
    """
    meshes = []

    # Enclosed room: x in [-2.5,2.5], z in [-2,2.5], y in [0,3], open front (+z).
    p, n, u, i = make_quad(
        [[-2.5, 0, -2], [2.5, 0, -2], [2.5, 0, 2.5], [-2.5, 0, 2.5]], [0, 1, 0])
    meshes.append(("Floor", 0, p, n, u, i, None))
    p, n, u, i = make_quad(
        [[-2.5, 3, 2.5], [2.5, 3, 2.5], [2.5, 3, -2], [-2.5, 3, -2]], [0, -1, 0])
    meshes.append(("Ceiling", 0, p, n, u, i, None))
    p, n, u, i = make_quad(
        [[-2.5, 0, -2], [2.5, 0, -2], [2.5, 3, -2], [-2.5, 3, -2]], [0, 0, 1])
    meshes.append(("BackWall", 0, p, n, u, i, None))
    p, n, u, i = make_quad(
        [[-2.5, 0, 2.5], [-2.5, 0, -2], [-2.5, 3, -2], [-2.5, 3, 2.5]], [1, 0, 0])
    meshes.append(("LeftWall", 0, p, n, u, i, None))
    p, n, u, i = make_quad(
        [[2.5, 0, -2], [2.5, 0, 2.5], [2.5, 3, 2.5], [2.5, 3, -2]], [-1, 0, 0])
    meshes.append(("RightWall", 0, p, n, u, i, None))

    # Clear glass sphere raised off the floor so the light it focuses lands on
    # the floor as a caustic spot separated from the contact shadow.
    p, n, u, i = make_uv_sphere(0.7, 48, 48)
    meshes.append(("GlassSphere", 1, p, n, u, i, [0.0, 1.35, 0.0]))

    materials = [
        {"name": "WhiteDiffuse",
         "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                                  "metallicFactor": 0.0, "roughnessFactor": 1.0},
         "alphaMode": "OPAQUE"},
        # Clear glass (BLEND -> transmission, roughness 0 -> smooth dielectric).
        {"name": "Glass",
         "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 0.4],
                                  "metallicFactor": 0.0, "roughnessFactor": 0.0},
         "alphaMode": "BLEND"},
    ]

    # Bright light in the ceiling straight above the sphere, small for a focused
    # caustic.
    lights = {
        "lights": [
            {
                "type": "rect",
                "position": [0.0, 2.98, 0.0],
                "orientation": [-90.0, 0.0, 0.0],
                "color": [1.0, 0.97, 0.92],
                "intensity": 3000.0,
                "width": 0.7,
                "height": 0.7,
            }
        ]
    }

    # Eye-level camera looking slightly down to frame the sphere and the caustic
    # spot it casts on the floor below it.
    camera_pos = [0.0, 1.5, 3.9]
    camera_target = [0.0, 0.55, -0.2]
    return meshes, materials, lights, camera_pos, camera_target


# ─── glTF binary builder ────────────────────────────────────────────────────


def pack_floats(data):
    return struct.pack(f"<{len(data)}f", *data)


def pack_uint16(data):
    return struct.pack(f"<{len(data)}H", *data)


def pack_uint32(data):
    return struct.pack(f"<{len(data)}I", *data)


def pad_to_4(data):
    remainder = len(data) % 4
    if remainder:
        data += b"\x00" * (4 - remainder)
    return data


def compute_min_max(flat, stride=3):
    mn = [float("inf")] * stride
    mx = [float("-inf")] * stride
    for i in range(0, len(flat), stride):
        for c in range(stride):
            mn[c] = min(mn[c], flat[i + c])
            mx[c] = max(mx[c], flat[i + c])
    return mn, mx


def rotation_quat_look_at(eye, target):
    """Compute a glTF rotation quaternion [x,y,z,w] to orient camera from eye toward target."""
    ex, ey, ez = eye
    tx, ty, tz = target
    fwd = [tx - ex, ty - ey, tz - ez]
    flen = math.sqrt(sum(c * c for c in fwd))
    fwd = [c / flen for c in fwd]

    src = [0.0, 0.0, -1.0]
    dot = sum(a * b for a, b in zip(src, fwd))

    if dot < -0.9999:
        return [0.0, 1.0, 0.0, 0.0]
    elif dot > 0.9999:
        return [0.0, 0.0, 0.0, 1.0]

    cx = src[1] * fwd[2] - src[2] * fwd[1]
    cy = src[2] * fwd[0] - src[0] * fwd[2]
    cz = src[0] * fwd[1] - src[1] * fwd[0]
    w = 1.0 + dot

    qlen = math.sqrt(cx * cx + cy * cy + cz * cz + w * w)
    return [round(cx / qlen, 6), round(cy / qlen, 6), round(cz / qlen, 6), round(w / qlen, 6)]


def build_glb(output_path, meshes, materials_json, camera_pos, camera_target):
    """
    Build a binary glTF (.glb) file.

    meshes: list of (name, material_idx, positions, normals, uvs, indices, translation_or_None)
    """

    # Pack each mesh's buffers
    mesh_buffers = []
    for name, mat_idx, pos, nrm, uv, idx, trans in meshes:
        max_idx = max(idx)
        use_u32 = max_idx > 65535

        pos_bin = pack_floats(pos)
        nrm_bin = pack_floats(nrm)
        uv_bin = pack_floats(uv)
        idx_bin = pack_uint32(idx) if use_u32 else pack_uint16(idx)

        mesh_buffers.append({
            "name": name,
            "mat_idx": mat_idx,
            "translation": trans,
            "pos": pos, "nrm": nrm, "uv": uv, "idx": idx,
            "pos_bin": pos_bin,
            "nrm_bin": nrm_bin,
            "uv_bin": uv_bin,
            "idx_bin": idx_bin,
            "use_u32": use_u32,
        })

    # Concatenate all binary chunks with 4-byte alignment
    chunks = []
    for mb in mesh_buffers:
        chunks.append(mb["pos_bin"])
        chunks.append(mb["nrm_bin"])
        chunks.append(mb["uv_bin"])
        chunks.append(pad_to_4(mb["idx_bin"]))

    offsets = []
    offset = 0
    for chunk in chunks:
        offsets.append(offset)
        offset += len(chunk)

    buffer_data = b"".join(chunks)

    # Build accessors & buffer views
    accessors = []
    buffer_views = []
    gltf_meshes = []

    bv_idx = 0
    acc_idx = 0

    for i, mb in enumerate(mesh_buffers):
        chunk_base = i * 4  # 4 chunks per mesh (pos, nrm, uv, idx)

        pos_min, pos_max = compute_min_max(mb["pos"])

        # Position buffer view + accessor
        buffer_views.append({
            "buffer": 0, "byteOffset": offsets[chunk_base],
            "byteLength": len(mb["pos_bin"]), "target": 34962,
        })
        accessors.append({
            "bufferView": bv_idx, "componentType": 5126,
            "count": len(mb["pos"]) // 3, "type": "VEC3",
            "min": pos_min, "max": pos_max,
        })
        pos_acc = acc_idx
        bv_idx += 1; acc_idx += 1

        # Normal buffer view + accessor
        buffer_views.append({
            "buffer": 0, "byteOffset": offsets[chunk_base + 1],
            "byteLength": len(mb["nrm_bin"]), "target": 34962,
        })
        accessors.append({
            "bufferView": bv_idx, "componentType": 5126,
            "count": len(mb["nrm"]) // 3, "type": "VEC3",
        })
        nrm_acc = acc_idx
        bv_idx += 1; acc_idx += 1

        # UV buffer view + accessor
        buffer_views.append({
            "buffer": 0, "byteOffset": offsets[chunk_base + 2],
            "byteLength": len(mb["uv_bin"]), "target": 34962,
        })
        accessors.append({
            "bufferView": bv_idx, "componentType": 5126,
            "count": len(mb["uv"]) // 2, "type": "VEC2",
        })
        uv_acc = acc_idx
        bv_idx += 1; acc_idx += 1

        # Index buffer view + accessor
        idx_raw_size = len(mb["idx_bin"])
        buffer_views.append({
            "buffer": 0, "byteOffset": offsets[chunk_base + 3],
            "byteLength": idx_raw_size, "target": 34963,
        })
        accessors.append({
            "bufferView": bv_idx,
            "componentType": 5125 if mb["use_u32"] else 5123,
            "count": len(mb["idx"]), "type": "SCALAR",
        })
        idx_acc = acc_idx
        bv_idx += 1; acc_idx += 1

        gltf_meshes.append({
            "name": mb["name"],
            "primitives": [{
                "attributes": {
                    "POSITION": pos_acc,
                    "NORMAL": nrm_acc,
                    "TEXCOORD_0": uv_acc,
                },
                "indices": idx_acc,
                "material": mb["mat_idx"],
            }],
        })

    # Build nodes
    nodes = []
    scene_nodes = []
    for i, mb in enumerate(mesh_buffers):
        node = {"name": mb["name"], "mesh": i}
        if mb["translation"]:
            node["translation"] = mb["translation"]
        nodes.append(node)
        scene_nodes.append(i)

    # Camera node
    cam_node_idx = len(nodes)
    nodes.append({
        "name": "Camera",
        "camera": 0,
        "translation": camera_pos,
        "rotation": rotation_quat_look_at(camera_pos, camera_target),
    })
    scene_nodes.append(cam_node_idx)

    gltf = {
        "asset": {"version": "2.0", "generator": "strelka_bdpt_test_gen"},
        "scene": 0,
        "scenes": [{"name": "TestScene", "nodes": scene_nodes}],
        "nodes": nodes,
        "cameras": [{
            "name": "Camera",
            "type": "perspective",
            "perspective": {
                "yfov": 0.7854,
                "znear": 0.1,
                "zfar": 100.0,
                "aspectRatio": 1.333,
            },
        }],
        "meshes": gltf_meshes,
        "materials": materials_json,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buffer_data)}],
    }

    # Write GLB
    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bin = json_str.encode("utf-8")
    # GLB spec: pad JSON with spaces (0x20)
    while len(json_bin) % 4:
        json_bin += b" "

    buffer_data = pad_to_4(buffer_data)

    total_length = 12 + (8 + len(json_bin)) + (8 + len(buffer_data))

    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        f.write(struct.pack("<II", len(json_bin), 0x4E4F534A))
        f.write(json_bin)
        f.write(struct.pack("<II", len(buffer_data), 0x004E4942))
        f.write(buffer_data)

    print(f"  GLB: {output_path}")


# ─── TOML config generation ─────────────────────────────────────────────────


def write_toml_config(path, scene_glb, output_exr, integrator, camera_pos, camera_target,
                      spp=256, max_depth=10, width=512, height=384):
    """Write a TOML render config."""
    content = f"""[scene]
path = "{scene_glb}"

[output]
path = "{output_exr}"
width = {width}
height = {height}

[render]
integrator = "{integrator}"
spp = {spp}
spp_per_launch = 1
max_depth = {max_depth}
sampler = "sobol"

[camera]
index = 0
position = [{camera_pos[0]}, {camera_pos[1]}, {camera_pos[2]}]
target = [{camera_target[0]}, {camera_target[1]}, {camera_target[2]}]
fov = 45.0

[tonemap]
type = "aces"
gamma = 0.0
exposure_iso = 400.0
exposure_fstop = 2.8
exposure_shutter = 60.0
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"  TOML: {path}")


# ─── Main ────────────────────────────────────────────────────────────────────


SCENES = {
    "cornell_box": {
        "desc": "Classic Cornell Box (diffuse walls, two blocks)",
        "func": make_cornell_box,
        "spp": 256,
        "depth": 8,
    },
    "glass_caustics": {
        "desc": "Glass sphere on diffuse floor (caustics test)",
        "func": make_glass_caustics,
        "spp": 512,
        "depth": 12,
    },
    "metal_sphere": {
        "desc": "Gold mirror sphere on diffuse floor",
        "func": make_metal_sphere,
        "spp": 256,
        "depth": 8,
    },
    "mixed_materials": {
        "desc": "Diffuse, rough metal, polished metal, glass spheres",
        "func": make_mixed_materials,
        "spp": 512,
        "depth": 12,
    },
    "small_light": {
        "desc": "Tiny bright light in enclosed room with glass sphere",
        "func": make_small_light,
        "spp": 512,
        "depth": 12,
    },
    "indirect_cove": {
        "desc": "Hidden cove light, diffuse room lit indirectly (BDPT/VCM vs PT)",
        "func": make_indirect_cove,
        "spp": 512,
        "depth": 12,
    },
    "caustic_pool": {
        "desc": "Refractive caustic through glass onto diffuse floor (VCM core test)",
        "func": make_caustic_pool,
        "spp": 512,
        "depth": 12,
    },
}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    base_dir = os.path.join(repo_root, "scenes", "bdpt_tests")

    for scene_name, scene_info in SCENES.items():
        print(f"\n=== {scene_name}: {scene_info['desc']} ===")

        scene_dir = os.path.join(base_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        meshes, materials, lights, cam_pos, cam_target = scene_info["func"]()

        glb_path = os.path.join(scene_dir, f"{scene_name}.glb")
        light_path = os.path.join(scene_dir, f"{scene_name}_light.json")

        # Write GLB
        build_glb(glb_path, meshes, materials, cam_pos, cam_target)

        # Write light JSON
        with open(light_path, "w") as f:
            json.dump(lights, f, indent=4)
        print(f"  Light: {light_path}")

        # Write TOML configs for each integrator
        # Use relative paths from build/Release/ where CLI runs
        rel_glb = f"../../scenes/bdpt_tests/{scene_name}/{scene_name}.glb"

        for integrator in ["pt", "bdpt", "vcm"]:
            toml_path = os.path.join(scene_dir, f"{scene_name}_{integrator}.toml")
            output_exr = f"output/{scene_name}_{integrator}.exr"

            write_toml_config(
                toml_path,
                rel_glb,
                output_exr,
                integrator,
                cam_pos, cam_target,
                spp=scene_info["spp"],
                max_depth=scene_info["depth"],
            )

    print(f"\n{'='*60}")
    print(f"All scenes generated in: {base_dir}")
    print(f"\nTo render all tests:")
    print(f"  cd build/Release")
    print(f"  bash ../../scripts/render_bdpt_tests.sh")


if __name__ == "__main__":
    main()
