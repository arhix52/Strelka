#!/usr/bin/env python3
"""
Generate a minimal test scene for comparing PT vs BDPT liquid material rendering.

Scene: ground plane + liquid sphere + rect area light + perspective camera.
Output: scenes/test_liquid/test_liquid.glb + test_liquid_light.json

No external dependencies — uses only Python stdlib.
"""

import json
import math
import os
import struct

# ─── Geometry helpers ────────────────────────────────────────────────────────

def make_plane(half_extent=5.0):
    """10x10 ground plane at Y=0, facing up (+Y normal)."""
    h = half_extent
    positions = [
        -h, 0.0, -h,
         h, 0.0, -h,
         h, 0.0,  h,
        -h, 0.0,  h,
    ]
    normals = [0.0, 1.0, 0.0] * 4
    uvs = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    indices = [0, 1, 2, 0, 2, 3]
    return positions, normals, uvs, indices


def make_uv_sphere(radius=1.0, rings=32, sectors=32):
    """UV sphere centered at origin."""
    positions = []
    normals = []
    uvs = []
    indices = []

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


# ─── glTF binary builder ────────────────────────────────────────────────────

def pack_floats(data):
    return struct.pack(f"<{len(data)}f", *data)


def pack_uint16(data):
    return struct.pack(f"<{len(data)}H", *data)


def pack_uint32(data):
    return struct.pack(f"<{len(data)}I", *data)


def pad_to_4(data):
    """Pad bytes to 4-byte alignment."""
    remainder = len(data) % 4
    if remainder:
        data += b"\x00" * (4 - remainder)
    return data


def compute_min_max(flat, stride=3):
    """Compute component-wise min/max of vec3 data."""
    mn = [float("inf")] * stride
    mx = [float("-inf")] * stride
    for i in range(0, len(flat), stride):
        for c in range(stride):
            mn[c] = min(mn[c], flat[i + c])
            mx[c] = max(mx[c], flat[i + c])
    return mn, mx


def build_glb(output_path):
    """Build a binary glTF (.glb) file."""

    # Generate geometry
    plane_pos, plane_nrm, plane_uv, plane_idx = make_plane(5.0)
    sphere_pos, sphere_nrm, sphere_uv, sphere_idx = make_uv_sphere(1.0, 32, 32)

    # Decide index type: uint16 if possible, else uint32
    sphere_max_idx = max(sphere_idx)
    use_uint32_sphere = sphere_max_idx > 65535

    # Pack binary buffers
    plane_pos_bin = pack_floats(plane_pos)
    plane_nrm_bin = pack_floats(plane_nrm)
    plane_uv_bin = pack_floats(plane_uv)
    plane_idx_bin = pack_uint16(plane_idx)

    sphere_pos_bin = pack_floats(sphere_pos)
    sphere_nrm_bin = pack_floats(sphere_nrm)
    sphere_uv_bin = pack_floats(sphere_uv)
    if use_uint32_sphere:
        sphere_idx_bin = pack_uint32(sphere_idx)
    else:
        sphere_idx_bin = pack_uint16(sphere_idx)

    # Concatenate all buffers (pad each to 4-byte alignment)
    chunks = [
        plane_pos_bin, plane_nrm_bin, plane_uv_bin, pad_to_4(plane_idx_bin),
        sphere_pos_bin, sphere_nrm_bin, sphere_uv_bin, pad_to_4(sphere_idx_bin),
    ]

    # Compute byte offsets
    offsets = []
    offset = 0
    for chunk in chunks:
        offsets.append(offset)
        offset += len(chunk)

    buffer_data = b"".join(chunks)

    # Sizes
    plane_pos_size = len(plane_pos_bin)
    plane_nrm_size = len(plane_nrm_bin)
    plane_uv_size = len(plane_uv_bin)
    plane_idx_size = len(plane_idx_bin)
    plane_idx_padded = len(pad_to_4(plane_idx_bin))

    sphere_pos_size = len(sphere_pos_bin)
    sphere_nrm_size = len(sphere_nrm_bin)
    sphere_uv_size = len(sphere_uv_bin)
    sphere_idx_size = len(sphere_idx_bin)

    plane_pos_min, plane_pos_max = compute_min_max(plane_pos)
    sphere_pos_min, sphere_pos_max = compute_min_max(sphere_pos)

    # ── Build glTF JSON ──

    gltf = {
        "asset": {"version": "2.0", "generator": "strelka_test_gen"},
        "scene": 0,
        "scenes": [{"name": "TestLiquid", "nodes": [0, 1, 2]}],
        "nodes": [
            {"name": "Ground", "mesh": 0},
            {"name": "LiquidSphere", "mesh": 1, "translation": [0.0, 1.0, 0.0]},
            {
                "name": "Camera",
                "camera": 0,
                "translation": [3.0, 3.0, 5.0],
                "rotation": rotation_quat_look_at([3, 3, 5], [0, 1, 0]),
            },
        ],
        "cameras": [
            {
                "name": "Camera",
                "type": "perspective",
                "perspective": {
                    "yfov": 0.7854,
                    "znear": 0.1,
                    "zfar": 100.0,
                    "aspectRatio": 1.333,
                },
            }
        ],
        "meshes": [
            {
                "name": "Ground",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                        "indices": 3,
                        "material": 0,
                    }
                ],
            },
            {
                "name": "LiquidSphere",
                "primitives": [
                    {
                        "attributes": {"POSITION": 4, "NORMAL": 5, "TEXCOORD_0": 6},
                        "indices": 7,
                        "material": 1,
                    }
                ],
            },
        ],
        "materials": [
            {
                "name": "GroundDiffuse",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.7, 0.7, 0.7, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
                "alphaMode": "OPAQUE",
            },
            {
                "name": "Liquid",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.9, 0.95, 1.0, 0.5],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.0,
                },
                "alphaMode": "BLEND",
            },
        ],
        "accessors": [
            # 0: plane positions
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(plane_pos) // 3,
                "type": "VEC3",
                "min": plane_pos_min,
                "max": plane_pos_max,
            },
            # 1: plane normals
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(plane_nrm) // 3,
                "type": "VEC3",
            },
            # 2: plane UVs
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": len(plane_uv) // 2,
                "type": "VEC2",
            },
            # 3: plane indices
            {
                "bufferView": 3,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(plane_idx),
                "type": "SCALAR",
            },
            # 4: sphere positions
            {
                "bufferView": 4,
                "componentType": 5126,
                "count": len(sphere_pos) // 3,
                "type": "VEC3",
                "min": sphere_pos_min,
                "max": sphere_pos_max,
            },
            # 5: sphere normals
            {
                "bufferView": 5,
                "componentType": 5126,
                "count": len(sphere_nrm) // 3,
                "type": "VEC3",
            },
            # 6: sphere UVs
            {
                "bufferView": 6,
                "componentType": 5126,
                "count": len(sphere_uv) // 2,
                "type": "VEC2",
            },
            # 7: sphere indices
            {
                "bufferView": 7,
                "componentType": 5125 if use_uint32_sphere else 5123,
                "count": len(sphere_idx),
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            # 0: plane pos
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": plane_pos_size, "target": 34962},
            # 1: plane nrm
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": plane_nrm_size, "target": 34962},
            # 2: plane uv
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": plane_uv_size, "target": 34962},
            # 3: plane idx
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": plane_idx_size, "target": 34963},
            # 4: sphere pos
            {"buffer": 0, "byteOffset": offsets[4], "byteLength": sphere_pos_size, "target": 34962},
            # 5: sphere nrm
            {"buffer": 0, "byteOffset": offsets[5], "byteLength": sphere_nrm_size, "target": 34962},
            # 6: sphere uv
            {"buffer": 0, "byteOffset": offsets[6], "byteLength": sphere_uv_size, "target": 34962},
            # 7: sphere idx
            {"buffer": 0, "byteOffset": offsets[7], "byteLength": sphere_idx_size, "target": 34963},
        ],
        "buffers": [{"byteLength": len(buffer_data)}],
    }

    # ── Write GLB ──

    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bin = json_str.encode("utf-8")
    json_bin = pad_to_4(json_bin)  # pad JSON chunk to 4 bytes with spaces
    # GLB spec says JSON padding should be 0x20 (space), not 0x00
    while len(json_bin) % 4:
        json_bin += b" "

    buffer_data = pad_to_4(buffer_data)

    # GLB header: magic + version + length
    total_length = 12 + (8 + len(json_bin)) + (8 + len(buffer_data))

    with open(output_path, "wb") as f:
        # GLB header
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))  # glTF magic, version 2
        # JSON chunk
        f.write(struct.pack("<II", len(json_bin), 0x4E4F534A))  # JSON chunk type
        f.write(json_bin)
        # BIN chunk
        f.write(struct.pack("<II", len(buffer_data), 0x004E4942))  # BIN chunk type
        f.write(buffer_data)

    print(f"Written: {output_path}")


def rotation_quat_look_at(eye, target):
    """Compute a glTF rotation quaternion [x,y,z,w] to orient camera from eye toward target.
    glTF cameras look down -Z in their local space."""
    # Forward direction (camera looks down -Z)
    ex, ey, ez = eye
    tx, ty, tz = target
    fwd = [tx - ex, ty - ey, tz - ez]
    flen = math.sqrt(sum(c * c for c in fwd))
    fwd = [c / flen for c in fwd]

    # Camera -Z should point along fwd, so camera +Z points along -fwd
    # We need rotation from (0,0,-1) to fwd
    # Using quaternion from two vectors:
    src = [0.0, 0.0, -1.0]
    dot = sum(a * b for a, b in zip(src, fwd))

    if dot < -0.9999:
        # Nearly opposite, rotate 180 around up
        return [0.0, 1.0, 0.0, 0.0]
    elif dot > 0.9999:
        return [0.0, 0.0, 0.0, 1.0]

    # cross product
    cx = src[1] * fwd[2] - src[2] * fwd[1]
    cy = src[2] * fwd[0] - src[0] * fwd[2]
    cz = src[0] * fwd[1] - src[1] * fwd[0]
    w = 1.0 + dot

    # normalize quaternion
    qlen = math.sqrt(cx * cx + cy * cy + cz * cz + w * w)
    return [round(cx / qlen, 6), round(cy / qlen, 6), round(cz / qlen, 6), round(w / qlen, 6)]


def write_light_json(output_path):
    """Write the companion light file for the scene."""
    data = {
        "lights": [
            {
                "type": "rect",
                "position": [0.0, 4.0, 0.0],
                "orientation": [90.0, 0.0, 0.0],
                "color": [1.0, 1.0, 1.0],
                "intensity": 100.0,
                "width": 2.0,
                "height": 2.0,
            }
        ]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Written: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    out_dir = os.path.join(repo_root, "scenes", "test_liquid")
    os.makedirs(out_dir, exist_ok=True)

    glb_path = os.path.join(out_dir, "test_liquid.glb")
    light_path = os.path.join(out_dir, "test_liquid_light.json")

    build_glb(glb_path)
    write_light_json(light_path)

    print(f"\nTo render:")
    print(f"  cd build/Release")
    print(f"  ./StrelkaEditor -s ../../scenes/test_liquid/test_liquid.glb")


if __name__ == "__main__":
    main()
