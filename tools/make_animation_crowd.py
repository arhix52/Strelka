#!/usr/bin/env python3
"""Build a self-contained animated crowd GLB from one skinned glTF asset."""

import argparse
import copy
import json
import math
import struct
from pathlib import Path


GLB_JSON = 0x4E4F534A
GLB_BIN = 0x004E4942


def read_glb(path: Path) -> tuple[dict, bytearray]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"{path}: truncated GLB")
    magic, version, total = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError(f"{path}: expected a complete GLB 2.0 file")

    chunks: dict[int, bytes] = {}
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks[kind] = data[offset : offset + length]
        offset += length
    if GLB_JSON not in chunks or GLB_BIN not in chunks:
        raise ValueError(f"{path}: embedded JSON and BIN chunks are required")

    document = json.loads(chunks[GLB_JSON].rstrip(b" \0"))
    buffers = document.get("buffers", [])
    if len(buffers) != 1 or "uri" in buffers[0]:
        raise ValueError(f"{path}: crowd generation supports one embedded buffer")
    return document, bytearray(chunks[GLB_BIN][: buffers[0]["byteLength"]])


def write_glb(path: Path, document: dict, binary: bytearray) -> None:
    document["buffers"][0]["byteLength"] = len(binary)
    json_bytes = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(struct.pack("<4sII", b"glTF", 2, total))
        output.write(struct.pack("<II", len(json_bytes), GLB_JSON))
        output.write(json_bytes)
        output.write(struct.pack("<II", len(binary), GLB_BIN))
        output.write(binary)


def accessor_floats(document: dict, binary: bytearray, accessor_id: int) -> list[float]:
    accessor = document["accessors"][accessor_id]
    if accessor.get("componentType") != 5126 or accessor.get("type") != "SCALAR" or "sparse" in accessor:
        raise ValueError(f"animation input accessor {accessor_id} must be dense float SCALAR")
    view = document["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise ValueError(f"animation input accessor {accessor_id} is not in buffer 0")
    stride = view.get("byteStride", 4)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return [struct.unpack_from("<f", binary, start + index * stride)[0] for index in range(accessor["count"])]


def scaled_time_accessor(document: dict, binary: bytearray, accessor_id: int, scale: float) -> int:
    values = [value * scale for value in accessor_floats(document, binary, accessor_id)]
    binary += b"\0" * (-len(binary) % 4)
    offset = len(binary)
    binary.extend(struct.pack(f"<{len(values)}f", *values))

    view_id = len(document["bufferViews"])
    document["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(values) * 4})
    accessor = copy.deepcopy(document["accessors"][accessor_id])
    accessor["bufferView"] = view_id
    accessor["byteOffset"] = 0
    accessor["min"] = [min(values)]
    accessor["max"] = [max(values)]
    new_id = len(document["accessors"])
    document["accessors"].append(accessor)
    return new_id


def translation_accessor(document: dict, binary: bytearray,
                         translations: list[tuple[float, float, float]]) -> int:
    binary += b"\0" * (-len(binary) % 4)
    offset = len(binary)
    flat = [component for translation in translations for component in translation]
    binary.extend(struct.pack(f"<{len(flat)}f", *flat))
    view_id = len(document["bufferViews"])
    document["bufferViews"].append({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": len(flat) * 4,
    })
    accessor_id = len(document["accessors"])
    document["accessors"].append({
        "bufferView": view_id,
        "componentType": 5126,
        "count": len(translations),
        "type": "VEC3",
    })
    return accessor_id


def strip_scene_owned_objects(node: dict) -> None:
    node.pop("camera", None)
    extensions = node.get("extensions")
    if extensions:
        extensions.pop("KHR_lights_punctual", None)
        if not extensions:
            node.pop("extensions")


def add_crowd_camera(document: dict, side: int, rows: int, spacing: float) -> None:
    extent = max(side * spacing, rows * spacing)
    position = (0.0, max(5.0, extent * 0.55), max(8.0, extent * 1.05))
    target = (0.0, 0.9, 0.0)

    def normalized(vector: tuple[float, float, float]) -> tuple[float, float, float]:
        length = math.sqrt(sum(component * component for component in vector))
        return tuple(component / length for component in vector)

    def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    forward = normalized(tuple(target[i] - position[i] for i in range(3)))
    right = normalized(cross(forward, (0.0, 1.0, 0.0)))
    up = cross(right, forward)
    matrix = [
        right[0], right[1], right[2], 0.0,
        up[0], up[1], up[2], 0.0,
        -forward[0], -forward[1], -forward[2], 0.0,
        position[0], position[1], position[2], 1.0,
    ]
    document["cameras"] = [{
        "name": "Crowd overview",
        "type": "perspective",
        "perspective": {"yfov": math.radians(45.0), "znear": 0.05, "zfar": max(100.0, extent * 10.0)},
    }]
    camera_node = len(document["nodes"])
    document["nodes"].append({"name": "Crowd overview", "camera": 0, "matrix": matrix})
    document["scenes"][0]["nodes"].append(camera_node)


def duplicate_crowd(document: dict,
                    binary: bytearray,
                    count: int,
                    spacing: float,
                    animated_ratio: float,
                    duration_range: tuple[float, float],
                    duration_buckets: int,
                    clip_id: int) -> dict:
    source_nodes = copy.deepcopy(document.get("nodes", []))
    source_skins = copy.deepcopy(document.get("skins", []))
    source_animations = copy.deepcopy(document.get("animations", []))
    source_scene = document.get("scene", 0)
    source_roots = document["scenes"][source_scene].get("nodes", [])
    if not source_nodes or not source_skins or not source_animations:
        raise ValueError("source must contain nodes, skins, and animations")
    if clip_id < 0 or clip_id >= len(source_animations):
        raise ValueError(f"clip {clip_id} is outside 0..{len(source_animations) - 1}")

    document["nodes"] = []
    document["skins"] = []
    document["animations"] = []
    document["scenes"] = [{"name": "Strelka animation crowd", "nodes": []}]
    document["scene"] = 0

    side = math.ceil(math.sqrt(count))
    rows = math.ceil(count / side)
    animated_count = round(count * animated_ratio)
    low, high = duration_range
    input_cache: dict[tuple[int, int], int] = {}

    for copy_id in range(count):
        node_base = len(document["nodes"])
        skin_base = len(document["skins"])
        for source_id, source_node in enumerate(source_nodes):
            node = copy.deepcopy(source_node)
            strip_scene_owned_objects(node)
            node["name"] = f"crowd_{copy_id:04d}/{node.get('name', source_id)}"
            if "children" in node:
                node["children"] = [node_base + child for child in node["children"]]
            if "skin" in node:
                node["skin"] = skin_base + node["skin"]
            document["nodes"].append(node)

        for source_skin in source_skins:
            skin = copy.deepcopy(source_skin)
            skin["name"] = f"crowd_{copy_id:04d}/{skin.get('name', 'skin')}"
            skin["joints"] = [node_base + joint for joint in skin["joints"]]
            if "skeleton" in skin:
                skin["skeleton"] = node_base + skin["skeleton"]
            document["skins"].append(skin)

        column = copy_id % side
        row = copy_id // side
        placement = {
            "name": f"crowd_{copy_id:04d}/placement",
            "translation": [(column - (side - 1) * 0.5) * spacing, 0.0,
                            (row - (rows - 1) * 0.5) * spacing],
            "children": [node_base + root for root in source_roots],
        }
        placement_id = len(document["nodes"])
        document["nodes"].append(placement)
        document["scenes"][0]["nodes"].append(placement_id)

        if copy_id >= animated_count:
            continue
        bucket = copy_id % duration_buckets
        alpha = 0.0 if duration_buckets == 1 else bucket / (duration_buckets - 1)
        duration_scale = low + (high - low) * alpha
        animation = copy.deepcopy(source_animations[clip_id])
        source_name = animation.get("name") or f"clip_{clip_id}"
        animation["name"] = f"crowd_{copy_id:04d}/{source_name}/duration_{duration_scale:.3f}"
        for channel in animation.get("channels", []):
            target = channel.get("target", {})
            if "node" in target:
                target["node"] = node_base + target["node"]
        for sampler in animation.get("samplers", []):
            source_input = sampler["input"]
            key = (source_input, bucket)
            if key not in input_cache:
                input_cache[key] = scaled_time_accessor(document, binary, source_input, duration_scale)
            sampler["input"] = input_cache[key]
        document["animations"].append(animation)

    add_crowd_camera(document, side, rows, spacing)
    extras = document.setdefault("asset", {}).setdefault("extras", {})
    extras["strelkaAnimationCrowd"] = {
        "characters": count,
        "animatedCharacters": animated_count,
        "durationBuckets": duration_buckets,
        "durationScale": [low, high],
    }
    return {"side": side, "rows": rows, "animated": animated_count}


def duplicate_shared_pose_crowd(document: dict,
                                binary: bytearray,
                                count: int,
                                spacing: float,
                                animated_ratio: float,
                                duration_range: tuple[float, float],
                                duration_buckets: int,
                                pose_buckets: int,
                                clip_id: int) -> dict:
    source_nodes = copy.deepcopy(document.get("nodes", []))
    source_skins = copy.deepcopy(document.get("skins", []))
    source_animations = copy.deepcopy(document.get("animations", []))
    source_scene = document.get("scene", 0)
    source_roots = document["scenes"][source_scene].get("nodes", [])
    if not source_nodes or not source_skins or not source_animations:
        raise ValueError("source must contain nodes, skins, and animations")
    if clip_id < 0 or clip_id >= len(source_animations):
        raise ValueError(f"clip {clip_id} is outside 0..{len(source_animations) - 1}")

    document["nodes"] = []
    document["skins"] = []
    document["animations"] = []
    document["scenes"] = [{"name": "Strelka shared-pose animation crowd", "nodes": []}]
    document["scene"] = 0

    side = math.ceil(math.sqrt(count))
    rows = math.ceil(count / side)
    animated_count = round(count * animated_ratio)
    group_count = min(animated_count, pose_buckets)
    low, high = duration_range
    input_cache: dict[tuple[int, int], int] = {}
    positions = [
        ((copy_id % side - (side - 1) * 0.5) * spacing, 0.0,
         (copy_id // side - (rows - 1) * 0.5) * spacing)
        for copy_id in range(count)
    ]

    def add_graph(name: str, placements: list[tuple[float, float, float]]) -> int:
        node_base = len(document["nodes"])
        skin_base = len(document["skins"])
        placement_accessor = translation_accessor(document, binary, placements)
        for source_id, source_node in enumerate(source_nodes):
            node = copy.deepcopy(source_node)
            strip_scene_owned_objects(node)
            node["name"] = f"{name}/{node.get('name', source_id)}"
            if "children" in node:
                node["children"] = [node_base + child for child in node["children"]]
            if "skin" in node:
                node["skin"] = skin_base + node["skin"]
            if "mesh" in node:
                extensions = node.setdefault("extensions", {})
                extensions["EXT_mesh_gpu_instancing"] = {
                    "attributes": {"TRANSLATION": placement_accessor}
                }
            document["nodes"].append(node)

        for source_skin in source_skins:
            skin = copy.deepcopy(source_skin)
            skin["name"] = f"{name}/{skin.get('name', 'skin')}"
            skin["joints"] = [node_base + joint for joint in skin["joints"]]
            if "skeleton" in skin:
                skin["skeleton"] = node_base + skin["skeleton"]
            document["skins"].append(skin)
        document["scenes"][0]["nodes"].extend(node_base + root for root in source_roots)
        return node_base

    for group_id in range(group_count):
        group_positions = [positions[copy_id] for copy_id in range(group_id, animated_count, group_count)]
        node_base = add_graph(f"pose_{group_id:04d}", group_positions)
        bucket = group_id % duration_buckets
        alpha = 0.0 if duration_buckets == 1 else bucket / (duration_buckets - 1)
        duration_scale = low + (high - low) * alpha
        animation = copy.deepcopy(source_animations[clip_id])
        source_name = animation.get("name") or f"clip_{clip_id}"
        animation["name"] = f"pose_{group_id:04d}/{source_name}/duration_{duration_scale:.3f}"
        for channel in animation.get("channels", []):
            target = channel.get("target", {})
            if "node" in target:
                target["node"] = node_base + target["node"]
        for sampler in animation.get("samplers", []):
            source_input = sampler["input"]
            key = (source_input, bucket)
            if key not in input_cache:
                input_cache[key] = scaled_time_accessor(document, binary, source_input, duration_scale)
            sampler["input"] = input_cache[key]
        document["animations"].append(animation)

    if animated_count < count:
        add_graph("idle", positions[animated_count:])

    used = set(document.get("extensionsUsed", []))
    used.add("EXT_mesh_gpu_instancing")
    document["extensionsUsed"] = sorted(used)
    add_crowd_camera(document, side, rows, spacing)
    extras = document.setdefault("asset", {}).setdefault("extras", {})
    extras["strelkaAnimationCrowd"] = {
        "characters": count,
        "animatedCharacters": animated_count,
        "sharedPoseBuckets": group_count,
        "durationBuckets": duration_buckets,
        "durationScale": [low, high],
    }
    return {"side": side, "rows": rows, "animated": animated_count, "pose_buckets": group_count}


def geometry_counts(document: dict) -> tuple[int, int, int]:
    vertices = 0
    triangles = 0
    primitives = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            position = primitive.get("attributes", {}).get("POSITION")
            if position is not None:
                vertices += document["accessors"][position]["count"]
            if "indices" in primitive:
                triangles += document["accessors"][primitive["indices"]]["count"] // 3
            primitives += 1
    return vertices, triangles, primitives


def write_light(path: Path, side: int, rows: int, spacing: float) -> None:
    width = max(4.0, side * spacing)
    depth = max(4.0, rows * spacing)
    payload = {
        "lights": [{
            "type": "rect",
            "name": "crowd_key",
            "position": [0.0, max(5.0, 0.75 * max(width, depth)), 0.0],
            "orientation": [-90.0, 0.0, 0.0],
            "color": [1.0, 0.97, 0.92],
            "intensity": 1400.0,
            "width": width,
            "height": depth,
        }]
    }
    path.with_name(path.stem + "_light.json").write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path,
                        default=root / "scenes/validation/brainstem/BrainStem.glb")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--spacing", type=float, default=3.0)
    parser.add_argument("--animated-ratio", type=float, default=0.75)
    parser.add_argument("--duration-scale", type=float, nargs=2, default=(0.8, 1.2), metavar=("MIN", "MAX"))
    parser.add_argument("--duration-buckets", type=int, default=7)
    parser.add_argument("--shared-pose-buckets", type=int, default=0,
                        help="share skinned geometry within N quantized animation poses")
    parser.add_argument("--clip", type=int, default=0)
    parser.add_argument("--no-light", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.duration_buckets < 1 or args.shared_pose_buckets < 0 or args.spacing <= 0.0:
        parser.error("count, duration-buckets, and spacing must be positive")
    if not 0.0 <= args.animated_ratio <= 1.0:
        parser.error("animated-ratio must be in [0, 1]")
    if args.duration_scale[0] <= 0.0 or args.duration_scale[1] < args.duration_scale[0]:
        parser.error("duration-scale must be positive and ordered")
    if args.output is None:
        args.output = root / f"build/profiles/brainstem-crowd-{args.count}.glb"
    return args


def main() -> None:
    args = parse_args()
    document, binary = read_glb(args.source)
    vertices, triangles, primitives = geometry_counts(document)
    if args.shared_pose_buckets:
        result = duplicate_shared_pose_crowd(document, binary, args.count, args.spacing, args.animated_ratio,
                                             tuple(args.duration_scale), args.duration_buckets,
                                             args.shared_pose_buckets, args.clip)
    else:
        result = duplicate_crowd(document, binary, args.count, args.spacing, args.animated_ratio,
                                 tuple(args.duration_scale), args.duration_buckets, args.clip)
    write_glb(args.output, document, binary)
    if not args.no_light:
        write_light(args.output, result["side"], result["rows"], args.spacing)

    geometry_copies = result.get("pose_buckets", result["animated"]) + (result["animated"] < args.count)
    estimated_geometry = geometry_copies * (vertices * (32 + 64 + 32 + 32) + triangles * 3 * 4)
    print(f"wrote {args.output}")
    print(f"characters={args.count} animated={result['animated']} clips={len(document['animations'])} "
          f"shared_pose_buckets={result.get('pose_buckets', 0)}")
    print(f"source vertices={vertices} triangles={triangles} primitives={primitives}")
    print(f"geometry estimate={estimated_geometry / (1024 ** 2):.1f} MiB before BLAS")


if __name__ == "__main__":
    main()
