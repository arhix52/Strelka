#!/usr/bin/env python3
"""Read the useful public structure of a 3ds Max scene without 3ds Max.

    python inspect_max.py scene.max --objects Sofa Cushion
    python inspect_max.py scene.max --all-materials --output materials.json

The native format is an OLE compound file whose Scene stream contains nested
chunks.  Autodesk documents the scene and ParamBlock2 concepts but not the
complete byte layout.  This tool deliberately exports raw IDs alongside the
few value encodings that can be verified from the scene, so unknown data is
never silently guessed.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

NULL_REF = 0xFFFFFFFF


def chunks(data: bytes, start: int = 0, end: int | None = None):
    """Yield (type, is_container, payload_start, payload_end) Max chunks."""
    end = len(data) if end is None else end
    offset = start
    while offset < end:
        if offset + 6 > end:
            raise ValueError(f"truncated chunk header at {offset}")
        chunk_type, signed_size = struct.unpack_from("<Hi", data, offset)
        header_size = 6
        if signed_size == 0:
            if offset + 14 > end:
                raise ValueError(f"truncated extended chunk header at {offset}")
            signed_size = struct.unpack_from("<q", data, offset + 6)[0]
            header_size = 14
            chunk_size = signed_size & 0x7FFFFFFFFFFFFFFF
        else:
            chunk_size = signed_size & 0x7FFFFFFF
        if chunk_size < header_size or offset + chunk_size > end:
            raise ValueError(
                f"invalid chunk 0x{chunk_type:04x} at {offset}: size={chunk_size}, end={end}"
            )
        yield chunk_type, signed_size < 0, offset + header_size, offset + chunk_size
        offset += chunk_size


def utf16(data: bytes, start: int, end: int) -> str:
    return data[start:end].decode("utf-16le", "replace").rstrip("\0")


def uints(data: bytes, start: int, end: int) -> list[int]:
    size = end - start
    if size % 4:
        return []
    return list(struct.unpack_from("<" + "I" * (size // 4), data, start))


def direct_metadata(data: bytes, node: tuple[int, bool, int, int]):
    _, _, start, end = node
    name = ""
    refs: list[int] = []
    typed: list[int] = []
    for chunk_type, _, payload_start, payload_end in chunks(data, start, end):
        if chunk_type == 0x0962:
            name = utf16(data, payload_start, payload_end)
        elif chunk_type == 0x2034:
            refs = uints(data, payload_start, payload_end)
        elif chunk_type == 0x2035:
            typed = uints(data, payload_start, payload_end)
    typed_refs = {
        str(typed[index]): typed[index + 1]
        for index in range(1, len(typed) - 1, 2)
    }
    return name, refs, typed_refs


def first_guid(data: bytes, start: int, end: int) -> bytes | None:
    for chunk_type, is_container, payload_start, payload_end in chunks(data, start, end):
        if chunk_type == 0x0002 and payload_end - payload_start == 16:
            return data[payload_start:payload_end]
        if is_container:
            found = first_guid(data, payload_start, payload_end)
            if found:
                return found
    return None


def asset_metadata(metadata: bytes, guid: bytes) -> dict | None:
    offset = metadata.find(guid)
    if offset < 0:
        return None
    offset += 16

    def read_string() -> str:
        nonlocal offset
        length = struct.unpack_from("<I", metadata, offset)[0]
        offset += 4
        value = metadata[offset : offset + length * 2].decode("utf-16le", "replace")
        offset += (length + 1) * 2  # stored strings have a trailing wide NUL
        return value

    try:
        return {"kind": read_string(), "path": read_string()}
    except (struct.error, UnicodeDecodeError):
        return {"guid": guid.hex()}


def decode_pb_value(param_type: int, raw: bytes):
    if not raw:
        return None
    if param_type & 0x0800 and len(raw) >= 4:
        count = struct.unpack_from("<I", raw, 0)[0]
        base_type = param_type & 0x07FF
        offset = 4
        values = []
        for _ in range(count):
            if offset >= len(raw):
                break
            offset += 1  # per-element storage flag
            if offset + 4 > len(raw):
                break
            if base_type == 0:
                values.append(struct.unpack_from("<f", raw, offset)[0])
            elif base_type in (1, 4):
                values.append(struct.unpack_from("<i", raw, offset)[0])
            else:
                break
            offset += 4
        if len(values) == count:
            return values
    if len(raw) == 1:
        return raw[0]
    if len(raw) == 4:
        if param_type in (0, 7):
            return struct.unpack("<f", raw)[0]
        return struct.unpack("<i", raw)[0]
    if len(raw) in (8, 12, 16) and param_type in (2, 3):
        return list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
    return {"hex": raw.hex()}


def paramblock2(data: bytes, node: tuple[int, bool, int, int], typed_refs: dict[str, int]):
    _, _, start, end = node
    params = []
    explicit_ref_slots: dict[int, int] = {}
    for chunk_type, _, payload_start, payload_end in chunks(data, start, end):
        raw = data[payload_start:payload_end]
        if chunk_type == 0x100E and len(raw) >= 15:
            param_id = struct.unpack_from("<H", raw, 0)[0]
            param_type = struct.unpack_from("<I", raw, 2)[0]
            value_raw = raw[15:]
            params.append(
                {
                    "id": f"0x{param_id:04x}",
                    "type": f"0x{param_type:04x}",
                    "value": decode_pb_value(param_type, value_raw),
                    "raw": raw.hex(),
                }
            )
        elif chunk_type == 0x0013 and len(raw) >= 4:
            count = struct.unpack_from("<I", raw, 0)[0]
            offset = 4
            for _ in range(count):
                if offset + 6 > len(raw):
                    break
                param_id, ref_slot = struct.unpack_from("<IH", raw, offset)
                explicit_ref_slots[param_id] = ref_slot
                offset += 6

    reference_params = [p for p in params if p["type"] == "0x000f"]
    used_slots = set(explicit_ref_slots.values())
    free_slots = (slot for slot in range(256) if slot not in used_slots)
    for param in reference_params:
        param_id = int(param["id"], 16)
        slot = explicit_ref_slots.get(param_id)
        if slot is None:
            slot = next(free_slots)
        param["reference_slot"] = slot
        param["reference_node"] = typed_refs.get(str(slot), NULL_REF)
    return params


def paramblock1(data: bytes, node: tuple[int, bool, int, int]):
    _, _, start, end = node
    params = []
    for chunk_type, is_container, payload_start, payload_end in chunks(data, start, end):
        if chunk_type != 0x0002 or not is_container:
            continue
        param_id = None
        value = None
        raw_value = None
        for child_type, _, child_start, child_end in chunks(data, payload_start, payload_end):
            raw = data[child_start:child_end]
            if child_type == 0x0003 and len(raw) == 4:
                param_id = struct.unpack("<I", raw)[0]
            elif child_type == 0x0100 and len(raw) == 4:
                value = struct.unpack("<f", raw)[0]
                raw_value = raw.hex()
            elif child_type in (0x0101, 0x0102) and len(raw) == 4:
                value = struct.unpack("<i", raw)[0]
                raw_value = raw.hex()
        if param_id is not None:
            params.append({"id": param_id, "value": value, "raw": raw_value})
    return params


def corona_legacy_values(graph: list[dict]) -> dict | None:
    """Return the stable CoronaLegacy surface constants used by its converter."""
    ids = {
        "diffuse_color": "0x0065",
        "reflection_color": "0x0066",
        "refraction_color": "0x0067",
        "diffuse_level": "0x0079",
        "reflection_level": "0x007a",
        "refraction_level": "0x007b",
        "reflection_glossiness": "0x00b4",
        "refraction_glossiness": "0x00b5",
        "fresnel_ior": "0x00b6",
        "refraction_ior": "0x00b7",
    }
    block = next(
        (
            node.get("parameters", [])
            for node in graph
            if node.get("class", {}).get("name") == "ParamBlock2"
            and any(param.get("id") == ids["diffuse_color"] for param in node.get("parameters", []))
        ),
        None,
    )
    if block is None:
        return None
    params = {param["id"]: param for param in block}
    by_id = {param_id: param.get("value") for param_id, param in params.items()}
    if not all(param_id in by_id for param_id in ids.values()):
        return None
    result = {name: by_id[param_id] for name, param_id in ids.items()}
    reflect = sum(result["reflection_color"]) / 3.0 * result["reflection_level"]
    # This is the exact constant-IOR branch in Chaos' CoronaLegacy -> Physical
    # converter. Texture-driven reflection stays in the raw graph inventory.
    result["physical_specular_ior"] = 1.0 + (result["fresnel_ior"] - 1.0) * reflect

    nodes = {node["index"]: node for node in graph if "index" in node}

    def bitmap_nodes(index: int):
        """Bitmap identities below one map slot, preserving shared Max maps."""
        pending = [index]
        visited = set()
        result = set()
        while pending:
            current = pending.pop()
            if current == NULL_REF or current in visited or current not in nodes:
                continue
            visited.add(current)
            node = nodes[current]
            if node.get("class", {}).get("name") in {"Bitmap", "位图"}:
                result.add(current)
            pending.extend(node.get("references", []))
            pending.extend(node.get("typed_references", {}).values())
        return result

    def map_source(index: int):
        node = nodes.get(index)
        if not node:
            return index
        if node.get("class", {}).get("name") in {"Output", "输出"}:
            return next((ref for ref in node.get("references", []) if ref in nodes), index)
        return index

    def constant_color(index: int):
        node = nodes.get(index)
        if not node or node.get("class", {}).get("name") != "CoronaColor":
            return None
        for ref in (*node.get("references", []), *node.get("typed_references", {}).values()):
            block = nodes.get(ref)
            if not block or block.get("class", {}).get("name") != "ParamBlock2":
                continue
            color = next((param.get("value") for param in block.get("parameters", [])
                          if param.get("id") == "0x0034"), None)
            if isinstance(color, list) and len(color) == 3:
                return color
        return None

    def output_gain(index: int):
        node = nodes.get(index)
        if not node or node.get("class", {}).get("name") not in {"Output", "输出"}:
            return 1.0
        for ref in node.get("references", []):
            controller = nodes.get(ref)
            if not controller or controller.get("class", {}).get("name") not in {"Output", "输出"}:
                continue
            for child in controller.get("references", []):
                pb = nodes.get(child)
                if pb and pb.get("class", {}).get("name") == "ParamBlock":
                    values = {param["id"]: param.get("value") for param in pb.get("parameters", [])}
                    if 2 in values:
                        return values[2]
        return 1.0

    diffuse_map = params.get("0x008d", {}).get("reference_node", NULL_REF)
    reflection_map = params.get("0x008e", {}).get("reference_node", NULL_REF)
    gloss_map = params.get("0x008f", {}).get("reference_node", NULL_REF)
    bump_map = params.get("0x0090", {}).get("reference_node", NULL_REF)
    diffuse_bitmaps = bitmap_nodes(diffuse_map)
    data_bitmaps = bitmap_nodes(gloss_map) | bitmap_nodes(bump_map)
    result["shared_diffuse_data_bitmaps"] = bool(diffuse_bitmaps & data_bitmaps)
    result["diffuse_map"] = diffuse_map != NULL_REF
    if diffuse_map != NULL_REF:
        result["diffuse_map_amount"] = by_id.get("0x0122", 1.0)
        color = constant_color(diffuse_map)
        if color is not None:
            result["diffuse_map_color"] = color
    if reflection_map != NULL_REF:
        source = map_source(reflection_map)
        result["reflection_map"] = True
        result["reflection_map_amount"] = by_id.get("0x0123", 1.0)
        result["reflection_map_gain"] = output_gain(reflection_map)
        color = constant_color(reflection_map)
        if color is not None:
            result["reflection_map_color"] = color
        result["reflection_map_source"] = (
            "color" if source == map_source(diffuse_map) else
            "data" if source in {map_source(gloss_map), map_source(bump_map)} else
            "unsupported"
        )
    else:
        result["reflection_map"] = False
    if gloss_map != NULL_REF:
        source = map_source(gloss_map)
        result["gloss_map"] = True
        result["gloss_map_amount"] = by_id.get("0x0124", 1.0)
        result["gloss_map_gain"] = output_gain(gloss_map)
        result["gloss_map_source"] = (
            "color" if source == map_source(diffuse_map) else
            "data" if source in {map_source(reflection_map), map_source(bump_map)} else
            "unsupported"
        )
    else:
        result["gloss_map"] = False
    result["bump_map"] = bump_map != NULL_REF
    result["bump_map_amount"] = by_id.get("0x0128", 1.0)
    return result


class MaxScene:
    def __init__(self, path: Path):
        try:
            import olefile
        except ImportError as exc:
            raise SystemExit("inspect_max.py requires 'olefile' (python -m pip install olefile)") from exc
        with olefile.OleFileIO(str(path)) as ole:
            self.scene = ole.openstream("Scene").read()
            self.class_directory = ole.openstream("ClassDirectory3").read()
            self.assets = ole.openstream("FileAssetMetaData3").read()

        root_type, is_container, start, end = next(chunks(self.scene))
        if root_type != 0x2028 or not is_container:
            raise ValueError("Scene stream has no 0x2028 root container")
        self.nodes = list(chunks(self.scene, start, end))
        self.classes = self._read_classes()

    def _read_classes(self):
        result = []
        for _, _, start, end in chunks(self.class_directory):
            entry = {"name": "", "guid": None, "super_id": None, "dll_index": None}
            for chunk_type, _, payload_start, payload_end in chunks(
                self.class_directory, start, end
            ):
                if chunk_type == 0x2042:
                    entry["name"] = utf16(self.class_directory, payload_start, payload_end)
                elif chunk_type == 0x2060 and payload_end - payload_start >= 16:
                    dll_index, guid, super_id = struct.unpack_from(
                        "<IQI", self.class_directory, payload_start
                    )
                    entry.update(
                        dll_index=dll_index,
                        guid=f"0x{guid:016x}",
                        super_id=f"0x{super_id:08x}",
                    )
            result.append(entry)
        return result

    def named_nodes(self, names: set[str]):
        found = {}
        for index, node in enumerate(self.nodes):
            name, _, _ = direct_metadata(self.scene, node)
            if name in names:
                found[name] = index
        return found

    def node_json(self, index: int):
        node = self.nodes[index]
        class_index, _, start, end = node
        name, refs, typed_refs = direct_metadata(self.scene, node)
        class_info = (
            self.classes[class_index]
            if class_index < len(self.classes)
            else {"name": f"class_{class_index}"}
        )
        result = {
            "index": index,
            "class_index": class_index,
            "class": class_info,
            "name": name or None,
            "references": refs,
            "typed_references": typed_refs,
        }
        guid = first_guid(self.scene, start, end)
        if guid:
            result["asset"] = asset_metadata(self.assets, guid)
        if class_info.get("name") == "ParamBlock2":
            result["parameters"] = paramblock2(self.scene, node, typed_refs)
        elif class_info.get("name") == "ParamBlock":
            result["parameters"] = paramblock1(self.scene, node)
        return result

    def material_graph(self, object_index: int):
        object_json = self.node_json(object_index)
        material_index = object_json["typed_references"].get("3")
        if material_index is None:
            return {"object": object_json, "material_root": None, "nodes": []}

        return {
            "object": object_json,
            "material_root": material_index,
            "nodes": self.graph_from_root(material_index),
        }

    def graph_from_root(self, material_index: int):
        """Follow one material root without duplicating its object bindings."""

        pending = [material_index]
        visited = set()
        graph = []
        while pending:
            index = pending.pop()
            if index == NULL_REF or index in visited or index >= len(self.nodes):
                continue
            visited.add(index)
            node = self.node_json(index)
            graph.append(node)
            pending.extend(node["references"])
            pending.extend(node["typed_references"].values())
        graph.sort(key=lambda item: item["index"])
        return graph

    def material_graphs(self):
        """Return every unique material graph and the scene nodes using it."""
        users: dict[int, list[str]] = {}
        node_class_indices = {
            index for index, info in enumerate(self.classes) if info.get("name") == "Node"
        }
        for index, node in enumerate(self.nodes):
            if node[0] not in node_class_indices:
                continue
            name, _, typed_refs = direct_metadata(self.scene, node)
            material_index = typed_refs.get("3")
            if material_index is not None and material_index != NULL_REF:
                users.setdefault(material_index, []).append(name or f"node_{index}")

        return {
            str(root): {
                "objects": sorted(names),
                "nodes": self.graph_from_root(root),
            }
            for root, names in sorted(users.items())
        }

    def material_audit(self, gltf_path: Path | None = None):
        """Compact inventory of every root graph, without the 5 MB raw PB dump."""
        gltf_users = {}
        if gltf_path:
            gltf = json.loads(gltf_path.read_text(encoding="utf-8"))
            materials = [item.get("name", "") for item in gltf.get("materials", [])]
            meshes = gltf.get("meshes", [])
            for node in gltf.get("nodes", []):
                mesh_index = node.get("mesh")
                if mesh_index is None or mesh_index >= len(meshes):
                    continue
                names = set()
                for primitive in meshes[mesh_index].get("primitives", []):
                    material_index = primitive.get("material")
                    if material_index is not None and material_index < len(materials):
                        names.add(materials[material_index])
                gltf_users.setdefault(node.get("name", ""), set()).update(names)

        direct_classes = {
            "", "ParamBlock", "ParamBlock2", "CoronaLegacyMtl", "Color Correction",
            "位图", "Bitmap", "放置", "Placement", "输出", "Output", "曲线控制",
            "Curve", "合成", "Composite",
        }
        roots = []
        statuses = Counter()
        classes = Counter()
        for root, item in self.material_graphs().items():
            graph_classes = Counter(node["class"].get("name") or "" for node in item["nodes"])
            classes.update(graph_classes)
            class_names = set(graph_classes)
            if class_names <= direct_classes:
                status = "bitmap_stack"
            elif "多维/子对象" in class_names or "Multi/Sub-Object" in class_names:
                status = "multi_sub"
            else:
                status = "procedural"
            statuses[status] += 1
            assets = sorted({
                Path(node["asset"]["path"].replace("\\", "/")).name
                for node in item["nodes"] if (node.get("asset") or {}).get("path")
            })
            root_node = self.node_json(int(root))
            submaterials = []
            if status == "multi_sub":
                for slot, child in enumerate(root_node.get("references", [])):
                    if child == NULL_REF:
                        continue
                    child_graph = self.graph_from_root(child)
                    submaterials.append({
                        "slot": slot,
                        "root": child,
                        "assets": sorted({
                            Path(node["asset"]["path"].replace("\\", "/")).name
                            for node in child_graph if (node.get("asset") or {}).get("path")
                        }),
                        "corona_legacy": corona_legacy_values(child_graph),
                    })
            roots.append({
                "root": int(root),
                "status": status,
                "objects": item["objects"],
                "gltf_materials": sorted({
                    material for obj in item["objects"] for material in gltf_users.get(obj, set())
                }),
                "classes": dict(sorted(graph_classes.items())),
                "assets": assets,
                "corona_legacy": corona_legacy_values(item["nodes"]),
                "corona_submaterials": submaterials,
            })
        return {
            "root_count": len(roots),
            "status_counts": dict(sorted(statuses.items())),
            "class_counts": dict(sorted(classes.items())),
            "roots": roots,
        }

    def summary(self):
        counts = Counter(node[0] for node in self.nodes)
        classes = []
        for class_index, count in counts.most_common():
            info = self.classes[class_index] if class_index < len(self.classes) else {}
            classes.append({"index": class_index, "name": info.get("name"), "count": count})
        return {"node_count": len(self.nodes), "classes": classes}


def self_test():
    primitive = struct.pack("<HiI", 0x1234, 10, 7)
    container_size = 6 + len(primitive)
    container = struct.pack("<HI", 0x2028, 0x80000000 | container_size) + primitive
    parsed = list(chunks(container))
    assert parsed == [(0x2028, True, 6, len(container))]
    assert list(chunks(container, parsed[0][2], parsed[0][3]))[0][0] == 0x1234
    assert decode_pb_value(0, struct.pack("<f", 0.5)) == 0.5
    fake = [{"index": 1, "class": {"name": "ParamBlock2"}, "parameters": [
        {"id": "0x0065", "value": [0.5, 0.5, 0.5]},
        {"id": "0x0066", "value": [0.25, 0.25, 0.25]},
        {"id": "0x0067", "value": [1.0, 1.0, 1.0]},
        {"id": "0x0079", "value": 1.0}, {"id": "0x007a", "value": 1.0},
        {"id": "0x007b", "value": 0.0}, {"id": "0x00b4", "value": 0.8},
        {"id": "0x00b5", "value": 1.0}, {"id": "0x00b6", "value": 1.52},
        {"id": "0x00b7", "value": 1.52},
        {"id": "0x008d", "reference_node": 2}, {"id": "0x0122", "value": 0.5},
        {"id": "0x008e", "reference_node": 4}, {"id": "0x0123", "value": 0.75},
    ]},
        {"index": 2, "class": {"name": "CoronaColor"}, "references": [3]},
        {"index": 3, "class": {"name": "ParamBlock2"}, "parameters": [
            {"id": "0x0034", "value": [0.1, 0.2, 0.3]}]},
        {"index": 4, "class": {"name": "CoronaColor"}, "references": [5]},
        {"index": 5, "class": {"name": "ParamBlock2"}, "parameters": [
            {"id": "0x0034", "value": [0.6, 0.5, 0.4]}]},
    ]
    values = corona_legacy_values(fake)
    assert abs(values["physical_specular_ior"] - 1.13) < 1e-6
    assert values["diffuse_map_color"] == [0.1, 0.2, 0.3]
    assert values["reflection_map_color"] == [0.6, 0.5, 0.4]
    print("inspect_max self-test: OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", nargs="?", type=Path)
    parser.add_argument("--objects", nargs="+", default=[])
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--audit-materials", action="store_true")
    parser.add_argument("--gltf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.scene:
        parser.error("scene is required unless --self-test is used")

    scene = MaxScene(args.scene)
    if args.audit_materials:
        output = {
            "source": str(args.scene.resolve()),
            **scene.material_audit(args.gltf),
        }
    elif args.all_materials:
        output = {
            "source": str(args.scene.resolve()),
            "materials": scene.material_graphs(),
        }
    elif args.objects:
        named = scene.named_nodes(set(args.objects))
        missing = sorted(set(args.objects) - named.keys())
        if missing:
            raise SystemExit("objects not found: " + ", ".join(missing))
        output = {
            "source": str(args.scene.resolve()),
            "objects": {
                name: scene.material_graph(index) for name, index in sorted(named.items())
            },
        }
    else:
        output = scene.summary()

    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
