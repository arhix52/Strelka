#!/usr/bin/env python3
"""Export the non-procedural part of imported 3ds Max materials without baking.

Run through Blender so the original material graphs are available::

    blender -b interior.blend --python export_native_materials.py -- \
        --gltf scenes/interior/interior.gltf

Only a deliberately small, exact graph family is accepted: Bitmap, one Max
Color Correction, one Max Output, Corona's usual inverted-glossiness mix, and
height Bump. Every rejected material and the first unsupported node is written
to an audit JSON beside the MaterialX document.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gltf_to_mtlx import convert, fmt  # noqa: E402


BEGIN = "  <!-- BEGIN STRELKA_NATIVE_MATERIALS -->"
END = "  <!-- END STRELKA_NATIVE_MATERIALS -->"
AUTHORED_NATIVE_UV = {"mint0006", "mint0126"}
AUTHORED_MATERIALX = {"mint0006"}


class Unsupported(ValueError):
    pass


def _value(socket):
    if socket is None or socket.links:
        raise Unsupported("animated/connected parameter")
    value = socket.default_value
    if hasattr(value, "__len__"):
        return tuple(float(x) for x in value)
    return float(value)


def _source(socket):
    if socket is None or not socket.links:
        return None
    node = socket.links[0].from_node
    while node.type == "REROUTE":
        if not node.inputs[0].links:
            return None
        node = node.inputs[0].links[0].from_node
    return node


def _input(node, identifier, name=None):
    return next((socket for socket in node.inputs
                 if socket.identifier == identifier or (name and socket.name == name)), None)


def _texture(socket, correction=None, gain=1.0):
    node = _source(socket)
    if node is None:
        raise Unsupported("texture input is not connected")

    if node.type == "GROUP" and node.node_tree and node.node_tree.name == "MAX_Output":
        if gain != 1.0:
            raise Unsupported("nested Max Output")
        return _texture(node.inputs.get("Color"), correction, _value(node.inputs.get("Value")))

    if node.type == "GROUP" and node.node_tree and node.node_tree.name == "MAX_Color_Correction":
        if correction is not None:
            raise Unsupported("nested Max Color Correction")
        keys = ("Hue", "Saturation", "Value", "Brightness", "Contrast", "Gamma")
        values = tuple(_value(node.inputs.get(key)) for key in keys)
        correction = {
            "adjust": (values[0] - 0.5, values[1], values[2]),
            "tone": (values[5], values[3], values[4]),
        }
        return _texture(node.inputs.get("Input"), correction, gain)

    if node.type != "TEX_IMAGE" or node.image is None:
        raise Unsupported(node.type if node.type != "GROUP" else
                          "group:" + (node.node_tree.name if node.node_tree else node.name))

    scale = (1.0, 1.0)
    offset = (0.0, 0.0)
    rotation = 0.0
    vector = node.inputs.get("Vector")
    mapping = _source(vector)
    if mapping is not None:
        if not (mapping.type == "GROUP" and mapping.node_tree and
                mapping.node_tree.name == "MAX_Texture_Mapping"):
            raise Unsupported("image vector:" + mapping.type)
        scale3 = _value(mapping.inputs.get("Scale"))
        location = _value(mapping.inputs.get("Location"))
        rotation = _value(mapping.inputs.get("Angle"))
        for key, expected in (("Crop Enable", 1.0), ("CropU", 0.0), ("CropV", 0.0),
                              ("CropWidth", 1.0), ("CropHeight", 1.0)):
            if abs(_value(mapping.inputs.get(key)) - expected) > 1.0e-7:
                raise Unsupported("cropped Max mapping")
        scale = scale3[:2]
        # MAX_Texture_Mapping negates Location before Blender's Mapping node.
        offset = (-location[0], -location[1])

    correction = correction or {"adjust": (0.0, 1.0, 1.0), "tone": (1.0, 0.0, 0.0)}
    return {
        "basename": os.path.basename(bpy.path.abspath(node.image.filepath)),
        "colorspace": "srgb_texture" if node.image.colorspace_settings.name == "sRGB" else "lin_rec709",
        "scale": scale,
        "offset": offset,
        "rotation": rotation,
        "adjust": correction["adjust"],
        "tone": correction["tone"],
        "gain": gain,
    }


def _same_texture(a, b):
    keys = ("basename", "colorspace", "scale", "offset", "rotation", "adjust", "tone")
    return all(a[key] == b[key] for key in keys)


def _identity_correction(texture):
    return texture["adjust"] == (0.0, 1.0, 1.0) and texture["tone"] == (1.0, 0.0, 0.0)


def _stack(socket):
    """Compile the exact Corona Bitmap/Composite subset used by this scene."""
    node = _source(socket)
    if node is None:
        value = _value(socket)
        if hasattr(value, "__len__"):
            return {"base": tuple(value[:3]), "layers": [], "gain": 1.0}
        return {"base": (value, value, value), "layers": [], "gain": 1.0}

    if node.type == "TEX_IMAGE" or (node.type == "GROUP" and node.node_tree and
                                     node.node_tree.name == "MAX_Color_Correction"):
        texture = _texture(socket)
        texture.update({"opacity": 1.0, "blend": "multiply"})
        return {"base": None, "layers": [texture], "gain": texture.pop("gain")}

    if node.type == "GROUP" and node.node_tree and node.node_tree.name == "MAX_Output":
        result = _stack(node.inputs.get("Color"))
        if result["gain"] != 1.0:
            raise Unsupported("nested Max Output")
        result["gain"] = _value(node.inputs.get("Value"))
        return result

    if node.type == "RGB":
        value = node.outputs[0].default_value
        return {"base": tuple(float(x) for x in value[:3]), "layers": [], "gain": 1.0}

    if node.type != "MIX_RGB" or node.blend_type not in {"MIX", "MULTIPLY"}:
        raise Unsupported(node.type if node.type != "GROUP" else
                          "group:" + (node.node_tree.name if node.node_tree else node.name))
    factor = node.inputs.get("Fac")
    if factor is None:
        factor = node.inputs.get("Factor")
    left = node.inputs.get("Color1")
    right = node.inputs.get("Color2")
    if factor is None or factor.links:
        raise Unsupported("textured Composite opacity")
    opacity = _value(factor)
    if node.blend_type == "MIX" and abs(opacity - 1.0) <= 1.0e-7:
        return _stack(right)

    result = _stack(left)
    incoming = _stack(right)
    if incoming["base"] is not None or len(incoming["layers"]) != 1 or incoming["gain"] != 1.0:
        raise Unsupported("Composite layer is not one bitmap")
    layer = incoming["layers"][0]
    if result["layers"] and not _identity_correction(layer):
        raise Unsupported("per-layer Color Correction")
    if result["base"] is not None and not _identity_correction(layer):
        raise Unsupported("Color Correction after constant Composite base")
    layer["opacity"] = opacity
    layer["blend"] = "mix" if node.blend_type == "MIX" else "multiply"
    result["layers"].append(layer)
    return result


def _same_stack(a, b):
    if a["base"] != b["base"] or len(a["layers"]) != len(b["layers"]):
        return False
    for left, right in zip(a["layers"], b["layers"]):
        if left["opacity"] != right["opacity"] or left["blend"] != right["blend"] or \
                not _same_texture(left, right):
            return False
    return True


def _restore_shared_map_colorspace(graph, corona):
    """A shared Max Bitmap has one decode, irrespective of the consuming slot."""
    if not corona or not corona.get("shared_diffuse_data_bitmaps"):
        return 0
    color = graph.get("color")
    data = graph.get("data")
    if not color or not data:
        return 0
    changed = 0
    for data_layer in data["layers"]:
        color_layer = next((layer for layer in color["layers"]
                            if layer["basename"].casefold() == data_layer["basename"].casefold()
                            and layer["scale"] == data_layer["scale"]
                            and layer["offset"] == data_layer["offset"]
                            and layer["rotation"] == data_layer["rotation"]), None)
        if color_layer and data_layer["colorspace"] != color_layer["colorspace"]:
            data_layer["colorspace"] = color_layer["colorspace"]
            changed += 1
    return changed


def _source_repair(material_name):
    """Restore source graphs max2blender could not represent, without baking."""

    def bitmap(name, colorspace="srgb_texture", scale=(1.0, 1.0),
               adjust=(0.0, 1.0, 1.0), tone=(1.0, 0.0, 0.0), **extra):
        return {
            "basename": name, "colorspace": colorspace, "scale": scale,
            "offset": (0.0, 0.0), "rotation": 0.0,
            "adjust": adjust, "tone": tone, "opacity": 1.0,
            "blend": "multiply", **extra,
        }

    if material_name == "mint0125":
        # Exact straight-line form of the supplied Composite/Falloff graph.
        # Facing is Blender Layer Weight at Blend=.5: 1-|N.V|.
        color = [
            bitmap("mint0042.jpg", adjust=(-0.00166112, 1.568106, 1.0), tone=(1.2, 0.0, 0.0)),
            bitmap("mint0043.jpg", opacity=0.2, blend="overlay"),
            bitmap("mint0042.jpg", adjust=(0.00498337, 1.348837, 2.0), tone=(1.8, 0.0, 0.0),
                   blend="screen", factor_base=0.0, factor_facing=0.539,
                   factor_facing_data=0.161, factor_data_layer=1),
            bitmap("mint0044.jpg", adjust=(0.00830567, 1.395349, 2.0), tone=(1.7, 0.0, 0.0),
                   opacity=0.8, factor_base=0.8, factor_facing=-0.8,
                   factor_facing_data=0.144, factor_data_layer=1),
        ]
        data = [
            bitmap("mint0040.jpg", "lin_rec709", scale=(1.5, 1.5)),
            bitmap("mint0045.png", "srgb_texture", scale=(5.0, 5.0), opacity=0.0),
        ]
        return {
            "color": {"base": None, "layers": color, "gain": 1.0,
                      "post_adjust": (0.00830555, 0.458472, 2.0),
                      "post_tone": (3.0, 0.0, 0.0)},
            "data": {"base": None, "layers": data, "gain": 1.0},
            "roughness": None,
            "bump": {"scale": 0.001, "gain": 1.0, "data_layer": 0,
                     "procedural": "voronoi_ridge", "procedural_scale": 100.0,
                     "texture_mix": 0.5},
        }

    if material_name == "mint0129":
        color = [
            bitmap("mint0079.jpg", adjust=(0.0, 1.0, 1.5), tone=(0.7, 0.0, 0.0)),
            bitmap("mint0079.jpg", adjust=(0.0, 1.0, 2.0), tone=(0.7, 0.0, 0.0),
                   blend="mix", factor_base=0.0, factor_facing=1.0),
        ]
        data = [bitmap("mint0079.jpg", "lin_rec709", adjust=(0.0, 1.0, 1.5),
                       tone=(0.7, 0.0, 0.0))]
        return {
            "color": {"base": None, "layers": color, "gain": 1.0},
            "data": {"base": None, "layers": data, "gain": 1.0},
            "roughness": None,
            "bump": {"scale": 0.001, "gain": 1.0, "data_layer": 0},
        }

    if material_name == "mint0074":
        return {
            "color": {"base": (0.02745098, 0.02745098, 0.02745098),
                      "layers": [], "gain": 1.0},
            "data": None,
            "roughness": None,
            "bump": {"scale": 0.001, "gain": 1.0, "data_layer": 4,
                     "procedural": "noise", "procedural_scale": 100.0,
                     "procedural_detail": 1.0, "procedural_roughness": 0.5,
                     "procedural_lacunarity": 2.0, "texture_mix": 0.0},
        }

    if material_name == "mint0128":
        # Tube: the Max graph is a constant CoronaLegacy material with a
        # Noise-driven bump. max2blender represented the noise correctly, but
        # its linked Mix node is outside the bitmap-stack subset above.
        return {
            "color": {"base": (0.0039215689,) * 3, "layers": [], "gain": 1.0},
            "data": None,
            "roughness": None,
            "bump": {"scale": 0.001, "gain": 1.0, "data_layer": 4,
                     "procedural": "noise", "procedural_scale": 100.0,
                     "procedural_detail": 1.0, "procedural_roughness": 0.5,
                     "procedural_lacunarity": 2.0, "texture_mix": 0.0},
        }

    if material_name != "mint0046":
        return None

    # The cabinet-top Bitmap is node 989 in the .max file, but its ParamBlock
    # has no FileAssetMetaData GUID and max2blender consequently left both
    # Color Correction inputs disconnected.  The supplied gray wood map is the
    # one visible on that surface in the authored render.
    color = bitmap("mint0008.jpg", adjust=(0.0, 1.0, 0.5), tone=(0.5, 0.0, 0.0))
    data = bitmap("mint0008.jpg", adjust=(0.0, 1.0, 1.0), tone=(1.0, 0.0, 0.0))
    return {
        "color": {"base": None, "layers": [color], "gain": 1.0},
        "data": {"base": None, "layers": [data], "gain": 1.0},
        "roughness": {"base": 0.2, "mix": 0.7, "gain": 1.0},
        "bump": {"scale": 0.001, "gain": 1.0},
    }


def _roughness(socket):
    node = _source(socket)
    if node is None:
        return None

    base = 0.0
    mix = 1.0
    if node.type == "MIX" and node.data_type == "FLOAT" and node.blend_type == "MIX":
        factor = _input(node, "Factor_Float")
        a = _input(node, "A_Float")
        b = _input(node, "B_Float")
        if factor is None or a is None or b is None or factor.links or a.links:
            raise Unsupported("non-constant roughness mix")
        mix = _value(factor)
        base = _value(a)
        node = _source(b)

    if node is None or node.type != "INVERT":
        raise Unsupported("roughness is not inverted glossiness")
    stack = _stack(node.inputs.get("Color"))
    return stack, base, mix


def _normal(socket):
    node = _source(socket)
    if node is None:
        return None
    if node.type != "BUMP" or _source(node.inputs.get("Normal")) is not None:
        raise Unsupported("normal is not a plain height bump")
    stack = _stack(node.inputs.get("Height"))
    if getattr(node, "invert", False):
        stack["gain"] = -stack["gain"]
    scale = _value(node.inputs.get("Strength")) * _value(node.inputs.get("Distance"))
    return stack, scale


def _surface(material):
    if not material.use_nodes or material.node_tree is None:
        raise Unsupported("no node tree")
    output = next((node for node in material.node_tree.nodes
                   if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
    surface = _source(output.inputs.get("Surface") if output else None)
    if surface is not None and surface.type == "BSDF_PRINCIPLED":
        return surface

    # CoronaRaySwitch exported by max2blender wraps its camera/direct material
    # in a node group and selects GI/reflection branches with Light Path nodes.
    # The direct and GI groups in this scene are byte-for-byte the same graph;
    # compiling that graph is exact for visible and diffuse paths and avoids a
    # UV-atlas bake entirely.
    direct = next((node for node in material.node_tree.nodes
                   if node.type == "GROUP" and node.node_tree and
                   node.node_tree.name.endswith("_convert_direct")), None)
    if direct is not None:
        group_output = next((node for node in direct.node_tree.nodes if node.type == "GROUP_OUTPUT"), None)
        nested = _source(group_output.inputs.get("Shader") if group_output else None)
        if nested is not None and nested.type == "BSDF_PRINCIPLED":
            return nested

    # Shadow-ray transparency is transport metadata, not a different visible
    # shader. The glTF extension already carries it; inspect the Principled arm.
    if surface is not None and surface.type == "MIX_SHADER":
        shaders = [socket for socket in surface.inputs if socket.type == "SHADER" and socket.links]
        principled = next((socket.links[0].from_node for socket in shaders
                           if socket.links[0].from_node.type == "BSDF_PRINCIPLED"), None)
        if principled is not None:
            return principled
    raise Unsupported("surface is not a supported Principled graph")


def inspect_material(material):
    principled = _surface(material)
    base_socket = principled.inputs.get("Base Color")
    base = _stack(base_socket)

    rough = _roughness(principled.inputs.get("Roughness"))
    normal = _normal(principled.inputs.get("Normal"))
    data = rough[0] if rough else normal[0] if normal else None
    if rough and normal and not _same_stack(rough[0], normal[0]):
        raise Unsupported("roughness and bump use different data graphs")

    for name in ("Metallic", "IOR", "Alpha", "Transmission Weight",
                 "Emission Color", "Emission Strength"):
        socket = principled.inputs.get(name)
        if socket is not None and _source(socket) is not None:
            raise Unsupported(name + " is textured/procedural")
    return {
        "color": base,
        "data": data,
        "roughness": {"base": rough[1], "mix": rough[2], "gain": rough[0]["gain"]} if rough else None,
        "bump": {"scale": normal[1], "gain": normal[0]["gain"]} if normal else None,
    }


def _number(value):
    return f"{float(value):.9g}"


def _vector(values):
    return ", ".join(_number(value) for value in values)


def _find_sources(root):
    files = {}
    for path in root.rglob("*"):
        if not path.is_file() or "_baked_" in path.name or "_uv_" in path.name:
            continue
        files.setdefault(path.name.casefold(), []).append(path)
    return files


def _resolve(texture, root, files):
    matches = files.get(texture["basename"].casefold(), [])
    if not matches:
        raise Unsupported("missing source bitmap:" + texture["basename"])
    matches.sort(key=lambda path: ("maps" not in path.parts, len(path.parts), str(path)))
    texture["file"] = matches[0].relative_to(root).as_posix()


def _material_xml(name, graph, inputs):
    safe = "".join(char if (char.isalnum() or char == "_") else "_" for char in name)
    shader = "S_native_" + safe
    texture = "G_native_" + safe
    lines = [f'  <surfacematerial name="{html.escape(name, quote=True)}" type="material">',
             f'    <input name="surfaceshader" type="surfaceshader" nodename="{shader}" />',
             '  </surfacematerial>',
             f'  <open_pbr_surface name="{shader}" type="surfaceshader">']
    for key, (kind, value) in sorted(inputs.items()):
        if (key == "base_color" and graph["color"]) or \
                (key == "specular_roughness" and graph["roughness"]) or \
                (key == "specular_color" and graph.get("specular_color")):
            continue
        lines.append(f'    <input name="{key}" type="{kind}" value="{fmt(kind, value)}" />')
    if graph["color"]:
        if graph["color"]["layers"]:
            lines.append(f'    <input name="base_color" type="color3" nodename="{texture}" output="base_color" />')
        else:
            lines.append(f'    <input name="base_color" type="color3" value="{_vector(graph["color"]["base"])}" />')
    if graph["roughness"]:
        lines.append(f'    <input name="specular_roughness" type="float" nodename="{texture}" output="roughness" />')
    if graph.get("specular_color"):
        lines.append(f'    <input name="specular_color" type="color3" nodename="{texture}" output="specular_color" />')
    if graph["bump"]:
        lines.append(f'    <input name="geometry_normal" type="vector3" nodename="{texture}" output="normal" />')
    lines.extend(['  </open_pbr_surface>',
                  f'  <strelka_layered_texture name="{texture}" type="multioutput">'])
    for prefix in ("color", "data"):
        stack = graph[prefix]
        if not stack:
            continue
        if not stack["layers"]:
            continue
        if stack["base"] is not None:
            lines.append(f'    <input name="{prefix}_base" type="vector3" value="{_vector(stack["base"])}" />')
        for index, source in enumerate(stack["layers"]):
            lines.extend([
                f'    <input name="{prefix}_file{index}" type="filename" value="{html.escape(source["file"], quote=True)}" colorspace="{source["colorspace"]}" />',
                f'    <input name="{prefix}_uv_scale{index}" type="vector2" value="{_vector(source["scale"])}" />',
                f'    <input name="{prefix}_uv_offset{index}" type="vector2" value="{_vector(source["offset"])}" />',
                f'    <input name="{prefix}_uv_rotation{index}" type="float" value="{_number(source["rotation"])}" />',
            ])
        first = stack["layers"][0]
        lines.extend([
            f'    <input name="{prefix}_adjust" type="vector3" value="{_vector(first["adjust"])}" />',
            f'    <input name="{prefix}_tone" type="vector3" value="{_vector(first["tone"])}" />',
        ])
        for index, source in enumerate(stack["layers"]):
            lines.extend([
                f'    <input name="{prefix}_adjust{index}" type="vector3" value="{_vector(source["adjust"])}" />',
                f'    <input name="{prefix}_tone{index}" type="vector3" value="{_vector(source["tone"])}" />',
            ])
        opacity = [layer["opacity"] for layer in stack["layers"]] + [0.0] * (4 - len(stack["layers"]))
        lines.append(f'    <input name="{prefix}_opacity" type="vector4" value="{_vector(opacity)}" />')
        if prefix == "color":
            blend_ids = {"multiply": 0.0, "mix": 1.0, "overlay": 2.0, "screen": 3.0}
            modes = [blend_ids[layer["blend"]] for layer in stack["layers"]] + [0.0] * (4 - len(stack["layers"]))
            base = [layer.get("factor_base", layer["opacity"]) for layer in stack["layers"]]
            data_factor = [layer.get("factor_data", 0.0) for layer in stack["layers"]]
            facing = [layer.get("factor_facing", 0.0) for layer in stack["layers"]]
            facing_data = [layer.get("factor_facing_data", 0.0) for layer in stack["layers"]]
            data_layer = [layer.get("factor_data_layer", 4.0) for layer in stack["layers"]]
            for values in (base, data_factor, facing, facing_data, data_layer):
                values += [0.0 if values is not data_layer else 4.0] * (4 - len(values))
            lines.extend([
                f'    <input name="color_blend_mode" type="vector4" value="{_vector(modes)}" />',
                f'    <input name="color_factor_base" type="vector4" value="{_vector(base)}" />',
                f'    <input name="color_factor_data" type="vector4" value="{_vector(data_factor)}" />',
                f'    <input name="color_factor_facing" type="vector4" value="{_vector(facing)}" />',
                f'    <input name="color_factor_facing_data" type="vector4" value="{_vector(facing_data)}" />',
                f'    <input name="color_factor_data_layer" type="vector4" value="{_vector(data_layer)}" />',
            ])
            if "post_adjust" in stack:
                lines.append(f'    <input name="color_post_adjust" type="vector3" value="{_vector(stack["post_adjust"])}" />')
            if "post_tone" in stack:
                lines.append(f'    <input name="color_post_tone" type="vector3" value="{_vector(stack["post_tone"])}" />')
        else:
            modes = [1.0 if layer["blend"] == "mix" else 0.0 for layer in stack["layers"]]
            modes += [0.0] * (4 - len(modes))
            lines.append(f'    <input name="data_blend_mode" type="vector4" value="{_vector(modes)}" />')
    if graph["roughness"]:
        rough = graph["roughness"]
        lines.extend([
            f'    <input name="roughness_gain" type="float" value="{_number(rough["gain"])}" />',
            f'    <input name="roughness_base" type="float" value="{_number(rough["base"])}" />',
            f'    <input name="roughness_mix" type="float" value="{_number(rough["mix"])}" />',
        ])
    if graph["bump"]:
        bump = graph["bump"]
        procedurals = {None: 0, "voronoi_ridge": 1, "noise": 2}
        lines.extend([
            f'    <input name="bump_gain" type="float" value="{_number(bump["gain"])}" />',
            f'    <input name="bump_scale" type="float" value="{_number(bump["scale"])}" />',
            f'    <input name="bump_data_layer" type="integer" value="{int(bump.get("data_layer", 4))}" />',
            f'    <input name="bump_procedural" type="integer" value="{procedurals[bump.get("procedural")]}" />',
            f'    <input name="bump_procedural_scale" type="float" value="{_number(bump.get("procedural_scale", 1.0))}" />',
            f'    <input name="bump_procedural_detail" type="float" value="{_number(bump.get("procedural_detail", 1.0))}" />',
            f'    <input name="bump_procedural_roughness" type="float" value="{_number(bump.get("procedural_roughness", 0.5))}" />',
            f'    <input name="bump_procedural_lacunarity" type="float" value="{_number(bump.get("procedural_lacunarity", 2.0))}" />',
            f'    <input name="bump_texture_mix" type="float" value="{_number(bump.get("texture_mix", 1.0))}" />',
        ])
    if graph.get("specular_color"):
        specular = graph["specular_color"]
        lines.extend([
            f'    <input name="specular_gain" type="float" value="{_number(specular["gain"])}" />',
            f'    <input name="specular_color_base" type="color3" value="{_vector(specular["base"])}" />',
            f'    <input name="specular_color_mix" type="float" value="{_number(specular["mix"])}" />',
            f'    <input name="specular_color_uses_color" type="integer" value="{int(specular["source"] == "color")}" />',
        ])
    layer_count = max(len(graph[prefix]["layers"]) if graph[prefix] else 0 for prefix in ("color", "data"))
    lines.extend([f'    <input name="layer_count" type="integer" value="{layer_count}" />',
                  '  </strelka_layered_texture>'])
    return lines


def _replace_block(text, lines):
    if BEGIN in text:
        start = text.index(BEGIN)
        end = text.index(END, start) + len(END)
        text = text[:start] + text[end:]
    block = "\n".join([BEGIN, *lines, END])
    if "</materialx>" not in text:
        raise ValueError("not a MaterialX document")
    return text.replace("</materialx>", block + "\n</materialx>")


def _remove_materials(text, names):
    """Remove generic glTF shaders superseded by exact native graphs."""
    for name in names:
        material = re.search(
            rf'\s*<surfacematerial\s+name="{re.escape(name)}".*?</surfacematerial>',
            text,
            re.DOTALL,
        )
        if not material:
            continue
        shader = re.search(r'nodename="([^"]+)"', material.group())
        text = text[:material.start()] + text[material.end():]
        if shader:
            text = re.sub(
                rf'\s*<open_pbr_surface\s+name="{re.escape(shader.group(1))}".*?</open_pbr_surface>',
                "",
                text,
                count=1,
                flags=re.DOTALL,
            )
    return text


def _match_corona_submaterial(candidates, assets, base=None, roughness=None, ior=None):
    """Match a Max Multi/Sub slot without requiring texture inputs to be constants."""
    assets = {name.casefold() for name in assets}
    ranked = []
    for item in candidates:
        candidate_assets = {name.casefold() for name in item.get("assets", [])}
        overlap = len(assets & candidate_assets)
        if overlap:
            ranked.append((-overlap, len(assets ^ candidate_assets), item))
    if ranked:
        ranked.sort(key=lambda entry: entry[:2])
        if len(ranked) == 1 or ranked[0][:2] < ranked[1][:2]:
            return ranked[0][2].get("corona_legacy")

    if base is None or roughness is None or ior is None:
        return None

    def score(item):
        values = item["corona_legacy"]
        roughness_error = (roughness - (1.0 - values["reflection_glossiness"])) ** 2
        if ior > 3.0 and values["fresnel_ior"] > 3.0:
            return roughness_error + (ior - values["fresnel_ior"]) ** 2
        expected = values["reflection_color"] if values["fresnel_ior"] > 3.0 else values["diffuse_color"]
        color_error = sum((base[index] - expected[index]) ** 2 for index in range(3))
        ior_error = 0.0 if values["fresnel_ior"] > 3.0 else (ior - values["fresnel_ior"]) ** 2
        return color_error + roughness_error + ior_error

    candidates = [item for item in candidates if item.get("corona_legacy")]
    match = min(candidates, key=score) if candidates else None
    return match["corona_legacy"] if match is not None and score(match) < 1.0e-5 else None


def _corona_materials(path):
    if not path.is_file():
        return {}
    result = {}
    for root in json.loads(path.read_text(encoding="utf-8")).get("roots", []):
        if root.get("status") == "multi_sub":
            candidates = [item for item in root.get("corona_submaterials", [])
                          if item.get("corona_legacy")]
            for name in root.get("gltf_materials", []):
                material = bpy.data.materials.get(name)
                if material is None:
                    continue
                assets = set()
                graph = None
                try:
                    graph = _source_repair(name) or inspect_material(material)
                    assets = {
                        layer["basename"]
                        for stack in (graph.get("color"), graph.get("data")) if stack
                        for layer in stack["layers"]
                    }
                except Unsupported:
                    pass
                base = roughness = ior = None
                try:
                    principled = _surface(material)
                    if graph and graph.get("color") and not graph["color"]["layers"]:
                        base = graph["color"]["base"]
                    else:
                        base = _value(principled.inputs.get("Base Color"))[:3]
                except (Unsupported, TypeError):
                    principled = None
                try:
                    roughness = float(_value(principled.inputs.get("Roughness")))
                except (Unsupported, TypeError, AttributeError):
                    pass
                try:
                    ior = float(_value(principled.inputs.get("IOR")))
                except (Unsupported, TypeError, AttributeError):
                    pass
                match = _match_corona_submaterial(candidates, assets, base, roughness, ior)
                if match:
                    result.setdefault(name, match)
            continue
        values = root.get("corona_legacy")
        if not values:
            continue
        for name in root.get("gltf_materials", []):
            if name not in result:
                result[name] = values
    return result


def _apply_corona_legacy(inputs, values, graph=None):
    """Preserve CoronaLegacy's Fresnel, reflection multiplier and glossiness."""
    if not values or values["refraction_level"] != 0.0:
        return False
    inputs["base_weight"] = ("float", values["diffuse_level"])
    inputs["base_metalness"] = ("float", 0.0)
    inputs["specular_weight"] = ("float", 1.0)
    inputs["specular_ior"] = ("float", values["fresnel_ior"])
    reflection = tuple(values["reflection_level"] * channel for channel in values["reflection_color"])
    if values.get("diffuse_map_color") is not None and graph is not None and graph.get("color"):
        amount = values["diffuse_map_amount"]
        graph["color"] = {
            "base": tuple((1.0 - amount) * base + amount * mapped
                          for base, mapped in zip(values["diffuse_color"], values["diffuse_map_color"])),
            "layers": [], "gain": 1.0,
        }
    if values.get("reflection_map_color") is not None:
        amount = values["reflection_map_amount"]
        reflection = tuple(values["reflection_level"] * ((1.0 - amount) * base + amount * mapped)
                           for base, mapped in zip(values["reflection_color"],
                                                   values["reflection_map_color"]))
        inputs["specular_color"] = ("color3", reflection)
    elif values.get("reflection_map") and values.get("reflection_map_source") in {"color", "data"} and graph is not None:
        graph["specular_color"] = {
            "base": reflection,
            "mix": values["reflection_map_amount"],
            "gain": values["reflection_map_gain"] * values["reflection_level"],
            "source": values["reflection_map_source"],
        }
    else:
        inputs["specular_color"] = ("color3", reflection)

    # max2blender's Principled reconstruction is only an approximation here:
    # for example the leather's source (0.8, 0.3, 5.0) arrived as
    # (0.8, 0.5, 5.5), producing near-zero roughness and fireflies.  The Max
    # values describe this straight-line branch exactly, so compile those.
    roughness = 1.0 - values["reflection_glossiness"]
    if values.get("gloss_map") and values.get("gloss_map_source") == "data" and graph is not None:
        graph["roughness"] = {
            "base": roughness,
            "mix": values["gloss_map_amount"],
            "gain": values["gloss_map_gain"],
        }
    else:
        if graph is not None:
            graph["roughness"] = None
        inputs["specular_roughness"] = ("float", roughness)
    if graph is not None and graph.get("bump") and values.get("bump_map"):
        # max2blender preserved the sign as Bump.invert, but discarded the
        # Corona map amount's magnitude (all imported distances are 0.001).
        graph["bump"]["scale"] *= abs(values["bump_map_amount"])
    return True


def _swap_native_uv(doc, material_names):
    names = [material.get("name", "") for material in doc.get("materials", [])]
    swapped = 0
    already = 0
    for mesh in doc.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            index = primitive.get("material")
            if index is None or index >= len(names) or names[index] not in material_names:
                continue
            extras = primitive.setdefault("extras", {})
            attrs = primitive.get("attributes", {})
            # Blender exports the authored UV set second. Older hand-authored
            # scene revisions swapped it before the marker existed, so the
            # accessor order is the only idempotent migration signal available.
            needs_swap = "TEXCOORD_1" in attrs and attrs["TEXCOORD_0"] < attrs["TEXCOORD_1"]
            if extras.get("strelkaNativeUv") and not needs_swap:
                already += 1
                continue
            if needs_swap:
                attrs["TEXCOORD_0"], attrs["TEXCOORD_1"] = attrs["TEXCOORD_1"], attrs.get("TEXCOORD_0")
                attrs.pop("TANGENT", None)
                swapped += 1
            extras["strelkaNativeUv"] = True
    return swapped, already


def self_test():
    source = {"basename": "a.png", "colorspace": "srgb_texture", "scale": (1.0, 2.0), "offset": (0.0, 0.0), "rotation": 0.0,
              "adjust": (0.0, 1.0, 1.0),
              "tone": (1.0, 0.0, 0.0), "gain": 1.0}
    assert _same_texture(source, dict(source, gain=7.0))
    assert not _same_texture(source, dict(source, scale=(2.0, 2.0)))
    assert _replace_block("<materialx>\n</materialx>", ["  x"]).count(BEGIN) == 1
    generic = ('<materialx>\n  <surfacematerial name="mat" type="material">\n'
               '    <input name="surfaceshader" type="surfaceshader" nodename="S_mat" />\n'
               '  </surfacematerial>\n  <open_pbr_surface name="S_mat" type="surfaceshader">\n'
               '  </open_pbr_surface>\n</materialx>')
    assert 'name="mat"' not in _remove_materials(generic, {"mat"})
    assert _source_repair("other") is None
    assert _source_repair("mint0046")["color"]["layers"][0]["basename"] == "mint0008.jpg"
    pillow = _source_repair("mint0125")
    assert [layer["blend"] for layer in pillow["color"]["layers"]] == [
        "multiply", "overlay", "screen", "multiply"]
    assert pillow["color"]["layers"][2]["factor_facing_data"] == 0.161
    assert pillow["bump"]["procedural"] == "voronoi_ridge"
    for stack in (pillow["color"], pillow["data"]):
        for source in stack["layers"]:
            source["file"] = source["basename"]
    pillow_xml = "\n".join(_material_xml("mint0125", pillow, {}))
    assert 'name="data_adjust1"' in pillow_xml
    assert _source_repair("mint0129")["color"]["layers"][1]["factor_facing"] == 1.0
    assert _source_repair("mint0074")["bump"]["procedural"] == "noise"
    assert _source_repair("mint0128")["bump"]["procedural"] == "noise"
    assert _match_corona_submaterial([
        {"assets": ["wall.png"], "corona_legacy": {"id": "wall"}},
        {"assets": ["trim.jpg"], "corona_legacy": {"id": "trim"}},
    ], {"WALL.PNG"})["id"] == "wall"
    assert _match_corona_submaterial([{
        "assets": [], "corona_legacy": {
            "diffuse_color": [0.1, 0.2, 0.3], "reflection_color": [0.6, 0.5, 0.4],
            "reflection_glossiness": 0.8, "fresnel_ior": 20.0,
        },
    }], set(), (0.25, 0.15, 0.05), 0.2, 20.0)["fresnel_ior"] == 20.0
    shared = {
        "color": {"layers": [source], "base": None},
        "data": {"layers": [dict(source, colorspace="lin_rec709")], "base": None},
    }
    assert _restore_shared_map_colorspace(shared, {"shared_diffuse_data_bitmaps": True}) == 1
    assert shared["data"]["layers"][0]["colorspace"] == "srgb_texture"
    constant = {"color": {"base": (0.1, 0.2, 0.3), "layers": [], "gain": 1.0},
                "data": None, "roughness": None, "bump": None}
    xml = "\n".join(_material_xml("constant", constant, {}))
    assert 'name="base_color" type="color3" value="0.1, 0.2, 0.3"' in xml
    inputs = {}
    graph = {}
    assert _apply_corona_legacy(inputs, {
        "diffuse_level": 0.8, "refraction_level": 0.0, "fresnel_ior": 1.52,
        "reflection_glossiness": 0.8,
        "reflection_level": 0.5, "reflection_color": [0.25, 0.25, 0.25],
    }, graph)
    assert inputs["specular_ior"] == ("float", 1.52)
    assert inputs["specular_color"] == ("color3", (0.125, 0.125, 0.125))
    assert inputs["specular_weight"] == ("float", 1.0)
    assert inputs["specular_roughness"][0] == "float"
    assert abs(inputs["specular_roughness"][1] - 0.2) < 1.0e-7
    mapped = {}
    mapped_graph = {"bump": {"scale": 0.001, "gain": 5.5}}
    assert _apply_corona_legacy(mapped, {
        "diffuse_level": 1.0, "refraction_level": 0.0, "fresnel_ior": 1.52,
        "reflection_glossiness": 0.8,
        "reflection_level": 1.0, "reflection_color": [0.25, 0.25, 0.25],
        "reflection_map": True, "reflection_map_amount": 0.3,
        "reflection_map_gain": 5.0, "reflection_map_source": "data",
        "gloss_map": True, "gloss_map_amount": 0.5,
        "gloss_map_gain": 5.5, "gloss_map_source": "data",
        "bump_map": True, "bump_map_amount": 0.3,
    }, mapped_graph)
    assert mapped["specular_ior"] == ("float", 1.52)
    assert mapped_graph["specular_color"]["gain"] == 5.0
    assert abs(mapped_graph["roughness"]["base"] - 0.2) < 1.0e-7
    assert mapped_graph["roughness"]["mix"] == 0.5
    assert mapped_graph["roughness"]["gain"] == 5.5
    assert abs(mapped_graph["bump"]["scale"] - 0.0003) < 1.0e-12
    metal = {"base_metalness": ("float", 1.0), "specular_ior": ("float", 2.0)}
    assert _apply_corona_legacy(metal, {
        "diffuse_level": 1.0, "refraction_level": 0.0, "fresnel_ior": 2.0,
        "reflection_glossiness": 0.5,
        "reflection_level": 1.0, "reflection_color": [0.5, 0.5, 0.5],
    })
    assert metal["base_metalness"] == ("float", 0.0)
    assert metal["specular_ior"] == ("float", 2.0)
    assert metal["specular_color"] == ("color3", (0.5, 0.5, 0.5))
    high_ior = {}
    high_graph = {"color": {"base": (0.2, 0.2, 0.2), "layers": [], "gain": 1.0},
                  "bump": None}
    assert _apply_corona_legacy(high_ior, {
        "diffuse_level": 1.0, "refraction_level": 0.0, "fresnel_ior": 20.0,
        "reflection_glossiness": 0.8, "reflection_level": 1.0,
        "reflection_color": [0.6, 0.5, 0.4], "diffuse_color": [0.1, 0.2, 0.3],
        "diffuse_map_color": [0.0, 0.0, 0.0], "diffuse_map_amount": 1.0,
        "reflection_map_color": [0.7, 0.6, 0.5], "reflection_map_amount": 1.0,
    }, high_graph)
    assert high_graph["color"]["base"] == (0.0, 0.0, 0.0)
    assert high_ior["specular_color"] == ("color3", (0.7, 0.6, 0.5))
    print("export_native_materials self-test: OK")


def main():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--gltf", type=Path)
    parser.add_argument("--mtlx", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-audit", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(raw)
    if args.self_test:
        self_test()
        return
    if args.gltf is None:
        parser.error("--gltf is required")
    gltf_path = args.gltf.resolve()
    mtlx_path = (args.mtlx or gltf_path.with_suffix(".mtlx")).resolve()
    report_path = (args.report or gltf_path.with_name(gltf_path.stem + "_native_materials.json")).resolve()
    max_audit_path = (args.max_audit or gltf_path.with_name(gltf_path.stem + "_max_material_audit.json")).resolve()

    gltf = json.loads(gltf_path.read_text(encoding="utf-8"))
    gltf_materials = {material.get("name"): material for material in gltf.get("materials", [])
                      if material.get("name")}
    sources = _find_sources(mtlx_path.parent)
    corona_materials = _corona_materials(max_audit_path)
    audit = []
    accepted = {}
    for material in sorted(bpy.data.materials, key=lambda item: item.name):
        try:
            if material.name not in gltf_materials:
                raise Unsupported("not present in exported glTF")
            graph = _source_repair(material.name) or inspect_material(material)
            _restore_shared_map_colorspace(graph, corona_materials.get(material.name))
            for stack in (graph["color"], graph["data"]):
                for source in stack["layers"] if stack else ():
                    _resolve(source, mtlx_path.parent, sources)
            accepted[material.name] = graph
            audit.append({"material": material.name, "status": "direct", "graph": graph})
        except Unsupported as error:
            audit.append({"material": material.name, "status": "fallback", "reason": str(error)})

    old_text = mtlx_path.read_text(encoding="utf-8")
    without_old_block = old_text
    if BEGIN in without_old_block:
        start = without_old_block.index(BEGIN)
        end = without_old_block.index(END, start) + len(END)
        without_old_block = without_old_block[:start] + without_old_block[end:]
    existing = set(re.findall(r'<surfacematerial\s+name="([^"]+)"', without_old_block))
    preserved = existing & AUTHORED_MATERIALX
    notes = []
    generated = []
    for name in sorted(accepted):
        if name in preserved:
            notes.append(f"{name}: kept existing authored MaterialX")
            continue
        material = gltf_materials[name]
        inputs, _ = convert(material, gltf.get("images", []), gltf.get("textures", []), notes)
        if _apply_corona_legacy(inputs, corona_materials.get(name), accepted[name]):
            notes.append(f"{name}: CoronaLegacy reflection multiplier and Fresnel IOR preserved independently")
        generated.extend(_material_xml(name, accepted[name], inputs))

    generated_names = set(accepted) - preserved
    new_text = _replace_block(_remove_materials(old_text, generated_names), generated)
    mtlx_path.write_text(new_text + ("" if old_text.endswith("\n") else "\n"),
                         encoding="utf-8")
    swapped, already = _swap_native_uv(gltf, generated_names | AUTHORED_NATIVE_UV)
    gltf_path.write_text(json.dumps(gltf, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    report = {
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "gltf": str(gltf_path),
        "direct_count": len(generated_names),
        "fallback_count": len(audit) - len(accepted),
        "native_uv_primitives": swapped,
        "already_native_uv_primitives": already,
        "corona_legacy_dielectrics": sum(
            1 for name in generated_names if name in corona_materials
            and corona_materials[name]["refraction_level"] == 0.0
        ),
        "notes": notes,
        "materials": audit,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"native MaterialX: {len(generated_names)} direct, {report['fallback_count']} fallback")
    print(f"native UV: {swapped} primitive(s) swapped, {already} already marked")
    print(f"audit -> {report_path}")


if __name__ == "__main__":
    main()
