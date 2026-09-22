#!/usr/bin/env python3
"""Build a deterministic geometry-only chart for pixel reconstruction filters."""

import base64
import json
import math
import struct
from pathlib import Path


PIXEL = 5.0 / 512.0


class Geometry:
    def __init__(self):
        self.positions = []
        self.indices = []

    def triangle(self, a, b, c):
        base = len(self.positions)
        self.positions.extend(((*a, 0.0), (*b, 0.0), (*c, 0.0)))
        self.indices.extend((base, base + 1, base + 2))

    def quad(self, a, b, c, d):
        base = len(self.positions)
        self.positions.extend(((*a, 0.0), (*b, 0.0), (*c, 0.0), (*d, 0.0)))
        self.indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    def rectangle(self, center, size, angle=0.0):
        cx, cy = center
        hx, hy = size[0] * 0.5, size[1] * 0.5
        cs, sn = math.cos(angle), math.sin(angle)

        def point(x, y):
            return cx + cs * x - sn * y, cy + sn * x + cs * y

        self.quad(point(-hx, -hy), point(hx, -hy), point(hx, hy), point(-hx, hy))

    def annulus(self, center, inner, outer, segments=256):
        cx, cy = center
        for i in range(segments):
            a0 = 2.0 * math.pi * i / segments
            a1 = 2.0 * math.pi * (i + 1) / segments
            self.quad((cx + inner * math.cos(a0), cy + inner * math.sin(a0)),
                      (cx + outer * math.cos(a0), cy + outer * math.sin(a0)),
                      (cx + outer * math.cos(a1), cy + outer * math.sin(a1)),
                      (cx + inner * math.cos(a1), cy + inner * math.sin(a1)))


def add_siemens_star(g):
    center = (-2.45, 1.15)
    radius = 0.95
    wedges = 192
    for i in range(0, wedges, 2):
        a0 = 2.0 * math.pi * i / wedges
        a1 = 2.0 * math.pi * (i + 1) / wedges
        g.triangle(center, (center[0] + radius * math.cos(a0), center[1] + radius * math.sin(a0)),
                   (center[0] + radius * math.cos(a1), center[1] + radius * math.sin(a1)))


def add_zone_plate(g):
    center = (0.0, 1.15)
    radius = 0.95
    bands = 40
    for i in range(0, bands, 2):
        inner = radius * math.sqrt(i / bands)
        outer = radius * math.sqrt((i + 1) / bands)
        g.annulus(center, inner, outer)


def add_line_widths(g):
    for row, width in enumerate((0.5, 0.75, 1.0, 1.5, 2.0)):
        g.rectangle((-2.45, -0.45 - 0.34 * row), (1.85, width * PIXEL), math.radians(11.0))


def add_checker_patches(g):
    for patch, pixels in enumerate((4.0, 2.0, 1.0)):
        x0 = -1.05 + patch * 0.72
        x1 = x0 + 0.62
        y0, y1 = -2.15, -0.30
        cell = pixels * PIXEL
        rows = math.ceil((y1 - y0) / cell)
        cols = math.ceil((x1 - x0) / cell)
        for row in range(rows):
            for col in range(cols):
                if (row + col) & 1:
                    continue
                xa, xb = x0 + col * cell, min(x0 + (col + 1) * cell, x1)
                ya, yb = y0 + row * cell, min(y0 + (row + 1) * cell, y1)
                g.quad((xa, ya), (xb, ya), (xb, yb), (xa, yb))


def add_frequency_sweep(g):
    x, x1 = 1.25, 3.55
    while x < x1:
        alpha = (x - 1.25) / (x1 - 1.25)
        period = PIXEL * (8.0 * (0.75 / 8.0) ** alpha)
        width = 0.5 * period
        g.rectangle((x + 0.5 * width, 1.15), (width, 1.9))
        x += period


def add_diagonal_grating(g):
    for row in range(58):
        g.rectangle((2.40, -2.15 + row * 0.033), (2.20, 0.75 * PIXEL), math.radians(4.0))


def add_panel_frames(g):
    for x in (-3.62, -1.20, 1.20, 3.62):
        g.rectangle((x, 0.0), (PIXEL, 4.65))
    for y in (-2.32, 2.32):
        g.rectangle((0.0, y), (7.25, PIXEL))


def build_document(g):
    positions = [component for position in g.positions for component in position]
    normals = [component for _ in g.positions for component in (0.0, 0.0, 1.0)]
    position_bytes = struct.pack(f"<{len(positions)}f", *positions)
    normal_bytes = struct.pack(f"<{len(normals)}f", *normals)
    index_bytes = struct.pack(f"<{len(g.indices)}I", *g.indices)
    binary = position_bytes + normal_bytes + index_bytes
    encoded = base64.b64encode(binary).decode("ascii")
    xs = [p[0] for p in g.positions]
    ys = [p[1] for p in g.positions]

    return {
        "asset": {"version": "2.0", "generator": "Strelka reconstruction-filter chart"},
        "scene": 0,
        "extensionsUsed": ["KHR_lights_punctual"],
        "extensions": {"KHR_lights_punctual": {"lights": [
            {"name": "No illumination", "type": "directional", "intensity": 0.0}
        ]}},
        "scenes": [{"nodes": [0, 1, 2, 3]}],
        "nodes": [
            {"name": "AA chart", "mesh": 0},
            {"name": "Camera", "camera": 0, "translation": [0, 0, 5]},
            {"name": "No illumination", "extensions": {"KHR_lights_punctual": {"light": 0}}},
            {"name": "Camera shifted half a pixel", "camera": 1, "translation": [0.5 * PIXEL, 0, 5]},
        ],
        "cameras": [
            {"name": "Camera", "type": "orthographic",
             "orthographic": {"xmag": 3.75, "ymag": 2.5, "znear": 0.1, "zfar": 10.0}},
            {"name": "Camera shifted half a pixel", "type": "orthographic",
             "orthographic": {"xmag": 3.75, "ymag": 2.5, "znear": 0.1, "zfar": 10.0}},
        ],
        "meshes": [{"name": "White emissive chart", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0
        }]}],
        "materials": [{"name": "White emitter", "doubleSided": True,
                       "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1], "metallicFactor": 0,
                                                "roughnessFactor": 1},
                       "emissiveFactor": [1.0, 1.0, 1.0]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(g.positions), "type": "VEC3",
             "min": [min(xs), min(ys), 0], "max": [max(xs), max(ys), 0]},
            {"bufferView": 1, "componentType": 5126, "count": len(g.positions), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(g.indices), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(normal_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes), "byteLength": len(index_bytes),
             "target": 34963},
        ],
        "buffers": [{"byteLength": len(binary), "uri": f"data:application/octet-stream;base64,{encoded}"}],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "scenes/validation/reconstruction_filter/reconstruction_filter.gltf"
    geometry = Geometry()
    add_siemens_star(geometry)
    add_zone_plate(geometry)
    add_line_widths(geometry)
    add_checker_patches(geometry)
    add_frequency_sweep(geometry)
    add_diagonal_grating(geometry)
    add_panel_frames(geometry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_document(geometry), separators=(",", ":")), encoding="utf-8")
    print(f"{output}: {len(geometry.positions)} vertices, {len(geometry.indices) // 3} triangles")


if __name__ == "__main__":
    main()
