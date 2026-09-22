#!/usr/bin/env python3
"""Apply the recoverable Corona VFB operators to a linear EXR.

Run with Blender so EXR IO and the final sRGB display transform are identical
to the reference-render tooling::

    blender -b --python apply_corona_vfb.py -- input.exr output.png
"""

from __future__ import annotations

import argparse
import math
import sys


# Fit once to Corona's public Contrast=10 response curve.  Neutral Contrast=1
# remains identity; unlike fitting the interior, this has no scene pixels in it.
_CONTRAST_EXPONENT = math.log(2.7) / math.log(10.0)


def highlight_compression(x, amount):
    return x * (1.0 + x / (amount * amount)) / (1.0 + x)


def contrast_power(amount):
    return amount ** _CONTRAST_EXPONENT


def self_test():
    assert highlight_compression(7.0, 1.0) == 7.0
    assert abs(highlight_compression(1.0, 8.0) - 0.5078125) < 1.0e-12
    assert contrast_power(1.0) == 1.0
    assert abs(contrast_power(10.0) - 2.7) < 1.0e-12
    print("apply_corona_vfb self-test: OK")


def main():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--exposure", type=float, default=5.6)
    parser.add_argument("--highlight-compression", type=float, default=8.0)
    parser.add_argument("--contrast", type=float, default=3.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(raw)
    if args.self_test:
        self_test()
        return
    if not args.input or not args.output:
        parser.error("input and output are required")

    import bpy
    import numpy as np

    source = bpy.data.images.load(args.input, check_existing=False)
    pixels = np.empty(len(source.pixels), dtype=np.float32)
    source.pixels.foreach_get(pixels)
    rgba = pixels.reshape((-1, 4))
    rgb = np.maximum(rgba[:, :3] * args.exposure, 0.0)
    hc = args.highlight_compression
    rgb = rgb * (1.0 + rgb / (hc * hc)) / (1.0 + rgb)
    p = contrast_power(args.contrast)
    rgb = np.clip(rgb, 0.0, 1.0)
    a = np.power(rgb, p)
    b = np.power(1.0 - rgb, p)
    rgba[:, :3] = a / np.maximum(a + b, 1.0e-20)

    result = bpy.data.images.new("Corona VFB", source.size[0], source.size[1], alpha=True, float_buffer=True)
    result.colorspace_settings.name = "Linear Rec.709"
    result.pixels.foreach_set(pixels)
    scene = bpy.context.scene
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    result.save_render(args.output, scene=scene)


if __name__ == "__main__":
    main()
