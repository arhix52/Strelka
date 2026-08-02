#!/usr/bin/env python3
"""
Turn an EXR into a PNG and print what is in it.

    blender -b -P tools/feature_tests/exr_view.py -- IN.exr OUT.png [--gain 1.0]

Exists because "is the image wrong" and "is the image empty" are different
questions and the second one is answered by numbers, not by looking. The
percentiles say whether a render is uniformly too bright or has a few fireflies;
the coverage says whether rays are hitting anything at all.
"""

import bpy
import numpy as np
import struct
import sys
import zlib


def write_png(path, rows, w, h):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(h))
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n"
                           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6))
                           + chunk(b"IEND", b""))


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    src, dst = a[0], a[1]
    gain = float(a[a.index("--gain") + 1]) if "--gain" in a else 1.0

    im = bpy.data.images.load(src)
    w, h = im.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    im.pixels.foreach_get(buf)
    bpy.data.images.remove(im)
    rgb = buf.reshape(h, w, 4)[::-1, :, :3]   # Blender stores bottom-up

    lum = rgb.mean(axis=2)
    print("%s  %dx%d" % (src, w, h))
    print("  non-black %.2f%%   mean %.4f" % (100 * (lum > 1e-6).mean(), lum.mean()))
    print("  p1 %.4f  p50 %.4f  p90 %.4f  p99 %.4f  max %.3f"
          % tuple(np.percentile(lum, [1, 50, 90, 99]).tolist() + [lum.max()]))

    x = np.clip(rgb * gain, 0, 1)
    s = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    write_png(dst, (s * 255 + 0.5).astype(np.uint8), w, h)
    print("  wrote %s (gain %.3g)" % (dst, gain))


main()
