#!/usr/bin/env python3
"""
Compare Strelka's normal debug output against the Cycles reference.

    blender -b -P tools/feature_tests/normals_compare.py -- REF.exr STRELKA.exr OUT.png

Both images hold (N + 1) / 2, but in different frames: Blender is Z-up and
Strelka sees the Y-up glTF, so the Strelka normals are rotated into Blender's
frame before anything is measured.

Reports coverage agreement separately from orientation agreement, because they
fail for different reasons. Coverage says whether the same pixels have geometry
at all -- a mismatch there is missing objects, a wrong camera or a broken
acceleration structure. Orientation says whether the surfaces face the same way
where both agree there is one -- a mismatch there is a transform, a winding or a
normal-packing problem.
"""

import bpy
import numpy as np
import struct
import sys
import zlib


def load(path):
    im = bpy.data.images.load(path)
    w, h = im.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    im.pixels.foreach_get(buf)
    bpy.data.images.remove(im)
    return buf.reshape(h, w, 4)[::-1, :, :3]


def write_png(path, rgb_rows, w, h):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + rgb_rows[y].tobytes() for y in range(h))
    blob = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))
    open(path, "wb").write(blob)


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    ref_p, tst_p, out_p = a[0], a[1], a[2]

    ref = load(ref_p)
    tst = load(tst_p)
    if ref.shape != tst.shape:
        print("shape mismatch: %s vs %s" % (ref.shape, tst.shape))
        return
    h, w, _ = ref.shape

    # Decode both back to unit vectors.
    n_ref = ref * 2.0 - 1.0
    n_tst = tst * 2.0 - 1.0
    # glTF (x, y, z) is Blender (x, -z, y).
    n_tst = np.stack([n_tst[:, :, 0], -n_tst[:, :, 2], n_tst[:, :, 1]], axis=-1)

    # A pixel that hit nothing is exactly 0 in both, which decodes to a vector of
    # length sqrt(3) pointing at (-1,-1,-1) -- so test the raw image, not the
    # decoded normal.
    hit_ref = ref.sum(axis=2) > 1e-6
    hit_tst = tst.sum(axis=2) > 1e-6

    both = hit_ref & hit_tst
    only_ref = hit_ref & ~hit_tst
    only_tst = hit_tst & ~hit_ref
    total = float(h * w)

    print("coverage: reference %.2f%%, strelka %.2f%%, agree %.2f%%"
          % (100 * hit_ref.mean(), 100 * hit_tst.mean(), 100 * both.mean()))
    print("          only reference %.2f%%, only strelka %.2f%%"
          % (100 * only_ref.sum() / total, 100 * only_tst.sum() / total))

    if both.any():
        r = n_ref[both]
        t = n_tst[both]
        r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        cos = np.clip((r * t).sum(axis=1), -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        print("orientation over agreeing pixels: median %.2f deg, mean %.2f deg, "
              "within 5 deg %.1f%%, within 30 deg %.1f%%"
              % (np.median(ang), ang.mean(), 100 * (ang < 5).mean(), 100 * (ang < 30).mean()))
    else:
        print("orientation: no pixel has geometry in both images")

    def enc(x):
        x = np.clip(x, 0, 1)
        s = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
        return (s * 255 + 0.5).astype(np.uint8)

    # Third panel: green where both hit, red where only the reference did, blue
    # where only Strelka did -- so a coverage failure is legible at a glance.
    cov = np.zeros((h, w, 3), dtype=np.float32)
    cov[..., 1] = both
    cov[..., 0] = only_ref
    cov[..., 2] = only_tst
    panel = np.concatenate([enc(ref), enc(tst), enc(cov)], axis=1)
    write_png(out_p, panel, w * 3, h)
    print("wrote %s (reference | strelka | coverage)" % out_p)


main()
