#!/usr/bin/env python3
"""
Compare Cycles reference EXRs against Strelka EXRs, scene by scene.

Run headless (uses Blender's bundled numpy + EXR reader, so no extra deps):
    blender -b -P tools/feature_tests/compare.py -- --out scenes/feature_tests

For each scene that has both images it prints a row and writes
<scene>_compare.png: reference | strelka | amplified difference.

Everything is measured on LINEAR radiance. Both renderers were told not to
apply a tone curve, so a difference here is a difference in the light
transport, not in the display transform.
"""

import bpy
import numpy as np
import os
import struct
import sys
import zlib

# Relative error below which a scene counts as matching. 2% absorbs Monte Carlo
# noise at the sample counts build_features.py uses; it is not a claim about
# either renderer being correct to 2%.
TOL_OK = 0.02
TOL_CLOSE = 0.10

# Pixels darker than this in the reference are excluded from the ratio, because
# dividing by near-zero radiance produces meaningless numbers.
RATIO_FLOOR = 1e-3


def write_png(path, width, height, rgb_rows):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(r) for r in rgb_rows)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(blob)


def load_exr(path):
    """Returns float32 array (h, w, 3), top-down."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    # Blender hands back bottom-up rows; flip so index 0 is the top scanline.
    return buf.reshape(h, w, 4)[::-1, :, :3].copy()


def encode(arr, gain=1.0):
    """Linear -> 8-bit sRGB-ish, for the contact sheet only."""
    x = np.clip(arr * gain, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def contact_sheet(path, ref, tst, diff_gain=8.0):
    h, w, _ = ref.shape
    diff = np.abs(ref - tst)
    panel = np.concatenate([encode(ref), encode(tst), encode(diff, diff_gain)], axis=1)
    rows = [panel[y].tobytes() for y in range(h)]
    write_png(path, w * 3, h, rows)


def stats(ref, tst):
    """Relative error, RMSE and mean ratio.

    The ratio is the interesting one for calibration: a scene that is
    geometrically identical but uniformly N times brighter reports rel ~= N-1
    and ratio ~= N, which points straight at light units or exposure rather
    than at shading.
    """
    mask = ref.mean(axis=2) > RATIO_FLOOR
    ref_mean = ref[mask].mean() if mask.any() else 0.0
    tst_mean = tst[mask].mean() if mask.any() else 0.0
    abs_err = np.abs(ref - tst).mean()
    rmse = float(np.sqrt(((ref - tst) ** 2).mean()))
    rel = float(abs_err / ref_mean) if ref_mean > 0 else float("inf")
    ratio = float(tst_mean / ref_mean) if ref_mean > 0 else float("nan")
    return rel, rmse, ratio


def verdict(rel):
    if rel < TOL_OK:
        return "OK"
    if rel < TOL_CLOSE:
        return "CLOSE"
    return "FAIL"


def compare_pair(ref_p, tst_p, out_p):
    """Two loose images, for production scenes that have no harness directory.

    Also reports the median of each, because a reference rendered under a wall
    clock is noisy: a handful of fireflies move the mean by a lot and the median
    by nothing, so the two together separate "different brightness" from "same
    brightness, different noise".
    """
    ref, tst = load_exr(ref_p), load_exr(tst_p)
    if ref.shape != tst.shape:
        print("shape mismatch: %s vs %s" % (ref.shape, tst.shape))
        return
    rel, rmse, ratio = stats(ref, tst)
    print("rel %.4f  rmse %.5f  mean ratio %.3f  %s" % (rel, rmse, ratio, verdict(rel)))
    print("median: reference %.4f  strelka %.4f  ratio %.3f"
          % (np.median(ref), np.median(tst),
             np.median(tst) / max(np.median(ref), 1e-9)))
    contact_sheet(out_p, ref, tst)
    print("wrote %s (reference | strelka | 8x diff)" % out_p)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    if "--pair" in argv:
        i = argv.index("--pair")
        compare_pair(argv[i + 1], argv[i + 2], argv[i + 3])
        return

    out_root = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv \
        else os.path.abspath("scenes/feature_tests")

    if not os.path.isdir(out_root):
        print("No such directory: %s" % out_root)
        return

    print("%-22s %8s %9s %8s  %s" % ("scene", "rel", "rmse", "ratio", "verdict"))
    print("-" * 62)

    missing = []
    for name in sorted(os.listdir(out_root)):
        scene_dir = os.path.join(out_root, name)
        if not os.path.isdir(scene_dir) or name.startswith("_"):
            continue

        ref_path = os.path.join(scene_dir, name + "_cycles.exr")
        tst_path = os.path.join(scene_dir, name + "_strelka.exr")
        if not os.path.exists(ref_path):
            missing.append("%s (no cycles reference)" % name)
            continue
        if not os.path.exists(tst_path):
            missing.append("%s (not rendered by Strelka yet)" % name)
            continue

        ref = load_exr(ref_path)
        tst = load_exr(tst_path)
        if ref.shape != tst.shape:
            print("%-22s  shape mismatch %s vs %s" % (name, ref.shape, tst.shape))
            continue

        rel, rmse, ratio = stats(ref, tst)

        # tinyexr and Blender do not have to agree on scanline order. If the
        # vertically mirrored image matches far better, the images are fine and
        # the writer's row order is what differs -- worth knowing, and it would
        # otherwise masquerade as a total shading failure.
        rel_f, rmse_f, ratio_f = stats(ref, tst[::-1])
        flipped = rel_f < rel * 0.5
        if flipped:
            rel, rmse, ratio = rel_f, rmse_f, ratio_f
            tst = tst[::-1]

        print("%-22s %8.4f %9.5f %8.3f  %s%s"
              % (name, rel, rmse, ratio, verdict(rel),
                 "  [v-flipped]" if flipped else ""))

        contact_sheet(os.path.join(scene_dir, name + "_compare.png"), ref, tst)

    if missing:
        print("\nSkipped:")
        for m in missing:
            print("  %s" % m)

    print("\nEach scene also has <name>_compare.png: reference | strelka | 8x diff")
    print("Read 00_calibration first. If its ratio is not ~1.0, every other")
    print("row is measuring that same offset and nothing else.")


if __name__ == "__main__":
    main()
