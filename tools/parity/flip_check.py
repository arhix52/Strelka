#!/usr/bin/env python3
"""Grade a render against a reference with FLIP, not with a mean.

    ~/flip-venv/bin/python tools/parity/flip_check.py ref.exr test.exr
    ~/flip-venv/bin/python tools/parity/flip_check.py ref.exr a.exr b.exr --map /tmp/err

FLIP (NVlabs/flip, Andersson et al.) models what an observer flipping between two
images would notice: it filters both through the spatial contrast sensitivity of
the visual system before differencing, and adds a feature term for edges and
points, which are what the eye locks onto. The result is per pixel in [0,1].

Why it earns its place beside `rel`. `rel` is a mean of absolute differences, so
it is blind in both directions that matter here. A one-pixel displacement of the
whole image is invisible in the mean of a busy frame and glaring to an observer;
noise at a level the eye integrates away costs `rel` exactly as much as the same
energy sitting in a single band. Reporting the 95th percentile alongside the mean
separates "wrong everywhere a little" from "wrong somewhere a lot", which no
single number does.

HDR mode is the one to use on our EXRs: FLIP then evaluates over a range of
exposures and takes the worst per pixel, so a scene is graded where it is bright
and where it is dark rather than only where the tone curve happens to land.

Install:  python3 -m venv ~/flip-venv && ~/flip-venv/bin/pip install flip-evaluator
The system python is externally managed on this machine, hence the venv.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feature_tests"))

import numpy as np  # noqa: E402

from exr_io import load_exr  # noqa: E402

try:
    import flip_evaluator
except ImportError:  # pragma: no cover - the message is the point
    sys.exit("flip_evaluator not importable. Run this under ~/flip-venv/bin/python; "
             "see the module docstring for how that venv is made.")


def load(path):
    """EXR through our own reader, anything else through FLIP's."""
    if path.lower().endswith(".exr"):
        return np.ascontiguousarray(load_exr(path).astype(np.float32))
    img = flip_evaluator.load(path)
    return np.ascontiguousarray(np.asarray(img, dtype=np.float32))


def grade(ref, test, hdr, ppd=None):
    params = {"ppd": ppd} if ppd else {}
    err, mean, _ = flip_evaluator.evaluate(
        ref, test, "HDR" if hdr else "LDR", applyMagma=False, parameters=params)
    err = np.asarray(err, dtype=np.float32)
    if err.ndim == 3:
        err = err[..., 0]
    return err, float(mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("test", nargs="+")
    ap.add_argument("--ldr", action="store_true", help="grade as sRGB LDR instead of linear HDR")
    ap.add_argument("--ppd", type=float, default=0.0, help="pixels per degree (default: FLIP's 67)")
    ap.add_argument("--match-exposure", action="store_true",
                    help="scale each test image to the reference's mean first")
    ap.add_argument("--map", default="", help="write <prefix>_<name>.png error maps")
    args = ap.parse_args()

    ref = load(args.reference)
    print("reference %s  %dx%d" % (os.path.basename(args.reference), ref.shape[1], ref.shape[0]))
    print("%-34s %8s %8s %8s %8s" % ("test", "mean", "p95", "p99", "max"))
    for path in args.test:
        test = load(path)
        if test.shape != ref.shape:
            print("%-34s shape %s != %s" % (os.path.basename(path), test.shape, ref.shape))
            continue
        if args.match_exposure:
            # A whole-image scale is a different complaint from a wrong image, and
            # FLIP does not forgive it: HDR mode sweeps exposures and keeps the
            # worst per pixel, so a uniform factor shows up everywhere at once.
            # Taking it out first is what separates "the environment bake is dim"
            # from "the render disagrees".
            m = float(test.mean())
            if m > 0.0:
                test = np.ascontiguousarray(test * (float(ref.mean()) / m))
        err, mean = grade(ref, test, not args.ldr, args.ppd or None)
        print("%-34s %8.4f %8.4f %8.4f %8.4f" % (
            os.path.basename(path), mean,
            float(np.percentile(err, 95)), float(np.percentile(err, 99)), float(err.max())))
        if args.map:
            # Magma, the way the FLIP tool presents it, so the maps here and the
            # ones the reference implementation writes read the same.
            import flip_evaluator.flip_python_api as api  # noqa: F401
            rgb, _, _ = flip_evaluator.evaluate(
                ref, test, "LDR" if args.ldr else "HDR", applyMagma=True,
                parameters={"ppd": args.ppd} if args.ppd else {})
            out = "%s_%s.png" % (args.map, os.path.splitext(os.path.basename(path))[0])
            _write_png(out, np.asarray(rgb, dtype=np.float32))
            print("%-34s -> %s" % ("", out))
    return 0


def _write_png(path, rgb):
    """8-bit RGB PNG, zlib only -- no image library in the venv to depend on."""
    import struct
    import zlib

    data = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    h, w, _ = data.shape
    raw = b"".join(b"\x00" + data[y].tobytes() for y in range(h))

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


if __name__ == "__main__":
    sys.exit(main())
