#!/usr/bin/env python3
"""How noisy is a render, separated from how wrong it is.

    tools/parity/noise_check.py --test optix --ref conv

`rel` in run_ladder.py is one number over two different things: sampling noise,
which falls as 1/sqrt(spp) and averages out, and bias, which does not. A backend
can lose that column to either, and the two want opposite fixes. This splits them.

Against a converged reference of the *same* renderer, the residual `X - ref` is
noise plus whatever that image gets wrong. Monte Carlo noise is independent
between neighbouring pixels; a shading or geometry disagreement is not -- it sits
on a surface, an edge or a lobe, and survives a blur. So the residual is split by
a 5x5 box:

    noise = rms(R - blur(R))         high frequency, per pixel, uncorrelated
    bias  = rms(blur(R))             what a blur keeps

Both are normalised by the reference mean so the columns compare across scenes.
The blur passes 1/25th of the noise power through to the bias column, which is
subtracted; scenes where the correction matters are the ones already reading
near zero.

The reference does not have to be the backend under test -- Metal's checked-in
`_strelka.exr` graded against an OptiX convergence is exactly the measurement
that says whether the two backends differ in noise or in what they converge to.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feature_tests"))

import numpy as np  # noqa: E402
from exr_io import load_exr  # noqa: E402

DEFAULT_ASSETS = os.path.expanduser("~/strelka_assets/feature_tests")


def box5(img):
    """5x5 box blur with edge clamping, via a separable running sum."""
    out = img
    for axis in (0, 1):
        padded = np.concatenate(
            [np.take(out, [0, 0], axis=axis), out, np.take(out, [-1, -1], axis=axis)], axis=axis
        )
        acc = np.cumsum(padded, axis=axis)
        acc = np.concatenate([np.zeros_like(np.take(acc, [0], axis=axis)), acc], axis=axis)
        n = out.shape[axis]
        out = (np.take(acc, range(5, 5 + n), axis=axis) - np.take(acc, range(0, n), axis=axis)) / 5.0
    return out


def smooth_mask(ref, quantile=0.75):
    """Pixels where the converged image has no edge under them.

    A silhouette or a texel boundary puts energy in the high-frequency band
    whether or not either render is noisy, and half a pixel of disagreement
    between two renderers then reads as noise. Grade only where the reference
    is locally flat: the gradient magnitude's own quantile, so the threshold
    follows the scene instead of being a constant nobody can defend.
    """
    lum = ref.mean(axis=2)
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    grad = np.maximum(gx, gy)
    grad = np.maximum(grad, box5(grad[:, :, None])[:, :, 0])
    return grad <= np.quantile(grad, quantile)


def split(test, ref, mask=None):
    """noise / bias / rel, each relative to the reference mean."""
    ref_mean = float(ref.mean())
    if ref_mean <= 0.0:
        return None
    residual = test - ref
    low = box5(residual)
    high = residual - low
    if mask is not None:
        high = high[mask]
        low = low[mask]

    noise_power = float((high**2).mean())
    # A 5x5 box passes 1/25 of white-noise power; take it back out of the bias
    # column so a purely noisy image does not read as slightly biased.
    bias_power = max(float((low**2).mean()) - noise_power / 24.0, 0.0)
    return (
        np.sqrt(noise_power) / ref_mean,
        np.sqrt(bias_power) / ref_mean,
        float(np.abs(residual).mean()) / ref_mean,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    ap.add_argument("--ref", default="conv", help="suffix of the converged reference")
    ap.add_argument("--test", action="append", default=[],
                    help="suffix to grade, repeatable (default: optix, strelka)")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--mask", action="store_true",
                    help="grade only where the reference is locally flat")
    args = ap.parse_args()
    tests = args.test or ["optix", "strelka"]

    scenes = sorted(
        d for d in os.listdir(args.assets)
        if os.path.isdir(os.path.join(args.assets, d)) and not d.startswith("_")
    )
    if args.only:
        scenes = [s for s in scenes if any(f in s for f in args.only)]

    head = "%-24s" % "scene"
    for t in tests:
        head += " %-22s" % ("%s  noise / bias" % t)
    print(head + " noise ratio")
    print("-" * (25 + 23 * len(tests) + 12))

    totals = {t: [] for t in tests}
    for name in scenes:
        d = os.path.join(args.assets, name)
        ref_path = os.path.join(d, "%s_%s.exr" % (name, args.ref))
        if not os.path.exists(ref_path):
            continue
        ref = load_exr(ref_path)
        mask = smooth_mask(ref) if args.mask else None
        row, vals = "%-24s" % name, []
        for t in tests:
            p = os.path.join(d, "%s_%s.exr" % (name, t))
            s = split(load_exr(p), ref, mask) if os.path.exists(p) else None
            if s and s[0] == s[0]:
                row += " %-22s" % ("%.4f / %.4f" % (s[0], s[1]))
                totals[t].append(s)
                vals.append(s[0])
            else:
                row += " %-22s" % "n/a"
                vals.append(None)
        if len(vals) == 2 and vals[0] and vals[1]:
            row += " %.2fx" % (vals[0] / vals[1])
        print(row)

    print("-" * (25 + 23 * len(tests) + 12))
    means = []
    row = "%-24s" % "mean"
    for t in tests:
        if totals[t]:
            n = float(np.mean([s[0] for s in totals[t]]))
            b = float(np.mean([s[1] for s in totals[t]]))
            means.append(n)
            row += " %-22s" % ("%.4f / %.4f" % (n, b))
        else:
            row += " %-22s" % "n/a"
    if len(means) == 2:
        row += " %.2fx" % (means[0] / means[1])
    print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
