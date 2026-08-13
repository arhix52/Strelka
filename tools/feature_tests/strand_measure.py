"""Read the isolated-strand probe: cross-section against Cycles, and by depth.

Run under Blender for the EXR reader in compare.py:

    blender -b -P tools/feature_tests/strand_measure.py

What to look for is documented in strand_probe.py. In short: the integrated
cross-section should be within about a percent of Cycles, and it must stop moving
after depth 2, because one convex fibre in an otherwise empty scene has nothing to
scatter off twice.
"""
import os
import sys

import numpy as np

# Blender leaves __file__ pointing at its own wrapper under --python-expr, so fall
# back to the repository layout rather than failing on the import below.
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
if not os.path.exists(os.path.join(_here, "compare.py")):
    _here = os.path.join(os.getcwd(), "tools", "feature_tests")
sys.path.insert(0, _here)
from compare import load_exr  # noqa: E402

OUT = os.environ.get("PROBE_OUT", "/tmp/strand_probe")
DEPTHS = (1, 2, 3, 4, 8)
W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def profile(path):
    """Mean luminance per row over the middle of the strand's length.

    The central third only: the root cap and the foreshortened tip are not a
    cross-section of anything.
    """
    img = load_exr(path) @ W
    h, w = img.shape
    return img[:, int(w * 0.40):int(w * 0.60)].mean(axis=1)


def main():
    ref = profile(os.path.join(OUT, "strand_cycles.exr"))
    rows = np.nonzero(ref > 1e-5)[0]
    lo, hi = rows.min(), rows.max()

    runs = []
    for d in DEPTHS:
        path = os.path.join(OUT, "strand_d%d.exr" % d)
        if os.path.exists(path):
            runs.append((d, profile(path)))
    if not runs:
        raise SystemExit("No strand_d*.exr in %s -- render the probe configs first." % OUT)

    print("integrated cross-section (rows %d..%d)" % (lo, hi))
    print("  cycles            %.4f" % ref.sum())
    for d, p in runs:
        print("  strelka depth %-2d  %.4f   ratio %.4f" % (d, p.sum(), p.sum() / ref.sum()))
    deep = [p.sum() for _, p in runs if _ >= 2]
    if len(deep) > 1:
        drift = max(deep) / min(deep)
        print("  drift over depth >= 2: %.4f %s"
              % (drift, "OK" if drift < 1.01 else "GROWING -- rays are re-entering the strand"))

    last = runs[-1][1]
    print("\nprofile: row, cycles, strelka (depth %d), ratio" % runs[-1][0])
    for y in range(lo, hi + 1):
        r, t = ref[y], last[y]
        print("  %4d  %.6f  %.6f  %s" % (y, r, t, "%.3f" % (t / r) if r > 1e-6 else "-"))

    idx = np.arange(ref.size)
    for label, p in (("cycles", ref), ("strelka", last)):
        m = p / p.sum()
        c = (idx * m).sum()
        print("%-8s peak %.6f  centroid %.3f  rms spread %.3f px"
              % (label, p.max(), c, np.sqrt(((idx - c) ** 2 * m).sum())))


main()
