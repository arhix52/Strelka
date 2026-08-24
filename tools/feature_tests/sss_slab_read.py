"""Read the slab rows: transmittance against Cycles, and against Beer-Lambert.

    /Applications/Blender.app/Contents/Resources/<v>/python/bin/python3.x \
        tools/feature_tests/sss_slab_read.py

The second column is the one that matters. Every factor this measurement does not
control -- the entry and exit lobes, the light, the solid angle, the camera -- is
identical between thicknesses, so it cancels in the ratio of two of them, and

    T(d1) / T(d2) = exp(-sigma_t * (d1 - d2))

is exact. That makes the row gradeable with no reference at all, which is what
separates a defect from the chord distribution of a curved body: on a sphere
every scale factor that fixed the shadowed half wrecked the lit one, and there
was no way to tell which of the two was wrong.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from exr_io import load_exr  # noqa: E402  (path set above)
from sss_slab import MFP, THICKNESSES, slab_name  # noqa: E402

ROOT = os.path.join(HERE, "..", "..", "scenes", "feature_tests")


def centre_mean(path):
    """Mean over a disc well inside the plate, so no edge enters the average."""
    img = load_exr(path)[..., :3]
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (w * 0.18) ** 2
    return float(img[mask].mean())


def main():
    ref, ours = [], []
    print("%6s%6s%11s%11s%8s" % ("d", "tau", "cycles", "strelka", "ratio"))
    for d in THICKNESSES:
        name = slab_name(d)
        scene = os.path.join(ROOT, name)
        c = centre_mean(os.path.join(scene, name + "_cycles.exr"))
        s = centre_mean(os.path.join(scene, name + "_strelka.exr"))
        ref.append(c)
        ours.append(s)
        print("%6.2f%6.2f%11.5f%11.5f%8.3f" % (d, d / MFP, c, s, s / c))

    print("\nattenuation between neighbouring thicknesses -- no reference needed")
    print("%12s%10s%10s%10s%12s%12s"
          % ("pair", "analytic", "cycles", "strelka", "sigma cycles", "sigma ours"))
    for i in range(len(THICKNESSES) - 1):
        dd = THICKNESSES[i + 1] - THICKNESSES[i]
        analytic = math.exp(-dd / MFP)
        rc = ref[i + 1] / ref[i]
        rs = ours[i + 1] / ours[i]
        print("%6.2f->%.2f%10.4f%10.4f%10.4f%12.2f%12.2f"
              % (THICKNESSES[i], THICKNESSES[i + 1], analytic, rc, rs,
                 -math.log(rc) / dd, -math.log(rs) / dd))
    print("\nnominal sigma_t = %.2f" % (1.0 / MFP))


main()
