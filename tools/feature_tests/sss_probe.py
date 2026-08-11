"""Per-sphere radiance readout for 25_subsurface.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/feature_tests/sss_probe.py

Prints the mean linear RGB inside a disc at the centre of each sphere for the
Cycles reference and the Strelka render, so the two can be compared per material
instead of through a single whole-frame number.
"""
import os
import sys

import bpy
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SCENE = os.path.join(ROOT, "scenes", "feature_tests", "25_subsurface")


def load(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return px[::-1, :, :3]


def main():
    which = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "strelka"
    ref = load(os.path.join(SCENE, "25_subsurface_cycles.exr"))
    ours = load(os.path.join(SCENE, f"25_subsurface_{which}.exr"))
    print(f"-- {which}")
    h, w, _ = ref.shape

    # Five spheres evenly spaced across the frame; sample a disc well inside each
    # so the silhouette and the floor contact stay out of the average.
    yy, xx = np.mgrid[0:h, 0:w]
    print(f"{'sphere':>8} {'cycles RGB':>28} {'strelka RGB':>28} {'ratio':>22}")
    for i in range(5):
        cx = w * (i + 0.5) / 5.0
        cy = h * 0.52
        r = w / 5.0 * 0.28
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        a = ref[m].mean(0)
        b = ours[m].mean(0)
        rat = b / np.maximum(a, 1e-6)
        # The whole silhouette, and its lit and shadowed halves: a walk that
        # carries light the wrong distance redistributes between the halves while
        # the total holds, which is a different fault from losing it.
        big = ((xx - cx) ** 2 + (yy - cy) ** 2) < (w * 0.093) ** 2
        top = big & (yy < cy)
        bot = big & (yy >= cy)
        whole = ours[big].mean() / max(ref[big].mean(), 1e-6)
        tr = ours[top].mean() / max(ref[top].mean(), 1e-6)
        br = ours[bot].mean() / max(ref[bot].mean(), 1e-6)
        print(f"{i:>8} {a[0]:8.4f}{a[1]:9.4f}{a[2]:9.4f}   "
              f"{b[0]:8.4f}{b[1]:9.4f}{b[2]:9.4f}   "
              f"{rat[0]:6.3f}{rat[1]:7.3f}{rat[2]:7.3f}   "
              f"disc {whole:5.3f} top {tr:5.3f} bot {br:5.3f}")

    # Backdrop, away from the spheres: isolates whether the lighting itself agrees.
    m = (yy < h * 0.12)
    a, b = ref[m].mean(0), ours[m].mean(0)
    print(f"{'backdrop':>8} {a[0]:8.4f}{a[1]:9.4f}{a[2]:9.4f}   "
          f"{b[0]:8.4f}{b[1]:9.4f}{b[2]:9.4f}   "
          f"{b[0]/a[0]:6.3f}{b[1]/a[1]:7.3f}{b[2]/a[2]:7.3f}")


main()
