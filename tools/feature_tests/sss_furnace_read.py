"""Read the white-furnace renders sss_furnace.py produced.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/feature_tests/sss_furnace_read.py

Prints the mean over the sphere's disc against the sky it sits in. Both are 1.0
for a walk that conserves energy.
"""
import glob
import os

import bpy
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "scenes", "feature_tests", "_sss_furnace")


def load(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return px[::-1, :, :3]


print(f"{'mfp':>8} {'sphere':>9} {'sky':>9} {'ratio':>8}")
for path in sorted(glob.glob(os.path.join(OUT, "furnace_*.exr"))):
    a = load(path)
    h, w, _ = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # Well inside the silhouette: the sky is the same radiance as a perfect
    # sphere would be, so any of it in the disc hides exactly what is measured.
    disc = ((xx - w * 0.5) ** 2 + (yy - h * 0.52) ** 2) < (w * 0.10) ** 2
    sky = (yy < h * 0.06)
    s, k = a[disc].mean(), a[sky].mean()
    mfp = os.path.basename(path)[len("furnace_"):-4].replace("p", ".")
    print(f"{mfp:>8} {s:9.4f} {k:9.4f} {s / k:8.3f}")
