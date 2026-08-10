#!/usr/bin/env python3
"""Mean linear luminance over a rectangle of a rendered PNG, for A/B rows.

Every measured claim in docs/open-defects.md is a patch mean against the same
patch of the reference, and up to now each one was taken by hand. This is that
step, so the numbers in the table can be reproduced by reading the command
rather than by trusting the row.

The mean is taken on *linear* radiance: the images are display-encoded, so each
channel is put back through the sRGB inverse first. A mean of sRGB bytes is not
a mean of anything physical -- it weights the dark half of the range far too
heavily -- and two images that differ in exposure compare differently under it.

Pure stdlib, like bubble_profile.py next to it: this tree has no numpy outside
Blender.

  python3 tools/iso_bathroom/patch_mean.py a.png b.png -- x y w h [name]  ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bubble_profile import read_png


def to_linear(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def patch_mean(pixels, width, height, x, y, w, h):
    total = 0.0
    n = 0
    for j in range(y, min(y + h, height)):
        row = j * width
        for i in range(x, min(x + w, width)):
            r, g, b = pixels[row + i][:3]
            total += 0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)
            n += 1
    return total / max(n, 1)


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        print(__doc__)
        return 1
    split = argv.index("--")
    images = argv[:split]
    rest = argv[split + 1:]

    patches = []
    while rest:
        x, y, w, h = (int(v) for v in rest[:4])
        rest = rest[4:]
        name = ""
        if rest and not rest[0].lstrip("-").isdigit():
            name, rest = rest[0], rest[1:]
        patches.append((name or f"{x},{y}", x, y, w, h))

    loaded = [(Path(p).name,) + read_png(p) for p in images]

    head = f"{'patch':<20}" + "".join(f"{n[:16]:>18}" for n, _, _, _ in loaded)
    print(head)
    for name, x, y, w, h in patches:
        row = f"{name:<20}"
        first = None
        for _, wd, ht, px in loaded:
            m = patch_mean(px, wd, ht, x, y, w, h)
            if first is None:
                first = m
                row += f"{m:>18.4f}"
            else:
                row += f"{m:>12.4f}{100.0 * (m / first - 1.0):>+6.0f}%"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
