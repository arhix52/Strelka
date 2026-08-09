#!/usr/bin/env python3
"""Radial luminance profile across a bubble, for images this repo renders.

The soap bubbles are the one place in the Isometric Bathroom conversion where
the defect has a radius rather than a hue: a thin-walled sphere read from the
wrong side of its interface total-internally-reflects everything outside
r / R = 1 / ior, so what goes wrong is an annulus with a computable inner edge.
A radial profile is therefore the measurement that can tell "the bubble is
dark" from "the bubble is dark outside 0.625 R", and only the second one names
a cause.

Pure stdlib on purpose: this tree has no numpy, and a PNG that sips wrote is a
non-interlaced 8-bit RGB(A) file, which is forty lines of zlib and struct.

  python3 tools/iso_bathroom/bubble_profile.py /tmp/iso.png 377 410 11.5
  python3 tools/iso_bathroom/bubble_profile.py before.png after.png 377 410 11.5
"""

import struct
import sys
import zlib


def read_png(path):
    """Return (width, height, [(r, g, b), ...]) for an 8-bit non-interlaced PNG."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")

    pos = 8
    idat = bytearray()
    width = height = depth = color = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or color not in (2, 6) or interlace != 0:
                raise ValueError(f"{path}: need 8-bit non-interlaced RGB/RGBA")
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break

    channels = 3 if color == 2 else 4
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    out = []
    prev = bytearray(stride)
    at = 0
    for _ in range(height):
        filt = raw[at]
        at += 1
        line = bytearray(raw[at : at + stride])
        at += stride
        # PNG per-scanline filters, in the order the spec numbers them.
        if filt == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filt == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif filt == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif filt == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        elif filt != 0:
            raise ValueError(f"{path}: unknown filter {filt}")
        for x in range(width):
            o = x * channels
            out.append((line[o], line[o + 1], line[o + 2]))
        prev = line
    return width, height, out


def luminance(px):
    return (0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2]) / 255.0


def profile(width, height, pixels, cx, cy, radius, bins=10, outer=1.6):
    """Mean luminance per r/R bin, plus the background ring past the sphere."""
    sums = [0.0] * bins
    counts = [0] * bins
    bg_sum, bg_count = 0.0, 0
    span = int(radius * outer) + 2
    for y in range(max(0, int(cy) - span), min(height, int(cy) + span + 1)):
        for x in range(max(0, int(cx) - span), min(width, int(cx) + span + 1)):
            dx, dy = x - cx, y - cy
            r = (dx * dx + dy * dy) ** 0.5 / radius
            lum = luminance(pixels[y * width + x])
            if r < 1.0:
                b = min(bins - 1, int(r * bins))
                sums[b] += lum
                counts[b] += 1
            elif 1.15 <= r <= outer:
                bg_sum += lum
                bg_count += 1
    means = [(sums[i] / counts[i]) if counts[i] else float("nan") for i in range(bins)]
    return means, (bg_sum / bg_count if bg_count else float("nan"))


def main():
    args = sys.argv[1:]
    images = []
    while args and not args[0].replace(".", "").replace("-", "").isdigit():
        images.append(args.pop(0))
    if len(args) < 3 or not images:
        print(__doc__)
        return 1
    cx, cy, radius = float(args[0]), float(args[1]), float(args[2])
    bins = int(args[3]) if len(args) > 3 else 10

    results = []
    for path in images:
        w, h, px = read_png(path)
        results.append((path, *profile(w, h, px, cx, cy, radius, bins)))

    print(f"bubble at ({cx:g}, {cy:g}) r={radius:g}px, mean luminance per r/R bin")
    print(f"{'r/R':>12} " + " ".join(f"{p.split('/')[-1]:>18}" for p, _, _ in results))
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        cells = " ".join(
            f"{m[i]:9.4f} ({m[i] / bg:5.2f})" if bg == bg else f"{m[i]:9.4f}"
            for _, m, bg in results
        )
        print(f"{lo:5.2f}-{hi:4.2f} {cells}")
    print(
        f"{'background':>12} "
        + " ".join(f"{bg:9.4f} ({1.0:5.2f})" for _, _, bg in results)
    )
    print("\n(ratio in parentheses is the bin over the wall just outside the bubble)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
