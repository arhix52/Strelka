#!/usr/bin/env python3
"""
Rebuild the merged base-colour/alpha textures a previous export produced.

    blender -b <file.blend> -P tools/feature_tests/remerge_textures.py -- --out DIR

glTF carries alpha in baseColorTexture.A and nowhere else, so the exporter
combines a foliage material's diffuse and its mask into one RGBA image named
"<diffuse>-<alpha>.png". Those files belong to the export, not to the .blend, and
anything that clears the output directory takes them with it -- after which the
glTF still references them and the render comes back with untextured foliage.

Regenerating them is a matter of image arithmetic, so this does that rather than
asking for a full re-export, which on a scene this size needs several gigabytes
and a few minutes.
"""

import bpy
import numpy as np
import os
import sys


def pixels(img):
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4), w, h


def resample(a, h, w):
    """Nearest neighbour; a mask only has to line up, not interpolate."""
    sh, sw = a.shape[:2]
    if (sh, sw) == (h, w):
        return a
    yi = (np.arange(h) * sh // h).clip(0, sh - 1)
    xi = (np.arange(w) * sw // w).clip(0, sw - 1)
    return a[yi][:, xi]


def find_image(name):
    """A .blend image by name, tolerating the exporter's habit of keeping the
    extension in the middle of a merged name."""
    for candidate in (name, name + ".png", name.replace(".png", "") + ".png"):
        img = bpy.data.images.get(candidate)
        if img is not None:
            return img
    # Blender uniquifies a repeated name with a .001 suffix, and the exporter
    # writes the name it was given -- so an exact lookup misses exactly the
    # images that appear in more than one material.
    stem = name.replace(".png", "")
    for img in bpy.data.images:
        if img.name.replace(".png", "").split(".")[0] == stem.split(".")[0] and img.size[0]:
            return img
    return None


def rebuild(target, out_dir):
    """One merged texture, from its two halves, named as the exporter named it."""
    stem = os.path.splitext(os.path.basename(target))[0]
    if "-" not in stem:
        return "not a merged name"
    base_name, alpha_name = stem.rsplit("-", 1)

    invert = alpha_name.endswith("_inv")
    if invert:
        alpha_name = alpha_name[: -len("_inv")]

    base_img = find_image(base_name)
    alpha_img = find_image(alpha_name)
    if base_img is None or alpha_img is None:
        return "missing source: %s" % (base_name if base_img is None else alpha_name)

    base, w, h = pixels(base_img)
    alpha = resample(pixels(alpha_img)[0], h, w)[:, :, 0]
    if invert:
        alpha = 1.0 - alpha

    out = bpy.data.images.new(stem, width=w, height=h, alpha=True)
    # Colour space before the pixels, and a byte buffer: a float buffer saved to
    # PNG goes through a transfer function, and setting the space afterwards
    # drops the write entirely. Both fail silently. See blender-reference-traps.
    out.colorspace_settings.name = base_img.colorspace_settings.name
    out.alpha_mode = "CHANNEL_PACKED"
    rgba = np.empty((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = base[:, :, :3]
    rgba[:, :, 3] = alpha
    out.pixels.foreach_set(rgba.ravel())
    out.update()
    out.filepath_raw = os.path.join(out_dir, stem + ".png")
    out.file_format = "PNG"
    out.save()
    return "rebuilt %dx%d, alpha mean %.3f" % (w, h, float(alpha.mean()))


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    out = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv else "."

    import json
    name = os.path.splitext(os.path.basename(bpy.data.filepath))[0] or "scene"
    gltf = os.path.join(out, name + ".gltf")
    with open(gltf) as f:
        doc = json.load(f)

    missing = [i["uri"] for i in doc.get("images", [])
               if "uri" in i and not os.path.exists(os.path.join(out, i["uri"]))]
    if not missing:
        print("REMERGE nothing missing")
        return
    for uri in missing:
        print("REMERGE %-52s %s" % (uri, rebuild(uri, out)))


if __name__ == "__main__":
    main()
