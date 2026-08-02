#!/usr/bin/env python3
"""
Rewrite shader graphs the glTF exporter cannot express into ones it can.

Imported by export_scene.py; not run on its own.

The glTF exporter understands one shape: a Principled BSDF wired straight to the
material output. Anything else -- and production foliage is always something else
-- exports as a flat opaque material with, at best, the base colour texture. The
pine forest's needle clusters came through as solid green cards, 7 800 of them,
and the render was unrecognisable.

Two idioms are handled here, both being the same thing written two ways:

    Output <- Add(Mix(mask, Principled, Transparent), Translucent)      pine_cover
    Output <- Mix(alpha, Transparent, Add(Principled, Mix(..)))         fir_twig

Note the operand order differs, and so does the polarity: Mix Shader takes its
first input at factor 0, so a Transparent in slot 1 means alpha = factor and a
Transparent in slot 0 means alpha = 1 - factor. Guessing this wrong inverts every
leaf in the scene, which does not look like an inverted mask -- it looks like the
foliage is missing.

What is lost, and deliberately: the Translucent BSDF. Light passing through a
leaf is a large part of how foliage reads, and glTF has no diffuse-transmission
parameter -- KHR_materials_transmission is specular transmission and would make
needles look like glass. Dropping it is the honest option of the ones available;
the alternative is a Strelka-specific extension and a matching BSDF lobe, which
is worth doing and is not this.
"""

import bpy
import numpy as np
import os


def _surface(mat):
    out = next((n for n in mat.node_tree.nodes
                if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if out is None or not out.inputs["Surface"].links:
        return None
    return out.inputs["Surface"].links[0].from_node, out


def _find_principled(node, depth=0):
    """The Principled BSDF under a tree of Mix/Add shader nodes."""
    if node is None or depth > 8:
        return None
    if node.type == "BSDF_PRINCIPLED":
        return node
    if node.type in {"MIX_SHADER", "ADD_SHADER"}:
        for i in node.inputs:
            if i.type != "SHADER" or not i.links:
                continue
            found = _find_principled(i.links[0].from_node, depth + 1)
            if found is not None:
                return found
    return None


def _alpha_source(node, depth=0):
    """(socket driving the mix, transparent_is_first) for the Transparent branch.

    Returns None when the graph has no Transparent BSDF, which is the common
    case and not a problem -- an opaque material needs no rewriting.
    """
    if node is None or depth > 8:
        return None
    if node.type == "MIX_SHADER":
        shader_inputs = [i for i in node.inputs if i.type == "SHADER"]
        for slot, i in enumerate(shader_inputs):
            if i.links and i.links[0].from_node.type == "BSDF_TRANSPARENT":
                fac = node.inputs["Fac"] if "Fac" in node.inputs else node.inputs[0]
                if not fac.links:
                    return None
                return fac.links[0], slot == 0
    if node.type in {"MIX_SHADER", "ADD_SHADER"}:
        for i in node.inputs:
            if i.type != "SHADER" or not i.links:
                continue
            found = _alpha_source(i.links[0].from_node, depth + 1)
            if found is not None:
                return found
    return None


def _image_of(link):
    """The image behind a link, if it is a texture read and nothing else."""
    n = link.from_node
    if n.type == "TEX_IMAGE" and n.image is not None:
        return n.image, n
    return None, None


def _resample(a, h, w):
    """Nearest-neighbour, because a mask only has to line up, not interpolate."""
    sh, sw = a.shape[:2]
    if (sh, sw) == (h, w):
        return a
    yi = (np.arange(h) * sh // h).clip(0, sh - 1)
    xi = (np.arange(w) * sw // w).clip(0, sw - 1)
    return a[yi][:, xi]


def _pixels(img):
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)


def _merge_rgba(base_img, mask_img, invert, out_dir):
    """Write base RGB with the mask in alpha, because glTF has nowhere else for it.

    glTF carries alpha in baseColorTexture.A and only there. Foliage is authored
    with the mask as its own file, so without this the exporter has a base colour
    and an alpha that cannot travel together, and drops the alpha.
    """
    base = _pixels(base_img)
    h, w, _ = base.shape
    mask = _resample(_pixels(mask_img), h, w)[:, :, 0]
    if invert:
        mask = 1.0 - mask

    name = "%s_rgba" % os.path.splitext(base_img.name)[0]
    merged = bpy.data.images.new(name, width=w, height=h, alpha=True)
    rgba = np.empty((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = base[:, :, :3]
    rgba[:, :, 3] = mask
    merged.pixels.foreach_set(rgba.ravel())
    # sRGB and 8-bit to match what was there: pixels come back linear, and
    # Blender re-encodes on save, so this round-trips the original values.
    merged.colorspace_settings.name = base_img.colorspace_settings.name
    merged.alpha_mode = "CHANNEL_PACKED"
    merged.filepath_raw = os.path.join(out_dir, name + ".png")
    merged.file_format = "PNG"
    merged.save()
    return merged


def flatten(out_dir):
    """Rewrite every material that needs it. Returns a per-material report."""
    report = []
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        found = _surface(mat)
        if found is None:
            continue
        surface, out_node = found
        if surface.type == "BSDF_PRINCIPLED":
            continue  # already the shape the exporter wants

        principled = _find_principled(surface)
        if principled is None:
            report.append((mat.name, "no Principled BSDF under %s" % surface.type))
            continue

        nt = mat.node_tree
        alpha = _alpha_source(surface)
        note = "flattened"

        if alpha is not None:
            link, transparent_first = alpha
            mask_img, mask_node = _image_of(link)
            base_in = principled.inputs["Base Color"]
            base_img, _ = _image_of(base_in.links[0]) if base_in.links else (None, None)

            if mask_img is not None and base_img is not None and mask_img != base_img:
                merged = _merge_rgba(base_img, mask_img, transparent_first, out_dir)
                tex = nt.nodes.new("ShaderNodeTexImage")
                tex.image = merged
                # The base colour may run through Hue/Saturation or similar; that
                # chain is left alone and only its texture swapped, so the look is
                # preserved rather than approximated.
                nt.links.new(tex.outputs["Color"], base_in.links[0].to_socket)
                nt.links.new(tex.outputs["Alpha"], principled.inputs["Alpha"])
                note = "alpha from %s merged into base colour" % mask_img.name
            else:
                src = link.from_socket
                if transparent_first:
                    inv = nt.nodes.new("ShaderNodeMath")
                    inv.operation = "SUBTRACT"
                    inv.inputs[0].default_value = 1.0
                    nt.links.new(src, inv.inputs[1])
                    src = inv.outputs["Value"]
                nt.links.new(src, principled.inputs["Alpha"])
                note = "alpha wired directly"
        else:
            note = "flattened, no transparency found"

        nt.links.new(principled.outputs["BSDF"], out_node.inputs["Surface"])
        report.append((mat.name, note))
    return report
