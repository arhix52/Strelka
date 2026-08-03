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

Note the operand order differs, and so does the polarity; the reasoning is at the
rewiring itself. Once the graph is a plain Principled BSDF the exporter merges the
base colour and the alpha into one RGBA texture on its own, which is the only
place glTF has to put alpha.

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


def _invert_image(img, out_dir):
    """1 - value, written to disk next to the export.

    Cached per source image, so a mask shared by several materials is inverted
    once rather than once per use.
    """
    name = "%s_inv" % os.path.splitext(img.name)[0]
    existing = bpy.data.images.get(name)
    if existing is not None:
        return existing

    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    px = buf.reshape(-1, 4)
    px[:, :3] = 1.0 - px[:, :3]

    # Byte buffer, and the colour space set before the pixels are written. Both
    # matter, and neither fails loudly. A float buffer saved to PNG puts the
    # values through a transfer function on the way out (0.75 arrives as 0.52),
    # and setting the colour space afterwards drops the write entirely -- the
    # file comes out uniformly black, which reads as a mask covering nothing
    # rather than as a save that did not happen. Measured, all four ways.
    out = bpy.data.images.new(name, width=w, height=h, alpha=False)
    out.colorspace_settings.name = img.colorspace_settings.name
    out.pixels.foreach_set(px.ravel())
    out.update()
    out.filepath_raw = os.path.join(out_dir, name + ".png")
    out.file_format = "PNG"
    out.save()
    return out


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
            # Mix Shader outputs (1 - Fac) * slot0 + Fac * slot1. A Transparent
            # in slot 0 therefore means opacity *is* the factor; in slot 1 it
            # means opacity is one minus it.
            #
            # fir_twig settles which way round that is without any guessing: its
            # Principled node already drives Alpha from the same texture that
            # feeds the mix, and its Transparent sits in slot 0. So slot 0 means
            # no inversion.
            #
            # Backwards, this does not look like an inverted mask. It looks like
            # the canopy is the wrong density, and the error arrives as light on
            # the ground far from anything to do with foliage.
            src = link.from_socket
            note = "alpha from the %s mix branch" % ("first" if transparent_first else "second")
            if not transparent_first:
                # Inverted in the image rather than with a Math node, because the
                # exporter does not evaluate the graph feeding Alpha -- it looks
                # for a texture and takes it whole. A SUBTRACT node in between is
                # silently skipped, and the mask exports at exactly the polarity
                # it was meant not to have.
                mask_img, _ = _image_of(link)
                if mask_img is None:
                    report.append((mat.name, "alpha needs inverting but is not a plain texture"))
                    continue
                inverted = _invert_image(mask_img, out_dir)
                tex = nt.nodes.new("ShaderNodeTexImage")
                tex.image = inverted
                for old in list(link.from_node.inputs):
                    if old.name == "Vector" and old.links:
                        nt.links.new(old.links[0].from_socket, tex.inputs["Vector"])
                src = tex.outputs["Color"]
                note += ", inverted into %s" % inverted.name
            nt.links.new(src, principled.inputs["Alpha"])
        else:
            note = "flattened, no transparency found"

        nt.links.new(principled.outputs["BSDF"], out_node.inputs["Surface"])
        report.append((mat.name, note))
    return report
