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

The Translucent BSDF these materials add is carried too, as
KHR_materials_diffuse_transmission -- light through a leaf is a large part of how
foliage reads, and it is emphatically not KHR_materials_transmission, which is
specular and would make needles look like glass. The exporter has no idea about
either, so the weight and tint are collected here and injected into the glTF
afterwards by export_scene.py.
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


def _find_translucent(node, depth=0):
    """The Translucent BSDF under a tree of Mix/Add shader nodes, if any."""
    if node is None or depth > 8:
        return None
    if node.type == "BSDF_TRANSLUCENT":
        return node
    if node.type in {"MIX_SHADER", "ADD_SHADER"}:
        for i in node.inputs:
            if i.type != "SHADER" or not i.links:
                continue
            found = _find_translucent(i.links[0].from_node, depth + 1)
            if found is not None:
                return found
    return None


def _mean_colour(socket, depth=0):
    """The average colour a socket produces, following one texture back if need be.

    A constant is returned as-is. A texture is averaged over its pixels, which is
    a real approximation and a defensible one here: light diffusing through a
    leaf comes out low-frequency, so the transmitted tint carries almost none of
    the reflected texture's detail. glTF has a diffuseTransmissionColorTexture
    for the exact answer; this renderer samples only the factor.
    """
    if depth > 6:
        return None
    if not socket.links:
        v = getattr(socket, "default_value", None)
        if v is None:
            return None
        if hasattr(v, "__len__"):
            return [float(v[0]), float(v[1]), float(v[2])]
        # A scalar socket -- a Fac or a Value -- read as grey.
        return [float(v)] * 3
    node = socket.links[0].from_node
    if node.type == "TEX_IMAGE" and node.image is not None:
        w, h = node.image.size
        if w == 0 or h == 0:
            return None
        buf = np.empty(w * h * 4, dtype=np.float32)
        node.image.pixels.foreach_get(buf)
        px = buf.reshape(-1, 4)
        # Weighted by alpha where there is one: the transparent margin of a
        # foliage card is black, and averaging it in makes every leaf too dark.
        alpha = px[:, 3]
        if alpha.max() > 0.0 and alpha.min() < 1.0:
            weight = alpha.sum()
            return [float((px[:, c] * alpha).sum() / weight) for c in range(3)]
        return [float(px[:, c].mean()) for c in range(3)]
    # Colour inputs only, and named ones first. Wandering into a Vector or a Fac
    # returns a number that is not a colour, and it is the sort of wrong answer
    # that renders rather than raises.
    for name in ("Color", "Base Color", "Color1", "A"):
        if name in node.inputs:
            got = _mean_colour(node.inputs[name], depth + 1)
            if got is not None:
                return got
    for i in node.inputs:
        if i.type == "RGBA":
            got = _mean_colour(i, depth + 1)
            if got is not None:
                return got
    return None


def _reaches_texture(socket, depth=0):
    """Whether a socket's chain ends at an image the exporter can carry itself."""
    if depth > 6 or not socket.links:
        return False
    node = socket.links[0].from_node
    if node.type == "TEX_IMAGE" and node.image is not None:
        return True
    return any(_reaches_texture(i, depth + 1) for i in node.inputs)


def _mean_scalar(socket, depth=0):
    """The average value a scalar socket produces.

    Colour chains first, then scalar ones, then the socket's own value. That last
    fallback is not a guess dressed up: Blender keeps default_value on a linked
    socket, and it is what the author set before wiring something in, so for the
    river's Transmission Weight -- driven by noise, defaulting to 1.0 -- it is
    exactly right. Where it is not, it is at least the value the material would
    have had with the graph deleted, which is what the exporter was going to do
    anyway.
    """
    colour = _mean_colour(socket, depth)
    if colour is not None:
        return sum(colour) / 3.0
    if socket.links and depth < 6:
        node = socket.links[0].from_node
        for i in node.inputs:
            if i.type != "VALUE":
                continue
            got = _mean_scalar(i, depth + 1)
            if got is not None:
                return got
    value = getattr(socket, "default_value", None)
    if value is None or hasattr(value, "__len__"):
        return None
    return float(value)


# Inputs the exporter reads as a plain value and silently drops when a node
# drives them. Base Color is handled separately because a texture behind it is
# carried fine and only a procedural chain is not.
_SCALAR_INPUTS = ("Metallic", "Roughness", "IOR", "Transmission Weight", "Transmission",
                  "Specular IOR Level", "Anisotropic")


def _reduce_inputs(principled):
    """Collapse node-driven inputs to their average, where they would be lost.

    The exporter carries a constant or a texture and nothing in between. The
    river's water is a Principled whose Base Color and Transmission Weight are
    both driven by procedural noise, so it exported as a smooth *opaque white*
    surface -- which rendered as a bright sandbank where the reference has dark
    water, and looked like missing geometry rather than a dropped input.

    An average is a real approximation and is stated as one. It is the same
    trade as the translucency tint above, and the alternative is baking every
    such input to a texture, which is a much larger piece of work for inputs
    that are usually near-constant anyway.
    """
    changed = []
    base = principled.inputs.get("Base Color")
    if base is not None and base.links and not _reaches_texture(base):
        colour = _mean_colour(base)
        if colour is not None:
            for link in list(base.links):
                principled.id_data.links.remove(link)
            base.default_value = (colour[0], colour[1], colour[2], 1.0)
            changed.append("base colour")

    for name in _SCALAR_INPUTS:
        socket = principled.inputs.get(name)
        if socket is None or not socket.links or _reaches_texture(socket):
            continue
        value = _mean_scalar(socket)
        if value is None:
            continue
        for link in list(socket.links):
            principled.id_data.links.remove(link)
        socket.default_value = value
        changed.append("%s=%.3f" % (name, value))
    return changed


def _volume_absorption(mat):
    """(attenuation colour, attenuation distance) from a Volume Absorption node.

    glTF has KHR_materials_volume for exactly this and the exporter writes it
    only from setups it recognises, which a Volume Absorption wired straight to
    the output is not. Blender's density is an absorption coefficient, so the
    distance light travels before the colour is reached is its reciprocal.
    """
    out = next((n for n in mat.node_tree.nodes
                if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if out is None:
        return None
    socket = out.inputs.get("Volume")
    if socket is None or not socket.links:
        return None
    node = socket.links[0].from_node
    if node.type not in {"VOLUME_ABSORPTION", "PRINCIPLED_VOLUME"}:
        return None
    colour = node.inputs["Color"].default_value
    density = float(node.inputs["Density"].default_value)
    if density <= 0.0:
        return None
    return [float(colour[0]), float(colour[1]), float(colour[2])], 1.0 / density


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
    """Rewrite every material that needs it.

    Returns (report, translucency, volumes): the two extension payloads the
    exporter cannot derive on its own, keyed by material name.
    """
    report = []
    translucency = {}
    volumes = {}
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        found = _surface(mat)
        if found is None:
            continue
        surface, out_node = found

        volume = _volume_absorption(mat)
        if volume is not None:
            volumes[mat.name] = volume

        if surface.type == "BSDF_PRINCIPLED":
            # Already the shape the exporter wants -- but its inputs may still be
            # driven by nodes it cannot read.
            reduced = _reduce_inputs(surface)
            if reduced or volume is not None:
                note = "inputs reduced: %s" % ", ".join(reduced) if reduced else "volume only"
                if volume is not None:
                    note += "; volume %s over %.2f" % (
                        "".join("%.2f " % c for c in volume[0]).strip(), volume[1])
                report.append((mat.name, note))
            continue

        principled = _find_principled(surface)
        if principled is None:
            report.append((mat.name, "no Principled BSDF under %s" % surface.type))
            continue

        nt = mat.node_tree

        # Foliage translucency, before the graph is rewritten and the node is
        # orphaned. Blender writes it as a Translucent BSDF added alongside the
        # Principled; glTF has KHR_materials_diffuse_transmission, and the
        # exporter knows nothing about either.
        translucent = _find_translucent(surface)
        if translucent is not None:
            colour = _mean_colour(translucent.inputs["Color"]) or [1.0, 1.0, 1.0]
            # 0.5, because the graph adds the two lobes at full strength and the
            # glTF model splits one lobe between them. Half each is the
            # energy-conserving reading of "reflect and transmit equally", which
            # is what the author expressed; carrying the addition across
            # literally would make a canopy brighter than the sky behind it.
            # A judgement call, and the only one available without a weight in
            # the graph to read.
            translucency[mat.name] = (0.5, colour)

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
        reduced = _reduce_inputs(principled)
        if reduced:
            note += ", reduced " + ", ".join(reduced)
        if mat.name in translucency:
            note += ", translucency %.2f %s" % (
                translucency[mat.name][0],
                "".join("%.2f " % c for c in translucency[mat.name][1]).strip())
        report.append((mat.name, note))
    return report, translucency, volumes
