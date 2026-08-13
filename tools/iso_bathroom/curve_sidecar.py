"""Writer for the ``<stem>_curves.bin`` sidecar Strelka reads beside a glTF.

glTF has no curve primitive, and hair is the one thing in these scenes that must
not be triangulated: a strand is a handful of control points, and the ribbon that
would replace it costs an order of magnitude more memory for a worse silhouette.
Both renderer backends take curves natively, so the sidecar is only a way to get
them off disk.

The format is documented once, in
``src/sceneloader/include/strelka/sceneloader/curve_sidecar.h``. This module is
the other end of it; keep the two in step.
"""

import struct

MAGIC = b"STRKCRV1"

BASIS_LINEAR = 0
BASIS_BSPLINE = 1

# Blender's particle hair properties named ``root_radius`` and ``tip_radius`` are
# *diameters*: the UI labels them "Diameter Root" / "Diameter Tip", and Cycles
# halves them when it builds the curve (``blender_curves.cpp`` folds a 0.5 into
# ``radius_scale``). Passing the raw property through as a radius therefore makes
# every strand twice as thick as the Cycles render it gets compared against.
#
# That is not only a silhouette error. Doubling the radius doubles the chord a
# ray crosses inside a strand, so the Chiang lobe absorbs over twice the path and
# the groom shifts toward the pigment's dominant channel. On 28_hair it read as a
# warm tilt worth 7% in blue, and reading it as a BSDF absorption bug is the trap
# this constant exists to close.
DIAMETER_TO_RADIUS = 0.5


def hair_radii(settings, gain=1.0):
    """``(root, tip)`` strand radii for a Blender HAIR particle system.

    ``settings`` is a ``ParticleSettings`` (only three float attributes are
    read, so any object carrying them works). ``gain`` is an author-facing
    multiplier for converters that need to fatten a groom deliberately.
    """
    scale = settings.radius_scale * DIAMETER_TO_RADIUS * gain
    return settings.root_radius * scale, settings.tip_radius * scale


def hair_strand_radii(settings, count, gain=1.0):
    """Per-control-point radii for one strand of ``count`` points.

    Four properties shape a strand and all four have to be read together, or the
    groom is the wrong thickness somewhere along its length. Every number below
    was measured against Cycles rather than read out of its source: one strand of
    a known property value, an orthographic camera, and the silhouette's width in
    pixels off the alpha channel.

    ``root_radius`` / ``tip_radius`` are diameters -- see ``hair_radii``.

    ``shape`` bends the root-to-tip interpolation:
    ``r(t) = (1 - t)**p * (root - tip) + tip``, with ``p = 1 + shape`` below zero
    and ``p = 1 / (1 - shape)`` above it. Measured at shape -0.5, 0 and +0.5, the
    formula holds to under a percent at every t.

    ``use_close_tip`` is on by default and forces the *last* control point to
    zero, so a strand ends in a point rather than a flat cap. It is the one that
    is easy to miss and it is not small: the taper lands entirely in the last
    segment, which for 28_hair is about 3% of the groom's projected area and all
    of it in the outer ring -- exactly where a hair comparison is most sensitive,
    and the ring that read 5% bright before this was applied.
    """
    root, tip = hair_radii(settings, gain)
    shape = float(settings.shape)
    if shape < 0.0:
        power = 1.0 + shape
    elif shape > 0.0:
        power = 1.0 / (1.0 - shape)
    else:
        power = 1.0

    radii = []
    for i in range(count):
        t = i / (count - 1) if count > 1 else 0.0
        radii.append(((1.0 - t) ** power) * (root - tip) + tip)
    # A one-point strand is not a curve, and zeroing its only radius would make
    # it invisible rather than pointed.
    if settings.use_close_tip and count > 1:
        radii[-1] = 0.0
    return radii


class CurveSet:
    """One material's worth of strands.

    ``strands`` is a list of point lists; each point is ``(x, y, z, radius)``.
    Radii, not diameters -- that is what both OptiX's width buffer and Metal's
    radius buffer mean.

    ``transform`` is a 16-float column-major object-to-world matrix, or None for
    the identity, which is what a converter that already baked the placement into
    the points wants.
    """

    def __init__(self, material, strands, basis=BASIS_LINEAR, transform=None):
        self.material = material
        self.strands = strands
        self.basis = basis
        self.transform = transform


def write_curve_sidecar(path, sets):
    """Write the sets to ``path``. Returns (strand count, control point count)."""
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]

    written = [s for s in sets if s.strands]
    total_strands = 0
    total_points = 0

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(written)))
        for cs in written:
            name = cs.material.encode("utf-8")
            f.write(struct.pack("<I", len(name)))
            f.write(name)
            point_count = sum(len(s) for s in cs.strands)
            f.write(struct.pack("<III", cs.basis, len(cs.strands), point_count))
            f.write(struct.pack("<16f", *(cs.transform or identity)))
            f.write(struct.pack("<%dI" % len(cs.strands), *(len(s) for s in cs.strands)))

            # Points and radii are separate arrays on disk because they are
            # separate buffers on the GPU: the acceleration structure takes a
            # control point buffer and a radius buffer with independent strides.
            coords = []
            radii = []
            for strand in cs.strands:
                for x, y, z, r in strand:
                    coords.extend((x, y, z))
                    radii.append(r)
            f.write(struct.pack("<%df" % len(coords), *coords))
            f.write(struct.pack("<%df" % len(radii), *radii))

            total_strands += len(cs.strands)
            total_points += point_count

    return total_strands, total_points
