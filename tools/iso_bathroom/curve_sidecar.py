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
