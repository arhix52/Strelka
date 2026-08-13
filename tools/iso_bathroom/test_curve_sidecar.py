#!/usr/bin/env python3
"""Tests for the curve sidecar writer and Blender's diameter convention.

Run directly (stdlib only, no Blender):

    python3 tools/iso_bathroom/test_curve_sidecar.py

Two things are pinned here, and they are the two ends of the same contract.

The first is `hair_radii`. Blender's particle properties `root_radius` and
`tip_radius` are *diameters*, and Cycles halves them before it builds a curve.
Both sidecar exporters once passed them through as radii, so every groom
rendered at twice its reference thickness -- and because doubling a strand's
radius doubles the chord a ray crosses inside it, the Chiang lobe absorbed over
twice the path and the whole groom drifted toward the pigment's dominant
channel. That presented as a shading error: on `28_hair` the per-channel ratio
against Cycles read R 1.003 / G 0.962 / B 0.932, which looks exactly like an
absorption coefficient a few percent high. Fixing it in the BSDF would have been
a fit to a geometry bug. The conversion is one multiply, so this test is here to
make sure nobody has to diagnose it a second time.

The second is the byte layout. The format is documented in
`src/sceneloader/include/strelka/sceneloader/curve_sidecar.h` and read by C++
that trusts it completely; a writer that silently disagrees produces a corrupt
acceleration structure rather than an error. So the payload is parsed back here
independently of the writer's own code.
"""

import struct
import tempfile
import unittest
from pathlib import Path
import sys

# The module under test is a script beside this file, not an installed package,
# so it is imported by path. No bytecode cache for it: the cache is keyed on the
# source's mtime to the second and its size in bytes, and editing a constant like
# 0.5 to 1.0 changes neither -- which is enough to run a stale copy of the very
# line this file exists to check.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_sidecar import (  # noqa: E402
    BASIS_BSPLINE,
    BASIS_LINEAR,
    DIAMETER_TO_RADIUS,
    CurveSet,
    hair_radii,
    hair_strand_radii,
    write_curve_sidecar,
)

MAGIC = b"STRKCRV1"


class FakeParticleSettings:
    """The handful of properties the radius helpers read off ParticleSettings.

    Defaults match Blender's for the two that are easy to forget: `shape` flat
    and `use_close_tip` on.
    """

    def __init__(self, root_radius, tip_radius, radius_scale,
                 shape=0.0, use_close_tip=True):
        self.root_radius = root_radius
        self.tip_radius = tip_radius
        self.radius_scale = radius_scale
        self.shape = shape
        self.use_close_tip = use_close_tip


def parse_sidecar(path):
    """Independent reader, mirroring the documented layout field for field."""
    data = Path(path).read_bytes()
    assert data[:8] == MAGIC, "bad magic"
    off = 8
    (set_count,) = struct.unpack_from("<I", data, off)
    off += 4

    sets = []
    for _ in range(set_count):
        (name_len,) = struct.unpack_from("<I", data, off)
        off += 4
        material = data[off:off + name_len].decode("utf-8")
        off += name_len
        basis, strand_count, point_count = struct.unpack_from("<III", data, off)
        off += 12
        transform = struct.unpack_from("<16f", data, off)
        off += 64
        counts = struct.unpack_from("<%dI" % strand_count, data, off)
        off += 4 * strand_count
        coords = struct.unpack_from("<%df" % (point_count * 3), data, off)
        off += 4 * point_count * 3
        radii = struct.unpack_from("<%df" % point_count, data, off)
        off += 4 * point_count
        sets.append({
            "material": material,
            "basis": basis,
            "counts": list(counts),
            "transform": list(transform),
            "points": [tuple(coords[i * 3:i * 3 + 3]) for i in range(point_count)],
            "radii": list(radii),
        })
    assert off == len(data), "trailing bytes: writer and layout disagree"
    return sets


class TestHairRadii(unittest.TestCase):
    """Blender authors diameters; the sidecar carries radii."""

    def test_halves_blenders_diameter(self):
        # The 28_hair groom: the numbers the regression was found on.
        s = FakeParticleSettings(root_radius=0.004, tip_radius=0.0015, radius_scale=1.0)
        root, tip = hair_radii(s)
        self.assertAlmostEqual(root, 0.002, places=9)
        self.assertAlmostEqual(tip, 0.00075, places=9)

    def test_conversion_constant_is_a_half(self):
        # Named so a reader who finds the 0.5 in a diff knows what it is for.
        self.assertEqual(DIAMETER_TO_RADIUS, 0.5)

    def test_radius_scale_multiplies_both_ends(self):
        s = FakeParticleSettings(root_radius=1.0, tip_radius=0.5, radius_scale=0.01)
        root, tip = hair_radii(s)
        self.assertAlmostEqual(root, 0.005, places=9)
        self.assertAlmostEqual(tip, 0.0025, places=9)

    def test_gain_is_an_author_facing_multiplier(self):
        s = FakeParticleSettings(root_radius=0.004, tip_radius=0.002, radius_scale=1.0)
        base_root, base_tip = hair_radii(s)
        root, tip = hair_radii(s, gain=3.0)
        self.assertAlmostEqual(root, base_root * 3.0, places=9)
        self.assertAlmostEqual(tip, base_tip * 3.0, places=9)

    def test_a_zero_tip_stays_zero(self):
        # Blender's default tip diameter is 0: a needle, not a cylinder.
        s = FakeParticleSettings(root_radius=0.01, tip_radius=0.0, radius_scale=1.0)
        root, tip = hair_radii(s)
        self.assertGreater(root, 0.0)
        self.assertEqual(tip, 0.0)


class TestHairStrandRadii(unittest.TestCase):
    """The profile along a strand, not just its two ends.

    The expected numbers here are what Cycles was measured to render: one strand
    of a known property value against an orthographic camera, silhouette width
    read off the alpha channel. See the docstring on `hair_strand_radii`.
    """

    def assertRadii(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for i, (a, e) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(a, e, places=7, msg="point %d" % i)

    def test_close_tip_zeroes_only_the_last_point(self):
        # Blender's default. A strand ends in a point, not a flat cap.
        s = FakeParticleSettings(0.4, 0.4, 1.0, use_close_tip=True)
        r = hair_strand_radii(s, 5)
        self.assertRadii(r[:-1], [0.2] * 4)
        self.assertEqual(r[-1], 0.0)

    def test_without_close_tip_the_strand_is_a_cylinder_to_its_end(self):
        s = FakeParticleSettings(0.4, 0.4, 1.0, use_close_tip=False)
        self.assertRadii(hair_strand_radii(s, 5), [0.2] * 5)

    def test_linear_taper_between_root_and_tip(self):
        s = FakeParticleSettings(0.4, 0.1, 1.0, use_close_tip=False)
        # Radii are half the properties: 0.2 down to 0.05.
        self.assertRadii(hair_strand_radii(s, 5), [0.2, 0.1625, 0.125, 0.0875, 0.05])

    def test_shape_below_zero_bulges_the_strand(self):
        # shape -0.5 -> p = 0.5. Measured diameter at t=0.5 was 0.3126.
        s = FakeParticleSettings(0.4, 0.1, 1.0, shape=-0.5, use_close_tip=False)
        r = hair_strand_radii(s, 5)
        self.assertAlmostEqual(r[2], 0.3121 / 2.0, places=4)
        self.assertGreater(r[2], 0.125)  # fatter than linear at the midpoint

    def test_shape_above_zero_pinches_the_strand(self):
        # shape +0.5 -> p = 2. Measured diameter at t=0.5 was 0.1754.
        s = FakeParticleSettings(0.4, 0.1, 1.0, shape=0.5, use_close_tip=False)
        r = hair_strand_radii(s, 5)
        self.assertAlmostEqual(r[2], 0.1750 / 2.0, places=4)
        self.assertLess(r[2], 0.125)

    def test_ends_are_exact_whatever_the_shape(self):
        for shape in (-0.5, 0.0, 0.5):
            s = FakeParticleSettings(0.4, 0.1, 1.0, shape=shape, use_close_tip=False)
            r = hair_strand_radii(s, 9)
            self.assertAlmostEqual(r[0], 0.2, places=7, msg="shape %s root" % shape)
            self.assertAlmostEqual(r[-1], 0.05, places=7, msg="shape %s tip" % shape)

    def test_the_28_hair_groom(self):
        # The scene's own numbers, with Blender's defaults for the rest.
        s = FakeParticleSettings(0.004, 0.0015, 1.0)
        r = hair_strand_radii(s, 9)
        self.assertAlmostEqual(r[0], 0.002, places=9)
        self.assertAlmostEqual(r[7], 0.00090625, places=9)
        self.assertEqual(r[-1], 0.0)
        for a, b in zip(r, r[1:-1]):
            self.assertGreaterEqual(a, b)

    def test_gain_scales_the_whole_profile(self):
        s = FakeParticleSettings(0.004, 0.0015, 1.0, use_close_tip=False)
        base = hair_strand_radii(s, 5)
        scaled = hair_strand_radii(s, 5, gain=2.0)
        self.assertRadii(scaled, [x * 2.0 for x in base])

    def test_a_single_point_strand_keeps_its_radius(self):
        # Zeroing the only point would make it invisible instead of pointed.
        s = FakeParticleSettings(0.004, 0.0015, 1.0, use_close_tip=True)
        self.assertRadii(hair_strand_radii(s, 1), [0.002])


class TestSidecarLayout(unittest.TestCase):
    """The bytes the C++ reader is entitled to expect."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.dir.name) / "curves.bin")

    def tearDown(self):
        self.dir.cleanup()

    def assertFloatsEqual(self, actual, expected):
        """Element-wise, at float32 precision -- the payload is 32-bit."""
        self.assertEqual(len(actual), len(expected))
        for i, (a, e) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(a, e, places=7, msg="element %d" % i)

    def test_round_trip_preserves_points_and_radii(self):
        strands = [
            [(0.0, 0.0, 0.0, 0.004), (0.0, 0.1, 0.0, 0.003), (0.0, 0.2, 0.0, 0.002)],
            [(1.0, 0.0, 0.0, 0.005), (1.0, 0.1, 0.0, 0.004), (1.0, 0.2, 0.0, 0.001)],
        ]
        counts = write_curve_sidecar(self.path, [CurveSet("hair0", strands, BASIS_LINEAR)])
        self.assertEqual(counts, (2, 6))

        (s,) = parse_sidecar(self.path)
        self.assertEqual(s["material"], "hair0")
        self.assertEqual(s["basis"], BASIS_LINEAR)
        self.assertEqual(s["counts"], [3, 3])
        # Radii are written verbatim: no unit conversion lives in the writer, so a
        # caller that handed it diameters gets a file that is wrong by two.
        self.assertFloatsEqual(s["radii"], [0.004, 0.003, 0.002, 0.005, 0.004, 0.001])
        self.assertFloatsEqual(s["points"][0], (0.0, 0.0, 0.0))
        self.assertFloatsEqual(s["points"][4], (1.0, 0.1, 0.0))

    def test_strands_may_have_different_lengths(self):
        strands = [
            [(0.0, 0.0, 0.0, 0.001)] * 2,
            [(0.0, 0.0, 0.0, 0.001)] * 5,
            [(0.0, 0.0, 0.0, 0.001)] * 3,
        ]
        write_curve_sidecar(self.path, [CurveSet("m", strands, BASIS_LINEAR)])
        (s,) = parse_sidecar(self.path)
        self.assertEqual(s["counts"], [2, 5, 3])
        self.assertEqual(sum(s["counts"]), len(s["radii"]))

    def test_bspline_basis_survives(self):
        strands = [[(0.0, 0.0, 0.0, 0.002)] * 4]
        write_curve_sidecar(self.path, [CurveSet("m", strands, BASIS_BSPLINE)])
        (s,) = parse_sidecar(self.path)
        self.assertEqual(s["basis"], BASIS_BSPLINE)

    def test_identity_transform_when_none_given(self):
        write_curve_sidecar(self.path, [CurveSet("m", [[(0.0, 0.0, 0.0, 0.001)] * 2])])
        (s,) = parse_sidecar(self.path)
        self.assertFloatsEqual(s["transform"],
                               [1.0, 0.0, 0.0, 0.0,
                                0.0, 1.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, 0.0, 1.0])

    def test_transform_is_written_column_major_as_given(self):
        xform = [float(i) for i in range(16)]
        write_curve_sidecar(
            self.path,
            [CurveSet("m", [[(0.0, 0.0, 0.0, 0.001)] * 2], BASIS_LINEAR, transform=xform)])
        (s,) = parse_sidecar(self.path)
        self.assertFloatsEqual(s["transform"], xform)

    def test_multiple_sets_keep_their_own_material_and_basis(self):
        a = CurveSet("hair_a", [[(0.0, 0.0, 0.0, 0.001)] * 2], BASIS_LINEAR)
        b = CurveSet("hair_b", [[(0.0, 0.0, 0.0, 0.002)] * 4], BASIS_BSPLINE)
        strands, points = write_curve_sidecar(self.path, [a, b])
        self.assertEqual((strands, points), (2, 6))
        parsed = parse_sidecar(self.path)
        self.assertEqual([p["material"] for p in parsed], ["hair_a", "hair_b"])
        self.assertEqual([p["basis"] for p in parsed], [BASIS_LINEAR, BASIS_BSPLINE])

    def test_empty_sets_are_dropped_not_written(self):
        # A particle system whose cache never filled must not reach the file as a
        # zero-strand set; the reader counts sets, not strands.
        kept = CurveSet("kept", [[(0.0, 0.0, 0.0, 0.001)] * 2], BASIS_LINEAR)
        write_curve_sidecar(self.path, [CurveSet("empty", [], BASIS_LINEAR), kept])
        parsed = parse_sidecar(self.path)
        self.assertEqual([p["material"] for p in parsed], ["kept"])

    def test_utf8_material_name_length_is_in_bytes(self):
        # nameLength is a byte count, so a multi-byte name must not shift the
        # payload that follows it.
        name = "волосы"
        write_curve_sidecar(self.path, [CurveSet(name, [[(0.0, 0.0, 0.0, 0.001)] * 2])])
        (s,) = parse_sidecar(self.path)
        self.assertEqual(s["material"], name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
