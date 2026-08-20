// Curve geometry layout for the OptiX backend: which control points a segment
// spans, which segments a set of strands produces, and where along its strand a
// hit landed.
//
// These were both wrong and neither was visible as a crash. `createCurve`
// hardcoded degree 3, so every linear groom in the tree -- which is all of them
// -- was built as a cubic B-spline over the same control points: a curve that
// does not pass through them, one segment shorter per strand at each end.
// And the curve uv was pinned to (0.5, 0.5), so a strand could not be textured
// and a root-to-tip ramp was a constant.
//
// Nothing here needs a GPU, so nothing here is checked by rendering.

#include <doctest/doctest.h>

#include <curve_layout.h>

#include <vector>

using namespace oka::curve_layout;

TEST_CASE("a segment spans two control points when linear and four when cubic")
{
    CHECK(controlPointsPerSegment(true) == 2u);
    CHECK(controlPointsPerSegment(false) == 4u);
}

TEST_CASE("a linear strand of n points makes n-1 segments, a cubic one n-3")
{
    const std::vector<uint32_t> counts = { 9 };

    const std::vector<int> linear = segmentIndices(counts, 0, 1, 0, true);
    CHECK(linear.size() == 8);
    for (size_t i = 0; i < linear.size(); ++i)
    {
        CHECK(linear[i] == (int)i);
    }

    const std::vector<int> cubic = segmentIndices(counts, 0, 1, 0, false);
    CHECK(cubic.size() == 6);
    CHECK(cubic.front() == 0);
    CHECK(cubic.back() == 5); // last segment spans points 5..8
}

TEST_CASE("segments of consecutive strands do not run into each other")
{
    // Two strands of four points each. The linear set must not emit a segment
    // that starts at the last point of strand 0 and ends at the first of strand
    // 1 -- that is a strand-long spike across the groom, and the sort of thing
    // that reads as a rendering artefact rather than an indexing bug.
    const std::vector<uint32_t> counts = { 4, 4 };
    const std::vector<int> linear = segmentIndices(counts, 0, 2, 0, true);
    REQUIRE(linear.size() == 6);
    CHECK(linear == std::vector<int>{ 0, 1, 2, 4, 5, 6 });

    const std::vector<int> cubic = segmentIndices(counts, 0, 2, 0, false);
    REQUIRE(cubic.size() == 2);
    CHECK(cubic == std::vector<int>{ 0, 4 });
}

TEST_CASE("indices are global, because OptiX is handed the whole point buffer")
{
    const std::vector<uint32_t> counts = { 3, 3 };
    // This set starts 100 points into the scene's shared curve point buffer.
    const std::vector<int> linear = segmentIndices(counts, 0, 2, 100, true);
    CHECK(linear == std::vector<int>{ 100, 101, 103, 104 });
}

TEST_CASE("a strand too short for one segment contributes nothing")
{
    const std::vector<uint32_t> counts = { 1, 5, 2 };
    // Cubic needs four points: only the middle strand qualifies, and it makes
    // two segments starting at the point after the one-point strand.
    const std::vector<int> cubic = segmentIndices(counts, 0, 3, 0, false);
    CHECK(cubic == std::vector<int>{ 1, 2 });

    // Linear needs two: the one-point strand still contributes nothing, and the
    // two-point strand contributes exactly one segment.
    const std::vector<int> linear = segmentIndices(counts, 0, 3, 0, true);
    CHECK(linear == std::vector<int>{ 1, 2, 3, 4, 6 });
}

TEST_CASE("a set can start part way into the shared strand-count array")
{
    const std::vector<uint32_t> counts = { 7, 7, 3, 3 };
    // The second set owns strands 2 and 3 and starts at point 14.
    const std::vector<int> linear = segmentIndices(counts, 2, 2, 14, true);
    CHECK(linear == std::vector<int>{ 14, 15, 17, 18 });
}

TEST_CASE("segments per strand is reported only when every strand agrees")
{
    CHECK(segmentsPerStrand({ 9, 9, 9 }, 0, 3, true) == 8u);
    CHECK(segmentsPerStrand({ 9, 9, 9 }, 0, 3, false) == 6u);
    CHECK(segmentsPerStrand({ 9, 8, 9 }, 0, 3, true) == 0u);
    // Uniform, but too short for the basis: no gradient rather than a negative
    // count wrapped into a large unsigned one.
    CHECK(segmentsPerStrand({ 3, 3 }, 0, 2, false) == 0u);
    CHECK(segmentsPerStrand({ 1, 1 }, 0, 2, true) == 0u);
    CHECK(segmentsPerStrand({}, 0, 0, true) == 0u);
}

TEST_CASE("the strand coordinate runs root to tip across a whole strand")
{
    // Eight segments per strand, which is what a nine-point linear groom gives.
    const uint32_t perStrand = 8;

    // Start of the first segment of the first strand is the root.
    CHECK(strandCoordinate(0, perStrand, 0.0f) == doctest::Approx(0.0f));
    // End of the last segment of the first strand is the tip.
    CHECK(strandCoordinate(7, perStrand, 1.0f) == doctest::Approx(1.0f));
    // Half way along the third segment.
    CHECK(strandCoordinate(2, perStrand, 0.5f) == doctest::Approx(2.5f / 8.0f));

    // The second strand restarts at the root: the coordinate is a position
    // within a strand, not within the set.
    CHECK(strandCoordinate(perStrand, perStrand, 0.0f) == doctest::Approx(0.0f));
    CHECK(strandCoordinate(perStrand + 3, perStrand, 0.25f) == doctest::Approx(3.25f / 8.0f));

    // It is monotonic along a strand, which is what makes a ramp a ramp.
    float prev = -1.0f;
    for (uint32_t seg = 0; seg < perStrand; ++seg)
    {
        for (int k = 0; k < 4; ++k)
        {
            const float c = strandCoordinate(seg, perStrand, 0.25f * static_cast<float>(k));
            CHECK(c > prev);
            prev = c;
        }
    }
}

TEST_CASE("a set with mixed strand lengths reports the root everywhere")
{
    // 0 means "the segment index says nothing about position", and every hit
    // then reads the root. A visible loss of gradient, rather than a modulo by
    // a count that does not describe this strand.
    CHECK(strandCoordinate(0, 0, 0.5f) == doctest::Approx(0.0f));
    CHECK(strandCoordinate(1234, 0, 1.0f) == doctest::Approx(0.0f));
}
