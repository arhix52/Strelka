#pragma once

// How a set of strands becomes a list of curve segments, and how a hit on one
// of those segments becomes a position along its strand.
//
// Both are pure arithmetic over the sidecar's numbers, so they live here rather
// than inside the acceleration-structure builder: the builder needs a GPU and a
// context to run at all, while these two answers are exactly the ones that were
// wrong -- the basis was hardcoded to cubic, and the strand coordinate did not
// exist, so every curve hit read uv (0.5, 0.5). `tests/render/test_curve_layout.cpp`
// pins them.
//
// No CUDA and no OptiX in here. The closest-hit shader includes it for the
// strand coordinate so the host and the device cannot drift apart on where
// along a strand a segment index lands.

#include <cstdint>

#if defined(__CUDACC__)
#    define OKA_CURVE_FN __host__ __device__ inline
#else
#    define OKA_CURVE_FN inline
#endif

#if !defined(__CUDACC__)
#    include <vector>
#endif

namespace oka
{
namespace curve_layout
{

/// Control points one segment spans. A round linear segment is a cylinder
/// between two points; a cubic B-spline segment is defined by four, and
/// consecutive segments overlap in three of them.
OKA_CURVE_FN uint32_t controlPointsPerSegment(bool isLinear)
{
    return isLinear ? 2u : 4u;
}

/// Where along its strand a hit landed, in [0, 1).
///
/// Segments are laid out strand after strand, so with a uniform segment count
/// the index modulo that count is the segment's position within its strand and
/// the curve parameter interpolates inside it. A set whose strands differ in
/// length carries 0, and every hit reports the root -- a ramp with no gradient,
/// which is a visible and explicable failure rather than an index into the
/// wrong strand.
OKA_CURVE_FN float strandCoordinate(uint32_t primitiveIndex, uint32_t segmentsPerStrand, float u)
{
    if (segmentsPerStrand == 0u)
    {
        return 0.0f;
    }
    return ((float)(primitiveIndex % segmentsPerStrand) + u) / (float)segmentsPerStrand;
}

#if !defined(__CUDACC__)
/// The index buffer an OptiX curve build wants: one entry per segment, holding
/// the index of that segment's first control point in the shared point buffer.
///
/// `pointsStart` is the set's offset into the scene-wide point buffer, because
/// OptiX is handed the whole buffer and indexes into it globally.
///
/// A strand with fewer control points than one segment needs contributes
/// nothing. The version this replaces computed `count - degree` as a signed
/// int, which is the same answer for the cubic case it was hardcoded to and a
/// negative one -- an empty loop by accident rather than by intent -- for a
/// strand too short.
inline std::vector<int> segmentIndices(const std::vector<uint32_t>& vertexCounts,
                                       uint32_t firstStrand,
                                       uint32_t strandCount,
                                       uint32_t pointsStart,
                                       bool isLinear)
{
    const uint32_t perSegment = controlPointsPerSegment(isLinear);
    std::vector<int> indices;
    uint32_t pointCursor = pointsStart;
    for (uint32_t s = 0; s < strandCount; ++s)
    {
        const uint32_t n = vertexCounts[firstStrand + s];
        for (uint32_t seg = 0; seg + perSegment <= n; ++seg)
        {
            indices.push_back(static_cast<int>(pointCursor + seg));
        }
        pointCursor += n;
    }
    return indices;
}

/// Segments per strand when every strand in the set has the same count, else 0.
/// The sidecar computes this too; recomputing it here is what lets a backend
/// check the number it was handed rather than trust it.
inline uint32_t segmentsPerStrand(const std::vector<uint32_t>& vertexCounts,
                                  uint32_t firstStrand,
                                  uint32_t strandCount,
                                  bool isLinear)
{
    if (strandCount == 0u)
    {
        return 0u;
    }
    const uint32_t perSegment = controlPointsPerSegment(isLinear);
    const uint32_t n = vertexCounts[firstStrand];
    for (uint32_t s = 1; s < strandCount; ++s)
    {
        if (vertexCounts[firstStrand + s] != n)
        {
            return 0u;
        }
    }
    return (n >= perSegment) ? (n - perSegment + 1u) : 0u;
}
#endif // !__CUDACC__

} // namespace curve_layout
} // namespace oka
