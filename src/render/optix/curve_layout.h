#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#    define OKA_CURVE_FN __host__ __device__ inline
#else
#    define OKA_CURVE_FN inline
#endif

#if !defined(__CUDACC__)
#    include <vector>
#endif

namespace oka::curve_layout
{

/// Control points one segment spans. A round linear segment is a cylinder
/// between two points; a cubic B-spline segment is defined by four, and
/// consecutive segments overlap in three of them.
OKA_CURVE_FN uint32_t controlPointsPerSegment(bool isLinear)
{
    return isLinear ? 2u : 4u;
}

OKA_CURVE_FN float strandCoordinate(uint32_t primitiveIndex, uint32_t segmentsPerStrand, float u)
{
    if (segmentsPerStrand == 0u)
    {
        return 0.0f;
    }
    return ((float)(primitiveIndex % segmentsPerStrand) + u) / (float)segmentsPerStrand;
}

#if !defined(__CUDACC__)
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

} // namespace oka::curve_layout

