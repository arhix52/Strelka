#pragma once

#include <strelka/scene/glm_wrapper.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace oka::sceneloader
{

enum class AlphaGeometryMode : uint8_t
{
    Mask,
    Blend,
};

enum class AlphaGeometryFilter : uint8_t
{
    Linear,
    Nearest,
};

struct AlphaUvTriangle
{
    std::array<glm::float2, 3> uv{};
    // Number of traversal candidates this unique triangle represents. The
    // geometry budget remains unweighted, while the greedy allocator prefers
    // resolving geometry reused by more instances.
    uint64_t candidateWeight = 1u;
};

struct AlphaGeometryAnalysisOptions
{
    AlphaGeometryMode mode = AlphaGeometryMode::Blend;
    AlphaGeometryFilter filter = AlphaGeometryFilter::Linear;
    uint32_t maxSubdivisionLevel = 3u;
    float alphaFactor = 1.0f;
    float alphaCutoff = 0.5f;
    // Zero is an exact BLEND decomposition. A caller may deliberately admit a
    // small bias (normally one 8-bit step) to discard effectively empty texels.
    float epsilon = 0.0f;
    // Maximum opaque + unknown triangles relative to the source. A useful
    // subdivision that removes enough transparent children can have zero or
    // negative cost and is therefore still accepted at the limit.
    float growthLimit = 3.0f;
    // The production heuristic avoids spending four triangles when the first
    // split resolves nothing. Turning it off is useful as an offline upper
    // bound: it shows whether another level could recover enough coverage to
    // justify a contour-based implementation.
    bool stopWhenAllChildrenUnknown = true;
};

struct AlphaGeometryLevelStats
{
    uint32_t subdivisionLevel = 0u;
    uint64_t transparentTriangles = 0u;
    uint64_t opaqueTriangles = 0u;
    uint64_t unknownTriangles = 0u;
    uint64_t survivingTriangles = 0u;
    uint64_t subdivisionsAccepted = 0u;
    uint64_t subdivisionsStoppedMixed = 0u;
    uint64_t subdivisionsStoppedByBudget = 0u;

    // A level-L child covers 4^-L of its source triangle. These sums therefore
    // remain comparable when the adaptive frontier contains several levels.
    double transparentSourceArea = 0.0;
    double opaqueSourceArea = 0.0;
    double unknownSourceArea = 0.0;
    double transparentWeightedArea = 0.0;
    double opaqueWeightedArea = 0.0;
    double unknownWeightedArea = 0.0;
};

struct AlphaGeometryAnalysis
{
    uint64_t sourceTriangles = 0u;
    double sourceWeight = 0.0;
    std::vector<AlphaGeometryLevelStats> levels;
};

// Conservatively classify transformed UV triangles against the alpha values
// sampled by the renderer. `alpha` is row-major level-zero UNORM alpha. The
// query includes every texel that the selected nearest/linear filter can touch
// and applies repeat addressing, so a transparent/opaque answer is safe for
// every point in the triangle; an axis-aligned UV bound can only turn a
// resolvable triangle into unknown, never the reverse.
AlphaGeometryAnalysis analyzeAlphaGeometry(std::span<const AlphaUvTriangle> triangles,
                                           std::span<const uint8_t> alpha,
                                           uint32_t width,
                                           uint32_t height,
                                           const AlphaGeometryAnalysisOptions& options);

} // namespace oka::sceneloader
