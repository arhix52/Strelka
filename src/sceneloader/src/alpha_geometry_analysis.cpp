#include <strelka/sceneloader/alpha_geometry_analysis.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>

namespace oka::sceneloader
{
namespace
{

enum class Coverage : uint8_t
{
    Transparent,
    Opaque,
    Unknown,
};

struct AxisIntervals
{
    std::array<std::array<uint32_t, 2>, 2> ranges{}; // inclusive
    uint32_t count = 0u;
};

int64_t positiveModulo(int64_t value, int64_t divisor)
{
    const int64_t remainder = value % divisor;
    return remainder < 0 ? remainder + divisor : remainder;
}

AxisIntervals repeatIntervals(int64_t lo, int64_t hi, uint32_t dimension)
{
    AxisIntervals result;
    if (dimension == 0u || hi < lo)
    {
        return result;
    }
    const uint64_t span = static_cast<uint64_t>(hi - lo) + 1u;
    if (span >= dimension)
    {
        result.ranges[0] = { 0u, dimension - 1u };
        result.count = 1u;
        return result;
    }

    const uint32_t first = static_cast<uint32_t>(positiveModulo(lo, dimension));
    const uint32_t last = static_cast<uint32_t>(positiveModulo(hi, dimension));
    if (first <= last)
    {
        result.ranges[0] = { first, last };
        result.count = 1u;
    }
    else
    {
        result.ranges[0] = { first, dimension - 1u };
        result.ranges[1] = { 0u, last };
        result.count = 2u;
    }
    return result;
}

class CoverageTable
{
public:
    CoverageTable(std::span<const uint8_t> alpha, uint32_t width, uint32_t height, const AlphaGeometryAnalysisOptions& options)
        : mWidth(width),
          mHeight(height),
          mFilter(options.filter),
          mStride(static_cast<size_t>(width) + 1u),
          mPossiblyVisible((static_cast<size_t>(height) + 1u) * mStride),
          mPossiblyTransparent((static_cast<size_t>(height) + 1u) * mStride)
    {
        const float factor = std::clamp(options.alphaFactor, 0.0f, 1.0f);
        const float epsilon = std::clamp(options.epsilon, 0.0f, 0.5f);
        const float cutoff = std::clamp(options.alphaCutoff, 0.0f, 1.0f);
        for (uint32_t y = 0u; y < height; ++y)
        {
            uint32_t visibleRow = 0u;
            uint32_t transparentRow = 0u;
            for (uint32_t x = 0u; x < width; ++x)
            {
                const float value = factor * (static_cast<float>(alpha[static_cast<size_t>(y) * width + x]) / 255.0f);
                const bool possiblyVisible = options.mode == AlphaGeometryMode::Mask ? value >= cutoff : value > epsilon;
                const bool possiblyTransparent =
                    options.mode == AlphaGeometryMode::Mask ? value < cutoff : value < 1.0f - epsilon;
                visibleRow += possiblyVisible ? 1u : 0u;
                transparentRow += possiblyTransparent ? 1u : 0u;
                const size_t here = (static_cast<size_t>(y) + 1u) * mStride + x + 1u;
                mPossiblyVisible[here] = mPossiblyVisible[here - mStride] + visibleRow;
                mPossiblyTransparent[here] = mPossiblyTransparent[here - mStride] + transparentRow;
            }
        }
    }

    Coverage classify(const AlphaUvTriangle& triangle) const
    {
        float minU = std::numeric_limits<float>::infinity();
        float minV = std::numeric_limits<float>::infinity();
        float maxU = -std::numeric_limits<float>::infinity();
        float maxV = -std::numeric_limits<float>::infinity();
        for (const glm::float2 uv : triangle.uv)
        {
            if (!std::isfinite(uv.x) || !std::isfinite(uv.y))
            {
                return Coverage::Unknown;
            }
            minU = std::min(minU, uv.x);
            minV = std::min(minV, uv.y);
            maxU = std::max(maxU, uv.x);
            maxV = std::max(maxV, uv.y);
        }

        // A normalized Metal linear sample maps u to u*width - 0.5. Include
        // both neighboring texels at both extrema. The rectangle is a
        // conservative superset of the triangle's true filter footprint.
        const AxisIntervals xs = supportIntervals(minU, maxU, mWidth);
        const AxisIntervals ys = supportIntervals(minV, maxV, mHeight);
        if (xs.count == 0u || ys.count == 0u)
        {
            return Coverage::Unknown;
        }
        if (!queryAny(mPossiblyVisible, xs, ys))
        {
            return Coverage::Transparent;
        }
        if (!queryAny(mPossiblyTransparent, xs, ys))
        {
            return Coverage::Opaque;
        }
        return Coverage::Unknown;
    }

private:
    AxisIntervals supportIntervals(float minimum, float maximum, uint32_t dimension) const
    {
        const double filterOffset = mFilter == AlphaGeometryFilter::Linear ? -0.5 : 0.0;
        const double scaledLo = static_cast<double>(minimum) * dimension + filterOffset;
        const double scaledHi = static_cast<double>(maximum) * dimension + filterOffset;
        constexpr double kSafeInteger = static_cast<double>(std::numeric_limits<int64_t>::max()) / 4.0;
        if (!std::isfinite(scaledLo) || !std::isfinite(scaledHi) || scaledLo < -kSafeInteger || scaledHi > kSafeInteger)
        {
            return repeatIntervals(0, static_cast<int64_t>(dimension) - 1, dimension);
        }
        const int64_t lo = static_cast<int64_t>(std::floor(scaledLo));
        const int64_t hi = static_cast<int64_t>(std::floor(scaledHi)) + (mFilter == AlphaGeometryFilter::Linear ? 1 : 0);
        return repeatIntervals(lo, hi, dimension);
    }

    uint32_t rectangleSum(const std::vector<uint32_t>& table, uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1) const
    {
        const size_t ax = x0;
        const size_t bx = static_cast<size_t>(x1) + 1u;
        const size_t ay = y0;
        const size_t by = static_cast<size_t>(y1) + 1u;
        return (table[by * mStride + bx] - table[ay * mStride + bx]) -
               (table[by * mStride + ax] - table[ay * mStride + ax]);
    }

    bool queryAny(const std::vector<uint32_t>& table, const AxisIntervals& xs, const AxisIntervals& ys) const
    {
        for (uint32_t yi = 0u; yi < ys.count; ++yi)
        {
            for (uint32_t xi = 0u; xi < xs.count; ++xi)
            {
                if (rectangleSum(table, xs.ranges[xi][0], xs.ranges[xi][1], ys.ranges[yi][0], ys.ranges[yi][1]) != 0u)
                {
                    return true;
                }
            }
        }
        return false;
    }

    uint32_t mWidth = 0u;
    uint32_t mHeight = 0u;
    AlphaGeometryFilter mFilter = AlphaGeometryFilter::Linear;
    size_t mStride = 0u;
    std::vector<uint32_t> mPossiblyVisible;
    std::vector<uint32_t> mPossiblyTransparent;
};

std::array<AlphaUvTriangle, 4> subdivide(const AlphaUvTriangle& triangle)
{
    const glm::float2 ab = (triangle.uv[0] + triangle.uv[1]) * 0.5f;
    const glm::float2 bc = (triangle.uv[1] + triangle.uv[2]) * 0.5f;
    const glm::float2 ca = (triangle.uv[2] + triangle.uv[0]) * 0.5f;
    return { AlphaUvTriangle{ { triangle.uv[0], ab, ca }, triangle.candidateWeight },
             AlphaUvTriangle{ { ab, triangle.uv[1], bc }, triangle.candidateWeight },
             AlphaUvTriangle{ { ca, bc, triangle.uv[2] }, triangle.candidateWeight },
             AlphaUvTriangle{ { ab, bc, ca }, triangle.candidateWeight } };
}

struct Frontier
{
    uint64_t transparentTriangles = 0u;
    uint64_t opaqueTriangles = 0u;
    uint64_t stoppedUnknownTriangles = 0u;
    uint64_t subdivisionsAccepted = 0u;
    uint64_t subdivisionsStoppedMixed = 0u;
    uint64_t subdivisionsStoppedByBudget = 0u;
    double transparentArea = 0.0;
    double opaqueArea = 0.0;
    double stoppedUnknownArea = 0.0;
    double transparentWeightedArea = 0.0;
    double opaqueWeightedArea = 0.0;
    double stoppedUnknownWeightedArea = 0.0;
    std::vector<AlphaUvTriangle> active;
};

AlphaGeometryLevelStats snapshot(const Frontier& frontier, uint32_t level, double activeArea)
{
    AlphaGeometryLevelStats stats;
    stats.subdivisionLevel = level;
    stats.transparentTriangles = frontier.transparentTriangles;
    stats.opaqueTriangles = frontier.opaqueTriangles;
    stats.unknownTriangles = frontier.stoppedUnknownTriangles + frontier.active.size();
    stats.survivingTriangles = stats.opaqueTriangles + stats.unknownTriangles;
    stats.subdivisionsAccepted = frontier.subdivisionsAccepted;
    stats.subdivisionsStoppedMixed = frontier.subdivisionsStoppedMixed;
    stats.subdivisionsStoppedByBudget = frontier.subdivisionsStoppedByBudget;
    stats.transparentSourceArea = frontier.transparentArea;
    stats.opaqueSourceArea = frontier.opaqueArea;
    stats.unknownSourceArea = frontier.stoppedUnknownArea + static_cast<double>(frontier.active.size()) * activeArea;
    stats.transparentWeightedArea = frontier.transparentWeightedArea;
    stats.opaqueWeightedArea = frontier.opaqueWeightedArea;
    stats.unknownWeightedArea = frontier.stoppedUnknownWeightedArea;
    for (const AlphaUvTriangle& triangle : frontier.active)
    {
        stats.unknownWeightedArea += static_cast<double>(triangle.candidateWeight) * activeArea;
    }
    return stats;
}

struct Decision
{
    std::array<Coverage, 4> coverage{};
    uint8_t survivingChildren = 0u;
    uint8_t unknownChildren = 0u;
    bool allUnknown = false;
    bool accepted = false;
    double benefitPerAddedTriangle = 0.0;
};

} // namespace

AlphaGeometryAnalysis analyzeAlphaGeometry(std::span<const AlphaUvTriangle> triangles,
                                           std::span<const uint8_t> alpha,
                                           uint32_t width,
                                           uint32_t height,
                                           const AlphaGeometryAnalysisOptions& options)
{
    const uint64_t pixelCount = static_cast<uint64_t>(width) * height;
    if (width == 0u || height == 0u || pixelCount > std::numeric_limits<uint32_t>::max() || alpha.size() != pixelCount)
    {
        throw std::invalid_argument("alpha image dimensions do not match its data");
    }
    if (!(options.growthLimit >= 1.0f) || !std::isfinite(options.growthLimit))
    {
        throw std::invalid_argument("alpha geometry growth limit must be finite and at least one");
    }
    if (options.maxSubdivisionLevel > 12u)
    {
        throw std::invalid_argument("alpha geometry subdivision level must be at most 12");
    }

    AlphaGeometryAnalysis analysis;
    analysis.sourceTriangles = triangles.size();
    for (const AlphaUvTriangle& triangle : triangles)
    {
        analysis.sourceWeight += static_cast<double>(triangle.candidateWeight);
    }
    analysis.levels.reserve(static_cast<size_t>(options.maxSubdivisionLevel) + 1u);
    if (triangles.empty())
    {
        for (uint32_t level = 0u; level <= options.maxSubdivisionLevel; ++level)
        {
            analysis.levels.push_back(AlphaGeometryLevelStats{ .subdivisionLevel = level });
        }
        return analysis;
    }

    const CoverageTable coverage(alpha, width, height, options);
    Frontier frontier;
    frontier.active.reserve(triangles.size());
    for (const AlphaUvTriangle& triangle : triangles)
    {
        switch (coverage.classify(triangle))
        {
        case Coverage::Transparent:
            ++frontier.transparentTriangles;
            frontier.transparentArea += 1.0;
            frontier.transparentWeightedArea += static_cast<double>(triangle.candidateWeight);
            break;
        case Coverage::Opaque:
            ++frontier.opaqueTriangles;
            frontier.opaqueArea += 1.0;
            frontier.opaqueWeightedArea += static_cast<double>(triangle.candidateWeight);
            break;
        case Coverage::Unknown:
            frontier.active.push_back(triangle);
            break;
        }
    }

    const double requestedSurviving = options.growthLimit * static_cast<double>(triangles.size());
    const double safeSurviving = std::min(requestedSurviving, static_cast<double>(std::numeric_limits<int64_t>::max()));
    const uint64_t maxSurviving = std::max<uint64_t>(triangles.size(), static_cast<uint64_t>(std::floor(safeSurviving)));
    double activeArea = 1.0;
    for (uint32_t level = 0u; level <= options.maxSubdivisionLevel; ++level)
    {
        analysis.levels.push_back(snapshot(frontier, level, activeArea));
        if (level == options.maxSubdivisionLevel || frontier.active.empty())
        {
            continue;
        }

        const double childArea = activeArea * 0.25;
        std::vector<Decision> decisions(frontier.active.size());
        std::vector<size_t> costly;
        costly.reserve(frontier.active.size());
        int64_t surviving =
            static_cast<int64_t>(frontier.opaqueTriangles + frontier.stoppedUnknownTriangles + frontier.active.size());
        for (size_t parent = 0u; parent < frontier.active.size(); ++parent)
        {
            Decision& decision = decisions[parent];
            const auto children = subdivide(frontier.active[parent]);
            for (uint32_t child = 0u; child < 4u; ++child)
            {
                decision.coverage[child] = coverage.classify(children[child]);
                decision.survivingChildren += decision.coverage[child] == Coverage::Transparent ? 0u : 1u;
                decision.unknownChildren += decision.coverage[child] == Coverage::Unknown ? 1u : 0u;
            }
            decision.allUnknown = decision.unknownChildren == 4u;
            if (decision.allUnknown && options.stopWhenAllChildrenUnknown)
            {
                continue;
            }
            const int32_t added = static_cast<int32_t>(decision.survivingChildren) - 1;
            const double benefit = static_cast<double>(4u - decision.unknownChildren) * childArea *
                                   static_cast<double>(frontier.active[parent].candidateWeight);
            decision.benefitPerAddedTriangle = added > 0 ? benefit / added : std::numeric_limits<double>::infinity();
            if (added <= 0)
            {
                decision.accepted = true;
                surviving += added;
            }
            else
            {
                costly.push_back(parent);
            }
        }

        std::ranges::sort(costly, [&](size_t a, size_t b) {
            if (decisions[a].benefitPerAddedTriangle != decisions[b].benefitPerAddedTriangle)
            {
                return decisions[a].benefitPerAddedTriangle > decisions[b].benefitPerAddedTriangle;
            }
            return a < b;
        });
        for (const size_t parent : costly)
        {
            const int32_t added = static_cast<int32_t>(decisions[parent].survivingChildren) - 1;
            if (surviving + added <= static_cast<int64_t>(maxSurviving))
            {
                decisions[parent].accepted = true;
                surviving += added;
            }
        }

        std::vector<AlphaUvTriangle> next;
        next.reserve(std::min<uint64_t>(maxSurviving, frontier.active.size() * 2u));
        for (size_t parent = 0u; parent < frontier.active.size(); ++parent)
        {
            const Decision& decision = decisions[parent];
            if ((decision.allUnknown && options.stopWhenAllChildrenUnknown) || !decision.accepted)
            {
                ++frontier.stoppedUnknownTriangles;
                frontier.stoppedUnknownArea += activeArea;
                frontier.stoppedUnknownWeightedArea +=
                    static_cast<double>(frontier.active[parent].candidateWeight) * activeArea;
                if (decision.allUnknown && options.stopWhenAllChildrenUnknown)
                {
                    ++frontier.subdivisionsStoppedMixed;
                }
                else
                {
                    ++frontier.subdivisionsStoppedByBudget;
                }
                continue;
            }

            ++frontier.subdivisionsAccepted;
            const auto children = subdivide(frontier.active[parent]);
            for (uint32_t child = 0u; child < 4u; ++child)
            {
                switch (decision.coverage[child])
                {
                case Coverage::Transparent:
                    ++frontier.transparentTriangles;
                    frontier.transparentArea += childArea;
                    frontier.transparentWeightedArea += static_cast<double>(children[child].candidateWeight) * childArea;
                    break;
                case Coverage::Opaque:
                    ++frontier.opaqueTriangles;
                    frontier.opaqueArea += childArea;
                    frontier.opaqueWeightedArea += static_cast<double>(children[child].candidateWeight) * childArea;
                    break;
                case Coverage::Unknown:
                    next.push_back(children[child]);
                    break;
                }
            }
        }
        frontier.active = std::move(next);
        activeArea = childArea;
    }
    return analysis;
}

} // namespace oka::sceneloader
