#include <doctest/doctest.h>

#include <strelka/sceneloader/alpha_geometry_analysis.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace oka::sceneloader;

namespace
{

AlphaUvTriangle triangle(glm::float2 a, glm::float2 b, glm::float2 c)
{
    return AlphaUvTriangle{ { a, b, c } };
}

void checkAreaPartition(const AlphaGeometryAnalysis& analysis)
{
    for (const AlphaGeometryLevelStats& level : analysis.levels)
    {
        CHECK(level.transparentSourceArea + level.opaqueSourceArea + level.unknownSourceArea ==
              doctest::Approx(static_cast<double>(analysis.sourceTriangles)));
        CHECK(level.transparentWeightedArea + level.opaqueWeightedArea + level.unknownWeightedArea ==
              doctest::Approx(analysis.sourceWeight));
    }
}

} // namespace

TEST_CASE("uniform alpha geometry is removed or made opaque without subdivision")
{
    const std::vector<AlphaUvTriangle> triangles = {
        triangle({ 0.1f, 0.1f }, { 0.4f, 0.1f }, { 0.1f, 0.4f }),
    };
    AlphaGeometryAnalysisOptions options;
    options.maxSubdivisionLevel = 3u;

    std::vector<uint8_t> alpha(16u, 0u);
    AlphaGeometryAnalysis analysis = analyzeAlphaGeometry(triangles, alpha, 4u, 4u, options);
    CHECK(analysis.levels[0].transparentTriangles == 1u);
    CHECK(analysis.levels[0].survivingTriangles == 0u);
    checkAreaPartition(analysis);

    std::ranges::fill(alpha, 255u);
    analysis = analyzeAlphaGeometry(triangles, alpha, 4u, 4u, options);
    CHECK(analysis.levels[0].opaqueTriangles == 1u);
    CHECK(analysis.levels[0].survivingTriangles == 1u);
    CHECK(analysis.levels.back().subdivisionsAccepted == 0u);
    checkAreaPartition(analysis);
}

TEST_CASE("mask classification uses the renderer cutoff inequality")
{
    const std::vector<AlphaUvTriangle> triangles = {
        triangle({ 0.2f, 0.2f }, { 0.3f, 0.2f }, { 0.2f, 0.3f }),
    };
    const std::vector<uint8_t> alpha(16u, 128u);
    AlphaGeometryAnalysisOptions options;
    options.mode = AlphaGeometryMode::Mask;
    options.maxSubdivisionLevel = 0u;
    options.alphaCutoff = 0.5f;
    CHECK(analyzeAlphaGeometry(triangles, alpha, 4u, 4u, options).levels[0].opaqueTriangles == 1u);

    options.alphaCutoff = 0.51f;
    CHECK(analyzeAlphaGeometry(triangles, alpha, 4u, 4u, options).levels[0].transparentTriangles == 1u);
}

TEST_CASE("candidate weights affect benefit but not the geometry budget")
{
    std::vector<uint8_t> alpha(size_t{ 16 } * 16u, 0u);
    for (uint32_t y = 0u; y < 16u; ++y)
    {
        for (uint32_t x = 12u; x < 16u; ++x)
        {
            alpha[y * 16u + x] = 255u;
        }
    }
    std::vector<AlphaUvTriangle> triangles = {
        triangle({ 0.05f, 0.1f }, { 0.95f, 0.1f }, { 0.05f, 0.9f }),
        triangle({ 0.05f, 0.1f }, { 0.95f, 0.1f }, { 0.05f, 0.9f }),
    };
    triangles[0].candidateWeight = 100u;
    triangles[1].candidateWeight = 1u;
    AlphaGeometryAnalysisOptions options;
    options.maxSubdivisionLevel = 1u;
    options.growthLimit = 1.0f;
    const AlphaGeometryAnalysis analysis = analyzeAlphaGeometry(triangles, alpha, 16u, 16u, options);
    CHECK(analysis.sourceWeight == doctest::Approx(101.0));
    CHECK(analysis.levels.back().survivingTriangles <= 2u);
    CHECK(analysis.levels.back().unknownWeightedArea < analysis.sourceWeight);
    checkAreaPartition(analysis);
}

TEST_CASE("repeat and bilinear support keep a wrap-edge triangle unknown")
{
    const std::vector<AlphaUvTriangle> triangles = {
        triangle({ 0.99f, 0.2f }, { 1.01f, 0.2f }, { 1.0f, 0.3f }),
    };
    const std::vector<uint8_t> alpha = { 0u, 255u, 0u, 255u };
    AlphaGeometryAnalysisOptions options;
    options.maxSubdivisionLevel = 0u;
    const AlphaGeometryAnalysis analysis = analyzeAlphaGeometry(triangles, alpha, 2u, 2u, options);
    CHECK(analysis.levels[0].unknownTriangles == 1u);
    CHECK(analysis.levels[0].transparentTriangles == 0u);
    CHECK(analysis.levels[0].opaqueTriangles == 0u);
}

TEST_CASE("nearest filtering excludes the linear neighbor texel")
{
    const std::vector<AlphaUvTriangle> triangles = {
        triangle({ 0.01f, 0.01f }, { 0.20f, 0.01f }, { 0.01f, 0.20f }),
    };
    const std::vector<uint8_t> alpha = {
        0u, 255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u,
    };
    AlphaGeometryAnalysisOptions options;
    options.maxSubdivisionLevel = 0u;
    options.filter = AlphaGeometryFilter::Nearest;
    CHECK(analyzeAlphaGeometry(triangles, alpha, 4u, 4u, options).levels[0].transparentTriangles == 1u);

    options.filter = AlphaGeometryFilter::Linear;
    CHECK(analyzeAlphaGeometry(triangles, alpha, 4u, 4u, options).levels[0].unknownTriangles == 1u);
}

TEST_CASE("adaptive subdivision resolves uniform children within the growth budget")
{
    std::vector<uint8_t> alpha(size_t{ 16 } * 16u, 0u);
    for (uint32_t y = 0u; y < 16u; ++y)
    {
        for (uint32_t x = 12u; x < 16u; ++x)
        {
            alpha[y * 16u + x] = 255u;
        }
    }
    const std::vector<AlphaUvTriangle> triangles = {
        triangle({ 0.05f, 0.1f }, { 0.95f, 0.1f }, { 0.05f, 0.9f }),
    };
    AlphaGeometryAnalysisOptions options;
    options.maxSubdivisionLevel = 3u;
    options.growthLimit = 3.0f;
    const AlphaGeometryAnalysis analysis = analyzeAlphaGeometry(triangles, alpha, 16u, 16u, options);
    CHECK(analysis.levels.back().subdivisionsAccepted > 0u);
    for (const AlphaGeometryLevelStats& level : analysis.levels)
    {
        CHECK(level.survivingTriangles <= 3u);
    }
    CHECK(analysis.levels.back().unknownSourceArea < analysis.levels.front().unknownSourceArea);
    checkAreaPartition(analysis);
}

TEST_CASE("four unresolved children stop subdivision immediately")
{
    std::vector<uint8_t> alpha(size_t{ 16 } * 16u);
    for (uint32_t y = 0u; y < 16u; ++y)
    {
        for (uint32_t x = 0u; x < 16u; ++x)
        {
            alpha[y * 16u + x] = ((x + y) & 1u) != 0u ? 255u : 0u;
        }
    }
    const std::vector<AlphaUvTriangle> triangles = {
        triangle({ 0.05f, 0.05f }, { 0.95f, 0.05f }, { 0.05f, 0.95f }),
    };
    AlphaGeometryAnalysisOptions options;
    options.maxSubdivisionLevel = 3u;
    const AlphaGeometryAnalysis analysis = analyzeAlphaGeometry(triangles, alpha, 16u, 16u, options);
    CHECK(analysis.levels[1].subdivisionsStoppedMixed == 1u);
    CHECK(analysis.levels[1].unknownTriangles == 1u);
    CHECK(analysis.levels.back().subdivisionsAccepted == 0u);
    checkAreaPartition(analysis);

    options.stopWhenAllChildrenUnknown = false;
    options.growthLimit = 4.0f;
    const AlphaGeometryAnalysis diagnostic = analyzeAlphaGeometry(triangles, alpha, 16u, 16u, options);
    CHECK(diagnostic.levels[1].subdivisionsStoppedMixed == 0u);
    CHECK(diagnostic.levels[1].subdivisionsAccepted == 1u);
    CHECK(diagnostic.levels[1].unknownTriangles == 4u);
    checkAreaPartition(diagnostic);
}
