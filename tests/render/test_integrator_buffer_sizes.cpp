#include <doctest/doctest.h>

#include "integrator_buffer_sizes.h"

using oka::metal::kWavefrontControlUints;
using oka::metal::kWavefrontCurveTraversalBatchesPerGroup;
using oka::metal::kWavefrontCurveTraversalBatchThreads;
using oka::metal::kWavefrontMinDiagnosticTraversalBatchThreads;
using oka::metal::kWavefrontStageDiagnosticBase;
using oka::metal::kWavefrontStageDiagnosticBounces;
using oka::metal::kWavefrontStageDiagnosticStride;
using oka::metal::kWavefrontStageStatsUints;
using oka::metal::kWavefrontTraversalBatchThreads;
using oka::metal::kWavefrontTriangleTraversalBatchThreads;
using oka::metal::wavefrontBufferLayout;
using oka::metal::wavefrontTraversalBatchCount;
using oka::metal::WavefrontElementSizes;

TEST_CASE("wavefrontBufferLayout scales with pixel count")
{
    WavefrontElementSizes sz;
    sz.pathState = 64;
    sz.sharcPathState = 28;
    sz.pathRay = 32;
    sz.hitRecord = 48;
    sz.iorStack = 16;
    sz.radiance = 16;
    sz.shadowRay = 40;
    sz.aovSample = 80;

    const auto a = wavefrontBufferLayout(64, 48, sz);
    CHECK(a.pixels == 64u * 48u);
    CHECK(a.pathStateBytes == (size_t)a.pixels * 64);
    CHECK(a.sharcPathStateBytes == (size_t)a.pixels * 28);
    CHECK(a.pathRayBytes == (size_t)a.pixels * 32);
    CHECK(a.hitBytes == (size_t)a.pixels * 48);
    CHECK(a.iorStackBytes == (size_t)a.pixels * 16);
    CHECK(a.radianceBytes == (size_t)a.pixels * 16);
    CHECK(a.guideRadianceBytes == a.radianceBytes);
    CHECK(a.pathQueueBytes == (size_t)a.pixels * sizeof(uint32_t));
    CHECK(a.controlBytes == (size_t)kWavefrontControlUints * sizeof(uint32_t));
    CHECK(a.traversalDispatchBytes ==
          (size_t)wavefrontTraversalBatchCount(a.pixels,
                                               kWavefrontMinDiagnosticTraversalBatchThreads) *
              3 * sizeof(uint32_t));
    CHECK(a.stageStatsBytes == (size_t)kWavefrontStageStatsUints * sizeof(uint32_t));
    CHECK(kWavefrontStageStatsUints ==
          kWavefrontStageDiagnosticBase +
              kWavefrontStageDiagnosticBounces * kWavefrontStageDiagnosticStride);
    CHECK(a.shadowRayBytes == (size_t)a.pixels * 40);
    CHECK(a.aovBytes == (size_t)a.pixels * 80);
    CHECK(a.hitQueueBytes == a.pathQueueBytes);
    CHECK(a.missQueueBytes == a.pathQueueBytes);

    const auto b = wavefrontBufferLayout(128, 96, sz);
    CHECK(b.pixels == 4 * a.pixels);
    CHECK(b.pathStateBytes == 4 * a.pathStateBytes);
    CHECK(b.sharcPathStateBytes == 4 * a.sharcPathStateBytes);
    CHECK(b.controlBytes == a.controlBytes);
    CHECK(b.traversalDispatchBytes == 3 * a.traversalDispatchBytes);
    CHECK(b.stageStatsBytes == a.stageStatsBytes);
}

TEST_CASE("preview presets make wavefront memory growth explicit")
{
    WavefrontElementSizes sz;
    sz.pathState = 32;
    sz.sharcPathState = 28;
    sz.pathRay = 24;
    sz.hitRecord = 32;
    sz.iorStack = 52;
    sz.radiance = 16;
    sz.shadowRay = 48;
    sz.aovSample = 64;

    const auto preview = wavefrontBufferLayout(960, 540, sz);
    const auto fullHd = wavefrontBufferLayout(1920, 1080, sz);

    CHECK(preview.pixels == 518400);
    CHECK(fullHd.pixels == 4 * preview.pixels);
    CHECK(fullHd.pathStateBytes == 4 * preview.pathStateBytes);
    CHECK(fullHd.sharcPathStateBytes == 4 * preview.sharcPathStateBytes);
    CHECK(fullHd.pathStateBytes + fullHd.sharcPathStateBytes == (size_t)fullHd.pixels * 60);
    CHECK(fullHd.shadowRayBytes == 4 * preview.shadowRayBytes);
    CHECK(fullHd.aovBytes == 4 * preview.aovBytes);
    CHECK(fullHd.controlBytes == preview.controlBytes);
    CHECK(preview.traversalDispatchBytes == 127 * 3 * sizeof(uint32_t));
    CHECK(fullHd.traversalDispatchBytes == 507 * 3 * sizeof(uint32_t));
    CHECK(fullHd.stageStatsBytes == preview.stageStatsBytes);
}

TEST_CASE("wavefront traversal batches cap one hardware dispatch")
{
    CHECK(wavefrontTraversalBatchCount(0) == 1);
    CHECK(wavefrontTraversalBatchCount(kWavefrontTraversalBatchThreads) == 1);
    CHECK(wavefrontTraversalBatchCount(kWavefrontTraversalBatchThreads + 1) == 2);
    CHECK(wavefrontTraversalBatchCount(kWavefrontTriangleTraversalBatchThreads,
                                       kWavefrontTriangleTraversalBatchThreads) == 1);
    CHECK(wavefrontTraversalBatchCount(kWavefrontTriangleTraversalBatchThreads + 1,
                                       kWavefrontTriangleTraversalBatchThreads) == 2);
    CHECK(wavefrontTraversalBatchCount(1920 * 1080,
                                       kWavefrontTriangleTraversalBatchThreads) == 4);
    CHECK(wavefrontTraversalBatchCount(1440 * 810) == 5);
    CHECK(wavefrontTraversalBatchCount(1920 * 1080) == 8);
    CHECK(wavefrontTraversalBatchCount(1920 * 1080,
                                       kWavefrontCurveTraversalBatchThreads) == 16);
    CHECK(kWavefrontCurveTraversalBatchThreads *
              kWavefrontCurveTraversalBatchesPerGroup ==
          256u * 1024u);
}
