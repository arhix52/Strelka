#include <doctest/doctest.h>

#include <host/integrator_buffer_sizes.h>

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
using oka::metal::WavefrontElementSizes;
using oka::metal::wavefrontTraversalBatchCount;

TEST_CASE("wavefrontBufferLayout scales with pixel count")
{
    WavefrontElementSizes sz;
    sz.pathState = 64;
    sz.mediumPathState = 8;
    sz.sharcUpdateState = 168;
    sz.pathRay = 32;
    sz.hitRecord = 48;
    sz.iorStack = 16;
    sz.radiance = 16;
    sz.guideRay = 32;
    sz.shadowRay = 40;
    sz.aovSample = 80;
    sz.restirReservoir = 32;
    sz.restirSurfaceHistory = 20;
    sz.restirShadingPoint = 104;

    const auto a = wavefrontBufferLayout(64, 48, sz);
    CHECK(a.pixels == 64u * 48u);
    CHECK(a.pathStateBytes == (size_t)a.pixels * 64);
    CHECK(a.mediumPathStateBytes == (size_t)a.pixels * 8);
    CHECK(a.sharcUpdateStateBytes == (size_t)a.pixels * 168);
    CHECK(a.pathRayBytes == (size_t)a.pixels * 32);
    CHECK(a.hitBytes == (size_t)a.pixels * 48);
    CHECK(a.iorStackBytes == (size_t)a.pixels * 16);
    CHECK(a.radianceBytes == (size_t)a.pixels * 16);
    CHECK(a.guideRayBytes == (size_t)a.pixels * 32);
    CHECK(a.guideQueueBytes == (size_t)a.pixels * sizeof(uint32_t));
    CHECK(a.pathQueueBytes == (size_t)a.pixels * sizeof(uint32_t));
    CHECK(a.controlBytes == (size_t)kWavefrontControlUints * sizeof(uint32_t));
    CHECK(a.traversalDispatchBytes ==
          (size_t)wavefrontTraversalBatchCount(a.pixels, kWavefrontMinDiagnosticTraversalBatchThreads) * 3 *
              sizeof(uint32_t));
    CHECK(a.stageStatsBytes == (size_t)kWavefrontStageStatsUints * sizeof(uint32_t));
    CHECK(kWavefrontStageStatsUints ==
          kWavefrontStageDiagnosticBase + kWavefrontStageDiagnosticBounces * kWavefrontStageDiagnosticStride);
    CHECK(a.shadowRayBytes == (size_t)a.pixels * 40);
    CHECK(a.aovBytes == (size_t)a.pixels * 80);
    CHECK(a.hitQueueBytes == a.pathQueueBytes);
    CHECK(a.missQueueBytes == a.pathQueueBytes);
    CHECK(a.restirReservoirBytes == (size_t)a.pixels * 32);
    CHECK(a.restirSurfaceHistoryBytes == (size_t)a.pixels * 20);
    CHECK(a.restirShadingPointBytes == (size_t)a.pixels * 104);
    CHECK(2 * a.restirReservoirBytes + 2 * a.restirSurfaceHistoryBytes + a.restirShadingPointBytes ==
          (size_t)a.pixels * 208);
    CHECK(2 * a.restirReservoirBytes + 2 * a.restirSurfaceHistoryBytes + 2 * a.restirShadingPointBytes ==
          (size_t)a.pixels * 312);

    const auto nee = wavefrontBufferLayout(64, 48, sz, 1, false);
    CHECK(nee.restirReservoirBytes == 0);
    CHECK(nee.restirSurfaceHistoryBytes == 0);
    CHECK(nee.restirShadingPointBytes == 0);

    const auto b = wavefrontBufferLayout(128, 96, sz);
    CHECK(b.pixels == 4 * a.pixels);
    CHECK(b.pathStateBytes == 4 * a.pathStateBytes);
    CHECK(b.mediumPathStateBytes == 4 * a.mediumPathStateBytes);
    CHECK(b.sharcUpdateStateBytes == 4 * a.sharcUpdateStateBytes);
    CHECK(b.controlBytes == a.controlBytes);
    CHECK(b.traversalDispatchBytes == 3 * a.traversalDispatchBytes);
    CHECK(b.stageStatsBytes == a.stageStatsBytes);
}

TEST_CASE("preview presets make wavefront memory growth explicit")
{
    WavefrontElementSizes sz;
    sz.pathState = 24;
    sz.mediumPathState = 8;
    sz.sharcUpdateState = 168;
    sz.pathRay = 24;
    sz.hitRecord = 32;
    sz.iorStack = 52;
    sz.radiance = 16;
    sz.guideRay = 32;
    sz.shadowRay = 48;
    sz.aovSample = 64;

    const auto preview = wavefrontBufferLayout(960, 540, sz);
    const auto fullHd = wavefrontBufferLayout(1920, 1080, sz);

    CHECK(preview.pixels == 518400);
    CHECK(fullHd.pixels == 4 * preview.pixels);
    CHECK(fullHd.pathStateBytes == 4 * preview.pathStateBytes);
    CHECK(fullHd.mediumPathStateBytes == 4 * preview.mediumPathStateBytes);
    CHECK(fullHd.sharcUpdateStateBytes == 4 * preview.sharcUpdateStateBytes);
    CHECK(fullHd.pathStateBytes + fullHd.mediumPathStateBytes + fullHd.sharcUpdateStateBytes ==
          (size_t)fullHd.pixels * 200);
    CHECK(fullHd.shadowRayBytes == 4 * preview.shadowRayBytes);
    CHECK(fullHd.aovBytes == 4 * preview.aovBytes);
    CHECK(fullHd.controlBytes == preview.controlBytes);
    CHECK(preview.traversalDispatchBytes == static_cast<size_t>(127) * 3 * sizeof(uint32_t));
    CHECK(fullHd.traversalDispatchBytes == static_cast<size_t>(507) * 3 * sizeof(uint32_t));
    CHECK(fullHd.stageStatsBytes == preview.stageStatsBytes);
}

TEST_CASE("SHARC update state follows the sparse update grid")
{
    WavefrontElementSizes sz;
    sz.sharcUpdateState = 168;

    const auto layout = wavefrontBufferLayout(1920, 1080, sz, 5);
    CHECK(layout.sharcUpdatePaths == 384u * 216u);
    CHECK(layout.sharcUpdateStateBytes == (size_t)layout.sharcUpdatePaths * 168);
}

TEST_CASE("wavefront traversal batches cap one hardware dispatch")
{
    CHECK(wavefrontTraversalBatchCount(0) == 1);
    CHECK(wavefrontTraversalBatchCount(kWavefrontTraversalBatchThreads) == 1);
    CHECK(wavefrontTraversalBatchCount(kWavefrontTraversalBatchThreads + 1) == 2);
    CHECK(wavefrontTraversalBatchCount(
              kWavefrontTriangleTraversalBatchThreads, kWavefrontTriangleTraversalBatchThreads) == 1);
    CHECK(wavefrontTraversalBatchCount(
              kWavefrontTriangleTraversalBatchThreads + 1, kWavefrontTriangleTraversalBatchThreads) == 2);
    CHECK(wavefrontTraversalBatchCount(1920 * 1080, kWavefrontTriangleTraversalBatchThreads) == 4);
    CHECK(wavefrontTraversalBatchCount(1440 * 810) == 5);
    CHECK(wavefrontTraversalBatchCount(1920 * 1080) == 8);
    CHECK(wavefrontTraversalBatchCount(1920 * 1080, kWavefrontCurveTraversalBatchThreads) == 16);
    CHECK(kWavefrontCurveTraversalBatchThreads * kWavefrontCurveTraversalBatchesPerGroup == 256u * 1024u);
}
