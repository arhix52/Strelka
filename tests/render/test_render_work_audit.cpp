#include <doctest/doctest.h>

#include <host/render_work_audit.h>

using oka::metal::CommandBufferAuditSample;
using oka::metal::RenderWorkInvariantSample;

TEST_CASE("render-work guides reuse the radiance traversal")
{
    RenderWorkInvariantSample off{ .primaryRays = 76800, .extendRays = 76800 };
    RenderWorkInvariantSample on = off;
    CHECK(oka::metal::guidesShareSurfaceTraversal(off, on));
    ++on.extendRays;
    CHECK_FALSE(oka::metal::guidesShareSurfaceTraversal(off, on));
}

TEST_CASE("render-work ReSTIR excludes first-bounce NEE")
{
    RenderWorkInvariantSample sample;
    CHECK(oka::metal::restirExcludesFirstBounceNee(true, sample));
    sample.firstBounceNee = 1;
    CHECK_FALSE(oka::metal::restirExcludesFirstBounceNee(true, sample));
}

TEST_CASE("render-work candidate and reuse stages do not trace")
{
    RenderWorkInvariantSample sample;
    CHECK(oka::metal::restirReuseDoesNotTrace(sample));
    sample.reuseQueries = 1;
    CHECK_FALSE(oka::metal::restirReuseDoesNotTrace(sample));
}

TEST_CASE("render-work final visibility is bounded by eligible hits")
{
    RenderWorkInvariantSample sample{ .eligibleHits = 10, .finalVisibilityRays = 10 };
    CHECK(oka::metal::finalVisibilityIsBounded(sample));
    ++sample.finalVisibilityRays;
    CHECK_FALSE(oka::metal::finalVisibilityIsBounded(sample));
}

TEST_CASE("render-work static frames do not update acceleration structures")
{
    RenderWorkInvariantSample sample;
    CHECK(oka::metal::staticAccelerationStructuresAreStable(sample));
    sample.tlasRefits = 1;
    CHECK_FALSE(oka::metal::staticAccelerationStructuresAreStable(sample));
}

TEST_CASE("render-work manual analytic-light tests do not scale with light count")
{
    RenderWorkInvariantSample one;
    RenderWorkInvariantSample many;
    CHECK(oka::metal::analyticLightWorkIsCountIndependent(one, many));
    many.manualAnalyticLightTests = 1;
    CHECK_FALSE(oka::metal::analyticLightWorkIsCountIndependent(one, many));
}

TEST_CASE("render-work ReSTIR dispatches follow configuration")
{
    RenderWorkInvariantSample sample{ .restirFusedDispatches = 1 };
    CHECK(oka::metal::restirDispatchesMatch(true, true, 2, sample));
    CHECK(oka::metal::restirDispatchesMatch(true, false, 0, sample));
}

TEST_CASE("render-work empty guide queues do not dispatch")
{
    RenderWorkInvariantSample sample;
    CHECK(oka::metal::guideDispatchesMatchActiveQueue(sample));
    sample.guideActiveItems = 1;
    sample.guideDispatches = 1;
    CHECK(oka::metal::guideDispatchesMatchActiveQueue(sample));
    sample.guideDispatches = 0;
    CHECK_FALSE(oka::metal::guideDispatchesMatchActiveQueue(sample));
}

TEST_CASE("render-work command buffers commit once")
{
    CommandBufferAuditSample sample{
        .creations = 1, .encoderCreations = 1, .dispatches = 1, .endEncodings = 1, .commits = 1, .waits = 1, .readbacks = 1
    };
    CHECK(oka::metal::commandBufferCommitsOnce(sample));
    ++sample.commits;
    CHECK_FALSE(oka::metal::commandBufferCommitsOnce(sample));
}

TEST_CASE("render-work moving lights update once per frame")
{
    RenderWorkInvariantSample sample{ .frames = 32,
                                      .maxTlasRefitsPerFrame = 1,
                                      .maxLightUploadsPerFrame = 1,
                                      .maxTemporalMappingsPerFrame = 1,
                                      .primaryDispatches = 32 };
    CHECK(oka::metal::movingLightFrameWorkIsBounded(sample));
    sample.maxTlasRefitsPerFrame = 2;
    CHECK_FALSE(oka::metal::movingLightFrameWorkIsBounded(sample));
}
