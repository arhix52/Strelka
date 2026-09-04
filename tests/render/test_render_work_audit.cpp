#include <doctest/doctest.h>

#include <host/render_work_audit.h>

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
    RenderWorkInvariantSample sample{ .restirSpatialDispatches = 1, .restirFinalDispatches = 1 };
    CHECK(oka::metal::restirDispatchesMatch(true, true, 2, sample));
    sample.restirSpatialDispatches = 0;
    CHECK(oka::metal::restirDispatchesMatch(true, false, 0, sample));
}
