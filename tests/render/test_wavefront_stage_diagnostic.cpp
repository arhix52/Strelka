#include <doctest/doctest.h>

#include "wavefront_stage_diagnostic.h"

using oka::metal::WavefrontStageFailure;
using oka::metal::inferWavefrontStageFailure;

TEST_CASE("wavefront stage diagnosis identifies the interrupted stage")
{
    const WavefrontStageFailure failure = inferWavefrontStageFailure(2, 4);

    CHECK(failure.lastCompletedStage == 1);
    CHECK(failure.suspectedStage == 2);
    CHECK_FALSE(failure.postIntegrator);
}

TEST_CASE("wavefront stage diagnosis recognizes post-integrator failure")
{
    const WavefrontStageFailure failure = inferWavefrontStageFailure(3, 3);

    CHECK(failure.lastCompletedStage == 2);
    CHECK(failure.suspectedStage == -1);
    CHECK(failure.postIntegrator);
}

TEST_CASE("wavefront stage diagnosis handles a missing opening breadcrumb")
{
    const WavefrontStageFailure missing =
        inferWavefrontStageFailure(oka::metal::kWavefrontStageNotStarted, 2);

    CHECK(missing.lastCompletedStage == -1);
    CHECK(missing.suspectedStage == 0);
    CHECK_FALSE(missing.postIntegrator);
}
