#include <doctest/doctest.h>

#include "wavefront_stage_diagnostic.h"

using oka::metal::WavefrontStageFailure;
using oka::metal::inferWavefrontStageFailure;

TEST_CASE("wavefront stage diagnosis identifies the interrupted stage")
{
    const WavefrontStageFailure failure = inferWavefrontStageFailure({ 100, 200, 300, 0, 0 }, 4);

    CHECK(failure.validMarks == 3);
    CHECK(failure.lastCompletedStage == 1);
    CHECK(failure.suspectedStage == 2);
    CHECK_FALSE(failure.postIntegrator);
}

TEST_CASE("wavefront stage diagnosis recognizes post-integrator failure")
{
    const WavefrontStageFailure failure = inferWavefrontStageFailure({ 100, 200, 300, 400 }, 3);

    CHECK(failure.validMarks == 4);
    CHECK(failure.lastCompletedStage == 2);
    CHECK(failure.suspectedStage == -1);
    CHECK(failure.postIntegrator);
}

TEST_CASE("wavefront stage diagnosis rejects missing and stale marks")
{
    const WavefrontStageFailure missing = inferWavefrontStageFailure({ 0, 0, 0 }, 2);
    const WavefrontStageFailure stale = inferWavefrontStageFailure({ 100, 200, 150, 400 }, 3);

    CHECK(missing.validMarks == 0);
    CHECK(missing.lastCompletedStage == -1);
    CHECK(missing.suspectedStage == -1);
    CHECK(stale.validMarks == 2);
    CHECK(stale.lastCompletedStage == 0);
    CHECK(stale.suspectedStage == 1);
}
