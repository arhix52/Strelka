#include <doctest/doctest.h>

#include "editor_frame_budget.h"

using namespace oka::editor_frame_budget;

TEST_CASE("frame budget prediction requires a completed timing sample")
{
    const RenderSettingsSnapshot settings{ 960, 540, false, 1.0f };

    CHECK_FALSE(assess(sampleFrom(0.0, settings), settings).hasPrediction);
    CHECK_FALSE(predictGpuTimeMs({ 100.0, 0 }, 1000) > 0.0);
}

TEST_CASE("frame budget prediction scales with PT pixel count")
{
    const RenderSettingsSnapshot current{ 960, 540, false, 1.0f };
    const RenderSettingsSnapshot proposed{ 1920, 1080, false, 1.0f };
    const Assessment assessment = assess(sampleFrom(150.0, current), proposed);

    CHECK(assessment.pathTraceWidth == 1920);
    CHECK(assessment.pathTraceHeight == 1080);
    CHECK(assessment.predictedGpuTimeMs == doctest::Approx(600.0));
    CHECK(assessment.exceedsBudget);
}

TEST_CASE("frame budget uses internal PT rather than preview resolution")
{
    const RenderSettingsSnapshot current{ 1280, 720, false, 1.0f };
    const RenderSettingsSnapshot proposed{ 1920, 1080, true, 0.5f };
    const Assessment assessment = assess(sampleFrom(400.0, current), proposed);

    CHECK(assessment.pathTraceWidth == 960);
    CHECK(assessment.pathTraceHeight == 540);
    CHECK(assessment.predictedGpuTimeMs == doctest::Approx(225.0));
    CHECK_FALSE(assessment.exceedsBudget);
}

TEST_CASE("exact frame budget is accepted")
{
    const RenderSettingsSnapshot settings{ 960, 540, false, 1.0f };
    const Assessment assessment = assess(sampleFrom(kInteractiveBudgetMs, settings), settings);

    CHECK(assessment.predictedGpuTimeMs == doctest::Approx(kInteractiveBudgetMs));
    CHECK_FALSE(assessment.exceedsBudget);
}

TEST_CASE("frame budget follows renderer truncation for odd sizes")
{
    const RenderSettingsSnapshot current{ 200, 150, false, 1.0f };
    const RenderSettingsSnapshot proposed{ 801, 601, true, 0.25f };
    const Assessment assessment = assess(sampleFrom(100.0, current), proposed);

    CHECK(assessment.pathTraceWidth == 200);
    CHECK(assessment.pathTraceHeight == 150);
    CHECK(assessment.predictedGpuTimeMs == doctest::Approx(100.0));
}

TEST_CASE("recommended scale chooses the highest tier inside budget")
{
    const RenderSettingsSnapshot current{ 1920, 1080, false, 1.0f };
    const FrameSample moderate = sampleFrom(800.0, current);
    const FrameSample heavy = sampleFrom(4000.0, current);

    CHECK(recommendedScale(moderate, 1920, 1080) == doctest::Approx(0.75f));
    CHECK(recommendedScale(heavy, 1920, 1080) == doctest::Approx(0.25f));
}
