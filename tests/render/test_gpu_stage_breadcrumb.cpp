#include <doctest/doctest.h>

#include "gpu_stage_breadcrumb.h"

#include <string>

using oka::optix::GpuStage;
using oka::optix::GpuStageFailure;
using oka::optix::inferGpuStageFailure;
using oka::optix::kGpuStageCount;

namespace
{
// A frame's worth of marks, built the way the renderer builds them: everything
// up to and including `lastCompleted` finished, and everything up to and
// including `lastSubmitted` was enqueued.
struct Frame
{
    uint8_t completed[kGpuStageCount] = {};
    uint8_t submitted[kGpuStageCount] = {};

    void submit(GpuStage stage)
    {
        submitted[static_cast<size_t>(stage)] = 1;
    }
    void complete(GpuStage stage)
    {
        completed[static_cast<size_t>(stage)] = 1;
    }
    GpuStageFailure infer() const
    {
        return inferGpuStageFailure(completed, submitted, kGpuStageCount);
    }
};
} // namespace

TEST_CASE("the stage with no mark is the one that faulted")
{
    Frame f;
    f.submit(GpuStage::ParamsUpload);
    f.submit(GpuStage::PathTrace);
    f.submit(GpuStage::Tonemap);
    f.complete(GpuStage::ParamsUpload);
    // The launch never wrote its mark, so the stream died inside it.

    const GpuStageFailure failure = f.infer();
    CHECK(failure.suspectedStage == static_cast<int32_t>(GpuStage::PathTrace));
    CHECK(failure.lastCompletedStage == static_cast<int32_t>(GpuStage::ParamsUpload));
    CHECK_FALSE(failure.allSubmittedCompleted);
}

TEST_CASE("a stage that was never submitted is never blamed")
{
    Frame f;
    // No skinning and no acceleration build this frame: a static scene, steady
    // state, only the per-frame work.
    f.submit(GpuStage::ParamsUpload);
    f.submit(GpuStage::PathTrace);
    f.complete(GpuStage::ParamsUpload);
    f.complete(GpuStage::PathTrace);

    const GpuStageFailure failure = f.infer();
    CHECK(failure.suspectedStage == -1);
    CHECK(failure.lastCompletedStage == static_cast<int32_t>(GpuStage::PathTrace));
    CHECK(failure.allSubmittedCompleted);
}

TEST_CASE("a fault before anything completed names the first submitted stage")
{
    Frame f;
    f.submit(GpuStage::AccelBuild);
    f.submit(GpuStage::ParamsUpload);
    f.submit(GpuStage::PathTrace);

    const GpuStageFailure failure = f.infer();
    CHECK(failure.suspectedStage == static_cast<int32_t>(GpuStage::AccelBuild));
    CHECK(failure.lastCompletedStage == -1);
    CHECK_FALSE(failure.allSubmittedCompleted);
}

TEST_CASE("marks above the gap do not move the blame past it")
{
    Frame f;
    f.submit(GpuStage::ParamsUpload);
    f.submit(GpuStage::PathTrace);
    f.submit(GpuStage::Tonemap);
    f.complete(GpuStage::ParamsUpload);
    f.complete(GpuStage::Tonemap); // corrupt: the launch it follows never finished

    const GpuStageFailure failure = f.infer();
    CHECK(failure.suspectedStage == static_cast<int32_t>(GpuStage::PathTrace));
    CHECK(failure.lastCompletedStage == static_cast<int32_t>(GpuStage::ParamsUpload));
}

// "Nothing was submitted" and "everything submitted succeeded" are different
// facts, and only the second one licenses the renderer to say the error came
// from outside this frame.
TEST_CASE("an empty frame implicates nothing and exonerates nothing")
{
    const Frame f;
    const GpuStageFailure failure = f.infer();
    CHECK(failure.suspectedStage == -1);
    CHECK(failure.lastCompletedStage == -1);
    CHECK_FALSE(failure.allSubmittedCompleted);
}

TEST_CASE("a null or empty mark buffer is not a diagnosis")
{
    uint8_t marks[kGpuStageCount] = {};
    CHECK(inferGpuStageFailure(nullptr, marks, kGpuStageCount).suspectedStage == -1);
    CHECK(inferGpuStageFailure(marks, nullptr, kGpuStageCount).suspectedStage == -1);
    CHECK(inferGpuStageFailure(marks, marks, 0).suspectedStage == -1);
}

TEST_CASE("every stage has a name")
{
    for (size_t i = 0; i < kGpuStageCount; ++i)
    {
        const char* name = oka::optix::gpuStageName(static_cast<GpuStage>(i));
        REQUIRE(name != nullptr);
        CHECK(std::string(name) != "unknown");
    }
}
