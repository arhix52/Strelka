#pragma once

#include <cstddef>
#include <cstdint>

namespace oka::optix
{

enum class GpuStage : uint32_t
{
    Skinning = 0,   ///< cuApplySkinning over the deforming meshes
    AccelBuild,     ///< optixAccelBuild / optixAccelCompact
    EnvCdf,         ///< environment alias-table upload; the enum name is historical
    ParamsUpload,   ///< the launch parameter block
    PathTrace,      ///< optixLaunch -- the raygen and everything it calls
    Tonemap,        ///< the post kernels writing the output image
    Count,
};

inline constexpr size_t kGpuStageCount = static_cast<size_t>(GpuStage::Count);

inline const char* gpuStageName(GpuStage stage)
{
    switch (stage)
    {
    case GpuStage::Skinning:
        return "skinning";
    case GpuStage::AccelBuild:
        return "acceleration structure build";
    case GpuStage::EnvCdf:
        return "environment CDF";
    case GpuStage::ParamsUpload:
        return "launch parameter upload";
    case GpuStage::PathTrace:
        return "path trace launch";
    case GpuStage::Tonemap:
        return "tonemap";
    case GpuStage::Count:
        break;
    }
    return "unknown";
}

/// What the marks say about where a frame died.
struct GpuStageFailure
{
    /// The last stage whose completion mark the device wrote, or -1 if none did.
    int32_t lastCompletedStage = -1;
    /// The stage that was running when it stopped, or -1 when nothing is
    /// implicated -- which happens when every submitted stage completed and the
    /// failure is therefore somewhere else entirely.
    int32_t suspectedStage = -1;
    bool allSubmittedCompleted = false;
};

inline GpuStageFailure inferGpuStageFailure(const uint8_t* completed, const uint8_t* submitted, size_t stageCount)
{
    GpuStageFailure result;
    if (completed == nullptr || submitted == nullptr || stageCount == 0)
    {
        return result;
    }

    for (size_t i = 0; i < stageCount; ++i)
    {
        if (submitted[i] == 0)
        {
            continue;
        }
        if (completed[i] == 0)
        {
            result.suspectedStage = static_cast<int32_t>(i);
            return result;
        }
        result.lastCompletedStage = static_cast<int32_t>(i);
    }

    // Nothing submitted is nothing to blame; that is not the same as everything
    // submitted having succeeded, and the caller has to be able to tell them
    // apart before it reports "the error came from outside this frame".
    result.allSubmittedCompleted = result.lastCompletedStage >= 0;
    return result;
}

} // namespace oka::optix

