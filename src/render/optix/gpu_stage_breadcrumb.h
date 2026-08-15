#pragma once

#include <cstddef>
#include <cstdint>

namespace oka
{
namespace optix
{

/// The GPU submissions one frame is made of, in the order they are enqueued.
///
/// CUDA reports a fault at the next synchronisation point, not at the call that
/// caused it, and a frame queues several things onto one stream before anything
/// waits. So "the launch failed" is what the host always sees, whichever of them
/// actually faulted -- and on a shared machine the difference between a bad
/// acceleration structure and a bad tonemap kernel is the difference between two
/// entirely separate investigations.
///
/// This is the CUDA analogue of `oka::metal::wavefront_stage_diagnostic`: the
/// GPU marks a stage *completed* by writing one byte into a device array, in
/// stream order, immediately after that stage's work. Whatever the host last
/// submitted is known already; what the marks add is how far the device actually
/// got, and the gap between the two is the suspect.
enum class GpuStage : uint32_t
{
    Skinning = 0,   ///< cuApplySkinning over the deforming meshes
    AccelBuild,     ///< optixAccelBuild / optixAccelCompact
    EnvCdf,         ///< the environment map's 2D CDF kernel
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
    /// Every stage the host submitted also completed. The error came from
    /// outside the frame -- an earlier asynchronous fault surfacing here, or a
    /// host-side call. Worth saying out loud, because the alternative is
    /// blaming whichever stage happened to be last.
    bool allSubmittedCompleted = false;
};

/// Decide which stage to blame.
///
/// `completed[i]` is non-zero when the device wrote stage `i`'s mark.
/// `submitted[i]` is non-zero when the host enqueued stage `i` this frame.
///
/// Both are needed, and the second is the one that is easy to leave out. A frame
/// with nothing skinned never submits the skinning stage, so its mark is absent
/// on a perfectly healthy device: an inference that read the marks alone would
/// blame skinning on every static scene, every time, and be believed.
///
/// The marks are written in stream order, so on a healthy device the completed
/// set is a prefix of the submitted one. A corrupt device can produce anything,
/// and this must not then name a stage that never ran, so the scan reports the
/// *first* submitted stage without a mark rather than the highest mark it finds.
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

} // namespace optix
} // namespace oka
