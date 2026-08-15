#pragma once

#include "optix_denoise_plan.h"

#include <optix.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace oka
{

/// The OptiX AI denoiser, standing in for MetalFX on this backend.
///
/// It owns its own layers -- colour, albedo, normal, flow, and the internal
/// guide layers a temporal model keeps between frames -- so that the renderer
/// hands it four device pointers and gets one back. Which model runs is decided
/// by `DenoisePlan`, not here: this class is the plumbing and the plan is the
/// policy, which is what makes the policy testable without a GPU.
///
/// The layer buffers are reallocated only when the plan's resolution or model
/// changes, because `optixDenoiserSetup` is not cheap and a render loop calls
/// this every frame.
class OptixDenoiserContext
{
public:
    OptixDenoiserContext() = default;
    ~OptixDenoiserContext();

    OptixDenoiserContext(const OptixDenoiserContext&) = delete;
    OptixDenoiserContext& operator=(const OptixDenoiserContext&) = delete;

    /// Build (or rebuild) for this plan. Returns false and leaves the object
    /// inactive if the denoiser could not be created -- the caller then shows the
    /// un-denoised image rather than nothing, which is the failure mode worth
    /// having.
    bool configure(OptixDeviceContext context, CUstream stream, const DenoisePlan& plan);

    /// Throw away the temporal history. The next frame is treated as the first
    /// of a sequence: nothing is reprojected onto it.
    void resetHistory()
    {
        mFirstFrame = true;
    }

    /// Run the network. All four inputs are at the plan's render resolution;
    /// `flow` may be null when there is no motion to report.
    bool denoise(CUstream stream, CUdeviceptr color, CUdeviceptr albedo, CUdeviceptr normal, CUdeviceptr flow);

    /// The denoised image, at the plan's *output* resolution. Zero until
    /// denoise() has run at least once.
    CUdeviceptr output() const
    {
        return mOutput;
    }

    bool active() const
    {
        return mDenoiser != nullptr;
    }

    bool hasOutput() const
    {
        return mDenoiser != nullptr && mOutput != 0 && mFramesDenoised > 0;
    }

    uint32_t outputWidth() const
    {
        return mPlan.outputWidth;
    }
    uint32_t outputHeight() const
    {
        return mPlan.outputHeight;
    }

    /// Bytes of device memory this object is holding.
    size_t deviceBytes() const
    {
        return mStateSize + mScratchSize + mOwnedImageBytes;
    }

    void release();

private:
    static OptixImage2D makeImage(CUdeviceptr data, uint32_t width, uint32_t height, OptixPixelFormat format);

    OptixDenoiser mDenoiser = nullptr;
    DenoisePlan mPlan{};

    CUdeviceptr mState = 0;
    size_t mStateSize = 0;
    CUdeviceptr mScratch = 0;
    size_t mScratchSize = 0;

    CUdeviceptr mOutput = 0;
    CUdeviceptr mPreviousOutput = 0;
    CUdeviceptr mInternalPrev = 0;
    CUdeviceptr mInternalNext = 0;
    size_t mInternalPixelBytes = 0;
    size_t mOwnedImageBytes = 0;

    bool mFirstFrame = true;
    uint64_t mFramesDenoised = 0;
};

} // namespace oka
