#include "OptixDenoiser.h"

#include "cuda_checks.h"

#include <optix_stubs.h>

#include <log.h>

namespace oka
{

namespace
{

/// Free and null in one step. Every buffer here is optional and reconfiguration
/// runs on a live renderer, so the "already null" case is the common one.
void freeDevice(CUdeviceptr& ptr)
{
    if (ptr != 0)
    {
        cudaFree(reinterpret_cast<void*>(ptr));
        ptr = 0;
    }
}

bool allocDevice(CUdeviceptr& ptr, size_t bytes)
{
    void* raw = nullptr;
    if (cudaMalloc(&raw, bytes) != cudaSuccess)
    {
        return false;
    }
    ptr = reinterpret_cast<CUdeviceptr>(raw);
    return true;
}

OptixDenoiserModelKind toOptixModelKind(DenoiseModelKind kind)
{
    switch (kind)
    {
    case DenoiseModelKind::eTemporalAov:
        return OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV;
    case DenoiseModelKind::eUpscale2x:
        return OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
    case DenoiseModelKind::eTemporalUpscale2x:
        return OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X;
    case DenoiseModelKind::eAov:
    case DenoiseModelKind::eNone:
    default:
        return OPTIX_DENOISER_MODEL_KIND_AOV;
    }
}

bool samePlan(const DenoisePlan& a, const DenoisePlan& b)
{
    return a.kind == b.kind && a.renderWidth == b.renderWidth && a.renderHeight == b.renderHeight &&
           a.outputWidth == b.outputWidth && a.outputHeight == b.outputHeight;
}

} // namespace

OptixDenoiserContext::~OptixDenoiserContext()
{
    release();
}

OptixImage2D OptixDenoiserContext::makeImage(CUdeviceptr data, uint32_t width, uint32_t height, OptixPixelFormat format)
{
    unsigned int pixelStride = 0;
    switch (format)
    {
    case OPTIX_PIXEL_FORMAT_FLOAT1:
        pixelStride = sizeof(float);
        break;
    case OPTIX_PIXEL_FORMAT_FLOAT2:
        pixelStride = 2 * sizeof(float);
        break;
    case OPTIX_PIXEL_FORMAT_FLOAT3:
        pixelStride = 3 * sizeof(float);
        break;
    case OPTIX_PIXEL_FORMAT_FLOAT4:
    default:
        pixelStride = 4 * sizeof(float);
        break;
    }
    OptixImage2D image = {};
    image.data = data;
    image.width = width;
    image.height = height;
    image.pixelStrideInBytes = pixelStride;
    image.rowStrideInBytes = width * pixelStride;
    image.format = format;
    return image;
}

void OptixDenoiserContext::release()
{
    if (mDenoiser)
    {
        optixDenoiserDestroy(mDenoiser);
        mDenoiser = nullptr;
    }
    freeDevice(mState);
    freeDevice(mScratch);
    freeDevice(mOutput);
    freeDevice(mPreviousOutput);
    freeDevice(mInternalPrev);
    freeDevice(mInternalNext);
    mStateSize = 0;
    mScratchSize = 0;
    mOwnedImageBytes = 0;
    mInternalPixelBytes = 0;
    mFirstFrame = true;
    mFramesDenoised = 0;
    mPlan = DenoisePlan{};
}

bool OptixDenoiserContext::configure(OptixDeviceContext context, CUstream stream, const DenoisePlan& plan)
{
    if (!plan.enabled())
    {
        if (mDenoiser)
        {
            release();
        }
        return false;
    }
    if (mDenoiser && samePlan(mPlan, plan))
    {
        return true;
    }

    release();
    mPlan = plan;

    OptixDenoiserOptions options = {};
    options.guideAlbedo = 1u;
    options.guideNormal = 1u;
    // Alpha is a coverage mask here, not a noisy estimate: every path writes 1.
    // Denoising it would only blur the edges of an image that has none.
    options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

    if (optixDenoiserCreate(context, toOptixModelKind(plan.kind), &options, &mDenoiser) != OPTIX_SUCCESS)
    {
        STRELKA_ERROR("OptiX denoiser: optixDenoiserCreate failed; rendering without it");
        mDenoiser = nullptr;
        return false;
    }

    OptixDenoiserSizes sizes = {};
    if (optixDenoiserComputeMemoryResources(mDenoiser, plan.renderWidth, plan.renderHeight, &sizes) != OPTIX_SUCCESS)
    {
        STRELKA_ERROR("OptiX denoiser: could not size its own memory; rendering without it");
        release();
        return false;
    }

    // No tiling. A tile boundary is visible in a denoised image unless the tiles
    // overlap, and at the resolutions this renders at the whole frame fits.
    mScratchSize = sizes.withoutOverlapScratchSizeInBytes;
    mStateSize = sizes.stateSizeInBytes;
    if (!allocDevice(mScratch, mScratchSize) || !allocDevice(mState, mStateSize))
    {
        STRELKA_ERROR("OptiX denoiser: out of device memory for state/scratch; rendering without it");
        release();
        return false;
    }

    const DenoiseBufferLayout layout = denoiseBufferLayout(plan, 0);
    if (!allocDevice(mOutput, layout.denoisedBytes))
    {
        STRELKA_ERROR("OptiX denoiser: out of device memory for the output image; rendering without it");
        release();
        return false;
    }
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(mOutput), 0, layout.denoisedBytes));
    mOwnedImageBytes = layout.denoisedBytes;

    if (plan.temporal)
    {
        if (!allocDevice(mPreviousOutput, layout.denoisedBytes))
        {
            STRELKA_ERROR("OptiX denoiser: out of device memory for the temporal history; rendering without it");
            release();
            return false;
        }
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(mPreviousOutput), 0, layout.denoisedBytes));
        mOwnedImageBytes += layout.denoisedBytes;

        mInternalPixelBytes = sizes.internalGuideLayerPixelSizeInBytes;
        const size_t internalBytes =
            static_cast<size_t>(plan.outputWidth) * plan.outputHeight * mInternalPixelBytes;
        if (!allocDevice(mInternalPrev, internalBytes) || !allocDevice(mInternalNext, internalBytes))
        {
            STRELKA_ERROR("OptiX denoiser: out of device memory for the internal guide layer; rendering without it");
            release();
            return false;
        }
        // Zeroed for the first frame, which is what "no previous layers" means.
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(mInternalPrev), 0, internalBytes));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(mInternalNext), 0, internalBytes));
        mOwnedImageBytes += 2 * internalBytes;
    }

    if (optixDenoiserSetup(mDenoiser, stream, plan.renderWidth, plan.renderHeight, mState, mStateSize, mScratch,
                           mScratchSize) != OPTIX_SUCCESS)
    {
        STRELKA_ERROR("OptiX denoiser: optixDenoiserSetup failed; rendering without it");
        release();
        return false;
    }

    mFirstFrame = true;
    mFramesDenoised = 0;
    STRELKA_INFO("OptiX denoiser: {}x{} -> {}x{}, model {}{}", plan.renderWidth, plan.renderHeight, plan.outputWidth,
                 plan.outputHeight, plan.temporal ? "temporal " : "", plan.upscale ? "upscale2x" : "aov");
    return true;
}

bool OptixDenoiserContext::denoise(CUstream stream,
                                   CUdeviceptr color,
                                   CUdeviceptr albedo,
                                   CUdeviceptr normal,
                                   CUdeviceptr flow,
                                   CUdeviceptr flowTrust)
{
    if (!mDenoiser || color == 0)
    {
        return false;
    }

    const uint32_t w = mPlan.renderWidth;
    const uint32_t h = mPlan.renderHeight;

    OptixDenoiserGuideLayer guideLayer = {};
    guideLayer.albedo = makeImage(albedo, w, h, OPTIX_PIXEL_FORMAT_FLOAT4);
    guideLayer.normal = makeImage(normal, w, h, OPTIX_PIXEL_FORMAT_FLOAT4);
    if (flow != 0)
    {
        guideLayer.flow = makeImage(flow, w, h, OPTIX_PIXEL_FORMAT_FLOAT2);
        // Where the motion vector is known to be a lie -- a mirror, a pane of
        // glass, anything whose guides describe a surface the camera cannot see
        // directly -- the reactive mask says so, and this is the input that acts
        // on it. Only meaningful alongside a flow layer.
        if (flowTrust != 0)
        {
            guideLayer.flowTrustworthiness = makeImage(flowTrust, w, h, OPTIX_PIXEL_FORMAT_FLOAT1);
        }
    }
    if (mPlan.temporal)
    {
        guideLayer.previousOutputInternalGuideLayer =
            makeImage(mInternalPrev, mPlan.outputWidth, mPlan.outputHeight, OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER);
        guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes =
            static_cast<unsigned int>(mInternalPixelBytes);
        guideLayer.previousOutputInternalGuideLayer.rowStrideInBytes =
            mPlan.outputWidth * static_cast<unsigned int>(mInternalPixelBytes);
        guideLayer.outputInternalGuideLayer = guideLayer.previousOutputInternalGuideLayer;
        guideLayer.outputInternalGuideLayer.data = mInternalNext;
    }

    OptixDenoiserLayer layer = {};
    layer.input = makeImage(color, w, h, OPTIX_PIXEL_FORMAT_FLOAT4);
    layer.output = makeImage(mOutput, mPlan.outputWidth, mPlan.outputHeight, OPTIX_PIXEL_FORMAT_FLOAT4);
    layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;
    if (mPlan.temporal)
    {
        // The first frame of a sequence has no denoised predecessor. The SDK
        // sample seeds it with the noisy input, which is a better starting point
        // than black; the upscaling models size their previous output
        // differently and are given nothing instead.
        if (mFirstFrame && !mPlan.upscale)
        {
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(mPreviousOutput), reinterpret_cast<const void*>(color),
                                       static_cast<size_t>(w) * h * 4 * sizeof(float), cudaMemcpyDeviceToDevice,
                                       stream));
        }
        layer.previousOutput =
            makeImage(mPreviousOutput, mPlan.outputWidth, mPlan.outputHeight, OPTIX_PIXEL_FORMAT_FLOAT4);
    }

    OptixDenoiserParams params = {};
    params.temporalModeUsePreviousLayers = (mPlan.temporal && !mFirstFrame) ? 1u : 0u;
    // Motion vectors are already in pixels of the render resolution, which is
    // the unit the flow layer is defined in, so no rescale.
    params.flowMulX = 1.0f;
    params.flowMulY = 1.0f;
    // Null hdrIntensity / hdrAverageColor: the denoiser computes its own, which
    // is what an accumulating render wants -- the exposure of the image changes
    // as it converges, and pinning it to a value measured on the first frame
    // would make the network's idea of "bright" wrong for every frame after.

    const OptixResult result = optixDenoiserInvoke(mDenoiser, stream, &params, mState, mStateSize, &guideLayer, &layer,
                                                   1u, 0u, 0u, mScratch, mScratchSize);
    if (result != OPTIX_SUCCESS)
    {
        STRELKA_ERROR("OptiX denoiser: optixDenoiserInvoke failed ({}); rendering without it",
                      static_cast<int>(result));
        release();
        return false;
    }

    if (mPlan.temporal)
    {
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(mPreviousOutput), reinterpret_cast<const void*>(mOutput),
                                   static_cast<size_t>(mPlan.outputWidth) * mPlan.outputHeight * 4 * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        CUdeviceptr tmp = mInternalPrev;
        mInternalPrev = mInternalNext;
        mInternalNext = tmp;
    }

    mFirstFrame = false;
    ++mFramesDenoised;
    return true;
}

} // namespace oka
