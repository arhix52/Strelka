#pragma once

#include <algorithm>
#include <cstdint>


namespace oka::render_resolution
{

struct Resolution
{
    uint32_t outputWidth = 1;
    uint32_t outputHeight = 1;
    uint32_t pathTraceWidth = 1;
    uint32_t pathTraceHeight = 1;
    float appliedScale = 1.0f;
    bool upscaling = false;
};

struct DenoiserPolicy
{
    bool useDenoiser = false;
    bool useSpatialFallback = false;
    float lowestSupportedScale = 1.0f;
};

inline Resolution resolve(uint32_t outputWidth, uint32_t outputHeight, bool enableUpscale, float requestedScale)
{
    Resolution result;
    result.outputWidth = std::max(1u, outputWidth);
    result.outputHeight = std::max(1u, outputHeight);
    result.appliedScale = enableUpscale ? std::clamp(requestedScale, 0.25f, 1.0f) : 1.0f;
    result.pathTraceWidth =
        std::max(1u, static_cast<uint32_t>(static_cast<float>(result.outputWidth) * result.appliedScale));
    result.pathTraceHeight =
        std::max(1u, static_cast<uint32_t>(static_cast<float>(result.outputHeight) * result.appliedScale));
    result.upscaling =
        result.pathTraceWidth != result.outputWidth || result.pathTraceHeight != result.outputHeight;
    return result;
}

inline DenoiserPolicy resolveDenoiserPolicy(bool wantDenoise,
                                             const Resolution& resolution,
                                             float maximumScaleFactor)
{
    DenoiserPolicy result;
    result.lowestSupportedScale = maximumScaleFactor > 0.0f ? 1.0f / maximumScaleFactor : 0.5f;
    if (!wantDenoise)
    {
        return result;
    }
    const float horizontalScale =
        static_cast<float>(resolution.pathTraceWidth) / static_cast<float>(resolution.outputWidth);
    const float verticalScale =
        static_cast<float>(resolution.pathTraceHeight) / static_cast<float>(resolution.outputHeight);
    if (resolution.upscaling &&
        (horizontalScale < result.lowestSupportedScale || verticalScale < result.lowestSupportedScale))
    {
        result.useSpatialFallback = true;
        return result;
    }
    result.useDenoiser = true;
    return result;
}

} // namespace oka::render_resolution

