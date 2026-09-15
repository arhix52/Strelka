#pragma once

// Spelled relative to src/render because this header is included by host code
// as well as by the .cu, and only the device build has src/render/optix itself
// on the include path.
#include <optix/OptixRenderParams.h>

#include <stdint.h>

extern "C" void resolveDenoiseGuides(const AovSample* aov,
                                     const float4* color,
                                     uint32_t width,
                                     uint32_t height,
                                     float3 exposure,
                                     float fireflyClamp,
                                     float4* outColor,
                                     float4* outAlbedo,
                                     float4* outNormal,
                                     float2* outFlow,
                                     float* outFlowTrust);

/// Copy an RGB image into a float4 buffer of the same size, alpha 1. Used to put
/// the denoised result back where the display and the EXR writer look for it.
extern "C" void copyDenoisedToImage(const float4* denoised, float4* image, uint32_t width, uint32_t height);

extern "C" void upscalePointSample(
    const float4* src, uint32_t srcWidth, uint32_t srcHeight, float4* dst, uint32_t dstWidth, uint32_t dstHeight);
