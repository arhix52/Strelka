#pragma once

// Spelled relative to src/render because this header is included by host code
// as well as by the .cu, and only the device build has src/render/optix itself
// on the include path.
#include <optix/OptixRenderParams.h>

#include <stdint.h>

/// Spread the packed guide records into the separate images the OptiX denoiser
/// reads, and prepare the colour layer beside them.
///
/// One kernel so every format decision lives in one place, and so the shading
/// programs stay free of it: `__closesthit__radiance` is the most
/// register-pressured program in the pipeline and one packed buffer write costs
/// it far less than four.
///
/// `color` is linear radiance, already divided by the sample count -- the
/// denoiser works in the space the light arrived in, and exposure comes after
/// it. `fireflyClamp` is a luminance ceiling in *exposed* units, so the threshold
/// means the same thing at any exposure; zero turns it off.
extern "C" void resolveDenoiseGuides(const AovSample* aov,
                                     const float4* color,
                                     uint32_t width,
                                     uint32_t height,
                                     float3 exposure,
                                     float fireflyClamp,
                                     float4* outColor,
                                     float4* outAlbedo,
                                     float4* outNormal,
                                     float2* outFlow);

/// Copy an RGB image into a float4 buffer of the same size, alpha 1. Used to put
/// the denoised result back where the display and the EXR writer look for it.
extern "C" void copyDenoisedToImage(const float4* denoised, float4* image, uint32_t width, uint32_t height);

/// Point-sample a smaller image up to a larger one.
///
/// Only used when an upscaling plan was configured and the denoiser then refused
/// to run: the tracer has written a half-size image and the caller's buffer is
/// full size, so without this the frame is whatever was in that buffer before.
/// A blocky image of the right scene is a better failure than a black one.
extern "C" void upscalePointSample(
    const float4* src, uint32_t srcWidth, uint32_t srcHeight, float4* dst, uint32_t dstWidth, uint32_t dstHeight);
