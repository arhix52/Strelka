#include "DenoiseGuides.h"

#include <optix_denoise_plan.h>

#include <sutil/vec_math.h>

__global__ void resolveDenoiseGuidesKernel(const AovSample* __restrict__ aov,
                                           const float4* __restrict__ color,
                                           uint32_t width,
                                           uint32_t height,
                                           float3 exposure,
                                           float fireflyClamp,
                                           float4* __restrict__ outColor,
                                           float4* __restrict__ outAlbedo,
                                           float4* __restrict__ outNormal,
                                           float2* __restrict__ outFlow)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height)
    {
        return;
    }
    const AovSample a = aov[i];

    float3 c = make_float3(color[i]);
    const float3 exposed = c * exposure;
    const float lum = dot(exposed, make_float3(0.2126f, 0.7152f, 0.0722f));
    c *= oka::guides::fireflyScale(lum, fireflyClamp);
    // A NaN reaching the network poisons a whole tile of the output, and a
    // single bad sample is exactly the thing a denoiser is being asked to hide.
    if (isnan(c.x) || isnan(c.y) || isnan(c.z))
    {
        c = make_float3(0.0f);
    }
    outColor[i] = make_float4(c, 1.0f);

    // One albedo layer, unlike MetalFX's separate diffuse and specular inputs.
    // The sum is what the network demodulates against, and it is bounded by
    // construction: the two lobes are complementary weights of the same base
    // colour, so their sum cannot exceed it.
    const float3 albedo = a.diffuseAlbedo + a.specularAlbedo;
    outAlbedo[i] = make_float4(clamp(albedo, make_float3(0.0f), make_float3(1.0f)), 1.0f);

    // World space, which is what every model but the two deprecated ones wants.
    outNormal[i] = make_float4(a.normal, 0.0f);
    outFlow[i] = make_float2(a.motionX, a.motionY);
}

__global__ void copyDenoisedToImageKernel(const float4* __restrict__ denoised,
                                          float4* __restrict__ image,
                                          uint32_t width,
                                          uint32_t height)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height)
    {
        return;
    }
    const float4 d = denoised[i];
    image[i] = make_float4(d.x, d.y, d.z, 1.0f);
}

extern "C" void resolveDenoiseGuides(const AovSample* aov,
                                     const float4* color,
                                     uint32_t width,
                                     uint32_t height,
                                     float3 exposure,
                                     float fireflyClamp,
                                     float4* outColor,
                                     float4* outAlbedo,
                                     float4* outNormal,
                                     float2* outFlow)
{
    const uint32_t pixels = width * height;
    const dim3 blockSize(256, 1, 1);
    const dim3 gridSize((pixels + 255) / 256, 1, 1);
    resolveDenoiseGuidesKernel<<<gridSize, blockSize, 0>>>(
        aov, color, width, height, exposure, fireflyClamp, outColor, outAlbedo, outNormal, outFlow);
}

extern "C" void copyDenoisedToImage(const float4* denoised, float4* image, uint32_t width, uint32_t height)
{
    const uint32_t pixels = width * height;
    const dim3 blockSize(256, 1, 1);
    const dim3 gridSize((pixels + 255) / 256, 1, 1);
    copyDenoisedToImageKernel<<<gridSize, blockSize, 0>>>(denoised, image, width, height);
}

__global__ void upscalePointSampleKernel(const float4* __restrict__ src,
                                         uint32_t srcWidth,
                                         uint32_t srcHeight,
                                         float4* __restrict__ dst,
                                         uint32_t dstWidth,
                                         uint32_t dstHeight)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dstWidth * dstHeight)
    {
        return;
    }
    const uint32_t x = i % dstWidth;
    const uint32_t y = i / dstWidth;
    const uint32_t sx = min(x * srcWidth / dstWidth, srcWidth - 1u);
    const uint32_t sy = min(y * srcHeight / dstHeight, srcHeight - 1u);
    dst[i] = src[sy * srcWidth + sx];
}

extern "C" void upscalePointSample(
    const float4* src, uint32_t srcWidth, uint32_t srcHeight, float4* dst, uint32_t dstWidth, uint32_t dstHeight)
{
    if (srcWidth == 0 || srcHeight == 0 || dstWidth == 0 || dstHeight == 0)
    {
        return;
    }
    const uint32_t pixels = dstWidth * dstHeight;
    const dim3 blockSize(256, 1, 1);
    const dim3 gridSize((pixels + 255) / 256, 1, 1);
    upscalePointSampleKernel<<<gridSize, blockSize, 0>>>(src, srcWidth, srcHeight, dst, dstWidth, dstHeight);
}
