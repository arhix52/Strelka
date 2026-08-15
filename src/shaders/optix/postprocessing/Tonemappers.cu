#include "Tonemappers.h"

#include <sutil/Matrix.h>
#include <sutil/vec_math_adv.h>

// The bounds test in every kernel below is `>=` and used to be `>`, so the one
// thread at index width*height wrote a pixel past the end of the image on every
// tonemap of every frame. It is a 16-byte overrun of a cudaMalloc'd buffer, which
// is exactly the kind of thing that does no visible damage for years and then
// corrupts whatever the allocator happened to put next.

__device__ __inline__ float calcLuminance(const float3 color)
{
    return dot(color, make_float3(0.299f, 0.587f, 0.114f));
}

__device__ __inline__ float3 reinhard(const float3 color)
{
    float luminance = calcLuminance(color);
    return color / (luminance + 1);
}

__global__ void tonemapReinhard(float4* image, const float3 exposure, uint32_t width, uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearPixelIndex >= height * width)
    {
        return;
    }
    const float3 radiance = make_float3(image[linearPixelIndex]) * exposure;
    image[linearPixelIndex] = make_float4(reinhard(radiance), 1.0f);
    return;
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
__device__ __inline__ float3 ACESFilm(const float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

__global__ void tonemapACESFilm(float4* image, const float3 exposure, uint32_t width, uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearPixelIndex >= height * width)
    {
        return;
    }
    const float3 radiance = make_float3(image[linearPixelIndex]) * exposure;
    image[linearPixelIndex] = make_float4(ACESFilm(radiance), 1.0f);
    return;
}

__device__ __inline__ float3 RRTAndODTFit(float3 v)
{
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

__device__ __inline__ float3 ACESFitted(float3 color)
{
    // https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    const sutil::Matrix3x3 ACESInputMat =
    {
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777
    };

    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    const sutil::Matrix3x3 ACESOutputMat =
    {
        1.60475, -0.53108, -0.07367,
        -0.10208,  1.10813, -0.00605,
        -0.00327, -0.07276,  1.07602
    };

    color = ACESInputMat * color;
    // Apply RRT and ODT
    color = RRTAndODTFit(color);
    color = ACESOutputMat * color;
    // Clamp to [0, 1]
    color = saturate(color);
    return color;
}

__global__ void tonemapACESFitted(float4* image, const float3 exposure, uint32_t width, uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearPixelIndex >= height * width)
    {
        return;
    }
    const float3 radiance = make_float3(image[linearPixelIndex]) * exposure;
    image[linearPixelIndex] = make_float4(ACESFitted(radiance), 1.0f);
    return;
}

/// The sRGB transfer function, one channel.
///
/// This used to be a bare `pow(c, 1/gamma)`, which is not what the other two
/// implementations of the same step do: `srgbGamma` in src/shaders/common is
/// what Metal's tonemapper and StrelkaCLI's PNG writer both call, and it has a
/// linear segment below 0.0031308 and the 1.055/-0.055 scale above it. The
/// difference is largest exactly where a display image spends most of its
/// pixels: at 0.05 linear the two answer 0.246 and 0.287.
///
/// A copy rather than an include. The shared header declares its functions
/// without a `__device__` qualifier and pulls in glm on anything that is not
/// Metal, so nvcc cannot call them from a kernel; making it callable means a
/// qualifier macro on every function in a header that also compiles into the
/// Metal backend, and this machine has no Mac to prove that change harmless on.
/// The numbers are pinned by tests/render/test_tonemappers.cpp on the host side.
__device__ __inline__ float srgbGammaChannel(const float c, const float gamma)
{
    if (isnan(c) || c < 0.0f)
    {
        return 0.0f;
    }
    if (c < 0.0031308f)
    {
        return 12.92f * c;
    }
    return 1.055f * powf(c, 1.0f / gamma) - 0.055f;
}

__global__ void gammaCorrection(const float gamma, float4* image, uint32_t width, uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearPixelIndex >= height * width)
    {
        return;
    }
    const float3 color = make_float3(image[linearPixelIndex]);
    image[linearPixelIndex] = make_float4(srgbGammaChannel(color.x, gamma), srgbGammaChannel(color.y, gamma),
                                          srgbGammaChannel(color.z, gamma), 1.0f);
    return;
}

void tonemap(const ToneMapperType type, const float3 exposure, const float gamma, float4* image, const uint32_t width, const uint32_t height)
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((width * height + 255) / 256, 1, 1);
    switch (type)
    {
    case ToneMapperType::eReinhard:
        tonemapReinhard<<<gridSize, blockSize, 0>>>(image, exposure, width, height);
        break;
    case ToneMapperType::eACES:
        tonemapACESFitted<<<gridSize, blockSize, 0>>>(image, exposure, width, height);
        break;
    case ToneMapperType::eFilmic:
        tonemapACESFilm<<<gridSize, blockSize, 0>>>(image, exposure, width, height);
        break;
    case ToneMapperType::eNone:
        break;
    default:
        break;
    }
    if (gamma > 0.0f)
    {
        gammaCorrection<<<gridSize, blockSize, 0>>>(gamma, image, width, height);
    }
}
