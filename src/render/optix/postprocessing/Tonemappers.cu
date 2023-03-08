#include "Tonemappers.h"

#include <sutil/Matrix.h>
#include <sutil/vec_math_adv.h>

__device__ __inline__ float calcLuminance(const float3 color)
{
    return dot(color, make_float3(0.299f, 0.587f, 0.114f));
}

__device__ __inline__ float3 reinhard(const float3 color)
{
    float luminance = calcLuminance(color);
    float reinhard = luminance / (luminance + 1);
    return color * (reinhard / luminance);
}

__global__ void tonemapReinhard(float4* image, uint32_t width, uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearPixelIndex > height * width)
    {
        return;
    }
    const float3 radiance = make_float3(image[linearPixelIndex]);
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

__global__ void tonemapACESFilm(float4* image, uint32_t width, uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearPixelIndex > height * width)
    {
        return;
    }
    const float3 radiance = make_float3(image[linearPixelIndex]);
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

__global__ void tonemapACESFitted(float4* image, uint32_t width, uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearPixelIndex > height * width)
    {
        return;
    }
    const float3 radiance = make_float3(image[linearPixelIndex]);
    image[linearPixelIndex] = make_float4(ACESFitted(radiance), 1.0f);
    return;
}

__global__ void gammaCorrection(const float gamma, float4* image, uint32_t width, uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearPixelIndex > height * width)
    {
        return;
    }
    const float3 color = make_float3(image[linearPixelIndex]);
    image[linearPixelIndex] = make_float4(powf(color, (1.0f / gamma)), 1.0f);
    return;
}

void tonemap(const ToneMapperType type, const float gamma, float4* image, const uint32_t width, const uint32_t height)
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((width * height + 255) / 256, 1, 1);
    switch (type)
    {
    case ToneMapperType::eReinhard:
        tonemapReinhard<<<gridSize, blockSize, 0>>>(image, width, height);
        break;
    case ToneMapperType::eACES:
        tonemapACESFitted<<<gridSize, blockSize, 0>>>(image, width, height);
        break;
    case ToneMapperType::eFilmic:
        tonemapACESFilm<<<gridSize, blockSize, 0>>>(image, width, height);
        break;
    case ToneMapperType::eNone:
    default:
        break;
    }
    if (gamma > 0.0f)
    {
        gammaCorrection<<<gridSize, blockSize, 0>>>(gamma, image, width, height);
    }
}
