#include "Tonemappers.h"

#include <cuda_fp16.h>
#include <sutil/Matrix.h>
#include <sutil/vec_math_adv.h>

// Bounds checks use >= so the width*height thread cannot write past the image.

__device__ __inline__ float calcLuminance(const float3 color)
{
    return dot(color, make_float3(0.299f, 0.587f, 0.114f));
}

__device__ __inline__ float3 reinhard(const float3 color)
{
    float luminance = calcLuminance(color);
    return color / (luminance + 1);
}

__global__ void tonemapReinhard(float4 *image,
                                const float3 exposure,
                                const float maxOutput,
                                uint32_t width,
                                uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float3 radiance;
    float output;

    if (linearPixelIndex >= height * width)
    {
        return;
    }
    output = fmaxf(maxOutput, 1.0f);
    radiance = make_float3(image[linearPixelIndex]) * exposure / output;
    image[linearPixelIndex] = make_float4(reinhard(radiance) * output, 1.0f);
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

__global__ void tonemapACESFilm(float4 *image,
                               const float3 exposure,
                               const float maxOutput,
                               uint32_t width,
                               uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float3 radiance;
    float output;

    if (linearPixelIndex >= height * width)
    {
        return;
    }
    output = fmaxf(maxOutput, 1.0f);
    radiance = make_float3(image[linearPixelIndex]) * exposure / output;
    image[linearPixelIndex] = make_float4(ACESFilm(radiance) * output, 1.0f);
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

__global__ void tonemapACESFitted(float4 *image,
                                 const float3 exposure,
                                 const float maxOutput,
                                 uint32_t width,
                                 uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float3 radiance;
    float output;

    if (linearPixelIndex >= height * width)
    {
        return;
    }
    output = fmaxf(maxOutput, 1.0f);
    radiance = make_float3(image[linearPixelIndex]) * exposure / output;
    image[linearPixelIndex] = make_float4(ACESFitted(radiance) * output, 1.0f);
    return;
}

__global__ void applyExposure(float4 *image, const float3 exposure, uint32_t width, uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float3 radiance;

    if (linearPixelIndex >= height * width)
    {
        return;
    }
    radiance = make_float3(image[linearPixelIndex]) * exposure;
    image[linearPixelIndex] = make_float4(radiance, 1.0f);
}

__device__ __inline__ float3 applyLinearPresentation(
    const float3 color,
    const oka::PresentationMetadata metadata)
{
    float3 exposure;
    float3 exposed;
    float output;
    ToneMapperType type;

    if (metadata.content == oka::PresentationContent::DebugDisplayLinear)
    {
        return color;
    }

    exposure = make_float3(metadata.exposure[0], metadata.exposure[1], metadata.exposure[2]);
    exposed = color * exposure;
    output = fmaxf(metadata.maxOutput, 1.0f);
    type = static_cast<ToneMapperType>(metadata.tonemapper);
    switch (type)
    {
    case ToneMapperType::eReinhard:
        return reinhard(exposed / output) * output;
    case ToneMapperType::eACES:
        return ACESFitted(exposed / output) * output;
    case ToneMapperType::eFilmic:
        return ACESFilm(exposed / output) * output;
    case ToneMapperType::eNone:
        return exposed;
    default:
        return exposed;
    }
}

__global__ void tonemapToSurfaceLinearKernel(
    const float4 *source,
    cudaSurfaceObject_t destination,
    uint32_t width,
    uint32_t height,
    oka::PresentationMetadata metadata)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t linearPixelIndex;
    float3 color;
    ushort4 encoded;

    if (x >= width || y >= height)
    {
        return;
    }

    linearPixelIndex = static_cast<size_t>(y) * width + x;
    color = applyLinearPresentation(make_float3(source[linearPixelIndex]), metadata);
    encoded.x = __half_as_ushort(__float2half(color.x));
    encoded.y = __half_as_ushort(__float2half(color.y));
    encoded.z = __half_as_ushort(__float2half(color.z));
    encoded.w = __half_as_ushort(__float2half(1.0f));
    surf2Dwrite(encoded, destination, x * sizeof(ushort4), y);
}

cudaError_t tonemapToSurfaceLinear(
    const float4 *source,
    cudaSurfaceObject_t destination,
    const uint32_t width,
    const uint32_t height,
    const oka::PresentationMetadata *metadata,
    cudaStream_t stream)
{
    dim3 blockSize(16, 16, 1);
    dim3 gridSize;

    if (source == nullptr || destination == 0 || width == 0 || height == 0 ||
        metadata == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    gridSize = dim3((width + blockSize.x - 1) / blockSize.x,
                    (height + blockSize.y - 1) / blockSize.y, 1);
    tonemapToSurfaceLinearKernel<<<gridSize, blockSize, 0, stream>>>(
        source, destination, width, height, *metadata);
    return cudaGetLastError();
}

/// Piecewise sRGB transfer matching the other backends. This local copy is
/// required because the shared host/Metal header is not CUDA-callable.
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

__global__ void gammaCorrection(const float gamma, float4 *image, uint32_t width, uint32_t height)
{
    const uint32_t linearPixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float3 color;

    if (linearPixelIndex >= height * width)
    {
        return;
    }
    color = make_float3(image[linearPixelIndex]);
    image[linearPixelIndex] = make_float4(srgbGammaChannel(color.x, gamma), srgbGammaChannel(color.y, gamma),
                                          srgbGammaChannel(color.z, gamma), 1.0f);
    return;
}

void tonemap(const ToneMapperType type,
             const float3 exposure,
             const float maxOutput,
             const float gamma,
             float4 *image,
             const uint32_t width,
             const uint32_t height)
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((width * height + 255) / 256, 1, 1);

    switch (type)
    {
    case ToneMapperType::eReinhard:
        tonemapReinhard<<<gridSize, blockSize, 0>>>(image, exposure, maxOutput, width, height);
        break;
    case ToneMapperType::eACES:
        tonemapACESFitted<<<gridSize, blockSize, 0>>>(image, exposure, maxOutput, width, height);
        break;
    case ToneMapperType::eFilmic:
        tonemapACESFilm<<<gridSize, blockSize, 0>>>(image, exposure, maxOutput, width, height);
        break;
    case ToneMapperType::eNone:
        applyExposure<<<gridSize, blockSize, 0>>>(image, exposure, width, height);
        break;
    default:
        break;
    }
    if (gamma > 0.0f)
    {
        gammaCorrection<<<gridSize, blockSize, 0>>>(gamma, image, width, height);
    }
}
