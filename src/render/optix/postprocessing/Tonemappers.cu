#include "Tonemappers.h"

__device__ float calcLuminance(float3 color)
{
    return dot(color, make_float3(0.299f, 0.587f, 0.114f));
}

__device__ float3 reinhard(float3 color)
{
    float luminance = calcLuminance(color);
    float reinhard = luminance / (luminance + 1);
    return color * (reinhard / luminance);
}

__global__ void tonemapKernel(float4* image, uint32_t width, uint32_t height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > height || y > width)
    {
        return;
    }

    float3 radiance = make_float3(image[x * width + y]);

    image[x * width + y] = make_float4(reinhard(radiance), 1.0f);
    return;
}

__host__ void tonemap(float4* image, uint32_t width, uint32_t height)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(height / blockSize.x, width / blockSize.y);
    tonemapKernel<<<gridSize, blockSize, 0>>>(image, width, height);
}
