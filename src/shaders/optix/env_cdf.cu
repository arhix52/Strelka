#include "env_cdf.h"
#include <math_constants.h>

// Each block processes one row of the env map.
// Thread count must be >= width (launched with blockDim.x = min(width, 1024)).
// For maps wider than 1024, each thread handles multiple pixels.
__global__ void envCdfBuildKernel(
    const float* __restrict__ envData, // float4 RGBA per pixel, tightly packed
    int w, int h,
    float* __restrict__ cdfX,    // conditional CDF: [y * w + x]
    float* __restrict__ cdfY,    // marginal CDF: [y]
    float* __restrict__ rowSums) // temporary per-row sum: [y]
{
    const int y = blockIdx.x;
    if (y >= h) return;

    // sin(theta) Jacobian weight for equirectangular map
    const float v = ((float)y + 0.5f) / (float)h;
    const float sinTheta = sinf(v * CUDART_PI_F);

    // Compute luminance for each pixel in this row and write to cdfX as initial values
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Phase 1: compute weighted luminance for each pixel
    for (int x = tid; x < w; x += stride)
    {
        const int pixelIdx = (y * w + x) * 4; // float4 layout
        const float r = envData[pixelIdx + 0];
        const float g = envData[pixelIdx + 1];
        const float b = envData[pixelIdx + 2];
        const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        cdfX[y * w + x] = lum * sinTheta;
    }
    __syncthreads();

    // Phase 2: sequential inclusive prefix sum (simple and correct for any width)
    // Only thread 0 does this to avoid complexity of parallel scan for variable widths
    if (tid == 0)
    {
        float sum = 0.0f;
        for (int x = 0; x < w; ++x)
        {
            sum += cdfX[y * w + x];
            cdfX[y * w + x] = sum;
        }
        rowSums[y] = sum;

        // Normalize to [0, 1]
        if (sum > 0.0f)
        {
            const float invSum = 1.0f / sum;
            for (int x = 0; x < w; ++x)
            {
                cdfX[y * w + x] *= invSum;
            }
        }
        else
        {
            // Uniform distribution for zero-luminance rows
            for (int x = 0; x < w; ++x)
            {
                cdfX[y * w + x] = (float)(x + 1) / (float)w;
            }
        }
    }
}

// Build marginal CDF from row sums
__global__ void envCdfMarginalKernel(
    const float* __restrict__ rowSums,
    int h,
    float* __restrict__ cdfY,
    float* __restrict__ totalPowerOut)
{
    // Single thread builds marginal CDF
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float sum = 0.0f;
    for (int y = 0; y < h; ++y)
    {
        sum += rowSums[y];
        cdfY[y] = sum;
    }

    *totalPowerOut = sum;

    // Normalize
    if (sum > 0.0f)
    {
        const float invSum = 1.0f / sum;
        for (int y = 0; y < h; ++y)
        {
            cdfY[y] *= invSum;
        }
    }
    else
    {
        for (int y = 0; y < h; ++y)
        {
            cdfY[y] = (float)(y + 1) / (float)h;
        }
    }
}

void buildEnvMapCdf(
    const float* d_envData,
    int w, int h,
    float* d_cdfX,
    float* d_cdfY,
    float* totalPower)
{
    // Temporary buffer for per-row luminance sums
    float* d_rowSums = nullptr;
    cudaMalloc(&d_rowSums, h * sizeof(float));

    // One block per row, single thread per block for simplicity
    // (env maps are typically 2K-8K wide, sequential scan per row is fast enough)
    int threadsPerBlock = 1;
    envCdfBuildKernel<<<h, threadsPerBlock, 0>>>(
        d_envData, w, h, d_cdfX, d_cdfY, d_rowSums);

    // Device-side total power output
    float* d_totalPower = nullptr;
    cudaMalloc(&d_totalPower, sizeof(float));

    // Build marginal CDF from row sums
    envCdfMarginalKernel<<<1, 1>>>(d_rowSums, h, d_cdfY, d_totalPower);

    // Copy total power back to host
    cudaMemcpy(totalPower, d_totalPower, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rowSums);
    cudaFree(d_totalPower);
}
