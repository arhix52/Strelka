#pragma once
// Environment map (dome light) sampling utilities for OptiX shaders.
// Provides equirectangular mapping, importance sampling via 2D CDF, and PDF evaluation.

#include <vector_types.h>
#include <sutil/vec_math.h>

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

// Convert a world-space direction to equirectangular UV coordinates.
// rotation: Y-axis rotation in radians applied to the environment map.
static __forceinline__ __device__ float2 dirToEnvUV(const float3& dir, float rotation)
{
    // Apply inverse Y-rotation to the direction
    const float cosR = cosf(-rotation);
    const float sinR = sinf(-rotation);
    const float rx = cosR * dir.x + sinR * dir.z;
    const float rz = -sinR * dir.x + cosR * dir.z;

    // Equirectangular: phi = atan2(rx, rz), theta = acos(dir.y)
    float phi = atan2f(rx, rz); // [-pi, pi]
    float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f)); // [0, pi]

    float u = (phi + M_PIf) / (2.0f * M_PIf); // [0, 1]
    float v = theta / M_PIf;                    // [0, 1]
    return make_float2(u, v);
}

// Convert equirectangular UV to world-space direction.
static __forceinline__ __device__ float3 envUVToDir(const float2& uv, float rotation)
{
    float phi = uv.x * 2.0f * M_PIf - M_PIf; // [-pi, pi]
    float theta = uv.y * M_PIf;                // [0, pi]

    float sinTheta = sinf(theta);
    float cosTheta = cosf(theta);

    // Direction before rotation
    float x = sinTheta * sinf(phi);
    float y = cosTheta;
    float z = sinTheta * cosf(phi);

    // Apply Y-rotation
    const float cosR = cosf(rotation);
    const float sinR = sinf(rotation);
    float rx = cosR * x + sinR * z;
    float rz = -sinR * x + cosR * z;

    return make_float3(rx, y, rz);
}

// Binary search in a CDF array. Returns index i such that cdf[i-1] < xi <= cdf[i].
// cdf: CDF array of length n (cdf[n-1] == 1.0)
static __forceinline__ __device__ int binarySearchCdf(const float* cdf, int n, float xi)
{
    int lo = 0;
    int hi = n - 1;
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] < xi)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// Sample the environment map using the 2D CDF.
// xi: two uniform random numbers in [0, 1)
// Returns: world-space direction, writes pdf.
// Forward-declared Params struct is expected to contain env map fields.
struct Params; // forward decl (already defined in OptixRenderParams.h)

static __forceinline__ __device__ float3 sampleEnvMap(
    const float2& xi,
    const float* cdfX,
    const float* cdfY,
    uint32_t envMapWidth,
    uint32_t envMapHeight,
    float envMapRotation,
    float& pdf)
{
    const int w = (int)envMapWidth;
    const int h = (int)envMapHeight;

    // Sample marginal CDF to get row y
    const int y = binarySearchCdf(cdfY, h, xi.y);

    // Sample conditional CDF for row y to get column x
    const int x = binarySearchCdf(&cdfX[y * w], w, xi.x);

    // Compute continuous UV with sub-pixel offset
    // PDF of selecting this pixel = p_y * p_x
    float pdfY = (y == 0) ? cdfY[0] : (cdfY[y] - cdfY[y - 1]);
    float pdfX = (x == 0) ? cdfX[y * w] : (cdfX[y * w + x] - cdfX[y * w + x - 1]);

    // UV at pixel center
    float u = ((float)x + 0.5f) / (float)w;
    float v = ((float)y + 0.5f) / (float)h;

    float3 dir = envUVToDir(make_float2(u, v), envMapRotation);

    // sin(theta) for Jacobian
    float theta = v * M_PIf;
    float sinTheta = sinf(theta);

    // PDF: pdfX * pdfY * (w * h) / (2 * pi^2 * sinTheta)
    if (sinTheta > 1e-6f && pdfX > 0.0f && pdfY > 0.0f)
    {
        pdf = (pdfX * pdfY * (float)(w * h)) / (2.0f * M_PIf * M_PIf * sinTheta);
    }
    else
    {
        pdf = 0.0f;
    }

    return dir;
}

// Evaluate PDF for a given world-space direction against the env map CDF.
static __forceinline__ __device__ float envMapPdf(
    const float3& dir,
    const float* cdfX,
    const float* cdfY,
    uint32_t envMapWidth,
    uint32_t envMapHeight,
    float envMapRotation)
{
    const int w = (int)envMapWidth;
    const int h = (int)envMapHeight;

    float2 uv = dirToEnvUV(dir, envMapRotation);

    int x = (int)(uv.x * w);
    int y = (int)(uv.y * h);
    x = max(0, min(x, w - 1));
    y = max(0, min(y, h - 1));

    float pdfY = (y == 0) ? cdfY[0] : (cdfY[y] - cdfY[y - 1]);
    float pdfX = (x == 0) ? cdfX[y * w] : (cdfX[y * w + x] - cdfX[y * w + x - 1]);

    float theta = uv.y * M_PIf;
    float sinTheta = sinf(theta);

    if (sinTheta > 1e-6f && pdfX > 0.0f && pdfY > 0.0f)
    {
        return (pdfX * pdfY * (float)(w * h)) / (2.0f * M_PIf * M_PIf * sinTheta);
    }
    return 0.0f;
}
