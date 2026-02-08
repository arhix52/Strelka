#pragma once
// Environment map (dome light) sampling utilities for Metal shaders.
// Ported from common/env_light.h (CUDA/OptiX version).

#include <metal_stdlib>
using namespace metal;

// Convert a world-space direction to equirectangular UV coordinates.
// rotation: Y-axis rotation in radians applied to the environment map.
static inline float2 dirToEnvUV(const float3 dir, float rotation)
{
    // Apply inverse Y-rotation to the direction
    const float cosR = cos(-rotation);
    const float sinR = sin(-rotation);
    const float rx = cosR * dir.x + sinR * dir.z;
    const float rz = -sinR * dir.x + cosR * dir.z;

    // Equirectangular: phi = atan2(rx, rz), theta = acos(dir.y)
    float phi = atan2(rx, rz);  // [-pi, pi]
    float theta = acos(clamp(dir.y, -1.0f, 1.0f)); // [0, pi]

    float u = (phi + M_PI_F) / (2.0f * M_PI_F); // [0, 1]
    float v = theta / M_PI_F;                     // [0, 1]
    return float2(u, v);
}

// Convert equirectangular UV to world-space direction.
static inline float3 envUVToDir(const float2 uv, float rotation)
{
    float phi = uv.x * 2.0f * M_PI_F - M_PI_F; // [-pi, pi]
    float theta = uv.y * M_PI_F;                 // [0, pi]

    float sinTheta = sin(theta);
    float cosTheta = cos(theta);

    float x = sinTheta * sin(phi);
    float y = cosTheta;
    float z = sinTheta * cos(phi);

    // Apply Y-rotation
    const float cosR = cos(rotation);
    const float sinR = sin(rotation);
    float rx = cosR * x + sinR * z;
    float rz = -sinR * x + cosR * z;

    return float3(rx, y, rz);
}

// Binary search in a CDF array. Returns index i such that cdf[i-1] < xi <= cdf[i].
static inline int binarySearchCdf(device const float* cdf, int n, float xi)
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
static inline float3 sampleEnvMap(
    const float2 xi,
    device const float* cdfX,
    device const float* cdfY,
    uint32_t envMapWidth,
    uint32_t envMapHeight,
    float envMapRotation,
    thread float& pdf)
{
    const int w = (int)envMapWidth;
    const int h = (int)envMapHeight;

    // Sample marginal CDF to get row y
    const int y = binarySearchCdf(cdfY, h, xi.y);

    // Sample conditional CDF for row y to get column x
    const int x = binarySearchCdf(&cdfX[y * w], w, xi.x);

    // PDF of selecting this pixel = p_y * p_x
    float pdfY = (y == 0) ? cdfY[0] : (cdfY[y] - cdfY[y - 1]);
    float pdfX = (x == 0) ? cdfX[y * w] : (cdfX[y * w + x] - cdfX[y * w + x - 1]);

    // UV at pixel center
    float u = ((float)x + 0.5f) / (float)w;
    float v = ((float)y + 0.5f) / (float)h;

    float3 dir = envUVToDir(float2(u, v), envMapRotation);

    // sin(theta) for Jacobian
    float theta = v * M_PI_F;
    float sinTheta = sin(theta);

    // PDF: pdfX * pdfY * (w * h) / (2 * pi^2 * sinTheta)
    if (sinTheta > 1e-6f && pdfX > 0.0f && pdfY > 0.0f)
    {
        pdf = (pdfX * pdfY * (float)(w * h)) / (2.0f * M_PI_F * M_PI_F * sinTheta);
    }
    else
    {
        pdf = 0.0f;
    }

    return dir;
}

// Evaluate PDF for a given world-space direction against the env map CDF.
static inline float envMapPdf(
    const float3 dir,
    device const float* cdfX,
    device const float* cdfY,
    uint32_t envMapWidth,
    uint32_t envMapHeight,
    float envMapRotation)
{
    const int w = (int)envMapWidth;
    const int h = (int)envMapHeight;

    float2 uv = dirToEnvUV(dir, envMapRotation);

    int x = (int)(uv.x * w);
    int y = (int)(uv.y * h);
    x = clamp(x, 0, w - 1);
    y = clamp(y, 0, h - 1);

    float pdfY = (y == 0) ? cdfY[0] : (cdfY[y] - cdfY[y - 1]);
    float pdfX = (x == 0) ? cdfX[y * w] : (cdfX[y * w + x] - cdfX[y * w + x - 1]);

    float theta = uv.y * M_PI_F;
    float sinTheta = sin(theta);

    if (sinTheta > 1e-6f && pdfX > 0.0f && pdfY > 0.0f)
    {
        return (pdfX * pdfY * (float)(w * h)) / (2.0f * M_PI_F * M_PI_F * sinTheta);
    }
    return 0.0f;
}
