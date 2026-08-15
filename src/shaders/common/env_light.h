#pragma once
// Environment map (dome light) sampling utilities for OptiX shaders.
//
// CUDA only, despite living under shaders/common: Metal keeps its own copy in
// src/shaders/metal/env_light_metal.h. The two must stay behaviourally identical
// -- both now sample from the alias table that src/render/metal/ibl_alias_table.h
// builds on the host -- but they are separate translation units and nothing here
// is compiled for Metal.

#include <vector_types.h>
#include <sutil/vec_math.h>

#include <OptixRenderParams.h>

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

// Luminance used to build the sampling distribution. Must match the host weight
// in buildIblAliasTable() exactly, or sampling and pdf disagree.
static __forceinline__ __device__ float envLuminance(const float3& rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

// Solid-angle pdf of the texel a direction falls into.
//
// The discrete probability of texel i is  w_i / W  with  w_i = lum_i * sin(theta_row),
// and the texel subtends  dOmega = 2*pi^2 * sin(theta_row) / (w*h).
// Dividing them cancels sin(theta) outright, so the whole pdf collapses to the
// texel luminance times one precomputed constant:
//     envPdfScale = (w*h) / (2*pi^2 * totalPower)
// That is why no CDF or per-texel pdf array has to be stored or searched.
static __forceinline__ __device__ float envTexelPdf(const float3& radiance, float envPdfScale)
{
    return envLuminance(radiance) * envPdfScale;
}

// Point fetch of one texel. The sampling distribution is built from unfiltered
// texels on the host, so sampling and pdf have to read unfiltered texels too --
// a bilinear tap here makes the two disagree along every luminance edge.
static __forceinline__ __device__ float3 envTexelFetch(cudaTextureObject_t pointTex, int x, int y)
{
    const float4 t = tex2D<float4>(pointTex, (float)x + 0.5f, (float)y + 0.5f);
    return make_float3(t.x, t.y, t.z);
}

// Sample the environment map with an alias table (Walker/Vose).
//
// The 2D-CDF sampler this replaces needed two binary searches per sample: ~11
// dependent loads in the marginal CDF plus ~11 scattered dependent loads into
// the conditional CDF, which for a 2K map is an 8 MB buffer -- 22 cache-missing
// round trips, all serialised, for every NEE sample at every bounce. An alias
// table answers the same query with a single 8-byte load, and it is the same
// table and the same host builder Metal uses, so the two backends now draw their
// environment samples from one distribution rather than two.
//
// xi: two uniform random numbers in [0, 1). Returns a world-space direction and
// writes the solid-angle pdf.
static __forceinline__ __device__ float3 sampleEnvMap(
    const float2& xi,
    const EnvAliasEntry* aliasTable,
    cudaTextureObject_t envMapPointTexture,
    uint32_t envMapWidth,
    uint32_t envMapHeight,
    float envMapRotation,
    float envPdfScale,
    float& pdf)
{
    const uint32_t w = envMapWidth;
    const uint32_t h = envMapHeight;

    // Shared with the host: see src/render/optix/env_alias_sampling.h, which
    // tests/render/test_env_alias_sampling.cpp exercises without a GPU.
    const EnvAliasDraw draw = envAliasDraw(aliasTable, w * h, xi.x);

    const uint32_t x = draw.texel % w;
    const uint32_t y = draw.texel / w;

    // Jitter inside the texel, on the variate the alias draw handed back.
    const float u = ((float)x + draw.frac) / (float)w;
    const float v = ((float)y + xi.y) / (float)h;

    const float3 dir = envUVToDir(make_float2(u, v), envMapRotation);

    const float3 radiance = envTexelFetch(envMapPointTexture, (int)x, (int)y);
    pdf = envTexelPdf(radiance, envPdfScale);

    return dir;
}

// Evaluate the solid-angle pdf for a direction -- used for MIS against BSDF
// sampling. One texel fetch, no search.
static __forceinline__ __device__ float envMapPdf(
    const float3& dir,
    cudaTextureObject_t envMapPointTexture,
    uint32_t envMapWidth,
    uint32_t envMapHeight,
    float envMapRotation,
    float envPdfScale)
{
    const float2 uv = dirToEnvUV(dir, envMapRotation);

    const int w = (int)envMapWidth;
    const int h = (int)envMapHeight;
    const int x = max(0, min((int)(uv.x * (float)w), w - 1));
    const int y = max(0, min((int)(uv.y * (float)h), h - 1));

    return envTexelPdf(envTexelFetch(envMapPointTexture, x, y), envPdfScale);
}
