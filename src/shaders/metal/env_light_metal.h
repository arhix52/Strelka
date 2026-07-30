#pragma once
// Environment map (dome light) sampling utilities for Metal shaders.
// Ported from common/env_light.h (CUDA/OptiX version).

#include <metal_stdlib>
#include "ShaderTypes.h"
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

// Luminance used to build the sampling distribution. Must match the CPU-side
// weight in MetalRender::loadEnvMap exactly, or sampling and PDF disagree.
static inline float envLuminance(const float3 rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

// Solid-angle PDF of the texel a direction falls into.
//
// The discrete probability of texel i is  w_i / W  with  w_i = lum_i * sin(theta_row),
// and the texel subtends  dOmega = 2*pi^2 * sin(theta_row) / (w*h).
// Dividing them cancels sin(theta) outright, so the whole PDF collapses to the
// texel luminance times one precomputed constant:
//     envPdfScale = (w*h) / (2*pi^2 * totalPower)
// That is why no CDF or per-texel PDF array has to be stored or searched.
static inline float envTexelPdf(const float3 radiance, float envPdfScale)
{
    return envLuminance(radiance) * envPdfScale;
}

// Sample the environment map with an alias table (Walker/Vose).
//
// The previous 2D-CDF sampler needed two binary searches per sample: ~10
// dependent loads in the marginal CDF plus ~11 scattered dependent loads into
// the conditional CDF, which for a 2K map is an 8 MB buffer — 21 cache-missing
// round trips, all serialised, for every NEE sample at every bounce. An alias
// table answers the same query with a single 8-byte load.
//
// xi: two uniform random numbers in [0, 1). Returns a world-space direction and
// writes the solid-angle pdf.
static inline float3 sampleEnvMap(
    const float2 xi,
    device const EnvAliasEntry* aliasTable,
    texture2d<float> envMapTexture,
    uint32_t envMapWidth,
    uint32_t envMapHeight,
    float envMapRotation,
    float envPdfScale,
    thread float& pdf)
{
    const uint32_t w = envMapWidth;
    const uint32_t h = envMapHeight;
    const uint32_t n = w * h;

    // Scale one variate up to bucket index + a fractional part.
    const float scaled = min(xi.x * (float)n, (float)n - 1e-6f);
    const uint32_t bucket = (uint32_t)scaled;
    float frac = scaled - (float)bucket;

    const EnvAliasEntry entry = aliasTable[bucket];

    // Recycle `frac` as both the alias coin flip and the first jitter axis:
    // remapping it back onto [0,1) conditioned on the branch taken keeps it
    // exactly uniform, so no third random number is needed.
    uint32_t texel;
    if (frac < entry.prob)
    {
        texel = bucket;
        frac = (entry.prob > 0.0f) ? (frac / entry.prob) : 0.0f;
    }
    else
    {
        texel = entry.alias;
        const float rest = 1.0f - entry.prob;
        frac = (rest > 0.0f) ? ((frac - entry.prob) / rest) : 0.0f;
    }
    frac = clamp(frac, 0.0f, 0.9999999f);

    const uint32_t x = texel % w;
    const uint32_t y = texel / w;

    // Jitter inside the texel. The old sampler always returned the texel centre,
    // so it could only ever generate w*h distinct directions while its pdf was a
    // continuous density — visible as quantised highlights and inconsistent MIS.
    const float u = ((float)x + frac) / (float)w;
    const float v = ((float)y + xi.y) / (float)h;

    const float3 dir = envUVToDir(float2(u, v), envMapRotation);

    // Read the same texel the distribution was built from (point sampling, not
    // the bilinear tap used for radiance) so sampling and pdf agree.
    const float3 radiance = envMapTexture.read(uint2(x, y)).xyz;
    pdf = envTexelPdf(radiance, envPdfScale);

    return dir;
}

// Evaluate the solid-angle PDF for a direction — used for MIS against BSDF
// sampling. One texel fetch, no search.
static inline float envMapPdf(
    const float3 dir,
    texture2d<float> envMapTexture,
    uint32_t envMapWidth,
    uint32_t envMapHeight,
    float envMapRotation,
    float envPdfScale)
{
    const float2 uv = dirToEnvUV(dir, envMapRotation);

    const int w = (int)envMapWidth;
    const int h = (int)envMapHeight;
    const int x = clamp((int)(uv.x * (float)w), 0, w - 1);
    const int y = clamp((int)(uv.y * (float)h), 0, h - 1);

    const float3 radiance = envMapTexture.read(uint2((uint)x, (uint)y)).xyz;
    return envTexelPdf(radiance, envPdfScale);
}
