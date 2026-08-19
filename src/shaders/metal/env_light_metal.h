#pragma once
// Environment map (dome light) sampling utilities for Metal shaders.
//
// Metal only, because the fetches below take a texture2d. The parametrisation,
// the luminance and the density are not Metal at all and now come from
// common/env_map_math.h; the alias draw comes from common/env_alias_sampling.h.
// Both used to be transcribed into this file, and the alias copy had already
// drifted -- it was missing the NaN and negative-variate guards the shared one
// grew.

#include <metal_stdlib>
#include <simd/simd.h>

#include "ShaderTypes.h"
#include <env_alias_sampling.h>
#include <env_map_math.h>

using namespace metal;

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

    // Shared with OptiX and the host: see common/env_alias_sampling.h, which
    // tests/render/test_env_alias_sampling.cpp exercises without a GPU.
    const EnvAliasDraw draw = envAliasDraw(aliasTable, w * h, xi.x);

    const uint32_t x = draw.texel % w;
    const uint32_t y = draw.texel / w;

    // Jitter inside the texel, on the variate the alias draw handed back. The
    // old sampler always returned the texel centre, so it could only ever
    // generate w*h distinct directions while its pdf was a continuous density --
    // visible as quantised highlights and inconsistent MIS.
    const float u = ((float)x + draw.frac) / (float)w;
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
