#pragma once
// Metal texture fetches for environment sampling; mapping, density and alias draws live in shared common headers.

#include <metal_stdlib>
#include <simd/simd.h>

#include "ShaderTypes.h"
#include <env_alias_sampling.h>
#include <env_map_math.h>

using namespace metal;

// Sample the environment map with an alias table (Walker/Vose).
//
// The alias table avoids the serial dependent loads of a two-dimensional CDF search.
//
// xi: two uniform random numbers in [0, 1). Returns a world-space direction and
// writes the solid-angle pdf.
static inline float3 sampleEnvMap(const float2 xi,
                                  device const EnvAliasEntry* aliasTable,
                                  uint32_t envMapWidth,
                                  uint32_t envMapHeight,
                                  float envMapRotation,
                                  thread float& pdf)
{
    const uint32_t w = envMapWidth;
    const uint32_t h = envMapHeight;

    // Shared with OptiX and the host: see common/env_alias_sampling.h, which
    // tests/render/test_env_alias_sampling.cpp exercises without a GPU.
    const EnvAliasDraw draw = envAliasDraw(aliasTable, w * h, xi.x);

    const uint32_t x = draw.texel % w;
    const uint32_t y = draw.texel / w;

    // Jitter with the alias draw's residual variate so sampled directions match the continuous density.
    const float u = ((float)x + draw.frac) / (float)w;
    const float v = envSampleSolidAngleV((int)y, (int)h, xi.y);

    const float3 dir = envUVToDir(float2(u, v), envMapRotation);

    pdf = aliasTable[draw.texel].solidAnglePdf;

    return dir;
}

// Evaluate the solid-angle PDF for a direction — used for MIS against BSDF
// sampling. One texel fetch, no search.
static inline float envMapPdf(const float3 dir,
                              device const EnvAliasEntry* aliasTable,
                              uint32_t envMapWidth,
                              uint32_t envMapHeight,
                              float envMapRotation)
{
    const float2 uv = dirToEnvUV(dir, envMapRotation);

    const int w = (int)envMapWidth;
    const int h = (int)envMapHeight;
    const int x = clamp((int)(uv.x * (float)w), 0, w - 1);
    const int y = clamp((int)(uv.y * (float)h), 0, h - 1);

    return aliasTable[(uint)y * envMapWidth + (uint)x].solidAnglePdf;
}
