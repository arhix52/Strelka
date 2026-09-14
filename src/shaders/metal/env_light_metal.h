#pragma once
// Metal texture fetches for environment sampling; mapping, density and alias draws live in shared common headers.

#include <metal_stdlib>
#include <simd/simd.h>

#include "ShaderTypes.h"
#include <env_alias_sampling.h>
#include <env_map_math.h>

using namespace metal;

// Production Metal environment mapping. The random dimensions are generated on
// a 23-bit [0, 1) lattice, so adding half a lattice step already places them in
// the open interval. Do not pay for the cross-backend audit path's boundary
// retries or pole reconstruction here.
static inline float envOpenRandom(float xi)
{
    return xi + 0x1p-24f;
}

static inline float2 metalDirToEnvUv(float3 dir, float2 rotationSinCos)
{
    const float sinR = rotationSinCos.x;
    const float cosR = rotationSinCos.y;
    const float rx = cosR * dir.x - sinR * dir.z;
    const float rz = sinR * dir.x + cosR * dir.z;
    const float phi = atan2(rx, rz);
    const float theta = atan2(length(float2(rx, rz)), dir.y);
    return float2((phi + M_PI_F) * (0.5f / M_PI_F), theta * (1.0f / M_PI_F));
}

static inline float3 metalEnvDirection(float u, float cosTheta, float2 rotationSinCos)
{
    const float phi = u * (2.0f * M_PI_F) - M_PI_F;
    const float sinTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f));
    const float sinPhi = sin(phi);
    const float cosPhi = cos(phi);
    const float x = sinTheta * sinPhi;
    const float z = sinTheta * cosPhi;
    const float sinR = rotationSinCos.x;
    const float cosR = rotationSinCos.y;
    return float3(cosR * x + sinR * z, cosTheta, -sinR * x + cosR * z);
}

// Sample the environment map with an alias table (Walker/Vose).
//
// The alias table avoids the serial dependent loads of a two-dimensional CDF search.
//
// The two full-width words select the alias bucket and branch. The two floats
// independently jitter within the selected texel.
static inline float3 sampleEnvMap(const uint2 aliasWords,
                                  const float2 jitter,
                                  device const EnvAliasEntry* aliasTable,
                                  device const packed_float2* rowCosBounds,
                                  uint32_t envMapWidth,
                                  uint32_t envMapHeight,
                                  float2 rotationSinCos,
                                  thread float2& uv,
                                  thread float& pdf)
{
    const uint32_t w = envMapWidth;
    const uint32_t h = envMapHeight;

    // Shared with OptiX and the host: see common/env_alias_sampling.h, which
    // tests/render/test_env_alias_sampling.cpp exercises without a GPU.
    const EnvAliasDraw draw = envAliasDraw(aliasTable, w * h, aliasWords.x, aliasWords.y);

    const uint32_t x = draw.texel % w;
    const uint32_t y = draw.texel / w;

    const float u = (float(x) + envOpenRandom(jitter.x)) / float(w);
    const float2 row = float2(rowCosBounds[y]);
    const float cosTheta = mix(row.x, row.y, envOpenRandom(jitter.y));
    const float v = acos(clamp(cosTheta, -1.0f, 1.0f)) * (1.0f / M_PI_F);
    uv = float2(u, v);
    const float3 dir = metalEnvDirection(u, cosTheta, rotationSinCos);

    pdf = aliasTable[draw.texel].solidAnglePdf;

    return dir;
}

// Evaluate the solid-angle PDF for a direction — used for MIS against BSDF
// sampling. One texel fetch, no search.
static inline float envMapPdf(const float3 dir,
                              device const EnvAliasEntry* aliasTable,
                              uint32_t envMapWidth,
                              uint32_t envMapHeight,
                              float2 rotationSinCos)
{
    const float2 uv = metalDirToEnvUv(dir, rotationSinCos);

    const int w = (int)envMapWidth;
    const int h = (int)envMapHeight;
    const int x = clamp((int)(uv.x * (float)w), 0, w - 1);
    const int y = clamp((int)(uv.y * (float)h), 0, h - 1);

    return aliasTable[(uint)y * envMapWidth + (uint)x].solidAnglePdf;
}
