#pragma once
// Metal texture fetches for environment sampling; mapping, density and alias draws live in shared common headers.

#include <metal_stdlib>
#include <simd/simd.h>

#include "ShaderTypes.h"
#include <env_alias_sampling.h>
#include <env_map_math.h>

using namespace metal;

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

// Metal keeps the random alias walk separate from the PDF table. A sampled
// direction needs one random 8-byte selection record followed by one 4-byte
// PDF load; MIS-only evaluation touches the dense PDF array alone.
static inline uint32_t metalEnvAliasTexel(device const uint2* aliasTable,
                                          uint32_t texelCount,
                                          uint32_t bucketWord,
                                          uint32_t coinWord)
{
    const uint32_t bucket = discreteUniformIndex(texelCount, bucketWord);
    const uint2 entry = aliasTable[bucket];
    return discreteAliasSelect(texelCount, bucket, coinWord, entry.x, entry.y);
}

static inline float3 sampleEnvMap(const uint2 aliasWords,
                                  const float2 jitter,
                                  device const uint2* aliasTable,
                                  device const float* pdfTable,
                                  device const packed_float2* rowCosBounds,
                                  uint32_t envMapWidth,
                                  uint32_t envMapHeight,
                                  float2 rotationSinCos,
                                  thread float2& uv,
                                  thread float& pdf)
{
    const uint32_t w = envMapWidth;
    const uint32_t h = envMapHeight;

    const uint32_t texel = metalEnvAliasTexel(aliasTable, w * h, aliasWords.x, aliasWords.y);

    const uint32_t x = texel % w;
    const uint32_t y = texel / w;

    const float u = (float(x) + envOpenRandom(jitter.x)) / float(w);
    const float2 row = float2(rowCosBounds[y]);
    const float cosTheta = mix(row.x, row.y, envOpenRandom(jitter.y));
    const float v = acos(clamp(cosTheta, -1.0f, 1.0f)) * (1.0f / M_PI_F);
    uv = float2(u, v);
    const float3 dir = metalEnvDirection(u, cosTheta, rotationSinCos);

    pdf = pdfTable[texel];

    return dir;
}

// Evaluate the solid-angle PDF for a direction — used for MIS against BSDF
// sampling. One texel fetch, no search.
static inline float envMapPdf(
    const float3 dir, device const float* pdfTable, uint32_t envMapWidth, uint32_t envMapHeight, float2 rotationSinCos)
{
    const float2 uv = metalDirToEnvUv(dir, rotationSinCos);

    const int w = (int)envMapWidth;
    const int h = (int)envMapHeight;
    const int x = clamp((int)(uv.x * (float)w), 0, w - 1);
    const int y = clamp((int)(uv.y * (float)h), 0, h - 1);

    return pdfTable[(uint)y * envMapWidth + (uint)x];
}
