#pragma once
// Environment map (dome light) sampling utilities for OptiX shaders.
//
// CUDA only, despite living under shaders/common: the texture fetches below take
// a cudaTextureObject_t. The parametrisation and the density are not CUDA at
// all, and they used to be duplicated here and in
// src/shaders/metal/env_light_metal.h with a comment on each asking that the two
// stay identical. They now come from common/env_map_math.h, which the Metal
// header includes as well and which the host tests compile.

#include <vector_types.h>
#include <sutil/vec_math.h>

#include <env_alias_sampling.h>
#include <env_map_math.h>

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
// The two full-width words select the alias bucket and branch. The two floats
// independently jitter within the selected texel.
static __forceinline__ __device__ float3 sampleEnvMap(const uint2& aliasWords,
                                                      const float2& jitter,
                                                      const uint2& retryWords,
                                                      const EnvAliasEntry* aliasTable,
                                                      uint32_t envMapWidth,
                                                      uint32_t envMapHeight,
                                                      float envMapRotation,
                                                      float& pdf)
{
    const uint32_t w = envMapWidth;
    const uint32_t h = envMapHeight;

    // Shared with Metal and the host: see common/env_alias_sampling.h, which
    // tests/render/test_env_alias_sampling.cpp exercises without a GPU.
    const EnvAliasDraw draw = envAliasDraw(aliasTable, w * h, aliasWords.x, aliasWords.y);

    const uint32_t x = draw.texel % w;
    const uint32_t y = draw.texel / w;

    const float3 dir = envSampleTexelDirection(
        (int)x, (int)y, (int)w, (int)h, jitter.x, jitter.y, retryWords.x, retryWords.y, envMapRotation);

    pdf = aliasTable[draw.texel].solidAnglePdf;

    return dir;
}

// Evaluate the solid-angle pdf for a direction -- used for MIS against BSDF
// sampling. One texel fetch, no search.
static __forceinline__ __device__ float envMapPdf(const float3& dir,
                                                  const EnvAliasEntry* aliasTable,
                                                  uint32_t envMapWidth,
                                                  uint32_t envMapHeight,
                                                  float envMapRotation)
{
    const float2 uv = dirToEnvUV(dir, envMapRotation);

    const int w = (int)envMapWidth;
    const int h = (int)envMapHeight;
    const int x = max(0, min((int)(uv.x * (float)w), w - 1));
    const int y = max(0, min((int)(uv.y * (float)h), h - 1));

    return aliasTable[(uint32_t)y * envMapWidth + (uint32_t)x].solidAnglePdf;
}
