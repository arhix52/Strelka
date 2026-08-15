#pragma once
// Alias-table draw for environment importance sampling.
//
// One copy, compiled twice: as a __device__ function into the OptiX modules
// (src/shaders/common/env_light.h calls it) and as a plain inline function into
// the host, where tests/render/test_env_alias_sampling.cpp exercises it without
// a GPU. A second implementation would be a second thing to keep in step with
// the host builder, and the whole point of this table is that one distribution
// is shared -- across two backends already, and now across two compilers.
//
// Deliberately free of CUDA and Metal headers, and of any math beyond compare
// and divide, so that "the test passes" says something about the code the GPU
// actually runs.

#include <stdint.h>

#if defined(__CUDACC__)
#    define STRELKA_ENV_SAMPLING_FN static __forceinline__ __device__
#else
#    define STRELKA_ENV_SAMPLING_FN inline
#endif

/// One bucket of a Walker/Vose alias table over the environment map's texels.
///
/// Layout must stay identical to oka::metal::EnvAliasEntry in
/// src/render/metal/ibl_alias_table.h, which is the single host-side builder both
/// backends use; OptixRender.cpp static_asserts that it does.
struct EnvAliasEntry
{
    float prob;
    uint32_t alias;
};

/// A texel index plus a uniform variate left over from choosing it.
struct EnvAliasDraw
{
    uint32_t texel;
    float frac;
};

/// Draw one texel from the table using a single uniform variate.
///
/// The fractional part of `xi * texelCount` does double duty: first as the alias
/// coin flip, then -- remapped back onto [0,1) conditioned on the branch it took
/// -- as a jitter axis inside the chosen texel. Conditioning is what keeps it
/// exactly uniform, so no second random number is needed for either job. Without
/// the jitter the sampler can only ever return texel centres, which is w*h
/// distinct directions carrying a continuous density: quantised highlights, and
/// a MIS weight that disagrees with what was actually sampled.
STRELKA_ENV_SAMPLING_FN EnvAliasDraw envAliasDraw(const EnvAliasEntry* table, uint32_t texelCount, float xi)
{
    EnvAliasDraw out;
    out.texel = 0u;
    out.frac = 0.0f;
    if (table == nullptr || texelCount == 0u)
    {
        return out;
    }

    // The upper bound matters: xi is nominally below one, but a variate of
    // 0.99999997 times a few million texels rounds to exactly texelCount in
    // float, which would index one past the end of the table.
    float scaled = xi * (float)texelCount;
    const float limit = (float)texelCount - 1e-6f;
    if (!(scaled >= 0.0f))
    {
        scaled = 0.0f; // also catches NaN
    }
    if (scaled > limit)
    {
        scaled = limit;
    }

    const uint32_t bucket = (uint32_t)scaled;
    float frac = scaled - (float)bucket;

    const EnvAliasEntry entry = table[bucket];
    if (frac < entry.prob)
    {
        out.texel = bucket;
        frac = (entry.prob > 0.0f) ? (frac / entry.prob) : 0.0f;
    }
    else
    {
        out.texel = entry.alias;
        const float rest = 1.0f - entry.prob;
        frac = (rest > 0.0f) ? ((frac - entry.prob) / rest) : 0.0f;
    }

    if (!(frac > 0.0f))
    {
        frac = 0.0f;
    }
    if (frac > 0.9999999f)
    {
        frac = 0.9999999f;
    }
    out.frac = frac;
    return out;
}
