#pragma once

#ifndef __METAL_VERSION__
#    include <stdint.h>
#endif

#include <discrete_sampling.h>

// Which address space the table lives in. Metal needs it spelled out on the
// pointer; the other two compilers have one address space and want nothing.
#if defined(__METAL_VERSION__)
#    define STRELKA_ENV_TABLE_PTR device
#else
#    define STRELKA_ENV_TABLE_PTR
#endif

#if defined(__CUDACC__)
#    define STRELKA_ENV_SAMPLING_FN static __forceinline__ __device__
#elif defined(__METAL_VERSION__)
#    define STRELKA_ENV_SAMPLING_FN static inline
#else
#    define STRELKA_ENV_SAMPLING_FN inline
#endif

struct EnvAliasEntry
{
    uint32_t threshold;
    uint32_t alias;
    float solidAnglePdf;
};

/// A texel selected from the represented alias distribution.
struct EnvAliasDraw
{
    uint32_t texel;
    // Retained for the external 1x1 audit ABI. Production sampling uses a
    // separate within-texel random dimension and leaves this at zero.
    float frac;
};

/// Draw one texel using independent bucket and alias-coin variates.
STRELKA_ENV_SAMPLING_FN EnvAliasDraw envAliasDraw(STRELKA_ENV_TABLE_PTR const EnvAliasEntry* table,
                                                  uint32_t texelCount,
                                                  uint32_t bucketWord,
                                                  uint32_t coinWord)
{
    EnvAliasDraw out;
    out.texel = 0u;
    out.frac = 0.0f;
    if (table == nullptr || texelCount == 0u)
    {
        return out;
    }

    const uint32_t bucket = discreteUniformIndex(texelCount, bucketWord);
    const EnvAliasEntry entry = table[bucket];
    out.texel = discreteAliasSelect(texelCount, bucket, coinWord, entry.threshold, entry.alias);
    return out;
}

// Compatibility for the external 1x1 environment audit source. Production
// samplers must use the independent-variate overload above; returning an invalid
// index for a larger table prevents accidental reuse of the old correlated draw.
STRELKA_ENV_SAMPLING_FN EnvAliasDraw envAliasDraw(STRELKA_ENV_TABLE_PTR const EnvAliasEntry* table,
                                                  uint32_t texelCount,
    float bucketUniform)
{
    EnvAliasDraw out = texelCount == 1u ? envAliasDraw(table, texelCount, 0u, 0u) :
                                         EnvAliasDraw{ texelCount, 0.0f };
    if (texelCount == 1u)
    {
        out.frac = bucketUniform;
    }
    return out;
}
