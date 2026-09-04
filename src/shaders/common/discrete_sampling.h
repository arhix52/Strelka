#ifndef STRELKA_DISCRETE_SAMPLING_H
#define STRELKA_DISCRETE_SAMPLING_H

#if !defined(__METAL_VERSION__)
#    include <stdint.h>
#endif

#if defined(__CUDACC__)
#    define STRELKA_DISCRETE_FN static __forceinline__ __host__ __device__
#elif defined(__METAL_VERSION__)
#    define STRELKA_DISCRETE_FN static inline
#else
#    define STRELKA_DISCRETE_FN inline
#endif

// Map all 2^32 input words onto `count` adjacent buckets. Unlike
// floor(float(word) * count), this retains enough states for every uint32-sized
// table. Multiplication-high also avoids the float rounding that can produce
// the out-of-range endpoint `count`.
STRELKA_DISCRETE_FN uint32_t discreteUniformIndex(uint32_t count, uint32_t word)
{
    if (count == 0u)
    {
        return 0u;
    }
#if defined(__METAL_VERSION__)
    return uint32_t((ulong(word) * ulong(count)) >> 32u);
#else
    return uint32_t((uint64_t(word) * uint64_t(count)) >> 32u);
#endif
}

// A self alias denotes the complete bucket. Otherwise threshold is the exact
// number of the 2^32 coin words that retain the bucket.
STRELKA_DISCRETE_FN uint32_t discreteAliasSelect(
    uint32_t count, uint32_t bucket, uint32_t coinWord, uint32_t threshold, uint32_t alias)
{
    if (count == 0u || bucket >= count)
    {
        return count;
    }
    if (alias == bucket || coinWord < threshold)
    {
        return bucket;
    }
    return alias < count ? alias : count;
}

// Convert a float class probability to its nearest strict-comparison count.
// Splitting at 16 bits avoids spelling 2^32 as a float, whose rounded endpoint
// cannot be converted safely to uint32 on every device compiler.
STRELKA_DISCRETE_FN uint32_t discreteProbabilityThreshold(float probability)
{
    if (!(probability > 0.0f))
    {
        return 0u;
    }
    if (!(probability < 1.0f))
    {
        return 0xffffffffu;
    }
    const float scaledHigh = probability * 65536.0f;
    uint32_t high = uint32_t(scaledHigh);
    const float scaledLow = (scaledHigh - float(high)) * 65536.0f;
    uint32_t low = uint32_t(scaledLow + 0.5f);
    if (low == 65536u)
    {
        low = 0u;
        ++high;
    }
    return (high << 16u) | low;
}

STRELKA_DISCRETE_FN float discreteThresholdProbability(uint32_t threshold)
{
    return float(threshold >> 16u) * 0x1p-16f + float(threshold & 0xffffu) * 0x1p-32f;
}

STRELKA_DISCRETE_FN bool discreteBernoulli(uint32_t word, float probability)
{
    if (!(probability > 0.0f))
    {
        return false;
    }
    if (!(probability < 1.0f))
    {
        return true;
    }
    return word < discreteProbabilityThreshold(probability);
}

// BSDF entry points historically receive floats, and their continuous draws
// still do. Their two categorical coordinates use the common 23-bit subset of
// every production sampler so the represented branch masses are exact floats
// and identical on Metal and OptiX.
#define STRELKA_FLOAT_LATTICE_STATES 8388608u

STRELKA_DISCRETE_FN uint32_t discreteFloatLatticeCount(float probability)
{
    if (!(probability > 0.0f))
    {
        return 0u;
    }
    if (!(probability < 1.0f))
    {
        return STRELKA_FLOAT_LATTICE_STATES;
    }
    uint32_t count = uint32_t(probability * float(STRELKA_FLOAT_LATTICE_STATES) + 0.5f);
    count = count > 0u ? count : 1u;
    return count < STRELKA_FLOAT_LATTICE_STATES ? count : STRELKA_FLOAT_LATTICE_STATES - 1u;
}

STRELKA_DISCRETE_FN float discreteFloatLatticeProbability(float probability)
{
    return float(discreteFloatLatticeCount(probability)) * 0x1p-23f;
}

STRELKA_DISCRETE_FN uint32_t discreteFloatLatticeWord(float uniform)
{
    if (!(uniform > 0.0f))
    {
        return 0u;
    }
    if (!(uniform < 1.0f))
    {
        return STRELKA_FLOAT_LATTICE_STATES - 1u;
    }
    const uint32_t word = uint32_t(uniform * float(STRELKA_FLOAT_LATTICE_STATES));
    return word < STRELKA_FLOAT_LATTICE_STATES ? word : STRELKA_FLOAT_LATTICE_STATES - 1u;
}

STRELKA_DISCRETE_FN bool discreteFloatLatticeBernoulli(uint32_t word, float probability)
{
    return word < discreteFloatLatticeCount(probability);
}

#if !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__)
inline uint64_t discreteBucketStateCount(uint32_t count, uint32_t bucket)
{
    if (count == 0u || bucket >= count)
    {
        return 0u;
    }
    constexpr uint64_t stateCount = uint64_t{ 1 } << 32u;
    const auto ceilRatio = [count](uint64_t numerator) {
        return numerator == 0u ? uint64_t{ 0 } : uint64_t{ 1 } + (numerator - 1u) / uint64_t(count);
    };
    const uint64_t begin = ceilRatio(uint64_t(bucket) * stateCount);
    const uint64_t end = ceilRatio(uint64_t(bucket + 1u) * stateCount);
    return end - begin;
}
#endif

#undef STRELKA_DISCRETE_FN

#endif // STRELKA_DISCRETE_SAMPLING_H
