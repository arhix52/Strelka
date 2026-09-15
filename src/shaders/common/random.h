#pragma once
#include <stdint.h>

// source:
// https://github.com/mmp/pbrt-v4/blob/5acc5e46cf4b5c3382babd6a3b93b87f54d79b0a/src/pbrt/util/float.h#L46C1-L47C1
static constexpr float FloatOneMinusEpsilon = 0x1.fffffep-1;

__constant__ const unsigned int primeNumbers[32] = {
    2,  3,  5,  7,  11, 13, 17, 19, 23, 29,  31,  37,  41,  43,  47,  53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
};

enum struct SampleDimension : uint32_t
{
    ePixelX,
    ePixelY,
    // Independent random variables for each level of light selection.
    eLightId,
    eLightClass,
    eLightBucket,
    eLightAlias,
    eTriangleBucket,
    eTriangleAlias,
    eTime,
    eLightPointX,
    eLightPointY,
    eBSDF0,
    eBSDF1,
    eBSDF2,
    eBSDF3,
    eRussianRoulette,
    // Coverage test for MASK/BLEND surfaces. Its own dimension, in the position
    // Metal has it, so a transparent hit does not consume or correlate with the
    // BSDF or roulette draws.
    eOpacity,
    eLensU,
    eLensV,
    eFogDistance,
    eFogPhaseU,
    eFogPhaseV,
    // Russian roulette on a shadow ray's accumulated transmittance. Its own
    // dimension so that killing a ray that is already almost blocked does not
    // correlate with which light was chosen or where on it the point landed.
    eShadowRR,
    eSssChannel,
    eSssDistance,
    eSssPhaseU,
    eSssPhaseV,
    // Independent retry entropy for finite-precision light samples that fall
    // just outside their analytically defined support.
    eLightRetryU,
    eLightRetryV,
    eNUM_DIMENSIONS
};

struct SamplerState
{
    uint32_t seed;
    uint32_t sampleIdx;
    uint32_t depthAndBlueNoise;
};

#define kBlueNoiseTile 128u
#define kSamplerDepthMask 0xFFu

__device__ __inline__ uint32_t samplerDepth(const SamplerState& state)
{
    return state.depthAndBlueNoise & kSamplerDepthMask;
}

__device__ __inline__ void samplerAdvanceDepth(SamplerState& state)
{
    const uint32_t next = (samplerDepth(state) + 1u) & kSamplerDepthMask;
    state.depthAndBlueNoise = (state.depthAndBlueNoise & ~kSamplerDepthMask) | next;
}

__device__ __inline__ void samplerSetDepthWithoutBlueNoise(SamplerState& state, uint32_t depth)
{
    state.depthAndBlueNoise = depth & kSamplerDepthMask;
}

__device__ __inline__ bool samplerHasBlueNoiseRank(const SamplerState& state)
{
    return (state.depthAndBlueNoise >> 8u) != 0u;
}

static __device__ bool samplerBlueNoiseEnabled();

/// The pixel's rank as a shift in [0, 1). Only meaningful when
/// samplerHasBlueNoise() is true.
__device__ __inline__ float samplerBlueNoise(const SamplerState& state)
{
    const uint32_t rank = (state.depthAndBlueNoise >> 8u) - 1u;
    return ((float)rank + 0.5f) / (float)(kBlueNoiseTile * kBlueNoiseTile);
}

#define MAX_BOUNCES 128

// Based on: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
__device__ inline unsigned int pcg_hash(unsigned int seed)
{
    unsigned int state = seed * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __inline__ uint32_t hash_combine(uint32_t seed, uint32_t v)
{
    return seed ^ (v + (seed << 6) + (seed >> 2));
}

__device__ inline unsigned int hash_with(unsigned int seed, unsigned int hash)
{
    // Wang hash
    seed = (seed ^ 61u) ^ hash;
    seed += seed << 3;
    seed ^= seed >> 4;
    seed *= 0x27d4eb2du;
    return seed;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int& prev)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

// jenkins hash
static __device__ unsigned int jenkins_hash(unsigned int a)
{
  a = (a + 0x7ED55D16) + (a << 12);
  a = (a ^ 0xC761C23C) ^ (a >> 19);
  a = (a + 0x165667B1) + (a <<  5);
  a = (a + 0xD3A2646C) ^ (a <<  9);
  a = (a + 0xFD7046C5) + (a <<  3);
  a = (a ^ 0xB55A4F09) ^ (a >> 16);
  return a;
}

__device__ __inline__ uint32_t hash(uint32_t x)
{
    // finalizer from murmurhash3
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

static __device__ float halton(uint32_t index, uint32_t base)
{
    const float s = 1.0f / float(base);
    unsigned int i = index;
    float result = 0.0f;
    float f = s;
    while (i)
    {
        const unsigned int digit = i % base;
        result += f * float(digit);
        i = (i - digit) / base;
        f *= s;
    }
    return clamp(result, 0.0f, FloatOneMinusEpsilon);
}

// source https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// "Insert" a 0 bit after each of the 16 low bits of x
__device__ __inline__ uint32_t Part1By1(uint32_t x)
{
  x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}

__device__ __inline__ uint32_t EncodeMorton2(uint32_t x, uint32_t y)
{
  return (Part1By1(y) << 1) + Part1By1(x);
}

#include "bluenoise_mask.h"

// One sequence for the whole screen: the construction depends on the pixels
// sharing it, so this seed must not vary per pixel.
#define kBlueNoiseGlobalSeed 0x9e3779b9u
#define kGoldenRatioConjugate 0.61803398875f

__device__ __inline__ float blueNoiseShift(float bn, uint32_t dimension)
{
    const float v = bn + (float)dimension * kGoldenRatioConjugate;
    return v - floorf(v);
}

__device__ const unsigned char kQuadrantPermutations[24][4] = {
    { 0u, 1u, 2u, 3u },
    { 0u, 1u, 3u, 2u },
    { 0u, 2u, 1u, 3u },
    { 0u, 2u, 3u, 1u },
    { 0u, 3u, 1u, 2u },
    { 0u, 3u, 2u, 1u },
    { 1u, 0, 2u, 3u },
    { 1u, 0, 3u, 2u },
    { 1u, 2u, 0, 3u },
    { 1u, 2u, 3u, 0u },
    { 1u, 3u, 0, 2u },
    { 1u, 3u, 2u, 0u },
    { 2u, 0, 1u, 3u },
    { 2u, 0, 3u, 1u },
    { 2u, 1u, 0, 3u },
    { 2u, 1u, 3u, 0u },
    { 2u, 3u, 0, 1u },
    { 2u, 3u, 1u, 0u },
    { 3u, 0, 1u, 2u },
    { 3u, 0, 2u, 1u },
    { 3u, 1u, 0, 2u },
    { 3u, 1u, 2u, 0u },
    { 3u, 2u, 0, 1u },
    { 3u, 2u, 1u, 0u }
};

__device__ __inline__ uint32_t owenScrambleMorton(uint32_t morton, uint32_t levels, uint32_t seed)
{
    uint32_t out = 0u;
    uint32_t prefix = seed;
    for (uint32_t level = levels; level-- > 0u;)
    {
        const uint32_t digit = (morton >> (2u * level)) & 3u;
        const uint32_t which = hash(prefix) % 24u;
        out |= (uint32_t)kQuadrantPermutations[which][digit] << (2u * level);
        prefix = hash_combine(prefix, digit + 1u);
    }
    return out;
}

static __device__ SamplerState initSampler(uint32_t pixelX,
                                           uint32_t pixelY,
                                           uint32_t linearPixelIndex,
                                           uint32_t pixelSampleIndex,
                                           uint32_t maxSampleCount,
                                           uint32_t mortonLevels,
                                           uint32_t blueNoiseSwitch)
{
    SamplerState sampler{};
    if (pixelSampleIndex < blueNoiseSwitch)
    {
        sampler.seed = hash(linearPixelIndex);
        sampler.sampleIdx = pixelSampleIndex;
        const uint32_t cell = (pixelY % kBlueNoiseTile) * kBlueNoiseTile + (pixelX % kBlueNoiseTile);
        sampler.depthAndBlueNoise = ((uint32_t)kBlueNoiseRank[cell] + 1u) << 8u;
    }
    else
    {
        sampler.seed = 52u;
        sampler.sampleIdx = owenScrambleMorton(EncodeMorton2(pixelX, pixelY), mortonLevels, 0x5ed1a17bu) *
                                maxSampleCount +
                            (pixelSampleIndex - blueNoiseSwitch);
        sampler.depthAndBlueNoise = 0u;
    }
    return sampler;
}

__device__ __inline__ uint32_t ReverseBits(uint32_t value)
{
#ifdef __CUDACC__
    return __brev(value);
#else
    value = (((value & 0xaaaaaaaa) >> 1) | ((value & 0x55555555) << 1));
    value = (((value & 0xcccccccc) >> 2) | ((value & 0x33333333) << 2));
    value = (((value & 0xf0f0f0f0) >> 4) | ((value & 0x0f0f0f0f) << 4));
    value = (((value & 0xff00ff00) >> 8) | ((value & 0x00ff00ff) << 8));
    return ((value >> 16) | (value << 16));
#endif
}

__device__ __inline__ uint32_t sobol_dim0(uint32_t index)
{
    return ReverseBits(index);
}

__device__ __inline__ uint32_t sobol_dim1(uint32_t index)
{
    uint32_t x = ReverseBits(index);
    x ^= (x & 0x55555555u) << 1;
    x ^= (x & 0x33333333u) << 2;
    x ^= (x & 0x0f0f0f0fu) << 4;
    x ^= (x & 0x00ff00ffu) << 8;
    x ^= (x & 0x0000ffffu) << 16;
    return x;
}

__device__ __inline__ uint32_t laine_karras_permutation(uint32_t value, uint32_t seed)
{
    value ^= value * 0x3d20adeau;
    value += seed;
    value *= (seed >> 16) | 1u;
    value ^= value * 0x05526c56u;
    value ^= value * 0x53a22864u;
    return value;
}

__device__ __inline__ uint32_t nested_uniform_scramble(uint32_t value, uint32_t seed)
{
    value = ReverseBits(value);
    value = laine_karras_permutation(value, seed);
    value = ReverseBits(value);
    return value;
}

struct SampleSlot
{
    uint32_t decision;
    uint32_t axis;
};

__host__ __device__ constexpr SampleSlot sampleSlot(SampleDimension d)
{
    switch (d)
    {
    case SampleDimension::ePixelX:          return { 0u, 0u };
    case SampleDimension::ePixelY:          return { 0u, 1u };
    case SampleDimension::eLightId:         return { 1u, 0u };
    case SampleDimension::eLightClass:      return { 2u, 0u };
    case SampleDimension::eLightBucket:     return { 3u, 0u };
    case SampleDimension::eLightAlias:      return { 4u, 0u };
    case SampleDimension::eTriangleBucket:  return { 5u, 0u };
    case SampleDimension::eTriangleAlias:   return { 6u, 0u };
    case SampleDimension::eTime:            return { 7u, 0u };
    case SampleDimension::eLightPointX:     return { 8u, 0u };
    case SampleDimension::eLightPointY:     return { 8u, 1u };
    case SampleDimension::eBSDF0:           return { 9u, 0u };
    case SampleDimension::eBSDF1:           return { 9u, 1u };
    case SampleDimension::eBSDF2:           return { 10u, 0u };
    case SampleDimension::eBSDF3:           return { 10u, 1u };
    case SampleDimension::eRussianRoulette: return { 11u, 0u };
    case SampleDimension::eOpacity:         return { 12u, 0u };
    case SampleDimension::eLensU:           return { 13u, 0u };
    case SampleDimension::eLensV:           return { 13u, 1u };
    case SampleDimension::eFogDistance:     return { 14u, 0u };
    case SampleDimension::eFogPhaseU:       return { 15u, 0u };
    case SampleDimension::eFogPhaseV:       return { 15u, 1u };
    case SampleDimension::eShadowRR:        return { 16u, 0u };
    case SampleDimension::eSssChannel:      return { 17u, 0u };
    case SampleDimension::eSssDistance:     return { 18u, 0u };
    case SampleDimension::eSssPhaseU:       return { 19u, 0u };
    case SampleDimension::eSssPhaseV:       return { 19u, 1u };
    case SampleDimension::eLightRetryU:     return { 20u, 0u };
    case SampleDimension::eLightRetryV:     return { 20u, 1u };
    default:                                return { 21u, 0u };
    }
}

__device__ __inline__ uint32_t sobol_padded_bits(uint32_t index, SampleSlot slot, uint32_t seed)
{
    const uint32_t decisionSeed = hash(hash_combine(seed, slot.decision + 1u));
    const uint32_t shuffled = nested_uniform_scramble(index, decisionSeed);
    const uint32_t v = slot.axis == 0u ? sobol_dim0(shuffled) : sobol_dim1(shuffled);
    return nested_uniform_scramble(v, hash_combine(decisionSeed, slot.axis + 1u));
}

/// Categorical decisions consume the unrounded Owen-scrambled word: converting
/// to float first would discard the low bits and make tables larger than the
/// float mantissa impossible to cover.
template <SampleDimension Dim>
__device__ __inline__ uint32_t randomBits(SamplerState& state)
{
    constexpr SampleSlot slot = sampleSlot(Dim);
    const uint32_t depth = samplerDepth(state);
    const bool masked = depth == 0u && samplerBlueNoiseEnabled() && samplerHasBlueNoiseRank(state);
    const uint32_t seed = masked ? kBlueNoiseGlobalSeed : state.seed + depth;
    const uint32_t word = sobol_padded_bits(state.sampleIdx, slot, seed);
    if (!masked)
    {
        return word;
    }
    // Applied to the word rather than to a float, so a categorical decision sees
    // the same rotation the continuous draw would. 24 bits of it: the low 8 are
    // below the mask's resolution and carrying them would only add rounding.
    return word + (uint32_t(blueNoiseShift(samplerBlueNoise(state), uint32_t(Dim)) * 16777216.0f) << 8u);
}

template <SampleDimension Dim>
__device__ __inline__ float random(SamplerState& state)
{
    return min(randomBits<Dim>(state) * 0x1p-32f, FloatOneMinusEpsilon);
}
