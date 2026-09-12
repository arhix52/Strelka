#pragma once
#include <simd/simd.h>
#include "bluenoise_mask.h"

using namespace metal;

constant uint32_t kFcSamplerType [[function_constant(18)]];
constant uint32_t FIXED_SAMPLER_TYPE = is_function_constant_defined(kFcSamplerType) ? kFcSamplerType : 0xffffffffu;

// source: https://github.com/mmp/pbrt-v4
constant constexpr float FloatOneMinusEpsilon = 0x1.fffffep-1;

float uintToFloat(uint x)
{
    return as_type<float>(0x3f800000 | (x >> 9)) - 1.f;
}

float2 uintToFloat(uint2 x)
{
    return as_type<float2>(0x3f800000u | (x >> 9u)) - 1.0f;
}

float4 uintToFloat(uint4 x)
{
    return as_type<float4>(0x3f800000u | (x >> 9u)) - 1.0f;
}

enum class SampleDimension : uint32_t
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
    eTime, // motion blur time [0, 1]
    eLightPointX,
    eLightPointY,
    eBSDF0,
    eBSDF1,
    eBSDF2,
    eBSDF3,
    eRussianRoulette,
    // Coverage test for MASK/BLEND surfaces. Its own dimension so a transparent
    // hit does not consume, or correlate with, the BSDF or roulette draws.
    eOpacity,
    eLensU,
    eLensV,
    // Atmospheric scattering: the free-flight distance, and the two draws that
    // pick a direction out of the phase function. Their own dimensions for the
    // same reason eOpacity has one -- a scattering event must not correlate with
    // the BSDF draws of the surface the ray was heading for.
    eFogDistance,
    eFogPhaseU,
    eFogPhaseV,
    // Russian roulette on a shadow ray's accumulated transmittance. Its own
    // dimension so that killing a ray that is already almost blocked does not
    // correlate with which light was chosen or where on it the point landed.
    eShadowRR,
    // The subsurface random walk: which colour channel drives free flight, how far
    // it goes, and the phase-function draw at the scattering event.
    //
    // Distinct from the fog dimensions even though the two never scatter at the
    // same vertex, because the channel choice happens in `extend` and the phase
    // draw in `shade` at the same walk step -- sharing eFogPhaseU between them
    // would tie which channel was picked to which way the walk turned.
    //
    // Adding dimensions changes the stride in random<>(), so every scene's noise
    // is realised differently from here on. That moves the noise, not the image
    // the samples converge to.
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
    uint32_t depth;
    /// This pixel's position in the blue-noise mask, in [0, 1). Only the
    /// blue-noise samplers read it; it is one register, and computing it costs one
    /// table lookup at path start rather than one per dimension.
    float bn;
    /// Sample count at which the hybrid sampler hands over from the blue-noise
    /// sequence to the per-pixel scrambled one. Zero disables the handover.
    uint32_t bnSwitch;
};

// Based on: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
inline unsigned pcg_hash(unsigned seed)
{
    unsigned state = seed * 747796405u + 2891336453u;
    unsigned word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline unsigned hash_with(unsigned seed, unsigned hash)
{
    // Wang hash
    seed = (seed ^ 61) ^ hash;
    seed += seed << 3;
    seed ^= seed >> 4;
    seed *= 0x27d4eb2d;
    return seed;
}

// jenkins hash
unsigned int hash(unsigned int a)
{
    a = (a + 0x7ED55D16) + (a << 12);
    a = (a ^ 0xC761C23C) ^ (a >> 19);
    a = (a + 0x165667B1) + (a << 5);
    a = (a + 0xD3A2646C) ^ (a << 9);
    a = (a + 0xFD7046C5) + (a << 3);
    a = (a ^ 0xB55A4F09) ^ (a >> 16);
    return a;
}

inline uint32_t hash_combine(uint32_t seed, uint32_t v)
{
    return seed ^ (v + (seed << 6) + (seed >> 2));
}

constant unsigned int primeNumbers[32] = { 2,  3,  5,  7,  11, 13, 17, 19, 23, 29,  31,  37,  41,  43,  47,  53,
                                           59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131 };
float halton(uint32_t index, uint32_t base)
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
    return clamp(result, 0.0f, 1.0f - 1e-6f);
}

constant constexpr uint32_t kBlueNoiseTile = 128u;

inline uint32_t part1By1(uint32_t x)
{
    x &= 0x0000ffffu;
    x = (x ^ (x << 8u)) & 0x00ff00ffu;
    x = (x ^ (x << 4u)) & 0x0f0f0f0fu;
    x = (x ^ (x << 2u)) & 0x33333333u;
    x = (x ^ (x << 1u)) & 0x55555555u;
    return x;
}

inline uint32_t encodeMorton2(uint32_t x, uint32_t y)
{
    return (part1By1(y) << 1u) + part1By1(x);
}

static SamplerState initSampler(uint32_t linearPixelIndex,
                                uint32_t pixelSampleIndex,
                                uint32_t width,
                                uint32_t bnSwitch,
                                uint32_t sampleBlockBits,
                                uint32_t samplerType)
{
    SamplerState sampler{};
    samplerType = FIXED_SAMPLER_TYPE != 0xffffffffu ? FIXED_SAMPLER_TYPE : samplerType;
    const bool blueNoisePrefix = samplerType == 3u || (samplerType == 4u && pixelSampleIndex < bnSwitch);
    if (samplerType >= 2u && samplerType <= 4u && !blueNoisePrefix)
    {
        // One global sequence, split into per-pixel blocks along a Z-curve. A
        // complete block is stratified jointly with its neighbours; a per-pixel
        // seed loses that anti-correlation and measured much noisier on
        // kids_room after the sampler was padded to two dimensions.
        const uint32_t safeWidth = max(width, 1u);
        const uint32_t px = linearPixelIndex % safeWidth;
        const uint32_t py = linearPixelIndex / safeWidth;
        const uint32_t tailIndex = samplerType == 4u ? pixelSampleIndex - bnSwitch : pixelSampleIndex;
        const uint32_t blockMask = (1u << sampleBlockBits) - 1u;
        const uint32_t epoch = tailIndex >> sampleBlockBits;
        sampler.seed = epoch == 0u ? 52u : 52u + hash(epoch);
        sampler.sampleIdx = (encodeMorton2(px, py) << sampleBlockBits) | (tailIndex & blockMask);
    }
    else
    {
        sampler.seed = hash(linearPixelIndex);
        sampler.sampleIdx = pixelSampleIndex;
    }
    sampler.depth = 0;
    if (blueNoisePrefix)
    {
        const uint32_t safeWidth = max(width, 1u);
        const uint32_t px = linearPixelIndex % safeWidth;
        const uint32_t py = linearPixelIndex / safeWidth;
        const uint32_t cell = (py % kBlueNoiseTile) * kBlueNoiseTile + (px % kBlueNoiseTile);
        sampler.bn = (float(kBlueNoiseRank[cell]) + 0.5f) / float(kBlueNoiseTile * kBlueNoiseTile);
    }
    // A tail index is already rebased and blocked above; prevent randomHybrid
    // from subtracting the switch a second time.
    sampler.bnSwitch = blueNoisePrefix ? bnSwitch : 0u;
    return sampler;
}

// The primary blue-noise prefix deliberately shares one Sobol sequence across
// the screen. Pixel decorrelation comes from bn, so no per-pixel scramble seed
// is constructed on this path.
static SamplerState initPrimaryBlueNoiseSampler(uint32_t linearPixelIndex,
                                                uint32_t pixelSampleIndex,
                                                uint32_t width,
                                                uint32_t bnSwitch)
{
    SamplerState sampler{};
    sampler.seed = 0u;
    sampler.sampleIdx = pixelSampleIndex;
    sampler.depth = 0u;
    const uint32_t safeWidth = max(width, 1u);
    const uint32_t px = linearPixelIndex % safeWidth;
    const uint32_t py = linearPixelIndex / safeWidth;
    const uint32_t cell = (py % kBlueNoiseTile) * kBlueNoiseTile + (px % kBlueNoiseTile);
    sampler.bn = (float(kBlueNoiseRank[cell]) + 0.5f) / float(kBlueNoiseTile * kBlueNoiseTile);
    sampler.bnSwitch = bnSwitch;
    return sampler;
}

template <SampleDimension Dim>
static float randomHalton(thread SamplerState& state)
{
    const uint32_t dimension = uint32_t(Dim) + state.depth * uint32_t(SampleDimension::eNUM_DIMENSIONS);
    const uint32_t base = primeNumbers[dimension & 31u];
    // Only 32 bases exist, so dimension 32 reuses base(0), 33 reuses base(1), etc.
    // With a shared sequence index that made e.g. eBSDF0@depth2 return exactly the
    // same number as ePixelX@depth0, correlating the pixel filter with a BSDF
    // lobe choice. Offsetting the sequence index per dimension breaks the tie.
    return halton(state.seed + state.sampleIdx + hash(dimension), base);
}

template <SampleDimension Dim>
static uint32_t randomPCGBits(thread SamplerState& state)
{
    const uint32_t dimension = uint32_t(Dim) + state.depth * uint32_t(SampleDimension::eNUM_DIMENSIONS);
    uint32_t h = hash_with(state.seed + state.sampleIdx, dimension);
    return pcg_hash(h);
}

template <SampleDimension Dim>
static float randomPCG(thread SamplerState& state)
{
    return uintToFloat(randomPCGBits<Dim>(state));
}

// ── Sobol + Owen scrambling (ported from OptiX random.h) ────────────────────

// Ablation: Owen scrambling with van der Corput only -- no sb_matrix reads.
// Dimension 0 of the Joe-Kuo table is van der Corput; Owen's per-dimension seed
// still separates draws. Not bit-identical to Sobol; isolates table load cost.
inline uint32_t sobol_uint_notable(uint32_t index, uint32_t dim)
{
    (void)dim;
    return reverse_bits(index);
}

// Owen scrambling needs every output bit to depend on lower input bits; multiplicative seed mixing
// decorrelates consecutive dimension seeds used by the screen-wide blue-noise samplers.
inline uint32_t laine_karras_permutation(uint32_t value, uint32_t seed)
{
    value ^= value * 0x3d20adeau;
    value += seed;
    value *= (seed >> 16) | 1u;
    value ^= value * 0x05526c56u;
    value ^= value * 0x53a22864u;
    return value;
}

inline uint32_t nested_uniform_scramble(uint32_t value, uint32_t seed)
{
    value = reverse_bits(value);
    value = laine_karras_permutation(value, seed);
    value = reverse_bits(value);
    return value;
}

inline uint2 nested_uniform_scramble(uint2 value, uint2 seed)
{
    value = reverse_bits(value);
    value ^= value * 0x3d20adeau;
    value += seed;
    value *= (seed >> 16u) | 1u;
    value ^= value * 0x05526c56u;
    value ^= value * 0x53a22864u;
    return reverse_bits(value);
}

// Sobol' dimensions 0 and 1 in closed form. A padded sampler only needs this
// pair, so the 32 KB direction-number table and its data-dependent gather can
// stay out of the hot path entirely.
inline uint32_t sobol_dim0(uint32_t index)
{
    return reverse_bits(index);
}

inline uint32_t sobol_dim1(uint32_t index)
{
    uint32_t x = reverse_bits(index);
    x ^= (x & 0x55555555u) << 1u;
    x ^= (x & 0x33333333u) << 2u;
    x ^= (x & 0x0f0f0f0fu) << 4u;
    x ^= (x & 0x00ff00ffu) << 8u;
    x ^= (x & 0x0000ffffu) << 16u;
    return x;
}

// A path is a list of sampling decisions. Genuinely two-dimensional decisions
// share a (0, 2)-sequence; unrelated categorical decisions get independent
// Owen scrambles instead of pretending to form a meaningful 2D projection.
struct SampleSlot
{
    uint32_t decision;
    uint32_t axis;
};

constexpr SampleSlot sampleSlot(SampleDimension d)
{
    switch (d)
    {
    case SampleDimension::ePixelX:
        return { 0u, 0u };
    case SampleDimension::ePixelY:
        return { 0u, 1u };
    case SampleDimension::eLightId:
        return { 1u, 0u };
    case SampleDimension::eLightClass:
        return { 2u, 0u };
    case SampleDimension::eLightBucket:
        return { 3u, 0u };
    case SampleDimension::eLightAlias:
        return { 4u, 0u };
    case SampleDimension::eTriangleBucket:
        return { 5u, 0u };
    case SampleDimension::eTriangleAlias:
        return { 6u, 0u };
    case SampleDimension::eTime:
        return { 7u, 0u };
    case SampleDimension::eLightPointX:
        return { 8u, 0u };
    case SampleDimension::eLightPointY:
        return { 8u, 1u };
    case SampleDimension::eBSDF0:
        return { 9u, 0u };
    case SampleDimension::eBSDF1:
        return { 9u, 1u };
    case SampleDimension::eBSDF2:
        return { 10u, 0u };
    case SampleDimension::eBSDF3:
        return { 10u, 1u };
    case SampleDimension::eRussianRoulette:
        return { 11u, 0u };
    case SampleDimension::eOpacity:
        return { 12u, 0u };
    case SampleDimension::eLensU:
        return { 13u, 0u };
    case SampleDimension::eLensV:
        return { 13u, 1u };
    case SampleDimension::eFogDistance:
        return { 14u, 0u };
    case SampleDimension::eFogPhaseU:
        return { 15u, 0u };
    case SampleDimension::eFogPhaseV:
        return { 15u, 1u };
    case SampleDimension::eShadowRR:
        return { 16u, 0u };
    case SampleDimension::eSssChannel:
        return { 17u, 0u };
    case SampleDimension::eSssDistance:
        return { 18u, 0u };
    case SampleDimension::eSssPhaseU:
        return { 19u, 0u };
    case SampleDimension::eSssPhaseV:
        return { 19u, 1u };
    case SampleDimension::eLightRetryU:
        return { 20u, 0u };
    case SampleDimension::eLightRetryV:
        return { 20u, 1u };
    default:
        return { 21u, 0u };
    }
}

inline uint32_t sobol_decision_seed(uint32_t seed, uint32_t decision)
{
    uint32_t value = hash_combine(seed, decision + 1u);
    value ^= value >> 16u;
    value *= 0x85ebca6bu;
    value ^= value >> 13u;
    value *= 0xc2b2ae35u;
    value ^= value >> 16u;
    return value;
}

inline uint32_t sobol_padded_bits(uint32_t index, SampleSlot slot, uint32_t seed)
{
    // Scrambling only the output leaves any two decisions on a bijective curve.
    // The per-decision index shuffle is required for their joint projection to
    // cover the square; omitting it made the OptiX kids_room result 8.2% dark.
    const uint32_t decisionSeed = sobol_decision_seed(seed, slot.decision);
    const uint32_t shuffled = nested_uniform_scramble(index, decisionSeed);
    const uint32_t value = slot.axis == 0u ? sobol_dim0(shuffled) : sobol_dim1(shuffled);
    return nested_uniform_scramble(value, hash_combine(decisionSeed, slot.axis + 1u));
}

template <SampleDimension Dim>
inline uint32_t sobol_padded_bits(uint32_t index, uint32_t seed)
{
    constexpr SampleSlot slot = sampleSlot(Dim);
    return sobol_padded_bits(index, slot, seed);
}

template <SampleDimension Dim0, SampleDimension Dim1>
inline uint2 sobol_padded_bits2(uint32_t index, uint32_t seed)
{
    constexpr SampleSlot slot0 = sampleSlot(Dim0);
    constexpr SampleSlot slot1 = sampleSlot(Dim1);
    if (slot0.decision != slot1.decision)
    {
        return uint2(sobol_padded_bits(index, slot0, seed), sobol_padded_bits(index, slot1, seed));
    }

    // The two axes of one decision share the expensive index shuffle. This is
    // the useful part of the old random2 batching, retained without the table.
    const uint32_t decisionSeed = sobol_decision_seed(seed, slot0.decision);
    const uint32_t shuffled = nested_uniform_scramble(index, decisionSeed);
    const uint32_t axis0 = sobol_dim0(shuffled);
    const uint32_t axis1 = sobol_dim1(shuffled);
    const uint2 value = uint2(slot0.axis == 0u ? axis0 : axis1, slot1.axis == 0u ? axis0 : axis1);
    return nested_uniform_scramble(
        value, uint2(hash_combine(decisionSeed, slot0.axis + 1u), hash_combine(decisionSeed, slot1.axis + 1u)));
}

inline uint32_t sobol_scramble_bits_notable(uint32_t index, uint32_t matrixIndex, uint32_t scrambleDim, uint32_t seed)
{
    seed = hash(seed);
    index = nested_uniform_scramble(index, seed);
    return nested_uniform_scramble(sobol_uint_notable(index, matrixIndex), hash_combine(seed, scrambleDim));
}

template <SampleDimension Dim>
static float randomSobol(thread SamplerState& state)
{
    return min(sobol_padded_bits<Dim>(state.sampleIdx, state.seed + state.depth) * 0x1p-32f, FloatOneMinusEpsilon);
}

template <SampleDimension Dim>
static uint32_t randomSobolBits(thread SamplerState& state)
{
    return sobol_padded_bits<Dim>(state.sampleIdx, state.seed + state.depth);
}

template <SampleDimension Dim>
static float randomSobolNoTable(thread SamplerState& state)
{
    const uint32_t dimension = uint32_t(Dim) + state.depth * uint32_t(SampleDimension::eNUM_DIMENSIONS);
    return min(
        sobol_scramble_bits_notable(state.sampleIdx, dimension % 256u, dimension, state.seed + state.depth) * 0x1p-32f,
        FloatOneMinusEpsilon);
}

template <SampleDimension Dim>
static uint32_t randomSobolNoTableBits(thread SamplerState& state)
{
    const uint32_t dimension = uint32_t(Dim) + state.depth * uint32_t(SampleDimension::eNUM_DIMENSIONS);
    return sobol_scramble_bits_notable(state.sampleIdx, dimension % 256u, dimension, state.seed + state.depth);
}

// ── Sobol with a blue-noise screen-space error distribution ─────────────────
//
// randomSobol gives every pixel its own scramble, so the error at neighbouring
// pixels is independent: white noise. Total error is right, but white noise is
// the *worst* spectrum to look at and the worst for any reconstruction filter to
// remove, because it puts as much energy at low frequencies -- where the eye is
// sensitive and where blurring cannot reach -- as at high ones.
//
// The alternative (Georgiev & Fajardo 2016; Heitz & Belcour 2019): let every
// pixel draw the *same* point set, and give each pixel a toroidal shift of it.
// The error is then a smooth function of that pixel's shift, so the error field
// inherits the spectrum of the shift field. A blue-noise shift field therefore
// buys a blue-noise error field -- the same total error, moved into the high
// frequencies that a filter and the eye both discard.
//
// This is a low-sample-count technique and does not pretend otherwise: the shift
// is a rotation, and a rotation is a weaker randomisation than a scramble for
// the discontinuous integrands a path tracer actually has. Past a few dozen
// samples per pixel randomSobol converges faster. See randomHybrid.

constant constexpr float kGoldenRatioConjugate = 0.61803398875f;
// One sequence for the whole screen: the construction depends on the pixels
// sharing it, so this seed must not vary per pixel.
constant constexpr uint32_t kBlueNoiseGlobalSeed = 0x9e3779b9u;

inline float blueNoiseShift(float bn, uint32_t dimension)
{
    // One mask, advanced per dimension along an additive recurrence. The golden
    // ratio's continued fraction makes it the slowest-approximated irrational,
    // so successive dimensions are as far apart as an additive step can put
    // them -- and an additive step, unlike a hash, leaves the mask's spatial
    // spectrum intact, which is the whole point of using the mask.
    return fract(bn + float(dimension) * kGoldenRatioConjugate);
}

// Shift only primary dimensions so error varies smoothly with the mask; deeper paths hash away its spatial spectrum.
template <SampleDimension Dim>
static float randomSobolBlueNoise(thread SamplerState& state)
{
    if (state.depth != 0u)
    {
        return randomSobol<Dim>(state);
    }
    // One sequence shared by the whole screen -- the pixels have to be drawing
    // the same points for their shifts to be comparable.
    const uint32_t dimension = uint32_t(Dim);
    const float v = min(sobol_padded_bits<Dim>(state.sampleIdx, kBlueNoiseGlobalSeed) * 0x1p-32f, FloatOneMinusEpsilon);
    return fract(v + blueNoiseShift(state.bn, dimension));
}

template <SampleDimension Dim>
static uint32_t randomSobolBlueNoiseBits(thread SamplerState& state)
{
    if (state.depth != 0u)
    {
        return randomSobolBits<Dim>(state);
    }
    const uint32_t dimension = uint32_t(Dim);
    const uint32_t word = sobol_padded_bits<Dim>(state.sampleIdx, kBlueNoiseGlobalSeed);
    const uint32_t shift = uint32_t(blueNoiseShift(state.bn, dimension) * 16777216.0f) << 8u;
    return word + shift;
}

// Blue noise while the frame is young, per-pixel scrambling once it is not.
//
// The two are unbiased estimates of the same integral, so an accumulator can
// average across the handover without correcting anything. The second stage
// restarts its sequence index at zero rather than continuing from bnSwitch, so
// it gets a whole stratified block instead of the tail of one.
template <SampleDimension Dim>
static float randomHybrid(thread SamplerState& state)
{
    if (state.sampleIdx < state.bnSwitch)
    {
        return randomSobolBlueNoise<Dim>(state);
    }
    SamplerState tail = state;
    tail.sampleIdx = state.sampleIdx - state.bnSwitch;
    return randomSobol<Dim>(tail);
}

template <SampleDimension Dim>
static uint32_t randomHybridBits(thread SamplerState& state)
{
    if (state.sampleIdx < state.bnSwitch)
    {
        return randomSobolBlueNoiseBits<Dim>(state);
    }
    SamplerState tail = state;
    tail.sampleIdx = state.sampleIdx - state.bnSwitch;
    return randomSobolBits<Dim>(tail);
}

// ── Sampler dispatch ────────────────────────────────────────────────────────
// 0 = Halton, 1 = PCG, 2 = Sobol (Owen scrambled), 3 = Sobol + blue noise,
// 4 = hybrid (3 below bnSwitch samples, 2 above), 5 = Owen + VDC (no table, ablation)

template <SampleDimension Dim>
static float random(thread SamplerState& state, uint32_t samplerType)
{
    samplerType = FIXED_SAMPLER_TYPE != 0xffffffffu ? FIXED_SAMPLER_TYPE : samplerType;
    if (samplerType == 5)
        return randomSobolNoTable<Dim>(state);
    if (samplerType == 4)
        return randomHybrid<Dim>(state);
    if (samplerType == 3)
        return randomSobolBlueNoise<Dim>(state);
    if (samplerType == 2)
        return randomSobol<Dim>(state);
    if (samplerType == 1)
        return randomPCG<Dim>(state);
    return randomHalton<Dim>(state);
}

template <SampleDimension Dim>
static uint32_t randomBits(thread SamplerState& state, uint32_t samplerType)
{
    samplerType = FIXED_SAMPLER_TYPE != 0xffffffffu ? FIXED_SAMPLER_TYPE : samplerType;
    if (samplerType == 5)
        return randomSobolNoTableBits<Dim>(state);
    if (samplerType == 4)
        return randomHybridBits<Dim>(state);
    if (samplerType == 3)
        return randomSobolBlueNoiseBits<Dim>(state);
    if (samplerType == 2)
        return randomSobolBits<Dim>(state);
    // Halton is constructed as a float radical inverse and has no full-width
    // word to preserve. Use the same dimensioned PCG permutation as sampler 1
    // for categorical decisions; continuous Halton dimensions are unchanged.
    return randomPCGBits<Dim>(state);
}

struct RandomSample4
{
    float4 value;
    uint4 bits;
};

struct RandomSample2
{
    float2 value;
    uint2 bits;
};

template <SampleDimension Dim0, SampleDimension Dim1>
static RandomSample2 random2(thread SamplerState& state, uint32_t samplerType)
{
    samplerType = FIXED_SAMPLER_TYPE != 0xffffffffu ? FIXED_SAMPLER_TYPE : samplerType;
    if (samplerType == 1u)
    {
        RandomSample2 result;
        result.bits = uint2(randomPCGBits<Dim0>(state), randomPCGBits<Dim1>(state));
        result.value = uintToFloat(result.bits);
        return result;
    }
    if (samplerType == 5u)
    {
        RandomSample2 result;
        result.bits = uint2(randomSobolNoTableBits<Dim0>(state), randomSobolNoTableBits<Dim1>(state));
        result.value = min(float2(result.bits) * 0x1p-32f, float2(FloatOneMinusEpsilon));
        return result;
    }
    if (samplerType >= 2u && samplerType <= 4u)
    {
        uint32_t index = state.sampleIdx;
        uint32_t seed = state.seed + state.depth;
        bool useBlueNoise = samplerType == 3u && state.depth == 0u;
        if (samplerType == 4u)
        {
            const bool inBlueNoisePrefix = state.sampleIdx < state.bnSwitch;
            useBlueNoise = inBlueNoisePrefix && state.depth == 0u;
            if (!inBlueNoisePrefix)
            {
                index -= state.bnSwitch;
            }
        }
        if (useBlueNoise)
        {
            seed = kBlueNoiseGlobalSeed;
        }

        const uint2 words = sobol_padded_bits2<Dim0, Dim1>(index, seed);

        RandomSample2 result;
        result.bits = words;
        result.value = min(float2(words) * 0x1p-32f, float2(FloatOneMinusEpsilon));
        if (useBlueNoise)
        {
            const uint2 dimensions = uint2(uint32_t(Dim0), uint32_t(Dim1));
            const float2 shifts = float2(blueNoiseShift(state.bn, dimensions.x), blueNoiseShift(state.bn, dimensions.y));
            result.value = fract(result.value + shifts);
            result.bits += uint2(shifts * 16777216.0f) << 8u;
        }
        return result;
    }

    RandomSample2 result;
    result.value = float2(random<Dim0>(state, samplerType), random<Dim1>(state, samplerType));
    result.bits = uint2(randomBits<Dim0>(state, samplerType), randomBits<Dim1>(state, samplerType));
    return result;
}

template <SampleDimension Dim0, SampleDimension Dim1, SampleDimension Dim2, SampleDimension Dim3>
static RandomSample4 random4(thread SamplerState& state, uint32_t samplerType)
{
    samplerType = FIXED_SAMPLER_TYPE != 0xffffffffu ? FIXED_SAMPLER_TYPE : samplerType;
    if (samplerType == 1u)
    {
        RandomSample4 result;
        result.bits = uint4(randomPCGBits<Dim0>(state), randomPCGBits<Dim1>(state), randomPCGBits<Dim2>(state),
                            randomPCGBits<Dim3>(state));
        result.value = uintToFloat(result.bits);
        return result;
    }
    if (samplerType == 5u)
    {
        RandomSample4 result;
        result.bits = uint4(randomSobolNoTableBits<Dim0>(state), randomSobolNoTableBits<Dim1>(state),
                            randomSobolNoTableBits<Dim2>(state), randomSobolNoTableBits<Dim3>(state));
        result.value = min(float4(result.bits) * 0x1p-32f, float4(FloatOneMinusEpsilon));
        return result;
    }
    if (samplerType >= 2u && samplerType <= 4u)
    {
        uint32_t index = state.sampleIdx;
        uint32_t seed = state.seed + state.depth;
        bool useBlueNoise = samplerType == 3u && state.depth == 0u;
        if (samplerType == 4u)
        {
            const bool inBlueNoisePrefix = state.sampleIdx < state.bnSwitch;
            useBlueNoise = inBlueNoisePrefix && state.depth == 0u;
            if (!inBlueNoisePrefix)
            {
                index -= state.bnSwitch;
            }
        }
        if (useBlueNoise)
        {
            seed = kBlueNoiseGlobalSeed;
        }

        const uint4 words =
            uint4(sobol_padded_bits2<Dim0, Dim1>(index, seed), sobol_padded_bits2<Dim2, Dim3>(index, seed));

        RandomSample4 result;
        result.bits = words;
        result.value = min(float4(words) * 0x1p-32f, float4(FloatOneMinusEpsilon));
        if (useBlueNoise)
        {
            const uint4 dimensions = uint4(uint32_t(Dim0), uint32_t(Dim1), uint32_t(Dim2), uint32_t(Dim3));
            const float4 shifts = float4(blueNoiseShift(state.bn, dimensions.x), blueNoiseShift(state.bn, dimensions.y),
                                         blueNoiseShift(state.bn, dimensions.z), blueNoiseShift(state.bn, dimensions.w));
            result.value = fract(result.value + shifts);
            result.bits += uint4(shifts * 16777216.0f) << 8u;
        }
        return result;
    }

    RandomSample4 result;
    result.value = float4(random<Dim0>(state, samplerType), random<Dim1>(state, samplerType),
                          random<Dim2>(state, samplerType), random<Dim3>(state, samplerType));
    result.bits = uint4(randomBits<Dim0>(state, samplerType), randomBits<Dim1>(state, samplerType),
                        randomBits<Dim2>(state, samplerType), randomBits<Dim3>(state, samplerType));
    return result;
}
