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
    // Atmospheric scattering: the free-flight distance, and the two draws that
    // pick a direction out of the phase function. Their own dimensions for the
    // same reason eOpacity has one -- a scattering event must not correlate with
    // the BSDF draws of the surface the ray was heading for. Metal's positions.
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
    /// Bounce count in the low 8 bits, the pixel's blue-noise rank plus one in
    /// the rest (zero when the mask is off).
    ///
    /// Packed rather than given a word of its own because SamplerState lives in
    /// PerRayData, whose 136 bytes are continuation-stack ABI and are charged
    /// per thread: three extra words there measured 4% on kids_room, and a
    /// register cache of the same state measured worse still (docs/open-perf.md).
    /// The rank needs 14 bits and the depth 8 of the 32, so nothing here is
    /// tight.
    ///
    /// Named so that the old `state.depth` no longer compiles. A packed field
    /// that still answers to its unpacked name is the kind of change that reads
    /// as working and returns a bounce count with a mask rank on top of it.
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

/// Set the dimension offset and drop the blue-noise rank together.
///
/// The two go together because the mask only ever applies to a path's primary
/// draws. A caller that re-seeds a sampler to decorrelate sub-steps -- a medium
/// walk does exactly this -- would otherwise still take the blue-noise branch on
/// a step that lands at depth 0, and that branch ignores the seed: every step of
/// the walk would draw the same number and the re-seeding would do nothing.
__device__ __inline__ void samplerSetDepthWithoutBlueNoise(SamplerState& state, uint32_t depth)
{
    state.depthAndBlueNoise = depth & kSamplerDepthMask;
}

/// Whether this path's primary draws take the mask.
///
/// Spelled through a macro the backend may override because the branch is not
/// free: it inlines into every one of the ~117 draws a path makes, and the
/// launch is issue-bound -- 255 registers, four warps per scheduler, 86-90% of
/// cycles with no eligible warp. Leaving it as a runtime test measured 3-9%
/// across four scenes *with the mask off*, on output that is bit-identical to
/// not having it. Given a compile-time constant instead, OptiX folds the whole
/// blue-noise path out and the cost is zero; see STRELKA_SAMPLER_BLUE_NOISE in
/// the OptiX modules, which ands this with a bound value.
__device__ __inline__ bool samplerHasBlueNoiseRank(const SamplerState& state)
{
    return (state.depthAndBlueNoise >> 8u) != 0u;
}

/// Whether this module was compiled with the mask at all.
///
/// Declared here and defined by each OptiX module, because that is where
/// `params` is visible -- OptixRenderParams.h includes this header for
/// SamplerState, so the launch parameters cannot be in scope at this point. The
/// definition reads a bound value, which is the whole reason for the indirection:
/// with it constant the compiler deletes the blue-noise path outright, and a
/// runtime test in its place measured 3-9% on four scenes while producing
/// bit-identical images.
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

// __device__ inline unsigned hash_combine(unsigned a, unsigned b)
// {
//     return a ^ (b + 0x9e3779b9 + (a << 6) + (a >> 2));
// }

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

// ── Blue-noise screen-space error distribution ──────────────────────────────
//
// Per-pixel scrambling gives every pixel an independent estimate, so the error
// between neighbours is white noise. The total error is right, but white is the
// worst spectrum both to look at and to filter: it puts as much energy at the
// low frequencies the eye is sensitive to, and that no blur can reach, as at the
// high ones.
//
// The alternative (Georgiev & Fajardo 2016; Heitz & Belcour 2019): let the whole
// screen draw one point set, and give each pixel a toroidal shift of it taken
// from a blue-noise mask. The estimator's error is a smooth function of that
// shift, so a shift field with a blue-noise spectrum produces an error field
// with one -- the same total error, moved to the frequencies that are cheapest
// to hide.
//
// This is a low-sample-count technique and does not pretend otherwise: a
// rotation is a weaker randomisation than a scramble for the discontinuous
// integrands a path tracer actually has, and past a few dozen samples per pixel
// plain per-pixel scrambling converges faster. That is why it is a switch and
// not the default, and why a headless still frame leaves it off.
#include "bluenoise_mask.h"

// One sequence for the whole screen: the construction depends on the pixels
// sharing it, so this seed must not vary per pixel.
#define kBlueNoiseGlobalSeed 0x9e3779b9u
#define kGoldenRatioConjugate 0.61803398875f

__device__ __inline__ float blueNoiseShift(float bn, uint32_t dimension)
{
    // One mask, advanced per dimension along an additive recurrence. The golden
    // ratio's continued fraction makes it the slowest-approximated irrational,
    // so successive dimensions are as far apart as an additive step can put
    // them -- and an additive step, unlike a hash, leaves the mask's spatial
    // spectrum intact, which is the whole point of using the mask.
    const float v = bn + (float)dimension * kGoldenRatioConjugate;
    return v - floorf(v);
}

// Owen scrambling of the pixel's Morton index, in base 4.
//
// The sampler hands each pixel a block of one global sequence -- pixel i gets
// samples [i*spp, (i+1)*spp) -- and orders the pixels along a Z-curve so that
// neighbours on screen get neighbouring blocks. That ordering is what makes the
// error blue without anything optimising it to be: any 2^m prefix of a
// (0,2)-sequence is a stratified grid, so the four pixels of a 2x2 quad hold
// between them the four child blocks of one parent block, and their sample sets
// complete each other. Where one pixel leaves a gap its neighbour has a point,
// which is anti-correlated error, which is high-frequency error.
//
// What the Z-curve also has is regularity, and that shows up as a visible
// pattern. Permuting the four quadrants at every level removes it while keeping
// every locality relation the argument above depends on: a quad still maps to a
// quad, siblings stay siblings, only the labels move. Four quadrants rather than
// two halves is why this is base 4 and not the base-2 scramble the sample index
// already gets.
//
// `levels` is how many levels of the curve the image actually uses; the digits
// above it are zero and scrambling them would push the block index past what
// `* maxSampleCount` can hold.
/// The 24 permutations of four quadrant labels, for the scramble below.
///
/// 96 bytes, read once per level of the curve when a path starts -- eleven times
/// at 1280x720 -- and not once per draw. That is the distinction that matters
/// after the direction-number table came out: a table in the sampler's inner
/// loop cost 9-19% of the frame, a table read eleven times per path costs
/// nothing measurable.
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
        // A full permutation of the four quadrants, not a two-bit XOR. The XOR
        // reaches four of the twenty-four and is what Ahmed and Wonka's
        // hierarchical scramble is a proper version of; measured against it here
        // the difference was nothing, which is worth recording as a result
        // rather than as a reason to keep the weaker one.
        //
        // The permutation has to depend on the digits above this one and on
        // nothing below, which is what makes the scramble nested rather than a
        // hash of the whole index -- a hash would break the sibling relation
        // that the anti-correlation rests on.
        const uint32_t which = hash(prefix) % 24u;
        out |= (uint32_t)kQuadrantPermutations[which][digit] << (2u * level);
        prefix = hash_combine(prefix, digit + 1u);
    }
    return out;
}


// Where a pixel's decorrelation lives, and why it is in the seed rather than in
// the index.
//
// This used to read `sampleIdx = EncodeMorton2(x, y) * maxSampleCount + sample`,
// with a constant seed, so the whole per-pixel difference sat inside the Sobol'
// index. Two things followed, and both are why it changed.
//
// The index is Owen-scrambled before `sobol_uint` walks it, so a per-pixel index
// makes the walk's row -- `lowestSetBit(bits)` -- differ in every lane. The walk
// then scatters ~16 dependent loads across the table instead of reading one row,
// and the warp waits for whichever lane has the most set bits. `sb_matrix` is
// laid out `[bit][dimension]` precisely so that a warp on one row reads adjacent
// words, and that premise was false. Replacing the table with arithmetic and
// leaving everything else alone measured 1.1-2.4 ms of a 1280x720 frame,
// 9-19% (docs/open-perf.md).
//
// The second is that a screen-wide toroidal shift is impossible while the pixel
// is in the index: the construction needs every pixel drawing the *same* point
// set, and a global scramble seed cannot deliver that if the index already
// differs. Blue noise had no way to exist here.
//
// Matching Metal's `initSampler` fixes both, and makes the two backends' samplers
// the same construction, which is what the parity column is for.
///
/// `blueNoiseSwitch` is how many of a pixel's first samples take the mask: 0
/// turns it off, a count runs blue noise for that prefix and per-pixel
/// scrambling afterwards, and 0xffffffff never switches. The handover is decided
/// here, once per path, rather than inside every draw -- the two are unbiased
/// estimates of the same integral, so an accumulator can average across it
/// without correcting anything, and nothing but the sampler's own construction
/// needs to know which side of it this sample fell on. The tail restarts its
/// index at zero so it gets a whole stratified block rather than the end of one.
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
        // The screen shares one sequence, which is what makes the shifts
        // comparable, so the index must not carry the pixel. The seed still
        // must: the mask only shifts depth 0, and every dimension past it falls
        // back to per-pixel scrambling -- a global seed there would have every
        // pixel tracing the same indirect path.
        sampler.seed = hash(linearPixelIndex);
        sampler.sampleIdx = pixelSampleIndex;
        const uint32_t cell = (pixelY % kBlueNoiseTile) * kBlueNoiseTile + (pixelX % kBlueNoiseTile);
        sampler.depthAndBlueNoise = ((uint32_t)kBlueNoiseRank[cell] + 1u) << 8u;
    }
    else
    {
        // The construction this backend has always used, kept bit-exact for the
        // default sampler. Moving the pixel from the index into the seed to
        // match Metal reads like the obvious tidy-up and measured 7-9% *slower*
        // on all four scenes: the Sobol' index is Owen-scrambled with the seed,
        // so a per-pixel seed leaves the walk exactly as divergent as a
        // per-pixel index did, and makes the value scramble divergent too --
        // which it was not before. Parity of the two constructions is worth
        // having, but not at that price and not without a measurement that says
        // where the cost went.
        // One global sequence, blocked per pixel along a scrambled Z-curve. The
        // seed is a constant on purpose: the pixels have to be scrambling the
        // *same* sequence for their blocks to complete each other, and a
        // per-pixel seed here would make neighbouring errors independent --
        // white noise -- which is the property this construction exists to
        // avoid. Measured: giving it one costs the arrangement entirely.
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

// Sobol' dimensions 0 and 1, closed form, no direction numbers stored.
//
// The 256-dimension Joe-Kuo table that used to live here was 32 KB gathered from
// memory on every draw, and the gather was the sampler's whole cost: replacing
// it with arithmetic and changing nothing else measured 9-19% of a 1280x720
// frame (docs/open-perf.md). It existed because the sampler climbed to
// dimension `Dim + depth * eNUM_DIMENSIONS` -- 117 of them on a depth-8 path.
// It no longer does; see sampleSlot() below.
//
// Two dimensions is all a padded sampler ever needs, and both have a closed
// form. Checked against the columns they replace, all 32 rows each:
//
//   dimension 0: direction numbers are 1 << (31 - k), so the walk is a bit
//                reversal and nothing else.
//   dimension 1: direction numbers are v0 = 1 << 31, vk = vk-1 ^ (vk-1 >> 1),
//                i.e. Pascal's triangle mod 2. That matrix is the fifth
//                Kronecker power of [[1,0],[1,1]], so the product decomposes
//                into five masked shift-XOR stages -- verified exhaustively on
//                the 32 basis vectors, which by linearity settles every index,
//                and again on 20000 random ones.
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

/// Which decision a draw belongs to, and which of the two Sobol' dimensions it
/// takes.
///
/// The sampler used to give every draw its own dimension, `Dim + depth *
/// eNUM_DIMENSIONS`, which reaches 117 on a depth-8 path and is the only reason
/// a 256-dimension table had to exist. It pads instead: a path is a list of
/// *decisions*, every decision draws from the same two-dimensional sequence, and
/// what tells two decisions apart is their Owen scramble seed rather than their
/// position in a high-dimensional one. Cycles is built this way for the same
/// reason -- two dimensions of direction numbers fit in registers, 256 do not.
///
/// Paired are the draws that are genuinely two-dimensional and consumed
/// together: a pixel offset, a point on a lens, a point on a light, a BSDF
/// direction. Those keep the (0,2)-sequence's joint stratification, which is
/// where it is worth something. The categorical draws stay one-dimensional --
/// pairing them would be a claim about how they correlate and there is no
/// measurement behind one.
///
/// What is given up is stratification *between* decisions. Past a few dimensions
/// that is worth very little to a path tracer, but it is a real trade and it is
/// what the FLIP and SSIM columns are for.
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

/// One decision's draw: the shared two-dimensional sequence, with both the index
/// and the value randomised by a seed belonging to this decision, this bounce
/// and nothing else.
///
/// Both, and this is not optional. Scrambling only the value gives each decision
/// a different *bijection* of the same point, so two decisions plotted against
/// each other trace a curve rather than covering the square -- the degeneracy
/// the note above the old table describes, reached by a different road. Written
/// that way it converged kids_room 8.2% dark, which is what a light selection
/// taking two coordinates off one curve looks like.
///
/// Shuffling the index costs nothing the Z-curve argument depends on. A nested
/// scramble maps dyadic blocks to dyadic blocks, so a pixel's run of
/// `maxSampleCount` consecutive indices stays a contiguous block of that size,
/// and the four blocks of a 2x2 quad stay the four children of one parent. Only
/// their labels move.
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
    // Shift only the primary draws. The mask's value is its spatial spectrum,
    // and a path that has bounced once has already been sent somewhere
    // uncorrelated with its neighbours, so past depth 0 there is no structure
    // left for the shift to preserve.
    //
    // The mask picks the *seed* rather than which branch to take. Written as an
    // if/else, both sides inlined their own copy of the draw and two draws went
    // from 112 SASS instructions to 224 -- and a draw happens ~117 times a path.
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
