#pragma once
#include <simd/simd.h>

using namespace metal;

// source: https://github.com/mmp/pbrt-v4
constant constexpr float FloatOneMinusEpsilon = 0x1.fffffep-1;

float uintToFloat(uint x) 
{
    return as_type<float>(0x3f800000 | (x >> 9)) - 1.f;
}

enum class SampleDimension : uint32_t
{
  ePixelX,
  ePixelY,
  eLightId,
  eTime, // motion blur time [0, 1]
  eLightPointX,
  eLightPointY,
  eBSDF0,
  eBSDF1,
  eBSDF2,
  eBSDF3,
  eRussianRoulette,
  eLensU,
  eLensV,
  eNUM_DIMENSIONS
};

struct SamplerState 
{
  uint32_t seed;
  uint32_t sampleIdx;
  uint32_t depth;
};

#define MAX_BOUNCES 128

// Based on: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
inline unsigned pcg_hash(unsigned seed) {
	unsigned state = seed * 747796405u + 2891336453u;
	unsigned word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

inline unsigned hash_with(unsigned seed, unsigned hash) {
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
  a = (a + 0x165667B1) + (a <<  5);
  a = (a + 0xD3A2646C) ^ (a <<  9);
  a = (a + 0xFD7046C5) + (a <<  3);
  a = (a ^ 0xB55A4F09) ^ (a >> 16);
  return a;
}

uint32_t hash2(uint32_t x, uint32_t seed)
{
  x ^= x * 0x3d20adea;
  x += seed;
  x *= (seed >> 16) | 1;
  x ^= x * 0x05526c56;
  x ^= x * 0x53a22864;
  return x;
}

inline uint32_t hash_combine(uint32_t seed, uint32_t v)
{
    return seed ^ (v + (seed << 6) + (seed >> 2));
}

uint jenkinsHash(uint x) 
{
    x += x << 10;
    x ^= x >> 6; 
    x += x << 3; 
    x ^= x >> 11; 
    x += x << 15; 
    return x;
}

constant unsigned int primeNumbers[32] =
{
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
  31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
  73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  127, 131
};

// Sobol direction matrices (5 dimensions x 32 bits)
constant const uint32_t sb_matrix[5][32] = {
    {0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000, 0x01000000,
     0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000, 0x00020000, 0x00010000,
     0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800, 0x00000400, 0x00000200, 0x00000100,
     0x00000080, 0x00000040, 0x00000020, 0x00000010, 0x00000008, 0x00000004, 0x00000002, 0x00000001},

    {0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
     0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
     0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
     0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0, 0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff},

    {0x80000000, 0xc0000000, 0x60000000, 0x90000000, 0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
     0x68800000, 0x9cc00000, 0xee600000, 0x55900000, 0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
     0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000, 0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
     0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590, 0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555},

    {0x80000000, 0xc0000000, 0x20000000, 0x50000000, 0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
     0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000, 0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
     0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000, 0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
     0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050, 0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093},

    {0x80000000, 0x40000000, 0x20000000, 0xb0000000, 0xf8000000, 0xdc000000, 0x7a000000, 0x9d000000,
     0x5a800000, 0x2fc00000, 0xa1600000, 0xf0b00000, 0xda880000, 0x6fc40000, 0x81620000, 0x40bb0000,
     0x22878000, 0xb3c9c000, 0xfb65a000, 0xddb2d000, 0x78022800, 0x9c0b3c00, 0x5a0fb600, 0x2d0ddb00,
     0xa2878080, 0xf3c9c040, 0xdb65a020, 0x6db2d0b0, 0x800228f8, 0x400b3cdc, 0x200fb67a, 0xb00ddb9d},
};

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
    return clamp(result, 0.0f, 1.0f - 1e-6f); // TODO: 1minusEps
}

static SamplerState initSampler(uint32_t linearPixelIndex, uint32_t pixelSampleIndex, uint32_t seed)
{
  SamplerState sampler {};
  sampler.seed = hash(linearPixelIndex); //^ 0x736caf6fu;
  sampler.sampleIdx = pixelSampleIndex;
  sampler.depth = 0;
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
static float randomPCG(thread SamplerState& state)
{
    const uint32_t dimension = uint32_t(Dim) + state.depth * uint32_t(SampleDimension::eNUM_DIMENSIONS);
    uint32_t h = hash_with(state.seed + state.sampleIdx, dimension);
    h = pcg_hash(h);
    return uintToFloat(h);
}

// ── Sobol + Owen scrambling (ported from OptiX random.h) ────────────────────

inline uint32_t sobol_uint(uint32_t index, uint32_t dim)
{
    uint32_t X = 0;
    for (int bit = 0; index != 0; ++bit, index >>= 1)
    {
        X ^= (index & 1) * sb_matrix[dim][bit];
    }
    return X;
}

inline uint32_t laine_karras_permutation(uint32_t value, uint32_t seed)
{
    value += seed;
    value ^= value * 0x6c50b47cu;
    value ^= value * 0xb82f1e52u;
    value ^= value * 0xc7afe638u;
    value ^= value * 0x8d22f6e6u;
    return value;
}

inline uint32_t nested_uniform_scramble(uint32_t value, uint32_t seed)
{
    value = reverse_bits(value);
    value = laine_karras_permutation(value, seed);
    value = reverse_bits(value);
    return value;
}

// `matrixIndex` selects one of the 5 available Sobol direction matrices;
// `scrambleDim` is the true (unreduced) sample dimension and only feeds the Owen
// scramble seed.
inline float sobol_scramble(uint32_t index, uint32_t matrixIndex, uint32_t scrambleDim, uint32_t seed)
{
    seed = hash(seed);
    index = nested_uniform_scramble(index, seed);
    uint32_t result = nested_uniform_scramble(sobol_uint(index, matrixIndex), hash_combine(seed, scrambleDim));
    return min(result * 0x1p-32f, FloatOneMinusEpsilon);
}

template <SampleDimension Dim>
static float randomSobol(thread SamplerState& state)
{
    const uint32_t dimension = uint32_t(Dim) + state.depth * uint32_t(SampleDimension::eNUM_DIMENSIONS);
    // Only 5 direction matrices are tabulated, so dimensions alias modulo 5.
    // Previously the reduced index was also used as the Owen scramble seed, which
    // made aliased dimensions produce *identical* values — at depth 0, ePixelY
    // (1) and eLensU (11 % 5 == 1) returned the same number, locking the pixel
    // jitter to the lens sample and structuring DoF bokeh. Seeding the scramble
    // with the unreduced dimension decorrelates them.
    return sobol_scramble(state.sampleIdx, dimension % 5u, dimension, state.seed + state.depth);
}

// ── Sampler dispatch ────────────────────────────────────────────────────────
// 0 = Halton, 1 = PCG, 2 = Sobol (Owen scrambled)

template <SampleDimension Dim>
static float random(thread SamplerState& state, uint32_t samplerType)
{
    if (samplerType == 2)
        return randomSobol<Dim>(state);
    if (samplerType == 1)
        return randomPCG<Dim>(state);
    return randomHalton<Dim>(state);
}

uint xorshift(thread uint& rngState) 
{
    rngState ^= rngState << 13; 
    rngState ^= rngState >> 17; 
    rngState ^= rngState << 5; 
    return rngState;
}

template<unsigned int N>
static  __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ )
  {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
static  __inline__ unsigned int lcg(thread unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

static  __inline__ unsigned int lcg2(thread unsigned int &prev)
{
  prev = (prev*8121 + 28411)  % 134456;
  return prev;
}

// Generate random float in [0, 1)
static __inline__ float rnd(thread unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

static  __inline__ unsigned int rot_seed( unsigned int seed, unsigned int frame )
{
    return seed ^ frame;
}

// Implementetion from Ray Tracing gems
// https://github.com/boksajak/referencePT/blob/master/shaders/PathTracer.hlsl
uint initRNG(uint2 pixelCoords, uint2 resolution, uint frameNumber)
{
    uint t = dot(float2(pixelCoords), float2(1, resolution.x));
    uint seed = t ^ jenkinsHash(frameNumber);
    // uint seed = dot(pixelCoords, uint2(1, resolution.x)) ^ jenkinsHash(frameNumber);
    return jenkinsHash(seed); 
}

uint owen_scramble_rev(uint x, uint seed)
{
    x ^= x * 0x3d20adea;
    x += seed;
    x *= (seed >> 16) | 1;
    x ^= x * 0x05526c56;
    x ^= x * 0x53a22864;
    return x;
}
