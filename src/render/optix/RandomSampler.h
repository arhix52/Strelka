#pragma once
#include <stdint.h>

// source: https://github.com/mmp/pbrt-v4/blob/5acc5e46cf4b5c3382babd6a3b93b87f54d79b0a/src/pbrt/util/float.h#L46C1-L47C1
static constexpr float FloatOneMinusEpsilon = 0x1.fffffep-1;

__device__ const unsigned int primeNumbers[32] = 
{
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
  31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
  73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  127, 131,
};

enum struct SampleDimension : uint32_t
{
  ePixelX,
  ePixelY,
  eLightId,
  eLightPointX,
  eLightPointY,
  eBSDF0,
  eBSDF1,
  eBSDF2,
  eBSDF3,
  eRussianRoulette,
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
__device__ inline unsigned int pcg_hash(unsigned int seed) {
	unsigned int state = seed * 747796405u + 2891336453u;
	unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ inline unsigned hash_combine(unsigned a, unsigned b) {
	return a ^ (b + 0x9e3779b9 + (a << 6) + (a >> 2));
}

__device__ inline unsigned int hash_with(unsigned int seed, unsigned int hash) {
	// Wang hash
	seed = (seed ^ 61u) ^ hash;
	seed += seed << 3;
	seed ^= seed >> 4;
	seed *= 0x27d4eb2du;
	return seed;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// // jenkins hash
// static __device__ unsigned int hash(unsigned int a)
// {
//   a = (a + 0x7ED55D16) + (a << 12);
//   a = (a ^ 0xC761C23C) ^ (a >> 19);
//   a = (a + 0x165667B1) + (a <<  5);
//   a = (a + 0xD3A2646C) ^ (a <<  9);
//   a = (a + 0xFD7046C5) + (a <<  3);
//   a = (a ^ 0xB55A4F09) ^ (a >> 16);
//   return a;
// }

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

static __device__ SamplerState initSampler(uint32_t linearPixelIndex, uint32_t pixelSampleIndex, uint32_t seed)
{
  SamplerState sampler {};
  sampler.seed = seed;
  sampler.sampleIdx = pixelSampleIndex + linearPixelIndex * 64;
  return sampler;
}

__device__ const uint32_t sb_matrix[5][32] = {
0x80000000, 0x40000000, 0x20000000, 0x10000000,
0x08000000, 0x04000000, 0x02000000, 0x01000000,
0x00800000, 0x00400000, 0x00200000, 0x00100000,
0x00080000, 0x00040000, 0x00020000, 0x00010000,
0x00008000, 0x00004000, 0x00002000, 0x00001000,
0x00000800, 0x00000400, 0x00000200, 0x00000100,
0x00000080, 0x00000040, 0x00000020, 0x00000010,
0x00000008, 0x00000004, 0x00000002, 0x00000001,

0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,

0x80000000, 0xc0000000, 0x60000000, 0x90000000,
0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
0x68800000, 0x9cc00000, 0xee600000, 0x55900000,
0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000,
0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590,
0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555,

0x80000000, 0xc0000000, 0x20000000, 0x50000000,
0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000,
0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000,
0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050,
0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,

0x80000000, 0x40000000, 0x20000000, 0xb0000000,
0xf8000000, 0xdc000000, 0x7a000000, 0x9d000000,
0x5a800000, 0x2fc00000, 0xa1600000, 0xf0b00000,
0xda880000, 0x6fc40000, 0x81620000, 0x40bb0000,
0x22878000, 0xb3c9c000, 0xfb65a000, 0xddb2d000,
0x78022800, 0x9c0b3c00, 0x5a0fb600, 0x2d0ddb00,
0xa2878080, 0xf3c9c040, 0xdb65a020, 0x6db2d0b0,
0x800228f8, 0x400b3cdc, 0x200fb67a, 0xb00ddb9d,
};

__device__ __inline__ uint32_t sobol_uint(uint32_t index, uint32_t dim)
{
    uint32_t X = 0;
    if (dim > 4)
    {
        return X;
    }
    for (int bit = 0; bit < 32; bit++)
    {
        int mask = (index >> bit) & 1;
        X ^= mask * sb_matrix[dim][bit];
    }
    return X;
}

__device__ __inline__ float sobol(uint32_t index, uint32_t dim)
{
    return sobol_uint(index, dim) * float(1.0f / float(UINT32_MAX));
    // return sobol_uint(index, dim) * float(1.0f / float(1ul << 31));
}

__device__ __inline__ uint32_t laine_karras_permutation(uint32_t value, uint32_t seed)
{
    value += seed;
    value ^= value * 0x6c50b47cu;
    value ^= value * 0xb82f1e52u;
    value ^= value * 0xc7afe638u;
    value ^= value * 0x8d22f6e6u;
    return value;
}

__device__ __inline__ uint32_t ReverseBits(uint32_t value)
{

    value = (((value & 0xaaaaaaaa) >> 1) | ((value & 0x55555555) << 1));
    value = (((value & 0xcccccccc) >> 2) | ((value & 0x33333333) << 2));
    value = (((value & 0xf0f0f0f0) >> 4) | ((value & 0x0f0f0f0f) << 4));
    value = (((value & 0xff00ff00) >> 8) | ((value & 0x00ff00ff) << 8));
    return ((value >> 16) | (value << 16));
}

__device__ __inline__ uint32_t nested_uniform_scramble(uint32_t value, uint32_t seed)
{
    value = ReverseBits(value);
    value = laine_karras_permutation(value, seed);
    value = ReverseBits(value);
    return value;
}

__device__ __inline__ float sobol_scramble(uint32_t index, uint32_t dim, uint32_t seed)
{
    // сомнительно, но окей
    // seed += dim / 5;
    // dim = dim % 5;

    index = nested_uniform_scramble(index, hash(seed));

    uint32_t result = nested_uniform_scramble(sobol_uint(index, dim), hash_combine(hash(seed), dim));

    return result * float(1.0f / float(UINT32_MAX));
    // return result * float(1.0f / float(1ul << 31));
}

template <SampleDimension Dim>
__device__ __inline__ float random(SamplerState& state)
{
    const uint32_t dimension = (uint32_t(Dim) + state.depth * uint32_t(SampleDimension::eNUM_DIMENSIONS)) % 5;
    // const uint32_t base = primeNumbers[dimension & 31u];
    // return halton(state.seed + state.sampleIdx, base);
    // const uint32_t dimension = uint32_t(Dim);
    return sobol_scramble(state.sampleIdx, dimension, state.seed);
}
