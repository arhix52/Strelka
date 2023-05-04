#pragma once

__device__ const unsigned int primeNumbers[32] = 
{
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
  31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
  73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  127, 131,
};

enum struct SampleDimension
{
    ePixel,
    eLightId,
    eLightPoint,
    eBSDF0,
    eBSDF1,
    eRussianRoulette,
    eNUM_DIMENSIONS
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

// jenkins hash
__device__ unsigned int hash(unsigned int a)
{
  a = (a + 0x7ED55D16) + (a << 12);
  a = (a ^ 0xC761C23C) ^ (a >> 19);
  a = (a + 0x165667B1) + (a <<  5);
  a = (a + 0xD3A2646C) ^ (a <<  9);
  a = (a + 0xFD7046C5) + (a <<  3);
  a = (a ^ 0xB55A4F09) ^ (a >> 16);
  return a;
}

__device__ float halton(uint32_t index, uint32_t base)
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
    return clamp(result, 0.0f, 1.0f);
}

template <SampleDimension Dim>
__device__ __inline__ float2 random(uint32_t linearPixelIndex, uint32_t bounce, uint32_t sampleIndex)
{
    uint32_t dimension = uint32_t(Dim) * 2;
    uint32_t seed = hash(linearPixelIndex);
    uint32_t index = seed + sampleIndex;
    const unsigned int baseX = primeNumbers[dimension & 31u];
    const unsigned int baseY = primeNumbers[(dimension + 1) & 31u];

    return make_float2(halton(index, baseX), halton(index, baseY));
}
