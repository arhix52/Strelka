#pragma once

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
__device__ __inline__ float2 random(uint32_t pixel_index, uint32_t bounce, uint32_t sample_index)
{
	uint32_t dimension = uint32_t(0);
	// uint32_t scramble = pcg_hash(pixel_index + bounce);
	uint32_t index = sample_index;// + scramble;
    const unsigned int base = 2; //primeNumbers[dimension++ & 1023];

	return make_float2(halton(index, base), halton(index, base));
}

__device__ float2 random2(uint32_t pixel_index, uint32_t bounce, const uint32_t sample_index)
{
	// uint32_t dimension = uint32_t(0);
	// uint32_t scramble = pcg_hash(pixel_index + bounce);
	const uint32_t index = 0;// + scramble;
    const unsigned int base = 2; //primeNumbers[dimension++ & 1023];
	// if (sample_index == 0)
	// {
	// 	return make_float2(0.0f, 0.0f);
	// }

	return make_float2(halton(index, base), halton(index, base));
}
