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
__device__ inline unsigned pcg_hash(unsigned seed) {
	unsigned state = seed * 747796405u + 2891336453u;
	unsigned word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ inline unsigned hash_combine(unsigned a, unsigned b) {
	return a ^ (b + 0x9e3779b9 + (a << 6) + (a >> 2));
}

__device__ inline unsigned hash_with(unsigned seed, unsigned hash) {
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

template <SampleDimension Dim>
__device__ float2 random(unsigned pixel_index, unsigned bounce, unsigned sample_index)
{
    unsigned hash =
        pcg_hash((pixel_index * unsigned(SampleDimension::eNUM_DIMENSIONS) + unsigned(Dim)) * 128 + bounce);

    const unsigned int xx = 0x2f7fffffu;
    const float one_over_max_unsigned = *((float*)(&xx)); // Constant such that 0xffffffff will map to a
                                                                     // float strictly less than 1.0f

    // const float one_over_max_unsigned = 1.0f / (4294967295u);
    // ((float) lcg(prev) / (float) 0x01000000);

    float x = (float) lcg(hash) / (float) 0x01000000;
    float y = (float) lcg(hash) / (float) 0x01000000;

    return make_float2(x, y);
    // return make_float2(one_over_max_unsigned, one_over_max_unsigned);
}
