#pragma once
#include <simd/simd.h>

using namespace metal;

float uintToFloat(uint x) 
{
    return as_type<float>(0x3f800000 | (x >> 9)) - 1.f;
}

// Code from optix samples

enum class SampleDimension : uint32_t
{
  ePixel,
  eLightId,
  eLightPoint,
  eBSDF0,
  eBSDF1,
  eRussianRoulette,
  eNUM_DIMENSIONS
};


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

template<SampleDimension Dim>
static float2 random(unsigned pixel_index, unsigned bounce, unsigned sample_index)
{
  	unsigned hash = pcg_hash((pixel_index * unsigned(SampleDimension::eNUM_DIMENSIONS) + unsigned(Dim)) * MAX_BOUNCES + bounce);

		const float one_over_max_unsigned = as_type<float>(0x2f7fffffu); // Constant such that 0xffffffff will map to a float strictly less than 1.0f

		float x = hash_with(sample_index,              hash) * one_over_max_unsigned;
		float y = hash_with(sample_index + 0xdeadbeef, hash) * one_over_max_unsigned;

		return float2(x, y);
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

uint jenkinsHash(uint x) 
{
    x += x << 10;
    x ^= x >> 6; 
    x += x << 3; 
    x ^= x >> 11; 
    x += x << 15; 
    return x;
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



uint xorshift(thread uint& rngState) 
{
    rngState ^= rngState << 13; 
    rngState ^= rngState >> 17; 
    rngState ^= rngState << 5; 
    return rngState;
}

// float rnd(thread uint& rngState) 
// {
//     return uintToFloat(xorshift(rngState));
// }

// uint hash_u32(uint n)
// {
//     n ^= 0xe6fe3beb;
//     n ^= n >> 16;
//     n = n.wrapping_mul(0x7feb352d);
//     n ^= n >> 15;
//     n = n.wrapping_mul(0x846ca68b);
//     n ^= n >> 16;
//     return n
// }

uint owen_scramble_rev(uint x, uint seed)
{
    x ^= x * 0x3d20adea;
    x += seed;
    x *= (seed >> 16) | 1;
    x ^= x * 0x05526c56;
    x ^= x * 0x53a22864;
    return x;
}
