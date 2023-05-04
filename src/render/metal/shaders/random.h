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
    // return clamp(result, 0.0f, 1.0f);
    return result;
}

template <SampleDimension Dim>
static float random(unsigned linearPixelIndex, unsigned bounce, unsigned sampleIndex)
{
    uint32_t seed = hash(linearPixelIndex);
    uint32_t dimension = uint32_t(Dim);
    const uint32_t base = primeNumbers[dimension & 31u];
    float x = halton(sampleIndex + seed, base);
    return x;
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
