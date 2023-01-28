#pragma once

using namespace metal;

// Code from optix samples

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

// // Generate random float in [0, 1)
// static __inline__ float rnd(thread unsigned int &prev)
// {
//   return ((float) lcg(prev) / (float) 0x01000000);
// }

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

float uintToFloat(uint x) 
{
    return as_type<float>(0x3f800000 | (x >> 9)) - 1.f;
}

uint xorshift(thread uint& rngState) 
{
    rngState ^= rngState << 13; 
    rngState ^= rngState >> 17; 
    rngState ^= rngState << 5; 
    return rngState;
}

float rnd(thread uint& rngState) 
{
    return uintToFloat(xorshift(rngState));
}
