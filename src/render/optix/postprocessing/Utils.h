#pragma once
#include <sutil/vec_math.h>

// utility function for accumulation and HDR <=> LDR
__device__ __inline__ float3 tonemap(float3 color, const float3 exposure)
{
    color *= exposure;
    return color / (color + 1.0f);
}

__device__ __inline__ float3 inverseTonemap(const float3 color, const float3 exposure)
{
    return color / (exposure - color * exposure);
}
