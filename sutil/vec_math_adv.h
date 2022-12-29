#pragma once

#include "sutil/vec_math.h"

#include <sutil/Preprocessor.h>

#include <vector_functions.h>
#include <vector_types.h>

#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <cstdlib>
#endif

// SUTIL_INLINE SUTIL_HOSTDEVICE float clamp( const float f, const float a, const float b )
// {
//     return fmaxf( a, fminf( f, b ) );
// }

// SUTIL_INLINE SUTIL_HOSTDEVICE float4 make_float4(const float3& a, float w)
// {
//   return make_float4(a.x, a.y, a.z, w);
// }

SUTIL_INLINE SUTIL_HOSTDEVICE float3 saturate(const float3 v)
{
    float3 r = v;
    r.x = clamp(r.x, 0.0f, 1.0f);
    r.y = clamp(r.y, 0.0f, 1.0f);
    r.z = clamp(r.z, 0.0f, 1.0f);
    return r;
}

SUTIL_INLINE SUTIL_HOSTDEVICE bool all(const float3 v)
{
    return v.x != 0.0f && v.y != 0.0f && v.z != 0.0f;
}

SUTIL_INLINE SUTIL_HOSTDEVICE bool any(const float3 v)
{
    return v.x != 0.0f || v.y != 0.0f || v.z != 0.0f;
}

SUTIL_INLINE SUTIL_HOSTDEVICE bool isnan(const float3 v)
{
    return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

