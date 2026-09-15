#ifndef STRELKA_MATERIAL_OPENPBR_SHIM_H
#define STRELKA_MATERIAL_OPENPBR_SHIM_H

// (1) -- see above.
#ifndef OPENPBR_USE_TEXTURE_LUTS
#    define OPENPBR_USE_TEXTURE_LUTS 0
#endif

// (5) -- the host is the only backend that compiles more than one translation
// unit against this header.
#if !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__) && !defined(__CUDACC__)
#    define STRELKA_OPENPBR_NEEDS_INTERNAL_LINKAGE 1
#endif

#include <strelka/material/material_math.h>

#if defined(__METAL_VERSION__)
#    define OPENPBR_USE_CUSTOM_VEC_TYPES 1
using vec2 = float2;
using vec3 = packed_float3;
using vec4 = float4;

template <typename T, int N>
inline vec<bool, N> equal(vec<T, N> a, vec<T, N> b)
{
    return a == b;
}
inline bool3 equal(vec3 a, vec3 b)
{
    return float3(a) == float3(b);
}
template <typename T, int N>
inline vec<bool, N> notEqual(vec<T, N> a, vec<T, N> b)
{
    return a != b;
}
inline bool3 notEqual(vec3 a, vec3 b)
{
    return float3(a) != float3(b);
}
template <typename T, int N>
inline vec<bool, N> greaterThan(vec<T, N> a, vec<T, N> b)
{
    return a > b;
}
inline bool3 greaterThan(vec3 a, vec3 b)
{
    return float3(a) > float3(b);
}
template <typename T, int N>
inline vec<bool, N> greaterThanEqual(vec<T, N> a, vec<T, N> b)
{
    return a >= b;
}
inline bool3 greaterThanEqual(vec3 a, vec3 b)
{
    return float3(a) >= float3(b);
}
#endif

// (6) -- see above. Before openpbr.h, and before the interop layer it pulls in,
// because it defines OPENPBR_USE_CUSTOM_VEC_TYPES.
#if defined(__CUDACC__)
#    include <strelka/material/openpbr/openpbr_cuda_vec.h>
#endif

// (2) -- see above.
#if !defined(__METAL_VERSION__)
#    define OPENPBR_USE_CUSTOM_SATURATE 1
#    if !defined(__CUDA_ARCH__) && !defined(__CUDACC__) && !defined(STRELKA_MATERIAL_CUDA_HOST)
// Host build only: material_math.h has saturate(float), nothing has the vec3
// overload openpbr calls. On CUDA both come from sutil/vec_math_adv.h.
#        ifndef STRELKA_MATERIAL_HAS_SATURATE_FLOAT3
#            define STRELKA_MATERIAL_HAS_SATURATE_FLOAT3
DEVICE_FUNC float3 saturate(float3 v)
{
    return { saturate(v.x), saturate(v.y), saturate(v.z) };
}
#        endif
#    endif
#endif

// (5) -- see above.
#if defined(STRELKA_OPENPBR_NEEDS_INTERNAL_LINKAGE)
#    include <cassert>
#    include <cstdint>
#endif

// (4) -- see above.
#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wc++17-extensions"
#endif
#if defined(STRELKA_OPENPBR_NEEDS_INTERNAL_LINKAGE)
// NOLINTBEGIN(cert-dcl59-cpp, misc-anonymous-namespace-in-header)
namespace
{
#endif

#include "openpbr.h"

#if defined(STRELKA_OPENPBR_NEEDS_INTERNAL_LINKAGE)
} // anonymous namespace
// NOLINTEND(cert-dcl59-cpp, misc-anonymous-namespace-in-header)
#endif
#if defined(__clang__)
#    pragma clang diagnostic pop
#endif

// (3) -- see above. Undefining names openpbr did not define is legal and a no-op,
// so this needs no per-backend guard.
#undef mix
#undef all
#undef any
#undef equal
#undef notEqual
#undef greaterThan
#undef greaterThanEqual

#endif // STRELKA_MATERIAL_OPENPBR_SHIM_H
