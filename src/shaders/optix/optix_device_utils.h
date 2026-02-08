#pragma once
// Shared OptiX device utilities used by raygen, closest-hit, and miss programs.

#include <optix.h>
#include <OptixRenderParams.h>

// ---- Pointer packing for payload transport ---------------------------------

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    return reinterpret_cast<void*>(uptr);
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ PerRayData* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<PerRayData*>(unpackPointer(u0, u1));
}

// ---- Attribute interpolation -----------------------------------------------

static __forceinline__ __device__ float3 interpolateAttrib(const float3 attr1,
                                                           const float3 attr2,
                                                           const float3 attr3,
                                                           const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

static __forceinline__ __device__ float2 interpolateAttrib(const float2 attr1,
                                                           const float2 attr2,
                                                           const float2 attr3,
                                                           const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

// ---- Vertex attribute unpacking --------------------------------------------

// Unpack normal from uint32_t. Valid range: [-1, 1]
// Packing: 10 bits per component (x in low bits, z in high bits), biased by +1.0 and scaled by 256
static __forceinline__ __device__ float3 unpackNormal(uint32_t val)
{
    constexpr float scale = 1.0f / 256.0f;
    float3 normal;
    normal.z = ((val & 0xfff00000) >> 20) * scale - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) * scale - 1.0f;
    normal.x = (val & 0x000003ff) * scale - 1.0f;
    return normal;
}

// Unpack UV from uint32_t. Valid range: [-10, 10]
// Packing: 16 bits per component, mapped to [-10, 10]
static __forceinline__ __device__ float2 unpackUV(uint32_t val)
{
    float2 uv;
    uv.y = ((val & 0xffff0000) >> 16) / 16383.99999f * 20.0f - 10.0f;
    uv.x = (val & 0x0000ffff) / 16383.99999f * 20.0f - 10.0f;
    return uv;
}

// ---- Ray offset for self-intersection avoidance ----------------------------
// From Ray Tracing Gems, Chapter 6
static __forceinline__ __device__ float3 offset_ray(const float3 p, const float3 n)
{
    static const float origin = 1.0f / 32.0f;
    static const float float_scale = 1.0f / 65536.0f;
    static const float int_scale = 256.0f;

    int3 of_i = make_int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = make_float3(__int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                             __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                             __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return make_float3(fabs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                       fabs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                       fabs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}
