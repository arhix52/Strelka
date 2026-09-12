#pragma once
// Shared OptiX device utilities used by raygen, closest-hit, and miss programs.

#include <optix.h>
#include <OptixRenderParams.h>

// ---- Firefly bound on indirect paths ---------------------------------------
//
// A firefly is a sample with an enormous weight and a tiny probability -- a
// caustic that found the light through a specular chain. Averaging it in is
// unbiased and does converge; the estimator is correct and the sample budget is
// not. Clamping trades that for bias, so it is off by default and applied only
// past the first bounce, where those paths live: clamping depth 0 as well would
// dim every directly visible emitter and the environment behind it.
static __forceinline__ __device__ float3 clampIndirectContribution(const float3 radiance,
                                                                   unsigned int depth,
                                                                   float limit)
{
    if (limit <= 0.0f || depth == 0u)
    {
        return radiance;
    }
    const float m = fmaxf(radiance.x, fmaxf(radiance.y, radiance.z));
    return (m > limit) ? radiance * (limit / m) : radiance;
}

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

// Unpack the shared RGB10A2-unorm vertex direction. A2 carries tangent
// handedness and is deliberately excluded from z.
static __forceinline__ __device__ float3 unpackNormal(uint32_t val)
{
    constexpr float scale = 2.0f / 1023.0f;
    float3 normal;
    // 10 bits for z, not 12: bit 30 holds the tangent handedness sign written
    // by packTangent(), and must not leak into the coordinate.
    normal.z = ((val & 0x3ff00000) >> 20) * scale - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) * scale - 1.0f;
    normal.x = (val & 0x000003ff) * scale - 1.0f;
    return normal;
}

// glTF TANGENT.w: +1 or -1, deciding which way the bitangent points.
static __forceinline__ __device__ float unpackTangentSign(uint32_t val)
{
    return (val & (1u << 30)) ? -1.0f : 1.0f;
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

// ---- This launch thread's pixel --------------------------------------------
//
// The launch index identifies the pixel and follows the ray through optixReorder.
static __forceinline__ __device__ uint32_t launchPixelIndex(const Params& p)
{
    const uint3 idx = optixGetLaunchIndex();
    return idx.y * p.image_width + idx.x;
}
