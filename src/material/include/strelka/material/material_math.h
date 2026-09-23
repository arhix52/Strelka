#ifndef STRELKA_MATERIAL_MATH_H
#define STRELKA_MATERIAL_MATH_H

// ============================================================================
// material_math.h -- Cross-platform math primitives for CUDA, Metal, and CPU
// ============================================================================

#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(STRELKA_MATERIAL_CUDA_HOST)
#    include <math.h>
#    if defined(__CUDACC__)
#        include <vector_types.h>
#        ifndef STRELKA_CUDA_UINT_TYPEDEF
#            define STRELKA_CUDA_UINT_TYPEDEF
typedef unsigned int uint;
#        endif
#    endif
#    include <sutil/vec_math_adv.h>
#    ifdef __CUDA_ARCH__
#        define DEVICE_FUNC __device__ __forceinline__
#    else
#        define DEVICE_FUNC inline
#    endif
#    if defined(__CUDACC__)
#        define DEVICE_CONST static __device__ const
#    else
#        define DEVICE_CONST static const
#    endif
#    define THREAD_REF
#    define M_PI_F 3.14159265358979323846f
#    define M_1_PI_F 0.31830988618379067154f
#    define M_2_PI_F 0.63661977236758134308f
// float2, float3, float4 and make_float* are CUDA built-ins.
// dot, cross, normalize, length are available via sutil/vec_math.h.

#    ifndef STRELKA_MATERIAL_MATH_CUDA_FUNCS
#        define STRELKA_MATERIAL_MATH_CUDA_FUNCS
// saturate() is provided by sutil/vec_math_adv.h in CUDA builds
DEVICE_FUNC float sqr(float v)
{
    return v * v;
}

DEVICE_FUNC float3 mix(float3 a, float3 b, float t)
{
    return a + (b - a) * t;
}

DEVICE_FUNC float mix(float a, float b, float t)
{
    return a + (b - a) * t;
}

DEVICE_FUNC float luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

DEVICE_FUNC float3 safe_normalize(float3 v)
{
    float len = length(v);
    return len > 1e-8f ? v / len : make_float3(0.0f, 1.0f, 0.0f);
}

DEVICE_FUNC float3 reflect_dir(float3 incident, float3 normal)
{
    return incident - 2.0f * dot(incident, normal) * normal;
}

#    endif

#elif defined(__METAL_VERSION__)
// ---- Metal Shading Language ------------------------------------------------
#    define DEVICE_FUNC inline
// Metal rejects a plain `const` array at module scope; it wants an explicit
// address space. See sheen_albedo_lut.h.
#    define DEVICE_CONST constant
#    define THREAD_REF thread
// M_PI_F is defined by <metal_stdlib>; only define if missing
#    ifndef M_PI_F
#        define M_PI_F 3.14159265358979323846f
#    endif
#    ifndef M_1_PI_F
#        define M_1_PI_F 0.31830988618379067154f
#    endif
#    ifndef M_2_PI_F
#        define M_2_PI_F 0.63661977236758134308f
#    endif

// C math compat aliases so shared headers (sampling.h, fresnel.h) compile
#    define sqrtf(x) metal::sqrt(x)
#    define cosf(x) metal::cos(x)
#    define sinf(x) metal::sin(x)
#    define fmaxf(x, y) metal::fmax(x, y)
#    define fminf(x, y) metal::fmin(x, y)
#    define fabsf(x) metal::fabs(x)
#    define floorf(x) metal::floor(x)
#    define fmodf(x, y) metal::fmod(x, y)
#    define copysignf(x, y) metal::copysign(x, y)
#    define acosf(x) metal::acos(x)
#    define asinf(x) metal::asin(x)
#    define tanf(x) metal::tan(x)
#    define atan2f(y, x) metal::atan2(y, x)
#    define expf(x) metal::exp(x)
#    define logf(x) metal::log(x)
#    define powf(x, y) metal::pow(x, y)
#    define fmaf(x, y, z) metal::fma(x, y, z)

inline float3 make_float3(float x, float y, float z)
{
    return float3(x, y, z);
}
inline float3 make_float3(float v)
{
    return float3(v);
}
inline float2 make_float2(float x, float y)
{
    return float2(x, y);
}
inline float4 make_float4(float x, float y, float z, float w)
{
    return float4(x, y, z, w);
}

// Use metal::saturate directly; do NOT define a wrapper (ambiguous with using namespace metal)
#    define saturate(v) metal::saturate(v)
inline float sqr(float v)
{
    return v * v;
}
inline float luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

inline float3 safe_normalize(float3 v)
{
    float len = metal::length(v);
    return len > 1e-8f ? v / len : float3(0.0f, 1.0f, 0.0f);
}

inline float3 reflect_dir(float3 incident, float3 normal)
{
    return metal::reflect(incident, normal);
}

#else
// ---- CPU (tests, previews) ------------------------------------------------
#    define DEVICE_FUNC inline
#    define DEVICE_CONST static const
#    define THREAD_REF

#    include <glm/glm.hpp>
#    include <glm/gtc/constants.hpp>
#    include <cmath>
#    include <algorithm>

#    ifndef STRELKA_MATERIAL_FLOAT_TYPES
#        define STRELKA_MATERIAL_FLOAT_TYPES
using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;
#    endif

// NOLINTBEGIN(modernize-return-braced-init-list)
// than trailing NOLINTs because clang-format splits a one-liner it is asked
inline float3 make_float3(float x, float y, float z)
{
    return float3(x, y, z);
}
inline float3 make_float3(float v)
{
    return float3(v);
}
inline float2 make_float2(float x, float y)
{
    return float2(x, y);
}
inline float4 make_float4(float x, float y, float z, float w)
{
    return float4(x, y, z, w);
}
// NOLINTEND(modernize-return-braced-init-list)

using glm::cross;
using glm::dot;
using glm::length;
using glm::mix;
using glm::normalize;
using glm::reflect;
using glm::refract;

inline float clamp(float v, float lo, float hi)
{
    return std::max(lo, std::min(hi, v));
}

inline float saturate(float v)
{
    return clamp(v, 0.0f, 1.0f);
}
inline float sqr(float v)
{
    return v * v;
}

inline float luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

inline float3 safe_normalize(float3 v)
{
    const float len = glm::length(v);
    return len > 1e-8f ? v / len : float3(0.0f, 1.0f, 0.0f);
}

#    define M_PI_F 3.14159265358979323846f
#    define M_1_PI_F 0.31830988618379067154f
#    define M_2_PI_F 0.63661977236758134308f

inline float3 reflect_dir(float3 incident, float3 normal)
{
    return incident - 2.0f * glm::dot(incident, normal) * normal;
}

#endif

// Scene loading rejects non-finite or unrepresentable geometry.
DEVICE_FUNC float3 normalizeFiniteVectorOrZero(float3 v)
{
    const float lengthSquared = dot(v, v);
    return lengthSquared > 0.0f ? v / sqrtf(lengthSquared) : make_float3(0.0f);
}

DEVICE_FUNC float3 orthonormalizeTangent(float3 normal, float3 transformedTangent)
{
    const float3 n = normalizeFiniteVectorOrZero(normal);
    if (!(dot(n, n) > 0.0f))
    {
        return make_float3(0.0f);
    }
    const float projection = dot(transformedTangent, n);
    const float3 tangent =
        make_float3(fmaf(-projection, n.x, transformedTangent.x), fmaf(-projection, n.y, transformedTangent.y),
                    fmaf(-projection, n.z, transformedTangent.z));
    return normalizeFiniteVectorOrZero(tangent);
}

DEVICE_FUNC float finiteVectorLength(float3 v)
{
    return sqrtf(dot(v, v));
}

DEVICE_FUNC float3 finiteDirectionAndDistance(float3 offset, THREAD_REF float& distance)
{
    distance = finiteVectorLength(offset);
    return distance > 0.0f ? offset / distance : make_float3(0.0f);
}

DEVICE_FUNC bool refract_dir(float3 incident, float3 normal, float eta, float incidentCosineMagnitude, THREAD_REF float3& out)
{
    if (!(eta > 0.0f) || !(eta <= 3.402823466e38f))
        return false;

    incidentCosineMagnitude = saturate(incidentCosineMagnitude);
    const float transmittedCosineSquared =
        fmaf(eta * eta, incidentCosineMagnitude * incidentCosineMagnitude - 1.0f, 1.0f);
    if (!(transmittedCosineSquared > 0.0f))
        return false;

    const float transmittedCosine = sqrtf(transmittedCosineSquared);
    const float signedIncidentCosine = dot(normal, incident) < 0.0f ? -incidentCosineMagnitude : incidentCosineMagnitude;
    const float normalScale = fmaf(eta, signedIncidentCosine, transmittedCosine);
    out = make_float3(fmaf(eta, incident.x, -normalScale * normal.x), fmaf(eta, incident.y, -normalScale * normal.y),
                      fmaf(eta, incident.z, -normalScale * normal.z));
    return true;
}

DEVICE_FUNC bool refract_dir(float3 incident, float3 normal, float eta, THREAD_REF float3& out)
{
    return refract_dir(incident, normal, eta, fabsf(dot(normal, incident)), out);
}

#endif // STRELKA_MATERIAL_MATH_H
