#ifndef STRELKA_MATERIAL_MATH_H
#define STRELKA_MATERIAL_MATH_H

// ============================================================================
// material_math.h -- Cross-platform math primitives for CUDA, Metal, and CPU
// ============================================================================

// STRELKA_MATERIAL_CUDA_HOST selects this branch for the OptiX backend's plain
// g++ translation units. They are host code, but they interoperate with device
// structs and already include CUDA's vector types and sutil, so they need the
// same spellings the device gets -- not the GLM ones the CPU branch installs.
// Taking the CPU branch there redefined make_float3/clamp/saturate on top of
// CUDA's and sutil's, which is what broke the Linux build.
#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(STRELKA_MATERIAL_CUDA_HOST)
// ---- CUDA (device code, nvcc host pass, and OptiX host code) ---------------
    #ifdef __CUDA_ARCH__
    #define DEVICE_FUNC   __device__ __forceinline__
    #else
    #define DEVICE_FUNC   inline
    #endif
    // Storage class for a module-scope constant table; see sheen_albedo_lut.h.
    //
    // Under nvcc the table has to carry __device__ or it lands in host memory and
    // every device function referencing it fails to resolve -- which is exactly
    // what "identifier kSheenAlbedoLut is undefined" meant when the closest-hit
    // module was compiled. `static` keeps it internal to the translation unit, so
    // relocatable device code does not end up with duplicate definitions.
    //
    // Plain g++ building the OptiX host side reaches this branch too (see
    // STRELKA_MATERIAL_CUDA_HOST) and does not know __device__, hence the split.
    #if defined(__CUDACC__)
    #define DEVICE_CONST  static __device__ const
    #else
    #define DEVICE_CONST  static const
    #endif
    #define THREAD_REF
    #define M_PI_F        3.14159265358979323846f
    #define M_1_PI_F      0.31830988618379067154f
    #define M_2_PI_F      0.63661977236758134308f
    // float2, float3, float4 and make_float* are CUDA built-ins.
    // dot, cross, normalize, length are available via sutil/vec_math.h.

    #ifndef STRELKA_MATERIAL_MATH_CUDA_FUNCS
    #define STRELKA_MATERIAL_MATH_CUDA_FUNCS
    // saturate() is provided by sutil/vec_math_adv.h in CUDA builds
    DEVICE_FUNC float  sqr(float v)       { return v * v; }

    DEVICE_FUNC float3 mix(float3 a, float3 b, float t)
    {
        return a + (b - a) * t;
    }

    DEVICE_FUNC float mix(float a, float b, float t)
    {
        return a + (b - a) * t;
    }

    DEVICE_FUNC float  luminance(float3 c)
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

    DEVICE_FUNC bool refract_dir(float3 incident, float3 normal, float eta, float3& out)
    {
        float cosi  = dot(normal, incident);
        float sin2t = eta * eta * (1.0f - cosi * cosi);
        if (sin2t > 1.0f) return false;
        out = eta * incident - (eta * cosi + sqrtf(1.0f - sin2t)) * normal;
        return true;
    }
    #endif

#elif defined(__METAL_VERSION__)
// ---- Metal Shading Language ------------------------------------------------
    #define DEVICE_FUNC   inline
    // Metal rejects a plain `const` array at module scope; it wants an explicit
    // address space. See sheen_albedo_lut.h.
    #define DEVICE_CONST  constant
    #define THREAD_REF    thread
    // M_PI_F is defined by <metal_stdlib>; only define if missing
    #ifndef M_PI_F
    #define M_PI_F        3.14159265358979323846f
    #endif
    #ifndef M_1_PI_F
    #define M_1_PI_F      0.31830988618379067154f
    #endif
    #ifndef M_2_PI_F
    #define M_2_PI_F      0.63661977236758134308f
    #endif

    // C math compat aliases so shared headers (sampling.h, fresnel.h) compile
    #define sqrtf(x)   metal::sqrt(x)
    #define cosf(x)    metal::cos(x)
    #define sinf(x)    metal::sin(x)
    #define fmaxf(x,y) metal::fmax(x,y)
    #define fminf(x,y) metal::fmin(x,y)
    #define fabsf(x)   metal::fabs(x)
    #define fmodf(x,y) metal::fmod(x, y)
    #define copysignf(x,y) metal::copysign(x, y)
    #define acosf(x)   metal::acos(x)
    #define asinf(x)   metal::asin(x)
    #define tanf(x)    metal::tan(x)
    #define atan2f(y,x) metal::atan2(y, x)
    #define expf(x)    metal::exp(x)
    #define logf(x)    metal::log(x)
    #define powf(x,y)  metal::pow(x,y)

    inline float3 make_float3(float x, float y, float z) { return float3(x, y, z); }
    inline float3 make_float3(float v)                    { return float3(v); }
    inline float2 make_float2(float x, float y)           { return float2(x, y); }
    inline float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }

    // Use metal::saturate directly; do NOT define a wrapper (ambiguous with using namespace metal)
    #define saturate(v) metal::saturate(v)
    inline float  sqr(float v)       { return v * v; }
    inline float  luminance(float3 c){ return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

    inline float3 safe_normalize(float3 v)
    {
        float len = metal::length(v);
        return len > 1e-8f ? v / len : float3(0.0f, 1.0f, 0.0f);
    }

    inline float3 reflect_dir(float3 incident, float3 normal)
    {
        return metal::reflect(incident, normal);
    }

    inline bool refract_dir(float3 incident, float3 normal, float eta, thread float3& out)
    {
        float cosi  = metal::dot(normal, incident);
        float sin2t = eta * eta * (1.0f - cosi * cosi);
        if (sin2t > 1.0f) return false;
        out = eta * incident - (eta * cosi + metal::sqrt(1.0f - sin2t)) * normal;
        return true;
    }

#else
// ---- CPU (tests, previews) ------------------------------------------------
    #define DEVICE_FUNC   inline
    #define DEVICE_CONST  static const
    #define THREAD_REF

    #include <glm/glm.hpp>
    #include <glm/gtc/constants.hpp>
    #include <cmath>
    #include <algorithm>

    // Guarded with the same macro material_params.h uses, so whichever of the two
    // is included first wins and the other is a no-op. They must agree on GLM --
    // see the note at the top of material_params.h for what happens when they do
    // not.
    #ifndef STRELKA_MATERIAL_FLOAT_TYPES
    #define STRELKA_MATERIAL_FLOAT_TYPES
    using float2 = glm::vec2;
    using float3 = glm::vec3;
    using float4 = glm::vec4;
    #endif

    // NOLINTBEGIN(modernize-return-braced-init-list)
    //
    // Naming the type is the whole point of a shim three compilers share, so a
    // braced return would delete the only thing these lines say. Markers rather
    // than trailing NOLINTs because clang-format splits a one-liner it is asked
    // to format and carries the comment to the closing brace, where it suppresses
    // nothing -- and this block is hand-aligned, so it is not formatted at all.
    inline float3 make_float3(float x, float y, float z) { return float3(x, y, z); }
    inline float3 make_float3(float v)                    { return float3(v); }
    inline float2 make_float2(float x, float y)           { return float2(x, y); }
    inline float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }
    // NOLINTEND(modernize-return-braced-init-list)

    using glm::dot;
    using glm::cross;
    using glm::normalize;
    using glm::mix;
    using glm::reflect;
    using glm::refract;
    using glm::length;

    inline float clamp(float v, float lo, float hi)
    {
        return std::max(lo, std::min(hi, v));
    }

    inline float saturate(float v) { return clamp(v, 0.0f, 1.0f); }
    inline float sqr(float v)      { return v * v; }

    inline float luminance(float3 c)
    {
        return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    }

    inline float3 safe_normalize(float3 v)
    {
        const float len = glm::length(v);
        return len > 1e-8f ? v / len : float3(0.0f, 1.0f, 0.0f);
    }

    #define M_PI_F   3.14159265358979323846f
    #define M_1_PI_F 0.31830988618379067154f
    #define M_2_PI_F 0.63661977236758134308f

    inline float3 reflect_dir(float3 incident, float3 normal)
    {
        return incident - 2.0f * glm::dot(incident, normal) * normal;
    }

    inline bool refract_dir(float3 incident, float3 normal, float eta, float3& out)
    {
        const float cosi = glm::dot(normal, incident);
        const float sin2t = eta * eta * (1.0f - cosi * cosi);
        if (sin2t > 1.0f) return false;
        out = eta * incident - (eta * cosi + std::sqrt(1.0f - sin2t)) * normal;
        return true;
    }
#endif

// Float vectors can have finite components while dot(v,v) overflows. These
// helpers scale before squaring, and are shared by every backend so geometry
// support never depends on which compiler implements length().
DEVICE_FUNC float3 normalizeFiniteVectorOrZero(float3 v)
{
    const float scale = fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
    if (!(scale > 0.0f) || !(scale <= 3.402823466e38f))
    {
        return make_float3(0.0f);
    }
    const float3 scaled = v / scale;
    const float lengthSquared = dot(scaled, scaled);
    return lengthSquared > 0.0f ? scaled / sqrtf(lengthSquared) : make_float3(0.0f);
}

DEVICE_FUNC float finiteVectorLength(float3 v)
{
    const float scale = fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
    if (!(scale > 0.0f) || !(scale <= 3.402823466e38f))
    {
        return 0.0f;
    }
    const float3 scaled = v / scale;
    const float normalizedLength = sqrtf(dot(scaled, scaled));
    return normalizedLength <= 3.402823466e38f / scale ? scale * normalizedLength : 0.0f;
}

#endif // STRELKA_MATERIAL_MATH_H
