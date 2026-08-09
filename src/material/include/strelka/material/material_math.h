#ifndef STRELKA_MATERIAL_MATH_H
#define STRELKA_MATERIAL_MATH_H

// ============================================================================
// material_math.h -- Cross-platform math primitives for CUDA, Metal, and CPU
// ============================================================================

#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
// ---- CUDA (device code and nvcc host pass) ---------------------------------
    #ifdef __CUDA_ARCH__
    #define DEVICE_FUNC   __device__ __forceinline__
    #else
    #define DEVICE_FUNC   inline
    #endif
    // Storage class for a module-scope constant table; see sheen_albedo_lut.h.
    #define DEVICE_CONST  static const
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

    using float2 = glm::vec2;
    using float3 = glm::vec3;
    using float4 = glm::vec4;

    inline float3 make_float3(float x, float y, float z) { return float3(x, y, z); }
    inline float3 make_float3(float v)                    { return float3(v); }
    inline float2 make_float2(float x, float y)           { return float2(x, y); }
    inline float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }

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
        float len = glm::length(v);
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
        float cosi  = glm::dot(normal, incident);
        float sin2t = eta * eta * (1.0f - cosi * cosi);
        if (sin2t > 1.0f) return false;
        out = eta * incident - (eta * cosi + std::sqrt(1.0f - sin2t)) * normal;
        return true;
    }
#endif

#endif // STRELKA_MATERIAL_MATH_H
