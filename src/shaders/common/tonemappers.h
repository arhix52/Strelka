#pragma once

#ifdef __METAL_VERSION__
using namespace metal;
#define TONEMAP_CONST constant
#define MAKE_FLOAT3(a, b, c) float3(a, b, c)
#define TONEMAP_NS_BEGIN
#define TONEMAP_NS_END
#else
#include <cmath>
#include <algorithm>

#if defined(__APPLE__)
#include <simd/simd.h>
#define MAKE_FLOAT3(a, b, c) simd_make_float3(a, b, c)
#else
#include <glm/glm.hpp>
#define MAKE_FLOAT3(a, b, c) ::oka::tonemap::float3(a, b, c)
#endif

#define TONEMAP_CONST const
#define TONEMAP_NS_BEGIN namespace oka { namespace tonemap {
#define TONEMAP_NS_END } }

#if defined(__CUDACC__)
namespace oka
{
namespace tonemap
{
#else
namespace oka::tonemap
{
#endif
#if defined(__APPLE__)
using float3 = simd_float3;
#else
using float3 = glm::vec3;
#endif

// Three columns, in the order Metal's float3x3 stores them, so `transpose(M) * v`
// means the same thing on both sides.
struct float3x3
{
    float3 c0;
    float3 c1;
    float3 c2;
};

inline float3x3 transpose(const float3x3& m)
{
    return { MAKE_FLOAT3(m.c0.x, m.c1.x, m.c2.x), MAKE_FLOAT3(m.c0.y, m.c1.y, m.c2.y),
             MAKE_FLOAT3(m.c0.z, m.c1.z, m.c2.z) };
}

inline float3 operator*(const float3x3& m, const float3& v)
{
    return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z;
}

inline float3 make_float3(float x, float y, float z)
{
    return MAKE_FLOAT3(x, y, z);
}

inline float saturate(float v)
{
    return std::min(std::max(v, 0.0f), 1.0f);
}
inline float3 saturate(const float3& v)
{
    return MAKE_FLOAT3(saturate(v.x), saturate(v.y), saturate(v.z));
}
inline bool isnan(float v)
{
    return std::isnan(v);
}
inline float pow(float a, float b)
{
    return std::pow(a, b);
}
inline float dot(const float3& a, const float3& b)
{
#if defined(__APPLE__)
    return simd_dot(a, b);
#else
    return glm::dot(a, b);
#endif
}
#if defined(__CUDACC__)
} // namespace tonemap
} // namespace oka
#else
} // namespace oka::tonemap
#endif

#endif

TONEMAP_NS_BEGIN

enum class ToneMapperType : uint32_t
{
    eNone = 0,
    eReinhard,
    eACES,
    eFilmic,
    eAgX,
};

inline float3 nonnegativeDisplayRadiance(const float3 color)
{
    return MAKE_FLOAT3(color.x > 0.0f ? color.x : 0.0f, color.y > 0.0f ? color.y : 0.0f,
                       color.z > 0.0f ? color.z : 0.0f);
}

// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
static TONEMAP_CONST float3x3 ACESInputMat =
{
    {0.59719, 0.35458, 0.04823},
    {0.07600, 0.90834, 0.01566},
    {0.02840, 0.13383, 0.83777}
};

// ODT_SAT => XYZ => D60_2_D65 => sRGB
static TONEMAP_CONST float3x3 ACESOutputMat =
{
    { 1.60475, -0.53108, -0.07367},
    {-0.10208,  1.10813, -0.00605},
    {-0.00327, -0.07276,  1.07602}
};

inline float3 RRTAndODTFit(float3 v)
{
    const float3 a = v * (v + 0.0245786f) - 0.000090537f;
    const float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

inline float3 ACESFitted(float3 color)
{
    color = nonnegativeDisplayRadiance(color);
    color = transpose(ACESInputMat) * color;
    // Apply RRT and ODT
    color = RRTAndODTFit(color);
    color = transpose(ACESOutputMat) * color;
    // Clamp to [0, 1]
    color = saturate(color);
    return color;
}

inline float3 spendHeadroomOnHighlights(const float3 linearColor, const float3 sdr, const float maxOutput)
{
    // Linear 1.0 is diffuse white: both exposure paths are photographic -- the
    // auto-exposure anchors the scene mean at middle grey 0.18, the sidecar sets
    // ISO/f-stop/shutter -- and white sits the usual two and a half stops over it.
    const float whitePoint = 1.0f;
    // Where the SDR curve stops being the whole answer. Expressed in curve
    // output rather than scene linear because each curve puts white at a
    // different input; 0.5 out is exactly where Reinhard puts linear 1.0.
    const float knee = 0.5f;
    float sdrPeak = sdr.x;
    float linearPeak = linearColor.x;
    float excess = 0.0f;
    float lift = 0.0f;
    float t = 0.0f;

    if (sdr.y > sdrPeak)
    {
        sdrPeak = sdr.y;
    }
    if (sdr.z > sdrPeak)
    {
        sdrPeak = sdr.z;
    }
    if (linearColor.y > linearPeak)
    {
        linearPeak = linearColor.y;
    }
    if (linearColor.z > linearPeak)
    {
        linearPeak = linearColor.z;
    }

    const float available = maxOutput > sdrPeak ? maxOutput - sdrPeak : 0.0f;

    excess = linearPeak > whitePoint ? linearPeak - whitePoint : 0.0f;
    lift = available > 0.0f ? available * excess / (excess + available) : 0.0f;
    // Smoothstep, so the lift is admitted with zero slope and rejoins the SDR
    // curve without a crease. A step in the first derivative at white shows up as
    // a hard edge drawn across every smooth falloff in the frame.
    t = saturate((sdrPeak - knee) / (1.0f - knee));
    return sdr + lift * (t * t * (3.0f - 2.0f * t));
}

inline float3 ACESFitted(float3 color, const float maxOutput)
{
    color = nonnegativeDisplayRadiance(color);
    const float3 sdr = ACESFitted(color);

    // Not only an optimisation: it is what keeps the SDR path bit-identical, so
    // the headless writer, the reference EXRs and the Cycles ladder do not move.
    if (maxOutput <= 1.0f)
    {
        return sdr;
    }
    return spendHeadroomOnHighlights(color, sdr, maxOutput);
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
inline float3 ACESFilm(float3 x)
{
    x = nonnegativeDisplayRadiance(x);
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

inline float3 ACESFilm(float3 x, const float maxOutput)
{
    x = nonnegativeDisplayRadiance(x);
    const float3 sdr = ACESFilm(x);

    if (maxOutput <= 1.0f)
    {
        return sdr;
    }
    return spendHeadroomOnHighlights(x, sdr, maxOutput);
}

// Blender 5.x AgX Base Contrast, expressed analytically so CPU and GPU display
// paths use the same transform without shipping an OCIO runtime or a 57^3 LUT.
// The formation curve and matrices come from Blender's AgX LUT generator.
static TONEMAP_CONST float3x3 AgXInputMat =
{
    {0.5448147465f, 0.3737873984f, 0.0813978551f},
    {0.1404169485f, 0.7541375546f, 0.1054454970f},
    {0.0888104196f, 0.1788717564f, 0.7323178240f}
};

static TONEMAP_CONST float3x3 AgXOutputMat =
{
    { 1.9648874117f, -0.8559884957f, -0.1088989160f},
    {-0.2993133649f,  1.3263979646f, -0.0270845997f},
    {-0.1643527425f, -0.2381839694f,  1.4025367120f}
};

inline float agxLog2(const float v)
{
#ifdef __METAL_VERSION__
    return metal::log2(v);
#else
    return std::log2(v);
#endif
}

inline float agxAbs(const float v)
{
#ifdef __METAL_VERSION__
    return metal::abs(v);
#else
    return std::abs(v);
#endif
}

inline float agxSqrt(const float v)
{
#ifdef __METAL_VERSION__
    return metal::sqrt(v);
#else
    return std::sqrt(v);
#endif
}

inline float agxFormation(float x)
{
    constexpr float minEv = -12.4739311883f;
    constexpr float maxEv = 4.0260688117f;
    constexpr float pivot = 0.6060606061f;
    constexpr float pivotValue = 0.4894370896f;
    constexpr float d = -80.0f / 55.0f;
    constexpr float e = 132.0f / 55.0f;

    x = saturate((agxLog2(x) - minEv) / (maxEv - minEv));
    const bool upper = x >= pivot;
    const float a = upper ? 0.9049684268f : -1.1441749659f;
    const float b = upper ? -27.9642728229f : 35.3559527134f;
    const float c = upper ? 46.1410501578f : -58.3373219771f;
    const float base = agxAbs(1.0f + a * (x - pivot) * agxSqrt(agxAbs(b + c * x)));
    return pivotValue + (d + e * x) / pow(base, 1.0f / 1.5f);
}

inline float3 AgX(float3 color)
{
    color = MAKE_FLOAT3(color.x > 2.0e-10f ? color.x : 2.0e-10f,
                        color.y > 2.0e-10f ? color.y : 2.0e-10f,
                        color.z > 2.0e-10f ? color.z : 2.0e-10f);
    color = transpose(AgXInputMat) * color;
    color = MAKE_FLOAT3(agxFormation(color.x), agxFormation(color.y), agxFormation(color.z));
    color = MAKE_FLOAT3(pow(color.x, 2.4f), pow(color.y, 2.4f), pow(color.z, 2.4f));
    return transpose(AgXOutputMat) * color;
}

inline float3 AgX(float3 color, const float maxOutput)
{
    const float3 sdr = AgX(color);
    if (maxOutput <= 1.0f)
    {
        return sdr;
    }
    return spendHeadroomOnHighlights(color, sdr, maxOutput);
}

// original implementation https://github.com/NVIDIAGameWorks/Falcor/blob/5236495554f57a734cc815522d95ae9a7dfe458a/Source/RenderPasses/ToneMapper/ToneMapping.ps.slang
inline float calcLuminance(float3 color)
{
    return dot(color, MAKE_FLOAT3(0.299f, 0.587f, 0.114f));
}

inline float3 reinhard(float3 color)
{
    color = nonnegativeDisplayRadiance(color);
    const float luminance = calcLuminance(color);
    // float reinhard = luminance / (luminance + 1);
    return color / (luminance + 1.0f);
}

inline float3 reinhard(float3 color, const float maxOutput)
{
    color = nonnegativeDisplayRadiance(color);
    const float3 sdr = reinhard(color);

    if (maxOutput <= 1.0f)
    {
        return sdr;
    }
    return spendHeadroomOnHighlights(color, sdr, maxOutput);
}

inline float gammaFloat(const float c, const float gamma)
{
    if (isnan(c))
    {
        return 0.0f;
    }
    if (c < 0.0f)
    {
        return 0.0f;
    }
    if (c < 0.0031308f)
    {
        return 12.92f * c;
    }
    return 1.055f * pow(c, 1.0f / gamma) - 0.055f;
}

inline float3 srgbGamma(const float3 color, const float gamma)
{
    return MAKE_FLOAT3(gammaFloat(color.x, gamma), gammaFloat(color.y, gamma), gammaFloat(color.z, gamma));
}

inline float inverseGammaFloat(const float c, const float gamma)
{
    if (isnan(c) || c < 0.0f)
    {
        return 0.0f;
    }
    if (c <= 0.04045f)
    {
        return c / 12.92f;
    }
    return pow((c + 0.055f) / 1.055f, gamma);
}

// utility function for accumulation and HDR <=> LDR
inline float3 tonemap(float3 color, const float3 exposure)
{
    color *= exposure;
    return color / (color + 1.0f);
}

inline float3 inverseTonemap(const float3 color, const float3 exposure)
{
    return color / (exposure - color * exposure);
}

TONEMAP_NS_END
