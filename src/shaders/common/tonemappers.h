#pragma once

#include <simd/simd.h>

// Compiled by the Metal compiler and by the host C++ compiler both.
//
// The host needs the curve because a display image has to go through it, and the
// GPU writes its tonemapped result to a texture rather than back into the buffer
// -- deliberately, so the buffer keeps the linear radiance an EXR wants. That
// left the headless PNG writer reading a buffer nothing had tonemapped, and
// `--tonemap none` and `--tonemap aces` produced byte-identical files.
//
// One definition rather than two, for the reason material_math.h gives: a tone
// curve copied to the host is a tone curve that drifts from the one on screen.
#ifdef __METAL_VERSION__
using namespace metal;
#define TONEMAP_CONST constant
#define MAKE_FLOAT3(a, b, c) float3(a, b, c)
#define TONEMAP_NS_BEGIN
#define TONEMAP_NS_END
#else
#include <cmath>
#include <algorithm>

// Namespaced on the host, unqualified on the GPU where there is nothing to
// collide with. `float3` is already taken in this codebase -- glm has one -- so
// injecting simd's into the global namespace turns every translation unit that
// includes both into a pile of ambiguity errors.
#define TONEMAP_CONST const
#define MAKE_FLOAT3(a, b, c) simd_make_float3(a, b, c)
#define TONEMAP_NS_BEGIN namespace oka { namespace tonemap {
#define TONEMAP_NS_END } }

namespace oka
{
namespace tonemap
{
using float3 = simd_float3;

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
    return { simd_make_float3(m.c0.x, m.c1.x, m.c2.x), simd_make_float3(m.c0.y, m.c1.y, m.c2.y),
             simd_make_float3(m.c0.z, m.c1.z, m.c2.z) };
}

inline float3 operator*(const float3x3& m, const float3& v)
{
    return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z;
}

inline float saturate(float v)
{
    return std::min(std::max(v, 0.0f), 1.0f);
}
inline float3 saturate(const float3& v)
{
    return simd_make_float3(saturate(v.x), saturate(v.y), saturate(v.z));
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
    return simd_dot(a, b);
}
} // namespace tonemap
} // namespace oka
#endif

TONEMAP_NS_BEGIN

enum class ToneMapperType : uint32_t
{
    eNone = 0,
    eReinhard,
    eACES,
    eFilmic,
};

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

float3 RRTAndODTFit(float3 v)
{
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

float3 ACESFitted(float3 color)
{
    color = transpose(ACESInputMat) * color;
    // Apply RRT and ODT
    color = RRTAndODTFit(color);
    color = transpose(ACESOutputMat) * color;
    // Clamp to [0, 1]
    color = saturate(color);
    return color;
}

inline float3 ACESFitted(float3 color, const float maxOutput)
{
    // Keep the SDR curve unchanged at maxOutput == 1 while moving its shoulder
    // to the display peak for EDR. Scaling the result only would brighten paper
    // white; scaling both axes preserves the curve's slope near black.
    return ACESFitted(color / maxOutput) * maxOutput;
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
float3 ACESFilm(float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

inline float3 ACESFilm(float3 x, const float maxOutput)
{
    return ACESFilm(x / maxOutput) * maxOutput;
}

// original implementation https://github.com/NVIDIAGameWorks/Falcor/blob/5236495554f57a734cc815522d95ae9a7dfe458a/Source/RenderPasses/ToneMapper/ToneMapping.ps.slang
float calcLuminance(float3 color)
{
    return dot(color, MAKE_FLOAT3(0.299f, 0.587f, 0.114f));
}

float3 reinhard(float3 color)
{
    float luminance = calcLuminance(color);
    // float reinhard = luminance / (luminance + 1);
    return color / (luminance + 1.0f);
}

inline float3 reinhard(float3 color, const float maxOutput)
{
    return reinhard(color / maxOutput) * maxOutput;
}

float gammaFloat(const float c, const float gamma)
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

float3 srgbGamma(const float3 color, const float gamma)
{
    return MAKE_FLOAT3(gammaFloat(color.x, gamma), gammaFloat(color.y, gamma), gammaFloat(color.z, gamma));
}

// utility function for accumulation and HDR <=> LDR
float3 tonemap(float3 color, const float3 exposure)
{
    color *= exposure;
    return color / (color + 1.0f);
}

float3 inverseTonemap(const float3 color, const float3 exposure)
{
    return color / (exposure - color * exposure);
}

TONEMAP_NS_END
