#pragma once

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

// The host vector type.
//
// Apple's simd is kept on macOS so the shipping tone curve stays bit-for-bit what
// it has always been. Everywhere else <simd/simd.h> does not exist, and this
// header sitting in shaders/common while being unbuildable off Apple is what kept
// StrelkaCLI and test_tonemappers from compiling on Linux at all.
//
// GLM is the substitute rather than a hand-rolled vector because it is already a
// dependency of every target that includes this, and because the operations used
// below -- vec*vec, vec*scalar, vec+scalar, vec/vec -- are exactly the ones a
// hand-rolled type would get subtly wrong. The arithmetic is the same scalar IEEE
// arithmetic either way; only the storage type differs.
#if defined(__APPLE__)
#include <simd/simd.h>
#define MAKE_FLOAT3(a, b, c) simd_make_float3(a, b, c)
#else
#include <glm/glm.hpp>
#define MAKE_FLOAT3(a, b, c) ::oka::tonemap::float3(a, b, c)
#endif

// Namespaced on the host, unqualified on the GPU where there is nothing to
// collide with. `float3` is already taken in this codebase -- glm has one -- so
// injecting simd's into the global namespace turns every translation unit that
// includes both into a pile of ambiguity errors.
#define TONEMAP_CONST const
#define TONEMAP_NS_BEGIN namespace oka { namespace tonemap {
#define TONEMAP_NS_END } }

namespace oka
{
namespace tonemap
{
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

/// Build a host float3 without naming the underlying vector library.
///
/// Callers used to spell this `simd_make_float3`, which compiled only on Apple
/// and is why the headless PNG path and test_tonemappers were macOS-only.
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

inline float3 RRTAndODTFit(float3 v)
{
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

inline float3 ACESFitted(float3 color)
{
    color = transpose(ACESInputMat) * color;
    // Apply RRT and ODT
    color = RRTAndODTFit(color);
    color = transpose(ACESOutputMat) * color;
    // Clamp to [0, 1]
    color = saturate(color);
    return color;
}

/// Spend display headroom on the highlights, and only on the highlights.
///
/// The curve maps into [0, 1]; a display with headroom can show `maxOutput`
/// times SDR white, and this adds back, above white, what the curve had to
/// compress away to fit.
///
/// The obvious way to reach the display peak is to scale both of the curve's
/// axes -- `f(x / maxOutput) * maxOutput` -- and that is what these overloads used
/// to return outright. It rebuilds the whole tone rather than the part that was
/// clipping: at 3.54x headroom ACES took a middle-grey 0.18 from 0.106 down to
/// 0.050, so picking an HDR display mode crushed the shadows by a stop and a half
/// instead of adding highlights. Reinhard alone escaped it, because its toe is
/// exactly linear and the two scalings cancel there -- which is why that curve
/// looked like the control did nothing at all.
///
/// Blending toward the rescaled curve above a knee fixes the shadows but not the
/// rest: it assumes the rescaled curve is the brighter of the two, which holds
/// only for a curve that is concave through the origin. ACES is not -- it has an
/// S-curve toe, so dividing the input by a large headroom lands in it, and
/// `f(x/16)*16` comes out *below* `f(x)` from roughly one third of a stop under
/// white to half a stop over. On the iso_bathroom frame that turned 16x headroom
/// into a mean lift of x0.97: more headroom, darker picture.
///
/// So the lift is built from the linear input instead of from a second pass
/// through the curve, and bounded by the range the curve actually left unspent.
/// It is non-negative by construction, which makes the two properties that
/// matter hold for any curve shape at all: the result is never darker than the
/// SDR curve, and never above the display peak -- except where the SDR curve was
/// already past it, which no amount of headroom can undo and which clamping
/// would only turn into a pixel darker than SDR.
///
/// One lift for the pixel, from its brightest channel, rather than one per
/// channel. Adding the same lift to all three walks a bright saturated highlight
/// toward white as it brightens, which is what a highlight does; a per-channel
/// lift extends only the channel that clipped and walks it further away.
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

    // Range left between what the curve already produced and what the display can
    // show, not simply maxOutput - 1.
    //
    // A curve's own output is not bounded by white. Reinhard divides by the
    // pixel's luminance, so a channel far brighter than that luminance comes out
    // above 1 at any headroom -- on a Cornell box frame that is thousands of
    // pixels. Budgeting from white instead of from where the curve actually
    // landed put those past the display peak, where the window server clips them
    // per channel and shifts their hue on the way.
    //
    // Clamped at zero rather than allowed to go negative: when the curve has
    // already overshot the peak there is nothing left to spend, and the pixel
    // must still not come back darker than the SDR curve left it.
    const float available = maxOutput > sdrPeak ? maxOutput - sdrPeak : 0.0f;

    excess = linearPeak > whitePoint ? linearPeak - whitePoint : 0.0f;
    // Reinhard-shaped on the excess: follows it one for one just over white,
    // where the range is not yet scarce, and rolls off to what is available
    // rather than running past the peak the display can show. The guard is for
    // available == 0, where the quotient would be 0/0.
    lift = available > 0.0f ? available * excess / (excess + available) : 0.0f;
    // Smoothstep, so the lift is admitted with zero slope and rejoins the SDR
    // curve without a crease. A step in the first derivative at white shows up as
    // a hard edge drawn across every smooth falloff in the frame.
    t = saturate((sdrPeak - knee) / (1.0f - knee));
    return sdr + lift * (t * t * (3.0f - 2.0f * t));
}

inline float3 ACESFitted(float3 color, const float maxOutput)
{
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
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

inline float3 ACESFilm(float3 x, const float maxOutput)
{
    const float3 sdr = ACESFilm(x);

    if (maxOutput <= 1.0f)
    {
        return sdr;
    }
    return spendHeadroomOnHighlights(x, sdr, maxOutput);
}

// original implementation https://github.com/NVIDIAGameWorks/Falcor/blob/5236495554f57a734cc815522d95ae9a7dfe458a/Source/RenderPasses/ToneMapper/ToneMapping.ps.slang
inline float calcLuminance(float3 color)
{
    return dot(color, MAKE_FLOAT3(0.299f, 0.587f, 0.114f));
}

inline float3 reinhard(float3 color)
{
    float luminance = calcLuminance(color);
    // float reinhard = luminance / (luminance + 1);
    return color / (luminance + 1.0f);
}

inline float3 reinhard(float3 color, const float maxOutput)
{
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
