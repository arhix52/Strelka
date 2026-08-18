#pragma once

// Shared by Metal shaders and host tests. Input RGB values are linear Rec.709;
// PQ values use the absolute ST 2084 scale where code value 1 is 10,000 nits.
#include "tonemappers.h"

#ifdef __METAL_VERSION__
#    define HDR_OUTPUT_NS_BEGIN
#    define HDR_OUTPUT_NS_END
#    define HDR_OUTPUT_FLOAT3 float3
#    define HDR_OUTPUT_MAKE_FLOAT3(a, b, c) float3(a, b, c)
#else
#    include <algorithm>
#    include <cmath>

#    define HDR_OUTPUT_NS_BEGIN                                                                                        \
        namespace oka                                                                                                  \
        {                                                                                                              \
        namespace hdr_output                                                                                           \
        {
#    define HDR_OUTPUT_NS_END                                                                                          \
        }                                                                                                              \
        }
#    define HDR_OUTPUT_FLOAT3 ::oka::tonemap::float3
#    define HDR_OUTPUT_MAKE_FLOAT3(a, b, c) ::oka::tonemap::make_float3(a, b, c)
#endif

HDR_OUTPUT_NS_BEGIN

using HdrFloat3 = HDR_OUTPUT_FLOAT3;

inline float sanitizePaperWhiteNits(const float paperWhiteNits)
{
#ifdef __METAL_VERSION__
    return max(paperWhiteNits, 1.0f);
#else
    return std::max(paperWhiteNits, 1.0f);
#endif
}

inline float sanitizePeakNits(const float paperWhiteNits,
                              const float peakNits)
{
    const float paperWhite = sanitizePaperWhiteNits(paperWhiteNits);

#ifdef __METAL_VERSION__
    return max(peakNits, paperWhite);
#else
    return std::max(peakNits, paperWhite);
#endif
}

inline float displayHeadroom(const float paperWhiteNits, const float peakNits)
{
    const float paperWhite = sanitizePaperWhiteNits(paperWhiteNits);
    const float peak = sanitizePeakNits(paperWhite, peakNits);

    return peak / paperWhite;
}

inline float linearDisplayToNits(const float value,
                                 const float paperWhiteNits,
                                 const float peakNits)
{
    const float paperWhite = sanitizePaperWhiteNits(paperWhiteNits);
    const float peak = sanitizePeakNits(paperWhite, peakNits);

#ifdef __METAL_VERSION__
    return clamp(value * paperWhite, 0.0f, peak);
#else
    return std::clamp(value * paperWhite, 0.0f, peak);
#endif
}

inline HdrFloat3 rec709ToRec2020(const HdrFloat3 color)
{
    return HDR_OUTPUT_MAKE_FLOAT3(0.6274040f * color.x + 0.3292820f * color.y + 0.0433136f * color.z,
                                  0.0690970f * color.x + 0.9195400f * color.y + 0.0113612f * color.z,
                                  0.0163916f * color.x + 0.0880132f * color.y + 0.8955950f * color.z);
}

inline float st2084Encode(const float normalizedLuminance)
{
    const float m1 = 2610.0f / 16384.0f;
    const float m2 = 2523.0f / 32.0f;
    const float c1 = 3424.0f / 4096.0f;
    const float c2 = 2413.0f / 128.0f;
    const float c3 = 2392.0f / 128.0f;
    float luminance;
    float lm1;
    float numerator;
    float denominator;

#ifdef __METAL_VERSION__
    luminance = clamp(normalizedLuminance, 0.0f, 1.0f);
#else
    luminance = std::clamp(normalizedLuminance, 0.0f, 1.0f);
#endif
    if (luminance == 0.0f)
    {
        return 0.0f;
    }
#ifdef __METAL_VERSION__
    lm1 = pow(luminance, m1);
#else
    lm1 = std::pow(luminance, m1);
#endif
    numerator = c1 + c2 * lm1;
    denominator = 1.0f + c3 * lm1;
#ifdef __METAL_VERSION__
    return pow(numerator / denominator, m2);
#else
    return std::pow(numerator / denominator, m2);
#endif
}

inline float st2084EncodeNits(const float luminanceNits)
{
    return st2084Encode(luminanceNits / 10000.0f);
}

inline HdrFloat3 st2084EncodeNits(const HdrFloat3 luminanceNits)
{
    return HDR_OUTPUT_MAKE_FLOAT3(
        st2084EncodeNits(luminanceNits.x), st2084EncodeNits(luminanceNits.y), st2084EncodeNits(luminanceNits.z));
}

HDR_OUTPUT_NS_END

#undef HDR_OUTPUT_MAKE_FLOAT3
#undef HDR_OUTPUT_FLOAT3
#undef HDR_OUTPUT_NS_END
#undef HDR_OUTPUT_NS_BEGIN
