#pragma once

#include <simd/simd.h>

using namespace metal;

enum class ToneMapperType : uint32_t
{
    eNone = 0,
    eReinhard,
    eACES,
    eFilmic,
};

// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
static constant float3x3 ACESInputMat =
{
    {0.59719, 0.35458, 0.04823},
    {0.07600, 0.90834, 0.01566},
    {0.02840, 0.13383, 0.83777}
};

// ODT_SAT => XYZ => D60_2_D65 => sRGB
static constant float3x3 ACESOutputMat =
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

// original implementation https://github.com/NVIDIAGameWorks/Falcor/blob/5236495554f57a734cc815522d95ae9a7dfe458a/Source/RenderPasses/ToneMapper/ToneMapping.ps.slang
float calcLuminance(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
}

float3 reinhard(float3 color)
{
    float luminance = calcLuminance(color);
    // float reinhard = luminance / (luminance + 1);
    return color / (luminance + 1.0f);
}

float gammaFloat(const float c, const float gamma)
{
    if (isnan(c))
    {
        return 0.0f;
    }
    if (c > 1.0f)
    {
        return 1.0f;
    }
    else if (c < 0.0f)
    {
        return 0.0f;
    }
    else if (c < 0.0031308f)
    {
        return 12.92f * c;
    }
    else
    {
        return 1.055f * pow(c, 1.0f / gamma) - 0.055f;
    }
}

float3 srgbGamma(const float3 color, const float gamma)
{
    return float3(gammaFloat(color.r, gamma), gammaFloat(color.g, gamma), gammaFloat(color.b, gamma));
}
