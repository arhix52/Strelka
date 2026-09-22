#ifndef STRELKA_RECONSTRUCTION_FILTER_H
#define STRELKA_RECONSTRUCTION_FILTER_H

#include <strelka/material/material_math.h>

#ifndef RECONSTRUCTION_FILTER_BOX
#define RECONSTRUCTION_FILTER_BOX 0u
#define RECONSTRUCTION_FILTER_MITCHELL 1u
#define RECONSTRUCTION_FILTER_TENT 2u
#define RECONSTRUCTION_FILTER_LANCZOS2 3u
#define RECONSTRUCTION_FILTER_GAUSSIAN 4u
#define RECONSTRUCTION_FILTER_BLACKMAN_HARRIS 5u
#endif

struct ReconstructionFilterSample
{
    float offset;
    float weight;
};

DEVICE_FUNC ReconstructionFilterSample sampleTent(float xi)
{
    constexpr float radius = 2.0f;
    const float u = fminf(fmaxf(xi, 0.0f), 1.0f);
    const float side = 2.0f * u - 1.0f;
    const float magnitude = radius * (1.0f - sqrtf(1.0f - fabsf(side)));
    return { copysignf(magnitude, side), 1.0f };
}

DEVICE_FUNC float reconstructionErf(float x)
{
    // Abramowitz-Stegun 7.1.26 is accurate enough for the inverse-CDF search.
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    x = fabsf(x);
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float p = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t +
                     0.254829592f) *
                    t;
    return sign * (1.0f - p * expf(-x * x));
}

DEVICE_FUNC float gaussianCdf(float x)
{
    constexpr float radius = 1.5f;
    constexpr float edge = 0.0111089965f;
    constexpr float gaussianIntegralScale = 0.6266570687f;
    return gaussianIntegralScale * (reconstructionErf(1.4142135624f * x) +
                                    reconstructionErf(1.4142135624f * radius)) -
           edge * (x + radius);
}

DEVICE_FUNC ReconstructionFilterSample sampleGaussian(float xi)
{
    constexpr float radius = 1.5f;
    constexpr float integral = 1.2166034551f;
    const float target = fminf(fmaxf(xi, 0.0f), 1.0f) * integral;
    float lo = -radius;
    float hi = radius;
    for (int i = 0; i < 12; ++i)
    {
        const float mid = 0.5f * (lo + hi);
        if (gaussianCdf(mid) < target)
            lo = mid;
        else
            hi = mid;
    }
    return { 0.5f * (lo + hi), 1.0f };
}

DEVICE_FUNC float blackmanHarrisCdf(float x)
{
    constexpr float radius = 2.0f;
    const float phase = M_PI_F * x / radius;
    return 0.35875f * (x + radius) + 0.48829f * radius / M_PI_F * sinf(phase) +
           0.14128f * radius / (2.0f * M_PI_F) * sinf(2.0f * phase) +
           0.01168f * radius / (3.0f * M_PI_F) * sinf(3.0f * phase);
}

DEVICE_FUNC ReconstructionFilterSample sampleBlackmanHarris(float xi)
{
    constexpr float radius = 2.0f;
    constexpr float integral = 1.435f;
    const float target = fminf(fmaxf(xi, 0.0f), 1.0f) * integral;
    float lo = -radius;
    float hi = radius;
    for (int i = 0; i < 12; ++i)
    {
        const float mid = 0.5f * (lo + hi);
        if (blackmanHarrisCdf(mid) < target)
            lo = mid;
        else
            hi = mid;
    }
    return { 0.5f * (lo + hi), 1.0f };
}

// clang-format off
DEVICE_CONST float kLanczos2PositiveInverseCdf[65] = {
    0.000000000f, 0.008590713f, 0.017184035f, 0.025782580f,
    0.034388974f, 0.043005865f, 0.051635921f, 0.060281847f,
    0.068946384f, 0.077632318f, 0.086342490f, 0.095079800f,
    0.103847217f, 0.112647790f, 0.121484651f, 0.130361032f,
    0.139280269f, 0.148245819f, 0.157261268f, 0.166330348f,
    0.175456948f, 0.184645133f, 0.193899160f, 0.203223497f,
    0.212622845f, 0.222102165f, 0.231666697f, 0.241321996f,
    0.251073964f, 0.260928886f, 0.270893471f, 0.280974904f,
    0.291180898f, 0.301519756f, 0.312000443f, 0.322632670f,
    0.333426988f, 0.344394902f, 0.355548998f, 0.366903104f,
    0.378472464f, 0.390273964f, 0.402326392f, 0.414650755f,
    0.427270668f, 0.440212835f, 0.453507644f, 0.467189919f,
    0.481299881f, 0.495884381f, 0.510998533f, 0.526707873f,
    0.543091322f, 0.560245287f, 0.578289532f, 0.597375828f,
    0.617701145f, 0.639528722f, 0.663223570f, 0.689316646f,
    0.718632108f, 0.752574758f, 0.793922757f, 0.849970379f,
    1.000000000f,
};

DEVICE_CONST float kLanczos2NegativeInverseCdf[65] = {
    1.000000000f, 1.048528813f, 1.069676633f, 1.086385175f,
    1.100826746f, 1.113841561f, 1.125860231f, 1.137138589f,
    1.147843503f, 1.158091129f, 1.167966335f, 1.177533508f,
    1.186843006f, 1.195935224f, 1.204843276f, 1.213594821f,
    1.222213356f, 1.230719148f, 1.239129928f, 1.247461411f,
    1.255727701f, 1.263941611f, 1.272114914f, 1.280258550f,
    1.288382796f, 1.296497414f, 1.304611770f, 1.312734945f,
    1.320875836f, 1.329043239f, 1.337245940f, 1.345492795f,
    1.353792811f, 1.362155227f, 1.370589606f, 1.379105921f,
    1.387714662f, 1.396426940f, 1.405254620f, 1.414210461f,
    1.423308285f, 1.432563171f, 1.441991693f, 1.451612199f,
    1.461445154f, 1.471513561f, 1.481843491f, 1.492464743f,
    1.503411691f, 1.514724390f, 1.526450028f, 1.538644874f,
    1.551376960f, 1.564729830f, 1.578807953f, 1.593744780f,
    1.609715181f, 1.626955567f, 1.645798322f, 1.666735174f,
    1.690545609f, 1.718594921f, 1.753686485f, 1.803634190f,
    2.000000000f,
};
// clang-format on

DEVICE_FUNC float lanczos2Kernel(float x)
{
    const float ax = fabsf(x);
    if (ax >= 2.0f)
    {
        return 0.0f;
    }
    if (ax < 1.0e-6f)
    {
        return 1.0f;
    }
    const float pix = M_PI_F * ax;
    return (sinf(pix) / pix) * (sinf(0.5f * pix) / (0.5f * pix));
}

DEVICE_FUNC float sampleLanczos2Lobe(float quantile, bool negativeLobe)
{
    const float scaled = fminf(fmaxf(quantile, 0.0f), 1.0f) * 64.0f;
    const unsigned int bin = (unsigned int)fminf(scaled, 63.0f);
    float t = scaled - float(bin);
    if (bin == 63u)
    {
        t = 1.0f - sqrtf(1.0f - t);
    }
    else if (negativeLobe && bin == 0u)
    {
        t = sqrtf(t);
    }
    const float a = negativeLobe ? kLanczos2NegativeInverseCdf[bin] : kLanczos2PositiveInverseCdf[bin];
    const float b = negativeLobe ? kLanczos2NegativeInverseCdf[bin + 1u] : kLanczos2PositiveInverseCdf[bin + 1u];
    return a + (b - a) * t;
}

DEVICE_FUNC ReconstructionFilterSample sampleLanczos2(float xi)
{
    constexpr float positiveLobeProbability = 0.924523478f;
    constexpr float normalizedAbsoluteIntegral = 1.177791162f;
    const float u = fminf(fmaxf(xi, 0.0f), 1.0f);
    const bool left = u < 0.5f;
    const float sideQuantile = left ? 2.0f * u : 2.0f * (u - 0.5f);
    const bool negativeLobe = sideQuantile >= positiveLobeProbability;
    const float lobeQuantile = negativeLobe ?
                                   (sideQuantile - positiveLobeProbability) / (1.0f - positiveLobeProbability) :
                                   sideQuantile / positiveLobeProbability;
    const float radius = sampleLanczos2Lobe(lobeQuantile, negativeLobe);
    const float weight = negativeLobe ? -normalizedAbsoluteIntegral : normalizedAbsoluteIntegral;
    return { left ? -radius : radius, weight };
}

DEVICE_FUNC float reconstructionBatchScale(float previousWeight, float nextWeight)
{
    // Every signed filter samples |kernel|, so every sample from one filter has
    // the same weight magnitude. Rebasing the encoded partial sum therefore
    // only needs to preserve or flip its sign.
    return (previousWeight < 0.0f) == (nextWeight < 0.0f) ? 1.0f : -1.0f;
}

// Standard Mitchell-Netravali (B = C = 1/3). Sampling |filter| makes
// the importance weight a signed constant rather than another path float.
DEVICE_FUNC float mitchellPrimitiveInner(float x)
{
    return (7.0f / 24.0f) * x * x * x * x - (2.0f / 3.0f) * x * x * x + (8.0f / 9.0f) * x;
}

DEVICE_FUNC float mitchellPrimitiveOuter(float x)
{
    return (-7.0f / 72.0f) * x * x * x * x + (2.0f / 3.0f) * x * x * x - (5.0f / 3.0f) * x * x + (16.0f / 9.0f) * x;
}

DEVICE_FUNC float mitchellAbsoluteHalfIntegral(float radius)
{
    const float r = fminf(fmaxf(radius, 0.0f), 2.0f);
    constexpr float zeroCrossing = 8.0f / 7.0f;
    constexpr float innerIntegral = 37.0f / 72.0f;
    constexpr float outerAtOne = 49.0f / 72.0f;
    constexpr float positiveIntegral = 355.0f / 686.0f;
    constexpr float outerAtZeroCrossing = 704.0f / 1029.0f;
    if (r <= 1.0f)
    {
        return mitchellPrimitiveInner(r);
    }
    if (r <= zeroCrossing)
    {
        return innerIntegral + mitchellPrimitiveOuter(r) - outerAtOne;
    }
    return positiveIntegral + outerAtZeroCrossing - mitchellPrimitiveOuter(r);
}

DEVICE_FUNC ReconstructionFilterSample sampleMitchell(float xi)
{
    constexpr float halfAbsoluteIntegral = 367.0f / 686.0f;
    constexpr float zeroCrossing = 8.0f / 7.0f;
    const bool negativeSide = xi < 0.5f;
    const float quantile = negativeSide ? xi * 2.0f : (xi - 0.5f) * 2.0f;
    const float target = quantile * halfAbsoluteIntegral;

    float lo = 0.0f;
    float hi = 2.0f;
    for (int i = 0; i < 12; ++i)
    {
        const float mid = 0.5f * (lo + hi);
        if (mitchellAbsoluteHalfIntegral(mid) < target)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    const float radius = 0.5f * (lo + hi);
    ReconstructionFilterSample result;
    result.offset = negativeSide ? -radius : radius;
    result.weight = radius > zeroCrossing ? -(367.0f / 343.0f) : (367.0f / 343.0f);
    return result;
}

DEVICE_FUNC ReconstructionFilterSample reconstructionFilterSample(unsigned int filter, float xi)
{
    if (filter == RECONSTRUCTION_FILTER_MITCHELL)
        return sampleMitchell(xi);
    if (filter == RECONSTRUCTION_FILTER_TENT)
        return sampleTent(xi);
    if (filter == RECONSTRUCTION_FILTER_LANCZOS2)
        return sampleLanczos2(xi);
    if (filter == RECONSTRUCTION_FILTER_GAUSSIAN)
        return sampleGaussian(xi);
    if (filter == RECONSTRUCTION_FILTER_BLACKMAN_HARRIS)
        return sampleBlackmanHarris(xi);
    return { xi - 0.5f, 1.0f };
}

DEVICE_FUNC bool reconstructionFilterIsSigned(unsigned int filter)
{
    return filter == RECONSTRUCTION_FILTER_MITCHELL || filter == RECONSTRUCTION_FILTER_LANCZOS2;
}

#endif
