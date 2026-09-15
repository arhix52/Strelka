#pragma once

// Host-testable arithmetic for the OptiX spectral free-flight walk. The device
// wrapper in shading/medium.h adds geometry.

#if defined(__CUDACC__)
#    define STRELKA_MEDIUM_FN __host__ __device__ inline
#else
#    define STRELKA_MEDIUM_FN inline
#endif

#include <cmath>

// NOLINTBEGIN(cppcoreguidelines-pro-type-member-init, cppcoreguidelines-init-variables)

namespace oka::medium
{

/// Three channels of anything -- extinction, albedo, a throughput. Laid out to
/// match float3 / packed_float3 so the device wrapper is a reinterpretation and
/// not a conversion.
struct Spectrum
{
    float x;
    float y;
    float z;
};

STRELKA_MEDIUM_FN Spectrum makeSpectrum(float x, float y, float z)
{
    Spectrum s;
    s.x = x;
    s.y = y;
    s.z = z;
    return s;
}

STRELKA_MEDIUM_FN float channel(const Spectrum& s, int i)
{
    return (i == 0) ? s.x : ((i == 1) ? s.y : s.z);
}

STRELKA_MEDIUM_FN float dot3(const Spectrum& a, const Spectrum& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

STRELKA_MEDIUM_FN Spectrum sigmaTFromRadius(const Spectrum& radius)
{
    const float rx = radius.x > 1e-5f ? radius.x : 1e-5f;
    const float ry = radius.y > 1e-5f ? radius.y : 1e-5f;
    const float rz = radius.z > 1e-5f ? radius.z : 1e-5f;
    return makeSpectrum(1.0f / rx, 1.0f / ry, 1.0f / rz);
}

STRELKA_MEDIUM_FN Spectrum channelPdf(const Spectrum& throughput, const Spectrum& albedo)
{
    const float wx = std::fabs(throughput.x * albedo.x);
    const float wy = std::fabs(throughput.y * albedo.y);
    const float wz = std::fabs(throughput.z * albedo.z);
    const float sum = wx + wy + wz;
    if (!(sum > 0.0f))
    {
        return makeSpectrum(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);
    }
    return makeSpectrum(wx / sum, wy / sum, wz / sum);
}

/// The channel a canonical sample selects from `pdf`.
STRELKA_MEDIUM_FN int selectChannel(const Spectrum& pdf, float u)
{
    float cdf = pdf.x;
    if (u < cdf)
    {
        return 0;
    }
    cdf += pdf.y;
    if (u < cdf)
    {
        return 1;
    }
    return 2;
}

STRELKA_MEDIUM_FN bool sampleDistance(const Spectrum& sigmaT,
                                      const Spectrum& pdf,
                                      float surfaceT,
                                      float uChannel,
                                      float uDist,
                                      float& t)
{
    const float st = channel(sigmaT, selectChannel(pdf, uChannel));
    if (!(st > 0.0f))
    {
        return false;
    }
    // -log(1 - u) rather than -log(u): u == 0 is a legitimate draw from most
    // samplers and log(0) is not.
    const float oneMinusU = (1.0f - uDist) > 1e-7f ? (1.0f - uDist) : 1e-7f;
    t = -std::log(oneMinusU) / st;
    return t < surfaceT;
}

STRELKA_MEDIUM_FN Spectrum scatterWeight(const Spectrum& sigmaT,
                                         const Spectrum& albedo,
                                         const Spectrum& pdf,
                                         float t)
{
    const Spectrum tr =
        makeSpectrum(std::exp(-sigmaT.x * t), std::exp(-sigmaT.y * t), std::exp(-sigmaT.z * t));
    const Spectrum pdfPerChannel =
        makeSpectrum(sigmaT.x * tr.x, sigmaT.y * tr.y, sigmaT.z * tr.z);
    const float density = dot3(pdf, pdfPerChannel);
    if (!(density > 0.0f))
    {
        return makeSpectrum(0.0f, 0.0f, 0.0f);
    }
    // sigma_s = albedo * sigma_t: the fraction of an extinction event that
    // scatters rather than absorbs.
    return makeSpectrum(albedo.x * pdfPerChannel.x / density, albedo.y * pdfPerChannel.y / density,
                        albedo.z * pdfPerChannel.z / density);
}

STRELKA_MEDIUM_FN Spectrum boundaryWeight(const Spectrum& sigmaT, const Spectrum& pdf, float t)
{
    const Spectrum tr =
        makeSpectrum(std::exp(-sigmaT.x * t), std::exp(-sigmaT.y * t), std::exp(-sigmaT.z * t));
    const float density = dot3(pdf, tr);
    if (!(density > 0.0f))
    {
        return makeSpectrum(0.0f, 0.0f, 0.0f);
    }
    return makeSpectrum(tr.x / density, tr.y / density, tr.z / density);
}

/// Henyey-Greenstein. g > 0 scatters forward, which is what haze around a low
/// sun does and why the effect is a glow rather than a uniform wash.
STRELKA_MEDIUM_FN float hgPhase(float cosTheta, float g)
{
    const float gg = g * g;
    const float denom = 1.0f + gg - 2.0f * g * cosTheta;
    const float safe = denom > 1e-8f ? denom : 1e-8f;
    return (1.0f - gg) / (4.0f * 3.14159265358979323846f * denom * std::sqrt(safe));
}

STRELKA_MEDIUM_FN float hgSampleCosine(float g, float u)
{
    float cosTheta;
    if (std::fabs(g) < 1e-3f)
    {
        cosTheta = 1.0f - 2.0f * u;
    }
    else
    {
        const float s = (1.0f - g * g) / (1.0f + g - 2.0f * g * u);
        cosTheta = (1.0f + g * g - s * s) / (2.0f * g);
    }
    return cosTheta < -1.0f ? -1.0f : (cosTheta > 1.0f ? 1.0f : cosTheta);
}

} // namespace oka::medium

// NOLINTEND(cppcoreguidelines-pro-type-member-init, cppcoreguidelines-init-variables)
