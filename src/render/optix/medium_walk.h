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
//
// NVCC and host tests compile this header. Initialising immediate out-parameters
// adds dead stores, while default member initialisers alter GPU-copyable structs.

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

/// Extinction per channel from the mean free path.
///
/// The colour of subsurface scattering lives here rather than in an albedo: red
/// travels furthest, so a thin edge goes red before it goes bright.
STRELKA_MEDIUM_FN Spectrum sigmaTFromRadius(const Spectrum& radius)
{
    const float rx = radius.x > 1e-5f ? radius.x : 1e-5f;
    const float ry = radius.y > 1e-5f ? radius.y : 1e-5f;
    const float rz = radius.z > 1e-5f ? radius.z : 1e-5f;
    return makeSpectrum(1.0f / rx, 1.0f / ry, 1.0f / rz);
}

/// Which channel drives the next free flight, in proportion to what the path is
/// still carrying.
///
/// Weighting by throughput times albedo avoids compounding balance-heuristic
/// weights when channel extinction differs.
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

/// Free flight, with one channel chosen per step.
///
/// A single scalar extinction would lose the colour the medium is for, and
/// sampling all three at once is not a thing free flight can do -- so one
/// channel drives the distance and the weights below are multiple-importance-
/// sampled across all three, which is what keeps the estimator unbiased for the
/// other two.
///
/// Returns true when the walk scatters before reaching `surfaceT`.
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

/// Throughput weight for scattering at `t`, balance-heuristic over the three
/// channels that could have produced that distance. `pdf` has to be the same
/// distribution the channel was drawn from, or the two stop cancelling --
/// sampling from one density and weighting by another is how an unbiased
/// estimator stops being one.
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

/// Throughput weight for reaching a boundary at `t` without scattering, over
/// the same three channels.
///
/// This is one for grey extinction and preserves spectral throughput otherwise.
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

/// The cosine HG puts a canonical sample at, measured against the direction of
/// *travel*.
///
/// The standard inversion returns the cosine against the direction the ray came
/// from, and at g = 0.8 the median draw is -0.944 -- so taken at face value a
/// forward-scattering medium scatters backwards. It does not show up in single
/// scattering, where the outgoing direction is fixed by the camera; only once a
/// path continues does the lobe point the wrong way.
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
