#ifndef STRELKA_OPTIX_SHADING_MEDIUM_H
#define STRELKA_OPTIX_SHADING_MEDIUM_H

// ============================================================================
// medium.h -- participating media on the OptiX backend
//
// Two features, one machine. A subsurface random walk and a bounded fog volume
// differ in where light enters, not in what happens once it is inside: both
// sample a free flight against a spectral extinction, scatter off a
// Henyey-Greenstein phase function, and leave through a boundary. This is the
// CUDA counterpart of src/shaders/metal/subsurface.h plus the medium half of
// wavefront.metal, ported behaviour-for-behaviour.
//
// The arithmetic proper lives in src/render/optix/medium_walk.h, which compiles
// on the host and is covered by tests/render/test_medium_walk.cpp. What is here
// is the part that needs float3 and a ray.
// ============================================================================

#include <optix.h>

#include <OptixRenderParams.h>
#include <medium_walk.h>

#include <sutil/vec_math.h>

// The device wrapper reinterprets rather than converts, so a change to either
// layout has to be noticed here rather than silently reordering a spectrum.
static_assert(sizeof(oka::medium::Spectrum) == sizeof(float3),
              "medium::Spectrum must be layout-compatible with float3");

static __forceinline__ __device__ oka::medium::Spectrum toSpectrum(float3 v)
{
    return oka::medium::makeSpectrum(v.x, v.y, v.z);
}

static __forceinline__ __device__ float3 fromSpectrum(const oka::medium::Spectrum& s)
{
    return make_float3(s.x, s.y, s.z);
}

/// Everything the free-flight decision needs about the medium a path is inside.
struct MediumSample
{
    /// Extinction per channel.
    float3 sigmaT;
    /// Single-scattering albedo: the fraction of an extinction event that
    /// scatters rather than absorbs.
    float3 albedo;
    /// The distribution the driving channel was drawn from. The weights have to
    /// be taken against the same one.
    float3 channelPdf;
    /// Distance to the sampled scattering event.
    float t;
    /// Whether that event precedes the surface the ray would otherwise reach.
    bool scattered;
};

/// Draw a free flight for a path inside `sigmaT`, bounded by `surfaceT`.
static __forceinline__ __device__ MediumSample sampleMedium(
    float3 sigmaT, float3 albedo, float3 throughput, float surfaceT, float uChannel, float uDist)
{
    MediumSample m;
    m.sigmaT = sigmaT;
    m.albedo = albedo;
    m.channelPdf = fromSpectrum(oka::medium::channelPdf(toSpectrum(throughput), toSpectrum(albedo)));
    m.t = 0.0f;
    m.scattered = oka::medium::sampleDistance(toSpectrum(sigmaT), toSpectrum(m.channelPdf), surfaceT,
                                              uChannel, uDist, m.t);
    return m;
}

static __forceinline__ __device__ float3 mediumScatterWeight(const MediumSample& m, float t)
{
    return fromSpectrum(oka::medium::scatterWeight(toSpectrum(m.sigmaT), toSpectrum(m.albedo),
                                                   toSpectrum(m.channelPdf), t));
}

static __forceinline__ __device__ float3 mediumBoundaryWeight(const MediumSample& m, float t)
{
    return fromSpectrum(
        oka::medium::boundaryWeight(toSpectrum(m.sigmaT), toSpectrum(m.channelPdf), t));
}

static __forceinline__ __device__ float hgPhaseFunction(float cosTheta, float g)
{
    return oka::medium::hgPhase(cosTheta, g);
}

/// Sample Henyey-Greenstein about the direction of travel. pdf == phase, so the
/// two cancel and the throughput carries only the albedo.
///
/// `wo` points back the way the ray came, matching the surface convention, so
/// the forward lobe is built around -wo.
static __forceinline__ __device__ float3 hgSampleDirection(
    float3 wo, float g, float u1, float u2, float& pdf)
{
    const float cosTheta = oka::medium::hgSampleCosine(g, u1);
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = 2.0f * M_PIf * u2;

    const float3 w = -wo;
    const float3 up =
        (fabsf(w.z) < 0.999f) ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    const float3 v = normalize(cross(up, w));
    const float3 u = cross(w, v);

    pdf = oka::medium::hgPhase(cosTheta, g);
    return normalize(sinTheta * cosf(phi) * u + sinTheta * sinf(phi) * v + cosTheta * w);
}

/// Cosine-distributed direction about `n`, for leaving the medium at the
/// boundary.
///
/// The interface is treated as rough on the way out for the same reason it is
/// on the way in: a specular exit would need the walk to track which side of a
/// refracting interface it is on, and the materials this serves are not
/// polished glass.
static __forceinline__ __device__ float3 mediumCosineDirection(float3 n, float u1, float u2)
{
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    const float3 t = normalize((fabsf(n.z) < 0.999f) ? cross(make_float3(0.0f, 0.0f, 1.0f), n)
                                                     : cross(make_float3(1.0f, 0.0f, 0.0f), n));
    const float3 b = cross(n, t);
    return normalize(t * (r * cosf(phi)) + b * (r * sinf(phi)) + n * sqrtf(fmaxf(1.0f - u1, 0.0f)));
}

/// A sampler decorrelated from the path's own sequence by the walk step.
///
/// A walk step deliberately does not spend a bounce, so the raygen loop does not
/// advance `sampler.depth` across it and every step of a walk would otherwise
/// draw the same numbers. Metal reaches the same place from the other side --
/// `samplerFor(..., depth + step)`.
static __forceinline__ __device__ SamplerState mediumSampler(const SamplerState& base,
                                                             uint32_t step)
{
    SamplerState s = base;
    s.depth = base.depth + step;
    s.seed = hash_combine(base.seed, step * 0x9E3779B9u + 0x2545F491u);
    return s;
}

#endif // STRELKA_OPTIX_SHADING_MEDIUM_H
