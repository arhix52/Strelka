#pragma once

// Subsurface scattering as a bounded interior medium.
//
// A path that samples the diffuse-transmission lobe of a material with
// `subsurface > 0` does not leave on the far side; it enters a scattering medium
// bounded by the surface and random-walks until it comes back out. That is what
// makes wax, marble, soap and a leaf read as translucent rather than as matte
// paint with a darker back.
//
// Deliberately *not* a BSSRDF. The separable diffusion approximations need a
// disk probe around the entry point -- extra traversal, and a shape assumption
// the geometry here does not honour: the cactus spines and the sink worktop are
// nothing like a half-space. The walk is what the wavefront can already do, and
// it reuses the free-flight machinery the fog path established.
//
// What the walk does not do: next-event estimation from inside the medium. The
// boundary occludes almost every shadow ray a dense medium would spawn, so the
// cost is real and the contribution is not. Cycles' random walk makes the same
// call. Light enters and leaves through the surface, where NEE does run.

#include <metal_stdlib>

using namespace metal;

// Extinction per channel from the mean free path. The colour of subsurface
// scattering lives here rather than in an albedo: red travels furthest, so a
// thin edge goes red before it goes bright.
static inline float3 sssSigmaT(float3 radius)
{
    return 1.0f / max(radius, float3(1e-5f));
}

// Which channel drives the next free flight, in proportion to what the path is
// still carrying.
//
// Choosing uniformly is unbiased and unusable: in a medium whose extinction
// differs threefold between channels, the balance-heuristic weight below can
// exceed one for whichever channel the sampled distance happened to suit, and
// over a walk of tens of steps those factors compound into fireflies. Weighting
// the choice by throughput times albedo makes the dominant channel the one whose
// distance is sampled, which is what collapses that tail. Cycles picks its
// channel the same way, after Chiang et al.'s production subsurface paper.
static inline float3 sssChannelPdf(float3 throughput, float3 albedo)
{
    const float3 w = abs(throughput * albedo);
    const float sum = w.x + w.y + w.z;
    return (sum > 0.0f) ? (w / sum) : float3(1.0f / 3.0f);
}

// Free flight, with one channel chosen per step.
//
// A single scalar extinction would lose the colour the medium is for, and
// sampling all three at once is not a thing free flight can do -- so one channel
// drives the distance and the weight below is multiple-importance-sampled across
// all three, which is what keeps the estimator unbiased for the other two.
//
// Returns true when the walk scatters before reaching `surfaceT`.
static inline bool sssSampleDistance(
    float3 sigmaT, float3 channelPdf, float surfaceT, float uChannel, float uDist, thread float& t)
{
    int c = 2;
    float cdf = channelPdf.x;
    if (uChannel < cdf)
    {
        c = 0;
    }
    else if (uChannel < (cdf += channelPdf.y))
    {
        c = 1;
    }
    const float st = sigmaT[c];
    if (!(st > 0.0f))
    {
        return false;
    }
    t = -log(max(1.0f - uDist, 1e-7f)) / st;
    return t < surfaceT;
}

// Throughput weight for scattering at `t`, balance-heuristic over the three
// channels that could have produced that distance. `channelPdf` has to be the
// same distribution the channel was drawn from, or the two stop cancelling.
static inline float3 sssScatterWeight(float3 sigmaT, float3 albedo, float3 channelPdf, float t)
{
    const float3 tr = exp(-sigmaT * t);
    const float3 pdfPerChannel = sigmaT * tr;
    const float pdf = dot(channelPdf, pdfPerChannel);
    if (!(pdf > 0.0f))
    {
        return float3(0.0f);
    }
    // sigma_s = albedo * sigma_t: the fraction of an extinction event that
    // scatters rather than absorbs.
    return (albedo * pdfPerChannel) / pdf;
}

// Throughput weight for reaching the boundary at `t` without scattering, over
// the same three channels.
static inline float3 sssBoundaryWeight(float3 sigmaT, float3 channelPdf, float t)
{
    const float3 tr = exp(-sigmaT * t);
    const float pdf = dot(channelPdf, tr);
    if (!(pdf > 0.0f))
    {
        return float3(0.0f);
    }
    return tr / pdf;
}

// Cosine-distributed direction about `n`, for leaving the medium at the
// boundary. The interface is treated as rough on the way out for the same reason
// it is on the way in: a specular exit would need the walk to track which side
// of a refracting interface it is on, and the materials this serves are not
// polished glass.
static inline float3 sssCosineDirection(float3 n, float u1, float u2)
{
    const float r = sqrt(u1);
    const float phi = 2.0f * M_PI_F * u2;
    const float3 t = normalize(abs(n.z) < 0.999f ? cross(float3(0.0f, 0.0f, 1.0f), n)
                                                 : cross(float3(1.0f, 0.0f, 0.0f), n));
    const float3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0f - u1, 0.0f)));
}
