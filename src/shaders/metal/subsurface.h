#pragma once

#include <metal_stdlib>

using namespace metal;

// Extinction per channel from the mean free path. The colour of subsurface
// scattering lives here rather than in an albedo: red travels furthest, so a
// thin edge goes red before it goes bright.
static inline float3 sssSigmaT(float3 radius)
{
    return 1.0f / max(radius, float3(1e-5f));
}

static inline float3 sssChannelPdf(float3 throughput, float3 albedo)
{
    const float3 w = abs(throughput * albedo);
    const float sum = w.x + w.y + w.z;
    return (sum > 0.0f) ? (w / sum) : float3(1.0f / 3.0f);
}

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

static inline float3 sssCosineDirection(float3 n, float u1, float u2)
{
    const float r = sqrt(u1);
    const float phi = 2.0f * M_PI_F * u2;
    const float3 t = normalize(abs(n.z) < 0.999f ? cross(float3(0.0f, 0.0f, 1.0f), n)
                                                 : cross(float3(1.0f, 0.0f, 0.0f), n));
    const float3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0f - u1, 0.0f)));
}
