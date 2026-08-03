#pragma once

// Homogeneous atmospheric scattering.
//
// A slab rather than a bounded volume: the medium fills everything below a
// height, and outside that there is nothing. That is a deliberate narrowing.
// Production scenes describe haze with a box the size of the set containing the
// camera -- the pine forest's is 209 x 209 x 29 m at the origin -- and a slab
// reproduces it exactly while costing one comparison instead of a second
// traversal of a volume boundary, per ray, including shadow rays.
//
// What it cannot do: a medium that is not the atmosphere. Smoke in a corner, a
// beam through a doorway, fog that ends at a wall. Those need the bounded kind,
// and this is not a step toward it so much as the common case taken on its own.

#include <metal_stdlib>

using namespace metal;

// The stretch of a ray that lies inside the slab, clipped to [0, tMax].
// Returns false when the ray never enters it.
static inline bool fogSegment(float3 origin, float3 direction, float tMax, float height,
                              thread float& t0, thread float& t1)
{
    t0 = 0.0f;
    t1 = tMax;

    const float oy = origin.y;
    const float dy = direction.y;
    if (fabs(dy) < 1e-6f)
    {
        // Travelling along the slab: either wholly in or wholly out.
        return oy < height;
    }

    const float tPlane = (height - oy) / dy;
    if (dy > 0.0f)
    {
        // Rising: inside until it crosses the ceiling.
        if (oy >= height)
            return false;
        t1 = min(t1, tPlane);
    }
    else
    {
        // Falling: outside until it crosses the ceiling.
        if (oy < height)
        {
            // already inside
        }
        else
        {
            t0 = max(t0, tPlane);
        }
    }
    return t1 > t0;
}

// Optical depth along a ray, for shadow rays and any other transmittance query.
static inline float fogOpticalDepth(float3 origin, float3 direction, float tMax,
                                    float height, float sigmaT)
{
    float t0, t1;
    if (!fogSegment(origin, direction, tMax, height, t0, t1))
        return 0.0f;
    return sigmaT * (t1 - t0);
}

// Free-flight distance sampling, analog: the probability of reaching the far end
// is exactly exp(-sigma_t * L), so the throughput needs no correction on the
// surface branch and only the single-scattering albedo on the scatter branch.
// Getting this wrong is invisible in a thin medium and doubles the haze in a
// thick one.
//
// Returns true when the ray scatters before tMax, with `distance` set.
static inline bool fogSampleDistance(float3 origin, float3 direction, float tMax,
                                     float height, float sigmaT, float u,
                                     thread float& distance)
{
    float t0, t1;
    if (sigmaT <= 0.0f || !fogSegment(origin, direction, tMax, height, t0, t1))
        return false;

    const float span = t1 - t0;
    // -log(1 - u) rather than -log(u): u == 0 is a legitimate draw from most
    // samplers and log(0) is not.
    const float t = -log(max(1.0f - u, 1e-7f)) / sigmaT;
    if (t >= span)
        return false;

    distance = t0 + t;
    return true;
}

// Henyey-Greenstein. g > 0 scatters forward, which is what haze around a low sun
// does and why the effect is a glow rather than a uniform wash.
static inline float hgPhase(float cosTheta, float g)
{
    const float gg = g * g;
    const float denom = 1.0f + gg - 2.0f * g * cosTheta;
    return (1.0f - gg) / (4.0f * M_PI_F * denom * sqrt(max(denom, 1e-8f)));
}

// Sample HG about `wo`, returning the new direction. pdf == phase, so the two
// cancel and the throughput carries only the albedo.
static inline float3 hgSample(float3 wo, float g, float u1, float u2, thread float& pdf)
{
    float cosTheta;
    if (fabs(g) < 1e-3f)
    {
        cosTheta = 1.0f - 2.0f * u1;
    }
    else
    {
        const float s = (1.0f - g * g) / (1.0f + g - 2.0f * g * u1);
        cosTheta = -(1.0f + g * g - s * s) / (2.0f * g);
    }
    cosTheta = clamp(cosTheta, -1.0f, 1.0f);

    const float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = 2.0f * M_PI_F * u2;

    // A frame about the direction of travel. wo points back the way the ray
    // came, so the forward lobe is around -wo.
    const float3 w = -wo;
    const float3 up = fabs(w.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    const float3 v = normalize(cross(up, w));
    const float3 u = cross(w, v);

    pdf = hgPhase(cosTheta, g);
    return normalize(sinTheta * cos(phi) * u + sinTheta * sin(phi) * v + cosTheta * w);
}
