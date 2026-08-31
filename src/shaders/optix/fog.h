#pragma once

// Homogeneous atmospheric scattering in a slab below a fixed height. Bounded
// media use shading/medium.h, which also owns the shared phase-function helpers.

#include <sutil/vec_math.h>

/// The stretch of a ray that lies inside the slab, clipped to [0, tMax].
/// Returns false when the ray never enters it.
static __forceinline__ __device__ bool fogSegment(
    float3 origin, float3 direction, float tMax, float height, float& t0, float& t1)
{
    t0 = 0.0f;
    t1 = tMax;

    const float oy = origin.y;
    const float dy = direction.y;
    if (fabsf(dy) < 1e-6f)
    {
        // Travelling along the slab: either wholly in or wholly out.
        return oy < height;
    }

    const float tPlane = (height - oy) / dy;
    if (dy > 0.0f)
    {
        // Rising: inside until it crosses the ceiling.
        if (oy >= height)
        {
            return false;
        }
        t1 = fminf(t1, tPlane);
    }
    else if (oy >= height)
    {
        // Falling from above: outside until it crosses the ceiling.
        t0 = fmaxf(t0, tPlane);
    }
    return t1 > t0;
}

/// Optical depth along a ray, for shadow rays and any other transmittance query.
static __forceinline__ __device__ float fogOpticalDepth(
    float3 origin, float3 direction, float tMax, float height, float sigmaT)
{
    float t0, t1;
    if (sigmaT <= 0.0f || !fogSegment(origin, direction, tMax, height, t0, t1))
    {
        return 0.0f;
    }
    return sigmaT * (t1 - t0);
}

/// Free-flight distance sampling, analog: the probability of reaching the far end
/// is exactly exp(-sigma_t * L), so the throughput needs no correction on the
/// surface branch and only the single-scattering albedo on the scatter branch.
/// Getting this wrong is invisible in a thin medium and doubles the haze in a
/// thick one.
///
/// Returns true when the ray scatters before tMax, with `distance` set.
static __forceinline__ __device__ bool fogSampleDistance(float3 origin,
                                                         float3 direction,
                                                         float tMax,
                                                         float height,
                                                         float sigmaT,
                                                         float u,
                                                         float& distance)
{
    float t0, t1;
    if (sigmaT <= 0.0f || !fogSegment(origin, direction, tMax, height, t0, t1))
    {
        return false;
    }

    const float span = t1 - t0;
    // -log(1 - u) rather than -log(u): u == 0 is a legitimate draw from most
    // samplers and log(0) is not.
    const float t = -logf(fmaxf(1.0f - u, 1e-7f)) / sigmaT;
    if (t >= span)
    {
        return false;
    }

    distance = t0 + t;
    return true;
}
