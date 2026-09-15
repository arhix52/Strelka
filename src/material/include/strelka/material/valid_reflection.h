#ifndef STRELKA_MATERIAL_VALID_REFLECTION_H
#define STRELKA_MATERIAL_VALID_REFLECTION_H

#include <strelka/material/material_math.h>

DEVICE_FUNC float3 ensureValidSpecularReflection(float3 Ng, float3 I, float3 N)
{
    const float3 R = 2.0f * dot(N, I) * N - I;

    const float Iz = dot(I, Ng);
    if (Iz <= 0.0f)
    {
        return Ng;
    }

    // A reflection may always be at least as grazing as the ray that produced
    // it, capped so a head-on view still admits a nearly tangent one.
    const float threshold = fminf(0.9f * Iz, 0.01f);
    if (dot(Ng, R) >= threshold)
    {
        return N;
    }

    // The plane the correction happens in: Ng as z, and the part of N
    // perpendicular to it as x, so the rotation is two numbers.
    const float3 Xv = N - dot(N, Ng) * Ng;
    const float xLen = length(Xv);
    const float3 X = (xLen > 1e-8f) ? (Xv / xLen) : N;

    const float Ix = dot(I, X);

    const float a = Ix * Ix + Iz * Iz;
    const float b = 2.0f * (a + Iz * threshold);
    const float c = (threshold + Iz) * (threshold + Iz);

    // The root that turns N the shorter way, which is the smaller correction.
    const float disc = b * b - 4.0f * a * c;
    const float root = (disc > 0.0f) ? sqrtf(disc) : 0.0f;
    const float Nz2 = (Ix < 0.0f) ? 0.25f * (b + root) / a : 0.25f * (b - root) / a;

    const float Nx = sqrtf(fmaxf(0.0f, 1.0f - Nz2));
    const float Nz = sqrtf(fmaxf(0.0f, Nz2));

    return Nx * X + Nz * Ng;
}

#endif // STRELKA_MATERIAL_VALID_REFLECTION_H
