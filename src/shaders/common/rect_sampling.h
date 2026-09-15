#ifndef STRELKA_RECT_SAMPLING_H
#define STRELKA_RECT_SAMPLING_H

#include <strelka/material/material_math.h>

struct SphQuad
{
    float3 o;
    float3 x;
    float3 y;
    float3 z;
    float z0;
    float x0;
    float y0;
    float x1;
    float y1;
    float b0;
    float b1;
    float b0sq;
    float g2;
    float g3;
    float S;
    bool useAreaFallback;
};

/// Build the spherical-rectangle frame for the light spanned by `p0`, `p0 + ex`
/// and `p0 + ey`, as seen from `o`.
DEVICE_FUNC SphQuad sphQuadInit(float3 p0, float3 ex, float3 ey, float3 o)
{
    SphQuad squad = {};

    const float exl = finiteVectorLength(ex);
    const float eyl = finiteVectorLength(ey);

    if (!(exl > 0.0f) || !(eyl > 0.0f))
    {
        squad.useAreaFallback = true;
        return squad;
    }

    squad.o = o;
    squad.x = normalizeFiniteVectorOrZero(ex);
    squad.y = normalizeFiniteVectorOrZero(ey);
    if (dot(squad.x, squad.y) != 0.0f)
    {
        squad.S = 1.0f; // positive sentinel; unused by the area fallback
        squad.useAreaFallback = true;
        return squad;
    }
    squad.z = cross(squad.x, squad.y);

    const float3 d = p0 - o;
    constexpr float maxSquaredCoordinate = 3.402823466e38f / 8.0f;
    const float coordinateLimit = sqrtf(maxSquaredCoordinate);
    if (exl > coordinateLimit || eyl > coordinateLimit || finiteVectorLength(d) > coordinateLimit)
    {
        squad.S = 1.0f;
        squad.useAreaFallback = true;
        return squad;
    }
    squad.z0 = dot(d, squad.z);
    if (squad.z0 > 0.0f)
    {
        squad.z = squad.z * -1.0f;
        squad.z0 = -squad.z0;
    }

    squad.x0 = dot(d, squad.x);
    squad.y0 = dot(d, squad.y);
    squad.x1 = squad.x0 + exl;
    squad.y1 = squad.y0 + eyl;

    // Compact edge-normal z components: (-y0, x1, y1, -x0) / hypot(*, z0).
    float nz0 = -squad.y0;
    float nz1 = squad.x1;
    float nz2 = squad.y1;
    float nz3 = -squad.x0;
    nz0 /= sqrtf(nz0 * nz0 + squad.z0 * squad.z0);
    nz1 /= sqrtf(nz1 * nz1 + squad.z0 * squad.z0);
    nz2 /= sqrtf(nz2 * nz2 + squad.z0 * squad.z0);
    nz3 /= sqrtf(nz3 * nz3 + squad.z0 * squad.z0);

    const float g0 = asinf(fminf(fmaxf(-nz0 * nz1, -1.0f), 1.0f));
    const float g1 = asinf(fminf(fmaxf(-nz1 * nz2, -1.0f), 1.0f));
    const float g2 = asinf(fminf(fmaxf(-nz2 * nz3, -1.0f), 1.0f));
    const float g3 = asinf(fminf(fmaxf(-nz3 * nz0, -1.0f), 1.0f));

    squad.S = -(g0 + g1 + g2 + g3);
    squad.g2 = g2;
    squad.g3 = g3;
    squad.b0 = nz0;
    squad.b1 = nz2;
    squad.b0sq = squad.b0 * squad.b0;

    const float nzMinSq = fminf(fminf(nz0 * nz0, nz1 * nz1), fminf(nz2 * nz2, nz3 * nz3));
    squad.useAreaFallback = (squad.S < 1e-5f) || (nzMinSq > 0.99999f);
    return squad;
}

/// A point on the rectangle, drawn uniformly in solid angle from `o`.
DEVICE_FUNC float3 sphQuadSample(const THREAD_REF SphQuad& squad, float u, float v)
{
    const float au = u * squad.S + squad.g2 + squad.g3;
    const float sinAu = sinf(au);
    const float fu = (fabsf(sinAu) > 1e-8f) ? (cosf(au) * squad.b0 + squad.b1) / sinAu : 0.0f;
    float cu = copysignf(1.0f / sqrtf(fu * fu + squad.b0sq), fu);
    cu = fminf(fmaxf(cu, -1.0f), 1.0f);

    float xu = -(cu * squad.z0) / fmaxf(sqrtf(1.0f - cu * cu), 1e-7f);
    xu = fminf(fmaxf(xu, squad.x0), squad.x1);

    const float d2 = xu * xu + squad.z0 * squad.z0;
    const float h0 = squad.y0 / sqrtf(d2 + squad.y0 * squad.y0);
    const float h1 = squad.y1 / sqrtf(d2 + squad.y1 * squad.y1);
    const float hv = h0 + v * (h1 - h0);
    const float hv2 = hv * hv;
    const float yv = (hv2 < 1.0f - 1e-6f) ? hv * sqrtf(d2 / (1.0f - hv2)) : squad.y1;

    return squad.o + xu * squad.x + yv * squad.y + squad.z0 * squad.z;
}

DEVICE_FUNC float sphQuadSolidAngle(float3 p0, float3 ex, float3 ey, float3 o, THREAD_REF bool& useAreaFallback)
{
    const SphQuad squad = sphQuadInit(p0, ex, ey, o);
    useAreaFallback = squad.useAreaFallback;
    return squad.S;
}

#endif // STRELKA_RECT_SAMPLING_H
