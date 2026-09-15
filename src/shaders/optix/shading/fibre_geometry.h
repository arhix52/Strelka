#ifndef STRELKA_OPTIX_FIBRE_GEOMETRY_H
#define STRELKA_OPTIX_FIBRE_GEOMETRY_H

#include <strelka/material/material_math.h>

// NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)

struct FibreExit
{
    float3 position; // where the ray leaves the strand
    float3 normal;   // outward normal there -- the direction to offset along
    bool crossed;    // false when the ray leaves on the side it arrived from
};

DEVICE_FUNC FibreExit fibre_exit(float3 position,
                                 float3 tangent,
                                 float3 normal,
                                 float radius,
                                 float3 dir)
{
    FibreExit exit;
    exit.position = position;
    exit.normal = normal;
    exit.crossed = false;

    const float3 dPerp = dir - tangent * dot(dir, tangent);
    const float m2 = dot(dPerp, dPerp);
    // Straight along the strand there is no far wall, and the chord below would
    // divide by zero on the way to saying so.
    if (m2 < 1e-8f || radius <= 0.0f)
    {
        return exit;
    }
    const float m = sqrtf(m2);
    const float3 u = dPerp * (1.0f / m);
    const float chord = -2.0f * radius * dot(normal, u);
    // Leaving on the side it arrived from: an ordinary surface offset is enough.
    if (chord <= 0.0f)
    {
        return exit;
    }
    exit.position = position + dir * (chord / m);
    // The outward normal at the *exit*, not at the entry: they point to opposite
    // sides of the strand.
    exit.normal = safe_normalize(normal * radius + u * chord);
    exit.crossed = true;
    return exit;
}

#endif // STRELKA_OPTIX_FIBRE_GEOMETRY_H

// NOLINTEND(cppcoreguidelines-pro-type-member-init)
