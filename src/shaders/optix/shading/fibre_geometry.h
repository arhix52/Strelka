#ifndef STRELKA_OPTIX_FIBRE_GEOMETRY_H
#define STRELKA_OPTIX_FIBRE_GEOMETRY_H

// ============================================================================
// fibre_geometry.h -- where a ray that scattered *through* a strand comes out.
//
// Deliberately free of CUDA and OptiX: it is plain vector arithmetic over
// material_math.h's float3, so the same code compiles into the closest-hit
// program and into the unit tests (tests/render/test_fibre_chord.cpp), which is
// the only way any of this can be checked without a GPU.
// ============================================================================

#include <strelka/material/material_math.h>

// NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)
//
// NVCC and host tests compile this header, while clang-tidy sees only the host
// build. Initialising out-parameters or GPU-bound structs would add dead stores.

// The Chiang hair lobe is a whole-fibre model: its transmission term is the
// absorption over the chord *inside* the strand, so a direction leaving on the
// far side has already been charged for the crossing. Starting such a ray on the
// surface it came from puts the strand in its way -- a shadow ray dies on its own
// fibre, and a bounce ray hits the far wall and buys a second whole-fibre event
// that the first one already contains.
//
// The strand is a cylinder of radius r about `tangent`, and the hit sits on its
// surface along `normal`. In the plane across the axis the chord from that point
// along the ray is -2r(n.u), where u is the ray direction projected into that
// plane and renormalised; the distance travelled to cover it is that chord over
// the length the direction itself has in the plane.
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
