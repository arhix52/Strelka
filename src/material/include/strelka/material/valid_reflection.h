#ifndef STRELKA_MATERIAL_VALID_REFLECTION_H
#define STRELKA_MATERIAL_VALID_REFLECTION_H

// ============================================================================
// valid_reflection.h -- keeping a normal-mapped normal in front of the surface.
//
// A normal map is a lie the surface tells about which way it faces, and at a
// grazing angle the lie can outrun the geometry: the perturbed normal ends up
// pointing away from the viewer, so dot(N, wo) goes negative on a triangle that
// is squarely facing the camera.
//
// Nothing downstream can answer that. standard_pbr reads dot(N, wo) <= 0 the
// way a dielectric exit reads it -- only a transmission lobe may reply -- and an
// opaque rock has none, so the sample came back BSDF_EVENT_ABSORB. The
// closest-hit program terminates on absorb above next-event estimation, so the
// pixel lost its direct lighting too and came out exactly black. On the pine
// forest's mossy rock that is a patch of solid black following the normal map's
// own pattern, appearing and disappearing as the camera moves, which is what it
// was reported as.
//
// The correction is Cycles', and deliberately so: Cycles is this renderer's
// reference, and it turns the normal rather than clamping it or falling back to
// the geometric one. This finds the *smallest* rotation of N, in the plane
// spanned by Ng and the view ray, that puts the mirror direction back above the
// surface -- so the map's relief survives everywhere except where it was making
// a physically impossible claim, which is the difference between this and a
// clamp that flattens the rock.
//
// Not Schuessler et al. 2017, which is the energy-preserving microfacet
// treatment of the same problem and the academic state of the art: it would
// move this renderer *away* from Cycles, which does not implement it.
//
// Deliberately free of CUDA, OptiX and Metal: float3 arithmetic only, so the
// device code and tests/material/test_valid_reflection.cpp compile one copy.
// ============================================================================

#include <strelka/material/material_math.h>

/// The shading normal, bent just far enough that reflecting `wo` about it stays
/// on the outside of `geometricNormal`.
///
/// `wo` points back along the incoming ray, so it is on the outside of the
/// surface, and `geometricNormal` is expected to have been flipped to agree with
/// it already. Returns `shadingNormal` untouched whenever the mirror direction
/// already clears the surface, which is almost every shading point in a frame.
DEVICE_FUNC float3 ensureValidSpecularReflection(float3 Ng, float3 I, float3 N)
{
    const float3 R = 2.0f * dot(N, I) * N - I;

    const float Iz = dot(I, Ng);
    if (Iz <= 0.0f)
    {
        // Cycles asserts this away -- it only calls the function with a
        // geometric normal already turned to face the ray. Guarded rather than
        // assumed because the callers here are two backends, and the geometric
        // normal is the one answer that cannot be worse than the input.
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
