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
DEVICE_FUNC float3 ensureValidSpecularReflection(float3 geometricNormal, float3 wo, float3 shadingNormal)
{
    const float3 R = 2.0f * dot(shadingNormal, wo) * shadingNormal - wo;

    const float Iz = dot(wo, geometricNormal);
    if (Iz <= 0.0f)
    {
        // The view ray is behind the geometry the caller said it was in front
        // of. Nothing here can be trusted, and the geometric normal is the one
        // answer that cannot be worse than the input.
        return geometricNormal;
    }

    // How shallow a reflection is still allowed to be. Cycles' bound: a
    // reflection may always be at least as grazing as the ray that produced it,
    // capped so that a head-on view still admits a nearly tangent one.
    const float threshold = fminf(0.9f * Iz, 0.01f);
    if (dot(geometricNormal, R) >= threshold)
    {
        return shadingNormal;
    }

    // Work in the plane the correction has to happen in: geometricNormal as z,
    // and the part of the shading normal perpendicular to it as x. The rotation
    // is then two numbers instead of a quaternion.
    const float NdotNg = dot(shadingNormal, geometricNormal);
    const float3 tangentAxis = shadingNormal - NdotNg * geometricNormal;
    const float tangentLen = length(tangentAxis);
    if (tangentLen < 1e-8f)
    {
        // The shading normal is parallel to the geometric one, so there is no
        // plane to rotate in -- and nothing to correct either.
        return geometricNormal;
    }
    const float3 X = tangentAxis / tangentLen;

    const float Ix = dot(wo, X);
    const float Ix2 = Ix * Ix;
    const float Iz2 = Iz * Iz;
    const float a = Ix2 + Iz2;

    const float b2 = Ix2 * (a - threshold * threshold);
    const float b = (b2 > 0.0f) ? sqrtf(b2) : 0.0f;
    const float c = Iz * threshold + a;

    // The two normals whose reflection lands exactly on the threshold. Both are
    // expressed by the square of their z component in the frame above.
    const float fac = 0.5f / a;
    const float N1_z2 = fac * (b + c);
    const float N2_z2 = fac * (-b + c);
    bool valid1 = (N1_z2 > 1e-5f) && (N1_z2 <= 1.0f + 1e-5f);
    bool valid2 = (N2_z2 > 1e-5f) && (N2_z2 <= 1.0f + 1e-5f);

    float Nx = 0.0f;
    float Nz = 0.0f;
    if (valid1 && valid2)
    {
        // Both are geometrically possible, so pick by what they do to the
        // reflection rather than by which root came first.
        const float N1x = sqrtf(fmaxf(0.0f, 1.0f - N1_z2));
        const float N1z = sqrtf(fmaxf(0.0f, N1_z2));
        const float N2x = sqrtf(fmaxf(0.0f, 1.0f - N2_z2));
        const float N2z = sqrtf(fmaxf(0.0f, N2_z2));

        const float R1 = 2.0f * (N1x * Ix + N1z * Iz) * N1z - Iz;
        const float R2 = 2.0f * (N2x * Ix + N2z * Iz) * N2z - Iz;

        valid1 = (R1 >= 1e-5f);
        valid2 = (R2 >= 1e-5f);
        if (valid1 && valid2)
        {
            // The shallower of the two is the smaller correction.
            const bool takeFirst = (R1 < R2);
            Nx = takeFirst ? N1x : N2x;
            Nz = takeFirst ? N1z : N2z;
        }
        else if (valid1)
        {
            Nx = N1x;
            Nz = N1z;
        }
        else if (valid2)
        {
            Nx = N2x;
            Nz = N2z;
        }
        else
        {
            return geometricNormal;
        }
    }
    else if (valid1 || valid2)
    {
        const float Nz2 = valid1 ? N1_z2 : N2_z2;
        Nx = sqrtf(fmaxf(0.0f, 1.0f - Nz2));
        Nz = sqrtf(fmaxf(0.0f, Nz2));
    }
    else
    {
        return geometricNormal;
    }

    return Nx * X + Nz * geometricNormal;
}

#endif // STRELKA_MATERIAL_VALID_REFLECTION_H
