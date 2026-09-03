#ifndef STRELKA_LIGHT_PDF_H
#define STRELKA_LIGHT_PDF_H

// ============================================================================
// light_pdf.h -- the densities both halves of the MIS estimate divide by.
//
// Multiple importance sampling balances two strategies only while they agree on
// the density of every direction both can produce, and each half is unbiased
// only while the density it divides by is the one its sampler actually drew
// from. Neither statement is about a backend, so the numbers live here: OptiX
// reads them through common/lights.h, Metal through metal/lights_metal.h, and
// tests/render/test_light_pdf.cpp compiles the same lines on the host.
//
// They were two hand-maintained copies before, and the copies drifted:
//   - the sphere light reported 1/(4pi) for a point drawn uniformly over the
//     sphere's *area*, which is not a density in either measure. Both halves
//     used the same wrong number so the weights still summed to one, and the
//     next-event estimator was over a hundred times too bright at four radii of
//     distance -- an error that grows as (d/r)^2, so it hides in a closeup;
//   - the distant light's density was written 1/(2pi(1-cos a)) on one side and
//     1/(4pi sin^2(a/2)) on the other. Algebraically equal, but 1 - cos loses
//     five digits to cancellation at sun-sized angles, so the two halves
//     disagreed by 0.2% at 0.0046 rad and by 4.9% at 0.001;
//   - LIGHT_TYPE_DOME was missing from the Metal switch entirely, which a
//     fall-through turned into a silently black light.
//
// Deliberately free of CUDA, Metal and UniformLight: scalars in, density out.
// A backend unpacks its own light struct and calls these.
// ============================================================================

#include <light_types.h>
#include <strelka/material/material_math.h>

// A point or spot light whose radius is at or below this is a delta light: it
// is sampled as a single point and has no solid angle for a BSDF ray to find.
// Above it the emitter is a sphere of that radius. The two sides of the test
// have to agree on the number, so it is stated once.
#define STRELKA_SOFT_LIGHT_RADIUS_MIN 1e-4f

// ---------------------------------------------------------------------------
// MIS heuristics
//
// Written as 1/(1 + (b/a)^k) rather than a^k/(a^k + b^k): a light pdf and a BSDF
// pdf routinely differ by ten orders of magnitude, and the ratio form neither
// overflows on a peaked GGX lobe nor underflows on a distant light.
//
// The a <= 0 branch is not defensive noise. `a` is the strategy being weighted,
// `b` the one it is weighed against, and both can legitimately be zero: a light
// whose spherical-quadrilateral solid angle underflowed reports pdf 0, and so
// does a BSDF sample that ended on a delta lobe. b/a is then 0/0, and the NaN
// that produced propagated straight into the pixel. A direction neither
// strategy claims is given to the one that actually produced it, which is the
// caller of this function.
// ---------------------------------------------------------------------------
DEVICE_FUNC float misWeightBalance(float a, float b)
{
    if (!(a > 0.0f))
    {
        return (b > 0.0f) ? 0.0f : 1.0f;
    }
    return 1.0f / (1.0f + (b / a));
}

DEVICE_FUNC float misWeightPower(float a, float b)
{
    if (!(a > 0.0f))
    {
        return (b > 0.0f) ? 0.0f : 1.0f;
    }
    const float r = b / a;
    return 1.0f / (1.0f + r * r);
}

// Dispatch: 0 = balance heuristic, 1 = power heuristic.
DEVICE_FUNC float computeMisWeight(float a, float b, unsigned int heuristic)
{
    return (heuristic == 1u) ? misWeightPower(a, b) : misWeightBalance(a, b);
}

// ---------------------------------------------------------------------------
// Densities, all with respect to solid angle at the shading vertex
// ---------------------------------------------------------------------------

/// True when a sampled point on a light faces the shading vertex, given
/// `cosAtLight` = -dot(L, lightNormal).
///
/// Strictly greater than zero, and both halves of the MIS estimate have to ask
/// it exactly this way. Next-event estimation on Metal used to require 1e-3
/// while the light hit accepted anything above 0, so a sliver of grazing
/// directions was one the bounce ray was discounted for and the connection never
/// offered -- a deduction with no delivery. The band is narrow; the rule that it
/// has to be the same band is not.
DEVICE_FUNC bool lightSampleFacesVertex(float cosAtLight)
{
    return cosAtLight > 0.0f;
}

/// A point drawn from an emitter's area measure, converted to solid angle.
///
/// `area` is the reciprocal local area density, `1/p_A`. It is the total area
/// for uniform-area emitters; for an affine ellipsoid sampled by mapping a
/// uniform unit-sphere direction it is `4*pi*J_A(n_object)`. Thus
/// p_omega = d^2 / (cos(theta_light) * area).
DEVICE_FUNC float areaLightSolidAnglePdf(float distToLight, float cosAtLight, float area)
{
    if (!lightSampleFacesVertex(cosAtLight) || !(area > 0.0f))
    {
        return 0.0f;
    }
    return (distToLight * distToLight) / (cosAtLight * area);
}

/// Emissive area of a sphere of radius r. Named because the estimator, the pdf
/// and the host's radiometric bake all have to use the same one.
DEVICE_FUNC float sphereLightArea(float radius)
{
    return 4.0f * M_PI_F * radius * radius;
}

/// A sphere light sampled uniformly over its surface.
///
/// The whole sphere, not the visible cap: the sampler picks a direction on the
/// unit sphere and places the point at centre + r*dir, so p_A = 1/(4 pi r^2) and
/// the far half is rejected by the caller's facing test rather than never drawn.
/// That costs half the samples to variance and nothing to correctness.
DEVICE_FUNC float sphereLightSolidAnglePdf(float distToLight, float cosAtLight, float radius)
{
    return areaLightSolidAnglePdf(distToLight, cosAtLight, sphereLightArea(radius));
}

/// A dome: uniform over the whole sphere of directions.
///
/// The full sphere and not a hemisphere, because a dome is the analytic form of
/// an environment and an environment lights a surface from below as well as
/// above once anything reflects.
DEVICE_FUNC float domeLightSolidAnglePdf()
{
    return 1.0f / (4.0f * M_PI_F);
}

/// Clamp an authored distant-light half angle to the only geometrically
/// meaningful range. `!(a > 0)` deliberately also catches NaN: a malformed
/// angle must not reach sin/cos and contaminate a path.
DEVICE_FUNC float distantLightHalfAngle(float halfAngle)
{
    return (halfAngle > 0.0f) ? fminf(halfAngle, M_PI_F) : 0.0f;
}

/// A zero-width distant light is a singular direction, not an arbitrarily
/// narrow continuous cone.
DEVICE_FUNC bool distantLightIsDelta(float halfAngle)
{
    return distantLightHalfAngle(halfAngle) == 0.0f;
}

/// The solid angle of a cone of the given half angle.
///
/// 4 pi sin^2(a/2), never 2 pi (1 - cos a). The two are equal in exact
/// arithmetic and not at all in floats at the angles that matter: the sun is
/// 0.0046 rad, where 1 - cos loses five digits to cancellation, and anything
/// narrower rounds towards zero. coneSolidAngle() in scene/light_desc.h bakes a
/// distant light's radiance by dividing by this quantity and the shader divides
/// it back out, so the two have to be the same number.
DEVICE_FUNC float coneSolidAngleFromHalfAngle(float halfAngle)
{
    const float s = sinf(0.5f * distantLightHalfAngle(halfAngle));
    return 4.0f * M_PI_F * s * s;
}

/// A distant light sampled uniformly inside its cone.
DEVICE_FUNC float coneLightSolidAnglePdf(float halfAngle)
{
    const float omega = coneSolidAngleFromHalfAngle(halfAngle);
    return (omega > 0.0f) ? (1.0f / omega) : 0.0f;
}

/// The placeholder a delta light carries in the pdf field.
///
/// Not a density: a sharp point has no solid angle. It is 1 so that dividing the
/// contribution by it is a no-op, and it must never reach a MIS heuristic --
/// see lightIsDeltaForMis().
DEVICE_FUNC float deltaLightPdf()
{
    return 1.0f;
}

/// Radiance of a sphere of radius r that emits a given radiant intensity.
///
/// A point light's colour is intensity (W/sr) and a sphere light's is radiance,
/// so a point light given a radius has to be converted or the two disagree.
/// A uniformly emitting sphere of radiance L has intensity I = pi r^2 L, hence
/// L = I / (pi r^2). Getting this wrong is not a subtle shift: the code that
/// kept the inverse-square law *and* divided by the sphere's solid-angle density
/// made a lamp jump by 4 pi the moment its radius crossed the softness
/// threshold, in a direction nothing about the geometry justified.
DEVICE_FUNC float sphereRadianceFromIntensity(float radius)
{
    const float r = fmaxf(radius, STRELKA_SOFT_LIGHT_RADIUS_MIN);
    return 1.0f / (M_PI_F * r * r);
}

/// True when a point or spot light of this radius is sampled as a sphere rather
/// than as a single point.
DEVICE_FUNC bool punctualLightIsSoft(float radius)
{
    return radius > STRELKA_SOFT_LIGHT_RADIUS_MIN;
}

/// True when a light is a lamp at a point rather than a surface: point, spot and
/// projector.
///
/// The three share a packing (UniformLight::points[1] is the position, points[2]
/// and [3] the local axes, normal the emission axis), a sampler, and a colour
/// that means radiant intensity rather than radiance. They differ only in the
/// angular profile applied on top -- isotropic, a cone, or an image -- so
/// everything that asks "is there a surface here" wants this one question and
/// not a list that a fourth kind of lamp would have to be added to.
DEVICE_FUNC bool lightIsPunctual(int type)
{
    return type == LIGHT_TYPE_POINT || type == LIGHT_TYPE_SPOT || type == LIGHT_TYPE_PROJECTOR;
}

/// True when next-event estimation owns every direction reaching this light, so
/// its contribution must not be weighed against the BSDF pdf.
///
/// Every point, spot and projector, whatever its radius, plus a zero-angle
/// distant light. Punctual proxy geometry has a zero visibility mask in both
/// backends, so the BSDF strategy cannot reach it. The distant case is singular
/// by definition. Neither kind has a continuous solid-angle competitor.
DEVICE_FUNC bool lightIsDeltaForMis(int type, float halfAngle)
{
    return lightIsPunctual(type) || (type == LIGHT_TYPE_DISTANT && distantLightIsDelta(halfAngle));
}

DEVICE_FUNC bool lightIsInfinite(int type)
{
    return type == LIGHT_TYPE_DISTANT || type == LIGHT_TYPE_DOME;
}

/// Conditional solid-angle density for evaluating an analytic infinite light
/// along a direction selected by the BSDF. `cosToAxis` is dot(W, -normal) for a
/// distant light and is ignored for a dome.
DEVICE_FUNC float infiniteLightConditionalPdf(int type, float halfAngle, float cosToAxis)
{
    if (type == LIGHT_TYPE_DOME)
    {
        return domeLightSolidAnglePdf();
    }
    if (type != LIGHT_TYPE_DISTANT || distantLightIsDelta(halfAngle))
    {
        return 0.0f;
    }

    const float angle = distantLightHalfAngle(halfAngle);
    return cosToAxis >= cosf(angle) ? coneLightSolidAnglePdf(angle) : 0.0f;
}

/// Everything the density of one light depends on, unpacked from whichever
/// light struct the backend holds.
struct LightPdfQuery
{
    int type;
    float distToLight; ///< shading vertex to the point on the light
    float cosAtLight; ///< -dot(L, light normal); <= 0 means the sample faces away
    float area; ///< reciprocal local area density; total area when p_A is uniform
    float radius; ///< point/spot/projector soft radius
    float halfAngle; ///< distant light's cone half angle
    float solidAngle; ///< rect solid-angle sampling: > 0 selects 1/S over the area form
};

DEVICE_FUNC LightPdfQuery makeLightPdfQuery(int type)
{
    LightPdfQuery q = {};
    q.type = type;
    q.distToLight = 0.0f;
    q.cosAtLight = 0.0f;
    q.area = 0.0f;
    q.radius = 0.0f;
    q.halfAngle = 0.0f;
    q.solidAngle = 0.0f;
    return q;
}

/// The density the light-sampling strategy has for the direction described by
/// `q`. One switch for both backends and for the host tests, so a light type
/// cannot be sampled by one of them and be invisible to the other -- which is
/// exactly what a missing LIGHT_TYPE_DOME case did on Metal.
DEVICE_FUNC float lightSolidAnglePdf(const THREAD_REF LightPdfQuery& q)
{
    switch (q.type)
    {
    case LIGHT_TYPE_RECT:
        // Solid-angle sampling when the spherical quadrilateral is well
        // conditioned, the area form when the sampler fell back to it. The
        // caller decides which by passing S or leaving it at zero, and it has to
        // make that decision the same way in both places or the halves disagree.
        return (q.solidAngle > 0.0f) ? (1.0f / q.solidAngle) :
                                       areaLightSolidAnglePdf(q.distToLight, q.cosAtLight, q.area);
    case LIGHT_TYPE_DISC:
        return areaLightSolidAnglePdf(q.distToLight, q.cosAtLight, q.area);
    case LIGHT_TYPE_SPHERE:
        return areaLightSolidAnglePdf(q.distToLight, q.cosAtLight, q.area);
    case LIGHT_TYPE_DISTANT:
        return coneLightSolidAnglePdf(q.halfAngle);
    case LIGHT_TYPE_DOME:
        return domeLightSolidAnglePdf();
    case LIGHT_TYPE_POINT:
    case LIGHT_TYPE_SPOT:
    case LIGHT_TYPE_PROJECTOR:
        // A soft one is a sphere and is divided by a real density; a sharp one
        // carries the placeholder. Both are exempt from the MIS heuristic --
        // see lightIsDeltaForMis(). The projector's image modulates how much
        // light leaves in a direction, not how the sampler picked it, so its
        // density is the point light's.
        return punctualLightIsSoft(q.radius) ? sphereLightSolidAnglePdf(q.distToLight, q.cosAtLight, q.radius) :
                                               deltaLightPdf();
    default:
        break;
    }
    return 0.0f;
}

/// A direction drawn uniformly over the unit sphere from two canonical variates.
///
/// Shared by the sphere and dome samplers in both backends so that the density
/// above describes the points they actually produce.
DEVICE_FUNC float3 uniformSphereDirection(float u1, float u2)
{
    const float cosTheta = 1.0f - 2.0f * u1; // uniform on [-1, 1]
    const float sinTheta = sqrtf(fmaxf(1.0f - cosTheta * cosTheta, 0.0f));
    const float phi = 2.0f * M_PI_F * u2;
    return make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

#endif // STRELKA_LIGHT_PDF_H
