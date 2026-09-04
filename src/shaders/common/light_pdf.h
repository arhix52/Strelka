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

// Smallest cap whose half-angle sine squared and solid-angle density remain
// normal binary32 values in every supported GPU arithmetic mode. At 2^-62 the
// half-angle sine is 2^-63 and its square is FLT_MIN even if fast math
// reassociates the solid-angle product. Narrower authored caps are represented as the same directional atom
// as an exact zero-angle distant light instead of being continuous on the CPU
// and flush-to-zero on the GPU.
#define STRELKA_MIN_CONTINUOUS_DISTANT_HALF_ANGLE 2.168404344971009e-19f

/// Represent a continuous light variate at the centre of its 23-bit float
/// lattice cell. This preserves the cell's probability while avoiding
/// finite-mass atoms on emitter edges, sphere poles, and cone boundaries.
DEVICE_FUNC float lightOpenUnitInterval(float xi)
{
    const float centred = xi + 0x1p-24f;
    return fminf(fmaxf(centred, 0x1p-24f), 0x1.fffffep-1f);
}

struct OrthonormalLightFrame
{
    float3 x;
    float3 y;
    float3 emissionAxis;
    bool valid;
};

/// Turn the affine images of local +X, +Y, and emission -Z into the rigid
/// frame an angular profile is defined in. Modified Gram-Schmidt retains the
/// transformed X/Y orientation (including mirrors) without letting scale or
/// shear distort the profile's spherical measure.
DEVICE_FUNC OrthonormalLightFrame makeOrthonormalLightFrame(float3 axisX, float3 axisY, float3 emissionAxis)
{
    OrthonormalLightFrame frame{};
    frame.emissionAxis = normalizeFiniteVectorOrZero(emissionAxis);
    frame.x = orthonormalizeTangent(frame.emissionAxis, axisX);
    const float yAlongNormal = dot(axisY, frame.emissionAxis);
    const float yAlongX = dot(axisY, frame.x);
    const float3 yResidual = make_float3(
        fmaf(-yAlongX, frame.x.x, fmaf(-yAlongNormal, frame.emissionAxis.x, axisY.x)),
        fmaf(-yAlongX, frame.x.y, fmaf(-yAlongNormal, frame.emissionAxis.y, axisY.y)),
        fmaf(-yAlongX, frame.x.z, fmaf(-yAlongNormal, frame.emissionAxis.z, axisY.z)));
    frame.y = normalizeFiniteVectorOrZero(yResidual);
    frame.valid = dot(frame.emissionAxis, frame.emissionAxis) > 0.0f && dot(frame.x, frame.x) > 0.0f &&
                  dot(frame.y, frame.y) > 0.0f;
    return frame;
}

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
    return cosAtLight > 0.0f && cosAtLight <= 3.402823466e38f;
}

/// A point drawn from an emitter's area measure, converted to solid angle.
///
/// `area` is the reciprocal local area density, `1/p_A`. It is the total area
/// for uniform-area emitters; for an affine ellipsoid sampled by mapping a
/// uniform unit-sphere direction it is `4*pi*J_A(n_object)`. Thus
/// p_omega = d^2 / (cos(theta_light) * area).
DEVICE_FUNC float areaLightSolidAnglePdf(float distToLight, float cosAtLight, float area)
{
    if (!lightSampleFacesVertex(cosAtLight) || !(distToLight > 0.0f) || !(area > 0.0f))
    {
        return 0.0f;
    }
    constexpr float maxFinite = 3.402823466e38f;
    cosAtLight = fminf(cosAtLight, 1.0f);
    const float scaledDistance = distToLight / sqrtf(area);
    const float largestFiniteDistance = sqrtf(maxFinite * cosAtLight);
    if (!(scaledDistance > 0.0f))
    {
        return 0.0f;
    }
    return scaledDistance <= largestFiniteDistance ? scaledDistance * scaledDistance / cosAtLight : maxFinite;
}

/// Area density converted to solid angle without first materialising its
/// reciprocal area/Jacobian. The multiplication order keeps p_A*d^2 finite in
/// cases where d^2 alone overflows.
DEVICE_FUNC float areaPdfToSolidAnglePdf(float distToLight, float cosAtLight, float areaPdf)
{
    if (!lightSampleFacesVertex(cosAtLight) || !(distToLight > 0.0f) || !(areaPdf > 0.0f))
    {
        return 0.0f;
    }
    constexpr float maxFinite = 3.402823466e38f;
    cosAtLight = fminf(cosAtLight, 1.0f);
    if (!(distToLight <= maxFinite) || !(areaPdf <= maxFinite))
    {
        return maxFinite;
    }
    int distanceExponent = 0;
    int cosineExponent = 0;
    int pdfExponent = 0;
#if defined(__METAL_VERSION__)
    const float distanceMantissa = metal::frexp(distToLight, distanceExponent);
    const float cosineMantissa = metal::frexp(cosAtLight, cosineExponent);
    const float pdfMantissa = metal::frexp(areaPdf, pdfExponent);
    const float result = metal::ldexp(pdfMantissa * distanceMantissa * distanceMantissa / cosineMantissa,
                                      pdfExponent + 2 * distanceExponent - cosineExponent);
#else
    const float distanceMantissa = frexpf(distToLight, &distanceExponent);
    const float cosineMantissa = frexpf(cosAtLight, &cosineExponent);
    const float pdfMantissa = frexpf(areaPdf, &pdfExponent);
    const float result = ldexpf(pdfMantissa * distanceMantissa * distanceMantissa / cosineMantissa,
                                pdfExponent + 2 * distanceExponent - cosineExponent);
#endif
    return result <= maxFinite ? result : maxFinite;
}

/// Emissive area of a sphere of radius r. Named because the estimator, the pdf
/// and the host's radiometric bake all have to use the same one.
DEVICE_FUNC float sphereLightArea(float radius)
{
    return 4.0f * M_PI_F * radius * radius;
}

DEVICE_FUNC float sphereLightAreaPdf(float radius)
{
    return radius > 0.0f ? (1.0f / (4.0f * M_PI_F * radius)) / radius : 0.0f;
}

/// A sphere light sampled uniformly over its surface.
///
/// The whole sphere, not the visible cap: the sampler picks a direction on the
/// unit sphere and places the point at centre + r*dir, so p_A = 1/(4 pi r^2) and
/// the far half is rejected by the caller's facing test rather than never drawn.
/// That costs half the samples to variance and nothing to correctness.
DEVICE_FUNC float sphereLightSolidAnglePdf(float distToLight, float cosAtLight, float radius)
{
    return areaPdfToSolidAnglePdf(distToLight, cosAtLight, sphereLightAreaPdf(radius));
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
    return distantLightHalfAngle(halfAngle) < STRELKA_MIN_CONTINUOUS_DISTANT_HALF_ANGLE;
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
    if (distantLightIsDelta(halfAngle))
    {
        return 0.0f;
    }
    const float omega = coneSolidAngleFromHalfAngle(halfAngle);
    return (omega > 0.0f) ? (1.0f / omega) : 0.0f;
}

DEVICE_FUNC bool distantLightContainsDirection(float halfAngle, float3 direction, float3 axisDirection);

/// Stable uniform spherical-cap inversion shared by CPU tests, Metal and OptiX.
///
/// `cosTheta = 1 - 2 q sin^2(a/2)` is retained for the axial component, but
/// recovering the tangent length with `sqrt(1-cosTheta^2)` loses the complete
/// sample once cosTheta rounds to one. The identity below computes sinTheta
/// directly from the half angle and the same solid-angle variate.
DEVICE_FUNC float3 sampleDistantLightDirection(float uPhi, float uSolidAngle, float halfAngle, float3 direction)
{
    // Scene packing stores a canonical unit axis. Do not renormalize it here:
    // an additional binary32 normalization can move a narrow cap by more than
    // its radius and make sample() disagree with miss-side PDF lookup.
    const float3 axis = direction;
    if (!(finiteVectorLength(axis) > 0.0f) || distantLightIsDelta(halfAngle))
    {
        return axis;
    }

    const float angle = distantLightHalfAngle(halfAngle);
    float q = fminf(fmaxf(uSolidAngle, 0.0f), 1.0f);
    const float halfSin = sinf(0.5f * angle);
    const float halfSinSquared = halfSin * halfSin;

    float3 tangent;
    if (fabsf(axis.x) > fabsf(axis.y))
    {
        tangent = normalizeFiniteVectorOrZero(make_float3(-axis.z, 0.0f, axis.x));
    }
    else
    {
        tangent = normalizeFiniteVectorOrZero(make_float3(0.0f, axis.z, -axis.y));
    }
    const float3 bitangent = cross(axis, tangent);
    const float phi = 2.0f * M_PI_F * uPhi;
    const float3 azimuth = cosf(phi) * tangent + sinf(phi) * bitangent;
    float3 sampledDirection = axis;
    for (int attempt = 0; attempt < 8; ++attempt)
    {
        const float cosTheta = 1.0f - 2.0f * q * halfSinSquared;
        const float sinTheta = 2.0f * halfSin * sqrtf(fmaxf(q * (1.0f - q * halfSinSquared), 0.0f));
        sampledDirection = normalizeFiniteVectorOrZero(azimuth * sinTheta + axis * cosTheta);
        if (distantLightContainsDirection(angle, sampledDirection, axis))
        {
            return sampledDirection;
        }
        // The final finite RNG cell can round just beyond the mathematical cap
        // after frame rotation and normalization. Map only that rejected cell
        // to a strict interior representative; never widen PDF support.
        q *= 0.5f;
    }
    // A direction equal to the axis is always inside every non-degenerate cap.
    return axis;
}

/// Cancellation-free cap membership for a represented direction.
///
/// For unit vectors, `|w-axis| = 2 sin(theta/2)`. Unlike a cosine threshold,
/// the chord remains nonzero when both cos(theta) and cos(halfAngle) round to
/// one. Both inputs are the canonical unit vectors produced by scene packing
/// and ray generation; renormalising either here would change that represented
/// event at tiny cap widths.
DEVICE_FUNC bool distantLightContainsDirection(float halfAngle, float3 direction, float3 axisDirection)
{
    if (distantLightIsDelta(halfAngle))
    {
        return false;
    }
    if (!(finiteVectorLength(direction) > 0.0f) || !(finiteVectorLength(axisDirection) > 0.0f))
    {
        return false;
    }
    const float maxChord = 2.0f * sinf(0.5f * distantLightHalfAngle(halfAngle));
    return finiteVectorLength(direction - axisDirection) <= maxChord;
}

/// Exact represented atom equality. There is deliberately no angular epsilon:
/// widening a sharp distant into a cone would invent continuous support.
DEVICE_FUNC bool distantLightDeltaDirectionMatches(float3 direction, float3 axisDirection)
{
    if (!(finiteVectorLength(direction) > 0.0f) || !(finiteVectorLength(axisDirection) > 0.0f))
    {
        return false;
    }
    return direction.x == axisDirection.x && direction.y == axisDirection.y && direction.z == axisDirection.z;
}

DEVICE_FUNC float infiniteLightDistance()
{
    return 1e16f;
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
/// A point, spot or projector only while its radius is at or below the shared
/// softness threshold, plus a sharp distant light. A positive-radius punctual
/// source is sampled and intersected as the same analytic sphere, so it has a
/// continuous BSDF competitor and must participate in ordinary solid-angle MIS.
/// `shapeParameter` is the punctual radius or the distant half angle.
DEVICE_FUNC bool lightIsDeltaForMis(int type, float shapeParameter)
{
    return (lightIsPunctual(type) && !punctualLightIsSoft(shapeParameter)) ||
           (type == LIGHT_TYPE_DISTANT && distantLightIsDelta(shapeParameter));
}

DEVICE_FUNC bool lightIsInfinite(int type)
{
    return type == LIGHT_TYPE_DISTANT || type == LIGHT_TYPE_DOME;
}

/// Whether the emitter itself can send the sampled direction toward a vertex.
/// Infinite lights have directional support but no emitting surface normal;
/// applying an area-light facing test would cut wide distant caps in half.
DEVICE_FUNC bool lightConnectionFacesVertex(int type, float cosAtLight, float punctualRadius = 0.0f)
{
    return (lightIsPunctual(type) && !punctualLightIsSoft(punctualRadius)) || lightIsInfinite(type) ||
           lightSampleFacesVertex(cosAtLight);
}

/// Conditional solid-angle density for evaluating an analytic infinite light
/// along a direction selected by the BSDF. This uses the same cancellation-free
/// geometry as the sampler.
DEVICE_FUNC float infiniteLightConditionalPdf(int type, float halfAngle, float3 direction, float3 axisDirection)
{
    if (type == LIGHT_TYPE_DOME)
    {
        return domeLightSolidAnglePdf();
    }
    if (type != LIGHT_TYPE_DISTANT || !distantLightContainsDirection(halfAngle, direction, axisDirection))
    {
        return 0.0f;
    }
    return coneLightSolidAnglePdf(halfAngle);
}

/// Everything the density of one light depends on, unpacked from whichever
/// light struct the backend holds.
struct LightPdfQuery
{
    int type;
    float distToLight; ///< shading vertex to the point on the light
    float cosAtLight; ///< -dot(L, light normal); <= 0 means the sample faces away
    float areaPdf; ///< density with respect to world area
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
    q.areaPdf = 0.0f;
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
                                       areaPdfToSolidAnglePdf(q.distToLight, q.cosAtLight, q.areaPdf);
    case LIGHT_TYPE_DISC:
    case LIGHT_TYPE_SPHERE:
        return areaPdfToSolidAnglePdf(q.distToLight, q.cosAtLight, q.areaPdf);
    case LIGHT_TYPE_DISTANT:
        return distantLightIsDelta(q.halfAngle) ? deltaLightPdf() : coneLightSolidAnglePdf(q.halfAngle);
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
