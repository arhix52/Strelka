#ifndef STRELKA_LIGHT_PDF_H
#define STRELKA_LIGHT_PDF_H

#include <light_types.h>
#include <strelka/material/material_math.h>

#define STRELKA_SOFT_LIGHT_RADIUS_MIN 1e-4f

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

DEVICE_FUNC OrthonormalLightFrame makeOrthonormalLightFrame(float3 axisX, float3 axisY, float3 emissionAxis)
{
    OrthonormalLightFrame frame{};
    frame.emissionAxis = normalizeFiniteVectorOrZero(emissionAxis);
    frame.x = orthonormalizeTangent(frame.emissionAxis, axisX);
    const float yAlongNormal = dot(axisY, frame.emissionAxis);
    const float yAlongX = dot(axisY, frame.x);
    const float3 yResidual = make_float3(fmaf(-yAlongX, frame.x.x, fmaf(-yAlongNormal, frame.emissionAxis.x, axisY.x)),
                                         fmaf(-yAlongX, frame.x.y, fmaf(-yAlongNormal, frame.emissionAxis.y, axisY.y)),
                                         fmaf(-yAlongX, frame.x.z, fmaf(-yAlongNormal, frame.emissionAxis.z, axisY.z)));
    frame.y = normalizeFiniteVectorOrZero(yResidual);
    frame.valid = dot(frame.emissionAxis, frame.emissionAxis) > 0.0f && dot(frame.x, frame.x) > 0.0f &&
                  dot(frame.y, frame.y) > 0.0f;
    return frame;
}

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

DEVICE_FUNC bool lightSampleFacesVertex(float cosAtLight)
{
    return cosAtLight > 0.0f && cosAtLight <= 3.402823466e38f;
}

struct PdfProductAccumulator
{
    float mantissa;
    int exponent;
};

DEVICE_FUNC bool multiplyPdfFactor(THREAD_REF PdfProductAccumulator& product, float factor)
{
    constexpr float maxFinite = 3.402823466e38f;
    if (!(factor > 0.0f) || !(factor <= maxFinite))
    {
        return false;
    }
#if defined(STRELKA_FAST_FINITE_GPU_MATH) && STRELKA_FAST_FINITE_GPU_MATH
    product.mantissa *= factor;
#else
    int exponent = 0;
    product.mantissa *= decomposeFloatExponent(factor, exponent);
    product.exponent += exponent;
#endif
    return true;
}

DEVICE_FUNC bool dividePdfFactor(THREAD_REF PdfProductAccumulator& product, float factor)
{
    constexpr float maxFinite = 3.402823466e38f;
    if (!(factor > 0.0f) || !(factor <= maxFinite))
    {
        return false;
    }
#if defined(STRELKA_FAST_FINITE_GPU_MATH) && STRELKA_FAST_FINITE_GPU_MATH
    product.mantissa /= factor;
#else
    int exponent = 0;
    product.mantissa /= decomposeFloatExponent(factor, exponent);
    product.exponent -= exponent;
#endif
    return true;
}

DEVICE_FUNC bool multiplySquaredPdfFactor(THREAD_REF PdfProductAccumulator& product, float factor)
{
    constexpr float maxFinite = 3.402823466e38f;
    if (!(factor > 0.0f) || !(factor <= maxFinite))
    {
        return false;
    }
#if defined(STRELKA_FAST_FINITE_GPU_MATH) && STRELKA_FAST_FINITE_GPU_MATH
    product.mantissa *= factor * factor;
#else
    int exponent = 0;
    const float mantissa = decomposeFloatExponent(factor, exponent);
    product.mantissa *= mantissa * mantissa;
    product.exponent += 2 * exponent;
#endif
    return true;
}

DEVICE_FUNC float finishPdfProduct(const THREAD_REF PdfProductAccumulator& product)
{
    constexpr float maxFinite = 3.402823466e38f;
#if defined(STRELKA_FAST_FINITE_GPU_MATH) && STRELKA_FAST_FINITE_GPU_MATH
    const float result = product.mantissa;
#else
    const float result = scaleFloatExponent(product.mantissa, product.exponent);
#endif
    if (!(result > 0.0f))
    {
        return 0.0f;
    }
    return result <= maxFinite ? result : maxFinite;
}

DEVICE_FUNC float scalePdfBySelection(
    float conditionalPdf, float selection0, float selection1, float selection2, float selection3)
{
    PdfProductAccumulator product{ 1.0f, 0 };
    if (!multiplyPdfFactor(product, conditionalPdf) || !multiplyPdfFactor(product, selection0) ||
        !multiplyPdfFactor(product, selection1) || !multiplyPdfFactor(product, selection2) ||
        !multiplyPdfFactor(product, selection3))
    {
        return 0.0f;
    }
    return finishPdfProduct(product);
}

DEVICE_FUNC float reciprocalPdfWithSelection(
    float denominator, float selection0, float selection1, float selection2, float selection3)
{
    PdfProductAccumulator product{ 1.0f, 0 };
    if (!dividePdfFactor(product, denominator) || !multiplyPdfFactor(product, selection0) ||
        !multiplyPdfFactor(product, selection1) || !multiplyPdfFactor(product, selection2) ||
        !multiplyPdfFactor(product, selection3))
    {
        return 0.0f;
    }
    return finishPdfProduct(product);
}

DEVICE_FUNC float areaLightSolidAnglePdf(float distToLight, float cosAtLight, float area)
{
    if (!lightSampleFacesVertex(cosAtLight) || !(distToLight > 0.0f) || !(area > 0.0f))
    {
        return 0.0f;
    }
    cosAtLight = fminf(cosAtLight, 1.0f);
    PdfProductAccumulator product{ 1.0f, 0 };
    if (!multiplySquaredPdfFactor(product, distToLight) || !dividePdfFactor(product, cosAtLight) ||
        !dividePdfFactor(product, area))
    {
        return 0.0f;
    }
    return finishPdfProduct(product);
}

DEVICE_FUNC float areaPdfToSolidAngleMarginalPdf(float distToLight,
                                                 float cosAtLight,
                                                 float areaPdf,
                                                 float selection0,
                                                 float selection1,
                                                 float selection2,
                                                 float selection3)
{
    if (!lightSampleFacesVertex(cosAtLight) || !(distToLight > 0.0f) || !(areaPdf > 0.0f))
    {
        return 0.0f;
    }
    cosAtLight = fminf(cosAtLight, 1.0f);
    PdfProductAccumulator product{ 1.0f, 0 };
    if (!multiplyPdfFactor(product, areaPdf) || !multiplySquaredPdfFactor(product, distToLight) ||
        !dividePdfFactor(product, cosAtLight) || !multiplyPdfFactor(product, selection0) ||
        !multiplyPdfFactor(product, selection1) || !multiplyPdfFactor(product, selection2) ||
        !multiplyPdfFactor(product, selection3))
    {
        return 0.0f;
    }
    return finishPdfProduct(product);
}

/// Area density converted to solid angle without first materialising its
/// reciprocal area/Jacobian. The multiplication order keeps p_A*d^2 finite in
/// cases where d^2 alone overflows.
DEVICE_FUNC float areaPdfToSolidAnglePdf(float distToLight, float cosAtLight, float areaPdf)
{
    return areaPdfToSolidAngleMarginalPdf(distToLight, cosAtLight, areaPdf, 1.0f, 1.0f, 1.0f, 1.0f);
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

DEVICE_FUNC float sphereLightSolidAnglePdf(float distToLight, float cosAtLight, float radius)
{
    return areaPdfToSolidAnglePdf(distToLight, cosAtLight, sphereLightAreaPdf(radius));
}

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

DEVICE_FUNC float lightRetryUniform(uint32_t word)
{
    return (float)(word >> 8u) * 0x1p-24f;
}

DEVICE_FUNC float3 sampleDistantLightDirection(
    float uPhi, float uSolidAngle, uint32_t retryPhi, uint32_t retrySolidAngle, float halfAngle, float3 direction)
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
    uint32_t phiState = retryPhi;
    uint32_t solidAngleState = retrySolidAngle;
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
    float3 sampledDirection = axis;
    for (int attempt = 0; attempt < 9; ++attempt)
    {
        const float phi = 2.0f * M_PI_F * uPhi;
        const float3 azimuth = cosf(phi) * tangent + sinf(phi) * bitangent;
        const float cosTheta = 1.0f - 2.0f * q * halfSinSquared;
        const float sinTheta = 2.0f * halfSin * sqrtf(fmaxf(q * (1.0f - q * halfSinSquared), 0.0f));
        sampledDirection = normalizeFiniteVectorOrZero(azimuth * sinTheta + axis * cosTheta);
        if (distantLightContainsDirection(angle, sampledDirection, axis))
        {
            return sampledDirection;
        }
        phiState = phiState * 1664525u + 1013904223u;
        solidAngleState = solidAngleState * 1664525u + 1013904223u;
        uPhi = lightRetryUniform(phiState);
        q = lightRetryUniform(solidAngleState);
    }
    // A direction equal to the axis is always inside every non-degenerate cap.
    return axis;
}

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

DEVICE_FUNC bool distantLightDeltaPathMatches(uint32_t depth, bool specularBounce, float3 direction, float3 axisDirection)
{
    return (depth == 0u || specularBounce) && distantLightDeltaDirectionMatches(direction, axisDirection);
}

DEVICE_FUNC float infiniteLightDistance()
{
    return 1e16f;
}

DEVICE_FUNC float deltaLightPdf()
{
    return 1.0f;
}

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

DEVICE_FUNC bool lightIsPunctual(int type)
{
    return type == LIGHT_TYPE_POINT || type == LIGHT_TYPE_SPOT || type == LIGHT_TYPE_PROJECTOR;
}

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

DEVICE_FUNC float lightSolidAnglePdf(const THREAD_REF LightPdfQuery& q)
{
    switch (q.type)
    {
    case LIGHT_TYPE_RECT:
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
        return punctualLightIsSoft(q.radius) ? sphereLightSolidAnglePdf(q.distToLight, q.cosAtLight, q.radius) :
                                               deltaLightPdf();
    default:
        break;
    }
    return 0.0f;
}

/// Complete analytic-light density, including every discrete selection mass.
/// Area conditionals are evaluated jointly with those masses so a finite
/// marginal cannot be corrupted by an overflowing conditional intermediate.
DEVICE_FUNC float marginalLightSolidAnglePdf(const THREAD_REF LightPdfQuery& q,
                                             float localSelectionPdf,
                                             float analyticSelectionPdf,
                                             float lightSelectionPdf)
{
    const bool areaConditional =
        (q.type == LIGHT_TYPE_RECT && !(q.solidAngle > 0.0f)) || q.type == LIGHT_TYPE_DISC ||
        q.type == LIGHT_TYPE_SPHERE ||
        ((q.type == LIGHT_TYPE_POINT || q.type == LIGHT_TYPE_SPOT || q.type == LIGHT_TYPE_PROJECTOR) &&
         punctualLightIsSoft(q.radius));
    if (areaConditional)
    {
        return areaPdfToSolidAngleMarginalPdf(
            q.distToLight, q.cosAtLight, q.areaPdf, localSelectionPdf, analyticSelectionPdf, lightSelectionPdf, 1.0f);
    }
    if (q.type == LIGHT_TYPE_RECT && q.solidAngle > 0.0f)
    {
        return reciprocalPdfWithSelection(q.solidAngle, localSelectionPdf, analyticSelectionPdf, lightSelectionPdf, 1.0f);
    }
    return scalePdfBySelection(lightSolidAnglePdf(q), localSelectionPdf, analyticSelectionPdf, lightSelectionPdf, 1.0f);
}

DEVICE_FUNC float3 uniformSphereDirection(float u1, float u2)
{
    const float cosTheta = 1.0f - 2.0f * u1; // uniform on [-1, 1]
    const float sinTheta = sqrtf(fmaxf(1.0f - cosTheta * cosTheta, 0.0f));
    const float phi = 2.0f * M_PI_F * u2;
    return make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

#endif // STRELKA_LIGHT_PDF_H
