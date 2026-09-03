#ifndef STRELKA_ANALYTIC_LIGHT_H
#define STRELKA_ANALYTIC_LIGHT_H

// Shared analytic disc/ellipsoid measure and intersection math. The axes are
// the columns of the authored object-to-world linear transform, including the
// light radius. Sampling and intersection therefore describe the same smooth
// geometry even under shear, non-uniform scale, or a mirrored transform.

#include <strelka/material/material_math.h>
#include <light_types.h>

enum : unsigned int
{
    STRELKA_ANALYTIC_LIGHT_CAMERA_BIT = 1u,
    STRELKA_ANALYTIC_LIGHT_SECONDARY_BIT = 2u
};

DEVICE_FUNC bool lightUsesAnalyticAreaIntersection(int lightType)
{
    return lightType == LIGHT_TYPE_DISC || lightType == LIGHT_TYPE_SPHERE;
}

struct AnalyticLightSample
{
    float3 point{};
    float3 normal{};
    // Reciprocal of the world-area density. For a disc this is its total area;
    // for an ellipsoid it is 4*pi times the local affine area Jacobian.
    float areaPdfDenominator = 0.0f;
};

struct AnalyticLightIntersection
{
    float distance = 0.0f;
    float3 normal{};
    bool hit = false;
};

DEVICE_FUNC float analyticDiscArea(float3 axisX, float3 axisY)
{
    return M_PI_F * length(cross(axisX, axisY));
}

DEVICE_FUNC float3 affineSphereCofactor(float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal)
{
    return objectNormal.x * cross(axisY, axisZ) + objectNormal.y * cross(axisZ, axisX) +
           objectNormal.z * cross(axisX, axisY);
}

// Unit inverse-transpose normal for an affine map whose columns are the three
// axes. The determinant sign matters for mirrored transforms; its magnitude
// cancels during normalization.
DEVICE_FUNC float3 transformAffineNormal(float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal)
{
    const float determinant = dot(axisX, cross(axisY, axisZ));
    const float3 cofactorNormal = affineSphereCofactor(axisX, axisY, axisZ, objectNormal);
    const float lengthSquared = dot(cofactorNormal, cofactorNormal);
    if (!(fabsf(determinant) > 0.0f) || !(lengthSquared > 0.0f) || !(lengthSquared <= 3.402823466e38f))
    {
        return make_float3(0.0f);
    }
    return copysignf(1.0f, determinant) * cofactorNormal / sqrtf(lengthSquared);
}

DEVICE_FUNC bool analyticAffineTransformIsNonsingular(float3 axisX, float3 axisY, float3 axisZ)
{
    const float determinantMagnitude = fabsf(dot(axisX, cross(axisY, axisZ)));
    return determinantMagnitude > 0.0f && determinantMagnitude <= 3.402823466e38f;
}

DEVICE_FUNC float3 affineSphereCoordinates(float3 axisX, float3 axisY, float3 axisZ, float3 worldOffset)
{
    const float determinant = dot(axisX, cross(axisY, axisZ));
    if (!(fabsf(determinant) > 0.0f))
    {
        return make_float3(0.0f);
    }
    return make_float3(dot(worldOffset, cross(axisY, axisZ)), dot(worldOffset, cross(axisZ, axisX)),
                       dot(worldOffset, cross(axisX, axisY))) /
           determinant;
}

DEVICE_FUNC float3 affineSphereCoordinateNumerators(float3 axisX, float3 axisY, float3 axisZ, float3 worldOffset)
{
    return make_float3(dot(worldOffset, cross(axisY, axisZ)), dot(worldOffset, cross(axisZ, axisX)),
                       dot(worldOffset, cross(axisX, axisY)));
}

DEVICE_FUNC AnalyticLightSample
sampleAnalyticDisc(float3 center, float3 axisX, float3 axisY, float3 emissionNormal, float u1, float u2)
{
    AnalyticLightSample sample;
    const float area = analyticDiscArea(axisX, axisY);
    if (!(area > 0.0f) || !(dot(emissionNormal, emissionNormal) > 0.0f))
    {
        return sample;
    }
    const float radius = sqrtf(fmaxf(u1, 0.0f));
    const float phi = 2.0f * M_PI_F * u2;
    sample.point = center + radius * (cosf(phi) * axisX + sinf(phi) * axisY);
    sample.normal = emissionNormal;
    sample.areaPdfDenominator = area;
    return sample;
}

DEVICE_FUNC AnalyticLightSample
sampleAnalyticEllipsoid(float3 center, float3 axisX, float3 axisY, float3 axisZ, float u1, float u2)
{
    AnalyticLightSample sample;
    if (!analyticAffineTransformIsNonsingular(axisX, axisY, axisZ))
    {
        return sample;
    }
    const float z = 1.0f - 2.0f * u1;
    const float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    const float phi = 2.0f * M_PI_F * u2;
    float3 objectNormal = make_float3(radial * cosf(phi), radial * sinf(phi), z);
    const float objectLengthSquared = dot(objectNormal, objectNormal);
    if (objectLengthSquared > 0.0f)
    {
        objectNormal /= sqrtf(objectLengthSquared);
    }
    sample.point = center + objectNormal.x * axisX + objectNormal.y * axisY + objectNormal.z * axisZ;
    const float3 cofactorNormal = affineSphereCofactor(axisX, axisY, axisZ, objectNormal);
    const float jacobian = length(cofactorNormal);
    const float orientation = dot(axisX, cross(axisY, axisZ)) < 0.0f ? -1.0f : 1.0f;
    sample.normal = (jacobian > 0.0f) ? (orientation * cofactorNormal / jacobian) : make_float3(0.0f);
    sample.areaPdfDenominator = 4.0f * M_PI_F * jacobian;
    return sample;
}

DEVICE_FUNC float analyticEllipsoidAreaPdfDenominator(
    float3 center, float3 axisX, float3 axisY, float3 axisZ, float3 point, THREAD_REF float3& normal)
{
    const float determinant = dot(axisX, cross(axisY, axisZ));
    float3 objectNormal = affineSphereCoordinateNumerators(axisX, axisY, axisZ, point - center);
    const float objectLengthSquared = dot(objectNormal, objectNormal);
    if (!(objectLengthSquared > 0.0f))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    objectNormal *= copysignf(1.0f, determinant) / sqrtf(objectLengthSquared);
    const float3 cofactorNormal = affineSphereCofactor(axisX, axisY, axisZ, objectNormal);
    const float jacobian = length(cofactorNormal);
    const float orientation = dot(axisX, cross(axisY, axisZ)) < 0.0f ? -1.0f : 1.0f;
    normal = (jacobian > 0.0f) ? (orientation * cofactorNormal / jacobian) : make_float3(0.0f);
    return 4.0f * M_PI_F * jacobian;
}

// Deterministic equal-area quadrature of the smooth ellipsoid's surface area.
// Sampling/PDF evaluation does not depend on this approximation; it is used
// only to power-weight the outer light-selection PMF.
DEVICE_FUNC float analyticEllipsoidSurfaceArea(float3 axisX, float3 axisY, float3 axisZ)
{
    if (!analyticAffineTransformIsNonsingular(axisX, axisY, axisZ))
    {
        return 0.0f;
    }
    const unsigned int sampleCount = 256u;
    const float goldenAngle = 2.39996322972865332f;
    float jacobianSum = 0.0f;
    for (unsigned int i = 0u; i < sampleCount; ++i)
    {
        const float z = 1.0f - 2.0f * (float(i) + 0.5f) / float(sampleCount);
        const float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
        const float phi = goldenAngle * float(i);
        const float3 objectNormal = make_float3(radial * cosf(phi), radial * sinf(phi), z);
        jacobianSum += length(affineSphereCofactor(axisX, axisY, axisZ, objectNormal));
    }
    return 4.0f * M_PI_F * jacobianSum / float(sampleCount);
}

DEVICE_FUNC AnalyticLightIntersection intersectAnalyticDisc(float3 rayOrigin,
                                                            float3 rayDirection,
                                                            float minDistance,
                                                            float maxDistance,
                                                            float3 center,
                                                            float3 axisX,
                                                            float3 axisY,
                                                            float3 emissionNormal)
{
    AnalyticLightIntersection result;
    result.distance = maxDistance;
    result.normal = make_float3(0.0f);
    result.hit = false;

    const float3 planeNormal = cross(axisX, axisY);
    const float normalLengthSquared = dot(planeNormal, planeNormal);
    const float denominator = dot(rayDirection, planeNormal);
    if (!(normalLengthSquared > 0.0f) || !(dot(emissionNormal, emissionNormal) > 0.0f) ||
        !(fabsf(denominator) > 0.0f))
    {
        return result;
    }
    const float distance = dot(center - rayOrigin, planeNormal) / denominator;
    if (!(distance >= minDistance && distance < maxDistance))
    {
        return result;
    }

    const float3 offset = rayOrigin + distance * rayDirection - center;
    const float x = dot(offset, cross(axisY, planeNormal)) / normalLengthSquared;
    const float y = dot(offset, cross(planeNormal, axisX)) / normalLengthSquared;
    if (!(x * x + y * y <= 1.0f))
    {
        return result;
    }

    result.distance = distance;
    result.normal = emissionNormal;
    result.hit = true;
    return result;
}

DEVICE_FUNC AnalyticLightIntersection intersectAnalyticEllipsoid(float3 rayOrigin,
                                                                 float3 rayDirection,
                                                                 float minDistance,
                                                                 float maxDistance,
                                                                 float3 center,
                                                                 float3 axisX,
                                                                 float3 axisY,
                                                                 float3 axisZ)
{
    AnalyticLightIntersection result;
    result.distance = maxDistance;
    result.normal = make_float3(0.0f);
    result.hit = false;

    if (!analyticAffineTransformIsNonsingular(axisX, axisY, axisZ))
    {
        return result;
    }
    const float3 cofactorX = cross(axisY, axisZ);
    const float3 cofactorY = cross(axisZ, axisX);
    const float3 cofactorZ = cross(axisX, axisY);
    const float determinant = dot(axisX, cofactorX);
    float3 objectOrigin = make_float3(dot(rayOrigin - center, cofactorX), dot(rayOrigin - center, cofactorY),
                                      dot(rayOrigin - center, cofactorZ));
    float3 objectDirection =
        make_float3(dot(rayDirection, cofactorX), dot(rayDirection, cofactorY), dot(rayDirection, cofactorZ));
    // Solve |adj(A) * (O + tD)|^2 = det(A)^2 directly. Dividing by a tiny
    // determinant first can overflow even though this homogeneous equation is
    // well represented. A common scale keeps all quadratic coefficients finite
    // without changing either root.
    float coefficientScale = fabsf(determinant);
    coefficientScale = fmaxf(coefficientScale, fmaxf(fabsf(objectOrigin.x), fabsf(objectOrigin.y)));
    coefficientScale = fmaxf(coefficientScale, fmaxf(fabsf(objectOrigin.z), fabsf(objectDirection.x)));
    coefficientScale = fmaxf(coefficientScale, fmaxf(fabsf(objectDirection.y), fabsf(objectDirection.z)));
    if (!(coefficientScale > 0.0f))
    {
        return result;
    }
    objectOrigin /= coefficientScale;
    objectDirection /= coefficientScale;
    const float scaledDeterminant = determinant / coefficientScale;
    const float a = dot(objectDirection, objectDirection);
    const float b = dot(objectOrigin, objectDirection);
    const float c = dot(objectOrigin, objectOrigin) - scaledDeterminant * scaledDeterminant;
    const float discriminant = b * b - a * c;
    if (!(a > 0.0f) || !(discriminant >= 0.0f))
    {
        return result;
    }

    const float root = sqrtf(fmaxf(discriminant, 0.0f));
    // Stable quadratic roots for a*t^2 + 2*b*t + c. The direct near root
    // subtracts almost equal numbers for grazing rays, precisely where a small
    // t error becomes a large normal/PDF disagreement.
    const float q = -b - copysignf(root, b);
    float t0 = q / a;
    float t1 = (fabsf(q) > 0.0f) ? (c / q) : ((-b + root) / a);
    if (t1 < t0)
    {
        const float swap = t0;
        t0 = t1;
        t1 = swap;
    }
    float distance = t0;
    if (!(distance >= minDistance && distance < maxDistance))
    {
        distance = t1;
    }
    if (!(distance >= minDistance && distance < maxDistance))
    {
        return result;
    }

    const float3 point = rayOrigin + distance * rayDirection;
    float3 normal;
    const float areaPdfDenominator = analyticEllipsoidAreaPdfDenominator(center, axisX, axisY, axisZ, point, normal);
    if (!(areaPdfDenominator > 0.0f))
    {
        return result;
    }
    result.distance = distance;
    result.normal = normal;
    result.hit = true;
    return result;
}

#endif // STRELKA_ANALYTIC_LIGHT_H
