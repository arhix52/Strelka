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
    // Density with respect to world area. Keeping p_A rather than 1/p_A is
    // essential when a very large finite surface has a representable density
    // although its area Jacobian itself exceeds float range.
    float areaPdf = 0.0f;
};

struct AnalyticLightIntersection
{
    float distance = 0.0f;
    float3 normal{};
    bool hit = false;
};

DEVICE_FUNC float analyticDiscArea(float3 axisX, float3 axisY)
{
    const float twiceParallelogramArea = finiteVectorLength(cross(axisX, axisY));
    return twiceParallelogramArea > 0.0f && twiceParallelogramArea <= 3.402823466e38f / M_PI_F ?
               M_PI_F * twiceParallelogramArea :
               0.0f;
}

DEVICE_FUNC float analyticDiscAreaPdf(float3 axisX, float3 axisY)
{
    return finiteCrossReciprocal(axisX, axisY, 1.0f / M_PI_F);
}

DEVICE_FUNC float3 affineSphereCofactor(float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal)
{
    return objectNormal.x * cross(axisY, axisZ) + objectNormal.y * cross(axisZ, axisX) +
           objectNormal.z * cross(axisX, axisY);
}

DEVICE_FUNC float analyticAffineOrientation(float3 axisX, float3 axisY, float3 axisZ)
{
    const float3 x = normalizeFiniteVectorOrZero(axisX);
    const float3 y = normalizeFiniteVectorOrZero(axisY);
    const float3 z = normalizeFiniteVectorOrZero(axisZ);
    const float determinant = dot(x, cross(y, z));
    return determinant > 0.0f ? 1.0f : (determinant < 0.0f ? -1.0f : 0.0f);
}

DEVICE_FUNC void objectTangentPlane(float3 objectNormal, THREAD_REF float3& tangent, THREAD_REF float3& bitangent)
{
    const float3 n = normalizeFiniteVectorOrZero(objectNormal);
    if (!(dot(n, n) > 0.0f))
    {
        tangent = make_float3(0.0f);
        bitangent = make_float3(0.0f);
        return;
    }
    const float3 reference = fabsf(n.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f, 1.0f, 0.0f);
    tangent = normalizeFiniteVectorOrZero(cross(reference, n));
    bitangent = cross(n, tangent);
}

DEVICE_FUNC float affineAxisScale(float3 axisX, float3 axisY, float3 axisZ)
{
    float scale = fmaxf(fabsf(axisX.x), fmaxf(fabsf(axisX.y), fabsf(axisX.z)));
    scale = fmaxf(scale, fmaxf(fabsf(axisY.x), fmaxf(fabsf(axisY.y), fabsf(axisY.z))));
    return fmaxf(scale, fmaxf(fabsf(axisZ.x), fmaxf(fabsf(axisZ.y), fabsf(axisZ.z))));
}

DEVICE_FUNC float affineSphereAreaPdfAndNormal(
    float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal, THREAD_REF float3& normal)
{
    normal = make_float3(0.0f);
    const float axisScale = affineAxisScale(axisX, axisY, axisZ);
    if (!(axisScale > 0.0f) || !(axisScale <= 3.402823466e38f))
    {
        return 0.0f;
    }
    float3 objectTangent;
    float3 objectBitangent;
    objectTangentPlane(objectNormal, objectTangent, objectBitangent);
    const float3 worldTangent = objectTangent.x * axisX + objectTangent.y * axisY + objectTangent.z * axisZ;
    const float3 worldBitangent = objectBitangent.x * axisX + objectBitangent.y * axisY + objectBitangent.z * axisZ;
    const float orientation = analyticAffineOrientation(axisX, axisY, axisZ);
    normal = orientation * finiteCrossDirection(worldTangent, worldBitangent);
    const float areaPdf = finiteCrossReciprocal(worldTangent, worldBitangent, 1.0f / (4.0f * M_PI_F));
    if (orientation == 0.0f || !(dot(normal, normal) > 0.0f) || !(areaPdf > 0.0f))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    return areaPdf;
}

DEVICE_FUNC float3 transformedNormalCofactor(float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal)
{
    float3 objectTangent;
    float3 objectBitangent;
    objectTangentPlane(objectNormal, objectTangent, objectBitangent);
    const float3 worldTangent = objectTangent.x * axisX + objectTangent.y * axisY + objectTangent.z * axisZ;
    const float3 worldBitangent = objectBitangent.x * axisX + objectBitangent.y * axisY + objectBitangent.z * axisZ;
    return finiteCrossDirection(worldTangent, worldBitangent);
}

// Unit inverse-transpose normal for an affine map whose columns are the three
// axes. The determinant sign matters for mirrored transforms; its magnitude
// cancels during normalization.
DEVICE_FUNC float3 transformAffineNormal(float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal)
{
    const float orientation = analyticAffineOrientation(axisX, axisY, axisZ);
    const float3 cofactorNormal = transformedNormalCofactor(axisX, axisY, axisZ, objectNormal);
    const float3 normal = normalizeFiniteVectorOrZero(cofactorNormal);
    if (orientation == 0.0f || !(dot(normal, normal) > 0.0f))
    {
        return make_float3(0.0f);
    }
    return orientation * normal;
}

DEVICE_FUNC bool analyticAffineTransformIsNonsingular(float3 axisX, float3 axisY, float3 axisZ)
{
    if (analyticAffineOrientation(axisX, axisY, axisZ) == 0.0f)
    {
        return false;
    }
    const float axisScale = affineAxisScale(axisX, axisY, axisZ);
    if (!(axisScale > 0.0f))
    {
        return false;
    }
    const float3 scaledX = axisX / axisScale;
    const float3 scaledY = axisY / axisScale;
    const float3 scaledZ = axisZ / axisScale;
    const float scaledDeterminant = dot(scaledX, cross(scaledY, scaledZ));
    if (!(fabsf(scaledDeterminant) > 0.0f))
    {
        return false;
    }

    // For a unit object normal n, J_A(n)=|cofactor(A)n|. Its maximum is at
    // most the sum of the three column norms. Express the reciprocal bound in
    // terms of representable basis densities, so no overflowing Jacobian is
    // formed and every point on an accepted ellipsoid has positive p_A.
    float3 ignoredNormal;
    const float px = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, make_float3(1.0f, 0.0f, 0.0f), ignoredNormal);
    const float py = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, make_float3(0.0f, 1.0f, 0.0f), ignoredNormal);
    const float pz = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, make_float3(0.0f, 0.0f, 1.0f), ignoredNormal);
    const float minimumPdf = fminf(px, fminf(py, pz));
    if (!(minimumPdf > 0.0f))
    {
        return false;
    }
    const float conservativePdf = minimumPdf / (minimumPdf / px + minimumPdf / py + minimumPdf / pz);
    return conservativePdf > 0.0f;
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
    const float areaPdf = analyticDiscAreaPdf(axisX, axisY);
    if (!(areaPdf > 0.0f) || !(dot(emissionNormal, emissionNormal) > 0.0f))
    {
        return sample;
    }
    const float radius = sqrtf(fmaxf(u1, 0.0f));
    const float phi = 2.0f * M_PI_F * u2;
    sample.point = center + radius * (cosf(phi) * axisX + sinf(phi) * axisY);
    sample.normal = emissionNormal;
    sample.areaPdf = areaPdf;
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
    sample.areaPdf = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, objectNormal, sample.normal);
    return sample;
}

DEVICE_FUNC float analyticEllipsoidAreaPdf(
    float3 center, float3 axisX, float3 axisY, float3 axisZ, float3 point, THREAD_REF float3& normal)
{
    if (!analyticAffineTransformIsNonsingular(axisX, axisY, axisZ))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    const float axisScale = affineAxisScale(axisX, axisY, axisZ);
    if (!(axisScale > 0.0f) || !(axisScale <= 3.402823466e38f))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    const float3 scaledX = axisX / axisScale;
    const float3 scaledY = axisY / axisScale;
    const float3 scaledZ = axisZ / axisScale;
    const float orientation = analyticAffineOrientation(scaledX, scaledY, scaledZ);
    float3 objectNormal = affineSphereCoordinateNumerators(scaledX, scaledY, scaledZ, point - center);
    objectNormal = orientation * normalizeFiniteVectorOrZero(objectNormal);
    if (!(dot(objectNormal, objectNormal) > 0.0f))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    return affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, objectNormal, normal);
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
    float jacobianMean = 0.0f;
    for (unsigned int i = 0u; i < sampleCount; ++i)
    {
        const float z = 1.0f - 2.0f * (float(i) + 0.5f) / float(sampleCount);
        const float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
        const float phi = goldenAngle * float(i);
        const float3 objectNormal = make_float3(radial * cosf(phi), radial * sinf(phi), z);
        float3 normal;
        const float areaPdf = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, objectNormal, normal);
        if (!(areaPdf > 0.0f))
        {
            return 0.0f;
        }
        const float jacobian = 1.0f / (4.0f * M_PI_F * areaPdf);
        if (!(jacobian > 0.0f) || !(jacobian <= 3.402823466e38f))
        {
            return 0.0f;
        }
        jacobianMean += jacobian / float(sampleCount);
    }
    return 4.0f * M_PI_F * jacobianMean;
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

    const float3 planeNormal = normalizeFiniteVectorOrZero(cross(axisX, axisY));
    const float denominator = dot(rayDirection, planeNormal);
    if (!(dot(planeNormal, planeNormal) > 0.0f) || !(dot(emissionNormal, emissionNormal) > 0.0f) ||
        !(fabsf(denominator) > 0.0f))
    {
        return result;
    }
    const float distance = dot(center - rayOrigin, planeNormal) / denominator;
    if (!(distance >= minDistance && distance < maxDistance))
    {
        return result;
    }

    float axisScale = fmaxf(fabsf(axisX.x), fmaxf(fabsf(axisX.y), fabsf(axisX.z)));
    axisScale = fmaxf(axisScale, fmaxf(fabsf(axisY.x), fmaxf(fabsf(axisY.y), fabsf(axisY.z))));
    if (!(axisScale > 0.0f) || !(axisScale <= 3.402823466e38f))
    {
        return result;
    }
    const float3 scaledX = axisX / axisScale;
    const float3 scaledY = axisY / axisScale;
    const float3 offset = (rayOrigin + distance * rayDirection - center) / axisScale;
    const float3 dualNormal = cross(scaledX, scaledY);
    const float determinant = dot(dualNormal, planeNormal);
    if (!(determinant > 0.0f))
    {
        return result;
    }
    const float x = dot(cross(offset, scaledY), planeNormal) / determinant;
    const float y = dot(cross(scaledX, offset), planeNormal) / determinant;
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
    float axisScale = fmaxf(fabsf(axisX.x), fmaxf(fabsf(axisX.y), fabsf(axisX.z)));
    axisScale = fmaxf(axisScale, fmaxf(fabsf(axisY.x), fmaxf(fabsf(axisY.y), fabsf(axisY.z))));
    axisScale = fmaxf(axisScale, fmaxf(fabsf(axisZ.x), fmaxf(fabsf(axisZ.y), fabsf(axisZ.z))));
    if (!(axisScale > 0.0f) || !(axisScale <= 3.402823466e38f))
    {
        return result;
    }
    const float3 scaledX = axisX / axisScale;
    const float3 scaledY = axisY / axisScale;
    const float3 scaledZ = axisZ / axisScale;
    const float3 cofactorX = cross(scaledY, scaledZ);
    const float3 cofactorY = cross(scaledZ, scaledX);
    const float3 cofactorZ = cross(scaledX, scaledY);
    const float determinant = axisScale * dot(scaledX, cofactorX);
    float3 objectOrigin = make_float3(
        dot(rayOrigin - center, cofactorX), dot(rayOrigin - center, cofactorY), dot(rayOrigin - center, cofactorZ));
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
    const float areaPdf = analyticEllipsoidAreaPdf(center, axisX, axisY, axisZ, point, normal);
    if (!(areaPdf > 0.0f))
    {
        return result;
    }
    result.distance = distance;
    result.normal = normal;
    result.hit = true;
    return result;
}

#endif // STRELKA_ANALYTIC_LIGHT_H
