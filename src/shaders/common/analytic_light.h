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
    const float twiceParallelogramArea = finiteVectorLength(cross(axisX, axisY));
    return twiceParallelogramArea > 0.0f && twiceParallelogramArea <= 3.402823466e38f / M_PI_F ?
               M_PI_F * twiceParallelogramArea :
               0.0f;
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

DEVICE_FUNC float3 scaledAffineSphereCofactor(float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal)
{
    const float scaleX = fmaxf(fabsf(axisX.x), fmaxf(fabsf(axisX.y), fabsf(axisX.z)));
    const float scaleY = fmaxf(fabsf(axisY.x), fmaxf(fabsf(axisY.y), fabsf(axisY.z)));
    const float scaleZ = fmaxf(fabsf(axisZ.x), fmaxf(fabsf(axisZ.y), fabsf(axisZ.z)));
    const float largest = fmaxf(scaleX, fmaxf(scaleY, scaleZ));
    const float secondLargest = fmaxf(fminf(scaleX, scaleY), fminf(fmaxf(scaleX, scaleY), scaleZ));
    if (!(scaleX > 0.0f) || !(scaleY > 0.0f) || !(scaleZ > 0.0f) || !(secondLargest > 0.0f) ||
        !(largest <= 3.402823466e38f))
    {
        return make_float3(0.0f);
    }
    const float3 x = axisX / scaleX;
    const float3 y = axisY / scaleY;
    const float3 z = axisZ / scaleZ;
    const float yzScale = (fmaxf(scaleY, scaleZ) / largest) * (fminf(scaleY, scaleZ) / secondLargest);
    const float zxScale = (fmaxf(scaleZ, scaleX) / largest) * (fminf(scaleZ, scaleX) / secondLargest);
    const float xyScale = (fmaxf(scaleX, scaleY) / largest) * (fminf(scaleX, scaleY) / secondLargest);
    return objectNormal.x * yzScale * cross(y, z) + objectNormal.y * zxScale * cross(z, x) +
           objectNormal.z * xyScale * cross(x, y);
}

// Unit inverse-transpose normal for an affine map whose columns are the three
// axes. The determinant sign matters for mirrored transforms; its magnitude
// cancels during normalization.
DEVICE_FUNC float3 transformAffineNormal(float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal)
{
    const float orientation = analyticAffineOrientation(axisX, axisY, axisZ);
    const float3 cofactorNormal = scaledAffineSphereCofactor(axisX, axisY, axisZ, objectNormal);
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
    const float cofactorXLength = finiteVectorLength(cross(axisY, axisZ));
    const float cofactorYLength = finiteVectorLength(cross(axisZ, axisX));
    const float cofactorZLength = finiteVectorLength(cross(axisX, axisY));
    constexpr float maxJacobian = 3.402823466e38f / (4.0f * M_PI_F);
    return cofactorXLength > 0.0f && cofactorYLength > 0.0f && cofactorZLength > 0.0f &&
           cofactorXLength <= maxJacobian - cofactorYLength &&
           cofactorXLength + cofactorYLength <= maxJacobian - cofactorZLength;
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
    const float jacobian = finiteVectorLength(cofactorNormal);
    const float orientation = analyticAffineOrientation(axisX, axisY, axisZ);
    sample.normal = jacobian > 0.0f ? orientation * normalizeFiniteVectorOrZero(cofactorNormal) : make_float3(0.0f);
    sample.areaPdfDenominator = 4.0f * M_PI_F * jacobian;
    return sample;
}

DEVICE_FUNC float analyticEllipsoidAreaPdfDenominator(
    float3 center, float3 axisX, float3 axisY, float3 axisZ, float3 point, THREAD_REF float3& normal)
{
    if (!analyticAffineTransformIsNonsingular(axisX, axisY, axisZ))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    float axisScale = fmaxf(fabsf(axisX.x), fmaxf(fabsf(axisX.y), fabsf(axisX.z)));
    axisScale = fmaxf(axisScale, fmaxf(fabsf(axisY.x), fmaxf(fabsf(axisY.y), fabsf(axisY.z))));
    axisScale = fmaxf(axisScale, fmaxf(fabsf(axisZ.x), fmaxf(fabsf(axisZ.y), fabsf(axisZ.z))));
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
    const float3 cofactorNormal = affineSphereCofactor(axisX, axisY, axisZ, objectNormal);
    const float jacobian = finiteVectorLength(cofactorNormal);
    normal = jacobian > 0.0f ? orientation * normalizeFiniteVectorOrZero(cofactorNormal) : make_float3(0.0f);
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
    float jacobianMean = 0.0f;
    for (unsigned int i = 0u; i < sampleCount; ++i)
    {
        const float z = 1.0f - 2.0f * (float(i) + 0.5f) / float(sampleCount);
        const float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
        const float phi = goldenAngle * float(i);
        const float3 objectNormal = make_float3(radial * cosf(phi), radial * sinf(phi), z);
        jacobianMean += finiteVectorLength(affineSphereCofactor(axisX, axisY, axisZ, objectNormal)) / float(sampleCount);
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
    const float xx = dot(scaledX, scaledX);
    const float xy = dot(scaledX, scaledY);
    const float yy = dot(scaledY, scaledY);
    const float determinant = xx * yy - xy * xy;
    if (!(determinant > 0.0f))
    {
        return result;
    }
    const float offsetX = dot(offset, scaledX);
    const float offsetY = dot(offset, scaledY);
    const float x = (yy * offsetX - xy * offsetY) / determinant;
    const float y = (xx * offsetY - xy * offsetX) / determinant;
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
