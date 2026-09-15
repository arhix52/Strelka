#ifndef STRELKA_ANALYTIC_LIGHT_H
#define STRELKA_ANALYTIC_LIGHT_H

#include <strelka/material/material_math.h>
#include <light_types.h>
#include <light_pdf.h>

enum : unsigned int
{
    STRELKA_ANALYTIC_LIGHT_CAMERA_BIT = 1u,
    STRELKA_ANALYTIC_LIGHT_SECONDARY_BIT = 2u
};

DEVICE_FUNC bool analyticLightVisibilityAllowsRay(float packedVisibility, bool includeCameraHidden)
{
    if (!(packedVisibility >= 1.0f) || !(packedVisibility <= 3.0f))
    {
        return false;
    }
    // Scene packing writes the bit mask as an exact small integer float.
    const unsigned int visibility = (unsigned int)packedVisibility;
    return (visibility & STRELKA_ANALYTIC_LIGHT_CAMERA_BIT) != 0u ||
           (includeCameraHidden && (visibility & STRELKA_ANALYTIC_LIGHT_SECONDARY_BIT) != 0u);
}

DEVICE_FUNC bool lightUsesAnalyticAreaIntersection(int lightType)
{
    return lightType == LIGHT_TYPE_RECT || lightType == LIGHT_TYPE_DISC || lightType == LIGHT_TYPE_SPHERE;
}

DEVICE_FUNC bool lightUsesAnalyticSurfaceIntersection(int lightType, float radius)
{
    return lightUsesAnalyticAreaIntersection(lightType) || (lightIsPunctual(lightType) && punctualLightIsSoft(radius));
}

struct AnalyticLightSample
{
    float3 point{};
    float3 normal{};
    float areaPdf = 0.0f;
};

struct AnalyticLightIntersection
{
    float distance = 0.0f;
    float3 point{};
    float3 normal{};
    float areaPdf = 0.0f;
    bool hit = false;
};

struct CanonicalAnalyticIntersection
{
    float distance = 0.0f;
    bool hit = false;
};

DEVICE_FUNC float2 sampleCanonicalDisc(float u1, float u2)
{
    const float radius = sqrtf(fminf(fmaxf(u1, 0.0f), 1.0f));
    const float phi = 2.0f * M_PI_F * u2;
    return make_float2(radius * cosf(phi), radius * sinf(phi));
}

DEVICE_FUNC float3 sampleCanonicalSphere(float u1, float u2)
{
    const float z = 1.0f - 2.0f * u1;
    const float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    const float phi = 2.0f * M_PI_F * u2;
    return make_float3(radial * cosf(phi), radial * sinf(phi), z);
}

DEVICE_FUNC CanonicalAnalyticIntersection intersectCanonicalSphere(float3 origin,
                                                                   float3 direction,
                                                                   float minDistance,
                                                                   float maxDistance)
{
    CanonicalAnalyticIntersection result;
    const float a = dot(direction, direction);
    const float halfB = dot(origin, direction);
    const float c = dot(origin, origin) - 1.0f;
    const float discriminant = halfB * halfB - a * c;
    if (discriminant < 0.0f)
        return result;
    const float root = sqrtf(discriminant);
    float distance = (-halfB - root) / a;
    if (distance < minDistance || distance > maxDistance)
        distance = (-halfB + root) / a;
    result.distance = distance;
    result.hit = distance >= minDistance && distance <= maxDistance;
    return result;
}

DEVICE_FUNC CanonicalAnalyticIntersection intersectCanonicalDisc(float3 origin,
                                                                 float3 direction,
                                                                 float minDistance,
                                                                 float maxDistance)
{
    CanonicalAnalyticIntersection result;
    if (direction.z == 0.0f)
        return result;
    const float distance = -origin.z / direction.z;
    const float2 point = make_float2(origin.x + distance * direction.x, origin.y + distance * direction.y);
    result.distance = distance;
    result.hit = distance >= minDistance && distance <= maxDistance && dot(point, point) <= 1.0f;
    return result;
}

DEVICE_FUNC bool analyticLightIntersectionSharesEvent(float distance, AnalyticLightIntersection candidate)
{
    return candidate.hit && candidate.distance == distance;
}

DEVICE_FUNC float analyticDiscArea(float3 axisX, float3 axisY)
{
    const float3 areaVector = cross(axisX, axisY);
    return M_PI_F * sqrtf(dot(areaVector, areaVector));
}

DEVICE_FUNC float analyticDiscAreaPdf(float3 axisX, float3 axisY)
{
    const float3 areaVector = cross(axisX, axisY);
    const float area = sqrtf(dot(areaVector, areaVector));
    return area > 0.0f ? 1.0f / (M_PI_F * area) : 0.0f;
}

DEVICE_FUNC float affineSphereAreaPdfAndNormal(
    float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal, THREAD_REF float3& normal)
{
    normal = make_float3(0.0f);
    const float3 cofactorX = cross(axisY, axisZ);
    const float3 cofactor =
        objectNormal.x * cofactorX + objectNormal.y * cross(axisZ, axisX) + objectNormal.z * cross(axisX, axisY);
    const float cofactorLengthSquared = dot(cofactor, cofactor);
    const float determinant = dot(axisX, cofactorX);
    if (!(cofactorLengthSquared > 0.0f) || determinant == 0.0f)
    {
        return 0.0f;
    }
    const float inverseCofactorLength = 1.0f / sqrtf(cofactorLengthSquared);
    normal = copysignf(1.0f, determinant) * cofactor * inverseCofactorLength;
    return inverseCofactorLength * (1.0f / (4.0f * M_PI_F));
}

DEVICE_FUNC float3 transformAffineNormal(float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal)
{
    const float3 cofactorX = cross(axisY, axisZ);
    const float3 cofactor =
        objectNormal.x * cofactorX + objectNormal.y * cross(axisZ, axisX) + objectNormal.z * cross(axisX, axisY);
    const float lengthSquared = dot(cofactor, cofactor);
    const float determinant = dot(axisX, cofactorX);
    if (!(lengthSquared > 0.0f) || determinant == 0.0f)
    {
        return make_float3(0.0f);
    }
    return copysignf(1.0f, determinant) * cofactor / sqrtf(lengthSquared);
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

DEVICE_FUNC AnalyticLightSample
sampleAnalyticDisc(float3 center, float3 axisX, float3 axisY, float3 emissionNormal, float u1, float u2)
{
    AnalyticLightSample sample;
    const float areaPdf = analyticDiscAreaPdf(axisX, axisY);
    if (!(areaPdf > 0.0f) || !(dot(emissionNormal, emissionNormal) > 0.0f))
    {
        return sample;
    }
    const float2 objectPoint = sampleCanonicalDisc(u1, u2);
    sample.point = center + objectPoint.x * axisX + objectPoint.y * axisY;
    sample.normal = emissionNormal;
    sample.areaPdf = areaPdf;
    return sample;
}

DEVICE_FUNC AnalyticLightSample
sampleAnalyticEllipsoidUnchecked(float3 center, float3 axisX, float3 axisY, float3 axisZ, float u1, float u2)
{
    AnalyticLightSample sample;
    float3 objectNormal = sampleCanonicalSphere(u1, u2);
    const float objectLengthSquared = dot(objectNormal, objectNormal);
    if (objectLengthSquared > 0.0f)
    {
        objectNormal /= sqrtf(objectLengthSquared);
    }
    sample.point = make_float3(
        fmaf(objectNormal.z, axisZ.x, fmaf(objectNormal.y, axisY.x, fmaf(objectNormal.x, axisX.x, center.x))),
        fmaf(objectNormal.z, axisZ.y, fmaf(objectNormal.y, axisY.y, fmaf(objectNormal.x, axisX.y, center.y))),
        fmaf(objectNormal.z, axisZ.z, fmaf(objectNormal.y, axisY.z, fmaf(objectNormal.x, axisX.z, center.z))));
    sample.areaPdf = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, objectNormal, sample.normal);
    return sample;
}

DEVICE_FUNC float analyticEllipsoidAreaPdfUnchecked(
    float3 center, float3 axisX, float3 axisY, float3 axisZ, float3 point, THREAD_REF float3& normal)
{
    float3 objectNormal = affineSphereCoordinates(axisX, axisY, axisZ, point - center);
    const float objectLengthSquared = dot(objectNormal, objectNormal);
    if (!(objectLengthSquared > 0.0f))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    objectNormal *= 1.0f / sqrtf(objectLengthSquared);
    return affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, objectNormal, normal);
}

DEVICE_FUNC float analyticEllipsoidSurfaceArea(float3 axisX, float3 axisY, float3 axisZ)
{
    if (!(fabsf(dot(axisX, cross(axisY, axisZ))) > 0.0f))
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
    result.point = make_float3(0.0f);
    result.normal = make_float3(0.0f);
    result.areaPdf = 0.0f;
    result.hit = false;

    const float3 plane = cross(axisX, axisY);
    const float area = sqrtf(dot(plane, plane));
    const float denominator = dot(rayDirection, plane);
    if (!(area > 0.0f) || denominator == 0.0f)
    {
        return result;
    }
    const float distance = dot(center - rayOrigin, plane) / denominator;
    if (!(distance >= minDistance && distance < maxDistance))
    {
        return result;
    }
    const float3 offset = rayOrigin + distance * rayDirection - center;
    const float xx = dot(axisX, axisX);
    const float xy = dot(axisX, axisY);
    const float yy = dot(axisY, axisY);
    const float gramDeterminant = xx * yy - xy * xy;
    const float projectedX = dot(offset, axisX);
    const float projectedY = dot(offset, axisY);
    const float u = (yy * projectedX - xy * projectedY) / gramDeterminant;
    const float v = (xx * projectedY - xy * projectedX) / gramDeterminant;
    if (!(u * u + v * v <= 1.0f))
    {
        return result;
    }
    result.distance = distance;
    result.point = center + u * axisX + v * axisY;
    result.normal = emissionNormal;
    result.areaPdf = 1.0f / (M_PI_F * area);
    result.hit = true;
    return result;
}

DEVICE_FUNC AnalyticLightIntersection intersectAnalyticRectangle(float3 rayOrigin,
                                                                 float3 rayDirection,
                                                                 float minDistance,
                                                                 float maxDistance,
                                                                 float3 corner,
                                                                 float3 edgeX,
                                                                 float3 edgeY,
                                                                 float3 emissionNormal)
{
    AnalyticLightIntersection result;
    result.distance = maxDistance;
    result.point = make_float3(0.0f);
    result.normal = make_float3(0.0f);
    result.areaPdf = 0.0f;
    result.hit = false;

    const float3 plane = cross(edgeX, edgeY);
    const float area = sqrtf(dot(plane, plane));
    const float denominator = dot(rayDirection, plane);
    if (!(area > 0.0f) || denominator == 0.0f)
    {
        return result;
    }
    const float distance = dot(corner - rayOrigin, plane) / denominator;
    if (!(distance >= minDistance && distance < maxDistance))
    {
        return result;
    }
    const float3 offset = rayOrigin + distance * rayDirection - corner;
    const float xx = dot(edgeX, edgeX);
    const float xy = dot(edgeX, edgeY);
    const float yy = dot(edgeY, edgeY);
    const float gramDeterminant = xx * yy - xy * xy;
    const float projectedX = dot(offset, edgeX);
    const float projectedY = dot(offset, edgeY);
    const float u = (yy * projectedX - xy * projectedY) / gramDeterminant;
    const float v = (xx * projectedY - xy * projectedX) / gramDeterminant;
    if (!(u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f))
    {
        return result;
    }
    result.distance = distance;
    result.point = corner + u * edgeX + v * edgeY;
    result.normal = emissionNormal;
    result.areaPdf = 1.0f / area;
    result.hit = true;
    return result;
}

DEVICE_FUNC AnalyticLightIntersection intersectAnalyticEllipsoidUnchecked(float3 rayOrigin,
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
    result.point = make_float3(0.0f);
    result.normal = make_float3(0.0f);
    result.areaPdf = 0.0f;
    result.hit = false;

    const float3 cofactorX = cross(axisY, axisZ);
    const float3 cofactorY = cross(axisZ, axisX);
    const float3 cofactorZ = cross(axisX, axisY);
    const float determinant = dot(axisX, cofactorX);
    if (!(fabsf(determinant) > 0.0f))
    {
        return result;
    }
    const float inverseDeterminant = 1.0f / determinant;
    const float3 relativeOrigin = rayOrigin - center;
    const float3 objectOrigin =
        make_float3(dot(relativeOrigin, cofactorX), dot(relativeOrigin, cofactorY), dot(relativeOrigin, cofactorZ)) *
        inverseDeterminant;
    const float3 objectDirection =
        make_float3(dot(rayDirection, cofactorX), dot(rayDirection, cofactorY), dot(rayDirection, cofactorZ)) *
        inverseDeterminant;
    const float a = dot(objectDirection, objectDirection);
    const float halfB = dot(objectOrigin, objectDirection);
    const float c = dot(objectOrigin, objectOrigin) - 1.0f;
    const float discriminant = fmaf(-a, c, halfB * halfB);
    if (!(a > 0.0f) || !(discriminant >= 0.0f))
    {
        return result;
    }
    const float root = sqrtf(discriminant);
    float distance = (-halfB - root) / a;
    if (!(distance >= minDistance && distance < maxDistance))
    {
        distance = (-halfB + root) / a;
    }
    if (!(distance >= minDistance && distance < maxDistance))
    {
        return result;
    }
    float3 objectNormal = objectOrigin + distance * objectDirection;
    const float objectLengthSquared = dot(objectNormal, objectNormal);
    if (!(objectLengthSquared > 0.0f))
    {
        return result;
    }
    objectNormal *= 1.0f / sqrtf(objectLengthSquared);
    const float3 cofactorNormal = objectNormal.x * cofactorX + objectNormal.y * cofactorY + objectNormal.z * cofactorZ;
    const float cofactorLengthSquared = dot(cofactorNormal, cofactorNormal);
    if (!(cofactorLengthSquared > 0.0f))
    {
        return result;
    }
    const float inverseCofactorLength = 1.0f / sqrtf(cofactorLengthSquared);
    result.distance = distance;
    result.point = center + objectNormal.x * axisX + objectNormal.y * axisY + objectNormal.z * axisZ;
    result.normal = copysignf(1.0f, determinant) * cofactorNormal * inverseCofactorLength;
    result.areaPdf = inverseCofactorLength * (1.0f / (4.0f * M_PI_F));
    result.hit = true;
    return result;
}

DEVICE_FUNC AnalyticLightIntersection intersectAnalyticLightSurfaceUnchecked(int lightType,
                                                                             float3 point0,
                                                                             float3 point1,
                                                                             float3 point2,
                                                                             float3 point3,
                                                                             float3 emissionNormal,
                                                                             float3 rayOrigin,
                                                                             float3 rayDirection,
                                                                             float minDistance,
                                                                             float maxDistance)
{
    if (lightType == LIGHT_TYPE_RECT)
    {
        return intersectAnalyticRectangle(rayOrigin, rayDirection, minDistance, maxDistance, point0, point1 - point0,
                                          point3 - point0, emissionNormal);
    }
    if (lightType == LIGHT_TYPE_DISC)
    {
        return intersectAnalyticDisc(
            rayOrigin, rayDirection, minDistance, maxDistance, point1, point2, point3, emissionNormal);
    }
    if (lightType == LIGHT_TYPE_SPHERE)
    {
        return intersectAnalyticEllipsoidUnchecked(
            rayOrigin, rayDirection, minDistance, maxDistance, point1, point0, point2, point3);
    }
    const float radius = point0.x;
    if (lightIsPunctual(lightType) && punctualLightIsSoft(radius))
    {
        return intersectAnalyticEllipsoidUnchecked(rayOrigin, rayDirection, minDistance, maxDistance, point1,
                                                   make_float3(radius, 0.0f, 0.0f), make_float3(0.0f, radius, 0.0f),
                                                   make_float3(0.0f, 0.0f, radius));
    }
    AnalyticLightIntersection miss;
    miss.distance = maxDistance;
    return miss;
}

DEVICE_FUNC bool analyticLightSurfaceOccludesSegment(int lightType,
                                                     float3 point0,
                                                     float3 point1,
                                                     float3 point2,
                                                     float3 point3,
                                                     float3 emissionNormal,
                                                     float packedVisibility,
                                                     float3 rayOrigin,
                                                     float3 rayDirection,
                                                     float minDistance,
                                                     float maxDistance)
{
    return analyticLightVisibilityAllowsRay(packedVisibility, true) &&
           intersectAnalyticLightSurfaceUnchecked(lightType, point0, point1, point2, point3, emissionNormal, rayOrigin,
                                                  rayDirection, minDistance, maxDistance)
               .hit;
}

#endif // STRELKA_ANALYTIC_LIGHT_H
