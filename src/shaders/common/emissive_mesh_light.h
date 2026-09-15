#ifndef STRELKA_EMISSIVE_MESH_LIGHT_H
#define STRELKA_EMISSIVE_MESH_LIGHT_H

#include <light_pdf.h>

struct EmissiveMeshLight
{
    unsigned int instanceId;
    unsigned int geometryId;
    unsigned int triangleOffset;
    unsigned int triangleCount;

    unsigned int vertexOffset;
    unsigned int indexOffset;
    unsigned int materialId;
    float selectionPdf;

    unsigned int aliasThreshold;
    unsigned int alias;
    // Metal: descriptor-buffer record containing this geometry's transform.
    // OptiX keeps its separate transform table and leaves this as instanceId.
    unsigned int transformIndex;
    unsigned int pad1;
};
static_assert(sizeof(EmissiveMeshLight) == 48, "EmissiveMeshLight host/GPU ABI changed");

// One primitive in an EmissiveMeshLight. All primitives remain addressable so
// hit lookup is triangleOffset + primitiveId; zero-area triangles simply have
// zero conditional probability and are never returned by the alias draw.
struct EmissiveTriangleLight
{
    float selectionPdf;
    unsigned int aliasThreshold;
    unsigned int alias;
    unsigned int pad;
};
static_assert(sizeof(EmissiveTriangleLight) == 16, "EmissiveTriangleLight host/GPU ABI changed");

struct EmissiveInstanceTransform
{
    float matrix[12];
};
static_assert(sizeof(EmissiveInstanceTransform) == 48, "EmissiveInstanceTransform host/GPU ABI changed");

struct EmissiveTriangleSample
{
    float3 point;
    float3 normal;
    float2 uv;
    float areaPdf;
    bool valid;
};

struct EmissiveTriangleMeasure
{
    float3 normal;
    float areaPdf;
};

struct EmissiveVisibilitySegment
{
    float3 direction;
    float maxDistance;
    bool valid;
};

DEVICE_FUNC EmissiveTriangleMeasure emissiveTriangleMeasure(float3 p0, float3 p1, float3 p2)
{
    EmissiveTriangleMeasure measure{};
    const float3 areaVector = cross(p1 - p0, p2 - p0);
    const float twiceAreaSquared = dot(areaVector, areaVector);
    if (twiceAreaSquared > 0.0f)
    {
        const float twiceArea = sqrtf(twiceAreaSquared);
        measure.normal = areaVector / twiceArea;
        measure.areaPdf = 2.0f / twiceArea;
    }
    return measure;
}

DEVICE_FUNC float emissiveTriangleAreaPdf(float3 p0, float3 p1, float3 p2)
{
    return emissiveTriangleMeasure(p0, p1, p2).areaPdf;
}

DEVICE_FUNC EmissiveVisibilitySegment emissiveVisibilitySegment(float3 sourceOffset, float3 targetOffset)
{
    EmissiveVisibilitySegment segment;
    segment.direction = make_float3(0.0f);
    segment.maxDistance = 0.0f;
    segment.valid = false;
    const float3 delta = targetOffset - sourceOffset;
    float distance;
    const float3 direction = finiteDirectionAndDistance(delta, distance);
    if (distance > 0.0f)
    {
        segment.direction = direction;
        segment.maxDistance = distance;
        segment.valid = true;
    }
    return segment;
}

DEVICE_FUNC EmissiveTriangleSample
sampleEmissiveTriangle(float3 p0, float3 p1, float3 p2, float2 uv0, float2 uv1, float2 uv2, float u0, float u1)
{
    EmissiveTriangleSample sample;
    sample.point = make_float3(0.0f);
    sample.normal = make_float3(0.0f);
    sample.uv = make_float2(0.0f, 0.0f);
    sample.areaPdf = 0.0f;
    sample.valid = false;

    const EmissiveTriangleMeasure measure = emissiveTriangleMeasure(p0, p1, p2);
    if (!(measure.areaPdf > 0.0f) || !(dot(measure.normal, measure.normal) > 0.0f))
    {
        return sample;
    }

    const float root = sqrtf(fminf(fmaxf(u0, 0.0f), 1.0f));
    const float edgeCoordinate = fminf(fmaxf(u1, 0.0f), 1.0f);
    const float3 edgePoint = make_float3(fmaf(edgeCoordinate, p2.x - p1.x, p1.x), fmaf(edgeCoordinate, p2.y - p1.y, p1.y),
                                         fmaf(edgeCoordinate, p2.z - p1.z, p1.z));
    sample.point = make_float3(fmaf(root, edgePoint.x - p0.x, p0.x), fmaf(root, edgePoint.y - p0.y, p0.y),
                               fmaf(root, edgePoint.z - p0.z, p0.z));
    sample.normal = measure.normal;
    const float2 edgeUv =
        make_float2(fmaf(edgeCoordinate, uv2.x - uv1.x, uv1.x), fmaf(edgeCoordinate, uv2.y - uv1.y, uv1.y));
    sample.uv = make_float2(fmaf(root, edgeUv.x - uv0.x, uv0.x), fmaf(root, edgeUv.y - uv0.y, uv0.y));
    constexpr float maxFinite = 3.402823466e38f;
    if (!(fabsf(sample.point.x) <= maxFinite) || !(fabsf(sample.point.y) <= maxFinite) ||
        !(fabsf(sample.point.z) <= maxFinite) || !(fabsf(sample.uv.x) <= maxFinite) || !(fabsf(sample.uv.y) <= maxFinite))
    {
        return sample;
    }
    sample.areaPdf = measure.areaPdf;
    sample.valid = true;
    return sample;
}

DEVICE_FUNC float emissiveTriangleSolidAnglePdf(float areaPdf, float3 shadingPoint, float3 pointOnLight, float3 lightNormal)
{
    const float3 offset = pointOnLight - shadingPoint;
    const float distance = finiteVectorLength(offset);
    const float3 direction = normalizeFiniteVectorOrZero(offset);
    if (!(areaPdf > 0.0f) || !(distance > 0.0f) || !(dot(direction, direction) > 0.0f))
    {
        return 0.0f;
    }
    return areaPdfToSolidAnglePdf(distance, fabsf(dot(lightNormal, -direction)), areaPdf);
}

DEVICE_FUNC float emissiveMeshMarginalSolidAnglePdf(float localSelectionPdf,
                                                    float meshClassPdf,
                                                    float meshSelectionPdf,
                                                    float triangleSelectionPdf,
                                                    float areaPdf,
                                                    float3 shadingPoint,
                                                    float3 pointOnLight,
                                                    float3 lightNormal)
{
    if (!(localSelectionPdf > 0.0f) || !(meshClassPdf > 0.0f) || !(meshSelectionPdf > 0.0f) ||
        !(triangleSelectionPdf > 0.0f))
    {
        return 0.0f;
    }
    const float3 offset = pointOnLight - shadingPoint;
    const float distance = finiteVectorLength(offset);
    const float3 direction = normalizeFiniteVectorOrZero(offset);
    if (!(distance > 0.0f) || !(dot(direction, direction) > 0.0f))
    {
        return 0.0f;
    }
    return areaPdfToSolidAngleMarginalPdf(distance, fabsf(dot(lightNormal, -direction)), areaPdf, localSelectionPdf,
                                          meshClassPdf, meshSelectionPdf, triangleSelectionPdf);
}

DEVICE_FUNC bool emissiveMeshKeyLess(unsigned int lightInstanceId,
                                     unsigned int lightGeometryId,
                                     unsigned int instanceId,
                                     unsigned int geometryId)
{
    return lightInstanceId < instanceId || (lightInstanceId == instanceId && lightGeometryId < geometryId);
}

#endif // STRELKA_EMISSIVE_MESH_LIGHT_H
