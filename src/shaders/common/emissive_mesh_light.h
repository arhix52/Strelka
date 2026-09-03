#ifndef STRELKA_EMISSIVE_MESH_LIGHT_H
#define STRELKA_EMISSIVE_MESH_LIGHT_H

#include <strelka/material/material_math.h>

// One material/geometry instance in the emissive-mesh hierarchy. The outer
// alias table selects this record; its contiguous triangle range owns a second
// alias table. IDs are renderer traversal IDs, so a BSDF hit can recover the
// same record without guessing from material identity.
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

    float aliasProbability;
    unsigned int alias;
    unsigned int pad0;
    unsigned int pad1;
};
static_assert(sizeof(EmissiveMeshLight) == 48, "EmissiveMeshLight host/GPU ABI changed");

// One primitive in an EmissiveMeshLight. All primitives remain addressable so
// hit lookup is triangleOffset + primitiveId; zero-area triangles simply have
// zero conditional probability and are never returned by the alias draw.
struct EmissiveTriangleLight
{
    float selectionPdf;
    float aliasProbability;
    unsigned int alias;
    unsigned int pad;
};
static_assert(sizeof(EmissiveTriangleLight) == 16, "EmissiveTriangleLight host/GPU ABI changed");

// Row-major affine transform for an emissive instance. OptiX cannot query an
// arbitrary TLAS instance transform while constructing a next-event proposal,
// so the host publishes the same current/previous matrices used by traversal.
// Metal reads its instance descriptor directly and does not need this table.
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

// A mesh emitter remains part of the traversed scene, so a visibility ray must
// stop at an offset point on its near side instead of running through the true
// sample and letting the emitter occlude itself.  Both endpoints are supplied
// by the backend's scale-aware offset_ray(); this helper only constructs the
// exact finite segment between them.
struct EmissiveVisibilitySegment
{
    float3 direction;
    float maxDistance;
    bool valid;
};

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

    const float3 edge1 = p1 - p0;
    const float3 edge2 = p2 - p0;
    const float inverseTwiceArea = inverseFiniteCrossLength(edge1, edge2);
    const float3 normal = finiteCrossDirection(edge1, edge2);
    if (!(inverseTwiceArea > 0.0f) || !(dot(normal, normal) > 0.0f))
    {
        return sample;
    }

    const float root = sqrtf(fminf(fmaxf(u0, 0.0f), 1.0f));
    const float b0 = 1.0f - root;
    const float b1 = root * (1.0f - fminf(fmaxf(u1, 0.0f), 1.0f));
    const float b2 = root - b1;
    sample.point = b0 * p0 + b1 * p1 + b2 * p2;
    sample.normal = normal;
    sample.uv = b0 * uv0 + b1 * uv1 + b2 * uv2;
    constexpr float maxFinite = 3.402823466e38f;
    sample.areaPdf = fminf(2.0f * inverseTwiceArea, maxFinite);
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
    const float cosine = fabsf(dot(lightNormal, -direction));
    if (!(cosine > 0.0f))
    {
        return 0.0f;
    }
    constexpr float maxFinite = 3.402823466e38f;
    const float scaledDistance = distance * sqrtf(areaPdf);
    const float largestFiniteDistance = sqrtf(maxFinite * cosine);
    if (!(scaledDistance > 0.0f))
    {
        return 0.0f;
    }
    return scaledDistance <= largestFiniteDistance ? scaledDistance * scaledDistance / cosine : maxFinite;
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
    constexpr float maxFinite = 3.402823466e38f;
    const float conditionalPdf = emissiveTriangleSolidAnglePdf(areaPdf, shadingPoint, pointOnLight, lightNormal);
    const float pdf = localSelectionPdf * meshClassPdf * meshSelectionPdf * triangleSelectionPdf * conditionalPdf;
    return pdf > 0.0f ? fminf(pdf, maxFinite) : 0.0f;
}

DEVICE_FUNC bool emissiveMeshKeyLess(unsigned int lightInstanceId,
                                     unsigned int lightGeometryId,
                                     unsigned int instanceId,
                                     unsigned int geometryId)
{
    return lightInstanceId < instanceId || (lightInstanceId == instanceId && lightGeometryId < geometryId);
}

#endif // STRELKA_EMISSIVE_MESH_LIGHT_H
