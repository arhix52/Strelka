#ifndef STRELKA_EMISSIVE_MESH_LIGHT_H
#define STRELKA_EMISSIVE_MESH_LIGHT_H

#include <light_pdf.h>

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

struct EmissiveTriangleMeasure
{
    float3 normal;
    float areaPdf;
};

struct EmissiveScaledTerm
{
    float mantissa;
    int exponent;
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

DEVICE_FUNC EmissiveScaledTerm emissiveNormalizeTerm(float value, int exponent)
{
    EmissiveScaledTerm term{};
    if (value != 0.0f)
    {
        int localExponent = 0;
        term.mantissa = decomposeFloatExponent(value, localExponent);
        term.exponent = exponent + localExponent;
    }
    return term;
}

DEVICE_FUNC void emissiveAppendProduct(float a,
                                       float b,
                                       float sign,
                                       THREAD_REF EmissiveScaledTerm* terms,
                                       THREAD_REF unsigned int& count)
{
    if (a == 0.0f || b == 0.0f)
    {
        return;
    }
    int exponentA = 0;
    int exponentB = 0;
    const float mantissaA = decomposeFloatExponent(a, exponentA);
    const float mantissaB = decomposeFloatExponent(b, exponentB);
    const CompensatedFloat product = compensatedProduct(mantissaA, mantissaB);
    const EmissiveScaledTerm high = emissiveNormalizeTerm(sign * product.high, exponentA + exponentB);
    const EmissiveScaledTerm low = emissiveNormalizeTerm(sign * product.low, exponentA + exponentB);
    if (high.mantissa != 0.0f)
    {
        terms[count++] = high;
    }
    if (low.mantissa != 0.0f)
    {
        terms[count++] = low;
    }
}

DEVICE_FUNC void emissiveAddTerms(EmissiveScaledTerm a,
                                  EmissiveScaledTerm b,
                                  THREAD_REF EmissiveScaledTerm& high,
                                  THREAD_REF EmissiveScaledTerm& low)
{
    if (a.mantissa == 0.0f)
    {
        high = b;
        low = a;
        return;
    }
    if (b.mantissa == 0.0f)
    {
        high = a;
        low = b;
        return;
    }
    const EmissiveScaledTerm larger = a.exponent >= b.exponent ? a : b;
    const EmissiveScaledTerm smaller = a.exponent >= b.exponent ? b : a;
    if (larger.exponent - smaller.exponent > 25)
    {
        high = larger;
        low = smaller;
        return;
    }
    const float alignedSmaller = scaleFloatExponent(smaller.mantissa, smaller.exponent - larger.exponent);
    const CompensatedFloat sum = compensatedSum(larger.mantissa, alignedSmaller);
    high = emissiveNormalizeTerm(sum.high, larger.exponent);
    low = emissiveNormalizeTerm(sum.low, larger.exponent);
}

// Exact products of the original float vertices are retained as separately
// exponent-scaled terms. This avoids both an overflowing p1-p0 and the loss of
// a small endpoint that determines the area after large products cancel.
DEVICE_FUNC EmissiveScaledTerm
emissiveTriangleCrossComponent(float a0, float a1, float a2, float b0, float b1, float b2)
{
    EmissiveScaledTerm terms[12];
    unsigned int count = 0u;
    emissiveAppendProduct(a1, b2, 1.0f, terms, count);
    emissiveAppendProduct(a1, b0, -1.0f, terms, count);
    emissiveAppendProduct(a0, b2, -1.0f, terms, count);
    emissiveAppendProduct(b1, a2, -1.0f, terms, count);
    emissiveAppendProduct(b1, a0, 1.0f, terms, count);
    emissiveAppendProduct(b0, a2, 1.0f, terms, count);
    if (count == 0u)
    {
        return EmissiveScaledTerm{};
    }

    for (unsigned int i = 1u; i < count; ++i)
    {
        const EmissiveScaledTerm term = terms[i];
        unsigned int j = i;
        while (j > 0u && (term.exponent < terms[j - 1u].exponent ||
                          (term.exponent == terms[j - 1u].exponent &&
                           fabsf(term.mantissa) < fabsf(terms[j - 1u].mantissa))))
        {
            terms[j] = terms[j - 1u];
            --j;
        }
        terms[j] = term;
    }

    EmissiveScaledTerm expansion[12];
    unsigned int expansionCount = 0u;
    EmissiveScaledTerm sum = terms[0];
    for (unsigned int i = 1u; i < count; ++i)
    {
        EmissiveScaledTerm high{};
        EmissiveScaledTerm low{};
        emissiveAddTerms(sum, terms[i], high, low);
        if (low.mantissa != 0.0f)
        {
            expansion[expansionCount++] = low;
        }
        sum = high;
    }
    if (sum.mantissa != 0.0f)
    {
        expansion[expansionCount++] = sum;
    }
    return expansionCount > 0u ? expansion[expansionCount - 1u] : EmissiveScaledTerm{};
}

DEVICE_FUNC EmissiveTriangleMeasure emissiveTriangleMeasure(float3 p0, float3 p1, float3 p2)
{
    EmissiveTriangleMeasure measure{};
    measure.normal = make_float3(0.0f);
    measure.areaPdf = 0.0f;

    constexpr float maxFinite = 3.402823466e38f;
    if (!(fabsf(p0.x) <= maxFinite) || !(fabsf(p0.y) <= maxFinite) || !(fabsf(p0.z) <= maxFinite) ||
        !(fabsf(p1.x) <= maxFinite) || !(fabsf(p1.y) <= maxFinite) || !(fabsf(p1.z) <= maxFinite) ||
        !(fabsf(p2.x) <= maxFinite) || !(fabsf(p2.y) <= maxFinite) || !(fabsf(p2.z) <= maxFinite))
    {
        return measure;
    }
    const EmissiveScaledTerm crossX =
        emissiveTriangleCrossComponent(p0.y, p1.y, p2.y, p0.z, p1.z, p2.z);
    const EmissiveScaledTerm crossY =
        emissiveTriangleCrossComponent(p0.z, p1.z, p2.z, p0.x, p1.x, p2.x);
    const EmissiveScaledTerm crossZ =
        emissiveTriangleCrossComponent(p0.x, p1.x, p2.x, p0.y, p1.y, p2.y);
    int commonExponent = crossX.mantissa != 0.0f ? crossX.exponent : -1000000;
    if (crossY.mantissa != 0.0f)
    {
        commonExponent = commonExponent > crossY.exponent ? commonExponent : crossY.exponent;
    }
    if (crossZ.mantissa != 0.0f)
    {
        commonExponent = commonExponent > crossZ.exponent ? commonExponent : crossZ.exponent;
    }
    if (commonExponent == -1000000)
    {
        return measure;
    }

    const float3 scaledCross = make_float3(scaleFloatExponent(crossX.mantissa, crossX.exponent - commonExponent),
                                           scaleFloatExponent(crossY.mantissa, crossY.exponent - commonExponent),
                                           scaleFloatExponent(crossZ.mantissa, crossZ.exponent - commonExponent));
    const float scaledLength = finiteVectorLength(scaledCross);
    if (!(scaledLength > 0.0f))
    {
        return measure;
    }

    int lengthExponent = 0;
    const float lengthMantissa = decomposeFloatExponent(scaledLength, lengthExponent);
    const float areaPdf = scaleFloatExponent(2.0f / lengthMantissa, -lengthExponent - commonExponent);
    constexpr float minNormal = 1.175494351e-38f;
    if (!(areaPdf >= minNormal) || !(areaPdf <= maxFinite))
    {
        return measure;
    }
    measure.normal = scaledCross / scaledLength;
    measure.areaPdf = areaPdf;
    return measure;
}

DEVICE_FUNC float emissiveTriangleAreaPdf(float3 p0, float3 p1, float3 p2)
{
    return emissiveTriangleMeasure(p0, p1, p2).areaPdf;
}

DEVICE_FUNC float finiteConvexLerp(float a, float b, float t)
{
    const bool crossesZero = (a < 0.0f && b > 0.0f) || (a > 0.0f && b < 0.0f);
    const float value = crossesZero ? fmaf(t, b, (1.0f - t) * a) : fmaf(t, b - a, a);
    return fminf(fmaxf(value, fminf(a, b)), fmaxf(a, b));
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
    const float3 edgePoint = make_float3(finiteConvexLerp(p1.x, p2.x, edgeCoordinate),
                                         finiteConvexLerp(p1.y, p2.y, edgeCoordinate),
                                         finiteConvexLerp(p1.z, p2.z, edgeCoordinate));
    sample.point = make_float3(finiteConvexLerp(p0.x, edgePoint.x, root),
                               finiteConvexLerp(p0.y, edgePoint.y, root),
                               finiteConvexLerp(p0.z, edgePoint.z, root));
    sample.normal = measure.normal;
    const float2 edgeUv = make_float2(finiteConvexLerp(uv1.x, uv2.x, edgeCoordinate),
                                      finiteConvexLerp(uv1.y, uv2.y, edgeCoordinate));
    sample.uv = make_float2(finiteConvexLerp(uv0.x, edgeUv.x, root), finiteConvexLerp(uv0.y, edgeUv.y, root));
    constexpr float maxFinite = 3.402823466e38f;
    if (!(fabsf(sample.point.x) <= maxFinite) || !(fabsf(sample.point.y) <= maxFinite) ||
        !(fabsf(sample.point.z) <= maxFinite) || !(fabsf(sample.uv.x) <= maxFinite) ||
        !(fabsf(sample.uv.y) <= maxFinite))
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
