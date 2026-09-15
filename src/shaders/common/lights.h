#pragma once
#include <vector_types.h>
#include <sutil/vec_math.h>
#include <light_types.h>
// The densities and the MIS heuristics themselves; this file only unpacks
// UniformLight and hands them scalars. Metal's lights_metal.h includes the same
// header, and tests/render/test_light_pdf.cpp compiles it on the host.
#include <light_pdf.h>
#include <analytic_light.h>
#include <rect_sampling.h>
#include <ies_math.h>
#include <projector.h>

struct UniformLight
{
    float4 points[4];
    float4 color;
    float4 normal;
    int type;
    float halfAngle;
    float pad0;
    float pad1;
    // Walker/Vose analytic-light distribution. color.w stores the represented
    // marginal PMF, matching Metal's UniformLight convention.
    unsigned int selectionAliasThreshold;
    unsigned int selectionAlias;
};

struct IesGpuBufferHeader
{
    unsigned int profileCount;
    unsigned int floatOffset; // byte offset of the float blob from the buffer start
    unsigned int pad0;
    unsigned int pad1;
};

struct IesGpuProfileHeader
{
    unsigned int nVertical;
    unsigned int nHorizontal;
    unsigned int anglesOffset; // index into the float blob: vertical then horizontal
    unsigned int candelaOffset; // index into the float blob
    float maxCandela;
    float pad0;
    float pad1;
    float pad2;
};

struct LightSampleData
{
    float3 pointOnLight;
    float pdf;

    float3 normal;
    float areaPdf;

    float3 L;
    float distToLight;

    float solidAngle = 0.0f;
};

struct AnalyticAreaLightHit
{
    float distance;
    float3 point;
    float3 normal;
    float areaPdf;
    unsigned int lightId;
    bool hit;
};

static __inline__ __device__ float calcLightAreaPdf(const UniformLight& l, const float3 hitPoint)
{
    float areaPdf = 0.0f;

    if (l.type == LIGHT_TYPE_RECT)
    {
        // Precomputed where the light is packed; see Scene::setLight.
        areaPdf = l.pad0;
    }
    else if (l.type == LIGHT_TYPE_DISC)
    {
        areaPdf = l.pad0;
    }
    else if (l.type == LIGHT_TYPE_SPHERE)
    {
        float3 normal;
        areaPdf = analyticEllipsoidAreaPdfUnchecked(make_float3(l.points[1]), make_float3(l.points[0]), make_float3(l.points[2]),
                                           make_float3(l.points[3]), hitPoint, normal);
    }
    else if (punctualLightIsSoft(l.points[0].x) && lightIsPunctual(l.type))
    {
        areaPdf = sphereLightAreaPdf(l.points[0].x);
    }
    return areaPdf;
}

static __inline__ __device__ float3 calcLightNormal(const UniformLight& l, const float3 hitPoint)
{
    float3 norm = make_float3(0.0f);

    if (l.type == LIGHT_TYPE_RECT)
    {
        norm = make_float3(l.normal);
    }
    else if (l.type == LIGHT_TYPE_DISC)
    {
        norm = make_float3(l.normal);
    }
    else if (l.type == LIGHT_TYPE_SPHERE)
    {
        analyticEllipsoidAreaPdfUnchecked(make_float3(l.points[1]), make_float3(l.points[0]), make_float3(l.points[2]),
                                 make_float3(l.points[3]), hitPoint, norm);
    }
    else if (lightIsPunctual(l.type))
    {
        // points[1] is the centre for all of them. A soft point is a sphere and
        // needs a real surface normal for the area-to-solid-angle conversion; a
        // sharp one never reaches a density that uses it.
        norm = normalize(hitPoint - make_float3(l.points[1]));
    }
    return norm;
}

static __inline__ __device__ void fillLightData(const UniformLight& l,
                                                const float3 hitPoint,
                                                LightSampleData& lightSampleData)
{
    const float3 toLight = lightSampleData.pointOnLight - hitPoint;
    float lenToLight;
    lightSampleData.L = finiteDirectionAndDistance(toLight, lenToLight);
    lightSampleData.distToLight = lenToLight;
    if (l.type == LIGHT_TYPE_SPHERE)
    {
        const AnalyticLightIntersection hit =
            intersectAnalyticEllipsoidUnchecked(hitPoint, lightSampleData.L, 0.0f, 3.402823466e38f, make_float3(l.points[1]),
                                       make_float3(l.points[0]), make_float3(l.points[2]), make_float3(l.points[3]));
        lightSampleData.areaPdf = hit.areaPdf;
        lightSampleData.normal = hit.normal;
        return;
    }
    lightSampleData.areaPdf = calcLightAreaPdf(l, lightSampleData.pointOnLight);
    lightSampleData.normal = calcLightNormal(l, lightSampleData.pointOnLight);
}

// The spherical-rectangle frame, its sample and its solid angle all come from
// common/rect_sampling.h, which Metal and the host tests compile as well. These
// wrappers only unpack UniformLight's four corners into a corner and two edges.
static __device__ SphQuad init(const UniformLight& l, const float3 o)
{
    const float3 p0 = make_float3(l.points[0]);
    return sphQuadInit(p0, make_float3(l.points[1]) - p0, make_float3(l.points[3]) - p0, o);
}

static __device__ float3 SphQuadSample(const SphQuad& squad, const float2 uv)
{
    return sphQuadSample(squad, uv.x, uv.y);
}

// MIS needs only the solid angle; sphQuadSolidAngle() drops the sample basis and
// the inverse-CDF constants so they do not consume registers on light hits.
static __inline__ __device__ float rectSolidAngle(const UniformLight& l, const float3 o, bool& useAreaFallback)
{
    const float3 p0 = make_float3(l.points[0]);
    return sphQuadSolidAngle(p0, make_float3(l.points[1]) - p0, make_float3(l.points[3]) - p0, o, useAreaFallback);
}

static __inline__ __device__ bool emitsLight(const float3 radiance)
{
    return radiance.x > 0.0f || radiance.y > 0.0f || radiance.z > 0.0f;
}

static __inline__ __device__ LightPdfQuery buildLightPdfQuery(const UniformLight& l, const LightSampleData& d)
{
    LightPdfQuery q = makeLightPdfQuery(l.type);
    q.distToLight = d.distToLight;
    q.cosAtLight = -dot(d.L, d.normal);
    q.areaPdf = d.areaPdf;
    q.halfAngle = l.halfAngle;
    if (lightIsPunctual(l.type))
    {
        q.radius = l.points[0].x;
    }
    return q;
}

/// The light-sampling density for a direction that arrived at `lightHitPoint`
/// from `surfaceHitPoint`. This is the number the BSDF half of the MIS estimate
/// weighs itself against, and it has to be the one the sampler below drew from.
static __inline__ __device__ float getLightPdf(const UniformLight& l,
                                               const float3 lightHitPoint,
                                               const float3 surfaceHitPoint,
                                               unsigned int rectLightSamplingMethod,
                                               float localSelectionPdf,
                                               float analyticSelectionPdf,
                                               float lightSelectionPdf)
{
    LightSampleData d{};
    d.pointOnLight = lightHitPoint;
    fillLightData(l, surfaceHitPoint, d);

    LightPdfQuery q = buildLightPdfQuery(l, d);
    if (l.type == LIGHT_TYPE_RECT && rectLightSamplingMethod != 0)
    {
        bool useAreaFallback = false;
        const float S = rectSolidAngle(l, surfaceHitPoint, useAreaFallback);
        if (S <= 0.0f)
        {
            return 0.0f;
        }
        q.solidAngle = useAreaFallback ? 0.0f : S;
    }
    return marginalLightSolidAnglePdf(q, localSelectionPdf, analyticSelectionPdf, lightSelectionPdf);
}

static __inline__ __device__ LightSampleData SampleRectLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
    float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);
    // lightSampleData.pointOnLight = make_float3(l.points[0]) + e1 * u.x + e2 * u.y;
    // https://www.arnoldrenderer.com/research/egsr2013_spherical_rectangle.pdf
    SphQuad quad = init(l, hitPoint);
    if (quad.S <= 0.0f)
    {
        lightSampleData.pdf = 0.0f;
        lightSampleData.pointOnLight = make_float3(l.points[0]) + e1 * u.x + e2 * u.y;
        fillLightData(l, hitPoint, lightSampleData);
        return lightSampleData;
    }
    if (quad.useAreaFallback)
    {
        lightSampleData.pointOnLight = make_float3(l.points[0]) + e1 * u.x + e2 * u.y;
        fillLightData(l, hitPoint, lightSampleData);
        lightSampleData.pdf = areaPdfToSolidAnglePdf(
            lightSampleData.distToLight, -dot(lightSampleData.L, lightSampleData.normal), lightSampleData.areaPdf);
        return lightSampleData;
    }

    lightSampleData.pointOnLight = SphQuadSample(quad, u);
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = 1.0f / quad.S;
    lightSampleData.solidAngle = quad.S;

    return lightSampleData;
}

static __inline__ __device__ LightSampleData SampleRectLightUniform(const UniformLight& l,
                                                                    const float2 u,
                                                                    const float3 hitPoint)
{
    LightSampleData lightSampleData;
    // // uniform sampling
    float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
    float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);
    lightSampleData.pointOnLight = make_float3(l.points[0]) + e1 * u.x + e2 * u.y;
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = areaPdfToSolidAnglePdf(
        lightSampleData.distToLight, -dot(lightSampleData.L, lightSampleData.normal), lightSampleData.areaPdf);
    return lightSampleData;
}

static __inline__ __device__ LightSampleData SampleDistantLight(const UniformLight& l,
                                                                const float2 u,
                                                                const uint2 retryWords,
                                                                const float3 hitPoint)
{
    LightSampleData lightSampleData;
    float pdf = 0.0f;
    const float3 axis = -make_float3(l.normal);
    float3 coneSample;
    if (distantLightIsDelta(l.halfAngle))
    {
        coneSample = axis;
        pdf = deltaLightPdf();
    }
    else
    {
        coneSample = sampleDistantLightDirection(u.x, u.y, retryWords.x, retryWords.y, l.halfAngle, axis);
        pdf = coneLightSolidAnglePdf(l.halfAngle);
    }

    lightSampleData.areaPdf = 0.0f;
    lightSampleData.distToLight = infiniteLightDistance();
    lightSampleData.L = coneSample;
    lightSampleData.normal = make_float3(l.normal);
    lightSampleData.pdf = pdf;
    lightSampleData.pointOnLight = coneSample;

    return lightSampleData;
}

// Uniform on the analytically transformed unit disc.
static __inline__ __device__ LightSampleData SampleDiscLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    const float3 center = make_float3(l.points[1]);
    const AnalyticLightSample sample =
        sampleAnalyticDisc(center, make_float3(l.points[2]), make_float3(l.points[3]), make_float3(l.normal), u.x, u.y);
    lightSampleData.pointOnLight = sample.point;
    const float3 toLight = sample.point - hitPoint;
    lightSampleData.L = finiteDirectionAndDistance(toLight, lightSampleData.distToLight);
    lightSampleData.normal = sample.normal;
    lightSampleData.areaPdf = sample.areaPdf;
    lightSampleData.pdf = areaPdfToSolidAnglePdf(
        lightSampleData.distToLight, -dot(lightSampleData.L, lightSampleData.normal), lightSampleData.areaPdf);
    return lightSampleData;
}

/// A unit-sphere direction mapped analytically to the affine ellipsoid.
static __inline__ __device__ LightSampleData SampleSphereLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    const float3 center = make_float3(l.points[1]);
    const AnalyticLightSample sample = sampleAnalyticEllipsoidUnchecked(
        center, make_float3(l.points[0]), make_float3(l.points[2]), make_float3(l.points[3]), u.x, u.y);
    lightSampleData.pointOnLight = sample.point;
    const float3 toLight = sample.point - hitPoint;
    lightSampleData.L = finiteDirectionAndDistance(toLight, lightSampleData.distToLight);
    lightSampleData.normal = sample.normal;
    lightSampleData.areaPdf = sample.areaPdf;
    lightSampleData.pdf = areaPdfToSolidAnglePdf(
        lightSampleData.distToLight, -dot(lightSampleData.L, lightSampleData.normal), lightSampleData.areaPdf);

    return lightSampleData;
}

static __inline__ __device__ LightSampleData SampleDomeLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;

    lightSampleData.L = uniformSphereDirection(u.x, u.y);
    lightSampleData.distToLight = infiniteLightDistance();
    lightSampleData.areaPdf = 0.0f;
    // Faces the shading point by construction, so the caller's -dot(L, normal)
    // test passes for every sampled direction.
    lightSampleData.normal = -lightSampleData.L;
    lightSampleData.pdf = domeLightSolidAnglePdf();
    lightSampleData.pointOnLight = hitPoint + lightSampleData.L * lightSampleData.distToLight;

    return lightSampleData;
}

static __inline__ __device__ LightSampleData SamplePointLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    const float radius = l.points[0].x;
    if (punctualLightIsSoft(radius))
    {
        LightSampleData lightSampleData;
        const float3 sphereDirection = uniformSphereDirection(u.x, u.y);
        const float3 lightPoint = make_float3(l.points[1]) + radius * sphereDirection;
        lightSampleData.pointOnLight = lightPoint;
        lightSampleData.L = finiteDirectionAndDistance(lightPoint - hitPoint, lightSampleData.distToLight);
        lightSampleData.normal = sphereDirection;
        lightSampleData.areaPdf = sphereLightAreaPdf(radius);
        lightSampleData.pdf = sphereLightSolidAnglePdf(
            lightSampleData.distToLight, -dot(lightSampleData.L, lightSampleData.normal), radius);
        return lightSampleData;
    }

    LightSampleData lightSampleData;
    const float3 center = make_float3(l.points[1]);
    const float3 toLight = center - hitPoint;
    float dist;
    lightSampleData.pointOnLight = center;
    lightSampleData.L = finiteDirectionAndDistance(toLight, dist);
    lightSampleData.distToLight = dist;
    lightSampleData.normal = -lightSampleData.L;
    lightSampleData.areaPdf = 0.0f;
    lightSampleData.pdf = deltaLightPdf();
    return lightSampleData;
}

static __inline__ __device__ float spotAttenuation(const UniformLight& l, const float3 dirFromLight)
{
    const float3 axis = normalize(make_float3(l.normal));
    const float cosOuter = cosf(l.halfAngle);
    const float cosInner = cosf(l.pad0);
    const float cosTheta = dot(axis, dirFromLight);
    if (cosTheta < cosOuter)
        return 0.0f;
    if (cosInner <= cosOuter)
        return 1.0f;
    return clamp((cosTheta - cosOuter) / (cosInner - cosOuter), 0.0f, 1.0f);
}

static __inline__ __device__ float rangeWindow(const UniformLight& l, float dist)
{
    if (l.pad1 <= 0.0f)
        return 1.0f;
    const float x = clamp(dist / l.pad1, 0.0f, 1.0f);
    const float y = 1.0f - x * x * x * x;
    return y * y;
}

static __inline__ __device__ float areaFalloff(const UniformLight& l, float dist)
{
    if (l.pad1 <= 0.0f)
    {
        return 1.0f;
    }
    if (l.type != LIGHT_TYPE_RECT && l.type != LIGHT_TYPE_DISC && l.type != LIGHT_TYPE_SPHERE)
    {
        return 1.0f;
    }
    const float x = clamp(dist / l.pad1, 0.0f, 1.0f);
    // 1 - smootherstep(x), the Map Range (SMOOTHERSTEP) node feeding the Mix.
    const float s = x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
    return 1.0f - s;
}

// The index of the image a projector light throws, or a negative number when it
// throws a plain white frame. points[0].z, the slot Scene::updateLight fills
// from UniformLightDesc::projectorImage.
static __inline__ __device__ int projectorImageIndex(const UniformLight& l)
{
    return packedNonnegativeIndex(l.points[0].z);
}

static __inline__ __device__ ProjectorSample projectorSampleForLight(const UniformLight& l, const float3 dirFromLight)
{
    const OrthonormalLightFrame frame = makeOrthonormalLightFrame(
        make_float3(l.points[2]), make_float3(l.points[3]), make_float3(l.normal));
    const float tanX = projectorTanHalfX(l.halfAngle);
    const float tanY = projectorTanHalfY(tanX, l.points[0].w);
    if (!frame.valid)
    {
        return projectorProject(0.0f, 0.0f, -1.0f, tanX, tanY, l.pad0);
    }
    const float3 d = normalizeFiniteVectorOrZero(dirFromLight);
    return projectorProject(
        dot(d, frame.x), dot(d, frame.y), dot(d, frame.emissionAxis), tanX, tanY, l.pad0);
}

static __inline__ __device__ float sampleIesCandela(const IesGpuBufferHeader* iesBuffer,
                                                    const UniformLight& l,
                                                    const float3 dirFromLight)
{
    const int profileIdx = packedNonnegativeIndex(l.points[0].y);
    if (!iesBuffer || profileIdx < 0 || (unsigned int)profileIdx >= iesBuffer->profileCount)
    {
        return 1.0f;
    }

    const IesGpuProfileHeader* headers =
        (const IesGpuProfileHeader*)((const char*)iesBuffer + sizeof(IesGpuBufferHeader));
    const IesGpuProfileHeader& h = headers[profileIdx];
    if (h.nVertical < 2u || h.nHorizontal < 2u)
    {
        return 0.0f;
    }

    const float* floats = (const float*)((const char*)iesBuffer + iesBuffer->floatOffset);

    // World -> light local. Columns of the light's basis; -Z is the photometric
    // axis, matching iesloader.cpp::sampleIesCandela.
    const OrthonormalLightFrame frame = makeOrthonormalLightFrame(
        make_float3(l.points[2]), make_float3(l.points[3]), make_float3(l.normal));
    if (!frame.valid)
    {
        return 0.0f;
    }
    const float3 d = normalizeFiniteVectorOrZero(dirFromLight);
    const float3 local = make_float3(dot(d, frame.x), dot(d, frame.y), -dot(d, frame.emissionAxis));

    const float vertDeg = acosf(clamp(-local.z, -1.0f, 1.0f)) * (180.0f / M_PIf);
    const float horizDeg = atan2f(local.x, -local.y) * (180.0f / M_PIf);

    // The same evaluation the host and Metal run -- see common/ies_math.h.
    return iesEvaluate(floats + h.anglesOffset, (int)h.nVertical, floats + h.anglesOffset + h.nVertical,
                       (int)h.nHorizontal, floats + h.candelaOffset, vertDeg, horizDeg);
}
