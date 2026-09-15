#pragma once
#include <metal_stdlib>
#include <simd/simd.h>

#include "ShaderTypes.h"
// Densities and MIS heuristics shared with the OptiX backend and with the host
// tests. This file only unpacks UniformLight and hands them scalars.
#include <light_types.h>
#include <ies_math.h>
#include <rect_sampling.h>
#include <light_pdf.h>
#include <analytic_light.h>
#include <projector.h>

using namespace metal;

constant bool kFcFastRectLightData [[function_constant(32)]];
constant bool SPEC_FAST_RECT_LIGHT_DATA =
    is_function_constant_defined(kFcFastRectLightData) ? kFcFastRectLightData : true;

struct LightSampleData
{
    float3 pointOnLight;
    // Solid angle of a successful spherical-rectangle draw; zero selects the
    // area form. Reuses the old `pdf` slot without growing per-thread state.
    float solidAngle;

    float3 normal;
    float areaPdf;

    float3 L;
    float distToLight;
};

struct PackedAnalyticHit
{
    float distance;
    bool hit;
};

static __inline__ PackedAnalyticHit intersectPackedRectangle(
    device const UniformLight& light, float3 rayOrigin, float3 rayDirection, float minDistance, float maxDistance)
{
    const float3 corner = float3(light.points[0]);
    const float3 edgeX = float3(light.points[1]) - corner;
    const float3 edgeY = float3(light.points[3]) - corner;
    const float3 planeNormal = float3(light.normal);
    const float denominator = dot(rayDirection, planeNormal);
    if (denominator == 0.0f)
    {
        return { maxDistance, false };
    }

    const float distance = dot(corner - rayOrigin, planeNormal) / denominator;
    if (!(distance >= minDistance && distance < maxDistance))
    {
        return { maxDistance, false };
    }

    const float3 offset = fma(rayDirection, float3(distance), rayOrigin - corner);
    const float projectedX = dot(offset, edgeX);
    const float projectedY = dot(offset, edgeY);
    const float3 inverseGram = float3(light.points[2]);
    const float u = fma(inverseGram.x, projectedX, inverseGram.y * projectedY);
    const float v = fma(inverseGram.y, projectedX, inverseGram.z * projectedY);
    const bool inside = u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f;
    return { inside ? distance : maxDistance, inside };
}

static float calcLightAreaPdf(device const UniformLight& l, const float3 hitPoint)
{
    float areaPdf = 0.0f;
    switch (l.type)
    {
    case LIGHT_TYPE_RECT: // rectangle area
    {
        // Precomputed where the light is packed; see Scene::setLight.
        areaPdf = l.pad0;
        break;
    }
    case LIGHT_TYPE_DISC: {
        areaPdf = l.pad0;
        break;
    }
    case LIGHT_TYPE_SPHERE: {
        float3 normal;
        areaPdf = analyticEllipsoidAreaPdfUnchecked(
            float3(l.points[1]), float3(l.points[0]), float3(l.points[2]), float3(l.points[3]), hitPoint, normal);
        break;
    }
    case LIGHT_TYPE_POINT:
    case LIGHT_TYPE_SPOT:
    case LIGHT_TYPE_PROJECTOR: {
        // A point, spot or projector with a radius is sampled as a sphere and
        // needs the same area its density divides by. A sharp one has none and
        // never reaches a density that reads this.
        if (punctualLightIsSoft(l.points[0].x))
        {
            areaPdf = sphereLightAreaPdf(l.points[0].x);
        }
        break;
    }
    }
    return areaPdf;
}

static float3 calcLightNormal(device const UniformLight& l, thread const float3 hitPoint)
{
    float3 norm = float3(0.0f);
    switch (l.type)
    {
    case LIGHT_TYPE_RECT: {
        norm = float3(l.normal);
        break;
    }
    case LIGHT_TYPE_DISC: {
        norm = float3(l.normal);
        break;
    }
    case LIGHT_TYPE_SPHERE: {
        analyticEllipsoidAreaPdfUnchecked(
            float3(l.points[1]), float3(l.points[0]), float3(l.points[2]), float3(l.points[3]), hitPoint, norm);
        break;
    }
    case LIGHT_TYPE_POINT:
    case LIGHT_TYPE_SPOT:
    case LIGHT_TYPE_PROJECTOR: {
        // points[1] is the centre for all of them.
        norm = normalize(hitPoint - float3(l.points[1]));
        break;
    }
    }
    return norm;
}

static void fillLightData(device const UniformLight& l,
                          thread const float3 hitPoint,
                          thread LightSampleData& lightSampleData)
{
    const float3 toLight = lightSampleData.pointOnLight - hitPoint;
    float lenToLight;
    lightSampleData.L = finiteDirectionAndDistance(toLight, lenToLight);
    lightSampleData.distToLight = lenToLight;
    if (l.type == LIGHT_TYPE_SPHERE)
    {
        const AnalyticLightIntersection hit =
            intersectAnalyticEllipsoidUnchecked(hitPoint, lightSampleData.L, 0.0f, 3.402823466e38f, float3(l.points[1]),
                                                float3(l.points[0]), float3(l.points[2]), float3(l.points[3]));
        lightSampleData.areaPdf = hit.areaPdf;
        lightSampleData.normal = hit.normal;
        return;
    }
    lightSampleData.areaPdf = calcLightAreaPdf(l, lightSampleData.pointOnLight);
    lightSampleData.normal = calcLightNormal(l, lightSampleData.pointOnLight);
}

static __inline__ void fillRectLightData(device const UniformLight& l,
                                         thread const float3 hitPoint,
                                         thread LightSampleData& lightSampleData)
{
    const float3 toLight = lightSampleData.pointOnLight - hitPoint;
    const float distance2 = dot(toLight, toLight);
    const float inverseDistance = rsqrt(max(distance2, 1e-20f));
    lightSampleData.L = toLight * inverseDistance;
    lightSampleData.distToLight = distance2 * inverseDistance;
    lightSampleData.areaPdf = l.pad0;
    lightSampleData.normal = float3(l.normal);
}

static __inline__ void fillSelectedRectLightData(device const UniformLight& l,
                                                 thread const float3 hitPoint,
                                                 thread LightSampleData& lightSampleData)
{
    if (SPEC_FAST_RECT_LIGHT_DATA)
    {
        fillRectLightData(l, hitPoint, lightSampleData);
    }
    else
    {
        fillLightData(l, hitPoint, lightSampleData);
    }
}

// The spherical-rectangle frame, its sample and its solid angle all come from
// common/rect_sampling.h, which the OptiX modules and the host tests compile as
// well. These wrappers only unpack UniformLight's four corners.
static __inline__ SphQuad initSphQuad(device const UniformLight& l, const float3 o)
{
    const float3 p0 = float3(l.points[0]);
    return sphQuadInit(p0, float3(l.points[1]) - p0, float3(l.points[3]) - p0, o);
}

static __inline__ float3 sampleSphQuad(thread const SphQuad& squad, const float2 uv)
{
    return sphQuadSample(squad, uv.x, uv.y);
}

// The MIS path needs only 1/S; sphQuadSolidAngle() leaves the sample basis and
// the inverse-CDF constants out of that path.
static __inline__ float rectSolidAngle(device const UniformLight& l, const float3 o, thread bool& useAreaFallback)
{
    const float3 p0 = float3(l.points[0]);
    return sphQuadSolidAngle(p0, float3(l.points[1]) - p0, float3(l.points[3]) - p0, o, useAreaFallback);
}

static LightSampleData SampleRectLight(device const UniformLight& l, thread const float2 u, thread const float3 hitPoint)
{
    LightSampleData lightSampleData;
    const float3 ex = float3(l.points[1]) - float3(l.points[0]);
    const float3 ey = float3(l.points[3]) - float3(l.points[0]);
    const SphQuad quad = initSphQuad(l, hitPoint);

    if (quad.S <= 0.0f)
    {
        return {};
    }
    if (quad.useAreaFallback)
    {
        // Light too small / too grazing for float32 SphQuad — area sampling is
        // indistinguishable and numerically safe.
        lightSampleData.pointOnLight = float3(l.points[0]) + ex * u.x + ey * u.y;
        fillSelectedRectLightData(l, hitPoint, lightSampleData);
        lightSampleData.solidAngle = 0.0f;
        return lightSampleData;
    }

    lightSampleData.pointOnLight = sampleSphQuad(quad, u);
    fillSelectedRectLightData(l, hitPoint, lightSampleData);
    lightSampleData.solidAngle = quad.S;
    return lightSampleData;
}

static __inline__ LightSampleData SampleRectLightUniform(device const UniformLight& l,
                                                         thread const float2 u,
                                                         thread const float3 hitPoint)
{
    LightSampleData lightSampleData;
    // uniform sampling
    float3 e1 = float3(l.points[1]) - float3(l.points[0]);
    float3 e2 = float3(l.points[3]) - float3(l.points[0]);
    lightSampleData.pointOnLight = float3(l.points[0]) + e1 * u.x + e2 * u.y;
    fillSelectedRectLightData(l, hitPoint, lightSampleData);
    lightSampleData.solidAngle = 0.0f;
    return lightSampleData;
}

// Area-to-solid-angle pdf of a point sampled uniformly on a flat light. Both the
// rectangle and the disc reach it: calcLightAreaPdf() and calcLightNormal() already
// know the shape, so nothing here is specific to one.
static __inline__ LightSampleData SampleDistantLight(device const UniformLight& l,
                                                     const float2 u,
                                                     const uint2 retryWords,
                                                     const float3 hitPoint)
{
    LightSampleData lightSampleData;
    const float3 axis = -float3(l.normal);
    float3 coneSample;
    if (distantLightIsDelta(l.halfAngle))
    {
        coneSample = axis;
    }
    else
    {
        coneSample = sampleDistantLightDirection(u.x, u.y, retryWords.x, retryWords.y, l.halfAngle, axis);
    }

    lightSampleData.areaPdf = 0.0f;
    lightSampleData.distToLight = infiniteLightDistance();
    lightSampleData.L = coneSample;
    lightSampleData.normal = float3(l.normal);
    lightSampleData.solidAngle = 0.0f;
    lightSampleData.pointOnLight = coneSample;

    return lightSampleData;
}

// Uniform on the analytically transformed unit disc.
static __inline__ LightSampleData SampleDiscLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    const float3 center = float3(l.points[1]);
    const float2 objectPoint = sampleCanonicalDisc(u.x, u.y);
    lightSampleData.pointOnLight = center + objectPoint.x * float3(l.points[2]) + objectPoint.y * float3(l.points[3]);
    const float3 toLight = lightSampleData.pointOnLight - hitPoint;
    lightSampleData.L = finiteDirectionAndDistance(toLight, lightSampleData.distToLight);
    lightSampleData.normal = float3(l.normal);
    lightSampleData.areaPdf = l.pad0;
    lightSampleData.solidAngle = 0.0f;
    return lightSampleData;
}

/// A unit-sphere direction mapped analytically to the affine ellipsoid.
static __inline__ LightSampleData SampleSphereLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    const float3 center = float3(l.points[1]);
    const AnalyticLightSample sample = sampleAnalyticEllipsoidUnchecked(
        center, float3(l.points[0]), float3(l.points[2]), float3(l.points[3]), u.x, u.y);
    lightSampleData.pointOnLight = sample.point;
    const float3 toLight = sample.point - hitPoint;
    lightSampleData.L = finiteDirectionAndDistance(toLight, lightSampleData.distToLight);
    lightSampleData.normal = sample.normal;
    lightSampleData.areaPdf = sample.areaPdf;
    lightSampleData.solidAngle = 0.0f;

    return lightSampleData;
}

static __inline__ LightSampleData SampleDomeLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;

    lightSampleData.L = uniformSphereDirection(u.x, u.y);
    lightSampleData.distToLight = infiniteLightDistance();
    lightSampleData.areaPdf = 0.0f;
    // Faces the shading point by construction, so the caller's -dot(L, normal)
    // test passes for every sampled direction.
    lightSampleData.normal = -lightSampleData.L;
    lightSampleData.solidAngle = 0.0f;
    lightSampleData.pointOnLight = hitPoint + lightSampleData.L * lightSampleData.distToLight;

    return lightSampleData;
}

static __inline__ LightSampleData SamplePointLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
{
    const float radius = l.points[0].x;
    if (punctualLightIsSoft(radius))
    {
        LightSampleData lightSampleData;
        const float3 sphereDirection = uniformSphereDirection(u.x, u.y);
        const float3 lightPoint = float3(l.points[1]) + radius * sphereDirection;
        lightSampleData.pointOnLight = lightPoint;
        lightSampleData.L = finiteDirectionAndDistance(lightPoint - hitPoint, lightSampleData.distToLight);
        lightSampleData.normal = sphereDirection;
        lightSampleData.areaPdf = sphereLightAreaPdf(radius);
        lightSampleData.solidAngle = 0.0f;
        return lightSampleData;
    }

    LightSampleData lightSampleData;
    const float3 center = float3(l.points[1]);
    const float3 toLight = center - hitPoint;
    float dist;
    lightSampleData.pointOnLight = center;
    lightSampleData.L = finiteDirectionAndDistance(toLight, dist);
    lightSampleData.distToLight = dist;
    lightSampleData.normal = -lightSampleData.L;
    lightSampleData.areaPdf = 0.0f;
    lightSampleData.solidAngle = 0.0f;
    return lightSampleData;
}

static __inline__ float spotAttenuation(device const UniformLight& l, const float3 dirFromLight)
{
    // halfAngle = outer, pad0 = inner. Smoothstep between the two cosines.
    const float3 axis = normalize(float3(l.normal));
    const float cosOuter = cos(l.halfAngle);
    const float cosInner = cos(l.pad0);
    const float cosTheta = dot(axis, dirFromLight);
    if (cosTheta < cosOuter)
        return 0.0f;
    if (cosInner <= cosOuter)
        return 1.0f;
    return saturate((cosTheta - cosOuter) / (cosInner - cosOuter));
}

static __inline__ ProjectorSample projectorSampleForLight(device const UniformLight& l, const float3 dirFromLight)
{
    const OrthonormalLightFrame frame =
        makeOrthonormalLightFrame(float3(l.points[2]), float3(l.points[3]), float3(l.normal));
    const float tanX = projectorTanHalfX(l.halfAngle);
    const float tanY = projectorTanHalfY(tanX, l.points[0].w);
    if (!frame.valid)
    {
        return projectorProject(0.0f, 0.0f, -1.0f, tanX, tanY, l.pad0);
    }
    const float3 d = normalizeFiniteVectorOrZero(dirFromLight);
    return projectorProject(dot(d, frame.x), dot(d, frame.y), dot(d, frame.emissionAxis), tanX, tanY, l.pad0);
}

static __inline__ float3 projectorEmission(device const UniformLight& l, const float3 dirFromLight)
{
    const ProjectorSample p = projectorSampleForLight(l, dirFromLight);
    if (!p.inside)
    {
        return float3(0.0f);
    }
    if (is_null_texture(l.projectorTexture))
    {
        return float3(p.falloff);
    }
    // clamp_to_edge on both axes: a slide has an edge, and repeating it would
    // tile the wall with copies of the frame the moment a sample lands a texel
    // outside from the bilinear tap.
    constexpr sampler projectorSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge, coord::normalized);
    return p.falloff * l.projectorTexture.sample(projectorSampler, float2(p.u, p.v), level(0.0f)).rgb;
}

// Bilinear sample of an IES candela table. `dirFromLight` is world-space; the
// light's local frame is rebuilt from points[2..3] (X/Y axes) and normal (−Z),
// the same packing Scene::updateLight writes for the CPU sampler.
static __inline__ float sampleIesCandela(device const IesGpuBufferHeader* iesBuffer,
                                         device const UniformLight& l,
                                         const float3 dirFromLight)
{
    const int profileIdx = packedNonnegativeIndex(l.points[0].y);
    if (!iesBuffer || profileIdx < 0 || (uint32_t)profileIdx >= iesBuffer->profileCount)
    {
        return 1.0f;
    }

    device const IesGpuProfileHeader* headers =
        (device const IesGpuProfileHeader*)((device const char*)iesBuffer + sizeof(IesGpuBufferHeader));
    device const IesGpuProfileHeader& h = headers[profileIdx];
    if (h.nVertical < 2u || h.nHorizontal < 2u)
    {
        return 0.0f;
    }

    device const float* floats = (device const float*)((device const char*)iesBuffer + iesBuffer->floatOffset);

    // World -> light local. -Z is the photometric axis.
    const OrthonormalLightFrame frame =
        makeOrthonormalLightFrame(float3(l.points[2]), float3(l.points[3]), float3(l.normal));
    if (!frame.valid)
    {
        return 0.0f;
    }
    const float3 d = normalizeFiniteVectorOrZero(dirFromLight);
    const float3 local = float3(dot(d, frame.x), dot(d, frame.y), -dot(d, frame.emissionAxis));

    const float vertDeg = acos(clamp(-local.z, -1.0f, 1.0f)) * (180.0f / M_PI_F);
    const float horizDeg = atan2(local.x, -local.y) * (180.0f / M_PI_F);

    // The same evaluation the host and OptiX run -- see common/ies_math.h.
    return iesEvaluate(floats + h.anglesOffset, (int)h.nVertical, floats + h.anglesOffset + h.nVertical,
                       (int)h.nHorizontal, floats + h.candelaOffset, vertDeg, horizDeg);
}

static __inline__ float rangeWindow(device const UniformLight& l, float dist)
{
    // KHR_lights_punctual range window: 1 at d=0, 0 at d=range.
    if (l.pad1 <= 0.0f)
        return 1.0f;
    const float x = saturate(dist / l.pad1);
    const float y = 1.0f - x * x * x * x;
    return y * y;
}

static __inline__ float areaFalloff(device const UniformLight& l, float dist, int lightType)
{
    if (l.pad1 <= 0.0f)
    {
        return 1.0f;
    }
    if (lightType != LIGHT_TYPE_RECT && lightType != LIGHT_TYPE_DISC && lightType != LIGHT_TYPE_SPHERE)
    {
        return 1.0f;
    }
    const float x = saturate(dist / l.pad1);
    // 1 - smootherstep(x), the Map Range (SMOOTHERSTEP) node feeding the Mix.
    const float s = x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
    return 1.0f - s;
}

static __inline__ float areaFalloff(device const UniformLight& l, float dist)
{
    return areaFalloff(l, dist, l.type);
}

static __inline__ LightPdfQuery buildLightPdfQuery(device const UniformLight& l,
                                                   thread const LightSampleData& d,
                                                   int lightType)
{
    LightPdfQuery q = makeLightPdfQuery(lightType);
    q.distToLight = d.distToLight;
    q.cosAtLight = -dot(d.L, d.normal);
    q.areaPdf = d.areaPdf;
    q.halfAngle = l.halfAngle;
    q.solidAngle = d.solidAngle;
    if (lightIsPunctual(lightType))
    {
        q.radius = l.points[0].x;
    }
    return q;
}

static __inline__ LightPdfQuery buildLightPdfQuery(device const UniformLight& l, thread const LightSampleData& d)
{
    return buildLightPdfQuery(l, d, l.type);
}

static __inline__ float getLightPdf(device const UniformLight& l,
                                    const float3 lightHitPoint,
                                    const float3 surfaceHitPoint,
                                    uint32_t rectLightSamplingMethod,
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
        // Decided by the same predicate and the same rectSolidAngle() call the
        // sampler uses. Ask it differently here and the two halves weigh against
        // pdfs neither of them drew from.
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
