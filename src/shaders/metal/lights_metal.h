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

struct LightSampleData
{
    float3 pointOnLight;
    float pdf;

    float3 normal;
    float areaPdf;

    float3 L;
    float distToLight;
};

struct AnalyticAreaLightHit
{
    float distance;
    float3 point;
    float3 normal;
    float areaPdf;
    uint32_t lightId;
    bool hit;
};

static AnalyticAreaLightHit findAnalyticAreaLightHit(device const UniformLight* lights,
                                                     uint32_t lightCount,
                                                     float3 rayOrigin,
                                                     float3 rayDirection,
                                                     float minDistance,
                                                     float maxDistance,
                                                     bool includeCameraHidden)
{
    AnalyticAreaLightHit closest;
    closest.distance = maxDistance;
    closest.point = float3(0.0f);
    closest.normal = float3(0.0f);
    closest.areaPdf = 0.0f;
    closest.lightId = 0u;
    closest.hit = false;
    for (uint32_t lightId = 0u; lightId < lightCount; ++lightId)
    {
        device const UniformLight& light = lights[lightId];
        if (!analyticLightVisibilityAllowsRay(light.normal.w, includeCameraHidden))
        {
            continue;
        }
        const AnalyticLightIntersection candidate = intersectAnalyticLightSurface(
            light.type, float3(light.points[0]), float3(light.points[1]), float3(light.points[2]),
            float3(light.points[3]), float3(light.normal), rayOrigin, rayDirection, minDistance, closest.distance);
        if (candidate.hit)
        {
            closest.distance = candidate.distance;
            closest.point = candidate.point;
            closest.normal = candidate.normal;
            closest.areaPdf = candidate.areaPdf;
            closest.lightId = lightId;
            closest.hit = true;
        }
    }
    return closest;
}

static bool analyticLightsOccludeSegment(device const UniformLight* lights,
                                         uint32_t lightCount,
                                         float3 rayOrigin,
                                         float3 rayDirection,
                                         float minDistance,
                                         float maxDistance)
{
    for (uint32_t lightId = 0u; lightId < lightCount; ++lightId)
    {
        device const UniformLight& light = lights[lightId];
        if (analyticLightSurfaceOccludesSegment(light.type, float3(light.points[0]), float3(light.points[1]),
                                                float3(light.points[2]), float3(light.points[3]), float3(light.normal),
                                                light.normal.w, rayOrigin, rayDirection, minDistance, maxDistance))
        {
            return true;
        }
    }
    return false;
}

static float calcLightAreaPdf(device const UniformLight& l, const float3 hitPoint)
{
    float areaPdf = 0.0f;
    switch (l.type)
    {
    case LIGHT_TYPE_RECT: // rectangle area
    {
        float3 e1 = float3(l.points[1]) - float3(l.points[0]);
        float3 e2 = float3(l.points[3]) - float3(l.points[0]);
        areaPdf = inverseFiniteCrossLength(e1, e2);
        break;
    }
    case LIGHT_TYPE_DISC: {
        areaPdf = analyticDiscAreaPdf(float3(l.points[2]), float3(l.points[3]));
        break;
    }
    case LIGHT_TYPE_SPHERE: {
        float3 normal;
        areaPdf = analyticEllipsoidAreaPdf(
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
        analyticEllipsoidAreaPdf(
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
            intersectAnalyticEllipsoid(hitPoint, lightSampleData.L, 0.0f, 3.402823466e38f, float3(l.points[1]),
                                       float3(l.points[0]), float3(l.points[2]), float3(l.points[3]));
        lightSampleData.areaPdf = hit.areaPdf;
        lightSampleData.normal = hit.normal;
        return;
    }
    lightSampleData.areaPdf = calcLightAreaPdf(l, lightSampleData.pointOnLight);
    lightSampleData.normal = calcLightNormal(l, lightSampleData.pointOnLight);
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
        lightSampleData.pdf = 0.0f;
        lightSampleData.pointOnLight = float3(l.points[0]) + ex * u.x + ey * u.y;
        fillLightData(l, hitPoint, lightSampleData);
        return lightSampleData;
    }
    if (quad.useAreaFallback)
    {
        // Light too small / too grazing for float32 SphQuad — area sampling is
        // indistinguishable and numerically safe.
        lightSampleData.pointOnLight = float3(l.points[0]) + ex * u.x + ey * u.y;
        fillLightData(l, hitPoint, lightSampleData);
        lightSampleData.pdf = areaPdfToSolidAnglePdf(
            lightSampleData.distToLight, -dot(lightSampleData.L, lightSampleData.normal), lightSampleData.areaPdf);
        return lightSampleData;
    }

    lightSampleData.pointOnLight = sampleSphQuad(quad, u);
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = 1.0f / quad.S;
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
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = areaPdfToSolidAnglePdf(
        lightSampleData.distToLight, -dot(lightSampleData.L, lightSampleData.normal), lightSampleData.areaPdf);
    return lightSampleData;
}

// Area-to-solid-angle pdf of a point sampled uniformly on a flat light. Both the
// rectangle and the disc reach it: calcLightAreaPdf() and calcLightNormal() already
// know the shape, so nothing here is specific to one.
static __inline__ LightSampleData SampleDistantLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    float pdf = 0.0f;
    const float3 axis = -float3(l.normal);
    float3 coneSample;
    if (distantLightIsDelta(l.halfAngle))
    {
        coneSample = axis;
        pdf = deltaLightPdf();
    }
    else
    {
        coneSample = sampleDistantLightDirection(u.x, u.y, l.halfAngle, axis);
        pdf = coneLightSolidAnglePdf(l.halfAngle);
    }

    lightSampleData.areaPdf = 0.0f;
    lightSampleData.distToLight = infiniteLightDistance();
    lightSampleData.L = coneSample;
    lightSampleData.normal = float3(l.normal);
    lightSampleData.pdf = pdf;
    lightSampleData.pointOnLight = coneSample;

    return lightSampleData;
}

// Uniform on the analytically transformed unit disc.
static __inline__ LightSampleData SampleDiscLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    const float3 center = float3(l.points[1]);
    const AnalyticLightSample sample =
        sampleAnalyticDisc(center, float3(l.points[2]), float3(l.points[3]), float3(l.normal), u.x, u.y);
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
static __inline__ LightSampleData SampleSphereLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    const float3 center = float3(l.points[1]);
    const AnalyticLightSample sample =
        sampleAnalyticEllipsoid(center, float3(l.points[0]), float3(l.points[2]), float3(l.points[3]), u.x, u.y);
    lightSampleData.pointOnLight = sample.point;
    const float3 toLight = sample.point - hitPoint;
    lightSampleData.L = finiteDirectionAndDistance(toLight, lightSampleData.distToLight);
    lightSampleData.normal = sample.normal;
    lightSampleData.areaPdf = sample.areaPdf;
    lightSampleData.pdf = areaPdfToSolidAnglePdf(
        lightSampleData.distToLight, -dot(lightSampleData.L, lightSampleData.normal), lightSampleData.areaPdf);

    return lightSampleData;
}

/// An infinitely distant, uniform-radiance dome.
///
/// Uniform over the whole sphere rather than the upper hemisphere: a dome is the
/// analytic form of an environment, and an environment lights a surface from
/// below as well as above once anything reflects. `color` is radiance, so there
/// is no distance falloff and no area.
static __inline__ LightSampleData SampleDomeLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
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


// Point and spot lights store radiant intensity in colour. connectLight turns
// that into what the surface receives, differently for the two cases -- see the
// note there. A soft radius falls back to sphere sampling so the light has a
// visible size and a penumbra.
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
        lightSampleData.pdf = sphereLightSolidAnglePdf(
            lightSampleData.distToLight, -dot(lightSampleData.L, lightSampleData.normal), radius);
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
    // Delta light: the BSDF never hits it, so the NEE pdf is 1 in the measure
    // connectLight divides by.
    lightSampleData.pdf = deltaLightPdf();
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


/// Where a direction leaving a projector lands on the image it throws.
///
/// The light's own frame is rebuilt from points[2..3] and normal, exactly as
/// sampleIesCandela does below -- a projector is the same lamp with a different
/// angular profile, and it inherits that packing rather than a second one.
/// halfAngle is half the horizontal field of view, points[0].w the frame's
/// aspect, pad0 the edge feather.
///
/// The fetch is the caller's: on this backend the image handle rides in the
/// light struct itself (see ShaderTypes.h), because the shade kernel has no
/// binding slot left for a table -- all 31 are spoken for.
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
    return projectorProject(
        dot(d, frame.x), dot(d, frame.y), dot(d, frame.emissionAxis), tanX, tanY, l.pad0);
}

/// What a projector emits in a direction, as a multiplier on its intensity.
///
/// The image, faded at the frame's edge, and black outside the frame -- so the
/// caller can multiply unconditionally the way it does with an IES table. A
/// projector with no image throws a plain white rectangle, which is a usable
/// light in its own right and is what an unresolved path degrades to rather than
/// darkness.
///
/// Level 0, deliberately. A compute kernel has no derivatives, and the footprint
/// that would set the level here is the *camera's* on the receiving surface, not
/// anything this function can see -- the same reason the material fetches take
/// an explicitly computed level. Sampling the top level makes a projector as
/// sharp as its image and leaves minification aliasing to be resolved by the
/// pixel samples, which converge on the right answer because they are jittered
/// across the pixel.
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

// Blender's "controlled falloff" for area lights: a Light Path "Ray Length"
// divided by a cutoff distance, run through a smootherstep Map Range, fades the
// emission out with a Mix Shader. pad1 carries that cutoff distance (0 = none),
// and only area lights read it -- punctual lights use pad1 as the KHR range in
// rangeWindow() above, so scaling them here as well would apply two windows.
static __inline__ float areaFalloff(device const UniformLight& l, float dist)
{
    if (l.pad1 <= 0.0f)
    {
        return 1.0f;
    }
    if (l.type != LIGHT_TYPE_RECT && l.type != LIGHT_TYPE_DISC && l.type != LIGHT_TYPE_SPHERE)
    {
        return 1.0f;
    }
    const float x = saturate(dist / l.pad1);
    // 1 - smootherstep(x), the Map Range (SMOOTHERSTEP) node feeding the Mix.
    const float s = x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
    return 1.0f - s;
}

/// Unpack one light into the scalars lightSolidAnglePdf() needs.
///
/// `radius` is only read for punctual types; every area light carries its local
/// world-area density in `d.areaPdf`.
static __inline__ LightPdfQuery buildLightPdfQuery(device const UniformLight& l, thread const LightSampleData& d)
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
/// weighs itself against, and it has to be the one the samplers above drew from.
///
/// For rectangles that depends on rectLightSamplingMethod: solid-angle sampling
/// gives 1/S, area sampling gives d^2/(cos A). Answering with the wrong one
/// silently breaks the MIS weight on every BSDF hit of a rect light.
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
