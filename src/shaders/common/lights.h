#pragma once
#include <vector_types.h>
#include <sutil/vec_math.h>
#include <light_types.h>
// The densities and the MIS heuristics themselves; this file only unpacks
// UniformLight and hands them scalars. Metal's lights_metal.h includes the same
// header, and tests/render/test_light_pdf.cpp compiles it on the host.
#include <light_pdf.h>
#include <rect_sampling.h>
#include <ies_math.h>

// GPU side structure
// pad0: spot inner cone (rad) or point soft radius.
// pad1: KHR attenuation range (0 = infinite).
// points[0].y for point/spot: IES profile index, or -1 when isotropic.
struct UniformLight
{
    float4 points[4];
    float4 color;
    float4 normal;
    int type;
    float halfAngle;
    float pad0;
    float pad1;
};

// Packed IES candela tables for the GPU. OptiXRender::createIesBuffer lays the
// buffer out as:
//   IesGpuBufferHeader
//   IesGpuProfileHeader[profileCount]
//   float blob (angles then candela, offsets relative to the blob start)
// Sampled by sampleIesCandela() below; intensity on the light is a multiplier
// on top of the table. Field for field this is the Metal ShaderTypes.h pair, so
// a profile packed by either backend reads the same on the other.
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
    unsigned int anglesOffset;  // index into the float blob: vertical then horizontal
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
    float area;

    float3 L;
    float distToLight;
};

static __inline__ __device__ float calcLightArea(const UniformLight& l)
{
    float area = 0.0f;

    if (l.type == LIGHT_TYPE_RECT)
    {
        float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
        float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);
        area = length(cross(e1, e2));
    }
    else if (l.type == LIGHT_TYPE_DISC)
    {
        area = M_PIf * l.points[0].x * l.points[0].x; // pi * radius^2
    }
    else if (l.type == LIGHT_TYPE_SPHERE)
    {
        area = sphereLightArea(l.points[0].x);
    }
    else if (punctualLightIsSoft(l.points[0].x) &&
             (l.type == LIGHT_TYPE_POINT || l.type == LIGHT_TYPE_SPOT))
    {
        // A point or spot with a radius is sampled as a sphere, so it needs the
        // same area the sphere's density divides by. Without this case
        // fillLightData() left the area at zero and the density collapsed to the
        // constant it used to be.
        area = sphereLightArea(l.points[0].x);
    }
    return area;
}

static __inline__ __device__ float3 calcLightNormal(const UniformLight& l, const float3 hitPoint)
{
    float3 norm = make_float3(0.0f);

    if (l.type == LIGHT_TYPE_RECT)
    {
        float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
        float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);

        norm = -normalize(cross(e1, e2));
    }
    else if (l.type == LIGHT_TYPE_DISC)
    {
        norm = make_float3(l.normal);
    }
    else if (l.type == LIGHT_TYPE_SPHERE || l.type == LIGHT_TYPE_POINT || l.type == LIGHT_TYPE_SPOT)
    {
        // points[1] is the centre for all three. A soft point is a sphere and
        // needs a real surface normal for the area-to-solid-angle conversion; a
        // sharp one never reaches a density that uses it.
        norm = normalize(hitPoint - make_float3(l.points[1]));
    }
    return norm;
}

static __inline__ __device__ void fillLightData(const UniformLight& l, const float3 hitPoint, LightSampleData& lightSampleData)
{
    lightSampleData.area = calcLightArea(l);
    lightSampleData.normal = calcLightNormal(l, hitPoint);
    const float3 toLight = lightSampleData.pointOnLight - hitPoint;
    const float lenToLight = length(toLight);
    lightSampleData.L = toLight / lenToLight;
    lightSampleData.distToLight = lenToLight;
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
static __inline__ __device__ float rectSolidAngle(const UniformLight& l,
                                                  const float3 o,
                                                  bool& useAreaFallback)
{
    const float3 p0 = make_float3(l.points[0]);
    return sphQuadSolidAngle(p0, make_float3(l.points[1]) - p0, make_float3(l.points[3]) - p0, o,
                             useAreaFallback);
}

static __inline__ __device__ bool emitsLight(const float3 radiance)
{
    return radiance.x > 0.0f || radiance.y > 0.0f || radiance.z > 0.0f;
}

/// Unpack one light into the scalars lightSolidAnglePdf() needs.
///
/// `radius` is only read for the types that have one, because points[0] means
/// something different on a rect (a corner) than on a sphere (the radius).
static __inline__ __device__ LightPdfQuery buildLightPdfQuery(const UniformLight& l,
                                                              const LightSampleData& d)
{
    LightPdfQuery q = makeLightPdfQuery(l.type);
    q.distToLight = d.distToLight;
    q.cosAtLight = -dot(d.L, d.normal);
    q.area = d.area;
    q.halfAngle = l.halfAngle;
    if (l.type == LIGHT_TYPE_SPHERE || l.type == LIGHT_TYPE_POINT || l.type == LIGHT_TYPE_SPOT)
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
                                               unsigned int rectLightSamplingMethod = 0)
{
    LightSampleData d {};
    d.pointOnLight = lightHitPoint;
    fillLightData(l, surfaceHitPoint, d);

    LightPdfQuery q = buildLightPdfQuery(l, d);
    if (l.type == LIGHT_TYPE_RECT && rectLightSamplingMethod != 0)
    {
        // Which of the two rect densities applies is decided by the same
        // predicate the sampler uses, from the same rectSolidAngle() call. Ask
        // it differently here and the two halves weigh against pdfs neither of
        // them drew from.
        bool useAreaFallback = false;
        const float S = rectSolidAngle(l, surfaceHitPoint, useAreaFallback);
        if (S <= 0.0f)
        {
            return 0.0f;
        }
        q.solidAngle = useAreaFallback ? 0.0f : S;
    }
    return lightSolidAnglePdf(q);
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
        lightSampleData.pdf = areaLightSolidAnglePdf(lightSampleData.distToLight,
                                                     -dot(lightSampleData.L, lightSampleData.normal),
                                                     lightSampleData.area);
        return lightSampleData;
    }

    lightSampleData.pointOnLight = SphQuadSample(quad, u);
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = 1.0f / quad.S;

    return lightSampleData;
}

static __inline__ __device__ LightSampleData SampleRectLightUniform(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    // // uniform sampling
    float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
    float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);
    lightSampleData.pointOnLight = make_float3(l.points[0]) + e1 * u.x + e2 * u.y;
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = areaLightSolidAnglePdf(lightSampleData.distToLight,
                                                 -dot(lightSampleData.L, lightSampleData.normal),
                                                 lightSampleData.area);
    return lightSampleData;
}

static __device__ void createCoordinateSystem(const float3& N, float3& Nt, float3& Nb) {
    if (fabs(N.x) > fabs(N.y)) {
        float invLen = 1.0f / sqrt(N.x * N.x + N.z * N.z);
        Nt = make_float3(-N.z * invLen, 0.0f, N.x * invLen);
    } else {
        float invLen = 1.0f / sqrt(N.y * N.y + N.z * N.z);
        Nt = make_float3(0.0f, N.z * invLen, -N.y * invLen);
    }
    Nb = cross(N, Nt);
}

static __device__ float3 SampleCone(float2 uv, float angle, float3 direction, float& pdf) {

    float phi = 2.0 * M_PIf * uv.x;
    const float halfSin = sin(0.5f * angle);
    float cosTheta = 1.0 - uv.y * (2.0f * halfSin * halfSin);

    // Convert spherical coordinates to 3D direction
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 u, v;
    createCoordinateSystem(direction, u, v);
    float3 sampledDir = normalize(cos(phi) * sinTheta * u + sin(phi) * sinTheta * v + cosTheta * direction);

    // See coneLightSolidAnglePdf() for why this is not 1/(2pi(1 - cos a)).
    pdf = coneLightSolidAnglePdf(angle);
    return sampledDir;
}

static __inline__ __device__ LightSampleData SampleDistantLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    float pdf = 0.0f;
    float3 coneSample = SampleCone(u, l.halfAngle, -make_float3(l.normal), pdf);

    lightSampleData.area = 0.0f;
    lightSampleData.distToLight = 1e9;
    lightSampleData.L = coneSample;
    lightSampleData.normal = make_float3(l.normal);
    lightSampleData.pdf = pdf;
    lightSampleData.pointOnLight = coneSample;

    return lightSampleData;
}

// Uniform over the disc's area. points[0].x carries the radius and points[2..3]
// the in-plane axes; the axes are normalized here because whether the light's
// transform already scaled them depends on how the light was authored, while the
// area is computed from the radius in points[0] either way.
static __inline__ __device__ LightSampleData SampleDiscLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;

    const float radius = l.points[0].x;
    const float3 center = make_float3(l.points[1]);
    const float3 axisX = normalize(make_float3(l.points[2]));
    const float3 axisY = normalize(make_float3(l.points[3]));

    // sqrt keeps the samples uniform per unit area rather than crowding the centre.
    const float r = radius * sqrtf(u.x);
    const float phi = 2.0f * M_PIf * u.y;
    lightSampleData.pointOnLight = center + r * (cosf(phi) * axisX + sinf(phi) * axisY);

    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = areaLightSolidAnglePdf(lightSampleData.distToLight,
                                                 -dot(lightSampleData.L, lightSampleData.normal),
                                                 lightSampleData.area);
    return lightSampleData;
}

/// A point drawn uniformly over the surface of a sphere light.
///
/// The density is the area one converted to solid angle, not the constant
/// 1/(4pi) this used to report. Uniform-area sampling has p_A = 1/(4 pi r^2),
/// and turning that into a solid-angle density needs the d^2 / cos Jacobian like
/// any other area light -- the sampler picks a point, not a direction. Reporting
/// 1/(4pi) made the next-event estimator scale with (d / r)^2: measured against
/// the analytic irradiance of a uniformly emitting sphere it was 111x too bright
/// at r = 0.5, d = 4, and the error grows with distance. Both halves of the MIS
/// estimate used the same wrong number, so the weights still summed to one and
/// nothing looked inconsistent -- the light was simply, quietly, wrong.
static __inline__ __device__ LightSampleData SampleSphereLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;

    const float radius = l.points[0].x;
    const float3 sphereDirection = uniformSphereDirection(u.x, u.y);
    const float3 lightPoint = make_float3(l.points[1]) + radius * sphereDirection;

    lightSampleData.pointOnLight = lightPoint;
    lightSampleData.distToLight = length(lightPoint - hitPoint);
    lightSampleData.L = normalize(lightPoint - hitPoint);
    lightSampleData.normal = sphereDirection;
    lightSampleData.area = sphereLightArea(radius);
    lightSampleData.pdf = sphereLightSolidAnglePdf(
        lightSampleData.distToLight, -dot(lightSampleData.L, lightSampleData.normal), radius);

    return lightSampleData;
}

/// An infinitely distant, uniform-radiance dome.
///
/// Uniform over the whole sphere rather than the upper hemisphere: a dome is the
/// analytic form of an environment, and an environment lights a surface from
/// below as well as above once anything reflects. `color` is radiance, so there
/// is no distance falloff and no area -- the pdf is the constant 1/4pi, which is
/// what getLightPdf() returns for this type so that MIS against a BSDF ray that
/// misses the scene agrees with what was sampled here.
///
/// Without this case the switch in sampleLight fell through, leaving a
/// zero-initialised LightSampleData: direction (0,0,0), pdf 0. The facing test
/// then rejected it, so a dome light contributed exactly nothing and did so
/// silently -- no NaN, no red pixel, just an unlit scene.
static __inline__ __device__ LightSampleData SampleDomeLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;

    lightSampleData.L = uniformSphereDirection(u.x, u.y);
    lightSampleData.distToLight = 1e9f;
    lightSampleData.area = 0.0f;
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
        return SampleSphereLight(l, u, hitPoint);
    }

    LightSampleData lightSampleData;
    const float3 center = make_float3(l.points[1]);
    const float3 toLight = center - hitPoint;
    const float dist = length(toLight);
    lightSampleData.pointOnLight = center;
    lightSampleData.L = toLight / fmaxf(dist, 1e-8f);
    lightSampleData.distToLight = dist;
    lightSampleData.normal = -lightSampleData.L;
    lightSampleData.area = 0.0f;
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
    // clamp(), not saturate(): saturate lives in sutil/vec_math_adv.h, which is
    // written for nvcc and does not compile in a plain host translation unit --
    // and this header is reached from host code through OptixRenderParams.h.
    // clamp() comes from vec_math.h, which is already included above.
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


// Bilinear sample of an IES candela table. `dirFromLight` is world-space; the
// light's local frame is rebuilt from points[2..3] (X/Y axes) and normal (-Z),
// the same packing Scene::updateLight writes for the CPU sampler in
// iesloader.cpp. Returns 1.0 when the light carries no profile, so the caller
// can multiply unconditionally.
static __inline__ __device__ float sampleIesCandela(const IesGpuBufferHeader* iesBuffer,
                                                    const UniformLight& l,
                                                    const float3 dirFromLight)
{
    const int profileIdx = (int)l.points[0].y;
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
    const float3 ax = normalize(make_float3(l.points[2]));
    const float3 ay = normalize(make_float3(l.points[3]));
    const float3 az = normalize(make_float3(l.normal)); // emission -Z
    const float3 d = normalize(dirFromLight);
    const float3 local = make_float3(dot(d, ax), dot(d, ay), -dot(d, az));

    const float vertDeg = acosf(clamp(-local.z, -1.0f, 1.0f)) * (180.0f / M_PIf);
    const float horizDeg = atan2f(local.x, -local.y) * (180.0f / M_PIf);

    // The same evaluation the host and Metal run -- see common/ies_math.h.
    return iesEvaluate(floats + h.anglesOffset, (int)h.nVertical,
                       floats + h.anglesOffset + h.nVertical, (int)h.nHorizontal,
                       floats + h.candelaOffset, vertDeg, horizDeg);
}
