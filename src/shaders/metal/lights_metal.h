#pragma once
#include <metal_stdlib>
#include <simd/simd.h>

#include "ShaderTypes.h"

using namespace metal;

struct LightSampleData
{
    float3 pointOnLight;
    float pdf;

    float3 normal;
    float area;

    float3 L;
    float distToLight;
};

__inline__ float misWeightBalance(const float a, const float b)
{
    return 1.0f / ( 1.0f + (b / a) );
}

__inline__ float misWeightPower(const float a, const float b)
{
    const float a2 = a * a;
    return a2 / (a2 + b * b);
}

static float calcLightArea(device const UniformLight& l)
{
    float area = 0.0f;
    switch (l.type)
    {
    case 0: // rectangle area
    {
        float3 e1 = float3(l.points[1]) - float3(l.points[0]);
        float3 e2 = float3(l.points[3]) - float3(l.points[0]);
        area = length(cross(e1, e2));
        break;
    }
    case 1: // disc area
    {
        area = l.points[0].x * l.points[0].x * M_PI_F; // pi * radius^2
        break;
    }
    case 2: // sphere area
    {
        area = l.points[0].x * l.points[0].x * 4.0f * M_PI_F; // 4 * pi * radius^2
        break;
    }
    }
    return area;
}

static float3 calcLightNormal(device const UniformLight& l, thread const float3 hitPoint)
{
    float3 norm = float3(0.0f);
    switch (l.type)
    {
    case 0: {
        float3 e1 = float3(l.points[1]) - float3(l.points[0]);
        float3 e2 = float3(l.points[3]) - float3(l.points[0]);
        norm = -normalize(cross(e1, e2));
        break;
    }
    case 1: {
        norm = float3(l.normal);
        break;
    }
    case 2: {
        norm = normalize(hitPoint - float3(l.points[1]));
        break;
    }
    }
    return norm;
}

static void fillLightData(device const UniformLight& l, thread const float3 hitPoint, thread LightSampleData& lightSampleData)
{
    lightSampleData.area = calcLightArea(l);
    lightSampleData.normal = calcLightNormal(l, hitPoint);
    const float3 toLight = lightSampleData.pointOnLight - hitPoint;
    const float lenToLight = length(toLight);
    lightSampleData.L = toLight / lenToLight;
    lightSampleData.distToLight = lenToLight;
}

struct SphQuad
{
    float3 o, x, y, z;
    float z0;
    float x0, y0;
    float x1, y1;
    float b0, b1, b0sq;
    float g2, g3;
    float S;
    bool useAreaFallback;
};

// Ureña / Fajardo / King, EGSR 2013, in the numerically stable asin form used by
// Cycles.
static __inline__ SphQuad initSphQuad(device const UniformLight& l, const float3 o)
{
    SphQuad squad;
    const float3 ex = float3(l.points[1]) - float3(l.points[0]);
    const float3 ey = float3(l.points[3]) - float3(l.points[0]);
    const float exl = length(ex);
    const float eyl = length(ey);
    squad.o = o;
    squad.x = ex / exl;
    squad.y = ey / eyl;
    squad.z = cross(squad.x, squad.y);
    const float3 d = float3(l.points[0]) - o;
    squad.z0 = dot(d, squad.z);
    if (squad.z0 > 0.0f)
    {
        squad.z = -squad.z;
        squad.z0 = -squad.z0;
    }

    squad.x0 = dot(d, squad.x);
    squad.y0 = dot(d, squad.y);
    squad.x1 = squad.x0 + exl;
    squad.y1 = squad.y0 + eyl;
    float4 nz = float4(-squad.y0, squad.x1, squad.y1, -squad.x0);
    nz /= sqrt(nz * nz + squad.z0 * squad.z0);

    const float g0 = asin(clamp(-nz.x * nz.y, -1.0f, 1.0f));
    const float g1 = asin(clamp(-nz.y * nz.z, -1.0f, 1.0f));
    squad.g2 = asin(clamp(-nz.z * nz.w, -1.0f, 1.0f));
    squad.g3 = asin(clamp(-nz.w * nz.x, -1.0f, 1.0f));
    squad.S = -(g0 + g1 + squad.g2 + squad.g3);
    squad.b0 = nz.x;
    squad.b1 = nz.z;
    squad.b0sq = squad.b0 * squad.b0;
    const float nzMinSq = min(min(nz.x * nz.x, nz.y * nz.y), min(nz.z * nz.z, nz.w * nz.w));
    squad.useAreaFallback = (squad.S < 1e-5f) || (nzMinSq > 0.99999f);
    return squad;
}

static __inline__ float3 sampleSphQuad(const SphQuad squad, const float2 uv)
{
    const float au = uv.x * squad.S + squad.g2 + squad.g3;
    const float sinAu = sin(au);
    const float fu = (abs(sinAu) > 1e-8f) ? (cos(au) * squad.b0 + squad.b1) / sinAu : 0.0f;
    float cu = copysign(1.0f / sqrt(fu * fu + squad.b0sq), fu);
    cu = clamp(cu, -1.0f, 1.0f);

    float xu = -(cu * squad.z0) / max(sqrt(1.0f - cu * cu), 1e-7f);
    xu = clamp(xu, squad.x0, squad.x1);
    const float d2 = xu * xu + squad.z0 * squad.z0;
    const float h0 = squad.y0 / sqrt(d2 + squad.y0 * squad.y0);
    const float h1 = squad.y1 / sqrt(d2 + squad.y1 * squad.y1);
    const float hv = h0 + uv.y * (h1 - h0);
    const float hv2 = hv * hv;
    const float yv = (hv2 < 1.0f - 1e-6f) ? hv * sqrt(d2 / (1.0f - hv2)) : squad.y1;
    return squad.o + xu * squad.x + yv * squad.y + squad.z0 * squad.z;
}

// The MIS path needs only 1/S. Avoid creating the sample basis and constants
// there; light hits are less frequent than NEE but still sit in the shade kernel.
static __inline__ float rectSolidAngle(device const UniformLight& l,
                                       const float3 o,
                                       thread bool& useAreaFallback)
{
    const float3 ex = float3(l.points[1]) - float3(l.points[0]);
    const float3 ey = float3(l.points[3]) - float3(l.points[0]);
    const float exl = length(ex);
    const float eyl = length(ey);
    const float3 x = ex / exl;
    const float3 y = ey / eyl;
    float3 z = cross(x, y);
    const float3 d = float3(l.points[0]) - o;
    float z0 = dot(d, z);
    if (z0 > 0.0f)
    {
        z0 = -z0;
    }

    const float x0 = dot(d, x);
    const float y0 = dot(d, y);
    float4 nz = float4(-y0, x0 + exl, y0 + eyl, -x0);
    nz /= sqrt(nz * nz + z0 * z0);
    const float g0 = asin(clamp(-nz.x * nz.y, -1.0f, 1.0f));
    const float g1 = asin(clamp(-nz.y * nz.z, -1.0f, 1.0f));
    const float g2 = asin(clamp(-nz.z * nz.w, -1.0f, 1.0f));
    const float g3 = asin(clamp(-nz.w * nz.x, -1.0f, 1.0f));
    const float S = -(g0 + g1 + g2 + g3);
    const float nzMinSq = min(min(nz.x * nz.x, nz.y * nz.y), min(nz.z * nz.z, nz.w * nz.w));
    useAreaFallback = (S < 1e-5f) || (nzMinSq > 0.99999f);
    return S;
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
        lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                              (dot(-lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
        return lightSampleData;
    }

    lightSampleData.pointOnLight = sampleSphQuad(quad, u);
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = 1.0f / quad.S;
    return lightSampleData;
}

static __inline__ LightSampleData SampleRectLightUniform(device const UniformLight& l, thread const float2 u, thread const float3 hitPoint)
{
    LightSampleData lightSampleData;
    // uniform sampling
    float3 e1 = float3(l.points[1]) - float3(l.points[0]);
    float3 e2 = float3(l.points[3]) - float3(l.points[0]);
    lightSampleData.pointOnLight = float3(l.points[0]) + e1 * u.x + e2 * u.y;
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                          (dot(-lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
    return lightSampleData;
}

// Area-to-solid-angle pdf of a point sampled uniformly on a flat light. Both the
// rectangle and the disc reach it: calcLightArea() and calcLightNormal() already
// know the shape, so nothing here is specific to one.
static __inline__ float getAreaLightPdf(device const UniformLight& l, const float3 lightHitPoint, const float3 surfaceHitPoint)
{
    LightSampleData lightSampleData {};
    lightSampleData.pointOnLight = lightHitPoint;
    fillLightData(l, surfaceHitPoint, lightSampleData);
    lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                            (dot(-lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
    return lightSampleData.pdf;
}

static void createCoordinateSystem(thread const float3 N, thread float3& Nt, thread float3& Nb) {
    if (fabs(N.x) > fabs(N.y)) {
        float invLen = 1.0f / sqrt(N.x * N.x + N.z * N.z);
        Nt = float3(-N.z * invLen, 0.0f, N.x * invLen);
    } else {
        float invLen = 1.0f / sqrt(N.y * N.y + N.z * N.z);
        Nt = float3(0.0f, N.z * invLen, -N.y * invLen);
    }
    Nb = cross(N, Nt);
}

static __inline__ float getDirectLightPdf(float angle)
{
    // 4pi sin^2(x/2), not 2pi (1 - cos x): see coneSolidAngle() in light_desc.h.
    // The CPU bakes the radiance as irradiance / solid angle and this divides it
    // back out, so the two have to be the same number, and at sun-sized angles
    // 1 - cos is not a number so much as a rounding artefact.
    const float s = sin(0.5f * angle);
    return 1.0f / (4.0f * M_PI_F * s * s);
}

static __inline__ float getSphereLightPdf() 
{ 
    return 1.0f / (4.0f * M_PI_F); 
} 

static float3 SampleCone(float2 uv, float angle, float3 direction, thread float& pdf) {

    float phi = 2.0 * M_PI_F * uv.x;
    const float halfSin = sin(0.5f * angle);
    float cosTheta = 1.0 - uv.y * (2.0f * halfSin * halfSin);

    // Convert spherical coordinates to 3D direction
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 u, v;
    createCoordinateSystem(direction, u, v);
    float3 sampledDir = normalize(cos(phi) * sinTheta * u + sin(phi) * sinTheta * v + cosTheta * direction);

    // Calculate the PDF for the sampled direction
    pdf = getDirectLightPdf(angle);
    return sampledDir;
}

static __inline__ LightSampleData SampleDistantLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    float pdf = 0.0f;
    float3 coneSample = SampleCone(u, l.halfAngle, -float3(l.normal), pdf);

    lightSampleData.area = 0.0f;
    lightSampleData.distToLight = 1e9;
    lightSampleData.L = coneSample;
    lightSampleData.normal = float3(l.normal);
    lightSampleData.pdf = pdf;
    lightSampleData.pointOnLight = coneSample;

    return lightSampleData;
}

// Uniform over the disc's area. points[0].x carries the radius and points[2..3]
// the in-plane axes; the axes are normalized here because whether the light's
// transform already scaled them depends on how the light was authored, while
// calcLightArea() reads the radius from points[0] either way.
static __inline__ LightSampleData SampleDiscLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;

    const float radius = l.points[0].x;
    const float3 center = float3(l.points[1]);
    const float3 axisX = normalize(float3(l.points[2]));
    const float3 axisY = normalize(float3(l.points[3]));

    // sqrt keeps the samples uniform per unit area rather than crowding the centre.
    const float r = radius * sqrt(u.x);
    const float phi = 2.0f * M_PI_F * u.y;
    lightSampleData.pointOnLight = center + r * (cos(phi) * axisX + sin(phi) * axisY);

    fillLightData(l, hitPoint, lightSampleData);
    const float cosAtLight = dot(-lightSampleData.L, lightSampleData.normal);
    lightSampleData.pdf = (cosAtLight > 0.0f && lightSampleData.area > 0.0f)
                              ? lightSampleData.distToLight * lightSampleData.distToLight /
                                    (cosAtLight * lightSampleData.area)
                              : 0.0f;
    return lightSampleData;
}

static __inline__ LightSampleData SampleSphereLight(device const UniformLight& l, const float2 u, const float3 hitPoint) 
{ 
    LightSampleData lightSampleData; 
 
    // Generate a random direction on the sphere using solid angle sampling 
    float cosTheta = 1.0f - 2.0f * u.x;  // cosTheta is uniformly distributed between [-1, 1] 
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta); 
    float phi = 2.0f * M_PI_F * u.y;  // phi is uniformly distributed between [0, 2*pi] 
     
    const float radius = l.points[0].x; 
 
    // Convert spherical coordinates to Cartesian coordinates 
    float3 sphereDirection = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta); 
    // Scale the direction by the radius of the sphere and move it to the light position 
    float3 lightPoint = float3(l.points[1]) + radius * sphereDirection; 
    // Calculate the direction from the hit point to the sampled point on the light 
    lightSampleData.L = normalize(lightPoint - hitPoint); 
     
    // Calculate the distance to the light 
    lightSampleData.distToLight = length(lightPoint - hitPoint); 
 
    lightSampleData.area = 0.0f; 
    lightSampleData.normal = sphereDirection; 
    lightSampleData.pdf = 1.0f / (4.0f * M_PI_F); 
    lightSampleData.pointOnLight = lightPoint; 
 
    return lightSampleData; 
}

// Point and spot lights store radiant intensity in colour. The contribution is
// I / r²; the inverse-square is applied in connectLight, not here. Soft radius
// (points[0].x > 0) falls back to sphere sampling so the light has a visible size.
static __inline__ LightSampleData SamplePointLight(device const UniformLight& l, const float2 u, const float3 hitPoint)
{
    const float radius = l.points[0].x;
    if (radius > 1e-4f)
    {
        return SampleSphereLight(l, u, hitPoint);
    }

    LightSampleData lightSampleData;
    const float3 center = float3(l.points[1]);
    const float3 toLight = center - hitPoint;
    const float dist = length(toLight);
    lightSampleData.pointOnLight = center;
    lightSampleData.L = toLight / max(dist, 1e-8f);
    lightSampleData.distToLight = dist;
    lightSampleData.normal = -lightSampleData.L;
    lightSampleData.area = 0.0f;
    // Delta light: the BSDF never hits it, so the NEE pdf is 1 in the measure
    // connectLight divides by.
    lightSampleData.pdf = 1.0f;
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

static __inline__ float rangeWindow(device const UniformLight& l, float dist)
{
    // KHR_lights_punctual range window: 1 at d=0, 0 at d=range.
    if (l.pad1 <= 0.0f)
        return 1.0f;
    const float x = saturate(dist / l.pad1);
    const float y = 1.0f - x * x * x * x;
    return y * y;
}

// PDF of the strategy NEE would have used for this light. For rectangles that
// depends on rectLightSamplingMethod: Advanced samples proportional to solid
// angle (pdf = 1/S), Uniform samples the area (pdf = r²/(cos θ A)). Using the
// wrong one here silently breaks MIS weights on BSDF hits of rect lights.
static __inline__ float getRectLightPdf(device const UniformLight& l,
                                        const float3 lightHitPoint,
                                        const float3 surfaceHitPoint,
                                        uint32_t rectLightSamplingMethod)
{
    if (rectLightSamplingMethod == 0)
    {
        return getAreaLightPdf(l, lightHitPoint, surfaceHitPoint);
    }
    bool useAreaFallback = false;
    const float S = rectSolidAngle(l, surfaceHitPoint, useAreaFallback);
    if (S <= 0.0f)
    {
        return 0.0f;
    }
    if (useAreaFallback)
    {
        return getAreaLightPdf(l, lightHitPoint, surfaceHitPoint);
    }
    return 1.0f / S;
}

static __inline__ float getLightPdf(device const UniformLight& l,
                                    const float3 lightHitPoint,
                                    const float3 surfaceHitPoint,
                                    uint32_t rectLightSamplingMethod)
{
    switch (l.type)
    {
    case 0:
        return getRectLightPdf(l, lightHitPoint, surfaceHitPoint, rectLightSamplingMethod);
    case 1:
        // Disc
        return getAreaLightPdf(l, lightHitPoint, surfaceHitPoint);
    case 2:
        // sphere
        return getSphereLightPdf();
    case 3:
        // Distant
        return getDirectLightPdf(l.halfAngle);
    case 5: // point
    case 6: // spot
        if (l.points[0].x > 1e-4f)
            return getSphereLightPdf();
        return 1.0f;
    default:
        break;
    }
    return 0.0f;
}
