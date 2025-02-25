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

static float calcLightArea(device const UniformLight& l)
{
    float area = 0.0f;

    if (l.type == 0) // rectangle area
    {
        float3 e1 = float3(l.points[1]) - float3(l.points[0]);
        float3 e2 = float3(l.points[3]) - float3(l.points[0]);
        area = length(cross(e1, e2));
    }
    else if (l.type == 1) // disc area
    {
        area = l.points[0].x * l.points[0].x * M_PI_F; // pi * radius^2
    }
    else if (l.type == 2) // sphere area
    {
        area = l.points[0].x * l.points[0].x * 4.0f * M_PI_F; // 4 * pi * radius^2
    }
    return area;
}

static float3 calcLightNormal(device const UniformLight& l, thread const float3& hitPoint)
{
    float3 norm = float3(0.0f);

    if (l.type == 0)
    {
        float3 e1 = float3(l.points[1]) - float3(l.points[0]);
        float3 e2 = float3(l.points[3]) - float3(l.points[0]);

        norm = -normalize(cross(e1, e2));
    }
    else if (l.type == 1)
    {
        norm = float3(l.normal);
    }
    else if (l.type == 2)
    {
        norm = normalize(hitPoint - float3(l.points[1]));
    }
    return norm;
}

static void fillLightData(device const UniformLight& l, thread const float3& hitPoint, thread LightSampleData& lightSampleData)
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
    float z0, z0sq;
    float x0, y0, y0sq; // rectangle coords in ’R’
    float x1, y1, y1sq;
    float b0, b1, b0sq, k;
    float S;
};

// Precomputation of constants for the spherical rectangle Q.
static SphQuad init(device const UniformLight& l, const float3 o)
{
    SphQuad squad;

    float3 ex = float3(l.points[1]) - float3(l.points[0]);
    float3 ey = float3(l.points[3]) - float3(l.points[0]);

    float3 s = float3(l.points[0]);

    float exl = length(ex);
    float eyl = length(ey);

    squad.o = o;
    squad.x = ex / exl;
    squad.y = ey / eyl;
    squad.z = cross(squad.x, squad.y);

    // compute rectangle coords in local reference system
    float3 d = s - o;
    squad.z0 = dot(d, squad.z);

    // flip ’z’ to make it point against ’Q’
    if (squad.z0 > 0.0f)
    {
        squad.z *= -1.0f;
        squad.z0 *= -1.0f;
    }

    squad.z0sq = squad.z0 * squad.z0;
    squad.x0 = dot(d, squad.x);
    squad.y0 = dot(d, squad.y);
    squad.x1 = squad.x0 + exl;
    squad.y1 = squad.y0 + eyl;
    squad.y0sq = squad.y0 * squad.y0;
    squad.y1sq = squad.y1 * squad.y1;

    // create vectors to four vertices
    float3 v00 = { squad.x0, squad.y0, squad.z0 };
    float3 v01 = { squad.x0, squad.y1, squad.z0 };
    float3 v10 = { squad.x1, squad.y0, squad.z0 };
    float3 v11 = { squad.x1, squad.y1, squad.z0 };

    // compute normals to edges
    float3 n0 = normalize(cross(v00, v10));
    float3 n1 = normalize(cross(v10, v11));
    float3 n2 = normalize(cross(v11, v01));
    float3 n3 = normalize(cross(v01, v00));

    // compute internal angles (gamma_i)
    float g0 = fast::acos(-dot(n0, n1));
    float g1 = fast::acos(-dot(n1, n2));
    float g2 = fast::acos(-dot(n2, n3));
    float g3 = fast::acos(-dot(n3, n0));

    // compute predefined constants
    squad.b0 = n0.z;
    squad.b1 = n2.z;
    squad.b0sq = squad.b0 * squad.b0;
    const float twoPi = 2.0f * M_PI_F;
    squad.k = twoPi - g2 - g3;

    // compute solid angle from internal angles
    squad.S = g0 + g1 - squad.k;

    return squad;
}

static float3 SphQuadSample(const SphQuad squad, const float2 uv)
{
    float u = uv.x;
    float v = uv.y;

    // 1. compute cu
    float au = u * squad.S + squad.k;
    float fu = (cos(au) * squad.b0 - squad.b1) / sin(au);
    float cu = 1 / sqrt(fu * fu + squad.b0sq) * (fu > 0 ? 1 : -1);
    cu = clamp(cu, -1.0f, 1.0f); // avoid NaNs

    // 2. compute xu
    float xu = -(cu * squad.z0) / sqrt(1 - cu * cu);
    xu = clamp(xu, squad.x0, squad.x1); // avoid Infs

    // 3. compute yv
    float d = sqrt(xu * xu + squad.z0sq);
    float h0 = squad.y0 / sqrt(d * d + squad.y0sq);
    float h1 = squad.y1 / sqrt(d * d + squad.y1sq);
    float hv = h0 + v * (h1 - h0);
    float hv2 = hv * hv;
    float eps = 1e-6f;
    float yv = (hv2 < 1.0f - eps) ? (hv * d) / sqrt(1.0f - hv2) : squad.y1;

    // 4. transform (xu, yv, z0) to world coords
    return (squad.o + xu * squad.x + yv * squad.y + squad.z0 * squad.z);
}

static LightSampleData SampleRectLight(device const UniformLight& l, thread const float2& u, thread const float3& hitPoint)
{
    LightSampleData lightSampleData;
    float3 e1 = float3(l.points[1]) - float3(l.points[0]);
    float3 e2 = float3(l.points[3]) - float3(l.points[0]);

    // https://www.arnoldrenderer.com/research/egsr2013_spherical_rectangle.pdf
    SphQuad quad = init(l, hitPoint);
    if (quad.S <= 0.0f)
    {
        lightSampleData.pdf = 0.0f;
        lightSampleData.pointOnLight = float3(l.points[0]) + e1 * u.x + e2 * u.y;
        fillLightData(l, hitPoint, lightSampleData);
        return lightSampleData;
    }
    if (quad.S < 1e-3f)
    {
        // light too small, use uniform
        lightSampleData.pointOnLight = float3(l.points[0]) + e1 * u.x + e2 * u.y;
        fillLightData(l, hitPoint, lightSampleData);
        lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                              (dot(-lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
        return lightSampleData;
    }
    lightSampleData.pointOnLight = SphQuadSample(quad, u);
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = max(0.0f, 1.0f / quad.S);
    return lightSampleData;
}

static __inline__ LightSampleData SampleRectLightUniform(device const UniformLight& l, thread const float2& u, thread const float3& hitPoint)
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

static __inline__ float getRectLightPdf(device const UniformLight& l, const float3 lightHitPoint, const float3 surfaceHitPoint)
{
    LightSampleData lightSampleData {};
    lightSampleData.pointOnLight = lightHitPoint;
    fillLightData(l, surfaceHitPoint, lightSampleData);
    lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                            (dot(-lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
    return lightSampleData.pdf;
}

static void createCoordinateSystem(thread const float3& N, thread float3& Nt, thread float3& Nb) {
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
    return 1.0f / (2.0f * M_PI_F * (1.0f - cos(angle)));
}

static __inline__ float getSphereLightPdf() 
{ 
    return 1.0f / (4.0f * M_PI_F); 
} 

static float3 SampleCone(float2 uv, float angle, float3 direction, thread float& pdf) {

    float phi = 2.0 * M_PI_F * uv.x;
    float cosTheta = 1.0 - uv.y * (1.0 - cos(angle));

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

static __inline__ float getLightPdf(device const UniformLight& l, const float3 lightHitPoint, const float3 surfaceHitPoint)
{
    switch (l.type)
    {
    case 0:
        // Rect
        return getRectLightPdf(l, lightHitPoint, surfaceHitPoint);
        break;
    case 2:
        // sphere
        return getSphereLightPdf();
        break;
    case 3:
        // Distant
        return getDirectLightPdf(l.halfAngle);
        break;
    default:
        break;
    }
    return 0.0f;
}
