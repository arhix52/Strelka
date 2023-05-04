#pragma once
#include <metal_stdlib>
#include <simd/simd.h>

#include "ShaderTypes.h"

using namespace metal;

#define M_PIf 3.1415926f;

// struct UniformLight
// {
//     float4 points[4];
//     float4 color;
//     float4 normal;
//     int type;
//     float pad0;
//     float pad2;
//     float pad3;
// };

struct LightSampleData
{
    float3 pointOnLight;
    float pdf;

    float3 normal;
    float area;

    float3 L;
    float distToLight;
};

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
        area = l.points[0].x * l.points[0].x * M_PIf; // pi * radius^2
    }
    else if (l.type == 2) // sphere area
    {
        area = l.points[0].x * l.points[0].x * 4.0f * M_PIf; // 4 * pi * radius^2
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
    if (squad.z0 > 0)
    {
        squad.z *= -1;
        squad.z0 *= -1;
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
    float g0 = acos(-dot(n0, n1));
    float g1 = acos(-dot(n1, n2));
    float g2 = acos(-dot(n2, n3));
    float g3 = acos(-dot(n3, n0));

    // compute predefined constants
    squad.b0 = n0.z;
    squad.b1 = n2.z;
    squad.b0sq = squad.b0 * squad.b0;
    const float twoPi = 2.0f * M_PIf;
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
    float eps = 1e-5;
    float yv = (hv < 1 - eps) ? (hv * d) / sqrt(1 - hv2) : squad.y1;

    // 4. transform (xu, yv, z0) to world coords
    return (squad.o + xu * squad.x + yv * squad.y + squad.z0 * squad.z);
}

static LightSampleData SampleRectLight(device const UniformLight& l, thread const float2& u, thread const float3& hitPoint)
{
    LightSampleData lightSampleData;
    // https://www.arnoldrenderer.com/research/egsr2013_spherical_rectangle.pdf
    SphQuad quad = init(l, hitPoint);
    lightSampleData.pointOnLight = SphQuadSample(quad, u);
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                          (-dot(lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
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
