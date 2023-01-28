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

static LightSampleData SampleRectLight(device const UniformLight& l, thread const float2& u, thread const float3& hitPoint)
{
    LightSampleData lightSampleData;
    // uniform sampling
    float3 e1 = float3(l.points[1]) - float3(l.points[0]);
    float3 e2 = float3(l.points[3]) - float3(l.points[0]);
    lightSampleData.pointOnLight = float3(l.points[0]) + e1 * u.x + e2 * u.y;
    // SphQuad quad = init(l, hitPoint);
    // lightSampleData.pointOnLight = SphQuadSample(quad, u);
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                          (-dot(lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
    return lightSampleData;
}
