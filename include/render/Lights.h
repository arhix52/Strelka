#pragma once
#include <vector_types.h>
#include <sutil/vec_math.h>

struct UniformLight
{
    float4 points[4];
    float4 color;
    float4 normal;
    int type;
    float pad0;
    float pad2;
    float pad3;
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

static __device__ float calcLightArea(const UniformLight& l)
{
    float area = 0.0f;

    if (l.type == 0) // rectangle area
    {
        float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
        float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);
        area = length(cross(e1, e2));
    }
    else if (l.type == 1) // disc area
    {
        area = M_PIf * l.points[0].x * l.points[0].x; // pi * radius^2
    }
    else if (l.type == 2) // sphere area
    {
        area = 4.0f * M_PIf * l.points[0].x * l.points[0].x; // 4 * pi * radius^2
    }
    return area;
}

static __device__ float3 calcLightNormal(const UniformLight& l, const float3& hitPoint)
{
    float3 norm = make_float3(0.0f);

    if (l.type == 0)
    {
        float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
        float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);

        norm = -normalize(cross(e1, e2));
    }
    else if (l.type == 1)
    {
        norm = make_float3(l.normal);
    }
    else if (l.type == 2)
    {
        norm = normalize(hitPoint - make_float3(l.points[1]));
    }
    return norm;
}

static __device__ void fillLightData(const UniformLight& l, const float3& hitPoint, LightSampleData& lightSampleData)
{
    lightSampleData.area = calcLightArea(l);
    lightSampleData.normal = calcLightNormal(l, hitPoint);
    const float3 toLight = lightSampleData.pointOnLight - hitPoint;
    const float lenToLight = length(toLight);
    lightSampleData.L = toLight / lenToLight;
    lightSampleData.distToLight = lenToLight;
}

static __device__ LightSampleData SampleRectLight(const UniformLight& l, const float2& u, const float3& hitPoint)
{
    LightSampleData lightSampleData;
    // uniform sampling
    float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
    float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);
    lightSampleData.pointOnLight = make_float3(l.points[0]) + e1 * u.x + e2 * u.y;
    // SphQuad quad = init(l, hitPoint);
    // lightSampleData.pointOnLight = SphQuadSample(quad, u);
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                          (-dot(lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
    return lightSampleData;
}
