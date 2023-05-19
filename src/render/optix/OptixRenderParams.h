#pragma once 
#include <optix_types.h>
#include <vector_types.h>
#include <sutil/Matrix.h>

#include "RandomSampler.h"
#include "Lights.h"

#define GEOMETRY_MASK_TRIANGLE 1
#define GEOMETRY_MASK_CURVE 2
#define GEOMETRY_MASK_LIGHT 4

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_CURVE)

#define RAY_MASK_PRIMARY (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT)
#define RAY_MASK_SHADOW GEOMETRY_MASK_GEOMETRY
#define RAY_MASK_SECONDARY GEOMETRY_MASK_GEOMETRY

struct Vertex
{
    float3 position;
    uint32_t tangent;

    uint32_t normal;
    uint32_t uv;
    float pad0;
    float pad1;
};

struct SceneData
{
    Vertex* vb;
    uint32_t* ib;
    UniformLight* lights;
    uint32_t numLights;
};

struct Params
{
    uint32_t subframe_index;
    uint32_t samples_per_launch;
    float4* image;
    float4* accum;
    uint32_t image_width;
    uint32_t image_height;

    uint32_t max_depth;

    uint32_t rectLightSamplingMethod;

    float clipToView[16];
    float viewToWorld[16];

    OptixTraversableHandle handle;
    SceneData scene;

    bool enableAccumulation;
    // developers settings:
    uint32_t debug;
    float shadowRayTmin;
    float materialRayTmin;
};

struct PerRayData
{
    SamplerState sampler;
    uint32_t linearPixelIndex;
    uint32_t sampleIndex;
    uint32_t depth; // bounce
    float3 radiance;
    float3 throughput;
    float3 origin;
    float3 dir;
    float3 prevHitPos;
    bool inside;
    bool specularBounce;
    float lastBsdfPdf;
};

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

struct RayGenData
{
    // No data needed
};

struct MissData
{
    float3 bg_color;
};

struct HitGroupData
{
    int32_t indexOffset;
    int32_t indexCount;
    int32_t vertexOffset;
    int32_t lightId; // only for lights. -1 for others
    CUdeviceptr argData;
    CUdeviceptr roData;
    CUdeviceptr resHandler;
    float4 world_to_object[4] = {};
    float4 object_to_world[4] = {};
};
