#pragma once 
#include <optix_types.h>

#include <vector_types.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/compatibility.hpp>

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
    float4* image;
    float4* accum;
    unsigned int image_width;
    unsigned int image_height;
    unsigned int subframe_index;
    unsigned int samples_per_launch;
    uint32_t max_depth;

    uint32_t rectLightSamplingMethod;

    glm::float4x4 clipToView;
    glm::float4x4 viewToWorld;

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
    uint32_t rndSeed;
    uint32_t depth;
    float3 radiance;
    float3 throughput;
    float3 origin;
    float3 dir;
    bool inside;
    bool specularBounce;
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
