#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>
#ifndef __METAL_VERSION__
#ifdef __cplusplus
#include <Metal/MTLTypes.hpp>
#endif
#endif

#define GEOMETRY_MASK_TRIANGLE 1
#define GEOMETRY_MASK_CURVE 2
#define GEOMETRY_MASK_LIGHT 4

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_CURVE)

#define RAY_MASK_PRIMARY (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT)
#define RAY_MASK_SHADOW GEOMETRY_MASK_GEOMETRY
#define RAY_MASK_SECONDARY GEOMETRY_MASK_GEOMETRY

#ifndef __METAL_VERSION__
struct packed_float3
{
#    ifdef __cplusplus
    packed_float3() = default;
    packed_float3(vector_float3 v) : x(v.x), y(v.y), z(v.z)
    {
    }
#    endif
    float x;
    float y;
    float z;
};
#endif

enum class DebugMode : uint32_t
{
    eNone = 0,
    eNormal,
};

struct Vertex
{
    vector_float3 pos;
    uint32_t tangent;

    uint32_t normal;
    uint32_t uv;
    float pad0;
    float pad1;
};

struct Uniforms
{
    simd::float4x4 viewToWorld;
    simd::float4x4 clipToView;
    vector_float3 missColor;

    uint32_t width;
    uint32_t height;
    uint32_t frameIndex;
    uint32_t subframeIndex;

    uint32_t numLights;
    uint32_t enableAccumulation;
    uint32_t samples_per_launch;
    uint32_t maxDepth;
    
    uint32_t rectLightSamplingMethod;

    uint32_t tonemapperType; // 0 - "None", "Reinhard", "ACES", "Filmic"
    float gamma; // 0 - off
    vector_float3 exposureValue;

    uint32_t debug;
};

struct UniformsTonemap
{
    uint32_t width;
    uint32_t height;
    
    uint32_t tonemapperType; // 0 - "None", "Reinhard", "ACES", "Filmic"
    float gamma; // 0 - off
    float maxEDR;
    vector_float3 exposureValue;
};

struct Triangle
{
    vector_float3 positions[3];
    uint32_t normals[3];
    uint32_t tangent[3];
    uint32_t uv[3];
};

// GPU side structure
struct UniformLight
{
    vector_float4 points[4];
    vector_float4 color;
    vector_float4 normal;
    int type;
    float halfAngle;
    float pad0;
    float pad1;
};

struct Material
{
    simd::float3 diffuse;

#ifdef __METAL_VERSION__
    texture2d<float>  diffuseTexture;
    texture2d<float>  normalTexture;
#else
    MTL::ResourceID diffuseTexture; // uint64_t - alignment 8
    MTL::ResourceID normalTexture;
#endif
};

#endif
