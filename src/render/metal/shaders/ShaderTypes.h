#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

#define GEOMETRY_MASK_TRIANGLE 1
#define GEOMETRY_MASK_SPHERE   2
#define GEOMETRY_MASK_LIGHT    4

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_SPHERE)

#define RAY_MASK_PRIMARY   (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT)
#define RAY_MASK_SHADOW    GEOMETRY_MASK_GEOMETRY
#define RAY_MASK_SECONDARY GEOMETRY_MASK_GEOMETRY

#ifndef __METAL_VERSION__
struct packed_float3 {
#ifdef __cplusplus
    packed_float3() = default;
    packed_float3(vector_float3 v) : x(v.x), y(v.y), z(v.z) {}
#endif
    float x;
    float y;
    float z;
};
#endif

struct Vertex
{
    vector_float3 pos;
    uint32_t tangent;

    uint32_t normal;
    uint32_t uv;
    float pad0;
    float pad1;
};

struct Uniforms {
    unsigned int width;
    unsigned int height;
    unsigned int frameIndex;
    unsigned int lightCount;
    simd::float4x4 viewToWorld;
    simd::float4x4 clipToView;
};

struct Triangle {
    uint32_t normals[3];
    uint32_t tangent[3];
    uint32_t uv[3];
};

#endif
