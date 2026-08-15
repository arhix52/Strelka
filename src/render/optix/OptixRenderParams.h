#pragma once
#include <optix_types.h>
#include <vector_types.h>
#include <sutil/Matrix.h>

#include <random.h>
#include <lights.h>
#include <strelka/material/material_params.h>
#include <strelka/material/ior_stack.h>

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
    Vertex* vb_prev;
    uint32_t* ib;
    UniformLight* lights;
    uint32_t numLights;
};

struct Params
{
    uint32_t subframe_index;
    uint32_t samples_per_launch;
    uint32_t maxSampleCount;
    float4* image;
    float4* accum;
    float4* diffuse;
    uint16_t* diffuseCounter;
    float4* specular;
    uint16_t* specularCounter;
    uint32_t image_width;
    uint32_t image_height;

    uint32_t max_depth;

    uint32_t rectLightSamplingMethod;

    float3 exposure;
    float clipToView[16];
    float viewToWorld[16];

    OptixTraversableHandle handle;
    SceneData scene;

    // Material data (indexed by materialId)
    MaterialParams* materials;
    cudaTextureObject_t* materialTextures; // flat array: [materialId * MAX_MATERIAL_TEXTURES + slot]

    bool enableAccumulation;
    // developers settings:
    bool enableMotionBlur;
    bool isMotionBlurVisible;
    uint32_t debug;
    float shadowRayTmin;
    float materialRayTmin;
    uint32_t misHeuristic; // 0 = balance, 1 = power

    // Environment map (dome light)
    bool hasEnvMap;
    cudaTextureObject_t envMapTexture;
    float* envCdfX;       // conditional CDF per row: [y * envMapWidth + x]
    float* envCdfY;       // marginal CDF: [y]
    uint32_t envMapWidth;
    uint32_t envMapHeight;
    float envMapIntensity;
    float envMapRotation; // Y-axis rotation in radians
    float3 envMapColorTint;
    float envMapTotalPower;

    // Depth of field
    int   useDof;
    float focalDistance;
    float lensRadius;
    int   apertureBlades;
    float bladeRotation;
    float anamorphicRatio;

    // Lens shift
    float shiftX;
    float shiftY;
};

enum class EventType: uint8_t
{
    eUndef,
    eAbsorb,
    eDiffuse,
    eSpecular,
    eLast,
};

/// How many cutout surfaces one path may slip through before it is given up on.
/// A hedge of alpha-tested leaves would otherwise let a path bounce forever:
/// passing through deliberately does not spend a bounce (see PerRayData::
/// passthrough), so `max_depth` cannot bound it. Same value as Metal's
/// PATH_PASSTHROUGH_MAX, for the same reason.
#define PATH_PASSTHROUGH_MAX 32u

struct PerRayData
{
    SamplerState sampler;
    uint32_t linearPixelIndex;
    uint32_t sampleIndex;
    uint32_t depth; // bounce
    /// How many transparent surfaces this path has already passed straight
    /// through. Counted separately from `depth` because a cutout is coverage,
    /// not scattering -- charging it a bounce empties the path budget on a
    /// canopy before any light transport happens.
    uint32_t passthrough;
    float3 radiance;
    float3 throughput;
    float3 origin;
    float3 dir;
    IorStack iorStack;
    bool specularBounce;
    /// Set by the closest hit when the ray went straight through the surface it
    /// hit. The raygen loop reads it to decide whether the segment counted as a
    /// bounce, and clears it before the next trace.
    bool passedThrough;
    float lastBsdfPdf;
    EventType firstEventType;
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

static constexpr int MAX_MATERIAL_TEXTURES = 6;

struct HitGroupData
{
    int32_t indexOffset;
    int32_t indexCount;
    int32_t vertexOffset;
    int32_t lightId;     // only for lights. -1 for others
    int32_t materialId;  // index into params.materials[] and params.materialTextures[]
    /// Segments per strand for a curve set, or 0 when the strands differ in
    /// length. It is the whole of what a root-to-tip UV needs: the segment index
    /// modulo this is where along the strand a hit landed, for no extra memory.
    uint32_t curveSegmentsPerStrand;
};
