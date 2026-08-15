#pragma once
#include <optix_types.h>
#include <vector_types.h>
#include <sutil/Matrix.h>

#include <random.h>
#include <lights.h>
#include <strelka/material/material_params.h>
#include <strelka/material/ior_stack.h>

#include "env_alias_sampling.h"

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
    cudaTextureObject_t envMapTexture;      // bilinear, normalized coords -- radiance
    cudaTextureObject_t envMapTexturePoint; // point, unnormalized coords -- sampling / pdf
    const EnvAliasEntry* envAliasTable;
    // (w*h) / (2*pi^2 * totalPower): turns a texel's luminance straight into its
    // solid-angle sampling density, so no CDF or pdf table is stored or searched.
    float envPdfScale;
    uint32_t envMapWidth;
    uint32_t envMapHeight;
    float envMapIntensity;
    float envMapRotation; // Y-axis rotation in radians
    float3 envMapColorTint;

    // A separate environment for camera rays. See Scene::EnvLightDesc: the
    // backdrop is what the camera sees, the map above is what lights the scene.
    bool hasEnvBackground;
    cudaTextureObject_t envBackgroundTexture;
    float envBackgroundIntensity;

    /// Light candidates drawn per shading point before one is resampled -- the M
    /// of resampled importance sampling. One is plain next-event estimation and
    /// the arithmetic reduces to exactly what it was.
    uint32_t risCandidates;
    /// 0 = NEE + MIS (normal), 1 = BSDF sampling only. Two independent unbiased
    /// estimators of the same integral, so at convergence they must agree.
    uint32_t estimatorMode;
    /// Upper bound on what one indirect path may contribute; 0 disables it.
    float clampIndirect;

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
    IorStack iorStack;
    bool specularBounce;
    /// Whether the vertex this ray left performed a next-event estimate. False
    /// under estimatorMode 1, and false when the scene has nothing to connect to;
    /// in both cases the BSDF strategy owns the whole contribution and the miss /
    /// light-hit shaders must not apply a MIS weight against an estimate that was
    /// never made.
    bool neeDone;
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
};
