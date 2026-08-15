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

// Params::projectionType. Mirrors oka::Camera::ProjectionType, which device code
// cannot include, and the identically-named constants in the Metal ShaderTypes.h.
#define PROJECTION_PERSPECTIVE 0u
#define PROJECTION_ORTHOGRAPHIC 1u

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

/// How `AovSample::depth` is encoded. Mirrors kDenoiseDepth* in the Metal
/// ShaderTypes.h, value for value, so a guide dumped from either backend means
/// the same thing.
#define STRELKA_DENOISE_DEPTH_DEVICE 0u ///< clip z / w, the value a depth buffer holds
#define STRELKA_DENOISE_DEPTH_VIEWZ 1u ///< distance along the camera's forward axis
#define STRELKA_DENOISE_DEPTH_RADIAL 2u ///< distance to the eye

/// Debug visualisations, in the order the editor's combo box lists them.
///
/// This is Metal's DebugMode from ShaderTypes.h, unchanged. It was already the
/// numbering the editor sent -- `RenderSettingsPanel` builds its list from that
/// enum for both backends -- while OptiX read 2 and 3 as "diffuse split" and
/// "specular split", so every AOV entry in that menu selected something else on
/// this backend.
enum class DebugMode : uint32_t
{
    eNone = 0,
    eNormal,
    eMotionBlur,
    // Denoiser guides. Selecting one of these turns guide production on for the
    // frame, so they cost nothing when nobody is looking at them.
    eAovDiffuseAlbedo,
    eAovSpecularAlbedo,
    eAovNormal,
    eAovRoughness,
    eAovDepth,
    eAovMotion,
    eAovReactive,
    eAovSpecularHitDistance,
};

#define DEBUG_MODE_FIRST_AOV 3u

/// What a denoiser needs to know about the primary hit, written once per pixel
/// by the program that shades it (or by the miss program, for background).
///
/// Field for field the Metal backend's AovSample, including the padding, so the
/// two backends' guides are the same 64 bytes and can be compared directly. The
/// packing differs only in spelling: Metal needs `packed_float3` because its
/// float3 is 16 bytes, CUDA's is already 12.
struct AovSample
{
    float3 diffuseAlbedo;
    float depth; ///< encoding selected by Params::denoiseDepthMode
    float3 specularAlbedo;
    float roughness;
    float3 normal; ///< world space
    float motionX; ///< previous-frame screen position minus current, in pixels
    float motionY;
    /// Distance from the primary hit to what its specular lobe sees, so a
    /// reflection reprojects at the depth of the thing being reflected rather
    /// than at the mirror's own. Zero when the surface is not specular.
    float specularHitDistance;
    /// 0 = trust the history here, 1 = ignore it. Raised where the motion vector
    /// is known to be a lie: mirrors, glass, and anything whose previous position
    /// could not be established.
    float reactive;
    float pad2;
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
    /// Projection of a world point onto this frame's and the previous frame's
    /// screen. The second is what a motion vector is measured against; without it
    /// a temporal denoiser has to assume the camera did not move, which is the
    /// one thing that is never true while anybody is looking at the image.
    float worldToClip[16];
    float prevWorldToClip[16];

    // Which projection generates the primary ray. An orthographic camera has no
    // centre of projection -- every ray runs down the view axis and the pixel
    // chooses where on the film it starts -- so it is a branch in ray generation
    // rather than a different matrix, and the half-extents have to come across
    // separately. Mirrors Uniforms::projectionType on the Metal side.
    uint32_t projectionType;
    float orthoHalfWidth;
    float orthoHalfHeight;

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

    // --- Denoiser guides -------------------------------------------------
    /// One record per pixel; null when guides are not being produced.
    AovSample* aov;
    /// Whether this launch writes guides at all. Raised by the denoiser, by the
    /// upscaler, and by any of the AOV debug views.
    bool writeAov;
    /// Take the material guides at the camera-visible surface instead of walking
    /// on to the first rough one. A switch rather than a default because the two
    /// answers differ by more than noise on a mirror; see
    /// tools/feature_tests/README.md, 20_mirror_and_floor.
    bool guidePrimaryHit;
    /// True when the previous frame's camera is meaningful, i.e. not the first
    /// frame and not just after a history reset.
    bool hasPrevFramePose;
    uint32_t denoiseDepthMode;
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
    /// Where in the image this path's camera ray actually went, y down and in
    /// pixels, jitter included. A motion vector is the difference between this
    /// and where the same surface point sat last frame; differencing against the
    /// pixel centre instead leaves the jitter inside every vector, which is a
    /// subpixel wobble on every pixel of a perfectly still image.
    float2 pixelSample;
    /// Whether this path is the one that writes the pixel's guide record.
    bool writeAov;
    /// Set once the guide record for this pixel has been written, so the bounce
    /// after the first describable surface cannot overwrite it.
    bool aovDone;
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
