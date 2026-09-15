#pragma once
#include <optix_types.h>
#include <vector_types.h>
#include <sutil/Matrix.h>

#include <random.h>
#include <lights.h>
#include <strelka/material/material_params.h>
#include <strelka/material/ior_stack.h>
// Parameters only, never openpbr.h: this header is included by the host, and the
// vendored library carries ~264 KB of lookup tables the host has no use for.
#include <strelka/material/openpbr/openpbr_params.h>

#include <env_alias_sampling.h>
#include <emissive_mesh_light.h>
#include <sharc.h>

// Values, not spellings, are what has to match Metal's ShaderTypes.h, which
// keeps its copy as #define. Enumerators here so the debugger and the analyzer
// can see them; the numbers below are the contract.
enum : uint32_t
{
    GEOMETRY_MASK_TRIANGLE = 1,
    GEOMETRY_MASK_CURVE = 2,
    GEOMETRY_MASK_LIGHT = 4,
    GEOMETRY_MASK_LIGHT_HIDDEN = 8,
    GEOMETRY_MASK_MEDIUM = 16,

    GEOMETRY_MASK_GEOMETRY = GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_CURVE,

    RAY_MASK_PRIMARY = GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_MEDIUM,
    RAY_MASK_SHADOW = GEOMETRY_MASK_GEOMETRY,
    RAY_MASK_SECONDARY = RAY_MASK_PRIMARY | GEOMETRY_MASK_LIGHT_HIDDEN,
};

// Params::projectionType. Mirrors oka::Camera::ProjectionType, which device code
// cannot include, and the identically-named constants in the Metal ShaderTypes.h.
enum : uint32_t
{
    PROJECTION_PERSPECTIVE = 0u,
    PROJECTION_ORTHOGRAPHIC = 1u
};

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
    const EmissiveMeshLight* emissiveMeshes;
    const EmissiveTriangleLight* emissiveTriangles;
    const EmissiveInstanceTransform* emissiveInstanceTransforms;
    const EmissiveInstanceTransform* prevEmissiveInstanceTransforms;
    uint32_t numEmissiveMeshes;
    // P(mesh emitter | local emitter). The complement selects an analytic
    // light; environment selection is the outer strategy.
    float meshLightSelectionPdf;
    /// Packed IES candela tables, indexed by each light's points[0].y. Never
    /// null once the scene is built -- a scene with no profile still gets a
    /// zero-count header, so sampleIesCandela() needs no null check per light.
    const IesGpuBufferHeader* iesProfiles;
    const cudaTextureObject_t* projectorTextures;
    uint32_t numProjectorTextures;
};

// Traversal only needs coverage and one UV transform. Keep it out of the
// shading-oriented MaterialParams table so foliage candidates fetch 40 bytes
// instead of several cache lines and do not evaluate sin/cos per intersection.
struct OptixAlphaMaterialData
{
    float baseColorAlpha;
    float alphaCutoff;
    uint32_t alphaMode;
    uint32_t reserved;
    float2 uvOffset;
    float2 uvTransformX;
    float2 uvTransformY;
};
static_assert(sizeof(OptixAlphaMaterialData) == 40, "OptiX alpha material ABI changed");

/// How `AovSample::depth` is encoded. Mirrors kDenoiseDepth* in the Metal
/// ShaderTypes.h, value for value, so a guide dumped from either backend means
/// the same thing.
enum : uint32_t
{
    STRELKA_DENOISE_DEPTH_DEVICE = 0u, ///< clip z / w, the value a depth buffer holds
    STRELKA_DENOISE_DEPTH_VIEWZ = 1u, ///< distance along the camera's forward axis
    STRELKA_DENOISE_DEPTH_RADIAL = 2u ///< distance to the eye
};

/// Debug visualisations, in the order the editor's combo box lists them.
///
/// Numbering is shared by the editor and both backends and must remain stable.
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
    eSharcGrid, ///< a stable colour per voxel, at the primary hit
    eSharcRadiance, ///< what the cache would answer at the primary hit
    eSharcOccupancy, ///< how much of the table is in use, as a screen overlay
    eSharcBounces, ///< bounces traced per pixel: what the cache actually saves
};

enum : uint32_t
{
    DEBUG_MODE_FIRST_AOV = 3u
};

#define DEBUG_MODE_IS_SINGLE_HIT(d)                                                                                    \
    ((d) == (uint32_t)DebugMode::eNormal || (d) == (uint32_t)DebugMode::eMotionBlur ||                                  \
     (d) == (uint32_t)DebugMode::eSharcGrid)

#define DEBUG_MODE_IS_SCENE_LINEAR(d)                                                                                  \
    ((d) == (uint32_t)DebugMode::eNone || (d) == (uint32_t)DebugMode::eSharcRadiance)

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

struct SharcPathState
{
    /// prd.radiance at the moment of the visit, so the difference at the end of
    /// the path is what the path gathered *after* it.
    float3 radianceAtVisit;
    /// 1 / throughput at the visit, floored per channel.
    float3 invThroughput;
    float3 responsiveRadiance;
    float launchRoughness;
    /// SHARC_NO_ENTRY until this path visits a voxel.
    uint32_t index;
    uint32_t responsiveIndex;
};

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct Params
{
    // Grouped by subsystem so camera, environment, SHARC and denoiser fields
    // remain readable; padding order is secondary for this per-frame upload.
    uint32_t subframe_index;
    uint32_t samples_per_launch;
    uint32_t maxSampleCount;
    uint32_t mortonLevels;
    uint32_t hasBlueNoise;
    uint32_t blueNoiseSwitch;
    float4* image;
    float4* accum;
    /// The first-event split. Null unless somebody asked for it -- see
    /// Params::writeSplitAov.
    float4* diffuse;
    uint16_t* diffuseCounter;
    float4* specular;
    uint16_t* specularCounter;
    bool writeSplitAov;
    uint32_t image_width;
    uint32_t image_height;

    uint32_t max_depth;

    uint32_t rectLightSamplingMethod;

    float3 exposure;
    float clipToView[16];
    float viewToWorld[16];
    float worldToClip[16];
    float prevWorldToClip[16];

    uint32_t projectionType;
    float orthoHalfWidth;
    float orthoHalfHeight;

    OptixTraversableHandle handle;
    SceneData scene;

    // Material data (indexed by materialId)
    MaterialParams* materials;
    OptixAlphaMaterialData* alphaMaterials;
    cudaTextureObject_t* materialTextures; // flat array: [materialId * MAX_MATERIAL_TEXTURES + slot]

    OpenPBRParams* openpbrParams;
    /// Flat, like materialTextures: [materialId * MAX_OPENPBR_TEXTURES + slot].
    /// Nineteen slots rather than six, and a wider set of transfer functions; the
    /// mapping lives in OptixRender.cpp beside its Metal counterpart's.
    cudaTextureObject_t* openpbrTextures;

    bool enableAccumulation;
    // developers settings:
    bool enableMotionBlur;
    bool isMotionBlurVisible;
    uint32_t debug;
    float shadowRayTmin;
    float materialRayTmin;
    uint32_t misHeuristic; // 0 = balance, 1 = power
    uint32_t volumeModel;

    bool enableShaderReorder;

    // Environment map (dome light)
    bool hasEnvMap;
    cudaTextureObject_t envMapTexture;      // bilinear, normalized coords -- radiance
    const EnvAliasEntry* envAliasTable;
    // 1 / integral(luminance dOmega): turns a texel's luminance straight into
    // its solid-angle sampling density.
    float envPdfScale;
    uint32_t envMapWidth;
    uint32_t envMapHeight;
    float envMapIntensity;
    float envMapRotation; // Y-axis rotation in radians
    float3 envMapColorTint;
    float envSelectionPdf;

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

    uint32_t subsurfaceIterations;
    /// Whether any material in the scene is a medium boundary. Gates the second
    /// traversal a shadow ray needs to accumulate optical depth across those
    /// boundaries, which is pure cost in the scenes that have none.
    bool hasBoundedMedium;
    bool hasSubsurface;

    bool hasCurves;
    /// Whether any material is MASK or BLEND. Gates the stochastic coverage test
    /// -- and with it an opacity texture fetch -- on every shaded vertex.
    bool hasCutout;
    bool hasOpenPBR;
    bool openpbrSheenAndCoat;
    bool openpbrDispersion;
    bool openpbrTranslucency;
    bool openpbrMetallic;

    bool hasFog;
    float fogSigmaT; ///< extinction, per world unit
    float fogAnisotropy; ///< Henyey-Greenstein g; > 0 scatters forward
    float fogHeight; ///< the medium fills everything below this y
    float3 fogAlbedo; ///< single-scattering albedo: what an event scatters rather than absorbs

    uint32_t* iorStats;
    SharcEntry* sharcEntries;
    /// One record per pixel. Null, and never touched, when sharcCapacity is 0 --
    /// which is a compile-time constant, so the whole cache path folds away.
    SharcPathState* sharcPath;
    uint32_t sharcCapacity; ///< entries; a power of two, 0 = off
    uint32_t sharcMinSamples; ///< deposits a voxel needs before it may be read
    uint32_t sharcDepth; ///< first bounce allowed to read the cache
    uint32_t sharcReadMaxSubframe;
    /// Non-zero when any light in the scene is marked responsive. Bound into the
    /// pipeline as a constant, so a scene without one compiles the second entry,
    /// the second probe and the second set of atomics out entirely.
    uint32_t sharcResponsive;
    /// One bit per light, set where UniformLightDesc::responsive is. Null when
    /// sharcResponsive is 0. Indexed by light id: word `id >> 5`, bit `id & 31`.
    const uint32_t* sharcResponsiveLights;
    /// World size of one pixel at unit distance times the pixels a voxel should
    /// span, so the setting behind it means the same thing at any resolution or
    /// field of view. See sharc_grid.h.
    float sharcBaseSize;

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

enum : uint32_t
{
    PATH_PASSTHROUGH_MAX = 32u
};

/// Hard ceiling on a walk, whatever `Params::subsurfaceIterations` asks for.
/// Metal's MEDIUM_MAX_STEPS, for the same reason.
enum : uint32_t
{
    MEDIUM_MAX_STEPS = 256u
};

enum : uint32_t
{
    IOR_STAT_OVERFLOW = 0,
    IOR_STAT_UNMATCHED = 1,
    IOR_STAT_ESCAPED_INSIDE = 2,
    IOR_STAT_COUNT = 3
};

enum : uint32_t
{
    STRELKA_PAYLOAD_COUNT = 2
};

struct OptixIorStack
{
    uint32_t materials[IOR_STACK_SIZE];
    int top;
};
static_assert(sizeof(OptixIorStack) == 20, "OptiX stores IOR and priority in the material table");

struct PerRayData
{
    // --- Touched every bounce --------------------------------------------
    float3 radiance;
    float3 throughput;
    /// The *next* ray, written by the shading program and read by the raygen
    /// loop. Not the current one -- that is optixGetWorldRayOrigin/Direction.
    float3 origin;
    float3 dir;
    float lastBsdfPdf;
    float misDistance;

    uint32_t depth : 9;
    /// Steps the current walk has taken. Reset when a medium is entered, and
    /// bounded by Params::subsurfaceIterations, itself clamped to
    /// MEDIUM_MAX_STEPS (256).
    uint32_t mediumStep : 9;
    uint32_t passthrough : 6;
    /// An EventType, held as bits because a bit field of enum type is not one
    /// storage unit with the rest on every compiler this builds under. Read and
    /// written through firstEventType() / setFirstEventType().
    uint32_t firstEventBits : 3;
    uint32_t specularBounce : 1;
    uint32_t neeDone : 1;
    /// Set by the closest hit when the ray went straight through the surface it
    /// hit. The raygen loop reads it to decide whether the segment counted as a
    /// bounce, and clears it before the next trace.
    uint32_t passedThrough : 1;
    /// Whether this path is the one that writes the pixel's guide record.
    uint32_t writeAov : 1;
    /// Set once the guide record for this pixel has been written, so the bounce
    /// after the first describable surface cannot overwrite it.
    uint32_t aovDone : 1;

    SamplerState sampler;
    OptixIorStack iorStack;

    uint32_t medium;
    float3 mediumAlbedo;

    DEVICE_FUNC EventType firstEventType() const
    {
        return (EventType)firstEventBits;
    }
    DEVICE_FUNC void setFirstEventType(EventType e)
    {
        firstEventBits = (uint32_t)e;
    }
};

/// The 108-byte size is continuation-stack ABI; adding a field increases
/// per-thread local state.
static_assert(sizeof(PerRayData) == 108, "PerRayData sizes the continuation stack; see docs/open-perf.md");

/// All three are spelled out because the first two are SBT record offsets that
/// the hit-group layout in OptixRender.cpp indexes by hand; a value here is not
/// an implementation detail of the enum.
enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT = 2
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
    const uint32_t* lightIndices;
    int32_t materialId;  // index into params.materials[] and params.materialTextures[]
    /// Segments per strand for a curve set, or 0 when the strands differ in
    /// length. It is the whole of what a root-to-tip UV needs: the segment index
    /// modulo this is where along the strand a hit landed, for no extra memory.
    uint32_t curveSegmentsPerStrand;
};
