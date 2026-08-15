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
// The boundary of a participating medium. Its own bit because a shadow ray must
// not be stopped by it -- RAY_MASK_SHADOW is the geometry bits alone, so a fog
// gizmo left on the triangle mask blacks out everything it encloses. Same value
// as Metal's GEOMETRY_MASK_MEDIUM, and 8 is skipped for the same reason: that
// is Metal's GEOMETRY_MASK_LIGHT_HIDDEN, which OptiX does not have yet.
#define GEOMETRY_MASK_MEDIUM 16

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_CURVE)

#define RAY_MASK_PRIMARY (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_MEDIUM)
#define RAY_MASK_SHADOW GEOMETRY_MASK_GEOMETRY
#define RAY_MASK_SECONDARY (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_MEDIUM)

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
    /// Packed IES candela tables, indexed by each light's points[0].y. Never
    /// null once the scene is built -- a scene with no profile still gets a
    /// zero-count header, so sampleIesCandela() needs no null check per light.
    const IesGpuBufferHeader* iesProfiles;
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
    /// Which reading of KHR_materials_volume the shading path uses: 0 = glTF
    /// (sigma_t = -ln(C)/d), 1 = Cycles ((1-C)/d). They disagree by a lot -- at an
    /// attenuation colour of 0.5, 0.69/d against 0.5/d -- and every ladder scene
    /// asks for the Cycles form, so pinning it was measuring the wrong density.
    uint32_t volumeModel;

    /// Whether this device's optixReorder() actually reorders. Queried once via
    /// OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING; on hardware without
    /// the sorting unit the call is a documented no-op, and skipping it there
    /// keeps the coherence key -- two dependent loads to reach material_type --
    /// off the critical path of a machine that cannot use it.
    bool enableShaderReorder;

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

    // --- Participating media ---------------------------------------------
    /// Ceiling on the steps of one random walk. A dense medium is a long walk
    /// that Russian roulette alone terminates slowly, and a path that never ends
    /// is a hang rather than a dim pixel. Drawn from
    /// `render/pt/subsurfaceIterations`, the same setting that sizes the Metal
    /// wavefront's extra dispatch iterations, so the two backends give a walk
    /// the same budget. 64 is where 25_subsurface stops moving.
    uint32_t subsurfaceIterations;
    /// Whether any material in the scene is a medium boundary. Gates the second
    /// traversal a shadow ray needs to accumulate optical depth across those
    /// boundaries, which is pure cost in the scenes that have none.
    bool hasBoundedMedium;

    /// The three ways the nested-dielectric stack loses a path, counted per
    /// launch: see IOR_STAT_* below. Null when the buffer has not been
    /// allocated, and then the counting is skipped rather than guessed at.
    /// Metal has reported these since its pop learned to match on the material
    /// being left; this is the other half of entry 5 of docs/open-defects.md.
    uint32_t* iorStats;

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

/// How many cutout surfaces one path may slip through before it is given up on.
/// A hedge of alpha-tested leaves would otherwise let a path bounce forever:
/// passing through deliberately does not spend a bounce (see PerRayData::
/// passthrough), so `max_depth` cannot bound it. Same value as Metal's
/// PATH_PASSTHROUGH_MAX, for the same reason.
#define PATH_PASSTHROUGH_MAX 32u

/// Hard ceiling on a walk, whatever `Params::subsurfaceIterations` asks for.
/// Metal's MEDIUM_MAX_STEPS, for the same reason.
#define MEDIUM_MAX_STEPS 256u

// The three ways the nested-dielectric stack loses a path, in the order Metal's
// ShaderTypes.h numbers them so the two backends' reports read the same.
//
// A push onto a full stack wants a deeper stack. An unmatched pop and a path
// that reaches the environment still inside a medium are both a mesh with a
// hole in it, seen from each side -- and the third is the one no exit event can
// catch, because the ray left through the hole. Each of them carries the wrong
// medium, and therefore the wrong absorption, for the rest of its life.
#define IOR_STAT_OVERFLOW 0
#define IOR_STAT_UNMATCHED 1
#define IOR_STAT_ESCAPED_INSIDE 2
#define IOR_STAT_COUNT 3

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
    /// Whether the vertex this ray left performed a next-event estimate. False
    /// under estimatorMode 1, and false when the scene has nothing to connect to;
    /// in both cases the BSDF strategy owns the whole contribution and the miss /
    /// light-hit shaders must not apply a MIS weight against an estimate that was
    /// never made.
    bool neeDone;
    /// Set by the closest hit when the ray went straight through the surface it
    /// hit. The raygen loop reads it to decide whether the segment counted as a
    /// bounce, and clears it before the next trace.
    bool passedThrough;
    float lastBsdfPdf;
    /// How far the ray has travelled since the vertex `lastBsdfPdf` was measured
    /// at. Zero for every path that has not passed through anything.
    ///
    /// Passing through a cutout, or crossing the boundary of a medium, restarts
    /// the ray at the surface it slipped past, and the multiple-importance weight
    /// at an area light needs the distance from the vertex that *scattered*, not
    /// from wherever the ray was last restarted. The direction does not change
    /// across either, so one number recovers that vertex:
    /// origin - direction * this. Metal's PathState carries the same field for
    /// the same reason.
    float misDistance;
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

    // --- Participating media ---------------------------------------------
    /// Which medium the path is inside: 0 for none, otherwise the material index
    /// plus one. Same encoding as the low bits of Metal's PathState::medium.
    ///
    /// A camera that starts inside a translucent object is not handled -- there
    /// is nothing to tell the path which medium it is in.
    uint32_t medium;
    /// Steps the current walk has taken. Reset when a medium is entered, and
    /// bounded by Params::subsurfaceIterations; past that the walk stops drawing
    /// free flights and the next surface is its boundary.
    uint32_t mediumStep;
    /// The walk's single-scattering albedo, resolved at the boundary the path
    /// entered through, because that is the last place a texture exists: inside
    /// the medium there is no surface to sample. A bounded volume has no entry
    /// surface to have textured and keeps the material's constant instead.
    float3 mediumAlbedo;
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
