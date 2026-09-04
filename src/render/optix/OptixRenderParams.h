#pragma once
#include <optix_types.h>
#include <vector_types.h>
#include <sutil/Matrix.h>

#include <random.h>
#include <lights.h>
#include <strelka/material/material_params.h>
#include <strelka/material/ior_stack.h>

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
// An emitter the camera must not see, but a bounce must. A light authored with
// visibleToCamera off still lights the scene and still has to be hit by a BSDF
// ray for the MIS estimate to balance -- it simply must not appear as a shape in
// the frame. Same value as Metal's GEOMETRY_MASK_LIGHT_HIDDEN.
    GEOMETRY_MASK_LIGHT_HIDDEN = 8,
// The boundary of a participating medium. Its own bit because a shadow ray must
// not be stopped by it -- RAY_MASK_SHADOW is the geometry bits alone, so a fog
// gizmo left on the triangle mask blacks out everything it encloses. Same value
// as Metal's GEOMETRY_MASK_MEDIUM.
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
    /// Images thrown by projector lights, indexed by each light's points[0].z.
    /// Null when the scene has no projector, so the fetch checks -- unlike the
    /// IES table this is an array of texture objects, and an empty one has no
    /// header to make a zero-length version of.
    const cudaTextureObject_t* projectorTextures;
    uint32_t numProjectorTextures;
};

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
    // Radiance cache. Appended, so every value above keeps its number and a
    // scene file or a `--debug` from before still selects what it used to.
    //
    // These are the SDK's HashGridDebug* family, and they exist for the reason
    // it gives: every parameter of a hash grid is invisible in the final image
    // until it is wrong, and then it is wrong in a way that reads as a shading
    // bug. Voxel size cannot be chosen without seeing the voxels.
    //
    eSharcGrid, ///< a stable colour per voxel, at the primary hit
    eSharcRadiance, ///< what the cache would answer at the primary hit
    eSharcOccupancy, ///< how much of the table is in use, as a screen overlay
    eSharcBounces, ///< bounces traced per pixel: what the cache actually saves
};

enum : uint32_t
{
    DEBUG_MODE_FIRST_AOV = 3u
};

/// The views that describe the first surface a camera ray reaches and nothing
/// past it. They are the ones the path is cut short for, and the ones nothing
/// volumetric may answer for, since a scattering event ends the path somewhere
/// that has no surface to report.
///
/// eSharcGrid belongs here because the grid is arithmetic -- it needs no cache
/// and no second bounce. The other two cache views deliberately do *not*: they
/// observe a table filled by the bounces that early termination would remove.
#define DEBUG_MODE_IS_SINGLE_HIT(d)                                                                                    \
    ((d) == (uint32_t)DebugMode::eNormal || (d) == (uint32_t)DebugMode::eMotionBlur ||                                  \
     (d) == (uint32_t)DebugMode::eSharcGrid)

/// The views whose output is radiance in scene units, and which therefore want
/// the same presentation transform -- exposure, curve, gamma -- as the render.
///
/// Everything else the debug menu offers is already a colour: normals, motion,
/// the denoiser guides, and the cache's grid, occupancy and bounce views. They
/// travel as PresentationContent::DebugDisplayLinear and a curve would only
/// crush them.
///
/// The list is short and has exactly one entry that is not obvious.
/// eSharcRadiance shows what the radiance cache holds, and it holds radiance:
/// it is only worth looking at beside the beauty render at the same exposure,
/// and presented as display-linear every voxel above one is the same white.
/// That it also cuts the path at the first surface, like the views that *are*
/// display-linear, is what made this worth naming separately.
#define DEBUG_MODE_IS_SCENE_LINEAR(d)                                                                                  \
    ((d) == (uint32_t)DebugMode::eNone || (d) == (uint32_t)DebugMode::eSharcRadiance)

/// What a denoiser needs to know about the primary hit, written once per pixel
/// by the program that shades it (or by the miss program, for background).
///
/// This shares Metal's 64-byte semantic contract, but not every field: Metal
/// ends with bounceDepth while OptiX ends with pad2.
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

/// One path's radiance-cache bookkeeping, held per pixel rather than per path.
///
/// One visit per path, at most. What the path gathers after the visit, divided
/// by its throughput there, is that vertex's outgoing radiance, and that is what
/// the deposit at the end of the raygen loop hands the cache.
///
/// It sits in its own buffer, allocated only when the cache is on, because in
/// PerRayData it was 28 bytes of continuation stack on every path in every
/// scene -- including the great majority that never turn the cache on at all.
struct SharcPathState
{
    /// prd.radiance at the moment of the visit, so the difference at the end of
    /// the path is what the path gathered *after* it.
    float3 radianceAtVisit;
    /// 1 / throughput at the visit, floored per channel.
    float3 invThroughput;
    /// How much of what the path gathered after the visit came from a light
    /// marked responsive, in the same units as `radianceAtVisit` -- i.e. before
    /// the division by throughput, so the two subtract cleanly.
    ///
    /// The cache stores this part in a separate entry with a much shorter
    /// temporal window, and the reader adds the two back together. Splitting by
    /// *component* rather than by path is what makes that sum correct: a mean of
    /// (total - responsive) plus a mean of (responsive) is the mean of the
    /// total, while sending whole paths to one entry or the other would make the
    /// two means overlap and the sum count some light twice.
    ///
    /// Only written while responsive lighting is on; the field costs 12 bytes of
    /// a per-pixel buffer that exists only when the cache does.
    float3 responsiveRadiance;
    /// Roughness of the surface this path's current segment was launched from.
    ///
    /// The cache's eligibility test needs the lobe that *sent* the ray, not the
    /// surface it arrived at -- see oka::sharc::mayReadCache. Carried here
    /// rather than in PerRayData because that struct sizes the continuation
    /// stack byte for byte in every scene (docs/open-perf.md), while this buffer
    /// exists only while the cache does. One path per pixel is in flight at a
    /// time, so a per-pixel slot is a per-path slot.
    float launchRoughness;
    /// SHARC_NO_ENTRY until this path visits a voxel.
    uint32_t index;
    /// The slot holding the same voxel's responsive half, resolved at the visit
    /// alongside `index` so the deposit does not have to hash again at the end
    /// of the path. SHARC_NO_ENTRY when responsive lighting is off, or when the
    /// probe run for it was full -- in which case the whole deposit goes to the
    /// ordinary entry, which is the right answer, just a slower-reacting one.
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
    float4* image;
    float4* accum;
    /// The first-event split. Null unless somebody asked for it -- see
    /// Params::writeSplitAov.
    float4* diffuse;
    uint16_t* diffuseCounter;
    float4* specular;
    uint16_t* specularCounter;
    /// Whether to accumulate the diffuse/specular split of the first event.
    ///
    /// Off by default, with no buffers allocated, because no host caller
    /// normally consumes these split AOVs.
    bool writeSplitAov;
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

    // --- Atmosphere ------------------------------------------------------
    //
    // The scene's `atmosphere` sidecar: a homogeneous slab below `fogHeight`,
    // with the same field meanings as MetalFrameUniforms.
    bool hasFog;
    float fogSigmaT; ///< extinction, per world unit
    float fogAnisotropy; ///< Henyey-Greenstein g; > 0 scatters forward
    float fogHeight; ///< the medium fills everything below this y
    float3 fogAlbedo; ///< single-scattering albedo: what an event scatters rather than absorbs

    /// The three ways the nested-dielectric stack loses a path, counted per
    /// launch: see IOR_STAT_* below. Null when the buffer has not been
    /// allocated, and then the counting is skipped rather than guessed at.
    /// Reports stack overflow, unmatched nested-dielectric exits, and paths that
    /// escape while still marked inside a dielectric.
    uint32_t* iorStats;
    // --- Radiance cache --------------------------------------------------
    //
    // Mirrors the four sharc* fields of Metal's Uniforms, and means the same
    // things. `sharcCapacity == 0` disables the cache outright, and nothing else
    // here is read when it is zero -- which is what makes the default a
    // byte-for-byte no-op rather than a path that happens to agree.
    SharcEntry* sharcEntries;
    /// One record per pixel. Null, and never touched, when sharcCapacity is 0 --
    /// which is a compile-time constant, so the whole cache path folds away.
    SharcPathState* sharcPath;
    uint32_t sharcCapacity; ///< entries; a power of two, 0 = off
    uint32_t sharcMinSamples; ///< deposits a voxel needs before it may be read
    uint32_t sharcDepth; ///< first bounce allowed to read the cache
    /// Stop *reading* the cache once this many samples have accumulated; 0 is
    /// no limit. Deposits carry on regardless, so the cache stays warm for the
    /// moment the camera moves again.
    ///
    /// Cache error eventually becomes the convergence floor, so reads stop
    /// after the configured crossover. This remains runtime state because it
    /// changes per frame.
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

// The three ways the nested-dielectric stack loses a path, in the order Metal's
// ShaderTypes.h numbers them so the two backends' reports read the same.
//
// A push onto a full stack wants a deeper stack. An unmatched pop and a path
// that reaches the environment still inside a medium are both a mesh with a
// hole in it, seen from each side -- and the third is the one no exit event can
// catch, because the ray left through the hole. Each of them carries the wrong
// medium, and therefore the wrong absorption, for the rest of its life.
enum : uint32_t
{
    IOR_STAT_OVERFLOW = 0,
    IOR_STAT_UNMATCHED = 1,
    IOR_STAT_ESCAPED_INSIDE = 2,
    IOR_STAT_COUNT = 3
};

/// How many payload registers the pipeline is compiled with. Two, holding a
/// packed pointer to PerRayData -- everything else the path carries is behind
/// that pointer.
///
/// More payload registers increased continuation-stack use and were reverted;
/// see docs/open-perf.md.
enum : uint32_t
{
    STRELKA_PAYLOAD_COUNT = 2
};

/// Everything one path carries between traversals.
///
/// The raygen holds this across `optixInvoke`, so its size directly determines
/// per-thread continuation-stack local state.
///
/// Hence the layout. The small fields are bit fields in one word and they sit
/// together rather than between the float3s, which is worth more than it looks:
/// interleaved, each one cost four bytes plus its own padding.
///
/// Two things are deliberately *not* here. `sampler.sampleIdx` is the path's
/// sample index, so nothing duplicates it; and which pixel the path belongs to
/// is answered by launchPixelIndex(), because every program runs under the
/// launch index of the ray that started it.
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

    // --- Counters and flags: one word ------------------------------------
    //
    // Bit fields rather than nine separate members. Laid out as bytes and bools
    // these cost twelve bytes of the record and its padding; the widths below
    // are each field's documented bound and they add to exactly 32 bits. The
    // mask-and-shift cost is preferable to a larger local-memory record.
    /// Bounce. `params.max_depth` is clamped to 255 so that the closest hit's
    /// "stop this path" idiom -- setting depth to max_depth, which the raygen
    /// then increments once more -- cannot overflow nine bits.
    uint32_t depth : 9;
    /// Steps the current walk has taken. Reset when a medium is entered, and
    /// bounded by Params::subsurfaceIterations, itself clamped to
    /// MEDIUM_MAX_STEPS (256).
    uint32_t mediumStep : 9;
    /// How many transparent surfaces this path has already passed straight
    /// through. Counted separately from `depth` because a cutout is coverage,
    /// not scattering -- charging it a bounce empties the path budget on a
    /// canopy before any light transport happens. Bounded by
    /// PATH_PASSTHROUGH_MAX (32), so six bits.
    uint32_t passthrough : 6;
    /// An EventType, held as bits because a bit field of enum type is not one
    /// storage unit with the rest on every compiler this builds under. Read and
    /// written through firstEventType() / setFirstEventType().
    uint32_t firstEventBits : 3;
    uint32_t specularBounce : 1;
    /// Whether the vertex this ray left performed a next-event estimate. False
    /// under estimatorMode 1, and false when the scene has nothing to connect to;
    /// in both cases the BSDF strategy owns the whole contribution and the miss /
    /// light-hit shaders must not apply a MIS weight against an estimate that was
    /// never made.
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

    // --- Sampling ---------------------------------------------------------
    //
    /// `sampler.depth` is the sampler's dimension offset. It is not `depth`
    /// above: the closest hit overloads that one as the path's stop signal, and
    /// unifying them would move the sampler's dimension when a path terminates.
    SamplerState sampler;
    /// Where in the image this path's camera ray actually went, y down and in
    /// pixels, jitter included. A motion vector is the difference between this
    /// and where the same surface point sat last frame; differencing against the
    /// pixel centre instead leaves the jitter inside every vector, which is a
    /// subpixel wobble on every pixel of a perfectly still image.
    float2 pixelSample;

    IorStack iorStack;

    // --- Participating media ---------------------------------------------
    /// Which medium the path is inside: 0 for none, otherwise the material index
    /// plus one. Same encoding as the low bits of Metal's PathState::medium.
    ///
    /// A camera that starts inside a translucent object is not handled -- there
    /// is nothing to tell the path which medium it is in.
    uint32_t medium;
    /// The walk's single-scattering albedo, resolved at the boundary the path
    /// entered through, because that is the last place a texture exists: inside
    /// the medium there is no surface to sample. A bounded volume has no entry
    /// surface to have textured and keeps the material's constant instead.
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

/// The 136-byte size is continuation-stack ABI; adding a field increases
/// per-thread local state.
static_assert(sizeof(PerRayData) == 136, "PerRayData sizes the continuation stack; see docs/open-perf.md");

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
    int32_t materialId;  // index into params.materials[] and params.materialTextures[]
    /// Segments per strand for a curve set, or 0 when the strands differ in
    /// length. It is the whole of what a root-to-tip UV needs: the segment index
    /// modulo this is where along the strand a hit landed, for no extra memory.
    uint32_t curveSegmentsPerStrand;
};
