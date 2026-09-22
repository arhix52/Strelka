#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>
#ifndef __METAL_VERSION__
#    ifdef __cplusplus
#        include <Metal/MTLTypes.hpp>
#    endif
#endif

#include <strelka/material/openpbr/openpbr_params.h>
#include <emissive_mesh_light.h>
#include <restir_reservoir.h>

#define GEOMETRY_MASK_TRIANGLE 1
#define GEOMETRY_MASK_CURVE 2
#define GEOMETRY_MASK_LIGHT 4
#define GEOMETRY_MASK_LIGHT_HIDDEN 8
// The boundary of a participating medium. Its own bit because a shadow ray must
// not be stopped by it -- RAY_MASK_SHADOW is the geometry bits alone, so a fog
// gizmo left on the triangle mask would black out everything it encloses.
#define GEOMETRY_MASK_MEDIUM 16
#define GEOMETRY_MASK_UNIFORM_ORTHOGONAL_TRANSFORM 128

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_CURVE)

#define RAY_MASK_PRIMARY (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_MEDIUM)
#define RAY_MASK_SHADOW (GEOMETRY_MASK_GEOMETRY)
#define RAY_MASK_SECONDARY (RAY_MASK_PRIMARY | GEOMETRY_MASK_LIGHT_HIDDEN)

#define ANALYTIC_INTERSECTION_SPHERE 0
#define ANALYTIC_INTERSECTION_DISC 1
#define CUTOUT_INTERSECTION_SHADOW 2
#define ANALYTIC_INTERSECTION_FUNCTION_COUNT 2
#define SHADOW_INTERSECTION_FUNCTION_COUNT 3

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

struct OpenPBRTextures
{
#ifdef __METAL_VERSION__
    metal::texture2d<float> tex[MAX_OPENPBR_TEXTURES];
#else
    MTL::ResourceID tex[MAX_OPENPBR_TEXTURES];
#endif
};
static_assert(sizeof(OpenPBRTextures) == MAX_OPENPBR_TEXTURES * 8, "OpenPBR texture handles must stay eight bytes");

enum class DebugMode : uint32_t
{
    eNone = 0,
    eNormal,
    eMotionBlur,
    // Denoiser guides, visualised by the resolve pass. Selecting any of these
    // turns their production on, so they cost nothing when not being looked at.
    eAovDiffuseAlbedo,
    eAovSpecularAlbedo,
    eAovNormal,
    eAovRoughness,
    eAovDepth,
    eAovMotion,
    // Guides whose visualization enables their production.
    eAovReactive,
    eAovSpecularHitDistance,
    eSharcGrid,
    eSharcRadiance,
    eSharcOccupancy,
    eSharcBounces,
};

#define DEBUG_MODE_FIRST_AOV 3
#define DEBUG_MODE_LAST_AOV ((uint32_t)DebugMode::eAovSpecularHitDistance)
#define DEBUG_MODE_IS_AOV(d) ((d) >= DEBUG_MODE_FIRST_AOV && (d) <= DEBUG_MODE_LAST_AOV)
// Debug views answered by the first surface a camera ray meets. The background
// and any emitter along the way have to stay out of them, or the one number the
// view exists to show competes with the sky for the pixel.
#define DEBUG_MODE_IS_SINGLE_HIT(d)                                                                                    \
    ((d) == (uint32_t)DebugMode::eNormal || (d) == (uint32_t)DebugMode::eMotionBlur ||                                 \
     (d) == (uint32_t)DebugMode::eSharcGrid || (d) == (uint32_t)DebugMode::eSharcRadiance)

struct Vertex
{
    packed_float3 pos;
    uint32_t tangent;

    uint32_t normal;
    uint32_t uv;
    uint32_t uv1;
    uint32_t color;
};
static_assert(sizeof(Vertex) == 32, "Vertex must match Scene::Vertex");

struct SurfaceGeometryPayload
{
    uint32_t shadingNormal;
    uint32_t geometryNormal;
    uint32_t tangentAndFlags;
    uint32_t uvAndLod;
};
static_assert(sizeof(SurfaceGeometryPayload) == 16, "Surface geometry payload must stay compact");

struct BaseLightConnectionPayload
{
    packed_float3 radiance;
    packed_float3 toLight;
    packed_float3 origin;
    packed_float3 visibilityTarget;
    float pdf;
    float tMax;
    uint32_t flags;
};
static_assert(sizeof(BaseLightConnectionPayload) == 60, "Base light connection payload size changed");

#define BASE_LIGHT_CONNECTION_NEEDS_RAY (1u << 0)
#define BASE_LIGHT_CONNECTION_VISIBILITY_TARGET (1u << 1)
#define BASE_LIGHT_CONNECTION_DELTA (1u << 2)

#define SURFACE_GEOMETRY_VALID (1u << 31)
#define SURFACE_GEOMETRY_TANGENT_NEGATIVE (1u << 30)

#define TEXTURE_LOD_MODE_MASK 0xffu
#define RECONSTRUCTION_FILTER_SHIFT 8u
#define RECONSTRUCTION_FILTER_MASK (0xffu << RECONSTRUCTION_FILTER_SHIFT)
#define RECONSTRUCTION_FILTER_BOX 0u
#define RECONSTRUCTION_FILTER_MITCHELL 1u
#define RECONSTRUCTION_FILTER_TENT 2u
#define RECONSTRUCTION_FILTER_LANCZOS2 3u

struct PrimitiveSurfaceData
{
    uint32_t normal[3];
    uint32_t geometryNormal;
};
static_assert(sizeof(PrimitiveSurfaceData) == 16, "Primitive surface data ABI changed");

#define PRIMITIVE_ALPHA_BLOCK_SHIFT 8u
#define PRIMITIVE_ALPHA_BLOCK_SIZE (1u << PRIMITIVE_ALPHA_BLOCK_SHIFT)
struct PrimitiveAlphaData
{
    uint32_t u;
    uint32_t v;
};
static_assert(sizeof(PrimitiveAlphaData) == 8, "Primitive alpha data ABI changed");

struct PrimitiveAlphaDecode
{
    // (u offset, v offset, u span, v span). The span includes RGB10's
    // normalisation, so shader reconstruction is two FMAs after interpolation.
    vector_float4 offsetScale;
};
static_assert(sizeof(PrimitiveAlphaDecode) == 16, "Primitive alpha decode ABI changed");

struct Uniforms
{
    simd::float4x4 viewToWorld;
    simd::float4x4 clipToView;
    simd::float4x4 prevViewToWorld;
    simd::float4x4 prevClipToView;
    vector_float3 missColor;

    uint32_t width;
    uint32_t height;
    /// Monotonic display-frame index. Used to keep one-sample interactive
    /// frames statistically independent when accumulation is disabled.
    uint32_t frameIndex;
    uint32_t subframeIndex;

    uint32_t numLights;
    uint32_t numEmissiveMeshes;
    // Probability of the mesh-emitter class conditioned on selecting a local
    // light. The complement selects the analytic-light alias table.
    float meshLightSelectionPdf;
    uint32_t enableAccumulation;
    uint32_t samples_per_launch;
    uint32_t maxDepth;

    uint32_t rectLightSamplingMethod;
    // VOLUME_MODEL_GLTF or VOLUME_MODEL_CYCLES; see volume.h for why this is a
    // setting rather than a constant.
    uint32_t volumeModel;
    // 0 - Halton, 1 - PCG, 2 - Sobol (Owen), 3 - Sobol + blue noise, 4 - hybrid,
    // 5 - Owen + VDC without sb_matrix (ablation)
    uint32_t samplerType;
    /// Sample count at which sampler 4 hands the frame from the blue-noise
    /// sequence to the per-pixel scrambled one.
    uint32_t blueNoiseSwitchSpp;

    uint32_t tonemapperType; // 0 - "None", "Reinhard", "ACES", "Filmic"
    float gamma; // 0 - off
    vector_float3 exposureValue;

    uint32_t debug;

    uint32_t enableMotionBlur;
    uint32_t isMotionBlurVisible;
    uint32_t enableCameraMotionBlur;

    // Depth of field
    int32_t useDof;
    float focalDistance;
    float lensRadius;
    int32_t apertureBlades;
    float bladeRotation;
    float anamorphicRatio;

    // Lens shift
    float shiftX;
    float shiftY;

    uint32_t projectionType;
    float orthoHalfWidth;
    float orthoHalfHeight;

    // Environment map (dome light)
    uint32_t hasEnvMap;
    // Atmospheric scattering, homogeneous below fogHeight. See fog.h for why a
    // slab and not a bounded volume.
    uint32_t hasFog;
    uint32_t hasBoundedMedium;
    // Scene-bounds diagonal caps subsurface free flights; longer draws leave the bounded medium.
    float sceneExtent;
    // Sparse Hash Radiance Cache (SHARC); see sharc.h. The cache is updated by
    // a sparse path pass, resolved, and only then queried by the image pass.
    uint32_t sharcCapacity; // 0 disables
    uint32_t sharcMinSamples; // before a voxel may be read
    uint32_t sharcDepth; // first bounce allowed to read the cache
    uint32_t sharcFlags;
    float sharcBaseSize; // world size of the requested perspective footprint at unit distance
    float sharcRoughnessThreshold;
    float sharcRadianceScale;
    uint32_t sharcUpdateDownscale;
    uint32_t sharcAccumulationFrames;
    uint32_t sharcResponsiveFrames;
    uint32_t sharcStaleFrameCount;
    uint32_t sharcPropagationDepth;
    uint32_t sharcDebug;
    uint32_t sharcUpdatePathCount;
    uint32_t sharcFrameIndex;
    int32_t sharcLevelBias;
    vector_float3 sharcCameraPrev;
    float fogSigmaT;
    float fogAnisotropy;
    float fogHeight;
    vector_float3 fogAlbedo;
    // A separate environment for camera rays. See Scene::EnvLightDesc.
    uint32_t hasEnvBackground;
    float envBackgroundIntensity;
    uint32_t envMapWidth;
    uint32_t envMapHeight;
    float envMapIntensity;
    float envMapRotation;
    uint32_t estimatorMode;
    // Ray mask for camera/secondary hardware traversal. `numLights` separately
    // gates smooth analytic-light intersections performed in the extend kernel.
    uint32_t primaryRayMask;
    // RGB tint plus the probability of choosing the environment over analytic
    // lights. float4 uses the same 16-byte slot float3 occupied, so this adds no
    // uniform traffic.
    vector_float4 envMapColorTint;
    // Previous frame's world-to-clip, for reprojecting a hit point into the last
    // frame's screen space. The motion-blur matrices are the inverses and cannot
    // be used for this.
    simd::float4x4 prevWorldToClip;
    // This frame's world-to-clip. Needed for the device-depth guide, which cannot
    // be derived from the inverses the camera-ray path carries.
    simd::float4x4 worldToClip;
    uint32_t writeAov;
    // Which convention the depth guide is written in; see kDenoiseDepth* below.
    uint32_t denoiseDepthMode;
    // Whether the previous frame's pose and instance transforms are valid to
    // read. False on the first frame of a scene and after anything that
    // invalidates the correspondence between frames.
    uint32_t hasPrevFramePose;
    /// Luminance ceiling, in exposed units, for the colour handed to the
    /// denoiser. Zero disables it.
    float denoiseFireflyClamp;
    float clampIndirect;
    // Sub-pixel offset applied to every pixel of this frame, in pixels. Temporal
    // upscaling needs the whole image shifted by a known amount it can undo; the
    // per-pixel random jitter that antialiases a still frame is noise to it.
    float jitterX;
    float jitterY;
    uint32_t useFrameJitter;
    /// Light candidates drawn per shading point before one is resampled -- the
    /// M of resampled importance sampling. One is plain next-event estimation
    /// and the arithmetic reduces to exactly what it was.
    uint32_t risCandidates;
    /// Which MIS heuristic weighs the two strategies: 0 = balance, 1 = power.
    uint32_t misHeuristic;
    /// Low byte: 0 = level 0, 1 = ray-cone LOD. The next byte stores the
    /// reconstruction filter without growing this hot, shared ABI block.
    uint32_t textureLodMode;
    uint32_t guidePrimaryHit;

    uint32_t restirDIEnabled;
    uint32_t initialCandidateCount;
    uint32_t temporalReuseEnabled;
    uint32_t spatialReuseEnabled;
    uint32_t spatialNeighborCount;
    uint32_t reservoirMaxAge;
    uint32_t restirDebugMode;
    uint32_t restirHistoryValid;
    uint32_t restirBiasCorrection;
    uint32_t restirInitialVisibility;
    uint32_t restirFinalVisibilityReuse;
    uint32_t restirFinalVisibilityMaxAge;
    uint32_t restirVisibilityRevision;
    /// Maximum scattering events in one dense subsurface random walk. This
    /// occupies the former ReSTIR alignment word, preserving the uniform ABI.
    uint32_t subsurfaceIterations;
    float restirProposalCollision;
    float restirProposalEntropy;

#ifdef __METAL_VERSION__
    device const OpenPBRParams* openpbrParams;
#else
    uint64_t openpbrParams;
#endif

    /// The bindless maps for material i, or null when no OpenPBR material in the
    /// scene has any. Same addressing story as openpbrParams above: no binding
    /// slot is free, so the table is reached by address and made resident by hand.
#ifdef __METAL_VERSION__
    device const OpenPBRTextures* openpbrTextures;
#else
    uint64_t openpbrTextures;
#endif

    /// Per-pixel continuation rays written by the primary radiance shade.
    /// Reached by address because wavefrontShade already uses all 31 bindings.
#ifdef __METAL_VERSION__
    device struct GuideRay* guideRays;
#else
    uint64_t guideRays;
#endif

    /// Compact geometry prepared by extend for the matching path slot. Like
    /// guideRays, this is addressed through the uniform block because shade
    /// already occupies every explicit Metal 4 buffer binding.
#ifdef __METAL_VERSION__
    device SurfaceGeometryPayload* surfaceGeometry;
#else
    uint64_t surfaceGeometry;
#endif

#ifdef __METAL_VERSION__
    device const EmissiveMeshLight* emissiveMeshes;
    device const EmissiveTriangleLight* emissiveTriangles;
#else
    uint64_t emissiveMeshes;
    uint64_t emissiveTriangles;
#endif

#ifdef __METAL_VERSION__
    device struct RestirReservoir* restirReservoir0;
    device struct RestirReservoir* restirReservoir1;
    device struct RestirSurfaceHistory* restirHistory0;
    device struct RestirSurfaceHistory* restirHistory1;
    device char* restirSurfaceData0;
    device char* restirSurfaceData1;
#else
    uint64_t restirReservoir0;
    uint64_t restirReservoir1;
    uint64_t restirHistory0;
    uint64_t restirHistory1;
    uint64_t restirSurfaceData0;
    uint64_t restirSurfaceData1;
#endif

    // Present only in the explicitly requested render-work audit variant. The
    // ordinary pipeline specialises every reference away.
#ifdef __METAL_VERSION__
    device atomic_uint* renderWorkCounters;
    device char* sharcHashData;
    device const char* curvePointData;
#else
    uint64_t renderWorkCounters;
    uint64_t sharcHashData;
    uint64_t curvePointData;
#endif

    uint32_t numInfiniteLights;
    /// log2 of the power-of-two sample block reserved for each pixel by the
    /// padded Sobol sampler. Occupies what was alignment padding before
    /// infiniteLightIndices.
    uint32_t sobolSampleBlockBits;
#ifdef __METAL_VERSION__
    device const uint32_t* infiniteLightIndices;
#else
    uint64_t infiniteLightIndices;
#endif
// Active guide pixel indices. Built by first-hit shading and consumed by one
// indirect guide dispatch, so scenes without guide continuations launch no grid.
#ifdef __METAL_VERSION__
    device uint32_t* guideQueue;
    device uint32_t* restirQueue;
#else
    uint64_t guideQueue;
    uint64_t restirQueue;
#endif

#ifdef __METAL_VERSION__
    device char* previousLights;
    device const uint32_t* previousToCurrentLight;
    device const uint32_t* currentToPreviousLight;
#else
    uint64_t previousLights;
    uint64_t previousToCurrentLight;
    uint64_t currentToPreviousLight;
#endif
    uint32_t previousNumLights;
    uint32_t previousNumEmissiveMeshes;
    float previousMeshLightSelectionPdf;
    float previousEnvSelectionPdf;
    uint32_t restirEnvironmentHistoryValid;
    uint32_t restirMeshHistoryValid;
    // Scratch proposals for the split Base NEE path. Kept at the end so adding
    // it does not shift the ABI of the existing uniform block.
#ifdef __METAL_VERSION__
    device BaseLightConnectionPayload* baseLightConnections;
#else
    uint64_t baseLightConnections;
#endif
    /// First transform-only record after the real TLAS instance descriptors.
    /// A world-space baked geometry indexes this tail by GeometryEntry id.
    uint32_t geometryTransformBase;
    // Consumes padding before the 16-byte-aligned camera basis.
    float textureLodBias;
    // Perspective camera unprojection expanded on the CPU. Appended so the
    // existing pointer offsets above remain stable across the shared ABI.
    vector_float3 cameraRayRight;
    vector_float3 cameraRayUp;
    vector_float3 cameraRayForward;
    // Append-only fields keep every existing resource pointer at its stable
    // host/Metal ABI offset.
    vector_float2 envMapRotationSinCos;
#ifdef __METAL_VERSION__
    device const packed_float2* envRowCosBounds;
    device const float* envPdfTable;
#else
    uint64_t envRowCosBounds;
    uint64_t envPdfTable;
#endif
    // Exact invariant division by width. samplerFor uses this to recover pixel
    // x/y without issuing two dynamic integer divides on every stage.
    uint32_t widthDivMultiplier;
    uint32_t widthDivShiftAdd;
};
static_assert(sizeof(Uniforms) == 1136, "Uniforms host/Metal ABI changed");

enum RenderWorkCounter : uint32_t
{
    WORK_PRIMARY_RAYS = 0,
    WORK_EXTEND_RAYS_BASE = 1,
    WORK_SHADE_ITEMS_BASE = 17,
    WORK_SHADOW_RAYS_BASE = 33,
    WORK_MISS_ITEMS_BASE = 49,
    WORK_GUIDE_ONLY_RAYS = 65,
    WORK_INTERSECTION_QUERIES = 66,
    WORK_RESTIR_ELIGIBLE_HITS = 67,
    WORK_RESTIR_INITIAL_CANDIDATES = 68,
    WORK_RESTIR_TEMPORAL_MERGES = 69,
    WORK_RESTIR_SPATIAL_MERGES = 70,
    WORK_RESTIR_FINAL_VISIBILITY_RAYS = 71,
    WORK_FIRST_BOUNCE_NEE_SAMPLES = 72,
    WORK_SECONDARY_NEE_SAMPLES = 73,
    WORK_RESTIR_SPATIAL_ITEMS = 74,
    WORK_RESTIR_FINAL_ITEMS = 75,
    WORK_MANUAL_ANALYTIC_LIGHT_TESTS = 76,
    WORK_GUIDE_ACTIVE_ITEMS = 77,
    WORK_RESTIR_CANDIDATE_QUERIES = 78,
    WORK_RESTIR_REUSE_QUERIES = 79,
    WORK_MISS_LIGHT_EVALUATIONS = 80,
    WORK_GUIDE_DISPATCHES = 81,
    WORK_RESTIR_DIAGNOSTIC_QUERIES = 82,
    WORK_RESTIR_EFFECTIVE_M = 83,
    WORK_RESTIR_TEMPORAL_REJECT_SURFACE = 84,
    WORK_RESTIR_TEMPORAL_REJECT_UNMAPPED = 85,
    WORK_RESTIR_TEMPORAL_REJECT_TYPE = 86,
    WORK_RESTIR_TEMPORAL_REJECT_ENVIRONMENT = 87,
    WORK_RESTIR_TEMPORAL_REJECT_MESH = 88,
    WORK_RESTIR_TEMPORAL_SELECTED = 89,
    WORK_RESTIR_TEMPORAL_TARGET_POSITIVE = 90,
    WORK_RESTIR_TEMPORAL_DUPLICATE_CURRENT = 91,
    WORK_RESTIR_TEMPORAL_SELECTED_DUPLICATE_CURRENT = 92,
    WORK_RESTIR_FINAL_SOURCE_INITIAL = 93,
    WORK_RESTIR_FINAL_SOURCE_TEMPORAL = 94,
    WORK_RESTIR_FINAL_SOURCE_SPATIAL = 95,
    WORK_RESTIR_FINAL_HISTORY = 96,
    WORK_RESTIR_FINAL_HISTORY_RAYS = 97,
    WORK_RESTIR_FINAL_HISTORY_VISIBLE = 98,
    WORK_RESTIR_AGE_HISTOGRAM_BASE = 99,
    WORK_RESTIR_M_HISTOGRAM_BASE = 121,
    WORK_RESTIR_TARGET_RATIO_HISTOGRAM_BASE = 129,
    WORK_RESTIR_INITIAL_VISIBILITY_QUERIES = 138,
    WORK_RESTIR_TEMPORAL_SAME_LIGHT = 139,
    WORK_RESTIR_INITIAL_VISIBILITY_REUSED = 140,
    WORK_RESTIR_VISIBILITY_CACHE_ATTEMPTS = 141,
    WORK_RESTIR_VISIBILITY_CACHE_HITS = 142,
    WORK_RESTIR_VISIBILITY_CACHE_REJECT_EMPTY = 143,
    WORK_RESTIR_VISIBILITY_CACHE_REJECT_AGE = 144,
    WORK_RESTIR_VISIBILITY_CACHE_REJECT_REVISION = 145,
    WORK_RESTIR_VISIBILITY_CACHE_REJECT_RECEIVER = 146,
    WORK_RESTIR_VISIBILITY_CACHE_REJECT_LIGHT = 147,
    WORK_RESTIR_VISIBILITY_CACHE_ORACLE_QUERIES = 148,
    WORK_RESTIR_VISIBILITY_CACHE_VISIBLE_VISIBLE = 149,
    WORK_RESTIR_VISIBILITY_CACHE_VISIBLE_OCCLUDED = 150,
    WORK_EXTENSION_QUERIES = 151,
    WORK_NEE_ELIGIBLE_HITS = 152,
    WORK_NEE_DELTA_HITS = 153,
    WORK_NEE_LIGHT_SAMPLER_CALLS = 154,
    WORK_NEE_FINITE_LIGHT_INSPECTIONS = 155,
    WORK_NEE_EMISSIVE_LIGHT_INSPECTIONS = 156,
    WORK_NEE_ENVIRONMENT_SAMPLES = 157,
    WORK_NEE_CONDITIONAL_PDF_EVALUATIONS = 158,
    WORK_NEE_VALID_CANDIDATES = 159,
    WORK_NEE_REJECT_CONNECTION = 160,
    WORK_NEE_REJECT_PDF = 161,
    WORK_NEE_REJECT_COSINE = 162,
    WORK_NEE_REJECT_TARGET = 163,
    WORK_NEE_SHADOW_APPENDS = 164,
    WORK_NEE_SHADOW_REJECT_NORMALIZATION = 165,
    WORK_NEE_SHADOW_REJECT_WEIGHT = 166,
    WORK_NEE_SHADOW_REJECT_SEGMENT = 167,
    WORK_ENVIRONMENT_EVALUATIONS = 168,
    WORK_PATH_CONTINUATIONS = 169,
    WORK_QUEUE_APPENDS = 170,
    WORK_QUEUE_OVERFLOWS = 171,
    WORK_ALPHA_TRAVERSAL_RESTARTS = 172,
    WORK_COUNTER_COUNT = 173,
    WORK_BOUNCE_SLOTS = 16
};

#define RESTIR_AUDIT_AGE_BINS 22u
#define RESTIR_AUDIT_M_BINS 8u
#define RESTIR_AUDIT_TARGET_RATIO_BINS 9u
#define RESTIR_AUDIT_LIGHT_ID_WORDS 128u
#define RESTIR_AUDIT_HISTORY_BIT 0x80000000u
#define RESTIR_VISIBILITY_UPDATE_BIT 0x40000000u
#define RESTIR_AUDIT_PATH_INDEX_MASK 0x3fffffffu

#define RESTIR_DIAGNOSTIC_PIXEL_COUNT 32u
#define RESTIR_DIAGNOSTIC_SOURCE_COUNT 3u

struct RestirDiagnosticRecord
{
    uint32_t pixelIndex;
    uint32_t stableLightId;
    uint32_t selectedSourceIndex;
    uint32_t sourceCount;
    uint32_t sourceIndices[RESTIR_DIAGNOSTIC_SOURCE_COUNT];
    uint32_t sourceM[RESTIR_DIAGNOSTIC_SOURCE_COUNT];
    float sourceTargets[RESTIR_DIAGNOSTIC_SOURCE_COUNT];
    float currentTarget;
    float weightSum;
    float basicDenominator;
    float normalization;
    uint32_t finalVisibility;
    float contribution[3];
};
static_assert(sizeof(RestirDiagnosticRecord) == 84, "ReSTIR diagnostic record ABI changed");

struct RestirCandidateAuditRecord
{
    uint32_t pixelIndex;
    uint32_t frameIndex;
    uint32_t sampleIndex;
    uint32_t rngSeed;
    uint32_t rngSampleIndex;
    uint32_t rngDepth;
    uint32_t lightDimension;
    uint32_t initialStreamSeed;
    uint32_t temporalStreamSeed;
    uint32_t spatialStreamSeed;
    RestirLightSample currentSample;
    RestirLightSample historySample;
    RestirLightSample mappedHistorySample;
    uint32_t mappedLightId;
    uint32_t currentBufferIndex;
    uint32_t historyBufferIndex;
    float proposalCollision;
    float proposalEntropy;
};
static_assert(sizeof(RestirCandidateAuditRecord) == 108, "ReSTIR candidate audit ABI changed");

#define PROJECTION_PERSPECTIVE 0u
#define PROJECTION_ORTHOGRAPHIC 1u

#define kDenoiseDepthDevice 0u ///< clip z / w, the value a depth buffer holds
#define kDenoiseDepthViewZ 1u ///< distance along the camera's forward axis
#define kDenoiseDepthRadial 2u ///< distance to the eye

static inline bool denoiseDepthReversed(uint32_t depthMode, uint32_t projectionType)
{
    return depthMode == kDenoiseDepthDevice && projectionType == PROJECTION_ORTHOGRAPHIC;
}

static inline float denoiseBackgroundDepth(uint32_t depthMode, uint32_t projectionType)
{
    if (depthMode != kDenoiseDepthDevice)
    {
        return 1e7f;
    }
    return denoiseDepthReversed(depthMode, projectionType) ? 0.0f : 1.0f;
}

struct AovSample
{
    packed_float3 diffuseAlbedo;
    float depth; // encoding selected by Uniforms::denoiseDepthMode
    packed_float3 specularAlbedo;
    float roughness;
    packed_float3 normal; // world space
    float motionX; // previous-frame screen position minus current, in pixels
    float motionY;
    /// Distance from the primary hit to what its specular lobe sees. MetalFX
    /// reprojects reflections with this instead of treating them as if they sat
    /// on the surface. Zero when the surface is not specular.
    float specularHitDistance;
    /// 0 = trust the history here, 1 = ignore it. Raised where the motion vector
    /// is known to be a lie: mirrors, glass, and anything whose previous position
    /// could not be established.
    float reactive;
    float guideStateOrBounceDepth;
};

struct UniformsTonemap
{
    // Render resolution: the linear radiance buffer is indexed with it.
    uint32_t width;
    uint32_t height;
    // Display resolution. Not the same thing once anything upscales, and the
    // texture-input tonemapper covers this, not the render size.
    uint32_t outWidth;
    uint32_t outHeight;

    uint32_t tonemapperType; // 0 - "None", "Reinhard", "ACES", "Filmic"
    float gamma; // 0 - off
    float maxEDR;
    vector_float3 exposureValue;
};

struct GeometryEntry
{
    uint32_t vbOffset; // mesh vertex buffer offset, or first control point of a curve set
    uint32_t indexOffset; // mesh index buffer offset, or first segment of a curve set
    uint32_t materialId;
    uint32_t flags;
};

#define GEOM_FLAG_CURVE (1u << 31)
#define GEOM_CURVE_CUBIC (1u << 30)
// Scheduling hint only: group similarly expensive BSDFs without a separate
// sort pass. It never selects shading behavior, so a stale hint is harmless.
#define GEOM_SHADE_BUCKET_SHIFT 28u
#define GEOM_SHADE_BUCKET_MASK (3u << GEOM_SHADE_BUCKET_SHIFT)
#define GEOM_FLAG_PRIMITIVE_SURFACE_DATA (1u << 27)
#define GEOM_FLAG_BAKED_TRANSFORM (1u << 26)
#define GEOM_FLAG_SURFACE_UV (1u << 25)
#define GEOM_FLAG_VERTEX_COLOR (1u << 24)
#define GEOM_PRIMITIVE_ALPHA_DATA_INDEX_MASK ((1u << 24) - 1u)
#define GEOM_CURVE_STRAND_MASK 0x0000FFFFu

// Wavefront path state is memory-traffic critical and fixed at 24 bytes; feature-specific state uses side tables.
// Pixel index is the path slot, and PathRay stays separate so extend fetches only traversal data.
struct PathRay
{
    packed_float3 origin;
    packed_float3 direction;
};

#define GUIDE_RAY_ACTIVE (1u << 0)
#define GUIDE_RAY_REPLACE_MATERIAL (1u << 1)

// Cold side state for the small subset of primary hits whose MetalFX guides
// need one or two more intersections. Keeping it separate preserves the
// bandwidth-critical 24-byte PathRay layout used by every radiance bounce.
struct GuideRay
{
    packed_float3 origin;
    uint32_t flags;
    packed_float3 direction;
    /// Two binary16 values: the ray's current medium and the medium outside it.
    uint32_t mediaIors;
};

struct PathState
{
    packed_float3 throughput;
    uint32_t depthAndFlags; // depth in bits 0..7, flags above
    float lastBsdfPdf;
    float misDistance;
};

struct MediumPathState
{
    uint32_t medium;
    uint32_t mediumAlbedo;
};

// The sparse update pass stores only the vertices that can still receive
// radiance. Four is the non-resampling propagation depth used by SHARC; the
// default resampling mode uses the first two records.
#define SHARC_MAX_PROPAGATION_DEPTH 4
struct SharcUpdateState
{
    uint32_t cacheIndices[SHARC_MAX_PROPAGATION_DEPTH];
    uint32_t responsiveIndices[SHARC_MAX_PROPAGATION_DEPTH];
    packed_float3 weights[SHARC_MAX_PROPAGATION_DEPTH];
    packed_float3 directions[SHARC_MAX_PROPAGATION_DEPTH];
    float directionWeights[SHARC_MAX_PROPAGATION_DEPTH];
    packed_float3 pendingThroughput;
    uint32_t pixelIndex;
    uint32_t pathLength;
    uint32_t flags;
};

#define MEDIUM_INDEX_MASK 0xFFFFu
#define MEDIUM_STEP_SHIFT 16u
/// Ceiling on one walk. A dense medium is a long walk that Russian roulette
/// alone terminates slowly, and a path that never ends is a hang rather than a
/// dim pixel.
#define MEDIUM_MAX_STEPS 256u
/// Dense SSS steps kept in registers by one traversal invocation. Eight cuts
/// queue traffic without serializing enough ray queries to hurt occupancy.
#define SSS_FUSED_STEPS 8u

#define SHARC_NO_ENTRY 0xFFFFFFFFu

#define SHARC_FLAG_MATERIAL_DEMODULATION (1u << 0)
#define SHARC_FLAG_SEPARATE_EMISSIVE (1u << 1)
#define SHARC_FLAG_DIRECTIONAL (1u << 2)
#define SHARC_FLAG_RESPONSIVE (1u << 3)
#define SHARC_FLAG_CACHE_RESAMPLING (1u << 4)
#define SHARC_FLAG_BLEND_ADJACENT_LEVELS (1u << 5)
#define SHARC_FLAG_FADE_ACCELERATION (1u << 6)
// Internal resolved-entry classification; not part of Uniforms::sharcFlags.
#define SHARC_RESOLVED_RESPONSIVE_ENTRY (1u << 31)

#define SHARC_STAT_INSERTION 0u
#define SHARC_STAT_INSERTION_FAILURE 1u
#define SHARC_STAT_QUERY_ATTEMPT 2u
#define SHARC_STAT_QUERY_HIT 3u
#define SHARC_STAT_EVICTION 4u
#define SHARC_STAT_COLLISION 5u
#define SHARC_STAT_SEGMENT_REJECT 6u
#define SHARC_STAT_FOOTPRINT_REJECT 7u
#define SHARC_STAT_RECEIVER_REJECT 8u
#define SHARC_STAT_ACCUMULATION_CLAMP 9u
#define SHARC_STAT_NONFINITE_REJECT 10u
#define SHARC_STAT_MAX_RADIANCE_FIXED 11u
#define SHARC_STAT_MAX_SAMPLE_COUNT 12u
#define SHARC_STAT_COUNT 13u
#define SHARC_HASH_ENTRY_STRIDE 4u
#define SHARC_ACCUMULATION_ENTRY_STRIDE 32u
#define SHARC_RESOLVED_ENTRY_STRIDE 32u

#define SHARC_DEBUG_OFF 0u
#define SHARC_DEBUG_CACHED_KEY 1u
#define SHARC_DEBUG_QUERY_RESULT 2u
#define SHARC_DEBUG_SAMPLE_COUNT 3u
#define SHARC_DEBUG_COUNTERS 4u
#define SHARC_DEBUG_COLLISIONS 5u
#define SHARC_DEBUG_LAST_VISUALIZATION SHARC_DEBUG_COLLISIONS

// Views answered by one surface. They replace the image rather than adding to
// it, so nothing else -- sky, emitter, or a later bounce -- may write the pixel.
#define SHARC_DEBUG_IS_SURFACE_VIEW(d)                                                                                 \
    ((d) != SHARC_DEBUG_OFF && (d) != SHARC_DEBUG_COUNTERS && (d) <= SHARC_DEBUG_LAST_VISUALIZATION)

#define IOR_STAT_OVERFLOW 0
#define IOR_STAT_UNMATCHED 1
#define IOR_STAT_ESCAPED_INSIDE 2
#define IOR_STAT_COUNT 3

#define PATH_FLAG_ALIVE (1u << 8)
#define PATH_FLAG_SPECULAR (1u << 9)
#define PATH_FLAG_NEE_DONE (1u << 10)
#define PATH_FLAG_AOV_DONE (1u << 11)
#define PATH_DEPTH_MASK 0xFFu
// Transparent hits are counted apart from bounces: passing through a cutout is
// not a scattering event and must not consume path depth. Bits 12+ are free.
#define PATH_PASSTHROUGH_SHIFT 12u
#define PATH_PASSTHROUGH_MASK (0x3fu << PATH_PASSTHROUGH_SHIFT)
#define PATH_PASSTHROUGH_MAX 32u
// Origin-lobe roughness for the SHARC footprint test. Pass-through uses bits
// 12..17, leaving this byte cold in the existing flags word.
#define PATH_SHARC_ROUGHNESS_SHIFT 18u
#define PATH_SHARC_ROUGHNESS_MASK (0xffu << PATH_SHARC_ROUGHNESS_SHIFT)
// Avoid touching the strided 36-byte IOR side table for the common path that
// has never entered a solid dielectric. The remaining high bits are free.
#define PATH_FLAG_IOR_STACK_ACTIVE (1u << 26)
static_assert((PATH_PASSTHROUGH_MASK & PATH_SHARC_ROUGHNESS_MASK) == 0u,
              "path pass-through count overlaps packed roughness");
static_assert((PATH_FLAG_IOR_STACK_ACTIVE & (PATH_PASSTHROUGH_MASK | PATH_SHARC_ROUGHNESS_MASK)) == 0u,
              "path IOR flag overlaps packed path data");

struct HitRecord
{
    // Triangle barycentrics (or curve t in x) as unorm16x2. The 1/65535 step is
    // below the source normal/UV quantisation and saves four streamed bytes per
    // hit without reducing any scene index range.
    uint32_t barycentrics;
    uint32_t geomEntryIndex; // instance userID + intersection.geometry_id
    // Carried because a BLAS may be shared by instances while only the intersection identifies the TLAS instance.
    uint32_t instanceIndex;
    uint32_t primitiveId;
    float distance; // < 0 means the ray escaped
};
static_assert(sizeof(HitRecord) == 20, "HitRecord must stay a compact wavefront record");

#define RESTIR_SURFACE_VALID (1u << 31)
#define RESTIR_SURFACE_MATERIAL_MASK 0x7fffffffu

struct RestirReservoir
{
    RestirLightSample sample;
    RestirReservoirState state;
    RestirVisibilityCache visibility;
};
static_assert(sizeof(RestirReservoir) == 40, "ReSTIR reservoir ABI changed");

struct RestirSurfaceHistory
{
    packed_float3 geometryNormal;
    float depth;
    uint32_t materialIdAndFlags;
};
static_assert(sizeof(RestirSurfaceHistory) == 20, "ReSTIR surface history ABI changed");

#define RESTIR_SHADING_VALID (1u << 31)
#define RESTIR_SHADING_CURVE (1u << 30)
#define RESTIR_SHADING_REPROJECTABLE (1u << 29)
#define RESTIR_SHADING_SAMPLE_MASK 0x1fffffffu

struct RestirShadingPoint
{
    packed_float3 position;
    packed_float3 shadingNormal;
    packed_float3 tangent;
    packed_float3 rayDirection;
    packed_float3 vertexColor;
    packed_float3 throughput;
    vector_float2 uv;
    float exteriorIor;
    float lodBase;
    float curveRadius;
    float tangentSign;
    uint32_t medium;
    uint32_t sampleIdxAndFlags;
};
static_assert(sizeof(RestirShadingPoint) == 104, "ReSTIR shading point ABI changed");

#define RESTIR_TARGET_SAMPLE_MASK 0xffffu
#define RESTIR_TARGET_MEDIUM_SHIFT 16u
#define RESTIR_TARGET_DIRECT (1u << 31)
#define RESTIR_TARGET_FRONT_FACE (1u << 30)
#define RESTIR_TARGET_THIN_WALLED (1u << 29)
#define RESTIR_TARGET_HAS_TRANSMISSION (1u << 28)
#define RESTIR_TARGET_HAS_DIFFUSE_TRANSMISSION (1u << 27)
#define RESTIR_TARGET_MATERIAL_SHIFT 24u
#define RESTIR_TARGET_MATERIAL_MASK 0x7u
#define RESTIR_TARGET_GEOMETRY_MASK 0x00ffffffu

struct RestirTargetSurface
{
    packed_float3 position;
    packed_float3 rayDirection;
    vector_float2 barycentrics;
    packed_float3 throughput;
    float lodBase;
    uint32_t geomEntryIndex;
    uint32_t instanceIndex;
    uint32_t primitiveId;
    uint32_t sampleIdxAndMedium;
};
static_assert(sizeof(RestirTargetSurface) == 64, "ReSTIR target surface ABI changed");

struct RestirDirectTargetSurface
{
    packed_float3 position;
    packed_float3 shadingNormal;
    packed_float3 rayDirection;
    packed_float3 albedo;
    uint32_t flags;
    float roughness;
    float metallicOrIor;
    uint32_t sampleIdxAndMedium;
};
static_assert(sizeof(RestirDirectTargetSurface) == 64, "ReSTIR direct target surface ABI changed");

// A deferred occlusion query produced by `shade` and consumed by `shadow`.
struct ShadowRay
{
    packed_float3 origin;
    packed_float3 direction;
    packed_float3 weight; // radiance already divided by pdf and multiplied by the BSDF
    float maxDistance;
    uint32_t pixelIndex;
    // Uniform random threshold for stochastic alpha visibility, drawn where
    // the ray was created because that is where the sampler knows the path's depth.
    float alphaThreshold;
    uint32_t medium;
    // Local (pre-path-throughput) direct-light estimate. The sparse update pass
    // propagates this only after visibility has been established by shadow.
    packed_float3 sharcRadiance;
    uint32_t sharcPathIndex;
};

// Plain path tracing never consumes the SHARC payload or the ReSTIR visibility
// flags. Its queue is dense, so omitting those four words saves both the shade
// write and the shadow read rather than merely shrinking the allocation.
struct CompactShadowTraversal
{
    packed_float3 origin;
    packed_float3 direction;
    float maxDistance;
    float alphaThreshold;
};
static_assert(sizeof(CompactShadowTraversal) == 32, "Compact shadow traversal ABI changed");

struct CompactShadowContribution
{
    packed_float3 weight;
    uint32_t pixelIndex;
};
static_assert(sizeof(CompactShadowContribution) == 16, "Compact shadow contribution ABI changed");

// EnvAliasEntry is shared by Metal, OptiX and host tests.
#include <env_alias_sampling.h>

struct SkinningParams
{
    uint32_t vbOffset; // vertex buffer offset for this mesh
    uint32_t sbOffset; // skin data buffer offset
    uint32_t jointMatOffset; // offset into joint matrices array
    uint32_t vertexCount;
};
static_assert(sizeof(SkinningParams) == 16, "SkinningParams host/Metal ABI changed");

struct UniformLight
{
    vector_float4 points[4];
    vector_float4 color;
    vector_float4 normal;
    int type;
    float halfAngle;
    float pad0;
    float pad1;
#ifdef __METAL_VERSION__
    texture2d<float> projectorTexture;
#else
    MTL::ResourceID projectorTexture;
#endif
    // Walker/Vose analytic-light distribution. The represented marginal PMF
    // lives in color.w (RGB consumers ignore it); these two fields replace the
    // old cumulative endpoint and PDF without growing the 128-byte ABI.
    uint32_t selectionAliasThreshold;
    uint32_t selectionAlias;
};
static_assert(sizeof(UniformLight) == 128, "UniformLight host/Metal ABI changed");

struct IesGpuBufferHeader
{
    uint32_t profileCount;
    uint32_t floatOffset; // byte offset of the float blob from the buffer start
    uint32_t pad0;
    uint32_t pad1;
};

struct IesGpuProfileHeader
{
    uint32_t nVertical;
    uint32_t nHorizontal;
    uint32_t anglesOffset; // index into the float blob: vertical then horizontal
    uint32_t candelaOffset; // index into the float blob
    float maxCandela;
    float pad0;
    float pad1;
    float pad2;
};

#define MATERIAL_TEX_BASE_COLOR (1u << 0)
#define MATERIAL_TEX_METALLIC_ROUGHNESS (1u << 1)
#define MATERIAL_TEX_NORMAL (1u << 2)
#define MATERIAL_TEX_EMISSION (1u << 3)
#define MATERIAL_TEX_OCCLUSION (1u << 4)
#define MATERIAL_TEXTURE_MASK ((1u << 5) - 1u)
#define MATERIAL_FEATURE_TRANSMISSION (1u << 8)
#define MATERIAL_FEATURE_CLEARCOAT (1u << 9)
#define MATERIAL_FEATURE_ANISOTROPY (1u << 10)
#define MATERIAL_FEATURE_DIFFUSE_TRANSMISSION (1u << 11)
#define MATERIAL_FEATURE_SHEEN (1u << 12)
#define MATERIAL_FEATURE_SUBSURFACE (1u << 13)
#define MATERIAL_FEATURE_IRIDESCENCE (1u << 14)
#define MATERIAL_FEATURE_SPECULAR_COLOR (1u << 15)
// Authored OpenPBR owns its factors and maps outright. A material translated
// from glTF still needs the generic texture path before the OpenPBR BSDF, so
// keep the two cases distinguishable after both use MATERIAL_TYPE_OPENPBR.
#define MATERIAL_FEATURE_NATIVE_OPENPBR (1u << 16)

struct Material
{
    // PBR parameters (layout uses packed_float3 for host/GPU compatibility)
    packed_float3 base_color; // 12 bytes
    float metallic; //  4 bytes  -- 16

    float roughness; //  4 bytes
    float ior; //  4 bytes
    float specular; //  4 bytes
    uint32_t features; // MaterialFeatureBits; occupies the old padding word -- 32

    float transmission; //  4 bytes
    float clearcoat; //  4 bytes
    float clearcoat_roughness; //  4 bytes
    float anisotropy; //  4 bytes  -- 48

    packed_float3 emission; // 12 bytes
    float emission_strength; //  4 bytes  -- 64

    float normal_scale; //  4 bytes
    float occlusion_strength; //  4 bytes
    float alpha_cutoff; //  4 bytes
    uint32_t material_type; //  4 bytes  -- 80

    uint32_t thin_walled; //  4 bytes
    uint32_t dielectric_priority; //  4 bytes  (nested dielectrics)
    uint32_t alpha_mode; //  4 bytes  (AlphaMode)
    float base_color_alpha; //  4 bytes  -- 96

    packed_float3 attenuation_color; // 12 bytes (KHR_materials_volume)
    float attenuation_distance; //  4 bytes -- 112

    vector_float2 uv_offset; //  8 bytes
    vector_float2 uv_transform_x; //  8 bytes
    vector_float2 uv_transform_y; //  8 bytes -- 136

    // KHR_materials_diffuse_transmission; see material_params.h.
    packed_float3 diffuse_transmission_color; // 12 bytes
    float diffuse_transmission; //  4 bytes -- 152

    // KHR_materials_sheen; see material_params.h.
    packed_float3 sheen_color; // 12 bytes
    float sheen; //  4 bytes -- 168

    // STRELKA_materials_subsurface; see material_params.h.
    packed_float3 subsurface_radius; // 12 bytes
    float sheen_roughness; //  4 bytes -- 184
    float subsurface; //  4 bytes
    float subsurface_anisotropy; //  4 bytes -- 192

    // STRELKA_materials_medium; see material_params.h.
    packed_float3 medium_emission; // 12 bytes
    uint32_t medium_flags; //  4 bytes -- 208
    float clearcoat_ior; //  4 bytes -- 212

    // KHR_materials_specular specularColorFactor; see material_params.h.
    packed_float3 specular_color; // 12 bytes -- 224

    // KHR_materials_iridescence; see material_params.h.
    float iridescence; //  4 bytes
    float iridescence_ior; //  4 bytes
    float iridescence_thickness; //  4 bytes -- 236

    packed_float3 subsurface_reference; // 12 bytes -- 248
    float _pad_irid[2]; //  8 bytes -- 256

    // Textures (8 bytes each: resource ID on CPU, texture handle on GPU)
#ifdef __METAL_VERSION__
    texture2d<float> baseColorTexture;
    texture2d<float> metallicRoughnessTexture;
    texture2d<float> normalTexture;
    texture2d<float> emissionTexture;
    texture2d<float> occlusionTexture;
#else
    MTL::ResourceID baseColorTexture;
    MTL::ResourceID metallicRoughnessTexture;
    MTL::ResourceID normalTexture;
    MTL::ResourceID emissionTexture;
    MTL::ResourceID occlusionTexture;
#endif
};
static_assert(sizeof(Material) == 296, "Material host/Metal ABI changed");

struct AlphaMaterialData
{
#ifdef __METAL_VERSION__
    texture2d<float> baseColorTexture;
#else
    MTL::ResourceID baseColorTexture;
#endif
    float baseColorAlpha;
    float alphaCutoff;
    uint32_t alphaMode;
    uint32_t features;
    vector_float2 uvOffset;
    vector_float2 uvTransformX;
    vector_float2 uvTransformY;
};
static_assert(sizeof(AlphaMaterialData) == 48, "Alpha material host/Metal ABI changed");

#endif
