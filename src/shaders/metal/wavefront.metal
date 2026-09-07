#include "shading_common.h"

// The OpenPBR BSDF. Included unconditionally, and it has to be: SPEC_OPENPBR is
// a function constant, so it selects code when the pipeline is specialised, not
// when this file is preprocessed. Its ~264 KB of lookup tables therefore land in
// the metallib whichever variant is built. That is data rather than
// instructions, so it does not compete for the instruction cache this kernel is
// bound by -- which is the whole reason the branch below is behind a constant.
#include <strelka/material/openpbr/openpbr_bridge.h>
#include <sharc_query_eligibility.h>

// Path state stays pixel-indexed while queues compact live indices between stages.
// Shadow traversal is deferred so its incoherent work stays out of shade.

#include "fog.h"
#include "subsurface.h"
#include "sharc.h"

// Every stage that resolves a hit reads the same instance descriptor buffer the
// top level was built from, so its element type has to be the one the host
// wrote -- MTLIndirectAccelerationStructureInstanceDescriptor.
//
// Not a choice: Metal 4's instance descriptor takes no array of bottom-level
// structures, so an instance can only name one by resource ID, which is what
// the indirect descriptor carries and the UserID one does not. The two differ
// in both stride (72 against 68) and in where userID sits (60 against 64), so
// reading one as the other silently turns userID into half a resource ID and
// indexes the geometry table with it -- every surface comes back black, with
// nothing for the validation layer to report.

// Bit 30 of HitRecord::geomEntryIndex marks a scattering event in the
// atmosphere: no surface was reached, the ray was stopped by the medium. Bit 31
// is the emissive-geometry flag; both live in the same word because a fog event
// is a third kind of "what did this ray hit" and the shade kernel already
// branches on that word.
#define HIT_FOG_BIT (1u << 30)

// Bit 29 marks a scattering event inside a subsurface medium: like the fog bit,
// the ray never reached a surface, but the medium is the one bounded by the
// object the path is currently inside rather than the atmosphere.
#define HIT_SSS_BIT (1u << 29)

// The walk's albedo, packed to one word. Eight bits a channel is more precision
// than a scattering albedo carries meaning at, and this rides on every live path.
static inline uint32_t packMediumAlbedo(float3 a)
{
    return (uint32_t)(saturate(a.x) * 255.0f + 0.5f) | ((uint32_t)(saturate(a.y) * 255.0f + 0.5f) << 8) |
           ((uint32_t)(saturate(a.z) * 255.0f + 0.5f) << 16);
}

static inline float3 unpackMediumAlbedo(uint32_t v)
{
    constexpr float s = 1.0f / 255.0f;
    return float3((v & 0xffu) * s, ((v >> 8) & 0xffu) * s, ((v >> 16) & 0xffu) * s);
}

// Bit 31 of HitRecord::geomEntryIndex marks a hit on emissive geometry, in which
// case the remaining bits hold the light index rather than a geometry entry.
#define HIT_LIGHT_BIT 0x80000000u

struct AnalyticPrimitiveIntersection
{
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

#define WF_ANALYTIC_INTERSECTION_ENTRY(NAME, BODY, ...)                                                                \
    [[intersection(bounding_box, __VA_ARGS__)]]                                                                        \
    AnalyticPrimitiveIntersection NAME(float3 origin [[origin]], float3 direction [[direction]],                       \
                                       float minDistance [[min_distance]], float maxDistance [[max_distance]])         \
    {                                                                                                                  \
        const CanonicalAnalyticIntersection hit = BODY(origin, direction, minDistance, maxDistance);                   \
        return { hit.hit, hit.distance };                                                                              \
    }

WF_ANALYTIC_INTERSECTION_ENTRY(analyticSphereIntersection, intersectCanonicalSphere, triangle_data, instancing)
WF_ANALYTIC_INTERSECTION_ENTRY(analyticDiscIntersection, intersectCanonicalDisc, triangle_data, instancing)
WF_ANALYTIC_INTERSECTION_ENTRY(
    analyticSphereIntersectionMotion, intersectCanonicalSphere, triangle_data, instancing, primitive_motion)
WF_ANALYTIC_INTERSECTION_ENTRY(
    analyticDiscIntersectionMotion, intersectCanonicalDisc, triangle_data, instancing, primitive_motion)
WF_ANALYTIC_INTERSECTION_ENTRY(analyticSphereIntersectionCurve, intersectCanonicalSphere, triangle_data, curve_data, instancing)
WF_ANALYTIC_INTERSECTION_ENTRY(analyticDiscIntersectionCurve, intersectCanonicalDisc, triangle_data, curve_data, instancing)
WF_ANALYTIC_INTERSECTION_ENTRY(analyticSphereIntersectionMotionCurve,
                               intersectCanonicalSphere,
                               triangle_data,
                               curve_data,
                               instancing,
                               primitive_motion)
WF_ANALYTIC_INTERSECTION_ENTRY(
    analyticDiscIntersectionMotionCurve, intersectCanonicalDisc, triangle_data, curve_data, instancing, primitive_motion)

// Layout of the control buffer, shared by every stage.
//   [0], [1] : live path count of each ping-pong queue
//   [2..4]   : MTLDispatchThreadgroupsIndirectArguments for extend/shade
//   [5]      : live path count for this bounce, so stages need no queue index
//   [6..10]  : shadow ray count, republished count, and its dispatch arguments
//   [11..15] : the same for rays that hit geometry
//   [16..20] : the same for rays that escaped
#define WF_CTRL_COUNT0 0
#define WF_CTRL_COUNT1 1
#define WF_CTRL_DISPATCH 2
#define WF_CTRL_ACTIVE 5
#define WF_CTRL_SHADOW 6
#define WF_CTRL_SHADOW_N 7
#define WF_CTRL_SHADOW_DIS 8
#define WF_CTRL_HIT 11
#define WF_CTRL_HIT_N 12
#define WF_CTRL_HIT_DIS 13
#define WF_CTRL_MISS 16
#define WF_CTRL_MISS_N 17
#define WF_CTRL_MISS_DIS 18
#define WF_CTRL_GUIDE 21
#define WF_CTRL_GUIDE_N 22
#define WF_CTRL_CAPACITY 24
#define WF_CTRL_GUIDE_DIS 25
#define WF_CTRL_RESTIR 28
#define WF_CTRL_RESTIR_N 28
#define WF_CTRL_RESTIR_DIS 29
// Profiling only: live path count and shadow ray count per bounce, so the
// per-stage timings can be read as a cost per ray rather than a cost per stage.
#define WF_CTRL_STATS_PATHS 32
#define WF_CTRL_STATS_SHADOW 64

// Shared pre-traversal breadcrumbs. The prepare dispatch completes before
// extend starts, so these survive even when one of the first few rays wedges in
// the ray tracing unit and the command buffer is terminated by the watchdog.
#define WF_DIAG_BOUNCES 96
#define WF_DIAG_LANES 4
#define WF_DIAG_LANE_WORDS 11
#define WF_DIAG_STRIDE (1 + WF_DIAG_LANES * WF_DIAG_LANE_WORDS)
#define WF_STAGE_BREADCRUMB 96
#define WF_DIAG_BASE 97

static inline void auditWork(constant Uniforms& uniforms, uint32_t counter, uint32_t amount = 1u)
{
    if (SPEC_RENDER_WORK_AUDIT)
    {
        atomic_fetch_add_explicit(&uniforms.renderWorkCounters[counter], amount, memory_order_relaxed);
    }
}

static inline device RestirDiagnosticRecord* restirDiagnosticRecord(constant Uniforms& uniforms, uint32_t pixelIndex)
{
    if (!SPEC_RENDER_WORK_AUDIT || uniforms.renderWorkCounters == nullptr)
    {
        return nullptr;
    }
    device uint32_t* words = (device uint32_t*)uniforms.renderWorkCounters;
    device uint32_t* pixels = words + WORK_COUNTER_COUNT + RESTIR_AUDIT_LIGHT_ID_WORDS;
    device RestirDiagnosticRecord* records = (device RestirDiagnosticRecord*)(pixels + RESTIR_DIAGNOSTIC_PIXEL_COUNT);
    for (uint32_t i = 0u; i < RESTIR_DIAGNOSTIC_PIXEL_COUNT; ++i)
    {
        if (pixels[i] == pixelIndex)
        {
            return &records[i];
        }
    }
    return nullptr;
}

static inline device RestirCandidateAuditRecord* restirCandidateAuditRecord(constant Uniforms& uniforms,
                                                                            uint32_t pixelIndex)
{
    device RestirDiagnosticRecord* diagnostic = restirDiagnosticRecord(uniforms, pixelIndex);
    if (diagnostic == nullptr)
        return nullptr;
    device uint32_t* words = (device uint32_t*)uniforms.renderWorkCounters;
    device uint32_t* pixels = words + WORK_COUNTER_COUNT + RESTIR_AUDIT_LIGHT_ID_WORDS;
    device RestirDiagnosticRecord* records = (device RestirDiagnosticRecord*)(pixels + RESTIR_DIAGNOSTIC_PIXEL_COUNT);
    const uint32_t slot = uint32_t(diagnostic - records);
    return ((device RestirCandidateAuditRecord*)(records + RESTIR_DIAGNOSTIC_PIXEL_COUNT)) + slot;
}

static inline void auditRestirLightId(constant Uniforms& uniforms, uint32_t lightId)
{
    if (SPEC_RENDER_WORK_AUDIT && lightId < RESTIR_AUDIT_LIGHT_ID_WORDS * 32u)
    {
        device atomic_uint* words = uniforms.renderWorkCounters + WORK_COUNTER_COUNT;
        atomic_fetch_or_explicit(&words[lightId >> 5u], 1u << (lightId & 31u), memory_order_relaxed);
    }
}

static inline uint32_t restirMHistogramBin(uint32_t M)
{
    return M <= 1u ? 0u : min(32u - clz(M - 1u), RESTIR_AUDIT_M_BINS - 1u);
}

static inline uint32_t restirTargetRatioHistogramBin(float ratio)
{
    return uint32_t(clamp(floor(log2(ratio)) + 4.0f, 0.0f, float(RESTIR_AUDIT_TARGET_RATIO_BINS - 1u)));
}

static inline void restirDiagnosticVisibility(constant Uniforms& uniforms, uint32_t pixelIndex, float3 contribution)
{
    device RestirDiagnosticRecord* record = restirDiagnosticRecord(uniforms, pixelIndex);
    if (record != nullptr)
    {
        record->finalVisibility = 1u;
        record->contribution[0] = contribution.x;
        record->contribution[1] = contribution.y;
        record->contribution[2] = contribution.z;
    }
}

// Reserve a run of output slots for the active lanes of one simdgroup. Every
// lane that reaches a call site belongs in that queue -- the others already
// returned or took the other branch, so they are inactive and the simdgroup
// reductions below see only the lanes being queued. One atomic per simdgroup
// instead of one per lane.
static inline void queuePush(constant Uniforms& uniforms,
                             device atomic_uint* counter,
                             device uint32_t* queueOut,
                             uint32_t pathIndex,
                             uint32_t capacity)
{
    const uint32_t rank = simd_prefix_exclusive_sum(1u);
    const uint32_t total = simd_sum(1u);
    uint32_t base = 0u;
    if (simd_is_first())
    {
        base = atomic_fetch_add_explicit(counter, total, memory_order_relaxed);
        auditWork(uniforms, WORK_QUEUE_APPENDS, total);
        if (base >= capacity || total > capacity - base)
        {
            auditWork(uniforms, WORK_QUEUE_OVERFLOWS, base >= capacity ? total : total - (capacity - base));
        }
    }
    base = simd_broadcast_first(base);
    if (base < capacity && rank < capacity - base)
    {
        queueOut[base + rank] = pathIndex;
    }
}


// Static and motion acceleration structures require different intersector types,
// so traversal specialization cannot be a runtime branch or function constant.

// The fraction of a light a cutout shadow ray is allowed to carry before Russian
// roulette ends its walk.
//
// Accepting on a collapsed transmittance keeps the early-out the walk had: once
// nothing measurable can get through, the ray is blocked and traversal stops.
// A shadow ray that is already almost blocked is allowed to stop.
//
// Through a canopy the expensive rays are not the blocked ones -- those accept
// at the first opaque leaf -- but the ones that keep slipping past cutouts with
// a hundredth of the light left, traversing the whole crown to deliver
// something that rounds to nothing. Below this fraction of the light the ray is
// killed by Russian roulette and what survives is scaled back up, so the
// estimate stays unbiased and only its variance moves.
//
// A fraction of the light, not a distance or a size, so it means the same thing
// in any scene.
constant float kShadowTransmittanceCutoff = 0.05f;

// Coverage is callable from the restart walk because Metal 4 cannot run its intersection function.
static inline float cutoutOpacityAt(uint primitive_id,
                                    uint geometry_id,
                                    uint instance_id,
                                    float2 barycentric_coord,
                                    constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                    device const Material* materials,
                                    device const GeometryEntry* geometryEntries,
                                    device const char* vertexBuffer,
                                    device const uint32_t* indexBuffer)
{
    const auto inst = instances[instance_id];
    const GeometryEntry entry = geometryEntries[inst.userID + geometry_id];
    device const Material& mat = materials[entry.materialId];

    if (mat.alpha_mode == ALPHA_MODE_OPAQUE)
    {
        return 1.0f; // blocks outright
    }

    constexpr uint32_t vtxStride = 32;
    constexpr uint32_t uvOff = 20;
    float2 uvv[3];
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t idx = indexBuffer[entry.indexOffset + primitive_id * 3 + k];
        uvv[k] = unpackUV(*(device const uint32_t*)(vertexBuffer + (entry.vbOffset + idx) * vtxStride + uvOff));
    }
    const float2 uv = interpolateAttrib(uvv[0], uvv[1], uvv[2], barycentric_coord);
    return resolveOpacity(mat, uv);
}

struct MotionTraversal
{
    // intersection_query rejects the motion tags, so this one keeps the
    // restart walk.
    enum
    {
        kInlineQuery = 0
    };
    using structure = acceleration_structure<instancing, primitive_motion>;
    using isect = intersector<triangle_data, instancing, primitive_motion>;
    using volume_isect = isect;
    using table = intersection_function_table<triangle_data, instancing, primitive_motion>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::bounding_box;
    }
    static float curveParameter(thread const isect::result_type&)
    {
        return 0.0f;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float time, table t)
    {
        return i.intersect(r, as, mask, time, t);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, structure as, uint32_t mask, float time)
    {
        return i.intersect(r, as, mask, time);
    }
};

struct StaticTraversal
{
    using structure = acceleration_structure<instancing>;
    using isect = intersector<triangle_data, instancing>;
    // Metal's inline traversal, used by the cutout shadow walk so the alpha test
    // can run in the kernel without giving up the single traversal. An enum
    // rather than a static constexpr: a program-scope constexpr under the Metal
    // compiler has to live in the constant address space.
    using query = intersection_query<triangle_data, instancing>;
    enum
    {
        kInlineQuery = 0
    };
    using volume_isect = isect;
    using table = intersection_function_table<triangle_data, instancing>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::bounding_box;
    }
    static float curveParameter(thread const isect::result_type&)
    {
        return 0.0f;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float, table t)
    {
        return i.intersect(r, as, mask, t);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, structure as, uint32_t mask, float)
    {
        return i.intersect(r, as, mask);
    }
};

// The same two, able to see curves.
//
// A separate pair rather than a flag, because `curve_data` is a *tag*: it decides
// what the intersector's result type carries, so it cannot be turned on by a
// function constant any more than the motion tag can. That is also why it is
// worth keeping apart -- a scene with no hair in it goes on traversing with the
// triangle-only intersector it always used, and the extra geometry type costs it
// nothing.
struct CurveMotionTraversal
{
    // intersection_query rejects the motion tags, so this one keeps the
    // restart walk.
    enum
    {
        kInlineQuery = 0
    };
    using structure = acceleration_structure<instancing, primitive_motion>;
    using isect = intersector<triangle_data, curve_data, instancing, primitive_motion>;
    using volume_isect = intersector<triangle_data, instancing, primitive_motion>;
    using table = intersection_function_table<triangle_data, curve_data, instancing, primitive_motion>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::curve | geometry_type::bounding_box;
    }
    static float curveParameter(thread const isect::result_type& r)
    {
        return r.curve_parameter;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float time, table t)
    {
        return i.intersect(r, as, mask, time, t);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, structure as, uint32_t mask, float time)
    {
        return i.intersect(r, as, mask, time);
    }
};

struct CurveStaticTraversal
{
    using structure = acceleration_structure<instancing>;
    using isect = intersector<triangle_data, curve_data, instancing>;
    using query = intersection_query<triangle_data, curve_data, instancing>;
    enum
    {
        kInlineQuery = 0
    };
    using volume_isect = intersector<triangle_data, instancing>;
    using table = intersection_function_table<triangle_data, curve_data, instancing>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::curve | geometry_type::bounding_box;
    }
    static float curveParameter(thread const isect::result_type& r)
    {
        return r.curve_parameter;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float, table t)
    {
        return i.intersect(r, as, mask, t);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, structure as, uint32_t mask, float)
    {
        return i.intersect(r, as, mask);
    }
};

// The curve and triangle intersectors return different result types even though
// extend consumes the same common fields. Copy those fields into one small
// value so an SSS lane can use a genuinely triangle-only intersector while the
// other lanes keep the curve-capable one.
struct ExtendIntersection
{
    intersection_type type;
    uint32_t instanceId;
    uint32_t geometryId;
    uint32_t primitiveId;
    float distance;
    float2 barycentrics;
    float curveParameter;
};

template <typename R>
static inline ExtendIntersection captureExtendIntersection(thread const R& r, float curveParameter)
{
    ExtendIntersection out;
    out.type = r.type;
    out.instanceId = 0u;
    out.geometryId = 0u;
    out.primitiveId = 0u;
    out.distance = INFINITY;
    out.barycentrics = float2(0.0f);
    out.curveParameter = 0.0f;
    if (r.type != intersection_type::none)
    {
        out.instanceId = r.instance_id;
        out.geometryId = r.geometry_id;
        out.primitiveId = r.primitive_id;
        out.distance = r.distance;
        if (r.type == intersection_type::triangle)
        {
            out.barycentrics = r.triangle_barycentric_coord;
        }
        else if (r.type == intersection_type::curve)
        {
            out.curveParameter = curveParameter;
        }
    }
    return out;
}

static inline uint32_t pathDepth(uint32_t depthAndFlags)
{
    return depthAndFlags & PATH_DEPTH_MASK;
}

// The sampler is never stored: it is a pure function of these three values, and
// recomputing it costs less than the 12 bytes it would add to every path.
static inline SamplerState samplerFor(constant Uniforms& uniforms, uint32_t pixelIndex, uint32_t sampleIdx, uint32_t depth)
{
    // Ordinary accumulation advances subframeIndex by the number of samples
    // already folded into the estimate. Temporal reconstruction instead follows
    // the monotonic display frame: camera motion resets subframeIndex, and using
    // it there would retrace the same path and hand MetalFX correlated noise.
    // SHARC has its own temporal sequence for the same reason.
    const uint32_t sequenceBase = restirSampleSequenceBase(
        uniforms.useFrameJitter != 0u, uniforms.enableAccumulation != 0u, uniforms.restirDIEnabled != 0u,
        uniforms.frameIndex, uniforms.samples_per_launch, uniforms.subframeIndex);
    const uint32_t sequenceIndex = SPEC_SHARC_UPDATE ? uniforms.sharcFrameIndex : sequenceBase + sampleIdx;
    SamplerState s = initSampler(pixelIndex, sequenceIndex, uniforms.width, uniforms.blueNoiseSwitchSpp);
    s.depth = depth;
    return s;
}

// A path has one time, sampled once when its camera ray is generated. The
// sampler is stateless, so every later stage can recover that exact value by
// drawing eTime at depth 0 again -- drawing it at the current depth would give a
// different time on every bounce and smear the path across the shutter.
static inline float motionTimeFor(constant Uniforms& uniforms, uint32_t pixelIndex, uint32_t sampleIdx)
{
    if (!SPEC_MOTION_BLUR || !uniforms.enableMotionBlur)
    {
        return 0.0f;
    }
    SamplerState s = samplerFor(uniforms, pixelIndex, sampleIdx, 0u);
    const uint32_t sampleCount = max(uniforms.samples_per_launch, 1u);
    const float t = ((float)sampleIdx + random<SampleDimension::eTime>(s, uniforms.samplerType)) / (float)sampleCount;
    return uniforms.isMotionBlurVisible ? t : 1.0f;
}

static inline bool shouldWriteAov(constant Uniforms& uniforms, uint32_t sampleIdx)
{
    return !SPEC_SHARC_UPDATE && uniforms.writeAov && sampleIdx == 0u;
}

static inline float backgroundDepth(constant Uniforms& uniforms);

static inline uint32_t sharcUpdateStateIndex(constant Uniforms& uniforms, uint32_t pixelIndex)
{
    const uint32_t scale = max(uniforms.sharcUpdateDownscale, 1u);
    const uint32_t tileWidth = (uniforms.width + scale - 1u) / scale;
    const uint2 pixel = uint2(pixelIndex % uniforms.width, pixelIndex / uniforms.width);
    return (pixel.y / scale) * tileWidth + pixel.x / scale;
}

static inline uint32_t packSharcRoughness(float roughness)
{
    return uint32_t(round(saturate(roughness) * 255.0f)) << PATH_SHARC_ROUGHNESS_SHIFT;
}

static inline float unpackSharcRoughness(uint32_t depthAndFlags)
{
    return float((depthAndFlags & PATH_SHARC_ROUGHNESS_MASK) >> PATH_SHARC_ROUGHNESS_SHIFT) / 255.0f;
}

// Angular bandwidth of the outgoing radiance represented by one SHARC entry.
// Use the narrowest active reflective layer: one sharp coat over a diffuse base
// is still view dependent when all layers share a single cached value.
static inline float sharcReceiverRoughness(bool isOpenPBR,
                                           thread const OpenPBRParams& openpbr,
                                           thread const SurfaceInteraction& si)
{
    float roughness = 1.0f;
    if (isOpenPBR)
    {
        roughness = sharcReceiverLobeRoughness(roughness, openpbr.specular_roughness, openpbr.specular_weight);
        roughness = sharcReceiverLobeRoughness(roughness, openpbr.coat_roughness, openpbr.coat_weight);
        roughness = sharcReceiverLobeRoughness(roughness, openpbr.fuzz_roughness, openpbr.fuzz_weight);
        return roughness;
    }

    // Standard PBR always has a Fresnel reflection lobe. Clearcoat and sheen
    // add their own view-dependent layers only when their weights are nonzero.
    roughness = sharcReceiverLobeRoughness(roughness, si.roughness, 1.0f);
    roughness = sharcReceiverLobeRoughness(roughness, si.clearcoat_roughness, si.clearcoat);
    roughness = sharcReceiverLobeRoughness(roughness, si.sheen_roughness, si.sheen);
    return roughness;
}

// ---------------------------------------------------------------------------
// generate -- camera rays
// ---------------------------------------------------------------------------

/// The medium a path is travelling inside, from whichever model describes it.
///
/// Two sites need this -- `extend`, to sample the next free flight, and `shade`,
/// to weight a scattering event -- and they must agree to the digit or the
/// estimator stops being unbiased: a flight sampled from one density and
/// weighted by another is exactly that mistake.
///
/// The OpenPBR branch does not read the material's subsurface fields at all. It
/// derives the medium from the OpenPBR parameters through the vendored
/// implementation, which blends subsurface scattering and transmission into one
/// volume the way the specification defines them jointly, and which applies a van
/// de Hulst mapping from the authored colour to a single-scattering albedo.
/// Reimplementing either here would be two formulas to keep in step instead of
/// none.
struct MediumProps
{
    float3 sigmaT;
    float3 albedo;
    float anisotropy;
};

/// Extinction alone, for the two places that need no albedo: the exit boundary
/// weight and the optical depth a shadow ray accumulates.
///
/// Its own function because those two sites read the material directly and had
/// no reason to know OpenPBR exists -- which is exactly how they went on using
/// Material::subsurface_radius, a glTF field an OpenPBR material never fills.
/// At zero that reciprocal is 1e5, so the exit boundary weight annihilated every
/// path leaving the medium, independent of how long its walk had been. That is
/// what made the chess kings dark at every iteration budget.
static float3 mediumSigmaT(constant Uniforms& uniforms, device const Material* materials, uint32_t materialIndex)
{
    device const Material& mm = materials[materialIndex];
    if (SPEC_OPENPBR && mm.material_type == MATERIAL_TYPE_OPENPBR && uniforms.openpbrParams != nullptr)
    {
        const OpenPBRParams mat = uniforms.openpbrParams[materialIndex];
        return openpbr_interior_volume(mat).extinction_coefficient;
    }
    return sssSigmaT(float3(mm.subsurface_radius));
}

static MediumProps mediumPropsFor(constant Uniforms& uniforms,
                                  device const Material* materials,
                                  uint32_t materialIndex,
                                  uint32_t packedAlbedo)
{
    device const Material& mm = materials[materialIndex];
    MediumProps out;

    if (SPEC_OPENPBR && mm.material_type == MATERIAL_TYPE_OPENPBR && uniforms.openpbrParams != nullptr)
    {
        // Copied out of device memory because the bridge takes a thread
        // reference -- it is portable code and cannot name Metal's address
        // spaces. The copy is nominal: openpbr_interior_volume() reads eleven
        // of these fields and the rest are dead, which scalar replacement
        // removes.
        const OpenPBRParams mat = uniforms.openpbrParams[materialIndex];
        const OpenPBR_HomogeneousVolume v = openpbr_interior_volume(mat);
        out.sigmaT = v.extinction_coefficient;
        // The texture-resolved single-scattering albedo was captured where the
        // path crossed the surface. There is no UV inside the volume with which
        // to reconstruct it from the material table here.
        out.albedo = unpackMediumAlbedo(packedAlbedo);
        out.anisotropy = v.anisotropy;
        return out;
    }

    out.sigmaT = mediumSigmaT(uniforms, materials, materialIndex);
    // A bounded volume has no entry surface to have textured, so it keeps the
    // material's constant; a subsurface walk takes what the boundary resolved.
    out.albedo = ((mm.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u) ? float3(mm.diffuse_transmission_color) :
                                                                    unpackMediumAlbedo(packedAlbedo);
    out.anisotropy = mm.subsurface_anisotropy;
    return out;
}

kernel void wavefrontGenerate(uint tid [[thread_position_in_grid]],
                              constant Uniforms& uniforms [[buffer(0)]],
                              device PathState* paths [[buffer(1)]],
                              device PathRay* rays [[buffer(8)]],
                              device float4* radianceOut [[buffer(2)]],
                              device IorStack* iorStacks [[buffer(3)]],
                              constant uint32_t& sampleIdx [[buffer(4)]],
                              device uint32_t* queueOut [[buffer(5)]],
                              device uint32_t* control [[buffer(6)]],
                              device AovSample* aov [[buffer(7)]],
                              // Zeroed here rather than on the host: `generate` already owns resetting the
                              // per-sample counters, and a host-side clear would race the frame in flight.
                              // The tally is therefore per sample, which is the rate rather than a total.
                              device uint32_t* iorStats [[buffer(9)]],
                              device SharcUpdateState* sharcUpdates [[buffer(10)]],
                              device MediumPathState* mediumPaths [[buffer(11)]])
{
    const uint32_t pixelCount = uniforms.width * uniforms.height;
    const uint32_t pathCount = SPEC_SHARC_UPDATE ? uniforms.sharcUpdatePathCount : pixelCount;
    if (tid == 0u)
    {
        control[WF_CTRL_COUNT0] = pathCount;
        control[WF_CTRL_COUNT1] = 0u;
        control[WF_CTRL_SHADOW] = 0u;
        control[WF_CTRL_HIT] = 0u;
        control[WF_CTRL_MISS] = 0u;
        if (!SPEC_SHARC_UPDATE && sampleIdx == 0u)
        {
            control[WF_CTRL_GUIDE] = 0u;
        }
        control[WF_CTRL_CAPACITY] = pathCount;
        iorStats[IOR_STAT_OVERFLOW] = 0u;
        iorStats[IOR_STAT_UNMATCHED] = 0u;
        iorStats[IOR_STAT_ESCAPED_INSIDE] = 0u;
    }
    if (tid >= pathCount)
    {
        return;
    }

    if (!SPEC_SHARC_UPDATE)
    {
        auditWork(uniforms, WORK_PRIMARY_RAYS);
    }

    uint32_t pixelIndex = tid;
    if (SPEC_SHARC_UPDATE)
    {
        const uint32_t scale = max(uniforms.sharcUpdateDownscale, 1u);
        const uint32_t tileWidth = (uniforms.width + scale - 1u) / scale;
        const uint2 tile = uint2(tid % tileWidth, tid / tileWidth);
        const uint32_t scramble = sharcHash(tid ^ (uniforms.sharcFrameIndex * 0x9e3779b9u));
        const uint2 offset = uint2(scramble % scale, (scramble / scale) % scale);
        const uint2 pixel = min(tile * scale + offset, uint2(uniforms.width - 1u, uniforms.height - 1u));
        pixelIndex = pixel.y * uniforms.width + pixel.x;
        SharcUpdateState updateState;
        sharcInitUpdateState(updateState, pixelIndex);
        sharcUpdates[tid] = updateState;
    }

    // Sparse updates preserve the same tile order while queueing the selected
    // full-resolution path slots.
    queueOut[tid] = pixelIndex;

    if (sampleIdx == 0u)
    {
        radianceOut[pixelIndex] = float4(0.0f);
    }
    if (uniforms.restirDIEnabled != 0u && !SPEC_SHARC_UPDATE)
    {
        device RestirReservoir* currentReservoirs =
            (uniforms.frameIndex & 1u) != 0u ? uniforms.restirReservoir1 : uniforms.restirReservoir0;
        device RestirSurfaceHistory* currentHistory =
            (uniforms.frameIndex & 1u) != 0u ? uniforms.restirHistory1 : uniforms.restirHistory0;
        currentReservoirs[pixelIndex] = {};
        currentHistory[pixelIndex] = {};
    }

    const uint2 pixel = uint2(pixelIndex % uniforms.width, pixelIndex / uniforms.width);
    SamplerState rng = samplerFor(uniforms, pixelIndex, sampleIdx, 0u);
    const float motionTime = motionTimeFor(uniforms, pixelIndex, sampleIdx);

    float3 origin, direction;
    generateCameraRay(pixel, rng, origin, direction, uniforms, motionTime);
    if (shouldWriteAov(uniforms, sampleIdx))
    {
        AovSample a;
        a.diffuseAlbedo = packed_float3(float3(0.0f));
        a.specularAlbedo = packed_float3(float3(0.0f));
        a.normal = packed_float3(-direction);
        a.roughness = 1.0f;
        a.depth = backgroundDepth(uniforms);
        a.motionX = 0.0f;
        a.motionY = 0.0f;
        a.specularHitDistance = 0.0f;
        // Until a primary surface replaces it, this is background. The AOV
        // resolve derives MetalFX's denoise-strength mask from its depth.
        a.reactive = 1.0f;
        a.guideStateOrBounceDepth = -1.0f;
        aov[pixelIndex] = a;
        uniforms.guideRays[pixelIndex].flags = 0u;
    }

    PathRay r;
    r.origin = packed_float3(origin);
    r.direction = packed_float3(direction);
    rays[pixelIndex] = r;

    PathState p;
    p.throughput = packed_float3(float3(1.0f));
    // The camera segment has no scattering footprint. The specular flag keeps
    // it non-queryable independently; subsequent bounces accumulate their cone
    // width into the packed SHARC roughness.
    p.depthAndFlags = PATH_FLAG_ALIVE | PATH_FLAG_SPECULAR | packSharcRoughness(0.0f);
    p.lastBsdfPdf = 0.0f;
    p.misDistance = 0.0f;
    paths[pixelIndex] = p;
    if (SPEC_SSS)
    {
        // Outside every medium. A camera that starts inside a translucent object
        // is not handled -- nothing tells the path which medium it is in.
        MediumPathState mediumState;
        mediumState.medium = 0u;
        mediumState.mediumAlbedo = 0u;
        mediumPaths[pixelIndex] = mediumState;
    }

    // ior_stack_* take a thread reference; device memory cannot bind to one.
    IorStack stack;
    ior_stack_init(stack);
    iorStacks[pixelIndex] = stack;
}

// ---------------------------------------------------------------------------
// extend -- closest hit
// ---------------------------------------------------------------------------
static inline device HitRecord* wavefrontHitRecord(device char* records, uint32_t index)
{
    const size_t stride = SPEC_RESTIR ? sizeof(RestirReservoir) : sizeof(HitRecord);
    return (device HitRecord*)(records + size_t(index) * stride);
}

template <typename T>
static void extendImpl(uint gid,
                       constant Uniforms& uniforms,
                       constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                       typename T::structure accelerationStructure,
                       typename T::structure volumeAccelerationStructure,
                       device const PathRay* rays,
                       device char* hits,
                       constant uint32_t& sampleIdx,
                       device const uint32_t* queue,
                       device const uint32_t* control,
                       device uint32_t* hitQueue,
                       device atomic_uint* hitCounter,
                       device uint32_t* missQueue,
                       device atomic_uint* missCounter,
                       // Only for the fog: the path's depth decorrelates the free-flight draw
                       // across bounces, and without it every bounce of a path scatters at the same
                       // fraction of its segment, which shows up as banding in the haze.
                       device const PathState* paths,
                       // Only for the subsurface walk, which needs the medium's mean free path to
                       // sample a free flight and reads it from the material the path is inside.
                       device const Material* materials,
                       device const MediumPathState* mediumPaths,
                       typename T::table functionTable,
                       // Chosen per dispatch rather than per ray: the only thing it distinguishes
                       // is the camera bounce from the rest, and `extend` is encoded once per
                       // bounce anyway. Reading the path's depth here to answer the same question
                       // would put a load in the hottest kernel in the renderer.
                       uint32_t rayMask)
{
    // Indirect dispatch can only launch whole threadgroups, so the tail of the
    // last one runs past the queue and has to be discarded here.
    if (gid >= control[WF_CTRL_ACTIVE])
    {
        return;
    }
    const uint32_t tid = queue[gid];
    // A corrupted append count must not turn into an out-of-bounds PathRay load
    // and then an invalid hardware traversal. This is cold-path protection: a
    // valid queue always contains the pixel slot its path owns.
    if (tid >= uniforms.width * uniforms.height)
    {
        return;
    }
    const PathRay pr = rays[tid];

    const uint32_t auditBounce = min(pathDepth(paths[tid].depthAndFlags), WORK_BOUNCE_SLOTS - 1u);
    auditWork(uniforms, WORK_EXTEND_RAYS_BASE + auditBounce);
    auditWork(uniforms, WORK_INTERSECTION_QUERIES);
    auditWork(uniforms, WORK_EXTENSION_QUERIES);

    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);

    ray r;
    // Match Apple's curve sample: an offset origin and a positive lower bound
    // solve different halves of self-intersection. Round curves can still report
    // the surface the secondary ray just left at an almost-zero t after several
    // fibre crossings.
    r.min_distance = pathDepth(paths[tid].depthAndFlags) == 0u ? 0.0f : 1e-6f;
    r.max_distance = INFINITY;
    r.origin = float3(pr.origin);
    r.direction = float3(pr.direction);

    // Never hand NaN, infinity, or a collapsed direction to the ray tracing
    // unit. Such a path cannot produce a finite contribution, while curve
    // traversal on malformed rays can fail to make progress and trip the GPU
    // watchdog instead of merely returning no intersection.
    const float directionLength2 = dot(r.direction, r.direction);
    if (!all(isfinite(r.origin)) || !all(isfinite(r.direction)) || !(directionLength2 > 0.25f && directionLength2 < 4.0f))
    {
        return;
    }

    // Draw a participating-medium event before traversal and use it as the
    // ray's upper bound. The old order traced to the closest surface first and
    // only then discovered that a sub-millimetre SSS free flight should have
    // stopped the ray. That is equivalent statistically, but catastrophically
    // more work in a curve-heavy scene: a single late random-walk ray could walk
    // the entire multi-million-segment AS and hold one dispatch past the GPU
    // watchdog. The same bound helps a fog ray in a large instanced scene.
    //
    // A surface at or before the sampled distance still wins below, so this is
    // only a scheduling bound; it does not change which event the path sees.
    bool insideSss = false;
    uint32_t mediumHitBit = 0u;
    float mediumScatterT = 0.0f;
    if (SPEC_SSS)
    {
        const MediumPathState mediumState = mediumPaths[tid];
        const uint32_t sss = mediumState.medium;
        const uint32_t medium = sss & MEDIUM_INDEX_MASK;
        insideSss = medium != 0u;
        if (insideSss)
        {
            const uint32_t step = sss >> MEDIUM_STEP_SHIFT;
            if (step < MEDIUM_MAX_STEPS)
            {
                const MediumProps mp = mediumPropsFor(uniforms, materials, medium - 1u, mediumState.mediumAlbedo);
                const float3 sigmaT = mp.sigmaT;
                const float3 albedo = mp.albedo;
                const float3 channelPdf = sssChannelPdf(float3(paths[tid].throughput), albedo);
                SamplerState srng = samplerFor(uniforms, tid, sampleIdx, pathDepth(paths[tid].depthAndFlags) + step);
                // Bound free flights by the scene: a longer draw left the medium and can wedge traversal on leaked
                // paths.
                if (sssSampleDistance(sigmaT, channelPdf, uniforms.sceneExtent,
                                      random<SampleDimension::eSssChannel>(srng, uniforms.samplerType),
                                      random<SampleDimension::eSssDistance>(srng, uniforms.samplerType), mediumScatterT))
                {
                    mediumHitBit = HIT_SSS_BIT;
                }
            }
        }
    }
    if (SPEC_FOG && uniforms.hasFog && !insideSss)
    {
        SamplerState frng = samplerFor(uniforms, tid, sampleIdx, pathDepth(paths[tid].depthAndFlags));
        if (fogSampleDistance(r.origin, r.direction, 1e16f, uniforms.fogHeight, uniforms.fogSigmaT,
                              random<SampleDimension::eFogDistance>(frng, uniforms.samplerType), mediumScatterT))
        {
            mediumHitBit = HIT_FOG_BIT;
        }
    }
    if (mediumHitBit != 0u)
    {
        // Keep max >= min even for a legitimate zero-valued random draw. Any
        // surface in this tiny padded interval is compared with the actual
        // sampled distance after traversal.
        r.max_distance = max(mediumScatterT, r.min_distance + 1e-6f);
    }

    // SSS random walks only need the closed triangle boundary of their volume.
    // Curves are fibre primitives, not volume boundaries. An instance mask was
    // not sufficient here: the curve-capable intersector still entered Metal's
    // pathological traversal mode for a handful of deep random-walk rays. Use
    // a triangle-only intersector type for those lanes, so curve intersection
    // code is absent from the operation rather than merely unable to return a
    // candidate.
    ExtendIntersection hit;
    if (insideSss)
    {
        typename T::volume_isect volumeIsect;
        volumeIsect.assume_geometry_type(geometry_type::triangle);
        volumeIsect.force_opacity(forced_opacity::opaque);
        volumeIsect.accept_any_intersection(false);
        const typename T::volume_isect::result_type volumeHit =
            T::traceVolume(volumeIsect, r, volumeAccelerationStructure, rayMask & ~GEOMETRY_MASK_CURVE, motionTime);
        hit = captureExtendIntersection(volumeHit, 0.0f);
    }
    else
    {
        typename T::isect isect;
        isect.assume_geometry_type(T::geometryTypes());
        // Cutout coverage is tested in shade, not extend.
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(false);
        const typename T::isect::result_type surfaceHit =
            T::trace(isect, r, accelerationStructure, rayMask, motionTime, functionTable);
        const float curveParameter = surfaceHit.type == intersection_type::curve ? T::curveParameter(surfaceHit) : 0.0f;
        hit = captureExtendIntersection(surfaceHit, curveParameter);
    }

    const float surfaceDistance = hit.type == intersection_type::none ? r.max_distance : hit.distance;

    // Escaped rays skip shade and go to miss. Fog/SSS are decided here
    // because only extend knows whether a surface precedes the medium event.
    if (mediumHitBit != 0u && (hit.type == intersection_type::none || mediumScatterT < surfaceDistance))
    {
        HitRecord mediumRec;
        mediumRec.geomEntryIndex = mediumHitBit;
        mediumRec.instanceIndex = 0u;
        mediumRec.primitiveId = 0u;
        mediumRec.barycentrics = vector_float2(0.0f, 0.0f);
        mediumRec.distance = mediumScatterT;
        *wavefrontHitRecord(hits, tid) = mediumRec;
        queuePush(uniforms, hitCounter, hitQueue, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    if (hit.type == intersection_type::none)
    {
        queuePush(uniforms, missCounter, missQueue, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    const auto inst = instances[hit.instanceId];
    const bool isLight = (inst.mask == GEOMETRY_MASK_LIGHT || inst.mask == GEOMETRY_MASK_LIGHT_HIDDEN);
    // For emissive geometry userID indexes the light table, not the geometry
    // table; the flag bit tells `shade` which one it is.
    HitRecord rec;
    rec.geomEntryIndex = isLight ? (HIT_LIGHT_BIT | inst.userID) : (inst.userID + hit.geometryId);
    rec.instanceIndex = hit.instanceId;
    rec.primitiveId = hit.primitiveId;
    // A curve hit has no barycentrics; what it has is one parameter along the
    // segment. It rides in the same two floats rather than in a field of its own,
    // because `shade` already has to read the geometry entry to find the
    // material and the entry says which kind of primitive this is.
    rec.barycentrics =
        (hit.type == intersection_type::curve) ? vector_float2(hit.curveParameter, 0.0f) : hit.barycentrics;
    rec.distance = hit.distance;
    *wavefrontHitRecord(hits, tid) = rec;
    queuePush(uniforms, hitCounter, hitQueue, tid, control[WF_CTRL_CAPACITY]);
}


#define WF_EXTEND_ENTRY(NAME, TRAITS)                                                                                  \
    kernel void NAME(                                                                                                  \
        uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]],                               \
        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(1)]],                          \
        TRAITS::structure accelerationStructure [[buffer(2)]], device const PathRay* rays [[buffer(3)]],               \
        device char* hits [[buffer(4)]], constant uint32_t& sampleIdx [[buffer(5)]],                                   \
        device const uint32_t* queue [[buffer(6)]], device const uint32_t* control [[buffer(7)]],                      \
        device uint32_t* hitQueue [[buffer(8)]], device atomic_uint* hitCounter [[buffer(9)]],                         \
        device uint32_t* missQueue [[buffer(10)]], device atomic_uint* missCounter [[buffer(11)]],                     \
        device const PathState* paths [[buffer(12)]], device const Material* materials [[buffer(13)]],                 \
        constant uint32_t& rayMask [[buffer(14)]], TRAITS::structure volumeAccelerationStructure [[buffer(15)]],       \
        constant uint32_t& queueOffset [[buffer(16)]], device const MediumPathState* mediumPaths [[buffer(17)]],       \
        TRAITS::table functionTable [[buffer(19)]])                                                                    \
    {                                                                                                                  \
        extendImpl<TRAITS>(gid + queueOffset, uniforms, instances, accelerationStructure, volumeAccelerationStructure, \
                           rays, hits, sampleIdx, queue, control, hitQueue, hitCounter, missQueue, missCounter, paths, \
                           materials, mediumPaths, functionTable, rayMask);                                            \
    }

WF_EXTEND_ENTRY(wavefrontExtend, MotionTraversal)
WF_EXTEND_ENTRY(wavefrontExtendStatic, StaticTraversal)
WF_EXTEND_ENTRY(wavefrontExtendCurve, CurveMotionTraversal)
WF_EXTEND_ENTRY(wavefrontExtendStaticCurve, CurveStaticTraversal)

// Rebuild the triangle's vertex attributes from the vertex buffer.
//
// intersection.primitive_data is only addressable inside the kernel that ran the
// intersect, so `shade` cannot reach the per-primitive record `extend` saw. The
// vertex buffer holds the same values — the skinning pass writes it, and the
// per-primitive record is derived from it — so refetching here is equivalent and
// keeps the hit record small.
static void fetchTriangle(device const char* vertexBuffer,
                          device const char* prevVertexBuffer,
                          device const uint32_t* indexBuffer,
                          GeometryEntry entry,
                          uint32_t primitiveId,
                          bool interpolateMotion,
                          float motionTime,
                          thread float3* p,
                          thread float3* n,
                          thread float3* t,
                          thread float2* uv,
                          thread float& tangentSign,
                          thread float3* vcol)
{
    // Scene::Vertex is 32 bytes; these offsets are shared with ShaderTypes.h and guarded by host static_asserts.
    constexpr uint32_t vtxStride = 32;
    constexpr uint32_t tangentOff = 12;
    constexpr uint32_t normalOff = 16;
    constexpr uint32_t uvOff = 20;
    constexpr uint32_t colorOff = 28; // Scene::Vertex::color, packed RGBA8

    uint32_t idx[3];
    idx[0] = indexBuffer[entry.indexOffset + primitiveId * 3 + 0];
    idx[1] = indexBuffer[entry.indexOffset + primitiveId * 3 + 1];
    idx[2] = indexBuffer[entry.indexOffset + primitiveId * 3 + 2];

    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t byteOff = (entry.vbOffset + idx[k]) * vtxStride;
        device const char* v = vertexBuffer + byteOff;

        const float3 pos = float3(*(device const packed_float3*)v);
        const float3 nrm = unpackNormal(*(device const uint32_t*)(v + normalOff));
        const uint32_t tanPacked = *(device const uint32_t*)(v + tangentOff);
        const float3 tan = unpackNormal(tanPacked);
        uv[k] = unpackUV(*(device const uint32_t*)(v + uvOff));
        // Vertex colour is not skinned and does not animate, so it is read from
        // the current frame even when the rest is motion-interpolated.
        vcol[k] = unpackVertexColor(*(device const uint32_t*)(v + colorOff));

        // Handedness is a per-mesh property in every exporter that writes it, so
        // one vertex settles it -- there is nothing sensible to interpolate.
        if (k == 0)
            tangentSign = unpackTangentSign(tanPacked);

        if (interpolateMotion)
        {
            device const char* pv = prevVertexBuffer + byteOff;
            const float3 posPrev = float3(*(device const packed_float3*)pv);
            const float3 nrmPrev = unpackNormal(*(device const uint32_t*)(pv + normalOff));
            const float3 tanPrev = unpackNormal(*(device const uint32_t*)(pv + tangentOff));
            // BVH keyframe 0 is prevVB at t=0 and keyframe 1 is VB at t=1.
            p[k] = mix(posPrev, pos, motionTime);
            n[k] = mix(nrmPrev, nrm, motionTime);
            t[k] = mix(tanPrev, tan, motionTime);
        }
        else
        {
            p[k] = pos;
            n[k] = nrm;
            t[k] = tan;
        }
    }
}

// Rebuild a curve hit from the segment it landed on.
//
// A curve intersection reports the segment and one parameter along it, and
// nothing else -- there is no vertex to interpolate and no uv authored anywhere.
// Everything a shading point needs comes back out of the three curve buffers:
//
//   tangent  the segment's own direction, which for hair is the strand's;
//   normal   the outward radial direction at the hit, which is what makes a
//            round curve shade as a cylinder rather than as a ribbon;
//   uv.x     where along the strand this is, root at 0 and tip at 1.
//
// The radius buffer is not read here at all: the distance from the hit to the
// axis is the radius the intersector used, so the one thing downstream does ask
// about thickness -- where a ray that scatters through the strand comes out --
// costs a square root of a quantity already in hand.
//
// uv.x is recovered from the segment index alone: strands from a particle system
// all have the same number of segments, so `segment % segmentsPerStrand` is the
// position within the strand and no per-point coordinate has to be stored. A set
// with strands of differing lengths reports 0 there, which is a flat root
// colour rather than a wrong gradient.
//
// The normal is derived from the hit point rather than from the parameter,
// because the hit point is exact and the parameter is what the intersector chose
// to report -- and at a sphere cap, where segments overlap, the two disagree.
static void fetchCurve(device const packed_float3* curvePoints,
                       device const uint32_t* curveSegments,
                       GeometryEntry entry,
                       uint32_t primitiveId,
                       float curveParam,
                       float3 worldHit,
                       float4x4 objectToWorld,
                       thread float3& outNormal,
                       thread float3& outTangent,
                       thread float2& outUv,
                       thread float& outRadius)
{
    const bool cubic = (entry.flags & GEOM_CURVE_CUBIC) != 0u;
    // Segment indices are local to the curve set, matching the range exposed to
    // its Metal geometry descriptor. Add the set's scene-wide point offset only
    // when refetching from the shared shader buffer.
    const uint32_t base = entry.vbOffset + curveSegments[entry.indexOffset + primitiveId];
    const float u = saturate(curveParam);
    // Control points are placed by the same transform as the hit, so the axis is
    // built in world space and the radial offset needs no change of basis.
#define WF_CURVE_CP(i) ((objectToWorld * float4(float3(curvePoints[base + (i)]), 1.0f)).xyz)

    float3 axis, dAxis;
    if (cubic)
    {
        // Cubic B-spline, the basis the acceleration structure was built with.
        const float3 p0 = WF_CURVE_CP(0), p1 = WF_CURVE_CP(1);
        const float3 p2 = WF_CURVE_CP(2), p3 = WF_CURVE_CP(3);
        const float u2 = u * u, u3 = u2 * u;
        const float b0 = (1.0f - 3.0f * u + 3.0f * u2 - u3) / 6.0f;
        const float b1 = (4.0f - 6.0f * u2 + 3.0f * u3) / 6.0f;
        const float b2 = (1.0f + 3.0f * u + 3.0f * u2 - 3.0f * u3) / 6.0f;
        const float b3 = u3 / 6.0f;
        const float d0 = (-1.0f + 2.0f * u - u2) * 0.5f;
        const float d1 = (-4.0f * u + 3.0f * u2) * 0.5f;
        const float d2 = (1.0f + 2.0f * u - 3.0f * u2) * 0.5f;
        const float d3 = u2 * 0.5f;
        axis = p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;
        dAxis = p0 * d0 + p1 * d1 + p2 * d2 + p3 * d3;
    }
    else
    {
        const float3 p0 = WF_CURVE_CP(0), p1 = WF_CURVE_CP(1);
        axis = mix(p0, p1, u);
        dAxis = p1 - p0;
    }
#undef WF_CURVE_CP

    outTangent = normalize(dAxis);
    const float3 radial = worldHit - axis;
    // Remove whatever component of the offset runs along the strand: at a
    // spherical cap the closest axis point is not the one the parameter names,
    // and leaving it in tilts the normal toward the tip.
    const float3 perp = radial - outTangent * dot(radial, outTangent);
    const float lenSq = dot(perp, perp);
    // A hit exactly on the axis has no radial direction; anything perpendicular
    // to the strand will do, and this is far rarer than a denormal guard.
    outNormal = lenSq > 1e-16f ? perp * rsqrt(lenSq) : normalize(cross(outTangent, float3(0.0f, 0.0f, 1.0f)));
    outRadius = sqrt(lenSq);

    const uint32_t perStrand = entry.flags & GEOM_CURVE_STRAND_MASK;
    const float alongStrand = perStrand != 0u ? ((float)(primitiveId % perStrand) + u) / (float)perStrand : 0.0f;
    outUv = float2(alongStrand, 0.0f);
}

// The whole-fibre lobe already accounts for absorption across the chord, so transmission starts at the far surface.
// For a cylinder, the chord distance is -2r(n.u) divided by the direction's length across the strand.
static inline float3 fibreExitOrigin(float3 position, float3 tangent, float3 normal, float radius, float3 dir)
{
    const float3 dPerp = dir - tangent * dot(dir, tangent);
    const float m2 = dot(dPerp, dPerp);
    // Straight along the strand there is no far wall, and the chord below would
    // divide by zero on the way to saying so.
    if (m2 < 1e-8f || radius <= 0.0f)
    {
        return offset_ray(position, normal);
    }
    const float m = sqrt(m2);
    const float3 u = dPerp / m;
    const float chord = -2.0f * radius * dot(normal, u);
    // Leaving on the side it arrived from: an ordinary surface offset is enough.
    if (chord <= 0.0f)
    {
        return offset_ray(position, normal);
    }
    // Offset along the outward normal at the exit, not at the entry: they point
    // to opposite sides of the strand.
    return offset_ray(position + dir * (chord / m), normalize(normal * radius + u * chord));
}

// The same fetch, interpolated on the way out.
//
// `shade` is the register-starved kernel of the three -- the pipeline caps it at
// 384 threads per threadgroup where `extend` takes 640, and Instruments records
// it spilling. Holding three vertices of position, normal, tangent, colour and
// uv keeps 42 floats live at once purely to feed a barycentric blend a few lines
// later; blending inside brings that down to twelve. The geometric normal is the
// only thing that needs the vertices themselves, so the two edges are kept and
// the third position is not.
static void fetchTriangleBlended(device const char* vertexBuffer,
                                 device const char* prevVertexBuffer,
                                 device const uint32_t* indexBuffer,
                                 GeometryEntry entry,
                                 uint32_t primitiveId,
                                 bool interpolateMotion,
                                 float motionTime,
                                 float2 bary,
                                 thread float3& outNormal,
                                 thread float3& outTangent,
                                 thread float2& outUv,
                                 thread float3& outColor,
                                 thread float& tangentSign,
                                 thread float3& outGeomNormal,
                                 // For the ray-cone texture LOD: the two object-space
                                 // edges and twice the triangle's area in uv. Both fall
                                 // out of loads this function already does, so the
                                 // footprint costs no extra memory traffic.
                                 thread float3& outEdge1,
                                 thread float3& outEdge2,
                                 thread float& outUvArea2)
{
    constexpr uint32_t vtxStride = 32;
    constexpr uint32_t tangentOff = 12;
    constexpr uint32_t normalOff = 16;
    constexpr uint32_t uvOff = 20;
    constexpr uint32_t colorOff = 28;

    const float w0 = 1.0f - bary.x - bary.y;
    const float3 weight = float3(w0, bary.x, bary.y);

    outNormal = float3(0.0f);
    outTangent = float3(0.0f);
    outColor = float3(0.0f);
    outUv = float2(0.0f);
    float3 p0 = float3(0.0f), e1 = float3(0.0f), e2 = float3(0.0f);
    float2 uv0 = float2(0.0f), uvE1 = float2(0.0f), uvE2 = float2(0.0f);

    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t idx = indexBuffer[entry.indexOffset + primitiveId * 3 + k];
        const uint32_t byteOff = (entry.vbOffset + idx) * vtxStride;
        device const char* v = vertexBuffer + byteOff;

        float3 pos = float3(*(device const packed_float3*)v);
        float3 nrm = unpackNormal(*(device const uint32_t*)(v + normalOff));
        const uint32_t tanPacked = *(device const uint32_t*)(v + tangentOff);
        float3 tan = unpackNormal(tanPacked);

        if (k == 0)
        {
            tangentSign = unpackTangentSign(tanPacked);
        }

        if (interpolateMotion)
        {
            device const char* pvb = prevVertexBuffer + byteOff;
            pos = mix(float3(*(device const packed_float3*)pvb), pos, motionTime);
            nrm = mix(unpackNormal(*(device const uint32_t*)(pvb + normalOff)), nrm, motionTime);
            tan = mix(unpackNormal(*(device const uint32_t*)(pvb + tangentOff)), tan, motionTime);
        }

        outNormal += nrm * weight[k];
        outTangent += tan * weight[k];
        const float2 vertUv = unpackUV(*(device const uint32_t*)(v + uvOff));
        outUv += vertUv * weight[k];
        if (k == 0)
            uv0 = vertUv;
        else if (k == 1)
            uvE1 = vertUv - uv0;
        else
            uvE2 = vertUv - uv0;
        outColor += unpackVertexColor(*(device const uint32_t*)(v + colorOff)) * weight[k];

        if (k == 0)
            p0 = pos;
        else if (k == 1)
            e1 = pos - p0;
        else
            e2 = pos - p0;
    }
    outGeomNormal = cross(e1, e2);
    outEdge1 = e1;
    outEdge2 = e2;
    outUvArea2 = abs(uvE1.x * uvE2.y - uvE2.x * uvE1.y);
}

// Where this hit point stood one frame ago, in world space.
//
// This is the whole difference between a motion vector that describes the scene
// and one that only describes the camera. Two things can have moved a surface
// between frames: the skinning pass rewrote its vertices, and the node holding it
// was re-placed. Both are read from the previous frame's copies, at the same
// barycentric coordinates, so what comes back is the same material point earlier
// in time. Without it a temporal denoiser reprojects a moving limb onto wherever
// that pixel used to be looking and smears the two together -- the ghosting that
// shows up on exactly the animated content the denoiser is supposed to help with.
static inline float3 previousWorldPosition(device const char* prevFrameVertexBuffer,
                                           device const uint32_t* indexBuffer,
                                           device const MTLIndirectAccelerationStructureInstanceDescriptor* prevInstances,
                                           GeometryEntry entry,
                                           uint32_t instanceIndex,
                                           uint32_t primitiveId,
                                           float2 bary)
{
    constexpr uint32_t vtxStride = 32; // see fetchTriangle for the layout

    float3 p[3];
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t idx = indexBuffer[entry.indexOffset + primitiveId * 3 + k];
        p[k] = float3(*(device const packed_float3*)(prevFrameVertexBuffer + (entry.vbOffset + idx) * vtxStride));
    }
    const float3 objectPos = interpolateAttrib(p[0], p[1], p[2], bary);

    const auto inst = prevInstances[instanceIndex];
    const float4x4 prevObjectToWorld =
        float4x4(float4(float3(inst.transformationMatrix[0]), 0.0f), float4(float3(inst.transformationMatrix[1]), 0.0f),
                 float4(float3(inst.transformationMatrix[2]), 0.0f), float4(float3(inst.transformationMatrix[3]), 1.0f));
    return (prevObjectToWorld * float4(objectPos, 1.0f)).xyz;
}

// Depth in whichever convention the denoiser is currently being fed; see
// kDenoiseDepth* for why this is a switch rather than a decision.
static inline float viewDepth(constant Uniforms& uniforms, float3 worldPosition)
{
    if (uniforms.denoiseDepthMode == kDenoiseDepthDevice)
    {
        const float4 clip = uniforms.worldToClip * float4(worldPosition, 1.0f);
        return clip.w > 0.0f ? clip.z / clip.w : 1.0f;
    }
    const float3 eye = (uniforms.viewToWorld * float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz;
    if (uniforms.denoiseDepthMode == kDenoiseDepthViewZ)
    {
        // Along the camera axis, which is what a depth buffer holds before the
        // projection is applied. The third column of viewToWorld is the camera's
        // backward axis, so forward is its negation.
        const float3 forward = -uniforms.viewToWorld[2].xyz;
        return dot(worldPosition - eye, forward);
    }
    return length(worldPosition - eye);
}

// The value the background writes. Device depth has a finite far plane, so the
// sentinel has to match the convention or the denoiser reads the sky as being
// nearer than the geometry.
static inline float backgroundDepth(constant Uniforms& uniforms)
{
    return denoiseBackgroundDepth(uniforms.denoiseDepthMode, uniforms.projectionType);
}

// Where a point was on screen last frame, in pixels, y down. That is the sign
// MetalFX documents: "the motion vectors for an object that moves down and to
// the right by 10 pixels would be (-10,-10)".
//
// The current position is the jittered sample position, not the pixel centre.
// The ray that produced this hit went through the jitter offset, so projecting
// the hit through either unjittered camera lands there. Subtracting the pixel
// centre would therefore leave this frame's jitter in every motion vector; a
// static scene must produce zero motion because MetalFX receives the sampling
// offset separately through jitterOffsetX/Y.
struct ScreenMotion
{
    float2 offset;
    float reactive;
};

static inline ScreenMotion screenMotion(constant Uniforms& uniforms, float4 prevClip, uint2 pixel)
{
    ScreenMotion result;
    result.offset = float2(0.0f);
    result.reactive = 0.0f;
    // Near-zero clip w has no valid reprojection. This is the exceptional case
    // the reactive mask is for: favor the current frame because there is no
    // meaningful history location to sample.
    const float kMinW = 1e-4f;
    if (prevClip.w <= kMinW)
    {
        result.reactive = 1.0f;
        return result;
    }
    const float2 prevNdc = prevClip.xy / prevClip.w;
    const float2 prevPixel = float2(
        (prevNdc.x * 0.5f + 0.5f) * (float)uniforms.width, (1.0f - (prevNdc.y * 0.5f + 0.5f)) * (float)uniforms.height);
    // This is the raster location of the world-space hit under the current
    // unjittered camera. Using the centre here would make a still scene report
    // exactly the jitter as motion, which is not a dejittered vector.
    const float2 currPixel = float2((float)pixel.x + 0.5f + uniforms.jitterX, (float)pixel.y + 0.5f + uniforms.jitterY);
    const float2 motion = prevPixel - currPixel;
    // Nothing that moved further than the frame is across in one frame can be
    // reprojected onto anything: past that the history lookup lands outside the
    // image, and the value is far more likely to be a reprojection artefact than
    // a real displacement. Clamped rather than zeroed so a genuinely fast object
    // still drags its history in the right direction.
    const float limit = (float)(uniforms.width + uniforms.height);
    result.offset = clamp(motion, -limit, limit);
    return result;
}

struct DenoiserMaterialGuides
{
    float3 diffuse;
    float3 specular;
    float roughness;
    float transmission;
};

static inline bool openpbrBaseMapDetailsSubsurface(thread const OpenPBRParams& p)
{
    return p.subsurface_weight > 0.0f && openpbrHasMap(p, OPENPBR_TEX_BASE_COLOR) &&
           !openpbrHasMap(p, OPENPBR_TEX_SUBSURFACE_COLOR);
}

// MetalFX asks for albedos that approximate the diffuse and specular radiance
// visible from this view, rather than the material's normal-incidence F0. Keep
// the layered lobes in that approximation: otherwise a grazing dielectric or a
// clear coat is described as almost black precisely where its highlight fills
// the pixel.
static inline DenoiserMaterialGuides standardDenoiserGuides(thread const SurfaceInteraction& si)
{
    DenoiserMaterialGuides g;
    const float3 base = float3(si.albedo);
    const float dielectric = 1.0f - si.metallic;
    g.diffuse = base * dielectric * (1.0f - si.transmission) * (1.0f - si.diffuse_transmission);

    const float cosView = saturate(abs(dot(float3(si.shading_normal), float3(si.wo))));
    const float3 f0 = gltf_f0(si.ior, si.specular, si.specular_color, si.albedo, si.metallic);
    g.specular = fresnel_schlick_roughness(f0, cosView, si.roughness);

    const float coatFresnel = si.clearcoat * fresnel_schlick_scalar(f0_from_ior(max(si.clearcoat_ior, 1.0f)), cosView);
    g.specular = g.specular + (float3(1.0f) - g.specular) * coatFresnel;
    const float3 fuzz = si.sheen_color * si.sheen * powr(1.0f - cosView, 5.0f);
    g.specular = saturate(g.specular + (float3(1.0f) - g.specular) * fuzz);

    const float mainEnergy = max(luminance(g.specular), 1e-4f);
    const float coatEnergy = max(coatFresnel, 0.0f);
    const float fuzzEnergy = max(luminance(fuzz), 0.0f);
    g.roughness =
        saturate((si.roughness * mainEnergy + si.clearcoat_roughness * coatEnergy + si.sheen_roughness * fuzzEnergy) /
                 (mainEnergy + coatEnergy + fuzzEnergy));
    g.transmission = max(si.transmission, si.diffuse_transmission);
    return g;
}

static inline DenoiserMaterialGuides openpbrDenoiserGuides(thread const OpenPBR_ResolvedInputs& in,
                                                           thread const SurfaceInteraction& si,
                                                           bool baseMapDetailsSubsurface)
{
    DenoiserMaterialGuides g;
    const float dielectric = 1.0f - in.base_metalness;
    const float opaque = 1.0f - in.transmission_weight;
    const float3 weightedBase = in.base_color * in.base_weight;
    // MaterialX exports in the test scenes use a mapped base plus a constant
    // subsurface tint. Preserve the mapped detail in that case; a genuinely
    // mapped subsurface colour remains an independent OpenPBR input.
    const float3 subsurfaceColor = baseMapDetailsSubsurface ? weightedBase * in.subsurface_color : in.subsurface_color;
    const float3 diffuseColor = mix(weightedBase, subsurfaceColor, in.subsurface_weight);
    g.diffuse = diffuseColor * dielectric * opaque;

    const float cosView = saturate(abs(dot(float3(si.shading_normal), float3(si.wo))));
    const float dielectricF0 = f0_from_ior(max(in.specular_ior, 1.0f)) * in.specular_weight;
    const float3 dielectricSpecular = saturate(in.specular_color * dielectricF0);
    const float3 metalSpecular = saturate(weightedBase * in.specular_weight);
    const float3 f0 = mix(dielectricSpecular, metalSpecular, in.base_metalness);
    g.specular = fresnel_schlick_roughness(f0, cosView, in.specular_roughness);

    const float coatFresnel = in.coat_weight * fresnel_schlick_scalar(f0_from_ior(max(in.coat_ior, 1.0f)), cosView);
    const float3 coated = g.specular + (float3(1.0f) - g.specular) * in.coat_color * coatFresnel;
    const float3 fuzz = in.fuzz_color * in.fuzz_weight * powr(1.0f - cosView, 5.0f);
    g.specular = saturate(coated + (float3(1.0f) - coated) * fuzz);

    const float mainEnergy = max(luminance(g.specular), 1e-4f);
    const float coatEnergy = max(coatFresnel, 0.0f);
    const float fuzzEnergy = max(luminance(fuzz), 0.0f);
    const float roughnessEnergy =
        in.specular_roughness * mainEnergy + in.coat_roughness * coatEnergy + in.fuzz_roughness * fuzzEnergy;
    g.roughness = saturate(roughnessEnergy / (mainEnergy + coatEnergy + fuzzEnergy));
    g.transmission = in.transmission_weight;
    return g;
}

static inline float denoiserInterfaceIor(bool isOpenPBR,
                                         thread const OpenPBRParams& openpbr,
                                         thread const SurfaceInteraction& si,
                                         bool entering)
{
    if (!isOpenPBR)
    {
        return max(si.ior, 1.0f);
    }

    // Match OpenPBR's preparation: specular_weight index-matches the base to
    // the surrounding medium, which is a partial coat while entering and the
    // tracked exterior otherwise. Using the authored IOR directly invents a
    // reflection when specular_weight is zero.
    const float exterior = max(si.exterior_ior, 1e-4f);
    const bool entersCoat = entering && openpbr.coat_weight > 0.0f;
    const float surrounding = entersCoat ? mix(exterior, max(openpbr.coat_ior, 1.0f), openpbr.coat_weight) : exterior;
    return surrounding *
           openpbr_apply_specular_weight_to_ior(max(openpbr.specular_ior, 1.0f) / surrounding, openpbr.specular_weight);
}

static inline uint32_t packGuideIors(float currentIor, float exteriorIor)
{
    return as_type<uint32_t>(half2(currentIor, exteriorIor));
}

static inline float2 unpackGuideIors(uint32_t packed)
{
    return float2(as_type<half2>(packed));
}

// ---------------------------------------------------------------------------
// miss -- rays that escaped the scene
//
// Split out of `shade` rather than branched inside it. The work is a texture
// fetch and one MIS weight, and keeping it here means `shade` neither dispatches
// threads for escaped rays nor carries the environment sampler in its register
// budget.
// ---------------------------------------------------------------------------
kernel void wavefrontMiss(uint gid [[thread_position_in_grid]],
                          constant Uniforms& uniforms [[buffer(0)]],
                          device const PathState* paths [[buffer(1)]],
                          device const PathRay* rays [[buffer(2)]],
                          device float4* radianceOut [[buffer(3)]],
                          device const uint32_t* queue [[buffer(4)]],
                          device const uint32_t* control [[buffer(5)]],
                          device AovSample* aov [[buffer(6)]],
                          constant uint32_t& sampleIdx [[buffer(7)]],
                          // The escape counter's other half. A path that reaches infinity while its
                          // dielectric stack still holds something left a volume without crossing its
                          // surface -- which is exactly what a hole in a refracting mesh does, and the
                          // failure `shade` cannot see because no exit event ever happens.
                          device const IorStack* iorStacks [[buffer(8)]],
                          device atomic_uint* iorStats [[buffer(9)]],
                          device SharcUpdateState* sharcUpdates [[buffer(10)]],
                          device SharcAccumulationEntry* sharcAccumulation [[buffer(11)]],
                          device const UniformLight* lights [[buffer(12)]],
                          device const EnvAliasEntry* envAliasTable [[buffer(13)]],
                          texture2d<float> envMapTexture [[texture(0)]],
                          texture2d<float> envBackgroundTexture [[texture(1)]])
{
    if (gid >= control[WF_CTRL_MISS_N])
    {
        return;
    }
    const uint32_t tid = queue[gid];
    const PathState p = paths[tid];
    const float3 rayDir = float3(rays[tid].direction);
    const float3 throughput = float3(p.throughput);
    const uint32_t depth = pathDepth(p.depthAndFlags);
    auditWork(uniforms, WORK_MISS_ITEMS_BASE + min(depth, WORK_BOUNCE_SLOTS - 1u));
    const bool specularBounce = (p.depthAndFlags & PATH_FLAG_SPECULAR) != 0u;
    const bool neeDone = (p.depthAndFlags & PATH_FLAG_NEE_DONE) != 0u;

    // Record path depth for the SHARC bounce heatmap.
    if (SPEC_DEBUG && (DebugMode)uniforms.debug == DebugMode::eSharcBounces)
    {
        aov[tid].guideStateOrBounceDepth = (float)depth;
    }

    // Counted here because here is the only place it is visible: the path is
    // gone and it still thinks it is inside glass. See ior_stack.h.
    if (iorStacks[tid].top >= 0)
    {
        atomic_fetch_add_explicit(&iorStats[IOR_STAT_ESCAPED_INSIDE], 1u, memory_order_relaxed);
    }

    // Escapes must overwrite stale denoiser guides, including those reached after a specular primary hit.
    if (shouldWriteAov(uniforms, sampleIdx) && (depth == 0u || (p.depthAndFlags & PATH_FLAG_AOV_DONE) == 0u))
    {
        AovSample a;
        a.diffuseAlbedo = packed_float3(float3(0.0f));
        a.specularAlbedo = packed_float3(float3(0.0f));
        a.normal = packed_float3(-rayDir);
        a.roughness = 1.0f;
        a.depth = backgroundDepth(uniforms);
        // Camera rays reproject the sky at infinity; bounced rays retain the primary surface's motion instead.
        const ScreenMotion motion = (depth == 0u) ?
                                        screenMotion(uniforms, uniforms.prevWorldToClip * float4(rayDir, 0.0f),
                                                     uint2(tid % uniforms.width, tid / uniforms.width)) :
                                        ScreenMotion{ float2(0.0f), 0.0f };
        a.specularHitDistance = 0.0f;
        // Primary background is noise-free but still has valid camera motion.
        // Denoise strength handles the former; rejecting temporal history here
        // throws away the subpixel samples the upscaler needs for the latter.
        a.reactive = depth == 0u ? motion.reactive : 0.0f;
        a.guideStateOrBounceDepth = depth == 0u ? -1.0f : 0.0f;
        if (depth == 0u)
        {
            a.motionX = motion.offset.x;
            a.motionY = motion.offset.y;
            aov[tid] = a;
        }
        else
        {
            // Bounced sky keeps the camera-visible surface's depth and motion for reprojection.
            const float state = aov[tid].guideStateOrBounceDepth;
            const bool blendTransmission = state >= 1.0f;
            const float interfaceFresnel = blendTransmission ? saturate(state - 1.0f) : 0.0f;
            if (blendTransmission)
            {
                const float3 primaryNormal = float3(aov[tid].normal);
                const float3 skyNormal = -rayDir;
                const float3 blendedNormal = primaryNormal * interfaceFresnel + skyNormal * (1.0f - interfaceFresnel);
                a.specularAlbedo = packed_float3(float3(interfaceFresnel));
                a.normal =
                    packed_float3(dot(blendedNormal, blendedNormal) > 1e-8f ? normalize(blendedNormal) : skyNormal);
                a.roughness = mix(aov[tid].roughness, 1.0f, 1.0f - interfaceFresnel);
            }
            a.depth = aov[tid].depth;
            a.motionX = aov[tid].motionX;
            a.motionY = aov[tid].motionY;
            a.specularHitDistance = aov[tid].specularHitDistance;
            a.reactive = aov[tid].reactive;
            aov[tid] = a;
        }
    }
    // A specular primary that escaped into the environment still has a hit
    // distance -- infinity -- and leaving it at zero tells MetalFX the
    // reflection sits on the mirror.
    if (shouldWriteAov(uniforms, sampleIdx) && depth > 0u && specularBounce &&
        (p.depthAndFlags & PATH_FLAG_AOV_DONE) == 0u)
    {
        aov[tid].specularHitDistance += max(uniforms.sceneExtent, 1e3f);
    }

    // A single-hit debug view and the SHaRC surface diagnostics are answers about
    // a surface. The background has nothing to say in either, and an environment
    // brighter than the debug colours drowns them where it does appear.
    if (!SPEC_SHARC_UPDATE &&
        ((SPEC_DEBUG && DEBUG_MODE_IS_SINGLE_HIT(uniforms.debug)) ||
         (SPEC_SHARC && uniforms.sharcCapacity != 0u && SHARC_DEBUG_IS_SURFACE_VIEW(uniforms.sharcDebug))))
    {
        return;
    }

    float3 radiance = float3(0.0f);
    float3 sharcEnvironment = float3(0.0f);
    if (SPEC_ENV_MAP && uniforms.hasEnvMap)
    {
        auditWork(uniforms, WORK_ENVIRONMENT_EVALUATIONS);
        // repeat in u, clamp in v. The map wraps in azimuth and does not wrap in
        // polar angle: with repeat on both axes the bilinear tap in the first row
        // blends the zenith with the last row, which is the nadir. OptiX has always
        // clamped v (cudaAddressModeClamp on axis 1); this backend wrapped it, so the
        // two disagreed on the one row where an equirectangular map has a seam that
        // is not a seam.
        constexpr sampler envSampler(
            mag_filter::linear, min_filter::linear, s_address::repeat, t_address::clamp_to_edge, coord::normalized);
        const float2 envUV = dirToEnvUV(rayDir, uniforms.envMapRotation);
        float3 envColor = envMapTexture.sample(envSampler, envUV).xyz;
        envColor *= uniforms.envMapIntensity * uniforms.envMapColorTint.xyz;

        if (depth == 0u || specularBounce || !neeDone)
        {
            // A ray still at depth 0 has scattered off nothing -- it may have
            // passed through cutout foliage, which Cycles also counts as a
            // camera ray -- so this is exactly where the backdrop belongs, and
            // the MIS branch below is left reading the lighting environment
            // because that is the one that was importance sampled.
            if (uniforms.hasEnvBackground && depth == 0u)
            {
                envColor = envBackgroundTexture.sample(envSampler, envUV).xyz * uniforms.envBackgroundIntensity *
                           uniforms.envMapColorTint.xyz;
            }
            radiance += throughput * envColor;
            sharcEnvironment = envColor;
        }
        else
        {
            const float envPdf =
                envMapPdf(rayDir, envAliasTable, uniforms.envMapWidth, uniforms.envMapHeight, uniforms.envMapRotation);
            const float envSelectionPdf =
                (uniforms.numLights > 0 || uniforms.numEmissiveMeshes > 0) ? uniforms.envMapColorTint.w : 1.0f;
            const float effectiveEnvPdf = envPdf * envSelectionPdf;
            // A texel of zero luminance has zero sampling density, so light
            // sampling could never have produced this direction and the BSDF
            // strategy owns it outright. Dropping the contribution instead --
            // which the guard used to do -- loses energy exactly along the edges
            // of dark regions, where the bilinear radiance is still non-zero.
            const float mis =
                effectiveEnvPdf > 0.0f ? computeMisWeight(p.lastBsdfPdf, effectiveEnvPdf, uniforms.misHeuristic) : 1.0f;
            radiance += throughput * envColor * mis;
            sharcEnvironment = envColor * mis;
        }
    }
    else
    {
        radiance += throughput * uniforms.missColor;
        sharcEnvironment = uniforms.missColor;
    }

    // Finite distant lights and domes are emitters at infinity. NEE samples
    // them in shade; the complementary BSDF strategy reaches them here. Use the
    // exact same selected-light density and cap support as connectToLight(). A
    // sharp distant is a delta and deliberately has no continuous miss term.
    if (SPEC_LIGHTS)
    {
        const float localSelectionPdf = (SPEC_ENV_MAP && uniforms.hasEnvMap) ? (1.0f - uniforms.envMapColorTint.w) : 1.0f;
        const float analyticClassPdf = uniforms.numEmissiveMeshes > 0u ? (1.0f - uniforms.meshLightSelectionPdf) : 1.0f;
        for (uint32_t infiniteIndex = 0; infiniteIndex < uniforms.numInfiniteLights; ++infiniteIndex)
        {
            auditWork(uniforms, WORK_MISS_LIGHT_EVALUATIONS);
            const uint32_t lightId = uniforms.infiniteLightIndices[infiniteIndex];
            device const UniformLight& light = lights[lightId];
            if (!analyticLightVisibilityAllowsRay(light.normal.w, depth != 0u))
            {
                continue;
            }
            const float3 axis = -float3(light.normal);
            if (light.type == LIGHT_TYPE_DISTANT && distantLightIsDelta(light.halfAngle))
            {
                // This is a discrete path-space event: only a preceding BSDF
                // atom can meet the represented light atom, and no continuous
                // MIS density participates. Exact equality avoids widening the
                // sharp distant into an artificial finite cone.
                if (distantLightDeltaPathMatches(depth, specularBounce, rayDir, axis))
                {
                    const float3 infiniteRadiance = float3(light.color);
                    radiance += throughput * infiniteRadiance;
                    sharcEnvironment += infiniteRadiance;
                }
                continue;
            }
            const float conditionalPdf = infiniteLightConditionalPdf(light.type, light.halfAngle, rayDir, axis);
            if (!(conditionalPdf > 0.0f))
            {
                continue;
            }
            const float effectivePdf =
                localSelectionPdf * analyticClassPdf * analyticLightSelectionPdf(light) * conditionalPdf;
            const float mis = (depth == 0u || specularBounce || !neeDone || !(effectivePdf > 0.0f)) ?
                                  1.0f :
                                  computeMisWeight(p.lastBsdfPdf, effectivePdf, uniforms.misHeuristic);
            const float3 infiniteRadiance = float3(light.color) * mis;
            radiance += throughput * infiniteRadiance;
            sharcEnvironment += infiniteRadiance;
        }
    }
    if (SPEC_SHARC_UPDATE)
    {
        const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
        SharcUpdateState updateState = sharcUpdates[updateIndex];
        sharcUpdateMiss(updateState, uniforms, sharcAccumulation, sharcEnvironment, -rayDir);
        sharcUpdates[updateIndex] = updateState;
    }
    radianceOut[tid] += float4(clampIndirectContribution(radiance, depth, uniforms.clampIndirect), 0.0f);
}

// ---------------------------------------------------------------------------
// shade -- material evaluation, next-event estimation, next ray
// ---------------------------------------------------------------------------
struct RestirEvaluation
{
    LightConnection connection;
    float3 integrand;
    float target;
};

static float restirTargetOnly(thread const LightConnection& connection,
                              thread SurfaceInteraction& si,
                              bool isFibre,
                              thread const ShadedFrame& neeFrame,
                              bool isOpenPBR,
                              thread const OpenPBR_PreparedBsdf& openpbrPrepared,
                              uint32_t misHeuristic)
{
    if (!connection.needsRay || !(connection.pdf > 0.0f) ||
        !neeProposesDirection(
            isFibre, neeFrame.frontFace, neeFrame.normalSign * dot(connection.toLight, si.shading_normal)))
    {
        return 0.0f;
    }
    const BsdfEvalResult evalResult =
        isOpenPBR ? openpbr_bsdf_eval(openpbrPrepared, si, connection.toLight) : bsdf_eval(si, connection.toLight);
    if (!(evalResult.pdf > 0.0f))
    {
        return 0.0f;
    }
    const float misWeight = connection.isDelta ? 1.0f : computeMisWeight(connection.pdf, evalResult.pdf, misHeuristic);
    return luminance(connection.radiance * evalResult.bsdf * misWeight);
}

static RestirEvaluation evaluateRestirConnection(thread const LightConnection& connection,
                                                 thread SurfaceInteraction& si,
                                                 bool isFibre,
                                                 thread const ShadedFrame& neeFrame,
                                                 bool isOpenPBR,
                                                 thread const OpenPBR_PreparedBsdf& openpbrPrepared,
                                                 uint32_t misHeuristic)
{
    RestirEvaluation result = {};
    result.connection = connection;
    // A fibre has no back side to reject, and neither does a leaf: see
    // neeCrossesSurface().
    if (!connection.needsRay || !(connection.pdf > 0.0f) ||
        !neeProposesDirection(neeCrossesSurface(isFibre, si.transmission, si.diffuse_transmission), neeFrame.frontFace,
                              neeFrame.normalSign * dot(connection.toLight, si.shading_normal)))
    {
        return result;
    }
    const BsdfEvalResult evalResult =
        isOpenPBR ? openpbr_bsdf_eval(openpbrPrepared, si, connection.toLight) : bsdf_eval(si, connection.toLight);
    if (!(evalResult.pdf > 0.0f))
    {
        return result;
    }
    const float misWeight = connection.isDelta ? 1.0f : computeMisWeight(connection.pdf, evalResult.pdf, misHeuristic);
    result.integrand = connection.radiance * evalResult.bsdf * misWeight;
    result.target = luminance(result.integrand);
    return result;
}

static bool restirSurfaceHistoryCompatible(thread const RestirSurfaceHistory& current,
                                           thread const RestirSurfaceHistory& previous)
{
    return restirSurfaceCompatible(current.depth, previous.depth,
                                   dot(float3(current.geometryNormal), float3(previous.geometryNormal)),
                                   current.materialIdAndFlags & RESTIR_SURFACE_MATERIAL_MASK,
                                   previous.materialIdAndFlags & RESTIR_SURFACE_MATERIAL_MASK,
                                   (previous.materialIdAndFlags & RESTIR_SURFACE_VALID) != 0u);
}

static uint32_t restirVisibilityQuantizedHash(float3 value, float inverseStep)
{
    const int3 q = int3(floor(value * inverseStep));
    uint32_t key = hash_combine(as_type<uint32_t>(q.x), as_type<uint32_t>(q.y));
    return hash_combine(key, as_type<uint32_t>(q.z));
}

static uint32_t restirVisibilityReceiverKey(constant Uniforms& uniforms,
                                            thread const SurfaceInteraction& si,
                                            uint32_t materialId,
                                            uint32_t geometryKey)
{
    const float positionStep = clamp(uniforms.sceneExtent * 0.005f, 0.02f, 0.1f);
    const int3 q = int3(floor(si.position / positionStep));
    uint32_t key = hash_combine(as_type<uint32_t>(q.x), as_type<uint32_t>(q.y));
    key = hash_combine(key, as_type<uint32_t>(q.z));
    key = hash_combine(key, restirVisibilityQuantizedHash(si.geometry_normal, 4.0f));
    key = hash_combine(key, materialId);
    key = hash_combine(key, geometryKey);
    return ((uniforms.restirVisibilityRevision & 0xffu) << 24u) | (key & RESTIR_VISIBILITY_LIGHT_KEY_MASK);
}

static uint32_t restirVisibilityGeometryKey(uint32_t geomEntryIndex, uint32_t instanceIndex, uint32_t primitiveId)
{
    return hash_combine(hash_combine(geomEntryIndex, instanceIndex), primitiveId) & RESTIR_TARGET_GEOMETRY_MASK;
}

static uint32_t restirVisibilityGeometryKey(thread const RestirTargetSurface& stored)
{
    if ((stored.geomEntryIndex & RESTIR_TARGET_DIRECT) != 0u)
        return ((thread const RestirDirectTargetSurface*)&stored)->flags & RESTIR_TARGET_GEOMETRY_MASK;
    return restirVisibilityGeometryKey(stored.geomEntryIndex, stored.instanceIndex, stored.primitiveId);
}

static uint32_t restirVisibilityLightKey(constant Uniforms& uniforms, thread const LightConnection& connection)
{
    uint32_t key = hash_combine(connection.sample.typeAndLightId, connection.sample.data0);
    key = hash_combine(key, connection.sample.data1);
    key = hash_combine(key, connection.sample.data2);
    const bool infinite = connection.tMax >= 1e15f;
    const float3 endpoint = infinite                       ? connection.toLight :
                            connection.hasVisibilityTarget ? connection.visibilityTarget :
                                                             connection.origin + connection.toLight * connection.tMax;
    const float positionStep = clamp(uniforms.sceneExtent * 0.01f, 0.1f, 0.25f);
    key = hash_combine(key, restirVisibilityQuantizedHash(endpoint, infinite ? 50.0f : 1.0f / positionStep));
    return key & RESTIR_VISIBILITY_LIGHT_KEY_MASK;
}

static bool restirVisibilityReceiverMatches(constant Uniforms& uniforms,
                                            thread const RestirReservoir& reservoir,
                                            thread const SurfaceInteraction& si,
                                            uint32_t materialId,
                                            uint32_t geometryKey)
{
    return reservoir.visibility.receiverKey == restirVisibilityReceiverKey(uniforms, si, materialId, geometryKey);
}

static bool restirVisibilityCacheMatches(constant Uniforms& uniforms,
                                         thread const RestirReservoir& reservoir,
                                         thread const SurfaceInteraction& si,
                                         uint32_t materialId,
                                         uint32_t geometryKey,
                                         uint32_t lightKey,
                                         bool countRejects)
{
    if (countRejects)
        auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_ATTEMPTS);
    if ((reservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBLE) == 0u)
    {
        if (countRejects)
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_REJECT_EMPTY);
        return false;
    }
    if (restirVisibilityAge(reservoir.visibility) > uniforms.restirFinalVisibilityMaxAge)
    {
        if (countRejects)
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_REJECT_AGE);
        return false;
    }
    if ((reservoir.visibility.receiverKey >> 24u) != (uniforms.restirVisibilityRevision & 0xffu))
    {
        if (countRejects)
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_REJECT_REVISION);
        return false;
    }
    if (!restirVisibilityReceiverMatches(uniforms, reservoir, si, materialId, geometryKey))
    {
        if (countRejects)
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_REJECT_RECEIVER);
        return false;
    }
    if ((reservoir.visibility.lightKeyAndAge & RESTIR_VISIBILITY_LIGHT_KEY_MASK) != lightKey)
    {
        if (countRejects)
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_REJECT_LIGHT);
        return false;
    }
    return true;
}

static bool restirCanStoreDirectTarget(thread const SurfaceInteraction& si, bool isFibre, bool isOpenPBR)
{
    if (isFibre || isOpenPBR || si.material_type == MATERIAL_TYPE_HAIR || any(si.shading_normal != si.geometry_normal))
    {
        return false;
    }
    if (si.material_type != MATERIAL_TYPE_STANDARD_PBR)
    {
        return true;
    }
    return si.transmission == 0.0f && si.clearcoat == 0.0f && si.anisotropy == 0.0f && si.specular == 0.5f &&
           all(si.specular_color == float3(1.0f)) && si.iridescence == 0.0f && si.diffuse_transmission == 0.0f &&
           si.sheen == 0.0f && si.subsurface == 0.0f && si.ior == 1.5f && !si.diffuse_faces_away;
}

static bool restirTargetIsDirect(thread const RestirTargetSurface& stored)
{
    return (stored.geomEntryIndex & RESTIR_TARGET_DIRECT) != 0u;
}

static float4x4 restirTargetObjectToWorld(constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                          thread const RestirTargetSurface& stored)
{
    return restirTargetIsDirect(stored) ? float4x4(1.0f) : emissiveObjectToWorld(instances, stored.instanceIndex);
}

static void rebuildRestirTargetSurface(constant Uniforms& uniforms,
                                       float4x4 objectToWorld,
                                       device const Material* materials,
                                       device const GeometryEntry* geometryEntries,
                                       device const char* vertexBuffer,
                                       device const char* prevVertexBuffer,
                                       device const uint32_t* indexBuffer,
                                       device const packed_float3* curvePoints,
                                       device const uint32_t* curveSegments,
                                       thread const RestirTargetSurface& stored,
                                       bool interpolateMotion,
                                       float motionTime,
                                       thread SurfaceInteraction& si,
                                       thread bool& isFibre,
                                       thread bool& isOpenPBR,
                                       thread ShadedFrame& neeFrame,
                                       thread OpenPBR_PreparedBsdf& openpbrPrepared,
                                       thread float& curveRadius);

static inline void auditNeeSampledConnection(constant Uniforms& uniforms, thread const LightConnection& conn)
{
    auditWork(uniforms, WORK_NEE_LIGHT_SAMPLER_CALLS);
    if (!conn.needsRay)
    {
        auditWork(uniforms, WORK_NEE_REJECT_CONNECTION);
        return;
    }
    auditWork(uniforms, WORK_NEE_CONDITIONAL_PDF_EVALUATIONS);
    const uint32_t sampleType = restirSampleType(conn.sample);
    if (sampleType == RESTIR_SAMPLE_ANALYTIC)
        auditWork(uniforms, WORK_NEE_FINITE_LIGHT_INSPECTIONS, 2u);
    else if (sampleType == RESTIR_SAMPLE_EMISSIVE_TRIANGLE)
        auditWork(uniforms, WORK_NEE_EMISSIVE_LIGHT_INSPECTIONS, 4u);
    else if (sampleType == RESTIR_SAMPLE_ENVIRONMENT)
        auditWork(uniforms, WORK_NEE_ENVIRONMENT_SAMPLES);
}

static bool restirDiagnosticVisible(constant Uniforms& uniforms,
                                    CurveStaticTraversal::structure accelerationStructure,
                                    CurveStaticTraversal::table functionTable,
                                    thread const LightConnection& connection,
                                    float3 shadowOrigin,
                                    constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                    device const Material* materials,
                                    device const GeometryEntry* geometryEntries,
                                    device const char* vertexBuffer,
                                    device const uint32_t* indexBuffer,
                                    thread float& transmittance);

kernel void wavefrontShade(uint gid [[thread_position_in_grid]],
                           constant Uniforms& uniforms [[buffer(0)]],
                           constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(1)]],
                           device const IesGpuBufferHeader* iesProfiles [[buffer(2)]],
                           device UniformLight* lights [[buffer(3)]],
                           device Material* materials [[buffer(4)]],
                           device PathState* paths [[buffer(5)]],
                           device PathRay* rays [[buffer(21)]],
                           device char* hits [[buffer(6)]],
                           device float4* radianceOut [[buffer(7)]],
                           device IorStack* iorStacks [[buffer(8)]],
                           device const GeometryEntry* geometryEntries [[buffer(9)]],
                           device const EnvAliasEntry* envAliasTable [[buffer(10)]],
                           device const char* vertexBuffer [[buffer(11)]],
                           device const char* prevVertexBuffer [[buffer(12)]],
                           device const uint32_t* indexBuffer [[buffer(13)]],
                           constant uint32_t& sampleIdx [[buffer(14)]],
                           device const uint32_t* queue [[buffer(15)]],
                           device uint32_t* queueOut [[buffer(16)]],
                           device atomic_uint* outCounter [[buffer(17)]],
                           device uint32_t* control [[buffer(18)]],
                           device ShadowRay* shadowRays [[buffer(19)]],
                           device atomic_uint* shadowCounter [[buffer(20)]],
                           device AovSample* aov [[buffer(22)]],
                           // The previous frame's pose, for motion vectors. Separate from
                           // prevVertexBuffer, which is a motion-blur shutter keyframe and is forced
                           // equal to the current pose whenever motion blur is off.
                           device char* sharcPassBuffer0 [[buffer(23)]],
                           device const char* sharcPassBuffer1 [[buffer(24)]],
                           CurveStaticTraversal::structure diagnosticAccelerationStructure [[buffer(25)]],
                           CurveStaticTraversal::table diagnosticFunctionTable [[buffer(26)]],
                           device const uint32_t* curveSegments [[buffer(27)]],
                           // Two counters for the ways the nested-dielectric stack loses a path; see
                           // ShaderTypes.h. Written only when one of them has already gone wrong.
                           device atomic_uint* iorStats [[buffer(28)]],
                           device char* sharcPassState [[buffer(29)]],
                           device MediumPathState* mediumPaths [[buffer(30)]],
                           texture2d<float> envMapTexture [[texture(0)]])
{
    if (gid >= control[WF_CTRL_HIT_N])
    {
        return;
    }
    const uint32_t tid = queue[gid];
    device SharcHashEntry* sharcHashEntries = (device SharcHashEntry*)uniforms.sharcHashData;
    device const packed_float3* curvePoints = (device const packed_float3*)uniforms.curvePointData;
    device const char* prevFrameVertexBuffer = sharcPassBuffer0;
    device const MTLIndirectAccelerationStructureInstanceDescriptor* prevInstances =
        (device const MTLIndirectAccelerationStructureInstanceDescriptor*)sharcPassBuffer1;
    device SharcUpdateState* sharcUpdates = (device SharcUpdateState*)sharcPassState;
    device SharcAccumulationEntry* sharcAccumulation = (device SharcAccumulationEntry*)sharcPassBuffer0;
    device const SharcResolvedEntry* sharcResolved = SPEC_SHARC_UPDATE ?
                                                         (device const SharcResolvedEntry*)sharcPassBuffer1 :
                                                         (device const SharcResolvedEntry*)sharcPassState;
    PathState p = paths[tid];
    MediumPathState mediumState = {};
    if (SPEC_SSS)
    {
        mediumState = mediumPaths[tid];
    }
    const PathRay pr = rays[tid];

    const uint32_t depth = pathDepth(p.depthAndFlags);
    auditWork(uniforms, WORK_SHADE_ITEMS_BASE + min(depth, WORK_BOUNCE_SLOTS - 1u));
    const bool specularBounce = (p.depthAndFlags & PATH_FLAG_SPECULAR) != 0u;
    const bool neeDone = (p.depthAndFlags & PATH_FLAG_NEE_DONE) != 0u;

    // Bounce heatmap. Every stage that sees a path records how deep it was, so
    // whichever one it dies in has left the answer behind.
    if (SPEC_DEBUG && (DebugMode)uniforms.debug == DebugMode::eSharcBounces)
    {
        aov[tid].guideStateOrBounceDepth = (float)depth;
    }

    SamplerState rng = samplerFor(uniforms, tid, sampleIdx, depth);
    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);

    const float3 rayOrigin = float3(pr.origin);
    const float3 rayDir = float3(pr.direction);
    float3 throughput = float3(p.throughput);
    const float3 throughputAtStageEntry = throughput;
    // The throughput `extend` drew this segment's medium channel from. The
    // nested-dielectric attenuation below moves `throughput` on, and weighting a
    // free flight by a density other than the one it was sampled from is how an
    // unbiased estimator stops being one.
    const float3 sampledThroughput = throughput;

    float3 radiance = float3(0.0f);
    const HitRecord rec = *wavefrontHitRecord(hits, tid);

    // Apply enclosing IOR absorption once before all vertex branches; rec.distance is the segment just travelled.
    // Subsurface walks carry their own extinction, while bounded volumes can still overlap enclosing glass.
    {
        const uint32_t walk = mediumState.medium & MEDIUM_INDEX_MASK;
        const bool inSubsurfaceWalk =
            SPEC_SSS && walk != 0u && (materials[walk - 1u].medium_flags & MEDIUM_FLAG_BOUNDARY) == 0u;
        IorStack preStack = iorStacks[tid];
        const uint32_t inside = ior_stack_current_material(preStack);
        if (!inSubsurfaceWalk && inside != 0xFFFFFFFFu)
        {
            device const Material& im = materials[inside];
            const float3 sigma_t =
                volume_extinction(float3(im.attenuation_color), im.attenuation_distance, uniforms.volumeModel);
            throughput *= beer_lambert_transmittance(sigma_t, rec.distance);
            // Every branch below that persists the path either recomputes this
            // or writes `p` unchanged, so it is written once here.
            p.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : throughput);
        }
    }

    // --- Atmospheric scattering ---------------------------------------------
    //
    // Handled before anything to do with surfaces: the ray never reached one.
    // Free-flight sampling was analog, so the only weight is the single-
    // scattering albedo -- the fraction of an extinction event that scatters
    // rather than absorbs.
    if (SPEC_FOG && (rec.geomEntryIndex & HIT_FOG_BIT) != 0u)
    {
        const float3 scatterPoint = rayOrigin + rayDir * rec.distance;
        throughput *= float3(uniforms.fogAlbedo);
        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
            sharcApplyPendingThroughput(updateState, uniforms);
            sharcUpdates[updateIndex] = updateState;
        }

        // A medium event has a position and no normal, which is what the
        // volumeEvent flag tells the light connection.
        SurfaceInteraction si = {};
        si.position = scatterPoint;
        si.shading_normal = -rayDir;
        si.geometry_normal = -rayDir;
        si.front_face = true;

        // Decided by what this vertex had available, never by what the draw
        // produced -- see volumeNeePairsWithBounce(). Setting it from the
        // outcome hands the bounce ray the whole contribution on exactly the
        // draws where the connection failed, and the two strategies stop
        // summing to one.
        const bool didNee = volumeNeePairsWithBounce(
            uniforms.estimatorMode == 0, (SPEC_LIGHTS && (uniforms.numLights > 0 || uniforms.numEmissiveMeshes > 0)) ||
                                             (SPEC_ENV_MAP && uniforms.hasEnvMap));
        if (didNee)
        {
            auditWork(uniforms, WORK_NEE_ELIGIBLE_HITS);
            const LightConnection conn = connectToLight(uniforms, uniforms.numLights, lights, instances, materials,
                                                        vertexBuffer, prevVertexBuffer, indexBuffer, motionTime, rng,
                                                        si, envAliasTable, envMapTexture, iesProfiles, true);
            auditNeeSampledConnection(uniforms, conn);
            if (conn.needsRay && conn.pdf > 0.0f)
            {
                // The phase cosine compares travel directions: dot(rayDir, toLight).
                const float phase = hgPhase(dot(rayDir, conn.toLight), uniforms.fogAnisotropy);
                // The phase function is the medium's BSDF and its own pdf, so
                // MIS pairs it against the light density exactly as a surface
                // lobe would.
                const float misWeight = conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, phase, uniforms.misHeuristic);
                const float3 weight = throughput * (conn.radiance / conn.pdf) * misWeight * phase;
                const EmissiveVisibilitySegment visibility = lightVisibilitySegment(conn, scatterPoint);
                if (any(weight > 1e-6f) && visibility.valid)
                {
                    auditWork(uniforms, WORK_NEE_VALID_CANDIDATES);
                    const uint32_t slot = atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
                    auditWork(uniforms, WORK_NEE_SHADOW_APPENDS);
                    ShadowRay sr;
                    sr.origin = packed_float3(scatterPoint);
                    sr.direction = packed_float3(visibility.direction);
                    sr.weight = packed_float3(clampIndirectContribution(weight, depth, uniforms.clampIndirect));
                    sr.maxDistance = visibility.maxDistance;
                    sr.pixelIndex = tid;
                    sr.medium = mediumState.medium & MEDIUM_INDEX_MASK;
                    sr.sharcRadiance =
                        packed_float3(SPEC_SHARC_UPDATE ? (conn.radiance / conn.pdf) * misWeight * phase : float3(0.0f));
                    sr.sharcPathIndex = tid;
                    sr.rrCutoff =
                        random<SampleDimension::eShadowRR>(rng, uniforms.samplerType) * kShadowTransmittanceCutoff;
                    shadowRays[slot] = sr;
                }
            }
            else if (conn.needsRay)
            {
                auditWork(uniforms, WORK_NEE_REJECT_PDF);
            }
        }

        float phasePdf = 0.0f;
        const float3 nextDir =
            hgSample(-rayDir, uniforms.fogAnisotropy, random<SampleDimension::eFogPhaseU>(rng, uniforms.samplerType),
                     random<SampleDimension::eFogPhaseV>(rng, uniforms.samplerType), phasePdf);

        radianceOut[tid] += float4(radiance, 0.0f);

        // Roulette on the medium's albedo, which is the only thing multiplying
        // the throughput here. Without it a thin haze is a very long random walk
        // that contributes almost nothing per step.
        const float survive = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05f, 1.0f);
        if (random<SampleDimension::eRussianRoulette>(rng, uniforms.samplerType) >= survive)
        {
            return;
        }
        throughput /= survive;

        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcSetThroughput(updateState, float3(1.0f / max(survive, 1e-5f)));
            sharcUpdates[updateIndex] = updateState;
        }

        PathRay nextRay;
        nextRay.origin = packed_float3(scatterPoint);
        nextRay.direction = packed_float3(nextDir);
        rays[tid] = nextRay;

        p.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : throughput);
        p.lastBsdfPdf = phasePdf;
        p.misDistance = 0.0f;

        // Depth advances: a scattering event is a bounce, and a medium with no
        // depth budget of its own would let a path wander forever.
        p.depthAndFlags =
            (depth + 1u) | PATH_FLAG_ALIVE |
            (p.depthAndFlags & ~(PATH_DEPTH_MASK | PATH_FLAG_ALIVE | PATH_FLAG_SPECULAR | PATH_FLAG_NEE_DONE)) |
            (didNee ? PATH_FLAG_NEE_DONE : 0u);
        if (depth + 1u >= uniforms.maxDepth)
        {
            return;
        }
        paths[tid] = p;
        auditWork(uniforms, WORK_PATH_CONTINUATIONS);
        queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    // --- Subsurface random walk ---------------------------------------------
    //
    // Dense random walks skip NEE; bounded media connect at the scatter vertex.
    // Surface NEE handles light entering and leaving the material.
    if (SPEC_SSS && (rec.geomEntryIndex & HIT_SSS_BIT) != 0u)
    {
        const uint32_t medium = mediumState.medium & MEDIUM_INDEX_MASK;
        const uint32_t step = mediumState.medium >> MEDIUM_STEP_SHIFT;
        device const Material& mm = materials[medium - 1u];
        const MediumProps mp = mediumPropsFor(uniforms, materials, medium - 1u, mediumState.mediumAlbedo);
        const float3 sigmaT = mp.sigmaT;
        const bool isBoundedMedium = (mm.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u;
        const float3 albedo = mp.albedo;

        throughput *= sssScatterWeight(sigmaT, albedo, sssChannelPdf(sampledThroughput, albedo), rec.distance);

        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
            sharcApplyPendingThroughput(updateState, uniforms);
            if (isBoundedMedium)
            {
                sharcPropagate(updateState, sharcAccumulation, float3(mm.medium_emission), uniforms,
                               (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u);
            }
            sharcUpdates[updateIndex] = updateState;
        }

        const float3 scatterPoint = rayOrigin + rayDir * rec.distance;
        SamplerState wrng = samplerFor(uniforms, tid, sampleIdx, depth + step);

        // A bounded volume is the one kind of medium worth connecting to a light
        // from: it is thin, it is lit from outside, and the shafts and the glow
        // are single scattering. A subsurface walk gets neither -- its boundary
        // occludes almost every shadow ray it would spawn.
        const bool isBounded = isBoundedMedium;
        // As in the fog path: available, not delivered.
        const bool didNeeVolume =
            isBounded &&
            volumeNeePairsWithBounce(uniforms.estimatorMode == 0,
                                     (SPEC_LIGHTS && (uniforms.numLights > 0 || uniforms.numEmissiveMeshes > 0)) ||
                                         (SPEC_ENV_MAP && uniforms.hasEnvMap));
        if (isBounded)
        {
            // Volumetric emission: what makes the bath water glow rather than
            // merely tint what is behind it.
            //
            // Clamped like every other contribution. Emission reached through a
            // glass or specular chain arrives with a throughput well above one,
            // and adding that unclamped put fireflies over the entire frame --
            // including the backdrop outside the room, which is what made it
            // obvious the term and not the medium was at fault.
            radiance += clampIndirectContribution(throughput * float3(mm.medium_emission), depth, uniforms.clampIndirect);

            if (didNeeVolume)
            {
                auditWork(uniforms, WORK_NEE_ELIGIBLE_HITS);
                SurfaceInteraction vsi = {};
                vsi.position = scatterPoint;
                vsi.shading_normal = -rayDir;
                vsi.geometry_normal = -rayDir;
                vsi.front_face = true;
                const LightConnection conn = connectToLight(uniforms, uniforms.numLights, lights, instances, materials,
                                                            vertexBuffer, prevVertexBuffer, indexBuffer, motionTime,
                                                            wrng, vsi, envAliasTable, envMapTexture, iesProfiles, true);
                auditNeeSampledConnection(uniforms, conn);
                if (conn.needsRay && conn.pdf > 0.0f)
                {
                    // Match the fog path's travel-direction phase convention.
                    const float phase = hgPhase(dot(rayDir, conn.toLight), mm.subsurface_anisotropy);
                    const float misWeight =
                        conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, phase, uniforms.misHeuristic);
                    const float3 weight = throughput * (conn.radiance / conn.pdf) * misWeight * phase;
                    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(conn, scatterPoint);
                    if (any(weight > 1e-6f) && visibility.valid)
                    {
                        auditWork(uniforms, WORK_NEE_VALID_CANDIDATES);
                        ShadowRay sr;
                        sr.origin = packed_float3(scatterPoint);
                        sr.direction = packed_float3(visibility.direction);
                        sr.weight = packed_float3(clampIndirectContribution(weight, depth, uniforms.clampIndirect));
                        sr.maxDistance = visibility.maxDistance;
                        sr.pixelIndex = tid;
                        sr.medium = mediumState.medium & MEDIUM_INDEX_MASK;
                        sr.sharcRadiance = packed_float3(
                            SPEC_SHARC_UPDATE ? (conn.radiance / conn.pdf) * misWeight * phase : float3(0.0f));
                        sr.sharcPathIndex = tid;
                        sr.rrCutoff =
                            random<SampleDimension::eShadowRR>(wrng, uniforms.samplerType) * kShadowTransmittanceCutoff;
                        const uint32_t slot = atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
                        auditWork(uniforms, WORK_NEE_SHADOW_APPENDS);
                        shadowRays[slot] = sr;
                    }
                }
                else if (conn.needsRay)
                {
                    auditWork(uniforms, WORK_NEE_REJECT_PDF);
                }
            }
        }

        float phasePdf = 0.0f;
        const float3 nextDir =
            hgSample(-rayDir, mm.subsurface_anisotropy, random<SampleDimension::eSssPhaseU>(wrng, uniforms.samplerType),
                     random<SampleDimension::eSssPhaseV>(wrng, uniforms.samplerType), phasePdf);

        radianceOut[tid] += float4(radiance, 0.0f);

        // Roulette on what the walk has left. The step ceiling is a backstop for
        // a medium dense enough that roulette alone would take thousands of
        // steps to end; this is what actually terminates the walk.
        const float survive = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05f, 1.0f);
        if (random<SampleDimension::eRussianRoulette>(wrng, uniforms.samplerType) >= survive)
        {
            return;
        }
        throughput /= survive;

        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcSetThroughput(updateState, float3(1.0f / max(survive, 1e-5f)));
            sharcUpdates[updateIndex] = updateState;
        }

        PathRay nextRay;
        nextRay.origin = packed_float3(scatterPoint);
        nextRay.direction = packed_float3(nextDir);
        rays[tid] = nextRay;

        p.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : throughput);
        p.lastBsdfPdf = phasePdf;
        p.misDistance = 0.0f;
        // Dense walks advance only their step counter; the whole walk consumes one path bounce.
        mediumState.medium = medium | ((step + 1u) << MEDIUM_STEP_SHIFT);
        // Bounded-volume scattering advances path depth so it cannot wander without a bounce budget.
        const uint32_t nextDepth = isBounded ? (depth + 1u) : depth;
        p.depthAndFlags =
            nextDepth | PATH_FLAG_ALIVE |
            (p.depthAndFlags & ~(PATH_DEPTH_MASK | PATH_FLAG_ALIVE | PATH_FLAG_SPECULAR | PATH_FLAG_NEE_DONE)) |
            (didNeeVolume ? PATH_FLAG_NEE_DONE : 0u);
        if (nextDepth >= uniforms.maxDepth)
        {
            return;
        }
        paths[tid] = p;
        mediumPaths[tid] = mediumState;
        auditWork(uniforms, WORK_PATH_CONTINUATIONS);
        queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    // --- Emissive geometry --------------------------------------------------
    if (SPEC_LIGHTS && (rec.geomEntryIndex & HIT_LIGHT_BIT) != 0u)
    {
        const uint32_t lightId = rec.geomEntryIndex & ~HIT_LIGHT_BIT;
        device const UniformLight& currLight = lights[lightId];
        // Traversal already decided the nearest surface and returned its
        // distance. Reconstruct the point and only evaluate normal/PDF; running
        // the full ray/surface intersection here made the same decision twice.
        const float3 hitPoint = rayOrigin + rayDir * rec.distance;
        float3 lightNormal;
        float lightAreaPdf;
        if (currLight.type == LIGHT_TYPE_SPHERE)
        {
            lightAreaPdf = analyticEllipsoidAreaPdfUnchecked(float3(currLight.points[1]), float3(currLight.points[0]),
                                                             float3(currLight.points[2]), float3(currLight.points[3]),
                                                             hitPoint, lightNormal);
        }
        else
        {
            lightNormal = calcLightNormal(currLight, hitPoint);
            lightAreaPdf = calcLightAreaPdf(currLight, hitPoint);
        }
        // A light's geometry is still a surface the denoiser has to reconstruct.
        // Its emission is unaffected by denoising, so it gets a black albedo and
        // its own geometry, which keeps the guides continuous across the edge.
        if (shouldWriteAov(uniforms, sampleIdx) && depth == 0u)
        {
            AovSample a;
            a.diffuseAlbedo = packed_float3(float3(0.0f));
            a.specularAlbedo = packed_float3(float3(0.0f));
            a.normal = packed_float3(-rayDir);
            a.roughness = 1.0f;
            a.depth = viewDepth(uniforms, hitPoint);
            // Analytic lights do not move, so the camera is the only thing that
            // can have displaced them.
            const ScreenMotion motion = screenMotion(uniforms, uniforms.prevWorldToClip * float4(hitPoint, 1.0f),
                                                     uint2(tid % uniforms.width, tid / uniforms.width));
            a.motionX = motion.offset.x;
            a.motionY = motion.offset.y;
            a.specularHitDistance = 0.0f;
            a.reactive = motion.reactive;
            // A directly visible emitter is deterministic. Let MetalFX scale it
            // temporally but do not ask the denoiser to suppress its energy.
            a.guideStateOrBounceDepth = -1.0f;
            aov[tid] = a;
        }
        else if (shouldWriteAov(uniforms, sampleIdx) && (p.depthAndFlags & PATH_FLAG_AOV_DONE) == 0u)
        {
            // An emitter reached through glass is a valid replacement surface,
            // just with no material albedo of its own. Retain the primary
            // interface's Fresnel share and reprojection data.
            const float state = aov[tid].guideStateOrBounceDepth;
            const float interfaceFresnel = state >= 1.0f ? saturate(state - 1.0f) : 0.0f;
            AovSample a;
            a.diffuseAlbedo = packed_float3(float3(0.0f));
            a.specularAlbedo = packed_float3(float3(interfaceFresnel));
            a.normal = packed_float3(-rayDir);
            a.roughness = mix(aov[tid].roughness, 1.0f, 1.0f - interfaceFresnel);
            a.depth = aov[tid].depth;
            a.motionX = aov[tid].motionX;
            a.motionY = aov[tid].motionY;
            a.specularHitDistance = aov[tid].specularHitDistance;
            a.reactive = 0.0f;
            a.guideStateOrBounceDepth = 0.0f;
            aov[tid] = a;
        }
        if (shouldWriteAov(uniforms, sampleIdx) && depth > 0u && specularBounce &&
            (p.depthAndFlags & PATH_FLAG_AOV_DONE) == 0u)
        {
            aov[tid].specularHitDistance += rec.distance;
        }
        // An emitter is a surface with a grid address like any other, so the
        // single-hit diagnostics answer here as well. Left to the branch below,
        // its emission -- orders of magnitude above any debug colour -- would
        // simply take the pixel.
        if (!SPEC_SHARC_UPDATE && depth == 0u)
        {
            if (SPEC_DEBUG && (DebugMode)uniforms.debug == DebugMode::eSharcGrid)
            {
                radianceOut[tid] = float4(sharcDebugColoredHash(uniforms, hitPoint, lightNormal), 0.0f);
                return;
            }
            if (SPEC_DEBUG && SPEC_SHARC && uniforms.sharcCapacity != 0u &&
                (DebugMode)uniforms.debug == DebugMode::eSharcRadiance)
            {
                radianceOut[tid] = float4(sharcDebugRadiance(uniforms, sharcHashEntries, sharcResolved, hitPoint,
                                                             lightNormal, -rayDir, float3(1.0f)),
                                          0.0f);
                return;
            }
            if (SPEC_SHARC && uniforms.sharcCapacity != 0u && SHARC_DEBUG_IS_SURFACE_VIEW(uniforms.sharcDebug))
            {
                radianceOut[tid] = float4(sharcDebugSurface(uniforms, sharcHashEntries, sharcResolved, hitPoint,
                                                            lightNormal, -rayDir, float3(1.0f)),
                                          0.0f);
                return;
            }
        }
        float3 sharcLight = float3(0.0f);
        const float3 scatteringOrigin = rayOrigin - rayDir * p.misDistance;
        const float lightCosine = -dot(rayDir, lightNormal);
        if (analyticLightVisibilityAllowsRay(currLight.normal.w, depth != 0u) &&
            lightConnectionFacesVertex(currLight.type, lightCosine, 0.0f))
        {
            const float hitDistance = finiteVectorLength(hitPoint - scatteringOrigin);
            const float3 Le = emittedLightRadiance(currLight, -rayDir, hitDistance, iesProfiles);
            float3 weightedLe;
            if (depth == 0u || specularBounce || !neeDone)
            {
                weightedLe = Le;
            }
            else
            {
                const float localSelectionPdf = uniforms.hasEnvMap ? 1.0f - uniforms.envMapColorTint.w : 1.0f;
                const float analyticClassPdf =
                    uniforms.numEmissiveMeshes > 0u ? 1.0f - uniforms.meshLightSelectionPdf : 1.0f;
                const float lightIdentityPdf = analyticLightSelectionPdf(currLight);
                const float lightPdf = areaPdfToSolidAngleMarginalPdf(
                    hitDistance, lightCosine, lightAreaPdf, localSelectionPdf, analyticClassPdf, lightIdentityPdf, 1.0f);
                const float mis = computeMisWeight(p.lastBsdfPdf, lightPdf, uniforms.misHeuristic);
                weightedLe = Le * mis;
            }
            radiance += throughput * weightedLe;
            sharcLight += weightedLe;
        }
        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
            sharcUpdateMiss(updateState, uniforms, sharcAccumulation, sharcLight, -rayDir);
            sharcUpdates[updateIndex] = updateState;
        }
        radianceOut[tid] += float4(clampIndirectContribution(radiance, depth, uniforms.clampIndirect), 0.0f);
        return;
    }

    // --- Surface ------------------------------------------------------------
    const GeometryEntry entry = geometryEntries[rec.geomEntryIndex];
    const bool interpolateMotion =
        SPEC_MOTION_BLUR && uniforms.enableMotionBlur && motionTime < 1.0f && prevVertexBuffer && indexBuffer;

    const float2 bary = rec.barycentrics;
    const bool isCurve = SPEC_CURVES && (entry.flags & GEOM_FLAG_CURVE) != 0u;
    float3 objectNormal, objectTangent, vertexColor, objectGeomNormal;
    float2 uv;
    float tangentSign = 1.0f;
    float3 objEdge1, objEdge2;
    float uvArea2 = 0.0f;
    // Only a curve hit has one, and only the fibre paths below read it.
    float curveRadius = 0.0f;

    const auto inst = instances[rec.instanceIndex];
    const float4x4 objectToWorld =
        float4x4(float4(float3(inst.transformationMatrix[0]), 0.0f), float4(float3(inst.transformationMatrix[1]), 0.0f),
                 float4(float3(inst.transformationMatrix[2]), 0.0f), float4(float3(inst.transformationMatrix[3]), 1.0f));

    const float3 worldPosition = rayOrigin + rayDir * rec.distance;

    float3 shadingNormal, shadingTangent, shadingGeomNormal;
    if (isCurve)
    {
        // Built in world space rather than fetched in object space and
        // transformed out, because the radial normal is taken *from the hit
        // point* -- and the hit point only exists in world space. Coming back the
        // other way would need the transform's inverse, which MSL does not
        // provide and which nothing else in this kernel wants.
        fetchCurve(curvePoints, curveSegments, entry, rec.primitiveId, bary.x, worldPosition, objectToWorld,
                   shadingNormal, shadingTangent, uv, curveRadius);
        // A strand has no separate geometric normal: the surface *is* the
        // cylinder, so the shading normal is the geometric one.
        shadingGeomNormal = shadingNormal;
        vertexColor = float3(1.0f);
        objEdge1 = shadingTangent;
        objEdge2 = shadingNormal;
        uvArea2 = 0.0f; // no uv derivatives, and a strand is thinner than a texel
    }
    else
    {
        fetchTriangleBlended(vertexBuffer, prevVertexBuffer, indexBuffer, entry, rec.primitiveId, interpolateMotion,
                             motionTime, bary, objectNormal, objectTangent, uv, vertexColor, tangentSign,
                             objectGeomNormal, objEdge1, objEdge2, uvArea2);
        shadingNormal = transformNormal(normalize(objectNormal), objectToWorld);
        shadingTangent =
            orthonormalizeTangent(shadingNormal, transformDirection(normalize(objectTangent), objectToWorld));
        shadingGeomNormal = transformNormal(objectGeomNormal, objectToWorld);
    }

    const float3 worldNormal = shadingNormal;
    const float3 worldTangent = shadingTangent;
    // glTF TANGENT.w. Without it the bitangent points the wrong way and every
    // normal map is mirrored along it -- bumps light from the opposite side.
    const float3 worldBinormal = cross(worldNormal, worldTangent) * tangentSign;

    const float3 geomNormal = shadingGeomNormal;

    // --- Crossing the boundary of a participating medium --------------------
    //
    // The gizmo of a bounded fog volume is not a surface: it is where the medium
    // starts and stops. A ray through it toggles which medium it is in and
    // carries on with the same direction and throughput -- unshaded, and without
    // spending a bounce, because a volume the light passes through twice would
    // otherwise cost two of them.
    if (SPEC_SSS && (materials[entry.materialId].medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u)
    {
        // The segment just travelled was attenuated at the top of this kernel,
        // which is the only place that sees all four kinds of vertex.
        //
        // Bounded by the same counter the cutout pass-through uses, and for the
        // same reason: neither advances `depth`, so neither has a natural end. A
        // boundary the ray re-hits through a self-intersection would otherwise
        // toggle the medium forever, and the path would stay in the queue burning
        // samples rather than stopping.
        const uint32_t passes = (p.depthAndFlags & PATH_PASSTHROUGH_MASK) >> PATH_PASSTHROUGH_SHIFT;
        if (passes >= PATH_PASSTHROUGH_MAX)
        {
            radianceOut[tid] += float4(radiance, 0.0f);
            return;
        }
        p.depthAndFlags = (p.depthAndFlags & ~PATH_PASSTHROUGH_MASK) |
                          (((passes + 1u) << PATH_PASSTHROUGH_SHIFT) & PATH_PASSTHROUGH_MASK);

        // Toggle, rather than deciding from the normal.
        //
        // A gizmo's winding is arbitrary: V-Ray decides inside from an
        // inside/outside test and never looks at the normal, so a box exported
        // from it may be wound either way. Reading `entering` off
        // dot(rayDir, geomNormal) therefore inverted one of the two volumes in
        // this scene -- and an inverted volume is not a subtle error, it is a
        // medium that fills all of space except the gizmo. Every path in the
        // frame then scattered in open air, which is what the speckle over the
        // backdrop and the haze through the window were.
        const uint32_t here = (entry.materialId + 1u) & MEDIUM_INDEX_MASK;
        const bool leaving = (mediumState.medium & MEDIUM_INDEX_MASK) == here;
        mediumState.medium = leaving ? 0u : here;

        // Push past the surface on the side the ray is heading, which needs the
        // sign of the normal and not its direction.
        const float3 exitSide = (dot(rayDir, geomNormal) > 0.0f) ? geomNormal : -geomNormal;

        PathRay nextRay;
        nextRay.origin = packed_float3(offset_ray(worldPosition, exitSide));
        nextRay.direction = packed_float3(rayDir);
        rays[tid] = nextRay;

        // The MIS distance has to keep counting: as far as the light at the end
        // of this ray is concerned, the scattering vertex is still the one before
        // the boundary, and resetting it here would inflate the weight the same
        // way a cutout pass-through would.
        p.misDistance += rec.distance;
        radianceOut[tid] += float4(radiance, 0.0f);
        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
            sharcUpdates[updateIndex] = updateState;
        }
        paths[tid] = p;
        mediumPaths[tid] = mediumState;
        auditWork(uniforms, WORK_PATH_CONTINUATIONS);
        queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    // --- Leaving a subsurface medium ----------------------------------------
    //
    // The walk reached the boundary. Everything below -- the material, the BSDF,
    // the cutout test -- describes what happens to a ray arriving from outside,
    // and none of it applies to one on its way out, so the exit is handled here
    // and the rest is skipped.
    //
    // Whatever surface the walk hit is treated as the boundary, not only the
    // object it entered. For the closed shapes this serves that is the same
    // surface; for geometry that interpenetrates it is a simplification, and the
    // alternative is carrying the entry instance and rejecting hits on anything
    // else -- which turns an open mesh into a light leak instead.
    if (SPEC_SSS && (mediumState.medium & MEDIUM_INDEX_MASK) != 0u &&
        (materials[(mediumState.medium & MEDIUM_INDEX_MASK) - 1u].medium_flags & MEDIUM_FLAG_BOUNDARY) == 0u)
    {
        const uint32_t medium = mediumState.medium & MEDIUM_INDEX_MASK;
        const uint32_t step = mediumState.medium >> MEDIUM_STEP_SHIFT;
        const float3 exitAlbedo = unpackMediumAlbedo(mediumState.mediumAlbedo);
        throughput *= sssBoundaryWeight(
            mediumSigmaT(uniforms, materials, medium - 1u), sssChannelPdf(sampledThroughput, exitAlbedo), rec.distance);
        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
            sharcApplyPendingThroughput(updateState, uniforms);
            sharcUpdates[updateIndex] = updateState;
        }

        // Use the geometric normal for offsets and the interpolated normal for the exit lobe and light connection.
        const float3 outwardGeom = (dot(geomNormal, rayDir) > 0.0f) ? geomNormal : -geomNormal;
        const float3 outward = (dot(worldNormal, outwardGeom) > 0.0f) ? worldNormal : -worldNormal;
        SamplerState xrng = samplerFor(uniforms, tid, sampleIdx, depth + step);

        // As in the fog path: available, not delivered. The exit lobe is a cosine
        // hemisphere, which is smooth at every parameter, so there is nothing
        // about the material to ask.
        const bool didNeeExit = volumeNeePairsWithBounce(
            uniforms.estimatorMode == 0, (SPEC_LIGHTS && (uniforms.numLights > 0 || uniforms.numEmissiveMeshes > 0)) ||
                                             (SPEC_ENV_MAP && uniforms.hasEnvMap));
        if (didNeeExit)
        {
            auditWork(uniforms, WORK_NEE_ELIGIBLE_HITS);
            // NEE here and not inside the walk: this is the vertex light can
            // actually reach, and leaving it to BSDF sampling alone is what makes
            // a translucent object the noisiest thing in a frame.
            SurfaceInteraction xsi = {};
            xsi.position = worldPosition;
            xsi.shading_normal = outward;
            xsi.geometry_normal = outward;
            xsi.wo = -rayDir;
            xsi.front_face = true;
            const LightConnection conn = connectToLight(uniforms, uniforms.numLights, lights, instances, materials,
                                                        vertexBuffer, prevVertexBuffer, indexBuffer, motionTime, xrng,
                                                        xsi, envAliasTable, envMapTexture, iesProfiles, false);
            auditNeeSampledConnection(uniforms, conn);
            if (conn.needsRay && conn.pdf > 0.0f)
            {
                const float cosOut = dot(outward, conn.toLight);
                if (cosOut > 0.0f)
                {
                    // The exit lobe is 1/pi; cos/pi is only its MIS density because conn.radiance includes the cosine.
                    const float lobePdf = cosOut * M_1_PI_F;
                    const float misWeight =
                        conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, lobePdf, uniforms.misHeuristic);
                    const float3 weight = throughput * (conn.radiance / conn.pdf) * misWeight * M_1_PI_F;
                    const float3 shadowOrigin = offset_ray(worldPosition, outwardGeom);
                    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(conn, shadowOrigin);
                    if (any(weight > 1e-6f) && visibility.valid)
                    {
                        auditWork(uniforms, WORK_NEE_VALID_CANDIDATES);
                        ShadowRay sr;
                        sr.origin = packed_float3(shadowOrigin);
                        sr.direction = packed_float3(visibility.direction);
                        sr.weight = packed_float3(clampIndirectContribution(weight, depth, uniforms.clampIndirect));
                        sr.maxDistance = visibility.maxDistance;
                        sr.pixelIndex = tid;
                        // Outside the medium: this vertex is the walk leaving it,
                        // and the ray starts on the far side of the boundary.
                        // Carrying the walk's medium here attenuated the whole
                        // distance to the light by a dense extinction that nothing
                        // ever cancelled -- the connection is what lights a
                        // translucent object, and it was arriving at zero.
                        sr.medium = 0u;
                        sr.sharcRadiance = packed_float3(
                            SPEC_SHARC_UPDATE ? (conn.radiance / conn.pdf) * misWeight * M_1_PI_F : float3(0.0f));
                        sr.sharcPathIndex = tid;
                        sr.rrCutoff =
                            random<SampleDimension::eShadowRR>(xrng, uniforms.samplerType) * kShadowTransmittanceCutoff;
                        const uint32_t slot = atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
                        auditWork(uniforms, WORK_NEE_SHADOW_APPENDS);
                        shadowRays[slot] = sr;
                    }
                }
            }
            else if (conn.needsRay)
            {
                auditWork(uniforms, WORK_NEE_REJECT_PDF);
            }
        }

        const float3 exitDir =
            sssCosineDirection(outward, random<SampleDimension::eSssPhaseU>(xrng, uniforms.samplerType),
                               random<SampleDimension::eSssPhaseV>(xrng, uniforms.samplerType));

        radianceOut[tid] += float4(radiance, 0.0f);

        const float survive = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05f, 1.0f);
        if (random<SampleDimension::eRussianRoulette>(xrng, uniforms.samplerType) >= survive)
        {
            return;
        }
        throughput /= survive;

        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcSetThroughput(updateState, float3(1.0f / max(survive, 1e-5f)));
            sharcUpdates[updateIndex] = updateState;
        }

        PathRay nextRay;
        nextRay.origin = packed_float3(offset_ray(worldPosition, outwardGeom));
        nextRay.direction = packed_float3(exitDir);
        rays[tid] = nextRay;

        p.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : throughput);
        p.lastBsdfPdf = fmax(dot(outward, exitDir), 0.0f) * M_1_PI_F;
        p.misDistance = 0.0f;
        mediumState.medium = 0u;
        // Depth advances once for the whole walk, here rather than at the entry:
        // charging it at both ends would cost a translucent surface two bounces
        // to do what an opaque one does in one.
        p.depthAndFlags =
            (depth + 1u) | PATH_FLAG_ALIVE |
            (p.depthAndFlags & ~(PATH_DEPTH_MASK | PATH_FLAG_ALIVE | PATH_FLAG_SPECULAR | PATH_FLAG_NEE_DONE)) |
            (didNeeExit ? PATH_FLAG_NEE_DONE : 0u);
        if (depth + 1u >= uniforms.maxDepth)
        {
            return;
        }
        paths[tid] = p;
        mediumPaths[tid] = mediumState;
        auditWork(uniforms, WORK_PATH_CONTINUATIONS);
        queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    // Derive texture LOD because compute kernels have no derivatives and carrying a cone would expand 24-byte
    // PathState. Specular paths keep pixel spread; diffuse events widen the cone to a hemisphere.
    const float3 worldEdge1 = transformDirection(objEdge1, objectToWorld);
    const float3 worldEdge2 = transformDirection(objEdge2, objectToWorld);
    const float worldArea2 = length(cross(worldEdge1, worldEdge2));
    const float pixelSpread = 2.0f * abs(uniforms.clipToView[1][1]) / float(max(uniforms.height, 1u));
    const bool coneIsPixelWide = (depth == 0u) || ((p.depthAndFlags & PATH_FLAG_SPECULAR) != 0u);
    const float coneSpread = coneIsPixelWide ? pixelSpread : 1.0f;
    const float coneWidthHere = coneSpread * rec.distance;
    float lodBase = -1e30f;
    if (uniforms.textureLodMode != 0u && uvArea2 > 0.0f && worldArea2 > 1e-20f && coneWidthHere > 0.0f)
    {
        const float ndotd = max(abs(dot(geomNormal, rayDir)), 1e-4f);
        lodBase = 0.5f * log2(uvArea2 / worldArea2) + log2(coneWidthHere) - log2(ndotd);
    }

    SurfaceInteraction si;
    initSurfaceInteraction(si, materials[entry.materialId], worldPosition, worldNormal, geomNormal, worldTangent,
                           worldBinormal, uv, rayDir, vertexColor, lodBase);

    // A strand shaded by the whole-fibre lobe: light crosses it in one event, so
    // neither the hemisphere tests nor the ray offsets below apply. Gated on the
    // geometry as well as the material because the chord needs a radius, and only
    // a curve hit has one.
    const bool fibreMaterial = scattersThroughFibre(si);
    const bool isFibre = isCurve && fibreMaterial;

    // Coverage. A MASK surface resolves to 0 or 1 and a BLEND one to its alpha,
    // so one stochastic test covers both: with probability (1 - opacity) the
    // path continues straight through, unchanged and unshaded.
    //
    // Bounces through transparent geometry deliberately do NOT advance `depth`.
    // A hedge of cutout leaves would otherwise exhaust maxDepth before any of
    // its light transport happened. They are bounded by their own counter in the
    // high bits of depthAndFlags, which PATH_DEPTH_MASK (0xFF) and the flags at
    // bits 8..11 leave free.
    //
    // Only when traversal did not already do it. With SPEC_ALPHA the
    // intersection function tests every candidate and a hit that arrives here
    // has already been accepted with probability `opacity` -- testing it a
    // second time makes the effective coverage opacity squared. That reads as
    // extra noise at low sample counts, which is what the harness first showed,
    // and only at 8k samples does it resolve into what it is: a blended plane
    // 27% too transparent.
    if (si.opacity < 1.0f)
    {
        const uint32_t layer = (p.depthAndFlags & PATH_PASSTHROUGH_MASK) >> PATH_PASSTHROUGH_SHIFT;
        SamplerState orng = samplerFor(uniforms, tid, sampleIdx, depth);
        float u = random<SampleDimension::eOpacity>(orng, uniforms.samplerType);
        // Rotated by which layer of cutout this is.
        //
        // Passing through deliberately does not advance `depth`, so the sampler
        // is rebuilt in the same state at every layer and hands back the same
        // number. Two leaves with the same alpha then make the same decision: a
        // ray that slipped through the first slips through the second as well,
        // and a canopy that should pass (1-a)^2 of what reaches it passes (1-a).
        // In a pine forest that is most of the geometry.
        //
        // A rotation rather than a fresh hash, because each layer on its own
        // stays stratified across samples -- white noise here is what turned a
        // blended plane into salt and pepper when this test first moved into
        // traversal.
        if (layer != 0u)
        {
            uint32_t r = sharcHash(layer * 2654435761u + tid * 2246822519u);
            u = fract(u + float(r) * (1.0f / 4294967296.0f));
        }
        if (u >= si.opacity)
        {
            const uint32_t passes = (p.depthAndFlags & PATH_PASSTHROUGH_MASK) >> PATH_PASSTHROUGH_SHIFT;
            radianceOut[tid] += float4(radiance, 0.0f);
            if (passes >= PATH_PASSTHROUGH_MAX)
            {
                return;
            }
            // Step off the surface on the side the ray was travelling, so the
            // next trace cannot re-hit the triangle it just passed through.
            const float3 faceNg = dot(geomNormal, rayDir) > 0.0f ? geomNormal : -geomNormal;
            PathRay through;
            through.origin = packed_float3(offset_ray(worldPosition, faceNg));
            through.direction = packed_float3(rayDir);
            rays[tid] = through;
            p.misDistance += rec.distance;
            p.depthAndFlags = (p.depthAndFlags & ~PATH_PASSTHROUGH_MASK) |
                              (((passes + 1u) << PATH_PASSTHROUGH_SHIFT) & PATH_PASSTHROUGH_MASK);
            if (SPEC_SHARC_UPDATE)
            {
                const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
                SharcUpdateState updateState = sharcUpdates[updateIndex];
                sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
                sharcUpdates[updateIndex] = updateState;
                p.throughput = packed_float3(float3(1.0f));
            }
            paths[tid] = p;
            auditWork(uniforms, WORK_PATH_CONTINUATIONS);
            queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
            return;
        }
    }

    // Resolve OpenPBR before producing guides. Its parameter block and maps are
    // the source of truth for base color, metalness, roughness, layered
    // specular response, and its normal map; the generic Material record only
    // mirrors the subset older shading paths require.
    const bool isOpenPBR = SPEC_OPENPBR && si.material_type == MATERIAL_TYPE_OPENPBR;
    OpenPBRParams openpbrMat;
    if (isOpenPBR)
    {
        openpbrMat = uniforms.openpbrParams[entry.materialId];
        if (openpbrMat.texture_mask != 0u && uniforms.openpbrTextures != nullptr)
        {
            applyOpenPBRTextures(openpbrMat, uniforms.openpbrTextures[entry.materialId], si, uv);
        }
    }

    const DebugMode debugMode = (DebugMode)uniforms.debug;
    if (!SPEC_SHARC_UPDATE && SPEC_DEBUG && debugMode == DebugMode::eSharcGrid && depth == 0u)
    {
        // NVIDIA's hash-grid view is evaluated at the primary world-space hit.
        // It visualizes addressing, not cache contents, and therefore works
        // whether the SHaRC resources are enabled or not. Assigned, not added:
        // a diagnostic answers for the pixel instead of tinting the render.
        radianceOut[tid] = float4(sharcDebugColoredHash(uniforms, si.position, si.geometry_normal), 0.0f);
        return;
    }
    if (SPEC_DEBUG && (debugMode == DebugMode::eMotionBlur || debugMode == DebugMode::eNormal))
    {
        // Debug views replace the radiance outright rather than accumulating.
        float3 dbg;
        if (debugMode == DebugMode::eNormal)
        {
            dbg = (si.shading_normal + float3(1.0f)) * 0.5f;
        }
        else
        {
            // Visualize displacement between the motion-interpolated and current-frame normals.
            float3 nCur[3], pCur[3], tCur[3], cCur[3];
            float2 uvCur[3];
            float signCur = 1.0f; // unused by this debug view
            fetchTriangle(vertexBuffer, prevVertexBuffer, indexBuffer, entry, rec.primitiveId, false, motionTime, pCur,
                          nCur, tCur, uvCur, signCur, cCur);
            dbg = float3(
                motionTime, clamp(length(normalize(objectNormal) - normalize(nCur[0])) * 10.0f, 0.0f, 1.0f), 0.0f);
        }
        radianceOut[tid] += float4(dbg, 0.0f);
        return;
    }

    IorStack iorStack = iorStacks[tid];
    const bool entering = si.front_face;
    si.exterior_ior =
        entering ? ior_stack_current_ior(iorStack) : ior_stack_peek_after_pop_material(iorStack, entry.materialId);

    const float3 surfaceEmission = float3(si.emission);

    // The radiance path owns the primary guides. A mirror or pane of glass may
    // still need the material behind it, and a glossy primary needs a reflected
    // hit distance, but those are geometric continuations rather than another
    // light path. Store one compact ray for the guide kernel instead of tracing
    // a second camera sample through every lighting and shadow stage.
    constexpr float kGuideRoughnessFloor = 0.05f;
    const bool writingAov = shouldWriteAov(uniforms, sampleIdx);
    DenoiserMaterialGuides materialGuides = {};
    bool guideOpaque = true;
    if (writingAov && depth == 0u)
    {
        if (isOpenPBR)
        {
            const OpenPBR_ResolvedInputs openpbrInputs = openpbr_resolve_inputs(openpbrMat, si);
            materialGuides = openpbrDenoiserGuides(openpbrInputs, si, openpbrBaseMapDetailsSubsurface(openpbrMat));
        }
        else
        {
            materialGuides = standardDenoiserGuides(si);
        }
        guideOpaque = materialGuides.transmission <= kGuideRoughnessFloor;

        // Depth and motion always belong to the camera-visible surface, even
        // when its material attributes will be replaced.
        const float3 prevPrimary = uniforms.hasPrevFramePose ?
                                       previousWorldPosition(prevFrameVertexBuffer, indexBuffer, prevInstances, entry,
                                                             rec.instanceIndex, rec.primitiveId, bary) :
                                       worldPosition;
        const ScreenMotion primaryMotion = screenMotion(uniforms, uniforms.prevWorldToClip * float4(prevPrimary, 1.0f),
                                                        uint2(tid % uniforms.width, tid / uniforms.width));
        aov[tid].depth = viewDepth(uniforms, worldPosition);
        aov[tid].motionX = primaryMotion.offset.x;
        aov[tid].motionY = primaryMotion.offset.y;
        aov[tid].reactive = primaryMotion.reactive;
        aov[tid].guideStateOrBounceDepth = 0.0f;

        aov[tid].diffuseAlbedo = packed_float3(materialGuides.diffuse);
        aov[tid].specularAlbedo = packed_float3(materialGuides.specular);
        aov[tid].normal = packed_float3(normalize(float3(si.shading_normal)));
        aov[tid].roughness = materialGuides.roughness;
        aov[tid].specularHitDistance = 0.0f;

        const bool hasPrimaryDiffuse = luminance(materialGuides.diffuse) > 1e-3f;
        const bool replaceMaterial =
            !uniforms.guidePrimaryHit &&
            (!guideOpaque || (!hasPrimaryDiffuse && materialGuides.roughness <= kGuideRoughnessFloor));
        const bool needsSpecularDistance =
            guideOpaque && materialGuides.roughness < 0.5f && luminance(materialGuides.specular) > 0.02f;

        float interfaceIor = max(si.exterior_ior, 1.0f);
        float interfaceFresnel = 0.0f;
        float interfaceCosine = 0.0f;
        if (!guideOpaque)
        {
            interfaceIor = denoiserInterfaceIor(isOpenPBR, openpbrMat, si, entering);
            const float eta = entering ? si.exterior_ior / interfaceIor : interfaceIor / max(si.exterior_ior, 1e-4f);
            interfaceCosine = abs(dot(float3(si.shading_normal), float3(si.wo)));
            interfaceFresnel = fresnel_dielectric(interfaceCosine, eta);
            aov[tid].diffuseAlbedo = packed_float3(float3(0.0f));
            aov[tid].specularAlbedo = packed_float3(float3(interfaceFresnel));
            // Values >= 1 carry the glass Fresnel into the replacement blend.
            aov[tid].guideStateOrBounceDepth = interfaceFresnel < 0.5f ? 1.0f + interfaceFresnel : 0.0f;
        }

        if (!any(surfaceEmission > 0.0f) && uniforms.writeAov && (replaceMaterial || needsSpecularDistance))
        {
            const float3 V = normalize(float3(si.wo));
            const float3 Ns = normalize(float3(si.shading_normal));
            const float3 Nf = dot(Ns, V) >= 0.0f ? Ns : -Ns;
            float3 guideDirection = reflect_dir(-V, Nf);
            bool transmitted = false;
            if (!guideOpaque)
            {
                const float eta = entering ? si.exterior_ior / interfaceIor : interfaceIor / max(si.exterior_ior, 1e-4f);
                float3 refracted;
                const bool validRefraction = si.thin_walled ? true : refract_dir(-V, Nf, eta, interfaceCosine, refracted);
                if (validRefraction && interfaceFresnel < 0.5f)
                {
                    guideDirection = si.thin_walled ? -V : refracted;
                    transmitted = true;
                }
            }

            const float3 faceNg =
                dot(float3(si.geometry_normal), V) > 0.0f ? float3(si.geometry_normal) : -float3(si.geometry_normal);
            GuideRay guide;
            guide.origin = packed_float3(offset_ray(si.position, transmitted ? -faceNg : faceNg));
            guide.flags = GUIDE_RAY_ACTIVE | (replaceMaterial ? GUIDE_RAY_REPLACE_MATERIAL : 0u);
            guide.direction = packed_float3(normalize(guideDirection));
            IorStack guideIorStack = iorStack;
            if (!si.thin_walled && !guideOpaque)
            {
                if (transmitted)
                {
                    if (entering)
                    {
                        ior_stack_push(guideIorStack, si.dielectric_priority, interfaceIor, entry.materialId);
                    }
                    else
                    {
                        ior_stack_pop_material(guideIorStack, entry.materialId);
                    }
                }
                else if (!entering && !ior_stack_has_material(guideIorStack, entry.materialId))
                {
                    // A camera may start inside a dielectric without ever
                    // having pushed it onto the path stack.
                    ior_stack_push(guideIorStack, si.dielectric_priority, interfaceIor, entry.materialId);
                }
            }
            const float guideCurrentIor = ior_stack_current_ior(guideIorStack);
            const float guideExteriorIor =
                guideIorStack.top > 0 ? guideIorStack.entries[guideIorStack.top - 1].ior : 1.0f;
            guide.mediaIors = packGuideIors(guideCurrentIor, guideExteriorIor);
            uniforms.guideRays[tid] = guide;
            queuePush(uniforms, (device atomic_uint*)&control[WF_CTRL_GUIDE], uniforms.guideQueue, tid,
                      control[WF_CTRL_CAPACITY]);
        }

        // The stochastic radiance continuation must not overwrite the primary
        // record. Any requested secondary information now belongs to GuideRay.
        p.depthAndFlags |= PATH_FLAG_AOV_DONE;
        paths[tid].depthAndFlags = p.depthAndFlags;

        if (any(surfaceEmission > 0.0f))
        {
            // Direct emission has no Monte Carlo variance. Preserve small HDR
            // emitters instead of filtering them toward their neighbourhood.
            aov[tid].guideStateOrBounceDepth = -1.0f;
        }
    }

    // --- Sparse Hash Radiance Cache ----------------------------------------
    const float3 diffuseAlbedo = float3(si.albedo) * (1.0f - si.metallic);
    const float3 specularF0 = gltf_f0(si.ior, si.specular, si.specular_color, si.albedo, si.metallic);
    const float3 materialDemodulation = sharcMaterialDemodulation(diffuseAlbedo, specularF0);
    const float receiverRoughness = sharcReceiverRoughness(isOpenPBR, openpbrMat, si);
    const bool cacheableReceiver = sharcReceiverCacheEligible(
        receiverRoughness, isOpenPBR ? openpbrMat.transmission_weight : max(si.transmission, si.diffuse_transmission),
        isFibre, uniforms.sharcRoughnessThreshold);

    if (SPEC_SHARC_UPDATE && uniforms.sharcCapacity != 0u)
    {
        // The upstream contract hands SharcUpdateHit this vertex's direct
        // lighting. A wavefront tracer does not have it yet: the connection is a
        // shadow ray that the next stage resolves, so `shadow` deposits it into
        // this same vertex -- state.cacheIndices[0], weight 1/demodulation --
        // and into the vertices behind it, which is exactly the pair of writes
        // SharcUpdateHit would have made. The direction weight is deferred for
        // the same reason: it is a property of the lobe the BSDF has not been
        // sampled from yet, and sharcSetRadianceDirectionWeight sets it below.
        const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
        SharcUpdateState updateState = sharcUpdates[updateIndex];
        sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
        const bool responsive = (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u;
        const bool continueTracing = sharcUpdateHit(
            updateState, uniforms, sharcHashEntries, sharcAccumulation, sharcResolved,
            uniforms.sharcDebug != 0u ? iorStats + IOR_STAT_COUNT : nullptr, si.position, si.geometry_normal, -rayDir,
            0.0f, materialDemodulation, float3(0.0f), surfaceEmission, cacheableReceiver,
            float(sharcHash(tid ^ uniforms.sharcFrameIndex)) * (1.0f / 4294967296.0f), responsive);
        sharcUpdates[updateIndex] = updateState;
        if (!continueTracing)
        {
            return;
        }
    }
    else if (SPEC_SHARC && uniforms.sharcCapacity != 0u)
    {
        // The cache diagnostics answer for the surface the camera sees, replace
        // the pixel outright, and end the path -- the same contract the upstream
        // sample's debug branch has. Everything they need is a query, so they run
        // ahead of the eligibility gates that the render path is bound by.
        if (SPEC_DEBUG && (DebugMode)uniforms.debug == DebugMode::eSharcRadiance && depth == 0u)
        {
            radianceOut[tid] = float4(sharcDebugRadiance(uniforms, sharcHashEntries, sharcResolved, si.position,
                                                         si.geometry_normal, -rayDir, materialDemodulation),
                                      0.0f);
            return;
        }
        if (SHARC_DEBUG_IS_SURFACE_VIEW(uniforms.sharcDebug) && depth == 0u)
        {
            radianceOut[tid] = float4(sharcDebugSurface(uniforms, sharcHashEntries, sharcResolved, si.position,
                                                        si.geometry_normal, -rayDir, materialDemodulation),
                                      0.0f);
            return;
        }
        if (depth >= uniforms.sharcDepth)
        {
            const SharcAddress address = sharcAddress(uniforms, si.position, si.geometry_normal, false);
            // Both gates are the upstream ones. A segment shorter than a voxel
            // diagonal would read the cell it is standing in, and a scattering
            // lobe whose footprint is finer than a voxel resolves detail the cell
            // cannot hold -- a mirror being the limiting case, with no footprint
            // at any distance.
            const float roughness = min(max(unpackSharcRoughness(p.depthAndFlags), 0.0f), 0.99f);
            const float alpha = roughness * roughness;
            const float alpha2 = alpha * alpha;
            const float footprint = rec.distance * sqrt(max(0.5f * alpha2 / max(1.0f - alpha2, 1e-5f), 0.0f));
            const bool validSegment = rec.distance > address.voxelSize * sqrt(3.0f);
            const bool validLobe = footprint > address.voxelSize;
            device atomic_uint* sharcStats = uniforms.sharcDebug != 0u ? iorStats + IOR_STAT_COUNT : nullptr;
            if (sharcStats && !validSegment)
            {
                atomic_fetch_add_explicit(&sharcStats[SHARC_STAT_SEGMENT_REJECT], 1u, memory_order_relaxed);
            }
            if (sharcStats && validSegment && !validLobe)
            {
                atomic_fetch_add_explicit(&sharcStats[SHARC_STAT_FOOTPRINT_REJECT], 1u, memory_order_relaxed);
            }
            if (sharcStats && validSegment && validLobe && !cacheableReceiver)
            {
                atomic_fetch_add_explicit(&sharcStats[SHARC_STAT_RECEIVER_REJECT], 1u, memory_order_relaxed);
            }
            float3 cached = float3(0.0f);
            uint32_t cachedSamples = 0u;
            if (validSegment && validLobe && cacheableReceiver &&
                sharcQuery(uniforms, sharcHashEntries, sharcResolved, sharcStats, si.position, si.geometry_normal,
                           -rayDir, materialDemodulation, cached, cachedSamples))
            {
                if ((uniforms.sharcFlags & SHARC_FLAG_SEPARATE_EMISSIVE) != 0u)
                {
                    cached += surfaceEmission;
                }
                radiance += throughput * cached;
                radianceOut[tid] += float4(radiance, 0.0f);
                paths[tid] = p;
                return;
            }
        }
    }

    if (any(surfaceEmission > 0.0f))
    {
        float emissionMis = 1.0f;
        if (SPEC_LIGHTS && depth > 0u && !specularBounce && neeDone && !isCurve && uniforms.numEmissiveMeshes > 0u)
        {
            const uint32_t geometryId = rec.geomEntryIndex - inst.userID;
            const float3 misOrigin = rayOrigin - rayDir * p.misDistance;
            const float lightPdf =
                emissiveMeshHitPdf(uniforms, instances, vertexBuffer, prevVertexBuffer, indexBuffer, rec.instanceIndex,
                                   geometryId, rec.primitiveId, misOrigin, worldPosition, motionTime);
            if (lightPdf > 0.0f)
            {
                emissionMis = computeMisWeight(p.lastBsdfPdf, lightPdf, uniforms.misHeuristic);
            }
        }
        radiance += throughput * surfaceEmission * emissionMis;
    }

    // Next-event estimation first, and decided by the material rather than by the
    // bounce.
    //
    // It used to run after bsdf_sample() and be gated on the event that came
    // back, which made the light half of the estimate depend on a draw belonging
    // to the other half. Two things were lost that way: the whole vertex
    // whenever the sample came back BSDF_EVENT_ABSORB (a microfacet draw that
    // landed below the horizon is not a material that absorbs), and the smooth
    // lobe's direct light on every draw the delta lobe won -- over half the
    // draws on a clearcoat with glTF's default coat roughness of 0. See
    // neeRunsAtVertex() and bsdf_has_smooth_lobe().
    //
    // Moving it costs nothing in sample values: every random<Dim>() here is a
    // pure function of (sampleIdx, dimension, seed, depth), so the order the
    // dimensions are drawn in does not change any of them.
    // OpenPBR, prepared once for this vertex.
    //
    // Once, because openpbr_prepare() builds the entire layered lobe stack and
    // both halves of the estimate run here -- eval for the light connection just
    // below, sample for the next segment further down. Preparing inside each
    // would do that work twice at every vertex.
    //
    // Placed after si is finished with rather than beside initSurfaceInteraction:
    // the nested-dielectric exterior IOR is resolved above, and prepare() reads
    // it. SHARC eligibility never mutates these shading parameters.
    OpenPBR_PreparedBsdf openpbrPrepared;
    if (isOpenPBR)
    {
        openpbrPrepared = openpbr_prepare_at(openpbrMat, si, throughput);
    }

    const bool hasEmitter = (SPEC_LIGHTS && (uniforms.numLights > 0 || uniforms.numEmissiveMeshes > 0)) ||
                            (SPEC_ENV_MAP && uniforms.hasEnvMap);
    const bool smoothLobe = isOpenPBR ? openpbr_has_smooth_lobe(openpbrMat) : bsdf_has_smooth_lobe(si);
    bool didNee = neeRunsAtVertex(uniforms.estimatorMode == 0, hasEmitter, smoothLobe);
    if (!smoothLobe)
    {
        auditWork(uniforms, WORK_NEE_DELTA_HITS);
    }
    const ShadedFrame neeFrame =
        shadedFrame(si.front_face, dot(si.shading_normal, si.wo), si.transmission, si.diffuse_transmission);
    if (didNee)
    {
        auditWork(uniforms, WORK_NEE_ELIGIBLE_HITS);
        const bool restirInitial = uniforms.restirDIEnabled != 0u && depth == 0u;
        const uint32_t candidates = restirInitial ?
                                        max(uniforms.initialCandidateCount, 1u) :
                                        (uniforms.restirDIEnabled != 0u ? 1u : max(uniforms.risCandidates, 1u));

        if (restirInitial)
        {
            auditWork(uniforms, WORK_RESTIR_ELIGIBLE_HITS);
            auditWork(uniforms, WORK_RESTIR_INITIAL_CANDIDATES, candidates);
        }
        else if (depth == 0u)
        {
            auditWork(uniforms, WORK_FIRST_BOUNCE_NEE_SAMPLES, candidates);
        }
        else
        {
            auditWork(uniforms, WORK_SECONDARY_NEE_SAMPLES, candidates);
        }

        LightConnection bestConn = makeEmptyConnection();
        float3 bestF = float3(0.0f);
        RestirReservoir reservoir = {};

        for (uint32_t i = 0; i < candidates; ++i)
        {
            // Candidates differ by their scramble, not by their dimension: each
            // one stays a stratified sequence across samples, and the first is
            // the sequence this code drew before RIS existed, so a single
            // candidate reproduces the old image exactly.
            SamplerState crng = rng;
            if (i != 0u)
            {
                crng.seed = hash_combine(rng.seed, i * 0x9E3779B9u);
            }

            const LightConnection conn = connectToLight(uniforms, uniforms.numLights, lights, instances, materials,
                                                        vertexBuffer, prevVertexBuffer, indexBuffer, motionTime, crng,
                                                        si, envAliasTable, envMapTexture, iesProfiles);
            auditNeeSampledConnection(uniforms, conn);
            const RestirEvaluation candidate =
                evaluateRestirConnection(conn, si, isFibre, neeFrame, isOpenPBR, openpbrPrepared, uniforms.misHeuristic);
            if (!(candidate.target > 0.0f))
            {
                if (conn.needsRay && !(conn.pdf > 0.0f))
                    auditWork(uniforms, WORK_NEE_REJECT_PDF);
                else if (conn.needsRay &&
                         !neeProposesDirection(
                             isFibre, neeFrame.frontFace, neeFrame.normalSign * dot(conn.toLight, si.shading_normal)))
                    auditWork(uniforms, WORK_NEE_REJECT_COSINE);
                else if (conn.needsRay)
                    auditWork(uniforms, WORK_NEE_REJECT_TARGET);
                continue;
            }
            auditWork(uniforms, WORK_NEE_VALID_CANDIDATES);
            const float w = candidate.target / conn.pdf;
            // Reuse eLightId under a new scramble so adding RIS does not shift later Sobol dimensions.
            SamplerState arng = crng;
            arng.seed = restirRngStreamSeed(crng.seed, RESTIR_RNG_INITIAL_SALT);
            if (restirReservoirUpdate(reservoir.state, w, candidate.target, 1u,
                                      random<SampleDimension::eLightId>(arng, uniforms.samplerType)))
            {
                bestConn = conn;
                bestF = candidate.integrand;
                reservoir.sample = conn.sample;
            }
        }

        reservoir.state.M = candidates;
        if (restirInitial && uniforms.restirInitialVisibility != 0u &&
            (reservoir.state.ageAndFlags & RESTIR_RESERVOIR_VALID) != 0u)
        {
            auditWork(uniforms, WORK_RESTIR_INITIAL_VISIBILITY_QUERIES);
            const float3 initialShadowOrigin =
                isFibre ? fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius, bestConn.toLight) :
                          bestConn.origin;
            float initialTransmittance;
            if (!restirDiagnosticVisible(uniforms, diagnosticAccelerationStructure, diagnosticFunctionTable, bestConn,
                                         initialShadowOrigin, instances, materials, geometryEntries, vertexBuffer,
                                         indexBuffer, initialTransmittance))
            {
                restirReservoirDiscardSample(reservoir.state);
            }
            else
            {
                restirReservoirStoreInitialVisibility(reservoir.state, initialTransmittance);
                const uint32_t geometryKey =
                    restirVisibilityGeometryKey(rec.geomEntryIndex, rec.instanceIndex, rec.primitiveId);
                restirVisibilityStore(reservoir.visibility,
                                      restirVisibilityReceiverKey(uniforms, si, entry.materialId, geometryKey),
                                      restirVisibilityLightKey(uniforms, bestConn), 0u);
            }
        }
        if (restirInitial)
        {
            const bool oddFrame = (uniforms.frameIndex & 1u) != 0u;
            device RestirCandidateAuditRecord* candidateAudit = restirCandidateAuditRecord(uniforms, tid);
            if (candidateAudit != nullptr)
            {
                candidateAudit->pixelIndex = tid;
                candidateAudit->frameIndex = uniforms.frameIndex;
                candidateAudit->sampleIndex = sampleIdx;
                candidateAudit->rngSeed = rng.seed;
                candidateAudit->rngSampleIndex = rng.sampleIdx;
                candidateAudit->rngDepth = rng.depth;
                candidateAudit->lightDimension = uint32_t(SampleDimension::eLightId);
                candidateAudit->initialStreamSeed = restirRngStreamSeed(rng.seed, RESTIR_RNG_INITIAL_SALT);
                candidateAudit->temporalStreamSeed = restirRngStreamSeed(rng.seed, RESTIR_RNG_TEMPORAL_SALT);
                candidateAudit->spatialStreamSeed = restirRngStreamSeed(rng.seed, RESTIR_RNG_SPATIAL_SALT);
                candidateAudit->currentSample = reservoir.sample;
                candidateAudit->currentBufferIndex = restirCurrentBufferIndex(uniforms.frameIndex);
                candidateAudit->historyBufferIndex = restirHistoryBufferIndex(uniforms.frameIndex);
                candidateAudit->proposalCollision = uniforms.restirProposalCollision;
                candidateAudit->proposalEntropy = uniforms.restirProposalEntropy;
            }
            device RestirSurfaceHistory* currentHistory = oddFrame ? uniforms.restirHistory1 : uniforms.restirHistory0;
            device char* currentSurfaceData = uniforms.restirBiasCorrection != 0u ?
                                                  (oddFrame ? uniforms.restirSurfaceData1 : uniforms.restirSurfaceData0) :
                                                  uniforms.restirSurfaceData0;

            const float3 previousPosition = uniforms.hasPrevFramePose != 0u && !isCurve ?
                                                previousWorldPosition(prevFrameVertexBuffer, indexBuffer, prevInstances,
                                                                      entry, rec.instanceIndex, rec.primitiveId, bary) :
                                                worldPosition;
            const uint2 pixel = uint2(tid % uniforms.width, tid / uniforms.width);
            const ScreenMotion motion =
                screenMotion(uniforms, uniforms.prevWorldToClip * float4(previousPosition, 1.0f), pixel);
            RestirSurfaceHistory surface;
            surface.geometryNormal = packed_float3(si.geometry_normal);
            surface.depth = viewDepth(uniforms, worldPosition);
            surface.materialIdAndFlags = entry.materialId | RESTIR_SURFACE_VALID;

            const float2 previousPixelPosition =
                float2(pixel) + float2(0.5f + uniforms.jitterX, 0.5f + uniforms.jitterY) + motion.offset;

            const uint32_t sampleAndMedium = (sampleIdx & RESTIR_TARGET_SAMPLE_MASK) |
                                             ((mediumState.medium & MEDIUM_INDEX_MASK) << RESTIR_TARGET_MEDIUM_SHIFT);
            if (restirCanStoreDirectTarget(si, isFibre, isOpenPBR))
            {
                RestirDirectTargetSurface targetSurface;
                targetSurface.position = packed_float3(worldPosition);
                targetSurface.shadingNormal = packed_float3(si.shading_normal);
                targetSurface.rayDirection = packed_float3(rayDir);
                targetSurface.albedo = packed_float3(si.albedo);
                targetSurface.flags = RESTIR_TARGET_DIRECT |
                                      ((si.material_type & RESTIR_TARGET_MATERIAL_MASK) << RESTIR_TARGET_MATERIAL_SHIFT) |
                                      (si.front_face ? RESTIR_TARGET_FRONT_FACE : 0u) |
                                      (si.thin_walled ? RESTIR_TARGET_THIN_WALLED : 0u) |
                                      (si.transmission > 0.0f ? RESTIR_TARGET_HAS_TRANSMISSION : 0u) |
                                      (si.diffuse_transmission > 0.0f ? RESTIR_TARGET_HAS_DIFFUSE_TRANSMISSION : 0u);
                targetSurface.flags |=
                    restirVisibilityGeometryKey(rec.geomEntryIndex, rec.instanceIndex, rec.primitiveId);
                targetSurface.roughness = si.roughness;
                targetSurface.metallicOrIor = si.material_type == MATERIAL_TYPE_STANDARD_PBR ? si.metallic : si.ior;
                targetSurface.sampleIdxAndMedium = sampleAndMedium;
                ((device RestirDirectTargetSurface*)currentSurfaceData)[tid] = targetSurface;
            }
            else
            {
                RestirTargetSurface targetSurface;
                targetSurface.position = packed_float3(worldPosition);
                targetSurface.rayDirection = packed_float3(rayDir);
                targetSurface.throughput = packed_float3(throughput);
                targetSurface.barycentrics = bary;
                targetSurface.lodBase = lodBase;
                targetSurface.geomEntryIndex = rec.geomEntryIndex;
                targetSurface.instanceIndex = rec.instanceIndex;
                targetSurface.primitiveId = rec.primitiveId;
                targetSurface.sampleIdxAndMedium = sampleAndMedium;
                ((device RestirTargetSurface*)currentSurfaceData)[tid] = targetSurface;
            }

            // Temporal reuse consumes the same surface interaction and prepared
            // BSDF as initial sampling. Keeping it here avoids reconstructing and
            // evaluating the first hit again in a full hit-queue dispatch.
            if (uniforms.temporalReuseEnabled != 0u && uniforms.restirHistoryValid != 0u && sampleIdx == 0u &&
                motion.reactive == 0.0f)
            {
                const int2 previousPixel = int2(floor(previousPixelPosition));
                if (all(previousPixel >= int2(0)) && previousPixel.x < int(uniforms.width) &&
                    previousPixel.y < int(uniforms.height))
                {
                    const uint32_t previousIndex = uint32_t(previousPixel.y) * uniforms.width + uint32_t(previousPixel.x);
                    device const RestirReservoir* previousReservoirs =
                        oddFrame ? uniforms.restirReservoir0 : uniforms.restirReservoir1;
                    device const RestirSurfaceHistory* previousHistory =
                        oddFrame ? uniforms.restirHistory0 : uniforms.restirHistory1;
                    device const char* previousSurfaceData =
                        oddFrame ? uniforms.restirSurfaceData0 : uniforms.restirSurfaceData1;
                    RestirReservoir previousReservoir = previousReservoirs[previousIndex];
                    const RestirSurfaceHistory oldSurface = previousHistory[previousIndex];
                    const uint32_t age = previousReservoir.state.ageAndFlags & RESTIR_RESERVOIR_AGE_MASK;
                    const bool previousValid = (previousReservoir.state.ageAndFlags & RESTIR_RESERVOIR_VALID) != 0u;
                    if (previousReservoir.state.M != 0u && (!previousValid || age < uniforms.reservoirMaxAge) &&
                        restirSurfaceHistoryCompatible(surface, oldSurface))
                    {
                        restirReservoirLimitHistoryM(
                            previousReservoir.state, reservoir.state.M, max(uniforms.reservoirMaxAge, 1u));
                        RestirEvaluation previousEvaluation = {};
                        RestirLightSample previousSampleAtCurrent = {};
                        const uint32_t previousLightMapping =
                            previousValid ? remapRestirSample(uniforms, lights, previousReservoir.sample, true,
                                                              previousSampleAtCurrent) :
                                            RESTIR_LIGHT_UNMAPPED;
                        if (candidateAudit != nullptr)
                        {
                            candidateAudit->historySample = previousReservoir.sample;
                            candidateAudit->mappedHistorySample = previousSampleAtCurrent;
                            candidateAudit->mappedLightId = previousLightMapping;
                        }
                        const bool previousLightValid = previousLightMapping != RESTIR_LIGHT_UNMAPPED &&
                                                        previousLightMapping != RESTIR_LIGHT_TYPE_CHANGED;
                        if (previousValid && !previousLightValid)
                        {
                            const uint32_t previousSampleType = restirSampleType(previousReservoir.sample);
                            auditWork(
                                uniforms,
                                previousLightMapping == RESTIR_LIGHT_TYPE_CHANGED ? WORK_RESTIR_TEMPORAL_REJECT_TYPE :
                                previousSampleType == RESTIR_SAMPLE_ENVIRONMENT ? WORK_RESTIR_TEMPORAL_REJECT_ENVIRONMENT :
                                previousSampleType == RESTIR_SAMPLE_EMISSIVE_TRIANGLE ? WORK_RESTIR_TEMPORAL_REJECT_MESH :
                                previousSampleType == RESTIR_SAMPLE_INVALID ? WORK_RESTIR_TEMPORAL_REJECT_SURFACE :
                                                                              WORK_RESTIR_TEMPORAL_REJECT_UNMAPPED);
                        }
                        bool previousDuplicateCurrent = false;
                        if (previousValid && previousLightValid)
                        {
                            if (previousSampleAtCurrent.typeAndLightId == reservoir.sample.typeAndLightId)
                                auditWork(uniforms, WORK_RESTIR_TEMPORAL_SAME_LIGHT);
                            previousDuplicateCurrent = restirSamplesEqual(previousSampleAtCurrent, reservoir.sample);
                            if (previousDuplicateCurrent)
                            {
                                auditWork(uniforms, WORK_RESTIR_TEMPORAL_DUPLICATE_CURRENT);
                            }
                            const LightConnection previousConnection = reconnectRestirSample(
                                uniforms, lights, instances, materials, vertexBuffer, prevVertexBuffer, indexBuffer,
                                motionTime, si, envAliasTable, envMapTexture, iesProfiles, previousSampleAtCurrent);
                            previousEvaluation =
                                evaluateRestirConnection(previousConnection, si, isFibre, neeFrame, isOpenPBR,
                                                         openpbrPrepared, uniforms.misHeuristic);
                            if (previousEvaluation.target > 0.0f)
                            {
                                auditWork(uniforms, WORK_RESTIR_TEMPORAL_TARGET_POSITIVE);
                                if (reservoir.state.target > 0.0f)
                                {
                                    const float targetRatio = previousEvaluation.target / reservoir.state.target;
                                    auditWork(uniforms, WORK_RESTIR_TARGET_RATIO_HISTOGRAM_BASE +
                                                            restirTargetRatioHistogramBin(targetRatio));
                                }
                            }
                        }
                        const bool temporalSourceCompatible = !previousValid || previousLightValid;
                        bool selectedHistory = false;
                        if (temporalSourceCompatible)
                        {
                            const uint32_t currentM = reservoir.state.M;
                            SamplerState temporalRng = rng;
                            temporalRng.seed = restirRngStreamSeed(temporalRng.seed, RESTIR_RNG_TEMPORAL_SALT);
                            const bool selectedSampleHasCurrentVisibility =
                                (reservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBLE) != 0u &&
                                previousLightValid && restirSamplesEqual(reservoir.sample, previousSampleAtCurrent);
                            const uint32_t currentVisibilityFlags =
                                reservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK;
                            const RestirVisibilityCache currentVisibility = reservoir.visibility;
                            selectedHistory = restirReservoirUpdate(
                                reservoir.state,
                                restirReservoirMergeWeight(previousReservoir.state, previousEvaluation.target),
                                previousEvaluation.target, previousReservoir.state.M,
                                random<SampleDimension::eLightId>(temporalRng, uniforms.samplerType));
                            if (previousLightValid)
                                auditWork(uniforms, WORK_RESTIR_TEMPORAL_MERGES);
                            if (selectedHistory)
                            {
                                reservoir.sample = previousSampleAtCurrent;
                                if (selectedSampleHasCurrentVisibility)
                                {
                                    reservoir.state.ageAndFlags |= currentVisibilityFlags;
                                    reservoir.visibility = currentVisibility;
                                }
                                else if ((previousReservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBLE) != 0u)
                                {
                                    reservoir.state.ageAndFlags |=
                                        previousReservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK;
                                    reservoir.visibility = previousReservoir.visibility;
                                    restirVisibilityStore(
                                        reservoir.visibility, reservoir.visibility.receiverKey,
                                        reservoir.visibility.lightKeyAndAge & RESTIR_VISIBILITY_LIGHT_KEY_MASK,
                                        min(restirVisibilityAge(previousReservoir.visibility) + 1u, 255u));
                                }
                                auditWork(uniforms, WORK_RESTIR_TEMPORAL_SELECTED);
                                if (previousDuplicateCurrent)
                                    auditWork(uniforms, WORK_RESTIR_TEMPORAL_SELECTED_DUPLICATE_CURRENT);
                            }
                            if (uniforms.restirBiasCorrection != 0u)
                            {
                                const float currentTarget = reservoir.state.target;
                                float previousTarget = previousReservoir.state.target;
                                LightConnection selectedAtPrevious = {};
                                RestirLightSample selectedAtPreviousSample = previousReservoir.sample;
                                const uint32_t selectedPreviousMapping =
                                    selectedHistory ? previousLightMapping :
                                                      remapRestirSample(uniforms, lights, reservoir.sample, false,
                                                                        selectedAtPreviousSample);
                                if (selectedPreviousMapping != RESTIR_LIGHT_UNMAPPED &&
                                    selectedPreviousMapping != RESTIR_LIGHT_TYPE_CHANGED &&
                                    (!selectedHistory ||
                                     (SPEC_RESTIR_RAY_TRACED_DIAGNOSTIC && uniforms.restirBiasCorrection == 2u)))
                                {
                                    const RestirTargetSurface previousStored =
                                        ((device const RestirTargetSurface*)previousSurfaceData)[previousIndex];
                                    SurfaceInteraction previousSi;
                                    bool previousIsFibre, previousIsOpenPBR;
                                    ShadedFrame previousNeeFrame;
                                    OpenPBR_PreparedBsdf previousOpenpbrPrepared;
                                    float previousCurveRadius;
                                    float4x4 previousObjectToWorld = float4x4(1.0f);
                                    if (!restirTargetIsDirect(previousStored))
                                    {
                                        const auto previousInstance = prevInstances[previousStored.instanceIndex];
                                        previousObjectToWorld =
                                            float4x4(float4(float3(previousInstance.transformationMatrix[0]), 0.0f),
                                                     float4(float3(previousInstance.transformationMatrix[1]), 0.0f),
                                                     float4(float3(previousInstance.transformationMatrix[2]), 0.0f),
                                                     float4(float3(previousInstance.transformationMatrix[3]), 1.0f));
                                    }
                                    rebuildRestirTargetSurface(uniforms, previousObjectToWorld, materials,
                                                               geometryEntries, prevFrameVertexBuffer,
                                                               prevFrameVertexBuffer, indexBuffer, curvePoints,
                                                               curveSegments, previousStored, false, 0.0f, previousSi,
                                                               previousIsFibre, previousIsOpenPBR, previousNeeFrame,
                                                               previousOpenpbrPrepared, previousCurveRadius);
                                    selectedAtPrevious = reconnectRestirSampleContext(
                                        uniforms, (device UniformLight*)uniforms.previousLights,
                                        uniforms.previousNumLights, uniforms.previousNumEmissiveMeshes,
                                        uniforms.previousMeshLightSelectionPdf,
                                        uniforms.hasEnvMap != 0u && uniforms.restirEnvironmentHistoryValid != 0u,
                                        uniforms.previousEnvSelectionPdf, prevInstances, materials,
                                        prevFrameVertexBuffer, prevFrameVertexBuffer, indexBuffer, 0.0f, previousSi,
                                        envAliasTable, envMapTexture, iesProfiles, selectedAtPreviousSample);
                                    previousTarget = restirTargetOnly(selectedAtPrevious, previousSi, previousIsFibre,
                                                                      previousNeeFrame, previousIsOpenPBR,
                                                                      previousOpenpbrPrepared, uniforms.misHeuristic);
                                }
                                else if (selectedPreviousMapping == RESTIR_LIGHT_UNMAPPED ||
                                         selectedPreviousMapping == RESTIR_LIGHT_TYPE_CHANGED)
                                {
                                    previousTarget = 0.0f;
                                }
                                if (SPEC_RESTIR_RAY_TRACED_DIAGNOSTIC && uniforms.restirBiasCorrection == 2u &&
                                    previousTarget > 0.0f)
                                {
                                    auditWork(uniforms, WORK_RESTIR_DIAGNOSTIC_QUERIES);
                                    float ignoredTransmittance;
                                    if (!restirDiagnosticVisible(
                                            uniforms, diagnosticAccelerationStructure, diagnosticFunctionTable,
                                            selectedAtPrevious, selectedAtPrevious.origin, instances, materials,
                                            geometryEntries, vertexBuffer, indexBuffer, ignoredTransmittance))
                                    {
                                        previousTarget = 0.0f;
                                    }
                                }
                                const float sourceTargetSum =
                                    float(currentM) * currentTarget + float(previousReservoir.state.M) * previousTarget;
                                restirReservoirApplyBasicNormalization(
                                    reservoir.state, selectedHistory ? previousTarget : currentTarget, sourceTargetSum);
                            }
                        }
                        reservoir.state.ageAndFlags = (reservoir.state.ageAndFlags & RESTIR_RESERVOIR_FLAGS_MASK) |
                                                      (selectedHistory ? min(age + 1u, RESTIR_RESERVOIR_AGE_MASK) : 0u);
                    }
                    else
                    {
                        auditWork(uniforms, WORK_RESTIR_TEMPORAL_REJECT_SURFACE);
                    }
                }
                else
                {
                    auditWork(uniforms, WORK_RESTIR_TEMPORAL_REJECT_SURFACE);
                }
            }
            *((device RestirReservoir*)(hits + size_t(tid) * sizeof(RestirReservoir))) = reservoir;
            currentHistory[tid] = surface;
            const uint32_t restirSlot =
                atomic_fetch_add_explicit((device atomic_uint*)&control[WF_CTRL_RESTIR], 1u, memory_order_relaxed);
            if (restirSlot < control[WF_CTRL_CAPACITY])
            {
                uniforms.restirQueue[restirSlot] = tid;
                if ((restirSlot & 63u) == 0u)
                {
                    atomic_fetch_add_explicit(
                        (device atomic_uint*)&control[WF_CTRL_RESTIR_DIS], 1u, memory_order_relaxed);
                }
            }
        }
        if (!restirInitial)
        {
            const float W = restirReservoirNormalization(reservoir.state);
            if (W > 0.0f)
            {
                const float3 weight = throughput * bestF * W;
                const float3 shadowOrigin = isFibre ? fibreExitOrigin(si.position, si.tangent, si.shading_normal,
                                                                      curveRadius, bestConn.toLight) :
                                                      bestConn.origin;
                const EmissiveVisibilitySegment visibility = lightVisibilitySegment(bestConn, shadowOrigin);
                if (any(weight != 0.0f) && visibility.valid)
                {
                    ShadowRay sr;
                    sr.origin = packed_float3(shadowOrigin);
                    sr.direction = packed_float3(visibility.direction);
                    sr.weight = packed_float3(clampIndirectContribution(weight, depth, uniforms.clampIndirect));
                    sr.maxDistance = visibility.maxDistance;
                    sr.pixelIndex = tid;
                    sr.medium = mediumState.medium & MEDIUM_INDEX_MASK;
                    sr.sharcRadiance = packed_float3(SPEC_SHARC_UPDATE ? bestF * W : float3(0.0f));
                    sr.sharcPathIndex = tid;
                    sr.rrCutoff =
                        random<SampleDimension::eShadowRR>(rng, uniforms.samplerType) * kShadowTransmittanceCutoff;
                    const uint32_t slot = atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
                    auditWork(uniforms, WORK_NEE_SHADOW_APPENDS);
                    shadowRays[slot] = sr;
                }
                else if (!any(weight != 0.0f))
                {
                    auditWork(uniforms, WORK_NEE_SHADOW_REJECT_WEIGHT);
                }
                else
                {
                    auditWork(uniforms, WORK_NEE_SHADOW_REJECT_SEGMENT);
                }
            }
            else
            {
                auditWork(uniforms, WORK_NEE_SHADOW_REJECT_NORMALIZATION);
            }
        }
    }

    const float4 xi = float4(random<SampleDimension::eBSDF0>(rng, uniforms.samplerType),
                             random<SampleDimension::eBSDF1>(rng, uniforms.samplerType),
                             random<SampleDimension::eBSDF2>(rng, uniforms.samplerType),
                             random<SampleDimension::eBSDF3>(rng, uniforms.samplerType));
    const uint32_t lobeWord = randomBits<SampleDimension::eBSDF2>(rng, uniforms.samplerType) >> 9u;
    const uint32_t fresnelWord = randomBits<SampleDimension::eBSDF3>(rng, uniforms.samplerType) >> 9u;
    BsdfSampleResult sampleResult =
        isOpenPBR ? openpbr_bsdf_sample(openpbrPrepared, xi) : bsdf_sample(si, xi, lobeWord, fresnelWord);

    if (sampleResult.event_type == BSDF_EVENT_ABSORB)
    {
        // Whatever next-event estimation queued above stays: it is this vertex's
        // direct lighting and does not depend on where the path went next.
        radianceOut[tid] += float4(radiance, 0.0f);
        return;
    }

    const bool nextSpecular = ((sampleResult.event_type & BSDF_EVENT_SPECULAR) != 0);
    if (SPEC_SHARC_UPDATE && cacheableReceiver)
    {
        const float directionWeight =
            (sampleResult.event_type & BSDF_EVENT_DIFFUSE) != 0 ? 0.0f : 1.0f - saturate(si.roughness);
        const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
        SharcUpdateState updateState = sharcUpdates[updateIndex];
        sharcSetRadianceDirectionWeight(updateState, directionWeight);
        sharcUpdates[updateIndex] = updateState;
    }

    // --- Next segment -------------------------------------------------------
    const float3 faceNg = (dot(si.geometry_normal, si.wo) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
    float3 nextOrigin;
    // Colour the diffuse-transmission lobe applied on the way into a subsurface
    // medium, divided back out below. The walk supplies the colour itself, once
    // per scattering event, so leaving the lobe's copy in charges the first event
    // twice: a sphere came out at albedo x its correct reflectance, which for a
    // deep-red medium is a third of the light it should return. Cycles divides
    // the same factor out for the same reason.
    float3 sssEntryTint = float3(1.0f);
    // Use the refracted entry direction instead of the lobe's cosine draw.
    bool sssRefractedEntry = false;
    // A fibre's transmission lobes do not put the path inside anything: the strand
    // is crossed within the one event, so there is no medium to enter and no entry
    // to match with an exit. Pushing the IOR stack here left every transmitted hair
    // path one level deeper than it came in, and a groom is thousands of hairs deep.
    if ((sampleResult.event_type & BSDF_EVENT_TRANSMISSION) != 0 && !isFibre)
    {
        // A diffuse-transmission event crosses an infinitesimally thin sheet.
        // A subsurface event enters the separately tracked random walk. Neither
        // makes the following surface segment part of a dielectric volume.
        const bool entersMedium = isOpenPBR ?
                                      (openpbrMat.subsurface_weight > 0.0f && openpbrMat.geometry_thin_walled == 0u) :
                                      (si.subsurface > 0.0f);
        const uint32_t entryEvent = isOpenPBR ? (sampleResult.event_type & BSDF_EVENT_TRANSMISSION) :
                                                (sampleResult.event_type & BSDF_EVENT_DIFFUSE_TRANSMISSION);
        const bool startsSubsurfaceWalk = SPEC_SSS && entersMedium && entryEvent != 0u;
        const bool diffuseTransmission = (sampleResult.event_type & BSDF_EVENT_DIFFUSE_TRANSMISSION) != 0u;

        // A thin-walled surface has no interior, so crossing it does not put the
        // path inside anything. Pushing the stack anyway left a ray that had gone
        // through the front of a bubble believing it was inside glass, so the far
        // side was an exit from a dense medium -- and every grazing angle there is
        // past the critical angle.
        if (!si.thin_walled && !startsSubsurfaceWalk && !diffuseTransmission && !fibreMaterial)
        {
            if (entering)
            {
                // Count stack failures because either leaves the path carrying the wrong medium.
                if (ior_stack_full(iorStack))
                {
                    atomic_fetch_add_explicit(&iorStats[IOR_STAT_OVERFLOW], 1u, memory_order_relaxed);
                }
                ior_stack_push(iorStack, si.dielectric_priority, si.ior, entry.materialId);
            }
            else
            {
                if (!ior_stack_has_material(iorStack, entry.materialId))
                {
                    atomic_fetch_add_explicit(&iorStats[IOR_STAT_UNMATCHED], 1u, memory_order_relaxed);
                }
                ior_stack_pop_material(iorStack, entry.materialId);
            }
        }
        nextOrigin = offset_ray(si.position, -faceNg);

        // OpenPBR enters on textured subsurface weight plus any transmission; glTF uses diffuse transmission.
        // Pure-absorption interiors stay Beer-Lambert instead of entering the random walk.
        if (startsSubsurfaceWalk)
        {
            mediumState.medium = (entry.materialId + 1u) & MEDIUM_INDEX_MASK;

            // The walk's albedo, resolved here because this is the last place a
            // texture exists: inside the medium there is no surface to sample.
            // Scaled by how far this point's albedo departs from the one the
            // material's scatter colour was derived from, so a flat material
            // takes the ratio 1 and is unchanged, and marble carries its veining
            // in.
            float3 walkAlbedo;
            if (isOpenPBR)
            {
                // MaterialX exports in the test scenes carry marble veining in
                // base_color and a constant subsurface tint. Fold the former
                // into the latter before OpenPBR maps it to single-scattering
                // albedo; a separately mapped subsurface colour stays intact.
                if (openpbrBaseMapDetailsSubsurface(openpbrMat))
                {
                    const float3 base = openpbr_color_to_float3(openpbrMat.base_color);
                    const float3 subsurface = openpbr_color_to_float3(openpbrMat.subsurface_color) * base;
                    openpbrMat.subsurface_color = OpenPBRColor{ subsurface.x, subsurface.y, subsurface.z };
                }
                walkAlbedo = openpbr_interior_volume(openpbrMat).albedo;
            }
            else
            {
                walkAlbedo = float3(materials[entry.materialId].diffuse_transmission_color);
                const float3 reference = float3(materials[entry.materialId].subsurface_reference);
                if (reference.x > 1e-4f && reference.y > 1e-4f && reference.z > 1e-4f)
                {
                    walkAlbedo *= si.albedo / reference;
                }
            }
            mediumState.mediumAlbedo = packMediumAlbedo(saturate(walkAlbedo));
            // This is the only common surface-shading branch that changes the
            // side record. Leave every other surface bounce read-only so an
            // SSS-capable scene does not turn eight cold bytes into an
            // unconditional write on every live path.
            mediumPaths[tid] = mediumState;
            // Only the glTF path divides by an entry tint. Its
            // diffuse-transmission lobe already carries the scatter colour in
            // bsdf_over_pdf, and the walk applies that colour again through the
            // medium's albedo, so one of the two has to come back out.
            //
            // OpenPBR does not double it: Adobe's subsurface lobe weight and the
            // single-scattering albedo its volume derives are already the split.
            // Dividing here would use si.diffuse_transmission_color, a glTF
            // field no OpenPBR material fills -- clamped to 1e-4, that is not a
            // small darkening but a factor of ten thousand, which is what the
            // fireflies around the chess kings were.
            if (!isOpenPBR)
            {
                sssEntryTint = max(float3(si.diffuse_transmission_color), float3(1e-4f));
            }

            // The lobe chooses and weights entry, but the random walk starts along the refracted direction.
            sssRefractedEntry = true;
        }
    }
    else
    {
        nextOrigin = offset_ray(si.position, faceNg);
    }
    iorStacks[tid] = iorStack;

    const float3 nextDir = sssRefractedEntry ?
                               subsurface_entry_direction(
                                   float3(si.wo),
                                   (dot(float3(si.shading_normal), float3(si.wo)) > 0.0f) ? float3(si.shading_normal) :
                                                                                            -float3(si.shading_normal),
                                   random<SampleDimension::eSssChannel>(rng, uniforms.samplerType),
                                   random<SampleDimension::eSssDistance>(rng, uniforms.samplerType)) :
                               normalize(sampleResult.wi);

    // A refracted entry that came back out on the viewer's side of the geometric
    // normal never entered anything, and the walk it would start is a walk
    // through the outside of the object. Cycles gives up on the bounce for the
    // same test -- `dot(sd->Ng, wo) >= 0.0f` in subsurface_bounce() -- and the
    // draws this rejects are the grazing ones, which is where the microfacet
    // normal can tilt far enough to refract along the surface rather than into
    // it. Keeping them lit a rim on every sphere that no reference has.
    if (sssRefractedEntry && dot(faceNg, nextDir) >= 0.0f)
    {
        radianceOut[tid] += float4(radiance, 0.0f);
        return;
    }

    if (isFibre)
    {
        // Both branches above assume a surface with an inside and an outside. A
        // strand has neither: the bounce leaves from wherever the crossing the lobe
        // already accounted for comes out.
        nextOrigin = fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius, nextDir);
    }
    float3 segmentThroughput = float3(sampleResult.bsdf_over_pdf) / sssEntryTint;
    // SHaRC treats every path segment independently: the cache stores radiance
    // per vertex and carries the connecting throughput itself, through
    // sharcSetThroughput. Leaving the path throughput accumulating on top of
    // that weights each deposit twice, and -- because Russian roulette then
    // divides by the whole path's throughput rather than this segment's -- puts
    // compensation factors of thousands into cells that see a handful of samples
    // a frame. That is the difference between a cache that converges and one
    // whose fireflies feed back through cache resampling.
    float3 nextThroughput = SPEC_SHARC_UPDATE ? segmentThroughput : throughput * segmentThroughput;

    // Record exactly the support NEE offered in the same shaded frame. A raw
    // back face may be an opaque two-sided surface whose BSDF flipped its frame,
    // or a transmissive exit that did not; shadedFrame distinguishes them.
    didNee = neePairsWithBounce(didNee, neeCrossesSurface(isFibre, si.transmission, si.diffuse_transmission),
                                neeFrame.frontFace, neeFrame.normalSign * dot(si.shading_normal, nextDir));

    radianceOut[tid] += float4(radiance, 0.0f);

    bool alive = dot(nextThroughput, nextThroughput) >= 1e-4f;
    if (alive && depth > 3u)
    {
        const float q = min(max(nextThroughput.x, max(nextThroughput.y, nextThroughput.z)), 1.0f);
        if (random<SampleDimension::eRussianRoulette>(rng, uniforms.samplerType) > q)
        {
            alive = false;
        }
        else
        {
            nextThroughput *= 1.0f / max(q, 1e-5f);
            segmentThroughput *= 1.0f / max(q, 1e-5f);
        }
    }
    if (depth + 1u >= uniforms.maxDepth)
    {
        alive = false;
    }

    if (!alive)
    {
        return;
    }

    if (SPEC_SHARC_UPDATE)
    {
        const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
        SharcUpdateState updateState = sharcUpdates[updateIndex];
        sharcSetThroughput(updateState, segmentThroughput);
        sharcUpdates[updateIndex] = updateState;
    }

    PathRay nextRay;
    nextRay.origin = packed_float3(nextOrigin);
    nextRay.direction = packed_float3(nextDir);
    rays[tid] = nextRay;

    p.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : nextThroughput);
    p.lastBsdfPdf = nextSpecular ? 1.0f : sampleResult.pdf;
    p.misDistance = 0.0f;
    // AOV_DONE is carried, not rebuilt: it records something that already
    // happened to this path, unlike the others, which describe the bounce being
    // set up. Dropping it let every escaping ray overwrite guides that a surface
    // had already written, which is most of the frame in an open scene.
    const float previousSharcRoughness = unpackSharcRoughness(p.depthAndFlags);
    const float sharcRoughness = min(
        previousSharcRoughness + (((sampleResult.event_type & BSDF_EVENT_DIFFUSE) != 0) ? 1.0f : si.roughness), 1.0f);
    p.depthAndFlags = (depth + 1u) | PATH_FLAG_ALIVE | (nextSpecular ? PATH_FLAG_SPECULAR : 0u) |
                      (didNee ? PATH_FLAG_NEE_DONE : 0u) | (p.depthAndFlags & PATH_FLAG_AOV_DONE) |
                      packSharcRoughness(sharcRoughness);
    paths[tid] = p;

    auditWork(uniforms, WORK_PATH_CONTINUATIONS);
    queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
}

static void rebuildRestirTargetSurface(constant Uniforms& uniforms,
                                       float4x4 objectToWorld,
                                       device const Material* materials,
                                       device const GeometryEntry* geometryEntries,
                                       device const char* vertexBuffer,
                                       device const char* prevVertexBuffer,
                                       device const uint32_t* indexBuffer,
                                       device const packed_float3* curvePoints,
                                       device const uint32_t* curveSegments,
                                       thread const RestirTargetSurface& stored,
                                       bool interpolateMotion,
                                       float motionTime,
                                       thread SurfaceInteraction& si,
                                       thread bool& isFibre,
                                       thread bool& isOpenPBR,
                                       thread ShadedFrame& neeFrame,
                                       thread OpenPBR_PreparedBsdf& openpbrPrepared,
                                       thread float& curveRadius)
{
    if (restirTargetIsDirect(stored))
    {
        const RestirDirectTargetSurface direct = *(thread const RestirDirectTargetSurface*)&stored;
        si = {};
        si.position = float3(direct.position);
        si.shading_normal = float3(direct.shadingNormal);
        si.geometry_normal = si.shading_normal;
        build_onb(si.shading_normal, si.tangent, si.bitangent);
        si.wo = -float3(direct.rayDirection);
        si.albedo = float3(direct.albedo);
        si.roughness = direct.roughness;
        si.material_type = (direct.flags >> RESTIR_TARGET_MATERIAL_SHIFT) & RESTIR_TARGET_MATERIAL_MASK;
        si.front_face = (direct.flags & RESTIR_TARGET_FRONT_FACE) != 0u;
        si.thin_walled = (direct.flags & RESTIR_TARGET_THIN_WALLED) != 0u;
        si.transmission = (direct.flags & RESTIR_TARGET_HAS_TRANSMISSION) != 0u ? 1.0f : 0.0f;
        si.diffuse_transmission = (direct.flags & RESTIR_TARGET_HAS_DIFFUSE_TRANSMISSION) != 0u ? 1.0f : 0.0f;
        si.exterior_ior = 1.0f;
        si.bump_normal = si.shading_normal;
        si.clearcoat_ior = 1.5f;
        si.iridescence_ior = 1.3f;
        si.specular_color = float3(1.0f);
        if (si.material_type == MATERIAL_TYPE_STANDARD_PBR)
        {
            si.metallic = direct.metallicOrIor;
            si.ior = 1.5f;
            si.specular = 0.5f;
        }
        else
        {
            si.ior = direct.metallicOrIor;
        }
        isFibre = false;
        isOpenPBR = false;
        curveRadius = 0.0f;
        neeFrame = shadedFrame(si.front_face, dot(si.shading_normal, si.wo), si.transmission, si.diffuse_transmission);
        return;
    }
    const GeometryEntry entry = geometryEntries[stored.geomEntryIndex];
    const float3 position = float3(stored.position);
    const float2 bary = stored.barycentrics;
    const bool isCurve = SPEC_CURVES && (entry.flags & GEOM_FLAG_CURVE) != 0u;
    float3 normal, tangent, geometryNormal, vertexColor;
    float2 uv;
    float tangentSign = 1.0f;
    curveRadius = 0.0f;
    if (isCurve)
    {
        fetchCurve(curvePoints, curveSegments, entry, stored.primitiveId, bary.x, position, objectToWorld, normal,
                   tangent, uv, curveRadius);
        geometryNormal = normal;
        vertexColor = float3(1.0f);
    }
    else
    {
        float3 objectNormal, objectTangent, edge1, edge2;
        float uvArea2;
        fetchTriangleBlended(vertexBuffer, prevVertexBuffer, indexBuffer, entry, stored.primitiveId, interpolateMotion,
                             motionTime, bary, objectNormal, objectTangent, uv, vertexColor, tangentSign,
                             geometryNormal, edge1, edge2, uvArea2);
        normal = transformNormal(normalize(objectNormal), objectToWorld);
        tangent = orthonormalizeTangent(normal, transformDirection(normalize(objectTangent), objectToWorld));
        geometryNormal = transformNormal(geometryNormal, objectToWorld);
    }
    initSurfaceInteraction(si, materials[entry.materialId], position, normal, geometryNormal, tangent,
                           cross(normal, tangent) * tangentSign, uv, float3(stored.rayDirection), vertexColor,
                           stored.lodBase);
    si.exterior_ior = 1.0f;
    isOpenPBR = SPEC_OPENPBR && si.material_type == MATERIAL_TYPE_OPENPBR;
    if (isOpenPBR)
    {
        OpenPBRParams openpbrMat = uniforms.openpbrParams[entry.materialId];
        if (openpbrMat.texture_mask != 0u && uniforms.openpbrTextures != nullptr)
        {
            applyOpenPBRTextures(openpbrMat, uniforms.openpbrTextures[entry.materialId], si, uv);
        }
        openpbrPrepared = openpbr_prepare_at(openpbrMat, si, float3(stored.throughput));
    }
    isFibre = isCurve && scattersThroughFibre(si);
    neeFrame = shadedFrame(si.front_face, dot(si.shading_normal, si.wo), si.transmission, si.diffuse_transmission);
}

constant int2 kRestirNeighborOffsets[] = { int2(-1, 0),  int2(0, -1), int2(1, 0), int2(0, 1),
                                           int2(-1, -1), int2(1, -1), int2(1, 1), int2(-1, 1),
                                           int2(-2, 0),  int2(0, -2), int2(2, 0), int2(0, 2),
                                           int2(-2, -1), int2(1, -2), int2(2, 1), int2(-1, 2) };

kernel void wavefrontRestirSpatialFinal(uint gid [[thread_position_in_grid]],
                                        constant Uniforms& uniforms [[buffer(0)]],
                                        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances
                                        [[buffer(1)]],
                                        device const IesGpuBufferHeader* iesProfiles [[buffer(2)]],
                                        device UniformLight* lights [[buffer(3)]],
                                        device const Material* materials [[buffer(4)]],
                                        device const EnvAliasEntry* envAliasTable [[buffer(5)]],
                                        device const char* vertexBuffer [[buffer(6)]],
                                        device const char* prevVertexBuffer [[buffer(7)]],
                                        device const uint32_t* indexBuffer [[buffer(8)]],
                                        device const uint32_t* hitQueue [[buffer(9)]],
                                        device ShadowRay* shadowRays [[buffer(10)]],
                                        device atomic_uint* shadowCounter [[buffer(11)]],
                                        device float4* radianceOut [[buffer(12)]],
                                        device const RestirReservoir* initialReservoirs [[buffer(13)]],
                                        device const uint32_t* control [[buffer(14)]],
                                        device const GeometryEntry* geometryEntries [[buffer(15)]],
                                        device const packed_float3* curvePoints [[buffer(16)]],
                                        device const uint32_t* curveSegments [[buffer(17)]],
                                        CurveStaticTraversal::structure diagnosticAccelerationStructure [[buffer(18)]],
                                        CurveStaticTraversal::table diagnosticFunctionTable [[buffer(19)]],
                                        texture2d<float> envMapTexture [[texture(0)]])
{
    if (gid >= control[WF_CTRL_RESTIR_N])
    {
        return;
    }
    const uint32_t tid = hitQueue[gid];
    const bool oddFrame = (uniforms.frameIndex & 1u) != 0u;
    device const char* surfaceData = uniforms.restirBiasCorrection != 0u ?
                                         (oddFrame ? uniforms.restirSurfaceData1 : uniforms.restirSurfaceData0) :
                                         uniforms.restirSurfaceData0;
    device RestirReservoir* currentReservoirs = oddFrame ? uniforms.restirReservoir1 : uniforms.restirReservoir0;
    device const RestirSurfaceHistory* history = oddFrame ? uniforms.restirHistory1 : uniforms.restirHistory0;
    const RestirSurfaceHistory currentSurface = history[tid];
    if (uniforms.restirDIEnabled == 0u || (currentSurface.materialIdAndFlags & RESTIR_SURFACE_VALID) == 0u)
    {
        return;
    }
    auditWork(uniforms, WORK_RESTIR_FINAL_ITEMS);

    device RestirDiagnosticRecord* diagnosticRecord = restirDiagnosticRecord(uniforms, tid);
    if (diagnosticRecord != nullptr)
    {
        RestirDiagnosticRecord empty = {};
        empty.pixelIndex = tid;
        *diagnosticRecord = empty;
    }

    RestirReservoir reservoir = initialReservoirs[tid];
    const uint32_t centerM = reservoir.state.M;
    uint32_t selectedAge = reservoir.state.ageAndFlags & RESTIR_RESERVOIR_AGE_MASK;
    uint32_t selectedSourceIndex = tid;

    SurfaceInteraction si;
    bool isFibre, isOpenPBR;
    ShadedFrame neeFrame;
    OpenPBR_PreparedBsdf openpbrPrepared;
    float3 storedThroughput;
    float curveRadius;
    uint32_t medium;
    uint32_t sampleIdx;
    const RestirTargetSurface stored = ((device const RestirTargetSurface*)surfaceData)[tid];
    sampleIdx = stored.sampleIdxAndMedium & RESTIR_TARGET_SAMPLE_MASK;
    medium = stored.sampleIdxAndMedium >> RESTIR_TARGET_MEDIUM_SHIFT;
    storedThroughput = restirTargetIsDirect(stored) ? float3(1.0f) : float3(stored.throughput);
    const float storedMotionTime = motionTimeFor(uniforms, tid, sampleIdx);
    const bool interpolateMotion =
        SPEC_MOTION_BLUR && uniforms.enableMotionBlur && storedMotionTime < 1.0f && prevVertexBuffer && indexBuffer;
    rebuildRestirTargetSurface(uniforms, restirTargetObjectToWorld(instances, stored), materials, geometryEntries,
                               vertexBuffer, prevVertexBuffer, indexBuffer, curvePoints, curveSegments, stored,
                               interpolateMotion, storedMotionTime, si, isFibre, isOpenPBR, neeFrame, openpbrPrepared,
                               curveRadius);

    const uint2 pixel = uint2(tid % uniforms.width, tid / uniforms.width);
    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);
    SamplerState rng = samplerFor(uniforms, tid, sampleIdx, 0u);
    const uint32_t rotation = hash_combine(tid, uniforms.frameIndex * 0x9e3779b9u) & 15u;
    const uint32_t neighborCount = uniforms.spatialReuseEnabled != 0u ? min(uniforms.spatialNeighborCount, 16u) : 0u;
    if (neighborCount != 0u)
    {
        auditWork(uniforms, WORK_RESTIR_SPATIAL_ITEMS);
    }
    for (uint32_t i = 0u; i < neighborCount; ++i)
    {
        const int2 neighborPixel = int2(pixel) + kRestirNeighborOffsets[(rotation + i * 5u) & 15u];
        if (any(neighborPixel < int2(0)) || neighborPixel.x >= int(uniforms.width) ||
            neighborPixel.y >= int(uniforms.height))
        {
            continue;
        }
        const uint32_t neighborIndex = uint32_t(neighborPixel.y) * uniforms.width + uint32_t(neighborPixel.x);
        const RestirReservoir neighbor = initialReservoirs[neighborIndex];
        const RestirSurfaceHistory neighborSurface = history[neighborIndex];
        if (neighbor.state.M == 0u || !restirSurfaceHistoryCompatible(currentSurface, neighborSurface))
        {
            continue;
        }

        auditWork(uniforms, WORK_RESTIR_SPATIAL_MERGES);

        RestirEvaluation evaluated = {};
        if ((neighbor.state.ageAndFlags & RESTIR_RESERVOIR_VALID) != 0u)
        {
            const LightConnection connection = reconnectRestirSample(
                uniforms, lights, instances, materials, vertexBuffer, prevVertexBuffer, indexBuffer, motionTime, si,
                envAliasTable, envMapTexture, iesProfiles, neighbor.sample);
            evaluated = evaluateRestirConnection(
                connection, si, isFibre, neeFrame, isOpenPBR, openpbrPrepared, uniforms.misHeuristic);
        }
        SamplerState neighborRng = rng;
        neighborRng.seed = restirRngStreamSeed(rng.seed, RESTIR_RNG_SPATIAL_SALT + i * 0x9e3779b9u);
        const bool selectedSampleHasCurrentVisibility =
            (reservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBLE) != 0u &&
            restirSamplesEqual(reservoir.sample, neighbor.sample);
        const uint32_t currentVisibilityFlags = reservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK;
        const RestirVisibilityCache currentVisibility = reservoir.visibility;
        if (restirReservoirUpdate(reservoir.state, restirReservoirMergeWeight(neighbor.state, evaluated.target),
                                  evaluated.target, neighbor.state.M,
                                  random<SampleDimension::eLightId>(neighborRng, uniforms.samplerType)))
        {
            reservoir.sample = neighbor.sample;
            if (selectedSampleHasCurrentVisibility)
            {
                reservoir.state.ageAndFlags |= currentVisibilityFlags;
                reservoir.visibility = currentVisibility;
            }
            else if ((neighbor.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBLE) != 0u)
            {
                reservoir.state.ageAndFlags |= neighbor.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK;
                reservoir.visibility = neighbor.visibility;
            }
            selectedAge = neighbor.state.ageAndFlags & RESTIR_RESERVOIR_AGE_MASK;
            selectedSourceIndex = neighborIndex;
        }
    }
    reservoir.state.ageAndFlags =
        (reservoir.state.ageAndFlags & RESTIR_RESERVOIR_FLAGS_MASK) | min(selectedAge, RESTIR_RESERVOIR_AGE_MASK);
    if ((reservoir.state.ageAndFlags & RESTIR_RESERVOIR_VALID) == 0u)
    {
        const uint32_t maxM = max(uniforms.initialCandidateCount, 1u) * max(uniforms.reservoirMaxAge, 1u) *
                              (min(uniforms.spatialNeighborCount, 16u) + 1u);
        restirReservoirLimitM(reservoir.state, maxM);
        auditWork(uniforms, WORK_RESTIR_EFFECTIVE_M, reservoir.state.M);
        auditWork(uniforms, WORK_RESTIR_M_HISTOGRAM_BASE + restirMHistogramBin(reservoir.state.M));
        currentReservoirs[tid] = reservoir;
        return;
    }

    const bool selectedSpatial = selectedSourceIndex != tid;
    const bool selectedHistory = selectedAge != 0u;
    auditWork(uniforms, selectedSpatial ? WORK_RESTIR_FINAL_SOURCE_SPATIAL :
                        selectedHistory ? WORK_RESTIR_FINAL_SOURCE_TEMPORAL :
                                          WORK_RESTIR_FINAL_SOURCE_INITIAL);
    if (selectedHistory)
        auditWork(uniforms, WORK_RESTIR_FINAL_HISTORY);
    auditWork(uniforms, WORK_RESTIR_AGE_HISTOGRAM_BASE + min(selectedAge, RESTIR_AUDIT_AGE_BINS - 1u));
    if (restirSampleType(reservoir.sample) == RESTIR_SAMPLE_ANALYTIC)
        auditRestirLightId(uniforms, restirSampleLightId(reservoir.sample));

    const LightConnection connection =
        reconnectRestirSample(uniforms, lights, instances, materials, vertexBuffer, prevVertexBuffer, indexBuffer,
                              motionTime, si, envAliasTable, envMapTexture, iesProfiles, reservoir.sample);
    const RestirEvaluation evaluated =
        evaluateRestirConnection(connection, si, isFibre, neeFrame, isOpenPBR, openpbrPrepared, uniforms.misHeuristic);
    reservoir.state.target = evaluated.target;
    if (diagnosticRecord != nullptr)
    {
        diagnosticRecord->stableLightId = restirSampleLightId(reservoir.sample);
        diagnosticRecord->selectedSourceIndex = selectedSourceIndex;
        diagnosticRecord->currentTarget = evaluated.target;
        diagnosticRecord->weightSum = reservoir.state.weightSum;
    }
    if (uniforms.restirBiasCorrection != 0u)
    {
        float selectedSourceTarget = selectedSourceIndex == tid ? evaluated.target : 0.0f;
        float sourceTargetSum = float(centerM) * evaluated.target;
        if (diagnosticRecord != nullptr)
        {
            diagnosticRecord->sourceCount = 1u;
            diagnosticRecord->sourceIndices[0] = tid;
            diagnosticRecord->sourceM[0] = centerM;
            diagnosticRecord->sourceTargets[0] = evaluated.target;
        }
        for (uint32_t i = 0u; i < neighborCount; ++i)
        {
            const int2 neighborPixel = int2(pixel) + kRestirNeighborOffsets[(rotation + i * 5u) & 15u];
            if (any(neighborPixel < int2(0)) || neighborPixel.x >= int(uniforms.width) ||
                neighborPixel.y >= int(uniforms.height))
            {
                continue;
            }
            const uint32_t neighborIndex = uint32_t(neighborPixel.y) * uniforms.width + uint32_t(neighborPixel.x);
            const RestirReservoir neighbor = initialReservoirs[neighborIndex];
            const RestirSurfaceHistory neighborSurface = history[neighborIndex];
            if (neighbor.state.M == 0u || !restirSurfaceHistoryCompatible(currentSurface, neighborSurface))
            {
                continue;
            }
            if (selectedSourceIndex == neighborIndex &&
                !(SPEC_RESTIR_RAY_TRACED_DIAGNOSTIC && uniforms.restirBiasCorrection == 2u))
            {
                selectedSourceTarget = neighbor.state.target;
                sourceTargetSum += float(neighbor.state.M) * selectedSourceTarget;
                if (diagnosticRecord != nullptr && diagnosticRecord->sourceCount < RESTIR_DIAGNOSTIC_SOURCE_COUNT)
                {
                    const uint32_t source = diagnosticRecord->sourceCount++;
                    diagnosticRecord->sourceIndices[source] = neighborIndex;
                    diagnosticRecord->sourceM[source] = neighbor.state.M;
                    diagnosticRecord->sourceTargets[source] = selectedSourceTarget;
                }
                continue;
            }
            const RestirTargetSurface neighborStored = ((device const RestirTargetSurface*)surfaceData)[neighborIndex];
            SurfaceInteraction neighborSi;
            bool neighborIsFibre, neighborIsOpenPBR;
            ShadedFrame neighborNeeFrame;
            OpenPBR_PreparedBsdf neighborOpenpbrPrepared;
            float neighborCurveRadius;
            const uint32_t neighborSampleIdx = neighborStored.sampleIdxAndMedium & RESTIR_TARGET_SAMPLE_MASK;
            const float neighborMotionTime = motionTimeFor(uniforms, neighborIndex, neighborSampleIdx);
            const bool neighborInterpolateMotion = SPEC_MOTION_BLUR && uniforms.enableMotionBlur &&
                                                   neighborMotionTime < 1.0f && prevVertexBuffer && indexBuffer;
            rebuildRestirTargetSurface(uniforms, restirTargetObjectToWorld(instances, neighborStored), materials,
                                       geometryEntries, vertexBuffer, prevVertexBuffer, indexBuffer, curvePoints,
                                       curveSegments, neighborStored, neighborInterpolateMotion, neighborMotionTime,
                                       neighborSi, neighborIsFibre, neighborIsOpenPBR, neighborNeeFrame,
                                       neighborOpenpbrPrepared, neighborCurveRadius);
            const LightConnection selectedAtNeighbor = reconnectRestirSample(
                uniforms, lights, instances, materials, vertexBuffer, prevVertexBuffer, indexBuffer, neighborMotionTime,
                neighborSi, envAliasTable, envMapTexture, iesProfiles, reservoir.sample);
            float neighborTarget = restirTargetOnly(selectedAtNeighbor, neighborSi, neighborIsFibre, neighborNeeFrame,
                                                    neighborIsOpenPBR, neighborOpenpbrPrepared, uniforms.misHeuristic);
            if (SPEC_RESTIR_RAY_TRACED_DIAGNOSTIC && uniforms.restirBiasCorrection == 2u && neighborTarget > 0.0f)
            {
                auditWork(uniforms, WORK_RESTIR_DIAGNOSTIC_QUERIES);
                float ignoredTransmittance;
                if (!restirDiagnosticVisible(uniforms, diagnosticAccelerationStructure, diagnosticFunctionTable,
                                             selectedAtNeighbor, selectedAtNeighbor.origin, instances, materials,
                                             geometryEntries, vertexBuffer, indexBuffer, ignoredTransmittance))
                {
                    neighborTarget = 0.0f;
                }
            }
            if (selectedSourceIndex == neighborIndex)
            {
                selectedSourceTarget = neighborTarget;
            }
            sourceTargetSum += float(neighbor.state.M) * neighborTarget;
            if (diagnosticRecord != nullptr && diagnosticRecord->sourceCount < RESTIR_DIAGNOSTIC_SOURCE_COUNT)
            {
                const uint32_t source = diagnosticRecord->sourceCount++;
                diagnosticRecord->sourceIndices[source] = neighborIndex;
                diagnosticRecord->sourceM[source] = neighbor.state.M;
                diagnosticRecord->sourceTargets[source] = neighborTarget;
            }
        }
        if (diagnosticRecord != nullptr)
        {
            diagnosticRecord->basicDenominator = sourceTargetSum;
        }
        restirReservoirApplyBasicNormalization(reservoir.state, selectedSourceTarget, sourceTargetSum);
    }
    const uint32_t maxM = max(uniforms.initialCandidateCount, 1u) * max(uniforms.reservoirMaxAge, 1u) *
                          (min(uniforms.spatialNeighborCount, 16u) + 1u);
    restirReservoirLimitM(reservoir.state, maxM);
    auditWork(uniforms, WORK_RESTIR_EFFECTIVE_M, reservoir.state.M);
    auditWork(uniforms, WORK_RESTIR_M_HISTOGRAM_BASE + restirMHistogramBin(reservoir.state.M));
    currentReservoirs[tid] = reservoir;
    if (diagnosticRecord != nullptr)
    {
        diagnosticRecord->normalization = restirReservoirNormalization(reservoir.state);
    }
    if (uniforms.restirDebugMode != 0u)
    {
        const float3 debugColor =
            uniforms.restirDebugMode == 1u ?
                float3(float(reservoir.state.ageAndFlags & RESTIR_RESERVOIR_AGE_MASK) /
                       float(max(uniforms.reservoirMaxAge, 1u))) :
                (restirSampleType(reservoir.sample) == RESTIR_SAMPLE_ANALYTIC    ? float3(1.0f, 0.2f, 0.1f) :
                 restirSampleType(reservoir.sample) == RESTIR_SAMPLE_ENVIRONMENT ? float3(0.1f, 0.4f, 1.0f) :
                                                                                   float3(0.1f, 1.0f, 0.2f));
        radianceOut[tid] = float4(debugColor, 0.0f);
        return;
    }
    const float W = restirReservoirNormalization(reservoir.state);
    if (!(W > 0.0f))
    {
        return;
    }

    const float3 shadowOrigin =
        isFibre ? fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius, connection.toLight) :
                  connection.origin;
    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(connection, shadowOrigin);
    const float3 weight =
        clampIndirectContribution(storedThroughput * evaluated.integrand * W, 0u, uniforms.clampIndirect);
    if (!any(weight != 0.0f) || !visibility.valid)
    {
        return;
    }
    const uint32_t receiverMaterial = currentSurface.materialIdAndFlags & RESTIR_SURFACE_MATERIAL_MASK;
    const uint32_t receiverGeometry = restirVisibilityGeometryKey(stored);
    uint32_t receiverKey = 0u;
    uint32_t lightKey = 0u;
    bool exactInitialVisibility = false;
    bool conservativeVisibility = false;
    const bool visibilityCacheAllowed = !SPEC_SSS || !uniforms.hasBoundedMedium;
    if (visibilityCacheAllowed && (uniforms.restirInitialVisibility != 0u || uniforms.restirFinalVisibilityReuse != 0u))
    {
        receiverKey = restirVisibilityReceiverKey(uniforms, si, receiverMaterial, receiverGeometry);
        lightKey = restirVisibilityLightKey(uniforms, connection);
        exactInitialVisibility =
            uniforms.restirInitialVisibility != 0u && restirVisibilityAge(reservoir.visibility) == 0u &&
            restirVisibilityCacheMatches(uniforms, reservoir, si, receiverMaterial, receiverGeometry, lightKey, false);
        conservativeVisibility =
            uniforms.restirFinalVisibilityReuse != 0u && !exactInitialVisibility &&
            restirVisibilityCacheMatches(uniforms, reservoir, si, receiverMaterial, receiverGeometry, lightKey, true);
    }
    const bool reuseVisibility = exactInitialVisibility || conservativeVisibility;
    if (reuseVisibility)
    {
        float3 visibleWeight = weight * restirReservoirInitialVisibility(reservoir.state);
        if (conservativeVisibility)
        {
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_HITS);
            if (SPEC_RENDER_WORK_AUDIT && restirReservoirInitialVisibility(reservoir.state) > 0.0f)
            {
                auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_ORACLE_QUERIES);
                float oracleTransmittance;
                const bool oracleVisible = restirDiagnosticVisible(
                    uniforms, diagnosticAccelerationStructure, diagnosticFunctionTable, connection, shadowOrigin,
                    instances, materials, geometryEntries, vertexBuffer, indexBuffer, oracleTransmittance);
                auditWork(uniforms, oracleVisible ? WORK_RESTIR_VISIBILITY_CACHE_VISIBLE_VISIBLE :
                                                    WORK_RESTIR_VISIBILITY_CACHE_VISIBLE_OCCLUDED);
            }
        }
        if (SPEC_FOG && uniforms.hasFog)
        {
            const float tau = fogOpticalDepth(
                shadowOrigin, visibility.direction, visibility.maxDistance, uniforms.fogHeight, uniforms.fogSigmaT);
            visibleWeight *= exp(-tau);
        }
        if (all(visibleWeight <= 1e-6f))
        {
            return;
        }
        radianceOut[tid] += float4(visibleWeight, 0.0f);
        restirDiagnosticVisibility(uniforms, tid, visibleWeight);
        if (restirVisibilityAge(reservoir.visibility) == 0u)
            auditWork(uniforms, WORK_RESTIR_INITIAL_VISIBILITY_REUSED);
        if (selectedHistory)
        {
            auditWork(uniforms, WORK_RESTIR_FINAL_HISTORY_VISIBLE);
        }
        return;
    }
    ShadowRay sr;
    sr.origin = packed_float3(shadowOrigin);
    sr.direction = packed_float3(visibility.direction);
    sr.weight = packed_float3(weight);
    sr.maxDistance = visibility.maxDistance;
    sr.pixelIndex = tid;
    sr.medium = medium;
    sr.sharcRadiance = packed_float3(float3(0.0f));
    const bool updateVisibilityCache = visibilityCacheAllowed && uniforms.restirFinalVisibilityReuse != 0u;
    if (updateVisibilityCache)
    {
        reservoir.state.ageAndFlags &= ~RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK;
        restirVisibilityStore(reservoir.visibility, receiverKey, lightKey, 0u);
        currentReservoirs[tid] = reservoir;
    }
    sr.sharcPathIndex = tid | (SPEC_RENDER_WORK_AUDIT && selectedHistory ? RESTIR_AUDIT_HISTORY_BIT : 0u) |
                        (updateVisibilityCache ? RESTIR_VISIBILITY_UPDATE_BIT : 0u);
    sr.rrCutoff = random<SampleDimension::eShadowRR>(rng, uniforms.samplerType) * kShadowTransmittanceCutoff;
    const uint32_t slot = atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
    shadowRays[slot] = sr;
    auditWork(uniforms, WORK_RESTIR_FINAL_VISIBILITY_RAYS);
    if (selectedHistory)
        auditWork(uniforms, WORK_RESTIR_FINAL_HISTORY_RAYS);
}

// ---------------------------------------------------------------------------
// guide -- compact deterministic continuation for MetalFX auxiliary buffers
// ---------------------------------------------------------------------------
static inline void storeGuideMaterial(
    device AovSample* aov, uint32_t tid, thread const DenoiserMaterialGuides& guides, float3 normal, float hitDistance)
{
    AovSample out = aov[tid];
    const float state = out.guideStateOrBounceDepth;
    const bool blendTransmission = state >= 1.0f;
    const float interfaceFresnel = blendTransmission ? saturate(state - 1.0f) : 0.0f;
    const float transmissionWeight = 1.0f - interfaceFresnel;
    const float3 guideNormal = normalize(normal);
    const float3 blendedNormal = float3(out.normal) * interfaceFresnel + guideNormal * transmissionWeight;
    out.diffuseAlbedo = packed_float3(guides.diffuse * transmissionWeight);
    out.specularAlbedo = packed_float3(float3(interfaceFresnel) + guides.specular * transmissionWeight);
    out.normal = packed_float3(
        blendTransmission && dot(blendedNormal, blendedNormal) > 1e-8f ? normalize(blendedNormal) : guideNormal);
    out.roughness = blendTransmission ? mix(out.roughness, guides.roughness, transmissionWeight) : guides.roughness;
    out.specularHitDistance = hitDistance;
    out.guideStateOrBounceDepth = 0.0f;
    aov[tid] = out;
}

static inline void storeGuideMiss(device AovSample* aov, uint32_t tid, float3 rayDirection, float hitDistance)
{
    DenoiserMaterialGuides sky;
    sky.diffuse = float3(0.0f);
    sky.specular = float3(0.0f);
    sky.roughness = 1.0f;
    sky.transmission = 0.0f;
    storeGuideMaterial(aov, tid, sky, -rayDirection, hitDistance);
}

template <typename T>
static void guideImpl(uint gid,
                      constant Uniforms& uniforms,
                      constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                      typename T::structure accelerationStructure,
                      device AovSample* aov,
                      device const Material* materials,
                      device const GeometryEntry* geometryEntries,
                      device const char* vertexBuffer,
                      device const char* prevVertexBuffer,
                      device const uint32_t* indexBuffer,
                      device const packed_float3* curvePoints,
                      device const uint32_t* curveSegments,
                      typename T::table functionTable,
                      device const uint32_t* control)
{
    if (gid >= control[WF_CTRL_GUIDE_N])
    {
        return;
    }
    if (gid == 0u)
    {
        auditWork(uniforms, WORK_GUIDE_DISPATCHES);
    }
    gid = uniforms.guideQueue[gid];

    GuideRay guide = uniforms.guideRays[gid];
    if ((guide.flags & GUIDE_RAY_ACTIVE) == 0u)
    {
        return;
    }
    auditWork(uniforms, WORK_GUIDE_ACTIVE_ITEMS);

    const bool replaceMaterial = (guide.flags & GUIDE_RAY_REPLACE_MATERIAL) != 0u;
    const float motionTime = motionTimeFor(uniforms, gid, 0u);
    float totalDistance = 0.0f;
    uint32_t describedSurfaces = 0u;

    // Six traversal attempts allow deterministic pass-through of a few alpha
    // layers while material continuation itself remains capped at two hits.
    for (uint32_t attempt = 0u; attempt < 6u; ++attempt)
    {
        const float3 origin = float3(guide.origin);
        const float3 direction = normalize(float3(guide.direction));
        if (!all(isfinite(origin)) || !all(isfinite(direction)))
        {
            return;
        }

        ray r;
        r.origin = origin;
        r.direction = direction;
        r.min_distance = 1e-6f;
        r.max_distance = INFINITY;

        typename T::isect isect;
        isect.assume_geometry_type(T::geometryTypes());
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(false);
        auditWork(uniforms, WORK_GUIDE_ONLY_RAYS);
        auditWork(uniforms, WORK_INTERSECTION_QUERIES);
        const typename T::isect::result_type rawHit =
            T::trace(isect, r, accelerationStructure, uniforms.primaryRayMask | GEOMETRY_MASK_LIGHT_HIDDEN, motionTime,
                     functionTable);
        const float curveParameter = rawHit.type == intersection_type::curve ? T::curveParameter(rawHit) : 0.0f;
        const ExtendIntersection hit = captureExtendIntersection(rawHit, curveParameter);
        if (hit.type == intersection_type::none)
        {
            const float missDistance = totalDistance + max(uniforms.sceneExtent, 1e3f);
            if (replaceMaterial)
            {
                storeGuideMiss(aov, gid, direction, missDistance);
            }
            else
            {
                aov[gid].specularHitDistance = missDistance;
            }
            return;
        }

        totalDistance += hit.distance;
        const auto inst = instances[hit.instanceId];
        if (inst.mask == GEOMETRY_MASK_LIGHT || inst.mask == GEOMETRY_MASK_LIGHT_HIDDEN)
        {
            if (replaceMaterial)
            {
                storeGuideMiss(aov, gid, direction, totalDistance);
            }
            else
            {
                aov[gid].specularHitDistance = totalDistance;
            }
            return;
        }

        const GeometryEntry entry = geometryEntries[inst.userID + hit.geometryId];
        // The common glossy case needs only the first opaque hit distance. It
        // avoids all vertex/material texture traffic here.
        if (!replaceMaterial && materials[entry.materialId].alpha_mode == ALPHA_MODE_OPAQUE)
        {
            aov[gid].specularHitDistance = totalDistance;
            return;
        }

        const float4x4 objectToWorld = float4x4(
            float4(float3(inst.transformationMatrix[0]), 0.0f), float4(float3(inst.transformationMatrix[1]), 0.0f),
            float4(float3(inst.transformationMatrix[2]), 0.0f), float4(float3(inst.transformationMatrix[3]), 1.0f));
        const float3 worldPosition = origin + direction * hit.distance;
        const bool isCurve = SPEC_CURVES && (entry.flags & GEOM_FLAG_CURVE) != 0u;

        float3 objectNormal, objectTangent, vertexColor, objectGeomNormal;
        float3 shadingNormal, shadingTangent, shadingGeomNormal;
        float2 uv;
        float tangentSign = 1.0f;
        if (isCurve)
        {
            float curveRadius = 0.0f;
            fetchCurve(curvePoints, curveSegments, entry, hit.primitiveId, hit.curveParameter, worldPosition,
                       objectToWorld, shadingNormal, shadingTangent, uv, curveRadius);
            shadingGeomNormal = shadingNormal;
            vertexColor = float3(1.0f);
        }
        else
        {
            float3 edge1, edge2;
            float uvArea2 = 0.0f;
            const bool interpolateMotion =
                SPEC_MOTION_BLUR && uniforms.enableMotionBlur && motionTime < 1.0f && prevVertexBuffer && indexBuffer;
            fetchTriangleBlended(vertexBuffer, prevVertexBuffer, indexBuffer, entry, hit.primitiveId, interpolateMotion,
                                 motionTime, hit.barycentrics, objectNormal, objectTangent, uv, vertexColor,
                                 tangentSign, objectGeomNormal, edge1, edge2, uvArea2);
            shadingNormal = transformNormal(normalize(objectNormal), objectToWorld);
            shadingTangent =
                orthonormalizeTangent(shadingNormal, transformDirection(normalize(objectTangent), objectToWorld));
            shadingGeomNormal = transformNormal(objectGeomNormal, objectToWorld);
        }

        const float3 binormal = cross(shadingNormal, shadingTangent) * tangentSign;
        SurfaceInteraction si;
        initSurfaceInteraction(si, materials[entry.materialId], worldPosition, shadingNormal, shadingGeomNormal,
                               shadingTangent, binormal, uv, direction, vertexColor, -1e30f);

        // Guides must be stable, so BLEND coverage uses a fixed threshold rather
        // than consuming the radiance path's random opacity decision.
        if (si.opacity < 0.5f)
        {
            const float3 faceNg = dot(shadingGeomNormal, direction) > 0.0f ? shadingGeomNormal : -shadingGeomNormal;
            guide.origin = packed_float3(offset_ray(worldPosition, faceNg));
            continue;
        }
        if (!replaceMaterial)
        {
            aov[gid].specularHitDistance = totalDistance;
            return;
        }

        const bool isOpenPBR = SPEC_OPENPBR && si.material_type == MATERIAL_TYPE_OPENPBR;
        OpenPBRParams openpbrMat;
        if (isOpenPBR)
        {
            openpbrMat = uniforms.openpbrParams[entry.materialId];
            if (openpbrMat.texture_mask != 0u && uniforms.openpbrTextures != nullptr)
            {
                applyOpenPBRTextures(openpbrMat, uniforms.openpbrTextures[entry.materialId], si, uv);
            }
        }

        const bool entering = si.front_face;
        const float2 mediaIors = unpackGuideIors(guide.mediaIors);
        const float currentIor = max(mediaIors.x, 1.0f);
        const float exteriorIor = max(mediaIors.y, 1.0f);
        si.exterior_ior = entering ? currentIor : exteriorIor;
        DenoiserMaterialGuides materialGuides;
        if (isOpenPBR)
        {
            const OpenPBR_ResolvedInputs openpbrInputs = openpbr_resolve_inputs(openpbrMat, si);
            materialGuides = openpbrDenoiserGuides(openpbrInputs, si, openpbrBaseMapDetailsSubsurface(openpbrMat));
        }
        else
        {
            materialGuides = standardDenoiserGuides(si);
        }

        constexpr float kGuideRoughnessFloor = 0.05f;
        const bool opaque = materialGuides.transmission <= kGuideRoughnessFloor;
        const bool hasDiffuse = luminance(materialGuides.diffuse) > 1e-3f;
        ++describedSurfaces;
        if ((opaque && (hasDiffuse || materialGuides.roughness > kGuideRoughnessFloor)) || describedSurfaces >= 2u)
        {
            storeGuideMaterial(aov, gid, materialGuides, float3(si.shading_normal), totalDistance);
            return;
        }

        const float3 V = normalize(float3(si.wo));
        const float3 Ns = normalize(float3(si.shading_normal));
        const float3 Nf = dot(Ns, V) >= 0.0f ? Ns : -Ns;
        float3 nextDirection = reflect_dir(-V, Nf);
        bool transmitted = false;
        float nextCurrentIor = currentIor;
        float nextExteriorIor = exteriorIor;
        if (!opaque)
        {
            const float materialIor = denoiserInterfaceIor(isOpenPBR, openpbrMat, si, entering);
            const float outsideIor = entering ? currentIor : exteriorIor;
            const float eta = entering ? outsideIor / materialIor : materialIor / outsideIor;
            const float interfaceCosine = abs(dot(Nf, V));
            const float fresnel = fresnel_dielectric(interfaceCosine, eta);
            float3 refracted;
            const bool validRefraction = si.thin_walled ? true : refract_dir(-V, Nf, eta, interfaceCosine, refracted);
            if (validRefraction && fresnel < 0.5f)
            {
                nextDirection = si.thin_walled ? -V : refracted;
                transmitted = true;
                if (!si.thin_walled)
                {
                    nextCurrentIor = entering ? materialIor : outsideIor;
                    nextExteriorIor = entering ? outsideIor : 1.0f;
                }
            }
        }

        const float3 faceNg =
            dot(float3(si.geometry_normal), V) > 0.0f ? float3(si.geometry_normal) : -float3(si.geometry_normal);
        guide.origin = packed_float3(offset_ray(si.position, transmitted ? -faceNg : faceNg));
        guide.direction = packed_float3(normalize(nextDirection));
        guide.mediaIors = packGuideIors(nextCurrentIor, nextExteriorIor);
    }

    // A pathological alpha stack is still a valid reflected depth. Using the
    // scene bound is safer for temporal reprojection than leaving zero.
    const float fallbackDistance = totalDistance + max(uniforms.sceneExtent, 1e3f);
    if (replaceMaterial)
    {
        storeGuideMiss(aov, gid, float3(guide.direction), fallbackDistance);
    }
    else
    {
        aov[gid].specularHitDistance = fallbackDistance;
    }
}

#define WF_GUIDE_ENTRY(NAME, TRAITS)                                                                                   \
    kernel void NAME(                                                                                                  \
        uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]],                               \
        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(1)]],                          \
        TRAITS::structure accelerationStructure [[buffer(2)]], device AovSample* aov [[buffer(4)]],                    \
        device const Material* materials [[buffer(5)]], device const GeometryEntry* geometryEntries [[buffer(6)]],     \
        device const char* vertexBuffer [[buffer(7)]], device const char* prevVertexBuffer [[buffer(8)]],              \
        device const uint32_t* indexBuffer [[buffer(9)]], device const packed_float3* curvePoints [[buffer(10)]],      \
        device const uint32_t* curveSegments [[buffer(11)]], TRAITS::table functionTable [[buffer(13)]],               \
        device const uint32_t* control [[buffer(14)]])                                                                 \
    {                                                                                                                  \
        guideImpl<TRAITS>(gid, uniforms, instances, accelerationStructure, aov, materials, geometryEntries,            \
                          vertexBuffer, prevVertexBuffer, indexBuffer, curvePoints, curveSegments, functionTable,      \
                          control);                                                                                    \
    }

WF_GUIDE_ENTRY(wavefrontGuide, MotionTraversal)
WF_GUIDE_ENTRY(wavefrontGuideStatic, StaticTraversal)
WF_GUIDE_ENTRY(wavefrontGuideCurve, CurveMotionTraversal)
WF_GUIDE_ENTRY(wavefrontGuideStaticCurve, CurveStaticTraversal)

// ---------------------------------------------------------------------------
// prepare -- turn the live path count into an indirect dispatch
//
// The count only exists on the GPU. Reading it back to size the next dispatch on
// the CPU would put a round trip in the middle of every bounce, which costs far
// more than the empty threadgroups an indirect dispatch occasionally launches.
// ---------------------------------------------------------------------------
// A timestamp query is surprisingly invasive on Metal 4: precise timestamps
// may split a compute pass internally, and hundreds of them made a healthy long
// frame trip the GPU watchdog. This one-word breadcrumb identifies the stage
// that was entered without using the counter-sampling path at all.
kernel void wavefrontStageBreadcrumb(device atomic_uint* stage [[buffer(0)]], constant uint32_t& stageIndex [[buffer(1)]])
{
    atomic_store_explicit(stage, stageIndex, memory_order_relaxed);
}

static inline void prepareTraversalDispatches(
    device uint32_t* args, uint32_t active, uint32_t threadsPerGroup, uint32_t batchThreads, uint32_t batchCount)
{
    for (uint32_t batch = 0u; batch < batchCount; ++batch)
    {
        const uint32_t offset = batch * batchThreads;
        const uint32_t n = active > offset ? min(active - offset, batchThreads) : 0u;
        args[batch * 3u + 0u] = (n + threadsPerGroup - 1u) / threadsPerGroup;
        args[batch * 3u + 1u] = 1u;
        args[batch * 3u + 2u] = 1u;
    }
}

kernel void wavefrontPrepare(device uint32_t& controlRef [[buffer(0)]],
                             constant uint32_t& srcIdx [[buffer(1)]],
                             constant uint32_t& threadsPerGroup [[buffer(2)]],
                             constant uint32_t& bounceIdx [[buffer(3)]],
                             device uint32_t* stageStats [[buffer(4)]],
                             device const PathState* paths [[buffer(5)]],
                             device const PathRay* rays [[buffer(6)]],
                             device const uint32_t* queue [[buffer(7)]],
                             constant uint32_t& diagnosticsEnabled [[buffer(8)]],
                             device uint32_t* traversalDispatches [[buffer(9)]],
                             constant uint32_t& traversalBatchThreads [[buffer(10)]],
                             constant uint32_t& traversalBatchCount [[buffer(11)]],
                             device const MediumPathState* mediumPaths [[buffer(12)]],
                             constant uint32_t& diagnosticsMediumEnabled [[buffer(13)]])
{
    device uint32_t* control = &controlRef;
    const uint32_t n = min(control[srcIdx], control[WF_CTRL_CAPACITY]);
    if (diagnosticsEnabled != 0u)
    {
        const uint32_t diagBase = WF_DIAG_BASE + min(bounceIdx, WF_DIAG_BOUNCES - 1u) * WF_DIAG_STRIDE;
        stageStats[diagBase] = n;
        for (uint32_t lane = 0u; lane < min(n, (uint32_t)WF_DIAG_LANES); ++lane)
        {
            const uint32_t laneBase = diagBase + 1u + lane * WF_DIAG_LANE_WORDS;
            const uint32_t tid = queue[lane];
            stageStats[laneBase] = tid;
            if (tid >= control[WF_CTRL_CAPACITY])
            {
                stageStats[laneBase + 1u] = 0xffffffffu;
                continue;
            }
            const PathState p = paths[tid];
            const PathRay r = rays[tid];
            const float3 origin = float3(r.origin);
            const float3 direction = float3(r.direction);
            // prepare is intentionally not specialised with the scene feature
            // constants. The runtime flag keeps non-SSS diagnostics from
            // reading the uninitialised cold table; this only runs for the
            // handful of diagnostic lanes and adds no traffic to rendering.
            stageStats[laneBase + 1u] = diagnosticsMediumEnabled != 0u ? mediumPaths[tid].medium : 0u;
            stageStats[laneBase + 2u] = p.depthAndFlags;
            stageStats[laneBase + 3u] = as_type<uint32_t>(origin.x);
            stageStats[laneBase + 4u] = as_type<uint32_t>(origin.y);
            stageStats[laneBase + 5u] = as_type<uint32_t>(origin.z);
            stageStats[laneBase + 6u] = as_type<uint32_t>(direction.x);
            stageStats[laneBase + 7u] = as_type<uint32_t>(direction.y);
            stageStats[laneBase + 8u] = as_type<uint32_t>(direction.z);
            stageStats[laneBase + 9u] = as_type<uint32_t>(dot(direction, direction));
            stageStats[laneBase + 10u] =
                as_type<uint32_t>(max(max(float3(p.throughput).x, float3(p.throughput).y), float3(p.throughput).z));
        }
        control[WF_CTRL_STATS_PATHS + min(bounceIdx, 31u)] = n;
    }
    control[WF_CTRL_ACTIVE] = n;
    control[WF_CTRL_DISPATCH + 0] = (n + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_DISPATCH + 1] = 1u;
    control[WF_CTRL_DISPATCH + 2] = 1u;
    prepareTraversalDispatches(traversalDispatches, n, threadsPerGroup, traversalBatchThreads, traversalBatchCount);
    // The stages about to run append into these, so clear their counts before
    // anything can add to them.
    control[1u - srcIdx] = 0u;
    control[WF_CTRL_SHADOW] = 0u;
    control[WF_CTRL_HIT] = 0u;
    control[WF_CTRL_MISS] = 0u;
}

// Between `extend` and the two stages that consume its classification.
kernel void wavefrontPrepareHitMiss(device uint32_t& controlRef [[buffer(0)]],
                                    constant uint32_t& threadsPerGroup [[buffer(1)]])
{
    device uint32_t* control = &controlRef;
    const uint32_t h = min(control[WF_CTRL_HIT], control[WF_CTRL_CAPACITY]);
    control[WF_CTRL_HIT_N] = h;
    control[WF_CTRL_HIT_DIS + 0] = (h + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_HIT_DIS + 1] = 1u;
    control[WF_CTRL_HIT_DIS + 2] = 1u;

    const uint32_t m = min(control[WF_CTRL_MISS], control[WF_CTRL_CAPACITY]);
    control[WF_CTRL_MISS_N] = m;
    control[WF_CTRL_MISS_DIS + 0] = (m + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_MISS_DIS + 1] = 1u;
    control[WF_CTRL_MISS_DIS + 2] = 1u;
    control[WF_CTRL_RESTIR] = 0u;
    control[WF_CTRL_RESTIR_DIS + 0] = 0u;
    control[WF_CTRL_RESTIR_DIS + 1] = 1u;
    control[WF_CTRL_RESTIR_DIS + 2] = 1u;
}

// Between `shade` and `shadow`: publish the number of shadow rays `shade`
// emitted and size their dispatch. Separate from wavefrontPrepare because the
// count does not exist until `shade` has run.
kernel void wavefrontPrepareShadow(device uint32_t& controlRef [[buffer(0)]],
                                   constant uint32_t& threadsPerGroup [[buffer(1)]],
                                   constant uint32_t& bounceIdx [[buffer(2)]],
                                   device uint32_t* traversalDispatches [[buffer(3)]],
                                   constant uint32_t& traversalBatchThreads [[buffer(4)]],
                                   constant uint32_t& traversalBatchCount [[buffer(5)]])
{
    device uint32_t* control = &controlRef;
    const uint32_t n = min(control[WF_CTRL_SHADOW], control[WF_CTRL_CAPACITY]);
    control[WF_CTRL_SHADOW_N] = n;
    control[WF_CTRL_STATS_SHADOW + min(bounceIdx, 31u)] = n;
    control[WF_CTRL_SHADOW_DIS + 0] = (n + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_SHADOW_DIS + 1] = 1u;
    control[WF_CTRL_SHADOW_DIS + 2] = 1u;
    prepareTraversalDispatches(traversalDispatches, n, threadsPerGroup, traversalBatchThreads, traversalBatchCount);
    const uint32_t guides = min(control[WF_CTRL_GUIDE], control[WF_CTRL_CAPACITY]);
    control[WF_CTRL_GUIDE_N] = guides;
    control[WF_CTRL_GUIDE_DIS + 0] = (guides + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_GUIDE_DIS + 1] = 1u;
    control[WF_CTRL_GUIDE_DIS + 2] = 1u;
}

// Transmittance of a shadow ray through whatever bounded media it crosses.
//
// The medium's boundary is not on the shadow mask -- a fog gizmo left there would
// black out everything it encloses -- so the segments inside it have to be found
// with a traversal of their own. That is the second traversal this feature was
// scoped to avoid, which is why it is gated on the scene having a bounded medium
// at all, and why it walks a bounded number of crossings rather than to
// completion.
//
// Alternating closest hits rather than an any-hit sweep: a convex volume answers
// in two, and the alternative is a payload that sorts an unbounded set of
// distances. `startMedium` is the one thing the ray cannot work out for itself --
// whether it began inside. A vertex within a fog volume and one just outside it
// produce the same origin and direction.
template <typename T>
static float3 mediumTransmittance(typename T::structure accelerationStructure,
                                  constant Uniforms& uniforms,
                                  device const Material* materials,
                                  device const GeometryEntry* geometryEntries,
                                  constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                  float3 origin,
                                  float3 direction,
                                  float maxDistance,
                                  uint32_t startMedium,
                                  float motionTime)
{
    constexpr uint32_t kMaxCrossings = 8u;

    float3 optical = float3(0.0f);
    float travelled = 0.0f;
    uint32_t medium = startMedium;

    for (uint32_t i = 0u; i < kMaxCrossings; ++i)
    {
        const float remaining = maxDistance - travelled;
        if (remaining <= 1e-5f)
        {
            break;
        }

        typename T::volume_isect isect;
        isect.assume_geometry_type(geometry_type::triangle);
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(false);

        ray r;
        r.origin = origin + direction * travelled;
        r.direction = direction;
        r.min_distance = 1e-4f;
        r.max_distance = remaining;

        const auto hit = T::traceVolume(isect, r, accelerationStructure, GEOMETRY_MASK_MEDIUM, motionTime);
        const bool escaped = (hit.type == intersection_type::none);
        const float segment = escaped ? remaining : hit.distance;

        if (medium != 0u)
        {
            optical += mediumSigmaT(uniforms, materials, medium - 1u) * segment;
        }
        if (escaped)
        {
            break;
        }

        // The same toggle the crossing in `shade` uses, for the same reason: a
        // gizmo's winding is arbitrary, so the normal cannot say which way the
        // ray is going.
        const auto inst = instances[hit.instance_id];
        const uint32_t here = (geometryEntries[inst.userID + hit.geometry_id].materialId + 1u) & MEDIUM_INDEX_MASK;
        medium = (medium == here) ? 0u : here;

        travelled += segment + 1e-4f;
    }

    return exp(-optical);
}

// Cutout shadow occlusion, computed in the kernel.
//
// The alpha test used to live in an any-hit intersection function bound through
// an MTLIntersectionFunctionTable. That is unusable on the Metal 4 queue: every
// device read an intersection function makes faults, because neither the
// residency set nor the argument table reaches the traversal's own execution
// context, and Metal 4 has no useResource to fall back on. Carrying the pointers
// in on the ray payload does not help either, so it is the execution context and
// not the binding. It reproduces on a 288-byte buffer in the two-triangle
// 07_alpha_clip scene, so it is not a matter of scale.
//
// What it looked like before it was understood: an in-range load of a page that
// is not resident wedges the ray tracing unit, and the queue reports whichever
// command buffer was current as kIOGPUCommandBufferCallbackErrorHang -- a hang
// with no long dispatch anywhere near it, at six milliseconds of GPU time,
// landing on a different bounce and a different stage each run.
//
// Two implementations, because the tags decide what is available:
//
//   * Static traversal uses Metal's inline intersection_query, which walks the
//     same single traversal the any-hit did and hands each candidate back to the
//     kernel -- where the tables are the kernel's own bindings. Same cost, same
//     result, no intersection function.
//   * Motion traversal cannot: intersection_query rejects the motion tags. It
//     restarts past each cutout instead, which is what this renderer did before
//     the any-hit existed. Bounded, because a canopy can stack more leaves than
//     any shadow ray needs to resolve.
//
// Returns false when the ray is blocked; `transmittance` is what survived.
// One cutout candidate's contribution to the shadow ray, and the roulette that
// ends the walk. Shared by both specialisations below: the compensation applied
// to the survivors in `shadowImpl` assumes exactly this termination rule, so the
// two must not drift apart.
static inline bool cutoutRouletteDone(thread float3& transmittance, float opacity, float cutoff)
{
    transmittance *= (1.0f - opacity);
    const float left = max(max(transmittance.x, transmittance.y), transmittance.z);
    return left <= cutoff || all(transmittance <= 1e-6f);
}

// How many cutout crossings the restart walk below may take before it gives up
// and reports what it has. It bounds that walk only -- the inline query needs no
// bound, because it answers the whole ray in one traversal.
constant uint32_t kMaxCutoutCrossings = 16u;

static inline bool shadowLightProxy(uint32_t instanceId,
                                    constant MTLIndirectAccelerationStructureInstanceDescriptor* instances)
{
    return (instances[instanceId].mask & (GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_LIGHT_HIDDEN)) != 0u;
}

template <typename T, bool Inline>
struct CutoutShadowWalk
{
    static bool run(constant Uniforms& uniforms,
                    typename T::structure as,
                    ray shadowRay,
                    float motionTime,
                    float cutoff,
                    constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                    device const Material* materials,
                    device const GeometryEntry* geometryEntries,
                    device const char* vertexBuffer,
                    device const uint32_t* indexBuffer,
                    typename T::table functionTable,
                    thread float3& transmittance)
    {
        transmittance = float3(1.0f);
        ray probe = shadowRay;
        // Only probe.min_distance changes between restarts, so the intersector is
        // configured once. Closest hit, because the walk has to meet the cutouts
        // in the order the ray does; forced opaque so traversal never looks for
        // an intersection function.
        typename T::isect isect;
        isect.assume_geometry_type(T::geometryTypes());
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(false);
        for (uint32_t crossing = 0u; crossing < kMaxCutoutCrossings; ++crossing)
        {
            if (crossing != 0u)
                auditWork(uniforms, WORK_ALPHA_TRAVERSAL_RESTARTS);
            const auto hit = T::trace(isect, probe, as, RAY_MASK_SHADOW, motionTime, functionTable);
            if (hit.type == intersection_type::none)
            {
                return true; // nothing else in the way
            }
            if (shadowLightProxy(hit.instance_id, instances))
            {
                return false;
            }
            // Curve geometry is built opaque and carries no cutout, so a strand
            // blocks outright rather than being alpha tested.
            float opacity = 1.0f;
            if (hit.type == intersection_type::triangle)
            {
                opacity =
                    cutoutOpacityAt(hit.primitive_id, hit.geometry_id, hit.instance_id, hit.triangle_barycentric_coord,
                                    instances, materials, geometryEntries, vertexBuffer, indexBuffer);
            }
            if (cutoutRouletteDone(transmittance, opacity, cutoff))
            {
                return false;
            }
            // Past the candidate just tested. Relative, because an absolute
            // epsilon is either too small to clear a distant hit or big enough
            // to step over a near one.
            probe.min_distance = hit.distance * (1.0f + 1e-5f) + 1e-5f;
            if (probe.min_distance >= probe.max_distance)
            {
                return true;
            }
        }
        return true;
    }
};

template <typename T>
struct CutoutShadowWalk<T, true>
{
    static bool run(constant Uniforms&,
                    typename T::structure as,
                    ray shadowRay,
                    float,
                    float cutoff,
                    constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                    device const Material* materials,
                    device const GeometryEntry* geometryEntries,
                    device const char* vertexBuffer,
                    device const uint32_t* indexBuffer,
                    typename T::table functionTable,
                    thread float3& transmittance)
    {
        transmittance = float3(1.0f);
        // Accept any hit and restrict geometry to avoid nearest-hit work and irrelevant bounding-box candidates.
        intersection_params params;
        params.accept_any_intersection(true);
        params.assume_geometry_type(T::geometryTypes());
        typename T::query q;
        q.reset(shadowRay, as, RAY_MASK_SHADOW, params, functionTable);
        while (q.next())
        {
            const uint32_t instanceId = q.get_candidate_instance_id();
            if (shadowLightProxy(instanceId, instances))
            {
                q.abort();
                return false;
            }
            // Only geometry the builder left non-opaque surfaces as a candidate.
            // Opaque geometry is committed by traversal itself and is never seen
            // here -- which is why the committed result has to be read after the
            // loop. Getting that wrong drops every opaque shadow caster in the
            // scene: the ground keeps its dappled canopy shade and loses the
            // trunks and rocks entirely, at 156 of this scene's 300 geometries.
            if (q.get_candidate_intersection_type() != intersection_type::triangle)
            {
                // A curve candidate carries no cutout -- curve geometry is built
                // opaque -- so it blocks.
                q.abort();
                return false;
            }
            const float opacity = cutoutOpacityAt(q.get_candidate_primitive_id(), q.get_candidate_geometry_id(),
                                                  instanceId, q.get_candidate_triangle_barycentric_coord(), instances,
                                                  materials, geometryEntries, vertexBuffer, indexBuffer);
            if (cutoutRouletteDone(transmittance, opacity, cutoff))
            {
                q.abort();
                return false;
            }
        }
        // Anything committed during that walk was opaque, and blocks outright.
        return q.get_committed_intersection_type() == intersection_type::none;
    }
};

static bool restirDiagnosticVisible(constant Uniforms& uniforms,
                                    CurveStaticTraversal::structure accelerationStructure,
                                    CurveStaticTraversal::table functionTable,
                                    thread const LightConnection& connection,
                                    float3 shadowOrigin,
                                    constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                    device const Material* materials,
                                    device const GeometryEntry* geometryEntries,
                                    device const char* vertexBuffer,
                                    device const uint32_t* indexBuffer,
                                    thread float& transmittance)
{
    const EmissiveVisibilitySegment segment = lightVisibilitySegment(connection, shadowOrigin);
    if (!segment.valid)
    {
        transmittance = 0.0f;
        return false;
    }
    ray shadowRay;
    shadowRay.origin = shadowOrigin;
    shadowRay.direction = segment.direction;
    shadowRay.min_distance = 0.0f;
    shadowRay.max_distance = segment.maxDistance;
    if (!SPEC_ALPHA)
    {
        transmittance = 1.0f;
        CurveStaticTraversal::isect isect;
        isect.assume_geometry_type(CurveStaticTraversal::geometryTypes());
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(true);
        return CurveStaticTraversal::trace(isect, shadowRay, accelerationStructure, RAY_MASK_SHADOW, 0.0f, functionTable)
                   .type == intersection_type::none;
    }
    float3 alphaTransmittance;
    const bool visible = CutoutShadowWalk<CurveStaticTraversal, false>::run(
        uniforms, accelerationStructure, shadowRay, 0.0f, 0.0f, instances, materials, geometryEntries, vertexBuffer,
        indexBuffer, functionTable, alphaTransmittance);
    transmittance = alphaTransmittance.x;
    return visible;
}

// ---------------------------------------------------------------------------
// shadow -- resolve the deferred connections
// ---------------------------------------------------------------------------
static void restirStoreFinalVisibility(constant Uniforms& uniforms, uint32_t pixelIndex, float transmittance)
{
    device RestirReservoir* reservoirs =
        (uniforms.frameIndex & 1u) != 0u ? uniforms.restirReservoir1 : uniforms.restirReservoir0;
    const uint32_t encoded = uint32_t(clamp(transmittance, 0.0f, 1.0f) * float(RESTIR_RESERVOIR_VISIBILITY_BITS) + 0.5f);
    device uint32_t& flags = reservoirs[pixelIndex].state.ageAndFlags;
    flags = (flags & ~RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK) | RESTIR_RESERVOIR_INITIAL_VISIBLE |
            (encoded << RESTIR_RESERVOIR_VISIBILITY_SHIFT);
}

template <typename T>
static void shadowImpl(uint gid,
                       constant Uniforms& uniforms,
                       typename T::structure accelerationStructure,
                       device const ShadowRay* shadowRays,
                       device float4* radianceOut,
                       device const uint32_t* control,
                       constant uint32_t& sampleIdx,
                       constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                       device const Material* materials,
                       device const GeometryEntry* geometryEntries,
                       device const char* vertexBuffer,
                       device const uint32_t* indexBuffer,
                       typename T::table functionTable,
                       device SharcUpdateState* sharcUpdates,
                       device SharcAccumulationEntry* sharcAccumulation,
                       uint32_t bounce)
{
    if (gid >= control[WF_CTRL_SHADOW_N])
    {
        return;
    }
    const ShadowRay sr = shadowRays[gid];
    const bool restirHistoryRay =
        SPEC_RENDER_WORK_AUDIT && bounce == 0u && (sr.sharcPathIndex & RESTIR_AUDIT_HISTORY_BIT) != 0u;
    const bool restirVisibilityUpdate = bounce == 0u && uniforms.restirFinalVisibilityReuse != 0u &&
                                        (sr.sharcPathIndex & RESTIR_VISIBILITY_UPDATE_BIT) != 0u;
    const uint32_t sharcPathIndex = sr.sharcPathIndex & RESTIR_AUDIT_PATH_INDEX_MASK;
    auditWork(uniforms, WORK_SHADOW_RAYS_BASE + min(bounce, WORK_BOUNCE_SLOTS - 1u));
    // Opaque validation scenes issue exactly one hardware query per shadow ray.
    // Alpha restart walks may issue more and remain classified separately in
    // the static ledger.
    auditWork(uniforms, WORK_INTERSECTION_QUERIES);

    ray shadowRay;
    shadowRay.origin = float3(sr.origin);
    shadowRay.direction = float3(sr.direction);
    // Zero, because connectLight/connectEnvLight now offset the origin along the
    // face the ray leaves through. A world-space epsilon standing in for that
    // offset is the wrong shape for the problem -- too small at architectural
    // scale, too large at prop scale -- and it was applied to connections whose
    // origin had already been offset, which is the same self-occlusion
    // asymmetry from the other side. OptiX runs its occlusion rays at
    // shadowRayTmin, which both apps seed to 0.
    shadowRay.min_distance = 0.0f;
    shadowRay.max_distance = sr.maxDistance;

    const float motionTime = motionTimeFor(uniforms, sr.pixelIndex, sampleIdx);
    float3 weight = float3(sr.weight);
    float3 sharcRadiance = float3(sr.sharcRadiance);

    if (!SPEC_ALPHA)
    {
        // No cutouts in this scene: one any-hit trace, exactly as before.
        typename T::isect isect;
        isect.assume_geometry_type(T::geometryTypes());
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(true);
        const bool visible =
            T::trace(isect, shadowRay, accelerationStructure, RAY_MASK_SHADOW, motionTime, functionTable).type ==
            intersection_type::none;
        if (restirVisibilityUpdate)
            restirStoreFinalVisibility(uniforms, sr.pixelIndex, visible ? 1.0f : 0.0f);
        if (visible)
        {
            // Geometry visibility and atmospheric transmittance are separate; surviving shadow rays still cross fog.
            if (SPEC_FOG && uniforms.hasFog)
            {
                const float tau = fogOpticalDepth(
                    float3(sr.origin), float3(sr.direction), sr.maxDistance, uniforms.fogHeight, uniforms.fogSigmaT);
                const float fogTransmittance = exp(-tau);
                weight *= fogTransmittance;
                sharcRadiance *= fogTransmittance;
            }
            if (SPEC_SSS && uniforms.hasBoundedMedium)
            {
                const float3 transmittance = mediumTransmittance<T>(
                    accelerationStructure, uniforms, materials, geometryEntries, instances, float3(sr.origin),
                    float3(sr.direction), sr.maxDistance, sr.medium, motionTime);
                weight *= transmittance;
                sharcRadiance *= transmittance;
            }
            if (bounce == 0u)
            {
                restirDiagnosticVisibility(uniforms, sr.pixelIndex, weight);
                if (restirHistoryRay)
                    auditWork(uniforms, WORK_RESTIR_FINAL_HISTORY_VISIBLE);
            }
            radianceOut[sr.pixelIndex] += float4(weight, 0.0f);
            if (SPEC_SHARC_UPDATE)
            {
                const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, sharcPathIndex);
                const SharcUpdateState updateState = sharcUpdates[updateIndex];
                sharcPropagate(updateState, sharcAccumulation, sharcRadiance, uniforms,
                               (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u);
            }
        }
        return;
    }

    // Cutouts make occlusion a product rather than a predicate, so a plain
    // any-hit does not answer the question: the nearest hit may be a hole.
    //
    // Deterministic rather than stochastic: a MASK surface contributes 0 or 1
    // exactly and a BLEND one its alpha, which is far quieter than rolling a
    // second random number per shadow ray.
    //
    // The walk itself is in CutoutShadowWalk -- inline intersection_query where
    // the tags allow it, a bounded restart otherwise. See the note there for why
    // this cannot be an any-hit intersection function on Metal 4.
    float3 transmittance;
    if (!CutoutShadowWalk<T, T::kInlineQuery != 0>::run(uniforms, accelerationStructure, shadowRay, motionTime,
                                                        sr.rrCutoff, instances, materials, geometryEntries,
                                                        vertexBuffer, indexBuffer, functionTable, transmittance))
    {
        if (restirVisibilityUpdate)
            restirStoreFinalVisibility(uniforms, sr.pixelIndex, 0.0f);
        return; // fully blocked
    }
    // The survivors of the roulette carry the weight of the ones it killed. A
    // ray ends below the cutoff only if it passed the test, which it does with
    // probability (its transmittance / cutoff), so scaling by the inverse of
    // that puts the expectation back where it was.
    const float m = max(max(transmittance.x, transmittance.y), transmittance.z);
    if (m < kShadowTransmittanceCutoff)
    {
        transmittance *= kShadowTransmittanceCutoff / max(m, 1e-20f);
    }
    if (restirVisibilityUpdate)
        restirStoreFinalVisibility(uniforms, sr.pixelIndex, transmittance.x);
    weight *= transmittance;
    sharcRadiance *= transmittance;
    if (all(weight <= 1e-6f))
    {
        return;
    }

    if (SPEC_FOG && uniforms.hasFog)
    {
        const float tau = fogOpticalDepth(
            float3(sr.origin), float3(sr.direction), sr.maxDistance, uniforms.fogHeight, uniforms.fogSigmaT);
        const float fogTransmittance = exp(-tau);
        weight *= fogTransmittance;
        sharcRadiance *= fogTransmittance;
    }
    if (SPEC_SSS && uniforms.hasBoundedMedium)
    {
        const float3 mediumTr =
            mediumTransmittance<T>(accelerationStructure, uniforms, materials, geometryEntries, instances,
                                   float3(sr.origin), float3(sr.direction), sr.maxDistance, sr.medium, motionTime);
        weight *= mediumTr;
        sharcRadiance *= mediumTr;
    }
    if (bounce == 0u)
    {
        restirDiagnosticVisibility(uniforms, sr.pixelIndex, weight);
        if (restirHistoryRay)
            auditWork(uniforms, WORK_RESTIR_FINAL_HISTORY_VISIBLE);
    }
    radianceOut[sr.pixelIndex] += float4(weight, 0.0f);
    if (SPEC_SHARC_UPDATE)
    {
        const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, sharcPathIndex);
        const SharcUpdateState updateState = sharcUpdates[updateIndex];
        sharcPropagate(updateState, sharcAccumulation, sharcRadiance, uniforms,
                       (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u);
    }
}

// Full reset clears all three resources. Responsive mode uses the same kernel
// with clearPersistent == 0 to clear only the per-frame atomic accumulation.
kernel void sharcClear(uint tid [[thread_position_in_grid]],
                       constant Uniforms& uniforms [[buffer(0)]],
                       device SharcHashEntry* hashEntries [[buffer(1)]],
                       device SharcAccumulationEntry* accumulationEntries [[buffer(2)]],
                       device SharcResolvedEntry* resolvedEntries [[buffer(3)]],
                       device atomic_uint* stats [[buffer(4)]],
                       constant uint32_t& clearPersistent [[buffer(5)]])
{
    if (tid < uniforms.sharcCapacity)
    {
        for (uint32_t channel = 0u; channel < 3u; ++channel)
        {
            atomic_store_explicit(&accumulationEntries[tid].radiance[channel], 0, memory_order_relaxed);
            atomic_store_explicit(&accumulationEntries[tid].direction[channel], 0, memory_order_relaxed);
        }
        atomic_store_explicit(&accumulationEntries[tid].sampleCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&accumulationEntries[tid].diagnosticFlags, 0u, memory_order_relaxed);
        if (clearPersistent != 0u)
        {
            atomic_store_explicit(&hashEntries[tid].key, 0u, memory_order_relaxed);
            SharcResolvedEntry empty = {};
            resolvedEntries[tid] = empty;
        }
    }
    if (tid < SHARC_STAT_COUNT)
    {
        atomic_store_explicit(&stats[tid], 0u, memory_order_relaxed);
    }
}

// The only pass that publishes update data to queries. Running one thread per
// cache entry also owns stale eviction, so no hash mutation races a query.
kernel void sharcResolve(uint tid [[thread_position_in_grid]],
                         constant Uniforms& uniforms [[buffer(0)]],
                         device SharcHashEntry* hashEntries [[buffer(1)]],
                         device SharcAccumulationEntry* accumulationEntries [[buffer(2)]],
                         device SharcResolvedEntry* resolvedEntries [[buffer(3)]],
                         device atomic_uint* stats [[buffer(4)]])
{
    if (tid >= uniforms.sharcCapacity)
    {
        return;
    }
    const uint32_t mainCapacity = sharcMainCapacity(uniforms);
    const bool responsiveEntry = (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u && tid >= mainCapacity;
    const bool preserveAccumulation = (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u;
    const uint32_t key = atomic_load_explicit(&hashEntries[tid].key, memory_order_relaxed);
    if (key == 0u)
    {
        return;
    }

    device SharcAccumulationEntry& accumulation = accumulationEntries[tid];
    const uint32_t entrySampleCount = preserveAccumulation ?
                                          atomic_load_explicit(&accumulation.sampleCount, memory_order_relaxed) :
                                          atomic_exchange_explicit(&accumulation.sampleCount, 0u, memory_order_relaxed);
    const uint32_t diagnosticFlags =
        preserveAccumulation ? atomic_load_explicit(&accumulation.diagnosticFlags, memory_order_relaxed) :
                               atomic_exchange_explicit(&accumulation.diagnosticFlags, 0u, memory_order_relaxed);
    float3 sum = float3(0.0f);
    float3 directionSum = float3(0.0f);
    uint32_t maxRadianceFixed = 0u;
    const bool directional = (uniforms.sharcFlags & SHARC_FLAG_DIRECTIONAL) != 0u;
    const float inverseScale = 1.0f / max(uniforms.sharcRadianceScale, 1.0f);
    for (uint32_t channel = 0u; channel < 3u; ++channel)
    {
        const int32_t radiance = preserveAccumulation ?
                                     atomic_load_explicit(&accumulation.radiance[channel], memory_order_relaxed) :
                                     atomic_exchange_explicit(&accumulation.radiance[channel], 0, memory_order_relaxed);
        const int32_t direction =
            preserveAccumulation ? atomic_load_explicit(&accumulation.direction[channel], memory_order_relaxed) :
                                   atomic_exchange_explicit(&accumulation.direction[channel], 0, memory_order_relaxed);
        sum[channel] = directional ? float(radiance) * inverseScale : float(as_type<uint32_t>(radiance)) * inverseScale;
        directionSum[channel] = float(direction) * inverseScale;
        maxRadianceFixed =
            max(maxRadianceFixed, directional ? sharcSignedMagnitude(radiance) : as_type<uint32_t>(radiance));
        if (directional)
        {
            maxRadianceFixed = max(maxRadianceFixed, sharcSignedMagnitude(direction));
        }
    }
    if (uniforms.sharcDebug != 0u)
    {
        if ((diagnosticFlags & kSharcDiagnosticAccumulationClamp) != 0u)
        {
            atomic_fetch_add_explicit(&stats[SHARC_STAT_ACCUMULATION_CLAMP], 1u, memory_order_relaxed);
        }
        if ((diagnosticFlags & kSharcDiagnosticNonfiniteReject) != 0u)
        {
            atomic_fetch_add_explicit(&stats[SHARC_STAT_NONFINITE_REJECT], 1u, memory_order_relaxed);
        }
        sharcAtomicMax(&stats[SHARC_STAT_MAX_RADIANCE_FIXED], maxRadianceFixed);
        sharcAtomicMax(&stats[SHARC_STAT_MAX_SAMPLE_COUNT], entrySampleCount);
    }

    SharcResolvedEntry previous = resolvedEntries[tid];
    uint32_t accumulatedFrames = (previous.accumulatedFramesAndFlags & ~SHARC_RESOLVED_RESPONSIVE_ENTRY) + 1u;
    uint32_t staleFrames = entrySampleCount != 0u ? 0u : previous.staleFrames + 1u;
    const uint32_t staleLimit =
        responsiveEntry ? max(uniforms.sharcResponsiveFrames, 1u) : clamp(uniforms.sharcStaleFrameCount, 8u, 1024u);
    if (staleFrames >= staleLimit)
    {
        SharcResolvedEntry empty = {};
        resolvedEntries[tid] = empty;
        // Resolve is the sole hash mutator in this phase; update and query are
        // separated by barriers, so eviction needs no fallible weak CAS.
        atomic_store_explicit(&hashEntries[tid].key, 0u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[SHARC_STAT_EVICTION], 1u, memory_order_relaxed);
        return;
    }

    const uint32_t fadeBit = 1u << (uniforms.sharcFrameIndex & 31u);
    uint32_t count = entrySampleCount;
    if (responsiveEntry)
    {
        uint32_t mainIndex = SHARC_NO_ENTRY;
        SharcAddress mainAddress;
        mainAddress.key = key;
        mainAddress.hash = sharcHash(key);
        mainAddress.voxelSize = 0.0f;
        mainAddress.levelBlend = 0.0f;
        mainAddress.level = 0;
        if (sharcFindEntry(uniforms, hashEntries, mainAddress, false, false, nullptr, mainIndex))
        {
            count = atomic_load_explicit(&accumulationEntries[mainIndex].sampleCount, memory_order_relaxed);
        }
    }

    if (count == 0u)
    {
        previous.accumulatedFramesAndFlags =
            min(accumulatedFrames, 1024u) | (responsiveEntry ? SHARC_RESOLVED_RESPONSIVE_ENTRY : 0u);
        previous.staleFrames = staleFrames;
        if ((uniforms.sharcFlags & SHARC_FLAG_FADE_ACCELERATION) != 0u && !responsiveEntry)
        {
            previous.fadeMask |= fadeBit;
        }
        resolvedEntries[tid] = previous;
        return;
    }

    float4 currentRadiance;
    float4 currentDirection = float4(0.0f);
    if (directional)
    {
        currentRadiance = float4(sharcResolveDirection(sum / float(count)), directionSum.x / float(count));
        currentDirection.xy = directionSum.yz / float(count);
    }
    else
    {
        currentRadiance = float4(max(sum / float(count), float3(0.0f)), 0.0f);
    }
    if (previous.sampleCount == 0u)
    {
        // A cell whose first insertion lost the race for its slot is re-inserted
        // further along the same bucket, and its history is sitting in the older
        // copy. Adopt it rather than restarting temporal accumulation from
        // nothing every time the hash map shuffles an entry.
        // Bounded by the region this entry lives in. Responsive companions carry
        // the same spatial key as their persistent partner -- the two are told
        // apart by which half of the table they sit in -- so a window that ran
        // past the split would adopt the wrong signal's history.
        const uint32_t regionEnd = responsiveEntry ? uniforms.sharcCapacity : mainCapacity;
        const uint32_t searchEnd = min(tid + 1u + kSharcLinearProbeWindow, regionEnd);
        for (uint32_t i = tid + 1u; i < searchEnd; ++i)
        {
            if (atomic_load_explicit(&hashEntries[i].key, memory_order_relaxed) != key)
            {
                continue;
            }
            const SharcResolvedEntry older = resolvedEntries[i];
            previous.radiance = older.radiance;
            previous.direction = older.direction;
            previous.sampleCount = older.sampleCount;
            previous.fadeMask = older.fadeMask;
            accumulatedFrames = (older.accumulatedFramesAndFlags & ~SHARC_RESOLVED_RESPONSIVE_ENTRY) + 1u;
            staleFrames = 0u;
            break;
        }
    }

    float previousCount = float(previous.sampleCount);
    const uint32_t historyLimit =
        clamp(responsiveEntry ? uniforms.sharcResponsiveFrames : uniforms.sharcAccumulationFrames, 1u, 1024u);
    if (accumulatedFrames > historyLimit)
    {
        previousCount *= float(historyLimit) / float(accumulatedFrames);
        accumulatedFrames = historyLimit;
    }

    if ((uniforms.sharcFlags & SHARC_FLAG_FADE_ACCELERATION) != 0u && !responsiveEntry)
    {
        const float currentLuminance = directional ? max(currentRadiance.w, 0.0f) : sharcLuminance(currentRadiance.xyz);
        const float previousLuminance =
            directional ? max(float(previous.radiance.w), 0.0f) : sharcLuminance(float3(previous.radiance.xyz));
        const bool fading = currentLuminance < previousLuminance;
        previous.fadeMask = (previous.fadeMask & ~fadeBit) | (fading ? fadeBit : 0u);
        if (popcount(previous.fadeMask) == 32u)
        {
            previousCount = float(count);
        }
    }

    float combinedCount = previousCount + float(count);
    const float alpha = float(count) / max(combinedCount, 1.0f);
    float4 resolvedRadiance = mix(float4(previous.radiance), currentRadiance, alpha);
    float4 resolvedDirection = mix(float4(previous.direction), currentDirection, alpha);

    const float3 cameraDelta = uniforms.viewToWorld[3].xyz - float3(uniforms.sharcCameraPrev);
    if ((uniforms.sharcFlags & SHARC_FLAG_BLEND_ADJACENT_LEVELS) != 0u && !responsiveEntry &&
        dot(cameraDelta, cameraDelta) > 1e-6f && accumulatedFrames <= 2u)
    {
        const SharcAddress adjacentAddress = sharcAdjacentLevelAddress(uniforms, key);
        uint32_t adjacentIndex = SHARC_NO_ENTRY;
        if (sharcFindEntry(uniforms, hashEntries, adjacentAddress, false, false, nullptr, adjacentIndex))
        {
            const SharcResolvedEntry adjacent = resolvedEntries[adjacentIndex];
            if (adjacent.sampleCount != 0u)
            {
                const float adjacentCount = float(adjacent.sampleCount);
                const float adjacentWeight = adjacentCount / (combinedCount + adjacentCount);
                resolvedRadiance = mix(resolvedRadiance, float4(adjacent.radiance), adjacentWeight);
                resolvedDirection = mix(resolvedDirection, float4(adjacent.direction), adjacentWeight);
                combinedCount += adjacentCount;
            }
        }
    }

    previous.radiance = half4(clamp(resolvedRadiance, float4(-65504.0f), float4(65504.0f)));
    previous.direction = half4(clamp(resolvedDirection, float4(-65504.0f), float4(65504.0f)));
    previous.sampleCount = uint32_t(min(round(combinedCount), 4294960000.0f));
    previous.accumulatedFramesAndFlags = accumulatedFrames | (responsiveEntry ? SHARC_RESOLVED_RESPONSIVE_ENTRY : 0u);
    previous.staleFrames = staleFrames;
    resolvedEntries[tid] = previous;
}

#define WF_SHADOW_ENTRY(NAME, TRAITS)                                                                                  \
    kernel void NAME(uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]],                  \
                     TRAITS::structure accelerationStructure [[buffer(1)]],                                            \
                     device const ShadowRay* shadowRays [[buffer(2)]], device float4* radianceOut [[buffer(3)]],       \
                     device const uint32_t* control [[buffer(4)]], constant uint32_t& sampleIdx [[buffer(5)]],         \
                     constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(6)]],             \
                     device const Material* materials [[buffer(7)]],                                                   \
                     device const GeometryEntry* geometryEntries [[buffer(8)]],                                        \
                     device const char* vertexBuffer [[buffer(9)]], device const uint32_t* indexBuffer [[buffer(10)]], \
                     device const UniformLight* lights [[buffer(11)]], constant uint32_t& queueOffset [[buffer(12)]],  \
                     device SharcUpdateState* sharcUpdates [[buffer(13)]],                                             \
                     device SharcAccumulationEntry* sharcAccumulation [[buffer(14)]],                                  \
                     TRAITS::table functionTable [[buffer(15)]], constant uint32_t& bounce [[buffer(16)]])             \
    {                                                                                                                  \
        shadowImpl<TRAITS>(gid + queueOffset, uniforms, accelerationStructure, shadowRays, radianceOut, control,       \
                           sampleIdx, instances, materials, geometryEntries, vertexBuffer, indexBuffer, functionTable, \
                           sharcUpdates, sharcAccumulation, bounce);                                                   \
    }

WF_SHADOW_ENTRY(wavefrontShadow, MotionTraversal)
WF_SHADOW_ENTRY(wavefrontShadowStatic, StaticTraversal)
WF_SHADOW_ENTRY(wavefrontShadowCurve, CurveMotionTraversal)
WF_SHADOW_ENTRY(wavefrontShadowStaticCurve, CurveStaticTraversal)

// ---------------------------------------------------------------------------
// resolve -- average the samples and fold into the accumulation buffer
//
// Fold this launch into the persistent accumulation buffer.
// ---------------------------------------------------------------------------
kernel void wavefrontResolve(uint tid [[thread_position_in_grid]],
                             constant Uniforms& uniforms [[buffer(0)]],
                             device const float4* radianceIn [[buffer(1)]],
                             device float4* res [[buffer(2)]],
                             device float4* accum [[buffer(3)]],
                             constant uint32_t& sampleCount [[buffer(4)]],
                             device const AovSample* aov [[buffer(5)]],
                             device SharcHashEntry* sharcHashEntries [[buffer(6)]],
                             device const SharcResolvedEntry* sharcResolved [[buffer(7)]])
{
    const uint32_t pixelCount = uniforms.width * uniforms.height;
    if (tid >= pixelCount)
    {
        return;
    }

    // Guide views. A denoiser fed a broken guide degrades quietly, so the guides
    // have to be inspectable on their own.
    const uint32_t debugMode = uniforms.debug;
    if (uniforms.sharcCapacity != 0u && debugMode == (uint32_t)DebugMode::eSharcOccupancy)
    {
        res[tid] = float4(sharcDebugOccupancy(tid, uniforms, sharcHashEntries, sharcResolved), 1.0f);
        return;
    }
    if (debugMode == (uint32_t)DebugMode::eSharcBounces)
    {
        res[tid] = float4(sharcDebugBounceColor((uint32_t)max(aov[tid].guideStateOrBounceDepth, 0.0f)), 1.0f);
        return;
    }
    if (DEBUG_MODE_IS_AOV(debugMode))
    {
        const AovSample a = aov[tid];
        float3 v = float3(0.0f);
        switch ((DebugMode)debugMode)
        {
        case DebugMode::eAovDiffuseAlbedo:
            v = float3(a.diffuseAlbedo);
            break;
        case DebugMode::eAovSpecularAlbedo:
            v = float3(a.specularAlbedo);
            break;
        case DebugMode::eAovNormal:
            v = float3(a.normal) * 0.5f + 0.5f;
            break;
        case DebugMode::eAovRoughness:
            v = float3(a.roughness);
            break;
        // d/(1+d): monotonic and scale-free, so a scene of any size is readable
        // and nothing crosses zero the way a logarithm does at d == 1.
        case DebugMode::eAovDepth:
            v = float3(a.depth / (1.0f + a.depth));
            break;
        // Red/green for the two axes, scaled so a few pixels of motion is visible.
        case DebugMode::eAovMotion:
            v = float3(a.motionX, a.motionY, 0.0f) * 0.05f + 0.5f;
            break;
        // Red where the denoiser is being told to distrust its history, so the
        // extent of the mask is a thing you can look at rather than infer.
        case DebugMode::eAovReactive:
            v = float3(a.reactive, 0.0f, 0.0f);
            break;
        // d/(1+d) again: scale-free, and zero stays zero.
        case DebugMode::eAovSpecularHitDistance:
            v = float3(a.specularHitDistance / (1.0f + a.specularHitDistance));
            break;
        default:
            break;
        }
        res[tid] = float4(v, 1.0f);
        return;
    }

    float3 result = radianceIn[tid].xyz / (float)max(sampleCount, 1u);

    if (uniforms.enableAccumulation)
    {
        float3 accumColor = result;
        if (uniforms.subframeIndex > 0)
        {
            // subframeIndex counts samples already folded in; this launch adds
            // sampleCount more, so the new mean carries weight m / (n + m).
            const float a = (float)sampleCount / (float)(uniforms.subframeIndex + sampleCount);
            accumColor = mix(float3(accum[tid]), accumColor, a);
        }
        accum[tid] = float4(accumColor, 1.0f);
        result = accumColor;
    }

    res[tid] = float4(result, 1.0f);
}

// ---------------------------------------------------------------------------
// aovResolve -- spread the packed guide records into the textures MetalFX wants
//
// One kernel so every pixel-format decision lives in one place; `shade` stays
// free of texture bindings, which matters because it is the kernel with the
// least register headroom.
// ---------------------------------------------------------------------------
kernel void wavefrontAovResolve(uint2 tid [[thread_position_in_grid]],
                                constant Uniforms& uniforms [[buffer(0)]],
                                device const AovSample* aov [[buffer(1)]],
                                device const float4* radiance [[buffer(2)]],
                                constant uint32_t& sampleCount [[buffer(3)]],
                                texture2d<float, access::write> colorTex [[texture(0)]],
                                texture2d<float, access::write> depthTex [[texture(1)]],
                                texture2d<float, access::write> motionTex [[texture(2)]],
                                texture2d<float, access::write> diffuseTex [[texture(3)]],
                                texture2d<float, access::write> specularTex [[texture(4)]],
                                texture2d<float, access::write> normalTex [[texture(5)]],
                                texture2d<float, access::write> roughTex [[texture(6)]],
                                texture2d<float, access::write> specHitTex [[texture(7)]],
                                texture2d<float, access::write> reactiveTex [[texture(8)]],
                                texture2d<float, access::write> denoiseStrengthTex [[texture(9)]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
    {
        return;
    }
    const uint32_t i = tid.y * uniforms.width + tid.x;
    const AovSample a = aov[i];

    // Linear radiance, not the tonemapped image: the denoiser works in the space
    // the light actually arrived in and exposure comes after it.
    //
    // Divided by the sample count, as the resolve pass does. The radiance buffer
    // holds the *sum* over this launch's samples, so handing it over raw makes the
    // denoiser's input brighter by a factor of spp -- invisible at one sample per
    // launch, which is why it survived, and wrong the moment anyone raises it.
    //
    // This launch is the MetalFX frame stream. PT accumulation is resolved into
    // its own buffer by wavefrontResolve and deliberately does not enter here:
    // the current color and the current auxiliary buffers must describe one and
    // the same jittered visibility sample.
    float3 color = radiance[i].xyz / (float)max(sampleCount, 1u);

    // Firefly clamp, in exposed units so the threshold means the same thing at
    // any exposure. A single unbounded sample is a bright dot that a temporal
    // filter then smears across many frames -- it costs far more than the energy
    // it carries. Clamped rather than dropped, so the pixel keeps its hue and
    // most of its brightness. Off when the threshold is zero.
    if (uniforms.denoiseFireflyClamp > 0.0f)
    {
        const float3 exposed = color * float3(uniforms.exposureValue);
        const float lum = dot(exposed, float3(0.2126f, 0.7152f, 0.0722f));
        if (lum > uniforms.denoiseFireflyClamp)
        {
            color *= uniforms.denoiseFireflyClamp / lum;
        }
    }
    colorTex.write(float4(color, 1.0f), tid);
    depthTex.write(float4(a.depth, 0.0f, 0.0f, 0.0f), tid);
    motionTex.write(float4(a.motionX, a.motionY, 0.0f, 0.0f), tid);
    diffuseTex.write(float4(float3(a.diffuseAlbedo), 1.0f), tid);
    specularTex.write(float4(float3(a.specularAlbedo), 1.0f), tid);
    normalTex.write(float4(float3(a.normal), 0.0f), tid);
    roughTex.write(float4(a.roughness, 0.0f, 0.0f, 0.0f), tid);
    // The texture is R16Float. Environment reflections use a large finite proxy
    // for infinity; clamp it before conversion so the guide stays finite rather
    // than becoming +inf and poisoning the network input.
    specHitTex.write(float4(clamp(a.specularHitDistance, 0.0f, 65504.0f), 0.0f, 0.0f, 0.0f), tid);
    reactiveTex.write(float4(saturate(a.reactive), 0.0f, 0.0f, 0.0f), tid);
    // Negative marks a noise-free primary: a camera ray that missed or directly
    // visible emission. Reactive pixels are deliberately not reused here:
    // invalid reprojection should reject history, not disable spatial denoising
    // of the current sample.
    const float denoiseStrength = a.guideStateOrBounceDepth < 0.0f ? 1.0f : 0.0f;
    denoiseStrengthTex.write(float4(denoiseStrength, 0.0f, 0.0f, 0.0f), tid);
}
