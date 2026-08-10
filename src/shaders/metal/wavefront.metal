#include "shading_common.h"

// ============================================================================
// Wavefront path tracer.
//
// Same shading as the megakernel (everything comes from shading_common.h); only
// the scheduling differs. A path's state lives in memory instead of registers so
// that each stage can later run over just the paths that are still alive, rather
// than making every lane of a simdgroup wait for the longest-lived path in it.
//
// Live paths are compacted between bounces into an index queue, so a stage only
// dispatches threads for work that is still alive. Compaction moves 4-byte
// indices, not 48-byte records, and reserves output slots one atomic per
// simdgroup rather than one per lane.
//
// Path state stays indexed by pixel; only the queue is compacted. That keeps the
// side tables (hits, IOR stacks, radiance) addressable by a single index and
// makes the queue's contents a permutation rather than a copy.
//
// Shadow rays are deferred into their own stage: they are a second, incoherent
// traversal that would otherwise run inside the most register-hungry kernel, and
// only a subset of the shaded paths emit one.
// ============================================================================

#include "fog.h"
#include "subsurface.h"
#include "sharc.h"

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
    return (uint32_t)(saturate(a.x) * 255.0f + 0.5f) |
           ((uint32_t)(saturate(a.y) * 255.0f + 0.5f) << 8) |
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

// Layout of the control buffer, shared by every stage.
//   [0], [1] : live path count of each ping-pong queue
//   [2..4]   : MTLDispatchThreadgroupsIndirectArguments for extend/shade
//   [5]      : live path count for this bounce, so stages need no queue index
//   [6..10]  : shadow ray count, republished count, and its dispatch arguments
//   [11..15] : the same for rays that hit geometry
//   [16..20] : the same for rays that escaped
#define WF_CTRL_COUNT0     0
#define WF_CTRL_COUNT1     1
#define WF_CTRL_DISPATCH   2
#define WF_CTRL_ACTIVE     5
#define WF_CTRL_SHADOW     6
#define WF_CTRL_SHADOW_N   7
#define WF_CTRL_SHADOW_DIS 8
#define WF_CTRL_HIT        11
#define WF_CTRL_HIT_N      12
#define WF_CTRL_HIT_DIS    13
#define WF_CTRL_MISS       16
#define WF_CTRL_MISS_N     17
#define WF_CTRL_MISS_DIS   18
// Profiling only: live path count and shadow ray count per bounce, so the
// per-stage timings can be read as a cost per ray rather than a cost per stage.
#define WF_CTRL_STATS_PATHS  32
#define WF_CTRL_STATS_SHADOW 64

// Reserve a run of output slots for the active lanes of one simdgroup. Every
// lane that reaches a call site belongs in that queue -- the others already
// returned or took the other branch, so they are inactive and the simdgroup
// reductions below see only the lanes being queued. One atomic per simdgroup
// instead of one per lane.
static inline void queuePush(device atomic_uint* counter, device uint32_t* queueOut,
                             uint32_t pathIndex)
{
    const uint32_t rank = simd_prefix_exclusive_sum(1u);
    const uint32_t total = simd_sum(1u);
    uint32_t base = 0u;
    if (simd_is_first())
    {
        base = atomic_fetch_add_explicit(counter, total, memory_order_relaxed);
    }
    base = simd_broadcast_first(base);
    queueOut[base + rank] = pathIndex;
}


// Traversal specialisation.
//
// A motion acceleration structure is a different type from a static one, and the
// intersector that walks it is a different type again — so this cannot be a
// runtime branch or a function constant, it has to be two compiled variants. It
// is worth it: every ray was paying for motion-BVH traversal, including in scenes
// with no deforming geometry at all, and that was 12-19% of the frame.
// What a cutout shadow ray carries through traversal. The intersection function
// multiplies into it and lets traversal continue, so one traversal answers the
// whole ray instead of eight restarts from the root.
struct ShadowPayload
{
    float3 transmittance;
    float cutoff;
};

// Alpha test during traversal.
//
// Returning false means "this was a hole, keep going", which is what an any-hit
// shader is for. The alternative -- and what this replaces -- is to find the
// closest hit, test it, move the ray past it and trace again, up to eight times.
// Each of those restarts walks the tree from the root, and on this forest that
// loop was a quarter of the whole frame.
//
// Accepting on a collapsed transmittance keeps the early-out the loop had: once
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

static bool shadowAlphaAnyHitImpl(uint primitive_id,
                                  uint geometry_id,
                                  uint instance_id,
                                  float2 barycentric_coord,
                                  ray_data ShadowPayload& payload,
                                  constant MTLAccelerationStructureUserIDInstanceDescriptor* instances,
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
        return true; // blocks outright
    }

    constexpr uint32_t vtxStride = 32;
    constexpr uint32_t uvOff = 20;
    float2 uvv[3];
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t idx = indexBuffer[entry.indexOffset + primitive_id * 3 + k];
        uvv[k] = unpackUV(
            *(device const uint32_t*)(vertexBuffer + (entry.vbOffset + idx) * vtxStride + uvOff));
    }
    const float2 uv = interpolateAttrib(uvv[0], uvv[1], uvv[2], barycentric_coord);
    const float opacity = resolveOpacity(mat, uv);

    payload.transmittance *= (1.0f - opacity);

    // Monotonically decreasing, so crossing the threshold once is the same
    // event as ending below it -- which is what lets the compensation be
    // applied at the end, from one draw taken before the ray was traced.
    const float3 t = payload.transmittance;
    if (max(max(t.x, t.y), t.z) <= payload.cutoff)
    {
        return true; // roulette says this ray is done
    }
    return all(payload.transmittance <= 1e-6f);
}

// Two entry points over one body. The tags an intersection function carries have
// to match the intersector and the table that will hold it, so a curve-capable
// shadow pipeline needs its own copy -- the test itself is identical, and curve
// geometry is built opaque, so this is never called on a strand.
#define WF_ANY_HIT_ENTRY(NAME, ...)                                                                  \
    [[intersection(triangle, __VA_ARGS__)]]                                                          \
    bool NAME(uint primitive_id [[primitive_id]], uint geometry_id [[geometry_id]],                  \
              uint instance_id [[instance_id]], float2 barycentric_coord [[barycentric_coord]],      \
              ray_data ShadowPayload& payload [[payload]],                                           \
              constant MTLAccelerationStructureUserIDInstanceDescriptor* instances [[buffer(0)]],    \
              device const Material* materials [[buffer(1)]],                                        \
              device const GeometryEntry* geometryEntries [[buffer(2)]],                             \
              device const char* vertexBuffer [[buffer(3)]],                                         \
              device const uint32_t* indexBuffer [[buffer(4)]])                                      \
    {                                                                                                \
        return shadowAlphaAnyHitImpl(primitive_id, geometry_id, instance_id, barycentric_coord,      \
                                     payload, instances, materials, geometryEntries, vertexBuffer,   \
                                     indexBuffer);                                                   \
    }

WF_ANY_HIT_ENTRY(shadowAlphaAnyHit, triangle_data, instancing)
WF_ANY_HIT_ENTRY(shadowAlphaAnyHitCurve, triangle_data, curve_data, instancing)

// Measured and not kept: the same stochastic alpha test as an intersection
// function on the *main* rays, so a canopy resolves in one traversal instead of
// one per leaf slipped past.
//
// It works and it is slightly more accurate on a single blended plane -- the
// harness scene went from 0.052 to 0.042 relative error -- but on the pine
// forest, which is the scene it was written for, it renders the same image
// (0.5% relative) 4% slower: 31.3 s against 30.0 s at 128 spp, twice each. An
// intersection-function callback per candidate costs more than the handful of
// traversal restarts it saves, because a ray meets few leaves, not many.
//
// The shadow case above is the opposite and stays: there the old path restarted
// a full closest-hit query up to eight times per ray.
//
// The first version of this measured as a 9% win, which it was not -- it also
// ran the coverage test a second time in `shade`, so foliage passed a^2 instead
// of a and rays escaped the canopy that should not have.

struct MotionTraversal
{
    using structure = acceleration_structure<instancing, primitive_motion>;
    using isect = intersector<triangle_data, instancing, primitive_motion>;
    using table = intersection_function_table<triangle_data, instancing, primitive_motion>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle;
    }
    static float curveParameter(thread const isect::result_type&)
    {
        return 0.0f;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float time)
    {
        return i.intersect(r, as, mask, time);
    }
    static isect::result_type traceAnyHit(thread isect& i, ray r, structure as, uint32_t mask,
                                          float time, table t, thread ShadowPayload& payload)
    {
        return i.intersect(r, as, mask, time, t, payload);
    }
};

struct StaticTraversal
{
    using structure = acceleration_structure<instancing>;
    using isect = intersector<triangle_data, instancing>;
    using table = intersection_function_table<triangle_data, instancing>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle;
    }
    static float curveParameter(thread const isect::result_type&)
    {
        return 0.0f;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float)
    {
        return i.intersect(r, as, mask);
    }
    static isect::result_type traceAnyHit(thread isect& i, ray r, structure as, uint32_t mask,
                                          float, table t, thread ShadowPayload& payload)
    {
        return i.intersect(r, as, mask, t, payload);
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
    using structure = acceleration_structure<instancing, primitive_motion>;
    using isect = intersector<triangle_data, curve_data, instancing, primitive_motion>;
    using table = intersection_function_table<triangle_data, curve_data, instancing, primitive_motion>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::curve;
    }
    static float curveParameter(thread const isect::result_type& r)
    {
        return r.curve_parameter;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float time)
    {
        return i.intersect(r, as, mask, time);
    }
    static isect::result_type traceAnyHit(thread isect& i, ray r, structure as, uint32_t mask,
                                          float time, table t, thread ShadowPayload& payload)
    {
        return i.intersect(r, as, mask, time, t, payload);
    }
};

struct CurveStaticTraversal
{
    using structure = acceleration_structure<instancing>;
    using isect = intersector<triangle_data, curve_data, instancing>;
    using table = intersection_function_table<triangle_data, curve_data, instancing>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::curve;
    }
    static float curveParameter(thread const isect::result_type& r)
    {
        return r.curve_parameter;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float)
    {
        return i.intersect(r, as, mask);
    }
    static isect::result_type traceAnyHit(thread isect& i, ray r, structure as, uint32_t mask,
                                          float, table t, thread ShadowPayload& payload)
    {
        return i.intersect(r, as, mask, t, payload);
    }
};

static inline uint32_t pathDepth(uint32_t depthAndFlags)
{
    return depthAndFlags & PATH_DEPTH_MASK;
}

// The sampler is never stored: it is a pure function of these three values, and
// recomputing it costs less than the 12 bytes it would add to every path.
static inline SamplerState samplerFor(constant Uniforms& uniforms, uint32_t pixelIndex,
                                      uint32_t sampleIdx, uint32_t depth)
{
    SamplerState s = initSampler(pixelIndex, uniforms.subframeIndex + sampleIdx, uniforms.width,
                                 uniforms.blueNoiseSwitchSpp);
    s.depth = depth;
    return s;
}

// A path has one time, sampled once when its camera ray is generated. The
// sampler is stateless, so every later stage can recover that exact value by
// drawing eTime at depth 0 again -- drawing it at the current depth would give a
// different time on every bounce and smear the path across the shutter.
static inline float motionTimeFor(constant Uniforms& uniforms, uint32_t pixelIndex, uint32_t sampleIdx)
{
    if (uniforms.canonicalGuideSample && sampleIdx == 0u)
    {
        return 1.0f;
    }
    if (!SPEC_MOTION_BLUR || !uniforms.enableMotionBlur)
    {
        return 0.0f;
    }
    SamplerState s = samplerFor(uniforms, pixelIndex, sampleIdx, 0u);
    const uint32_t firstRadianceSample = uniforms.canonicalGuideSample ? 1u : 0u;
    const uint32_t radianceSample = sampleIdx - firstRadianceSample;
    const uint32_t sampleCount = max(uniforms.samples_per_launch, 1u);
    const float t =
        ((float)radianceSample + random<SampleDimension::eTime>(s, uniforms.samplerType)) /
        (float)sampleCount;
    return uniforms.isMotionBlurVisible ? t : 1.0f;
}

static inline bool shouldWriteAov(constant Uniforms& uniforms, uint32_t sampleIdx)
{
    return uniforms.writeAov &&
           (!uniforms.canonicalGuideSample || sampleIdx == 0u);
}

// Measured and not kept: adaptive sampling -- retiring a pixel once its estimate
// stops moving, so that nothing downstream traces for it.
//
// It works, it is not worth it, and the reasons are structural rather than a
// matter of tuning. Pine forest, 640x480, against a 512 spp reference:
//
//   128 spp uniform    33.0 s   rel 0.0787  rmse 0.01404  median 0.994
//   116 spp uniform    29.4 s   rel 0.0853  rmse 0.01516  median 0.992
//   adaptive 0.002     30.0 s   rel 0.0876  rmse 0.01461  median 0.955
//   100 spp uniform    28.4 s   rel 0.0932  rmse 0.01654  median 0.992
//   adaptive 0.005     26.5 s   rel 0.0950  rmse 0.01515  median 0.854
//
// At equal wall clock it loses to simply asking for fewer samples, and it pays
// for that with a bias the uniform render does not have.
//
// Three things go wrong, and each is worth knowing before trying again.
//
// A pixel that converges cheaply is also cheap to sample. The sky retires first,
// and a sky ray misses almost immediately -- with a criterion relative to the
// pixel's own mean only a tenth of the frame ever retired, all of it sky: the
// camera-ray count fell from 307k to 275k while the bounce-1 count did not move
// at all. Nothing expensive stops.
//
// Retiring anything expensive needs an absolute criterion, and an absolute
// criterion retires dark pixels. Radiance is non-negative and heavy-tailed, so a
// dark pixel that has not yet found its rare bright path looks converged
// precisely because it has not found it. Freezing it there is a systematic
// darkening -- 15% at the median at a threshold of 0.005, far more than the
// noise it saved.
//
// And retirement needs a compacted queue, which this architecture charges for.
// `generate` writes the identity, and every later stage reads PathRay,
// HitRecord and PathState in pixel order because of it. Compacting a queue that
// had nothing to compact -- the machinery enabled, the threshold zero -- cost
// 15% of the frame on its own, the same coherence effect that made ray sorting
// lose.
//
// A tile-level version answers the second and third of those: a tile mean is
// estimated well enough not to freeze low, and whole tiles keep the queue
// ordered. It does not answer the first, which is the one that bounds the win.
//
// ---------------------------------------------------------------------------
// generate -- camera rays
// ---------------------------------------------------------------------------
kernel void wavefrontGenerate(
    uint                                                       tid            [[thread_position_in_grid]],
    constant Uniforms&                                         uniforms       [[buffer(0)]],
    device PathState*                                          paths          [[buffer(1)]],
    device PathRay*                                            rays           [[buffer(8)]],
    device float4*                                             radianceOut    [[buffer(2)]],
    device IorStack*                                           iorStacks      [[buffer(3)]],
    constant uint32_t&                                         sampleIdx      [[buffer(4)]],
    device uint32_t*                                           queueOut       [[buffer(5)]],
    device uint32_t*                                           control        [[buffer(6)]],
    device AovSample*                                          aov            [[buffer(7)]],
    // Zeroed here rather than on the host: `generate` already owns resetting the
    // per-sample counters, and a host-side clear would race the frame in flight.
    // The tally is therefore per sample, which is the rate rather than a total.
    device uint32_t*                                           iorStats       [[buffer(9)]])
{
    const uint32_t pixelCount = uniforms.width * uniforms.height;
    if (tid == 0u)
    {
        control[WF_CTRL_COUNT0] = pixelCount;
        control[WF_CTRL_COUNT1] = 0u;
        control[WF_CTRL_SHADOW] = 0u;
        control[WF_CTRL_HIT] = 0u;
        control[WF_CTRL_MISS] = 0u;
        iorStats[IOR_STAT_OVERFLOW] = 0u;
        iorStats[IOR_STAT_UNMATCHED] = 0u;
        iorStats[IOR_STAT_ESCAPED_INSIDE] = 0u;
    }
    if (tid >= pixelCount)
    {
        return;
    }
    // Every camera ray starts alive, so the first queue is the identity and
    // needs no compaction.
    queueOut[tid] = tid;

    const uint32_t firstRadianceSample = uniforms.canonicalGuideSample ? 1u : 0u;
    if (sampleIdx == firstRadianceSample)
    {
        radianceOut[tid] = float4(0.0f);
    }

    const uint2 pixel = uint2(tid % uniforms.width, tid / uniforms.width);
    SamplerState rng = samplerFor(uniforms, tid, sampleIdx, 0u);
    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);

    float3 origin, direction;
    generateCameraRay(pixel, rng, origin, direction, uniforms, motionTime);
    if (uniforms.canonicalGuideSample && sampleIdx == 0u)
    {
        AovSample a;
        a.diffuseAlbedo = packed_float3(float3(0.0f));
        a.specularAlbedo = packed_float3(float3(0.0f));
        a.normal = packed_float3(-direction);
        a.roughness = 1.0f;
        a.depth =
            uniforms.denoiseDepthMode == kDenoiseDepthDevice ? 0.0f : 1e7f;
        a.motionX = 0.0f;
        a.motionY = 0.0f;
        a.specularHitDistance = 0.0f;
        a.reactive = 1.0f;
        a.pad2 = 0.0f;
        aov[tid] = a;
    }

    PathRay r;
    r.origin = packed_float3(origin);
    r.direction = packed_float3(direction);
    rays[tid] = r;

    PathState p;
    p.throughput = packed_float3(float3(1.0f));
    p.depthAndFlags = PATH_FLAG_ALIVE; // depth 0, not specular, NEE not done
    p.lastBsdfPdf = 0.0f;
    p.misDistance = 0.0f;
    p.sharcIndex = SHARC_NO_ENTRY;
    p.sharcRadianceAtVisit = packed_float3(0.0f);
    p.sharcInvThroughput = packed_float3(0.0f);
    // Outside every medium. A camera that starts inside a translucent object is
    // not handled -- there is nothing to tell the path which medium it is in.
    p.medium = 0u;
    p.mediumAlbedo = 0u;
    paths[tid] = p;

    // ior_stack_* take a thread reference; device memory cannot bind to one.
    IorStack stack;
    ior_stack_init(stack);
    iorStacks[tid] = stack;
}

// ---------------------------------------------------------------------------
// extend -- closest hit
// ---------------------------------------------------------------------------
template <typename T>
static void extendImpl(
    uint gid,
    constant Uniforms&                                         uniforms,
    constant MTLAccelerationStructureUserIDInstanceDescriptor* instances,
    typename T::structure accelerationStructure,
    device const PathRay*                                      rays,
    device HitRecord*                                          hits,
    constant uint32_t&                                         sampleIdx,
    device const uint32_t*                                     queue,
    device const uint32_t*                                     control,
    device uint32_t*                                           hitQueue,
    device atomic_uint*                                        hitCounter,
    device uint32_t*                                           missQueue,
    device atomic_uint*                                        missCounter,
    // Only for the fog: the path's depth decorrelates the free-flight draw
    // across bounces, and without it every bounce of a path scatters at the same
    // fraction of its segment, which shows up as banding in the haze.
    device const PathState*                                    paths,
    // Only for the subsurface walk, which needs the medium's mean free path to
    // sample a free flight and reads it from the material the path is inside.
    device const Material*                                     materials,
    // Chosen per dispatch rather than per ray: the only thing it distinguishes
    // is the camera bounce from the rest, and `extend` is encoded once per
    // bounce anyway. Reading the path's depth here to answer the same question
    // would put a load in the hottest kernel in the renderer.
    uint32_t                                                   rayMask)
{
    // Indirect dispatch can only launch whole threadgroups, so the tail of the
    // last one runs past the queue and has to be discarded here.
    if (gid >= control[WF_CTRL_ACTIVE])
    {
        return;
    }
    const uint32_t tid = queue[gid];
    const PathRay pr = rays[tid];

    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);

    ray r;
    r.min_distance = 0.0f;
    r.max_distance = INFINITY;
    r.origin = float3(pr.origin);
    r.direction = float3(pr.direction);

    // Measured and not kept: `intersection_query`, Metal's inline traversal, in
    // place of the intersector object.
    //
    // The hardware counters say this kernel runs at 16% compute occupancy where
    // a saturating ALU kernel reaches 88%, and the hope was that stepping
    // traversal in the shader would ask for fewer registers and let more waves
    // stay resident. It does not: occupancy comes back 15.2% against 16.9%, and
    // the frame takes the same 174 ms. Occupancy is pinned by something below
    // this API -- most likely how many traversals the ray tracing unit will
    // carry at once -- and neither threadgroup size (64, 128, 256 all give
    // 16.9%) nor the API form moves it.
    //
    // It does move the *limiters*: MMU 44% -> 29% and last level cache 29% ->
    // 20%, with ALU going 34% -> 39%. Same work, differently constrained, same
    // wall clock. Worth knowing if the memory picture ever changes.
    //
    // It also cannot serve the whole renderer: `intersection_query` rejects the
    // primitive_motion tag, so a deforming scene would need the intersector kept
    // alongside it.
    typename T::isect isect;
    isect.assume_geometry_type(T::geometryTypes());
    // The coverage test for cutout geometry happens in `shade`, not here -- see
    // the note above extendAlphaAnyHit's replacement for the measurement.
    isect.force_opacity(forced_opacity::opaque);
    isect.accept_any_intersection(false);

    const typename T::isect::result_type hit =
        T::trace(isect, r, accelerationStructure, rayMask, motionTime);

    // Chased through the ray buffer -- 22 MB at 720p, past the caches -- because
    // the queue this used to walk is 3.7 MB and fits in them, which made the
    // probe report on cache hits rather than on memory.

    // A ray that escaped carries no information beyond the fact, so it goes
    // straight to the miss stage: no hit record is written and the path never
    // enters `shade`. On a scene with an open background that is most of the
    // secondary rays, and it was the whole cost of the bounce.
    // Atmospheric scattering, decided here because this is the one kernel that
    // knows both where the ray ended and whether it ended at all. A ray that
    // scatters never reaches its surface, and one that escapes can still scatter
    // on the way out -- so the miss branch is inside this test, not before it.
    // Everything below this line is on the wrong side of a register cliff, and
    // the cliff is the only lever the hardware counters leave.
    //
    // This kernel runs at 17% compute occupancy where a saturating ALU kernel
    // reaches 88%. The machine is empty, not busy: relieving memory pressure
    // without raising occupancy buys nothing at all -- Metal's inline
    // `intersection_query` cut the MMU limiter from 44% to 29% and the last
    // level cache from 29% to 20% for exactly the same 174 ms frame. What does
    // move is occupancy, and occupancy moves with registers: compiling the fog
    // out lifts the pipeline's threadgroup limit from 640 to 704, occupancy from
    // 16.5% to 18.8%, and the frame from ~175 ms to ~167 (three interleaved
    // runs each, swapping metallibs to cancel thermal drift).
    //
    // Reaching 704 with the fog still in was tried and cannot be done piecemeal.
    // Every one of these on its own leaves it at 640, and only removing the
    // whole block reaches 704:
    //   - writing the fog HitRecord field by field instead of through a local
    //   - calling randomSobol directly, so the runtime samplerType switch does
    //     not inline all five samplers here
    //   - merging the fog and surface exits into a single hit-queue push
    //   - carrying the free-flight draw on PathRay, taken where the ray was
    //     created, so no SamplerState is built here at all -- and that one is
    //     self-defeating: the extra word takes PathRay from 24 bytes to 28 and
    //     lives across the traversal, which costs what it saved
    //
    // The remaining route is a separate dispatch for the fog decision, which
    // buys at most those 4.5% and pays a barrier and a drain per bounce -- and
    // this renderer has already measured that dispatch boundaries are not free
    // (see the acceleration structure batching note). Not attempted.
    //
    // Inside a subsurface medium the path is inside a solid object, so the
    // atmosphere does not apply and this runs instead of the fog test rather
    // than alongside it.
    if (SPEC_SSS)
    {
        const uint32_t sss = paths[tid].medium;
        const uint32_t medium = sss & MEDIUM_INDEX_MASK;
        if (medium != 0u)
        {
            const float surfaceT = (hit.type == intersection_type::none) ? 1e16f : hit.distance;
            const float3 sigmaT = sssSigmaT(float3(materials[medium - 1u].subsurface_radius));
            // The walk step, not the path depth, indexes the sampler: a walk can
            // take hundreds of steps at one depth, and rebuilding the sampler in
            // the same state at each of them would have every step draw the same
            // distance and turn the same way.
            const uint32_t step = sss >> MEDIUM_STEP_SHIFT;
            SamplerState srng = samplerFor(uniforms, tid, sampleIdx,
                                           pathDepth(paths[tid].depthAndFlags) + step);
            float scatterT = 0.0f;
            if (step < MEDIUM_MAX_STEPS &&
                sssSampleDistance(sigmaT, surfaceT,
                                  random<SampleDimension::eSssChannel>(srng, uniforms.samplerType),
                                  random<SampleDimension::eSssDistance>(srng, uniforms.samplerType),
                                  scatterT))
            {
                HitRecord sssRec;
                sssRec.geomEntryIndex = HIT_SSS_BIT;
                sssRec.instanceIndex = 0u;
                sssRec.primitiveId = 0u;
                sssRec.barycentrics = vector_float2(0.0f, 0.0f);
                sssRec.distance = scatterT;
                hits[tid] = sssRec;
                queuePush(hitCounter, hitQueue, tid);
                return;
            }
        }
    }

    // A scene without fog already gets 704: SPEC_FOG is a function constant.
    if (SPEC_FOG && uniforms.hasFog && !(SPEC_SSS && (paths[tid].medium & MEDIUM_INDEX_MASK) != 0u))
    {
        const float surfaceT =
            (hit.type == intersection_type::none) ? 1e16f : hit.distance;
        SamplerState frng =
            samplerFor(uniforms, tid, sampleIdx, pathDepth(paths[tid].depthAndFlags));
        float scatterT = 0.0f;
        if (fogSampleDistance(r.origin, r.direction, surfaceT, uniforms.fogHeight, uniforms.fogSigmaT,
                              random<SampleDimension::eFogDistance>(frng, uniforms.samplerType),
                              scatterT))
        {
            HitRecord fogRec;
            fogRec.geomEntryIndex = HIT_FOG_BIT;
            fogRec.instanceIndex = 0u;
            fogRec.primitiveId = 0u;
            fogRec.barycentrics = vector_float2(0.0f, 0.0f);
            fogRec.distance = scatterT;
            hits[tid] = fogRec;
            queuePush(hitCounter, hitQueue, tid);
            return;
        }
    }

    if (hit.type == intersection_type::none)
    {
        queuePush(missCounter, missQueue, tid);
        return;
    }

    const auto inst = instances[hit.instance_id];
    const bool isLight = (inst.mask == GEOMETRY_MASK_LIGHT || inst.mask == GEOMETRY_MASK_LIGHT_HIDDEN);
    // For emissive geometry userID indexes the light table, not the geometry
    // table; the flag bit tells `shade` which one it is.
    HitRecord rec;
    rec.geomEntryIndex = isLight ? (HIT_LIGHT_BIT | inst.userID)
                                 : (inst.userID + hit.geometry_id);
    rec.instanceIndex = hit.instance_id;
    rec.primitiveId = hit.primitive_id;
    // A curve hit has no barycentrics; what it has is one parameter along the
    // segment. It rides in the same two floats rather than in a field of its own,
    // because `shade` already has to read the geometry entry to find the
    // material and the entry says which kind of primitive this is.
    rec.barycentrics = (hit.type == intersection_type::curve)
                           ? vector_float2(T::curveParameter(hit), 0.0f)
                           : hit.triangle_barycentric_coord;
    rec.distance = hit.distance;
    hits[tid] = rec;
    queuePush(hitCounter, hitQueue, tid);
}


#define WF_EXTEND_ENTRY(NAME, TRAITS)                                                                       \
    kernel void NAME(uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]],       \
                     constant MTLAccelerationStructureUserIDInstanceDescriptor* instances [[buffer(1)]],    \
                     TRAITS::structure accelerationStructure [[buffer(2)]],                                 \
                     device const PathRay* rays [[buffer(3)]], device HitRecord* hits [[buffer(4)]],        \
                     constant uint32_t& sampleIdx [[buffer(5)]],                                            \
                     device const uint32_t* queue [[buffer(6)]],                                            \
                     device const uint32_t* control [[buffer(7)]],                                          \
                     device uint32_t* hitQueue [[buffer(8)]],                                               \
                     device atomic_uint* hitCounter [[buffer(9)]],                                          \
                     device uint32_t* missQueue [[buffer(10)]],                                             \
                     device atomic_uint* missCounter [[buffer(11)]],                                        \
                     device const PathState* paths [[buffer(12)]],                                          \
                     device const Material* materials [[buffer(13)]],                                       \
                     constant uint32_t& rayMask [[buffer(14)]])                                            \
    {                                                                                                       \
        extendImpl<TRAITS>(gid, uniforms, instances, accelerationStructure, rays, hits, sampleIdx, queue,    \
                           control, hitQueue, hitCounter, missQueue, missCounter, paths, materials,       \
                           rayMask);                                                                        \
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
                          thread float3* p, thread float3* n, thread float3* t, thread float2* uv,
                          thread float& tangentSign, thread float3* vcol)
{
    // Scene::Vertex layout (32 bytes): pos@0 (packed_float3), tangent@12,
    // normal@16, uv@20 — all uint32 after the position. The `Vertex` struct in
    // ShaderTypes.h does *not* match this and must not be used.
    constexpr uint32_t vtxStride  = 32;
    constexpr uint32_t tangentOff = 12;
    constexpr uint32_t normalOff  = 16;
    constexpr uint32_t uvOff      = 20;
    constexpr uint32_t colorOff   = 28;   // Scene::Vertex::color, packed RGBA8

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
// The radius buffer is not read here at all -- it is what the intersector
// already used to find the hit, and nothing downstream asks how thick the
// strand was.
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
                       thread float2& outUv)
{
    const bool cubic = (entry.flags & GEOM_CURVE_CUBIC) != 0u;
    const uint32_t base = curveSegments[entry.indexOffset + primitiveId];
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
    outNormal = lenSq > 1e-16f ? perp * rsqrt(lenSq)
                               : normalize(cross(outTangent, float3(0.0f, 0.0f, 1.0f)));

    const uint32_t perStrand = entry.flags & GEOM_CURVE_STRAND_MASK;
    const float alongStrand =
        perStrand != 0u ? ((float)(primitiveId % perStrand) + u) / (float)perStrand : 0.0f;
    outUv = float2(alongStrand, 0.0f);
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
                                 thread float3& outNormal, thread float3& outTangent,
                                 thread float2& outUv, thread float3& outColor,
                                 thread float& tangentSign, thread float3& outGeomNormal,
                                 // For the ray-cone texture LOD: the two object-space
                                 // edges and twice the triangle's area in uv. Both fall
                                 // out of loads this function already does, so the
                                 // footprint costs no extra memory traffic.
                                 thread float3& outEdge1, thread float3& outEdge2,
                                 thread float& outUvArea2)
{
    constexpr uint32_t vtxStride  = 32;
    constexpr uint32_t tangentOff = 12;
    constexpr uint32_t normalOff  = 16;
    constexpr uint32_t uvOff      = 20;
    constexpr uint32_t colorOff   = 28;

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
        if (k == 0) uv0 = vertUv;
        else if (k == 1) uvE1 = vertUv - uv0;
        else uvE2 = vertUv - uv0;
        outColor += unpackVertexColor(*(device const uint32_t*)(v + colorOff)) * weight[k];

        if (k == 0) p0 = pos;
        else if (k == 1) e1 = pos - p0;
        else e2 = pos - p0;
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
static inline float3 previousWorldPosition(
    device const char* prevFrameVertexBuffer,
    device const uint32_t* indexBuffer,
    constant MTLAccelerationStructureUserIDInstanceDescriptor* prevInstances,
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
        p[k] = float3(*(device const packed_float3*)(prevFrameVertexBuffer +
                                                     (entry.vbOffset + idx) * vtxStride));
    }
    const float3 objectPos = interpolateAttrib(p[0], p[1], p[2], bary);

    const auto inst = prevInstances[instanceIndex];
    const float4x4 prevObjectToWorld = float4x4(
        float4(float3(inst.transformationMatrix[0]), 0.0f),
        float4(float3(inst.transformationMatrix[1]), 0.0f),
        float4(float3(inst.transformationMatrix[2]), 0.0f),
        float4(float3(inst.transformationMatrix[3]), 1.0f));
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
        const float3 forward =
            -float3(uniforms.viewToWorld[0][2], uniforms.viewToWorld[1][2], uniforms.viewToWorld[2][2]);
        return dot(worldPosition - eye, forward);
    }
    return length(worldPosition - eye);
}

// The value the background writes. Device depth has a finite far plane, so the
// sentinel has to match the convention or the denoiser reads the sky as being
// nearer than the geometry.
static inline float backgroundDepth(constant Uniforms& uniforms)
{
    return uniforms.denoiseDepthMode == kDenoiseDepthDevice ? 0.0f : 1e7f;
}

// Where a point was on screen last frame, in pixels, y down. That is the sign
// MetalFX documents: "the motion vectors for an object that moves down and to
// the right by 10 pixels would be (-10,-10)".
//
// The current position is the *jittered* sample position, not the pixel centre.
// The ray that produced this hit went through the jitter offset, so projecting
// the hit back through the current camera lands there, not at the centre.
// Differencing against the centre instead leaves the jitter inside every motion
// vector -- a subpixel wobble on every pixel of a perfectly still image, which is
// the one thing a temporal reconstruction must not be told, because MetalFX
// already accounts for the jitter itself through jitterOffsetX/Y.
static inline float2 screenMotion(constant Uniforms& uniforms, float4 prevClip, uint2 pixel)
{
    // A w at or near zero is a point on the previous camera's plane, and dividing
    // by it does not produce a large motion vector -- it produces a meaningless
    // one. Measured on the pine forest before this guard: 14% of pixels claimed
    // more than ten pixels of motion with the camera standing still, and the
    // worst saturated the half-float motion texture at 65504 in both channels.
    // The denoiser then fetched history from those coordinates, which is a
    // temporal filter that cannot converge by construction.
    //
    // Zero is the honest answer for a reprojection that has none: it says "this
    // pixel did not move", the history it blends is the one under the pixel, and
    // every caller that can reach this case already marks the pixel reactive.
    const float kMinW = 1e-4f;
    if (prevClip.w <= kMinW)
    {
        return float2(0.0f);
    }
    const float2 prevNdc = prevClip.xy / prevClip.w;
    const float2 prevPixel = float2((prevNdc.x * 0.5f + 0.5f) * (float)uniforms.width,
                                    (1.0f - (prevNdc.y * 0.5f + 0.5f)) * (float)uniforms.height);
    // generateCameraRay builds pixelPos.y as height - (y + 0.5 + jitterY), so the
    // flip cancels and the sample sits at row y + 0.5 + jitterY in screen space.
    const float2 currPixel = float2((float)pixel.x + 0.5f + uniforms.jitterX,
                                    (float)pixel.y + 0.5f + uniforms.jitterY);
    const float2 motion = prevPixel - currPixel;
    // Nothing that moved further than the frame is across in one frame can be
    // reprojected onto anything: past that the history lookup lands outside the
    // image, and the value is far more likely to be a reprojection artefact than
    // a real displacement. Clamped rather than zeroed so a genuinely fast object
    // still drags its history in the right direction.
    const float limit = (float)(uniforms.width + uniforms.height);
    return clamp(motion, -limit, limit);
}

// ---------------------------------------------------------------------------
// miss -- rays that escaped the scene
//
// Split out of `shade` rather than branched inside it. The work is a texture
// fetch and one MIS weight, and keeping it here means `shade` neither dispatches
// threads for escaped rays nor carries the environment sampler in its register
// budget.
// ---------------------------------------------------------------------------
kernel void wavefrontMiss(
    uint                    gid           [[thread_position_in_grid]],
    constant Uniforms&      uniforms      [[buffer(0)]],
    device const PathState* paths         [[buffer(1)]],
    device const PathRay*   rays          [[buffer(2)]],
    device float4*          radianceOut   [[buffer(3)]],
    device const uint32_t*  queue         [[buffer(4)]],
    device const uint32_t*  control       [[buffer(5)]],
    device AovSample*       aov           [[buffer(6)]],
    constant uint32_t&      sampleIdx     [[buffer(7)]],
    // The escape counter's other half. A path that reaches infinity while its
    // dielectric stack still holds something left a volume without crossing its
    // surface -- which is exactly what a hole in a refracting mesh does, and the
    // failure `shade` cannot see because no exit event ever happens.
    device const IorStack*  iorStacks     [[buffer(8)]],
    device atomic_uint*     iorStats      [[buffer(9)]],
    texture2d<float>        envMapTexture [[texture(0)]],
    texture2d<float>        envBackgroundTexture [[texture(1)]])
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
    const bool specularBounce = (p.depthAndFlags & PATH_FLAG_SPECULAR) != 0u;
    const bool neeDone = (p.depthAndFlags & PATH_FLAG_NEE_DONE) != 0u;

    // Counted here because here is the only place it is visible: the path is
    // gone and it still thinks it is inside glass. See ior_stack.h.
    if (iorStacks[tid].top >= 0)
    {
        atomic_fetch_add_explicit(&iorStats[IOR_STAT_ESCAPED_INSIDE], 1u, memory_order_relaxed);
    }

    // Background still needs a record, or the denoiser reads whatever the
    // previous frame left in the guides and smears the silhouette. Not only at
    // depth 0: a specular primary hit defers its guides, so if the reflected ray
    // is the one that escapes, this is the only chance to write them.
    if (shouldWriteAov(uniforms, sampleIdx) &&
        (depth == 0u || (p.depthAndFlags & PATH_FLAG_AOV_DONE) == 0u))
    {
        AovSample a;
        a.diffuseAlbedo = packed_float3(float3(0.0f));
        a.specularAlbedo = packed_float3(float3(0.0f));
        a.normal = packed_float3(-rayDir);
        a.roughness = 1.0f;
        a.depth = backgroundDepth(uniforms);
        // The sky moves on screen when the camera turns, and leaving this at zero
        // tells the denoiser it did not: the history is then blended from the
        // wrong place across the whole background. A direction reprojects like a
        // point at infinity -- w = 0 -- so the previous camera is all that is
        // needed, and no depth.
        // Only for a camera ray. Past a bounce, `rayDir` is the direction the
        // path left the surface in, and projecting it through the previous
        // camera answers "where would something infinitely far in that direction
        // have been on screen" -- a question about the bounce, not about this
        // pixel. Those directions point everywhere, including nearly across the
        // view, where the projection is degenerate; that is where most of the
        // absurd motion vectors came from.
        const float2 motion =
            (depth == 0u) ? screenMotion(uniforms, uniforms.prevWorldToClip * float4(rayDir, 0.0f),
                                         uint2(tid % uniforms.width, tid / uniforms.width))
                          : float2(0.0f);
        a.specularHitDistance = 0.0f;
        // Sky seen through a mirror moves with the reflection, not with the
        // reflector, so its history is not reliable either.
        a.reactive = (depth > 0u) ? 1.0f : 0.0f;
        a.pad2 = 0.0f;
        if (depth == 0u)
        {
            a.motionX = motion.x;
            a.motionY = motion.y;
            aov[tid] = a;
        }
        else
        {
            // Depth and motion belong to the surface the camera sees, and this is
            // not it -- the path got here through a bounce. Leaving the primary
            // hit's values in place keeps the denoiser reprojecting this pixel by
            // this pixel's own motion; overwriting them with the sky's asked it to
            // reproject by something several hundred pixels away.
            a.depth = aov[tid].depth;
            a.motionX = aov[tid].motionX;
            a.motionY = aov[tid].motionY;
            aov[tid] = a;
        }
    }

    float3 radiance = float3(0.0f);
    if (SPEC_ENV_MAP && uniforms.hasEnvMap)
    {
        constexpr sampler envSampler(mag_filter::linear, min_filter::linear, address::repeat, coord::normalized);
        const float2 envUV = dirToEnvUV(rayDir, uniforms.envMapRotation);
        float3 envColor = envMapTexture.sample(envSampler, envUV).xyz;
        envColor *= uniforms.envMapIntensity * float3(uniforms.envMapColorTint);

        if (depth == 0u || specularBounce || !neeDone)
        {
            // A ray still at depth 0 has scattered off nothing -- it may have
            // passed through cutout foliage, which Cycles also counts as a
            // camera ray -- so this is exactly where the backdrop belongs, and
            // the MIS branch below is left reading the lighting environment
            // because that is the one that was importance sampled.
            if (uniforms.hasEnvBackground && depth == 0u)
            {
                envColor = envBackgroundTexture.sample(envSampler, envUV).xyz *
                           uniforms.envBackgroundIntensity * float3(uniforms.envMapColorTint);
            }
            radiance += throughput * envColor;
        }
        else
        {
            const float envPdf = envMapPdf(rayDir, envMapTexture,
                                           uniforms.envMapWidth, uniforms.envMapHeight,
                                           uniforms.envMapRotation, uniforms.envPdfScale);
            const float envSelectionPdf = (uniforms.numLights > 0) ? 0.5f : 1.0f;
            const float effectiveEnvPdf = envPdf * envSelectionPdf;
            // A texel of zero luminance has zero sampling density, so light
            // sampling could never have produced this direction and the BSDF
            // strategy owns it outright. Dropping the contribution instead --
            // which the guard used to do -- loses energy exactly along the edges
            // of dark regions, where the bilinear radiance is still non-zero.
            radiance += throughput * envColor *
                        (effectiveEnvPdf > 0.0f ? misWeightBalance(p.lastBsdfPdf, effectiveEnvPdf) : 1.0f);
        }
    }
    else
    {
        radiance += throughput * uniforms.missColor;
    }
    radianceOut[tid] += float4(clampIndirectContribution(radiance, depth, uniforms.clampIndirect), 0.0f);
}

// ---------------------------------------------------------------------------
// shade -- material evaluation, next-event estimation, next ray
// ---------------------------------------------------------------------------
kernel void wavefrontShade(
    uint                                                       gid            [[thread_position_in_grid]],
    constant Uniforms&                                         uniforms       [[buffer(0)]],
    constant MTLAccelerationStructureUserIDInstanceDescriptor* instances      [[buffer(1)]],
    device UniformLight*                                       lights         [[buffer(3)]],
    device Material*                                           materials      [[buffer(4)]],
    device PathState*                                          paths          [[buffer(5)]],
    device PathRay*                                            rays           [[buffer(21)]],
    device const HitRecord*                                    hits           [[buffer(6)]],
    device float4*                                             radianceOut    [[buffer(7)]],
    device IorStack*                                           iorStacks      [[buffer(8)]],
    device const GeometryEntry*                                geometryEntries[[buffer(9)]],
    device const EnvAliasEntry*                                envAliasTable  [[buffer(10)]],
    device const char*                                         vertexBuffer   [[buffer(11)]],
    device const char*                                         prevVertexBuffer [[buffer(12)]],
    device const uint32_t*                                     indexBuffer    [[buffer(13)]],
    constant uint32_t&                                         sampleIdx      [[buffer(14)]],
    device const uint32_t*                                     queue          [[buffer(15)]],
    device uint32_t*                                           queueOut       [[buffer(16)]],
    device atomic_uint*                                        outCounter     [[buffer(17)]],
    device const uint32_t*                                     control        [[buffer(18)]],
    device ShadowRay*                                          shadowRays     [[buffer(19)]],
    device atomic_uint*                                        shadowCounter  [[buffer(20)]],
    device AovSample*                                          aov            [[buffer(22)]],
    // The previous frame's pose, for motion vectors. Separate from
    // prevVertexBuffer, which is a motion-blur shutter keyframe and is forced
    // equal to the current pose whenever motion blur is off.
    device const char*                                         prevFrameVertexBuffer [[buffer(23)]],
    constant MTLAccelerationStructureUserIDInstanceDescriptor* prevInstances  [[buffer(24)]],
    device SharcEntry*                                         sharcEntries   [[buffer(25)]],
    // Curves. `extend` cannot hand over what it saw -- primitive_data is only
    // addressable inside the kernel that ran the intersect -- so a strand hit is
    // rebuilt here from the same buffers the acceleration structure was built
    // from, exactly as a triangle hit is refetched from the vertex buffer.
    device const packed_float3*                                curvePoints    [[buffer(26)]],
    device const uint32_t*                                     curveSegments  [[buffer(27)]],
    // Two counters for the ways the nested-dielectric stack loses a path; see
    // ShaderTypes.h. Written only when one of them has already gone wrong.
    device atomic_uint*                                        iorStats       [[buffer(28)]],
    texture2d<float>                                           envMapTexture  [[texture(0)]])
{
    if (gid >= control[WF_CTRL_HIT_N])
    {
        return;
    }
    const uint32_t tid = queue[gid];
    PathState p = paths[tid];
    const PathRay pr = rays[tid];

    const uint32_t depth = pathDepth(p.depthAndFlags);
    const bool specularBounce = (p.depthAndFlags & PATH_FLAG_SPECULAR) != 0u;
    const bool neeDone = (p.depthAndFlags & PATH_FLAG_NEE_DONE) != 0u;

    SamplerState rng = samplerFor(uniforms, tid, sampleIdx, depth);
    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);

    const float3 rayOrigin = float3(pr.origin);
    const float3 rayDir = float3(pr.direction);
    float3 throughput = float3(p.throughput);

    float3 radiance = float3(0.0f);
    const HitRecord rec = hits[tid];

    // --- Absorption over the segment just travelled -------------------------
    //
    // The IOR stack already knows which medium the path is inside; it also
    // carries which material that medium came from, so the extinction can be
    // looked up here rather than threaded through the path state.
    //
    // Once, here, rather than inside each of the four branches below -- because
    // three of them return before the fourth is reached, and a branch that
    // forgets this does not lose a highlight, it loses the colour of everything
    // seen through the medium.
    //
    // That is what the fog gizmo around the bathtub was doing to the water. The
    // gizmo is dense enough that nearly every ray inside the water scatters in
    // it before reaching a surface, a scattering event returns from its own
    // branch, and the water's absorption over the segment leading to that event
    // was applied by nobody. It read as the fog washing the cyan out, which is
    // why four orders of magnitude of fog density barely moved it: the fog was
    // not tinting anything, it was replacing the vertex that would have.
    //
    // rec.distance is the segment in all four cases -- a scattering event
    // carries the free-flight distance it sampled, a surface hit carries the
    // hit distance.
    //
    // Skipped inside a subsurface walk, which is a different medium model:
    // sssScatterWeight already carries that medium's extinction, and a material
    // carrying both extensions would otherwise be attenuated twice for one
    // interior. A bounded volume is not skipped -- being inside a fog gizmo says
    // nothing about whether the path is also inside glass.
    {
        const uint32_t walk = p.medium & MEDIUM_INDEX_MASK;
        const bool inSubsurfaceWalk =
            SPEC_SSS && walk != 0u &&
            (materials[walk - 1u].medium_flags & MEDIUM_FLAG_BOUNDARY) == 0u;
        IorStack preStack = iorStacks[tid];
        const uint32_t inside = ior_stack_current_material(preStack);
        if (!inSubsurfaceWalk && inside != 0xFFFFFFFFu)
        {
            device const Material& im = materials[inside];
            const float3 sigma_t = volume_extinction(float3(im.attenuation_color),
                                                     im.attenuation_distance, uniforms.volumeModel);
            throughput *= beer_lambert_transmittance(sigma_t, rec.distance);
            // Every branch below that persists the path either recomputes this
            // or writes `p` unchanged, so it is written once here.
            p.throughput = packed_float3(throughput);
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

        // A medium event has a position and no normal, which is what the
        // volumeEvent flag tells the light connection.
        SurfaceInteraction si = {};
        si.position = scatterPoint;
        si.shading_normal = -rayDir;
        si.geometry_normal = -rayDir;
        si.front_face = true;

        bool didNee = false;
        if (SPEC_LIGHTS || (SPEC_ENV_MAP && uniforms.hasEnvMap))
        {
            const LightConnection conn =
                connectToLight(uniforms, uniforms.numLights, lights, rng, si, envAliasTable,
                               envMapTexture, true);
            if (conn.needsRay && conn.pdf > 0.0f)
            {
                // dot(rayDir, toLight), not dot(-rayDir, toLight). The phase
                // function takes the angle between the two directions of
                // *travel*: light arrives along -toLight and leaves toward the
                // camera along -rayDir, so their cosine is dot(rayDir, toLight).
                //
                // Negated, a forward-scattering medium becomes a backward-
                // scattering one. Looking into a low sun -- where the glow is --
                // it was six times too dim, and looking away from it, where
                // there should be almost nothing, thirty-seven times too bright.
                const float phase = hgPhase(dot(rayDir, conn.toLight), uniforms.fogAnisotropy);
                // The phase function is the medium's BSDF and its own pdf, so
                // MIS pairs it against the light density exactly as a surface
                // lobe would.
                const float misWeight =
                    conn.isDelta ? 1.0f : misWeightBalance(conn.pdf, phase);
                const float3 weight = throughput * (conn.radiance / conn.pdf) * misWeight * phase;
                if (any(weight > 1e-6f))
                {
                    const uint32_t slot =
                        atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
                    ShadowRay sr;
                    sr.origin = packed_float3(scatterPoint);
                    sr.direction = packed_float3(conn.toLight);
                    sr.weight = packed_float3(clampIndirectContribution(weight, depth, uniforms.clampIndirect));
                    sr.maxDistance = conn.tMax;
                    sr.pixelIndex = tid;
                    sr.medium = p.medium & MEDIUM_INDEX_MASK;
                    sr.rrCutoff = random<SampleDimension::eShadowRR>(rng, uniforms.samplerType) *
                                  kShadowTransmittanceCutoff;
                    shadowRays[slot] = sr;
                    didNee = true;
                }
            }
        }

        float phasePdf = 0.0f;
        const float3 nextDir =
            hgSample(-rayDir, uniforms.fogAnisotropy,
                     random<SampleDimension::eFogPhaseU>(rng, uniforms.samplerType),
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

        PathRay nextRay;
        nextRay.origin = packed_float3(scatterPoint);
        nextRay.direction = packed_float3(nextDir);
        rays[tid] = nextRay;

        p.throughput = packed_float3(throughput);
        p.lastBsdfPdf = phasePdf;
        p.misDistance = 0.0f;

        // Depth advances: a scattering event is a bounce, and a medium with no
        // depth budget of its own would let a path wander forever.
        p.depthAndFlags = (depth + 1u) | PATH_FLAG_ALIVE |
                          (p.depthAndFlags & ~(PATH_DEPTH_MASK | PATH_FLAG_ALIVE |
                                               PATH_FLAG_SPECULAR | PATH_FLAG_NEE_DONE)) |
                          (didNee ? PATH_FLAG_NEE_DONE : 0u);
        if (depth + 1u >= uniforms.maxDepth)
        {
            return;
        }
        paths[tid] = p;
        queuePush(outCounter, queueOut, tid);
        return;
    }

    // --- Subsurface random walk ---------------------------------------------
    //
    // A scattering event inside the medium bounded by the surface the path
    // entered through. Shaped like the fog case above and different from it in
    // two ways: the medium is per material rather than global, and there is no
    // next-event estimation -- the boundary occludes nearly every shadow ray a
    // dense medium would spawn, so the cost is real and the contribution is not.
    // Light gets in and out through the surface, where NEE does run.
    if (SPEC_SSS && (rec.geomEntryIndex & HIT_SSS_BIT) != 0u)
    {
        const uint32_t medium = p.medium & MEDIUM_INDEX_MASK;
        const uint32_t step = p.medium >> MEDIUM_STEP_SHIFT;
        device const Material& mm = materials[medium - 1u];
        const float3 sigmaT = sssSigmaT(float3(mm.subsurface_radius));
        // A bounded volume has no entry surface to have textured, so it keeps the
        // material's constant; a subsurface walk takes what the boundary resolved.
        const bool isBoundedMedium = (mm.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u;
        const float3 albedo = isBoundedMedium ? float3(mm.diffuse_transmission_color)
                                              : unpackMediumAlbedo(p.mediumAlbedo);

        throughput *= sssScatterWeight(sigmaT, albedo, rec.distance);

        const float3 scatterPoint = rayOrigin + rayDir * rec.distance;
        SamplerState wrng = samplerFor(uniforms, tid, sampleIdx, depth + step);

        // A bounded volume is the one kind of medium worth connecting to a light
        // from: it is thin, it is lit from outside, and the shafts and the glow
        // are single scattering. A subsurface walk gets neither -- its boundary
        // occludes almost every shadow ray it would spawn.
        const bool isBounded = isBoundedMedium;
        bool didNeeVolume = false;
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
            radiance += clampIndirectContribution(throughput * float3(mm.medium_emission), depth,
                                                  uniforms.clampIndirect);

            if (SPEC_LIGHTS || (SPEC_ENV_MAP && uniforms.hasEnvMap))
            {
                SurfaceInteraction vsi = {};
                vsi.position = scatterPoint;
                vsi.shading_normal = -rayDir;
                vsi.geometry_normal = -rayDir;
                vsi.front_face = true;
                const LightConnection conn =
                    connectToLight(uniforms, uniforms.numLights, lights, wrng, vsi, envAliasTable,
                                   envMapTexture, true);
                if (conn.needsRay && conn.pdf > 0.0f)
                {
                    // dot(rayDir, toLight): the phase function takes the angle
                    // between the two directions of travel, not between the two
                    // directions pointing away from the vertex. See the same
                    // note on the fog path.
                    const float phase = hgPhase(dot(rayDir, conn.toLight), mm.subsurface_anisotropy);
                    const float misWeight = conn.isDelta ? 1.0f : misWeightBalance(conn.pdf, phase);
                    const float3 weight = throughput * (conn.radiance / conn.pdf) * misWeight * phase;
                    if (any(weight > 1e-6f))
                    {
                        ShadowRay sr;
                        sr.origin = packed_float3(scatterPoint);
                        sr.direction = packed_float3(conn.toLight);
                        sr.weight = packed_float3(
                            clampIndirectContribution(weight, depth, uniforms.clampIndirect));
                        sr.maxDistance = conn.tMax;
                        sr.pixelIndex = tid;
                        sr.medium = p.medium & MEDIUM_INDEX_MASK;
                        sr.rrCutoff = random<SampleDimension::eShadowRR>(wrng, uniforms.samplerType) *
                                      kShadowTransmittanceCutoff;
                        const uint32_t slot =
                            atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
                        shadowRays[slot] = sr;
                        didNeeVolume = true;
                    }
                }
            }
        }

        float phasePdf = 0.0f;
        const float3 nextDir =
            hgSample(-rayDir, mm.subsurface_anisotropy,
                     random<SampleDimension::eSssPhaseU>(wrng, uniforms.samplerType),
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

        PathRay nextRay;
        nextRay.origin = packed_float3(scatterPoint);
        nextRay.direction = packed_float3(nextDir);
        rays[tid] = nextRay;

        p.throughput = packed_float3(throughput);
        p.lastBsdfPdf = phasePdf;
        p.misDistance = 0.0f;
        // The walk advances its own step counter and not the path depth: the
        // whole walk is one scattering event as far as the path budget is
        // concerned, and charging it per step would make a translucent object go
        // black at any sane maxDepth.
        p.medium = medium | ((step + 1u) << MEDIUM_STEP_SHIFT);
        // A subsurface walk keeps its depth: the whole walk is one scattering
        // event as far as the path budget is concerned, and charging it per step
        // would make a translucent object go black at any sane maxDepth. A
        // bounded volume does advance, for the reason the fog path does -- a
        // medium with no depth budget of its own is a path that wanders forever.
        const uint32_t nextDepth = isBounded ? (depth + 1u) : depth;
        p.depthAndFlags = nextDepth | PATH_FLAG_ALIVE |
                          (p.depthAndFlags & ~(PATH_DEPTH_MASK | PATH_FLAG_ALIVE |
                                               PATH_FLAG_SPECULAR | PATH_FLAG_NEE_DONE)) |
                          (didNeeVolume ? PATH_FLAG_NEE_DONE : 0u);
        if (nextDepth >= uniforms.maxDepth)
        {
            return;
        }
        paths[tid] = p;
        queuePush(outCounter, queueOut, tid);
        return;
    }

    // --- Emissive geometry --------------------------------------------------
    if (SPEC_LIGHTS && (rec.geomEntryIndex & HIT_LIGHT_BIT) != 0u)
    {
        const uint32_t lightId = rec.geomEntryIndex & ~HIT_LIGHT_BIT;
        const float3 hitPoint = rayOrigin + rayDir * rec.distance;
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
            const float2 motion =
                screenMotion(uniforms, uniforms.prevWorldToClip * float4(hitPoint, 1.0f),
                             uint2(tid % uniforms.width, tid / uniforms.width));
            a.motionX = motion.x;
            a.motionY = motion.y;
            a.specularHitDistance = 0.0f;
            a.reactive = (depth > 0u) ? 1.0f : 0.0f;
            a.pad2 = 0.0f;
            aov[tid] = a;
        }
        device const UniformLight& currLight = lights[lightId];
        const float3 lightNormal = calcLightNormal(currLight, hitPoint);
        if (-dot(rayDir, lightNormal) > 0.0f)
        {
            const float3 Le = float3(currLight.color);
            if (depth == 0u || specularBounce || !neeDone)
            {
                radiance += throughput * Le;
            }
            else
            {
                const float lightSelectionPdf = uniforms.hasEnvMap
                    ? 0.5f / (float)uniforms.numLights
                    : 1.0f / (float)uniforms.numLights;
                // From the vertex that scattered, which is not the ray's origin
                // once it has passed through a cutout on the way here. Using the
                // origin makes the light look nearer than the scattering vertex
                // saw it, which shrinks its solid-angle density, which inflates
                // this weight -- and the next-event estimate at that vertex has
                // already claimed the rest. The two then sum to more than one.
                const float3 misOrigin = rayOrigin - rayDir * p.misDistance;
                const float lightPdf = getLightPdf(currLight, hitPoint, misOrigin) * lightSelectionPdf;
                radiance += throughput * Le * misWeightBalance(p.lastBsdfPdf, lightPdf);
            }
        }
        radianceOut[tid] += float4(clampIndirectContribution(radiance, depth, uniforms.clampIndirect), 0.0f);
        return;
    }

    // --- Surface ------------------------------------------------------------
    const GeometryEntry entry = geometryEntries[rec.geomEntryIndex];
    const bool interpolateMotion = SPEC_MOTION_BLUR && uniforms.enableMotionBlur &&
                                   motionTime < 1.0f && prevVertexBuffer && indexBuffer;

    const float2 bary = rec.barycentrics;
    const bool isCurve = SPEC_CURVES && (entry.flags & GEOM_FLAG_CURVE) != 0u;
    float3 objectNormal, objectTangent, vertexColor, objectGeomNormal;
    float2 uv;
    float tangentSign = 1.0f;
    float3 objEdge1, objEdge2;
    float uvArea2 = 0.0f;

    const auto inst = instances[rec.instanceIndex];
    const float4x4 objectToWorld = float4x4(
        float4(float3(inst.transformationMatrix[0]), 0.0f),
        float4(float3(inst.transformationMatrix[1]), 0.0f),
        float4(float3(inst.transformationMatrix[2]), 0.0f),
        float4(float3(inst.transformationMatrix[3]), 1.0f));

    const float3 worldPosition = rayOrigin + rayDir * rec.distance;

    float3 shadingNormal, shadingTangent, shadingGeomNormal;
    if (isCurve)
    {
        // Built in world space rather than fetched in object space and
        // transformed out, because the radial normal is taken *from the hit
        // point* -- and the hit point only exists in world space. Coming back the
        // other way would need the transform's inverse, which MSL does not
        // provide and which nothing else in this kernel wants.
        fetchCurve(curvePoints, curveSegments, entry, rec.primitiveId, bary.x, worldPosition,
                   objectToWorld, shadingNormal, shadingTangent, uv);
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
        fetchTriangleBlended(vertexBuffer, prevVertexBuffer, indexBuffer, entry, rec.primitiveId,
                             interpolateMotion, motionTime, bary, objectNormal, objectTangent, uv,
                             vertexColor, tangentSign, objectGeomNormal, objEdge1, objEdge2, uvArea2);
        shadingNormal = normalize(transformDirection(normalize(objectNormal), objectToWorld));
        shadingTangent = normalize(transformDirection(normalize(objectTangent), objectToWorld));
        shadingGeomNormal = normalize(transformDirection(objectGeomNormal, objectToWorld));
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
        const uint32_t passes = p.depthAndFlags >> PATH_PASSTHROUGH_SHIFT;
        if (passes >= PATH_PASSTHROUGH_MAX)
        {
            radianceOut[tid] += float4(radiance, 0.0f);
            return;
        }
        p.depthAndFlags = (p.depthAndFlags & ((1u << PATH_PASSTHROUGH_SHIFT) - 1u)) |
                          ((passes + 1u) << PATH_PASSTHROUGH_SHIFT);

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
        const bool leaving = (p.medium & MEDIUM_INDEX_MASK) == here;
        p.medium = leaving ? 0u : here;

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
        paths[tid] = p;
        queuePush(outCounter, queueOut, tid);
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
    if (SPEC_SSS && (p.medium & MEDIUM_INDEX_MASK) != 0u &&
        (materials[(p.medium & MEDIUM_INDEX_MASK) - 1u].medium_flags & MEDIUM_FLAG_BOUNDARY) == 0u)
    {
        const uint32_t medium = p.medium & MEDIUM_INDEX_MASK;
        const uint32_t step = p.medium >> MEDIUM_STEP_SHIFT;
        device const Material& mm = materials[medium - 1u];
        throughput *= sssBoundaryWeight(sssSigmaT(float3(mm.subsurface_radius)), rec.distance);

        // The ray is travelling outwards, so the outward normal is the one it
        // agrees with.
        const float3 outward = (dot(geomNormal, rayDir) > 0.0f) ? geomNormal : -geomNormal;
        SamplerState xrng = samplerFor(uniforms, tid, sampleIdx, depth + step);

        bool didNeeExit = false;
        if (SPEC_LIGHTS || (SPEC_ENV_MAP && uniforms.hasEnvMap))
        {
            // NEE here and not inside the walk: this is the vertex light can
            // actually reach, and leaving it to BSDF sampling alone is what makes
            // a translucent object the noisiest thing in a frame.
            SurfaceInteraction xsi = {};
            xsi.position = worldPosition;
            xsi.shading_normal = outward;
            xsi.geometry_normal = outward;
            xsi.wo = -rayDir;
            xsi.front_face = true;
            const LightConnection conn =
                connectToLight(uniforms, uniforms.numLights, lights, xrng, xsi, envAliasTable,
                               envMapTexture, false);
            if (conn.needsRay && conn.pdf > 0.0f)
            {
                const float cosOut = dot(outward, conn.toLight);
                if (cosOut > 0.0f)
                {
                    // The exit is Lambertian and the medium's albedo was already
                    // paid for during the walk, so the lobe here is 1/pi and its
                    // own density is cos/pi.
                    const float lobePdf = cosOut * M_1_PI_F;
                    const float misWeight = conn.isDelta ? 1.0f : misWeightBalance(conn.pdf, lobePdf);
                    const float3 weight =
                        throughput * (conn.radiance / conn.pdf) * misWeight * lobePdf;
                    if (any(weight > 1e-6f))
                    {
                        ShadowRay sr;
                        sr.origin = packed_float3(offset_ray(worldPosition, outward));
                        sr.direction = packed_float3(conn.toLight);
                        sr.weight = packed_float3(
                            clampIndirectContribution(weight, depth, uniforms.clampIndirect));
                        sr.maxDistance = conn.tMax;
                        sr.pixelIndex = tid;
                        sr.medium = p.medium & MEDIUM_INDEX_MASK;
                        sr.rrCutoff = random<SampleDimension::eShadowRR>(xrng, uniforms.samplerType) *
                                      kShadowTransmittanceCutoff;
                        const uint32_t slot =
                            atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
                        shadowRays[slot] = sr;
                        didNeeExit = true;
                    }
                }
            }
        }

        const float3 exitDir =
            sssCosineDirection(outward,
                               random<SampleDimension::eSssPhaseU>(xrng, uniforms.samplerType),
                               random<SampleDimension::eSssPhaseV>(xrng, uniforms.samplerType));

        radianceOut[tid] += float4(radiance, 0.0f);

        const float survive = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05f, 1.0f);
        if (random<SampleDimension::eRussianRoulette>(xrng, uniforms.samplerType) >= survive)
        {
            return;
        }
        throughput /= survive;

        PathRay nextRay;
        nextRay.origin = packed_float3(offset_ray(worldPosition, outward));
        nextRay.direction = packed_float3(exitDir);
        rays[tid] = nextRay;

        p.throughput = packed_float3(throughput);
        p.lastBsdfPdf = fmax(dot(outward, exitDir), 0.0f) * M_1_PI_F;
        p.misDistance = 0.0f;
        p.medium = 0u;
        // Depth advances once for the whole walk, here rather than at the entry:
        // charging it at both ends would cost a translucent surface two bounces
        // to do what an opaque one does in one.
        p.depthAndFlags = (depth + 1u) | PATH_FLAG_ALIVE |
                          (p.depthAndFlags & ~(PATH_DEPTH_MASK | PATH_FLAG_ALIVE |
                                               PATH_FLAG_SPECULAR | PATH_FLAG_NEE_DONE)) |
                          (didNeeExit ? PATH_FLAG_NEE_DONE : 0u);
        if (depth + 1u >= uniforms.maxDepth)
        {
            return;
        }
        paths[tid] = p;
        queuePush(outCounter, queueOut, tid);
        return;
    }

    // Ray-cone footprint at this hit. The cone opened by `coneSpread` over the
    // distance just travelled; the triangle turns that width into texels via the
    // ratio of its uv area to its world area, and the grazing term accounts for a
    // footprint stretched by hitting the surface at an angle.
    const float3 worldEdge1 = transformDirection(objEdge1, objectToWorld);
    const float3 worldEdge2 = transformDirection(objEdge2, objectToWorld);
    const float worldArea2 = length(cross(worldEdge1, worldEdge2));
    // MEASURED: no faster. 45.15 ms against 45.25 at 1280x720 native, and 13.80
    // against 13.88 at half resolution with the radiance cache on (n=4 each,
    // ABBA, t=0.10 and t=0.20) -- zero either way, while the image moves on 29%
    // of pixels, so the level of detail is certainly being applied.
    //
    // The premise was that `shade` costs 26% of the frame because five level-0
    // fetches per hit pull against the cache the acceleration structures need.
    // It does not: cutting that traffic moves neither `shade` nor `extend`. The
    // same answer is already recorded above the extend kernel, reached from the
    // other side -- inline traversal cut the MMU limiter from 44% to 29% and the
    // last level cache from 29% to 20% for exactly the same frame time. This
    // renderer is latency-bound on the ray tracing unit at 17% occupancy, and
    // memory-side work of any kind measures as nothing until that moves.
    //
    // Kept, and off by default, because it is a filtering fix rather than a
    // performance one: the renderer builds mip chains, pays for them in memory,
    // and without this reads level 0 for every fetch -- a compute kernel has no
    // derivatives, so sample() takes the top level however the sampler is set.
    // That aliases minified surfaces. At high sample counts the supersampling
    // hides it and mip 0 converges correctly, which is why an offline render
    // should leave this alone; at a genuine 1 spp there is nothing to hide it.
    //
    // The cone is derived rather than carried. PathState is read and written for
    // every live path on every bounce and is guarded at 52 bytes, so two floats
    // there would cost more memory traffic than the mips they buy back.
    //
    // What the width needs is the spread times the segment, and past the primary
    // hit the spread is dominated by the last scattering event, not by the pixel
    // it started from: one diffuse bounce opens the cone over the hemisphere and
    // whatever it was before that stops mattering. So a specular path keeps the
    // pixel's own spread -- which is what keeps a mirror sharp -- and everything
    // else takes a hemisphere's worth.
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
    initSurfaceInteraction(si, materials[entry.materialId],
                           worldPosition, worldNormal, geomNormal,
                           worldTangent, worldBinormal, uv, rayDir, vertexColor, lodBase);

    // 0xFFFFFF words is 64 MB of chase, past any cache on this part; the earlier
    // version walked a 3.7 MB queue and was measuring cache hits, not memory.

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
        const uint32_t layer = p.depthAndFlags >> PATH_PASSTHROUGH_SHIFT;
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
            const uint32_t passes = p.depthAndFlags >> PATH_PASSTHROUGH_SHIFT;
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
            p.depthAndFlags = (p.depthAndFlags & ((1u << PATH_PASSTHROUGH_SHIFT) - 1u)) |
                              ((passes + 1u) << PATH_PASSTHROUGH_SHIFT);
            paths[tid] = p;
            queuePush(outCounter, queueOut, tid);
            return;
        }
    }

    const DebugMode debugMode = (DebugMode)uniforms.debug;
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
            // Same quantity as the megakernel: how far the motion-interpolated
            // normal has moved from the current-frame one.
            float3 nCur[3], pCur[3], tCur[3], cCur[3];
            float2 uvCur[3];
            float signCur = 1.0f; // unused by this debug view
            fetchTriangle(vertexBuffer, prevVertexBuffer, indexBuffer, entry, rec.primitiveId,
                          false, motionTime, pCur, nCur, tCur, uvCur, signCur, cCur);
            dbg = float3(motionTime,
                         clamp(length(normalize(objectNormal) - normalize(nCur[0])) * 10.0f, 0.0f, 1.0f),
                         0.0f);
        }
        radianceOut[tid] = float4(dbg, 0.0f);
        return;
    }

    // Denoiser guides, from the first surface that can actually be described.
    //
    // A mirror or a pane of glass has no albedo to demodulate against and a
    // roughness of nothing, so guides taken there tell the denoiser that a
    // featureless black surface sits where a whole reflected world is, and the
    // reflection is left to denoise itself with no help at all. Walking on to the
    // first rough surface gives it something to work with. PATH_FLAG_AOV_DONE
    // stops the bounce after that from overwriting it.
    const bool aovDone = (p.depthAndFlags & PATH_FLAG_AOV_DONE) != 0u;
    // Below this a surface reflects rather than scatters, and its own albedo is
    // not what the pixel's colour comes from.
    constexpr float kGuideRoughnessFloor = 0.05f;
    // With guidePrimaryHit the camera-visible surface is the answer by
    // definition, so the roughness floor -- and the flicker it causes where a
    // material sits on top of it -- does not enter into it.
    const bool guideWorthy = uniforms.guidePrimaryHit
                                 ? (depth == 0u)
                                 : (si.roughness > kGuideRoughnessFloor);
    // Never walk forever: past a couple of bounces the reflected surface has
    // little to do with this pixel, and no guides at all is worse than imperfect
    // ones.
    const bool guideLastChance = depth >= 2u;

    // Depth and motion, always from the surface the camera actually sees.
    //
    // These two are what the denoiser reprojects with, and unlike albedo or
    // roughness they belong to the pixel rather than to whatever surface the
    // material guides were eventually taken from. Written here, before the walk
    // to the first rough surface, so a specular or alpha-tested primary hit
    // cannot hand the pixel a secondary surface's screen position instead --
    // which is not a small error: the two sit in different places on screen, and
    // measured on the pine forest it gave a fifth of the frame motion vectors of
    // tens of pixels with the camera standing still.
    const bool writingAov = shouldWriteAov(uniforms, sampleIdx);
    if (writingAov && depth == 0u)
    {
        const float3 prevPrimary =
            uniforms.hasPrevFramePose
                ? previousWorldPosition(prevFrameVertexBuffer, indexBuffer, prevInstances, entry, rec.instanceIndex,
                                        rec.primitiveId, bary)
                : worldPosition;
        const float2 primaryMotion =
            screenMotion(uniforms, uniforms.prevWorldToClip * float4(prevPrimary, 1.0f),
                         uint2(tid % uniforms.width, tid / uniforms.width));
        aov[tid].depth = viewDepth(uniforms, worldPosition);
        aov[tid].motionX = primaryMotion.x;
        aov[tid].motionY = primaryMotion.y;
    }

    if (writingAov && !aovDone && (guideWorthy || guideLastChance))
    {
        AovSample a;
        // Metals put their colour in the specular lobe and have no diffuse one.
        const float3 base = float3(si.albedo);
        a.diffuseAlbedo = packed_float3(base * (1.0f - si.metallic));
        a.specularAlbedo = packed_float3(mix(float3(0.04f), base, si.metallic));
        a.normal = packed_float3(si.shading_normal);
        a.roughness = si.roughness;
        // Taken from the block above, which wrote them for the primary surface
        // whatever this one is.
        a.depth = aov[tid].depth;
        const float2 motion = float2(aov[tid].motionX, aov[tid].motionY);
        a.motionX = motion.x;
        a.motionY = motion.y;
        // Filled in by the bounce that follows a specular one; see below.
        a.specularHitDistance = 0.0f;
        // Reactive means "the history for this pixel is not valid", and the one
        // thing that makes it so is guides describing a surface other than the one
        // the camera sees -- which is exactly the deferred-guide case, a primary
        // hit too smooth to describe. Water, glass, a mirror.
        //
        // It used to also scale with the motion vector, saturating at four pixels
        // of movement. That marked the whole frame the moment the camera moved at
        // all, so every camera movement threw away the entire history and the
        // image fell back to single-sample noise -- everywhere, not just on the
        // reflective surfaces the mask is for. Motion is what motion vectors are
        // for; a pixel that moved is reprojectable, not untrustworthy.
        a.reactive = (depth > 0u) ? 1.0f : 0.0f;
        a.pad2 = 0.0f;
        aov[tid] = a;
        p.depthAndFlags |= PATH_FLAG_AOV_DONE;
        paths[tid].depthAndFlags = p.depthAndFlags;
    }

    // What the specular lobe of the primary hit is looking at. MetalFX takes this
    // separately so it can reproject a reflection at the depth of the thing being
    // reflected rather than at the mirror's own.
    if (shouldWriteAov(uniforms, sampleIdx) && depth == 1u && specularBounce)
    {
        aov[tid].specularHitDistance = rec.distance;
    }

    if (si.emission.x > 0.0f || si.emission.y > 0.0f || si.emission.z > 0.0f)
    {
        radiance += throughput * si.emission;
    }

    IorStack iorStack = iorStacks[tid];
    const bool entering = si.front_face;
    si.exterior_ior = entering ? ior_stack_current_ior(iorStack)
                               : ior_stack_peek_after_pop(iorStack, si.dielectric_priority);

    // --- Radiance cache ------------------------------------------------------
    //
    // Read only past the first few bounces and only off a rough surface: the
    // camera ray and the first bounce carry the detail a voxel average would
    // blur, and a mirror reflects a direction rather than a place.
    if (SPEC_SHARC && uniforms.sharcCapacity != 0u && depth >= uniforms.sharcDepth &&
        si.roughness > 0.3f)
    {
        uint32_t voxelHash = 0u, voxelKey = 0u;
        sharcVoxel(si.position, si.shading_normal, uniforms.viewToWorld[3].xyz, uniforms.sharcBaseSize,
                   voxelHash, voxelKey);

        uint32_t slot = 0u;
        // Insert only when this path is going to fill it in; a read that misses
        // simply carries on tracing.
        const bool wantVisit = (p.sharcIndex == SHARC_NO_ENTRY);
        // A fixed share of paths never read and always trace to the end, so the
        // cache keeps converging instead of freezing at whatever the first few
        // paths through a voxel happened to find. This is what SHARC gives its
        // separate update pass; here it is the same paths, thinned.
        const bool updatePath =
            (sharcHash(tid * 9781u + sampleIdx * 6271u) & 7u) == 0u;
        if (sharcFind(sharcEntries, uniforms.sharcCapacity, voxelHash, voxelKey, wantVisit, slot))
        {
            uint32_t cachedCount = 0u;
            const float3 cached = sharcRead(sharcEntries, slot, cachedCount);
            // A path either reads or records, never both. One that has recorded
            // a voxel owes it an honest estimate of the rest of the path, and a
            // cached read inside that estimate feeds the cache its own output --
            // a loop that amplifies whatever error it starts with. It showed as
            // a classroom 11% bright with no single step being wrong.
            if (!updatePath && p.sharcIndex == SHARC_NO_ENTRY &&
                cachedCount >= uniforms.sharcMinSamples)
            {
                // The rest of this path is what the cache already knows.
                radiance += throughput * cached;
                radianceOut[tid] += float4(radiance, 0.0f);
                p.sharcIndex = SHARC_NO_ENTRY;
                paths[tid] = p;
                return;
            }
            // Recorded only while the throughput is worth dividing by. The
            // deposit is what the path gathered divided by its throughput here,
            // and at a throughput of a thousandth that estimator has a variance
            // to match -- a handful of such deposits pulled a classroom 46%
            // bright. Below the threshold the path simply carries on untracked.
            if (wantVisit && luminance(throughput) > 0.05f)
            {
                p.sharcIndex = slot;
                p.sharcRadianceAtVisit = packed_float3(float3(radianceOut[tid].xyz) + radiance);
                p.sharcInvThroughput = packed_float3(1.0f / max(throughput, float3(0.02f)));
            }
        }
    }

    const float4 xi = float4(random<SampleDimension::eBSDF0>(rng, uniforms.samplerType),
                             random<SampleDimension::eBSDF1>(rng, uniforms.samplerType),
                             random<SampleDimension::eBSDF2>(rng, uniforms.samplerType),
                             random<SampleDimension::eBSDF3>(rng, uniforms.samplerType));
    BsdfSampleResult sampleResult = bsdf_sample(si, xi);

    if (sampleResult.event_type == BSDF_EVENT_ABSORB)
    {
        radianceOut[tid] += float4(radiance, 0.0f);
        return;
    }

    const bool nextSpecular = ((sampleResult.event_type & BSDF_EVENT_SPECULAR) != 0);

    bool didNee = (uniforms.estimatorMode == 0) &&
                  (sampleResult.event_type & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY)) &&
                  ((SPEC_LIGHTS && uniforms.numLights > 0) || (SPEC_ENV_MAP && uniforms.hasEnvMap));
    if (didNee)
    {
        // Resampled importance sampling over several light candidates: draw M of
        // them from the light-sampling density, weight each by how much it
        // would actually contribute, and keep one. The shadow ray count does not
        // change -- one candidate survives and one ray is traced -- but the
        // survivor is chosen knowing the BSDF, the cosine and the MIS weight,
        // none of which the light's own density knows about.
        //
        // That is the resampling half of ReSTIR. What it is worth, measured:
        //
        //   classroom, direct light only, 32 spp   rmse 0.0570 -> 0.0370 at M=8
        //   classroom, full render, 256 spp        rmse 10.02  -> 9.76
        //   pine, direct light only, 32 spp        rel  0.1289 -> 0.1277
        //   pine, full render, 128 spp             rel  0.0787 -> 0.0788
        //
        // A third off the direct-lighting error where light selection is the
        // hard part, and nothing at all where it is not. The full renders are
        // the same picture because their error is somewhere else: in the
        // classroom, twelve bounces of indirect; in the forest, visibility.
        // Resampling cannot touch visibility -- the target function is the
        // *unshadowed* contribution, by construction, and which gap in a canopy
        // a direction happens to find is exactly what it does not know.
        //
        // Which is also why the reuse half is not here. Reuse pays by
        // concentrating a reservoir on lights a neighbour already found
        // unoccluded, and that is worth having when visibility is smooth over a
        // few pixels. Dappled light through a canopy is not: the measurement
        // above bounds what any amount of it could recover on this scene at
        // around one percent.
        //
        // Default is one candidate, which reduces every line below to the
        // arithmetic this code had before, bit for bit.
        //
        // The target is the luminance of the unshadowed contribution, with the
        // MIS weight already folded in. Folding it in is what keeps this
        // unbiased against the BSDF strategy: the two weights still sum to one
        // at every direction, so RIS is simply a better estimator of the
        // next-event half and the other half is untouched.
        const uint32_t candidates = max(uniforms.risCandidates, 1u);

        LightConnection bestConn = makeEmptyConnection();
        float3 bestF = float3(0.0f);
        float bestTarget = 0.0f;
        float weightSum = 0.0f;

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

            const LightConnection conn = connectToLight(uniforms, uniforms.numLights, lights, crng,
                                                        si, envAliasTable, envMapTexture);
            const bool isNextEventValid =
                ((dot(conn.toLight, si.shading_normal) > 0.0f) == si.front_face) && conn.pdf > 0.0f;
            if (!isNextEventValid || !conn.needsRay)
            {
                continue;
            }
            BsdfEvalResult evalResult = bsdf_eval(si, conn.toLight);
            if (isnan(conn.pdf) || isnan(evalResult.pdf))
            {
                radianceOut[tid] = float4(1000000.0f, 0.0f, 0.0f, 0.0f);
                return;
            }
            if (!(evalResult.pdf > 0.0f))
            {
                continue;
            }
            const float misWeight =
                conn.isDelta ? 1.0f : misWeightBalance(conn.pdf, evalResult.pdf);
            const float3 f = conn.radiance * evalResult.bsdf * misWeight;
            const float target = luminance(f);
            if (!(target > 0.0f))
            {
                continue;
            }
            const float w = target / conn.pdf;
            weightSum += w;
            // The acceptance draw reuses eLightId under a different scramble
            // rather than taking a dimension of its own. Adding one to the
            // enum shifts every dimension index above it, and with only five
            // Sobol direction matrices to alias into, that alone cost 8% of the
            // relative error on the forest before a single candidate had been
            // resampled.
            SamplerState arng = crng;
            arng.seed = hash_combine(crng.seed, 0x51633e2du);
            if (random<SampleDimension::eLightId>(arng, uniforms.samplerType) * weightSum <= w)
            {
                bestConn = conn;
                bestF = f;
                bestTarget = target;
            }
        }

        if (bestTarget > 0.0f)
        {
            // The reservoir's contribution weight: the mean candidate weight
            // over the target the survivor was kept for. At one candidate it is
            // 1 / pdf and every line below is the arithmetic this code had.
            const float W = (weightSum / (float)candidates) / bestTarget;
            const float3 weight = throughput * bestF * W;
            if (any(weight != 0.0f))
            {
                ShadowRay sr;
                sr.origin = packed_float3(bestConn.origin);
                sr.direction = packed_float3(bestConn.toLight);
                sr.weight = packed_float3(clampIndirectContribution(weight, depth, uniforms.clampIndirect));
                sr.maxDistance = bestConn.tMax;
                sr.pixelIndex = tid;
                sr.medium = p.medium & MEDIUM_INDEX_MASK;
                sr.rrCutoff = random<SampleDimension::eShadowRR>(rng, uniforms.samplerType) *
                              kShadowTransmittanceCutoff;
                const uint32_t slot =
                    atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
                shadowRays[slot] = sr;
            }
        }
    }

    // --- Next segment -------------------------------------------------------
    const float3 faceNg = (dot(si.geometry_normal, si.wo) > 0.0f) ? si.geometry_normal
                                                                  : -si.geometry_normal;
    float3 nextOrigin;
    if ((sampleResult.event_type & BSDF_EVENT_TRANSMISSION) != 0)
    {
        // A thin-walled surface has no interior, so crossing it does not put the
        // path inside anything. Pushing the stack anyway left a ray that had gone
        // through the front of a bubble believing it was inside glass, so the far
        // side was an exit from a dense medium -- and every grazing angle there is
        // past the critical angle.
        if (!si.thin_walled)
        {
            if (entering)
            {
                // Counted, not merely survived. Both of these leave the path
                // carrying the wrong medium and neither used to say a word; see
                // entry 1 of docs/open-defects.md. The test is a loop over at
                // most four entries on a path that has already established it is
                // a transmission through a solid, and the atomic only runs when
                // something has actually gone wrong.
                if (ior_stack_full(iorStack))
                {
                    atomic_fetch_add_explicit(&iorStats[IOR_STAT_OVERFLOW], 1u,
                                              memory_order_relaxed);
                }
                ior_stack_push(iorStack, si.dielectric_priority, si.ior, entry.materialId);
            }
            else
            {
                if (!ior_stack_can_pop(iorStack, si.dielectric_priority, entry.materialId))
                {
                    atomic_fetch_add_explicit(&iorStats[IOR_STAT_UNMATCHED], 1u,
                                              memory_order_relaxed);
                }
                ior_stack_pop(iorStack, si.dielectric_priority, entry.materialId);
            }
        }
        nextOrigin = offset_ray(si.position, -faceNg);

        // Entering a subsurface medium. The lobe that got here is the diffuse
        // transmission one, which on its own puts the light straight out the far
        // side; what this adds is that it random-walks on the way. From here the
        // path is inside, and `extend` samples free flight instead of running to
        // the next surface.
        if (SPEC_SSS && si.subsurface > 0.0f &&
            (sampleResult.event_type & BSDF_EVENT_DIFFUSE_TRANSMISSION) != 0)
        {
            p.medium = (entry.materialId + 1u) & MEDIUM_INDEX_MASK;

            // The walk's albedo, resolved here because this is the last place a
            // texture exists: inside the medium there is no surface to sample.
            // Scaled by how far this point's albedo departs from the one the
            // material's scatter colour was derived from, so a flat material
            // takes the ratio 1 and is unchanged, and marble carries its veining
            // in.
            float3 walkAlbedo = float3(materials[entry.materialId].diffuse_transmission_color);
            const float3 reference = float3(materials[entry.materialId].subsurface_reference);
            if (reference.x > 1e-4f && reference.y > 1e-4f && reference.z > 1e-4f)
            {
                walkAlbedo *= si.albedo / reference;
            }
            p.mediumAlbedo = packMediumAlbedo(saturate(walkAlbedo));
        }
    }
    else
    {
        nextOrigin = offset_ray(si.position, faceNg);
    }
    iorStacks[tid] = iorStack;

    const float3 nextDir = normalize(sampleResult.wi);
    float3 nextThroughput = throughput * sampleResult.bsdf_over_pdf;

    // NEE only reaches directions above the shading normal of a front face, so a
    // hit anywhere else must not be weighted against it.
    didNee = didNee && si.front_face && dot(si.shading_normal, nextDir) > 0.0f;

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

    PathRay nextRay;
    nextRay.origin = packed_float3(nextOrigin);
    nextRay.direction = packed_float3(nextDir);
    rays[tid] = nextRay;

    p.throughput = packed_float3(nextThroughput);
    p.lastBsdfPdf = nextSpecular ? 1.0f : sampleResult.pdf;
    p.misDistance = 0.0f;
    // AOV_DONE is carried, not rebuilt: it records something that already
    // happened to this path, unlike the others, which describe the bounce being
    // set up. Dropping it let every escaping ray overwrite guides that a surface
    // had already written, which is most of the frame in an open scene.
    p.depthAndFlags = (depth + 1u) | PATH_FLAG_ALIVE |
                      (nextSpecular ? PATH_FLAG_SPECULAR : 0u) |
                      (didNee ? PATH_FLAG_NEE_DONE : 0u) |
                      (p.depthAndFlags & PATH_FLAG_AOV_DONE);
    paths[tid] = p;

    queuePush(outCounter, queueOut, tid);
}

// ---------------------------------------------------------------------------
// prepare -- turn the live path count into an indirect dispatch
//
// The count only exists on the GPU. Reading it back to size the next dispatch on
// the CPU would put a round trip in the middle of every bounce, which costs far
// more than the empty threadgroups an indirect dispatch occasionally launches.
// ---------------------------------------------------------------------------
kernel void wavefrontPrepare(
    device uint32_t&        controlRef    [[buffer(0)]],
    constant uint32_t&      srcIdx        [[buffer(1)]],
    constant uint32_t&      threadsPerGroup [[buffer(2)]],
    constant uint32_t&      bounceIdx       [[buffer(3)]])
{
    device uint32_t* control = &controlRef;
    const uint32_t n = control[srcIdx];
    control[WF_CTRL_ACTIVE] = n;
    control[WF_CTRL_STATS_PATHS + min(bounceIdx, 31u)] = n;
    control[WF_CTRL_DISPATCH + 0] = (n + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_DISPATCH + 1] = 1u;
    control[WF_CTRL_DISPATCH + 2] = 1u;
    // The stages about to run append into these, so clear their counts before
    // anything can add to them.
    control[1u - srcIdx] = 0u;
    control[WF_CTRL_SHADOW] = 0u;
    control[WF_CTRL_HIT] = 0u;
    control[WF_CTRL_MISS] = 0u;
}

// Measured and not kept: reordering the extend queue so that neighbouring
// threads trace similar rays.
//
// The profile invites it. A camera ray costs 0.090 us of traversal and a bounce
// ray costs 0.166 -- same structure, same scene, 1.8x the time -- and extend is
// two thirds of the frame, so closing that gap would be worth a fifth of it.
//
// A counting sort into octahedral direction bins (count, scan, scatter) costs
// almost nothing to run, 0.6 ms against extend's 95, and made extend slower
// both ways it was tried. Sorting the whole queue: 94.7 ms -> 103.5. Sorting
// within blocks of 8192, which keeps the origins together: 94.7 -> 98.2.
//
// Why it does not pay here: the queue already arrives in pixel order, because
// `generate` writes it that way and `shade` compacts it in place, so
// neighbouring entries already start from neighbouring points and read
// neighbouring PathRay, HitRecord and PathState. A permutation buys direction
// coherence by giving up both origin coherence and every coalesced access in
// the stage. The 1.8x is also not all divergence -- a bounce ray is simply
// longer than a camera ray into a canopy, and no reordering shortens it.

// Between `extend` and the two stages that consume its classification.
kernel void wavefrontPrepareHitMiss(
    device uint32_t&        controlRef      [[buffer(0)]],
    constant uint32_t&      threadsPerGroup [[buffer(1)]])
{
    device uint32_t* control = &controlRef;
    const uint32_t h = control[WF_CTRL_HIT];
    control[WF_CTRL_HIT_N] = h;
    control[WF_CTRL_HIT_DIS + 0] = (h + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_HIT_DIS + 1] = 1u;
    control[WF_CTRL_HIT_DIS + 2] = 1u;

    const uint32_t m = control[WF_CTRL_MISS];
    control[WF_CTRL_MISS_N] = m;
    control[WF_CTRL_MISS_DIS + 0] = (m + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_MISS_DIS + 1] = 1u;
    control[WF_CTRL_MISS_DIS + 2] = 1u;
}

// Between `shade` and `shadow`: publish the number of shadow rays `shade`
// emitted and size their dispatch. Separate from wavefrontPrepare because the
// count does not exist until `shade` has run.
kernel void wavefrontPrepareShadow(
    device uint32_t&        controlRef      [[buffer(0)]],
    constant uint32_t&      threadsPerGroup [[buffer(1)]],
    constant uint32_t&      bounceIdx       [[buffer(2)]])
{
    device uint32_t* control = &controlRef;
    const uint32_t n = control[WF_CTRL_SHADOW];
    control[WF_CTRL_SHADOW_N] = n;
    control[WF_CTRL_STATS_SHADOW + min(bounceIdx, 31u)] = n;
    control[WF_CTRL_SHADOW_DIS + 0] = (n + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_SHADOW_DIS + 1] = 1u;
    control[WF_CTRL_SHADOW_DIS + 2] = 1u;
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
                                  device const Material* materials,
                                  device const GeometryEntry* geometryEntries,
                                  constant MTLAccelerationStructureUserIDInstanceDescriptor* instances,
                                  float3 origin, float3 direction, float maxDistance,
                                  uint32_t startMedium, float motionTime)
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

        typename T::isect isect;
        isect.assume_geometry_type(T::geometryTypes());
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(false);

        ray r;
        r.origin = origin + direction * travelled;
        r.direction = direction;
        r.min_distance = 1e-4f;
        r.max_distance = remaining;

        const auto hit = T::trace(isect, r, accelerationStructure, GEOMETRY_MASK_MEDIUM, motionTime);
        const bool escaped = (hit.type == intersection_type::none);
        const float segment = escaped ? remaining : hit.distance;

        if (medium != 0u)
        {
            optical += sssSigmaT(float3(materials[medium - 1u].subsurface_radius)) * segment;
        }
        if (escaped)
        {
            break;
        }

        // The same toggle the crossing in `shade` uses, for the same reason: a
        // gizmo's winding is arbitrary, so the normal cannot say which way the
        // ray is going.
        const auto inst = instances[hit.instance_id];
        const uint32_t here =
            (geometryEntries[inst.userID + hit.geometry_id].materialId + 1u) & MEDIUM_INDEX_MASK;
        medium = (medium == here) ? 0u : here;

        travelled += segment + 1e-4f;
    }

    return exp(-optical);
}

// ---------------------------------------------------------------------------
// shadow -- resolve the deferred connections
// ---------------------------------------------------------------------------
template <typename T>
static void shadowImpl(
    uint                    gid,
    constant Uniforms&      uniforms,
    typename T::structure   accelerationStructure,
    device const ShadowRay* shadowRays,
    device float4*          radianceOut,
    device const uint32_t*  control,
    constant uint32_t&      sampleIdx,
    constant MTLAccelerationStructureUserIDInstanceDescriptor* instances,
    device const Material*  materials,
    device const GeometryEntry* geometryEntries,
    device const char*      vertexBuffer,
    device const uint32_t*  indexBuffer,
    typename T::table       functionTable)
{
    if (gid >= control[WF_CTRL_SHADOW_N])
    {
        return;
    }
    const ShadowRay sr = shadowRays[gid];

    typename T::isect isect;
    isect.assume_geometry_type(T::geometryTypes());

    ray shadowRay;
    shadowRay.origin = float3(sr.origin);
    shadowRay.direction = float3(sr.direction);
    shadowRay.min_distance = 0.001f;
    shadowRay.max_distance = sr.maxDistance;

    const float motionTime = motionTimeFor(uniforms, sr.pixelIndex, sampleIdx);
    float3 weight = float3(sr.weight);

    if (!SPEC_ALPHA)
    {
        // No cutouts in this scene: one any-hit trace, exactly as before.
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(true);
        if (T::trace(isect, shadowRay, accelerationStructure, RAY_MASK_SHADOW, motionTime).type ==
            intersection_type::none)
        {
            // Whatever survived the geometry still has to cross the atmosphere. Without
    // this a shadow ray is a hole in the fog, and every light reads as if the
    // haze were not there -- which is exactly the term that makes a low sun
    // through trees look like a low sun through trees.
    // Measured and not moved: this depends only on the ray, so it can be folded
    // into sr.weight where the ray is built, and doing so lifts this kernel's
    // threadgroup limit from 576 to 640. Six interleaved runs of each say the
    // stage does not care -- 36.7 ms against 37.5 -- which is what the model
    // predicts: the same 11% of limit was worth 4.5% of the frame on `extend`
    // at 63% of it, so on a stage at 28% it is around 2%, under the noise floor
    // of a machine that swings 5% between runs.
    if (SPEC_FOG && uniforms.hasFog)
    {
        const float tau = fogOpticalDepth(float3(sr.origin), float3(sr.direction),
                                          sr.maxDistance, uniforms.fogHeight, uniforms.fogSigmaT);
        weight *= exp(-tau);
    }
    if (SPEC_SSS && uniforms.hasBoundedMedium)
    {
        weight *= mediumTransmittance<T>(accelerationStructure, materials, geometryEntries,
                                         instances, float3(sr.origin), float3(sr.direction),
                                         sr.maxDistance, sr.medium, motionTime);
    }
    radianceOut[sr.pixelIndex] += float4(weight, 0.0f);
        }
        return;
    }

    // Cutouts make occlusion a product rather than a predicate, so a plain
    // any-hit does not answer the question: the nearest hit may be a hole. The
    // alpha test runs inside traversal instead, accumulating coverage into the
    // ray payload, so the whole ray is answered by one traversal.
    //
    // Deterministic rather than stochastic: a MASK surface contributes 0 or 1
    // exactly and a BLEND one its alpha, which is far quieter than rolling a
    // second random number per shadow ray.
    //
    // force_opacity is not set here -- the instance flag decides -- because
    // forcing opacity is exactly what makes traversal skip the function.
    //
    // It used to be set, a few lines up and before this branch existed, left
    // over from when a shadow ray was a plain predicate. The intersection
    // function was therefore never called once, cutouts blocked light outright,
    // and every measurement of this stage was measuring a closest-hit search
    // with no alpha test in it.
    // Any hit, not the closest one.
    //
    // The alpha test accumulates a product over the candidates it lets through,
    // and a product does not care in which order they arrive; the traversal
    // stops when the function accepts, which it does only once nothing
    // measurable is left. Asking for the closest accepted hit instead makes the
    // intersector keep candidates in distance order for an answer that is
    // discarded.
    isect.accept_any_intersection(true);

    ShadowPayload payload;
    payload.transmittance = float3(1.0f);
    payload.cutoff = sr.rrCutoff;
    const auto hit = T::traceAnyHit(isect, shadowRay, accelerationStructure, RAY_MASK_SHADOW,
                                    motionTime, functionTable, payload);
    // Above the early return, not below it: in a canopy most shadow rays are
    // blocked and return here, so a probe past this point runs on the minority
    // that got through and reports headroom the stage does not have.

    if (hit.type != intersection_type::none)
    {
        return; // something accepted: fully blocked
    }
    // The survivors of the roulette carry the weight of the ones it killed. A
    // ray ends below the cutoff only if it passed the test, which it does with
    // probability (its transmittance / cutoff), so scaling by the inverse of
    // that puts the expectation back where it was.
    float3 transmittance = payload.transmittance;
    const float m = max(max(transmittance.x, transmittance.y), transmittance.z);
    if (m < kShadowTransmittanceCutoff)
    {
        transmittance *= kShadowTransmittanceCutoff / max(m, 1e-20f);
    }
    weight *= transmittance;
    if (all(weight <= 1e-6f))
    {
        return;
    }

    if (SPEC_FOG && uniforms.hasFog)
    {
        const float tau = fogOpticalDepth(float3(sr.origin), float3(sr.direction),
                                          sr.maxDistance, uniforms.fogHeight, uniforms.fogSigmaT);
        weight *= exp(-tau);
    }
    if (SPEC_SSS && uniforms.hasBoundedMedium)
    {
        weight *= mediumTransmittance<T>(accelerationStructure, materials, geometryEntries,
                                         instances, float3(sr.origin), float3(sr.direction),
                                         sr.maxDistance, sr.medium, motionTime);
    }
    radianceOut[sr.pixelIndex] += float4(weight, 0.0f);
}

// One pass at the end of a sample: every path that passed through a cache voxel
// deposits what it gathered afterwards.
//
// Here rather than at each of the half-dozen places a path can end, because the
// path state survives to the end of the sample and this is one dispatch instead
// of six scattered edits that would each have to stay correct.
kernel void wavefrontSharcDeposit(uint tid [[thread_position_in_grid]],
                                  constant Uniforms& uniforms [[buffer(0)]],
                                  device PathState* paths [[buffer(1)]],
                                  device const float4* radianceOut [[buffer(2)]],
                                  device SharcEntry* sharcEntries [[buffer(3)]])
{
    if (tid >= uniforms.width * uniforms.height)
    {
        return;
    }
    PathState p = paths[tid];
    if (p.sharcIndex == SHARC_NO_ENTRY)
    {
        return;
    }
    const float3 gathered =
        (float3(radianceOut[tid].xyz) - float3(p.sharcRadianceAtVisit)) * float3(p.sharcInvThroughput);
    if (all(gathered >= 0.0f))
    {
        sharcWrite(sharcEntries, p.sharcIndex, gathered);
    }
    p.sharcIndex = SHARC_NO_ENTRY;
    paths[tid] = p;
}

#define WF_SHADOW_ENTRY(NAME, TRAITS)                                                                \
    kernel void NAME(uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]], \
                     TRAITS::structure accelerationStructure [[buffer(1)]],                          \
                     device const ShadowRay* shadowRays [[buffer(2)]],                               \
                     device float4* radianceOut [[buffer(3)]],                                       \
                     device const uint32_t* control [[buffer(4)]],                                   \
                     constant uint32_t& sampleIdx [[buffer(5)]],                                     \
                     constant MTLAccelerationStructureUserIDInstanceDescriptor* instances            \
                         [[buffer(6)]],                                                              \
                     device const Material* materials [[buffer(7)]],                                 \
                     device const GeometryEntry* geometryEntries [[buffer(8)]],                      \
                     device const char* vertexBuffer [[buffer(9)]],                                  \
                     device const uint32_t* indexBuffer [[buffer(10)]],                             \
                     TRAITS::table functionTable [[buffer(11)]])                                     \
    {                                                                                                \
        shadowImpl<TRAITS>(gid, uniforms, accelerationStructure, shadowRays, radianceOut, control,    \
                           sampleIdx, instances, materials, geometryEntries, vertexBuffer,           \
                           indexBuffer, functionTable);                                              \
    }

WF_SHADOW_ENTRY(wavefrontShadow, MotionTraversal)
WF_SHADOW_ENTRY(wavefrontShadowStatic, StaticTraversal)
WF_SHADOW_ENTRY(wavefrontShadowCurve, CurveMotionTraversal)
WF_SHADOW_ENTRY(wavefrontShadowStaticCurve, CurveStaticTraversal)

// ---------------------------------------------------------------------------
// resolve -- average the samples and fold into the accumulation buffer
//
// Byte-for-byte the same arithmetic as the megakernel's tail, so the two
// tracers produce identical images.
// ---------------------------------------------------------------------------
kernel void wavefrontResolve(
    uint                    tid           [[thread_position_in_grid]],
    constant Uniforms&      uniforms      [[buffer(0)]],
    device const float4*    radianceIn    [[buffer(1)]],
    device float4*          res           [[buffer(2)]],
    device float4*          accum         [[buffer(3)]],
    constant uint32_t&      sampleCount   [[buffer(4)]],
    device const AovSample* aov           [[buffer(5)]])
{
    const uint32_t pixelCount = uniforms.width * uniforms.height;
    if (tid >= pixelCount)
    {
        return;
    }

    // Guide views. A denoiser fed a broken guide degrades quietly, so the guides
    // have to be inspectable on their own.
    const uint32_t debugMode = uniforms.debug;
    if (debugMode >= DEBUG_MODE_FIRST_AOV)
    {
        const AovSample a = aov[tid];
        float3 v = float3(0.0f);
        switch ((DebugMode)debugMode)
        {
        case DebugMode::eAovDiffuseAlbedo:  v = float3(a.diffuseAlbedo); break;
        case DebugMode::eAovSpecularAlbedo: v = float3(a.specularAlbedo); break;
        case DebugMode::eAovNormal:         v = float3(a.normal) * 0.5f + 0.5f; break;
        case DebugMode::eAovRoughness:      v = float3(a.roughness); break;
        // d/(1+d): monotonic and scale-free, so a scene of any size is readable
        // and nothing crosses zero the way a logarithm does at d == 1.
        case DebugMode::eAovDepth:          v = float3(a.depth / (1.0f + a.depth)); break;
        // Red/green for the two axes, scaled so a few pixels of motion is visible.
        case DebugMode::eAovMotion:
            v = float3(a.motionX, a.motionY, 0.0f) * 0.05f + 0.5f;
            break;
        // Red where the denoiser is being told to distrust its history, so the
        // extent of the mask is a thing you can look at rather than infer.
        case DebugMode::eAovReactive:      v = float3(a.reactive, 0.0f, 0.0f); break;
        // d/(1+d) again: scale-free, and zero stays zero.
        case DebugMode::eAovSpecularHitDistance:
            v = float3(a.specularHitDistance / (1.0f + a.specularHitDistance));
            break;
        default: break;
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
kernel void wavefrontAovResolve(
    uint2                          tid       [[thread_position_in_grid]],
    constant Uniforms&             uniforms  [[buffer(0)]],
    device const AovSample*        aov       [[buffer(1)]],
    device const float4*           radiance  [[buffer(2)]],
    constant uint32_t&             sampleCount [[buffer(3)]],
    device const float4*           accumulated [[buffer(4)]],
    texture2d<float, access::write> colorTex   [[texture(0)]],
    texture2d<float, access::write> depthTex   [[texture(1)]],
    texture2d<float, access::write> motionTex  [[texture(2)]],
    texture2d<float, access::write> diffuseTex [[texture(3)]],
    texture2d<float, access::write> specularTex[[texture(4)]],
    texture2d<float, access::write> normalTex  [[texture(5)]],
    texture2d<float, access::write> roughTex   [[texture(6)]],
    texture2d<float, access::write> specHitTex [[texture(7)]],
    texture2d<float, access::write> reactiveTex[[texture(8)]])
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
    // When the estimator is accumulating, hand over the running mean rather than
    // this launch's samples. Nothing stops a paused, static scene from converging
    // -- the accumulator is already doing it -- but the denoiser was being fed a
    // one-sample frame forever, so what reached the screen never got better than
    // its own temporal filter could make it.
    const float3 launch = radiance[i].xyz / (float)max(sampleCount, 1u);
    float3 color = uniforms.useAccumulatedColor ? accumulated[i].xyz : launch;

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
    specHitTex.write(float4(a.specularHitDistance, 0.0f, 0.0f, 0.0f), tid);
    reactiveTex.write(float4(saturate(a.reactive), 0.0f, 0.0f, 0.0f), tid);
}
