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
struct MotionTraversal
{
    using structure = acceleration_structure<instancing, primitive_motion>;
    using isect = intersector<triangle_data, instancing, primitive_motion>;
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float time)
    {
        return i.intersect(r, as, mask, time);
    }
};

struct StaticTraversal
{
    using structure = acceleration_structure<instancing>;
    using isect = intersector<triangle_data, instancing>;
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float)
    {
        return i.intersect(r, as, mask);
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
    SamplerState s = initSampler(pixelIndex, uniforms.subframeIndex + sampleIdx, 0u);
    s.depth = depth;
    return s;
}

// A path has one time, sampled once when its camera ray is generated. The
// sampler is stateless, so every later stage can recover that exact value by
// drawing eTime at depth 0 again -- drawing it at the current depth would give a
// different time on every bounce and smear the path across the shutter.
static inline float motionTimeFor(constant Uniforms& uniforms, uint32_t pixelIndex, uint32_t sampleIdx)
{
    if (!uniforms.enableMotionBlur)
    {
        return 0.0f;
    }
    SamplerState s = samplerFor(uniforms, pixelIndex, sampleIdx, 0u);
    const float t = random<SampleDimension::eTime>(s, uniforms.samplerType);
    return uniforms.isMotionBlurVisible ? t : 1.0f;
}

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
    device uint32_t*                                           control        [[buffer(6)]])
{
    const uint32_t pixelCount = uniforms.width * uniforms.height;
    if (tid == 0u)
    {
        control[WF_CTRL_COUNT0] = pixelCount;
        control[WF_CTRL_COUNT1] = 0u;
        control[WF_CTRL_SHADOW] = 0u;
        control[WF_CTRL_HIT] = 0u;
        control[WF_CTRL_MISS] = 0u;
    }
    if (tid >= pixelCount)
    {
        return;
    }
    // Every camera ray starts alive, so the first queue is the identity and
    // needs no compaction.
    queueOut[tid] = tid;

    if (sampleIdx == 0u)
    {
        radianceOut[tid] = float4(0.0f);
    }

    const uint2 pixel = uint2(tid % uniforms.width, tid / uniforms.width);
    SamplerState rng = samplerFor(uniforms, tid, sampleIdx, 0u);
    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);

    float3 origin, direction;
    generateCameraRay(pixel, rng, origin, direction, uniforms, motionTime);

    PathRay r;
    r.origin = packed_float3(origin);
    r.direction = packed_float3(direction);
    rays[tid] = r;

    PathState p;
    p.throughput = packed_float3(float3(1.0f));
    p.depthAndFlags = PATH_FLAG_ALIVE; // depth 0, not specular, NEE not done
    p.lastBsdfPdf = 0.0f;
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
    device atomic_uint*                                        missCounter)
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

    typename T::isect isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    isect.accept_any_intersection(false);

    typename T::isect::result_type hit =
        T::trace(isect, r, accelerationStructure, uniforms.primaryRayMask, motionTime);

    // A ray that escaped carries no information beyond the fact, so it goes
    // straight to the miss stage: no hit record is written and the path never
    // enters `shade`. On a scene with an open background that is most of the
    // secondary rays, and it was the whole cost of the bounce.
    if (hit.type == intersection_type::none)
    {
        queuePush(missCounter, missQueue, tid);
        return;
    }

    const auto inst = instances[hit.instance_id];
    const bool isLight = (inst.mask == GEOMETRY_MASK_LIGHT);
    // For emissive geometry userID indexes the light table, not the geometry
    // table; the flag bit tells `shade` which one it is.
    HitRecord rec;
    rec.geomEntryIndex = isLight ? (HIT_LIGHT_BIT | inst.userID)
                                 : (inst.userID + hit.geometry_id);
    rec.primitiveId = hit.primitive_id;
    rec.barycentrics = hit.triangle_barycentric_coord;
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
                     device atomic_uint* missCounter [[buffer(11)]])                                        \
    {                                                                                                       \
        extendImpl<TRAITS>(gid, uniforms, instances, accelerationStructure, rays, hits, sampleIdx, queue,    \
                           control, hitQueue, hitCounter, missQueue, missCounter);                          \
    }

WF_EXTEND_ENTRY(wavefrontExtend, MotionTraversal)
WF_EXTEND_ENTRY(wavefrontExtendStatic, StaticTraversal)

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
                          thread float3* p, thread float3* n, thread float3* t, thread float2* uv)
{
    // Scene::Vertex layout (32 bytes): pos@0 (packed_float3), tangent@12,
    // normal@16, uv@20 — all uint32 after the position. The `Vertex` struct in
    // ShaderTypes.h does *not* match this and must not be used.
    constexpr uint32_t vtxStride  = 32;
    constexpr uint32_t tangentOff = 12;
    constexpr uint32_t normalOff  = 16;
    constexpr uint32_t uvOff      = 20;

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
        const float3 tan = unpackNormal(*(device const uint32_t*)(v + tangentOff));
        uv[k] = unpackUV(*(device const uint32_t*)(v + uvOff));

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
    texture2d<float>        envMapTexture [[texture(0)]])
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

    float3 radiance = float3(0.0f);
    if (uniforms.hasEnvMap)
    {
        constexpr sampler envSampler(mag_filter::linear, min_filter::linear, address::repeat, coord::normalized);
        const float2 envUV = dirToEnvUV(rayDir, uniforms.envMapRotation);
        float3 envColor = envMapTexture.sample(envSampler, envUV).xyz;
        envColor *= uniforms.envMapIntensity * float3(uniforms.envMapColorTint);

        if (depth == 0u || specularBounce || !neeDone)
        {
            radiance += throughput * envColor;
        }
        else
        {
            const float envPdf = envMapPdf(rayDir, envMapTexture,
                                           uniforms.envMapWidth, uniforms.envMapHeight,
                                           uniforms.envMapRotation, uniforms.envPdfScale);
            const float envSelectionPdf = (uniforms.numLights > 0) ? 0.5f : 1.0f;
            const float effectiveEnvPdf = envPdf * envSelectionPdf;
            if (effectiveEnvPdf > 0.0f)
            {
                radiance += throughput * envColor * misWeightBalance(p.lastBsdfPdf, effectiveEnvPdf);
            }
        }
    }
    else
    {
        radiance += throughput * uniforms.missColor;
    }
    radianceOut[tid] += float4(radiance, 0.0f);
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
    const float3 throughput = float3(p.throughput);

    float3 radiance = float3(0.0f);
    const HitRecord rec = hits[tid];

    // --- Emissive geometry --------------------------------------------------
    if ((rec.geomEntryIndex & HIT_LIGHT_BIT) != 0u)
    {
        const uint32_t lightId = rec.geomEntryIndex & ~HIT_LIGHT_BIT;
        const float3 hitPoint = rayOrigin + rayDir * rec.distance;
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
                const float lightPdf = getLightPdf(currLight, hitPoint, rayOrigin) * lightSelectionPdf;
                radiance += throughput * Le * misWeightBalance(p.lastBsdfPdf, lightPdf);
            }
        }
        radianceOut[tid] += float4(radiance, 0.0f);
        return;
    }

    // --- Surface ------------------------------------------------------------
    const GeometryEntry entry = geometryEntries[rec.geomEntryIndex];
    const bool interpolateMotion = uniforms.enableMotionBlur && motionTime < 1.0f &&
                                   prevVertexBuffer && indexBuffer;

    float3 pv[3], nv[3], tv[3];
    float2 uvv[3];
    fetchTriangle(vertexBuffer, prevVertexBuffer, indexBuffer, entry, rec.primitiveId,
                  interpolateMotion, motionTime, pv, nv, tv, uvv);

    const auto inst = instances[entry.instanceIndex];
    const float4x4 objectToWorld = float4x4(
        float4(float3(inst.transformationMatrix[0]), 0.0f),
        float4(float3(inst.transformationMatrix[1]), 0.0f),
        float4(float3(inst.transformationMatrix[2]), 0.0f),
        float4(float3(inst.transformationMatrix[3]), 1.0f));

    const float2 bary = rec.barycentrics;
    const float2 uv = interpolateAttrib(uvv[0], uvv[1], uvv[2], bary);
    const float3 worldPosition = rayOrigin + rayDir * rec.distance;

    const float3 objectNormal = normalize(interpolateAttrib(nv[0], nv[1], nv[2], bary));
    const float3 worldNormal = normalize(transformDirection(objectNormal, objectToWorld));
    const float3 worldTangent = normalize(transformDirection(
        normalize(interpolateAttrib(tv[0], tv[1], tv[2], bary)), objectToWorld));
    const float3 worldBinormal = cross(worldNormal, worldTangent);

    float3 geomNormal = cross(pv[1] - pv[0], pv[2] - pv[0]);
    geomNormal = normalize(transformDirection(geomNormal, objectToWorld));

    SurfaceInteraction si;
    initSurfaceInteraction(si, materials[entry.materialId],
                           worldPosition, worldNormal, geomNormal,
                           worldTangent, worldBinormal, uv, rayDir);

    const DebugMode debugMode = (DebugMode)uniforms.debug;
    if (debugMode == DebugMode::eMotionBlur || debugMode == DebugMode::eNormal)
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
            float3 nCur[3], pCur[3], tCur[3];
            float2 uvCur[3];
            fetchTriangle(vertexBuffer, prevVertexBuffer, indexBuffer, entry, rec.primitiveId,
                          false, motionTime, pCur, nCur, tCur, uvCur);
            dbg = float3(motionTime, clamp(length(nv[0] - nCur[0]) * 10.0f, 0.0f, 1.0f), 0.0f);
        }
        radianceOut[tid] = float4(dbg, 0.0f);
        return;
    }

    if (si.emission.x > 0.0f || si.emission.y > 0.0f || si.emission.z > 0.0f)
    {
        radiance += throughput * si.emission;
    }

    IorStack iorStack = iorStacks[tid];
    const bool entering = si.front_face;
    si.exterior_ior = entering ? ior_stack_current_ior(iorStack)
                               : ior_stack_peek_after_pop(iorStack, si.dielectric_priority);

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
                  (uniforms.numLights > 0 || uniforms.hasEnvMap);
    if (didNee)
    {
        // Build the connection here and hand the ray to the shadow stage, which
        // adds the contribution if nothing is in the way. Multiplying by a
        // visibility of 0 and adding the result is the same as not adding it, so
        // deferring changes no arithmetic.
        const LightConnection conn = connectToLight(uniforms, uniforms.numLights, lights, rng, si,
                                                    envAliasTable, envMapTexture);

        const bool isNextEventValid =
            ((dot(conn.toLight, si.shading_normal) > 0.0f) == si.front_face) && conn.pdf > 0.0f;
        if (isNextEventValid)
        {
            BsdfEvalResult evalResult = bsdf_eval(si, conn.toLight);
            if (isnan(conn.pdf) || isnan(evalResult.pdf))
            {
                radianceOut[tid] = float4(1000000.0f, 0.0f, 0.0f, 0.0f);
                return;
            }
            if (evalResult.pdf > 0.0f && conn.needsRay)
            {
                const float3 weight = throughput * (conn.radiance / conn.pdf) *
                                      misWeightBalance(conn.pdf, evalResult.pdf) * evalResult.bsdf;
                if (any(weight != 0.0f))
                {
                    ShadowRay sr;
                    sr.origin = packed_float3(conn.origin);
                    sr.direction = packed_float3(conn.toLight);
                    sr.weight = packed_float3(weight);
                    sr.maxDistance = conn.tMax;
                    sr.pixelIndex = tid;
                    const uint32_t slot =
                        atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
                    shadowRays[slot] = sr;
                }
            }
        }
    }

    // --- Next segment -------------------------------------------------------
    const float3 faceNg = (dot(si.geometry_normal, si.wo) > 0.0f) ? si.geometry_normal
                                                                  : -si.geometry_normal;
    float3 nextOrigin;
    if ((sampleResult.event_type & BSDF_EVENT_TRANSMISSION) != 0)
    {
        if (entering)
            ior_stack_push(iorStack, si.dielectric_priority, si.ior);
        else
            ior_stack_pop(iorStack, si.dielectric_priority);
        nextOrigin = offset_ray(si.position, -faceNg);
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
    p.depthAndFlags = (depth + 1u) | PATH_FLAG_ALIVE |
                      (nextSpecular ? PATH_FLAG_SPECULAR : 0u) |
                      (didNee ? PATH_FLAG_NEE_DONE : 0u);
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
    constant uint32_t&      sampleIdx)
{
    if (gid >= control[WF_CTRL_SHADOW_N])
    {
        return;
    }
    const ShadowRay sr = shadowRays[gid];

    typename T::isect isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    isect.accept_any_intersection(true);

    ray shadowRay;
    shadowRay.origin = float3(sr.origin);
    shadowRay.direction = float3(sr.direction);
    shadowRay.min_distance = 0.001f;
    shadowRay.max_distance = sr.maxDistance;

    const float motionTime = motionTimeFor(uniforms, sr.pixelIndex, sampleIdx);
    const bool occluded =
        T::trace(isect, shadowRay, accelerationStructure, RAY_MASK_SHADOW, motionTime).type !=
        intersection_type::none;
    if (!occluded)
    {
        radianceOut[sr.pixelIndex] += float4(float3(sr.weight), 0.0f);
    }
}

#define WF_SHADOW_ENTRY(NAME, TRAITS)                                                                \
    kernel void NAME(uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]], \
                     TRAITS::structure accelerationStructure [[buffer(1)]],                          \
                     device const ShadowRay* shadowRays [[buffer(2)]],                               \
                     device float4* radianceOut [[buffer(3)]],                                       \
                     device const uint32_t* control [[buffer(4)]],                                   \
                     constant uint32_t& sampleIdx [[buffer(5)]])                                     \
    {                                                                                                \
        shadowImpl<TRAITS>(gid, uniforms, accelerationStructure, shadowRays, radianceOut, control,    \
                           sampleIdx);                                                               \
    }

WF_SHADOW_ENTRY(wavefrontShadow, MotionTraversal)
WF_SHADOW_ENTRY(wavefrontShadowStatic, StaticTraversal)

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
    constant uint32_t&      sampleCount   [[buffer(4)]])
{
    const uint32_t pixelCount = uniforms.width * uniforms.height;
    if (tid >= pixelCount)
    {
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
