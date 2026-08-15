#include <optix.h>

#include <cuda.h>

#include <OptixRenderParams.h>
#include <cuda_helpers/helpers.h>
#include <cuda_helpers/curve.h>

#include <random.h>

#include <sutil/Matrix.h>
#include <sutil/vec_math.h>
#include <sutil/vec_math_adv.h>

#include <lights.h>
#include <env_light.h>

#include <strelka/material/bsdf.h>

#include "optix_device_utils.h"

extern "C"
{
    __constant__ Params params;
}

static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction, float tmin, float tmax)
{
    const float time = optixGetRayTime();

    unsigned int occluded = 0u;
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
               time, // rayTime
               OptixVisibilityMask(RAY_MASK_SHADOW), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               RAY_TYPE_OCCLUSION, // SBT offset
               RAY_TYPE_COUNT, // SBT stride
               RAY_TYPE_OCCLUSION, // missSBTIndex
               occluded);
    return occluded;
}

/// Uniform light choice from a canonical sample, clamped to the last light.
///
/// The clamp is not defensive noise: u is drawn from [0, 1) but the env-map branch
/// feeds it u*2 from a u already known to be below 0.5, and that product rounds to
/// exactly 1.0f for the largest such u. Without the clamp that one sample indexes
/// one past the end of the light buffer.
static __forceinline__ __device__ uint32_t selectLightIndex(float u, uint32_t numLights)
{
    const uint32_t index = (uint32_t)(numLights * u);
    return (index < numLights) ? index : (numLights - 1);
}

/// One proposed connection to a light, before visibility is known.
///
/// Separating the proposal from the shadow ray is what makes resampling possible:
/// several candidates can be drawn and weighted by what they would contribute,
/// and only the survivor costs a ray. It also fixes an ordering problem the old
/// code had -- it traced occlusion inside the light sampler, so a candidate that
/// loses the resampling draw would still have paid for a ray.
struct LightConnection
{
    float3 radiance; // Li times the shading cosine, unshadowed
    float3 toLight;
    float pdf; // solid-angle density, including the light-selection probability
    float tMax;
    bool needsRay;
    /// A sharp point or spot cannot be hit by a BSDF ray, so no other strategy
    /// can produce this direction and its MIS weight is exactly one. Weighting it
    /// against the BSDF pdf -- which is what this code used to do -- discards the
    /// share of the light the BSDF strategy is credited with and never delivers.
    bool isDelta;
};

static __forceinline__ __device__ LightConnection makeEmptyConnection()
{
    LightConnection c;
    c.radiance = make_float3(0.0f);
    c.toLight = make_float3(0.0f);
    c.pdf = 0.0f;
    c.tMax = 0.0f;
    c.needsRay = false;
    c.isDelta = false;
    return c;
}

static __device__ LightConnection connectLight(SamplerState& sampler,
                                               const UniformLight& light,
                                               const SurfaceInteraction& si)
{
    LightSampleData lightSampleData = {};
    const float2 uv =
        make_float2(random<SampleDimension::eLightPointX>(sampler), random<SampleDimension::eLightPointY>(sampler));
    switch (light.type)
    {
    case LIGHT_TYPE_RECT:
        if (params.rectLightSamplingMethod == 0)
        {
            lightSampleData = SampleRectLightUniform(light, uv, si.position);
        }
        else
        {
            lightSampleData = SampleRectLight(light, uv, si.position);
        }
        break;
    case LIGHT_TYPE_DISC:
        lightSampleData = SampleDiscLight(light, uv, si.position);
        break;
    case LIGHT_TYPE_SPHERE:
        lightSampleData = SampleSphereLight(light, uv, si.position);
        break;
    case LIGHT_TYPE_DISTANT:
        lightSampleData = SampleDistantLight(light, uv, si.position);
        break;
    case LIGHT_TYPE_DOME:
        lightSampleData = SampleDomeLight(light, uv, si.position);
        break;
    case LIGHT_TYPE_POINT:
    case LIGHT_TYPE_SPOT:
        lightSampleData = SamplePointLight(light, uv, si.position);
        break;
    }

    LightConnection c = makeEmptyConnection();
    c.toLight = lightSampleData.L;
    // Sharp point/spot only: give one a radius and it is sampled as a sphere,
    // which BSDF rays can hit and which therefore does need MIS.
    c.isDelta = (light.type == LIGHT_TYPE_POINT || light.type == LIGHT_TYPE_SPOT) &&
                !(light.points[0].x > 1e-4f);

    float3 Li = make_float3(light.color);
    if (light.type == LIGHT_TYPE_POINT || light.type == LIGHT_TYPE_SPOT)
    {
        const float dist = fmaxf(lightSampleData.distToLight, 1e-4f);
        Li *= rangeWindow(light, dist) / (dist * dist);
        if (light.type == LIGHT_TYPE_SPOT)
        {
            Li *= spotAttenuation(light, -lightSampleData.L);
        }
    }

    const bool facing = (light.type == LIGHT_TYPE_POINT || light.type == LIGHT_TYPE_SPOT)
                            ? (dot(si.shading_normal, lightSampleData.L) > 0.0f && emitsLight(Li))
                            : (dot(si.shading_normal, lightSampleData.L) > 0.0f &&
                               -dot(lightSampleData.L, lightSampleData.normal) > 0.0f && emitsLight(Li));
    if (facing)
    {
        // The cosine belongs here because bsdf_eval() returns f alone, unlike
        // bsdf_sample()'s bsdf_over_pdf which already carries it. See the note on
        // both result structs in bsdf_types.h.
        c.radiance = Li * saturate(dot(si.shading_normal, lightSampleData.L));
        c.pdf = lightSampleData.pdf;
        c.tMax = lightSampleData.distToLight;
        c.needsRay = true;
    }
    return c;
}

static __device__ LightConnection connectEnvLight(SamplerState& sampler, const SurfaceInteraction& si)
{
    const float2 xi = make_float2(
        random<SampleDimension::eLightPointX>(sampler),
        random<SampleDimension::eLightPointY>(sampler));

    float envPdf = 0.0f;
    const float3 dir = sampleEnvMap(xi,
                                    params.envAliasTable, params.envMapTexturePoint,
                                    params.envMapWidth, params.envMapHeight,
                                    params.envMapRotation, params.envPdfScale,
                                    envPdf);

    LightConnection c = makeEmptyConnection();
    c.toLight = dir;
    c.pdf = envPdf;

    if (envPdf <= 0.0f)
        return c;

    // Check if direction is above the surface
    if (dot(si.shading_normal, dir) <= 0.0f)
        return c;

    // Bilinear for the radiance carried down the ray; the point fetch inside
    // sampleEnvMap is for the sampling density only, and using it here would
    // quantise the lighting to the map's texels.
    const float2 uv = dirToEnvUV(dir, params.envMapRotation);
    const float4 envSample = tex2D<float4>(params.envMapTexture, uv.x, uv.y);
    float3 Li = make_float3(envSample.x, envSample.y, envSample.z);
    Li *= params.envMapIntensity * params.envMapColorTint;

    c.radiance = Li * fmaxf(dot(si.shading_normal, dir), 0.0f);
    c.tMax = 1e16f;
    c.needsRay = true;
    return c;
}

/// Choose a strategy and build the connection. Visibility is the caller's job.
static __device__ LightConnection connectToLight(SamplerState& sampler, const SurfaceInteraction& si)
{
    if (params.hasEnvMap)
    {
        const float u = random<SampleDimension::eLightId>(sampler);

        if (params.scene.numLights == 0 || u >= 0.5f)
        {
            // Sample environment map
            const float selectionPdf = (params.scene.numLights > 0) ? 0.5f : 1.0f;
            LightConnection c = connectEnvLight(sampler, si);
            c.pdf *= selectionPdf;
            return c;
        }
        // Sample local light (remap u from [0, 0.5) to [0, 1))
        const uint32_t lightId = selectLightIndex(u * 2.0f, params.scene.numLights);
        LightConnection c = connectLight(sampler, params.scene.lights[lightId], si);
        c.pdf *= 0.5f / params.scene.numLights;
        return c;
    }

    // A scene with neither an environment nor an analytic light has nothing to
    // connect to. This used to fall through and divide by numLights == 0, then
    // read lights[0] off a null device pointer -- mLightBuffer is constructed
    // empty, so the pointer really is null rather than merely unpopulated.
    if (params.scene.numLights == 0)
    {
        return makeEmptyConnection();
    }

    const float u = random<SampleDimension::eLightId>(sampler);
    const uint32_t lightId = selectLightIndex(u, params.scene.numLights);
    LightConnection c = connectLight(sampler, params.scene.lights[lightId], si);
    c.pdf *= 1.0f / params.scene.numLights;
    return c;
}

/// Next-event estimation, with resampled importance sampling over the candidates.
///
/// Draw M candidates from the light-sampling density, weight each by what it
/// would actually contribute -- BSDF, cosine and MIS weight included, none of
/// which the light's own density knows about -- and keep one. The shadow ray
/// count does not change: one candidate survives and one ray is traced.
///
/// The target is the luminance of the *unshadowed* contribution with the MIS
/// weight already folded in. Folding it in is what keeps this unbiased against
/// the BSDF strategy: the two weights still sum to one in every direction, so
/// resampling only improves the next-event half and leaves the other alone.
/// Resampling cannot help with visibility, by construction -- the target does not
/// know it.
///
/// The default of one candidate reduces every line below to plain next-event
/// estimation, bit for bit: the first candidate uses the sampler unmodified.
///
/// Returns the radiance to add at this vertex, already multiplied by throughput
/// and clamped, or zero.
static __device__ float3 estimateDirectLighting(PerRayData* prd, const SurfaceInteraction& si)
{
    const uint32_t candidates = max(params.risCandidates, 1u);

    LightConnection bestConn = makeEmptyConnection();
    float3 bestF = make_float3(0.0f);
    float bestTarget = 0.0f;
    float weightSum = 0.0f;

    for (uint32_t i = 0; i < candidates; ++i)
    {
        // Candidates differ by their scramble, not by their dimension: each one
        // stays a stratified sequence across samples, and the first is the
        // sequence this code drew before RIS existed.
        SamplerState crng = prd->sampler;
        if (i != 0u)
        {
            crng.seed = hash_combine(prd->sampler.seed, i * 0x9E3779B9u);
        }

        const LightConnection conn = connectToLight(crng, si);
        const bool isNextEventValid =
            ((dot(conn.toLight, si.shading_normal) > 0.0f) == si.front_face) && (conn.pdf > 0.0f);
        if (!isNextEventValid || !conn.needsRay)
        {
            continue;
        }

        const BsdfEvalResult evalData = bsdf_eval(si, conn.toLight);
        if (isnan(conn.radiance) || isnan(conn.pdf) || isnan(evalData.bsdf) || isnan(evalData.pdf))
        {
            // ERROR, terminate tracing
            prd->radiance = make_float3(10000.0f, 0.0f, 0.0f);
            prd->throughput = make_float3(0.0f);
            return make_float3(0.0f);
        }
        if (!(evalData.pdf > 0.0f))
        {
            continue;
        }

        const float misWeight =
            conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, evalData.pdf, params.misHeuristic);
        const float3 f = conn.radiance * evalData.bsdf * misWeight;
        const float target = dot(f, make_float3(0.2126f, 0.7152f, 0.0722f));
        if (!(target > 0.0f))
        {
            continue;
        }

        const float w = target / conn.pdf;
        weightSum += w;
        // The acceptance draw reuses eLightId under a different scramble rather
        // than taking a dimension of its own: adding one to the enum shifts every
        // dimension index above it.
        SamplerState arng = crng;
        arng.seed = hash_combine(crng.seed, 0x51633e2du);
        if (random<SampleDimension::eLightId>(arng) * weightSum <= w)
        {
            bestConn = conn;
            bestF = f;
            bestTarget = target;
        }
    }

    if (!(bestTarget > 0.0f))
    {
        return make_float3(0.0f);
    }

    // The reservoir's contribution weight: the mean candidate weight over the
    // target the survivor was kept for. At one candidate it is 1 / pdf and every
    // line here is the arithmetic this code had.
    const float W = (weightSum / (float)candidates) / bestTarget;
    const float3 weight = prd->throughput * bestF * W;
    if (weight.x == 0.0f && weight.y == 0.0f && weight.z == 0.0f)
    {
        return make_float3(0.0f);
    }

    // Offset along the face the shadow ray actually leaves from. The raw geometry
    // normal points to a fixed side of the triangle, so on a back-face hit it
    // pushes the origin *into* the surface and the ray immediately hits the
    // geometry it started on -- next-event estimation then reports occlusion the
    // BSDF strategy does not see, and the two halves of the MIS estimate stop
    // summing to the integral. The bounce ray below already orients its offset
    // this way; this is the same fix Metal's connectEnvLight carries.
    const float3 offsetNg = (dot(si.geometry_normal, bestConn.toLight) > 0.0f) ? si.geometry_normal
                                                                              : -si.geometry_normal;
    const bool occluded = traceOcclusion(params.handle, offset_ray(si.position, offsetNg), bestConn.toLight,
                                         params.shadowRayTmin, bestConn.tMax);
    if (occluded)
    {
        return make_float3(0.0f);
    }
    return clampIndirectContribution(weight, prd->depth, params.clampIndirect);
}

// Get curve hit-point in world coordinates.
static __forceinline__ __device__ float3 getHitPoint()
{
    const float t = optixGetRayTmax();
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDirection = optixGetWorldRayDirection();

    return rayOrigin + t * rayDirection;
}

// Compute surface normal of cubic primitive in world space.
static __forceinline__ __device__ float3 normalCubic(const int primitiveIndex)
{
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int gasSbtIndex = optixGetSbtGASIndex();
    float4 controlPoints[4];

    optixGetCubicBSplineVertexData(gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints);

    CubicInterpolator interpolator;
    interpolator.initializeFromBSpline(controlPoints);

    float3 hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint = optixTransformPointFromWorldToObjectSpace(hitPoint);
    const float3 normal = surfaceNormal(interpolator, optixGetCurveParameter(), hitPoint);
    return optixTransformNormalFromObjectToWorldSpace(normal);
}

struct SurfaceHitData
{
    float3 position;
    float3 normal;
    float3 geom_normal;
    float2 uv;
    float3 worldTangent;
    float3 worldBinormal;
};

static __forceinline__ __device__ SurfaceHitData fillTriangleGeomData(const HitGroupData* hit_data)
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const unsigned int primitiveId = optixGetPrimitiveIndex();
    const uint32_t i0 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 0)];
    const uint32_t i1 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 1)];
    const uint32_t i2 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 2)];

    const uint32_t baseVbOffset = hit_data->vertexOffset;

    float3 p0, p1, p2, n0, n1, n2, t0, t1, t2;
    float2 uv0, uv1, uv2;

    if (params.enableMotionBlur)
    {
        const float t = optixGetRayTime();
        const Vertex* vb0 = params.scene.vb_prev;
        const Vertex* vb1 = params.scene.vb;

        const Vertex v0_0 = vb0[baseVbOffset + i0];
        const Vertex v1_0 = vb0[baseVbOffset + i1];
        const Vertex v2_0 = vb0[baseVbOffset + i2];

        const Vertex v0_1 = vb1[baseVbOffset + i0];
        const Vertex v1_1 = vb1[baseVbOffset + i1];
        const Vertex v2_1 = vb1[baseVbOffset + i2];

        // Interpolate all vertex attributes
        p0 = lerp(v0_0.position, v0_1.position, t);
        p1 = lerp(v1_0.position, v1_1.position, t);
        p2 = lerp(v2_0.position, v2_1.position, t);

        n0 = lerp(unpackNormal(v0_0.normal), unpackNormal(v0_1.normal), t);
        n1 = lerp(unpackNormal(v1_0.normal), unpackNormal(v1_1.normal), t);
        n2 = lerp(unpackNormal(v2_0.normal), unpackNormal(v2_1.normal), t);

        t0 = lerp(unpackNormal(v0_0.tangent), unpackNormal(v0_1.tangent), t);
        t1 = lerp(unpackNormal(v1_0.tangent), unpackNormal(v1_1.tangent), t);
        t2 = lerp(unpackNormal(v2_0.tangent), unpackNormal(v2_1.tangent), t);

        uv0 = lerp(unpackUV(v0_0.uv), unpackUV(v0_1.uv), t);
        uv1 = lerp(unpackUV(v1_0.uv), unpackUV(v1_1.uv), t);
        uv2 = lerp(unpackUV(v2_0.uv), unpackUV(v2_1.uv), t);
    }
    else
    {
        const Vertex v0 = params.scene.vb[baseVbOffset + i0];
        const Vertex v1 = params.scene.vb[baseVbOffset + i1];
        const Vertex v2 = params.scene.vb[baseVbOffset + i2];
        p0 = v0.position;
        p1 = v1.position;
        p2 = v2.position;

        n0 = unpackNormal(v0.normal);
        n1 = unpackNormal(v1.normal);
        n2 = unpackNormal(v2.normal);

        t0 = unpackNormal(v0.tangent);
        t1 = unpackNormal(v1.tangent);
        t2 = unpackNormal(v2.tangent);

        uv0 = unpackUV(v0.uv);
        uv1 = unpackUV(v1.uv);
        uv2 = unpackUV(v2.uv);
    }

    const float2 uvCoord = interpolateAttrib(uv0, uv1, uv2, barycentrics);

    const float3 worldPosition = optixTransformPointFromObjectToWorldSpace(interpolateAttrib(p0, p1, p2, barycentrics));
    const float3 object_normal = interpolateAttrib(n0, n1, n2, barycentrics);
    float3 worldNormal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    float3 geomNormal = cross(p1 - p0, p2 - p0);
    geomNormal = normalize(optixTransformNormalFromObjectToWorldSpace(geomNormal));
    const float3 worldTangent =
        normalize(optixTransformNormalFromObjectToWorldSpace(interpolateAttrib(t0, t1, t2, barycentrics)));
    const float3 worldBinormal = cross(worldNormal, worldTangent);

    SurfaceHitData res;
    res.normal = worldNormal;
    res.geom_normal = geomNormal;
    res.position = worldPosition;
    res.uv = uvCoord;
    res.worldTangent = worldTangent;
    res.worldBinormal = worldBinormal;
    return res;
}

static __forceinline__ __device__ SurfaceHitData fillCurveGeomData(const HitGroupData* hit_data)
{
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int gasSbtIndex = optixGetSbtGASIndex();
    float4 controlPoints[4];
    optixGetCubicBSplineVertexData(gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints);
    CubicInterpolator interpolator;
    interpolator.initializeFromBSpline(controlPoints);
    float3 hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint = optixTransformPointFromWorldToObjectSpace(hitPoint); // interpolators work in object space
    float3 worldNormal = normalize(
        optixTransformNormalFromObjectToWorldSpace(surfaceNormal(interpolator, optixGetCurveParameter(), hitPoint)));
    const float3 worldTangent =
        normalize(optixTransformNormalFromObjectToWorldSpace(curveTangent(interpolator, optixGetCurveParameter())));
    const float3 worldBinormal = cross(worldNormal, worldTangent);
    const float3 worldPosition = optixTransformPointFromObjectToWorldSpace(hitPoint);
    SurfaceHitData res;
    res.normal = worldNormal;
    res.geom_normal = worldNormal;
    res.position = worldPosition;
    res.uv = make_float2(0.5f, 0.5f);
    res.worldTangent = worldTangent;
    res.worldBinormal = worldBinormal;

    return res;
}

extern "C" __global__ void __closesthit__radiance()
{
    OptixPrimitiveType primType = optixGetPrimitiveType();

    PerRayData* prd = getPRD();
    HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const float3 ray_dir = optixGetWorldRayDirection();

    SurfaceHitData surfaceHit;
    if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE)
    {
        surfaceHit = fillTriangleGeomData(hit_data);
    }
    else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE)
    {
        surfaceHit = fillCurveGeomData(hit_data);
    }

    // Fill SurfaceInteraction from hit data
    SurfaceInteraction si;
    si.position = surfaceHit.position;
    si.shading_normal = surfaceHit.normal;
    si.geometry_normal = surfaceHit.geom_normal;
    si.tangent = surfaceHit.worldTangent;
    si.bitangent = surfaceHit.worldBinormal;
    si.uv = surfaceHit.uv;
    si.wo = -ray_dir;
    si.front_face = dot(surfaceHit.geom_normal, -ray_dir) > 0.0f;

    // Look up material from device buffer (indexed by materialId)
    const int32_t matId = hit_data->materialId;
    const MaterialParams& matParams = params.materials[matId];
    const cudaTextureObject_t* textures = &params.materialTextures[matId * MAX_MATERIAL_TEXTURES];
    bsdf_init(si, matParams, textures);

    if (params.debug == 1)
    {
        prd->radiance = (si.geometry_normal + make_float3(1.0f)) * 0.5f;
        return;
    }

    // Add emission
    if (si.emission.x > 0.0f || si.emission.y > 0.0f || si.emission.z > 0.0f)
    {
        prd->radiance +=
            clampIndirectContribution(prd->throughput * si.emission, prd->depth, params.clampIndirect);
    }

    // Set exterior IOR from the IOR stack for nested dielectrics
    bool entering = si.front_face;
    if (entering)
    {
        si.exterior_ior = ior_stack_current_ior(prd->iorStack);
    }
    else
    {
        si.exterior_ior = ior_stack_peek_after_pop(prd->iorStack, si.dielectric_priority);
    }

    const float z1 = random<SampleDimension::eBSDF0>(prd->sampler);
    const float z2 = random<SampleDimension::eBSDF1>(prd->sampler);
    const float z3 = random<SampleDimension::eBSDF2>(prd->sampler);
    const float z4 = random<SampleDimension::eBSDF3>(prd->sampler);

    float4 xi = make_float4(z1, z2, z3, z4);
    BsdfSampleResult sample_data = bsdf_sample(si, xi);

    if (sample_data.event_type == BSDF_EVENT_ABSORB)
    {
        if (prd->depth == 0)
        {
            prd->firstEventType = EventType::eAbsorb;
        }
        // stop on absorb
        prd->throughput = make_float3(0.0f);
        return;
    }
    prd->specularBounce = ((sample_data.event_type & BSDF_EVENT_SPECULAR) != 0);

    if (prd->depth == 0)
    {
        if (sample_data.event_type & BSDF_EVENT_DIFFUSE)
        {
            prd->firstEventType = EventType::eDiffuse;
        }
        if (sample_data.event_type & BSDF_EVENT_GLOSSY)
        {
            prd->firstEventType = EventType::eSpecular;
        }
    }

    // estimatorMode 1 drops next-event estimation entirely and lets BSDF sampling
    // carry the whole integral. The two are independent unbiased estimators, so at
    // convergence they must agree; the difference between them measures estimator
    // inconsistency directly, which is the only reason the switch exists.
    const bool didNee = (params.estimatorMode == 0) &&
                        ((sample_data.event_type & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY)) != 0) &&
                        (params.scene.numLights > 0 || params.hasEnvMap);
    if (didNee)
    {
        prd->radiance += estimateDirectLighting(prd, si);
        if (prd->throughput.x == 0.0f && prd->throughput.y == 0.0f && prd->throughput.z == 0.0f)
        {
            // estimateDirectLighting() found a NaN and painted the pixel.
            return;
        }
    }
    prd->neeDone = didNee;

    // setup next path segment
    // Face normal oriented toward the incoming ray (wo)
    float3 faceNg = (dot(si.geometry_normal, si.wo) > 0.0f)
                  ? si.geometry_normal : -si.geometry_normal;
    // Update IOR stack on transmission
    if ((sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0)
    {
        if (entering)
            ior_stack_push(prd->iorStack, si.dielectric_priority, si.ior, (unsigned int)matId);
        else
            ior_stack_pop(prd->iorStack, si.dielectric_priority, (unsigned int)matId);
        prd->origin = offset_ray(si.position, -faceNg);
    }
    else
    {
        prd->origin = offset_ray(si.position, faceNg);
    }
    prd->lastBsdfPdf = (prd->specularBounce) ? 1.0f : sample_data.pdf;
    prd->dir = sample_data.wi;
    prd->throughput *= sample_data.bsdf_over_pdf;
}
