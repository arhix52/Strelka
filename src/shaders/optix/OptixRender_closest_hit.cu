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

#include <postprocessing/Guides.h>

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

static __device__ float3 sampleLight(SamplerState& sampler,
                                         const UniformLight& light,
                                         const SurfaceInteraction& si,
                                         float3& toLight,
                                         float& lightPdf)
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
    case LIGHT_TYPE_POINT:
    case LIGHT_TYPE_SPOT:
        lightSampleData = SamplePointLight(light, uv, si.position);
        break;
    }

    toLight = lightSampleData.L;
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
        const bool occluded =
            traceOcclusion(params.handle, offset_ray(si.position, si.geometry_normal), lightSampleData.L,
                           params.shadowRayTmin, // tmin
                           lightSampleData.distToLight // tmax
            );
        float visibility = occluded ? 0.0f : 1.0f;
        lightPdf = lightSampleData.pdf;
        // The cosine belongs here because bsdf_eval() returns f alone, unlike
        // bsdf_sample()'s bsdf_over_pdf which already carries it. See the note on
        // both result structs in bsdf_types.h.
        return visibility * Li * saturate(dot(si.shading_normal, lightSampleData.L));
    }

    return make_float3(0.0f);
}

static __device__ float3 sampleEnvLightNEE(SamplerState& sampler,
                                            const SurfaceInteraction& si,
                                            float3& toLight,
                                            float& lightPdf)
{
    const float2 xi = make_float2(
        random<SampleDimension::eLightPointX>(sampler),
        random<SampleDimension::eLightPointY>(sampler));

    float envPdf = 0.0f;
    float3 dir = sampleEnvMap(xi,
                              params.envCdfX, params.envCdfY,
                              params.envMapWidth, params.envMapHeight,
                              params.envMapRotation,
                              envPdf);

    toLight = dir;
    lightPdf = envPdf;

    if (envPdf <= 0.0f)
        return make_float3(0.0f);

    // Check if direction is above the surface
    if (dot(si.shading_normal, dir) <= 0.0f)
        return make_float3(0.0f);

    // Trace shadow ray to infinity
    const bool occluded = traceOcclusion(
        params.handle,
        offset_ray(si.position, si.geometry_normal),
        dir,
        params.shadowRayTmin,
        1e16f);

    if (occluded)
        return make_float3(0.0f);

    // Evaluate env map radiance at sampled direction
    const float2 uv = dirToEnvUV(dir, params.envMapRotation);
    const float4 envSample = tex2D<float4>(params.envMapTexture, uv.x, uv.y);
    float3 Li = make_float3(envSample.x, envSample.y, envSample.z);
    Li *= params.envMapIntensity * params.envMapColorTint;

    return Li * fmaxf(dot(si.shading_normal, dir), 0.0f);
}

__device__ float3 estimateDirectLighting(SamplerState& sampler,
                                         const SurfaceInteraction& si,
                                         float3& toLight,
                                         float& lightPdf)
{
    if (params.hasEnvMap)
    {
        const float u = random<SampleDimension::eLightId>(sampler);

        if (params.scene.numLights == 0 || u >= 0.5f)
        {
            // Sample environment map
            const float selectionPdf = (params.scene.numLights > 0) ? 0.5f : 1.0f;
            const float3 r = sampleEnvLightNEE(sampler, si, toLight, lightPdf);
            lightPdf *= selectionPdf;
            return r;
        }
        else
        {
            // Sample local light (remap u from [0, 0.5) to [0, 1))
            const float remappedU = u * 2.0f;
            const uint32_t lightId = selectLightIndex(remappedU, params.scene.numLights);
            const float lightSelectionPdf = 0.5f / params.scene.numLights;
            const UniformLight& currLight = params.scene.lights[lightId];
            const float3 r = sampleLight(sampler, currLight, si, toLight, lightPdf);
            lightPdf *= lightSelectionPdf;
            return r;
        }
    }
    else
    {
        // A scene with neither an environment nor an analytic light has nothing to
        // connect to. This used to fall through and divide by numLights == 0, then
        // read lights[0] off a null device pointer -- mLightBuffer is constructed
        // empty, so the pointer really is null rather than merely unpopulated.
        // Returning a zero contribution with a zero pdf is what the caller already
        // handles for a light that happens to face away.
        if (params.scene.numLights == 0)
        {
            toLight = make_float3(0.0f);
            lightPdf = 0.0f;
            return make_float3(0.0f);
        }
        const float u = random<SampleDimension::eLightId>(sampler);
        const uint32_t lightId = selectLightIndex(u, params.scene.numLights);
        const float lightSelectionPdf = 1.0f / params.scene.numLights;
        const UniformLight& currLight = params.scene.lights[lightId];
        const float3 r = sampleLight(sampler, currLight, si, toLight, lightPdf);
        lightPdf *= lightSelectionPdf;
        return r;
    }
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

/// Where this triangle hit was in the world one frame ago.
///
/// `vb_prev` is the previous frame's vertex buffer -- the same data motion blur
/// uses as its first key -- so a deforming or skinned mesh reprojects correctly.
/// What it does *not* carry is a rigid instance transform that moved between
/// frames: the previous frame's instance matrices are not on the device, so the
/// previous object-space position is put through the current transform. Camera
/// motion is exact either way, and camera motion is what a still scene's guides
/// are made of. Returns false when there is nothing better than "it did not
/// move" to say.
static __forceinline__ __device__ bool previousTriangleWorldPosition(const HitGroupData* hit_data, float3& outPosition)
{
    if (params.scene.vb_prev == nullptr)
    {
        return false;
    }
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const unsigned int primitiveId = optixGetPrimitiveIndex();
    const uint32_t i0 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 0)];
    const uint32_t i1 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 1)];
    const uint32_t i2 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 2)];
    const uint32_t base = hit_data->vertexOffset;
    const float3 objectPos = interpolateAttrib(params.scene.vb_prev[base + i0].position,
                                               params.scene.vb_prev[base + i1].position,
                                               params.scene.vb_prev[base + i2].position, barycentrics);
    outPosition = optixTransformPointFromObjectToWorldSpace(objectPos);
    return true;
}

/// Fill in the pixel's guide record from the surface being shaded.
///
/// Depth and motion are written first and separately, because unlike albedo or
/// roughness they belong to the pixel rather than to whatever surface the
/// material guides were eventually taken from. A specular primary hit hands its
/// material guides to the surface it reflects, and that surface sits somewhere
/// else on screen -- reprojecting the pixel by *its* motion is not a small
/// error.
static __forceinline__ __device__ void writeSurfaceGuide(const HitGroupData* hit_data,
                                                         PerRayData* prd,
                                                         const SurfaceInteraction& si,
                                                         const float3 worldPosition,
                                                         const bool isTriangle)
{
    const uint32_t pixelIndex = prd->linearPixelIndex;
    if (prd->depth == 0)
    {
        float3 prevPosition = worldPosition;
        if (isTriangle)
        {
            previousTriangleWorldPosition(hit_data, prevPosition);
        }
        const float2 motion = guideScreenMotion(params, make_float4(prevPosition, 1.0f), prd->pixelSample);
        params.aov[pixelIndex].depth = guideViewDepth(params, worldPosition);
        params.aov[pixelIndex].motionX = motion.x;
        params.aov[pixelIndex].motionY = motion.y;
    }

    if (oka::guides::shouldWriteGuide(true, prd->aovDone, params.guidePrimaryHit, prd->depth, si.roughness))
    {
        const AovSample previous = params.aov[pixelIndex];
        AovSample a;
        // Metals put their colour in the specular lobe and have no diffuse one.
        const float3 base = si.albedo;
        a.diffuseAlbedo = base * (1.0f - si.metallic);
        a.specularAlbedo = lerp(make_float3(0.04f), base, si.metallic);
        a.normal = si.shading_normal;
        a.roughness = si.roughness;
        // Taken from the block above, which wrote them for the primary surface
        // whatever this one is.
        a.depth = previous.depth;
        a.motionX = previous.motionX;
        a.motionY = previous.motionY;
        a.specularHitDistance = 0.0f;
        a.reactive = oka::guides::reactiveFor(prd->depth);
        a.pad2 = 0.0f;
        params.aov[pixelIndex] = a;
        prd->aovDone = true;
    }

    // What the specular lobe of the primary hit is looking at. A denoiser takes
    // this separately so it can reproject a reflection at the depth of the thing
    // being reflected rather than at the mirror's own. `specularBounce` still
    // describes the previous bounce here -- this program has not overwritten it
    // yet -- which is exactly the question being asked.
    if (prd->depth == 1 && prd->specularBounce)
    {
        params.aov[pixelIndex].specularHitDistance = optixGetRayTmax();
    }
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

    if (prd->writeAov && params.aov != nullptr)
    {
        writeSurfaceGuide(hit_data, prd, si, surfaceHit.position, primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE);
    }

    if (params.debug == (uint32_t)DebugMode::eNormal)
    {
        prd->radiance = (si.geometry_normal + make_float3(1.0f)) * 0.5f;
        return;
    }
    if (params.debug == (uint32_t)DebugMode::eMotionBlur)
    {
        // Red is where in the shutter this path was sampled; green is how far
        // this vertex moved between the two motion keys, so a mesh that deforms
        // and a mesh that only translates look different.
        float3 prevPosition = surfaceHit.position;
        const bool moved = (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE) &&
                           previousTriangleWorldPosition(hit_data, prevPosition);
        prd->radiance = make_float3(
            optixGetRayTime(), moved ? saturate(length(surfaceHit.position - prevPosition) * 10.0f) : 0.0f, 0.0f);
        return;
    }

    // Add emission
    if (si.emission.x > 0.0f || si.emission.y > 0.0f || si.emission.z > 0.0f)
    {
        prd->radiance += prd->throughput * si.emission;
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

    if (sample_data.event_type & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY))
    {
        float3 toLight; // return value for estimateDirectLighting()
        float lightPdf = 0.0f; // return value for estimateDirectLighting()
        const float3 radiance = estimateDirectLighting(prd->sampler, si, toLight, lightPdf);
        if (isnan(radiance) || isnan(lightPdf))
        {
            // ERROR, terminate tracing
            prd->radiance = make_float3(10000.0f, 0.0f, 0.0f);
            prd->throughput = make_float3(0.0f);
            return;
        }

        const bool isNextEventValid = ((dot(toLight, si.shading_normal) > 0.0f) == si.front_face) && (lightPdf != 0.0f);
        if (isNextEventValid)
        {
            BsdfEvalResult evalData = bsdf_eval(si, toLight);

            if (isnan(evalData.bsdf) || isnan(evalData.pdf))
            {
                // ERROR, terminate tracing
                prd->radiance = make_float3(10000.0f, 0.0f, 0.0f);
                prd->throughput = make_float3(0.0f);
                return;
            }

            // compute lighting for this light
            if (evalData.pdf > 0.0f)
            {
                const float3 radianceOverPdf = radiance / lightPdf;
                const float misWeight = computeMisWeight(lightPdf, evalData.pdf, params.misHeuristic);
                prd->radiance += prd->throughput * radianceOverPdf * misWeight * evalData.bsdf;
            }
        }
    }

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
