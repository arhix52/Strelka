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
#include <strelka/material/volume.h>

#include "optix_device_utils.h"
#include "shading/shading_common.h"

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

/// Where a next-event connection leaves from.
///
/// A surface offsets along the face the shadow ray actually departs through. A
/// fibre does not have one: the Chiang lobe has already paid for the crossing,
/// so a connection leaving through the strand has to start past it or the fibre
/// occludes itself and the dominant lobe is never connected to at all.
static __forceinline__ __device__ float3 shadowOrigin(const SurfaceInteraction& si,
                                                      float curveRadius,
                                                      float3 toLight)
{
    if (scattersThroughFibre(si) && curveRadius > 0.0f)
    {
        return fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius, toLight);
    }
    const float3 offsetNg =
        (dot(si.geometry_normal, toLight) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
    return offset_ray(si.position, offsetNg);
}

static __device__ float3 sampleLight(SamplerState& sampler,
                                         const UniformLight& light,
                                         const SurfaceInteraction& si,
                                         float curveRadius,
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

    // lightReachesShadingPoint() is `dot(N, L) > 0` for everything except a
    // fibre, where the hemisphere test is the wrong question -- see the note on
    // it in shading/shading_common.h.
    const bool lit = lightReachesShadingPoint(si, lightSampleData.L);
    const bool facing = (light.type == LIGHT_TYPE_POINT || light.type == LIGHT_TYPE_SPOT)
                            ? (lit && emitsLight(Li))
                            : (lit && -dot(lightSampleData.L, lightSampleData.normal) > 0.0f &&
                               emitsLight(Li));
    if (facing)
    {
        const bool occluded =
            traceOcclusion(params.handle, shadowOrigin(si, curveRadius, lightSampleData.L),
                           lightSampleData.L,
                           params.shadowRayTmin, // tmin
                           lightSampleData.distToLight // tmax
            );
        float visibility = occluded ? 0.0f : 1.0f;
        lightPdf = lightSampleData.pdf;
        // The cosine belongs here because bsdf_eval() returns f alone, unlike
        // bsdf_sample()'s bsdf_over_pdf which already carries it. See the note on
        // both result structs in bsdf_types.h.
        return visibility * Li * shadingCosine(si, lightSampleData.L);
    }

    return make_float3(0.0f);
}

static __device__ float3 sampleEnvLightNEE(SamplerState& sampler,
                                            const SurfaceInteraction& si,
                                            float curveRadius,
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

    // Check if direction is above the surface -- an identity for everything but
    // a fibre, which has no dark side.
    if (!lightReachesShadingPoint(si, dir))
        return make_float3(0.0f);

    // Trace shadow ray to infinity
    const bool occluded = traceOcclusion(
        params.handle,
        shadowOrigin(si, curveRadius, dir),
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

    return Li * shadingCosine(si, dir);
}

__device__ float3 estimateDirectLighting(SamplerState& sampler,
                                         const SurfaceInteraction& si,
                                         float curveRadius,
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
            const float3 r = sampleEnvLightNEE(sampler, si, curveRadius, toLight, lightPdf);
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
            const float3 r = sampleLight(sampler, currLight, si, curveRadius, toLight, lightPdf);
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
        const float3 r = sampleLight(sampler, currLight, si, curveRadius, toLight, lightPdf);
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
    /// glTF COLOR_0, interpolated. White when the mesh carries no colour set.
    float3 vertexColor;
    /// World-space radius of the strand at the hit. Zero for a triangle, and the
    /// one thing the fibre chord needs that a curve hit does not hand back.
    float curveRadius;
};

/// glTF COLOR_0 out of oka::Scene::Vertex.
///
/// The device-side `Vertex` in OptixRenderParams.h spells the last eight bytes
/// `pad0` / `pad1`, but the buffer it aliases is oka::Scene::Vertex, whose last
/// two words are `uv1` and `color`. Both structs are 32 bytes with the same
/// field offsets, and OptiXRender::createVertexBuffer uploads the scene struct
/// whole, so the colour really is there -- it is only spelled as padding.
/// Renaming those two fields is a hand-off; reading the bits is not.
static __forceinline__ __device__ float3 vertexColorOf(const Vertex& v)
{
    return unpack_vertex_color(__float_as_uint(v.pad1));
}

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
    float3 c0, c1, c2;
    // glTF TANGENT.w rides in bit 30 of the packed tangent. Handedness is a
    // per-mesh property in every exporter that writes it, so one vertex settles
    // it -- there is nothing sensible to interpolate.
    float tangentSign = 1.0f;

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

        // Vertex colour is not skinned and does not animate, so it is read from
        // the current frame even when the rest is motion-interpolated.
        c0 = vertexColorOf(v0_1);
        c1 = vertexColorOf(v1_1);
        c2 = vertexColorOf(v2_1);
        tangentSign = unpackTangentSign(v0_1.tangent);
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

        c0 = vertexColorOf(v0);
        c1 = vertexColorOf(v1);
        c2 = vertexColorOf(v2);
        tangentSign = unpackTangentSign(v0.tangent);
    }

    const float2 uvCoord = interpolateAttrib(uv0, uv1, uv2, barycentrics);

    const float3 worldPosition = optixTransformPointFromObjectToWorldSpace(interpolateAttrib(p0, p1, p2, barycentrics));
    const float3 object_normal = interpolateAttrib(n0, n1, n2, barycentrics);
    float3 worldNormal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    float3 geomNormal = cross(p1 - p0, p2 - p0);
    geomNormal = normalize(optixTransformNormalFromObjectToWorldSpace(geomNormal));
    const float3 worldTangent =
        normalize(optixTransformNormalFromObjectToWorldSpace(interpolateAttrib(t0, t1, t2, barycentrics)));
    // Without TANGENT.w the bitangent points the wrong way and every normal map
    // is mirrored along it -- bumps light from the opposite side.
    const float3 worldBinormal = cross(worldNormal, worldTangent) * tangentSign;

    SurfaceHitData res;
    res.normal = worldNormal;
    res.geom_normal = geomNormal;
    res.position = worldPosition;
    res.uv = uvCoord;
    res.worldTangent = worldTangent;
    res.worldBinormal = worldBinormal;
    res.vertexColor = interpolateAttrib(c0, c1, c2, barycentrics);
    res.curveRadius = 0.0f;
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
    const float u = optixGetCurveParameter();
    float3 hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint = optixTransformPointFromWorldToObjectSpace(hitPoint); // interpolators work in object space
    const float3 objectNormal = surfaceNormal(interpolator, u, hitPoint);
    float3 worldNormal = normalize(optixTransformNormalFromObjectToWorldSpace(objectNormal));
    const float3 worldTangent =
        normalize(optixTransformNormalFromObjectToWorldSpace(curveTangent(interpolator, u)));
    const float3 worldBinormal = cross(worldNormal, worldTangent);
    const float3 worldPosition = optixTransformPointFromObjectToWorldSpace(hitPoint);
    SurfaceHitData res;
    res.normal = worldNormal;
    res.geom_normal = worldNormal;
    res.position = worldPosition;
    res.uv = make_float2(0.5f, 0.5f);
    res.worldTangent = worldTangent;
    res.worldBinormal = worldBinormal;
    res.vertexColor = make_float3(1.0f);
    // The chord across the strand is measured in world units, and the
    // interpolator's radius is in object ones. Carrying the radial *vector*
    // through the transform rather than the scalar is what makes that survive an
    // instance transform with a scale on it -- and the object normal is already
    // the radial direction everywhere except the two flat endcaps, where the
    // chord degenerates to zero and fibre_exit() falls back to a surface offset.
    res.curveRadius =
        length(optixTransformVectorFromObjectToWorldSpace(objectNormal * interpolator.radius(u)));

    return res;
}

extern "C" __global__ void __closesthit__radiance()
{
    OptixPrimitiveType primType = optixGetPrimitiveType();

    PerRayData* prd = getPRD();
    HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const float3 ray_dir = optixGetWorldRayDirection();

    SurfaceHitData surfaceHit = {};
    const bool isCurveHit = (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE);
    if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE)
    {
        surfaceHit = fillTriangleGeomData(hit_data);
    }
    else if (isCurveHit)
    {
        surfaceHit = fillCurveGeomData(hit_data);
    }

    // Look up material from device buffer (indexed by materialId)
    const int32_t matId = hit_data->materialId;
    const MaterialParams& matParams = params.materials[matId];
    const cudaTextureObject_t* textures = &params.materialTextures[matId * MAX_MATERIAL_TEXTURES];

    // --- Absorption over the segment just travelled -------------------------
    //
    // The IOR stack already knows which medium the path is inside; it also
    // carries the material that medium came from, so the extinction is looked up
    // here rather than threaded through the payload. It has to happen before the
    // pop at the bottom of this program, and before emission and next-event
    // estimation, or everything seen through a dense medium keeps its own colour.
    //
    // volume.h was included by no .cu file at all before this, so the OptiX path
    // had no volumetric attenuation of any kind.
    {
        const unsigned int inside = ior_stack_current_material(prd->iorStack);
        if (inside != 0xFFFFFFFFu)
        {
            const MaterialParams& im = params.materials[inside];
            const float3 sigma_t = volume_extinction(im.attenuation_color, im.attenuation_distance,
                                                     kOptixVolumeModel);
            prd->throughput *= beer_lambert_transmittance(sigma_t, optixGetRayTmax());
        }
    }

    // Fill SurfaceInteraction from hit data, resolving textures, the uv
    // transform, the normal map, vertex colour and coverage on the way.
    SurfaceInteraction si;
    initSurfaceInteraction(si, matParams, textures, surfaceHit.position, surfaceHit.normal,
                           surfaceHit.geom_normal, surfaceHit.worldTangent,
                           surfaceHit.worldBinormal, surfaceHit.uv, ray_dir,
                           surfaceHit.vertexColor);

    // A strand shaded by the whole-fibre lobe: light crosses it in one event, so
    // neither the hemisphere tests nor the ray offsets below apply. Gated on the
    // geometry as well as the material because the chord needs a radius, and only
    // a curve hit has one.
    const bool isFibre = isCurveHit && scattersThroughFibre(si) && surfaceHit.curveRadius > 0.0f;
    const float curveRadius = isFibre ? surfaceHit.curveRadius : 0.0f;

    if (params.debug == 1)
    {
        prd->radiance = (si.geometry_normal + make_float3(1.0f)) * 0.5f;
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
        const float3 radiance =
            estimateDirectLighting(prd->sampler, si, curveRadius, toLight, lightPdf);
        if (isnan(radiance) || isnan(lightPdf))
        {
            // ERROR, terminate tracing
            prd->radiance = make_float3(10000.0f, 0.0f, 0.0f);
            prd->throughput = make_float3(0.0f);
            return;
        }

        // A fibre has no back side to reject: the Chiang lobe's TT term is light
        // that entered one side and left the other, and on a bright groom it is
        // four fifths of the albedo.
        const bool isNextEventValid =
            (isFibre || ((dot(toLight, si.shading_normal) > 0.0f) == si.front_face)) &&
            (lightPdf != 0.0f);
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
    // Update IOR stack on transmission.
    //
    // A fibre's transmission lobes do not put the path inside anything: the
    // strand is crossed within the one event, so there is no medium to enter and
    // no entry to match with an exit. Pushing here left every transmitted hair
    // path one level deeper than it came in, and a groom is thousands of hairs
    // deep.
    if ((sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0 && !isFibre)
    {
        // A thin-walled surface has no interior either, so crossing it does not
        // put the path inside anything. Pushing the stack anyway left a ray that
        // had gone through the front of a bubble believing it was inside glass,
        // so the far side read as an exit from a dense medium -- and every
        // grazing angle there is past the critical angle.
        if (!si.thin_walled)
        {
            if (entering)
                ior_stack_push(prd->iorStack, si.dielectric_priority, si.ior, (unsigned int)matId);
            else
                ior_stack_pop(prd->iorStack, si.dielectric_priority, (unsigned int)matId);
        }
        prd->origin = offset_ray(si.position, -faceNg);
    }
    else
    {
        prd->origin = offset_ray(si.position, faceNg);
    }
    prd->dir = sample_data.wi;
    if (isFibre)
    {
        // Both branches above assume a surface with an inside and an outside. A
        // strand has neither: the bounce leaves from wherever the crossing the
        // lobe has already accounted for comes out. Without this the ray hits the
        // far wall and buys a second whole-fibre event -- and a third, which is
        // what made an isolated strand's cross-section climb with depth instead
        // of going flat.
        prd->origin = fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius,
                                      normalize(prd->dir));
    }
    prd->lastBsdfPdf = (prd->specularBounce) ? 1.0f : sample_data.pdf;
    prd->throughput *= sample_data.bsdf_over_pdf;
}
