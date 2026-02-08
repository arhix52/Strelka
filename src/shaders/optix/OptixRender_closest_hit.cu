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

#include <strelka/material/bsdf.h>

static __forceinline__ __device__ float3 get_barycentrics()
{
    const float2 bary = optixGetTriangleBarycentrics();
    return make_float3(1.0f - bary.x - bary.y, bary.x, bary.y);
}

extern "C"
{
    __constant__ Params params;
}

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ PerRayData* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<PerRayData*>(unpackPointer(u0, u1));
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

static __forceinline__ __device__ float3 interpolateAttrib(const float3 attr1,
                                                           const float3 attr2,
                                                           const float3 attr3,
                                                           const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

static __forceinline__ __device__ float2 interpolateAttrib(const float2 attr1,
                                                           const float2 attr2,
                                                           const float2 attr3,
                                                           const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

// Clever offset_ray function from Ray Tracing Gems chapter 6
// Offsets the ray origin from current position p, along normal n (which must be geometric normal)
// so that no self-intersection can occur.
static __forceinline__ __device__ float3 offset_ray(const float3 p, const float3 n)
{
    static const float origin = 1.0f / 32.0f;
    static const float float_scale = 1.0f / 65536.0f;
    static const float int_scale = 256.0f;

    int3 of_i = make_int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = make_float3(__int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                             __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                             __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return make_float3(fabs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                       fabs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                       fabs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

//  valid range of coordinates [-1; 1]
static __forceinline__ __device__ float3 unpackNormal(uint32_t val)
{
    constexpr float scale = 1.0f / 256.0f;
    float3 normal;
    normal.z = ((val & 0xfff00000) >> 20) * scale - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) * scale - 1.0f;
    normal.x = (val & 0x000003ff) * scale - 1.0f;
    return normal;
}

//  valid range of coordinates [-10; 10]
static __forceinline__ __device__ float2 unpackUV(uint32_t val)
{
    float2 uv;
    uv.y = ((val & 0xffff0000) >> 16) / 16383.99999f * 20.0f - 10.0f;
    uv.x = (val & 0x0000ffff) / 16383.99999f * 20.0f - 10.0f;

    return uv;
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
    case 0:
        if (params.rectLightSamplingMethod == 0)
        {
            lightSampleData = SampleRectLightUniform(light, uv, si.position);
        }
        else
        {
            lightSampleData = SampleRectLight(light, uv, si.position);
        }
        break;
    case 2:
        lightSampleData = SampleSphereLight(light, uv, si.position);
        break;

    case 3:
        lightSampleData = SampleDistantLight(light, uv, si.position);
        break;
    }

    toLight = lightSampleData.L;
    float3 Li = make_float3(light.color);

    if (dot(si.shading_normal, lightSampleData.L) > 0.0f && -dot(lightSampleData.L, lightSampleData.normal) > 0.0 && all(Li))
    {
        const bool occluded =
            traceOcclusion(params.handle, offset_ray(si.position, si.geometry_normal), lightSampleData.L,
                           params.shadowRayTmin, // tmin
                           lightSampleData.distToLight // tmax
            );
        float visibility = occluded ? 0.0f : 1.0f;
        lightPdf = lightSampleData.pdf;
        return visibility * Li * saturate(dot(si.shading_normal, lightSampleData.L));
    }

    return make_float3(0.0f);
}

__device__ float3 estimateDirectLighting(SamplerState& sampler,
                                         const SurfaceInteraction& si,
                                         float3& toLight,
                                         float& lightPdf)
{
    const float u = random<SampleDimension::eLightId>(sampler);
    const uint32_t lightId = (uint32_t)(params.scene.numLights * u);
    const float lightSelectionPdf = 1.0f / params.scene.numLights;
    const UniformLight& currLight = params.scene.lights[lightId];
    const float3 r = sampleLight(sampler, currLight, si, toLight, lightPdf);
    lightPdf *= lightSelectionPdf;
    return r;
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

static __forceinline__ __device__ SurfaceHitData fillTriangleGeomData(const HitGroupData* hit_data, const bool inside)
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
    geomNormal *= (inside ? -1.0f : 1.0f);
    worldNormal *= (inside ? -1.0f : 1.0f);

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

static __forceinline__ __device__ SurfaceHitData fillCurveGeomData(const HitGroupData* hit_data, const bool inside)
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
    worldNormal *= (inside ? -1.0f : 1.0f);
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
    const bool isInside = prd->inside;
    HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const float3 ray_dir = optixGetWorldRayDirection();

    SurfaceHitData surfaceHit;
    if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE)
    {
        surfaceHit = fillTriangleGeomData(hit_data, isInside);
    }
    else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE)
    {
        surfaceHit = fillCurveGeomData(hit_data, isInside);
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
    si.front_face = !isInside;

    // Resolve material textures and parameters into the SurfaceInteraction
    bsdf_init(si, hit_data->materialParams, hit_data->textures);

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

        const bool isNextEventValid = ((dot(toLight, si.shading_normal) > 0.0f) != isInside) && (lightPdf != 0.0f);
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
                const float misWeight = misWeightBalance(lightPdf, evalData.pdf);
                prd->radiance += prd->throughput * radianceOverPdf * misWeight * evalData.bsdf;
            }
        }
    }

    // setup next path segment
    // flip inside/outside on transmission
    if ((sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0)
    {
        prd->inside = !prd->inside;
        prd->origin = offset_ray(si.position, -si.geometry_normal);
    }
    else
    {
        prd->origin = offset_ray(si.position, si.geometry_normal);
    }
    prd->lastBsdfPdf = (prd->specularBounce) ? 1.0f : sample_data.pdf;
    prd->dir = sample_data.wi;
    prd->throughput *= sample_data.bsdf_over_pdf;
}
