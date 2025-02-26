#include <optix.h>

#include <cuda.h>

#include "OptixRenderParams.h"
#include <cuda/helpers.h>
#include <cuda/curve.h>

#include "RandomSampler.h"

#include <sutil/Matrix.h>
#include <sutil/vec_math.h>
#include <sutil/vec_math_adv.h>

#include "Lights.h"

#define TEX_SUPPORT_NO_VTABLES
#define TEX_SUPPORT_NO_DUMMY_SCENEDATA
#include "texture_support_cuda.h" // texture runtime

#include <mi/neuraylib/target_code_types.h>

typedef mi::neuraylib::Material_expr_function Mat_expr_func;
typedef mi::neuraylib::Bsdf_init_function Bsdf_init_func;
typedef mi::neuraylib::Bsdf_sample_function Bsdf_sample_func;
typedef mi::neuraylib::Bsdf_evaluate_function Bsdf_evaluate_func;
typedef mi::neuraylib::Shading_state_material Mdl_state;
//
// Declarations of generated MDL functions
//
extern "C" __device__ Bsdf_init_func mdlcode_init;
extern "C" __device__ Bsdf_sample_func mdlcode_sample;
extern "C" __device__ Bsdf_evaluate_func mdlcode_evaluate;
// extern "C" __device__ Mat_expr_func      mdlcode_thin_walled;

//
// Functions needed by texture runtime when this file is compiled with Clang
//

extern "C" __device__ __inline__ void __itex2D_float(float* retVal, cudaTextureObject_t texObject, float x, float y)
{
    float4 tmp;
    asm volatile("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
                 : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                 : "l"(texObject), "f"(x), "f"(y));
    *retVal = (float)(tmp.x);
}

extern "C" __device__ __inline__ void __itex2D_float4(float4* retVal, cudaTextureObject_t texObject, float x, float y)
{
    float4 tmp;
    asm volatile("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
                 : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                 : "l"(texObject), "f"(x), "f"(y));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itex2DGrad_float4(
    float4* retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
    float4 tmp;
    asm volatile("tex.grad.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], {%7, %8}, {%9, %10};"
                 : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                 : "l"(texObject), "f"(x), "f"(y), "f"(dPdx.x), "f"(dPdx.y), "f"(dPdy.x), "f"(dPdy.y));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itex2DLod_float4(
    float4* retVal, cudaTextureObject_t texObject, float x, float y, float level)
{
    float4 tmp;
    asm volatile("tex.level.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
                 : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                 : "l"(texObject), "f"(x), "f"(y), "f"(level));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itex3D_float(float* retVal, cudaTextureObject_t texObject, float x, float y, float z)
{
    float4 tmp;
    asm volatile("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
                 : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                 : "l"(texObject), "f"(x), "f"(y), "f"(z));
    *retVal = (float)(tmp.x);
}

extern "C" __device__ __inline__ void __itex3D_float4(
    float4* retVal, cudaTextureObject_t texObject, float x, float y, float z)
{
    float4 tmp;
    asm volatile("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
                 : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                 : "l"(texObject), "f"(x), "f"(y), "f"(z));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itexCubemap_float4(
    float4* retVal, cudaTextureObject_t texObject, float x, float y, float z)
{
    float4 tmp;
    asm volatile("tex.cube.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
                 : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                 : "l"(texObject), "f"(x), "f"(y), "f"(z));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

static __forceinline__ __device__ float3 get_barycentrics()
{
    const float2 bary = optixGetTriangleBarycentrics();
    return make_float3(1.0f - bary.x - bary.y, bary.x, bary.y);
}

// Implementation of scene_data_lookup_float4().
extern "C" __device__ void scene_data_lookup_float4(
    float                                  result[4],
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    float                                  default_value[4],
    bool                                   uniform_lookup)
{
    if (scene_data_id == 0) {
        result[0] = default_value[0];
        result[1] = default_value[1];
        result[2] = default_value[2];
        result[3] = default_value[3];
        return;
    }


    result[0] = state->text_coords[0].x;
    result[1] = state->text_coords[0].y;
    result[2] = 0.0f;
    result[3] = 0.0f;
}

// Implementation of scene_data_lookup_float2().
extern "C" __device__ void scene_data_lookup_float2(
    float                                  result[2],
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    float                                  default_value[2],
    bool                                   uniform_lookup)
{
    float res4[4];
    float def_val4[4] = { default_value[0], default_value[1], 0.0f, 0.0f };
    scene_data_lookup_float4(
        res4,
        self_base,
        state,
        scene_data_id,
        def_val4,
        uniform_lookup);
    result[0] = res4[0];
    result[1] = res4[1];
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
    unsigned int occluded = 0u;
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
               0.0f, // rayTime
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
                                         const Mdl_state& state,
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
            lightSampleData = SampleRectLightUniform(light, uv, state.position);
        }
        else
        {
            lightSampleData = SampleRectLight(light, uv, state.position);
        }
        break;
        // case 1:
        //     lightSampleData = SampleDiscLight(light, float2(rand(rngState), rand(rngState)), state.position);
        //     break;
    case 2:
        lightSampleData = SampleSphereLight(light, uv, state.position);
        break;

    case 3:
        lightSampleData = SampleDistantLight(light, uv, state.position);
        break;
    }

    toLight = lightSampleData.L;
    float3 Li = make_float3(light.color);

    if (dot(state.normal, lightSampleData.L) > 0.0f && -dot(lightSampleData.L, lightSampleData.normal) > 0.0 && all(Li))
    {
        // Ray shadowRay
        const bool occluded =
            traceOcclusion(params.handle, offset_ray(state.position, state.geom_normal), lightSampleData.L,
                           params.shadowRayTmin, // tmin
                           lightSampleData.distToLight // tmax
            );
        float visibility = occluded ? 0.0f : 1.0f;
        lightPdf = lightSampleData.pdf;
        return visibility * Li * saturate(dot(state.normal, lightSampleData.L));
    }

    return make_float3(0.0f);
}

__device__ float3 estimateDirectLighting(SamplerState& sampler,
                                         const Mdl_state& state,
                                         float3& toLight,
                                         float& lightPdf)
{
    const float u = random<SampleDimension::eLightId>(sampler);
    const uint32_t lightId = (uint32_t)(params.scene.numLights * u);
    const float lightSelectionPdf = 1.0f / params.scene.numLights;
    const UniformLight& currLight = params.scene.lights[lightId];
    const float3 r = sampleLight(sampler, currLight, state, toLight, lightPdf);
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

// Compute surface normal of cubic pimitive in world space.
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
    float3 text_coords[2];
    float3 worldTangent[2];
    float3 worldBinormal[2];
};

static __forceinline__ __device__ SurfaceHitData fillTriangleGeomData(const HitGroupData* hit_data, const bool inside)
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const unsigned int primitiveId = optixGetPrimitiveIndex();
    const uint32_t i0 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 0)];
    const uint32_t i1 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 1)];
    const uint32_t i2 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 2)];

    const uint32_t baseVbOffset = hit_data->vertexOffset;

    const Vertex v0 = params.scene.vb[baseVbOffset + i0];
    const Vertex v1 = params.scene.vb[baseVbOffset + i1];
    const Vertex v2 = params.scene.vb[baseVbOffset + i2];

    const float3 p0 = v0.position;
    const float3 p1 = v1.position;
    const float3 p2 = v2.position;

    const float3 n0 = unpackNormal(v0.normal);
    const float3 n1 = unpackNormal(v1.normal);
    const float3 n2 = unpackNormal(v2.normal);

    const float3 t0 = unpackNormal(v0.tangent);
    const float3 t1 = unpackNormal(v1.tangent);
    const float3 t2 = unpackNormal(v2.tangent);

    const float2 uv0 = unpackUV(v0.uv);
    const float2 uv1 = unpackUV(v1.uv);
    const float2 uv2 = unpackUV(v2.uv);

    const float2 uvCoord = interpolateAttrib(uv0, uv1, uv2, barycentrics);
    const float3 text_coords = make_float3(uvCoord.x, uvCoord.y, 0.0f);

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
    res.text_coords[0] = text_coords;
    res.text_coords[1] = make_float3(0.0f);
    res.worldTangent[0] = worldTangent;
    res.worldBinormal[0] = worldBinormal;
    res.worldTangent[1] = worldTangent;
    res.worldBinormal[1] = worldBinormal;
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
    res.text_coords[0] = make_float3(0.5f, 0.5f, 0.5f);
    res.text_coords[1] = make_float3(0.5f);
    res.worldTangent[0] = worldTangent;
    res.worldBinormal[0] = worldBinormal;
    res.worldTangent[1] = worldTangent;
    res.worldBinormal[1] = worldBinormal;

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
    // setup MDL state
    // 16 - predefined value, that should match mdl setup
    // set_option("num_texture_results", "16")
    float4 texture_results[16] = {};
    Mdl_state state;
    state.position = surfaceHit.position;
    state.normal = surfaceHit.normal;
    state.geom_normal = surfaceHit.geom_normal;
    // Currently supports only 2 set of textures
    // see: set_option("num_texture_spaces", "2")
    state.text_coords = surfaceHit.text_coords;
    state.tangent_u = surfaceHit.worldTangent;
    state.tangent_v = surfaceHit.worldBinormal;
    // fill common state
    state.animation_time = 0.0f;
    state.text_results = texture_results;
    state.ro_data_segment = (const char*)hit_data->roData;
    state.world_to_object = (float4*)&hit_data->world_to_object;
    state.object_to_world = (float4*)&hit_data->object_to_world;
    state.object_id = 0;
    state.meters_per_scene_unit = 1.0f;

    const float3 ior1 = (isInside) ? make_float3(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR) : make_float3(1.0f); // material ->
                                                                                                          // air
    const float3 ior2 = (isInside) ? make_float3(1.0f) : make_float3(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);

    mi::neuraylib::Resource_data res_data = { nullptr, (Texture_handler*)hit_data->resHandler }; // TODO

    mdlcode_init(&state, &res_data, (const char*)hit_data->argData);

    if (params.debug == 1)
    {
        prd->radiance = (state.normal + make_float3(1.0f)) * 0.5f;
        return;
    }

    const float z1 = random<SampleDimension::eBSDF0>(prd->sampler);
    const float z2 = random<SampleDimension::eBSDF1>(prd->sampler);
    const float z3 = random<SampleDimension::eBSDF2>(prd->sampler);
    const float z4 = random<SampleDimension::eBSDF3>(prd->sampler);

    mi::neuraylib::Bsdf_sample_data sample_data;
    sample_data.ior1 = ior1;
    sample_data.ior2 = ior2;
    sample_data.k1 = -ray_dir;
    sample_data.xi = make_float4(z1, z2, z3, z4);

    mdlcode_sample(&sample_data, &state, &res_data, (const char*)hit_data->argData);

    if (sample_data.event_type == mi::neuraylib::BSDF_EVENT_ABSORB)
    {
        if (prd->depth == 0)
        {
            prd->firstEventType = EventType::eAbsorb;
        }
        // stop on absorb
        prd->throughput = make_float3(0.0f);
        return;
    }
    prd->specularBounce = ((sample_data.event_type & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0);

    if (prd->depth == 0)
    {
        if (sample_data.event_type & mi::neuraylib::BSDF_EVENT_DIFFUSE)
        {
            prd->firstEventType = EventType::eDiffuse;
        }
        if (sample_data.event_type & mi::neuraylib::BSDF_EVENT_GLOSSY)
        {
            prd->firstEventType = EventType::eSpecular;
        }
    }

    if (sample_data.event_type & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY))
    {
        float3 toLight; // return value for estimateDirectLighting()
        float lightPdf = 0.0f; // return value for estimateDirectLighting()
        const float3 radiance = estimateDirectLighting(prd->sampler, state, toLight, lightPdf);
        if (isnan(radiance) || isnan(lightPdf))
        {
            // ERROR, terminate tracing;
            prd->radiance = make_float3(10000.0f, 0.0f, 0.0f);
            prd->throughput = make_float3(0.0f);
            return;
        }

        const bool isNextEventValid = ((dot(toLight, state.normal) > 0.0f) != isInside) && (lightPdf != 0.0f);
        if (isNextEventValid)
        {
            mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> evalData = {};
            evalData.ior1 = ior1; // IOR current medium
            evalData.ior2 = ior2; // IOR other side
            evalData.k1 = -ray_dir; // outgoing direction
            evalData.k2 = toLight; // incoming direction
            evalData.bsdf_diffuse = make_float3(0.0f);
            evalData.bsdf_glossy = make_float3(0.0f);

            mdlcode_evaluate(&evalData, &state, &res_data, (const char*)hit_data->argData);

            if (isnan(evalData.bsdf_diffuse) || isnan(evalData.bsdf_glossy))
            {
                // ERROR, terminate tracing;
                prd->radiance = make_float3(10000.0f, 0.0f, 0.0f);
                prd->throughput = make_float3(0.0f);
                return;
            }

            // compute lighting for this light
            if (evalData.pdf > 0.0f)
            {
                const float3 radianceOverPdf = radiance / lightPdf;
                const float misWeight = misWeightBalance(lightPdf, evalData.pdf);
                prd->radiance += prd->throughput * radianceOverPdf * misWeight * (evalData.bsdf_diffuse + evalData.bsdf_glossy);
            }
        }
    }

    // setup next path segment
    // flip inside/outside on transmission
    if ((sample_data.event_type & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
    {
        prd->inside = !prd->inside;
        prd->origin = offset_ray(state.position, -state.geom_normal);
    }
    else
    {
        prd->origin = offset_ray(state.position, state.geom_normal);
    }
    // MDL returns pdf = 0.0 for specular (it should be infinite)
    prd->lastBsdfPdf = (prd->specularBounce) ? 1.0f : sample_data.pdf;
    prd->dir = sample_data.k2;
    prd->throughput *= sample_data.bsdf_over_pdf;
}
