#include <optix.h>

#include "OptixRenderParams.h"
#include <cuda/helpers.h>
#include <cuda/random.h>

#include <sutil/vec_math.h>

#include <glm/glm.hpp>

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

static __forceinline__ __device__ void generateCameraRay(uint2 pixelIndex,
                                                         uint32_t& seed,
                                                         glm::float3& origin,
                                                         glm::float3& direction)
{
    const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

    float2 pixelPos = make_float2(pixelIndex.x + subpixel_jitter.x, pixelIndex.y + subpixel_jitter.y);

    float2 dimension = make_float2(params.image_width, params.image_height);
    float2 pixelNDC = (pixelPos / dimension) * 2.0f - 1.0f;

    glm::float4 clip{ pixelNDC.x, pixelNDC.y, 1.0f, 1.0f };
    glm::float4 viewSpace = params.clipToView * clip;

    glm::float4 wdir = params.viewToWorld * glm::float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);

    origin = params.viewToWorld * glm::float4(0.0f, 0.0f, 0.0f, 1.0f);
    direction = glm::normalize(wdir);
}

extern "C" __global__ void __raygen__rg()
{
    const int w = params.image_width;
    const uint3 launch_index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int subframe_index = params.subframe_index;

    unsigned int seed = tea<4>(launch_index.y * w + launch_index.x, subframe_index);

    float3 result = make_float3(0.0f);

    for (int sampleIdx = 0; sampleIdx < params.samples_per_launch; ++sampleIdx)
    {
        float3 ray_origin, ray_direction;
        glm::float3 rayO, rayD;
        generateCameraRay({ launch_index.x, launch_index.y }, seed, rayO, rayD);

        ray_origin = { rayO.x, rayO.y, rayO.z };
        ray_direction = { rayD.x, rayD.y, rayD.z };

        PerRayData prd;
        prd.rndSeed = seed;
        prd.radiance = make_float3(0.0f);
        prd.throughput = make_float3(1.0f);
        prd.inside = false;
        prd.depth = 0;
        prd.specularBounce = false;

        unsigned int u0, u1;
        packPointer(&prd, u0, u1);

        int depth = 0;
        while (depth < params.max_depth)
        {
            optixTrace(params.handle, ray_origin, ray_direction,
                       params.materialRayTmin, // Min intersection distance
                       1e16f, // Max intersection distance
                       0.0f, // rayTime -- used for motion blur
                       OptixVisibilityMask(255), // Specify always visible
                       OPTIX_RAY_FLAG_NONE,
                       RAY_TYPE_RADIANCE, // SBT offset   -- See SBT discussion
                       RAY_TYPE_COUNT, // SBT stride   -- See SBT discussion
                       RAY_TYPE_RADIANCE, // missSBTIndex -- See SBT discussion
                       u0, u1);

            ray_origin = prd.origin;
            ray_direction = prd.dir;

            if (depth > 3)
            {
                const float p = max(prd.throughput.x, max(prd.throughput.y, prd.throughput.z));
                if (rnd(prd.rndSeed) > p)
                {
                    break;
                }
                prd.throughput *= 1.0f / (p + 1e-5f);
            }

            if (dot(prd.throughput, prd.throughput) < 1e-5f)
            {
                break;
            }

            ++depth;
            prd.depth = depth;

            if (params.debug == 1)
                break;
        }
        result += prd.radiance;
    }

    result /= static_cast<float>(params.samples_per_launch);

    const unsigned int image_index = launch_index.y * params.image_width + launch_index.x;

    if (params.enableAccumulation && params.debug == 0)
    {
        float3 accum_color = result;
        if (params.subframe_index > 0)
        {
            const float a = 1.0f / static_cast<float>(params.subframe_index + 1);
            const float3 accum_color_prev = make_float3(params.accum[image_index]);
            accum_color = lerp(accum_color_prev, accum_color, a);
        }
        params.accum[image_index] = make_float4(accum_color, 1.0f);
        params.image[image_index] = make_float4(accum_color, 1.0f);
    }
    else
    {
        params.image[image_index] = make_float4(result, 1.0f);
    }
}

extern "C" __global__ void __miss__ms()
{
    MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    PerRayData* prd = getPRD();
    prd->radiance += prd->throughput * miss_data->bg_color;
    prd->throughput = make_float3(0.0f);
    prd->depth = params.max_depth;
}

__device__ glm::float3 interpolateAttrib(const glm::float3 attr1,
                                         const glm::float3 attr2,
                                         const glm::float3 attr3,
                                         const float2 bary)
{
    return attr1 * (1 - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

//  valid range of coordinates [-1; 1]
__device__ glm::float3 unpackNormal(uint32_t val)
{
    glm::float3 normal;
    normal.z = ((val & 0xfff00000) >> 20) / 511.99999f * 2.0f - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) / 511.99999f * 2.0f - 1.0f;
    normal.x = (val & 0x000003ff) / 511.99999f * 2.0f - 1.0f;

    return normal;
}

extern "C" __global__ void __closesthit__ch()
{
    PerRayData* prd = getPRD();

    const float2 barycentrics = optixGetTriangleBarycentrics();
    const unsigned int primitiveId = optixGetPrimitiveIndex();

    HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    const uint32_t i0 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 0)];
    const uint32_t i1 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 1)];
    const uint32_t i2 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 2)];

    const uint32_t baseVbOffset = hit_data->vertexOffset;

    glm::float3 N0 = unpackNormal(params.scene.vb[baseVbOffset + i0].normal);
    glm::float3 N1 = unpackNormal(params.scene.vb[baseVbOffset + i1].normal);
    glm::float3 N2 = unpackNormal(params.scene.vb[baseVbOffset + i2].normal);

    glm::float3 object_normal = glm::normalize(interpolateAttrib(N0, N1, N2, barycentrics));
    // float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );

    glm::float3 res = (object_normal + glm::float3(1.0f)) * 0.5f;
    // setPayload(make_float3(barycentrics, 1.0f));
    prd->radiance = make_float3(res.x, res.y, res.z);
}

static __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
{
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion(true);
}

extern "C" __global__ void __closesthit__light()
{
    PerRayData* prd = getPRD();
    HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const int32_t lightId = hit_data->lightId;
    const UniformLight& currLight = params.scene.lights[lightId];
    prd->radiance += prd->throughput * make_float3(currLight.color);
    prd->throughput = make_float3(0.0f);
    // stop tracing
    return;
}
