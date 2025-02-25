#include <optix.h>

#include "OptixRenderParams.h"
#include <cuda/helpers.h>
#include "RandomSampler.h"

#include <sutil/vec_math.h>
#include <sutil/Matrix.h>

#include <postprocessing/Utils.h>

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

__device__ void generateCameraRay(
    const uint2 pixelIndex, SamplerState& sampler, float3& origin, float3& direction)
{
    float2 subpixel_jitter =
        make_float2(random<SampleDimension::ePixelX>(sampler), random<SampleDimension::ePixelY>(sampler));

    float2 pixelPos = make_float2(pixelIndex.x + subpixel_jitter.x, pixelIndex.y + subpixel_jitter.y);

    float2 dimension = make_float2(params.image_width, params.image_height);
    float2 pixelNDC = (pixelPos / dimension) * 2.0f - 1.0f;

    float4 clip{ pixelNDC.x, pixelNDC.y, 1.0f, 1.0f };
    const sutil::Matrix4x4 clipToView(params.clipToView);
    float4 viewSpace = clipToView * clip;

    const sutil::Matrix4x4 viewToWorld(params.viewToWorld);
    float4 wdir = viewToWorld * make_float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);

    origin = make_float3(viewToWorld * make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    direction = normalize(make_float3(wdir));
}

__device__ float4 accumulate(float4* history,
                             const float3 value,
                             const uint32_t linearPixelIndex,
                             const float3 exposure,
                             const uint32_t subFrameIndex)
{
    // Accumulation
    float3 accumColor = value;
    if (subFrameIndex > 0)
    {
        const float a = 1.0f / static_cast<float>(subFrameIndex + 1);
        const float3 accumColorPrev = make_float3(history[linearPixelIndex]);
        const float3 exposure = params.exposure;
        // perform lerp in ldr and back to hdr back
        accumColor = inverseTonemap(lerp(tonemap(accumColorPrev, exposure), tonemap(accumColor, exposure), a), exposure);
    }
    history[linearPixelIndex] = make_float4(accumColor, 1.0f);
    return make_float4(accumColor, 1.0f);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 launch_index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 result = make_float3(0.0f);
    float3 diffuse = make_float3(0.0f);
    float4 diffuseOut = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    uint32_t diffuseSamples = 0;
    float3 specular = make_float3(0.0f);
    float4 specularOut = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    uint32_t specularSamples = 0;
    const uint32_t linearPixelIndex = launch_index.y * params.image_width + launch_index.x;

    for (uint32_t sampleIdx = 0; sampleIdx < params.samples_per_launch; ++sampleIdx)
    {
        PerRayData prd = {};
        prd.firstEventType = EventType::eUndef;
        prd.linearPixelIndex = linearPixelIndex;
        prd.sampleIndex = params.subframe_index + sampleIdx;

        prd.sampler = initSampler(launch_index.x, params.image_height - launch_index.y, prd.linearPixelIndex, prd.sampleIndex, params.maxSampleCount, 52u);

        prd.radiance = make_float3(0.0f);
        prd.throughput = make_float3(1.0f);
        prd.inside = false;
        prd.depth = 0;
        prd.specularBounce = false;
        prd.lastBsdfPdf = 0.0f;
        
        float3 ray_origin, ray_direction;

        const uint2 pixelCoord = make_uint2(launch_index.x, params.image_height - launch_index.y);
        generateCameraRay(pixelCoord, prd.sampler, ray_origin, ray_direction);

        unsigned int payload0, payload1;
        packPointer(&prd, payload0, payload1);

        while (prd.depth < params.max_depth)
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
                       payload0, payload1);

            ray_origin = prd.origin;
            ray_direction = prd.dir;

            if (prd.depth > 3)
            {
                const float p = max(prd.throughput.x, max(prd.throughput.y, prd.throughput.z));
                if (random<SampleDimension::eRussianRoulette>(prd.sampler) > p)
                {
                    break;
                }
                prd.throughput *= 1.0f / (p + 1e-5f);
            }

            if (dot(prd.throughput, prd.throughput) < 1e-5f)
            {
                break;
            }

            ++prd.depth;

            if (params.debug == 1)
                break;
            prd.sampler.depth++;
        }
        result += prd.radiance;

        if (prd.firstEventType == EventType::eDiffuse)
        {
            diffuse += prd.radiance;
            ++diffuseSamples;
        }
        if (prd.firstEventType == EventType::eSpecular)
        {
            specular += prd.radiance;
            ++specularSamples;
        }
    }

    result /= static_cast<float>(params.samples_per_launch);
    if (diffuseSamples > 0)
    {
        diffuse /= static_cast<float>(diffuseSamples);
        uint32_t prevSamplesCount = params.subframe_index > 0 ? params.diffuseCounter[linearPixelIndex] : 0;
        diffuseOut = accumulate(params.diffuse, diffuse, linearPixelIndex, params.exposure, prevSamplesCount);
        params.diffuseCounter[linearPixelIndex] = uint16_t(prevSamplesCount + diffuseSamples);
    }
    else
    {
        if (params.subframe_index == 0)
        {
            // need to reset history
            params.diffuse[linearPixelIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
            params.diffuseCounter[linearPixelIndex] = 0;
        }
        diffuseOut = params.diffuse[linearPixelIndex];
    }
    if (specularSamples > 0)
    {
        specular /= static_cast<float>(specularSamples);
        uint32_t prevSamplesCount = params.subframe_index > 0 ? params.specularCounter[linearPixelIndex] : 0;
        specularOut = accumulate(params.specular, specular, linearPixelIndex, params.exposure, prevSamplesCount);
        params.specularCounter[linearPixelIndex] = uint16_t(prevSamplesCount + specularSamples);
    }
    else
    {
        if (params.subframe_index == 0)
        {
            // need to reset history
            params.specular[linearPixelIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
            params.specularCounter[linearPixelIndex] = 0;
        }
        if (params.specularCounter[linearPixelIndex] > 0)
        {
            specularOut = params.specular[linearPixelIndex];
        }
        else
        {
            specularOut = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    if (params.debug == 2)
    {
        params.image[linearPixelIndex] = diffuseOut;
        return;
    }
    if (params.debug == 3)
    {
        params.image[linearPixelIndex] = specularOut;
        return;
    }

    if (params.enableAccumulation && params.debug == 0)
    {
        // Accumulation
        params.image[linearPixelIndex] = accumulate(params.accum, result, linearPixelIndex, params.exposure, params.subframe_index);
    }
    else
    {
        params.image[linearPixelIndex] = make_float4(result, 1.0f);
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

__device__ float3 interpolateAttrib(const float3 attr1,
                                         const float3 attr2,
                                         const float3 attr3,
                                         const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

//  valid range of coordinates [-1; 1]
__device__ float3 unpackNormal(uint32_t val)
{
    float3 normal;
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

    float3 N0 = unpackNormal(params.scene.vb[baseVbOffset + i0].normal);
    float3 N1 = unpackNormal(params.scene.vb[baseVbOffset + i1].normal);
    float3 N2 = unpackNormal(params.scene.vb[baseVbOffset + i2].normal);

    float3 object_normal = normalize(interpolateAttrib(N0, N1, N2, barycentrics));

    float3 res = (object_normal + make_float3(1.0f)) * 0.5f;
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
    const float3 rayDir = optixGetWorldRayDirection();
    const float3 hitPoint = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;
    const float3 lightNormal = calcLightNormal(currLight, hitPoint);
    if (-dot(rayDir, lightNormal) > 0.0f)
    {
        if (prd->depth == 0 || prd->specularBounce)
        {
            prd->radiance += prd->throughput * make_float3(currLight.color) * -dot(rayDir, lightNormal);
        }
        else
        {
            float lightPdf = getLightPdf(currLight, hitPoint, optixGetWorldRayOrigin()) / (params.scene.numLights);
            const float misWeight = misWeightBalance(prd->lastBsdfPdf, lightPdf);
            prd->radiance += prd->throughput * make_float3(currLight.color) * -dot(rayDir, lightNormal) * misWeight;
        }
    }
    prd->throughput = make_float3(0.0f);
    // stop tracing
    return;
}
