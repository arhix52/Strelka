#include <optix.h>

#include <OptixRenderParams.h>
#include <cuda_helpers/helpers.h>
#include <random.h>

#include <sutil/vec_math.h>
#include <sutil/Matrix.h>

#include <postprocessing/Utils.h>
#include <env_light.h>

#include "optix_device_utils.h"

extern "C"
{
    __constant__ Params params;
}

// Concentric disk mapping (Shirley & Chiu 1997)
__device__ float2 concentricDiskSample(float u1, float u2)
{
    float2 offset = make_float2(2.0f * u1 - 1.0f, 2.0f * u2 - 1.0f);
    if (offset.x == 0.0f && offset.y == 0.0f)
        return make_float2(0.0f, 0.0f);

    float theta, r;
    if (fabsf(offset.x) > fabsf(offset.y))
    {
        r = offset.x;
        theta = (M_PIf / 4.0f) * (offset.y / offset.x);
    }
    else
    {
        r = offset.y;
        theta = (M_PIf / 2.0f) - (M_PIf / 4.0f) * (offset.x / offset.y);
    }
    return make_float2(r * cosf(theta), r * sinf(theta));
}

// Sample regular polygon aperture (blades >= 3)
__device__ float2 samplePolygonAperture(float u1, float u2, int blades)
{
    float sectorAngle = 2.0f * M_PIf / (float)blades;
    int sector = (int)(u1 * blades);
    if (sector >= blades) sector = blades - 1;
    float u = u1 * blades - (float)sector;

    // Sample triangle: center to two adjacent vertices
    float su = sqrtf(u);
    float bary0 = 1.0f - su;
    float bary1 = u2 * su;

    float angle0 = sectorAngle * sector;
    float angle1 = sectorAngle * (sector + 1);
    float cosA0 = cosf(angle0), sinA0 = sinf(angle0);
    float cosA1 = cosf(angle1), sinA1 = sinf(angle1);

    // Vertices on unit circle; interpolate from center (0,0)
    float x = bary1 * cosA0 + (1.0f - bary0 - bary1) * cosA1;
    float y = bary1 * sinA0 + (1.0f - bary0 - bary1) * sinA1;
    return make_float2(x, y);
}

__device__ float2 sampleAperture(SamplerState& sampler)
{
    float u1 = random<SampleDimension::eLensU>(sampler);
    float u2 = random<SampleDimension::eLensV>(sampler);

    float2 p;
    if (params.apertureBlades < 3)
    {
        p = concentricDiskSample(u1, u2);
    }
    else
    {
        p = samplePolygonAperture(u1, u2, params.apertureBlades);
    }

    // Blade rotation
    if (params.bladeRotation != 0.0f)
    {
        float cosR = cosf(params.bladeRotation);
        float sinR = sinf(params.bladeRotation);
        p = make_float2(p.x * cosR - p.y * sinR, p.x * sinR + p.y * cosR);
    }

    // Anamorphic ratio: stretch Y
    p.y *= params.anamorphicRatio;

    return p;
}

__device__ void generateCameraRay(
    const uint2 pixelIndex, SamplerState& sampler, float3& origin, float3& direction)
{
    float2 subpixel_jitter =
        make_float2(random<SampleDimension::ePixelX>(sampler), random<SampleDimension::ePixelY>(sampler));

    float2 pixelPos = make_float2(pixelIndex.x + subpixel_jitter.x, pixelIndex.y + subpixel_jitter.y);

    float2 dimension = make_float2(params.image_width, params.image_height);
    float2 pixelNDC = (pixelPos / dimension) * 2.0f - 1.0f;

    // Lens shift
    pixelNDC.x += params.shiftX * 2.0f;
    pixelNDC.y += params.shiftY * 2.0f;

    float4 clip{ pixelNDC.x, pixelNDC.y, 1.0f, 1.0f };
    const sutil::Matrix4x4 clipToView(params.clipToView);
    float4 viewSpace = clipToView * clip;

    const sutil::Matrix4x4 viewToWorld(params.viewToWorld);
    float4 wdir = viewToWorld * make_float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);

    origin = make_float3(viewToWorld * make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    direction = normalize(make_float3(wdir));

    // Thin lens DOF
    if (params.useDof && params.lensRadius > 0.0f)
    {
        // Camera basis vectors from viewToWorld matrix
        float3 camRight = make_float3(viewToWorld[0], viewToWorld[4], viewToWorld[8]);
        float3 camUp    = make_float3(viewToWorld[1], viewToWorld[5], viewToWorld[9]);
        float3 camFwd   = make_float3(-viewToWorld[2], -viewToWorld[6], -viewToWorld[10]);

        // Focal point along ray at focal distance (measured along camera forward axis)
        float t = params.focalDistance / fmaxf(dot(direction, camFwd), 1e-6f);
        float3 focalPoint = origin + direction * t;

        // Jitter origin on lens aperture
        float2 lensSample = sampleAperture(sampler) * params.lensRadius;
        origin += camRight * lensSample.x + camUp * lensSample.y;
        direction = normalize(focalPoint - origin);
    }
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
        ior_stack_init(prd.iorStack);
        prd.depth = 0;
        prd.passthrough = 0;
        prd.passedThrough = false;
        prd.specularBounce = false;
        prd.lastBsdfPdf = 0.0f;

        float3 ray_origin, ray_direction;

        const uint2 pixelCoord = make_uint2(launch_index.x, params.image_height - launch_index.y);
        generateCameraRay(pixelCoord, prd.sampler, ray_origin, ray_direction);

        unsigned int payload0, payload1;
        packPointer(&prd, payload0, payload1);

        float time = params.enableMotionBlur ? random<SampleDimension::eTime>(prd.sampler) : 0.0f;
        if (params.enableMotionBlur && !params.isMotionBlurVisible) time = 1.0f;

        while (prd.depth < params.max_depth)
        {
            optixTrace(params.handle, ray_origin, ray_direction,
                       params.materialRayTmin, // Min intersection distance
                       1e16f, // Max intersection distance
                       time, // rayTime -- used for motion blur
                       OptixVisibilityMask(255), // Specify always visible
                       OPTIX_RAY_FLAG_NONE,
                       RAY_TYPE_RADIANCE, // SBT offset   -- See SBT discussion
                       RAY_TYPE_COUNT, // SBT stride   -- See SBT discussion
                       RAY_TYPE_RADIANCE, // missSBTIndex -- See SBT discussion
                       payload0, payload1);

            ray_origin = prd.origin;
            ray_direction = prd.dir;

            // A cutout the path slipped through is coverage, not scattering: the
            // segment carries on in the same direction with the same throughput.
            // It deliberately does not spend a bounce -- a hedge of alpha-tested
            // leaves would otherwise exhaust max_depth before any light
            // transport happened -- and is bounded instead by
            // PATH_PASSTHROUGH_MAX, which the closest hit enforces.
            if (prd.passedThrough)
            {
                prd.passedThrough = false;
                continue;
            }

            if (prd.depth > 3)
            {
                const float lum = dot(prd.throughput, make_float3(0.2126f, 0.7152f, 0.0722f));
                const float p = clamp(lum, 0.05f, 0.95f);
                if (random<SampleDimension::eRussianRoulette>(prd.sampler) > p)
                {
                    break;
                }
                prd.throughput *= 1.0f / p;
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
    PerRayData* prd = getPRD();

    if (params.hasEnvMap)
    {
        const float3 ray_dir = optixGetWorldRayDirection();
        const float2 uv = dirToEnvUV(ray_dir, params.envMapRotation);
        const float4 envSample = tex2D<float4>(params.envMapTexture, uv.x, uv.y);
        float3 envColor = make_float3(envSample.x, envSample.y, envSample.z);
        envColor *= params.envMapIntensity * params.envMapColorTint;

        if (prd->depth == 0 || prd->specularBounce)
        {
            // Direct camera ray or specular bounce: add full env contribution
            prd->radiance += prd->throughput * envColor;
        }
        else
        {
            // MIS weight with BSDF sampling vs env map PDF
            const float envPdf = envMapPdf(ray_dir,
                                           params.envCdfX, params.envCdfY,
                                           params.envMapWidth, params.envMapHeight,
                                           params.envMapRotation);
            // Account for 50% selection probability when local lights exist
            const float envSelectionPdf = (params.scene.numLights > 0) ? 0.5f : 1.0f;
            const float effectiveEnvPdf = envPdf * envSelectionPdf;
            if (effectiveEnvPdf > 0.0f)
            {
                const float misWeight = computeMisWeight(prd->lastBsdfPdf, effectiveEnvPdf, params.misHeuristic);
                prd->radiance += prd->throughput * envColor * misWeight;
            }
        }
    }
    else
    {
        MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
        prd->radiance += prd->throughput * miss_data->bg_color;
    }

    prd->throughput = make_float3(0.0f);
    prd->depth = params.max_depth;
}

// Reached only for a hit the any-hit program accepted, which is an opaque one:
// nothing gets through, so the surviving fraction of the light is zero. The
// payload carries that fraction as a float, not a boolean, because a shadow ray
// can now cross several cutout surfaces and arrive dimmed rather than blocked.
extern "C" __global__ void __closesthit__occlusion()
{
    optixSetPayload_0(__float_as_uint(0.0f));
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
            // When env map is present, local lights are selected with 50% probability
            const float lightSelectionPdf = params.hasEnvMap
                ? 0.5f / params.scene.numLights
                : 1.0f / params.scene.numLights;
            float lightPdf =
                getLightPdf(currLight, hitPoint, optixGetWorldRayOrigin(), params.rectLightSamplingMethod) *
                lightSelectionPdf;
            const float misWeight = computeMisWeight(prd->lastBsdfPdf, lightPdf, params.misHeuristic);
            prd->radiance += prd->throughput * make_float3(currLight.color) * -dot(rayDir, lightNormal) * misWeight;
        }
    }
    prd->throughput = make_float3(0.0f);
    // stop tracing
    return;
}