#include <optix.h>

#include <OptixRenderParams.h>
#include <cuda_helpers/helpers.h>
#include <random.h>

#include <sutil/vec_math.h>
#include <sutil/Matrix.h>

#include <postprocessing/Guides.h>
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
    const uint2 pixelIndex, SamplerState& sampler, float3& origin, float3& direction, float2& screenSample)
{
    float2 subpixel_jitter =
        make_float2(random<SampleDimension::ePixelX>(sampler), random<SampleDimension::ePixelY>(sampler));

    float2 pixelPos = make_float2(pixelIndex.x + subpixel_jitter.x, pixelIndex.y + subpixel_jitter.y);

    // The same position expressed the way a screen-space motion vector needs it:
    // y down, origin at the top-left of the image. `pixelIndex.y` arrives already
    // flipped (the caller passes height - y), so the flip has to be undone here
    // rather than guessed at by whoever consumes it.
    screenSample = make_float2(pixelPos.x, (float)params.image_height - pixelPos.y);

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

/// Fold `value` -- the mean of `newSampleCount` fresh samples -- into a running
/// mean that already holds `prevSampleCount` of them.
///
/// Linearly, in radiance. This used to lerp in tone-mapped space and invert the
/// curve afterwards, which is not an average of the samples but an average of
/// `c/(c+1)` mapped back, and the two differ by Jensen's inequality: the
/// tone-mapped mean is always the darker one, by an amount that grows with the
/// variance among the samples. The whole ladder was low because of it --
/// 00_calibration at ratio 0.973 where Metal reads 1.010 -- and the mirror scene,
/// which has the most per-sample variance of any row, was 0.779. The inverse
/// also diverges: `c / (exposure - c*exposure)` goes to infinity as the
/// tone-mapped value approaches 1, so a bright pixel's history was one rounding
/// error away from an infinity.
///
/// The weight is the new samples' share of the total, not `1/(n+1)`: with more
/// than one sample per launch the old form counted a whole launch as a single
/// sample and left the first launch permanently over-weighted. Byte for byte
/// what wavefrontResolve does on Metal.
__device__ float4 accumulate(float4* history,
                             const float3 value,
                             const uint32_t linearPixelIndex,
                             const uint32_t prevSampleCount,
                             const uint32_t newSampleCount)
{
    float3 accumColor = value;
    if (prevSampleCount > 0 && newSampleCount > 0)
    {
        const float a = static_cast<float>(newSampleCount) / static_cast<float>(prevSampleCount + newSampleCount);
        accumColor = lerp(make_float3(history[linearPixelIndex]), value, a);
    }
    history[linearPixelIndex] = make_float4(accumColor, 1.0f);
    return make_float4(accumColor, 1.0f);
}

/// The guide views, drawn from the record the shading programs wrote.
///
/// A denoiser fed a broken guide degrades quietly, so every guide has to be
/// something you can look at. Same encodings as the Metal resolve pass.
__device__ float3 visualiseGuide(const AovSample& a, const uint32_t debugMode)
{
    switch ((DebugMode)debugMode)
    {
    case DebugMode::eAovDiffuseAlbedo:
        return a.diffuseAlbedo;
    case DebugMode::eAovSpecularAlbedo:
        return a.specularAlbedo;
    case DebugMode::eAovNormal:
        return a.normal * 0.5f + make_float3(0.5f);
    case DebugMode::eAovRoughness:
        return make_float3(a.roughness);
    // d/(1+d): monotonic and scale-free, so a scene of any size is readable and
    // nothing crosses zero the way a logarithm does at d == 1.
    case DebugMode::eAovDepth:
        return make_float3(a.depth / (1.0f + a.depth));
    // Red/green for the two axes, scaled so a few pixels of motion is visible.
    case DebugMode::eAovMotion:
        return make_float3(a.motionX, a.motionY, 0.0f) * 0.05f + make_float3(0.5f);
    // Red where the denoiser is being told to distrust its history, so the extent
    // of the mask is a thing you can look at rather than infer.
    case DebugMode::eAovReactive:
        return make_float3(a.reactive, 0.0f, 0.0f);
    case DebugMode::eAovSpecularHitDistance:
        return make_float3(a.specularHitDistance / (1.0f + a.specularHitDistance));
    default:
        return make_float3(0.0f);
    }
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
        prd.specularBounce = false;
        prd.lastBsdfPdf = 0.0f;
        // Guides describe a pixel, not a sample, so only the first sample of a
        // launch writes them; the rest would rewrite the same record through a
        // different jitter and pay the memory traffic for nothing.
        prd.writeAov = params.writeAov && sampleIdx == 0;
        prd.aovDone = false;

        float3 ray_origin, ray_direction;

        const uint2 pixelCoord = make_uint2(launch_index.x, params.image_height - launch_index.y);
        generateCameraRay(pixelCoord, prd.sampler, ray_origin, ray_direction, prd.pixelSample);

        if (prd.writeAov && params.aov != nullptr)
        {
            // Start from a complete record, so that a path which never reaches a
            // surface it can describe leaves defined values in every field
            // rather than last frame's. The shading and miss programs overwrite
            // whichever parts they can speak for.
            AovSample a;
            a.diffuseAlbedo = make_float3(0.0f);
            a.specularAlbedo = make_float3(0.0f);
            a.normal = -ray_direction;
            a.roughness = 1.0f;
            a.depth = oka::guides::backgroundDepth(params.denoiseDepthMode);
            a.motionX = 0.0f;
            a.motionY = 0.0f;
            a.specularHitDistance = 0.0f;
            a.reactive = 1.0f;
            a.pad2 = 0.0f;
            params.aov[linearPixelIndex] = a;
        }

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

            // The two single-hit views describe the first surface and nothing
            // past it, so there is no reason to keep tracing.
            if (params.debug == (uint32_t)DebugMode::eNormal || params.debug == (uint32_t)DebugMode::eMotionBlur)
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
        diffuseOut = accumulate(params.diffuse, diffuse, linearPixelIndex, prevSamplesCount, diffuseSamples);
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
        specularOut = accumulate(params.specular, specular, linearPixelIndex, prevSamplesCount, specularSamples);
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

    // The diffuse/specular split is still accumulated -- it is what a caller
    // asking for those two images reads -- but it no longer owns debug slots 2
    // and 3. Those are DebugMode::eMotionBlur and the first guide view, which is
    // what the editor's menu and the headless `render.debug` key have always
    // meant by them.
    (void)diffuseOut;
    (void)specularOut;

    if (params.debug >= DEBUG_MODE_FIRST_AOV && params.aov != nullptr)
    {
        params.image[linearPixelIndex] = make_float4(visualiseGuide(params.aov[linearPixelIndex], params.debug), 1.0f);
        return;
    }

    if (params.enableAccumulation && params.debug == 0)
    {
        // Accumulation
        params.image[linearPixelIndex] =
            accumulate(params.accum, result, linearPixelIndex, params.subframe_index, params.samples_per_launch);
    }
    else
    {
        params.image[linearPixelIndex] = make_float4(result, 1.0f);
    }
}

extern "C" __global__ void __miss__ms()
{
    PerRayData* prd = getPRD();

    // Background still needs a guide record, or the denoiser reads whatever the
    // previous frame left there and smears the silhouette across the sky.
    if (prd->writeAov && !prd->aovDone && params.aov != nullptr)
    {
        writeBackgroundGuide(params, prd->linearPixelIndex, optixGetWorldRayDirection(), prd->depth, prd->pixelSample);
        prd->aovDone = true;
    }

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

    // An emitter the camera can see is a surface like any other as far as the
    // guides are concerned; leaving it out puts a hole in the albedo and normal
    // buffers exactly where the brightest thing in the frame is.
    if (prd->writeAov && !prd->aovDone && params.aov != nullptr)
    {
        AovSample a;
        // An albedo guide is a reflectance, so an emitter's radiance cannot go
        // in it directly -- a 20 W/sr light would hand the network an albedo of
        // 20. Normalised by its own largest channel, which keeps the light's
        // hue and lands in the [0, 1] an albedo lives in.
        const float3 lightColor = make_float3(currLight.color);
        const float peak = fmaxf(fmaxf(lightColor.x, lightColor.y), fmaxf(lightColor.z, 1e-6f));
        a.diffuseAlbedo = lightColor / peak;
        a.specularAlbedo = make_float3(0.0f);
        a.normal = lightNormal;
        a.roughness = 1.0f;
        a.depth = prd->depth == 0 ? guideViewDepth(params, hitPoint) : params.aov[prd->linearPixelIndex].depth;
        if (prd->depth == 0)
        {
            const float2 motion = guideScreenMotion(params, make_float4(hitPoint, 1.0f), prd->pixelSample);
            a.motionX = motion.x;
            a.motionY = motion.y;
        }
        else
        {
            a.motionX = params.aov[prd->linearPixelIndex].motionX;
            a.motionY = params.aov[prd->linearPixelIndex].motionY;
        }
        a.specularHitDistance = 0.0f;
        a.reactive = oka::guides::reactiveFor(prd->depth);
        a.pad2 = 0.0f;
        params.aov[prd->linearPixelIndex] = a;
        prd->aovDone = true;
    }

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