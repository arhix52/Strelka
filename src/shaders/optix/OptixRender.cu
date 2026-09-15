#include <optix.h>

#include <OptixRenderParams.h>

extern "C"
{
    __constant__ Params params;
}

static __device__ bool samplerBlueNoiseEnabled()
{
    return params.hasBlueNoise != 0u;
}
#include <cuda_helpers/helpers.h>
#include <random.h>

#include <sutil/vec_math.h>
#include <sutil/Matrix.h>

#include <postprocessing/Guides.h>
#include <env_light.h>

#include "optix_device_utils.h"

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

    // The film Y flip includes subpixel jitter so every sample stays in its row.
    float2 pixelPos = make_float2(pixelIndex.x + subpixel_jitter.x,
                                  (float)params.image_height - (pixelIndex.y + subpixel_jitter.y));

    // The same position expressed the way a screen-space motion vector needs it:
    // y down, origin at the top-left of the image -- so, undoing the flip above,
    // the jittered sample position in launch order.
    screenSample = make_float2(pixelPos.x, (float)params.image_height - pixelPos.y);

    float2 dimension = make_float2(params.image_width, params.image_height);
    float2 pixelNDC = (pixelPos / dimension) * 2.0f - 1.0f;

    // Lens shift
    pixelNDC.x += params.shiftX * 2.0f;
    pixelNDC.y += params.shiftY * 2.0f;

    const sutil::Matrix4x4 viewToWorld(params.viewToWorld);

    if (params.projectionType == PROJECTION_ORTHOGRAPHIC)
    {
        const float3 filmPos =
            make_float3(pixelNDC.x * params.orthoHalfWidth, pixelNDC.y * params.orthoHalfHeight, 0.0f);
        origin = make_float3(viewToWorld * make_float4(filmPos.x, filmPos.y, filmPos.z, 1.0f));
        direction = normalize(make_float3(viewToWorld * make_float4(0.0f, 0.0f, -1.0f, 0.0f)));
    }
    else
    {
        float4 clip{ pixelNDC.x, pixelNDC.y, 1.0f, 1.0f };
        const sutil::Matrix4x4 clipToView(params.clipToView);
        float4 viewSpace = clipToView * clip;

        float4 wdir = viewToWorld * make_float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);

        origin = make_float3(viewToWorld * make_float4(0.0f, 0.0f, 0.0f, 1.0f));
        direction = normalize(make_float3(wdir));
    }

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

/// Fold fresh samples into the running mean in linear radiance.
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
static __forceinline__ __device__ unsigned int reorderCoherenceHint()
{
    if (!optixHitObjectIsHit())
        return 0u;

    const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixHitObjectGetSbtDataPointer());
    unsigned int hint = 1u;
    if (hit_data->lightId >= 0)
        return hint | 2u;

    const int32_t matId = hit_data->materialId;
    if (matId >= 0)
        hint |= (params.materials[matId].material_type & 0x7u) << 2;
    return hint;
}

static constexpr unsigned int kReorderHintBits = 5;

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
    uint32_t bounceSum = 0;
    const uint32_t linearPixelIndex = launch_index.y * params.image_width + launch_index.x;

    if (params.debug == (uint32_t)DebugMode::eSharcRadiance)
    {
        params.image[linearPixelIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    for (uint32_t sampleIdx = 0; sampleIdx < params.samples_per_launch; ++sampleIdx)
    {
        // Zero-initialise every field; PerRayData size directly determines the
        // per-thread continuation stack.
        PerRayData prd = {};
        prd.setFirstEventType(EventType::eUndef);
        const uint32_t sampleIndex = params.subframe_index + sampleIdx;

        prd.sampler = initSampler(launch_index.x, launch_index.y, linearPixelIndex, sampleIndex, params.maxSampleCount,
                                    params.mortonLevels, params.blueNoiseSwitch);

        prd.radiance = make_float3(0.0f);
        prd.throughput = make_float3(1.0f);
        prd.iorStack.top = -1;
        prd.depth = 0;
        prd.passthrough = 0;
        prd.passedThrough = false;
        prd.specularBounce = false;
        prd.neeDone = false;
        prd.lastBsdfPdf = 0.0f;
        // Guides describe a pixel, not a sample, so only the first sample of a
        // launch writes them; the rest would rewrite the same record through a
        // different jitter and pay the memory traffic for nothing.
        prd.writeAov = params.writeAov && sampleIdx == 0;
        prd.aovDone = false;
        if (params.sharcCapacity != 0u)
        {
            params.sharcPath[linearPixelIndex].index = SHARC_NO_ENTRY;
            // No segment has been launched yet, and zero blocks the eligibility
            // test -- which is correct: the camera ray is not a lobe.
            params.sharcPath[linearPixelIndex].launchRoughness = 0.0f;
        }

        float3 ray_origin, ray_direction;
        float2 pixelSample;

        const uint2 pixelCoord = make_uint2(launch_index.x, launch_index.y);
        generateCameraRay(pixelCoord, prd.sampler, ray_origin, ray_direction, pixelSample);

        if (prd.writeAov && params.aov != nullptr)
        {
            AovSample a;
            a.diffuseAlbedo = make_float3(0.0f);
            a.specularAlbedo = make_float3(0.0f);
            a.normal = -ray_direction;
            a.roughness = 1.0f;
            a.depth = oka::guides::backgroundDepth(params.denoiseDepthMode);
            // Temporary home for the jittered camera sample. The first guide
            // writer consumes it and replaces both fields with actual motion.
            a.motionX = pixelSample.x;
            a.motionY = pixelSample.y;
            a.specularHitDistance = 0.0f;
            a.reactive = 1.0f;
            a.pad2 = 0.0f;
            params.aov[linearPixelIndex] = a;
        }

        unsigned int payload0, payload1;
        packPointer(&prd, payload0, payload1);

        float time = params.enableMotionBlur ? random<SampleDimension::eTime>(prd.sampler) : 0.0f;
        if (params.enableMotionBlur && !params.isMotionBlurVisible) time = 1.0f;

        const uint32_t maxSegments =
            params.max_depth + PATH_PASSTHROUGH_MAX + min(params.subsurfaceIterations, MEDIUM_MAX_STEPS);
        uint32_t segments = 0;
        // Tracks prd.depth, but survives the cache overwriting it. See bounceSum.
        uint32_t bounces = 0;

        while (prd.depth < params.max_depth && segments < maxSegments)
        {
            ++segments;
            optixTraverse(params.handle, ray_origin, ray_direction,
                          params.materialRayTmin, // Min intersection distance
                          1e16f, // Max intersection distance
                          time, // rayTime -- used for motion blur
                          OptixVisibilityMask(prd.depth == 0 ? RAY_MASK_PRIMARY : RAY_MASK_SECONDARY),
                          OPTIX_RAY_FLAG_NONE,
                          RAY_TYPE_RADIANCE, // SBT offset   -- See SBT discussion
                          RAY_TYPE_COUNT, // SBT stride   -- See SBT discussion
                          RAY_TYPE_RADIANCE, // missSBTIndex -- See SBT discussion
                          payload0, payload1);
            if (params.enableShaderReorder)
            {
                optixReorder(reorderCoherenceHint(), kReorderHintBits);
            }
            optixInvoke(payload0, payload1);

            ray_origin = prd.origin;
            ray_direction = prd.dir;

            if (prd.passedThrough)
            {
                prd.passedThrough = false;
                continue;
            }

            // Russian roulette uses the largest throughput channel, capped at
            // one; the floor bounds the weight of surviving low-throughput paths.
            // See docs/open-defects.md.
            if (prd.depth > 3)
            {
                const float maxChannel =
                    fmaxf(prd.throughput.x, fmaxf(prd.throughput.y, prd.throughput.z));
                const float p = clamp(maxChannel, 0.05f, 1.0f);
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
            ++bounces;

            // The single-hit views describe the first surface and nothing
            // past it, so there is no reason to keep tracing.
            if (DEBUG_MODE_IS_SINGLE_HIT(params.debug))
                break;
            samplerAdvanceDepth(prd.sampler);
        }

        if (params.sharcCapacity != 0u)
        {
            const SharcPathState visit = params.sharcPath[linearPixelIndex];
            if (visit.index != SHARC_NO_ENTRY)
            {
                const float3 gathered = (prd.radiance - visit.radianceAtVisit) * visit.invThroughput;
                if (gathered.x >= 0.0f && gathered.y >= 0.0f && gathered.z >= 0.0f)
                {
                    if (params.sharcResponsive != 0u && visit.responsiveIndex != SHARC_NO_ENTRY)
                    {
                        const float3 responsive = visit.responsiveRadiance * visit.invThroughput;
                        const float3 steady = make_float3(fmaxf(gathered.x - responsive.x, 0.0f),
                                                          fmaxf(gathered.y - responsive.y, 0.0f),
                                                          fmaxf(gathered.z - responsive.z, 0.0f));
                        sharcWrite(params.sharcEntries, visit.index, steady);
                        sharcWrite(params.sharcEntries, visit.responsiveIndex, responsive);
                    }
                    else
                    {
                        sharcWrite(params.sharcEntries, visit.index, gathered);
                    }
                }
            }
        }

        result += prd.radiance;
        bounceSum += bounces;

        if (params.writeSplitAov)
        {
            if (prd.firstEventType() == EventType::eDiffuse)
            {
                diffuse += prd.radiance;
                ++diffuseSamples;
            }
            if (prd.firstEventType() == EventType::eSpecular)
            {
                specular += prd.radiance;
                ++specularSamples;
            }
        }
    }

    result /= static_cast<float>(params.samples_per_launch);

    if (params.writeSplitAov)
    {
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
    }
    (void)diffuseOut;
    (void)specularOut;

    if (params.debug >= DEBUG_MODE_FIRST_AOV && params.debug < (uint32_t)DebugMode::eSharcGrid &&
        params.aov != nullptr)
    {
        params.image[linearPixelIndex] = make_float4(visualiseGuide(params.aov[linearPixelIndex], params.debug), 1.0f);
        return;
    }

    // The two radiance-cache views that are not a property of one surface. The
    // other two (eSharcGrid, eSharcRadiance) are written by the closest hit,
    // where the surface is.
    if (params.debug == (uint32_t)DebugMode::eSharcBounces)
    {
        const float mean = (float)bounceSum / (float)params.samples_per_launch;
        const float t = fminf(mean, 3.0f);
        float3 heat;
        if (t < 1.0f)
        {
            heat = lerp(make_float3(0.0f, 0.0f, 0.4f), make_float3(0.0f, 1.0f, 0.0f), t);
        }
        else if (t < 2.0f)
        {
            heat = lerp(make_float3(0.0f, 1.0f, 0.0f), make_float3(1.0f, 1.0f, 0.0f), t - 1.0f);
        }
        else
        {
            heat = lerp(make_float3(1.0f, 1.0f, 0.0f), make_float3(1.0f, 0.0f, 0.0f), t - 2.0f);
        }
        params.image[linearPixelIndex] = make_float4(heat, 1.0f);
        return;
    }
    if (params.debug == (uint32_t)DebugMode::eSharcRadiance)
    {
        return; // already written, at the primary hit
    }
    if (params.debug == (uint32_t)DebugMode::eSharcOccupancy)
    {
        // A view of the table, not of the scene -- the table has no position in
        // the world, so there is nothing to show it over. Black where the grid
        // of blocks does not reach, as the SDK's own overlay does.
        float3 overlay = make_float3(0.0f);
        if (params.sharcCapacity != 0u)
        {
            sharcDebugOccupancy(params.sharcEntries, params.sharcCapacity,
                                make_uint2(launch_index.x, launch_index.y), make_uint2(dim.x, dim.y), overlay);
        }
        params.image[linearPixelIndex] = make_float4(overlay, 1.0f);
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

extern "C" __global__ void __closesthit__occlusion()
{
    optixSetPayload_0(__float_as_uint(0.0f));
}
