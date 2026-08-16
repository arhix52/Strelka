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

    // Film position, y up, from a launch index that counts down. The flip has to
    // take the jitter with it -- `height - (y + j)` -- and not be applied to the
    // integer alone. `height - y` then `+ j` is the same expression one row out:
    // row 0 samples the band [height, height+1), which is off the film entirely,
    // and every row lands one pixel from where Metal puts it.
    //
    // That was measurable and was being read as noise. Fitting the offset between
    // the two backends' ladder renders gives dy = +1.00 on every scene with an
    // edge in it (00 +0.91, 02 +0.98, 20 +1.00, 24 +1.02, 28 +1.04) and dx = 0.
    // A one-row displacement puts a full-contrast residual on every silhouette,
    // so `rel` against Cycles ran ~1.3x Metal's while the two backends' per-sample
    // variance, measured where the reference is locally flat, was the same to 2%.
    // tools/parity/noise_check.py is that measurement.
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
        // No centre of projection: every ray runs down the view axis and the pixel
        // picks where on the film it starts. clipToView is deliberately unused --
        // for an orthographic frame it is a scale, and going through it would only
        // re-derive the half-extents that are already here. Same derivation as the
        // Metal path in shading_common.h, so the two backends frame identically.
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
// The coherence key handed to optixReorder, read off the hit object between
// traversal and shading.
//
// What it sorts on, least significant bit first, because that is the order the
// sorting unit weights them in:
//
//   bit 0    hit / miss. The largest divergence in the loop: a miss runs an
//            environment lookup and ends the path, a hit runs a BSDF.
//   bit 1    the hit is an emitter. Light geometry has its own hit group, adds
//            emission and stops; it shares no code with a surface.
//   bits 2-4 MaterialType. Diffuse, conductor, dielectric, standard PBR and
//            hair are five different sets of lobes, and which one a warp is
//            running is what decides how much of it is idle.
//
// Instance or primitive identity is deliberately not in here. It sorts rays
// that shade identically into different buckets, and the hardware has only a
// few bits to spend.
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
    const uint32_t linearPixelIndex = launch_index.y * params.image_width + launch_index.x;

    for (uint32_t sampleIdx = 0; sampleIdx < params.samples_per_launch; ++sampleIdx)
    {
        PerRayData prd = {};
        prd.firstEventType = EventType::eUndef;
        prd.linearPixelIndex = linearPixelIndex;
        prd.sampleIndex = params.subframe_index + sampleIdx;

        // Launch coordinates as they are. The Morton code only decides which block
        // of the sequence this pixel draws from, so a flipped y was never wrong
        // here -- but it passed `height` for row 0, and having one expression for
        // "this pixel" is what keeps the film flip in generateCameraRay, where it
        // has to take the jitter with it.
        prd.sampler = initSampler(launch_index.x, launch_index.y, prd.linearPixelIndex, prd.sampleIndex, params.maxSampleCount, 52u);

        prd.radiance = make_float3(0.0f);
        prd.throughput = make_float3(1.0f);
        ior_stack_init(prd.iorStack);
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
        // Not zero: zero is a valid slot. The `= {}` above would otherwise leave
        // every path claiming to have visited entry 0.
        prd.sharcIndex = SHARC_NO_ENTRY;

        float3 ray_origin, ray_direction;

        const uint2 pixelCoord = make_uint2(launch_index.x, launch_index.y);
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

        // Segments, not bounces. A cutout the path slips through, a medium
        // boundary it crosses and a step of a subsurface walk each take a
        // traversal and deliberately spend no depth, so `max_depth` alone does
        // not bound this loop.
        //
        // The budget is Metal's, arrived at from the other side: the wavefront
        // encodes `maxDepth + PATH_PASSTHROUGH_MAX + subsurfaceIterations`
        // dispatch iterations and drops whatever paths are still alive when they
        // run out. A per-path cap of the same size drops exactly the same paths,
        // and it is the difference between a dim pixel and a GPU hang if a walk
        // ever fails to terminate.
        const uint32_t maxSegments =
            params.max_depth + PATH_PASSTHROUGH_MAX + min(params.subsurfaceIterations, MEDIUM_MAX_STEPS);
        uint32_t segments = 0;

        while (prd.depth < params.max_depth && segments < maxSegments)
        {
            ++segments;
            // Traversal and shading are split so that the warp can be sorted
            // between them. optixTraverse leaves a hit object behind without
            // running a program; optixReorder regroups the threads by what that
            // hit object says they are about to shade; optixInvoke then runs the
            // closest-hit or miss program on a warp whose threads mostly agree.
            //
            // Together they are exactly equivalent to the optixTrace this
            // replaces -- same payloads, same programs, same results. The payload
            // pair is a packed pointer to PerRayData in local memory, so nothing
            // rides in the registers that the split could drop.
            optixTraverse(params.handle, ray_origin, ray_direction,
                          params.materialRayTmin, // Min intersection distance
                          1e16f, // Max intersection distance
                          time, // rayTime -- used for motion blur
                          // Camera rays see what the camera should see; every
                          // bounce after also sees the emitters marked hidden, so
                          // a light authored out of frame still balances the MIS
                          // estimate it is deducted for.
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

            // Russian roulette on the largest channel the path still carries,
            // capped at one. Metal's `q`, arrived at from the other side.
            //
            // This used to be `clamp(luminance, 0.05, 0.95)`, and both ends of
            // that were paying for themselves in variance:
            //
            //   * The ceiling killed strong paths. A path whose throughput is at
            //     or above one -- which after four bounces in a bright interior
            //     is most of what is left -- was killed 5% of the time anyway and
            //     the survivors scaled by 1/0.95. Unbiased, and pure added
            //     variance, charged again at every bounce: over the twelve
            //     remaining at depth 16, 46% of such paths die and the survivors
            //     come back carrying 1.85x. That is a firefly generator with no
            //     upside, and `q = min(max_component, 1)` never kills a path that
            //     carries a full unit of light.
            //
            //   * Luminance is the wrong norm for a coloured path. A throughput
            //     of (0, 0, 5) -- what a blue-tinted glass leaves -- has
            //     luminance 0.36, so it was killed 64% of the time and the
            //     survivors multiplied by 2.8, while carrying five units of blue.
            //     That is where the bathroom's *coloured* speckle came from, and
            //     why clamp_indirect = 8 halved the frame's variance.
            //
            // Measured on the iso bathroom at 256 spp, graded against a 4096-spp
            // render of this backend on the pixels where it is locally flat:
            // relative noise 0.184 before, 0.175 after, converged mean unmoved at
            // 0.2169. So it is worth having and it is not what makes that scene
            // noisy -- half of that frame's variance is in samples above 8, which
            // `clamp_indirect = 8` removes at the cost of 6.5% of the mean, and
            // the roulette is not where they come from. See docs/open-defects.md.
            // The floor is kept -- Metal has none -- because without it a path
            // whose largest channel is 1e-6 survives one time in a million
            // carrying 1e6, and at 0.05 the same path survives one time in twenty
            // carrying 20. Both are unbiased; the second has a variance.
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

            // The two single-hit views describe the first surface and nothing
            // past it, so there is no reason to keep tracing.
            if (DEBUG_MODE_IS_SINGLE_HIT(params.debug))
                break;
            prd.sampler.depth++;
        }

        // The path is over, so what it gathered after the cache visit is known.
        // Divided by the throughput it carried there, that is the outgoing
        // radiance of the visited point -- which is what the cache stores.
        //
        // Here rather than at each of the half-dozen places a path can end,
        // which is the same reason Metal spends a whole dispatch on it: the
        // alternative is six scattered edits that would each have to stay
        // correct.
        if (params.sharcCapacity != 0u && prd.sharcIndex != SHARC_NO_ENTRY)
        {
            const float3 gathered = (prd.radiance - prd.sharcRadianceAtVisit) * prd.sharcInvThroughput;
            // A negative component means the difference is not what it claims --
            // a clamp or a NaN guard fired between the visit and here -- and a
            // negative deposit would wrap the unsigned accumulator.
            if (gathered.x >= 0.0f && gathered.y >= 0.0f && gathered.z >= 0.0f)
            {
                sharcWrite(params.sharcEntries, prd.sharcIndex, gathered);
            }
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

// __miss__ms lives in OptixRender_closest_hit.cu.
//
// Not for tidiness: a ray that reaches the environment has still travelled a
// segment, and with an atmosphere that segment can scatter before the sky is
// ever seen. Handling that needs next-event estimation, and connectToLight and
// its shadow ray are defined in that translation unit -- OptiX modules do not
// share device functions, so the program has to be where the light machinery is.
// createProgramGroups() points the miss group at `closest_hit_module`.

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
        // `color` is radiance, and radiance along a ray does not fall off with the
        // angle it leaves the emitter at: the cosine at the light belongs in the
        // area-to-solid-angle Jacobian that next-event estimation already applies,
        // not here. Multiplying by it a second time made the BSDF strategy darken
        // every emitter it hit off-axis while the next-event strategy did not, so
        // the two disagreed by exactly cos at every vertex. Metal's light hit has
        // never had the factor.
        const float3 Le = make_float3(currLight.color);
        float3 radiance;
        if (prd->depth == 0 || prd->specularBounce || !prd->neeDone)
        {
            radiance = prd->throughput * Le;
        }
        else
        {
            // When env map is present, local lights are selected with 50% probability
            const float lightSelectionPdf = params.hasEnvMap
                ? 0.5f / params.scene.numLights
                : 1.0f / params.scene.numLights;
            // From the vertex that scattered, which is not the ray's origin once
            // it has passed through a cutout or crossed a medium's boundary on
            // the way here. Using the origin makes the light look nearer than the
            // scattering vertex saw it, which shrinks its solid-angle density,
            // which inflates this weight -- and the next-event estimate at that
            // vertex has already claimed the rest, so the two sum to more than
            // one.
            const float3 misOrigin = optixGetWorldRayOrigin() - rayDir * prd->misDistance;
            float lightPdf =
                getLightPdf(currLight, hitPoint, misOrigin, params.rectLightSamplingMethod) *
                lightSelectionPdf;
            const float misWeight = computeMisWeight(prd->lastBsdfPdf, lightPdf, params.misHeuristic);
            radiance = prd->throughput * Le * misWeight;
        }
        prd->radiance += clampIndirectContribution(radiance, prd->depth, params.clampIndirect);
    }
    prd->throughput = make_float3(0.0f);
    // stop tracing
    return;
}