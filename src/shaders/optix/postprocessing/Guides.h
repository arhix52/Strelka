#pragma once

#include <OptixRenderParams.h>
#include <optix_denoise_plan.h>

#include <sutil/Matrix.h>
#include <sutil/vec_math.h>

/// Depth in whichever convention the denoiser is currently being fed; see
/// STRELKA_DENOISE_DEPTH_* for why this is a switch rather than a decision.
static __forceinline__ __device__ float guideViewDepth(const Params& params, const float3 worldPosition)
{
    if (params.denoiseDepthMode == STRELKA_DENOISE_DEPTH_DEVICE)
    {
        const sutil::Matrix4x4 worldToClip(params.worldToClip);
        const float4 clip = worldToClip * make_float4(worldPosition, 1.0f);
        return clip.w > 0.0f ? clip.z / clip.w : 1.0f;
    }
    const sutil::Matrix4x4 viewToWorld(params.viewToWorld);
    const float3 eye = make_float3(viewToWorld * make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    if (params.denoiseDepthMode == STRELKA_DENOISE_DEPTH_VIEWZ)
    {
        // Along the camera axis, which is what a depth buffer holds before the
        // projection is applied. The third column of viewToWorld is the camera's
        // backward axis, so forward is its negation.
        const float3 forward = -make_float3(viewToWorld[2], viewToWorld[6], viewToWorld[10]);
        return dot(worldPosition - eye, forward);
    }
    return length(worldPosition - eye);
}

/// Screen-space motion for a point that sat at `prevWorld` last frame, in pixels
/// and y down, measured against the sample position the camera ray went through.
static __forceinline__ __device__ float2 guideScreenMotion(const Params& params,
                                                           const float4 prevWorldHomogeneous,
                                                           const float2 currentSample)
{
    if (!params.hasPrevFramePose)
    {
        return make_float2(0.0f, 0.0f);
    }
    const sutil::Matrix4x4 prevWorldToClip(params.prevWorldToClip);
    const float4 clip = prevWorldToClip * prevWorldHomogeneous;
    const oka::guides::Vec2 m = oka::guides::screenMotion(
        clip.x, clip.y, clip.w, currentSample.x, currentSample.y, params.image_width, params.image_height);
    return make_float2(m.x, m.y);
}

/// The raygen parks its jittered pixel position in the guide record until the
/// primary hit replaces these fields with actual motion. Keeping it there
/// avoids carrying eight AOV-only bytes in every path's continuation stack.
static __forceinline__ __device__ float2 guideCurrentSample(const Params& params, const uint32_t pixelIndex)
{
    return make_float2(params.aov[pixelIndex].motionX, params.aov[pixelIndex].motionY);
}

static __forceinline__ __device__ void writeBackgroundGuide(const Params& params,
                                                            const uint32_t pixelIndex,
                                                            const float3 rayDir,
                                                            const uint32_t depth,
                                                            const float2 currentSample)
{
    AovSample a;
    a.diffuseAlbedo = make_float3(0.0f);
    a.specularAlbedo = make_float3(0.0f);
    a.normal = -rayDir;
    a.roughness = 1.0f;
    a.specularHitDistance = 0.0f;
    // Sky seen through a mirror moves with the reflection, not with the
    // reflector, so its history is not reliable either.
    a.reactive = oka::guides::reactiveFor(depth);
    a.pad2 = 0.0f;

    if (depth == 0)
    {
        a.depth = oka::guides::backgroundDepth(params.denoiseDepthMode);
        const float2 motion = guideScreenMotion(params, make_float4(rayDir, 0.0f), currentSample);
        a.motionX = motion.x;
        a.motionY = motion.y;
    }
    else
    {
        const AovSample prev = params.aov[pixelIndex];
        a.depth = prev.depth;
        a.motionX = prev.motionX;
        a.motionY = prev.motionY;
    }
    params.aov[pixelIndex] = a;
}
