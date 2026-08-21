#include "MetalFrameUniforms.h"

#include <log.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <simd/simd.h>


namespace oka::metal
{

MetalFrameUniforms::~MetalFrameUniforms()
{
    release();
}

void MetalFrameUniforms::init(MTL::Device* device)
{
    mDevice = device;
}

void MetalFrameUniforms::release()
{
    auto safeRelease = [](auto*& p) {
        if (p)
        {
            p->release();
            p = nullptr;
        }
    };
    for (auto*& buf : mUniformBuffers)
        safeRelease(buf);
    for (auto*& buf : mUniformTMBuffers)
        safeRelease(buf);
    safeRelease(mSharcHashBuffer);
    safeRelease(mSharcAccumulationBuffer);
    safeRelease(mSharcResolvedBuffer);
    mSharcCapacity = 0;
    mSharcFlags = 0;
    mSharcSceneScale = -1.0f;
    mSharcLevelBias = 0;
    mSharcResetPending = false;
    mPrevSettings = {};
}

void MetalFrameUniforms::allocateRings()
{
    for (MTL::Buffer*& uniformBuffer : mUniformBuffers)
    {
        if (!uniformBuffer)
            uniformBuffer = mDevice->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
    }
    for (MTL::Buffer*& uniformBuffer : mUniformTMBuffers)
    {
        if (!uniformBuffer)
            uniformBuffer = mDevice->newBuffer(sizeof(UniformsTonemap), MTL::ResourceStorageModeShared);
    }
}

void MetalFrameUniforms::ensureSharc(SettingsManager* settings, const oka::Camera& camera, uint32_t width, uint32_t height)
{
    (void)settings;
    (void)camera;
    (void)width;
    (void)height;
    // SHARC sizing lives inside fill() to preserve prior behavior (capacity change mid-fill).
}

MetalFrameUniforms::FillResult MetalFrameUniforms::fill(const FillInput& in)
{
    using simd::float3;
    using simd::float4;
    using simd::float4x4;

    const SettingsManager& settings = *in.settings;
    const uint32_t width = in.width;
    const uint32_t height = in.height;
    const uint32_t outWidth = in.outWidth;
    const uint32_t outHeight = in.outHeight;
    const uint32_t spp = in.spp;
    const uint32_t sspTotal = in.sspTotal;
    const uint32_t maxDepth = in.maxDepth;
    const uint32_t debug = in.debug;
    const uint32_t rectLightSamplingMethod = in.rectLightSamplingMethod;
    const uint32_t samplerType = in.samplerType;
    const uint32_t blueNoiseSwitchSpp = in.blueNoiseSwitchSpp;
    const bool enableAccumulation = in.enableAccumulation;
    const bool anyAnimationPlaying = in.anyAnimationPlaying;
    const bool denoising = in.denoising;
    const bool isMotionBlurVisible = in.isMotionBlurVisible;
    const bool enableCameraMotionBlur = in.enableCameraMotionBlur;
    const oka::Camera& camera = *in.camera;

    const bool playbackBlur = settings.getAs<bool>("render/pt/denoisePlaybackMotionBlur");
    const float shutterTime = settings.getAs<float>("render/motionBlur/shutterTime");
    const uint32_t shutterMode = settings.getAs<uint32_t>("render/motionBlur/shutterMode");
    const bool qualityPlaybackBlur =
        denoising && anyAnimationPlaying && in.enableMotionBlur && isMotionBlurVisible && playbackBlur;
    const bool effectiveAccumulation = enableAccumulation && !anyAnimationPlaying;
    const uint32_t accumulatedSamples = in.subframeIndex;
    const uint32_t remainingSamples = accumulatedSamples < sspTotal ? sspTotal - accumulatedSamples : 0u;
    const bool accumulationActive = effectiveAccumulation && remainingSamples > 0u;

    MTL::Buffer* pUniformBuffer = mUniformBuffers[in.frameSlot % kFrameUniformSlots];
    MTL::Buffer* pUniformTMBuffer = mUniformTMBuffers[in.frameSlot % kFrameUniformSlots];
    auto* pUniformData = static_cast<Uniforms*>(pUniformBuffer->contents());
    auto* pUniformTonemap = static_cast<UniformsTonemap*>(pUniformTMBuffer->contents());
    pUniformData->frameIndex = in.frameSlot;
    pUniformData->subframeIndex = in.subframeIndex;
    pUniformData->height = height;
    pUniformData->width = width;
    const bool analyticLightsEnabled = settings.getAs<bool>("render/validate/analyticLights");
    pUniformData->numLights = analyticLightsEnabled ? (uint32_t)in.scene->getLightsDesc().size() : 0u;
    pUniformData->primaryRayMask = analyticLightsEnabled ? RAY_MASK_PRIMARY : GEOMETRY_MASK_GEOMETRY;
    pUniformData->estimatorMode = settings.getAs<uint32_t>("render/validate/estimatorMode");
    // 0 = glTF (-ln(C)/d), 1 = Cycles ((1-C)/d). See volume.h.
    pUniformData->volumeModel = settings.getAs<uint32_t>("render/material/volumeModel");
    pUniformData->samples_per_launch = spp;
    pUniformData->enableAccumulation = (uint32_t)accumulationActive;
    pUniformData->risCandidates = std::max(settings.getAs<uint32_t>("render/pt/risCandidates"), 1u);
    // 0 = balance, 1 = power. The same key OptiX reads, so the two backends can
    // be compared under either heuristic.
    pUniformData->misHeuristic = settings.getAs<uint32_t>("render/pt/misHeuristic");
    pUniformData->textureLodMode = settings.getAs<uint32_t>("render/pt/textureLod");
    pUniformData->guidePrimaryHit = settings.getAs<uint32_t>("render/pt/guidePrimaryHit");
    pUniformData->missColor = float3(0.0f);
    pUniformData->maxDepth = maxDepth;
    pUniformData->debug = debug;
    // Denoiser guides. Off unless something downstream consumes them: writing
    // them costs a 64-byte store per pixel at the primary hit.
    // Looking at a guide implies producing it, and so does denoising.
    const bool denoiseOn = denoising;
    pUniformData->writeAov = settings.getAs<uint32_t>("render/pt/writeAov") || DEBUG_MODE_IS_AOV(debug) || denoiseOn;
    // Hand the denoiser the accumulated estimate whenever there is one.
    //
    // A camera that has stopped keeps tracing samples into the accumulation
    // buffer, and that buffer is a far better input than the frame's single
    // sample: it converges, and the denoiser then has almost nothing left to
    // invent. Restricted to the paused-motion-blur case before, so standing
    // still in the editor fed the denoiser one noisy sample per frame forever
    // and left MetalFX's short history as the only thing cleaning it up -- which
    // is why a still camera stayed visibly noisy no matter how long it sat there.
    //
    // Moving the camera resets the subframe counter, so this falls back to the
    // single-sample path by itself. Jitter is skipped while it is on (see just
    // below): the accumulation already samples the pixel area, and jittering on
    // top of an average is a second, uncontrolled blur.
    pUniformData->useAccumulatedColor =
        (effectiveAccumulation && in.subframeIndex > 0 && in.accumulationBuffer && !in.noAccumColor) ? 1u : 0u;
    const bool accumulating = pUniformData->useAccumulatedColor != 0u;
    float jx = 0.0f, jy = 0.0f;
    if (denoiseOn && !in.pausedBlurRefine && !accumulating)
    {
        const double ratio = (double)outWidth / (double)std::max(width, 1u);
        const uint32_t phaseCount = (uint32_t)std::clamp(std::lround(8.0 * ratio * ratio), 8L, 128L);
        metal::frameJitter(in.frameNumber, phaseCount, jx, jy);
    }
    pUniformData->jitterX = jx;
    pUniformData->jitterY = jy;
    // Temporal reconstruction of any kind needs the frame shifted by a known
    // amount it can undo; without jitter the scaler has nothing new to
    // accumulate between frames and degenerates to a blur.
    const bool temporalOn = denoiseOn || (settings.getAs<bool>("render/pt/enableUpscale") &&
                                          settings.getAs<uint32_t>("render/pt/upscaleMode") == 1u);
    pUniformData->useFrameJitter = temporalOn ? 1u : 0u;
    // Guides are assembled the same way whether the denoiser consumes them or a
    // debug view shows them, which needs saying because the obvious spelling --
    // tie it to `denoiseOn` -- cannot be right. `denoiseOn` is false whenever a
    // debug view is up, since a debug view replaces the image there would be to
    // filter. So every AOV view rendered its guides the *other* way: writeAov on,
    // canonical sample off, every sample overwriting the last into a buffer that
    // is assigned rather than accumulated.
    //
    // On a matte surface that is invisible, because every sample picks the same
    // one. On a glossy one the guide walk stops at a different surface per
    // sample, the last writer wins per pixel, and the view is salt and pepper --
    // over the floor tiles and the tiled walls of the bathroom, which is most of
    // its area. It reads exactly like a broken guide, and the guide is fine.
    pUniformData->canonicalGuideSample = pUniformData->writeAov ? 1u : 0u;
    pUniformData->denoiseFireflyClamp = settings.getAs<float>("render/pt/denoiseFireflyClamp");
    pUniformData->clampIndirect = settings.getAs<float>("render/pt/clampIndirect");
    pUniformData->hasBoundedMedium = in.materials->hasBoundedMedium() ? 1u : 0u;
    // The world scale the subsurface walk draws its free flights against. Cached
    // in Scene against the transform generation, so this is a comparison once
    // the scene has settled. The fallback matters for a scene with no bounded
    // geometry -- an environment-only frame -- where there is nothing to bound
    // and nothing to scatter in either.
    {
        glm::float3 boundsMin(0.0f);
        glm::float3 boundsMax(0.0f);
        pUniformData->sceneExtent = (in.scene != nullptr && in.scene->worldBounds(boundsMin, boundsMax)) ?
                                        std::max(glm::length(boundsMax - boundsMin), 1e-4f) :
                                        1e16f;
    }
    {
        // Previous frame's world-to-clip for screen-space reprojection. The
        // motion-blur uniforms hold the inverses and cannot serve here.
        const glm::float4x4 prevWorldToClip = in.prevView->mCamMatrices.perspective * in.prevView->mCamMatrices.view;
        std::memcpy(&pUniformData->prevWorldToClip, glm::value_ptr(prevWorldToClip), sizeof(float4x4));
        const glm::float4x4 worldToClip = in.currView->mCamMatrices.perspective * in.currView->mCamMatrices.view;
        std::memcpy(&pUniformData->worldToClip, glm::value_ptr(worldToClip), sizeof(float4x4));
    }
    pUniformData->denoiseDepthMode = settings.getAs<uint32_t>("render/pt/denoiseDepthMode");
    // A pose is only usable once one has been captured *and* the frame it belongs
    // to still corresponds to this one. Anything that resets the history has
    // already declared that it does not.
    pUniformData->hasPrevFramePose = (in.hasPrevFramePose && !in.resetDenoiseHistory && !in.noPrevPose) ? 1u : 0u;
    pUniformData->enableMotionBlur = in.enableMotionBlur ? 1 : 0;
    const bool stochasticShutter = isMotionBlurVisible && (!denoising || qualityPlaybackBlur || in.pausedBlurRefine);
    pUniformData->isMotionBlurVisible = (uint32_t)stochasticShutter;
    pUniformData->enableCameraMotionBlur = (uint32_t)enableCameraMotionBlur;
    pUniformData->rectLightSamplingMethod = rectLightSamplingMethod;
    pUniformData->samplerType = samplerType;
    pUniformData->blueNoiseSwitchSpp = blueNoiseSwitchSpp;

    // Depth of field
    pUniformData->useDof = camera.useDof ? 1 : 0;
    pUniformData->focalDistance = camera.focalDistance;
    pUniformData->lensRadius = camera.useDof ? camera.focalLengthMm / (2.0f * camera.fStopDof * 1000.0f) : 0.0f;
    pUniformData->apertureBlades = camera.apertureBlades;
    pUniformData->bladeRotation = camera.bladeRotation;
    pUniformData->anamorphicRatio = camera.anamorphicRatio;

    // Lens shift
    pUniformData->shiftX = camera.shiftX;
    pUniformData->shiftY = camera.shiftY;

    // Projection. The half-extents are adapted to the render aspect the same way
    // the perspective fov is (Camera::magForAspect), so a camera authored square
    // and rendered wide keeps its framing instead of stretching.
    pUniformData->projectionType = (uint32_t)camera.projection;
    {
        float halfWidth = camera.xmag, halfHeight = camera.ymag;
        const float aspect = (height > 0) ? (float)width / (float)height : 1.0f;
        camera.magForAspect(aspect, halfWidth, halfHeight);
        pUniformData->orthoHalfWidth = halfWidth;
        pUniformData->orthoHalfHeight = halfHeight;
    }

    // Radiance cache
    {
        const bool wantSharc = in.settings->getAs<bool>("render/pt/sharc");
        const uint32_t capacity =
            wantSharc ? std::max(1u << 16, in.settings->getAs<uint32_t>("render/pt/sharcCapacity")) : 0u;
        if (capacity != mSharcCapacity)
        {
            ++mSharcResourceGeneration;
            auto releaseBuffer = [](MTL::Buffer*& buffer) {
                if (buffer)
                {
                    buffer->release();
                    buffer = nullptr;
                }
            };
            releaseBuffer(mSharcHashBuffer);
            releaseBuffer(mSharcAccumulationBuffer);
            releaseBuffer(mSharcResolvedBuffer);
            mSharcCapacity = 0;
            if (capacity)
            {
                constexpr size_t kHashStride = 4;
                constexpr size_t kAccumulationStride = 32;
                constexpr size_t kResolvedStride = 32;
                mSharcHashBuffer = mDevice->newBuffer((size_t)capacity * kHashStride, MTL::ResourceStorageModePrivate);
                mSharcAccumulationBuffer =
                    mDevice->newBuffer((size_t)capacity * kAccumulationStride, MTL::ResourceStorageModePrivate);
                mSharcResolvedBuffer =
                    mDevice->newBuffer((size_t)capacity * kResolvedStride, MTL::ResourceStorageModePrivate);
                if (mSharcHashBuffer && mSharcAccumulationBuffer && mSharcResolvedBuffer)
                {
                    mSharcCapacity = capacity;
                    mSharcResetPending = true;
                    STRELKA_INFO(
                        "SHARC 1.8.3 Metal: {} entries ({:.1f} MB: hash {:.1f}, "
                        "accumulation {:.1f}, resolved {:.1f})",
                        capacity, capacity * 68.0 / 1e6, capacity * 4.0 / 1e6, capacity * 32.0 / 1e6,
                        capacity * 32.0 / 1e6);
                }
                else
                {
                    releaseBuffer(mSharcHashBuffer);
                    releaseBuffer(mSharcAccumulationBuffer);
                    releaseBuffer(mSharcResolvedBuffer);
                    STRELKA_ERROR("SHARC Metal allocation failed for {} entries ({:.1f} MiB); cache disabled", capacity,
                                  capacity * 68.0 / (1024.0 * 1024.0));
                }
            }
        }

        uint32_t flags = 0u;
        if (in.settings->getAs<bool>("render/pt/sharcMaterialDemodulation"))
            flags |= SHARC_FLAG_MATERIAL_DEMODULATION;
        if (in.settings->getAs<bool>("render/pt/sharcSeparateEmissive"))
            flags |= SHARC_FLAG_SEPARATE_EMISSIVE;
        if (in.settings->getAs<bool>("render/pt/sharcDirectional"))
            flags |= SHARC_FLAG_DIRECTIONAL;
        if (in.settings->getAs<bool>("render/pt/sharcMetalResponsive"))
            flags |= SHARC_FLAG_RESPONSIVE;
        if (in.settings->getAs<bool>("render/pt/sharcCacheResampling"))
            flags |= SHARC_FLAG_CACHE_RESAMPLING;
        if (in.settings->getAs<bool>("render/pt/sharcBlendAdjacentLevels"))
            flags |= SHARC_FLAG_BLEND_ADJACENT_LEVELS;
        if (in.settings->getAs<bool>("render/pt/sharcFadeAcceleration"))
            flags |= SHARC_FLAG_FADE_ACCELERATION;
        if (flags != mSharcFlags && mSharcCapacity != 0u)
        {
            mSharcResetPending = true;
        }
        mSharcFlags = flags;

        pUniformData->sharcCapacity = mSharcCapacity;
        pUniformData->sharcMinSamples = in.settings->getAs<uint32_t>("render/pt/sharcMetalMinSamples");
        pUniformData->sharcDepth = in.settings->getAs<uint32_t>("render/pt/sharcDepth");
        pUniformData->sharcFlags = flags;
        // The world size of one pixel at unit distance, times the number of
        // pixels a voxel should span. Everything scene-dependent -- field of
        // view, resolution -- is folded in here so the setting itself is not.
        const float tanHalfFov =
            std::tan(glm::radians(camera.fovForAspect(static_cast<float>(width) / static_cast<float>(height))) * 0.5f);
        const float pixelAngle = 2.0f * tanHalfFov / (float)height;
        const float sharcBaseSize = pixelAngle * std::max(1.0f, in.settings->getAs<float>("render/pt/sharcVoxelPixels"));
        const float sharcSceneScale = std::max(in.settings->getAs<float>("render/pt/sharcSceneScale"), 1e-5f);
        const float sharcRadianceScale = std::max(in.settings->getAs<float>("render/pt/sharcRadianceScale"), 1.0f);
        const int32_t sharcLevelBias = static_cast<int32_t>(in.settings->getAs<uint32_t>("render/pt/sharcLevelBias"));
        if (mSharcCapacity != 0u && (sharcSceneScale != mSharcSceneScale || sharcLevelBias != mSharcLevelBias))
        {
            mSharcResetPending = true;
        }
        mSharcSceneScale = sharcSceneScale;
        mSharcLevelBias = sharcLevelBias;
        pUniformData->sharcBaseSize = sharcBaseSize;
        pUniformData->sharcSceneScale = sharcSceneScale;
        pUniformData->sharcRoughnessThreshold =
            std::clamp(in.settings->getAs<float>("render/pt/sharcRoughnessThreshold"), 0.0f, 1.0f);
        pUniformData->sharcRadianceScale = sharcRadianceScale;
        pUniformData->sharcUpdateDownscale = std::max(in.settings->getAs<uint32_t>("render/pt/sharcUpdateDownscale"), 1u);
        pUniformData->sharcAccumulationFrames =
            std::max(in.settings->getAs<uint32_t>("render/pt/sharcAccumFrames"), 1u);
        pUniformData->sharcResponsiveFrames =
            std::max(in.settings->getAs<uint32_t>("render/pt/sharcResponsiveFrames"), 1u);
        pUniformData->sharcStaleFrameCount = std::max(in.settings->getAs<uint32_t>("render/pt/sharcStaleFrames"), 1u);
        pUniformData->sharcPropagationDepth = std::clamp(
            in.settings->getAs<uint32_t>("render/pt/sharcPropagationDepth"), 1u, uint32_t(SHARC_MAX_PROPAGATION_DEPTH));
        pUniformData->sharcDebug = in.settings->getAs<uint32_t>("render/pt/sharcDebug");
        pUniformData->sharcUpdatePathCount =
            ((width + pUniformData->sharcUpdateDownscale - 1u) / pUniformData->sharcUpdateDownscale) *
            ((height + pUniformData->sharcUpdateDownscale - 1u) / pUniformData->sharcUpdateDownscale);
        pUniformData->sharcFrameIndex = static_cast<uint32_t>(in.frameNumber);
        pUniformData->sharcLevelBias = sharcLevelBias;
        // The previous frame's camera, not the motion-blur shutter-open one: the
        // grid's level comes from distance to the camera, so a cell the camera
        // walked towards is re-keyed at a finer level, and resolve carries its
        // history over from whichever neighbour it came from.
        // On the first frame there is no previous pose; using the current one
        // keeps the delta at exactly zero so the transfer stays off until the
        // camera has actually moved.
        const bool hasPrevCamera = in.prevView != nullptr && in.frameNumber != 0;
        const glm::float4x4 prevCameraToWorld =
            glm::inverse(hasPrevCamera ? in.prevView->mCamMatrices.view : camera.matrices.view);
        pUniformData->sharcCameraPrev = float3{ prevCameraToWorld[3][0], prevCameraToWorld[3][1],
                                                prevCameraToWorld[3][2] };
    }

    // Atmosphere
    {
        const auto& atmosphere = in.scene->getAtmosphere();
        const bool on = atmosphere.has_value() && atmosphere->density > 0.0f;
        pUniformData->hasFog = on ? 1u : 0u;
        pUniformData->fogSigmaT = on ? atmosphere->density : 0.0f;
        pUniformData->fogAnisotropy = on ? atmosphere->anisotropy : 0.0f;
        pUniformData->fogHeight = on ? atmosphere->height : 0.0f;
        pUniformData->fogAlbedo = on ? vector_float3{ atmosphere->color.x, atmosphere->color.y, atmosphere->color.z } :
                                       vector_float3{ 0.0f, 0.0f, 0.0f };
    }

    // Environment map
    if (in.environment->state().loaded)
    {
        const auto& envLight = in.scene->getEnvLight();
        pUniformData->hasEnvMap = 1;
        pUniformData->envMapWidth = (uint32_t)in.environment->state().mapTexture->width();
        pUniformData->envMapHeight = (uint32_t)in.environment->state().mapTexture->height();
        const float userIntensity = envLight.has_value() ? envLight->intensity : 1.0f;
        pUniformData->envMapIntensity = in.environment->state().autoScale * userIntensity;
        pUniformData->envMapRotation =
            envLight.has_value() ? envLight->rotationY * static_cast<float>(M_PI / 180.0) : 0.0f;
        pUniformData->envPdfScale = in.environment->state().pdfScale;
        const bool hasBackdrop = in.environment->state().backgroundTexture != nullptr;
        pUniformData->hasEnvBackground = hasBackdrop ? 1u : 0u;
        pUniformData->envBackgroundIntensity = hasBackdrop && envLight.has_value() ? envLight->backgroundIntensity : 1.0f;
        if (envLight.has_value())
        {
            pUniformData->envMapColorTint = { envLight->color.x, envLight->color.y, envLight->color.z };
        }
        else
        {
            pUniformData->envMapColorTint = { 1.0f, 1.0f, 1.0f };
        }
    }
    else
    {
        pUniformData->hasEnvMap = 0;
        pUniformData->hasEnvBackground = 0;
        pUniformData->envPdfScale = 0.0f;
        // A dome with no texture is still a light: a uniform sky of one colour,
        // which is what a V-Ray dome with `use_dome_tex` off is, and what the
        // kids' bedroom is lit by. It used to be nothing at all -- hasEnvMap
        // needs a texture and missColor was hard zero -- so the scene rendered
        // black but for the lamps.
        //
        // Carried on the miss colour rather than as a sampled light, and that is
        // not a shortcut. Next-event estimation exists to importance sample a
        // distribution the BSDF cannot see, and a constant environment has none:
        // for a Lambertian surface the cosine-weighted BSDF sample *is* the
        // optimal strategy, so what is left to converge is visibility alone.
        const auto& envLight = in.scene->getEnvLight();
        if (envLight.has_value())
        {
            const glm::float3 c = envLight->color * envLight->intensity;
            pUniformData->missColor = { c.x, c.y, c.z };
        }
    }

    pUniformTonemap->width = width;
    pUniformTonemap->height = height;
    pUniformTonemap->outWidth = outWidth;
    pUniformTonemap->outHeight = outHeight;
    // A debug view is data, not a picture: a normal, a roughness or a motion
    // vector means what it means, and a tone curve would misreport it. So the
    // pass runs as a straight copy instead of being skipped -- it is the only
    // thing that writes the texture the display samples, and skipping it left
    // every debug view showing the last tonemapped frame, which reads as the
    // control doing nothing at all.
    // Debug views paint palette colours -- a hash, a probe depth, an occupancy
    // flag, a bounce count -- and a photographic exposure applied to those means
    // nothing. `eSharcRadiance` is the exception: it holds radiance, and it is
    // only worth reading next to the render, under the render's own exposure and
    // curve.
    const bool radianceView = debug == (uint32_t)DebugMode::eSharcRadiance;
    const bool sharcPaletteView =
        pUniformData->sharcCapacity != 0u && SHARC_DEBUG_IS_SURFACE_VIEW(pUniformData->sharcDebug);
    const bool debugView = (debug != 0 && !radianceView) || sharcPaletteView;
    // 0 is ToneMapperType::eNone, which lives in the shader-side tonemappers.h.
    pUniformTonemap->tonemapperType = debugView ? 0u : settings.getAs<uint32_t>("render/pt/tonemapperType");
    pUniformTonemap->gamma = debugView ? 0.0f : settings.getAs<float>("render/post/gamma");
    pUniformTonemap->maxEDR = settings.getAs<float>("render/post/tonemapper/maxEDR");

    // --- Detect settings changes (member-based, not static) ---
    bool settingsChanged = false;
    settingsChanged |= (mPrevSettings.rectLightSamplingMethod != rectLightSamplingMethod);
    settingsChanged |= (mPrevSettings.samplerType != samplerType);
    settingsChanged |= (mPrevSettings.blueNoiseSwitchSpp != blueNoiseSwitchSpp);
    settingsChanged |= (mPrevSettings.enableAccumulation != enableAccumulation);
    settingsChanged |= (mPrevSettings.sspTotal > sspTotal);
    settingsChanged |= (mPrevSettings.spp != spp);
    settingsChanged |= (mPrevSettings.playbackBlur != playbackBlur);
    settingsChanged |= (mPrevSettings.shutterTime != shutterTime);
    settingsChanged |= (mPrevSettings.shutterMode != shutterMode);
    settingsChanged |= (mPrevSettings.enableMotionBlur != in.enableMotionBlur);
    settingsChanged |= (mPrevSettings.isMotionBlurVisible != isMotionBlurVisible);
    settingsChanged |= (mPrevSettings.enableCameraMotionBlur != enableCameraMotionBlur);
    settingsChanged |= (mPrevSettings.useDof != pUniformData->useDof);
    settingsChanged |= (mPrevSettings.focalDistance != pUniformData->focalDistance);
    settingsChanged |= (mPrevSettings.lensRadius != pUniformData->lensRadius);
    settingsChanged |= (mPrevSettings.apertureBlades != pUniformData->apertureBlades);
    // The other two aperture controls change the bokeh shape as much as the blade
    // count does, and without them an edit kept averaging into the frames taken with
    // the old aperture -- the setting looked inert until something else reset history.
    settingsChanged |= (mPrevSettings.bladeRotation != pUniformData->bladeRotation);
    settingsChanged |= (mPrevSettings.anamorphicRatio != pUniformData->anamorphicRatio);
    settingsChanged |= (mPrevSettings.shiftX != pUniformData->shiftX) || (mPrevSettings.shiftY != pUniformData->shiftY);
    settingsChanged |= (mPrevSettings.maxDepth != maxDepth);
    settingsChanged |= (mPrevSettings.debug != debug);
    settingsChanged |= (mPrevSettings.clampIndirect != pUniformData->clampIndirect);

    mPrevSettings.rectLightSamplingMethod = rectLightSamplingMethod;
    mPrevSettings.samplerType = samplerType;
    mPrevSettings.blueNoiseSwitchSpp = blueNoiseSwitchSpp;
    mPrevSettings.enableAccumulation = enableAccumulation;
    mPrevSettings.sspTotal = sspTotal;
    mPrevSettings.spp = spp;
    mPrevSettings.playbackBlur = playbackBlur;
    mPrevSettings.shutterTime = shutterTime;
    mPrevSettings.shutterMode = shutterMode;
    mPrevSettings.enableMotionBlur = in.enableMotionBlur;
    mPrevSettings.isMotionBlurVisible = isMotionBlurVisible;
    mPrevSettings.enableCameraMotionBlur = enableCameraMotionBlur;
    mPrevSettings.useDof = pUniformData->useDof;
    mPrevSettings.focalDistance = pUniformData->focalDistance;
    mPrevSettings.lensRadius = pUniformData->lensRadius;
    mPrevSettings.apertureBlades = pUniformData->apertureBlades;
    mPrevSettings.bladeRotation = pUniformData->bladeRotation;
    mPrevSettings.anamorphicRatio = pUniformData->anamorphicRatio;
    mPrevSettings.shiftX = pUniformData->shiftX;
    mPrevSettings.shiftY = pUniformData->shiftY;
    mPrevSettings.maxDepth = maxDepth;
    mPrevSettings.debug = debug;
    mPrevSettings.clampIndirect = pUniformData->clampIndirect;

    /* settingsChanged reported via FillResult; orchestrator resets subframe/history */

    // Matrix copies: glm and simd both use column-major layout
    const glm::float4x4 invView = glm::inverse(camera.matrices.view);
    std::memcpy(&pUniformData->viewToWorld, glm::value_ptr(invView), sizeof(float4x4));
    std::memcpy(&pUniformData->clipToView, glm::value_ptr(camera.matrices.invPerspective), sizeof(float4x4));

    {
        const glm::float4x4 prevInvView = glm::inverse(in.prevMotionBlurView->mCamMatrices.view);
        std::memcpy(&pUniformData->prevViewToWorld, glm::value_ptr(prevInvView), sizeof(float4x4));
        std::memcpy(&pUniformData->prevClipToView, glm::value_ptr(in.prevMotionBlurView->mCamMatrices.invPerspective),
                    sizeof(float4x4));
    }

    pUniformData->subframeIndex = in.subframeIndex;

    // Photometric exposure
    const float filmIso = settings.getAs<float>("render/post/tonemapper/filmIso");
    const float cm2_factor = settings.getAs<float>("render/post/tonemapper/cm2_factor");
    const float fStop = settings.getAs<float>("render/post/tonemapper/fStop");
    const float shutterSpeed = settings.getAs<float>("render/post/tonemapper/shutterSpeed");
    // Specifies the main color temperature of the light sources; the color that will be mapped to “white” on output,
    // e.g., an incoming color of this hue/saturation will be mapped to grayscale, but its intensity will remain
    // unchanged. This is similar to white balance controls on digital cameras.
    float3 whitePoint{ 1.0f, 1.0f, 1.0f };
    auto all = [](float3 v) { return v.x > 0.0f && v.y > 0.0f && v.z > 0.0f; };
    float3 exposureValue = all(whitePoint) ? 1.0f / whitePoint : float3(1.0f);
    const float lum = simd::dot(exposureValue, float3{ 0.299f, 0.587f, 0.114f });
    if (filmIso > 0.0f)
    {
        // See https://www.nayuki.io/page/the-photographic-exposure-equation
        exposureValue *= cm2_factor * filmIso / (shutterSpeed * fStop * fStop) / 100.0f;
    }
    else
    {
        exposureValue *= cm2_factor;
    }
    exposureValue /= lum;
    // Exposure is part of the picture, not part of the data, so it goes too.
    pUniformTonemap->exposureValue = debugView ? float3(1.0f) : exposureValue;
    pUniformData->exposureValue = exposureValue; // need for proper accumulation


    FillResult out{};
    out.uniforms = pUniformData;
    out.tonemap = pUniformTonemap;
    out.uniformBuffer = pUniformBuffer;
    out.tonemapBuffer = pUniformTMBuffer;
    out.accumulationActive = accumulationActive;
    out.effectiveAccumulation = effectiveAccumulation;
    out.remainingSamples = remainingSamples;
    out.jitterX = pUniformData->jitterX;
    out.jitterY = pUniformData->jitterY;
    out.settingsChanged = settingsChanged;
    const auto samplesPerLaunch = pUniformData->samples_per_launch;
    // Nothing to trace once the budget is spent: the accumulated estimate is the
    // final one and the frames after it only re-run the display transform.
    //
    // That includes the denoised path, which used to keep tracing a sample per
    // frame forever. Its output is a texture the tone curve reads, so the frame
    // it produced at the last sample can be re-tonemapped as it stands -- which
    // is what makes exposure and the tone curve still respond after the estimator
    // has stopped, at the price of one dispatch rather than a whole re-render.
    out.samplesThisLaunch = accumulationActive ? std::min(samplesPerLaunch, remainingSamples) :
                                                 (effectiveAccumulation ? 0u : samplesPerLaunch);
    return out;
}

} // namespace oka::metal

