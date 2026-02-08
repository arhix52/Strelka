#include <cstddef>
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "MetalRender.h"
#include "MetalBuffer.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <unistd.h>

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <glm/ext/matrix_relational.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
#include <log.h>

#include <simd/simd.h>

#include "ShaderTypes.h"

using namespace oka;
namespace fs = std::filesystem;

MetalRender::MetalRender(/* args */) = default;

MetalRender::~MetalRender()
{
    @autoreleasepool
    {
        // Drain the GPU: submit a fence and wait for all prior work to finish.
        // Then spin until the async completion handler has set mRenderBusy=false,
        // ensuring no handler is still accessing members when we release resources.
        if (mCommandQueue)
        {
            MTL::CommandBuffer* fence = mCommandQueue->commandBuffer();
            if (fence)
            {
                fence->commit();
                fence->waitUntilCompleted();
            }
        }
        while (mRenderBusy.load(std::memory_order_acquire))
        {
            usleep(100);
        }

        // Delete C++ wrapper objects (MetalBuffer) for async output.
        // MetalBuffer::~MetalBuffer calls release() on its inner MTL::Buffer.
        delete mAsyncOutputBuffers[0];
        mAsyncOutputBuffers[0] = nullptr;
        delete mAsyncOutputBuffers[1];
        mAsyncOutputBuffers[1] = nullptr;

        // Free heap-allocated Mesh structs (but NOT their Metal resources,
        // which are ref-counted by the device and released below).
        for (auto* mesh : mMetalMeshes)
            delete mesh;
        mMetalMeshes.clear();

        // Metal objects: release only those we explicitly created with newXxx().
        // Some objects (e.g. acceleration structures returned by newAccelerationStructure)
        // may share internal references. Use a flat release-and-null pattern to
        // avoid double-release from pointer authentication failures.
        auto safeRelease = [](auto*& p) {
            if (p) { p->release(); p = nullptr; }
        };

        // Acceleration structures
        for (auto*& as : mPrimitiveAccelerationStructures)
            safeRelease(as);
        safeRelease(mInstanceAccelerationStructure);

        // Material textures
        for (auto*& tex : mMaterialTextures)
            safeRelease(tex);

        // Buffers
        safeRelease(mAccumulationBuffer);
        safeRelease(mLightBuffer);
        safeRelease(mVertexBuffer);
        safeRelease(mIndexBuffer);
        safeRelease(mInstanceBuffer);
        safeRelease(mMaterialBuffer);
        safeRelease(mSkinDataBuffer);
        safeRelease(mJointMatricesBuffer);
        safeRelease(mPrevVertexBuffer);
        safeRelease(mInstanceDataBuffer);
        for (auto*& buf : mUniformBuffers) safeRelease(buf);
        for (auto*& buf : mUniformTMBuffers) safeRelease(buf);

        // Environment map
        safeRelease(mEnvMapTexture);
        safeRelease(mEnvCdfXBuffer);
        safeRelease(mEnvCdfYBuffer);

        // Pipeline states
        safeRelease(mPathTracingPSO);
        safeRelease(mTonemapperPSO);
        safeRelease(mSkinningPSO);
        safeRelease(mTriangleUpdatePSO);

        // Queue & device (release last)
        safeRelease(mCommandQueue);
        safeRelease(mDevice);
    }
}

void MetalRender::triggerRenderIfIdle()
{
    if (mRenderBusy.load())
        return;

    const uint32_t w = getSettings()->getAs<uint32_t>("render/width");
    const uint32_t h = getSettings()->getAs<uint32_t>("render/height");

    // Pick the buffer that is NOT currently being displayed
    int ri = mReadyIndex.load();
    mWriteIndex = (ri >= 0) ? (1 - ri) : 0;

    // Create or resize the write buffer
    if (!mAsyncOutputBuffers[mWriteIndex])
    {
        BufferDesc desc{};
        desc.format = BufferFormat::FLOAT4;
        desc.width = w;
        desc.height = h;
        mAsyncOutputBuffers[mWriteIndex] = createBuffer(desc);
    }
    else if (mAsyncOutputBuffers[mWriteIndex]->width() != w ||
             mAsyncOutputBuffers[mWriteIndex]->height() != h)
    {
        mAsyncOutputBuffers[mWriteIndex]->resize(w, h);
    }

    // Also ensure the other buffer exists (display may need it)
    int otherIdx = 1 - mWriteIndex;
    if (!mAsyncOutputBuffers[otherIdx])
    {
        BufferDesc desc{};
        desc.format = BufferFormat::FLOAT4;
        desc.width = w;
        desc.height = h;
        mAsyncOutputBuffers[otherIdx] = createBuffer(desc);
    }
    else if (mAsyncOutputBuffers[otherIdx]->width() != w ||
             mAsyncOutputBuffers[otherIdx]->height() != h)
    {
        // Resize ready buffer too — display will pick up new size next frame
        mAsyncOutputBuffers[otherIdx]->resize(w, h);
        mReadyIndex.store(-1); // invalidate since we resized
    }

    mRenderBusy.store(true);
    render(mAsyncOutputBuffers[mWriteIndex]);
}

Buffer* MetalRender::getReadyBuffer()
{
    int ri = mReadyIndex.load();
    if (ri < 0)
        return nullptr;
    return mAsyncOutputBuffers[ri];
}

void MetalRender::init()
{
    mDevice = MTL::CreateSystemDefaultDevice();
    if (!mDevice)
    {
        STRELKA_FATAL("Failed to create Metal device");
        return;
    }
    mCommandQueue = mDevice->newCommandQueue();
    buildComputePipeline();
    buildTonemapperPipeline();
}

MTL::Texture* MetalRender::loadTextureFromFile(const std::string& fileName)
{
    int texWidth = 0;
    int texHeight = 0;
    int texChannels = 0;
    stbi_uc* data = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (data == nullptr)
    {
        STRELKA_ERROR("Unable to load texture from file: {}", fileName.c_str());
        return nullptr;
    }
    MTL::TextureDescriptor* pTextureDesc = MTL::TextureDescriptor::alloc()->init();
    pTextureDesc->setWidth(texWidth);
    pTextureDesc->setHeight(texHeight);
    pTextureDesc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    pTextureDesc->setTextureType(MTL::TextureType2D);
    pTextureDesc->setStorageMode(MTL::StorageModeManaged);
    pTextureDesc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead);

    MTL::Texture* pTexture = mDevice->newTexture(pTextureDesc);

    const MTL::Region region = MTL::Region::Make3D(0, 0, 0, texWidth, texHeight, 1);
    pTexture->replaceRegion(region, 0, data, 4ull * texWidth);

    pTextureDesc->release();
    return pTexture;
}

void MetalRender::createMetalMaterials()
{
    using simd::float3;
    const std::vector<Scene::MaterialDescription>& matDescs = mScene->getMaterials();
    std::vector<Material> gpuMaterials;
    const fs::path resourcePath = getSettings()->getAs<std::string>("resource/searchPath");

    auto loadTex = [&](const std::string& path) -> MTL::ResourceID {
        if (path.empty()) return MTL::ResourceID{};
        const fs::path fullPath = resourcePath / path;
        MTL::Texture* tex = loadTextureFromFile(fullPath.string());
        if (tex) mMaterialTextures.push_back(tex);
        return tex ? tex->gpuResourceID() : MTL::ResourceID{};
    };

    for (const Scene::MaterialDescription& currMatDesc : matDescs)
    {
        Material material = {};
        const auto& p = currMatDesc.params;
        material.base_color = packed_float3(simd_make_float3(p.base_color.x, p.base_color.y, p.base_color.z));
        material.metallic = p.metallic;
        material.roughness = p.roughness;
        material.ior = p.ior;
        material.specular = p.specular;
        material.specular_tint = p.specular_tint;
        material.transmission = p.transmission;
        material.clearcoat = p.clearcoat;
        material.clearcoat_roughness = p.clearcoat_roughness;
        material.anisotropy = p.anisotropy;
        material.emission = packed_float3(simd_make_float3(p.emission.x, p.emission.y, p.emission.z));
        material.emission_strength = p.emission_strength;
        material.normal_scale = p.normal_scale;
        material.occlusion_strength = p.occlusion_strength;
        material.alpha_cutoff = p.alpha_cutoff;
        material.material_type = p.material_type;
        material.thin_walled = p.thin_walled;

        material.baseColorTexture = loadTex(currMatDesc.baseColorTexPath);
        material.metallicRoughnessTexture = loadTex(currMatDesc.metallicRoughnessTexPath);
        material.normalTexture = loadTex(currMatDesc.normalTexPath);
        material.emissionTexture = loadTex(currMatDesc.emissionTexPath);
        material.occlusionTexture = loadTex(currMatDesc.occlusionTexPath);

        gpuMaterials.push_back(material);
    }

    const size_t materialsDataSize = sizeof(Material) * gpuMaterials.size();
    if (materialsDataSize > 0)
    {
        mMaterialBuffer = mDevice->newBuffer(materialsDataSize, MTL::ResourceStorageModeManaged);
        memcpy(mMaterialBuffer->contents(), gpuMaterials.data(), materialsDataSize);
        mMaterialBuffer->didModifyRange(NS::Range::Make(0, mMaterialBuffer->length()));
    }
    else
    {
        mMaterialBuffer = nullptr;
    }
}

void MetalRender::render(Buffer* output)
{
    using simd::float3;
    using simd::float4;
    using simd::float4x4;
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    SharedContext& ctx = getSharedContext();

    if (ctx.mFrameNumber == 0)
    {
        buildBuffers();
        createMetalMaterials();
        createAccelerationStructures();
        // create accum buffer, we don't need cpu access, make it device only
        mAccumulationBuffer = mDevice->newBuffer(
            output->width() * output->height() * output->getElementSize(), MTL::ResourceStorageModePrivate);

        // Initialize skinning pipeline if scene has skeletal data
        if (!mScene->getVerticesSkinData().empty())
        {
            buildSkinningPipeline();
            createSkinDataBuffer();
            allocJointMatrices();
        }

        // Load environment map if specified
        const auto& envLight = mScene->getEnvLight();
        if (envLight.has_value() && !envLight->texturePath.empty())
        {
            const std::string resourcePathStr = getSettings()->getAs<std::string>("resource/searchPath");
            const fs::path envTexPath = fs::path(resourcePathStr) / envLight->texturePath;
            loadEnvMap(envTexPath.string());
        }
    }

    mFrameIndex = (mFrameIndex + 1) % kMaxFramesInFlight;

    // Recreate accumulation buffer if output size changed
    const uint32_t width = output->width();
    const uint32_t height = output->height();
    const size_t requiredSize = width * height * output->getElementSize();
    if (mAccumulationBuffer && requiredSize != mAccumulationBuffer->length())
    {
        mAccumulationBuffer->release();
        mAccumulationBuffer = mDevice->newBuffer(requiredSize, MTL::ResourceStorageModePrivate);
        ctx.mSubframeIndex = 0;
    }

    // Update motion blur enable state from settings each frame
    mEnableMotionBlur = getSettings()->getAs<bool>("render/enableMotionBlur");

    bool motionBlurCameraSet = false; // track if animation block sets prev camera

    // Animation detection: two-pass at t_open / t_close for motion blur (CHANGED-only)
    {
        SettingsManager& animSettings = *getSettings();
        std::vector<oka::Scene::Animation>& animations = mScene->getAnimations();

        // Collect target times and detect which animations actually changed
        constexpr float EPSILON = 1e-6f;
        bool animStateChanged = false;
        float maxTimeDelta = 0.0f; // track largest time jump for scrub detection
        const size_t animCount = animations.size();
        mAnimTargetTimes.resize(animCount);
        mAnimChanged.resize(animCount);
        std::fill(mAnimChanged.begin(), mAnimChanged.end(), false);
        for (int i = 0; i < (int)animCount; ++i)
        {
            char key[64];
            snprintf(key, sizeof(key), "render/animation/anim%d/time", i);
            mAnimTargetTimes[i] = animSettings.getAs<float>(key);
            const float delta = std::abs(animations[i].current - mAnimTargetTimes[i]);
            if (delta > EPSILON)
            {
                mAnimChanged[i] = true;
                animStateChanged = true;
                maxTimeDelta = std::max(maxTimeDelta, delta);
            }
        }

        if (animStateChanged)
        {
            const float shutterDuration = mEnableMotionBlur
                ? animSettings.getAs<float>("render/motionBlur/shutterTime") : 0.0f;
            const uint32_t shutterMode = animSettings.getAs<uint32_t>("render/motionBlur/shutterMode");

            if (mEnableMotionBlur && mPrevVertexBuffer && shutterDuration > 0.0f)
            {
                // Shutter offset: Centered straddles t_anim, Leading closes at t_anim,
                // Trailing opens at t_anim
                float shutterOffset = 0.0f;
                switch (shutterMode)
                {
                case 0: shutterOffset = -shutterDuration * 0.5f; break; // Centered
                case 1: shutterOffset = -shutterDuration;        break; // Leading
                case 2: shutterOffset = 0.0f;                    break; // Trailing
                }

                // --- Pass 1: evaluate CHANGED animations at t_open ---
                bool pass1Skeletal = false;
                for (int i = 0; i < (int)animations.size(); ++i)
                {
                    if (!mAnimChanged[i]) continue;
                    float tOpen = mAnimTargetTimes[i] + shutterOffset;
                    tOpen = std::clamp(tOpen, animations[i].start, animations[i].end);
                    animations[i].current = tOpen;
                    pass1Skeletal |= mScene->applyAnimation(i);
                }

                // Capture camera state at t_open for camera motion blur
                const uint32_t selCam = animSettings.getAs<uint32_t>("render/selectedCamera");
                oka::Camera& prevCam = mScene->getCamera(selCam);
                prevCam.updateAspectRatio(width / (float)height);
                prevCam.updateViewMatrix();
                mPrevMotionBlurView.mCamMatrices = prevCam.matrices;
                motionBlurCameraSet = true;

                if (pass1Skeletal)
                {
                    applySkinning();
                }
                // Always copy current VB to prevVB — ensures prev state is consistent
                // even when only camera (not skeleton) animation changed
                copyVertexBufferToPrev();

                // --- Pass 2: evaluate CHANGED animations at t_close ---
                bool pass2Skeletal = false;
                for (int i = 0; i < (int)animations.size(); ++i)
                {
                    if (!mAnimChanged[i]) continue;
                    float tClose = mAnimTargetTimes[i] + shutterOffset + shutterDuration;
                    tClose = std::clamp(tClose, animations[i].start, animations[i].end);
                    animations[i].current = tClose;
                    pass2Skeletal |= mScene->applyAnimation(i);
                }

                if (pass2Skeletal)
                {
                    applySkinning();
                    // Full rebuild periodically or on large scrubs, but throttle to avoid
                    // back-to-back full rebuilds during rapid scrubbing (min 5 frames apart)
                    const bool wantsFullRebuild = (mBlasUpdateCount >= 10) ||
                                                  (maxTimeDelta > shutterDuration * 2.0f);
                    const bool fullRebuild = wantsFullRebuild && (mFramesSinceFullRebuild >= 5);
                    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
                    for (int mi = 0; mi < (int)meshes.size(); ++mi)
                    {
                        if (mMetalMeshes[mi]->mIsSkeletal)
                        {
                            if (fullRebuild)
                                rebuildBLAS(mi);
                            else
                                refitBLAS(mi);
                        }
                    }
                    mBlasUpdateCount = fullRebuild ? 0 : (mBlasUpdateCount + 1);
                    mFramesSinceFullRebuild = fullRebuild ? 0 : (mFramesSinceFullRebuild + 1);
                }
                updateInstanceTransforms();
                rebuildTLAS();

                // Restore target times so next-frame EPSILON check is stable
                for (int i = 0; i < (int)animations.size(); ++i)
                    animations[i].current = mAnimTargetTimes[i];
            }
            else
            {
                // Motion blur disabled or no shutter: single-pass at target time
                bool accelStructureDirty = false;
                for (int i = 0; i < (int)animations.size(); ++i)
                {
                    if (!mAnimChanged[i]) continue;
                    animations[i].current = mAnimTargetTimes[i];
                    accelStructureDirty |= mScene->applyAnimation(i);
                }

                if (accelStructureDirty)
                {
                    applySkinning();
                    // Sync prevVB with current VB — motion BVH needs both keyframes
                    // consistent when motion blur is off (otherwise keyframe 0 is stale)
                    copyVertexBufferToPrev();
                    // Full rebuild periodically or on large scrubs, throttled
                    const bool wantsFullRebuild = (mBlasUpdateCount >= 10) ||
                                                  (maxTimeDelta > 0.1f);
                    const bool fullRebuild = wantsFullRebuild && (mFramesSinceFullRebuild >= 5);
                    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
                    for (int mi = 0; mi < (int)meshes.size(); ++mi)
                    {
                        if (mMetalMeshes[mi]->mIsSkeletal)
                        {
                            if (fullRebuild)
                                rebuildBLAS(mi);
                            else
                                refitBLAS(mi);
                        }
                    }
                    mBlasUpdateCount = fullRebuild ? 0 : (mBlasUpdateCount + 1);
                    mFramesSinceFullRebuild = fullRebuild ? 0 : (mFramesSinceFullRebuild + 1);
                    rebuildTLAS();
                }
                else
                {
                    updateInstanceTransforms();
                    rebuildTLAS();
                }
            }
            ctx.mSubframeIndex = 0;
        }
    }

    SettingsManager& settings = *getSettings();

    const uint32_t selectedCamera = settings.getAs<uint32_t>("render/selectedCamera");
    oka::Camera& camera = mScene->getCamera(selectedCamera);
    camera.updateAspectRatio(width / (float)height);
    camera.updateViewMatrix();

    const View currView{camera.matrices};

    if (!motionBlurCameraSet && ctx.mFrameNumber == 0)
    {
        mPrevMotionBlurView.mCamMatrices = camera.matrices;
    }

    if (glm::any(glm::notEqual(currView.mCamMatrices.perspective, mPrevView.mCamMatrices.perspective)) ||
        glm::any(glm::notEqual(currView.mCamMatrices.view, mPrevView.mCamMatrices.view)))
    {
        ctx.mSubframeIndex = 0;
    }

    // --- Cache all settings once per frame ---
    const uint32_t spp = settings.getAs<uint32_t>("render/pt/spp");
    const bool enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    const uint32_t maxDepth = settings.getAs<uint32_t>("render/pt/depth");
    const uint32_t debug = settings.getAs<uint32_t>("render/pt/debug");
    const uint32_t rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    const uint32_t samplerType = settings.getAs<uint32_t>("render/pt/samplerType");
    const uint32_t sspTotal = settings.getAs<uint32_t>("render/pt/sppTotal");
    const bool isMotionBlurVisible = settings.getAs<bool>("render/isMotionBlurVisible");
    const bool enableCameraMotionBlur = settings.getAs<bool>("render/enableCameraMotionBlur");

    MTL::Buffer* pUniformBuffer = mUniformBuffers[mFrameIndex];
    MTL::Buffer* pUniformTMBuffer = mUniformTMBuffers[mFrameIndex];
    auto pUniformData = reinterpret_cast<Uniforms*>(pUniformBuffer->contents());
    auto pUniformTonemap = reinterpret_cast<UniformsTonemap*>(pUniformTMBuffer->contents());
    pUniformData->frameIndex = mFrameIndex;
    pUniformData->subframeIndex = ctx.mSubframeIndex;
    pUniformData->height = height;
    pUniformData->width = width;
    pUniformData->numLights = mScene->getLightsDesc().size();
    pUniformData->samples_per_launch = spp;
    pUniformData->enableAccumulation = (uint32_t)enableAccumulation;
    pUniformData->missColor = float3(0.0f);
    pUniformData->maxDepth = maxDepth;
    pUniformData->debug = debug;
    pUniformData->enableMotionBlur = mEnableMotionBlur ? 1 : 0;
    pUniformData->isMotionBlurVisible = (uint32_t)isMotionBlurVisible;
    pUniformData->enableCameraMotionBlur = (uint32_t)enableCameraMotionBlur;
    pUniformData->rectLightSamplingMethod = rectLightSamplingMethod;
    pUniformData->samplerType = samplerType;

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

    // Environment map
    if (mEnvMapLoaded)
    {
        const auto& envLight = mScene->getEnvLight();
        pUniformData->hasEnvMap = 1;
        pUniformData->envMapWidth = (uint32_t)mEnvMapTexture->width();
        pUniformData->envMapHeight = (uint32_t)mEnvMapTexture->height();
        const float userIntensity = envLight.has_value() ? envLight->intensity : 1.0f;
        pUniformData->envMapIntensity = mEnvMapAutoScale * userIntensity;
        pUniformData->envMapRotation = envLight.has_value() ? envLight->rotationY * (M_PI / 180.0f) : 0.0f;
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
    }

    pUniformTonemap->width = width;
    pUniformTonemap->height = height;
    pUniformTonemap->tonemapperType = settings.getAs<uint32_t>("render/pt/tonemapperType");
    pUniformTonemap->gamma = settings.getAs<float>("render/post/gamma");
    pUniformTonemap->maxEDR = settings.getAs<float>("render/post/tonemapper/maxEDR");

    // --- Detect settings changes (member-based, not static) ---
    bool settingsChanged = false;
    settingsChanged |= (mPrevSettings.rectLightSamplingMethod != rectLightSamplingMethod);
    settingsChanged |= (mPrevSettings.samplerType != samplerType);
    settingsChanged |= (mPrevSettings.enableAccumulation != enableAccumulation);
    settingsChanged |= (mPrevSettings.sspTotal > sspTotal);
    settingsChanged |= (mPrevSettings.spp != spp);
    settingsChanged |= (mPrevSettings.enableMotionBlur != mEnableMotionBlur);
    settingsChanged |= (mPrevSettings.isMotionBlurVisible != isMotionBlurVisible);
    settingsChanged |= (mPrevSettings.enableCameraMotionBlur != enableCameraMotionBlur);
    settingsChanged |= (mPrevSettings.useDof != pUniformData->useDof);
    settingsChanged |= (mPrevSettings.focalDistance != pUniformData->focalDistance);
    settingsChanged |= (mPrevSettings.lensRadius != pUniformData->lensRadius);
    settingsChanged |= (mPrevSettings.apertureBlades != pUniformData->apertureBlades);
    settingsChanged |= (mPrevSettings.shiftX != pUniformData->shiftX) || (mPrevSettings.shiftY != pUniformData->shiftY);
    settingsChanged |= (mPrevSettings.maxDepth != maxDepth);
    settingsChanged |= (mPrevSettings.debug != debug);

    mPrevSettings.rectLightSamplingMethod = rectLightSamplingMethod;
    mPrevSettings.samplerType = samplerType;
    mPrevSettings.enableAccumulation = enableAccumulation;
    mPrevSettings.sspTotal = sspTotal;
    mPrevSettings.spp = spp;
    mPrevSettings.enableMotionBlur = mEnableMotionBlur;
    mPrevSettings.isMotionBlurVisible = isMotionBlurVisible;
    mPrevSettings.enableCameraMotionBlur = enableCameraMotionBlur;
    mPrevSettings.useDof = pUniformData->useDof;
    mPrevSettings.focalDistance = pUniformData->focalDistance;
    mPrevSettings.lensRadius = pUniformData->lensRadius;
    mPrevSettings.apertureBlades = pUniformData->apertureBlades;
    mPrevSettings.shiftX = pUniformData->shiftX;
    mPrevSettings.shiftY = pUniformData->shiftY;
    mPrevSettings.maxDepth = maxDepth;
    mPrevSettings.debug = debug;

    if (settingsChanged)
    {
        ctx.mSubframeIndex = 0;
    }

    // Matrix copies: glm and simd both use column-major layout
    const glm::float4x4 invView = glm::inverse(camera.matrices.view);
    std::memcpy(&pUniformData->viewToWorld, glm::value_ptr(invView), sizeof(float4x4));
    std::memcpy(&pUniformData->clipToView, glm::value_ptr(camera.matrices.invPerspective), sizeof(float4x4));

    {
        const glm::float4x4 prevInvView = glm::inverse(mPrevMotionBlurView.mCamMatrices.view);
        std::memcpy(&pUniformData->prevViewToWorld, glm::value_ptr(prevInvView), sizeof(float4x4));
        std::memcpy(&pUniformData->prevClipToView, glm::value_ptr(mPrevMotionBlurView.mCamMatrices.invPerspective), sizeof(float4x4));
    }

    pUniformData->subframeIndex = ctx.mSubframeIndex;

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
    pUniformTonemap->exposureValue = exposureValue;
    pUniformData->exposureValue = exposureValue; // need for proper accumulation

    const auto samplesPerLaunch = pUniformData->samples_per_launch;
    const int32_t leftSpp = sspTotal - ctx.mSubframeIndex;
    // if accumulation is off then launch selected samples per pixel
    const uint32_t samplesThisLaunch =
        enableAccumulation ? std::min((int32_t)samplesPerLaunch, leftSpp) : samplesPerLaunch;
    if (samplesThisLaunch != 0)
    {
        pUniformData->samples_per_launch = samplesThisLaunch; // TODO: implement in pt kernel

        pUniformBuffer->didModifyRange(NS::Range::Make(0, sizeof(Uniforms)));
        pUniformTMBuffer->didModifyRange(NS::Range::Make(0, sizeof(UniformsTonemap)));

        MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* pComputeEncoder = pCmd->computeCommandEncoder();
        if (mMaterialBuffer != nullptr)
        {
            pComputeEncoder->useResource(mMaterialBuffer, MTL::ResourceUsageRead);
        }
        if (mLightBuffer != nullptr)
        {
            pComputeEncoder->useResource(mLightBuffer, MTL::ResourceUsageRead);
        }
        if (mInstanceAccelerationStructure != nullptr)
        {
            pComputeEncoder->useResource(mInstanceAccelerationStructure, MTL::ResourceUsageRead);
        }
        for (const MTL::AccelerationStructure* primitiveAccel : mPrimitiveAccelerationStructures)
        {
            pComputeEncoder->useResource(primitiveAccel, MTL::ResourceUsageRead);
        }
        for (auto& materialTexture : mMaterialTextures)
        {
            pComputeEncoder->useResource(materialTexture, MTL::ResourceUsageRead);
        }
        pComputeEncoder->useResource(((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageWrite);
        pComputeEncoder->useResource(mPrevVertexBuffer, MTL::ResourceUsageRead);
        pComputeEncoder->useResource(mIndexBuffer, MTL::ResourceUsageRead);
        pComputeEncoder->useResource(mInstanceDataBuffer, MTL::ResourceUsageRead);

        pComputeEncoder->setComputePipelineState(mPathTracingPSO);
        pComputeEncoder->setBuffer(pUniformBuffer, 0, 0);
        pComputeEncoder->setBuffer(mInstanceBuffer, 0, 1);
        pComputeEncoder->setAccelerationStructure(mInstanceAccelerationStructure, 2);
        pComputeEncoder->setBuffer(mLightBuffer, 0, 3);
        pComputeEncoder->setBuffer(mMaterialBuffer, 0, 4);
        // Output
        pComputeEncoder->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 5);
        pComputeEncoder->setBuffer(mAccumulationBuffer, 0, 6);
        // Motion blur buffers
        pComputeEncoder->setBuffer(mPrevVertexBuffer, 0, 7);
        pComputeEncoder->setBuffer(mIndexBuffer, 0, 8);
        pComputeEncoder->setBuffer(mInstanceDataBuffer, 0, 9);
        // Environment map buffers
        if (mEnvCdfXBuffer)
        {
            pComputeEncoder->useResource(mEnvCdfXBuffer, MTL::ResourceUsageRead);
            pComputeEncoder->setBuffer(mEnvCdfXBuffer, 0, 10);
        }
        if (mEnvCdfYBuffer)
        {
            pComputeEncoder->useResource(mEnvCdfYBuffer, MTL::ResourceUsageRead);
            pComputeEncoder->setBuffer(mEnvCdfYBuffer, 0, 11);
        }
        if (mEnvMapTexture)
        {
            pComputeEncoder->useResource(mEnvMapTexture, MTL::ResourceUsageRead);
            pComputeEncoder->setTexture(mEnvMapTexture, 0);
        }
        if (mInstanceBuffer != nullptr)
        {
            const MTL::Size gridSize = MTL::Size(width, height, 1);
            const MTL::Size threadgroupSize(8, 8, 1);
            pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);
        }
        // Disable tonemapping for debug output
        if (pUniformData->debug == 0)
        {
            pComputeEncoder->setComputePipelineState(mTonemapperPSO);
            pComputeEncoder->useResource(
                ((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
            pComputeEncoder->setBuffer(pUniformTMBuffer, 0, 0);
            pComputeEncoder->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
            {
                const MTL::Size gridSize = MTL::Size(width, height, 1);
                const MTL::Size threadgroupSize(8, 8, 1);
                pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);
            }
        }

        pComputeEncoder->endEncoding();

        if (enableAccumulation)
        {
            ctx.mSubframeIndex += samplesThisLaunch;
        }
        else
        {
            ctx.mSubframeIndex = 0;
        }

        // Completion handler for async double-buffered output
        if (mRenderBusy.load())
        {
            int writeIdx = mWriteIndex;
            pCmd->addCompletedHandler(MTL::HandlerFunction([this, writeIdx](MTL::CommandBuffer* cb) {
                double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
                mLastRenderTimeMs.store(gpuMs, std::memory_order_relaxed);
                mReadyIndex.store(writeIdx);
                mRenderBusy.store(false);
            }));
        }
        pCmd->commit();
    }
    else
    {
        MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();

        MTL::BlitCommandEncoder* pBlitEncoder = pCmd->blitCommandEncoder();
        pBlitEncoder->copyFromBuffer(
            mAccumulationBuffer, 0, ((MetalBuffer*)output)->getNativePtr(), 0, width * height * sizeof(float4));
        pBlitEncoder->endEncoding();

        // Disable tonemapping for debug output
        if (pUniformData->debug == 0)
        {
            MTL::ComputeCommandEncoder* pComputeEncoder = pCmd->computeCommandEncoder();

            pComputeEncoder->setComputePipelineState(mTonemapperPSO);
            pComputeEncoder->useResource(
                ((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
            pComputeEncoder->setBuffer(pUniformTMBuffer, 0, 0);
            pComputeEncoder->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
            {
                const MTL::Size gridSize = MTL::Size(width, height, 1);
                const MTL::Size threadgroupSize(8, 8, 1);
                pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);
            }
            pComputeEncoder->endEncoding();
        }

        // Completion handler for async double-buffered output
        if (mRenderBusy.load())
        {
            int writeIdx = mWriteIndex;
            pCmd->addCompletedHandler(MTL::HandlerFunction([this, writeIdx](MTL::CommandBuffer* cb) {
                double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
                mLastRenderTimeMs.store(gpuMs, std::memory_order_relaxed);
                mReadyIndex.store(writeIdx);
                mRenderBusy.store(false);
            }));
        }
        pCmd->commit();
    }
    pPool->release();

    mPrevView = currView;
    ctx.mFrameNumber++;
}

Buffer* MetalRender::createBuffer(const BufferDesc& desc)
{
    assert(mDevice);
    const size_t size = desc.height * desc.width * Buffer::getElementSize(desc.format);
    assert(size != 0);
    MTL::Buffer* buff = mDevice->newBuffer(size, MTL::ResourceStorageModeManaged);
    assert(buff);
    auto res = new MetalBuffer(buff, desc.format, desc.width, desc.height);
    assert(res);
    return res;
}

void MetalRender::buildComputePipeline()
{
    NS::Error* pError = nullptr;
    MTL::Library* pComputeLibrary =
        mDevice->newLibrary(NS::String::string("./metal/shaders/pathtrace.metallib", NS::UTF8StringEncoding), &pError);
    if (!pComputeLibrary)
    {
        STRELKA_FATAL("{}", pError->localizedDescription()->utf8String());
        assert(false);
    }
    MTL::Function* pPathTraceFn =
        pComputeLibrary->newFunction(NS::String::string("raytracingKernel", NS::UTF8StringEncoding));
    mPathTracingPSO = mDevice->newComputePipelineState(pPathTraceFn, &pError);
    if (!mPathTracingPSO)
    {
        STRELKA_FATAL("{}", pError->localizedDescription()->utf8String());
        assert(false);
    }

    pPathTraceFn->release();
    pComputeLibrary->release();
}

void MetalRender::buildTonemapperPipeline()
{
    NS::Error* pError = nullptr;
    MTL::Library* pComputeLibrary =
        mDevice->newLibrary(NS::String::string("./metal/shaders/tonemapper.metallib", NS::UTF8StringEncoding), &pError);
    if (!pComputeLibrary)
    {
        STRELKA_FATAL("{}", pError->localizedDescription()->utf8String());
        assert(false);
    }
    MTL::Function* pTonemapperFn =
        pComputeLibrary->newFunction(NS::String::string("toneMappingComputeShader", NS::UTF8StringEncoding));
    mTonemapperPSO = mDevice->newComputePipelineState(pTonemapperFn, &pError);
    if (!mTonemapperPSO)
    {
        STRELKA_FATAL("{}", pError->localizedDescription()->utf8String());
        assert(false);
    }

    pTonemapperFn->release();
    pComputeLibrary->release();
}

void MetalRender::buildBuffers()
{
    const std::vector<Scene::Vertex>& vertices = mScene->getVertices();
    const std::vector<uint32_t>& indices = mScene->getIndices();
    const std::vector<Scene::Light>& lightDescs = mScene->getLights();

    static_assert(sizeof(Scene::Light) == sizeof(UniformLight));
    const size_t lightBufferSize = sizeof(Scene::Light) * lightDescs.size();
    const size_t vertexDataSize = sizeof(Scene::Vertex) * vertices.size();
    const size_t indexDataSize = sizeof(uint32_t) * indices.size();

    MTL::Buffer* pLightBuffer = nullptr;
    if (lightBufferSize > 0)
    {
        pLightBuffer = mDevice->newBuffer(lightBufferSize, MTL::ResourceStorageModeManaged);
        memcpy(pLightBuffer->contents(), lightDescs.data(), lightBufferSize);
        pLightBuffer->didModifyRange(NS::Range::Make(0, pLightBuffer->length()));
    }
    MTL::Buffer* pVertexBuffer = nullptr;
    if (vertexDataSize > 0)
    {
        pVertexBuffer = mDevice->newBuffer(vertexDataSize, MTL::ResourceStorageModeManaged);
        memcpy(pVertexBuffer->contents(), vertices.data(), vertexDataSize);
        pVertexBuffer->didModifyRange(NS::Range::Make(0, pVertexBuffer->length()));
    };
    MTL::Buffer* pIndexBuffer = nullptr;
    if (indexDataSize > 0)
    {
        pIndexBuffer = mDevice->newBuffer(indexDataSize, MTL::ResourceStorageModeManaged);
        memcpy(pIndexBuffer->contents(), indices.data(), indexDataSize);
        pIndexBuffer->didModifyRange(NS::Range::Make(0, pIndexBuffer->length()));
    }

    mLightBuffer = pLightBuffer;
    mVertexBuffer = pVertexBuffer;
    mIndexBuffer = pIndexBuffer;

    // Allocate prevVertexBuffer as copy of VB (needed for motion BVH keyframes at init time)
    if (vertexDataSize > 0)
    {
        mPrevVertexBuffer = mDevice->newBuffer(vertexDataSize, MTL::ResourceStorageModeManaged);
        memcpy(mPrevVertexBuffer->contents(), vertices.data(), vertexDataSize);
        mPrevVertexBuffer->didModifyRange(NS::Range::Make(0, mPrevVertexBuffer->length()));
    }

    // Allocate InstanceData buffer for per-mesh vertex/index offset lookups (motion blur shader)
    {
        const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
        if (!meshes.empty())
        {
            std::vector<InstanceData> instanceData(meshes.size());
            for (size_t mi = 0; mi < meshes.size(); ++mi)
            {
                instanceData[mi].vbOffset = meshes[mi].mVbOffset;
                instanceData[mi].indexOffset = meshes[mi].mIndex;
            }
            mInstanceDataBuffer = mDevice->newBuffer(
                instanceData.size() * sizeof(InstanceData), MTL::ResourceStorageModeManaged);
            memcpy(mInstanceDataBuffer->contents(), instanceData.data(),
                   instanceData.size() * sizeof(InstanceData));
            mInstanceDataBuffer->didModifyRange(NS::Range::Make(0, mInstanceDataBuffer->length()));
        }
    }

    for (MTL::Buffer*& uniformBuffer : mUniformBuffers)
    {
        uniformBuffer = mDevice->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeManaged);
    }
    for (MTL::Buffer*& uniformBuffer : mUniformTMBuffers)
    {
        uniformBuffer = mDevice->newBuffer(sizeof(UniformsTonemap), MTL::ResourceStorageModeManaged);
    }
}

MTL::AccelerationStructure* MetalRender::createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    // Query for the sizes needed to store and build the acceleration structure.
    const MTL::AccelerationStructureSizes accelSizes = mDevice->accelerationStructureSizes(descriptor);
    // Allocate an acceleration structure large enough for this descriptor. This doesn't actually
    // build the acceleration structure, it just allocates memory.
    MTL::AccelerationStructure* accelerationStructure =
        mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
    // Allocate scratch space Metal uses to build the acceleration structure.
    // Use MTLResourceStorageModePrivate for best performance because the sample
    // doesn't need access to the buffer's contents.
    MTL::Buffer* scratchBuffer = mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
    // Create a command buffer to perform the acceleration structure build.
    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    // Create an acceleration structure command encoder.
    MTL::AccelerationStructureCommandEncoder* commandEncoder = commandBuffer->accelerationStructureCommandEncoder();
    // Allocate a buffer for Metal to write the compacted accelerated structure's size into.
    MTL::Buffer* compactedSizeBuffer = mDevice->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    // Schedule the actual acceleration structure build.
    commandEncoder->buildAccelerationStructure(accelerationStructure, descriptor, scratchBuffer, 0UL);
    // Compute and write the compacted acceleration structure size into the buffer. You
    // need to already have a built accelerated structure because Metal determines the compacted
    // size based on the final size of the acceleration structure. Compacting an acceleration
    // structure can potentially reclaim significant amounts of memory because Metal must
    // create the initial structure using a conservative approach.
    commandEncoder->writeCompactedAccelerationStructureSize(accelerationStructure, compactedSizeBuffer, 0UL);
    // End encoding and commit the command buffer so the GPU can start building the
    // acceleration structure.
    commandEncoder->endEncoding();
    commandBuffer->commit();

    // The sample waits for Metal to finish executing the command buffer so that it can
    // read back the compacted size.

    // Note: Don't wait for Metal to finish executing the command buffer if you aren't compacting
    // the acceleration structure because doing so requires CPU/GPU synchronization. You don't have
    // to compact acceleration structures, but it's helpful when creating large static acceleration
    // structures, such as static scene geometry. Avoid compacting acceleration structures that
    // you rebuild every frame because the synchronization cost may be significant.

    commandBuffer->waitUntilCompleted();

    const uint32_t compactedSize = *(uint32_t*)compactedSizeBuffer->contents();

    // commandBuffer->release();
    // commandEncoder->release();

    // Allocate a smaller acceleration structure based on the returned size.
    MTL::AccelerationStructure* compactedAccelerationStructure = mDevice->newAccelerationStructure(compactedSize);

    // Create another command buffer and encoder.
    commandBuffer = mCommandQueue->commandBuffer();
    commandEncoder = commandBuffer->accelerationStructureCommandEncoder();

    // Encode the command to copy and compact the acceleration structure into the
    // smaller acceleration structure.
    commandEncoder->copyAndCompactAccelerationStructure(accelerationStructure, compactedAccelerationStructure);

    // End encoding and commit the command buffer. You don't need to wait for Metal to finish
    // executing this command buffer as long as you synchronize any ray-intersection work
    // to run after this command buffer completes. The sample relies on Metal's default
    // dependency tracking on resources to automatically synchronize access to the new
    // compacted acceleration structure.
    commandEncoder->endEncoding();
    commandBuffer->commit();

    // commandEncoder->release();
    // commandBuffer->release();
    accelerationStructure->release();
    scratchBuffer->release();
    compactedSizeBuffer->release();

    pPool->release();

    return compactedAccelerationStructure->retain();
}

MTL::AccelerationStructure* MetalRender::createAccelerationStructureNoCompact(
    MTL::AccelerationStructureDescriptor* descriptor)
{
    // Allow refitting on this descriptor
    descriptor->setUsage(MTL::AccelerationStructureUsageRefit);

    const MTL::AccelerationStructureSizes accelSizes = mDevice->accelerationStructureSizes(descriptor);
    MTL::AccelerationStructure* accelerationStructure =
        mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
    MTL::Buffer* scratchBuffer =
        mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);

    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    MTL::AccelerationStructureCommandEncoder* commandEncoder = commandBuffer->accelerationStructureCommandEncoder();
    commandEncoder->buildAccelerationStructure(accelerationStructure, descriptor, scratchBuffer, 0UL);
    commandEncoder->endEncoding();
    commandBuffer->commit();
    // No waitUntilCompleted — Metal queue ordering guarantees subsequent
    // command buffers on the same queue see the built AS.

    scratchBuffer->release();
    return accelerationStructure;
}

MetalRender::Mesh* MetalRender::createMesh(const oka::Mesh& mesh)
{
    auto result = new MetalRender::Mesh();

    const uint32_t triangleCount = mesh.mCount / 3;
    result->mTriangleCount = triangleCount;
    result->mVbOffset = mesh.mVbOffset;
    result->mIndexOffset = mesh.mIndex;
    result->mIsSkeletal = mesh.isSkeletal;

    const std::vector<Scene::Vertex>& vertices = mScene->getVertices();
    const std::vector<uint32_t>& indices = mScene->getIndices();

    std::vector<Triangle> triangleData(triangleCount);
    for (int i = 0; i < triangleCount; ++i)
    {
        Triangle& curr = triangleData[i];
        const uint32_t i0 = indices[mesh.mIndex + i * 3 + 0];
        const uint32_t i1 = indices[mesh.mIndex + i * 3 + 1];
        const uint32_t i2 = indices[mesh.mIndex + i * 3 + 2];
        // Positions
        using simd::float3;

        curr.positions[0] = { vertices[mesh.mVbOffset + i0].pos.x, vertices[mesh.mVbOffset + i0].pos.y,
                              vertices[mesh.mVbOffset + i0].pos.z };
        curr.positions[1] = { vertices[mesh.mVbOffset + i1].pos.x, vertices[mesh.mVbOffset + i1].pos.y,
                              vertices[mesh.mVbOffset + i1].pos.z };
        curr.positions[2] = { vertices[mesh.mVbOffset + i2].pos.x, vertices[mesh.mVbOffset + i2].pos.y,
                              vertices[mesh.mVbOffset + i2].pos.z };
        // Normals
        curr.normals[0] = vertices[mesh.mVbOffset + i0].normal;
        curr.normals[1] = vertices[mesh.mVbOffset + i1].normal;
        curr.normals[2] = vertices[mesh.mVbOffset + i2].normal;
        // Tangents
        curr.tangent[0] = vertices[mesh.mVbOffset + i0].tangent;
        curr.tangent[1] = vertices[mesh.mVbOffset + i1].tangent;
        curr.tangent[2] = vertices[mesh.mVbOffset + i2].tangent;
        // UVs
        curr.uv[0] = vertices[mesh.mVbOffset + i0].uv;
        curr.uv[1] = vertices[mesh.mVbOffset + i1].uv;
        curr.uv[2] = vertices[mesh.mVbOffset + i2].uv;
    }

    MTL::Buffer* perPrimitiveBuffer =
        mDevice->newBuffer(triangleData.size() * sizeof(Triangle), MTL::ResourceStorageModeManaged);

    memcpy(perPrimitiveBuffer->contents(), triangleData.data(), sizeof(Triangle) * triangleData.size());

    perPrimitiveBuffer->didModifyRange(NS::Range(0, perPrimitiveBuffer->length()));

    if (mesh.isSkeletal)
    {
        // Motion BVH with 2 keyframes (prevVB @ t=0, VB @ t=1)
        MTL::PrimitiveAccelerationStructureDescriptor* primDescriptor =
            createMotionBLASDescriptor(mesh, perPrimitiveBuffer, triangleCount);
        result->mGas = createAccelerationStructureNoCompact(primDescriptor);
        primDescriptor->release();
    }
    else
    {
        // Static BVH for non-skeletal meshes
        auto* geomDescriptor =
            MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();

        geomDescriptor->setVertexBuffer(mVertexBuffer);
        geomDescriptor->setVertexBufferOffset(mesh.mVbOffset * sizeof(Scene::Vertex));
        geomDescriptor->setVertexStride(sizeof(Scene::Vertex));
        geomDescriptor->setIndexBuffer(mIndexBuffer);
        geomDescriptor->setIndexBufferOffset(mesh.mIndex * sizeof(uint32_t));
        geomDescriptor->setIndexType(MTL::IndexTypeUInt32);
        geomDescriptor->setTriangleCount(triangleCount);
        geomDescriptor->setPrimitiveDataBuffer(perPrimitiveBuffer);
        geomDescriptor->setPrimitiveDataBufferOffset(0);
        geomDescriptor->setPrimitiveDataElementSize(sizeof(Triangle));
        geomDescriptor->setPrimitiveDataStride(sizeof(Triangle));

        const NS::Array* geomDescriptors =
            NS::Array::array((const NS::Object* const*)&geomDescriptor, 1UL);

        MTL::PrimitiveAccelerationStructureDescriptor* primDescriptor =
            MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
        primDescriptor->setGeometryDescriptors(geomDescriptors);

        result->mGas = createAccelerationStructure(primDescriptor);

        primDescriptor->release();
        geomDescriptor->release();
    }

    // Keep per-primitive buffer alive for skeletal meshes (needed for triangle updates)
    result->mPerPrimitiveBuffer = perPrimitiveBuffer;
    if (!mesh.isSkeletal)
    {
        perPrimitiveBuffer->release();
        result->mPerPrimitiveBuffer = nullptr;
    }
    return result;
}

void MetalRender::createAccelerationStructures()
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
    const std::vector<oka::Curve>& curves = mScene->getCurves();
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    if (meshes.empty() && curves.empty())
    {
        return;
    }

    for (const oka::Mesh& currMesh : meshes)
    {
        MetalRender::Mesh* metalMesh = createMesh(currMesh);
        mMetalMeshes.push_back(metalMesh);
        mPrimitiveAccelerationStructures.push_back(metalMesh->mGas);
    }

    mInstanceBuffer = mDevice->newBuffer(
        sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor) * instances.size(), MTL::ResourceStorageModeManaged);
    auto instanceDescriptors = (MTL::AccelerationStructureUserIDInstanceDescriptor*)mInstanceBuffer->contents();
    for (int i = 0; i < instances.size(); ++i)
    {
        const Instance& curr = instances[i];
        instanceDescriptors[i].accelerationStructureIndex = curr.mMeshId;
        instanceDescriptors[i].options = MTL::AccelerationStructureInstanceOptionOpaque;
        instanceDescriptors[i].intersectionFunctionTableOffset = 0;
        instanceDescriptors[i].userID = curr.type == Instance::Type::eLight ? curr.mLightId : curr.mMaterialId;
        instanceDescriptors[i].mask = curr.type == Instance::Type::eLight ? GEOMETRY_MASK_LIGHT : GEOMETRY_MASK_TRIANGLE;

        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                instanceDescriptors[i].transformationMatrix.columns[column][row] = curr.transform[column][row];
            }
        }
    }
    mInstanceBuffer->didModifyRange(NS::Range::Make(0, mInstanceBuffer->length()));

    const NS::Array* instancedAccelerationStructures = NS::Array::array(
        (const NS::Object* const*)mPrimitiveAccelerationStructures.data(), mPrimitiveAccelerationStructures.size());
    MTL::InstanceAccelerationStructureDescriptor* accelDescriptor =
        MTL::InstanceAccelerationStructureDescriptor::descriptor();
    accelDescriptor->setInstancedAccelerationStructures(instancedAccelerationStructures);
    accelDescriptor->setInstanceCount(instances.size());
    accelDescriptor->setInstanceDescriptorBuffer(mInstanceBuffer);
    accelDescriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);

    mInstanceAccelerationStructure = createAccelerationStructure(accelDescriptor);
    pPool->release();
}

void MetalRender::buildSkinningPipeline()
{
    NS::Error* pError = nullptr;
    MTL::Library* pLibrary =
        mDevice->newLibrary(NS::String::string("./metal/shaders/skinning.metallib", NS::UTF8StringEncoding), &pError);
    if (!pLibrary)
    {
        STRELKA_FATAL("Failed to load skinning metallib: {}", pError->localizedDescription()->utf8String());
        assert(false);
    }

    MTL::Function* pSkinningFn =
        pLibrary->newFunction(NS::String::string("skinningKernel", NS::UTF8StringEncoding));
    mSkinningPSO = mDevice->newComputePipelineState(pSkinningFn, &pError);
    if (!mSkinningPSO)
    {
        STRELKA_FATAL("Failed to create skinning PSO: {}", pError->localizedDescription()->utf8String());
        assert(false);
    }
    pSkinningFn->release();

    MTL::Function* pTriUpdateFn =
        pLibrary->newFunction(NS::String::string("updateTriangleBufferKernel", NS::UTF8StringEncoding));
    mTriangleUpdatePSO = mDevice->newComputePipelineState(pTriUpdateFn, &pError);
    if (!mTriangleUpdatePSO)
    {
        STRELKA_FATAL("Failed to create triangle update PSO: {}", pError->localizedDescription()->utf8String());
        assert(false);
    }
    pTriUpdateFn->release();

    pLibrary->release();
}

void MetalRender::createSkinDataBuffer()
{
    const std::vector<Scene::vertexSkinData>& skinData = mScene->getVerticesSkinData();
    if (skinData.empty())
        return;

    const size_t dataSize = skinData.size() * sizeof(Scene::vertexSkinData);
    mSkinDataBuffer = mDevice->newBuffer(dataSize, MTL::ResourceStorageModeManaged);
    memcpy(mSkinDataBuffer->contents(), skinData.data(), dataSize);
    mSkinDataBuffer->didModifyRange(NS::Range::Make(0, mSkinDataBuffer->length()));
}

void MetalRender::allocJointMatrices()
{
    size_t jointMatSize = 0;
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            auto jointCount = mScene->mSkines[node.skin].joints.size();
            std::vector<glm::mat4> currJointMats;
            mScene->computeJointMatrices(&currJointMats, jointCount, node.skin);

            jointMatSize += currJointMats.size();
            mJointMatOffsets.push_back(currJointMats.size());
        }
    }

    if (jointMatSize > 0)
    {
        mJointMatricesBuffer =
            mDevice->newBuffer(jointMatSize * sizeof(simd::float4x4), MTL::ResourceStorageModeManaged);
    }
}

void MetalRender::applySkinning()
{
    if (!mSkinningPSO || !mSkinDataBuffer || !mJointMatricesBuffer)
        return;

    // Compute joint matrices on CPU
    std::vector<glm::mat4> jointMat;
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            auto jointCount = mScene->mSkines[node.skin].joints.size();
            std::vector<glm::mat4> currJointMats;
            mScene->computeJointMatrices(&currJointMats, jointCount, node.skin);
            jointMat.insert(jointMat.end(), currJointMats.begin(), currJointMats.end());
        }
    }

    // Convert glm::mat4 → simd::float4x4 (both column-major)
    std::vector<simd::float4x4> simdMatrices(jointMat.size());
    for (size_t i = 0; i < jointMat.size(); ++i)
    {
        const glm::mat4& m = jointMat[i];
        for (int col = 0; col < 4; col++)
        {
            for (int row = 0; row < 4; row++)
            {
                simdMatrices[i].columns[col][row] = m[col][row];
            }
        }
    }

    // Upload joint matrices
    memcpy(mJointMatricesBuffer->contents(), simdMatrices.data(), simdMatrices.size() * sizeof(simd::float4x4));
    mJointMatricesBuffer->didModifyRange(NS::Range::Make(0, simdMatrices.size() * sizeof(simd::float4x4)));

    // Dispatch skinning + triangle update kernels
    MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* pEncoder = pCmd->computeCommandEncoder();

    int skinIndex = 0;
    int jointMatOffset = 0;
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            if (skinIndex > 0)
            {
                jointMatOffset += mJointMatOffsets[skinIndex - 1];
            }
            skinIndex++;

            for (const auto instId : node.instanceIds)
            {
                auto& mesh = mScene->mMeshes[mScene->mInstances[instId].mMeshId];
                uint32_t meshId = mScene->mInstances[instId].mMeshId;

                // Dispatch skinning kernel
                SkinningParams skinParams = {};
                skinParams.vbOffset = mesh.mVbOffset;
                skinParams.sbOffset = mesh.mSbOffset;
                skinParams.jointMatOffset = jointMatOffset;
                skinParams.vertexCount = mesh.mVertexCount;

                pEncoder->setComputePipelineState(mSkinningPSO);
                pEncoder->setBuffer(mVertexBuffer, 0, 0);
                pEncoder->setBuffer(mSkinDataBuffer, 0, 1);
                pEncoder->setBuffer(mJointMatricesBuffer, 0, 2);
                pEncoder->setBytes(&skinParams, sizeof(SkinningParams), 3);

                const uint32_t threadsPerGroup = 256;
                const MTL::Size gridSize = MTL::Size(mesh.mVertexCount, 1, 1);
                const MTL::Size groupSize = MTL::Size(threadsPerGroup, 1, 1);
                pEncoder->dispatchThreads(gridSize, groupSize);

                // Dispatch triangle update kernel
                MetalRender::Mesh* metalMesh = mMetalMeshes[meshId];
                if (metalMesh->mPerPrimitiveBuffer)
                {
                    TriangleUpdateParams triParams = {};
                    triParams.triangleCount = metalMesh->mTriangleCount;
                    triParams.indexOffset = mesh.mIndex;
                    triParams.vbOffset = mesh.mVbOffset;

                    pEncoder->setComputePipelineState(mTriangleUpdatePSO);
                    pEncoder->setBuffer(metalMesh->mPerPrimitiveBuffer, 0, 0);
                    pEncoder->setBuffer(mVertexBuffer, 0, 1);
                    pEncoder->setBuffer(mIndexBuffer, 0, 2);
                    pEncoder->setBytes(&triParams, sizeof(TriangleUpdateParams), 3);

                    const MTL::Size triGridSize = MTL::Size(metalMesh->mTriangleCount, 1, 1);
                    pEncoder->dispatchThreads(triGridSize, groupSize);
                }
            }
        }
    }

    pEncoder->endEncoding();
    pCmd->commit();
    // No waitUntilCompleted — queue ordering guarantees subsequent AS operations
    // on the same queue see skinning results.
}

void MetalRender::copyVertexBufferToPrev()
{
    const size_t vertexDataSize = mVertexBuffer->length();
    MTL::CommandBuffer* blitCmd = mCommandQueue->commandBuffer();
    MTL::BlitCommandEncoder* blit = blitCmd->blitCommandEncoder();
    blit->copyFromBuffer(mVertexBuffer, 0, mPrevVertexBuffer, 0, vertexDataSize);
    blit->endEncoding();
    blitCmd->commit();
}

MTL::PrimitiveAccelerationStructureDescriptor* MetalRender::createMotionBLASDescriptor(
    const oka::Mesh& sceneMesh, MTL::Buffer* perPrimitiveBuffer, uint32_t triangleCount)
{
    auto* geomDescriptor =
        MTL::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()->init();

    MTL::MotionKeyframeData* kf0 = MTL::MotionKeyframeData::alloc()->init();
    kf0->setBuffer(mPrevVertexBuffer);
    kf0->setOffset(sceneMesh.mVbOffset * sizeof(Scene::Vertex));

    MTL::MotionKeyframeData* kf1 = MTL::MotionKeyframeData::alloc()->init();
    kf1->setBuffer(mVertexBuffer);
    kf1->setOffset(sceneMesh.mVbOffset * sizeof(Scene::Vertex));

    const NS::Object* keyframes[] = { kf0, kf1 };
    NS::Array* vertexBuffers = NS::Array::array(keyframes, 2UL);
    geomDescriptor->setVertexBuffers(vertexBuffers);
    geomDescriptor->setVertexStride(sizeof(Scene::Vertex));

    geomDescriptor->setIndexBuffer(mIndexBuffer);
    geomDescriptor->setIndexBufferOffset(sceneMesh.mIndex * sizeof(uint32_t));
    geomDescriptor->setIndexType(MTL::IndexTypeUInt32);
    geomDescriptor->setTriangleCount(triangleCount);
    geomDescriptor->setPrimitiveDataBuffer(perPrimitiveBuffer);
    geomDescriptor->setPrimitiveDataBufferOffset(0);
    geomDescriptor->setPrimitiveDataElementSize(sizeof(Triangle));
    geomDescriptor->setPrimitiveDataStride(sizeof(Triangle));

    const NS::Array* geomDescriptors = NS::Array::array((const NS::Object* const*)&geomDescriptor, 1UL);

    MTL::PrimitiveAccelerationStructureDescriptor* primDescriptor =
        MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    primDescriptor->setGeometryDescriptors(geomDescriptors);
    primDescriptor->setMotionKeyframeCount(2);
    primDescriptor->setMotionStartTime(0.0f);
    primDescriptor->setMotionEndTime(1.0f);
    primDescriptor->setMotionStartBorderMode(MTL::MotionBorderModeClamp);
    primDescriptor->setMotionEndBorderMode(MTL::MotionBorderModeClamp);

    kf0->release();
    kf1->release();
    geomDescriptor->release();

    return primDescriptor;
}

void MetalRender::refitBLAS(int meshIndex)
{
    MetalRender::Mesh* metalMesh = mMetalMeshes[meshIndex];
    if (!metalMesh->mIsSkeletal)
        return;

    const oka::Mesh& sceneMesh = mScene->getMeshes()[meshIndex];
    MTL::PrimitiveAccelerationStructureDescriptor* primDescriptor =
        createMotionBLASDescriptor(sceneMesh, metalMesh->mPerPrimitiveBuffer, metalMesh->mTriangleCount);
    primDescriptor->setUsage(MTL::AccelerationStructureUsageRefit);

    const MTL::AccelerationStructureSizes accelSizes = mDevice->accelerationStructureSizes(primDescriptor);
    MTL::Buffer* scratchBuffer =
        mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);

    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    MTL::AccelerationStructureCommandEncoder* commandEncoder = commandBuffer->accelerationStructureCommandEncoder();

    commandEncoder->refitAccelerationStructure(
        metalMesh->mGas, primDescriptor, metalMesh->mGas, scratchBuffer, 0UL);

    commandEncoder->endEncoding();
    commandBuffer->commit();

    scratchBuffer->release();
    primDescriptor->release();
}

void MetalRender::rebuildBLAS(int meshIndex)
{
    MetalRender::Mesh* metalMesh = mMetalMeshes[meshIndex];
    if (!metalMesh->mIsSkeletal)
        return;

    const oka::Mesh& sceneMesh = mScene->getMeshes()[meshIndex];
    MTL::PrimitiveAccelerationStructureDescriptor* primDescriptor =
        createMotionBLASDescriptor(sceneMesh, metalMesh->mPerPrimitiveBuffer, metalMesh->mTriangleCount);

    metalMesh->mGas->release();
    metalMesh->mGas = createAccelerationStructureNoCompact(primDescriptor);
    mPrimitiveAccelerationStructures[meshIndex] = metalMesh->mGas;

    primDescriptor->release();
}

void MetalRender::updateInstanceTransforms()
{
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    auto instanceDescriptors = (MTL::AccelerationStructureUserIDInstanceDescriptor*)mInstanceBuffer->contents();

    for (int i = 0; i < (int)instances.size(); ++i)
    {
        const Instance& curr = instances[i];
        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                instanceDescriptors[i].transformationMatrix.columns[column][row] = curr.transform[column][row];
            }
        }
    }
    mInstanceBuffer->didModifyRange(NS::Range::Make(0, mInstanceBuffer->length()));
}

void MetalRender::rebuildTLAS()
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    // Update instance transforms
    updateInstanceTransforms();

    // Release old TLAS
    if (mInstanceAccelerationStructure)
    {
        mInstanceAccelerationStructure->release();
        mInstanceAccelerationStructure = nullptr;
    }

    const std::vector<oka::Instance>& instances = mScene->getInstances();

    const NS::Array* instancedAccelerationStructures = NS::Array::array(
        (const NS::Object* const*)mPrimitiveAccelerationStructures.data(), mPrimitiveAccelerationStructures.size());
    MTL::InstanceAccelerationStructureDescriptor* accelDescriptor =
        MTL::InstanceAccelerationStructureDescriptor::descriptor();
    accelDescriptor->setInstancedAccelerationStructures(instancedAccelerationStructures);
    accelDescriptor->setInstanceCount(instances.size());
    accelDescriptor->setInstanceDescriptorBuffer(mInstanceBuffer);
    accelDescriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);

    // Rebuild without compaction for animation (avoid sync stall)
    mInstanceAccelerationStructure = createAccelerationStructureNoCompact(accelDescriptor);

    pPool->release();
}

void MetalRender::loadEnvMap(const std::string& texturePath)
{
    // Release previous env map resources to avoid leaks on reload
    if (mEnvMapTexture) { mEnvMapTexture->release(); mEnvMapTexture = nullptr; }
    if (mEnvCdfXBuffer) { mEnvCdfXBuffer->release(); mEnvCdfXBuffer = nullptr; }
    if (mEnvCdfYBuffer) { mEnvCdfYBuffer->release(); mEnvCdfYBuffer = nullptr; }
    mEnvMapLoaded = false;

    int width = 0, height = 0;
    float* pixelData = nullptr;
    bool isExr = false;

    const std::string ext = fs::path(texturePath).extension().string();
    if (ext == ".exr" || ext == ".EXR")
    {
        const char* err = nullptr;
        int ret = LoadEXR(&pixelData, &width, &height, texturePath.c_str(), &err);
        if (ret != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to load EXR env map: {} ({})", texturePath, err ? err : "unknown");
            if (err) FreeEXRErrorMessage(err);
            return;
        }
        isExr = true;
    }
    else
    {
        int channels = 0;
        pixelData = stbi_loadf(texturePath.c_str(), &width, &height, &channels, 4);
        if (!pixelData)
        {
            STRELKA_ERROR("Failed to load env map: {}", texturePath);
            return;
        }
    }

    STRELKA_INFO("Loaded env map: {} ({}x{})", texturePath, width, height);

    // Create Metal texture (RGBA32Float)
    MTL::TextureDescriptor* pTextureDesc = MTL::TextureDescriptor::alloc()->init();
    pTextureDesc->setWidth(width);
    pTextureDesc->setHeight(height);
    pTextureDesc->setPixelFormat(MTL::PixelFormatRGBA32Float);
    pTextureDesc->setTextureType(MTL::TextureType2D);
    pTextureDesc->setStorageMode(MTL::StorageModeManaged);
    pTextureDesc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead);

    mEnvMapTexture = mDevice->newTexture(pTextureDesc);
    pTextureDesc->release();

    const MTL::Region region = MTL::Region::Make3D(0, 0, 0, width, height, 1);
    mEnvMapTexture->replaceRegion(region, 0, pixelData, width * sizeof(float) * 4);

    // Build 2D CDF on CPU (sequential, runs once at load)
    const size_t cdfXSize = (size_t)width * height;
    const size_t cdfYSize = (size_t)height;
    std::vector<float> cdfX(cdfXSize);
    std::vector<float> cdfY(cdfYSize);
    std::vector<float> rowSums(height);

    // Phase 1: Build conditional CDF per row
    for (int y = 0; y < height; ++y)
    {
        const float v = ((float)y + 0.5f) / (float)height;
        const float sinTheta = std::sin(v * M_PI);

        float sum = 0.0f;
        for (int x = 0; x < width; ++x)
        {
            const int pixelIdx = (y * width + x) * 4;
            const float r = pixelData[pixelIdx + 0];
            const float g = pixelData[pixelIdx + 1];
            const float b = pixelData[pixelIdx + 2];
            const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            sum += lum * sinTheta;
            cdfX[y * width + x] = sum;
        }
        rowSums[y] = sum;

        // Normalize to [0, 1]
        if (sum > 0.0f)
        {
            const float invSum = 1.0f / sum;
            for (int x = 0; x < width; ++x)
                cdfX[y * width + x] *= invSum;
        }
        else
        {
            for (int x = 0; x < width; ++x)
                cdfX[y * width + x] = (float)(x + 1) / (float)width;
        }
    }

    // Phase 2: Build marginal CDF from row sums
    float totalPower = 0.0f;
    for (int y = 0; y < height; ++y)
    {
        totalPower += rowSums[y];
        cdfY[y] = totalPower;
    }
    if (totalPower > 0.0f)
    {
        const float invTotal = 1.0f / totalPower;
        for (int y = 0; y < height; ++y)
            cdfY[y] *= invTotal;
    }
    else
    {
        for (int y = 0; y < height; ++y)
            cdfY[y] = (float)(y + 1) / (float)height;
    }

    // Upload CDF buffers to GPU
    mEnvCdfXBuffer = mDevice->newBuffer(cdfX.data(), cdfXSize * sizeof(float), MTL::ResourceStorageModeManaged);
    mEnvCdfYBuffer = mDevice->newBuffer(cdfY.data(), cdfYSize * sizeof(float), MTL::ResourceStorageModeManaged);

    // Free host pixel data
    if (isExr)
        free(pixelData);
    else
        stbi_image_free(pixelData);

    // Auto-calibrate env map intensity
    const float avgWeightedLum = totalPower / (float)(width * height);
    const float kCalibrationTarget = 1000.0f;
    mEnvMapAutoScale = (avgWeightedLum > 1e-6f) ? kCalibrationTarget / avgWeightedLum : 1.0f;
    mEnvMapLoaded = true;

    STRELKA_INFO("Env map CDF built, total power: {}, avgLum: {:.4f}, autoScale: {:.1f}",
                 totalPower, avgWeightedLum, mEnvMapAutoScale);
}
