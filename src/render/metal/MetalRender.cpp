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

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <glm/ext/matrix_relational.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <log.h>

#include <simd/simd.h>

#include "shaders/ShaderTypes.h"

using namespace oka;
namespace fs = std::filesystem;

MetalRender::MetalRender(/* args */) = default;

MetalRender::~MetalRender() = default;

void MetalRender::init()
{
    mDevice = MTL::CreateSystemDefaultDevice();
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
    for (const Scene::MaterialDescription& currMatDesc : matDescs)
    {
        Material material = {};
        material.diffuse = { 1.0f, 1.0f, 1.0f };
        for (const auto& param : currMatDesc.params)
        {
            if (param.name == "diffuse_color" || param.name == "diffuseColor" || param.name == "diffuse_color_constant")
            {
                memcpy(&material.diffuse, param.value.data(), sizeof(float) * 3);
            }
            if (param.type == MaterialManager::Param::Type::eTexture)
            {
                std::string texPath(param.value.size(), 0);
                memcpy(texPath.data(), param.value.data(), param.value.size());
                const fs::path fullTextureFilePath = resourcePath / texPath;
                if (param.name == "diffuse_texture")
                {
                    MTL::Texture* diffuseTex = loadTextureFromFile(fullTextureFilePath.string());
                    mMaterialTextures.push_back(diffuseTex);
                    material.diffuseTexture = diffuseTex->gpuResourceID();
                }
                if (param.name == "normalmap_texture")
                {
                    MTL::Texture* normalTex = loadTextureFromFile(fullTextureFilePath.string());
                    mMaterialTextures.push_back(normalTex);
                    material.normalTexture = normalTex->gpuResourceID();
                }
            }
        }
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

    if (getSharedContext().mFrameNumber == 0)
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
        getSharedContext().mSubframeIndex = 0;
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
        std::vector<float> targetTimes(animations.size());
        std::vector<bool> changed(animations.size(), false);
        for (int i = 0; i < (int)animations.size(); ++i)
        {
            const std::string scrollNameStr = "render/animation/anim" + std::to_string(i) + "/time";
            targetTimes[i] = animSettings.getAs<float>(scrollNameStr.c_str());
            const float delta = std::abs(animations[i].current - targetTimes[i]);
            if (delta > EPSILON)
            {
                changed[i] = true;
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
                    if (!changed[i]) continue;
                    float tOpen = targetTimes[i] + shutterOffset;
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
                    if (!changed[i]) continue;
                    float tClose = targetTimes[i] + shutterOffset + shutterDuration;
                    tClose = std::clamp(tClose, animations[i].start, animations[i].end);
                    animations[i].current = tClose;
                    pass2Skeletal |= mScene->applyAnimation(i);
                }

                if (pass2Skeletal)
                {
                    applySkinning();
                    // Force full rebuild on large time jumps (scrubbing) — refit produces
                    // degenerate BVH when geometry changes dramatically between frames
                    const bool fullRebuild = (mBlasUpdateCount >= 10) ||
                                             (maxTimeDelta > shutterDuration * 2.0f);
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
                }
                updateInstanceTransforms();
                rebuildTLAS();

                // Restore target times so next-frame EPSILON check is stable
                for (int i = 0; i < (int)animations.size(); ++i)
                    animations[i].current = targetTimes[i];
            }
            else
            {
                // Motion blur disabled or no shutter: single-pass at target time
                bool accelStructureDirty = false;
                for (int i = 0; i < (int)animations.size(); ++i)
                {
                    if (!changed[i]) continue;
                    animations[i].current = targetTimes[i];
                    accelStructureDirty |= mScene->applyAnimation(i);
                }

                if (accelStructureDirty)
                {
                    applySkinning();
                    // Sync prevVB with current VB — motion BVH needs both keyframes
                    // consistent when motion blur is off (otherwise keyframe 0 is stale)
                    copyVertexBufferToPrev();
                    // Force full rebuild on large time jumps (scrubbing)
                    const bool fullRebuild = (mBlasUpdateCount >= 10) ||
                                             (maxTimeDelta > 0.1f);
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
                    rebuildTLAS();
                }
                else
                {
                    updateInstanceTransforms();
                    rebuildTLAS();
                }
            }
            getSharedContext().mSubframeIndex = 0;
        }
    }

    SettingsManager& settings = *getSettings();

    const uint32_t selectedCamera = settings.getAs<uint32_t>("render/selectedCamera");
    oka::Camera& camera = mScene->getCamera(selectedCamera);
    camera.updateAspectRatio(width / (float)height);
    camera.updateViewMatrix();

    View currView = {};
    currView.mCamMatrices = camera.matrices;

    // Initialize prev motion blur view on first frame if animation block didn't set it.
    // On subsequent frames, mPrevMotionBlurView retains the t_open camera from the last
    // animation frame so the converged image shows correct camera motion blur.
    if (!motionBlurCameraSet && getSharedContext().mFrameNumber == 0)
    {
        mPrevMotionBlurView.mCamMatrices = camera.matrices;
    }

    if (glm::any(glm::notEqual(currView.mCamMatrices.perspective, mPrevView.mCamMatrices.perspective)) ||
        glm::any(glm::notEqual(currView.mCamMatrices.view, mPrevView.mCamMatrices.view)))
    {
        // need reset
        getSharedContext().mSubframeIndex = 0;
    }

    MTL::Buffer* pUniformBuffer = mUniformBuffers[mFrameIndex];
    MTL::Buffer* pUniformTMBuffer = mUniformTMBuffers[mFrameIndex];
    auto pUniformData = reinterpret_cast<Uniforms*>(pUniformBuffer->contents());
    auto pUniformTonemap = reinterpret_cast<UniformsTonemap*>(pUniformTMBuffer->contents());
    pUniformData->frameIndex = mFrameIndex;
    pUniformData->subframeIndex = getSharedContext().mSubframeIndex;
    pUniformData->height = height;
    pUniformData->width = width;
    pUniformData->numLights = mScene->getLightsDesc().size();
    pUniformData->samples_per_launch = settings.getAs<uint32_t>("render/pt/spp");
    pUniformData->enableAccumulation = (uint32_t)settings.getAs<bool>("render/pt/enableAcc");
    pUniformData->missColor = float3(0.0f);
    pUniformData->maxDepth = settings.getAs<uint32_t>("render/pt/depth");
    pUniformData->debug = settings.getAs<uint32_t>("render/pt/debug");
    pUniformData->enableMotionBlur = mEnableMotionBlur ? 1 : 0;
    pUniformData->isMotionBlurVisible = (uint32_t)settings.getAs<bool>("render/isMotionBlurVisible");
    pUniformData->enableCameraMotionBlur = (uint32_t)settings.getAs<bool>("render/enableCameraMotionBlur");

    pUniformTonemap->width = width;
    pUniformTonemap->height = height;
    pUniformTonemap->tonemapperType = settings.getAs<uint32_t>("render/pt/tonemapperType");
    pUniformTonemap->gamma = settings.getAs<float>("render/post/gamma");
    pUniformTonemap->maxEDR = settings.getAs<float>("render/post/tonemapper/maxEDR");

    bool settingsChanged = false;

    static uint32_t rectLightSamplingMethodPrev = 0;
    pUniformData->rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    settingsChanged = (rectLightSamplingMethodPrev != pUniformData->rectLightSamplingMethod);
    rectLightSamplingMethodPrev = pUniformData->rectLightSamplingMethod;

    static uint32_t samplerTypePrev = 0;
    pUniformData->samplerType = settings.getAs<uint32_t>("render/pt/samplerType");
    settingsChanged |= (samplerTypePrev != pUniformData->samplerType);
    samplerTypePrev = pUniformData->samplerType;

    static bool enableAccumulationPrev = false;
    const bool enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    settingsChanged |= (enableAccumulationPrev != enableAccumulation);
    enableAccumulationPrev = enableAccumulation;

    static uint32_t sspTotalPrev = 0;
    const auto sspTotal = settings.getAs<uint32_t>("render/pt/sppTotal");
    settingsChanged |= (sspTotalPrev > sspTotal); // reset only if new spp less than already accumulated
    sspTotalPrev = sspTotal;

    static uint32_t sppPrev = 0;
    pUniformData->samples_per_launch = settings.getAs<uint32_t>("render/pt/spp");
    settingsChanged |= (sppPrev != pUniformData->samples_per_launch);
    sppPrev = pUniformData->samples_per_launch;

    static bool enableMotionBlurPrev = false;
    const bool enableMotionBlurCurr = mEnableMotionBlur;
    settingsChanged |= (enableMotionBlurPrev != enableMotionBlurCurr);
    enableMotionBlurPrev = enableMotionBlurCurr;

    static bool isMotionBlurVisiblePrev = true;
    const bool isMotionBlurVisibleCurr = settings.getAs<bool>("render/isMotionBlurVisible");
    settingsChanged |= (isMotionBlurVisiblePrev != isMotionBlurVisibleCurr);
    isMotionBlurVisiblePrev = isMotionBlurVisibleCurr;

    static bool enableCameraMotionBlurPrev = true;
    const bool enableCameraMotionBlurCurr = settings.getAs<bool>("render/enableCameraMotionBlur");
    settingsChanged |= (enableCameraMotionBlurPrev != enableCameraMotionBlurCurr);
    enableCameraMotionBlurPrev = enableCameraMotionBlurCurr;

    if (settingsChanged)
    {
        getSharedContext().mSubframeIndex = 0;
    }

    glm::float4x4 invView = glm::inverse(camera.matrices.view);
    for (int column = 0; column < 4; column++)
    {
        for (int row = 0; row < 4; row++)
        {
            pUniformData->viewToWorld.columns[column][row] = invView[column][row];
        }
    }
    for (int column = 0; column < 4; column++)
    {
        for (int row = 0; row < 4; row++)
        {
            pUniformData->clipToView.columns[column][row] = camera.matrices.invPerspective[column][row];
        }
    }

    // Previous camera matrices for camera motion blur
    {
        glm::float4x4 prevInvView = glm::inverse(mPrevMotionBlurView.mCamMatrices.view);
        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 4; row++)
            {
                pUniformData->prevViewToWorld.columns[column][row] = prevInvView[column][row];
                pUniformData->prevClipToView.columns[column][row] = mPrevMotionBlurView.mCamMatrices.invPerspective[column][row];
            }
        }
    }

    pUniformData->subframeIndex = getSharedContext().mSubframeIndex;

    // Photometric Units from iray documentation
    // Controls the sensitivity of the “camera film” and is expressed as an index; the ISO number of the film, also
    // known as “film speed.” The higher this value, the greater the exposure. If this is set to a non-zero value,
    // “Photographic” mode is enabled. If this is set to 0, “Arbitrary” mode is enabled, and all color scaling is then
    // strictly defined by the value of cm^2 Factor.
    auto filmIso = settings.getAs<float>("render/post/tonemapper/filmIso");
    // The candela per meter square factor
    auto cm2_factor = settings.getAs<float>("render/post/tonemapper/cm2_factor");
    // The fractional aperture number; e.g., 11 means aperture “f/11.” It adjusts the size of the opening of the “camera
    // iris” and is expressed as a ratio. The higher this value, the lower the exposure.
    auto fStop = settings.getAs<float>("render/post/tonemapper/fStop");
    // Controls the duration, in fractions of a second, that the “shutter” is open; e.g., the value 100 means that the
    // “shutter” is open for 1/100th of a second. The higher this value, the greater the exposure
    auto shutterSpeed = settings.getAs<float>("render/post/tonemapper/shutterSpeed");
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
    const int32_t leftSpp = sspTotal - getSharedContext().mSubframeIndex;
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

        pCmd->commit();

        if (enableAccumulation)
        {
            getSharedContext().mSubframeIndex += samplesThisLaunch;
        }
        else
        {
            getSharedContext().mSubframeIndex = 0;
        }
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

        pCmd->commit();
    }
    pPool->release();

    mPrevView = currView;
    getSharedContext().mFrameNumber++;
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
