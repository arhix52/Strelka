#pragma once

#define GLM_FORCE_SILENT_WARNINGS
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

// #define STB_IMAGE_STATIC
// #define STB_IMAGE_IMPLEMENTATION
// #include "accumulation.h"
// #include "bvh.h"
// #include "common.h"
// #include "debugview.h"
// #include "gbuffer.h"
// #include "pathtracer.h"
// #include "reduction.h"
// #include "tonemap.h"
// #include "upscalepass.h"
// #include "vkrender.h"

// #include <materialmanager/materialmanager.h>
// #include <scene/scene.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace oka
{

class PtRender
{
public:
    PtRender()
    {
    }

    void init();
    void cleanup();

    void reloadPt();

    void drawFrame(Image* result);

    void setSharedContext(SharedContext* ctx)
    {
        mSharedCtx = ctx;
    }

    SharedContext& getSharedContext()
    {
        return *mSharedCtx;
    }

    void setScene(Scene* scene)
    {
        mScene = scene;
    }

    Scene* getScene()
    {
        return mScene;
    }

    void setActiveCameraIndex(const uint32_t index)
    {
        mCurrentSceneRenderData->cameraIndex = index;
    }

    uint32_t getActiveCameraIndex()
    {
        return mCurrentSceneRenderData->cameraIndex;
    }

private:
    SharedContext* mSharedCtx;

    // MaterialManager* mMaterialManager = nullptr;
    // BvhBuilder mBvhBuilder;
    // PathTracer* mPathTracer;
    // Accumulation* mAccumulationPathTracer;
    // Tonemap* mTonemap;
    // UpscalePass* mUpscalePass;
    // ReductionPass* mReductionPass;
    // DebugView* mDebugView;

    struct RenderSettings
    {
        bool enableUpscale = false;
        bool enableAccumulation = false;
        uint32_t stratifiedSamplingType = 0;
        uint32_t debugView = 0;
    };

    RenderSettings mSettings;

    struct ViewData
    {
        // could be scaled
        uint32_t renderWidth; // = swapChainExtent.width * mRenderConfig.upscaleFactor;
        uint32_t renderHeight; // = swapChainExtent.height * mRenderConfig.upscaleFactor;
        // swapchain size
        uint32_t finalWidth; // = swapChainExtent.width;
        uint32_t finalHeight; // = swapChainExtent.height;
        uint32_t spp;
        GBuffer* gbuffer;
        Image* prevDepth;
        Image* textureTonemapImage;
        Image* textureUpscaleImage;
        Image* textureDebugViewImage;
        Image* mPathTracerImage;
        Image* mAccumulationPathTracerImage;
        Buffer* mSampleBuffer = nullptr;
        Buffer* mCompositingBuffer = nullptr;
        ResourceManager* mResManager = nullptr;
        uint32_t mPtIteration = 0;

        Camera::Matrices mCamMatrices;

        ~ViewData()
        {
            assert(mResManager);
            if (gbuffer)
            {
                delete gbuffer;
            }
            if (prevDepth)
            {
                mResManager->destroyImage(prevDepth);
            }
            if (textureTonemapImage)
            {
                mResManager->destroyImage(textureTonemapImage);
            }
            if (textureUpscaleImage)
            {
                mResManager->destroyImage(textureUpscaleImage);
            }
            if (textureDebugViewImage)
            {
                mResManager->destroyImage(textureDebugViewImage);
            }
            if (mPathTracerImage)
            {
                mResManager->destroyImage(mPathTracerImage);
            }
            if (mAccumulationPathTracerImage)
            {
                mResManager->destroyImage(mAccumulationPathTracerImage);
            }
        }
    };

    ViewData* mPrevView = nullptr;

    Image* mAccumulatedPt = nullptr;

    std::array<ViewData*, MAX_FRAMES_IN_FLIGHT> mView;
    std::array<bool, MAX_FRAMES_IN_FLIGHT> mNeedRecreateView = { false, false, false };

    struct SceneRenderData
    {
        float animationTime = 0;
        uint32_t cameraIndex = 0;
        uint32_t mIndicesCount = 0;
        uint32_t mInstanceCount = 0;
        Buffer* mVertexBuffer = nullptr;
        Buffer* mMaterialBuffer = nullptr;
        Buffer* mIndexBuffer = nullptr;
        Buffer* mInstanceBuffer = nullptr;
        Buffer* mLightsBuffer = nullptr;
        Buffer* mBvhNodeBuffer = nullptr;

        const MaterialManager::TargetCode* mMaterialTargetCode = nullptr;

        Buffer* mMdlArgBuffer = nullptr;
        Buffer* mMdlRoBuffer = nullptr;
        Buffer* mMdlInfoBuffer = nullptr;
        Buffer* mMdlMaterialBuffer = nullptr;

        ResourceManager* mResManager = nullptr;
        explicit SceneRenderData(ResourceManager* resManager)
        {
            mResManager = resManager;
        }
        ~SceneRenderData()
        {
            assert(mResManager);
            if (mVertexBuffer)
            {
                mResManager->destroyBuffer(mVertexBuffer);
            }
            if (mIndexBuffer)
            {
                mResManager->destroyBuffer(mIndexBuffer);
            }
            // if (mMaterialBuffer)
            //{
            //    mResManager->destroyBuffer(mMaterialBuffer);
            //}
            if (mInstanceBuffer)
            {
                mResManager->destroyBuffer(mInstanceBuffer);
            }
            if (mLightsBuffer)
            {
                mResManager->destroyBuffer(mLightsBuffer);
            }
            if (mBvhNodeBuffer)
            {
                mResManager->destroyBuffer(mBvhNodeBuffer);
            }
        }
    };

    SceneRenderData* mCurrentSceneRenderData = nullptr;
    SceneRenderData* mDefaultSceneRenderData = nullptr;

    oka::Scene* mScene = nullptr;

    ViewData* createView(uint32_t width, uint32_t height, uint32_t spp);
    GBuffer* createGbuffer(uint32_t width, uint32_t height);

    void createVertexBuffer(oka::Scene& scene);
    void createLightsBuffer(oka::Scene& scene);
    void createBvhBuffer(oka::Scene& scene);
    void createIndexBuffer(oka::Scene& scene);
    void createInstanceBuffer(oka::Scene& scene);
    void createMdlBuffers();

    void recordImageBarrier(VkCommandBuffer& cmd,
                            Image* image,
                            VkImageLayout newLayout,
                            VkAccessFlags srcAccess,
                            VkAccessFlags dstAccess,
                            VkPipelineStageFlags sourceStage,
                            VkPipelineStageFlags destinationStage,
                            VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);
    void recordBufferBarrier(VkCommandBuffer& cmd,
                             Buffer* buff,
                             VkAccessFlags srcAccess,
                             VkAccessFlags dstAccess,
                             VkPipelineStageFlags sourceStage,
                             VkPipelineStageFlags destinationStage);

    void initDefaultSettings();
    void readSettings();

public:
    oka::ResourceManager* getResManager()
    {
        return mSharedCtx->mResManager;
    }

    oka::TextureManager* getTexManager()
    {
        return mSharedCtx->mTextureManager;
    }

    SceneRenderData* getSceneData()
    {
        return mCurrentSceneRenderData;
    }

    ShaderManager* getShaderManager()
    {
        return mSharedCtx->mShaderManager;
    }

    SettingsManager* getSettingsManager()
    {
        return mSharedCtx->mSettingsManager;
    }
};

} // namespace oka
