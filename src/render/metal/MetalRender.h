#pragma once
#include <strelka/render/render.h>

#include <Metal/Metal.hpp>
#include <env.h>
#include <glm/glm.hpp>

#include "Metal4Context.h"
#include "MetalTextures.h"
#include "MetalEnvironment.h"
#include "MetalGeometry.h"
#include "MetalMaterials.h"
#include "MetalLights.h"
#include "MetalAccelStructure.h"
#include "MetalSkinning.h"
#include "MetalFrameUniforms.h"
#include "MetalPostProcess.h"
#include "MetalScenePreparation.h"
#include "scene_stream.h"
#include "MetalWavefrontIntegrator.h"
#include "MetalDomainMap.h"
#include "ShaderTypes.h" // GeometryEntry, shared with the path-trace kernel
#include <atomic>
#include <vector>

namespace oka
{

static constexpr size_t kMaxFramesInFlight = metal::kFrameUniformSlots;

class MetalRender : public Render
{
public:
    MetalRender(/* args */);
    ~MetalRender() override;

    void init() override;
    void render(Buffer* output) override;
    void renderSync(Buffer* output) override;
    void beginGpuCapture(const std::string& path) override;
    void endGpuCapture() override;
    Buffer* createBuffer(const BufferDesc& desc) override;

    void triggerRenderIfIdle() override;
    bool isRenderBusy() const override
    {
        return mRenderBusy.load(std::memory_order_acquire);
    }
    bool denoiserFallbackActive() const override
    {
        return mDenoiserFallbackActive.load(std::memory_order_relaxed);
    }
    Buffer* getReadyBuffer() override;
    void* getReadyTexture() override;
    ReadyFrame getReadyFrame() override
    {
        const int readyIndex = mReadyIndex.load(std::memory_order_acquire);
        if (readyIndex < 0 || readyIndex > 1)
        {
            return {};
        }
        return { mAsyncOutputBuffers[readyIndex], mPost.displayTexture(readyIndex) };
    }

    bool memoryReport(MemoryReport& report) const override;

    bool isBuildingScene() const override
    {
        return mScenePrep.isBuilding();
    }
    void resetTemporalHistory() override
    {
        mResetDenoiseHistory = true;
    }
    bool readDisplayTexture(std::vector<float>& rgba, uint32_t& width, uint32_t& height) override;
    bool readGuideTexture(Guide guide, std::vector<float>& rgba, uint32_t& width, uint32_t& height) override;
    float skinnedGeometryExtent() override;
    bool deviceError() const override
    {
        return mDeviceError || mMetal4FrameFailed.load(std::memory_order_relaxed);
    }

    bool motionGeometryActive() override
    {
        return mAccel.motionBlasBuilt() && mShutterIntervalActive;
    }

    void* getNativeDevicePtr() override
    {
        return mDevice;
    }

    void* getNativeCommandQueue() override
    {
        return mCommandQueue;
    }

    // Non-null only while frames go out through Metal 4: that is the case where
    // the display's queue is not the one that produced the texture.
    void* getNativeFrameEvent() override
    {
        return (mMetal4.isValid() && mMetal4FrameValue != 0) ? (void*)mMetal4.frameEvent() : nullptr;
    }
    uint64_t frameEventValue() const override
    {
        return mMetal4FrameValue;
    }

private:
    using Mesh = metal::MetalGeometry::Mesh;

    struct View
    {
        oka::Camera::Matrices mCamMatrices;
    };

    View mPrevView;
    MTL::Device* mDevice = nullptr;
    MTL::CommandQueue* mCommandQueue = nullptr;

    MTL::Buffer* mAccumulationBuffer = nullptr;

    // Domain-owned resources (see MetalDomainMap.h).
    metal::MetalTextures mTextures;
    metal::MetalEnvironment mEnvironment;
    metal::MetalGeometry mGeometry;
    metal::MetalMaterials mMaterials;
    metal::MetalLights mLights;
    metal::MetalAccelStructure mAccel;
    metal::MetalSkinning mSkinning;
    metal::MetalFrameUniforms mFrameUniforms;
    metal::MetalPostProcess mPost;
    metal::MetalScenePreparation mScenePrep;
    // Paces how often a scene that is still loading is republished.
    metal::PublishClock mPublishClock;
    double mBuildStartMs = 0.0;
    bool mReportedFirstPartialFrame = false;
    metal::MetalWavefrontIntegrator mIntegrator;
    void generateTextureMips();
    metal::IntegratorSceneBindings integratorSceneBindings();

    /// Extra extend/shade iterations a scene needs so that geometry which does
    /// not advance a path's depth cannot eat its bounce budget. See
    /// wavefrontIterations().
    static constexpr uint32_t kPassthroughIterations = 8;
    static constexpr uint32_t kSubsurfaceIterations = 64;
    /// Iterations of the wavefront loop for one sample at this path depth.
    uint32_t wavefrontIterations(uint32_t maxDepth) const;

    // A GPU command buffer failed. Kept so a headless run can exit non-zero
    // instead of writing a black image and reporting success.
    bool mDeviceError = false;
    bool mDeviceErrorReported = false;
    // Same, for Metal 4: the commit feedback runs off the render thread, so it
    // cannot touch the two above.
    std::atomic<bool> mMetal4FrameFailed{ false };
    std::atomic<bool> mMetal4FrameFailReported{ false };
    uint32_t mFrameIndex = 0;

    // Reusable per-frame vectors (avoid heap alloc each frame)
    std::vector<float> mAnimTargetTimes;
    std::vector<bool> mAnimChanged;

    // The previous frame's pose, for denoiser motion vectors.
    //
    // Deliberately not Geometry's prev VB: that one is a motion-blur shutter
    // keyframe, and when motion blur is off it is forced equal to the current
    // pose, which would make every motion vector describe a scene that never
    // deforms. Vertices still need a snapshot before skinning. Instance
    // descriptors do not: MetalAccelStructure swaps two fully initialized
    // buffers when transforms change, leaving the old current as previous.
    MTL::Buffer* mPrevFrameVertexBuffer = nullptr;
    bool mHasPrevFramePose = false;
    /// Prepare previous-pose storage and snapshot deforming vertices.
    bool capturePrevFramePose();

    bool mEnableMotionBlur = false;
    View mPrevMotionBlurView; // camera at T - shutter for camera motion blur

    // One capture scope per sample, so a profiler sees discrete frames.
    MTL::CaptureScope* mFrameScope = nullptr;
    /// The two pose keyframes currently hold different poses, so the shutter spans
    /// a real interval and the frame has motion blur in it -- true across a pause.
    bool mShutterIntervalActive = false;
    bool mWasAnimationPlaying = false;
    /// A freshly built scene has never been posed: the vertex buffer holds the
    /// bind pose the loader uploaded, and the animation block only acts on a
    /// change of time -- which a load does not produce, because the loader sets
    /// each animation's current time to its start and the editor asks for that
    /// same start. Raised by the build so the first frame past it poses once.
    bool mNeedsInitialPose = false;
    bool mPausedBlurRefine = false;
    void rebuildAccelerationStructures();

    // Metal 4 owns deformation, acceleration-structure maintenance and tracing.
    // The Metal 3 queue remains for the denoiser and display/readback utilities.
    Metal4Context mMetal4;
    // Bumped when the allocation set can have changed, so residency is
    // rebuilt then and not every frame.
    uint32_t mMetal4ResidencyGeneration = 0;
    // Still owned by MetalRender: residency spans every domain (geometry, lights,
    // textures, guides), not only wavefront queues. Integrator flags dirty when
    // a new variant builds intersection tables.
    void makeResourcesResidentForMetal4(Buffer* output);

    MTL::Library* loadShaderLibrary(const char* relativePath);
    void buildBuffers();
    void uploadLightBuffer();
    void handleSceneChanges();

    enum class TextureKind
    {
        Color,
        NonColor,
        Normal,
    };
    MTL::Texture* loadTextureFromFile(const std::string& fileName, bool srgb, TextureKind kind = TextureKind::Color);
    MTL::Texture* loadCachedTexture(const std::string& cachePath);
    std::string textureCacheKey(const std::string& fileName, bool srgb, TextureKind kind) const;
    void createMetalMaterials();
    bool stepMetalMaterials(double budgetMs);

    bool stepSceneBuild(Buffer* output);
    void finishSceneBuild(Buffer* output);
    metal::SceneBuildHooks makeSceneBuildHooks();
    void buildSceneEnvironment(Buffer* output);
    void buildSceneTail(Buffer* output);

    // Async render (double-buffered output)
    Buffer* mAsyncOutputBuffers[2] = { nullptr, nullptr };
    bool mResetDenoiseHistory = true;
    /// A denoised frame is present in mPost.denoisedTexture() and belongs to the
    /// scene and camera as they stand. What the sample budget freezes is that
    /// texture, so a post-only frame needs to know it exists.
    bool mHasDenoisedFrame = false;
    // Enough of the previous camera to tell a cut from a pan.
    glm::float3 mPrevCameraPos{ 0.0f };
    glm::float3 mPrevCameraForward{ 0.0f, 0.0f, -1.0f };
    float mPrevCameraStep = 0.0f;
    bool mHasPrevCamera = false;
    /// Force motion vectors back to camera-only, for measuring what the
    /// previous-frame pose is actually worth.
    const bool mNoPrevPose = envFlag("STRELKA_NO_PREV_POSE");
    const bool mNoAccumColor = envFlag("STRELKA_NO_ACCUM_COLOR");
    std::atomic<int> mReadyIndex{ -1 };
    std::atomic<bool> mRenderBusy{ false };
    std::atomic<bool> mDenoiserFallbackActive{ false };
    int mWriteIndex = 0;

    // Sync mode (headless CLI): retain the last committed buffer and wait on it.
    bool mSyncMode = false;
    MTL::CommandBuffer* mLastCommandBuffer = nullptr;
    // What renderSync blocks on when the frame went out through Metal 4: there is
    // no MTL4 command buffer to wait on, so the queue signals a shared event.
    uint64_t mMetal4FrameValue = 0;
    void retainCommandBufferForSync(MTL::CommandBuffer* pCmd);

    void loadEnvMap(const std::string& texturePath);
    void loadEnvBackground(const std::string& texturePath);
};

} // namespace oka
