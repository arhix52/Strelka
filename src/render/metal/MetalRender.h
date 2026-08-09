#pragma once
#include <strelka/render/render.h>

#include <Metal/Metal.hpp>
#include <glm/glm.hpp>

#include "Metal4Context.h"
#include "MetalFxContext.h"
#include "ShaderTypes.h" // GeometryEntry, shared with the path-trace kernel
#include <atomic>
#include <map>
#include <vector>

namespace oka
{
static constexpr size_t kMaxFramesInFlight = 3;

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
    Buffer* getReadyBuffer() override;
    void* getReadyTexture() override;
    void resetTemporalHistory() override
    {
        mResetDenoiseHistory = true;
    }
    bool readDisplayTexture(std::vector<float>& rgba, uint32_t& width, uint32_t& height) override;
    bool readGuideTexture(Guide guide, std::vector<float>& rgba, uint32_t& width, uint32_t& height) override;
    float skinnedGeometryExtent() override;
    bool deviceError() const override
    {
        return mDeviceError;
    }

    bool motionGeometryActive() override
    {
        return mMotionBlasBuilt && mShutterIntervalActive;
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
    // Per scene mesh (one glTF primitive): just the data the skinning pass and
    // the geometry descriptors need. Acceleration structures live in Blas below,
    // because many meshes now share one.
    struct Mesh
    {
        MTL::Buffer* mPerPrimitiveBuffer = nullptr;
        uint32_t mTriangleCount = 0;
        uint32_t mVbOffset = 0;
        bool mIsSkeletal = false;
    };

    // One acceleration structure covering N geometries that always move together
    // (in practice: every primitive of one glTF mesh node).
    struct Blas
    {
        MTL::AccelerationStructure* mAs = nullptr;
        // Kept alive for refit. Invariant: it only names buffers, offsets and
        // triangle counts, none of which change while the pose does.
        MTL::PrimitiveAccelerationStructureDescriptor* mDescriptor = nullptr;
        MTL::Buffer* mScratch = nullptr; // persistent, reused every refit/rebuild
        size_t mRefitScratchSize = 0;
        size_t mBuildScratchSize = 0;
        bool mIsSkeletal = false;
        uint32_t mGeometryBase = 0; // first index into mGeometryEntries
    };

    // One emitted TLAS instance. A merged group contributes a single instance,
    // so this no longer maps one-to-one onto Scene::Instance.
    struct EmittedInstance
    {
        uint32_t sceneInstanceId; // representative, supplies the transform
        uint32_t asIndex;
        uint32_t userID;
        uint32_t mask;
    };

    void createMeshData(size_t meshIndex);
    MTL::AccelerationStructureMotionTriangleGeometryDescriptor* createMotionGeometryDescriptor(
        const oka::Mesh& sceneMesh, MTL::Buffer* perPrimitiveBuffer, uint32_t triangleCount);
    MTL::AccelerationStructureTriangleGeometryDescriptor* createStaticGeometryDescriptor(
        const oka::Mesh& sceneMesh, MTL::Buffer* perPrimitiveBuffer, uint32_t triangleCount);
    size_t buildBlas(const std::vector<uint32_t>& sceneInstanceIds, bool skeletal);
    struct View
    {
        oka::Camera::Matrices mCamMatrices;
    };

    // --- Settings change detection (replaces 16 static locals in render()) ---
    struct PrevSettings
    {
        uint32_t rectLightSamplingMethod = 0;
        uint32_t samplerType = 0;
        uint32_t blueNoiseSwitchSpp = 0;
        bool enableAccumulation = false;
        uint32_t sspTotal = 0;
        uint32_t spp = 0;
        bool playbackBlur = false;
        uint32_t shutterMode = 0;
        float shutterTime = 0.0f;
        bool enableMotionBlur = false;
        bool isMotionBlurVisible = true;
        bool enableCameraMotionBlur = false;
        int32_t useDof = 0;
        float focalDistance = 0.0f;
        float lensRadius = 0.0f;
        int32_t apertureBlades = 0;
        float shiftX = 0.0f;
        float shiftY = 0.0f;
        uint32_t maxDepth = 0;
        uint32_t debug = 0;
    };
    PrevSettings mPrevSettings;

    View mPrevView;
    MTL::Device* mDevice = nullptr;
    MTL::CommandQueue* mCommandQueue = nullptr;
    MTL::ComputePipelineState* mTonemapperPSO = nullptr;
    MTL::ComputePipelineState* mSkinningPSO = nullptr;
    MTL::ComputePipelineState* mTriangleUpdatePSO = nullptr;

    MTL::Buffer* mAccumulationBuffer = nullptr;
    MTL::Buffer* mLightBuffer = nullptr;
    MTL::Buffer* mVertexBuffer = nullptr;
    MTL::Buffer* mUniformBuffers[kMaxFramesInFlight] = {};
    MTL::Buffer* mUniformTMBuffers[kMaxFramesInFlight] = {};

    MTL::Buffer* mIndexBuffer = nullptr;
    uint32_t mTriangleCount = 0;
    std::vector<MetalRender::Mesh*> mMetalMeshes;
    std::vector<MetalRender::Blas> mBlasList;
    std::vector<EmittedInstance> mEmittedInstances;
    std::vector<GeometryEntry> mGeometryEntries;
    std::vector<MTL::AccelerationStructure*> mPrimitiveAccelerationStructures;
    MTL::AccelerationStructure* mInstanceAccelerationStructure = nullptr;
    MTL::Buffer* mInstanceBuffer = nullptr;
    MTL::Buffer* mTlasScratchBuffer = nullptr; // persistent, reused every TLAS refit
    size_t mTlasInstanceCount = 0;

    // Textures awaiting one shared mipmap blit; see generateTextureMips().
    std::vector<MTL::Texture*> mTexturesNeedingMips;
    uint32_t mTextureCacheHits = 0;
    uint32_t mTextureCacheMisses = 0;
    void generateTextureMips();

    MTL::Buffer* mMaterialBuffer = nullptr;
    // Set while uploading materials. Gates the alpha function constant, so a
    // scene with no cutouts compiles the same kernels it always did.
    bool mSceneHasAlphaMaterials = false;
    // One byte per material: does it need the alpha test at all. Drives the
    // per-geometry opaque flag, so traversal skips the intersection function on
    // geometry that never had a cutout in it.
    std::vector<uint8_t> mMaterialIsCutout;
    uint32_t mOpaqueGeometryCount = 0;
    uint32_t mCutoutGeometryCount = 0;


    // A GPU command buffer failed. Kept so a headless run can exit non-zero
    // instead of writing a black image and reporting success.
    bool mDeviceError = false;
    bool mDeviceErrorReported = false;
    std::vector<MTL::Texture*> mMaterialTextures;
    uint32_t mFrameIndex = 0;

    // Skinning / animation
    MTL::Buffer* mSkinDataBuffer = nullptr;
    MTL::Buffer* mJointMatricesBuffer = nullptr;
    std::vector<uint32_t> mJointMatOffsets;
    std::vector<glm::mat4> mJointMatScratch; // reused across the two skinning passes

    // Reusable per-frame vectors (avoid heap alloc each frame)
    std::vector<float> mAnimTargetTimes;
    std::vector<bool> mAnimChanged;

    // The previous frame's pose, for denoiser motion vectors.
    //
    // Deliberately not mPrevVertexBuffer: that one is a motion-blur shutter
    // keyframe, and when motion blur is off it is forced equal to the current
    // pose, which would make every motion vector describe a scene that never
    // deforms. These two are snapshots taken at the top of a frame, before
    // skinning and before the instance transforms are re-uploaded, so during
    // frame N they hold frame N-1.
    MTL::Buffer* mPrevFrameVertexBuffer = nullptr;
    MTL::Buffer* mPrevFrameInstanceBuffer = nullptr;
    bool mHasPrevFramePose = false;
    /// Snapshot the current pose. No-op when nothing in the scene can deform,
    /// in which case the current vertex buffer is already the previous one.
    void capturePrevFramePose();

    // Motion blur
    // Whether the megakernel was the selected tracer when the scene was built.
    // Per-primitive attribute data exists only for it; the wavefront tracer
    // refetches from the vertex buffer and would otherwise pay 72 bytes per
    // triangle twice over for nothing.
    bool mNeedsPrimitiveData = false;

    MTL::Buffer* mPrevVertexBuffer = nullptr;
    bool mOwnsPrevVertexBuffer = false;
    // The radiance cache. Sized once and reused for the whole render: a still
    // frame wants every sample's deposits, and a moving one is invalidated by
    // the camera check in updateUniforms.
    MTL::Buffer* mSharcBuffer = nullptr;
    uint32_t mSharcCapacity = 0;
    std::pair<size_t, size_t> mHostGeometryBytes{ 0, 0 };
    MTL::Buffer* mGeometryEntryBuffer = nullptr;
    bool mEnableMotionBlur = false;
    View mPrevMotionBlurView; // camera at T - shutter for camera motion blur

    // Environment map (dome light)
    MTL::Texture* mEnvMapTexture = nullptr;
    MTL::Texture* mEnvBackgroundTexture = nullptr;
    // Flat alias table, one entry per texel — replaces the marginal/conditional
    // CDF pair, so importance sampling costs one load instead of two binary
    // searches.
    MTL::Buffer* mEnvAliasBuffer = nullptr;
    float mEnvPdfScale = 0.0f;
    float mEnvMapAutoScale = 1.0f;
    bool mEnvMapLoaded = false;

    // --- Frame splitting ----------------------------------------------------
    // Target wall-clock cost of a single path-trace command buffer. Keeping each
    // submission short is what keeps the display queue (and therefore the UI)
    // running at vsync while a heavy frame renders.
    // Upper bound on bands per frame. Each band is a separate command buffer with
    // its own binding + residency setup, so splitting past this costs more than
    // the interleaving it enables.

    // --- Wavefront tracer ---------------------------------------------------
    // One set of pipelines per combination of scene features. Every branch a
    // function constant removes is on something that cannot change between rays,
    // so it is compiled away rather than executed; the cost is that a change to
    // any of those facts needs a new pipeline, hence the cache.
    struct WavefrontVariant
    {
        MTL::ComputePipelineState* generate = nullptr;
        MTL::ComputePipelineState* extendMotion = nullptr;
        MTL::ComputePipelineState* extendStatic = nullptr;
        MTL::ComputePipelineState* shade = nullptr;
        MTL::ComputePipelineState* miss = nullptr;
        MTL::ComputePipelineState* shadowMotion = nullptr;
        MTL::ComputePipelineState* shadowStatic = nullptr;
        // Bound to the shadow dispatch so the alpha test can run inside
        // traversal rather than as a restart loop around it.
        MTL::IntersectionFunctionTable* shadowTableMotion = nullptr;
        MTL::IntersectionFunctionTable* shadowTableStatic = nullptr;
        MTL::ComputePipelineState* sharcDeposit = nullptr;
    };
    enum WavefrontFeature : uint32_t
    {
        kFeatureEnvMap = 1u << 0,
        kFeatureLights = 1u << 1,
        kFeatureAlpha = 1u << 6,
        kFeatureMotionBlur = 1u << 2,
        kFeatureDof = 1u << 3,
        kFeatureDebug = 1u << 4,
        // Not a shader feature: pipelines for the Metal 4 path are built by a
        // different compiler and are not interchangeable, so the mode has to
        // separate them in the cache.
        kFeatureMetal4 = 1u << 5,
        kFeatureFog = 1u << 7,
        kFeatureSharc = 1u << 8,
        kFeatureCount = 1u << 5,
    };
    std::map<uint32_t, WavefrontVariant> mWavefrontVariants;
    MTL::Library* mWavefrontLibrary = nullptr;
    const WavefrontVariant* wavefrontVariantFor(uint32_t features);

    MTL::ComputePipelineState* mWavefrontResolvePSO = nullptr;
    // One capture scope per sample, so a profiler sees discrete frames.
    MTL::CaptureScope* mFrameScope = nullptr;
    MTL::ComputePipelineState* mWavefrontPreparePSO = nullptr;
    MTL::ComputePipelineState* mWavefrontPrepareShadowPSO = nullptr;
    // What the acceleration structures were actually built for. A skeletal mesh
    // only needs two keyframes when the shutter is open across them; with motion
    // blur off the shader pins the sample time to keyframe 1 and the second one
    // is never read, so the structure can be a plain static one and traversed as
    // such. Flipping the setting has to rebuild them.
    bool mMotionBlasBuilt = false;
    /// The two pose keyframes currently hold different poses, so the shutter spans
    /// a real interval and the frame has motion blur in it -- true across a pause.
    bool mShutterIntervalActive = false;
    bool mWasAnimationPlaying = false;
    bool mPausedBlurRefine = false;
    bool mBuildMotionBlas = false;
    void rebuildAccelerationStructures();
    MTL::ComputePipelineState* mWavefrontPrepareHitMissPSO = nullptr;

    MTL::Buffer* mPathStateBuffer = nullptr;
    MTL::Buffer* mPathRayBuffer = nullptr;
    MTL::Buffer* mHitBuffer = nullptr;
    MTL::Buffer* mIorStackBuffer = nullptr;
    MTL::Buffer* mRadianceBuffer = nullptr;
    MTL::Buffer* mGuideRadianceBuffer = nullptr;
    // Ping-pong queues of live path indices, plus the counters and the indirect
    // dispatch arguments derived from them. All GPU-side: the counts are never
    // read back, or every bounce would carry a round trip.
    MTL::Buffer* mPathQueueBuffer[2] = { nullptr, nullptr };
    MTL::Buffer* mWavefrontControlBuffer = nullptr;
    MTL::Buffer* mShadowRayBuffer = nullptr;
    // `extend` splits its input into rays that hit geometry and rays that
    // escaped, so neither stage dispatches threads for the other's work.
    MTL::Buffer* mHitQueueBuffer = nullptr;
    MTL::Buffer* mMissQueueBuffer = nullptr;
    // Denoiser guides: one packed record per pixel, written at the primary hit.
    MTL::Buffer* mAovBuffer = nullptr;

    // Metal 4 submission. Created alongside the Metal 3 objects so both paths
    // exist and can be compared; selected by render/pt/metal4.
    Metal4Context mMetal4;
    // Bumped when the allocation set can have changed, so residency is
    // rebuilt then and not every frame.
    uint32_t mMetal4ResidencyGeneration = 0;
    // Set whenever an allocation the frame can touch appears outside the
    // resolution change the generation above tracks -- a newly built wavefront
    // variant and its intersection function tables, in practice.
    bool mMetal4ResidencyDirty = true;
    void makeResourcesResidentForMetal4(Buffer* output);

    MTL::CounterSampleBuffer* mStageTimestampBuffer = nullptr;
    // Metal 4 counts the same stages through a heap the encoder writes into.
    MTL4::CounterHeap* mStageCounterHeap = nullptr;
    void createStageCounterHeap();
    void reportStageTimingsMetal4();
    // Heap ticks are not nanoseconds and not the clock sampleTimestamps reports,
    // so the scale is anchored once against the command buffer's own GPU time.
    double mGpuTicksToMs = 0.0;
    MTL::Buffer* mStageStatsBuffer = nullptr; // shared copy of the control buffer, profiling only
    static constexpr uint32_t kMaxStageSamples = 256;

    // Stage kind of each timestamp, in encode order. A stage's duration is the
    // gap to the next timestamp, so there is always one more sample than stage.
    std::vector<uint8_t> mStageKinds;
    bool mProfileStages = false;
    uint32_t mWavefrontCapacity = 0; // pixels the buffers above are sized for

    void buildWavefrontPipelines();
    // Per-dispatch GPU timestamps for the wavefront stages. Without them the
    // question "which stage is the frame in" can only be answered by ablation,
    // and every ablation changes the workload it is trying to measure.
    void createStageTimestampBuffer();
    void reportStageTimings();
    void ensureWavefrontBuffers(uint32_t width, uint32_t height);
    // Returns the encoder to keep using: in profiling mode each stage gets its
    // own, because this hardware can only sample counters at encoder boundaries.
    MTL::ComputeCommandEncoder* encodeWavefront(MTL::CommandBuffer* pCmd, MTL::ComputeCommandEncoder* enc,
                         MTL::Buffer* uniformBuffer,
                         Buffer* output, uint32_t width, uint32_t height, uint32_t sampleCount,
                         uint32_t features);
    void encodeWavefrontMetal4(MTL4::CommandBuffer* cmd, MTL4::ComputeCommandEncoder*& enc, MTL::Buffer* uniformBuffer, Buffer* output,
                               uint32_t width, uint32_t height, uint32_t sampleCount, uint32_t features);
    /// Constant-free stages, built by the Metal 4 compiler.
    MTL::ComputePipelineState* mWavefrontResolvePSO4 = nullptr;
    MTL::ComputePipelineState* mWavefrontPreparePSO4 = nullptr;
    MTL::ComputePipelineState* mWavefrontPrepareShadowPSO4 = nullptr;
    MTL::ComputePipelineState* mWavefrontPrepareHitMissPSO4 = nullptr;
    MTL::ComputePipelineState* mTonemapperPSO4 = nullptr;
    MTL::ComputePipelineState* mTonemapperTexPSO = nullptr;
    MTL::ComputePipelineState* mSkinningPSO4 = nullptr;
    MTL::ComputePipelineState* mTriangleUpdatePSO4 = nullptr;

    MTL::Library* loadShaderLibrary(const char* relativePath);
    void buildTonemapperPipeline();
    void buildBuffers();
    void uploadLightBuffer();
    void handleSceneChanges();

    // srgb selects the transfer function the sampler applies. Colour maps are
    // authored sRGB-encoded; data maps (normal, metallic-roughness, occlusion)
    // are not and must stay linear.
    // What a texture is for, which decides whether it can be block compressed.
    // A normal map cannot: a two-bit index along a line through 5:6:5 space is
    // not enough for a direction, and the banding shows as facets on every
    // curved surface.
    enum class TextureKind
    {
        Color,
        NonColor,
        Normal,
    };
    MTL::Texture* loadTextureFromFile(const std::string& fileName, bool srgb,
                                      TextureKind kind = TextureKind::Color);
    MTL::Texture* loadCachedTexture(const std::string& cachePath);
    std::string textureCacheKey(const std::string& fileName, bool srgb, TextureKind kind) const;
    void createMetalMaterials();

    MTL::AccelerationStructure* createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor);
    // Acceleration structure builds are grouped into one encoder so the hardware
    // can run them in parallel; each build in a group needs its own scratch.
    MTL::CommandBuffer* mAsGroupCommandBuffer = nullptr;
    MTL::AccelerationStructureCommandEncoder* mAsGroupEncoder = nullptr;
    std::vector<MTL::Buffer*> mAsGroupScratch;
    uint32_t mAsGroupPending = 0;
    void flushAccelerationStructureGroup();
    MTL::AccelerationStructure* createAccelerationStructureNoCompact(MTL::AccelerationStructureDescriptor* descriptor);
    void createAccelerationStructures();

    // Animation / skinning
    void buildSkinningPipeline();
    void createSkinDataBuffer();
    void allocJointMatrices();
    void applySkinning();
    void copyVertexBufferToPrev();

    // Async render (double-buffered output)
    Buffer* mAsyncOutputBuffers[2] = {nullptr, nullptr};
    // What the screen shows: tonemapped, one per async slot. The buffers above
    // keep the linear radiance, which is what the reference capture writes out.
    MTL::Texture* mDisplayTextures[2] = { nullptr, nullptr };
    uint32_t mDisplayTextureWidth = 0;
    uint32_t mDisplayTextureHeight = 0;
    MTL::TextureUsage mDisplayTextureUsage = 0;
    void ensureDisplayTextures(uint32_t width, uint32_t height);
    // What the tracer renders into when upscaling: the reduced resolution. The
    // display textures stay at output resolution and MetalFX bridges the two.
    MTL::Texture* mUpscaleTextures[2] = { nullptr, nullptr };
    uint32_t mUpscaleTextureWidth = 0;
    uint32_t mUpscaleTextureHeight = 0;
    MetalFxContext mMetalFx;
    void ensureUpscaleTextures(uint32_t width, uint32_t height);
    /// Where the tonemapper should write this frame: the reduced-resolution
    /// texture when upscaling, the display texture otherwise.
    MTL::Texture* tonemapTarget(bool upscaling) const;

    // Denoiser guides at render resolution, plus the linear output the denoiser
    // writes and the tonemapper then reads. Allocated only when denoising is on.
    struct GuideTextures
    {
        MTL::Texture* color = nullptr;
        MTL::Texture* depth = nullptr;
        MTL::Texture* motion = nullptr;
        MTL::Texture* diffuse = nullptr;
        MTL::Texture* specular = nullptr;
        MTL::Texture* normal = nullptr;
        MTL::Texture* roughness = nullptr;
        MTL::Texture* specularHitDistance = nullptr;
        MTL::Texture* reactive = nullptr;
    };
    GuideTextures mGuides;
    MTL::Texture* mDenoisedTexture = nullptr;
    bool mResetDenoiseHistory = true;
    // Enough of the previous camera to tell a cut from a pan.
    glm::float3 mPrevCameraPos{ 0.0f };
    glm::float3 mPrevCameraForward{ 0.0f, 0.0f, -1.0f };
    float mPrevCameraStep = 0.0f;
    bool mHasPrevCamera = false;
    bool mPrevDenoiseEnabled = false;
    bool mLoggedMetal4DenoiserGap = false;
    bool mLoggedShaderValidationDenoiserGap = false;
    uint32_t mGuideWidth = 0;
    uint32_t mGuideHeight = 0;
    // The output size the denoised texture was built for. Part of the cache key:
    // guides live at render resolution but the result does not.
    uint32_t mGuideOutWidth = 0;
    uint32_t mGuideOutHeight = 0;
    bool mLoggedUpscaleClamp = false;
    bool mLoggedSkinningPipelineGap = false;
    /// Opt back into the Metal 4 skinning submission, which is still wrong.
    const bool mSkinMetal4 = getenv("STRELKA_SKIN_METAL4") != nullptr;
    /// Force motion vectors back to camera-only, for measuring what the
    /// previous-frame pose is actually worth.
    const bool mNoPrevPose = getenv("STRELKA_NO_PREV_POSE") != nullptr;
    const bool mNoAccumColor = getenv("STRELKA_NO_ACCUM_COLOR") != nullptr;
    void applySkinningMetal3();
    MTL::ComputePipelineState* mAovResolvePSO = nullptr;
    MTL::ComputePipelineState* mAovResolvePSO4 = nullptr;
    void releaseGuideTextures();
    void ensureGuideTextures(uint32_t width, uint32_t height, uint32_t outWidth, uint32_t outHeight);
    /// Halton (2,3), the standard temporal jitter sequence: MetalFX reconstructs
    /// detail from knowing exactly how each frame was displaced.
    void frameJitter(uint64_t frameIndex, uint32_t phaseCount, float& x, float& y) const;
    std::atomic<int> mReadyIndex{-1};
    std::atomic<bool> mRenderBusy{false};
    int mWriteIndex = 0;

    // Sync mode (headless CLI): retain the last committed buffer and wait on it.
    bool mSyncMode = false;
    MTL::CommandBuffer* mLastCommandBuffer = nullptr;
    // What renderSync blocks on when the frame went out through Metal 4: there is
    // no MTL4 command buffer to wait on, so the queue signals a shared event.
    uint64_t mMetal4FrameValue = 0;
    void retainCommandBufferForSync(MTL::CommandBuffer* pCmd);

    // Environment map
    void loadEnvMap(const std::string& texturePath);
    void loadEnvBackground(const std::string& texturePath);

    // BVH management
    void ensureScratchBuffer(MTL::Buffer*& buffer, size_t requiredSize);
    /// Refit every skeletal BLAS in one command buffer, rebuilding a bounded
    /// slice of them per frame to amortise the periodic quality refresh.
    void updateSkeletalBLAS();
    static constexpr size_t kMaxBlasRebuildsPerFrame = 8;
    size_t mNextBlasRebuildIndex = 0;
    void rebuildTLAS();
    void updateInstanceTransforms();
};

} // namespace oka
