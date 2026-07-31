#pragma once
#include <strelka/render/render.h>

#include <Metal/Metal.hpp>
#include <glm/glm.hpp>

#include "Metal4Context.h"
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
    Buffer* createBuffer(const BufferDesc& desc) override;

    void triggerRenderIfIdle() override;
    Buffer* getReadyBuffer() override;

    void* getNativeDevicePtr() override
    {
        return mDevice;
    }

    void* getNativeCommandQueue() override
    {
        return mCommandQueue;
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
        uint32_t mIndexOffset = 0;
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
        bool enableAccumulation = false;
        uint32_t sspTotal = 0;
        uint32_t spp = 0;
        bool enableMotionBlur = false;
        bool isMotionBlurVisible = true;
        bool enableCameraMotionBlur = true;
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

    MTL::ComputePipelineState* mPathTracingPSO = nullptr;
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

    MTL::Buffer* mMaterialBuffer = nullptr;
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

    // Motion blur
    MTL::Buffer* mPrevVertexBuffer = nullptr;
    MTL::Buffer* mGeometryEntryBuffer = nullptr;
    bool mEnableMotionBlur = false;
    View mPrevMotionBlurView; // camera at T - shutter for camera motion blur

    // Environment map (dome light)
    MTL::Texture* mEnvMapTexture = nullptr;
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
    static constexpr double kTargetSubmissionMs = 6.0;
    // Upper bound on bands per frame. Each band is a separate command buffer with
    // its own binding + residency setup, so splitting past this costs more than
    // the interleaving it enables.
    static constexpr uint32_t kMaxBands = 8;
    std::atomic<double> mFrameGpuStartSeconds{ 0.0 };
    uint32_t mLastBandTotalRows = 0;

    uint32_t computeBandHeight(uint32_t height) const;
    void encodePathTraceBindings(MTL::ComputeCommandEncoder* enc, MTL::Buffer* uniformBuffer, Buffer* output);

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
    };
    enum WavefrontFeature : uint32_t
    {
        kFeatureEnvMap = 1u << 0,
        kFeatureLights = 1u << 1,
        kFeatureMotionBlur = 1u << 2,
        kFeatureDof = 1u << 3,
        kFeatureDebug = 1u << 4,
        // Not a shader feature: pipelines for the Metal 4 path are built by a
        // different compiler and are not interchangeable, so the mode has to
        // separate them in the cache.
        kFeatureMetal4 = 1u << 5,
        kFeatureCount = 1u << 5,
    };
    std::map<uint32_t, WavefrontVariant> mWavefrontVariants;
    MTL::Library* mWavefrontLibrary = nullptr;
    const WavefrontVariant* wavefrontVariantFor(uint32_t features);

    MTL::ComputePipelineState* mWavefrontResolvePSO = nullptr;
    MTL::ComputePipelineState* mWavefrontPreparePSO = nullptr;
    MTL::ComputePipelineState* mWavefrontPrepareShadowPSO = nullptr;
    bool mSceneHasMotionBlas = false;
    // What the acceleration structures were actually built for. A skeletal mesh
    // only needs two keyframes when the shutter is open across them; with motion
    // blur off the shader pins the sample time to keyframe 1 and the second one
    // is never read, so the structure can be a plain static one and traversed as
    // such. Flipping the setting has to rebuild them.
    bool mMotionBlasBuilt = false;
    bool mBuildMotionBlas = false;
    uint32_t mMotionBlasSwitchFrames = 0;
    void rebuildAccelerationStructures();
    MTL::ComputePipelineState* mWavefrontPrepareHitMissPSO = nullptr;

    MTL::Buffer* mPathStateBuffer = nullptr;
    MTL::Buffer* mPathRayBuffer = nullptr;
    MTL::Buffer* mHitBuffer = nullptr;
    MTL::Buffer* mIorStackBuffer = nullptr;
    MTL::Buffer* mRadianceBuffer = nullptr;
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
    void makeResourcesResidentForMetal4(Buffer* output);

    MTL::CounterSampleBuffer* mStageTimestampBuffer = nullptr;
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
    void encodeWavefrontMetal4(MTL4::ComputeCommandEncoder* enc, MTL::Buffer* uniformBuffer, Buffer* output,
                               uint32_t width, uint32_t height, uint32_t sampleCount, uint32_t features);
    /// Constant-free stages, built by the Metal 4 compiler.
    MTL::ComputePipelineState* mWavefrontResolvePSO4 = nullptr;
    MTL::ComputePipelineState* mWavefrontPreparePSO4 = nullptr;
    MTL::ComputePipelineState* mWavefrontPrepareShadowPSO4 = nullptr;
    MTL::ComputePipelineState* mWavefrontPrepareHitMissPSO4 = nullptr;
    MTL::ComputePipelineState* mTonemapperPSO4 = nullptr;

    MTL::Library* loadShaderLibrary(const char* relativePath);
    void buildComputePipeline();
    void buildTonemapperPipeline();
    void buildBuffers();

    MTL::Texture* loadTextureFromFile(const std::string& fileName);
    void createMetalMaterials();

    MTL::AccelerationStructure* createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor);
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
    std::atomic<int> mReadyIndex{-1};
    std::atomic<bool> mRenderBusy{false};
    int mWriteIndex = 0;

    // Environment map
    void loadEnvMap(const std::string& texturePath);

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
