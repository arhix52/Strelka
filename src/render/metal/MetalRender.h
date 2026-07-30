#pragma once
#include <strelka/render/render.h>

#include <Metal/Metal.hpp>
#include <glm/glm.hpp>

#include "ShaderTypes.h" // GeometryEntry, shared with the path-trace kernel
#include <atomic>
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
    uint32_t mBlasUpdateCount = 0;
    uint32_t mFramesSinceFullRebuild = 0; // throttle full rebuilds during rapid scrubbing

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
    void updateSkeletalBLAS(bool largeTimeJump);
    static constexpr size_t kMaxBlasRebuildsPerFrame = 8;
    size_t mNextBlasRebuildIndex = 0;
    void rebuildTLAS();
    void updateInstanceTransforms();
};

} // namespace oka
