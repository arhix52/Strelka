#pragma once
#include <strelka/render/render.h>

#include <Metal/Metal.hpp>
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
    struct Mesh
    {
        MTL::AccelerationStructure* mGas = nullptr;
        MTL::Buffer* mPerPrimitiveBuffer = nullptr;
        uint32_t mTriangleCount = 0;
        uint32_t mVbOffset = 0;
        uint32_t mIndexOffset = 0;
        bool mIsSkeletal = false;
    };

    Mesh* createMesh(const oka::Mesh& mesh);
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
    std::vector<MTL::AccelerationStructure*> mPrimitiveAccelerationStructures;
    MTL::AccelerationStructure* mInstanceAccelerationStructure = nullptr;
    MTL::Buffer* mInstanceBuffer = nullptr;

    MTL::Buffer* mMaterialBuffer = nullptr;
    std::vector<MTL::Texture*> mMaterialTextures;
    uint32_t mFrameIndex = 0;

    // Skinning / animation
    MTL::Buffer* mSkinDataBuffer = nullptr;
    MTL::Buffer* mJointMatricesBuffer = nullptr;
    std::vector<uint32_t> mJointMatOffsets;
    uint32_t mBlasUpdateCount = 0;

    // Reusable per-frame vectors (avoid heap alloc each frame)
    std::vector<float> mAnimTargetTimes;
    std::vector<bool> mAnimChanged;

    // Motion blur
    MTL::Buffer* mPrevVertexBuffer = nullptr;
    MTL::Buffer* mInstanceDataBuffer = nullptr;
    bool mEnableMotionBlur = false;
    View mPrevMotionBlurView; // camera at T - shutter for camera motion blur

    // Environment map (dome light)
    MTL::Texture* mEnvMapTexture = nullptr;
    MTL::Buffer* mEnvCdfXBuffer = nullptr;
    MTL::Buffer* mEnvCdfYBuffer = nullptr;
    float mEnvMapAutoScale = 1.0f;
    bool mEnvMapLoaded = false;

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
    MTL::PrimitiveAccelerationStructureDescriptor* createMotionBLASDescriptor(
        const oka::Mesh& sceneMesh, MTL::Buffer* perPrimitiveBuffer, uint32_t triangleCount);
    void refitBLAS(int meshIndex);
    void rebuildBLAS(int meshIndex);
    void rebuildTLAS();
    void updateInstanceTransforms();
};

} // namespace oka
