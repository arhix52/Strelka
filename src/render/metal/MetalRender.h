#pragma once
#include "render.h"

#include <Metal/Metal.hpp>

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

    View mPrevView;
    MTL::Device* mDevice;
    MTL::CommandQueue* mCommandQueue;
    MTL::Library* mShaderLibrary;

    MTL::ComputePipelineState* mPathTracingPSO = nullptr;
    MTL::ComputePipelineState* mTonemapperPSO = nullptr;
    MTL::ComputePipelineState* mSkinningPSO = nullptr;
    MTL::ComputePipelineState* mTriangleUpdatePSO = nullptr;

    MTL::Buffer* mAccumulationBuffer;
    MTL::Buffer* mLightBuffer;
    MTL::Buffer* mVertexBuffer;
    MTL::Buffer* mUniformBuffers[kMaxFramesInFlight];
    MTL::Buffer* mUniformTMBuffers[kMaxFramesInFlight];
    
    MTL::Buffer* mIndexBuffer;
    uint32_t mTriangleCount;
    std::vector<MetalRender::Mesh*> mMetalMeshes;
    std::vector<MTL::AccelerationStructure*> mPrimitiveAccelerationStructures;
    MTL::AccelerationStructure* mInstanceAccelerationStructure;
    MTL::Buffer* mInstanceBuffer;

    MTL::Buffer* mMaterialBuffer = nullptr;
    std::vector<MTL::Texture*> mMaterialTextures;
    uint32_t mFrameIndex = 0;

    // Skinning / animation
    MTL::Buffer* mSkinDataBuffer = nullptr;
    MTL::Buffer* mJointMatricesBuffer = nullptr;
    std::vector<uint32_t> mJointMatOffsets;
    uint32_t mBlasUpdateCount = 0;

    // Motion blur
    MTL::Buffer* mPrevVertexBuffer = nullptr;
    MTL::Buffer* mInstanceDataBuffer = nullptr;
    bool mEnableMotionBlur = false;
    View mPrevMotionBlurView; // camera at T - shutter for camera motion blur

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

    // BVH management
    MTL::PrimitiveAccelerationStructureDescriptor* createMotionBLASDescriptor(
        const oka::Mesh& sceneMesh, MTL::Buffer* perPrimitiveBuffer, uint32_t triangleCount);
    void refitBLAS(int meshIndex);
    void rebuildBLAS(int meshIndex);
    void rebuildTLAS();
    void updateInstanceTransforms();
};

} // namespace oka
