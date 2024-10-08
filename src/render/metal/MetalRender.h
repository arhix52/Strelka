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

private:
    struct Mesh
    {
        MTL::AccelerationStructure* mGas;
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

    MTL::ComputePipelineState* mPathTracingPSO;
    MTL::ComputePipelineState* mTonemapperPSO;

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

    MTL::Buffer* mMaterialBuffer;
    std::vector<MTL::Texture*> mMaterialTextures;
    uint32_t mFrameIndex;

    void buildComputePipeline();
    void buildTonemapperPipeline();
    void buildBuffers();
    
    MTL::Texture* loadTextureFromFile(const std::string& fileName);
    void createMetalMaterials();

    MTL::AccelerationStructure* createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor);
    void createAccelerationStructures();
};

} // namespace oka
