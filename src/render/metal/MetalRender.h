#pragma once
#include "render.h"
#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>

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

    MTL::ComputePipelineState* mRayTracingPSO;

    MTL::Buffer* mAccumulationBuffer;
    MTL::Buffer* mLightBuffer;
    MTL::Buffer* mVertexBuffer;
    MTL::Buffer* mUniformBuffers[kMaxFramesInFlight];
    MTL::Buffer* mIndexBuffer;
    uint32_t mTriangleCount;
    std::vector<MetalRender::Mesh*> mMetalMeshes;
    std::vector<MTL::AccelerationStructure*> mPrimitiveAccelerationStructures;
    MTL::AccelerationStructure* mInstanceAccelerationStructure;
    MTL::Buffer* mInstanceBuffer;

    MTL::Buffer* mMaterialBuffer;

    uint32_t mFrameIndex;
    dispatch_semaphore_t mSemaphoreDispatch;

    void buildComputePipeline();
    void buildBuffers();

    void createMetalMaterials();

    MTL::AccelerationStructure* createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor);
    void createAccelerationStructures();
};

} // namespace oka
