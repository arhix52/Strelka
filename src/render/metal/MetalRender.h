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
    ~MetalRender();

    void init() override;
    void render(Buffer* output) override;
    Buffer* createBuffer(const BufferDesc& desc) override;

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

    MTL::RenderPipelineState* _pPSO;
    MTL::ComputePipelineState* mRayTracingPSO;

    MTL::Buffer* _pVertexDataBuffer;
    MTL::Buffer* _pInstanceDataBuffer;
    MTL::Buffer* _pUniformBuffer[kMaxFramesInFlight];
    MTL::Buffer* _pIndexBuffer;
    uint32_t _triangleCount;
    std::vector<MetalRender::Mesh*> mMetalMeshes;
    std::vector<MTL::AccelerationStructure*> _primitiveAccelerationStructures;
    MTL::AccelerationStructure* _instanceAccelerationStructure;
    MTL::Buffer* _instanceBuffer;

    MTL::Texture* mFramebufferTexture;

    int _frame;
    uint32_t width;
    uint32_t height;
    dispatch_semaphore_t _semaphore;

    void buildComputePipeline();
    void buildBuffers();
    void buildTexture(uint32_t width, uint32_t heigth);

    MTL::AccelerationStructure* createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor);
    void createAccelerationStructures();
};

} // namespace oka
