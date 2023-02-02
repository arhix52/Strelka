#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include "MetalRender.h"
#include "MetalBuffer.h"

#include <cassert>

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <glm/ext/matrix_relational.hpp>


#include <simd/simd.h>

#include "shaders/ShaderTypes.h"

using namespace oka;

MetalRender::MetalRender(/* args */)
{
}

MetalRender::~MetalRender()
{
}

void MetalRender::init()
{
    mDevice = MTL::CreateSystemDefaultDevice();
    mCommandQueue = mDevice->newCommandQueue();
    buildComputePipeline();
    // buildBuffers();
    // createAccelerationStructures();
    _semaphore = dispatch_semaphore_create(oka::kMaxFramesInFlight);
}

void MetalRender::buildTexture(uint32_t width, uint32_t heigth)
{
    MTL::TextureDescriptor* pTextureDesc = MTL::TextureDescriptor::alloc()->init();
    pTextureDesc->setWidth(width);
    pTextureDesc->setHeight(heigth);
    pTextureDesc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    pTextureDesc->setTextureType(MTL::TextureType2D);
    pTextureDesc->setStorageMode(MTL::StorageModeManaged);
    pTextureDesc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead | MTL::ResourceUsageWrite);

    MTL::Texture* pTexture = mDevice->newTexture(pTextureDesc);
    mFramebufferTexture = pTexture;

    pTextureDesc->release();
}

void MetalRender::render(Buffer* output)
{
    using simd::float3;
    using simd::float4;
    using simd::float4x4;
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    if (getSharedContext().mFrameNumber == 0)
    {
        buildBuffers();
        createAccelerationStructures();
        // create accum buffer, we don't need cpu access, make it device only
        mAccumulationBuffer = mDevice->newBuffer(
            output->width() * output->height() * output->getElementSize(), MTL::ResourceStorageModePrivate);
    }

    _frame = (_frame + 1) % oka::kMaxFramesInFlight;

    MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    MetalRender* pRenderer = this;
    pCmd->addCompletedHandler(^void(MTL::CommandBuffer* pCmd) {
      dispatch_semaphore_signal(pRenderer->_semaphore);
    });

    // Update camera state:
    width = output->width();
    height = output->height();

    oka::Camera& camera = mScene->getCamera(1);
    camera.updateAspectRatio(width / (float)height);
    camera.updateViewMatrix();

    View currView;

    currView.mCamMatrices = camera.matrices;

    if (glm::any(glm::notEqual(currView.mCamMatrices.perspective, mPrevView.mCamMatrices.perspective)) ||
        glm::any(glm::notEqual(currView.mCamMatrices.view, mPrevView.mCamMatrices.view)))
    {
        // need reset
    }

    MTL::Buffer* pUniformBuffer = _pUniformBuffer[_frame];
    Uniforms* pUniformData = reinterpret_cast<Uniforms*>(pUniformBuffer->contents());
    pUniformData->frameIndex = _frame;
    pUniformData->subframeIndex = getSharedContext().mFrameNumber;
    pUniformData->height = height;
    pUniformData->width = width;
    pUniformData->numLights = mScene->getLightsDesc().size();
    pUniformData->samples_per_launch = 1;
    pUniformData->enableAccumulation = 1;
    pUniformData->missColor = float3(0.0f);
    pUniformData->maxDepth = 4;

    glm::float4x4 invView = glm::inverse(camera.matrices.view);
    for (int column = 0; column < 4; column++)
    {
        for (int row = 0; row < 4; row++)
        {
            pUniformData->viewToWorld.columns[column][row] = invView[column][row];
        }
    }
    for (int column = 0; column < 4; column++)
    {
        for (int row = 0; row < 4; row++)
        {
            pUniformData->clipToView.columns[column][row] = camera.matrices.invPerspective[column][row];
        }
    }

    pUniformBuffer->didModifyRange(NS::Range::Make(0, sizeof(Uniforms)));

    MTL::ComputeCommandEncoder* pComputeEncoder = pCmd->computeCommandEncoder();
    pComputeEncoder->useResource(mLightBuffer, MTL::ResourceUsageRead);
    pComputeEncoder->useResource(_instanceAccelerationStructure, MTL::ResourceUsageRead);
    for (int i = 0; i < _primitiveAccelerationStructures.size(); ++i)
    {
        pComputeEncoder->useResource(_primitiveAccelerationStructures[i], MTL::ResourceUsageRead);
    }

    pComputeEncoder->setComputePipelineState(mRayTracingPSO);
    pComputeEncoder->setBuffer(_pUniformBuffer[_frame], 0, 0);
    pComputeEncoder->setBuffer(_instanceBuffer, 0, 1);
    pComputeEncoder->setAccelerationStructure(_instanceAccelerationStructure, 2);
    pComputeEncoder->setBuffer(mLightBuffer, 0, 3);
    // Output
    pComputeEncoder->setBuffer(((oka::MetalBuffer*)output)->getNativePtr(), 0, 4);
    pComputeEncoder->setBuffer(mAccumulationBuffer, 0, 5);

    MTL::Size gridSize = MTL::Size(width, height, 1);

    NS::UInteger threadGroupSize = mRayTracingPSO->maxTotalThreadsPerThreadgroup();
    MTL::Size threadgroupSize(threadGroupSize, 1, 1);

    pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);

    pComputeEncoder->endEncoding();

    pCmd->commit();
    pCmd->waitUntilCompleted();

    pPool->release();

    getSharedContext().mFrameNumber++;
    mPrevView = currView;
}

Buffer* oka::MetalRender::createBuffer(const BufferDesc& desc)
{
    assert(mDevice);
    const size_t size = desc.height * desc.width * Buffer::getElementSize(desc.format);
    assert(size != 0);
    MTL::Buffer* buff = mDevice->newBuffer(size, MTL::ResourceStorageModeManaged);
    assert(buff);
    MetalBuffer* res = new MetalBuffer(buff, desc.format, desc.width, desc.height);
    assert(res);
    return res;
}

void MetalRender::buildComputePipeline()
{
    NS::Error* pError = nullptr;
    MTL::Library* pComputeLibrary =
        mDevice->newLibrary(NS::String::string("./metal/shaders/pathtrace.metallib", NS::UTF8StringEncoding), &pError);
    if (!pComputeLibrary)
    {
        __builtin_printf("%s", pError->localizedDescription()->utf8String());
        assert(false);
    }
    MTL::Function* pPathTraceFn =
        pComputeLibrary->newFunction(NS::String::string("raytracingKernel", NS::UTF8StringEncoding));
    mRayTracingPSO = mDevice->newComputePipelineState(pPathTraceFn, &pError);
    if (!mRayTracingPSO)
    {
        __builtin_printf("%s", pError->localizedDescription()->utf8String());
        assert(false);
    }

    pPathTraceFn->release();
    pComputeLibrary->release();
}

void MetalRender::buildBuffers()
{
    const std::vector<oka::Scene::Vertex>& vertices = mScene->getVertices();
    const std::vector<uint32_t>& indices = mScene->getIndices();
    const std::vector<Scene::Light>& lightDescs = mScene->getLights();

    assert(sizeof(Scene::Light) == sizeof(UniformLight));
    const size_t lightBufferSize = sizeof(Scene::Light) * lightDescs.size();
    const size_t vertexDataSize = sizeof(oka::Scene::Vertex) * vertices.size();
    const size_t indexDataSize = sizeof(uint32_t) * indices.size();

    MTL::Buffer* pLightBuffer = mDevice->newBuffer(lightBufferSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer* pVertexBuffer = mDevice->newBuffer(vertexDataSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer* pIndexBuffer = mDevice->newBuffer(indexDataSize, MTL::ResourceStorageModeManaged);

    mLightBuffer = pLightBuffer;
    _pVertexDataBuffer = pVertexBuffer;
    _pIndexBuffer = pIndexBuffer;

    memcpy(mLightBuffer->contents(), lightDescs.data(), lightBufferSize);
    memcpy(_pVertexDataBuffer->contents(), vertices.data(), vertexDataSize);
    memcpy(_pIndexBuffer->contents(), indices.data(), indexDataSize);

    mLightBuffer->didModifyRange(NS::Range::Make(0, mLightBuffer->length()));
    _pVertexDataBuffer->didModifyRange(NS::Range::Make(0, _pVertexDataBuffer->length()));
    _pIndexBuffer->didModifyRange(NS::Range::Make(0, _pIndexBuffer->length()));

    for (size_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        _pUniformBuffer[i] = mDevice->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeManaged);
    }
}

MTL::AccelerationStructure* MetalRender::createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    // Query for the sizes needed to store and build the acceleration structure.
    MTL::AccelerationStructureSizes accelSizes = mDevice->accelerationStructureSizes(descriptor);
    // Allocate an acceleration structure large enough for this descriptor. This doesn't actually
    // build the acceleration structure, it just allocates memory.
    MTL::AccelerationStructure* accelerationStructure =
        mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
    // Allocate scratch space Metal uses to build the acceleration structure.
    // Use MTLResourceStorageModePrivate for best performance because the sample
    // doesn't need access to the buffer's contents.
    MTL::Buffer* scratchBuffer = mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
    // Create a command buffer to perform the acceleration structure build.
    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    // Create an acceleration structure command encoder.
    MTL::AccelerationStructureCommandEncoder* commandEncoder = commandBuffer->accelerationStructureCommandEncoder();
    // Allocate a buffer for Metal to write the compacted accelerated structure's size into.
    MTL::Buffer* compactedSizeBuffer = mDevice->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    // Schedule the actual acceleration structure build.
    commandEncoder->buildAccelerationStructure(accelerationStructure, descriptor, scratchBuffer, 0UL);
    // Compute and write the compacted acceleration structure size into the buffer. You
    // need to already have a built accelerated structure because Metal determines the compacted
    // size based on the final size of the acceleration structure. Compacting an acceleration
    // structure can potentially reclaim significant amounts of memory because Metal must
    // create the initial structure using a conservative approach.
    commandEncoder->writeCompactedAccelerationStructureSize(accelerationStructure, compactedSizeBuffer, 0UL);
    // End encoding and commit the command buffer so the GPU can start building the
    // acceleration structure.
    commandEncoder->endEncoding();
    commandBuffer->commit();

    // The sample waits for Metal to finish executing the command buffer so that it can
    // read back the compacted size.

    // Note: Don't wait for Metal to finish executing the command buffer if you aren't compacting
    // the acceleration structure because doing so requires CPU/GPU synchronization. You don't have
    // to compact acceleration structures, but it's helpful when creating large static acceleration
    // structures, such as static scene geometry. Avoid compacting acceleration structures that
    // you rebuild every frame because the synchronization cost may be significant.

    commandBuffer->waitUntilCompleted();

    const uint32_t compactedSize = *(uint32_t*)compactedSizeBuffer->contents();

    // commandBuffer->release();
    // commandEncoder->release();

    // Allocate a smaller acceleration structure based on the returned size.
    MTL::AccelerationStructure* compactedAccelerationStructure = mDevice->newAccelerationStructure(compactedSize);

    // Create another command buffer and encoder.
    commandBuffer = mCommandQueue->commandBuffer();
    commandEncoder = commandBuffer->accelerationStructureCommandEncoder();

    // Encode the command to copy and compact the acceleration structure into the
    // smaller acceleration structure.
    commandEncoder->copyAndCompactAccelerationStructure(accelerationStructure, compactedAccelerationStructure);

    // End encoding and commit the command buffer. You don't need to wait for Metal to finish
    // executing this command buffer as long as you synchronize any ray-intersection work
    // to run after this command buffer completes. The sample relies on Metal's default
    // dependency tracking on resources to automatically synchronize access to the new
    // compacted acceleration structure.
    commandEncoder->endEncoding();
    commandBuffer->commit();

    // commandEncoder->release();
    // commandBuffer->release();
    accelerationStructure->release();
    scratchBuffer->release();
    compactedSizeBuffer->release();

    pPool->release();

    return compactedAccelerationStructure->retain();
}

constexpr simd_float4x4 makeIdentity()
{
    using simd::float4;
    return (simd_float4x4){ (float4){ 1.f, 0.f, 0.f, 0.f }, (float4){ 0.f, 1.f, 0.f, 0.f },
                            (float4){ 0.f, 0.f, 1.f, 0.f }, (float4){ 0.f, 0.f, 0.f, 1.f } };
}

MetalRender::Mesh* MetalRender::createMesh(const oka::Mesh& mesh)
{
    MetalRender::Mesh* result = new MetalRender::Mesh();

    const uint32_t triangleCount = mesh.mCount / 3;

    const std::vector<oka::Scene::Vertex>& vertices = mScene->getVertices();
    const std::vector<uint32_t>& indices = mScene->getIndices();

    std::vector<Triangle> triangleData(triangleCount);
    for (int i = 0; i < triangleCount; ++i)
    {
        Triangle& curr = triangleData[i];
        const uint32_t i0 = indices[mesh.mIndex + i * 3 + 0];
        const uint32_t i1 = indices[mesh.mIndex + i * 3 + 1];
        const uint32_t i2 = indices[mesh.mIndex + i * 3 + 2];
        // Positions
        using simd::float3;

        curr.positions[0] = { vertices[mesh.mVbOffset + i0].pos.x, vertices[mesh.mVbOffset + i0].pos.y,
                              vertices[mesh.mVbOffset + i0].pos.z };
        curr.positions[1] = { vertices[mesh.mVbOffset + i1].pos.x, vertices[mesh.mVbOffset + i1].pos.y,
                              vertices[mesh.mVbOffset + i1].pos.z };
        curr.positions[2] = { vertices[mesh.mVbOffset + i2].pos.x, vertices[mesh.mVbOffset + i2].pos.y,
                              vertices[mesh.mVbOffset + i2].pos.z };
        // Normals
        curr.normals[0] = vertices[mesh.mVbOffset + i0].normal;
        curr.normals[1] = vertices[mesh.mVbOffset + i1].normal;
        curr.normals[2] = vertices[mesh.mVbOffset + i2].normal;
        // Tangents
        curr.tangent[0] = vertices[mesh.mVbOffset + i0].tangent;
        curr.tangent[1] = vertices[mesh.mVbOffset + i1].tangent;
        curr.tangent[2] = vertices[mesh.mVbOffset + i2].tangent;
        // UVs
        curr.uv[0] = vertices[mesh.mVbOffset + i0].uv;
        curr.uv[1] = vertices[mesh.mVbOffset + i1].uv;
        curr.uv[2] = vertices[mesh.mVbOffset + i2].uv;
    }

    MTL::Buffer* perPrimitiveBuffer =
        mDevice->newBuffer(triangleData.size() * sizeof(Triangle), MTL::ResourceStorageModeManaged);

    memcpy(perPrimitiveBuffer->contents(), triangleData.data(), sizeof(Triangle) * triangleData.size());

    perPrimitiveBuffer->didModifyRange(NS::Range(0, perPrimitiveBuffer->length()));

    MTL::AccelerationStructureTriangleGeometryDescriptor* geomDescriptor =
        MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();

    geomDescriptor->setVertexBuffer(_pVertexDataBuffer);
    geomDescriptor->setVertexBufferOffset(mesh.mVbOffset * sizeof(oka::Scene::Vertex));
    geomDescriptor->setVertexStride(sizeof(oka::Scene::Vertex));
    geomDescriptor->setIndexBuffer(_pIndexBuffer);
    geomDescriptor->setIndexBufferOffset(mesh.mIndex * sizeof(uint32_t));
    geomDescriptor->setIndexType(MTL::IndexTypeUInt32);
    geomDescriptor->setTriangleCount(triangleCount);
    // Setup per primitive data
    geomDescriptor->setPrimitiveDataBuffer(perPrimitiveBuffer);
    geomDescriptor->setPrimitiveDataBufferOffset(0);
    geomDescriptor->setPrimitiveDataElementSize(sizeof(Triangle));
    geomDescriptor->setPrimitiveDataStride(sizeof(Triangle));

    const NS::Array* geomDescriptors = NS::Array::array((const NS::Object* const*)&geomDescriptor, 1UL);

    MTL::PrimitiveAccelerationStructureDescriptor* primDescriptor =
        MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    primDescriptor->setGeometryDescriptors(geomDescriptors);

    result->mGas = createAccelerationStructure(primDescriptor);

    primDescriptor->release();
    geomDescriptor->release();
    perPrimitiveBuffer->release();
    return result;
}

void MetalRender::createAccelerationStructures()
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
    const std::vector<oka::Curve>& curves = mScene->getCurves();
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    if (meshes.empty() && curves.empty())
    {
        return;
    }

    for (int i = 0; i < meshes.size(); ++i)
    {
        MetalRender::Mesh* m = createMesh(meshes[i]);
        mMetalMeshes.push_back(m);
        _primitiveAccelerationStructures.push_back(m->mGas);
    }

    _instanceBuffer = mDevice->newBuffer(
        sizeof(MTL::AccelerationStructureInstanceDescriptor) * instances.size(), MTL::ResourceStorageModeManaged);
    MTL::AccelerationStructureInstanceDescriptor* instanceDescriptors =
        (MTL::AccelerationStructureInstanceDescriptor*)_instanceBuffer->contents();
    for (int i = 0; i < instances.size(); ++i)
    {
        const oka::Instance& curr = instances[i];
        instanceDescriptors[i].accelerationStructureIndex = curr.mMeshId;
        instanceDescriptors[i].options = MTL::AccelerationStructureInstanceOptionOpaque;
        instanceDescriptors[i].intersectionFunctionTableOffset = 0;
        instanceDescriptors[i].mask =
            curr.type == oka::Instance::Type::eLight ? GEOMETRY_MASK_LIGHT : GEOMETRY_MASK_TRIANGLE;

        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                instanceDescriptors[i].transformationMatrix.columns[column][row] = curr.transform[column][row];
            }
        }
    }

    simd_float4x4 identity = makeIdentity();

    _instanceBuffer->didModifyRange(NS::Range::Make(0, _instanceBuffer->length()));

    const NS::Array* instancedAccelerationStructures = NS::Array::array(
        (const NS::Object* const*)_primitiveAccelerationStructures.data(), _primitiveAccelerationStructures.size());
    MTL::InstanceAccelerationStructureDescriptor* accelDescriptor =
        MTL::InstanceAccelerationStructureDescriptor::descriptor();
    accelDescriptor->setInstancedAccelerationStructures(instancedAccelerationStructures);
    accelDescriptor->setInstanceCount(instances.size());
    accelDescriptor->setInstanceDescriptorBuffer(_instanceBuffer);

    _instanceAccelerationStructure = createAccelerationStructure(accelDescriptor);
    pPool->release();
}
