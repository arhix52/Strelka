#include "Metal4Context.h"

#include <log.h>

namespace oka
{

// --- ConstantRing ----------------------------------------------------------

bool ConstantRing::init(MTL::Device* device, size_t bytesPerFrame, uint32_t frameCount)
{
    release();
    mCapacity = bytesPerFrame;
    mBuffers.reserve(frameCount);
    for (uint32_t i = 0; i < frameCount; ++i)
    {
        MTL::Buffer* buffer = device->newBuffer(bytesPerFrame, MTL::ResourceStorageModeShared);
        if (!buffer)
        {
            release();
            return false;
        }
        mBuffers.push_back(buffer);
    }
    return true;
}

void ConstantRing::release()
{
    for (MTL::Buffer* buffer : mBuffers)
    {
        buffer->release();
    }
    mBuffers.clear();
    mCapacity = 0;
    mOffset = 0;
}

void ConstantRing::beginFrame(uint32_t frameIndex)
{
    mFrame = frameIndex % (uint32_t)std::max<size_t>(mBuffers.size(), 1);
    mOffset = 0;
}

MTL::GPUAddress ConstantRing::push(const void* data, size_t size)
{
    if (mBuffers.empty())
    {
        return 0;
    }
    // Metal wants argument addresses aligned; 16 covers every scalar and vector
    // type a kernel can take as a constant.
    constexpr size_t kAlign = 16;
    const size_t offset = (mOffset + kAlign - 1) & ~(kAlign - 1);
    if (offset + size > mCapacity)
    {
        STRELKA_ERROR("Metal 4 constant ring exhausted: {} + {} > {}", offset, size, mCapacity);
        return 0;
    }
    MTL::Buffer* buffer = mBuffers[mFrame];
    std::memcpy(static_cast<uint8_t*>(buffer->contents()) + offset, data, size);
    mOffset = offset + size;
    return buffer->gpuAddress() + offset;
}

// --- Metal4Context ---------------------------------------------------------

bool Metal4Context::init(MTL::Device* device, uint32_t frameCount, size_t constantBytesPerFrame)
{
    mDevice = device;

    NS::Error* error = nullptr;
    mQueue = device->newMTL4CommandQueue();
    if (!mQueue)
    {
        STRELKA_WARNING("Metal 4 unavailable on this device; keeping the Metal 3 path");
        return false;
    }

    MTL4::CompilerDescriptor* compilerDesc = MTL4::CompilerDescriptor::alloc()->init();
    mCompiler = device->newCompiler(compilerDesc, &error);
    compilerDesc->release();
    if (!mCompiler)
    {
        STRELKA_ERROR("Metal 4 compiler: {}",
                      error ? error->localizedDescription()->utf8String() : "unknown error");
        release();
        return false;
    }

    // One argument table shared by every stage. Sized to the largest binding
    // index any kernel uses, with headroom: an index past the end is a hard
    // failure at encode time, not a warning.
    MTL4::ArgumentTableDescriptor* tableDesc = MTL4::ArgumentTableDescriptor::alloc()->init();
    tableDesc->setMaxBufferBindCount(32);
    tableDesc->setMaxTextureBindCount(8);
    mArgumentTable = device->newArgumentTable(tableDesc, &error);
    tableDesc->release();
    if (!mArgumentTable)
    {
        STRELKA_ERROR("Metal 4 argument table: {}",
                      error ? error->localizedDescription()->utf8String() : "unknown error");
        release();
        return false;
    }

    // Residency is a queue-wide declaration in Metal 4, replacing the
    // per-encoder useResource calls the Metal 3 path makes on every band.
    MTL::ResidencySetDescriptor* residencyDesc = MTL::ResidencySetDescriptor::alloc()->init();
    residencyDesc->setInitialCapacity(256);
    mResidencySet = device->newResidencySet(residencyDesc, &error);
    residencyDesc->release();
    if (!mResidencySet)
    {
        STRELKA_ERROR("Metal 4 residency set: {}",
                      error ? error->localizedDescription()->utf8String() : "unknown error");
        release();
        return false;
    }
    mQueue->addResidencySet(mResidencySet);

    // An allocator owns the memory its command buffer is written into, so it can
    // only be reset once that work has finished on the GPU -- hence one per frame
    // in flight rather than one shared.
    mAllocators.reserve(frameCount);
    mCommandBuffers.reserve(frameCount);
    for (uint32_t i = 0; i < frameCount; ++i)
    {
        MTL4::CommandAllocator* allocator = device->newCommandAllocator();
        MTL4::CommandBuffer* commandBuffer = device->newCommandBuffer();
        if (!allocator || !commandBuffer)
        {
            STRELKA_ERROR("Metal 4 command allocator/buffer creation failed");
            release();
            return false;
        }
        mAllocators.push_back(allocator);
        mCommandBuffers.push_back(commandBuffer);
    }

    if (!mConstants.init(device, constantBytesPerFrame, frameCount))
    {
        STRELKA_ERROR("Metal 4 constant ring allocation failed");
        release();
        return false;
    }
    for (MTL::Buffer* buffer : mConstants.buffers())
    {
        addResident(buffer);
    }
    commitResidency();

    STRELKA_INFO("Metal 4 submission layer ready: {} frames in flight, {} KB of constants per frame",
                 frameCount, constantBytesPerFrame / 1024);
    return true;
}

void Metal4Context::release()
{
    mConstants.release();
    for (MTL4::CommandBuffer* commandBuffer : mCommandBuffers)
    {
        commandBuffer->release();
    }
    mCommandBuffers.clear();
    for (MTL4::CommandAllocator* allocator : mAllocators)
    {
        allocator->release();
    }
    mAllocators.clear();
    if (mResidencySet)
    {
        if (mQueue)
        {
            mQueue->removeResidencySet(mResidencySet);
        }
        mResidencySet->release();
        mResidencySet = nullptr;
    }
    if (mArgumentTable)
    {
        mArgumentTable->release();
        mArgumentTable = nullptr;
    }
    if (mCompiler)
    {
        mCompiler->release();
        mCompiler = nullptr;
    }
    if (mQueue)
    {
        mQueue->release();
        mQueue = nullptr;
    }
}

MTL4::CommandBuffer* Metal4Context::beginFrame(uint32_t frameIndex)
{
    if (mAllocators.empty())
    {
        return nullptr;
    }
    const uint32_t slot = frameIndex % (uint32_t)mAllocators.size();
    mConstants.beginFrame(slot);

    MTL4::CommandAllocator* allocator = mAllocators[slot];
    allocator->reset();

    MTL4::CommandBuffer* commandBuffer = mCommandBuffers[slot];
    commandBuffer->beginCommandBuffer(allocator);
    commandBuffer->useResidencySet(mResidencySet);
    return commandBuffer;
}

void Metal4Context::addResident(MTL::Allocation* allocation)
{
    if (!mResidencySet || !allocation)
    {
        return;
    }
    mResidencySet->addAllocation(allocation);
    mResidencyDirty = true;
}

void Metal4Context::commitResidency()
{
    if (!mResidencySet || !mResidencyDirty)
    {
        return;
    }
    mResidencySet->commit();
    mResidencySet->requestResidency();
    mResidencyDirty = false;
}

MTL::ComputePipelineState* Metal4Context::newComputePipelineState(MTL::Library* library,
                                                                  const char* functionName,
                                                                  MTL::FunctionConstantValues* constants)
{
    if (!mCompiler || !library)
    {
        return nullptr;
    }
    NS::Error* error = nullptr;

    MTL4::LibraryFunctionDescriptor* functionDesc = MTL4::LibraryFunctionDescriptor::alloc()->init();
    functionDesc->setLibrary(library);
    functionDesc->setName(NS::String::string(functionName, NS::UTF8StringEncoding));

    MTL4::ComputePipelineDescriptor* pipelineDesc = MTL4::ComputePipelineDescriptor::alloc()->init();
    // Function constants arrive wrapped in a specialising descriptor rather than
    // as an argument to the pipeline call, as they were in Metal 3.
    MTL4::SpecializedFunctionDescriptor* specialized = nullptr;
    if (constants)
    {
        specialized = MTL4::SpecializedFunctionDescriptor::alloc()->init();
        specialized->setFunctionDescriptor(functionDesc);
        specialized->setConstantValues(constants);
        pipelineDesc->setComputeFunctionDescriptor(specialized);
    }
    else
    {
        pipelineDesc->setComputeFunctionDescriptor(functionDesc);
    }

    MTL::ComputePipelineState* pipeline = mCompiler->newComputePipelineState(pipelineDesc, nullptr, &error);
    if (!pipeline)
    {
        STRELKA_ERROR("Metal 4 pipeline {}: {}", functionName,
                      error ? error->localizedDescription()->utf8String() : "unknown error");
    }

    if (specialized)
    {
        specialized->release();
    }
    pipelineDesc->release();
    functionDesc->release();
    return pipeline;
}

} // namespace oka
