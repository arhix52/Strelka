#include "Metal4Context.h"

#include <log.h>

namespace oka
{

// Generous on purpose: the longest immediate submission is a single BLAS build,
// and the largest structure in a heavy scene takes well under a second.
static constexpr uint32_t kImmediateTimeoutMs = 5000;

// --- ConstantRing ----------------------------------------------------------

bool ConstantRing::init(MTL::Device* device, size_t bytesPerPage, uint32_t frameCount, PageCallback onPage)
{
    release();
    mDevice = device;
    mCapacity = bytesPerPage;
    mOnPage = std::move(onPage);
    mPages.resize(frameCount);
    for (std::vector<MTL::Buffer*>& pages : mPages)
    {
        MTL::Buffer* page = mDevice->newBuffer(mCapacity, MTL::ResourceStorageModeShared);
        if (!page)
        {
            release();
            return false;
        }
        pages.push_back(page);
        if (mOnPage)
        {
            mOnPage(page);
        }
    }
    return true;
}

void ConstantRing::release()
{
    for (std::vector<MTL::Buffer*>& pages : mPages)
    {
        for (MTL::Buffer* page : pages)
        {
            page->release();
        }
    }
    mPages.clear();
    mOnPage = nullptr;
    mCapacity = 0;
    mOffset = 0;
    mPage = 0;
}

void ConstantRing::beginFrame(uint32_t frameIndex)
{
    mFrame = frameIndex % (uint32_t)std::max<size_t>(mPages.size(), 1);
    mOffset = 0;
    mPage = 0;
}

MTL::GPUAddress ConstantRing::push(const void* data, size_t size)
{
    if (mPages.empty() || size > mCapacity)
    {
        STRELKA_ERROR("Metal 4 constant of {} bytes does not fit a {} byte page", size, mCapacity);
        return 0;
    }
    // Metal wants argument addresses aligned; 16 covers every scalar and vector
    // type a kernel can take as a constant.
    constexpr size_t kAlign = 16;
    size_t offset = (mOffset + kAlign - 1) & ~(kAlign - 1);
    std::vector<MTL::Buffer*>& pages = mPages[mFrame];
    if (offset + size > mCapacity)
    {
        // Pages already allocated for this frame are reused; only the first frame
        // that needs a deeper chain pays for it.
        if (mPage + 1 >= pages.size())
        {
            MTL::Buffer* page = mDevice->newBuffer(mCapacity, MTL::ResourceStorageModeShared);
            if (!page)
            {
                STRELKA_ERROR("Metal 4 constant ring could not grow past {} pages", pages.size());
                return 0;
            }
            pages.push_back(page);
            if (mOnPage)
            {
                mOnPage(page);
            }
        }
        ++mPage;
        offset = 0;
    }
    MTL::Buffer* buffer = pages[mPage];
    std::memcpy(static_cast<uint8_t*>(buffer->contents()) + offset, data, size);
    mOffset = offset + size;
    return buffer->gpuAddress() + offset;
}

// --- Metal4Context ---------------------------------------------------------

bool Metal4Context::init(MTL::Device* device, uint32_t frameCount, size_t constantBytesPerFrame)
{
    mDevice = device;

    NS::Error* error = nullptr;
    // The queue has to be created with a descriptor naming a feedback queue.
    // Without one, commit feedback handlers are never delivered -- and this
    // renderer's asynchronous loop clears its in-flight flag from exactly that
    // handler, so the editor rendered one frame, never learned it had finished,
    // and showed a black viewport forever. The headless path did not notice
    // because it waits on the queue's shared event instead.
    MTL4::CommandQueueDescriptor* queueDesc = MTL4::CommandQueueDescriptor::alloc()->init();
    mFeedbackQueue = dispatch_queue_create("com.strelka.mtl4.feedback", DISPATCH_QUEUE_SERIAL);
    // A reference of our own, because the descriptor's ownership of this is
    // one-sided. Measured, since the header says nothing: setFeedbackQueue does
    // not retain and neither does newMTL4CommandQueue, but releasing the
    // descriptor *does* release the queue. So the create's +1 is consumed by
    // queueDesc->release() below, leaving the command queue -- which is still
    // running -- pointed at a deallocated dispatch queue, and leaving our own
    // dispatch_release in release() to trap as an over-release. It did: every
    // teardown of a Metal 4 renderer aborted in libdispatch.
    dispatch_retain(mFeedbackQueue);
    queueDesc->setFeedbackQueue(mFeedbackQueue);
    mQueue = device->newMTL4CommandQueue(queueDesc, &error);
    queueDesc->release();
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

    // One argument table shared by every stage, so it has to be as wide as the
    // highest index any of them binds: buffer(28) is the shade stage's IOR stats
    // and texture(8) the reactive guide (see wavefront.metal). A bind past the
    // declared count is a hard failure, and only the debug layer says so -- these
    // counts were left at the old maxima when the curve, IOR-stats and reactive
    // bindings were added, and without MTL_DEBUG_LAYER the writes simply went
    // past the end of the table.
    MTL4::ArgumentTableDescriptor* tableDesc = MTL4::ArgumentTableDescriptor::alloc()->init();
    tableDesc->setMaxBufferBindCount(29);
    tableDesc->setMaxTextureBindCount(9);
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

    mImmediateAllocator = device->newCommandAllocator();
    mImmediateBuffer = device->newCommandBuffer();
    // Metal 4 has no waitUntilCompleted; a shared event signalled by the queue is
    // how a submission is waited on.
    mImmediateEvent = device->newSharedEvent();
    mFrameEvent = device->newSharedEvent();
    if (!mImmediateAllocator || !mImmediateBuffer || !mImmediateEvent || !mFrameEvent)
    {
        STRELKA_ERROR("Metal 4 immediate submission objects failed");
        release();
        return false;
    }

    // Pages can also be added mid-frame when a launch needs more than one, and
    // the frame's pre-submit commitResidency() publishes those.
    ConstantRing::PageCallback residency = [this](MTL::Buffer* page) { addResident(page); };
    if (!mConstants.init(device, constantBytesPerFrame, frameCount, residency) ||
        !mImmediateConstants.init(device, constantBytesPerFrame, 1, residency))
    {
        STRELKA_ERROR("Metal 4 constant ring allocation failed");
        release();
        return false;
    }
    commitResidency();

    STRELKA_INFO("Metal 4 submission layer ready: {} frames in flight, {} KB of constants per frame",
                 frameCount, constantBytesPerFrame / 1024);
    return true;
}

MTL4::CommandBuffer* Metal4Context::beginImmediate()
{
    if (!mImmediateBuffer)
    {
        return nullptr;
    }
    mImmediateAllocator->reset();
    // submitAndWait() blocks, so one buffer's worth is always safe to reuse.
    mImmediateConstants.beginFrame(0);
    mImmediateBuffer->beginCommandBuffer(mImmediateAllocator);
    mImmediateBuffer->useResidencySet(mResidencySet);
    return mImmediateBuffer;
}

void Metal4Context::submitAndWait(MTL4::CommandBuffer* commandBuffer)
{
    if (!commandBuffer || !mQueue)
    {
        return;
    }
    // Encoding may allocate scratch or destination resources. Residency only
    // has to be committed before queue submission, not before encoding.
    commitResidency();
    commandBuffer->endCommandBuffer();
    const MTL4::CommandBuffer* buffers[] = { commandBuffer };
    MTL4::CommitOptions* options = MTL4::CommitOptions::alloc()->init();
    options->addFeedbackHandler(MTL4::CommitFeedbackHandlerFunction([](MTL4::CommitFeedback* feedback) {
        NS::Error* error = feedback ? feedback->error() : nullptr;
        if (error)
        {
            STRELKA_ERROR("Metal 4 immediate submission failed: {}",
                          error->localizedDescription() ? error->localizedDescription()->utf8String()
                                                       : "unknown error");
        }
    }));
    mQueue->commit(buffers, 1, options);
    options->release();
    mQueue->signalEvent(mImmediateEvent, ++mImmediateValue);
    // Not advisory: the next beginImmediate() resets the allocator this command
    // buffer was written into, so continuing past a timeout hands the GPU
    // commands that are being overwritten. Say so rather than corrupt silently.
    if (!mImmediateEvent->waitUntilSignaledValue(mImmediateValue, kImmediateTimeoutMs))
    {
        STRELKA_ERROR("Metal 4 immediate submission did not complete within {} ms; the GPU is still reading a "
                      "command buffer that is about to be reused",
                      kImmediateTimeoutMs);
    }
}

uint64_t Metal4Context::signalFrame()
{
    if (!mQueue || !mFrameEvent)
    {
        return 0;
    }
    mQueue->signalEvent(mFrameEvent, ++mFrameValue);
    return mFrameValue;
}

bool Metal4Context::waitForFrame(uint64_t value, uint32_t timeoutMs)
{
    if (!mFrameEvent || value == 0)
    {
        return false;
    }
    return mFrameEvent->waitUntilSignaledValue(value, timeoutMs);
}

void Metal4Context::release()
{
    if (mImmediateEvent)
    {
        mImmediateEvent->release();
        mImmediateEvent = nullptr;
    }
    if (mFrameEvent)
    {
        mFrameEvent->release();
        mFrameEvent = nullptr;
    }
    if (mImmediateBuffer)
    {
        mImmediateBuffer->release();
        mImmediateBuffer = nullptr;
    }
    if (mImmediateAllocator)
    {
        mImmediateAllocator->release();
        mImmediateAllocator = nullptr;
    }
    mConstants.release();
    mImmediateConstants.release();
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
    // Last, and after the queue that delivers feedback on it: our own reference
    // from init(), see the note there about who does and does not retain this.
    if (mFeedbackQueue)
    {
        dispatch_release(mFeedbackQueue);
        mFeedbackQueue = nullptr;
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

void Metal4Context::removeResident(MTL::Allocation* allocation)
{
    if (!mResidencySet || !allocation)
    {
        return;
    }
    mResidencySet->removeAllocation(allocation);
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

MTL::ComputePipelineState* Metal4Context::newComputePipelineStateLinked(
    MTL::Library* library, const char* functionName, const char* linkedFunctionName,
    MTL::FunctionConstantValues* constants)
{
    if (!mCompiler || !library)
    {
        return nullptr;
    }
    NS::Error* error = nullptr;

    auto describe = [&](const char* name) -> NS::Object* {
        MTL4::LibraryFunctionDescriptor* fd = MTL4::LibraryFunctionDescriptor::alloc()->init();
        fd->setLibrary(library);
        fd->setName(NS::String::string(name, NS::UTF8StringEncoding));
        if (!constants)
        {
            return fd;
        }
        // The intersection function reads the same function constants as its
        // caller -- SPEC_ALPHA among them -- so it has to be specialised with
        // the identical set, or the two disagree about what the scene contains.
        MTL4::SpecializedFunctionDescriptor* sd = MTL4::SpecializedFunctionDescriptor::alloc()->init();
        sd->setFunctionDescriptor(fd);
        sd->setConstantValues(constants);
        fd->release();
        return sd;
    };

    NS::Object* computeDesc = describe(functionName);
    NS::Object* linkedDesc = describe(linkedFunctionName);

    MTL4::ComputePipelineDescriptor* pipelineDesc = MTL4::ComputePipelineDescriptor::alloc()->init();
    pipelineDesc->setComputeFunctionDescriptor(static_cast<MTL4::FunctionDescriptor*>(computeDesc));

    const NS::Object* fns[] = { linkedDesc };
    MTL4::StaticLinkingDescriptor* linking = MTL4::StaticLinkingDescriptor::alloc()->init();
    linking->setFunctionDescriptors(NS::Array::array(fns, 1));
    pipelineDesc->setStaticLinkingDescriptor(linking);

    MTL::ComputePipelineState* pipeline = mCompiler->newComputePipelineState(pipelineDesc, nullptr, &error);
    if (!pipeline)
    {
        STRELKA_ERROR("Metal 4 pipeline {} (linking {}): {}", functionName, linkedFunctionName,
                      error ? error->localizedDescription()->utf8String() : "unknown error");
    }

    linking->release();
    pipelineDesc->release();
    computeDesc->release();
    linkedDesc->release();
    return pipeline;
}

} // namespace oka
