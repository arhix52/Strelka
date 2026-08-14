#pragma once

// Metal 4 submission layer.
//
// Metal 4 replaces four things the Metal 3 path relied on, and each replacement
// is a constraint rather than a convenience:
//
//   * There is no setBytes. Every constant reaches a shader through a GPU
//     address, so small per-dispatch values need a buffer to live in --
//     ConstantRing below.
//   * There is no automatic hazard tracking. Dependencies between dispatches are
//     stated with explicit barriers; forgetting one is a silent data race, not a
//     validation error.
//   * Bindings go through an argument table instead of setBuffer, and the table
//     holds raw addresses, so the caller owns resource lifetime.
//   * Residency is declared once for the whole queue with a residency set rather
//     than per encoder with useResource.
//
// Command allocators own the memory an encoder writes into, so one per frame in
// flight, reset only once that frame's work has actually completed.

#include <Metal/Metal.hpp>
#include <dispatch/dispatch.h>

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <functional>
#include <vector>

namespace oka
{

/// Per-frame bump allocator standing in for setBytes.
///
/// Shared storage, written by the CPU and read by the GPU in the same frame; the
/// caller must not reuse a frame's ring until that frame's commit feedback has
/// fired, which is the same rule the command allocators follow.
///
/// How much one frame needs is not bounded by the frame: a launch carrying many
/// samples encodes the whole wavefront loop that many times over, and each stage
/// of each iteration pushes its own constants. So a frame's storage is a chain of
/// equally sized pages that grows on demand rather than one fixed buffer.
class ConstantRing
{
public:
    /// Called for every page allocated, at init and on growth, so the owner can
    /// declare it resident. A page the residency set does not name is a GPU fault
    /// at the dispatch that reads it.
    using PageCallback = std::function<void(MTL::Buffer*)>;

    bool init(MTL::Device* device, size_t bytesPerPage, uint32_t frameCount, PageCallback onPage);
    void release();

    /// Start of a frame: hand the ring back to the beginning.
    void beginFrame(uint32_t frameIndex);

    /// Copy `size` bytes in and return the GPU address they landed at. Returns 0
    /// only when `size` exceeds a whole page or the allocation fails, both of
    /// which the caller must treat as a bug rather than recover from.
    MTL::GPUAddress push(const void* data, size_t size);

    template <typename T>
    MTL::GPUAddress push(const T& value)
    {
        return push(&value, sizeof(T));
    }

private:
    MTL::Device* mDevice = nullptr;
    /// Pages per frame in flight. Kept per frame rather than shared so growth
    /// never touches storage another frame is still reading.
    std::vector<std::vector<MTL::Buffer*>> mPages;
    PageCallback mOnPage;
    size_t mCapacity = 0;
    size_t mOffset = 0;
    uint32_t mFrame = 0;
    uint32_t mPage = 0;
};

/// Owns the Metal 4 objects and nothing else: it is deliberately separable from
/// MetalRender so the migration can be reviewed, measured and reverted on its
/// own.
class Metal4Context
{
public:
    /// Returns false when the device or OS predates Metal 4, in which case the
    /// caller keeps using the Metal 3 path.
    bool init(MTL::Device* device, uint32_t frameCount, size_t constantBytesPerFrame);
    void release();

    bool isValid() const
    {
        return mQueue != nullptr;
    }

    MTL4::CommandQueue* queue() const
    {
        return mQueue;
    }
    MTL4::Compiler* compiler() const
    {
        return mCompiler;
    }
    MTL4::ArgumentTable* argumentTable() const
    {
        return mArgumentTable;
    }
    /// The frame's ring, reset by beginFrame().
    ConstantRing& constants()
    {
        return mConstants;
    }

    /// A ring of its own for work outside the frame loop, reset by
    /// beginImmediate(). Sharing the frame's would either run it dry -- skinning
    /// pushes two constants per mesh per frame and never resets -- or overwrite
    /// constants a frame still in flight is reading.
    ConstantRing& immediateConstants()
    {
        return mImmediateConstants;
    }

    /// Begin recording into the given frame's allocator. The allocator is reset
    /// here, so the frame's previous work must already have completed.
    MTL4::CommandBuffer* beginFrame(uint32_t frameIndex);

    /// One-off work outside the frame loop -- scene load and acceleration
    /// structure rebuilds. Per-frame skinning and refits use beginFrame() so
    /// they remain pipelined with tracing.
    MTL4::CommandBuffer* beginImmediate();
    void submitAndWait(MTL4::CommandBuffer* commandBuffer);

    /// Per-frame skinning on its own allocator ring, committed without blocking
    /// the CPU.
    ///
    /// The alternative -- beginImmediate() + submitAndWait() -- costs a full
    /// submit-to-completion round trip in the middle of every animated frame
    /// (measured at ~19 ms median on BrainStem), and leaves the GPU with nothing
    /// queued for the duration. Consumers order behind this work by waiting on
    /// skinEvent() at the value returned here, on whichever queue they use.
    ///
    /// The ring must be at least as deep as the frames the renderer keeps in
    /// flight: beginSkin() resets the allocator for its slot, so a shallower ring
    /// would overwrite commands the GPU is still reading.
    MTL4::CommandBuffer* beginSkin(uint32_t frameIndex);
    ConstantRing& skinConstants()
    {
        return mSkinConstants;
    }
    /// Commit without waiting. Returns the skinEvent() value that, once reached,
    /// means the skinned vertices are visible to any other queue.
    uint64_t submitSkin(MTL4::CommandBuffer* commandBuffer);
    MTL::SharedEvent* skinEvent() const
    {
        return mSkinEvent;
    }

    /// Insert a wait/signal on the Metal 4 queue timeline (cross-queue sync).
    void wait(MTL::SharedEvent* event, uint64_t value);
    void signal(MTL::SharedEvent* event, uint64_t value);

    /// Frame-loop counterpart of submitAndWait's tail, split in two so the
    /// caller can commit, do other work, and block later -- which is what an
    /// interactive loop wants and a headless one does not.
    ///
    /// Metal 4 answers a committed command buffer through a commit feedback
    /// handler, never through the buffer itself, so there is no
    /// waitUntilCompleted to call. A queue-signalled shared event is the only
    /// thing a caller can block on.
    uint64_t signalFrame();
    /// The event signalFrame() signals, for a consumer on another queue to wait on.
    MTL::SharedEvent* frameEvent() const
    {
        return mFrameEvent;
    }
    bool waitForFrame(uint64_t value, uint32_t timeoutMs = 5000);

    /// Declare a resource resident for as long as it exists. Cheap to call
    /// repeatedly; commitResidency() must follow before the next submit.
    void addResident(MTL::Allocation* allocation);
    void removeResident(MTL::Allocation* allocation);
    void commitResidency();
    NS::UInteger residencyAllocationCount() const;
    uint64_t residencyAllocatedSize() const;

    /// Build a pipeline through the Metal 4 compiler. Pipelines built the
    /// Metal 3 way are not usable with an argument table.
    MTL::ComputePipelineState* newComputePipelineState(MTL::Library* library,
                                                       const char* functionName,
                                                       MTL::FunctionConstantValues* constants);

    /// Same, with an intersection function statically linked in, so the pipeline
    /// can hand out a table to bind it through. Metal 3 states this with
    /// MTLLinkedFunctions on the pipeline descriptor; Metal 4 replaces that with
    /// a StaticLinkingDescriptor carrying function *descriptors*, which is why
    /// this cannot just take the MTL::Function the other path builds.
    MTL::ComputePipelineState* newComputePipelineStateLinked(MTL::Library* library,
                                                             const char* functionName,
                                                             const char* linkedFunctionName,
                                                             MTL::FunctionConstantValues* constants);

private:
    MTL::Device* mDevice = nullptr;
    MTL4::CommandQueue* mQueue = nullptr;
    // Commit feedback is delivered here; a queue built without one delivers none.
    dispatch_queue_t mFeedbackQueue = nullptr;
    MTL4::Compiler* mCompiler = nullptr;
    MTL4::ArgumentTable* mArgumentTable = nullptr;
    MTL::ResidencySet* mResidencySet = nullptr;
    std::vector<MTL4::CommandAllocator*> mAllocators;
    std::vector<MTL4::CommandBuffer*> mCommandBuffers;
    MTL4::CommandAllocator* mImmediateAllocator = nullptr;
    MTL4::CommandBuffer* mImmediateBuffer = nullptr;
    MTL::SharedEvent* mImmediateEvent = nullptr;
    uint64_t mImmediateValue = 0;
    std::vector<MTL4::CommandAllocator*> mSkinAllocators;
    std::vector<MTL4::CommandBuffer*> mSkinBuffers;
    MTL::SharedEvent* mSkinEvent = nullptr;
    uint64_t mSkinValue = 0;
    ConstantRing mSkinConstants;
    MTL::SharedEvent* mFrameEvent = nullptr;
    uint64_t mFrameValue = 0;
    ConstantRing mConstants;
    ConstantRing mImmediateConstants;
    bool mResidencyDirty = false;
};

} // namespace oka
