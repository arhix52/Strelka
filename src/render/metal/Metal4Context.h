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
// Command allocators own the memory an encoder writes into, so every in-flight
// command buffer needs distinct allocator storage until its work completes.

#include <Metal/Metal.hpp>
#include <dispatch/dispatch.h>

#include <cstdint>
#include <cstring>
#include <string>
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

/// Width of the one argument table every Metal 4 stage shares. Counts, not
/// indices: the highest binding in wavefront.metal is buffer(30), the shade
/// stage's medium path state, and texture(8), the reactive guide. Raise these
/// with the shader, in the same commit -- a bind past the declared count writes
/// past the end of the table, and only MTL_DEBUG_LAYER=1 will tell you.
inline constexpr uint32_t kMetal4BufferBindCount = 31;
inline constexpr uint32_t kMetal4TextureBindCount = 9;

/// Names a Metal 4 command buffer or encoder, for Xcode captures and the debug
/// layer. A frame here is a dozen command buffers whose only distinguishing
/// feature is which bounce they carry, and a capture of unnamed ones is
/// unreadable.
///
/// Not a fix for MTL_SHADER_VALIDATION: labelling every encoder was tried for
/// that and does not help. MetalTools still dereferences a null label while
/// decoding a GPU error report (resolvedSharedPacketData, just before
/// setEncoderLabel:), so validation runs still die in the completion handler
/// before printing what they found. Stepping the program counter past that one
/// instruction in lldb is what makes the report readable.
void labelMetal4(MTL4::CommandBuffer* buffer, const std::string& name);
void labelMetal4(MTL4::CommandEncoder* encoder, const std::string& name);

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
    /// Schedule work on the serial feedback queue after the currently running
    /// feedback handler has returned. A commit made recursively from its own
    /// feedback callback can remain one continuously resident scheduler
    /// workload even when every command buffer is submitted separately.
    void afterFeedback(std::function<void()> work);
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

    /// Begin recording into the given frame's allocator. The allocator is reset
    /// here, so the frame's previous work must already have completed.
    MTL4::CommandBuffer* beginFrame(uint32_t frameIndex);
    /// Continue the same frame in another command buffer with its own allocator.
    /// Allocator storage remains live until the GPU finishes that chunk.
    MTL4::CommandBuffer* continueFrame(uint32_t frameIndex, uint32_t continuationIndex);

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

    /// Frame-loop counterpart of submitAndWait's tail, split in two so the
    /// caller can commit, do other work, and block later -- which is what an
    /// interactive loop wants and a headless one does not.
    ///
    /// Metal 4 answers a committed command buffer through a commit feedback
    /// handler, never through the buffer itself, so there is no
    /// waitUntilCompleted to call. A queue-signalled shared event is the only
    /// thing a caller can block on.
    uint64_t reserveFrameSignal();
    void signalFrame(uint64_t value);
    /// The event signalFrame(value) signals, for a consumer on another queue to wait on.
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

private:
    struct FrameContinuation
    {
        MTL4::CommandAllocator* allocator = nullptr;
        MTL4::CommandBuffer* commandBuffer = nullptr;
    };

    MTL::Device* mDevice = nullptr;
    MTL4::CommandQueue* mQueue = nullptr;
    // Commit feedback is delivered here; a queue built without one delivers none.
    dispatch_queue_t mFeedbackQueue = nullptr;
    MTL4::Compiler* mCompiler = nullptr;
    MTL4::ArgumentTable* mArgumentTable = nullptr;
    MTL::ResidencySet* mResidencySet = nullptr;
    std::vector<MTL4::CommandAllocator*> mAllocators;
    std::vector<MTL4::CommandBuffer*> mCommandBuffers;
    std::vector<std::vector<FrameContinuation>> mContinuationBuffers;
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
    bool mResidencyDirty = false;
};

} // namespace oka
