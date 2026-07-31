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

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

namespace oka
{

/// Per-frame bump allocator standing in for setBytes.
///
/// Shared storage, written by the CPU and read by the GPU in the same frame; the
/// caller must not reuse a frame's ring until that frame's commit feedback has
/// fired, which is the same rule the command allocators follow.
class ConstantRing
{
public:
    bool init(MTL::Device* device, size_t bytesPerFrame, uint32_t frameCount);
    void release();

    /// Start of a frame: hand the ring back to the beginning.
    void beginFrame(uint32_t frameIndex);

    /// Copy `size` bytes in and return the GPU address they landed at.
    /// Returns 0 if the frame's ring is exhausted, which the caller must treat
    /// as a bug rather than a condition to recover from.
    MTL::GPUAddress push(const void* data, size_t size);

    template <typename T>
    MTL::GPUAddress push(const T& value)
    {
        return push(&value, sizeof(T));
    }

    /// Every buffer, so they can be added to the residency set.
    const std::vector<MTL::Buffer*>& buffers() const
    {
        return mBuffers;
    }

private:
    std::vector<MTL::Buffer*> mBuffers;
    size_t mCapacity = 0;
    size_t mOffset = 0;
    uint32_t mFrame = 0;
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
    ConstantRing& constants()
    {
        return mConstants;
    }

    /// Begin recording into the given frame's allocator. The allocator is reset
    /// here, so the frame's previous work must already have completed.
    MTL4::CommandBuffer* beginFrame(uint32_t frameIndex);

    /// One-off work outside the frame loop -- scene load, acceleration structure
    /// builds, skinning. Uses an allocator of its own so it cannot collide with a
    /// frame still in flight, and submitAndWait() blocks until it is done, which
    /// is what every caller of this needs anyway.
    MTL4::CommandBuffer* beginImmediate();
    void submitAndWait(MTL4::CommandBuffer* commandBuffer);

    /// Declare a resource resident for as long as it exists. Cheap to call
    /// repeatedly; commitResidency() must follow before the next submit.
    void addResident(MTL::Allocation* allocation);
    void commitResidency();

    /// Build a pipeline through the Metal 4 compiler. Pipelines built the
    /// Metal 3 way are not usable with an argument table.
    MTL::ComputePipelineState* newComputePipelineState(MTL::Library* library,
                                                       const char* functionName,
                                                       MTL::FunctionConstantValues* constants);

private:
    MTL::Device* mDevice = nullptr;
    MTL4::CommandQueue* mQueue = nullptr;
    MTL4::Compiler* mCompiler = nullptr;
    MTL4::ArgumentTable* mArgumentTable = nullptr;
    MTL::ResidencySet* mResidencySet = nullptr;
    std::vector<MTL4::CommandAllocator*> mAllocators;
    std::vector<MTL4::CommandBuffer*> mCommandBuffers;
    MTL4::CommandAllocator* mImmediateAllocator = nullptr;
    MTL4::CommandBuffer* mImmediateBuffer = nullptr;
    MTL::SharedEvent* mImmediateEvent = nullptr;
    uint64_t mImmediateValue = 0;
    ConstantRing mConstants;
    bool mResidencyDirty = false;
};

} // namespace oka
