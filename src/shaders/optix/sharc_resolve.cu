// The SHARC resolve pass. See sharc_resolve.h for what it is for, and
// sharc_grid.h::resolveEntry for the arithmetic, which lives there so it can be
// tested on the host -- every rule in it fails silently on a GPU.

#include <sharc.h>
#include <sharc_resolve.h>

__global__ void sharcResolveKernel(SharcEntry* entries, SharcResolveParams params)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.capacity)
    {
        return;
    }

    SharcEntry& entry = entries[index];
    const unsigned long long key = entry.key;
    if (key == 0ull)
    {
        return; // empty slot; nothing to resolve and nothing to age
    }

    // Responsive lighting uses a shorter accumulation window.
    const bool responsive = oka::sharc::isResponsiveKey(key);
    const uint32_t accumFrameNumMax = responsive ? params.responsiveFrameNumMax : params.accumFrameNumMax;
    // Both halves are deposited together and go stale in lockstep, so they use
    // the same lifetime to prevent one half being evicted while the other remains.
    const uint32_t staleFrameNumMax = params.staleFrameNumMax;

    oka::sharc::ResolveInput input;
    input.accum[0] = entry.accum[0];
    input.accum[1] = entry.accum[1];
    input.accum[2] = entry.accum[2];
    input.accumCount = entry.accumCount;
    input.resolvedLo = entry.resolvedLo;
    input.resolvedHi = entry.resolvedHi;
    input.frameData = entry.frameData;

    oka::sharc::ResolveOutput output = oka::sharc::resolveEntry(input, accumFrameNumMax, staleFrameNumMax);

    if (output.evict)
    {
        // Order matters. The key is what a path probes on, so it is cleared
        // last: a slot whose key is already gone but whose payload still holds
        // the old voxel's radiance would be inserted into by another voxel and
        // answer with the evicted one's numbers.
        entry.accum[0] = 0u;
        entry.accum[1] = 0u;
        entry.accum[2] = 0u;
        entry.accumCount = 0u;
        entry.resolvedLo = 0u;
        entry.resolvedHi = 0u;
        entry.frameData = 0u;
        __threadfence();
        entry.key = 0ull;
        return;
    }

    // Reprojection across grid levels.
    //
    // The level under a point follows its distance to the eye, so moving the eye
    // re-quantises a world that has not moved: the point lands in a different
    // voxel, its new entry starts from nothing, and everything the cache learned
    // about it sits one level away waiting to be evicted unread. This finds that
    // entry and blends it in, which is the difference between a camera movement
    // costing the cache a few frames and costing it everything it knew.
    //
    // Only for an entry young enough that the camera's movement is the likely
    // reason it is young -- kReprojectFrameNumMax, the SDK's rule.
    //
    // The adjacent entry's resolved half is read while another thread may be
    // writing it, and that race is deliberate and the SDK's: taking a lock, or a
    // second pass, to make a two-frame-old blend exact would cost more than the
    // blend is worth. The worst outcome is one voxel blended against a value one
    // frame out of date.
    if (params.reproject)
    {
        uint32_t accumFrames = 0u;
        uint32_t staleFrames = 0u;
        oka::sharc::unpackFrameData(output.frameData, accumFrames, staleFrames);
        if (accumFrames <= oka::sharc::kReprojectFrameNumMax)
        {
            const unsigned long long adjacent = oka::sharc::adjacentLevelKey(
                key, params.cameraPosition[0], params.cameraPosition[1], params.cameraPosition[2],
                params.cameraPositionPrev[0], params.cameraPositionPrev[1], params.cameraPositionPrev[2]);
            uint32_t adjacentSlot = 0u;
            if (adjacent != key && sharcFind(entries, params.capacity, adjacent, false, adjacentSlot))
            {
                const oka::sharc::Resolved other = oka::sharc::unpackResolved(
                    entries[adjacentSlot].resolvedLo, entries[adjacentSlot].resolvedHi);
                if (other.sampleNum > 0.0f)
                {
                    const oka::sharc::Resolved own =
                        oka::sharc::unpackResolved(output.resolvedLo, output.resolvedHi);
                    oka::sharc::packResolved(oka::sharc::blendAdjacentLevel(own, other), output.resolvedLo,
                                             output.resolvedHi);
                }
            }
        }
    }

    entry.resolvedLo = output.resolvedLo;
    entry.resolvedHi = output.resolvedHi;
    entry.frameData = output.frameData;

    // Clear this frame's accumulator now that it has been folded in. The next
    // launch's atomics start from zero, which is what makes `accum` mean "this
    // frame" rather than "since the last clear".
    //
    // The SDK cannot do this when responsive lighting is on -- its responsive
    // entries read the *main* entry's accumulation during resolve, and one
    // thread per entry gives no ordering between the two -- so it makes the host
    // clear the buffer before every update instead. That does not apply here:
    // the split is made on the path, both entries are deposited into together
    // and by the same paths, so a responsive entry's own sample count is already
    // the right one and it never has to look at its neighbour's.
    entry.accum[0] = 0u;
    entry.accum[1] = 0u;
    entry.accum[2] = 0u;
    entry.accumCount = 0u;
}

__global__ void sharcCountOccupancyKernel(const SharcEntry* entries, uint32_t capacity, uint32_t* counter)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= capacity)
    {
        return;
    }
    // One atomic per occupied slot rather than a reduction: this runs for a
    // debug readout, not in the frame's critical path, and a warp-level scan
    // here would be code to maintain for a number nobody reads unless a panel
    // is open.
    if (entries[index].key != 0ull)
    {
        atomicAdd(counter, 1u);
    }
}

extern "C" void sharcResolve(SharcEntry* entries, const SharcResolveParams& params, cudaStream_t stream)
{
    if (params.capacity == 0u)
    {
        return;
    }
    const dim3 blockSize(256, 1, 1);
    const dim3 gridSize((params.capacity + 255u) / 256u, 1, 1);
    sharcResolveKernel<<<gridSize, blockSize, 0, stream>>>(entries, params);
}

extern "C" void sharcCountOccupancy(
    const SharcEntry* entries, uint32_t capacity, uint32_t* deviceCounter, cudaStream_t stream)
{
    if (capacity == 0u || deviceCounter == nullptr)
    {
        return;
    }
    cudaMemsetAsync(deviceCounter, 0, sizeof(uint32_t), stream);
    const dim3 blockSize(256, 1, 1);
    const dim3 gridSize((capacity + 255u) / 256u, 1, 1);
    sharcCountOccupancyKernel<<<gridSize, blockSize, 0, stream>>>(entries, capacity, deviceCounter);
}
