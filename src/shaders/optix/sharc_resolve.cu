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
