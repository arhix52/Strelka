#pragma once

#include <sharc_grid.h>

#if defined(__CUDACC__)
#    include <vector_functions.h>
#    include <vector_types.h>
#endif

/// One slot, in two halves. Forty bytes, which is also what the SDK spends per
/// voxel across its three buffers.
struct SharcEntry
{
    unsigned long long key;

    /// This frame's deposits. Written by atomicAdd from many paths at once,
    /// zeroed by the resolve pass once it has folded them in.
    unsigned int accum[3];
    unsigned int accumCount;

    /// What the voxel has concluded across frames: fp16 r|g, then fp16
    /// b|sampleNum. The only half a path reads. See oka::sharc::Resolved.
    unsigned int resolvedLo;
    unsigned int resolvedHi;

    /// Frames merged into `resolved`, and frames since anything was deposited.
    /// Sixteen bits each. See oka::sharc::packFrameData.
    unsigned int frameData;
};

static_assert(sizeof(SharcEntry) == 40, "the host allocates 40 bytes an entry");
static_assert(sizeof(((SharcEntry*)nullptr)->key) == 8, "the key is claimed by a 64-bit atomicCAS");

/// Marks a path that has not visited a voxel this sample.
#define SHARC_NO_ENTRY 0xFFFFFFFFu

#if defined(__CUDACC__)

static __forceinline__ __device__ uint64_t sharcVoxel(
    float3 position, float3 normal, float3 cameraPosition, float baseSize, bool responsive)
{
    // Spelled out rather than taken from sutil, so this header needs nothing but
    // the CUDA vector types and can be included from the params header.
    const float dx = position.x - cameraPosition.x;
    const float dy = position.y - cameraPosition.y;
    const float dz = position.z - cameraPosition.z;
    const oka::sharc::Voxel voxel = oka::sharc::voxelForDistance(sqrtf(dx * dx + dy * dy + dz * dz), baseSize);
    const int32_t x = oka::sharc::voxelCoordinate(position.x, voxel.size);
    const int32_t y = oka::sharc::voxelCoordinate(position.y, voxel.size);
    const int32_t z = oka::sharc::voxelCoordinate(position.z, voxel.size);
    const uint32_t bucket = oka::sharc::normalBucket(normal.x, normal.y, normal.z);
    return oka::sharc::voxelKey(x, y, z, voxel.level, bucket, responsive);
}

static __forceinline__ __device__ bool sharcFind(
    SharcEntry* entries, uint32_t capacity, unsigned long long key, bool insert, uint32_t& outIndex)
{
    const uint32_t hash = oka::sharc::keyHash(key);
    for (uint32_t probe = 0u; probe < oka::sharc::kProbeCount; ++probe)
    {
        const uint32_t index = oka::sharc::probeSlot(hash, capacity, probe);
        unsigned long long* slot = &entries[index].key;
        const unsigned long long existing = *(volatile unsigned long long*)slot;
        if (existing == key)
        {
            outIndex = index;
            return true;
        }
        if (existing == 0ull)
        {
            if (!insert)
            {
                // Eviction leaves holes in probe runs, so an empty slot cannot
                // terminate lookup; responsive halves must also remain visible.
                continue;
            }
            const unsigned long long previous = atomicCAS(slot, 0ull, key);
            // Either this thread took the slot, or another thread took it for
            // the same voxel; both are a hit.
            if (previous == 0ull || previous == key)
            {
                outIndex = index;
                return true;
            }
        }
    }
    return false;
}

static __forceinline__ __device__ float3 sharcRead(const SharcEntry* entries, uint32_t index, float& outSampleNum)
{
    const SharcEntry* entry = &entries[index];
    const oka::sharc::Resolved resolved =
        oka::sharc::unpackResolved(*(volatile unsigned int*)&entry->resolvedLo,
                                   *(volatile unsigned int*)&entry->resolvedHi);
    outSampleNum = resolved.sampleNum;
    return make_float3(resolved.r, resolved.g, resolved.b);
}

static __forceinline__ __device__ float3 sharcReadResponsive(const SharcEntry* entries,
                                                             uint32_t capacity,
                                                             unsigned long long mainKey)
{
    uint32_t slot = 0u;
    if (!sharcFind(const_cast<SharcEntry*>(entries), capacity, oka::sharc::responsiveKey(mainKey), false, slot))
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    float sampleNum = 0.0f;
    const float3 radiance = sharcRead(entries, slot, sampleNum);
    return sampleNum > 0.0f ? radiance : make_float3(0.0f, 0.0f, 0.0f);
}

static __forceinline__ __device__ void sharcWrite(SharcEntry* entries, uint32_t index, float3 radiance)
{
    SharcEntry* entry = &entries[index];
    if (*(volatile unsigned int*)&entry->accumCount >= oka::sharc::kMaxCount)
    {
        return;
    }
    atomicAdd(&entry->accum[0], oka::sharc::encode(radiance.x));
    atomicAdd(&entry->accum[1], oka::sharc::encode(radiance.y));
    atomicAdd(&entry->accum[2], oka::sharc::encode(radiance.z));
    atomicAdd(&entry->accumCount, 1u);
}

static __forceinline__ __device__ float3 sharcDebugColour(uint32_t hash)
{
    return make_float3((float)((hash >> 0) & 0x3FFu) / 1023.0f, (float)((hash >> 11) & 0x7FFu) / 2047.0f,
                       (float)((hash >> 22) & 0x7FFu) / 2047.0f);
}

static __forceinline__ __device__ bool sharcDebugOccupancy(
    const SharcEntry* entries, uint32_t capacity, uint2 pixel, uint2 screenSize, float3& outColour)
{
    const uint32_t elementSize = 7u;
    const uint32_t blockSize = elementSize + 1u; // one pixel of border

    const uint32_t rowNum = screenSize.y / blockSize;
    if (rowNum == 0u)
    {
        return false;
    }
    const uint32_t rowIndex = pixel.y / blockSize;
    const uint32_t columnIndex = pixel.x / blockSize;
    const uint32_t elementIndex = columnIndex * rowNum + rowIndex;

    if (elementIndex >= capacity || (pixel.x % blockSize) >= elementSize || (pixel.y % blockSize) >= elementSize)
    {
        return false;
    }

    if (*(volatile unsigned long long*)&entries[elementIndex].key == 0ull)
    {
        outColour = make_float3(0.0f, 0.0f, 0.0f);
        return true;
    }

    float sampleNum = 0.0f;
    sharcRead(entries, elementIndex, sampleNum);
    outColour = sampleNum > 0.0f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.75f, 0.0f);
    return true;
}

#endif // __CUDACC__
