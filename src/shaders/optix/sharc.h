#pragma once

// A spatially hashed radiance cache, after NVIDIA's SHARC. Port of
// src/shaders/metal/sharc.h; the arithmetic that decides which slot a point
// lands in and what number goes into it lives in sharc_grid.h, which carries no
// CUDA and has tests.
//
// Why: paths run to the depth limit although capping them at three or four
// changes the image by under a percent, and those late bounces are cheap in
// contribution and expensive in traversal. Caching lets a path stop after two or
// three and read what the rest of it would have gathered -- from an average over
// every path that has passed through the same place, which is also quieter than
// any single path's estimate.
//
// The cache is a hash grid rather than a spatial structure: no build, no
// hierarchy, and a voxel is addressed by arithmetic. Its resolution follows the
// distance to the camera, so a voxel covers roughly a constant angle -- fine
// where the eye is, coarse where it is not, without anything having to decide
// that per scene.
//
// What it is not: a cache of everything. Only diffuse-ish bounces past the first
// few read from it. The camera ray, the first bounce and any specular path are
// traced as before, because that is what carries the detail a cache would blur.
//
// The one shape difference from Metal is where the deposit happens. Metal is a
// wavefront, so it needs a separate pass at the end of a sample to walk the path
// states; here the path runs to completion inside the raygen loop, and the
// deposit is the last thing that loop does. Same estimator, one fewer buffer.

#include <sharc_grid.h>

#if defined(__CUDACC__)
#    include <vector_functions.h>
#    include <vector_types.h>
#endif

/// One slot: a key, three fixed-point radiance sums and a count. Twenty bytes,
/// which is what the Metal backend allocates per entry too.
struct SharcEntry
{
    unsigned int key;
    unsigned int accum[3];
    unsigned int count;
};

static_assert(sizeof(SharcEntry) == 20, "the host allocates 20 bytes an entry");

/// Marks a path that has not visited a voxel this sample.
#define SHARC_NO_ENTRY 0xFFFFFFFFu

#if defined(__CUDACC__)

/// The voxel a point belongs to, and the checksum that identifies it.
static __forceinline__ __device__ void sharcVoxel(
    float3 position, float3 normal, float3 cameraPosition, float baseSize, uint32_t& outHash, uint32_t& outKey)
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
    outHash = oka::sharc::voxelHash(x, y, z, voxel.level, bucket);
    outKey = oka::sharc::voxelKey(outHash);
}

/// Find the slot for a voxel, inserting it when `insert` and there is room.
///
/// Returns false when the probe run is exhausted, which means the table is
/// over-subscribed and the caller should simply carry on tracing.
static __forceinline__ __device__ bool sharcFind(
    SharcEntry* entries, uint32_t capacity, uint32_t hash, uint32_t key, bool insert, uint32_t& outIndex)
{
    for (uint32_t probe = 0u; probe < oka::sharc::kProbeCount; ++probe)
    {
        const uint32_t index = oka::sharc::probeSlot(hash, capacity, probe);
        unsigned int* slot = &entries[index].key;
        const unsigned int existing = *(volatile unsigned int*)slot;
        if (existing == key)
        {
            outIndex = index;
            return true;
        }
        if (existing == 0u)
        {
            if (!insert)
            {
                return false;
            }
            const unsigned int previous = atomicCAS(slot, 0u, key);
            // Either this thread took the slot, or another thread took it for
            // the same voxel; both are a hit.
            if (previous == 0u || previous == key)
            {
                outIndex = index;
                return true;
            }
        }
    }
    return false;
}

/// The mean radiance a slot holds, and how many deposits it is made of.
static __forceinline__ __device__ float3 sharcRead(const SharcEntry* entries, uint32_t index, uint32_t& outCount)
{
    const SharcEntry* entry = &entries[index];
    const unsigned int count = *(volatile unsigned int*)&entry->count;
    outCount = count;
    if (count == 0u)
    {
        return make_float3(0.0f);
    }
    return make_float3(oka::sharc::decode(*(volatile unsigned int*)&entry->accum[0], count),
                       oka::sharc::decode(*(volatile unsigned int*)&entry->accum[1], count),
                       oka::sharc::decode(*(volatile unsigned int*)&entry->accum[2], count));
}

/// Add one estimate of a voxel's outgoing radiance.
///
/// Clamped on the way in. One firefly deposited into a voxel is then read back
/// by every path that passes through it, which turns a single bright pixel into
/// a bright region -- the one failure mode of a cache that is worse than the
/// noise it replaces.
static __forceinline__ __device__ void sharcWrite(SharcEntry* entries, uint32_t index, float3 radiance)
{
    SharcEntry* entry = &entries[index];
    // A slot that has taken its fill stops taking rather than wrapping: a
    // wrapped sum reads back as a near-black voxel, which every path through it
    // then believes.
    if (*(volatile unsigned int*)&entry->count >= oka::sharc::kMaxCount)
    {
        return;
    }
    atomicAdd(&entry->accum[0], oka::sharc::encode(radiance.x));
    atomicAdd(&entry->accum[1], oka::sharc::encode(radiance.y));
    atomicAdd(&entry->accum[2], oka::sharc::encode(radiance.z));
    atomicAdd(&entry->count, 1u);
}

#endif // __CUDACC__
