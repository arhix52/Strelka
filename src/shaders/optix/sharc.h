#pragma once

// Spatially hashed radiance cache after NVIDIA SHARC. Entries split per-frame
// atomic deposits from temporally resolved radiance; sharc_grid.h owns the
// host-testable grid arithmetic. OptiX deposits inline when each path completes,
// while Metal's wavefront backend uses a separate pass.

#include <sharc_grid.h>

#if defined(__CUDACC__)
#    include <vector_functions.h>
#    include <vector_types.h>
#endif

/// One slot, in two halves. Forty bytes, which is also what the SDK spends per
/// voxel across its three buffers.
struct SharcEntry
{
    /// The voxel this slot holds: coordinates, level, normal bucket and the
    /// responsive flag, packed by oka::sharc::voxelKey. Zero means empty, which
    /// is also what eviction writes -- and the key layout guarantees no real
    /// voxel packs to zero. First in the struct so the 64-bit atomic that
    /// claims it is aligned.
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

/// The key of the voxel a point belongs to.
///
/// `responsive` selects between the two entries a voxel can have: the ordinary
/// one, and the one holding the part of its signal that is expected to change
/// fast. They are different keys, so they are different slots and different
/// probe runs -- see sharcFind.
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

/// Find the slot for a voxel, inserting it when `insert` and there is room.
///
/// Returns false when the probe run is exhausted, which means the table is
/// over-subscribed and the caller should simply carry on tracing.
///
/// Responsive lighting uses an independent key and probe run because OptiX does
/// not carry the SDK's adjacent-entry offset in path state.
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

/// What a slot has resolved to, and how many deposits stand behind it.
///
/// Reads the resolved half only. A path must not read `accum`: those are this
/// frame's partial sums, they are being written by other paths as this one
/// reads, and early in a frame they are an average of a handful of samples.
/// The resolved half is a whole frame behind and that is the point.
static __forceinline__ __device__ float3 sharcRead(const SharcEntry* entries, uint32_t index, float& outSampleNum)
{
    const SharcEntry* entry = &entries[index];
    const oka::sharc::Resolved resolved =
        oka::sharc::unpackResolved(*(volatile unsigned int*)&entry->resolvedLo,
                                   *(volatile unsigned int*)&entry->resolvedHi);
    outSampleNum = resolved.sampleNum;
    return make_float3(resolved.r, resolved.g, resolved.b);
}

/// The responsive half of a voxel's answer, or zero if it has none.
///
/// Responsive lighting stores the fast-changing part of a voxel's signal in a
/// second entry with a shorter temporal window, so it can react in a handful of
/// frames while the rest of the signal keeps averaging over dozens. The two are
/// an additive decomposition of the same radiance -- what goes into one is
/// subtracted from what goes into the other -- so a reader adds them, and a
/// voxel with no responsive entry is simply the whole signal in the main one.
///
/// A second probe run, and worth gating: this is only called when responsive
/// lighting is on, which is a launch-parameter constant the pipeline is
/// specialised against, so a scene that does not use it pays nothing at all.
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

/// Add one estimate of a voxel's outgoing radiance to this frame's accumulator.
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
    // then believes. The resolve pass zeroes these every frame, so reaching the
    // limit now takes a frame rather than a render -- but a 4K launch can put
    // millions of paths through one voxel, so the guard stays.
    if (*(volatile unsigned int*)&entry->accumCount >= oka::sharc::kMaxCount)
    {
        return;
    }
    atomicAdd(&entry->accum[0], oka::sharc::encode(radiance.x));
    atomicAdd(&entry->accum[1], oka::sharc::encode(radiance.y));
    atomicAdd(&entry->accum[2], oka::sharc::encode(radiance.z));
    atomicAdd(&entry->accumCount, 1u);
}

// --- Debug visualisation ---------------------------------------------------
//
// Ported from the SDK's HashGridDebug* family, and here for the same reason it
// is there: every parameter of a hash grid is invisible in the final image
// until it is wrong, and then it is wrong in a way that looks like a shading
// bug. Voxel size in particular cannot be chosen without seeing it.

/// A stable colour per voxel, so the grid itself is visible.
///
/// Port of HashGridGetColorFromHash32. The bit ranges are the SDK's: they are
/// chosen so that neighbouring hashes land far apart in colour, which is what
/// makes a voxel boundary a visible edge rather than a gradient.
static __forceinline__ __device__ float3 sharcDebugColour(uint32_t hash)
{
    return make_float3((float)((hash >> 0) & 0x3FFu) / 1023.0f, (float)((hash >> 11) & 0x7FFu) / 2047.0f,
                       (float)((hash >> 22) & 0x7FFu) / 2047.0f);
}

/// The occupancy overlay: one small block per entry, lit where the entry is in
/// use. Port of HashGridDebugOccupancy.
///
/// Returns false for pixels the overlay does not cover, so the caller can leave
/// them alone. What to look for is in the SDK's own note: with a static camera
/// roughly 10-20% of the table should be occupied. Much more than that and the
/// table is thrashing -- either raise the capacity or evict harder.
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

    // Green for an entry carrying resolved radiance, amber for one that has been
    // inserted but has nothing to answer with yet. The difference matters: a
    // table that is full of amber is being inserted into and evicted before it
    // ever resolves, which reads as "occupied" but caches nothing.
    float sampleNum = 0.0f;
    sharcRead(entries, elementIndex, sampleNum);
    outColour = sampleNum > 0.0f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.75f, 0.0f);
    return true;
}

#endif // __CUDACC__
