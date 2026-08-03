#pragma once

// A spatially hashed radiance cache, after NVIDIA's SHARC.
//
// The profile says why: `extend` is two thirds of the frame, and paths run to
// twelve bounces although capping them at four changes the image by under one
// percent. The bounces are cheap in contribution and expensive in traversal.
// Caching lets a path stop after two or three and read what the rest of it would
// have gathered -- from an average over every path that has passed through the
// same place, which is also quieter than any single path's estimate.
//
// The cache is a hash grid rather than a spatial structure: no build, no
// hierarchy, and a voxel is addressed by arithmetic. Its resolution follows the
// distance to the camera, so a voxel covers roughly a constant angle -- fine
// where the eye is, coarse where it is not, without anything having to decide
// that per scene.
//
// What it is not: a cache of everything. Only diffuse-ish bounces past the first
// few are read from it. The camera ray, the first bounce and any specular path
// are traced as before, because that is what carries the detail a cache would
// blur away.

#include <metal_stdlib>

using namespace metal;

// One slot. The key is a 32-bit checksum of the voxel, zero meaning empty; a
// collision between two different voxels that also agree on 32 bits costs one
// wrong voxel out of millions, which is a fairer trade than the memory a 64-bit
// key and its atomics would take.
struct SharcEntry
{
    atomic_uint key;
    atomic_uint accum[3]; // radiance sums, fixed point
    atomic_uint count;
};

// Fixed point rather than float atomics: a sum of 256 samples of radiance up to
// a few hundred stays inside 32 bits at this scale, and integer atomics are
// available everywhere.
// 64 rather than 1024: the quantity stored is outgoing radiance -- what a path
// gathered divided by its throughput at that point -- which in a lit interior is
// tens, not fractions. A finer scale would overflow the sum long before the
// count did.
constant float kSharcScale = 64.0f;
// The clamp exists for fireflies, and it has to sit well above anything the
// scene legitimately produces. At 32 it cut a classroom's outgoing radiance by a
// third and the render came back 34% dark -- a cache that clips is worse than no
// cache, because the error is systematic rather than noisy.
constant float kSharcClamp = 256.0f;

static inline uint32_t sharcHash(uint32_t x)
{
    // Finalizer from MurmurHash3: cheap and mixes the low bits, which is where
    // voxel coordinates differ.
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

// The voxel a point belongs to, and a checksum that identifies it.
//
// The level is chosen from the distance to the camera so that a voxel subtends
// roughly a constant angle. Quantising the normal as well keeps the two sides of
// a leaf, or a floor and the ceiling below it, from sharing radiance.
static inline bool sharcVoxel(float3 position, float3 normal, float3 cameraPosition,
                              float baseSize, thread uint32_t& outHash, thread uint32_t& outKey)
{
    // Voxel size follows the screen-space footprint: `baseSize` is the world
    // size of one pixel at unit distance times however many pixels a voxel
    // should span, so the same setting means the same thing in a forest and in a
    // classroom, at any resolution or field of view.
    //
    // An absolute size in metres does not: 0.25 m was a tenth of the distance in
    // a room -- a voxel fifty pixels across, and a 15% bias to match -- while in
    // the forest the same number was barely used at all.
    //
    // Quantised to powers of two so that a point near a level boundary lands in
    // one voxel or the other rather than smearing across both.
    const float distance = length(position - cameraPosition);
    const float footprint = max(distance * baseSize, 1e-4f);
    const float size = exp2(floor(log2(footprint)));
    const float level = log2(size);

    const int3 voxel = int3(floor(position / size));
    // Six buckets: the dominant axis and its sign. Finer than that starts
    // splitting a smooth surface into stripes.
    const float3 an = abs(normal);
    uint32_t axis = an.x > an.y ? (an.x > an.z ? 0u : 2u) : (an.y > an.z ? 1u : 2u);
    axis = axis * 2u + (normal[axis] < 0.0f ? 1u : 0u);

    uint32_t h = sharcHash((uint32_t)voxel.x * 73856093u);
    h ^= sharcHash((uint32_t)voxel.y * 19349663u);
    h ^= sharcHash((uint32_t)voxel.z * 83492791u);
    h ^= sharcHash((uint32_t)(int)level * 2654435761u);
    h ^= sharcHash(axis * 40503u);

    outHash = h;
    // A second, independent mix identifies the voxel within its slot. Never
    // zero, because zero marks an empty slot.
    outKey = max(sharcHash(h ^ 0x9e3779b9u), 1u);
    return true;
}

// Find the slot for a voxel, inserting it if there is room. Returns false when
// the probe sequence is full, which means the cache is over-subscribed and the
// caller should simply carry on tracing.
static inline bool sharcFind(device SharcEntry* entries, uint32_t capacity, uint32_t hash,
                             uint32_t key, bool insert, thread uint32_t& outIndex)
{
    const uint32_t mask = capacity - 1u;
    uint32_t index = hash & mask;
    for (uint32_t probe = 0u; probe < 8u; ++probe)
    {
        device atomic_uint* slot = &entries[index].key;
        const uint32_t existing = atomic_load_explicit(slot, memory_order_relaxed);
        if (existing == key)
        {
            outIndex = index;
            return true;
        }
        if (existing == 0u)
        {
            if (!insert)
                return false;
            uint32_t expected = 0u;
            if (atomic_compare_exchange_weak_explicit(slot, &expected, key, memory_order_relaxed,
                                                      memory_order_relaxed))
            {
                outIndex = index;
                return true;
            }
            // Someone else took it; it may even be the same voxel.
            if (expected == key)
            {
                outIndex = index;
                return true;
            }
        }
        index = (index + 1u) & mask;
    }
    return false;
}

static inline float3 sharcRead(device SharcEntry* entries, uint32_t index, thread uint32_t& outCount)
{
    const uint32_t count = atomic_load_explicit(&entries[index].count, memory_order_relaxed);
    outCount = count;
    if (count == 0u)
        return float3(0.0f);
    const float inv = 1.0f / (float(count) * kSharcScale);
    return float3(float(atomic_load_explicit(&entries[index].accum[0], memory_order_relaxed)),
                  float(atomic_load_explicit(&entries[index].accum[1], memory_order_relaxed)),
                  float(atomic_load_explicit(&entries[index].accum[2], memory_order_relaxed))) *
           inv;
}

static inline void sharcWrite(device SharcEntry* entries, uint32_t index, float3 radiance)
{
    // Clamped before it goes in. One firefly deposited into a voxel is then read
    // back by every path that passes through it, which turns a single bright
    // pixel into a bright region -- the one failure mode of a cache that is worse
    // than the noise it replaces.
    const float3 clamped = clamp(radiance, 0.0f, kSharcClamp);
    for (uint32_t c = 0; c < 3; ++c)
    {
        atomic_fetch_add_explicit(&entries[index].accum[c],
                                  (uint32_t)(clamped[c] * kSharcScale), memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&entries[index].count, 1u, memory_order_relaxed);
}
