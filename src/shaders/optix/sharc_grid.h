#pragma once

// The arithmetic of the SHARC hash grid, with no CUDA in it.
//
// Everything here is what decides *which* slot a shading point lands in and
// *what number* goes into it: the voxel quantisation, the two hashes, the probe
// sequence and the fixed-point encoding. None of it needs a GPU to be wrong, and
// all of the ways it goes wrong are silent -- a voxel that straddles a level
// boundary blurs across a wall, a fixed-point scale that overflows turns a lit
// room dark, a probe sequence that does not cover its table drops half the
// inserts. So it is here, where it can be tested, and src/shaders/optix/sharc.h
// is the thin layer of atomics on top.
//
// It is a port of src/shaders/metal/sharc.h and must stay one: two backends that
// quantise a voxel differently do not have a comparable cache, and the whole
// point of the parity ladder is that they answer alike.

#include <cstdint>

#if defined(__CUDACC__)
#    define STRELKA_SHARC_FN static __forceinline__ __device__
#else
#    include <cmath>
#    define STRELKA_SHARC_FN inline
#endif

namespace oka
{
namespace sharc
{

/// Fixed point rather than float atomics: a sum of a few hundred samples of
/// radiance stays inside 32 bits at this scale, and integer atomics are
/// available everywhere.
///
/// 64 rather than 1024 because the quantity stored is *outgoing* radiance --
/// what a path gathered divided by its throughput at that point -- which in a
/// lit interior is tens, not fractions. A finer scale would overflow the sum
/// long before the count did.
constexpr float kScale = 64.0f;

/// The clamp exists for fireflies, and it has to sit well above anything a scene
/// legitimately produces. Metal measured 32 costing a classroom a third of its
/// outgoing radiance and 34% of its brightness: a cache that clips is worse than
/// no cache, because the error is systematic rather than noisy.
constexpr float kClamp = 256.0f;

/// How far the linear probe walks before it gives up. A miss at the end of it
/// means the table is over-subscribed and the caller simply carries on tracing.
constexpr uint32_t kProbeCount = 8u;

/// Never fewer than this many entries, so that a capacity somebody typed in a
/// config cannot turn the cache into a single contended slot.
constexpr uint32_t kMinCapacity = 1u << 16;

/// One in eight paths never reads the cache and always traces to the end, so the
/// cache keeps converging instead of freezing at whatever the first paths
/// through a voxel happened to find. SHARC proper gives this its own update
/// pass; here it is the same paths, thinned.
constexpr uint32_t kUpdateShare = 8u;

/// Below this throughput a path is not worth recording.
///
/// The deposit is what the path gathered divided by its throughput at the visit,
/// so at a throughput of a thousandth the estimator has a variance to match --
/// Metal measured a handful of such deposits pulling a classroom 46% bright.
constexpr float kMinRecordThroughput = 0.05f;

/// Floor on the per-channel throughput the deposit divides by, so one dark
/// channel cannot turn a finite estimate into a huge one.
constexpr float kThroughputFloor = 0.02f;

/// Surfaces rougher than this may read the cache. A mirror reflects a direction
/// rather than a place, and a voxel average has no direction in it.
constexpr float kMinRoughness = 0.3f;

/// MurmurHash3's finalizer. Cheap, and it mixes the low bits, which is where
/// voxel coordinates differ.
STRELKA_SHARC_FN uint32_t hash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

/// The world size of a voxel at `distance` from the eye.
///
/// It follows the screen-space footprint: `baseSize` is the world size of one
/// pixel at unit distance times however many pixels a voxel should span, so the
/// same setting means the same thing in a forest and in a classroom, at any
/// resolution or field of view. An absolute size in metres does not -- Metal
/// measured 0.25 m being a tenth of the room's depth (a voxel fifty pixels
/// across, and a 15% bias to match) while in the forest the same number was
/// barely used.
///
/// Quantised to powers of two so that a point near a level boundary lands in one
/// voxel or the other rather than smearing across both.
struct Voxel
{
    float size = 1.0f;
    /// log2 of the size, which is the exponent the hash mixes in. Carried
    /// rather than recovered with a second logarithm: the size is an exact power
    /// of two, so the level is an exact integer, and taking log2 of it again is
    /// one more chance for a value near a boundary to fall the other way.
    int32_t level = 0;
};

STRELKA_SHARC_FN Voxel voxelForDistance(float distance, float baseSize)
{
#if defined(__CUDACC__)
    const float footprint = fmaxf(distance * baseSize, 1e-4f);
    const float level = floorf(log2f(footprint));
    Voxel v;
    v.level = (int32_t)level;
    v.size = exp2f(level);
#else
    const float footprint = std::fmax(distance * baseSize, 1e-4f);
    const float level = std::floor(std::log2(footprint));
    Voxel v;
    v.level = (int32_t)level;
    v.size = std::exp2(level);
#endif
    return v;
}

/// The integer coordinate of `coordinate` in a grid of cells `size` across.
STRELKA_SHARC_FN int32_t voxelCoordinate(float coordinate, float size)
{
#if defined(__CUDACC__)
    return (int32_t)floorf(coordinate / size);
#else
    return (int32_t)std::floor(coordinate / size);
#endif
}

/// Six buckets: the dominant axis of the normal and its sign.
///
/// Finer than that starts splitting a smooth surface into stripes; coarser lets
/// the two sides of a leaf, or a floor and the ceiling below it, share radiance.
STRELKA_SHARC_FN uint32_t normalBucket(float nx, float ny, float nz)
{
#if defined(__CUDACC__)
    const float ax = fabsf(nx), ay = fabsf(ny), az = fabsf(nz);
#else
    const float ax = std::fabs(nx), ay = std::fabs(ny), az = std::fabs(nz);
#endif
    uint32_t axis;
    float component;
    if (ax > ay)
    {
        axis = (ax > az) ? 0u : 2u;
    }
    else
    {
        axis = (ay > az) ? 1u : 2u;
    }
    component = (axis == 0u) ? nx : (axis == 1u) ? ny : nz;
    return axis * 2u + (component < 0.0f ? 1u : 0u);
}

/// Where the eight-probe run starts, and where it goes next.
STRELKA_SHARC_FN uint32_t probeSlot(uint32_t voxelHash, uint32_t capacity, uint32_t probe)
{
    const uint32_t mask = capacity - 1u;
    return (voxelHash + probe) & mask;
}

/// A voxel's hash, from its integer coordinates, its level and its normal
/// bucket. Separate from the position quantisation so a test can drive it with
/// coordinates rather than with a camera.
STRELKA_SHARC_FN uint32_t voxelHash(int32_t x, int32_t y, int32_t z, int32_t level, uint32_t bucket)
{
    uint32_t h = hash((uint32_t)x * 73856093u);
    h ^= hash((uint32_t)y * 19349663u);
    h ^= hash((uint32_t)z * 83492791u);
    h ^= hash((uint32_t)level * 2654435761u);
    h ^= hash(bucket * 40503u);
    return h;
}

/// A second, independent mix, identifying the voxel within its slot.
///
/// Never zero: zero marks an empty slot. A collision between two different
/// voxels that also agree on 32 bits costs one wrong voxel out of millions,
/// which is a fairer trade than the memory a 64-bit key and its atomics would
/// take.
STRELKA_SHARC_FN uint32_t voxelKey(uint32_t voxelHashValue)
{
    const uint32_t k = hash(voxelHashValue ^ 0x9e3779b9u);
    return k == 0u ? 1u : k;
}

/// Whether this path is one of the fixed share that records rather than reads.
///
/// A path either reads or records, never both. One that has recorded a voxel
/// owes it an honest estimate of the rest of the path, and a cached read inside
/// that estimate feeds the cache its own output -- a loop that amplifies
/// whatever error it starts with. Metal saw it as a classroom 11% bright with no
/// single step being wrong.
STRELKA_SHARC_FN bool isUpdatePath(uint32_t pixelIndex, uint32_t sampleIndex)
{
    return (hash(pixelIndex * 9781u + sampleIndex * 6271u) & (kUpdateShare - 1u)) == 0u;
}

/// Encode one channel of radiance for the accumulator.
///
/// Rounded, not truncated, and that is the one deliberate departure from the
/// Metal port. Truncation loses half a quantum on every deposit in the same
/// direction, and half a quantum is 1/128 of a unit of radiance: against an
/// outgoing radiance of a couple of units that is a quarter of a percent, every
/// time, for as long as the cache is on. Measured -- 00_calibration and
/// 02_basecolor both came back at a ratio of 0.997 against the same render with
/// the cache off, on two scenes that share nothing but this arithmetic. Rounding
/// makes the error zero-mean instead, and the same two rows then read 1.000.
/// The Metal side should take the same change; it is reported as a hand-off.
STRELKA_SHARC_FN uint32_t encode(float radiance)
{
    float v = radiance;
    if (!(v > 0.0f))
    {
        v = 0.0f; // catches NaN as well as negatives
    }
    if (v > kClamp)
    {
        v = kClamp;
    }
    return (uint32_t)(v * kScale + 0.5f);
}

/// Decode a channel's mean from its sum and the sample count.
STRELKA_SHARC_FN float decode(uint32_t sum, uint32_t count)
{
    if (count == 0u)
    {
        return 0.0f;
    }
    return (float)sum / ((float)count * kScale);
}

/// How many deposits a slot can take before its sum could overflow.
///
/// One deposit is at most kClamp * kScale, so the sum stays inside 32 bits for
/// this many of them. Reached in a long render, and a wrapped sum reads back as
/// a near-black voxel that every path through it then believes -- so the write
/// stops rather than wraps.
constexpr uint32_t kMaxCount = (uint32_t)(4294967295.0 / (double)(kClamp * kScale));

} // namespace sharc
} // namespace oka
