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
#    include <cstring>
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

/// Whether a path may read the cache at a hit it has just traced to.
///
/// This is the SDK's eligibility test, and it replaces what used to be a flat
/// `roughness > 0.3` on the surface being *hit*. That gate was crude in both
/// directions: it asked the wrong surface -- what matters is the lobe that
/// launched the segment, not what the segment landed on -- and it cost about
/// two thirds of the cache's benefit. Measured on iso_bathroom at one sample
/// with a warm cache, the share of paths the cache terminates after a single
/// bounce: 44.9% with no cache, 54.9% with the roughness gate, 75.9% with no
/// gate at all. Blurring is what the gate is *for*, so removing it is not an
/// option; asking the right question is.
///
/// Two conditions, both from the SDK:
///
///  * the segment just traced has to be longer than a voxel's diagonal, or the
///    hit is inside the same voxel the ray left and the cache would answer with
///    an average that includes the very point being shaded;
///  * the lobe that launched the segment has to have spread wider than a voxel
///    by the time it arrived. A tight lobe still carries an image -- that is
///    what a mirror is -- and a voxel average has no image in it. A wide one
///    carries an average already, so reading one costs nothing.
///
/// `launchRoughness` is the roughness of the surface the segment left, not of
/// the one it reached.
STRELKA_SHARC_FN bool mayReadCache(float segmentLength, float launchRoughness, float voxelSize)
{
    // sqrt(3) is the voxel diagonal: a segment shorter than that may not have
    // left the voxel at all.
    if (!(segmentLength > voxelSize * 1.7320508f))
    {
        return false;
    }
    // alpha is the squared roughness, as everywhere else in this renderer, and
    // the spread is the SDK's: hitDistance * sqrt(0.5 * a^2 / (1 - a^2)).
    // Clamped below one so a perfectly smooth surface gives zero rather than a
    // division by zero -- and zero is the right answer for it.
    const float clamped = launchRoughness < 0.99f ? launchRoughness : 0.99f;
    const float alpha = clamped * clamped;
    const float alpha2 = alpha * alpha;
    const float denominator = 1.0f - alpha2 > 1e-6f ? 1.0f - alpha2 : 1e-6f;
#if defined(__CUDACC__)
    const float footprint = segmentLength * sqrtf(0.5f * alpha2 / denominator);
#else
    const float footprint = segmentLength * std::sqrt(0.5f * alpha2 / denominator);
#endif
    return footprint > voxelSize;
}

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

// ---------------------------------------------------------------------------
// The key: which voxel a slot holds
// ---------------------------------------------------------------------------
//
// Sixty-four bits carrying the voxel's integer coordinates, its level, its
// normal bucket and one flag -- the SDK's HashGridComputeSpatialHash layout,
// bit for bit in spirit if not in constant.
//
// It used to be a 32-bit checksum of all of that, mixed through the hash, and
// the comment here defended the trade: one wrong voxel in millions against the
// memory a 64-bit key costs. That was true as far as it went, and it is not why
// the key is now what it is. An un-hashable key cannot answer *which* voxel a
// slot holds, and two things need to ask:
//
//   * reprojection, which takes an entry's voxel, works out where the same
//     region sat one grid level away, and blends the data the camera's movement
//     stranded there (adjacentLevelKey below);
//   * responsive lighting, which needs one bit to say "this entry holds the
//     fast-changing part of the signal" -- and a bit in a checksum is a bit
//     that changes the checksum.
//
// Both come out of the SDK, and neither is expressible over a hash. The cost is
// four bytes an entry (32 -> 40 with alignment) and the collisions go away.

/// 17 bits per axis, signed, which is the SDK's split and for the same reason.
/// Coordinates are position/voxelSize and the voxel size follows the distance to
/// the camera, so a point twice as far away has a voxel twice as large and lands
/// at roughly the same coordinate: the range in play is hundreds, not millions.
/// Out-of-range coordinates wrap rather than clamp, exactly as the SDK's do; two
/// voxels 131072 cells apart at the same level then share a key, which needs a
/// scene tens of thousands of voxels across to reach.
constexpr uint32_t kPositionBits = 17u;
constexpr uint32_t kLevelBits = 9u;
constexpr uint32_t kNormalBits = 3u;

constexpr uint64_t kPositionMask = (1ull << kPositionBits) - 1ull;
constexpr uint64_t kLevelMask = (1ull << kLevelBits) - 1ull;
constexpr uint64_t kNormalMask = (1ull << kNormalBits) - 1ull;

constexpr uint32_t kLevelOffset = kPositionBits * 3u;
constexpr uint32_t kNormalOffset = kLevelOffset + kLevelBits;
/// The top bit, left free by the layout above (17*3 + 9 + 3 = 63). The SDK
/// reserves the same bit for the same purpose.
constexpr uint32_t kResponsiveBit = 63u;

/// Our level is the exponent of a power-of-two voxel size and so is signed --
/// negative wherever the footprint is under a metre, which is most of an
/// interior. The field is unsigned, so it carries level + bias.
///
/// 256 leaves [-255, 255], against a level that in practice runs about [-20, 5].
/// The clamp below keeps the biased value at least 1, which is what guarantees a
/// key is never zero: zero is the empty slot, and a voxel that hashed to it
/// would be invisible to every probe that walks past.
constexpr int32_t kLevelBias = 256;
constexpr int32_t kLevelMin = -kLevelBias + 1;
constexpr int32_t kLevelMax = kLevelBias - 1;

/// A voxel's identity: coordinates, level, normal bucket, responsive flag.
STRELKA_SHARC_FN uint64_t voxelKey(int32_t x, int32_t y, int32_t z, int32_t level, uint32_t bucket, bool responsive)
{
    const int32_t clamped = level < kLevelMin ? kLevelMin : (level > kLevelMax ? kLevelMax : level);
    uint64_t key = ((uint64_t)(uint32_t)x & kPositionMask) << (kPositionBits * 0u);
    key |= ((uint64_t)(uint32_t)y & kPositionMask) << (kPositionBits * 1u);
    key |= ((uint64_t)(uint32_t)z & kPositionMask) << (kPositionBits * 2u);
    key |= ((uint64_t)(uint32_t)(clamped + kLevelBias) & kLevelMask) << kLevelOffset;
    key |= ((uint64_t)bucket & kNormalMask) << kNormalOffset;
    if (responsive)
    {
        key |= 1ull << kResponsiveBit;
    }
    return key;
}

/// Sign-extend a packed coordinate, without a branch.
STRELKA_SHARC_FN int32_t unpackCoordinate(uint64_t key, uint32_t axis)
{
    const int32_t raw = (int32_t)((key >> (kPositionBits * axis)) & kPositionMask);
    return (raw << (32u - kPositionBits)) >> (32u - kPositionBits);
}

STRELKA_SHARC_FN int32_t unpackLevel(uint64_t key)
{
    return (int32_t)((key >> kLevelOffset) & kLevelMask) - kLevelBias;
}

STRELKA_SHARC_FN uint32_t unpackBucket(uint64_t key)
{
    return (uint32_t)((key >> kNormalOffset) & kNormalMask);
}

STRELKA_SHARC_FN bool isResponsiveKey(uint64_t key)
{
    return ((key >> kResponsiveBit) & 1ull) != 0ull;
}

/// The same voxel's key, marked as holding the responsive part of the signal.
STRELKA_SHARC_FN uint64_t responsiveKey(uint64_t key)
{
    return key | (1ull << kResponsiveBit);
}

/// The 32 bits of a key that choose its slot.
///
/// Both halves are mixed and combined, as the SDK's HashGridHash32 does: the low
/// half alone is the coordinates, and neighbouring voxels differ only there.
STRELKA_SHARC_FN uint32_t keyHash(uint64_t key)
{
    return hash((uint32_t)key) ^ hash((uint32_t)(key >> 32));
}

/// Where the eight-probe run starts, and where it goes next.
STRELKA_SHARC_FN uint32_t probeSlot(uint32_t voxelHash, uint32_t capacity, uint32_t probe)
{
    const uint32_t mask = capacity - 1u;
    return (voxelHash + probe) & mask;
}

/// The world size of a voxel at `level`, which is what the level *is*.
STRELKA_SHARC_FN float voxelSizeForLevel(int32_t level)
{
#if defined(__CUDACC__)
    return exp2f((float)level);
#else
    return std::exp2((float)level);
#endif
}

/// The key of the voxel one grid level away that covers the same place.
///
/// Port of the SDK's SharcGetAdjacentLevelHashKey, and the answer to the thing
/// that makes a distance-driven grid awkward: the level follows the distance to
/// the eye, so when the eye moves, a point that has not moved at all quantises
/// into a different voxel. Everything the cache learned about it is still in the
/// table, under the old level, and without this it simply ages out unread while
/// the new voxel starts from nothing.
///
/// Which way to look is decided the SDK's way, by measuring the same voxel's
/// distance to both cameras in its own grid units: if the eye is nearer now, the
/// current level is the finer one and the history is one level coarser.
STRELKA_SHARC_FN uint64_t adjacentLevelKey(
    uint64_t key, float cameraX, float cameraY, float cameraZ, float previousX, float previousY, float previousZ)
{
    int32_t x = unpackCoordinate(key, 0u);
    int32_t y = unpackCoordinate(key, 1u);
    int32_t z = unpackCoordinate(key, 2u);
    int32_t level = unpackLevel(key);

    const float voxelSize = voxelSizeForLevel(level);
    const int32_t cx = voxelCoordinate(cameraX, voxelSize);
    const int32_t cy = voxelCoordinate(cameraY, voxelSize);
    const int32_t cz = voxelCoordinate(cameraZ, voxelSize);
    const int32_t px = voxelCoordinate(previousX, voxelSize);
    const int32_t py = voxelCoordinate(previousY, voxelSize);
    const int32_t pz = voxelCoordinate(previousZ, voxelSize);

    const int32_t dx = cx - x, dy = cy - y, dz = cz - z;
    const int32_t qx = px - x, qy = py - y, qz = pz - z;
    const int32_t distance = dx * dx + dy * dy + dz * dz;
    const int32_t distancePrev = qx * qx + qy * qy + qz * qz;

    if (distance < distancePrev)
    {
        // The eye came closer, so this voxel is finer than the one that held the
        // history. Halving the coordinate is the coarser level's cell, and it has
        // to be a floor: an arithmetic shift is one on every compiler this
        // builds under, where dividing by two truncates towards zero and would
        // fold the two cells either side of the origin into one.
        x >>= 1;
        y >>= 1;
        z >>= 1;
        level = level + 1 > kLevelMax ? kLevelMax : level + 1;
    }
    else
    {
        x <<= 1;
        y <<= 1;
        z <<= 1;
        level = level - 1 < kLevelMin ? kLevelMin : level - 1;
    }

    // The normal bucket and the responsive flag carry over: this is the same
    // surface, seen at a different resolution.
    return voxelKey(x, y, z, level, unpackBucket(key), isResponsiveKey(key));
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
    // roundf, not a truncated v + 0.5f: the two agree for every non-negative
    // value except the ones where v * kScale + 0.5f rounds up to the next
    // representable float on its own, and there the truncation lands a quantum
    // high. kClamp keeps v well below that, so this is the same number today --
    // it is spelled this way so it stays the same number if kScale grows.
    return (uint32_t)roundf(v * kScale);
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
/// twice this many of them. A wrapped sum reads back as a near-black voxel that
/// every path through it then believes, so the write stops rather than wraps.
///
/// Half the arithmetic bound, and the half is the point: the guard is a plain
/// read followed by four atomic adds, so every thread already past the read when
/// the limit is reached still deposits. On this hardware that is a hundred
/// thousand or so threads, which against a bound sitting exactly at UINT32_MAX
/// would wrap the very sum the guard exists to protect. Halving it costs
/// nothing -- the accumulator is cleared every frame by the resolve pass, so
/// even the reduced bound is a hundred thousand deposits into one voxel in one
/// frame.
constexpr uint32_t kMaxCount = (uint32_t)(4294967295.0 / (double)(kClamp * kScale) / 2.0);

// ---------------------------------------------------------------------------
// Resolve: what a voxel keeps between frames
// ---------------------------------------------------------------------------
//
// Everything above describes one frame's deposits. What follows is the part
// that was missing, and the reason the cache measured nothing: the entry held a
// single running mean and the host wiped the whole table whenever the
// accumulator restarted, which in the editor is every camera movement. A cache
// that starts from nothing on every frame the camera moves is not a cache.
//
// The fix is NVIDIA's, from SHARC v1.8.3 `SharcResolveEntry`: split what a
// voxel gathered *this frame* from what it has resolved *across* frames, merge
// the two once per frame under a bounded window, and let entries nobody visits
// age out on their own. Then a camera movement costs the entries it actually
// invalidates instead of all of them.
//
// Faithful to the SDK in the parts that decide a number -- the window
// normalisation, the staleness rule, the fp16 packing of the resolved half --
// and deliberately not in three others, each noted where it bites:
//   * no responsive-lighting or SH-directional path (both are SDK compile-time
//     options this backend does not set),
//   * no adjacent-level reprojection, which needs a key that carries the voxel
//     position; ours is a 32-bit checksum and cannot be un-hashed (see below),
//   * no linear-probe re-find, which needs a whole entry's neighbours and so
//     cannot live in a pure function of one entry.

/// Bounds on the temporal window, matching SHARC_ACCUMULATED_FRAME_NUM_MIN/MAX.
///
/// The window is what trades quality against response: a voxel that averages
/// over more frames is quieter and slower to notice that the lighting changed.
constexpr uint32_t kAccumFrameNumMin = 1u;
constexpr uint32_t kAccumFrameNumMax = 1024u;

/// Bounds on how long an unvisited entry survives, matching
/// SHARC_STALE_FRAME_NUM_MIN/MAX.
///
/// The minimum is not politeness: evicting aggressively means re-inserting
/// constantly, and the SDK's own note is that a small value costs performance.
constexpr uint32_t kStaleFrameNumMin = 8u;
constexpr uint32_t kStaleFrameNumMax = 1024u;

/// Both frame counters live in one word, sixteen bits each, so the entry stays
/// at 32 bytes. Sixteen bits is far more than either bound above needs.
constexpr uint32_t kFrameNumMask = 0xFFFFu;

/// Largest finite binary16. Values are clamped to it rather than allowed to
/// round up to an infinity, which a resolved voxel would then hand to every
/// path that reads it.
constexpr float kHalfMax = 65504.0f;

STRELKA_SHARC_FN uint32_t floatBits(float value)
{
#if defined(__CUDACC__)
    return __float_as_uint(value);
#else
    uint32_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value), "float is not 32 bits here");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

STRELKA_SHARC_FN float bitsToFloat(uint32_t bits)
{
#if defined(__CUDACC__)
    return __uint_as_float(bits);
#else
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

/// Encode a non-negative, finite value as IEEE binary16.
///
/// Spelled out rather than taken from `__half`, because this header has to give
/// the same answer on the host -- which is where its tests run -- as on the
/// device, and a host build has no CUDA half type. Every quantity stored this
/// way is a radiance or a sample count, so negatives and NaN are folded to
/// zero at the door rather than encoded.
STRELKA_SHARC_FN uint32_t packHalf(float value)
{
    if (!(value > 0.0f))
    {
        return 0u; // catches NaN as well as negatives
    }
    if (value >= kHalfMax)
    {
        return 0x7BFFu; // largest finite binary16
    }
    const uint32_t bits = floatBits(value);
    const int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent <= 0)
    {
        // Subnormal in binary16. Flushed to zero: the values here are radiance
        // and sample counts, and 6e-5 of either is not worth the arithmetic.
        return 0u;
    }
    uint32_t half = ((uint32_t)exponent << 10) | (mantissa >> 13);
    // Round to nearest, ties to even -- the same reason `encode` rounds rather
    // than truncates: a bias of half a quantum, applied every frame in the same
    // direction, is a bias in the image.
    const uint32_t dropped = mantissa & 0x1FFFu;
    if (dropped > 0x1000u || (dropped == 0x1000u && (half & 1u) != 0u))
    {
        ++half;
    }
    // The carry above can reach the exponent, which is intended; it cannot reach
    // infinity, because `value` was already clamped below kHalfMax.
    return half;
}

STRELKA_SHARC_FN float unpackHalf(uint32_t half)
{
    const uint32_t exponent = (half >> 10) & 0x1Fu;
    const uint32_t mantissa = half & 0x3FFu;
    if (exponent == 0u)
    {
        return 0.0f; // zero, or a subnormal packHalf never produces
    }
    return bitsToFloat(((exponent - 15u + 127u) << 23) | (mantissa << 13));
}

/// The half of an entry that survives the frame: a voxel's mean radiance and
/// the sample count behind it, in eight bytes.
///
/// Same four-component fp16 packing as the SDK's `SharcPackedData::radianceData`
/// -- three channels and the count -- because the quantities and their ranges
/// are the same. Eight bytes rather than the twelve a float3 would take is what
/// pays for splitting the entry in two without growing it past 32.
struct Resolved
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    /// Effective number of deposits behind the mean. Fractional, because the
    /// window normalisation scales it down rather than dropping samples.
    float sampleNum = 0.0f;
};

STRELKA_SHARC_FN void packResolved(const Resolved& value, uint32_t& outLo, uint32_t& outHi)
{
    outLo = packHalf(value.r) | (packHalf(value.g) << 16);
    outHi = packHalf(value.b) | (packHalf(value.sampleNum) << 16);
}

STRELKA_SHARC_FN Resolved unpackResolved(uint32_t lo, uint32_t hi)
{
    Resolved value;
    value.r = unpackHalf(lo & 0xFFFFu);
    value.g = unpackHalf(lo >> 16);
    value.b = unpackHalf(hi & 0xFFFFu);
    value.sampleNum = unpackHalf(hi >> 16);
    return value;
}

STRELKA_SHARC_FN uint32_t packFrameData(uint32_t accumFrames, uint32_t staleFrames)
{
    const uint32_t a = accumFrames > kFrameNumMask ? kFrameNumMask : accumFrames;
    const uint32_t s = staleFrames > kFrameNumMask ? kFrameNumMask : staleFrames;
    return a | (s << 16);
}

STRELKA_SHARC_FN void unpackFrameData(uint32_t packed, uint32_t& outAccumFrames, uint32_t& outStaleFrames)
{
    outAccumFrames = packed & kFrameNumMask;
    outStaleFrames = packed >> 16;
}

/// Fold a voxel's data from the adjacent grid level into its own.
///
/// Port of the SDK's SHARC_BLEND_ADJACENT_LEVELS arm. Weighted by the sample
/// counts behind each, which is the same rule the temporal merge uses and for
/// the same reason: the two are estimates of the same quantity, so the one with
/// more samples behind it should weigh more.
///
/// Reached only for an entry that is new -- a couple of frames old at most --
/// because that is the situation it exists for: the camera moved, the level
/// under a point changed, and everything the cache knew about that point is
/// sitting one level away about to age out unread.
STRELKA_SHARC_FN Resolved blendAdjacentLevel(const Resolved& own, const Resolved& adjacent)
{
    const float total = own.sampleNum + adjacent.sampleNum;
    if (!(total > 0.0f))
    {
        return own;
    }
    const float inverseTotal = 1.0f / total;
    Resolved blended;
    blended.r = (own.r * own.sampleNum + adjacent.r * adjacent.sampleNum) * inverseTotal;
    blended.g = (own.g * own.sampleNum + adjacent.g * adjacent.sampleNum) * inverseTotal;
    blended.b = (own.b * own.sampleNum + adjacent.b * adjacent.sampleNum) * inverseTotal;
    // The samples are inherited, not merely borrowed: without this the entry
    // stays below the read threshold and the reprojection buys nothing but a
    // slightly better number nobody is allowed to use yet.
    blended.sampleNum = total;
    return blended;
}

/// How many frames an entry may be old and still take a reprojected blend.
///
/// The SDK's constant, and the reasoning is that reprojection answers "this
/// entry is new *because* the camera moved". An entry that has been accumulating
/// for longer is not that entry, and blending a coarser level into it would only
/// smear it.
constexpr uint32_t kReprojectFrameNumMax = 2u;

/// One entry as the resolve pass sees it.
struct ResolveInput
{
    /// This frame's fixed-point radiance sums and the deposits behind them.
    uint32_t accum[3] = { 0u, 0u, 0u };
    uint32_t accumCount = 0u;
    /// What the entry resolved to at the end of the previous frame.
    uint32_t resolvedLo = 0u;
    uint32_t resolvedHi = 0u;
    uint32_t frameData = 0u;
};

struct ResolveOutput
{
    /// The entry has gone unvisited long enough to be given back to the table.
    /// The caller clears the key, which is what actually frees the slot.
    bool evict = false;
    uint32_t resolvedLo = 0u;
    uint32_t resolvedHi = 0u;
    uint32_t frameData = 0u;
};

/// Merge one frame of deposits into what a voxel already knows.
///
/// A pure function of a single entry, so it can be tested on the host without a
/// GPU -- which is the whole premise of this header, and matters more here than
/// anywhere else in it, because every one of these rules is a silent failure:
/// a window that does not normalise freezes the cache at its first answer, a
/// staleness rule that never fires leaks the table, and one that fires too
/// eagerly re-inserts every entry every frame.
STRELKA_SHARC_FN ResolveOutput resolveEntry(const ResolveInput& input, uint32_t accumFrameNumMax, uint32_t staleFrameNumMax)
{
    ResolveOutput output;

    uint32_t accumFrames = 0u;
    uint32_t staleFrames = 0u;
    unpackFrameData(input.frameData, accumFrames, staleFrames);

    // Staleness counts frames with no new deposit, and resets on any.
    staleFrames = (input.accumCount != 0u) ? 0u : staleFrames + 1u;

    uint32_t staleMax = staleFrameNumMax;
    staleMax = staleMax < kStaleFrameNumMin ? kStaleFrameNumMin : staleMax;
    staleMax = staleMax > kStaleFrameNumMax ? kStaleFrameNumMax : staleMax;

    if (staleFrames >= staleMax)
    {
        // Nobody has looked at this voxel for long enough that its radiance is
        // probably no longer true. Handing the slot back is what lets the table
        // survive a moving camera without the host clearing all of it.
        output.evict = true;
        return output;
    }

    accumFrames += 1u;

    if (input.accumCount == 0u)
    {
        // Aged, but still believed. Keeping the resolved value is the point:
        // a voxel just off screen, or behind the camera for a moment, is still
        // the answer when a path reaches it again.
        output.resolvedLo = input.resolvedLo;
        output.resolvedHi = input.resolvedHi;
        output.frameData = packFrameData(accumFrames, staleFrames);
        return output;
    }

    const Resolved previous = unpackResolved(input.resolvedLo, input.resolvedHi);
    float sampleNumPrev = previous.sampleNum;

    uint32_t window = accumFrameNumMax;
    window = window < kAccumFrameNumMin ? kAccumFrameNumMin : window;
    window = window > kAccumFrameNumMax ? kAccumFrameNumMax : window;

    if (accumFrames > window)
    {
        // The window is what keeps the cache able to change its mind. Without
        // this, the weight of the history grows without bound and a voxel that
        // has been averaging for a thousand frames cannot notice that somebody
        // turned a light on.
        sampleNumPrev *= (float)window / (float)accumFrames;
        accumFrames = window;
    }

    const float sampleNum = (float)input.accumCount;
    const float total = sampleNumPrev + sampleNum;
    const float inverseTotal = total > 0.0f ? 1.0f / total : 0.0f;

    Resolved resolved;
    resolved.r = (previous.r * sampleNumPrev + decode(input.accum[0], input.accumCount) * sampleNum) * inverseTotal;
    resolved.g = (previous.g * sampleNumPrev + decode(input.accum[1], input.accumCount) * sampleNum) * inverseTotal;
    resolved.b = (previous.b * sampleNumPrev + decode(input.accum[2], input.accumCount) * sampleNum) * inverseTotal;
    resolved.sampleNum = total;

    packResolved(resolved, output.resolvedLo, output.resolvedHi);
    output.frameData = packFrameData(accumFrames, staleFrames);
    return output;
}

} // namespace sharc
} // namespace oka
