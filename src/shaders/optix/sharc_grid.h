#pragma once

// Host-testable SHARC grid arithmetic. Its numeric contracts stay aligned with
// src/shaders/metal/sharc.h; src/shaders/optix/sharc.h adds CUDA atomics.

#include <cstdint>

// NOLINTBEGIN(cppcoreguidelines-init-variables)

#if defined(__CUDACC__)
#    define STRELKA_SHARC_FN static __forceinline__ __device__
#else
#    include <cmath>
#    include <cstring>
#    define STRELKA_SHARC_FN inline
#endif

#if defined(__CUDACC__)
namespace oka
{
namespace sharc
{
#else
namespace oka::sharc
{
#endif

constexpr float kScale = 64.0f;

/// The clamp rejects fireflies while preserving legitimate outgoing radiance.
constexpr float kClamp = 256.0f;

/// How far the linear probe walks before it gives up. A miss at the end of it
/// means the table is over-subscribed and the caller simply carries on tracing.
constexpr uint32_t kProbeCount = 8u;

/// Never fewer than this many entries, so that a capacity somebody typed in a
/// config cannot turn the cache into a single contended slot.
constexpr uint32_t kMinCapacity = 1u << 16;

constexpr uint32_t kUpdateShare = 8u;

/// Below this throughput, division at deposit produces excessive variance.
constexpr float kMinRecordThroughput = 0.05f;

/// Floor on the per-channel throughput the deposit divides by, so one dark
/// channel cannot turn a finite estimate into a huge one.
constexpr float kThroughputFloor = 0.02f;

STRELKA_SHARC_FN bool mayReadCache(float segmentLength, float launchRoughness, float voxelSize)
{
    // sqrt(3) is the voxel diagonal: a segment shorter than that may not have
    // left the voxel at all.
    if (!(segmentLength > voxelSize * 1.7320508f))
    {
        return false;
    }
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

struct Voxel
{
    float size = 1.0f;
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
    const float cell = floorf(coordinate / size);
#else
    const float cell = std::floor(coordinate / size);
#endif
    if (cell != cell)
    {
        return 0;
    }
    if (cell <= -2147483648.0f)
    {
        return (-2147483647 - 1);
    }
    if (cell >= 2147483648.0f)
    {
        return 2147483647;
    }
    return (int32_t)cell;
}

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

    const int64_t dx = (int64_t)cx - (int64_t)x;
    const int64_t dy = (int64_t)cy - (int64_t)y;
    const int64_t dz = (int64_t)cz - (int64_t)z;
    const int64_t qx = (int64_t)px - (int64_t)x;
    const int64_t qy = (int64_t)py - (int64_t)y;
    const int64_t qz = (int64_t)pz - (int64_t)z;
    const uint64_t distance = (uint64_t)(dx * dx) + (uint64_t)(dy * dy) + (uint64_t)(dz * dz);
    const uint64_t distancePrev = (uint64_t)(qx * qx) + (uint64_t)(qy * qy) + (uint64_t)(qz * qz);

    if (distance < distancePrev)
    {
        x >>= 1;
        y >>= 1;
        z >>= 1;
        level = level + 1 > kLevelMax ? kLevelMax : level + 1;
    }
    else
    {
        x *= 2;
        y *= 2;
        z *= 2;
        level = level - 1 < kLevelMin ? kLevelMin : level - 1;
    }

    // The normal bucket and the responsive flag carry over: this is the same
    // surface, seen at a different resolution.
    return voxelKey(x, y, z, level, unpackBucket(key), isResponsiveKey(key));
}

/// Whether this path is one of the fixed share that records rather than reads.
///
/// A path either reads or records, never both, preventing cache feedback.
STRELKA_SHARC_FN bool isUpdatePath(uint32_t pixelIndex, uint32_t sampleIndex)
{
    return (hash(pixelIndex * 9781u + sampleIndex * 6271u) & (kUpdateShare - 1u)) == 0u;
}

/// Encode one channel of radiance for the accumulator.
///
/// Rounded rather than truncated so quantisation error is zero-mean.
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

constexpr uint32_t kMaxCount = (uint32_t)(4294967295.0 / (double)(kClamp * kScale) / 2.0);

constexpr uint32_t kAccumFrameNumMin = 1u;
constexpr uint32_t kAccumFrameNumMax = 1024u;

constexpr uint32_t kStaleFrameNumMin = 8u;
constexpr uint32_t kStaleFrameNumMax = 1024u;

/// Both frame counters live in one word, sixteen bits each.
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

#if defined(__CUDACC__)
} // namespace sharc
} // namespace oka
#else
} // namespace oka::sharc
#endif

// NOLINTEND(cppcoreguidelines-init-variables)
