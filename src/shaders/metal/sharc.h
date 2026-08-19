#pragma once

// Independent Metal implementation of the public SHARC 1.8.3 integration
// contract. No NVIDIA SDK source is included here. The representation follows
// the documented split-resource design: a compact hash table, an atomic update
// buffer, and fp16 persistent resolved data. Update and query never access the
// same radiance storage.

#include <metal_stdlib>

using namespace metal;

constant uint32_t kSharcBucketSize = 16u;
// A lookup gives up after this many empty slots; insertion always claims the
// first empty slot in the bucket, so a sparse bucket cannot hide the key deeper.
constant uint32_t kSharcLimitEmptySlots = 2u;
// Resolve rescues an entry's history from a duplicate key this far ahead of it,
// which is where a collision during insertion would have put the older copy.
constant uint32_t kSharcLinearProbeWindow = 8u;
constant float kSharcMinimumVoxelSize = 1e-5f;
constant float kSharcPositionBias = 1e-4f;
constant float kSharcNormalBias = 1e-3f;

struct SharcHashEntry
{
    atomic_uint key;
};

// The same six 32-bit words carry unsigned RGB in the regular layout and
// signed YCoCg SH coefficients in the directional layout. Atomic integer add
// is modulo 2^32, so the regular path stores uint bit patterns in these words
// and resolves them as uint again. Explicit padding keeps the host-visible
// stride at 32 bytes.
struct SharcAccumulationEntry
{
    atomic_int radiance[3];
    atomic_int direction[3];
    atomic_uint sampleCount;
    // Diagnostic flags occupy the upstream layout's padding word, preserving
    // the 32-byte ABI. They are written only while SHaRC diagnostics are on.
    atomic_uint diagnosticFlags;
};

constant uint32_t kSharcDiagnosticAccumulationClamp = 1u << 0u;
constant uint32_t kSharcDiagnosticNonfiniteReject = 1u << 1u;

// Persistent data is fp16. In the regular layout radiance.xyz stores RGB. In
// the directional layout radiance stores luminance L1/L0 and direction.xy
// stores Co/Cg L0. The remaining words carry temporal state and keep the stride
// at 32 bytes.
struct SharcResolvedEntry
{
    half4 radiance;
    half4 direction;
    uint32_t sampleCount;
    uint32_t accumulatedFramesAndFlags;
    uint32_t staleFrames;
    uint32_t fadeMask;
};

struct SharcStats
{
    atomic_uint insertions;
    atomic_uint insertionFailures;
    atomic_uint queryAttempts;
    atomic_uint queryHits;
    atomic_uint evictions;
    atomic_uint collisions;
    atomic_uint segmentRejects;
    atomic_uint footprintRejects;
};

struct SharcAddress
{
    uint32_t key;
    uint32_t hash;
    float voxelSize;
    float levelBlend;
    int32_t level;
};

static inline uint32_t sharcHash(uint32_t x)
{
    // Jenkins' 32-bit integer hash, used by the public compact-grid contract.
    x = (x + 0x7ed55d16u) + (x << 12u);
    x = (x ^ 0xc761c23cu) ^ (x >> 19u);
    x = (x + 0x165667b1u) + (x << 5u);
    x = (x + 0xd3a2646cu) ^ (x << 9u);
    x = (x + 0xfd7046c5u) + (x << 3u);
    x = (x ^ 0xb55a4f09u) ^ (x >> 16u);
    return x;
}

static inline float3 sharcDebugColorFromHash(uint32_t hash)
{
    // Same bit slices as NVIDIA HashGridGetColorFromHash32(). Keeping all of
    // the available bits avoids the dark, low-entropy 8-bit/channel view.
    return float3(float((hash >> 0u) & 0x3ffu) / 1023.0f, float((hash >> 11u) & 0x7ffu) / 2047.0f,
                  float((hash >> 22u) & 0x7ffu) / 2047.0f);
}

static inline float sharcLuminance(float3 color)
{
    return dot(color, float3(0.213f, 0.715f, 0.072f));
}

static inline float3 sharcRgbToYCoCg(float3 color)
{
    return float3(0.25f * (color.x + 2.0f * color.y + color.z), color.x - color.z, color.y - 0.5f * (color.x + color.z));
}

static inline float3 sharcYCoCgToRgb(float3 color)
{
    return float3(color.x + 0.5f * (color.y - color.z), color.x + 0.5f * color.z, color.x - 0.5f * (color.y + color.z));
}

static inline uint32_t sharcNormalBits(float3 normal)
{
    return (normal.x + kSharcNormalBias < 0.0f ? 1u : 0u) | (normal.y + kSharcNormalBias < 0.0f ? 2u : 0u) |
           (normal.z + kSharcNormalBias < 0.0f ? 4u : 0u);
}

static inline float sharcVoxelSizeForLevel(constant Uniforms& uniforms, int32_t level)
{
    return max(exp2(float(level - uniforms.sharcLevelBias)) / max(uniforms.sharcSceneScale, kSharcMinimumVoxelSize),
               kSharcMinimumVoxelSize);
}

// Compact hash-grid layout: 8 signed bits per axis, 5 for the logarithmic
// level, and 3 for the normal octant. It is the portable Metal alternative to
// the upstream 64-bit key/lock-buffer permutations.
static inline SharcAddress sharcAddressAtLevel(
    constant Uniforms& uniforms, float3 position, float3 normal, int32_t level, bool responsive)
{
    (void)responsive;
    SharcAddress result;
    level = clamp(level, 1, 31);
    result.level = level;
    result.voxelSize = sharcVoxelSizeForLevel(uniforms, level);
    const int3 cell = int3(floor((position + kSharcPositionBias) / result.voxelSize));
    result.key = (uint32_t(cell.x) & 0xffu) | ((uint32_t(cell.y) & 0xffu) << 8u) | ((uint32_t(cell.z) & 0xffu) << 16u) |
                 ((uint32_t(level) & 0x1fu) << 24u) | (sharcNormalBits(normal) << 29u);
    result.hash = sharcHash(result.key);
    result.levelBlend = 0.0f;
    return result;
}

static inline SharcAddress sharcAddress(constant Uniforms& uniforms, float3 position, float3 normal, bool responsive)
{
    const float distanceToCamera =
        max(length(position + kSharcPositionBias - uniforms.viewToWorld[3].xyz), kSharcMinimumVoxelSize);
    const float continuousLevel = clamp(log2(distanceToCamera) + float(uniforms.sharcLevelBias), 1.0f, 31.0f);
    const int32_t level = int32_t(floor(continuousLevel));
    SharcAddress result = sharcAddressAtLevel(uniforms, position, normal, level, responsive);
    result.levelBlend = fract(continuousLevel);
    return result;
}

static inline float3 sharcDebugColoredHash(constant Uniforms& uniforms, float3 position, float3 normal)
{
    // Metal equivalent of NVIDIA HashGridDebugColoredHash(): color the compact
    // spatial key, then modulate it with a second hash of the logarithmic level.
    // It deliberately touches no cache buffer, so the grid is inspectable with
    // SHaRC disabled and exposes only address generation.
    const SharcAddress address = sharcAddress(uniforms, position, normal, false);
    return sharcDebugColorFromHash(address.hash) * sharcDebugColorFromHash(sharcHash(uint32_t(address.level)));
}

static inline uint32_t sharcBucketBase(uint32_t hash, uint32_t capacity);
static inline uint32_t sharcMainCapacity(constant Uniforms& uniforms);

static inline float3 sharcDebugOccupancy(uint32_t pixelIndex,
                                         constant Uniforms& uniforms,
                                         device SharcHashEntry* hashEntries,
                                         device const SharcResolvedEntry* resolvedEntries)
{
    // Metal port of HashGridDebugOccupancy(). Each 7x7 square visualizes one
    // table entry, and columns are grouped by the 16-entry probe bucket.
    constexpr uint32_t elementSize = 7u;
    constexpr uint32_t blockSize = 8u;
    const uint2 pixelPosition = uint2(pixelIndex % uniforms.width, pixelIndex / uniforms.width);
    const uint32_t rowCount = max(uniforms.height / blockSize, 1u);
    const uint32_t rowIndex = pixelPosition.y / blockSize;
    const uint32_t columnIndex = pixelPosition.x / blockSize;
    const uint32_t elementIndex = (columnIndex / kSharcBucketSize) * (rowCount * kSharcBucketSize) +
                                  rowIndex * kSharcBucketSize + (columnIndex % kSharcBucketSize);
    if (elementIndex < uniforms.sharcCapacity && pixelPosition.x % blockSize < elementSize &&
        pixelPosition.y % blockSize < elementSize)
    {
        const uint32_t key = atomic_load_explicit(&hashEntries[elementIndex].key, memory_order_relaxed);
        if (key != 0u)
        {
            // Green once the entry has something to answer a query with, amber
            // while it is only reserved: a table that stays amber is being
            // evicted before it ever resolves.
            return resolvedEntries[elementIndex].sampleCount >= max(uniforms.sharcMinSamples, 1u) ?
                       float3(0.0f, 1.0f, 0.0f) :
                       float3(1.0f, 0.75f, 0.0f);
        }
    }
    return float3(0.0f);
}

static inline bool sharcFindEntry(constant Uniforms& uniforms,
                                  device SharcHashEntry* hashEntries,
                                  SharcAddress address,
                                  bool responsiveEntry,
                                  bool insert,
                                  device atomic_uint* stats,
                                  thread uint32_t& outIndex,
                                  thread uint32_t& outBucketOffset);

static inline float3 sharcDebugCollisions(constant Uniforms& uniforms,
                                          device SharcHashEntry* hashEntries,
                                          float3 position,
                                          float3 normal)
{
    // Same bucket-offset palette as HashGridDebugHashCollisions(). Blue is the
    // base slot; cyan/green/yellow/orange/red indicate progressively deeper
    // probes, with red also covering a missing key.
    const SharcAddress address = sharcAddress(uniforms, position, normal, false);
    uint32_t index = 0u;
    uint32_t bucketOffset = kSharcBucketSize;
    sharcFindEntry(uniforms, hashEntries, address, false, false, nullptr, index, bucketOffset);
    if (bucketOffset == 0u)
    {
        return float3(0.0f, 0.0f, 1.0f);
    }
    if (bucketOffset == 1u)
    {
        return float3(0.0f, 0.5f, 0.5f);
    }
    if (bucketOffset == 2u)
    {
        return float3(0.0f, 1.0f, 0.0f);
    }
    if (bucketOffset == 3u)
    {
        return float3(1.0f, 1.0f, 0.0f);
    }
    if (bucketOffset == 4u)
    {
        return float3(0.75f, 0.25f, 0.0f);
    }
    return float3(1.0f, 0.0f, 0.0f);
}

static inline SharcAddress sharcAdjacentLevelAddress(constant Uniforms& uniforms, uint32_t key)
{
    int3 gridPosition = int3(int32_t(key & 0xffu), int32_t((key >> 8u) & 0xffu), int32_t((key >> 16u) & 0xffu));
    gridPosition = (gridPosition << 24) >> 24;
    int32_t level = int32_t((key >> 24u) & 0x1fu);
    const float voxelSize = sharcVoxelSizeForLevel(uniforms, level);
    const float3 cameraPosition = uniforms.viewToWorld[3].xyz;
    const float3 previousCameraPosition = float3(uniforms.sharcCameraPrev);
    const float3 cameraGrid = floor(cameraPosition / voxelSize);
    const float3 previousCameraGrid = floor(previousCameraPosition / voxelSize);
    const float3 currentOffset = cameraGrid - float3(gridPosition);
    const float3 previousOffset = previousCameraGrid - float3(gridPosition);
    const float currentDistance2 = dot(currentOffset, currentOffset);
    const float previousDistance2 = dot(previousOffset, previousOffset);
    if (currentDistance2 < previousDistance2)
    {
        gridPosition = int3(floor(float3(gridPosition) * 0.5f));
        level = min(level + 1, 31);
    }
    else
    {
        gridPosition *= 2;
        level = max(level - 1, 1);
    }

    SharcAddress result;
    result.key = (uint32_t(gridPosition.x) & 0xffu) | ((uint32_t(gridPosition.y) & 0xffu) << 8u) |
                 ((uint32_t(gridPosition.z) & 0xffu) << 16u) | ((uint32_t(level) & 0x1fu) << 24u) | (key & 0xe0000000u);
    result.hash = sharcHash(result.key);
    result.voxelSize = sharcVoxelSizeForLevel(uniforms, level);
    result.levelBlend = 0.0f;
    result.level = level;
    return result;
}

static inline uint32_t sharcBucketBase(uint32_t hash, uint32_t capacity)
{
    const uint32_t bucketRange = max(capacity - kSharcBucketSize + 1u, 1u);
    return hash % bucketRange;
}

// Compact keys consume all 32 bits, so there is no collision-free marker bit
// for responsive entries. When responsive lighting is enabled, split the table
// into two disjoint regions that use the same spatial key: persistent entries
// occupy the first half and responsive companions the second half.
static inline uint32_t sharcMainCapacity(constant Uniforms& uniforms)
{
    return (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u ? uniforms.sharcCapacity / 2u : uniforms.sharcCapacity;
}

// One bucket probe, shared by insertion, lookup and the collision view.
//
// Insertion compare-exchanges the key into every slot in turn: an occupied slot
// simply fails the exchange, and the first empty one wins. Lookup relies on that
// invariant to stop early -- once the bucket has shown kSharcLimitEmptySlots
// empty slots the key cannot be deeper in the chain, because insertion would
// have taken one of them.
static inline bool sharcFindRange(device SharcHashEntry* hashEntries,
                                  uint32_t rangeOffset,
                                  uint32_t rangeCapacity,
                                  SharcAddress address,
                                  bool insert,
                                  device atomic_uint* stats,
                                  thread uint32_t& outIndex,
                                  thread uint32_t& outBucketOffset)
{
    outBucketOffset = kSharcBucketSize;
    if (rangeCapacity < kSharcBucketSize)
    {
        return false;
    }
    const uint32_t base = rangeOffset + sharcBucketBase(address.hash, rangeCapacity);
    uint32_t emptySlots = 0u;
    for (uint32_t probe = 0u; probe < kSharcBucketSize; ++probe)
    {
        const uint32_t index = base + probe;
        device atomic_uint* slot = &hashEntries[index].key;
        if (insert)
        {
            // Metal has no strong compare-exchange, and a weak one may fail
            // spuriously. Retry the same slot while it still reads empty:
            // stepping to the next probe with this one unclaimed would publish
            // the same key twice and split one voxel's temporal history.
            uint32_t previous = 0u;
            bool exchanged = false;
            for (uint32_t attempt = 0u; attempt < 4u; ++attempt)
            {
                previous = 0u;
                exchanged = atomic_compare_exchange_weak_explicit(
                    slot, &previous, address.key, memory_order_relaxed, memory_order_relaxed);
                if (exchanged || previous != 0u)
                {
                    break;
                }
            }
            if (exchanged || previous == address.key)
            {
                outIndex = index;
                outBucketOffset = probe;
                if (stats && exchanged)
                {
                    atomic_fetch_add_explicit(&stats[SHARC_STAT_INSERTION], 1u, memory_order_relaxed);
                }
                return true;
            }
            if (stats)
            {
                atomic_fetch_add_explicit(&stats[SHARC_STAT_COLLISION], 1u, memory_order_relaxed);
            }
            continue;
        }
        const uint32_t key = atomic_load_explicit(slot, memory_order_relaxed);
        if (key == address.key)
        {
            outIndex = index;
            outBucketOffset = probe;
            return true;
        }
        if (key == 0u)
        {
            if (emptySlots >= kSharcLimitEmptySlots)
            {
                break;
            }
            ++emptySlots;
            continue;
        }
        if (stats)
        {
            atomic_fetch_add_explicit(&stats[SHARC_STAT_COLLISION], 1u, memory_order_relaxed);
        }
    }
    if (insert && stats)
    {
        atomic_fetch_add_explicit(&stats[SHARC_STAT_INSERTION_FAILURE], 1u, memory_order_relaxed);
    }
    return false;
}

static inline bool sharcFindEntry(constant Uniforms& uniforms,
                                  device SharcHashEntry* hashEntries,
                                  SharcAddress address,
                                  bool responsiveEntry,
                                  bool insert,
                                  device atomic_uint* stats,
                                  thread uint32_t& outIndex,
                                  thread uint32_t& outBucketOffset)
{
    const uint32_t mainCapacity = sharcMainCapacity(uniforms);
    const uint32_t offset = responsiveEntry ? mainCapacity : 0u;
    const uint32_t capacity = responsiveEntry ? uniforms.sharcCapacity - mainCapacity : mainCapacity;
    return sharcFindRange(hashEntries, offset, capacity, address, insert, stats, outIndex, outBucketOffset);
}

static inline bool sharcFindEntry(constant Uniforms& uniforms,
                                  device SharcHashEntry* hashEntries,
                                  SharcAddress address,
                                  bool responsiveEntry,
                                  bool insert,
                                  device atomic_uint* stats,
                                  thread uint32_t& outIndex)
{
    uint32_t bucketOffset = 0u;
    return sharcFindEntry(uniforms, hashEntries, address, responsiveEntry, insert, stats, outIndex, bucketOffset);
}

static inline float3 sharcMaterialDemodulation(float3 diffuseAlbedo, float3 specularF0)
{
    const float specularEnergy = sharcLuminance(specularF0 + (1.0f - specularF0) / 21.0f);
    return max(diffuseAlbedo, float3(0.05f)) + max(specularF0, float3(0.02f)) * specularEnergy;
}

static inline float3 sharcDecode(thread const SharcResolvedEntry& entry, float3 direction, bool directional)
{
    if (!directional)
    {
        return max(float3(entry.radiance.xyz), float3(0.0f));
    }
    // NVIDIA SHARC 1.8.3 directional representation: first-order luminance
    // moment plus L0 luminance and YCoCg chroma. Chroma follows the decoded
    // luminance, preventing a bright colored specular sample from being reused
    // in an unrelated direction.
    const float3 luminanceMoment = float3(entry.radiance.xyz);
    const float luminanceL0 = float(entry.radiance.w);
    const float directionalLuminance = min(length(luminanceMoment), max(luminanceL0, 0.0f));
    const float diffuseLuminance = max(luminanceL0 - directionalLuminance, 0.0f);
    const float luminance = diffuseLuminance + max(dot(luminanceMoment, normalize(direction)), 0.0f);
    const float chromaScale = luminanceL0 > 1e-6f ? luminance / luminanceL0 : 0.0f;
    const float2 chroma = float2(entry.direction.xy) * chromaScale;
    return max(sharcYCoCgToRgb(float3(luminance, chroma)), float3(0.0f));
}

static inline bool sharcQuerySingle(constant Uniforms& uniforms,
                                    device SharcHashEntry* hashEntries,
                                    device const SharcResolvedEntry* resolvedEntries,
                                    device atomic_uint* stats,
                                    SharcAddress address,
                                    bool responsiveEntry,
                                    float3 direction,
                                    thread float3& radiance,
                                    thread uint32_t& sampleCount)
{
    uint32_t index = 0u;
    if (!sharcFindEntry(uniforms, hashEntries, address, responsiveEntry, false, stats, index))
    {
        return false;
    }
    const SharcResolvedEntry entry = resolvedEntries[index];
    sampleCount = entry.sampleCount;
    if (sampleCount < uniforms.sharcMinSamples)
    {
        return false;
    }
    radiance = sharcDecode(entry, direction, (uniforms.sharcFlags & SHARC_FLAG_DIRECTIONAL) != 0u);
    return all(isfinite(radiance));
}

static inline bool sharcQuery(constant Uniforms& uniforms,
                              device SharcHashEntry* hashEntries,
                              device const SharcResolvedEntry* resolvedEntries,
                              device atomic_uint* stats,
                              float3 position,
                              float3 normal,
                              float3 direction,
                              float3 materialDemodulation,
                              thread float3& radiance,
                              thread uint32_t& sampleCount)
{
    if (stats)
    {
        atomic_fetch_add_explicit(&stats[SHARC_STAT_QUERY_ATTEMPT], 1u, memory_order_relaxed);
    }
    const bool responsive = (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u;
    const SharcAddress address = sharcAddress(uniforms, position, normal, false);
    float3 fine = float3(0.0f);
    uint32_t fineCount = 0u;
    const bool found =
        sharcQuerySingle(uniforms, hashEntries, resolvedEntries, stats, address, false, direction, fine, fineCount);
    // Reported even on a miss: a cell that exists but has not gathered enough
    // samples yet is what the sample-count diagnostic is for.
    sampleCount = fineCount;
    if (!found)
    {
        return false;
    }
    if (responsive)
    {
        const SharcAddress responsiveAddress = sharcAddressAtLevel(uniforms, position, normal, address.level, true);
        float3 responsiveRadiance = float3(0.0f);
        uint32_t responsiveCount = 0u;
        if (sharcQuerySingle(uniforms, hashEntries, resolvedEntries, stats, responsiveAddress, true, direction,
                             responsiveRadiance, responsiveCount))
        {
            fine += responsiveRadiance;
        }
    }

    radiance = fine;
    sampleCount = fineCount;
    if ((uniforms.sharcFlags & SHARC_FLAG_MATERIAL_DEMODULATION) != 0u)
    {
        radiance *= materialDemodulation;
    }
    if (stats)
    {
        atomic_fetch_add_explicit(&stats[SHARC_STAT_QUERY_HIT], 1u, memory_order_relaxed);
    }
    return true;
}

// Rounded, not truncated. Truncation biases every deposit towards zero by half a
// quantum, which the OptiX port measured as a quarter of a percent of systematic
// darkening for as long as the cache is on -- `00_calibration` and
// `02_basecolor` read 0.997 truncating and 1.000 rounding.
static inline int32_t sharcFixed(float value, float scale)
{
    const float limit = 2147480000.0f;
    return int32_t(clamp(round(value * scale), -limit, limit));
}

static inline uint32_t sharcUnsignedFixed(float value, float scale)
{
    const float limit = 4294960000.0f;
    return uint32_t(clamp(round(value * scale), 0.0f, limit));
}

static inline void sharcAtomicMax(device atomic_uint* destination, uint32_t value)
{
    uint32_t previous = atomic_load_explicit(destination, memory_order_relaxed);
    while (previous < value && !atomic_compare_exchange_weak_explicit(
                                   destination, &previous, value, memory_order_relaxed, memory_order_relaxed))
    {
    }
}

static inline uint32_t sharcSignedMagnitude(int32_t value)
{
    const uint32_t sign = uint32_t(value >> 31);
    return (uint32_t(value) ^ sign) - sign;
}

static inline void sharcAccumulate(device SharcAccumulationEntry* accumulationEntries,
                                   uint32_t index,
                                   float3 radiance,
                                   float3 direction,
                                   float directionWeight,
                                   bool addSample,
                                   constant Uniforms& uniforms)
{
    if (index == SHARC_NO_ENTRY)
    {
        return;
    }
    device SharcAccumulationEntry& entry = accumulationEntries[index];
    if (!all(isfinite(radiance)))
    {
        if (uniforms.sharcDebug != 0u)
        {
            atomic_fetch_or_explicit(&entry.diagnosticFlags, kSharcDiagnosticNonfiniteReject, memory_order_relaxed);
        }
        return;
    }
    const float scale = max(uniforms.sharcRadianceScale, 1.0f);
    const bool directional = (uniforms.sharcFlags & SHARC_FLAG_DIRECTIONAL) != 0u;
    // Match SHARC's 32-bit fixed-point layouts. Only guard the numeric
    // conversion boundary; the original implementation deliberately relies on
    // radianceScale, rather than a per-sample radiance heuristic, to keep sums
    // from overflowing the accumulator.
    const float accumulationLimit = (directional ? 2147480000.0f : 4294960000.0f) / scale;
    if (uniforms.sharcDebug != 0u && any(radiance > accumulationLimit))
    {
        atomic_fetch_or_explicit(&entry.diagnosticFlags, kSharcDiagnosticAccumulationClamp, memory_order_relaxed);
    }
    // Adding zero is still an atomic. Most deposits have at least one channel
    // that quantizes to nothing -- and a vertex whose own radiance is entirely
    // deferred, which is every hit under separate emissive, has three -- so the
    // upstream guard is worth keeping.
    if (directional)
    {
        const float3 ycocg = sharcRgbToYCoCg(radiance);
        const float3 luminanceMoment = normalize(direction) * ycocg.x * saturate(directionWeight);
        for (uint32_t channel = 0u; channel < 3u; ++channel)
        {
            const int32_t value = sharcFixed(luminanceMoment[channel], scale);
            if (value != 0)
            {
                atomic_fetch_add_explicit(&entry.radiance[channel], value, memory_order_relaxed);
            }
        }
        for (uint32_t channel = 0u; channel < 3u; ++channel)
        {
            const int32_t value = sharcFixed(ycocg[channel], scale);
            if (value != 0)
            {
                atomic_fetch_add_explicit(&entry.direction[channel], value, memory_order_relaxed);
            }
        }
    }
    else
    {
        for (uint32_t channel = 0u; channel < 3u; ++channel)
        {
            const uint32_t value = sharcUnsignedFixed(radiance[channel], scale);
            if (value != 0u)
            {
                atomic_fetch_add_explicit(&entry.radiance[channel], as_type<int32_t>(value), memory_order_relaxed);
            }
        }
    }
    if (addSample)
    {
        atomic_fetch_add_explicit(&entry.sampleCount, 1u, memory_order_relaxed);
    }
}

static inline void sharcInitUpdateState(thread SharcUpdateState& state, uint32_t pixelIndex)
{
    for (uint32_t i = 0u; i < SHARC_MAX_PROPAGATION_DEPTH; ++i)
    {
        state.cacheIndices[i] = SHARC_NO_ENTRY;
        state.responsiveIndices[i] = SHARC_NO_ENTRY;
        state.weights[i] = packed_float3(0.0f);
        state.directions[i] = packed_float3(0.0f);
        state.directionWeights[i] = 0.0f;
    }
    state.pendingThroughput = packed_float3(1.0f);
    state.pixelIndex = pixelIndex;
    state.pathLength = 0u;
    state.flags = 0u;
}

static inline void sharcApplyPendingThroughput(thread SharcUpdateState& state, constant Uniforms& uniforms)
{
    const float3 throughput = float3(state.pendingThroughput);
    const uint32_t depth = min(uniforms.sharcPropagationDepth, uint32_t(SHARC_MAX_PROPAGATION_DEPTH));
    for (uint32_t i = 0u; i < depth; ++i)
    {
        state.weights[i] = packed_float3(float3(state.weights[i]) * throughput);
    }
    state.pendingThroughput = packed_float3(1.0f);
}

static inline void sharcSetThroughput(thread SharcUpdateState& state, float3 throughput)
{
    state.pendingThroughput = packed_float3(throughput);
}

static inline void sharcSetRadianceDirectionWeight(thread SharcUpdateState& state, float directionWeight)
{
    if (state.pathLength != 0u)
    {
        state.directionWeights[0] = saturate(directionWeight);
    }
}

static inline void sharcMultiplyPendingThroughput(thread SharcUpdateState& state, float3 throughput)
{
    state.pendingThroughput = packed_float3(float3(state.pendingThroughput) * throughput);
}

static inline void sharcPropagate(thread const SharcUpdateState& state,
                                  device SharcAccumulationEntry* accumulationEntries,
                                  float3 localRadiance,
                                  constant Uniforms& uniforms,
                                  bool responsive)
{
    const uint32_t depth = min(uniforms.sharcPropagationDepth, uint32_t(SHARC_MAX_PROPAGATION_DEPTH));
    for (uint32_t i = 0u; i < depth; ++i)
    {
        const bool useResponsive = responsive && state.responsiveIndices[i] != SHARC_NO_ENTRY;
        const uint32_t index = useResponsive ? state.responsiveIndices[i] : state.cacheIndices[i];
        if (index == SHARC_NO_ENTRY)
        {
            continue;
        }
        sharcAccumulate(accumulationEntries, index, localRadiance * float3(state.weights[i]),
                        float3(state.directions[i]), state.directionWeights[i], false, uniforms);
    }
}

static inline bool sharcUpdateHit(thread SharcUpdateState& state,
                                  constant Uniforms& uniforms,
                                  device SharcHashEntry* hashEntries,
                                  device SharcAccumulationEntry* accumulationEntries,
                                  device const SharcResolvedEntry* resolvedEntries,
                                  device atomic_uint* stats,
                                  float3 position,
                                  float3 normal,
                                  float3 incidentDirection,
                                  float directionWeight,
                                  float3 materialDemodulation,
                                  float3 directLighting,
                                  float3 emissive,
                                  float randomValue,
                                  bool responsive)
{
    sharcApplyPendingThroughput(state, uniforms);
    const bool separateEmissive = (uniforms.sharcFlags & SHARC_FLAG_SEPARATE_EMISSIVE) != 0u;
    // What this vertex stores for itself. With separate emissive the cache holds
    // no emission at all: queries add the surface's own emission back, which is
    // what keeps an emitter sharp and lets it change without a cache reset.
    const float3 localRadiance = directLighting + (separateEmissive ? float3(0.0f) : emissive);
    float3 propagatedRadiance = localRadiance;
    bool continueTracing = true;
    const uint32_t resamplingDepth =
        uint32_t(round(mix(1.0f, float(max(uniforms.sharcPropagationDepth, 1u)), saturate(randomValue))));
    if ((uniforms.sharcFlags & SHARC_FLAG_CACHE_RESAMPLING) != 0u && resamplingDepth <= state.pathLength)
    {
        float3 cached = float3(0.0f);
        uint32_t count = 0u;
        if (sharcQuery(uniforms, hashEntries, resolvedEntries, nullptr, position, normal, incidentDirection,
                       materialDemodulation, cached, count))
        {
            // The cached value already carries everything this vertex reflects,
            // including its emission when the cache stores it.
            propagatedRadiance = cached;
            continueTracing = false;
        }
    }

    if (separateEmissive)
    {
        propagatedRadiance += emissive;
    }
    sharcPropagate(state, accumulationEntries, propagatedRadiance, uniforms, responsive);

    const uint32_t depth = min(uniforms.sharcPropagationDepth, uint32_t(SHARC_MAX_PROPAGATION_DEPTH));
    if (depth == 0u)
    {
        return continueTracing;
    }
    for (uint32_t i = depth - 1u; i > 0u; --i)
    {
        state.cacheIndices[i] = state.cacheIndices[i - 1u];
        state.responsiveIndices[i] = state.responsiveIndices[i - 1u];
        state.weights[i] = state.weights[i - 1u];
        state.directions[i] = state.directions[i - 1u];
        state.directionWeights[i] = state.directionWeights[i - 1u];
    }
    const SharcAddress address = sharcAddress(uniforms, position, normal, false);
    uint32_t index = SHARC_NO_ENTRY;
    if (!sharcFindEntry(uniforms, hashEntries, address, false, true, stats, index))
    {
        state.cacheIndices[0] = SHARC_NO_ENTRY;
        state.responsiveIndices[0] = SHARC_NO_ENTRY;
        state.weights[0] = packed_float3(0.0f);
        state.directions[0] = packed_float3(0.0f);
        state.directionWeights[0] = 0.0f;
        return false;
    }

    uint32_t responsiveIndex = SHARC_NO_ENTRY;
    if (responsive)
    {
        const SharcAddress responsiveAddress = sharcAddress(uniforms, position, normal, true);
        if (!sharcFindEntry(uniforms, hashEntries, responsiveAddress, true, true, stats, responsiveIndex))
        {
            return false;
        }
    }
    if (continueTracing)
    {
        const float3 demodulated = ((uniforms.sharcFlags & SHARC_FLAG_MATERIAL_DEMODULATION) != 0u) ?
                                       localRadiance / max(materialDemodulation, float3(1e-3f)) :
                                       localRadiance;
        if (responsiveIndex != SHARC_NO_ENTRY)
        {
            // Keep the persistent/main entry alive while placing the changing
            // signal in the short-history companion entry. Query adds both.
            sharcAccumulate(accumulationEntries, index, float3(0.0f), incidentDirection, 0.0f, true, uniforms);
            sharcAccumulate(
                accumulationEntries, responsiveIndex, demodulated, incidentDirection, directionWeight, true, uniforms);
        }
        else
        {
            sharcAccumulate(accumulationEntries, index, demodulated, incidentDirection, directionWeight, true, uniforms);
        }
    }
    state.cacheIndices[0] = index;
    state.responsiveIndices[0] = responsiveIndex;
    const float3 demod = (uniforms.sharcFlags & SHARC_FLAG_MATERIAL_DEMODULATION) != 0u ?
                             max(materialDemodulation, float3(1e-3f)) :
                             float3(1.0f);
    state.weights[0] = packed_float3(1.0f / demod);
    state.directions[0] = packed_float3(normalize(incidentDirection));
    state.directionWeights[0] = saturate(directionWeight);
    state.pathLength = min(state.pathLength + 1u, depth);
    return continueTracing;
}

static inline void sharcUpdateMiss(thread SharcUpdateState& state,
                                   constant Uniforms& uniforms,
                                   device SharcAccumulationEntry* accumulationEntries,
                                   float3 environmentRadiance,
                                   float3 direction)
{
    (void)direction;
    sharcApplyPendingThroughput(state, uniforms);
    const bool responsive = (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u;
    sharcPropagate(state, accumulationEntries, environmentRadiance, uniforms, responsive);
}

static inline float3 sharcResolveDirection(float3 directionMoment)
{
    // Keep the first-order luminance moment in radiance units. Normalizing it
    // before temporal blending would make dark and bright samples contribute
    // equally and breaks directional reconstruction.
    return clamp(directionMoment, float3(-65504.0f), float3(65504.0f));
}

// What a query would answer at this surface, which is the cross-backend
// `DebugMode::eSharcRadiance` view. Black where the voxel is missing or has not
// resolved yet.
static inline float3 sharcDebugRadiance(constant Uniforms& uniforms,
                                        device SharcHashEntry* hashEntries,
                                        device const SharcResolvedEntry* resolvedEntries,
                                        float3 position,
                                        float3 normal,
                                        float3 direction,
                                        float3 materialDemodulation)
{
    float3 radiance = float3(0.0f);
    uint32_t sampleCount = 0u;
    if (!sharcQuery(uniforms, hashEntries, resolvedEntries, nullptr, position, normal, direction, materialDemodulation,
                    radiance, sampleCount))
    {
        return float3(0.0f);
    }
    return radiance;
}

// The Metal-only diagnostics, all evaluated at one surface: what the hash map
// did, rather than what the cache holds.
static inline float3 sharcDebugSurface(constant Uniforms& uniforms,
                                       device SharcHashEntry* hashEntries,
                                       device const SharcResolvedEntry* resolvedEntries,
                                       float3 position,
                                       float3 normal,
                                       float3 direction,
                                       float3 materialDemodulation)
{
    if (uniforms.sharcDebug == SHARC_DEBUG_COLLISIONS)
    {
        return sharcDebugCollisions(uniforms, hashEntries, position, normal);
    }
    const SharcAddress address = sharcAddress(uniforms, position, normal, false);
    float3 radiance = float3(0.0f);
    uint32_t sampleCount = 0u;
    const bool hit = sharcQuery(uniforms, hashEntries, resolvedEntries, nullptr, position, normal, direction,
                                materialDemodulation, radiance, sampleCount);
    if (uniforms.sharcDebug == SHARC_DEBUG_QUERY_RESULT)
    {
        return hit ? float3(0.1f, 0.9f, 0.2f) : float3(0.9f, 0.1f, 0.1f);
    }
    if (uniforms.sharcDebug == SHARC_DEBUG_SAMPLE_COUNT)
    {
        // Blue is empty, white is warm; the square keeps the low end readable.
        const float v = min(float(sampleCount) / 32.0f, 1.0f);
        return float3(v, v * v, 1.0f - v);
    }
    // SHARC_DEBUG_CACHED_KEY: the resident cells, coloured by the key that found
    // them, so a cell that moved slot between frames changes colour.
    uint32_t index = 0u;
    if (!sharcFindEntry(uniforms, hashEntries, address, false, false, nullptr, index))
    {
        return float3(0.0f);
    }
    return sharcDebugColorFromHash(sharcHash(address.key ^ index));
}

// How deep a path actually went, as the panel describes it: blue none, green
// one, yellow two, red three or more. Comparing it with the cache off and on is
// the direct measurement of what the cache buys, and the only one that says
// where.
static inline float3 sharcDebugBounceColor(uint32_t depth)
{
    if (depth == 0u)
    {
        return float3(0.1f, 0.2f, 0.9f);
    }
    if (depth == 1u)
    {
        return float3(0.1f, 0.9f, 0.2f);
    }
    if (depth == 2u)
    {
        return float3(0.95f, 0.9f, 0.1f);
    }
    return float3(0.95f, 0.15f, 0.1f);
}
