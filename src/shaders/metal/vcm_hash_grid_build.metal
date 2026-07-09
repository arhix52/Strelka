// ============================================================================
// vcm_hash_grid_build.metal -- VCM Hash Grid Build Kernel
//
// Inserts non-delta light subpath vertices into the spatial hash grid
// for range queries during vertex merging.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"
#include "bdpt_types.h"
#include "vcm_types.h"

// Teschner spatial hash (Teschner et al. 2003)
static inline uint32_t teschnerHash(int3 cell)
{
    uint32_t h = (uint32_t)(cell.x * 73856093) ^
                 (uint32_t)(cell.y * 19349663) ^
                 (uint32_t)(cell.z * 83492791);
    return h & (VCM_HASH_SIZE - 1);
}

kernel void vcm_hash_grid_build(
    uint2                       tid              [[thread_position_in_grid]],
    constant Uniforms&          uniforms         [[buffer(0)]],
    device const BDPTVertex*    lightVertices    [[buffer(1)]],
    device const uint32_t*      lightPathLengths [[buffer(2)]],
    device uint32_t*            hashHeads        [[buffer(3)]],
    device VCMHashEntry*        hashEntries      [[buffer(4)]],
    device atomic_uint*         hashCounter      [[buffer(5)]]
)
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    const uint32_t linearPixelIndex = tid.y * uniforms.width + tid.x;
    const uint32_t pathLen = lightPathLengths[linearPixelIndex];
    const float cellSize = uniforms.vcmHashCellSize;

    if (cellSize <= 0.0f)
        return;

    const float invCellSize = 1.0f / cellSize;

    // Insert each non-delta, non-camera light vertex (skip vertex 0 = on light surface)
    for (uint32_t d = 1; d < pathLen; ++d)
    {
        const uint32_t vertIdx = linearPixelIndex * uniforms.bdptStride + d;
        device const BDPTVertex& lv = lightVertices[vertIdx];

        // Skip delta vertices (can't merge with delta)
        if (lv.is_delta)
            continue;
        // Skip camera vertices (shouldn't exist in light subpath, but guard)
        if (lv.is_on_camera)
            continue;

        float3 pos = float3(lv.position);
        int3 cell = int3(floor(pos * invCellSize));
        uint32_t bucket = teschnerHash(cell);

        // Allocate entry
        uint32_t entryIdx = atomic_fetch_add_explicit(hashCounter, 1, memory_order_relaxed);

        // Guard against overflow (numPixels * bdptStride entries max)
        uint32_t maxEntries = uniforms.width * uniforms.height * uniforms.bdptStride;
        if (entryIdx >= maxEntries)
            return;

        // Link-list insert: new entry points to old head, head points to new entry
        hashEntries[entryIdx].vertexIndex = vertIdx;
        hashEntries[entryIdx].next = atomic_exchange_explicit(
            (device atomic_uint*)&hashHeads[bucket], entryIdx, memory_order_relaxed);
    }
}
