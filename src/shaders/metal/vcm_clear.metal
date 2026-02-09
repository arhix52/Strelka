// ============================================================================
// vcm_clear.metal -- VCM Hash Grid Clear Kernel
//
// Resets the spatial hash grid head pointers to invalid and the
// global entry counter to 0.
// ============================================================================

#include <metal_stdlib>
#include "vcm_types.h"

using namespace metal;

kernel void vcm_clear(
    uint tid [[thread_position_in_grid]],
    device uint32_t* hashHeads [[buffer(0)]],
    device atomic_uint* hashCounter [[buffer(1)]]
)
{
    if (tid < VCM_HASH_SIZE)
        hashHeads[tid] = VCM_HASH_INVALID;
    if (tid == 0)
        atomic_store_explicit(hashCounter, 0, memory_order_relaxed);
}
