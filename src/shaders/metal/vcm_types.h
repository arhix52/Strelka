#pragma once
// ============================================================================
// vcm_types.h -- Data structures for Vertex Connection and Merging (VCM)
// ============================================================================

#include <simd/simd.h>

// Hash grid size: 1M buckets (~4 MB for head pointers)
#define VCM_HASH_SIZE (1 << 20)

// Invalid/sentinel value for hash linked list termination
#define VCM_HASH_INVALID 0xFFFFFFFF

// Entry in the VCM spatial hash grid linked list
struct VCMHashEntry
{
    uint32_t vertexIndex;   // index into light vertex buffer (linearPixel * MAX_DEPTH + depth)
    uint32_t next;          // next entry in linked list (VCM_HASH_INVALID = end)
};
