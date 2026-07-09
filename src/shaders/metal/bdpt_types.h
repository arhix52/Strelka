#pragma once
// ============================================================================
// bdpt_types.h -- Data structures for bidirectional path tracing
// ============================================================================

#include <simd/simd.h>

#ifndef __METAL_VERSION__
#ifndef BDPT_PACKED_FLOAT3_DEFINED
#define BDPT_PACKED_FLOAT3_DEFINED
// Use the packed_float3 from ShaderTypes.h on CPU side
#endif
#endif

// Hard upper bound on BDPT/VCM subpath vertex slots per pixel.
// The ACTUAL per-pixel stride is chosen at runtime as
//   uniforms.bdptStride = min(max_depth + 1, BDPT_MAX_DEPTH)
// so that the s=0/s=1 (unidirectional + NEE) strategies reach the same path
// length as the reference path tracer (otherwise deep bounces are silently
// truncated and BDPT/VCM render darker than PT), while vertex-buffer memory
// stays proportional to the configured max_depth instead of this hard cap.
// Buffers are allocated with the runtime stride; this constant only bounds it.
#define BDPT_MAX_DEPTH 16

// ---------------------------------------------------------------------------
// BDPTVertex -- stored per-vertex in camera and light subpath buffers
//
// Uses packed_float3 (12 bytes) to avoid Metal's 16-byte float3 alignment.
// Total size: 128 bytes, 16-byte aligned for GPU access.
// ---------------------------------------------------------------------------
struct BDPTVertex
{
    packed_float3 position;        // 12 bytes
    float         pdf_fwd;         //  4 bytes  -- 16

    packed_float3 geometry_normal;  // 12 bytes
    float         pdf_rev;         //  4 bytes  -- 32

    packed_float3 shading_normal;   // 12 bytes
    float         dVCM;            //  4 bytes  (partial MIS weight) -- 48

    packed_float3 throughput;       // 12 bytes
    float         dVC;             //  4 bytes  (partial MIS weight) -- 64

    packed_float3 wo;              // 12 bytes  (outgoing direction at this vertex)
    float         dVM;             //  4 bytes  (partial MIS weight for merging) -- 80

    float         uv_x;           //  4 bytes
    float         uv_y;           //  4 bytes
    uint32_t      material_index;  //  4 bytes
    uint32_t      event_type;      //  4 bytes  -- 96

    uint32_t      is_delta;        //  4 bytes
    uint32_t      is_on_light;     //  4 bytes
    uint32_t      is_on_camera;    //  4 bytes
    uint32_t      light_index;     //  4 bytes  -- 112

    float         exterior_ior;    //  4 bytes  (IOR of medium outside this vertex)
    uint32_t      _pad[3];         // 12 bytes  -- 128 (16-byte aligned)
};
