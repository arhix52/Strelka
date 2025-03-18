#pragma once
#include <sutil/Matrix.h>

extern "C" void cuApplySkinning(
    int threads_per_block,
    const int vbOffset,
    const int sbOffset,
    void* vertexPtr,
    const float3* d_initial_positions,
    const float3* d_initial_normals, 
    const float4* d_weights,
    const int4* d_joints,
    const sutil::Matrix4x4* d_jointMats,
    const uint32_t vertexCount);