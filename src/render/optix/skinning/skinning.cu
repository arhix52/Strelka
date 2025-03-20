#include "skinning.h"
#include <sutil/vec_math.h>

//  valid range of coordinates [-1; 1]
__device__ uint32_t packNormal(float3 normal)
{
    uint32_t packed = (uint32_t)((normal.x + 1.0f) / 2.0f * 511.99999f);
    packed += (uint32_t)((normal.y + 1.0f) / 2.0f * 511.99999f) << 10;
    packed += (uint32_t)((normal.z + 1.0f) / 2.0f * 511.99999f) << 20;
    return packed;
}

__global__ void skinningKernel(
    const int vbOffset,
    const int sbOffset,
    void* vertexPtr,
    const float3* d_initial_positions,
    const float3* d_initial_normals, 
    const float4* d_weights,
    const int4* d_joints,
    const sutil::Matrix4x4* d_jointMats,
    int jointMatOffset,
    const uint32_t vertexCount) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertexCount) return;

    int offsettedId = idx + sbOffset;
    float3 initialPos = d_initial_positions[offsettedId];
    float3 initialNorm = d_initial_normals[offsettedId];
    float4 weights = d_weights[offsettedId];
    int4 joints = d_joints[offsettedId];

    const sutil::Matrix4x4 skinMat =
          weights.x * d_jointMats[jointMatOffset + joints.x]
        + weights.y * d_jointMats[jointMatOffset + joints.y]
        + weights.z * d_jointMats[jointMatOffset + joints.z]
        + weights.w * d_jointMats[jointMatOffset + joints.w];

    char* vertexBase = static_cast<char*>(vertexPtr);

    float3* vertexPos = reinterpret_cast<float3*>(vertexBase + (vbOffset + idx) * 32);
    *vertexPos = make_float3(skinMat * make_float4(initialPos.x, initialPos.y, initialPos.z, 1.0f));

    uint32_t* vertexNorm = reinterpret_cast<uint32_t*>(vertexBase + (vbOffset + idx) * 32 + 16);
    *vertexNorm = packNormal(normalize(make_matrix3x3(skinMat) * initialNorm));
}

void cuApplySkinning(
    int threads_per_block,
    const int vbOffset,
    const int sbOffset,
    void* vertexPtr,
    const float3* d_initial_positions,
    const float3* d_initial_normals, 
    const float4* d_weights,
    const int4* d_joints,
    const sutil::Matrix4x4* d_jointMats,
    int jointMatOffset,
    const uint32_t vertexCount)
{
    int blocks_per_grid = (vertexCount + threads_per_block - 1) / threads_per_block;
    skinningKernel<<<blocks_per_grid, threads_per_block>>>(vbOffset, sbOffset, vertexPtr,
        d_initial_positions, d_initial_normals, d_weights, d_joints, d_jointMats, jointMatOffset, vertexCount);
}