#include "skinning.h"
#include <sutil/vec_math.h>

// Shared RGB10A2-unorm vertex direction. A2 stays clear here so tangent
// handedness can be copied into bit 30 below.
__device__ const uint32_t packNormal(float3 normal)
{
    constexpr float scale = 511.5f;
    const uint32_t x = __float2uint_rn((fminf(fmaxf(normal.x, -1.0f), 1.0f) + 1.0f) * scale);
    const uint32_t y = __float2uint_rn((fminf(fmaxf(normal.y, -1.0f), 1.0f) + 1.0f) * scale);
    const uint32_t z = __float2uint_rn((fminf(fmaxf(normal.z, -1.0f), 1.0f) + 1.0f) * scale);
    return (z << 20) | (y << 10) | x;
}

__global__ void skinningKernel(
    const int vbOffset,
    const int sbOffset,
    void* vertexPtr,
    const void* vertexSkinDataPtr,
    const sutil::Matrix4x4* d_jointMats,
    int jointMatOffset,
    const uint32_t vertexCount) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertexCount) return;

    int offsettedId = idx + sbOffset;

    const char* baseSkinData = static_cast<const char*>(vertexSkinDataPtr);

    constexpr int jointsOffset = 0;
    constexpr int weightsOffset = 16;
    constexpr int posOffset = weightsOffset + 16;
    constexpr int normalOffset = posOffset + 16;

    const char* skinData = baseSkinData + offsettedId * 64;

    const int4* joints = reinterpret_cast<const int4*>(skinData + jointsOffset);
    const float4* weights = reinterpret_cast<const float4*>(skinData + weightsOffset);
    const float3* initialPos = reinterpret_cast<const float3*>(skinData + posOffset);
    const float3* initialNorm = reinterpret_cast<const float3*>(skinData + normalOffset);
    constexpr int tangentOffset = normalOffset + 12; // uint32_t at offset 60
    const uint32_t* initialTangentPacked = reinterpret_cast<const uint32_t*>(skinData + tangentOffset);

    const sutil::Matrix4x4 skinMat =
          weights->x * d_jointMats[jointMatOffset + joints->x]
        + weights->y * d_jointMats[jointMatOffset + joints->y]
        + weights->z * d_jointMats[jointMatOffset + joints->z]
        + weights->w * d_jointMats[jointMatOffset + joints->w];

    char* vertexBase = static_cast<char*>(vertexPtr);

    float3* vertexPos = reinterpret_cast<float3*>(vertexBase + (vbOffset + idx) * 32);
    *vertexPos = make_float3(skinMat * make_float4(*initialPos, 1.0f));

    const sutil::Matrix3x3 skinMat3x3 = make_matrix3x3(skinMat);

    uint32_t* vertexNorm = reinterpret_cast<uint32_t*>(vertexBase + (vbOffset + idx) * 32 + 16);
    *vertexNorm = packNormal(normalize(skinMat3x3 * (*initialNorm)));

    // Transform tangent by the same 3x3 matrix
    constexpr float scale = 2.0f / 1023.0f;
    uint32_t tp = *initialTangentPacked;
    float3 initialTangent = make_float3(float(tp & 0x3FFu) * scale - 1.0f, float((tp >> 10) & 0x3FFu) * scale - 1.0f,
                                        float((tp >> 20) & 0x3FFu) * scale - 1.0f);
    uint32_t* vertexTangent = reinterpret_cast<uint32_t*>(vertexBase + (vbOffset + idx) * 32 + 12);
    *vertexTangent = packNormal(normalize(skinMat3x3 * initialTangent)) | (tp & (1u << 30));
}

void cuApplySkinning(
    int threads_per_block,
    const int vbOffset,
    const int sbOffset,
    void* vertexPtr,
    const void* vertexSkinDataPtr,
    const sutil::Matrix4x4* d_jointMats,
    int jointMatOffset,
    const uint32_t vertexCount)
{
    int blocks_per_grid = (vertexCount + threads_per_block - 1) / threads_per_block;
    skinningKernel<<<blocks_per_grid, threads_per_block>>>(vbOffset, sbOffset, vertexPtr, vertexSkinDataPtr, d_jointMats, jointMatOffset, vertexCount);
}
