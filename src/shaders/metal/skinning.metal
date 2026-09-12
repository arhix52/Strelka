#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// Pack normal into 10:10:10+2 bit format (matching Scene::Vertex encoding)
static uint32_t packNormal(float3 normal)
{
    return pack_float_to_unorm10a2(float4(normal * 0.5f + 0.5f, 0.0f));
}

// Skin data layout per vertex (64 bytes):
//   [0..15]  joints  (int4)
//   [16..31] weights (float4)
//   [32..47] initial position (float3 + pad)
//   [48..63] initial normal (float3) + packed tangent (uint32)
// Must match CPU-side Scene::vertexSkinData (64 bytes).
// Metal's float3 is 16 bytes in structs; packed_float3 is 12 bytes — matching glm::float3.
struct SkinData
{
    int4           joints;   // 16 bytes, offset 0
    float4         weights;  // 16 bytes, offset 16
    packed_float3  pos;      // 12 bytes, offset 32
    float          pad0;     //  4 bytes, offset 44
    packed_float3  normal;   // 12 bytes, offset 48
    uint32_t       tangent;  //  4 bytes, offset 60 (rest-pose packed tangent)
};                           // Total: 64 bytes

kernel void skinningKernel(
    device char*            vertexBuffer    [[buffer(0)]],
    const device SkinData*  skinDataBuffer  [[buffer(1)]],
    const device float4x4*  jointMatrices   [[buffer(2)]],
    constant SkinningParams& params         [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.vertexCount)
        return;

    const device SkinData& sd = skinDataBuffer[params.sbOffset + tid];

    // Load packed_float3 into float3 for math operations
    float3 restPos = float3(sd.pos);
    float3 restNorm = float3(sd.normal);

    float4x4 skinMat = sd.weights.x * jointMatrices[params.jointMatOffset + sd.joints.x]
                      + sd.weights.y * jointMatrices[params.jointMatOffset + sd.joints.y]
                      + sd.weights.z * jointMatrices[params.jointMatOffset + sd.joints.z]
                      + sd.weights.w * jointMatrices[params.jointMatOffset + sd.joints.w];

    // Transform position
    float4 skinnedPos = skinMat * float4(restPos, 1.0f);

    // Transform normal and tangent (use upper-left 3x3)
    float3x3 normalMat = float3x3(skinMat[0].xyz, skinMat[1].xyz, skinMat[2].xyz);
    float3 skinnedNormal = normalize(normalMat * restNorm);

    // Unpack rest-pose tangent and transform
    float3 restTangent = unpack_unorm10a2_to_float(sd.tangent).xyz * 2.0f - 1.0f;
    float3 skinnedTangent = normalize(normalMat * restTangent);

    // Write to vertex buffer: stride = 32 bytes per vertex (matching Scene::Vertex)
    // Layout: pos(float3=12) + tangent(uint32=4) + normal(uint32=4) + uv/uv1/color(uint32=4 each)
    uint vertexAddr = (params.vbOffset + tid) * 32;

    // Write position (12 bytes at offset 0)
    device float3* posPtr = (device float3*)(vertexBuffer + vertexAddr);
    *posPtr = skinnedPos.xyz;

    // Write packed tangent (4 bytes at offset 12). Skinning rotates the tangent
    // but cannot change which side of the surface the bitangent is on, so bit 30
    // is carried over from the rest pose rather than recomputed.
    device uint32_t* tanPtr = (device uint32_t*)(vertexBuffer + vertexAddr + 12);
    *tanPtr = packNormal(skinnedTangent) | (sd.tangent & (1u << 30));

    // Write packed normal (4 bytes at offset 16)
    device uint32_t* normPtr = (device uint32_t*)(vertexBuffer + vertexAddr + 16);
    *normPtr = packNormal(skinnedNormal);
}
