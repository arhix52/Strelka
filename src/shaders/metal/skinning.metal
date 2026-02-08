#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// Pack normal into 10:10:10+2 bit format (matching Scene::Vertex encoding)
static uint32_t packNormal(float3 normal)
{
    constexpr float scale = 256.0f;
    uint32_t x = (uint32_t)((normal.x + 1.0f) * scale);
    uint32_t y = (uint32_t)((normal.y + 1.0f) * scale);
    uint32_t z = (uint32_t)((normal.z + 1.0f) * scale);
    return (z << 20) | (y << 10) | x;
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
    constexpr float invScale = 1.0f / 256.0f;
    float3 restTangent = float3(
        float(sd.tangent & 0x3FFu) * invScale - 1.0f,
        float((sd.tangent >> 10) & 0x3FFu) * invScale - 1.0f,
        float((sd.tangent >> 20) & 0x3FFu) * invScale - 1.0f);
    float3 skinnedTangent = normalize(normalMat * restTangent);

    // Write to vertex buffer: stride = 32 bytes per vertex (matching Scene::Vertex)
    // Layout: pos(float3=12) + tangent(uint32=4) + normal(uint32=4) + uv(uint32=4) + pad(8)
    uint vertexAddr = (params.vbOffset + tid) * 32;

    // Write position (12 bytes at offset 0)
    device float3* posPtr = (device float3*)(vertexBuffer + vertexAddr);
    *posPtr = skinnedPos.xyz;

    // Write packed tangent (4 bytes at offset 12)
    device uint32_t* tanPtr = (device uint32_t*)(vertexBuffer + vertexAddr + 12);
    *tanPtr = packNormal(skinnedTangent);

    // Write packed normal (4 bytes at offset 16)
    device uint32_t* normPtr = (device uint32_t*)(vertexBuffer + vertexAddr + 16);
    *normPtr = packNormal(skinnedNormal);
}

kernel void updateTriangleBufferKernel(
    device Triangle*          triangleBuffer  [[buffer(0)]],
    const device char*        vertexBuffer    [[buffer(1)]],
    const device uint32_t*    indexBuffer     [[buffer(2)]],
    constant TriangleUpdateParams& params     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.triangleCount)
        return;

    uint32_t i0 = indexBuffer[params.indexOffset + tid * 3 + 0];
    uint32_t i1 = indexBuffer[params.indexOffset + tid * 3 + 1];
    uint32_t i2 = indexBuffer[params.indexOffset + tid * 3 + 2];

    // Read skinned vertices (stride 32 bytes)
    uint addr0 = (params.vbOffset + i0) * 32;
    uint addr1 = (params.vbOffset + i1) * 32;
    uint addr2 = (params.vbOffset + i2) * 32;

    const device float3* p0 = (const device float3*)(vertexBuffer + addr0);
    const device float3* p1 = (const device float3*)(vertexBuffer + addr1);
    const device float3* p2 = (const device float3*)(vertexBuffer + addr2);

    const device uint32_t* t0 = (const device uint32_t*)(vertexBuffer + addr0 + 12);
    const device uint32_t* t1 = (const device uint32_t*)(vertexBuffer + addr1 + 12);
    const device uint32_t* t2 = (const device uint32_t*)(vertexBuffer + addr2 + 12);

    const device uint32_t* n0 = (const device uint32_t*)(vertexBuffer + addr0 + 16);
    const device uint32_t* n1 = (const device uint32_t*)(vertexBuffer + addr1 + 16);
    const device uint32_t* n2 = (const device uint32_t*)(vertexBuffer + addr2 + 16);

    device Triangle& tri = triangleBuffer[tid];
    tri.positions[0] = *p0;
    tri.positions[1] = *p1;
    tri.positions[2] = *p2;
    tri.normals[0] = *n0;
    tri.normals[1] = *n1;
    tri.normals[2] = *n2;
    tri.tangent[0] = *t0;
    tri.tangent[1] = *t1;
    tri.tangent[2] = *t2;
    // uv unchanged by skinning
}
