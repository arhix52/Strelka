#pragma once
// CPU-side vertex attribute packing/unpacking utilities.
// GPU-side counterparts: src/shaders/optix/optix_device_utils.h (CUDA)
//                        src/shaders/metal/pathtrace.metal (Metal)

#include <cstdint>
#include <glm/glm.hpp>

namespace oka
{

// Pack normal to uint32_t. Valid range: [-1, 1]
// Format: 10 bits per component (x in low bits, z in high bits), bias +1.0, scale 256
inline uint32_t packNormal(const glm::float3& normal)
{
    constexpr float scale = 256.0f;
    auto x = (uint32_t)((normal.x + 1.0f) * scale);
    auto y = (uint32_t)((normal.y + 1.0f) * scale);
    auto z = (uint32_t)((normal.z + 1.0f) * scale);
    return (z << 20) | (y << 10) | x;
}

// Unpack normal from uint32_t.
inline glm::float3 unpackNormal(uint32_t val)
{
    constexpr float scale = 1.0f / 256.0f;
    glm::float3 normal;
    normal.z = ((val & 0xfff00000) >> 20) * scale - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) * scale - 1.0f;
    normal.x = (val & 0x000003ff) * scale - 1.0f;
    return normal;
}

// Pack UV to uint32_t. Valid range: [-10, 10]
// Format: 16 bits per component (x low, y high)
inline uint32_t packUV(const glm::float2& uv)
{
    int32_t packed = (uint32_t)((uv.x + 10.0f) / 20.0f * 16383.99999f);
    packed += (uint32_t)((uv.y + 10.0f) / 20.0f * 16383.99999f) << 16;
    return packed;
}

// Unpack UV from uint32_t.
inline glm::float2 unpackUV(uint32_t val)
{
    glm::float2 uv;
    uv.y = ((val & 0xffff0000) >> 16) / 16383.99999f * 20.0f - 10.0f;
    uv.x = (val & 0x0000ffff) / 16383.99999f * 20.0f - 10.0f;
    return uv;
}

} // namespace oka
