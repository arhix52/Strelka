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
//
// The z mask is 10 bits, not 12: packNormal only ever fills bits 0..29, and
// bit 30 now carries the tangent handedness sign (see packTangent). A wider
// mask would fold that sign into z.
inline glm::float3 unpackNormal(uint32_t val)
{
    constexpr float scale = 1.0f / 256.0f;
    glm::float3 normal;
    normal.z = ((val & 0x3ff00000) >> 20) * scale - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) * scale - 1.0f;
    normal.x = (val & 0x000003ff) * scale - 1.0f;
    return normal;
}

// glTF stores TANGENT as vec4 whose w is the bitangent handedness (+1/-1).
// Dropping it mirrors every normal map along the bitangent, so it rides along
// in bit 30 of the packed tangent -- free, since packNormal leaves it clear.
constexpr uint32_t kTangentSignBit = 1u << 30;

inline uint32_t packTangent(const glm::float3& tangent, float handedness)
{
    return packNormal(tangent) | (handedness < 0.0f ? kTangentSignBit : 0u);
}

inline float unpackTangentSign(uint32_t val)
{
    return (val & kTangentSignBit) ? -1.0f : 1.0f;
}

// glTF COLOR_0, packed RGBA8. The values are LINEAR -- COLOR_0 carries no
// transfer function, unlike a base-colour texture -- so nothing here decodes.
// 8 bits is ample: COLOR_0 is a multiplier on base colour, and a quarter-percent
// quantisation step is far below anything the surface it modulates can show.
inline uint32_t packColor(const glm::float4& c)
{
    auto q = [](float v) -> uint32_t {
        const float x = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return (uint32_t)(x * 255.0f + 0.5f);
    };
    return q(c.x) | (q(c.y) << 8) | (q(c.z) << 16) | (q(c.w) << 24);
}

inline glm::float4 unpackColor(uint32_t val)
{
    constexpr float s = 1.0f / 255.0f;
    return glm::float4((val & 0xffu) * s, ((val >> 8) & 0xffu) * s, ((val >> 16) & 0xffu) * s,
                       ((val >> 24) & 0xffu) * s);
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
