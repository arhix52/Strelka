#pragma once

#include <cmath>
#include <cstdint>

namespace oka::projector
{

// IEC 61966-2-1 EOTF used by Metal's *_sRGB texture formats. OptiX linearises
// the same 8-bit source on the host so both texture filters interpolate the
// same linear radiance values.
inline float srgb8ToLinear(uint8_t code)
{
    const float encoded = static_cast<float>(code) / 255.0f;
    return encoded <= 0.04045f ? encoded / 12.92f : std::pow((encoded + 0.055f) / 1.055f, 2.4f);
}

} // namespace oka::projector
