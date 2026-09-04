#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
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

inline void sanitizeLinearRgba(float* rgba, size_t texelCount)
{
    if (!rgba)
    {
        return;
    }
    for (size_t i = 0; i < texelCount; ++i)
    {
        float* texel = rgba + 4u * i;
        for (size_t channel = 0; channel < 3u; ++channel)
        {
            texel[channel] = std::isfinite(texel[channel]) && texel[channel] > 0.0f ? texel[channel] : 0.0f;
        }
        texel[3] = std::isfinite(texel[3]) ? std::clamp(texel[3], 0.0f, 1.0f) : 1.0f;
    }
}

} // namespace oka::projector
