#pragma once

#include <algorithm>
#include <cstdint>


namespace oka::metal
{

// Halton sequence sample in [0, 1). Index from 1 for jitter so the first entry
// is not zero (which would leave one frame in each cycle unjittered).
inline float haltonAt(uint64_t index, uint32_t base)
{
    float result = 0.0f;
    float f = 1.0f / (float)base;
    while (index > 0)
    {
        result += f * (float)(index % base);
        index /= base;
        f /= (float)base;
    }
    return result;
}

// Halton (2, 3) jitter centred on the pixel, wrapped to phaseCount. Temporal
// upscalers need a closed sequence over the output pixel grid.
inline void frameJitter(uint64_t frameIndex, uint32_t phaseCount, float& x, float& y)
{
    const uint64_t phase = frameIndex % std::max<uint32_t>(phaseCount, 1u);
    x = haltonAt(phase + 1, 2) - 0.5f;
    y = haltonAt(phase + 1, 3) - 0.5f;
}

} // namespace oka::metal

