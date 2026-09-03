#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace oka::metal
{

inline float sharcBaseSizeForPerspective(float verticalFovRadians, uint32_t imageHeight, float voxelPixels)
{
    const float pixelAngle = 2.0f * std::tan(verticalFovRadians * 0.5f) / static_cast<float>(std::max(imageHeight, 1u));
    return pixelAngle * std::max(voxelPixels, 1.0f);
}

// Host copy of the compact Metal grid's LOD calculation. Kept here so the
// screen-space sizing contract can be tested without compiling a Metal kernel.
inline float sharcVoxelSizeForDistance(float distance, float baseSize, int32_t levelBias)
{
    const float continuousLevel =
        std::clamp(std::log2(std::max(distance, 1e-5f)) + static_cast<float>(levelBias), 1.0f, 31.0f);
    const int32_t level = static_cast<int32_t>(std::floor(continuousLevel));
    return std::max(std::exp2(static_cast<float>(level - levelBias)) * baseSize, 1e-5f);
}

} // namespace oka::metal
