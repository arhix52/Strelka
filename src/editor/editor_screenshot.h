#pragma once

#include <algorithm>
#include <cmath>
#include <string_view>

namespace oka::editor_screenshot
{

enum class Source
{
    SceneLinear,
    DisplayReferredSdr,
    DisplayReferredHdr,
};

inline Source sourceForExtension(std::string_view extension, bool displayReferred = false)
{
    if (extension == ".png")
    {
        return Source::DisplayReferredSdr;
    }
    return displayReferred ? Source::DisplayReferredHdr : Source::SceneLinear;
}

inline float encodeSrgb(float linear)
{
    const float value = std::clamp(linear, 0.0f, 1.0f);

    if (value <= 0.0031308f)
    {
        return 12.92f * value;
    }
    return 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

} // namespace oka::editor_screenshot

