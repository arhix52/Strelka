#pragma once

#include <algorithm>
#include <cmath>
#include <string_view>

namespace oka
{
namespace editor_screenshot
{

enum class Source
{
    SceneLinear,
    DisplayReferredSdr,
};

inline Source sourceForExtension(std::string_view extension)
{
    return extension == ".png" ? Source::DisplayReferredSdr
                               : Source::SceneLinear;
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

} // namespace editor_screenshot
} // namespace oka
