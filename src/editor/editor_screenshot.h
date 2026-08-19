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
    DisplayReferredHdr,
};

/// What a screenshot should contain, given where it is going.
///
/// PNG has nowhere to put a value above white, so it is always the display
/// transform taken back down to SDR -- clipping the EDR range into it instead
/// would throw away exactly the highlights the headroom was spent on.
///
/// EXR can hold either, so it takes the caller's choice. Scene-linear radiance
/// is the default because that is what a screenshot is usually wanted for -- a
/// reference, a comparison against another renderer, an image to grade. The
/// display-referred form is for the other question: what the screen actually
/// showed, with the EDR range intact.
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

} // namespace editor_screenshot
} // namespace oka
