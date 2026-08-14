#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

namespace oka
{
namespace editor_viewport
{

enum class PresentationMode : uint32_t
{
    Fit = 0,
    OneToOne,
    Fill,
};

struct PreviewPreset
{
    const char* label;
    uint32_t width;
    uint32_t height;
};

inline constexpr std::array<PreviewPreset, 4> kPreviewPresets = {
    PreviewPreset{ "Draft 640 x 360", 640, 360 },
    PreviewPreset{ "Preview 960 x 540", 960, 540 },
    PreviewPreset{ "HD 1280 x 720", 1280, 720 },
    PreviewPreset{ "Full HD 1920 x 1080", 1920, 1080 },
};

inline constexpr uint32_t kDefaultPreviewWidth = 960;
inline constexpr uint32_t kDefaultPreviewHeight = 540;
inline constexpr uint32_t kMinPreviewDimension = 16;
inline constexpr uint32_t kMaxPreviewDimension = 8192;

struct Point
{
    float x = 0.0f;
    float y = 0.0f;
};

struct Rect
{
    Point min;
    Point max;

    float width() const { return max.x - min.x; }
    float height() const { return max.y - min.y; }
    bool valid() const { return width() > 0.0f && height() > 0.0f; }
};

struct Layout
{
    Rect panelRect;
    // Full uncropped camera image. It may extend outside panelRect in Fill
    // and OneToOne modes; overlays and picking must use this rect.
    Rect imageRect;
    Rect visibleRect;
};

inline Rect intersectRects(const Rect& a, const Rect& b)
{
    Rect result;
    result.min.x = std::max(a.min.x, b.min.x);
    result.min.y = std::max(a.min.y, b.min.y);
    result.max.x = std::min(a.max.x, b.max.x);
    result.max.y = std::min(a.max.y, b.max.y);
    if (!result.valid())
    {
        return {};
    }
    return result;
}

inline Layout computeLayout(const Rect& panelRect,
                            uint32_t imageWidth,
                            uint32_t imageHeight,
                            Point framebufferScale,
                            PresentationMode mode)
{
    Layout result;
    result.panelRect = panelRect;
    if (!panelRect.valid() || imageWidth == 0 || imageHeight == 0)
    {
        return result;
    }

    const float panelWidth = panelRect.width();
    const float panelHeight = panelRect.height();
    const float fitScale = std::min(panelWidth / static_cast<float>(imageWidth),
                                    panelHeight / static_cast<float>(imageHeight));
    const float fillScale = std::max(panelWidth / static_cast<float>(imageWidth),
                                     panelHeight / static_cast<float>(imageHeight));
    float scale = fitScale;
    if (mode == PresentationMode::Fill)
    {
        scale = fillScale;
    }
    else if (mode == PresentationMode::OneToOne)
    {
        const float sx = framebufferScale.x > 0.0f ? framebufferScale.x : 1.0f;
        const float sy = framebufferScale.y > 0.0f ? framebufferScale.y : 1.0f;
        // Non-uniform display scaling would distort a square pixel. Use the
        // smaller density so both axes remain visible and the aspect stays exact.
        scale = 1.0f / std::min(sx, sy);
    }

    const float width = static_cast<float>(imageWidth) * scale;
    const float height = static_cast<float>(imageHeight) * scale;
    const float x = panelRect.min.x + (panelWidth - width) * 0.5f;
    const float y = panelRect.min.y + (panelHeight - height) * 0.5f;
    result.imageRect = { { x, y }, { x + width, y + height } };
    result.visibleRect = intersectRects(panelRect, result.imageRect);
    return result;
}

inline std::optional<Point> screenToImageUv(Point screenPoint, const Layout& layout)
{
    if (!layout.imageRect.valid() || !layout.visibleRect.valid() ||
        screenPoint.x < layout.visibleRect.min.x || screenPoint.x > layout.visibleRect.max.x ||
        screenPoint.y < layout.visibleRect.min.y || screenPoint.y > layout.visibleRect.max.y)
    {
        return std::nullopt;
    }

    return Point{ (screenPoint.x - layout.imageRect.min.x) / layout.imageRect.width(),
                  (screenPoint.y - layout.imageRect.min.y) / layout.imageRect.height() };
}

inline int findPreset(uint32_t width, uint32_t height)
{
    for (size_t i = 0; i < kPreviewPresets.size(); ++i)
    {
        if (kPreviewPresets[i].width == width && kPreviewPresets[i].height == height)
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

inline uint32_t clampPreviewDimension(int value)
{
    return static_cast<uint32_t>(
        std::clamp(value, static_cast<int>(kMinPreviewDimension), static_cast<int>(kMaxPreviewDimension)));
}

} // namespace editor_viewport
} // namespace oka
