#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <fmt/format.h>

namespace oka
{
namespace editor_document
{

/// Window title: "Strelka [*] basename — render ms / spp"
inline std::string formatWindowTitle(bool dirty, const std::string& scenePath, float renderMs, uint32_t spp)
{
    std::string doc;
    if (scenePath.empty())
    {
        doc = "(empty)";
    }
    else
    {
        doc = std::filesystem::path(scenePath).filename().string();
        if (doc.empty())
        {
            doc = scenePath;
        }
    }
    return fmt::format("Strelka {}{} — [{:.1f} ms] [{} spp]", dirty ? "* " : "", doc, renderMs, spp);
}

/// After a failed/cancelled open, keep the previous document path (may be empty).
inline std::string restorePathAfterFailedLoad(const std::string& previousPath)
{
    return previousPath;
}

/// Framed "Main" is appended last on load; select it so the viewport matches the fit.
inline int selectMainCameraIndexAfterLoad(uint32_t cameraCount)
{
    if (cameraCount == 0)
    {
        return 0;
    }
    return static_cast<int>(cameraCount - 1);
}

inline int clampCameraIndex(int selected, uint32_t cameraCount)
{
    if (cameraCount == 0)
    {
        return 0;
    }
    if (selected < 0)
    {
        return 0;
    }
    if (static_cast<uint32_t>(selected) >= cameraCount)
    {
        return static_cast<int>(cameraCount - 1);
    }
    return selected;
}

} // namespace editor_document
} // namespace oka
