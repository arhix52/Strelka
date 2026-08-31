#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <fmt/format.h>


namespace oka::editor_document
{

/// Window title: "Strelka [*] basename". Render stats (ms/spp) live in the
/// viewport's own status line instead -- an OS title bar is not a HUD, and it
/// only ever shows the scene a user actually cares about identifying.
inline std::string formatWindowTitle(bool dirty, const std::string& scenePath)
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
    return fmt::format("Strelka {}{}", dirty ? "* " : "", doc);
}

/// After a failed/cancelled open, keep the previous document path (may be empty).
inline std::string restorePathAfterFailedLoad(const std::string& previousPath)
{
    return previousPath;
}

/// Which camera a freshly loaded document opens on. The cameras the scene authored
/// come first and the fitted "Main" is appended last, so this opens on the authored
/// shot when the scene has one -- that is what the scene was built around and what
/// StrelkaCLI renders for the same file -- and falls back to Main when it has none.
inline int selectCameraIndexAfterLoad(uint32_t authoredCameraCount, uint32_t totalCameraCount)
{
    if (totalCameraCount == 0)
    {
        return 0;
    }
    if (authoredCameraCount > 0)
    {
        return 0;
    }
    return static_cast<int>(totalCameraCount - 1);
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
    // Non-negative by the guard above, so the conversion is exact.
    const uint32_t selectedIndex = static_cast<uint32_t>(selected);
    if (selectedIndex >= cameraCount)
    {
        return static_cast<int>(cameraCount - 1);
    }
    return selected;
}

/// How many File → Open Recent entries the editor keeps. Ten is enough to cover
/// a working set without turning the submenu into a file browser.
inline constexpr size_t kRecentScenesCapacity = 10;

/// Absolute, weakly-canonical form so "/a/../b.glb" and "/b.glb" collide in the
/// recent list. Falls back to the input when the path cannot be resolved yet
/// (a file that has not been written, a missing drive), because the caller still
/// wants that string remembered.
inline std::string normalizeRecentPath(const std::string& path)
{
    if (path.empty())
    {
        return {};
    }
    std::error_code ec;
    std::filesystem::path resolved = std::filesystem::weakly_canonical(path, ec);
    if (ec)
    {
        ec.clear();
        resolved = std::filesystem::absolute(path, ec);
    }
    if (ec)
    {
        return path;
    }
    return resolved.string();
}

/// Move `path` to the front of `recent`, drop earlier duplicates of the same
/// file, and trim to `capacity`. Empty paths are ignored: an empty document is
/// not a scene that was opened.
inline void pushRecentScene(std::vector<std::string>& recent, const std::string& path,
                            size_t capacity = kRecentScenesCapacity)
{
    const std::string norm = normalizeRecentPath(path);
    if (norm.empty() || capacity == 0)
    {
        return;
    }
    // std::erase_if rather than the erase-remove pair: ranges::remove_if returns
    // a subrange, so the pair does not even compile against it, and the one-call
    // form is what it was always spelling out.
    std::erase_if(recent, [&](const std::string& existing) { return normalizeRecentPath(existing) == norm; });
    recent.insert(recent.begin(), norm);
    if (recent.size() > capacity)
    {
        recent.resize(capacity);
    }
}

/// One absolute path per line, most-recent first. Blank lines and entries that
/// normalize to empty are skipped so a hand-edited file cannot poison the menu.
/// Duplicates keep the earlier (more recent) line.
inline std::vector<std::string> loadRecentScenes(const std::filesystem::path& file,
                                                 size_t capacity = kRecentScenesCapacity)
{
    std::vector<std::string> recent;
    std::ifstream in(file);
    if (!in)
    {
        return recent;
    }
    std::string line;
    while (std::getline(in, line))
    {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
        {
            line.pop_back();
        }
        const std::string norm = normalizeRecentPath(line);
        if (norm.empty())
        {
            continue;
        }
        const bool already = std::ranges::any_of(
            recent, [&](const std::string& existing) { return normalizeRecentPath(existing) == norm; });
        if (already)
        {
            continue;
        }
        recent.push_back(norm);
        if (recent.size() >= capacity)
        {
            break;
        }
    }
    return recent;
}

inline bool saveRecentScenes(const std::filesystem::path& file, const std::vector<std::string>& recent)
{
    std::ofstream out(file, std::ios::trunc);
    if (!out)
    {
        return false;
    }
    for (const std::string& path : recent)
    {
        out << path << '\n';
    }
    return static_cast<bool>(out);
}

} // namespace oka::editor_document

