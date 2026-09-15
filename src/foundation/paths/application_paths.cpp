#include "application_paths.h"

#include <cstdlib>
#include <string_view>

namespace
{

std::filesystem::path environmentPath(const char* name)
{
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::filesystem::path(value) : std::filesystem::path();
}

std::filesystem::path homePath()
{
#if defined(_WIN32)
    return environmentPath("USERPROFILE");
#else
    return environmentPath("HOME");
#endif
}

std::filesystem::path fallbackRoot(std::string_view leaf)
{
    std::error_code ec;
    const std::filesystem::path temporary = std::filesystem::temp_directory_path(ec);
    return (ec ? std::filesystem::path(".") : temporary) / leaf;
}

} // namespace

std::filesystem::path oka::applicationSupportDirectory()
{
#if defined(_WIN32)
    const std::filesystem::path root = environmentPath("LOCALAPPDATA");
    return (root.empty() ? fallbackRoot("Strelka") : root / "Strelka");
#else
    const std::filesystem::path xdg = environmentPath("XDG_CONFIG_HOME");
    const std::filesystem::path home = homePath();
    const std::filesystem::path root = xdg.empty() && !home.empty() ? home / ".config" : xdg;
    return (root.empty() ? fallbackRoot("strelka") : root / "strelka");
#endif
}

std::filesystem::path oka::applicationCacheDirectory()
{
#if defined(_WIN32)
    const std::filesystem::path root = environmentPath("LOCALAPPDATA");
    return (root.empty() ? fallbackRoot("Strelka-cache") : root / "Strelka" / "Cache");
#else
    const std::filesystem::path xdg = environmentPath("XDG_CACHE_HOME");
    const std::filesystem::path home = homePath();
    const std::filesystem::path root = xdg.empty() && !home.empty() ? home / ".cache" : xdg;
    return (root.empty() ? fallbackRoot("strelka-cache") : root / "strelka");
#endif
}

std::filesystem::path oka::applicationLogDirectory()
{
#if defined(_WIN32)
    const std::filesystem::path root = environmentPath("LOCALAPPDATA");
    return (root.empty() ? fallbackRoot("Strelka-logs") : root / "Strelka" / "Logs");
#else
    const std::filesystem::path xdg = environmentPath("XDG_STATE_HOME");
    const std::filesystem::path home = homePath();
    const std::filesystem::path root = xdg.empty() && !home.empty() ? home / ".local" / "state" : xdg;
    return (root.empty() ? fallbackRoot("strelka-logs") : root / "strelka");
#endif
}
