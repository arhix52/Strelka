#pragma once

#include <filesystem>
#include <string>

#if defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <vector>
#elif defined(_WIN32)
#    include <windows.h>
#else
#    include <unistd.h>
#    include <vector>
#endif

namespace oka
{

/// Directory containing the running executable.
///
/// Runtime assets (metallibs, OptiX IR, the fullScreen.metal source) are laid
/// out next to the binary by the build system. Resolving against the executable
/// instead of the process working directory lets the app be launched from
/// anywhere, including via Finder / a debugger with a different CWD.
inline const std::filesystem::path& getExecutableDir()
{
    static const std::filesystem::path dir = [] {
        std::error_code ec;
#if defined(__APPLE__)
        uint32_t size = 0;
        _NSGetExecutablePath(nullptr, &size);
        std::vector<char> buf(size + 1, '\0');
        if (_NSGetExecutablePath(buf.data(), &size) != 0)
        {
            return std::filesystem::current_path(ec);
        }
        std::filesystem::path exe = std::filesystem::weakly_canonical(std::filesystem::path(buf.data()), ec);
#elif defined(_WIN32)
        wchar_t buf[MAX_PATH] = {};
        const DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
        if (len == 0)
        {
            return std::filesystem::current_path(ec);
        }
        std::filesystem::path exe = std::filesystem::weakly_canonical(std::filesystem::path(buf), ec);
#else
        std::vector<char> buf(4096, '\0');
        const ssize_t len = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
        if (len <= 0)
        {
            return std::filesystem::current_path(ec);
        }
        std::filesystem::path exe = std::filesystem::weakly_canonical(std::filesystem::path(buf.data()), ec);
#endif
        if (ec)
        {
            return std::filesystem::current_path(ec);
        }
        return exe.parent_path();
    }();
    return dir;
}

/// Resolve a build-tree-relative asset path (e.g. "metal/shaders/pathtrace.metallib").
///
/// Prefers the executable directory; falls back to the working directory so that
/// existing "run from the build root" workflows keep working.
inline std::string resolveResourcePath(const std::string& relative)
{
    std::error_code ec;
    const std::filesystem::path fromExe = getExecutableDir() / relative;
    if (std::filesystem::exists(fromExe, ec))
    {
        return fromExe.string();
    }
    return relative;
}

} // namespace oka
