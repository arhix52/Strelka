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
        const std::filesystem::path exe = std::filesystem::weakly_canonical(std::filesystem::path(buf.data()), ec);
#elif defined(_WIN32)
        wchar_t buf[MAX_PATH] = {};
        const DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
        if (len == 0)
        {
            return std::filesystem::current_path(ec);
        }
        const std::filesystem::path exe = std::filesystem::weakly_canonical(std::filesystem::path(buf), ec);
#else
        std::vector<char> buf(4096, '\0');
        const ssize_t len = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
        if (len <= 0)
        {
            return std::filesystem::current_path(ec);
        }
        const std::filesystem::path exe = std::filesystem::weakly_canonical(std::filesystem::path(buf.data()), ec);
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
/// Three candidates, in the order they are cheapest to be right about:
///
///  1. Next to the executable. This is the build tree, and the macOS package,
///     where the binaries and their assets share one directory.
///  2. ../share/strelka, relative to the executable. This is an installed tree
///     on Linux, where the binary is in bin/ and anything that is not a program
///     belongs under share/ -- which is what every packaging convention and
///     every distribution's policy expects, and what lets one prefix hold
///     several applications.
///  3. The path as given, relative to the working directory, so that "run it
///     from the build root" keeps working.
inline std::string resolveResourcePath(const std::string& relative)
{
    std::error_code ec;
    const std::filesystem::path fromExe = getExecutableDir() / relative;
    if (std::filesystem::exists(fromExe, ec))
    {
        return fromExe.string();
    }
    const std::filesystem::path fromShare = getExecutableDir().parent_path() / "share" / "strelka" / relative;
    if (std::filesystem::exists(fromShare, ec))
    {
        return fromShare.string();
    }
    return relative;
}

} // namespace oka
