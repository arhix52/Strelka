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

inline std::string resolveResourcePath(const std::string& relative)
{
    std::error_code ec;
#if defined(__APPLE__)
    // <bundle>.app/Contents/MacOS/<executable> -> Contents/Resources.
    // The same binary may also be run outside a bundle while developing, so
    // keep the flat lookup below as a fallback.
    const std::filesystem::path fromBundle = getExecutableDir().parent_path() / "Resources" / relative;
    if (std::filesystem::exists(fromBundle, ec))
    {
        return fromBundle.string();
    }
#endif
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
