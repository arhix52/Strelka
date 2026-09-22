#pragma once

#include "texture_asset.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace oka::texture
{

inline constexpr uint32_t kTextureCacheKeyVersion = 3;

struct TextureCacheKeyInputs
{
    std::string fileName;
    uint64_t fileSize = 0;
    int64_t writeTimeCount = 0; // last_write_time epoch count
    uint32_t maxDimension = 0;
    uint32_t downscale = 1;
    bool srgb = false;
    bool compressed = false;
    Semantic semantic = Semantic::Color;
    TargetProfile target = TargetProfile::AppleAstc;
    uint32_t encoderVersion = 0;
};

inline std::string textureIdentityPath(const std::filesystem::path& path)
{
    std::error_code error;
    std::filesystem::path identity;
    std::filesystem::path canonical;
    identity = std::filesystem::absolute(path, error);
    if (error)
    {
        error.clear();
        identity = path;
    }
    canonical = std::filesystem::weakly_canonical(identity, error);
    if (!error)
    {
        identity = canonical;
    }
    return identity.lexically_normal().string();
}

// FNV-1a over a stable description of how the texture will be uploaded. Returns
// a cache file name like "0123abcd....btex".
inline std::string textureCacheKey(const TextureCacheKeyInputs& in)
{
    std::string blob = textureIdentityPath(in.fileName);
    blob += "|" + std::to_string((unsigned long long)in.fileSize);
    blob += "|" + std::to_string((long long)in.writeTimeCount);
    blob += "|" + std::to_string(in.maxDimension) + "|" + std::to_string(in.downscale);
    blob += in.srgb ? "|srgb" : "|linear";
    blob += in.compressed ? "|compressed" : "|raw";
    blob += "|" + std::to_string(static_cast<int>(in.semantic));
    blob += "|" + std::to_string(static_cast<int>(in.target));
    blob += "|encoder" + std::to_string(in.encoderVersion);
    blob += "|key" + std::to_string(kTextureCacheKeyVersion);

    uint64_t hash = 1469598103934665603ull;
    for (const char c : blob)
    {
        hash ^= (uint64_t)(unsigned char)c;
        hash *= 1099511628211ull;
    }

    // Hand-rolled hex: avoids snprintf (vararg / ignored-return tidy hits) and
    // keeps this header free of fmt so the unit tests can include it alone.
    static constexpr char kHex[] = "0123456789abcdef";
    std::string name(16 + 5, '\0');
    for (int i = 15; i >= 0; --i)
    {
        name[static_cast<size_t>(i)] = kHex[hash & 0xfull];
        hash >>= 4;
    }
    name.replace(16, 5, ".btex");
    return name;
}

} // namespace oka::texture
