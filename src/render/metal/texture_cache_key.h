#pragma once

#include <cstdint>
#include <string>


namespace oka::metal
{

// Bumped whenever the .btex cache layout or encoder changes.
inline constexpr uint32_t kTextureCacheVersion = 2;

enum class TextureKind : int
{
    Color = 0,
    NonColor = 1,
    Normal = 2,
};

struct TextureCacheKeyInputs
{
    std::string fileName;
    uint64_t fileSize = 0;
    int64_t writeTimeCount = 0; // last_write_time epoch count
    uint32_t maxDimension = 0;
    uint32_t downscale = 0;
    bool srgb = false;
    TextureKind kind = TextureKind::Color;
};

// FNV-1a over a stable description of how the texture will be uploaded. Returns
// a cache file name like "0123abcd....btex".
inline std::string textureCacheKey(const TextureCacheKeyInputs& in)
{
    std::string blob = in.fileName;
    blob += "|" + std::to_string((unsigned long long)in.fileSize);
    blob += "|" + std::to_string((long long)in.writeTimeCount);
    blob += "|" + std::to_string(in.maxDimension) + "|" + std::to_string(in.downscale);
    blob += in.srgb ? "|srgb" : "|linear";
    blob += "|" + std::to_string((int)in.kind);
    blob += "|v" + std::to_string(kTextureCacheVersion);

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

} // namespace oka::metal

