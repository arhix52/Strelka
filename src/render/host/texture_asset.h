#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>

namespace oka::texture
{

enum class Semantic : uint8_t
{
    Color = 0,
    NonColor = 1,
    Normal = 2,
};

enum class TargetProfile : uint8_t
{
    AppleAstc = 0,
    NvidiaBc = 1,
};

enum class NativeFormat : uint8_t
{
    RGBA8 = 0,
    RGBA16 = 1,
    RGBA32F = 2,
    BC1 = 3,
    BC3 = 4,
    BC5 = 5,
    ASTC4x4 = 6,
    ASTC6x6 = 7,
};

struct FormatInfo
{
    uint8_t blockExtent = 1;
    uint8_t bytesPerBlock = 0;
    bool compressed = false;
};

constexpr FormatInfo formatInfo(NativeFormat format)
{
    switch (format)
    {
    case NativeFormat::RGBA8:
        return { 1, 4, false };
    case NativeFormat::RGBA16:
        return { 1, 8, false };
    case NativeFormat::RGBA32F:
        return { 1, 16, false };
    case NativeFormat::BC1:
        return { 4, 8, true };
    case NativeFormat::BC3:
    case NativeFormat::BC5:
    case NativeFormat::ASTC4x4:
        return { 4, 16, true };
    case NativeFormat::ASTC6x6:
        return { 6, 16, true };
    }
    return {};
}

inline size_t levelBytes(NativeFormat format, int width, int height)
{
    const FormatInfo info = formatInfo(format);
    const size_t w = static_cast<size_t>(std::max(1, width));
    const size_t h = static_cast<size_t>(std::max(1, height));
    return ((w + info.blockExtent - 1) / info.blockExtent) * ((h + info.blockExtent - 1) / info.blockExtent) *
           info.bytesPerBlock;
}

struct Extent
{
    int width = 0;
    int height = 0;

    bool operator==(const Extent&) const = default;
};

inline Extent resolveExtent(int srcWidth, int srcHeight, uint32_t maxDimension, uint32_t downscale)
{
    const int divisor = static_cast<int>(std::max(1u, downscale));
    const int maxEdge = static_cast<int>(std::min<uint32_t>(maxDimension, static_cast<uint32_t>(INT_MAX)));
    int width = std::max(1, srcWidth / divisor);
    int height = std::max(1, srcHeight / divisor);
    while (maxEdge > 0 && std::max(width, height) > maxEdge && (width > 1 || height > 1))
    {
        width = std::max(1, width / 2);
        height = std::max(1, height / 2);
    }
    return { width, height };
}

struct Recipe
{
    Semantic semantic = Semantic::Color;
    TargetProfile target = TargetProfile::AppleAstc;
    bool hasAlpha = false;
    bool sourceIsFloat = false;
    bool sourceIs16Bit = false;
    bool compress = true;
};

constexpr NativeFormat chooseNativeFormat(const Recipe& recipe)
{
    if (recipe.sourceIsFloat)
        return NativeFormat::RGBA32F;
    if (recipe.sourceIs16Bit)
        return NativeFormat::RGBA16;
    if (!recipe.compress)
        return NativeFormat::RGBA8;

    // BC1/ASTC colour encoders are not safe scalar storage. Their RGB endpoint
    // quantisation and block interpolation become visible steps after a
    // roughness lookup or a height derivative. Until a one-channel data format
    // is carried by both backends, keep authored data maps lossless at 8 bit.
    if (recipe.semantic == Semantic::NonColor)
        return NativeFormat::RGBA8;

    if (recipe.target == TargetProfile::AppleAstc)
        return recipe.semantic == Semantic::Normal ? NativeFormat::ASTC4x4 : NativeFormat::ASTC6x6;

    if (recipe.semantic == Semantic::Normal)
        return NativeFormat::BC5;
    return recipe.hasAlpha ? NativeFormat::BC3 : NativeFormat::BC1;
}

} // namespace oka::texture
