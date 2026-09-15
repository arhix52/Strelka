#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>

namespace oka::optix_tex
{

/// What the texture is for. Decides the transfer function and, when the texture
/// is compressed, the block format.
enum class Kind : int
{
    Color = 0, // base colour, emission -- sRGB encoded on disk
    NonColor = 1, // metallic-roughness, occlusion -- linear on disk
    Normal = 2, // tangent-space normal -- linear, and unit length
};

/// The texel format the CUDA array is created with.
enum class Format : int
{
    RGBA8 = 0, // uchar4, read as normalized float
    RGBA16 = 1, // ushort4, read as normalized float
    RGBA32F = 2, // float4, read as element type
    BC1 = 3, // opaque colour, 8 bytes / block
    BC3 = 4, // colour + alpha, 16 bytes / block
    BC5 = 5, // two channels, 16 bytes / block -- normal maps
};

inline bool isCompressed(Format f)
{
    return f == Format::BC1 || f == Format::BC3 || f == Format::BC5;
}

/// Bytes per 4x4 block; 0 for uncompressed formats.
inline size_t blockBytes(Format f)
{
    switch (f)
    {
    case Format::BC1:
        return 8;
    case Format::BC3:
    case Format::BC5:
        return 16;
    default:
        return 0;
    }
}

/// Bytes per texel; 0 for compressed formats.
inline size_t texelBytes(Format f)
{
    switch (f)
    {
    case Format::RGBA8:
        return 4;
    case Format::RGBA16:
        return 8;
    case Format::RGBA32F:
        return 16;
    default:
        return 0;
    }
}

/// Bytes one mip level of `width` x `height` occupies.
inline size_t levelBytes(Format f, int width, int height)
{
    const size_t w = (size_t)std::max(1, width);
    const size_t h = (size_t)std::max(1, height);
    if (isCompressed(f))
        return ((w + 3) / 4) * ((h + 3) / 4) * blockBytes(f);
    return w * h * texelBytes(f);
}

struct Extent
{
    int width = 0;
    int height = 0;

    bool operator==(const Extent& o) const
    {
        return width == o.width && height == o.height;
    }
};

/// `downscale` first, then halve until the longest edge is within
/// `maxDimension`. `maxDimension == 0` means unbounded, `downscale == 0` is read
/// as 1. Never returns an edge below 1.
inline Extent resolveExtent(int srcWidth, int srcHeight, uint32_t maxDimension, uint32_t downscale)
{
    const int divisor = (int)std::max(1u, downscale);
    // Both bounds in one signedness: a texture edge that does not fit in an int
    // is past every limit a GPU has anyway, so the clamp cannot change a
    // decision.
    const int maxEdge = (int)std::min<uint32_t>(maxDimension, (uint32_t)INT_MAX);
    int w = std::max(1, srcWidth / divisor);
    int h = std::max(1, srcHeight / divisor);
    while (maxEdge > 0 && std::max(w, h) > maxEdge && (w > 1 || h > 1))
    {
        w = std::max(1, w / 2);
        h = std::max(1, h / 2);
    }
    return Extent{ w, h };
}

/// Levels in a full chain down to 1x1, which is what an uncompressed array
/// takes.
inline uint32_t fullMipLevelCount(int width, int height)
{
    uint32_t levels = 1;
    const uint32_t longest = (uint32_t)std::max({ 1, width, height });
    while ((1u << levels) <= longest)
        ++levels;
    return levels;
}

inline uint32_t mipLevelCount(Format format, int width, int height)
{
    const uint32_t full = fullMipLevelCount(width, height);
    if (!isCompressed(format))
        return full;
    uint32_t levels = 1;
    for (uint32_t l = 1; l < full; ++l)
    {
        const int w = std::max(1, width >> l);
        const int h = std::max(1, height >> l);
        if (std::min(w, h) < 4)
            break;
        levels = l + 1;
    }
    return levels;
}

/// The four knobs the settings carry, read once per texture.
struct DecodeSettings
{
    uint32_t maxDimension = 0; // render/texture/maxDimension, 0 = unbounded
    uint32_t downscale = 1; // render/texture/downscale
    bool blockCompress = false; // render/texture/compress
    bool wantMips = false; // render/texture/mips
};

struct PlanInputs
{
    int srcWidth = 0;
    int srcHeight = 0;
    uint32_t maxDimension = 0;
    uint32_t downscale = 1;
    Kind kind = Kind::Color;
    /// The file decoded to floats (a .hdr or .exr used as a material texture).
    bool sourceIsFloat = false;
    /// The file decoded to 16 bits per channel.
    bool sourceIs16Bit = false;
    /// `render/texture/compress`, and the device agreeing to it.
    bool blockCompress = false;
    /// Whether any texel's alpha is below opaque. Only consulted for colour.
    bool hasAlpha = false;
    /// Whether mip levels are wanted. Off until ray cones select levels because
    /// `tex2D` has no ray-tracing derivatives and otherwise reads level 0.
    bool wantMips = false;
};

struct Plan
{
    Extent extent;
    uint32_t levels = 1;
    Format format = Format::RGBA8;
    /// Set the texture object's sRGB flag. Mutually exclusive with a
    /// `*_SRGB` block format, which does the same decode in the format.
    bool srgbTextureFlag = false;
    /// Ask CUDA for the sRGB block-compressed channel kind.
    bool srgbBlockFormat = false;
    /// Re-normalise every level after resampling. Normal maps only.
    bool normalizeLevels = false;
    /// Resample in sRGB space rather than on the encoded bytes.
    bool resampleInSrgb = false;

    size_t levelBytes(uint32_t level) const
    {
        return oka::optix_tex::levelBytes(format, std::max(1, extent.width >> level),
                                          std::max(1, extent.height >> level));
    }

    size_t totalBytes() const
    {
        size_t total = 0;
        for (uint32_t l = 0; l < levels; ++l)
            total += levelBytes(l);
        return total;
    }
};

inline Plan planTexture(const PlanInputs& in)
{
    Plan plan;
    plan.extent = resolveExtent(in.srcWidth, in.srcHeight, in.maxDimension, in.downscale);

    const bool wantsSrgb = in.kind == Kind::Color;
    const bool eightBit = !in.sourceIsFloat && !in.sourceIs16Bit;

    if (in.sourceIsFloat)
    {
        plan.format = Format::RGBA32F;
    }
    else if (in.sourceIs16Bit)
    {
        plan.format = Format::RGBA16;
    }
    else if (in.blockCompress)
    {
        if (in.kind == Kind::Normal)
            plan.format = Format::BC5;
        else if (in.hasAlpha)
            plan.format = Format::BC3;
        else
            plan.format = Format::BC1;
    }
    else
    {
        plan.format = Format::RGBA8;
    }

    plan.srgbBlockFormat = wantsSrgb && isCompressed(plan.format);
    plan.srgbTextureFlag = wantsSrgb && eightBit && !isCompressed(plan.format);
    plan.resampleInSrgb = wantsSrgb && eightBit;
    plan.normalizeLevels = in.kind == Kind::Normal;
    plan.levels = in.wantMips ? mipLevelCount(plan.format, plan.extent.width, plan.extent.height) : 1u;
    return plan;
}

} // namespace oka::optix_tex

