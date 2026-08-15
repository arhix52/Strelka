#pragma once

// How a material texture becomes a CUDA array: the arithmetic, with no CUDA in
// it.
//
// Every decision the OptiX texture loader makes that does not need a device --
// the resampled extent, how many mip levels that extent has, which texel format
// carries it, whether the texture object does the sRGB decode or the format
// does, and how many bytes each level occupies -- lives here so it can be
// tested without a GPU. `OptixTextures.cpp` holds the CUDA calls and nothing
// else that can be got wrong arithmetically.
//
// The extent rule is deliberately identical to MetalTextures::decodeToPayload:
// divide by `downscale`, then halve until the longest edge fits `maxDimension`.
// Two backends that disagree about which pixels they are filtering cannot be
// compared, and the ladder compares them.

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace oka
{
namespace optix_tex
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
    int w = std::max(1, srcWidth / divisor);
    int h = std::max(1, srcHeight / divisor);
    while (maxDimension > 0 && (uint32_t)std::max(w, h) > maxDimension && (w > 1 || h > 1))
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
    const uint32_t longest = (uint32_t)std::max(1, std::max(width, height));
    while ((1u << levels) <= longest)
        ++levels;
    return levels;
}

/// Levels for `format`. A block-compressed chain stops at the last level whose
/// shorter edge is still a whole 4x4 block: below that a level is mostly the
/// encoder's padding, and CUDA's block-compressed arrays are the one place
/// where a 2x2 level buys a driver-side special case for 8 bytes of saving.
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
    /// Whether mip levels are wanted at all. Off by default: `tex2D` from a ray
    /// tracing program has no derivatives, so it reads level 0 and a chain is
    /// 33% of memory for nothing until ray cones select a level.
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

/// The whole decision, in one place.
///
/// Rules, in the order they bind:
///  * Only Kind::Color is sRGB encoded. A roughness or an occlusion map read
///    through a transfer function is the classic silent shading error, and a
///    normal map read through one is not a direction any more.
///  * Float and 16-bit sources are never block compressed: BC1/3/5 are 8-bit
///    formats, and a source that bothered to carry more than 8 bits is the last
///    thing to quantise.
///  * A 16-bit sRGB source is decoded on the host, because CUDA's sRGB texture
///    flag is defined for 8-bit unorm formats only. The plan says so by leaving
///    `srgbTextureFlag` clear on a RGBA16 colour texture; the loader is
///    responsible for having linearised the bytes.
///  * Normal maps take BC5 when compressed, which stores X and Y only, and are
///    re-normalised at every level whether compressed or not so the two paths
///    shade alike. This is the same rule MetalTextures applies.
///
///    READ THIS BEFORE SAMPLING SLOT 2. A BC5 texture returns z = 0. Whoever
///    wires normal mapping into the OptiX closest-hit program must rebuild Z
///    from X and Y -- `sqrt(saturate(1 - dot(xy, xy)))`, before the glTF
///    normal scale is applied to X and Y -- exactly as
///    src/shaders/metal/shading_common.h does. Reading .xyz straight out of the
///    texture gives a flat normal on every compressed map and a correct one on
///    every uncompressed map, which is the kind of difference that gets blamed
///    on the tangent frame.
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

} // namespace optix_tex
} // namespace oka
