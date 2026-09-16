#pragma once

#include <host/texture_asset.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace oka::optix_tex
{

using Kind = texture::Semantic;
using Extent = texture::Extent;
using Format = texture::NativeFormat;
using texture::resolveExtent;

inline bool isCompressed(Format f)
{
    return texture::formatInfo(f).compressed;
}

/// Bytes per texel; 0 for compressed formats.
inline size_t texelBytes(Format f)
{
    const texture::FormatInfo info = texture::formatInfo(f);
    return info.compressed ? 0 : info.bytesPerBlock;
}

/// Decode settings read once per texture.
struct DecodeSettings
{
    uint32_t maxDimension = 0; // render/texture/maxDimension, 0 = unbounded
    uint32_t downscale = 1; // render/texture/downscale
    bool blockCompress = false; // render/texture/compress
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
        return texture::levelBytes(format, std::max(1, extent.width >> level), std::max(1, extent.height >> level));
    }

    size_t totalBytes() const
    {
        return levelBytes(0);
    }
};

inline Plan planTexture(const PlanInputs& in)
{
    Plan plan;
    plan.extent = resolveExtent(in.srcWidth, in.srcHeight, in.maxDimension, in.downscale);

    const bool wantsSrgb = in.kind == Kind::Color;
    const bool eightBit = !in.sourceIsFloat && !in.sourceIs16Bit;

    texture::Recipe recipe;
    recipe.semantic = in.kind;
    recipe.target = texture::TargetProfile::NvidiaBc;
    recipe.hasAlpha = in.hasAlpha;
    recipe.sourceIsFloat = in.sourceIsFloat;
    recipe.sourceIs16Bit = in.sourceIs16Bit;
    recipe.compress = in.blockCompress;
    plan.format = texture::chooseNativeFormat(recipe);

    plan.srgbBlockFormat = wantsSrgb && isCompressed(plan.format);
    plan.srgbTextureFlag = wantsSrgb && eightBit && !isCompressed(plan.format);
    plan.resampleInSrgb = wantsSrgb && eightBit;
    plan.normalizeLevels = in.kind == Kind::Normal;
    return plan;
}

} // namespace oka::optix_tex
