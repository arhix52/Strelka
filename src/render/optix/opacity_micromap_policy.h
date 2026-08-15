#pragma once

// How an opacity micromap is derived from the alpha test it is meant to
// shortcut, and the two rules that keep it from changing the image.
//
// An alpha cutout on this backend costs a shader invocation during traversal:
// shadow rays enter `__anyhit__occlusion` at every cutout triangle they cross,
// and radiance rays run the closest hit only to find they went straight
// through. An opacity micromap lets the traversal hardware answer for the
// regions that are wholly inside or wholly outside the cutout, and call the
// shader only where the answer actually varies.
//
// That is an acceleration, so the only interesting property is that it does not
// change what is shaded:
//
//   1. A microtriangle is called OPAQUE or TRANSPARENT only when *every* point
//      of it resolves that way under the same resolveOpacity() the shader runs.
//      Anything else -- including anything this file cannot bound exactly -- is
//      UNKNOWN, which is a fall-through to the shader and therefore always safe.
//
//   2. The bound is taken over texels, not over sample points. Sampling the
//      three corners of a microtriangle is what the SDK sample does and it is
//      not conservative: a cutout edge can cross the interior while missing all
//      three corners. Here the microtriangle's uv box is turned into the set of
//      texels a bilinear fetch anywhere inside it could read, and the classifier
//      is handed the min and max alpha over that set.
//
// The tolerance argument is the other half of rule 1. Alpha that reaches the
// GPU through BC3 comes back through an interpolated palette, and the exact
// rounding of that palette is not pinned down identically by every
// implementation. A band around the cutoff, inside which nothing is classified,
// costs a few microtriangles of shortcut and removes the entire question.
//
// The OptiX enumerator values are mirrored rather than included so that this
// file, and its tests, need no CUDA and no OptiX SDK. OptixRender.cpp
// static_asserts each mirror against the real enumerator.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace oka
{
namespace optix_omm
{

/// Mirrors of OPTIX_OPACITY_MICROMAP_STATE_*. Checked in OptixRender.cpp.
enum State : uint32_t
{
    kStateTransparent = 0u,
    kStateOpaque = 1u,
    kStateUnknownTransparent = 2u,
    kStateUnknownOpaque = 3u,
};

/// Mirrors of OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_*. A triangle in a
/// uniform state needs no micromap of its own; it names one of these instead
/// and costs four bytes rather than the whole subdivided array.
enum PredefinedIndex : int32_t
{
    kIndexFullyTransparent = -1,
    kIndexFullyOpaque = -2,
    kIndexFullyUnknownTransparent = -3,
    kIndexFullyUnknownOpaque = -4,
};

/// Mirror of OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL.
constexpr uint32_t kMaxSubdivisionLevel = 12u;

/// Mirrors of the ALPHA_MODE_* constants in material_params.h.
enum AlphaMode : uint32_t
{
    kAlphaOpaque = 0u,
    kAlphaMask = 1u,
    kAlphaBlend = 2u,
};

/// 4^level.
inline uint32_t microTriangleCount(uint32_t level)
{
    return 1u << (2u * level);
}

/// Bytes one triangle's micromap occupies in the 4-state format: two bits per
/// microtriangle. Level 0 is one microtriangle and still rounds up to a byte,
/// which is what the array's own alignment expects.
inline size_t microMapBytes(uint32_t level)
{
    const uint32_t states = microTriangleCount(level);
    return (size_t)((states * 2u + 7u) / 8u);
}

/// The finest subdivision `triangleCount` triangles can be given inside
/// `byteBudget`, never above `maxLevel`.
///
/// A budget rather than a constant because the cost is per triangle and the
/// benefit is not: level 4 is 64 bytes a triangle, which is nothing on the two
/// triangles of a cutout card and four gigabytes on a forest floor. Returns
/// kNoSubdivision when even level 1 does not fit, which the caller reads as
/// "do not build one".
constexpr uint32_t kNoSubdivision = 0xFFFFFFFFu;

inline uint32_t chooseSubdivisionLevel(size_t triangleCount, size_t byteBudget, uint32_t maxLevel)
{
    if (triangleCount == 0)
    {
        return kNoSubdivision;
    }
    maxLevel = std::min(maxLevel, kMaxSubdivisionLevel);
    for (uint32_t level = maxLevel + 1u; level-- > 1u;)
    {
        if (microMapBytes(level) * triangleCount <= byteBudget)
        {
            return level;
        }
    }
    return kNoSubdivision;
}

/// Write one microtriangle's state into the packed 4-state array.
///
/// Two bits each, four per byte, least significant pair first -- the layout
/// OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE reads. Written through bytes rather
/// than through the 16-bit words the SDK sample uses so the packing does not
/// depend on the host's endianness.
inline void setMicroState(uint8_t* states, uint32_t microIndex, uint32_t state)
{
    uint8_t& byte = states[microIndex >> 2];
    const uint32_t shift = (microIndex & 3u) * 2u;
    byte = (uint8_t)((byte & ~(0x3u << shift)) | ((state & 0x3u) << shift));
}

inline uint32_t getMicroState(const uint8_t* states, uint32_t microIndex)
{
    return (uint32_t)((states[microIndex >> 2] >> ((microIndex & 3u) * 2u)) & 0x3u);
}

/// The alpha test a material performs, as resolveOpacity() spells it.
struct AlphaRule
{
    uint32_t alphaMode = kAlphaOpaque;
    /// MaterialParams::base_color_alpha, the factor the texture multiplies.
    float baseAlpha = 1.0f;
    /// MaterialParams::alpha_cutoff. Read only in MASK.
    float cutoff = 0.5f;
    /// Whether a base-colour texture contributes an alpha channel at all. With
    /// no texture the coverage is constant over the whole material and every
    /// triangle classifies uniformly.
    bool hasTexture = false;
};

/// What a region of a surface resolves to under an AlphaRule.
enum class Coverage
{
    Transparent, ///< every point passes straight through
    Opaque, ///< every point is fully covered
    Mixed, ///< neither, or not provable -- the shader has to answer
};

/// Classify a region from the range of base-colour texture alpha over it.
///
/// `minTexAlpha` / `maxTexAlpha` bound the *texture's* alpha channel over
/// everything a bilinear fetch in the region could read; pass 1,1 when the
/// material has no texture. `tolerance` widens the Mixed band on both sides and
/// is therefore always safe to raise.
inline Coverage classifyCoverage(const AlphaRule& rule, float minTexAlpha, float maxTexAlpha, float tolerance)
{
    if (rule.alphaMode == kAlphaOpaque)
    {
        return Coverage::Opaque;
    }
    if (!(minTexAlpha <= maxTexAlpha))
    {
        return Coverage::Mixed; // an empty or NaN range says nothing
    }
    // base_color_alpha is a factor, and glTF does not allow it to be negative;
    // a scene that carries one anyway would invert the bound below, so refuse.
    if (!(rule.baseAlpha >= 0.0f))
    {
        return Coverage::Mixed;
    }
    const float lo = rule.baseAlpha * minTexAlpha;
    const float hi = rule.baseAlpha * maxTexAlpha;
    const float tol = std::max(0.0f, tolerance);

    if (rule.alphaMode == kAlphaMask)
    {
        // resolveOpacity: alpha >= cutoff ? 1 : 0.
        if (lo - tol >= rule.cutoff)
        {
            return Coverage::Opaque;
        }
        if (hi + tol < rule.cutoff)
        {
            return Coverage::Transparent;
        }
        return Coverage::Mixed;
    }

    // BLEND. resolveOpacity returns saturate(alpha), and the shader treats
    // anything below 1 as partially transparent, so the only opaque region is
    // the one that saturates. No slack in the other direction either: an alpha
    // of a thousandth is not zero, and calling it transparent would lose the
    // light it blocks.
    if (lo - tol >= 1.0f)
    {
        return Coverage::Opaque;
    }
    if (hi + tol <= 0.0f)
    {
        return Coverage::Transparent;
    }
    return Coverage::Mixed;
}

/// The microtriangle state for a coverage.
///
/// Mixed becomes UNKNOWN_OPAQUE rather than UNKNOWN_TRANSPARENT. The two are
/// identical while the 4-state format is in force -- both call the shader -- and
/// differ only if an instance ever asks for FORCE_OPACITY_MICROMAP_2_STATE, at
/// which point opaque is the reading that keeps the closest hit running and so
/// keeps the stochastic cutout test happening at all.
inline uint32_t microStateFor(Coverage coverage)
{
    switch (coverage)
    {
    case Coverage::Transparent:
        return kStateTransparent;
    case Coverage::Opaque:
        return kStateOpaque;
    case Coverage::Mixed:
        break;
    }
    return kStateUnknownOpaque;
}

/// The predefined index for a triangle that is uniform over its whole area.
inline int32_t predefinedIndexFor(Coverage coverage)
{
    switch (coverage)
    {
    case Coverage::Transparent:
        return kIndexFullyTransparent;
    case Coverage::Opaque:
        return kIndexFullyOpaque;
    case Coverage::Mixed:
        break;
    }
    return kIndexFullyUnknownOpaque;
}

/// A half-open run of texel indices along one axis, before wrapping.
struct TexelSpan
{
    int lo = 0;
    int hi = 0; ///< inclusive
    /// True when the span already covers the whole axis, so wrapping it would
    /// only revisit texels.
    bool full = false;

    int count() const
    {
        return full ? 0 : (hi - lo + 1);
    }
};

/// Every texel a bilinear fetch anywhere in [c0, c1] of a normalised coordinate
/// can read, for an axis of `size` texels.
///
/// CUDA's linear filter at normalised u reads the two texels either side of
/// `u * size - 0.5`, so the run starts one texel below the box and ends one
/// above it. `pad` is added to the box first, which absorbs the difference
/// between the host's evaluation of a barycentric interpolation and the
/// device's -- they are the same expression but not the same instruction
/// sequence, and a uv that lands exactly on a texel boundary must not depend on
/// which way the last bit went.
inline TexelSpan bilinearTexelSpan(float c0, float c1, int size, float pad)
{
    TexelSpan span;
    if (size <= 0)
    {
        span.full = true;
        return span;
    }
    if (c1 < c0)
    {
        std::swap(c0, c1);
    }
    const double scale = (double)size;
    const double lo = ((double)c0 - (double)pad) * scale - 0.5;
    const double hi = ((double)c1 + (double)pad) * scale + 0.5;

    // A box wider than the texture wraps onto itself; there is nothing left to
    // exclude, so say so rather than iterating millions of aliases of the same
    // texels.
    if (!(hi - lo < (double)size))
    {
        span.full = true;
        return span;
    }
    const double lowIndex = std::floor(lo);
    const double highIndex = std::floor(hi);
    // Outside what an int can hold the span is meaningless; treat it as the
    // whole axis, which is conservative.
    if (lowIndex < -1073741824.0 || highIndex > 1073741824.0)
    {
        span.full = true;
        return span;
    }
    span.lo = (int)lowIndex;
    span.hi = (int)highIndex;
    if (span.count() >= size)
    {
        span.full = true;
    }
    return span;
}

/// Wrap a possibly negative texel index onto [0, size), which is what
/// cudaAddressModeWrap does.
inline int wrapTexel(int index, int size)
{
    if (size <= 0)
    {
        return 0;
    }
    const int m = index % size;
    return m < 0 ? m + size : m;
}

/// Decode the eight alpha bytes of a BC4 / BC3 block into sixteen texels.
///
/// The GPU reads a block-compressed base colour through this palette, so a
/// micromap built from the *uncompressed* file would be describing a texture the
/// renderer does not have. Row-major within the block, matching the layout
/// oka::bc::compressBlockBC4 writes.
inline void decodeBc4AlphaBlock(const uint8_t block[8], uint8_t out[16])
{
    const int a0 = block[0];
    const int a1 = block[1];
    int palette[8];
    palette[0] = a0;
    palette[1] = a1;
    if (a0 > a1)
    {
        for (int k = 2; k < 8; ++k)
        {
            palette[k] = ((8 - k) * a0 + (k - 1) * a1) / 7;
        }
    }
    else
    {
        for (int k = 2; k < 6; ++k)
        {
            palette[k] = ((6 - k) * a0 + (k - 1) * a1) / 5;
        }
        palette[6] = 0;
        palette[7] = 255;
    }

    uint64_t indices = 0;
    for (int b = 0; b < 6; ++b)
    {
        indices |= (uint64_t)block[2 + b] << (8 * b);
    }
    for (int i = 0; i < 16; ++i)
    {
        out[i] = (uint8_t)palette[(indices >> (3 * i)) & 0x7u];
    }
}

/// How far a decoded alpha may be from the value the GPU's sampler returns.
///
/// Zero for a format the host and the device read the same bytes of. Two
/// levels for BC4-style alpha, which is an integer palette whose rounding the
/// specification states as an exact expression but which implementations have
/// been known to round rather than truncate. It buys back a band of
/// microtriangles around the cutoff and nothing else.
constexpr float kExactAlphaTolerance = 0.0f;
constexpr float kBlockCompressedAlphaTolerance = 2.0f / 255.0f;

/// What one mesh's micromaps came to, for the log and for the tests.
struct BuildSummary
{
    size_t triangles = 0;
    size_t uniformOpaque = 0;
    size_t uniformTransparent = 0;
    size_t uniformUnknown = 0;
    size_t subdivided = 0; ///< triangles that got a micromap of their own
    uint32_t subdivisionLevel = 0;
    /// Microtriangles the classifier looked at, and how many of them it could
    /// answer for. The ratio is the whole of what a micromap is worth:
    /// all-unknown is an array that costs memory and answers nothing, and the
    /// log has to be able to say which of the two happened.
    size_t microTriangles = 0;
    size_t microResolved = 0;

    size_t micromapCount() const
    {
        return subdivided;
    }

    /// Nothing was resolved, so the micromap would answer "ask the shader"
    /// everywhere -- which is what happens without one, at the cost of the
    /// array. The caller drops it.
    bool isPointless() const
    {
        return subdivided == 0 && uniformOpaque == 0 && uniformTransparent == 0;
    }
};

} // namespace optix_omm
} // namespace oka
