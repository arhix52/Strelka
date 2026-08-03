#pragma once

// Block compression for the texture cache: BC1 for opaque colour, BC3 where
// alpha carries something.
//
// Why these and not BC7, which is better: BC7 is a search over eight block modes
// and takes long enough that it wants a GPU encoder or a build step. BC1 and BC3
// are a range fit over a 4x4 block -- a few hundred instructions -- and they are
// what makes the difference between a scene fitting in memory and not. The
// results are cached, so the encode is paid once per texture per setting rather
// than once per launch.
//
// What they cost: BC1 quantises colour to 5:6:5 endpoints and four interpolated
// levels, which is visible on smooth gradients and invisible on the bark, moss
// and foliage that fill a scene like this. Normal maps are deliberately left
// uncompressed -- a two-bit index along a line through 5:6:5 space is not enough
// for a direction, and the banding shows up as facets on every curved surface.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace oka::bc
{

inline uint16_t pack565(int r, int g, int b)
{
    return (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}

inline void unpack565(uint16_t c, int& r, int& g, int& b)
{
    r = ((c >> 11) & 0x1F) * 255 / 31;
    g = ((c >> 5) & 0x3F) * 255 / 63;
    b = (c & 0x1F) * 255 / 31;
}

/// One 4x4 BC1 colour block, 8 bytes. `src` is RGBA8, `stride` in bytes.
///
/// Range fit: the endpoints are the per-channel extremes of the block, which is
/// what makes this fast. It is not optimal -- a least-squares fit along the
/// principal axis is better -- but the difference is small on photographic
/// texture and the cost is not.
inline void compressBlockBC1(const uint8_t* src, int width, int height, int x0, int y0,
                             size_t stride, uint8_t out[8])
{
    int lo[3] = { 255, 255, 255 };
    int hi[3] = { 0, 0, 0 };
    uint8_t pixels[16][3];

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            // Blocks run past the edge on non-multiple-of-four sizes; clamp
            // rather than read outside, which would sample the next row.
            const int sx = std::min(x0 + x, width - 1);
            const int sy = std::min(y0 + y, height - 1);
            const uint8_t* p = src + (size_t)sy * stride + (size_t)sx * 4;
            for (int c = 0; c < 3; ++c)
            {
                pixels[y * 4 + x][c] = p[c];
                lo[c] = std::min(lo[c], (int)p[c]);
                hi[c] = std::max(hi[c], (int)p[c]);
            }
        }
    }

    uint16_t c0 = pack565(hi[0], hi[1], hi[2]);
    uint16_t c1 = pack565(lo[0], lo[1], lo[2]);
    // c0 > c1 selects the four-colour mode; equal endpoints would select the
    // three-colour one and lose a quarter of the palette for a flat block.
    if (c0 < c1)
        std::swap(c0, c1);

    int e0[3], e1[3];
    unpack565(c0, e0[0], e0[1], e0[2]);
    unpack565(c1, e1[0], e1[1], e1[2]);

    int palette[4][3];
    for (int c = 0; c < 3; ++c)
    {
        palette[0][c] = e0[c];
        palette[1][c] = e1[c];
        palette[2][c] = (2 * e0[c] + e1[c]) / 3;
        palette[3][c] = (e0[c] + 2 * e1[c]) / 3;
    }

    uint32_t indices = 0;
    for (int i = 0; i < 16; ++i)
    {
        int best = 0;
        int bestDist = 1 << 30;
        for (int k = 0; k < 4; ++k)
        {
            int dist = 0;
            for (int c = 0; c < 3; ++c)
            {
                const int d = (int)pixels[i][c] - palette[k][c];
                dist += d * d;
            }
            if (dist < bestDist)
            {
                bestDist = dist;
                best = k;
            }
        }
        indices |= (uint32_t)best << (2 * i);
    }

    out[0] = (uint8_t)(c0 & 0xFF);
    out[1] = (uint8_t)(c0 >> 8);
    out[2] = (uint8_t)(c1 & 0xFF);
    out[3] = (uint8_t)(c1 >> 8);
    std::memcpy(out + 4, &indices, 4);
}

/// One 4x4 BC4 alpha block, 8 bytes: two endpoints and three bits per pixel.
inline void compressBlockAlpha(const uint8_t* src, int width, int height, int x0, int y0,
                               size_t stride, uint8_t out[8])
{
    uint8_t values[16];
    int lo = 255, hi = 0;
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            const int sx = std::min(x0 + x, width - 1);
            const int sy = std::min(y0 + y, height - 1);
            const uint8_t a = src[(size_t)sy * stride + (size_t)sx * 4 + 3];
            values[y * 4 + x] = a;
            lo = std::min(lo, (int)a);
            hi = std::max(hi, (int)a);
        }
    }

    out[0] = (uint8_t)hi;
    out[1] = (uint8_t)lo;

    // a0 > a1 selects the eight-value mode. Equal endpoints mean a flat block
    // and every index resolves to the same value either way.
    int palette[8];
    palette[0] = hi;
    palette[1] = lo;
    if (hi > lo)
    {
        for (int k = 1; k < 7; ++k)
            palette[k + 1] = ((7 - k) * hi + k * lo) / 7;
    }
    else
    {
        for (int k = 1; k < 5; ++k)
            palette[k + 1] = ((5 - k) * hi + k * lo) / 5;
        palette[6] = 0;
        palette[7] = 255;
    }

    uint64_t indices = 0;
    for (int i = 0; i < 16; ++i)
    {
        int best = 0;
        int bestDist = 1 << 30;
        for (int k = 0; k < 8; ++k)
        {
            const int d = std::abs((int)values[i] - palette[k]);
            if (d < bestDist)
            {
                bestDist = d;
                best = k;
            }
        }
        indices |= (uint64_t)best << (3 * i);
    }
    for (int b = 0; b < 6; ++b)
        out[2 + b] = (uint8_t)((indices >> (8 * b)) & 0xFF);
}

/// Whether any pixel's alpha is not fully opaque, which decides BC1 against BC3.
inline bool hasAlpha(const uint8_t* rgba, int width, int height)
{
    const size_t count = (size_t)width * height;
    for (size_t i = 0; i < count; ++i)
    {
        if (rgba[i * 4 + 3] != 255)
            return true;
    }
    return false;
}

/// Compress an RGBA8 image. `blockBytes` is 8 for BC1, 16 for BC3.
// compressImage, not compress: miniz -- pulled in by tinyexr -- takes that name
// as a macro, and the collision is reported at the call site rather than here.
inline std::vector<uint8_t> compressImage(const uint8_t* rgba, int width, int height, bool withAlpha)
{
    const int blocksX = (width + 3) / 4;
    const int blocksY = (height + 3) / 4;
    const int blockBytes = withAlpha ? 16 : 8;
    std::vector<uint8_t> out((size_t)blocksX * blocksY * blockBytes);
    const size_t stride = (size_t)width * 4;

    for (int by = 0; by < blocksY; ++by)
    {
        for (int bx = 0; bx < blocksX; ++bx)
        {
            uint8_t* dst = out.data() + ((size_t)by * blocksX + bx) * blockBytes;
            if (withAlpha)
            {
                compressBlockAlpha(rgba, width, height, bx * 4, by * 4, stride, dst);
                compressBlockBC1(rgba, width, height, bx * 4, by * 4, stride, dst + 8);
            }
            else
            {
                compressBlockBC1(rgba, width, height, bx * 4, by * 4, stride, dst);
            }
        }
    }
    return out;
}

/// Bytes per row of blocks, which is what Metal wants for a compressed level.
inline size_t bytesPerRow(int width, bool withAlpha)
{
    return (size_t)((width + 3) / 4) * (withAlpha ? 16 : 8);
}

} // namespace oka::bc
