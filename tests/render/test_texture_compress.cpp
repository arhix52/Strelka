#include <doctest/doctest.h>

#include <host/texture_compress.h>
#include <strelka/material/normal_filter.h>

#include "../support/sampling.h"

#include <cmath>
#include <vector>

using oka::test::unorm8;

namespace
{

struct Normal
{
    float x, y, z;
};

/// A field with both the smooth gradients BC1 cannot hold and the sharp creases
/// that make a block span a wide range of directions.
std::vector<uint8_t> makeNormalMap(int width, int height)
{
    std::vector<uint8_t> rgba((size_t)width * height * 4, 0);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            // Bumps on a 16-pixel grid, so a 4x4 block sees anything from a flat
            // area to the steep side of a bump.
            const float u = (float)(x % 16) / 16.0f * 2.0f - 1.0f;
            const float v = (float)(y % 16) / 16.0f * 2.0f - 1.0f;
            const float r2 = u * u + v * v;
            float nx = u * 0.9f;
            float ny = v * 0.9f;
            if (r2 > 1.0f)
            {
                // Between the bumps the surface tilts slowly: the smooth ramp.
                nx = 0.25f * (float)x / (float)width;
                ny = 0.25f * (float)y / (float)height;
            }
            const float nz = std::sqrt(std::max(0.0f, 1.0f - nx * nx - ny * ny));
            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            uint8_t* p = rgba.data() + ((size_t)y * width + x) * 4;
            p[0] = (uint8_t)std::lround((nx / len * 0.5f + 0.5f) * 255.0f);
            p[1] = (uint8_t)std::lround((ny / len * 0.5f + 0.5f) * 255.0f);
            p[2] = (uint8_t)std::lround((nz / len * 0.5f + 0.5f) * 255.0f);
            p[3] = 255;
        }
    }
    return rgba;
}

/// One channel of a BC4 block. Mirrors the endpoint modes the encoder writes.
void decodeBC4(const uint8_t block[8], uint8_t out[16])
{
    const int hi = block[0];
    const int lo = block[1];
    int palette[8];
    palette[0] = hi;
    palette[1] = lo;
    if (hi > lo)
    {
        for (int k = 1; k < 7; ++k)
        {
            palette[k + 1] = ((7 - k) * hi + k * lo) / 7;
        }
    }
    else
    {
        for (int k = 1; k < 5; ++k)
        {
            palette[k + 1] = ((5 - k) * hi + k * lo) / 5;
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
        out[i] = (uint8_t)palette[(indices >> (3 * i)) & 0x7];
    }
}

void decodeBC1(const uint8_t block[8], uint8_t out[16][3])
{
    const uint16_t c0 = (uint16_t)(block[0] | (block[1] << 8));
    const uint16_t c1 = (uint16_t)(block[2] | (block[3] << 8));
    int e0[3], e1[3];
    oka::bc::unpack565(c0, e0[0], e0[1], e0[2]);
    oka::bc::unpack565(c1, e1[0], e1[1], e1[2]);

    int palette[4][3];
    for (int c = 0; c < 3; ++c)
    {
        palette[0][c] = e0[c];
        palette[1][c] = e1[c];
        palette[2][c] = (2 * e0[c] + e1[c]) / 3;
        palette[3][c] = (e0[c] + 2 * e1[c]) / 3;
    }

    uint32_t indices = 0;
    std::memcpy(&indices, block + 4, 4);
    for (int i = 0; i < 16; ++i)
    {
        const uint32_t k = (indices >> (2 * i)) & 0x3u;
        for (int c = 0; c < 3; ++c)
        {
            out[i][c] = (uint8_t)palette[k][c];
        }
    }
}

Normal unpackNormal(float x, float y, float z)
{
    const float nx = x * 2.0f - 1.0f;
    const float ny = y * 2.0f - 1.0f;
    const float nz = z * 2.0f - 1.0f;
    const float len = std::max(1e-6f, std::sqrt(nx * nx + ny * ny + nz * nz));
    return { nx / len, ny / len, nz / len };
}

/// Z from X and Y, which is how the shader reads a BC5 normal map.
Normal reconstructNormal(float x, float y)
{
    const float nx = x * 2.0f - 1.0f;
    const float ny = y * 2.0f - 1.0f;
    const float nz = std::sqrt(std::max(0.0f, 1.0f - nx * nx - ny * ny));
    const float len = std::max(1e-6f, std::sqrt(nx * nx + ny * ny + nz * nz));
    return { nx / len, ny / len, nz / len };
}

float angleDegrees(const Normal& a, const Normal& b)
{
    const float d = std::clamp(a.x * b.x + a.y * b.y + a.z * b.z, -1.0f, 1.0f);
    return std::acos(d) * 57.2957795f;
}

struct Error
{
    float mean = 0.0f;
    float max = 0.0f;
};

Error measure(const std::vector<uint8_t>& rgba, int width, int height, oka::bc::Format format)
{
    const std::vector<uint8_t> encoded = oka::bc::compressImage(rgba.data(), width, height, format);
    const int blocksX = (width + 3) / 4;
    const size_t blockSize = oka::bc::blockBytes(format);

    Error err;
    double sum = 0.0;
    size_t count = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const uint8_t* block = encoded.data() + ((size_t)(y / 4) * blocksX + x / 4) * blockSize;
            const int i = (y % 4) * 4 + (x % 4);

            Normal decoded{};
            if (format == oka::bc::Format::BC5)
            {
                uint8_t xs[16], ys[16];
                decodeBC4(block, xs);
                decodeBC4(block + 8, ys);
                decoded = reconstructNormal(unorm8(xs[i]), unorm8(ys[i]));
            }
            else
            {
                uint8_t rgb[16][3];
                decodeBC1(block, rgb);
                decoded = unpackNormal(unorm8(rgb[i][0]), unorm8(rgb[i][1]), unorm8(rgb[i][2]));
            }

            const uint8_t* p = rgba.data() + ((size_t)y * width + x) * 4;
            const Normal want = unpackNormal(unorm8(p[0]), unorm8(p[1]), unorm8(p[2]));
            const float angle = angleDegrees(decoded, want);
            sum += angle;
            err.max = std::max(err.max, angle);
            ++count;
        }
    }
    err.mean = (float)(sum / (double)count);
    return err;
}

} // namespace

TEST_CASE("BC5 keeps a normal map within a degree of the source")
{
    const int width = 64;
    const int height = 64;
    const std::vector<uint8_t> rgba = makeNormalMap(width, height);

    const Error bc5 = measure(rgba, width, height, oka::bc::Format::BC5);
    CHECK(bc5.mean < 2.0f);
    CHECK(bc5.max < 8.0f);
}

TEST_CASE("BC1 is what normal maps are compressed with instead of")
{
    const int width = 64;
    const int height = 64;
    const std::vector<uint8_t> rgba = makeNormalMap(width, height);

    const Error bc1 = measure(rgba, width, height, oka::bc::Format::BC1);
    const Error bc5 = measure(rgba, width, height, oka::bc::Format::BC5);

    CHECK(bc1.mean > 2.0f * bc5.mean);
}

TEST_CASE("Compressed sizes are the block counts Metal expects")
{
    // 65 rounds up to 17 blocks, which is the case where a level's rows do not
    // divide by four and the encoder has to clamp instead of reading past.
    const int width = 65;
    const int height = 33;
    const std::vector<uint8_t> rgba = makeNormalMap(width, height);
    const size_t blocks = (size_t)((width + 3) / 4) * ((height + 3) / 4);

    CHECK(oka::bc::compressImage(rgba.data(), width, height, oka::bc::Format::BC1).size() == blocks * 8);
    CHECK(oka::bc::compressImage(rgba.data(), width, height, oka::bc::Format::BC3).size() == blocks * 16);
    CHECK(oka::bc::compressImage(rgba.data(), width, height, oka::bc::Format::BC5).size() == blocks * 16);
}

TEST_CASE("normalizeNormalMap makes Z reconstruction exact")
{
    // A texel a few percent off unit length, which is what a normal map off a
    // lossy encoder looks like. Without the pass, the rebuilt Z is not the Z that
    // was stored and the direction shifts.
    std::vector<uint8_t> rgba = { 200, 90, 230, 255 };
    const Normal source = unpackNormal(unorm8(rgba[0]), unorm8(rgba[1]), unorm8(rgba[2]));
    const Normal before = reconstructNormal(unorm8(rgba[0]), unorm8(rgba[1]));
    CHECK(angleDegrees(before, source) > 0.5f);

    oka::bc::normalizeNormalMap(rgba.data(), 1, 1);
    const Normal after = reconstructNormal(unorm8(rgba[0]), unorm8(rgba[1]));
    CHECK(angleDegrees(after, source) < 0.2f); // what is left is the 8-bit rounding
}

TEST_CASE("normalizeNormalMap keeps the direction of a non-unit texel")
{
    for (const uint8_t grey : { (uint8_t)200, (uint8_t)128 })
    {
        std::vector<uint8_t> rgba = { grey, grey, grey, 255 };
        const Normal source = unpackNormal(unorm8(grey), unorm8(grey), unorm8(grey));

        oka::bc::normalizeNormalMap(rgba.data(), 1, 1);
        const Normal after = reconstructNormal(unorm8(rgba[0]), unorm8(rgba[1]));
        CHECK(angleDegrees(after, source) < 0.5f);
    }
}

TEST_CASE("normal mipmaps retain directional concentration for specular filtering")
{
    // Two unit normals whose mean is (0, 0, 0.8). A conventional normalized
    // mip would keep only (0, 0, 1); the alpha moment must retain the missing
    // 0.2 so shading can broaden the microfacet lobe.
    std::vector<uint8_t> src = {
        204, 128, 230, 17,
        51, 128, 230, 93,
    };
    oka::bc::normalizeNormalMap(src.data(), 2, 1);
    CHECK(src[3] == 255);
    CHECK(src[7] == 255);

    std::vector<uint8_t> mip(4);
    oka::bc::downsampleNormalMoments(src.data(), 2, 1, mip.data(), 1, 1);
    const Normal direction = unpackNormal(unorm8(mip[0]), unorm8(mip[1]), unorm8(mip[2]));
    CHECK(direction.x == doctest::Approx(0.0f).epsilon(0.02f));
    CHECK(direction.z == doctest::Approx(1.0f).epsilon(0.01f));
    CHECK(unorm8(mip[3]) == doctest::Approx(0.8f).epsilon(0.02f));

    CHECK(normal_filter_roughness(0.02f, 0.82f, 0.15f) == doctest::Approx(0.3f).epsilon(0.001f));
    CHECK(normal_filter_roughness(0.2f, 1.0f, 0.15f) == doctest::Approx(0.2f));
    CHECK(normal_filter_roughness(0.2f, 0.0f, 0.15f) == doctest::Approx(0.2f));
}

TEST_CASE("A flat normal map survives BC5 exactly")
{
    // The common case: most of a normal map is (0.5, 0.5, 1). A block that is
    // one value has both endpoints equal, which selects the six-value mode --
    // where index 6 and 7 decode to 0 and 255 rather than to the endpoints.
    const int width = 8;
    const int height = 8;
    std::vector<uint8_t> rgba((size_t)width * height * 4, 255);
    for (size_t i = 0; i < (size_t)width * height; ++i)
    {
        rgba[i * 4 + 0] = 128;
        rgba[i * 4 + 1] = 128;
        rgba[i * 4 + 2] = 255;
    }

    const Error bc5 = measure(rgba, width, height, oka::bc::Format::BC5);
    CHECK(bc5.max < 0.01f);
}
