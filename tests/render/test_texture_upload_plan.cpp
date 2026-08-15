#include <doctest/doctest.h>

#include "texture_upload_plan.h"

using namespace oka::optix_tex;

// The OptiX texture loader's arithmetic. Everything here runs on the host with
// no CUDA and no GPU: the parts that need a device are in
// texture_support_cuda.h, and the parts that can be got wrong silently are
// here.

TEST_CASE("resolveExtent matches the Metal rule: downscale, then fit maxDimension")
{
    // Neither knob set: the file's own size.
    CHECK(resolveExtent(2048, 1024, 0, 1) == Extent{ 2048, 1024 });

    // downscale alone divides both edges.
    CHECK(resolveExtent(2048, 1024, 0, 2) == Extent{ 1024, 512 });
    CHECK(resolveExtent(2048, 1024, 0, 4) == Extent{ 512, 256 });

    // maxDimension alone halves until the LONGEST edge fits, keeping the aspect.
    CHECK(resolveExtent(2048, 1024, 512, 1) == Extent{ 512, 256 });
    CHECK(resolveExtent(2048, 1024, 2048, 1) == Extent{ 2048, 1024 });

    // Together: divide first, then halve. 2048/2 = 1024, still over 512, so once
    // more.
    CHECK(resolveExtent(2048, 1024, 512, 2) == Extent{ 512, 256 });

    // A degenerate downscale is read as 1 rather than dividing by zero.
    CHECK(resolveExtent(64, 64, 0, 0) == Extent{ 64, 64 });

    // Non-power-of-two, and the edge case that used to spin: a 1-pixel edge
    // cannot halve, so the loop has to stop on the other one.
    CHECK(resolveExtent(1000, 1, 256, 1) == Extent{ 250, 1 });
    CHECK(resolveExtent(1, 1, 1, 1) == Extent{ 1, 1 });
}

TEST_CASE("mip chains stop where the format stops being whole blocks")
{
    // Uncompressed goes all the way to 1x1.
    CHECK(fullMipLevelCount(256, 256) == 9);
    CHECK(mipLevelCount(Format::RGBA8, 256, 256) == 9);
    CHECK(mipLevelCount(Format::RGBA8, 1, 1) == 1);
    CHECK(mipLevelCount(Format::RGBA8, 640, 480) == 10); // driven by the longest edge

    // Compressed stops at 4x4: below that a level is mostly padding.
    CHECK(mipLevelCount(Format::BC1, 256, 256) == 7); // 256..4
    CHECK(mipLevelCount(Format::BC5, 256, 256) == 7);
    CHECK(mipLevelCount(Format::BC1, 4, 4) == 1);
    CHECK(mipLevelCount(Format::BC1, 2, 2) == 1);

    // A wide, short texture is limited by its short edge, not its long one.
    CHECK(mipLevelCount(Format::BC1, 1024, 8) == 2); // 1024x8, 512x4, then 256x2
}

TEST_CASE("levelBytes counts whole blocks, and rounds up")
{
    CHECK(levelBytes(Format::RGBA8, 16, 16) == 16 * 16 * 4);
    CHECK(levelBytes(Format::RGBA16, 16, 16) == 16 * 16 * 8);
    CHECK(levelBytes(Format::RGBA32F, 16, 16) == 16 * 16 * 16);

    // 16x16 is 4x4 blocks.
    CHECK(levelBytes(Format::BC1, 16, 16) == 16 * 8);
    CHECK(levelBytes(Format::BC3, 16, 16) == 16 * 16);
    CHECK(levelBytes(Format::BC5, 16, 16) == 16 * 16);

    // 17 texels is five blocks, not four and a quarter.
    CHECK(levelBytes(Format::BC1, 17, 16) == 5 * 4 * 8);

    // BC1 is an eighth of RGBA8 -- 8 bytes for a 4x4 block against 64 -- which
    // is the whole reason it is here. BC3 and BC5 are a quarter.
    CHECK(levelBytes(Format::BC1, 1024, 1024) * 8 == levelBytes(Format::RGBA8, 1024, 1024));
    CHECK(levelBytes(Format::BC3, 1024, 1024) * 4 == levelBytes(Format::RGBA8, 1024, 1024));
}

TEST_CASE("only a colour texture gets a transfer function")
{
    PlanInputs in;
    in.srcWidth = 256;
    in.srcHeight = 256;

    in.kind = Kind::Color;
    Plan colour = planTexture(in);
    CHECK(colour.format == Format::RGBA8);
    CHECK(colour.srgbTextureFlag);
    CHECK(colour.resampleInSrgb);
    CHECK_FALSE(colour.normalizeLevels);

    // A roughness or occlusion map read through sRGB is the classic silent
    // shading error; it must not happen.
    in.kind = Kind::NonColor;
    Plan linear = planTexture(in);
    CHECK_FALSE(linear.srgbTextureFlag);
    CHECK_FALSE(linear.srgbBlockFormat);
    CHECK_FALSE(linear.resampleInSrgb);

    // Nor a normal map, which is a direction rather than a colour.
    in.kind = Kind::Normal;
    Plan normal = planTexture(in);
    CHECK_FALSE(normal.srgbTextureFlag);
    CHECK_FALSE(normal.srgbBlockFormat);
    CHECK(normal.normalizeLevels);
}

TEST_CASE("format choice: what compresses, and to what")
{
    PlanInputs in;
    in.srcWidth = 256;
    in.srcHeight = 256;
    in.blockCompress = true;

    in.kind = Kind::Color;
    in.hasAlpha = false;
    CHECK(planTexture(in).format == Format::BC1);
    CHECK(planTexture(in).srgbBlockFormat);
    // The sRGB decode lives in the channel kind, but the texture object's flag
    // has to agree with it -- CUDA rejects the pair otherwise, which is why the
    // loader ORs the two.
    CHECK_FALSE(planTexture(in).srgbTextureFlag);

    in.hasAlpha = true;
    CHECK(planTexture(in).format == Format::BC3);

    // A normal map takes BC5 whether or not it has alpha: three channels do not
    // survive a shared 5:6:5 line, and Z is rebuilt in the shader.
    in.kind = Kind::Normal;
    CHECK(planTexture(in).format == Format::BC5);
    CHECK_FALSE(planTexture(in).srgbBlockFormat);

    // More than 8 bits per channel is the last thing to quantise: a float or
    // 16-bit source is never block compressed, whatever the setting says.
    in.kind = Kind::Color;
    in.hasAlpha = false;
    in.sourceIsFloat = true;
    CHECK(planTexture(in).format == Format::RGBA32F);
    CHECK_FALSE(planTexture(in).srgbTextureFlag); // float data is already linear

    in.sourceIsFloat = false;
    in.sourceIs16Bit = true;
    Plan sixteen = planTexture(in);
    CHECK(sixteen.format == Format::RGBA16);
    // CUDA's sRGB texture flag is an 8-bit-unorm feature, so a 16-bit colour map
    // must have been linearised on the host. The plan says so by leaving the
    // flag clear, and that is what the loader keys on.
    CHECK_FALSE(sixteen.srgbTextureFlag);
    CHECK_FALSE(sixteen.srgbBlockFormat);
}

TEST_CASE("levels are off until something asks for them")
{
    PlanInputs in;
    in.srcWidth = 512;
    in.srcHeight = 512;

    // tex2D from a ray tracing program has no derivatives, so it reads level 0;
    // a chain would be a third of the memory for nothing.
    CHECK(planTexture(in).levels == 1);

    in.wantMips = true;
    CHECK(planTexture(in).levels == 10);

    in.blockCompress = true;
    CHECK(planTexture(in).levels == 8); // 512..4
}

TEST_CASE("totalBytes: a mip chain is a third again, and BC1 is an eighth")
{
    PlanInputs in;
    in.srcWidth = 1024;
    in.srcHeight = 1024;
    in.kind = Kind::Color;

    const Plan flat = planTexture(in);
    CHECK(flat.totalBytes() == 1024ull * 1024 * 4);

    in.wantMips = true;
    const Plan chain = planTexture(in);
    // Sum of a geometric series in 1/4: strictly under 4/3 of the base.
    CHECK(chain.totalBytes() > flat.totalBytes());
    CHECK(chain.totalBytes() < flat.totalBytes() * 4 / 3 + 4);

    in.wantMips = false;
    in.blockCompress = true;
    CHECK(planTexture(in).totalBytes() == flat.totalBytes() / 8);
}

TEST_CASE("the cache file's format tags are fixed, because files outlive builds")
{
    // These values are written into .btex payload headers. Renumbering the enum
    // would make every cached texture decode as a different format, which reads
    // back as a valid file full of rubbish rather than as a miss.
    CHECK((int)Format::RGBA8 == 0);
    CHECK((int)Format::RGBA16 == 1);
    CHECK((int)Format::RGBA32F == 2);
    CHECK((int)Format::BC1 == 3);
    CHECK((int)Format::BC3 == 4);
    CHECK((int)Format::BC5 == 5);

    // Kind is cast straight to oka::metal::TextureKind to build the shared cache
    // key, so the two enumerations have to line up.
    CHECK((int)Kind::Color == 0);
    CHECK((int)Kind::NonColor == 1);
    CHECK((int)Kind::Normal == 2);
}
