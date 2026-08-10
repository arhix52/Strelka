#include <doctest/doctest.h>

#include "texture_cache_key.h"

using oka::metal::textureCacheKey;
using oka::metal::TextureCacheKeyInputs;
using oka::metal::TextureKind;

TEST_CASE("textureCacheKey is stable for identical inputs")
{
    TextureCacheKeyInputs in;
    in.fileName = "/tmp/albedo.png";
    in.fileSize = 12345;
    in.writeTimeCount = 999;
    in.maxDimension = 2048;
    in.downscale = 1;
    in.srgb = true;
    in.kind = TextureKind::Color;

    const std::string a = textureCacheKey(in);
    const std::string b = textureCacheKey(in);
    CHECK(a == b);
    CHECK(a.size() == 21); // 16 hex + ".btex"
    CHECK(a.substr(a.size() - 5) == ".btex");
}

TEST_CASE("textureCacheKey changes when any input changes")
{
    TextureCacheKeyInputs base;
    base.fileName = "tex.png";
    base.fileSize = 100;
    base.writeTimeCount = 1;
    base.maxDimension = 1024;
    base.downscale = 1;
    base.srgb = false;
    base.kind = TextureKind::NonColor;
    const std::string k0 = textureCacheKey(base);

    {
        auto in = base;
        in.fileName = "other.png";
        CHECK(textureCacheKey(in) != k0);
    }
    {
        auto in = base;
        in.fileSize = 101;
        CHECK(textureCacheKey(in) != k0);
    }
    {
        auto in = base;
        in.writeTimeCount = 2;
        CHECK(textureCacheKey(in) != k0);
    }
    {
        auto in = base;
        in.maxDimension = 512;
        CHECK(textureCacheKey(in) != k0);
    }
    {
        auto in = base;
        in.downscale = 2;
        CHECK(textureCacheKey(in) != k0);
    }
    {
        auto in = base;
        in.srgb = true;
        CHECK(textureCacheKey(in) != k0);
    }
    {
        auto in = base;
        in.kind = TextureKind::Normal;
        CHECK(textureCacheKey(in) != k0);
    }
}
