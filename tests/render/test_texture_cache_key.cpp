#include <doctest/doctest.h>

#include <host/texture_cache_key.h>

using oka::texture::Semantic;
using oka::texture::TargetProfile;
using oka::texture::textureCacheKey;
using oka::texture::TextureCacheKeyInputs;
using oka::texture::textureIdentityPath;

TEST_CASE("textureCacheKey is stable for identical inputs")
{
    TextureCacheKeyInputs in;
    in.fileName = "/tmp/albedo.png";
    in.fileSize = 12345;
    in.writeTimeCount = 999;
    in.maxDimension = 2048;
    in.downscale = 1;
    in.srgb = true;
    in.semantic = Semantic::Color;
    in.target = TargetProfile::AppleAstc;
    in.compressed = true;
    in.encoderVersion = 3;

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
    base.semantic = Semantic::NonColor;
    base.target = TargetProfile::NvidiaBc;
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
        in.semantic = Semantic::Normal;
        CHECK(textureCacheKey(in) != k0);
    }
    {
        auto in = base;
        in.target = TargetProfile::AppleAstc;
        CHECK(textureCacheKey(in) != k0);
    }
    {
        auto in = base;
        in.compressed = true;
        CHECK(textureCacheKey(in) != k0);
    }
    {
        auto in = base;
        in.encoderVersion = 2;
        CHECK(textureCacheKey(in) != k0);
    }
}

TEST_CASE("textureCacheKey gives relative and absolute spellings one identity")
{
    TextureCacheKeyInputs relative;
    relative.fileName = "textures/../textures/albedo.png";
    TextureCacheKeyInputs absolute = relative;
    absolute.fileName = (std::filesystem::current_path() / "textures/albedo.png").string();
    CHECK(textureCacheKey(relative) == textureCacheKey(absolute));
    CHECK(textureIdentityPath(relative.fileName) == textureIdentityPath(absolute.fileName));
}
