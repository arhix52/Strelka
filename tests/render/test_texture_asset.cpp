#include <doctest/doctest.h>

#include <host/texture_asset.h>

using namespace oka::texture;

TEST_CASE("native texture policy chooses ASTC for Apple and BC for NVIDIA")
{
    Recipe recipe;
    recipe.compress = true;
    recipe.target = TargetProfile::AppleAstc;
    CHECK(chooseNativeFormat(recipe) == NativeFormat::ASTC6x6);

    recipe.semantic = Semantic::Normal;
    CHECK(chooseNativeFormat(recipe) == NativeFormat::ASTC4x4);

    recipe = Recipe{};
    recipe.target = TargetProfile::NvidiaBc;
    recipe.compress = true;
    CHECK(chooseNativeFormat(recipe) == NativeFormat::BC1);
    recipe.hasAlpha = true;
    CHECK(chooseNativeFormat(recipe) == NativeFormat::BC3);
    recipe.semantic = Semantic::Normal;
    CHECK(chooseNativeFormat(recipe) == NativeFormat::BC5);
}

TEST_CASE("native texture size accounts for each block footprint")
{
    CHECK(levelBytes(NativeFormat::RGBA8, 17, 17) == 17 * 17 * 4);
    CHECK(levelBytes(NativeFormat::BC1, 17, 17) == 5 * 5 * 8);
    CHECK(levelBytes(NativeFormat::ASTC4x4, 17, 17) == 5 * 5 * 16);
    CHECK(levelBytes(NativeFormat::ASTC6x6, 17, 17) == 3 * 3 * 16);
}
