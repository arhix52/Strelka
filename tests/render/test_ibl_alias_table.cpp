#include <doctest/doctest.h>

#include "ibl_alias_table.h"

#include <cmath>
#include <vector>

using oka::metal::buildIblAliasTable;

TEST_CASE("buildIblAliasTable black map has unit probs and zero pdfScale")
{
    const int w = 2;
    const int h = 2;
    std::vector<float> pixels((size_t)w * h * 4, 0.0f);
    const auto result = buildIblAliasTable(pixels.data(), w, h);
    CHECK(result.alias.size() == 4);
    CHECK(result.totalPower == doctest::Approx(0.0));
    CHECK(result.envPdfScale == doctest::Approx(0.0f));
    for (const auto& e : result.alias)
    {
        CHECK(e.prob == doctest::Approx(1.0f));
    }
}

TEST_CASE("buildIblAliasTable bright texel gets mass and positive pdfScale")
{
    const int w = 2;
    const int h = 2;
    std::vector<float> pixels((size_t)w * h * 4, 0.0f);
    // Bright texel at (0,0)
    pixels[0] = 1.0f;
    pixels[1] = 1.0f;
    pixels[2] = 1.0f;
    pixels[3] = 1.0f;

    const auto result = buildIblAliasTable(pixels.data(), w, h);
    CHECK(result.totalPower > 0.0);
    CHECK(result.envPdfScale > 0.0f);
    CHECK(result.alias.size() == 4);

    // Every entry stays in valid Walker/Vose ranges.
    for (size_t i = 0; i < result.alias.size(); ++i)
    {
        CHECK(result.alias[i].prob >= 0.0f);
        CHECK(result.alias[i].prob <= 1.0f + 1e-5f);
        CHECK(result.alias[i].alias < result.alias.size());
    }
}

TEST_CASE("buildIblAliasTable null or empty returns empty")
{
    CHECK(buildIblAliasTable(nullptr, 4, 4).alias.empty());
    std::vector<float> pixels(4, 1.0f);
    CHECK(buildIblAliasTable(pixels.data(), 0, 1).alias.empty());
}
