#include <doctest/doctest.h>

#include <host/ibl_alias_table.h>

#include <cmath>
#include <limits>
#include <vector>

using oka::metal::buildIblAliasTable;
using oka::metal::EnvAliasEntry;
using oka::metal::IblAliasTableResult;

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

TEST_CASE("a NaN texel does not take the whole map with it")
{
    // A downloaded HDRI with one bad pixel is not exotic. Without a finiteness
    // test the NaN propagates through totalPower into envPdfScale, and every
    // density the shaders compute -- for sampling and for the MIS weight alike
    // -- comes back NaN. std::max does not clamp it, so the guard has to be
    // explicit.
    const int w = 8;
    const int h = 4;
    std::vector<float> px((size_t)w * h * 4, 0.0f);
    for (size_t i = 0; i < (size_t)w * h; ++i)
    {
        px[i * 4 + 0] = px[i * 4 + 1] = px[i * 4 + 2] = 1.0f;
        px[i * 4 + 3] = 1.0f;
    }
    px[5 * 4 + 1] = std::numeric_limits<float>::quiet_NaN();
    px[9 * 4 + 0] = std::numeric_limits<float>::infinity();

    const IblAliasTableResult r = buildIblAliasTable(px.data(), w, h);

    CHECK(std::isfinite(r.envPdfScale));
    CHECK(r.envPdfScale > 0.0f);
    CHECK(std::isfinite(r.totalPower));
    CHECK(r.totalPower > 0.0);
    for (const EnvAliasEntry& e : r.alias)
    {
        CHECK(std::isfinite(e.prob));
        CHECK(e.alias < (uint32_t)(w * h));
    }
}
