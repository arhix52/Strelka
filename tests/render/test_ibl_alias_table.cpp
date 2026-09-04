#include <doctest/doctest.h>

#include <host/ibl_alias_table.h>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

using oka::metal::buildSolidAngleIblAliasTable;
using oka::metal::EnvAliasEntry;
using oka::metal::IblAliasTableResult;

TEST_CASE("buildSolidAngleIblAliasTable black map has unit probs and zero pdfScale")
{
    const int w = 2;
    const int h = 2;
    std::vector<float> pixels((size_t)w * h * 4, 0.0f);
    const auto result = buildSolidAngleIblAliasTable(pixels.data(), w, h);
    CHECK(result.alias.size() == 4);
    CHECK(result.totalPower == doctest::Approx(0.0));
    CHECK(result.envPdfScale == doctest::Approx(0.0f));
    for (const auto& e : result.alias)
    {
        CHECK(e.threshold == 0u);
        CHECK(e.alias < result.alias.size());
    }
}

TEST_CASE("buildSolidAngleIblAliasTable bright texel gets mass and positive pdfScale")
{
    const int w = 2;
    const int h = 2;
    std::vector<float> pixels((size_t)w * h * 4, 0.0f);
    // Bright texel at (0,0)
    pixels[0] = 1.0f;
    pixels[1] = 1.0f;
    pixels[2] = 1.0f;
    pixels[3] = 1.0f;

    const auto result = buildSolidAngleIblAliasTable(pixels.data(), w, h);
    CHECK(result.totalPower > 0.0);
    CHECK(result.envPdfScale > 0.0f);
    CHECK(result.alias.size() == 4);

    // Every entry stays in valid Walker/Vose ranges.
    for (size_t i = 0; i < result.alias.size(); ++i)
    {
        CHECK(result.alias[i].alias < result.alias.size());
    }
}

TEST_CASE("buildSolidAngleIblAliasTable null or empty returns empty")
{
    CHECK(buildSolidAngleIblAliasTable(nullptr, 4, 4).alias.empty());
    std::vector<float> pixels(4, 1.0f);
    CHECK(buildSolidAngleIblAliasTable(pixels.data(), 0, 1).alias.empty());
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

    const IblAliasTableResult r = buildSolidAngleIblAliasTable(px.data(), w, h);

    CHECK(std::isfinite(r.envPdfScale));
    CHECK(r.envPdfScale > 0.0f);
    CHECK(std::isfinite(r.totalPower));
    CHECK(r.totalPower > 0.0);
    for (const EnvAliasEntry& e : r.alias)
    {
        CHECK(e.alias < (uint32_t)(w * h));
    }
}

TEST_CASE("constant maps have exact sphere normalization at degenerate resolutions")
{
    for (const auto [w, h] : { std::pair{ 1, 1 }, std::pair{ 1, 7 }, std::pair{ 9, 1 }, std::pair{ 7, 5 } })
    {
        CAPTURE(w);
        CAPTURE(h);
        std::vector<float> pixels((size_t)w * (size_t)h * 4, 1.0f);
        const IblAliasTableResult result = buildSolidAngleIblAliasTable(pixels.data(), w, h);
        CHECK(result.totalPower == doctest::Approx(4.0 * M_PI).epsilon(1e-13));
        CHECK(result.envPdfScale == doctest::Approx(1.0 / (4.0 * M_PI)).epsilon(2e-7));
    }
}

TEST_CASE("positive extreme-dynamic-range texels retain discrete support")
{
    constexpr int w = 4;
    constexpr int h = 3;
    std::vector<float> pixels((size_t)w * h * 4, 0.0f);
    const float values[] = { 0.0f, 1e-6f, 1.0f, 1e6f, 3.0f, 0.0f, 7e3f, 2e-4f, 9.0f, 2.0f, 0.0f, 5e5f };
    for (size_t i = 0; i < (size_t)w * h; ++i)
    {
        pixels[i * 4 + 0] = pixels[i * 4 + 1] = pixels[i * 4 + 2] = values[i];
        pixels[i * 4 + 3] = 1.0f;
    }

    const IblAliasTableResult result = buildSolidAngleIblAliasTable(pixels.data(), w, h);
    REQUIRE(result.alias.size() == (size_t)w * h);

    // Reconstruct the PMF represented by the integer alias table. This checks
    // support directly without hoping a finite random run visits a 1e-12 bin.
    std::vector<double> represented(result.alias.size(), 0.0);
    constexpr double integerStateCount = 4294967296.0;
    for (size_t bucket = 0; bucket < result.alias.size(); ++bucket)
    {
        const EnvAliasEntry& entry = result.alias[bucket];
        const double bucketMass = double(discreteBucketStateCount(uint32_t(result.alias.size()), uint32_t(bucket))) /
                                  integerStateCount;
        const double own = entry.alias == bucket ? 1.0 : double(entry.threshold) / integerStateCount;
        represented[bucket] += bucketMass * own;
        represented[entry.alias] += bucketMass * (1.0 - own);
    }
    for (size_t i = 0; i < represented.size(); ++i)
    {
        CAPTURE(i);
        const int y = static_cast<int>(i / w);
        const double theta0 = M_PI * static_cast<double>(y) / h;
        const double theta1 = M_PI * static_cast<double>(y + 1) / h;
        const double solidAngle = (2.0 * M_PI / w) * (std::cos(theta0) - std::cos(theta1));
        CHECK(result.alias[i].solidAnglePdf * solidAngle == doctest::Approx(represented[i]).epsilon(2e-7));
        if (values[i] > 0.0f)
        {
            CHECK(represented[i] > 0.0);
        }
        else
        {
            CHECK(represented[i] == doctest::Approx(0.0).epsilon(1e-12));
        }
    }
}

TEST_CASE("a finite positive channel retains support beside negative channels")
{
    const float pixels[] = { 1.0f, -1000.0f, 0.0f, 1.0f };
    const IblAliasTableResult result = buildSolidAngleIblAliasTable(pixels, 1, 1);
    CHECK(result.totalPower > 0.0);
    CHECK(result.envPdfScale > 0.0f);
}
