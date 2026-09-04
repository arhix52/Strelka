#include <doctest/doctest.h>

#include <host/ibl_alias_table.h>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

using oka::metal::buildSolidAngleIblAliasTable;
using oka::metal::EnvAliasEntry;
using oka::metal::IblAliasTableResult;

TEST_CASE("buildSolidAngleIblAliasTable black map has unit aliases and zero power")
{
    const int w = 2;
    const int h = 2;
    std::vector<float> pixels((size_t)w * h * 4, 0.0f);
    const auto result = buildSolidAngleIblAliasTable(pixels.data(), w, h);
    CHECK(result.alias.size() == 4);
    CHECK(result.totalPower == doctest::Approx(0.0));
    for (const auto& e : result.alias)
    {
        CHECK(e.threshold == 0u);
        CHECK(e.alias < result.alias.size());
    }
}

TEST_CASE("buildSolidAngleIblAliasTable bright texel gets positive density")
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
    // Without a finiteness test NaN reaches both the radiance integral and the
    // alias weights. std::max does not clamp it, so the guard is explicit.
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
        for (const EnvAliasEntry& entry : result.alias)
        {
            CHECK(entry.solidAnglePdf == doctest::Approx(1.0 / (4.0 * M_PI)).epsilon(2e-7));
        }
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
        const int x = static_cast<int>(i % w);
        double footprintMax = 0.0;
        for (int dy = -1; dy <= 1; ++dy)
        {
            const int sy = std::clamp(y + dy, 0, h - 1);
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int sx = (x + dx + w) % w;
                footprintMax = std::max(footprintMax, static_cast<double>(values[sy * w + sx]));
            }
        }
        if (footprintMax > 0.0)
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
    CHECK(result.alias[0].solidAnglePdf > 0.0f);
}

TEST_CASE("finite radiance channels survive invalid neighbours in the same texel")
{
    const float pixels[] = { std::numeric_limits<float>::quiet_NaN(), 1.0f,
                             std::numeric_limits<float>::infinity(), 1.0f };
    const IblAliasTableResult result = buildSolidAngleIblAliasTable(pixels, 1, 1);
    CHECK(result.totalPower > 0.0);
    CHECK(result.alias[0].solidAnglePdf == doctest::Approx(1.0 / (4.0 * M_PI)).epsilon(2e-7));
}

TEST_CASE("bilinear environment footprint gives every positive reconstruction support")
{
    constexpr int w = 3;
    constexpr int h = 3;
    std::vector<float> pixels((size_t)w * h * 4, 0.0f);
    const size_t centre = (size_t(1) * w + 1u) * 4u;
    pixels[centre + 0u] = pixels[centre + 1u] = pixels[centre + 2u] = 1.0f;
    pixels[centre + 3u] = 1.0f;

    const IblAliasTableResult result = buildSolidAngleIblAliasTable(pixels.data(), w, h);
    REQUIRE(result.alias.size() == size_t(w * h));
    for (const EnvAliasEntry& entry : result.alias)
    {
        // Every one of these bins contains UVs whose bilinear footprint reaches
        // the bright centre texel. The old centre-only weights leave eight of
        // those positive-radiance directions at PDF zero.
        CHECK(entry.solidAnglePdf > 0.0f);
    }
}

TEST_CASE("environment footprint wraps the seam but not the poles")
{
    constexpr int w = 5;
    constexpr int h = 5;
    std::vector<float> pixels((size_t)w * h * 4u, 0.0f);
    pixels[0] = pixels[1] = pixels[2] = pixels[3] = 1.0f;
    const IblAliasTableResult result = buildSolidAngleIblAliasTable(pixels.data(), w, h);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            CAPTURE(x);
            CAPTURE(y);
            const bool reconstructionCanReachBrightTexel = (x == 0 || x == 1 || x == w - 1) && y <= 1;
            const float pdf = result.alias[(size_t)y * w + x].solidAnglePdf;
            CHECK((pdf > 0.0f) == reconstructionCanReachBrightTexel);
        }
    }
}

TEST_CASE("subnormal environment power keeps a finite sampling representation")
{
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float pixels[] = { tiny, tiny, tiny, 1.0f };
    const IblAliasTableResult result = buildSolidAngleIblAliasTable(pixels, 1, 1);
    CHECK(result.totalPower > 0.0);
    CHECK(result.alias[0].solidAnglePdf == doctest::Approx(1.0 / (4.0 * M_PI)).epsilon(2e-7));

    const float oldReciprocal = static_cast<float>(1.0 / result.totalPower);
    CHECK(std::isinf(oldReciprocal));
    CHECK(1.0 / static_cast<double>(oldReciprocal) == 0.0);
    CHECK(oka::metal::environmentLightPower(result.totalPower, 1.0, 1.0, 1.0) > 0.0);
}

TEST_CASE("environment upload sanitization leaves no invalid radiance")
{
    float pixels[] = { std::numeric_limits<float>::quiet_NaN(), -1.0f,
                       std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
                       0.25f, 0.5f, 1.0f, 1.0f };
    oka::metal::sanitizeEnvironmentPixels(pixels, 2, 1);
    for (const float value : pixels)
    {
        CHECK(std::isfinite(value));
    }
    CHECK(pixels[0] == 0.0f);
    CHECK(pixels[1] == 0.0f);
    CHECK(pixels[2] == 0.0f);
    CHECK(pixels[3] == 1.0f);
    CHECK(pixels[4] == 0.25f);
}
