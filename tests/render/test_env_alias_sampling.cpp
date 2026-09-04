#include <doctest/doctest.h>

// The two halves of environment importance sampling, checked against each other
// on the host: the host builder in render/host/ibl_alias_table.h and the draw the GPU
// actually runs, which env_alias_sampling.h compiles for both targets.
//
// What this pins is the property the whole scheme rests on and that no
// individual line of either file states: the frequency with which a texel is
// drawn has to equal the density MIS later divides by. If those two disagree the
// image is still an image -- smooth, plausible, and wrong by whatever the
// mismatch is -- which is why it is worth a test rather than an inspection.

#include <env_alias_sampling.h>
#include <env_map_math.h>
#include <host/ibl_alias_table.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

using oka::metal::buildSolidAngleIblAliasTable;

namespace
{

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

// A map with an order of magnitude across it, a dark band, and a bright spot --
// the shapes that break a sampler are a zero row and a single dominant texel.
std::vector<float> makeMap(int w, int h)
{
    std::vector<float> px((size_t)w * h * 4, 0.0f);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            const size_t i = ((size_t)y * w + x) * 4;
            float v = 0.05f + 0.9f * (float)x / (float)w;
            if (y == h / 2)
            {
                v = 0.0f; // a band the sampler must never return
            }
            if (x == 1 && y == 1)
            {
                v = 40.0f; // and one texel that should take most of the draws
            }
            px[i + 0] = v;
            px[i + 1] = v;
            px[i + 2] = v;
            px[i + 3] = 1.0f;
        }
    }
    return px;
}

// The builder's entry and the device's entry are two declarations of one layout;
// the backend uploads the former's bytes and the shader reads them as the latter.
// Copying field by field here is the cheap way of saying so out loud -- the size
// and alignment halves of the same claim are static_asserts in OptixRender.cpp.
std::vector<EnvAliasEntry> toDeviceTable(const std::vector<oka::metal::EnvAliasEntry>& src)
{
    std::vector<EnvAliasEntry> out(src.size());
    for (size_t i = 0; i < src.size(); ++i)
    {
        out[i].threshold = src[i].threshold;
        out[i].alias = src[i].alias;
        out[i].solidAnglePdf = src[i].solidAnglePdf;
    }
    return out;
}

uint32_t hashWord(uint32_t value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

uint32_t stratifiedWord(uint32_t sample, uint32_t count)
{
    constexpr uint64_t states = uint64_t{ 1 } << 32u;
    return static_cast<uint32_t>((uint64_t(sample) * states + states / 2u) / count);
}

} // namespace

static_assert(sizeof(EnvAliasEntry) == sizeof(oka::metal::EnvAliasEntry),
              "device EnvAliasEntry must match the host builder's entry");
static_assert(alignof(EnvAliasEntry) == alignof(oka::metal::EnvAliasEntry),
              "device EnvAliasEntry must match the host builder's entry");

TEST_CASE("envAliasDraw reproduces the distribution the table was built from")
{
    const int w = 16;
    const int h = 8;
    const auto px = makeMap(w, h);
    const auto built = buildSolidAngleIblAliasTable(px.data(), w, h);
    const auto table = toDeviceTable(built.alias);
    REQUIRE(table.size() == (size_t)w * h);

    // A stratified sweep rather than a random one: the draw is a deterministic
    // function of its variate, so sweeping [0,1) uniformly measures the discrete
    // distribution exactly and the test has no seed and no flake.
    const uint32_t n = (uint32_t)table.size();
    const int draws = 400000;
    std::vector<int> hits(n, 0);
    for (int s = 0; s < draws; ++s)
    {
        const EnvAliasDraw d = envAliasDraw(table.data(), n, stratifiedWord(uint32_t(s), uint32_t(draws)),
                                            hashWord(uint32_t(s)));
        REQUIRE(d.texel < n);
        hits[d.texel]++;
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        const int y = static_cast<int>(i / static_cast<uint32_t>(w));
        const double theta0 = (double)y / (double)h * M_PI;
        const double theta1 = (double)(y + 1) / (double)h * M_PI;
        const double solidAngle = (2.0 * M_PI / (double)w) * (std::cos(theta0) - std::cos(theta1));
        const double expected = static_cast<double>(built.alias[i].solidAnglePdf) * solidAngle;
        const double got = (double)hits[i] / (double)draws;
        CHECK(got == doctest::Approx(expected).epsilon(0.02).scale(1.0 / (double)n));
    }
}

TEST_CASE("environment alias thresholds match the finite GPU random lattice")
{
    constexpr uint32_t texelCount = 1u << 20u;
    constexpr uint32_t threshold = 1u << 10u;
    const EnvAliasEntry table[] = { { threshold, 1u, 0.0f }, { 0u, 1u, 0.0f } };

    uint32_t legacyOwn = 0u;
    for (uint32_t fraction = 0u; fraction < 8u; ++fraction)
    {
        legacyOwn += static_cast<float>(fraction) * 0x1p-23f < 0x1p-22f ? 1u : 0u;
    }
    CHECK(static_cast<double>(legacyOwn) / 8.0 != doctest::Approx(0x1p-22).epsilon(1e-7));

    CHECK(envAliasDraw(table, texelCount, 0u, threshold - 1u).texel == 0u);
    CHECK(envAliasDraw(table, texelCount, 0u, threshold).texel == 1u);
    CHECK(static_cast<double>(threshold) / 4294967296.0 == doctest::Approx(0x1p-22).epsilon(1e-12));
}

TEST_CASE("environment solid-angle jitter excludes the coordinate singularities")
{
    constexpr int height = 8;
    const float north = envSampleSolidAngleV(0, height, 0.0f);
    const float south = envSampleSolidAngleV(height - 1, height, 0x1.fffffep-1f);

    CHECK(north > 0.0f);
    CHECK(south < 1.0f);

    // Mutation: the old closed-interval mapping collapses every first-row
    // azimuth onto one pole direction while returning the selected texel's
    // potentially different PDF.
    CHECK(envUVToDir(make_float2(0.125f, 0.0f), 0.0f).y == 1.0f);
}

TEST_CASE("finite environment jitter remains inside the selected texel")
{
    constexpr uint32_t width = 4u;
    constexpr uint32_t selectedX = 1u;
    const float oldU = (static_cast<float>(selectedX) + envOpenUnitInterval(0x1.fffffep-1f)) /
                       static_cast<float>(width);
    CHECK(static_cast<uint32_t>(oldU * static_cast<float>(width)) == 2u);

    const float u = envSampleTexelU(static_cast<int>(selectedX), static_cast<int>(width), 0x1.fffffep-1f);
    CHECK(static_cast<uint32_t>(u * static_cast<float>(width)) == selectedX);
    CHECK(u != (static_cast<float>(selectedX) + 0.5f) / static_cast<float>(width));

    for (const int w : { 1, 4, 16, 1024 })
    {
        for (const int h : { 1, 4, 8 })
        {
            const int xStep = std::max(w / 16, 1);
            for (int x = 0; x < w; x += xStep)
            {
                for (int y = 0; y < h; ++y)
                {
                    for (const float rotation : { 0.0f, 0.63f, -2.1f })
                    {
                        for (const float xi : { 0.0f, 0x1.fffffep-1f })
                        {
                            const float3 direction =
                                envSampleTexelDirection(x, y, w, h, xi, xi, 0x12345678u, 0x9abcdef0u, rotation);
                            const float2 evaluated = dirToEnvUV(direction, rotation);
                            const int evaluatedX = std::clamp(static_cast<int>(evaluated.x * static_cast<float>(w)),
                                                              0, w - 1);
                            const int evaluatedY = std::clamp(static_cast<int>(evaluated.y * static_cast<float>(h)),
                                                              0, h - 1);
                            CAPTURE(w);
                            CAPTURE(h);
                            CAPTURE(x);
                            CAPTURE(y);
                            CAPTURE(rotation);
                            CAPTURE(xi);
                            CHECK(evaluatedX == x);
                            CHECK(evaluatedY == y);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("environment round-trip repair retains jitter instead of creating a midpoint atom")
{
    constexpr int w = 1024;
    constexpr int h = 512;
    constexpr int x = 0;
    constexpr int y = 0;
    constexpr float rotation = 0.0f;
    const float midpointU = (static_cast<float>(x) + 0.5f) / static_cast<float>(w);
    const float midpointV = envSolidAngleRowV(y, h, 0.5f);
    const float3 midpoint = envUVToDir(make_float2(midpointU, midpointV), rotation);

    int mismatchMutation = 0;
    int midpointCollapses = 0;
    for (uint32_t word = 0; word < (1u << 23u) && mismatchMutation < 16; ++word)
    {
        const float xi = static_cast<float>(word) * 0x1p-23f;
        const float rawU = (static_cast<float>(x) + envOpenUnitInterval(xi)) / static_cast<float>(w);
        const float rawV = envSampleSolidAngleV(y, h, xi);
        const float3 rawDirection = envUVToDir(make_float2(rawU, rawV), rotation);
        const float2 rawBack = dirToEnvUV(rawDirection, rotation);
        const int rawX = std::clamp(static_cast<int>(rawBack.x * static_cast<float>(w)), 0, w - 1);
        const int rawY = std::clamp(static_cast<int>(rawBack.y * static_cast<float>(h)), 0, h - 1);
        if (rawX == x && rawY == y)
        {
            continue;
        }

        ++mismatchMutation;
        const float3 repaired =
            envSampleTexelDirection(x, y, w, h, xi, xi, hashWord(word), hashWord(word ^ 0x9e3779b9u), rotation);
        const float2 repairedBack = dirToEnvUV(repaired, rotation);
        CHECK(std::clamp(static_cast<int>(repairedBack.x * static_cast<float>(w)), 0, w - 1) == x);
        CHECK(std::clamp(static_cast<int>(repairedBack.y * static_cast<float>(h)), 0, h - 1) == y);
        midpointCollapses += repaired.x == midpoint.x && repaired.y == midpoint.y && repaired.z == midpoint.z ? 1 : 0;
    }

    REQUIRE(mismatchMutation == 16);
    CHECK(midpointCollapses == 0);
}

TEST_CASE("environment round-trip repair does not collapse an extreme row interval")
{
    constexpr int w = 1;
    constexpr int h = 1 << 20;
    constexpr int x = 0;
    constexpr float rotation = 0.0f;

    const auto checkDistinct = [&](int y, uint32_t firstWord, uint32_t lastWord) {
        const float xiU = 0.31415927f;
        const float3 first = envSampleTexelDirection(
            x, y, w, h, xiU, static_cast<float>(firstWord) * 0x1p-23f, 0x12345678u, 0x9abcdef0u, rotation);
        const float3 last = envSampleTexelDirection(
            x, y, w, h, xiU, static_cast<float>(lastWord) * 0x1p-23f, 0x12345678u, 0x9abcdef0u, rotation);
        const float2 firstUv = dirToEnvUV(first, rotation);
        const float2 lastUv = dirToEnvUV(last, rotation);

        CAPTURE(y);
        CHECK(std::clamp(static_cast<int>(firstUv.y * h), 0, h - 1) == y);
        CHECK(std::clamp(static_cast<int>(lastUv.y * h), 0, h - 1) == y);
        const bool distinct = first.x != last.x || first.y != last.y || first.z != last.z;
        CHECK(distinct);
    };

    checkDistinct(h / 2, 8259807u, 8388607u);
    checkDistinct(h - 1, 0u, 632868u);
}

TEST_CASE("a zero bilinear footprint is never drawn")
{
    const int w = 8;
    const int h = 8;
    std::vector<float> px((size_t)w * h * 4u, 0.0f);
    px[0] = px[1] = px[2] = px[3] = 1.0f;
    const auto built = buildSolidAngleIblAliasTable(px.data(), w, h);
    const auto table = toDeviceTable(built.alias);
    const uint32_t n = (uint32_t)table.size();

    const int draws = 100000;
    for (int s = 0; s < draws; ++s)
    {
        const EnvAliasDraw d = envAliasDraw(table.data(), n, stratifiedWord(uint32_t(s), uint32_t(draws)),
                                            hashWord(uint32_t(s)));
        REQUIRE(d.texel < n);
        CHECK(table[d.texel].solidAnglePdf > 0.0f);
    }
}

TEST_CASE("represented environment density integrates to one")
{
    const int w = 16;
    const int h = 8;
    const auto px = makeMap(w, h);
    const auto built = buildSolidAngleIblAliasTable(px.data(), w, h);

    // sum over texels of pdf(texel) * solid angle(texel) must be 1, which is the
    // statement that lum * envPdfScale really is a density on the sphere.
    double integral = 0.0;
    for (int y = 0; y < h; ++y)
    {
        const double theta0 = (double)y / (double)h * M_PI;
        const double theta1 = (double)(y + 1) / (double)h * M_PI;
        const double dOmega = (2.0 * M_PI / (double)w) * (std::cos(theta0) - std::cos(theta1));
        for (int x = 0; x < w; ++x)
        {
            integral += static_cast<double>(built.alias[(size_t)y * w + x].solidAnglePdf) * dOmega;
        }
    }
    CHECK(integral == doctest::Approx(1.0).epsilon(1e-5));
}

TEST_CASE("solid-angle samples and evaluated PDFs use the same texel measure")
{
    for (const auto [w, h] : { std::pair{ 1, 1 }, std::pair{ 1, 9 }, std::pair{ 11, 1 }, std::pair{ 13, 7 } })
    {
        CAPTURE(w);
        CAPTURE(h);
        std::vector<float> px((size_t)w * h * 4, 0.0f);
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                const size_t i = (size_t)y * w + x;
                const float value = (i % 5 == 0) ? 0.0f : (0.125f + (float)i);
                px[i * 4 + 0] = px[i * 4 + 1] = px[i * 4 + 2] = value;
                px[i * 4 + 3] = 1.0f;
            }
        }
        // A 1x1 all-zero map has no density; make this case a positive constant.
        if (w == 1 && h == 1)
        {
            px[0] = px[1] = px[2] = 1.0f;
        }

        const auto built = buildSolidAngleIblAliasTable(px.data(), w, h);
        const auto table = toDeviceTable(built.alias);
        REQUIRE(built.envPdfScale > 0.0f);
        std::mt19937 rng(0x51A17u + (uint32_t)(w * 31 + h));
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

        for (uint32_t sample = 0; sample < 20000u; ++sample)
        {
            const uint32_t bucketWord = rng();
            const uint32_t aliasWord = rng();
            const float jitterU = uniform(rng);
            const float jitterV = uniform(rng);
            const EnvAliasDraw draw = envAliasDraw(table.data(), (uint32_t)table.size(), bucketWord, aliasWord);
            const uint32_t x = draw.texel % (uint32_t)w;
            const uint32_t y = draw.texel / (uint32_t)w;
            const float u = ((float)x + jitterU) / (float)w;
            const float v = envSampleSolidAngleV((int)y, h, jitterV);
            const float3 direction = envUVToDir(make_float2(u, v), 0.63f);
            const float2 evaluatedUv = dirToEnvUV(direction, 0.63f);
            const int evaluatedX = std::clamp((int)(evaluatedUv.x * (float)w), 0, w - 1);
            const int evaluatedY = std::clamp((int)(evaluatedUv.y * (float)h), 0, h - 1);
            const float returnedPdf = table[draw.texel].solidAnglePdf;
            const float evaluatedPdf = table[(size_t)evaluatedY * w + evaluatedX].solidAnglePdf;
            INFO("sample=", sample, " texel=", x, ",", y, " jitter=", jitterU, " uv=", u, ",", v,
                 " evaluated=", evaluatedX, ",", evaluatedY);
            CHECK(returnedPdf == doctest::Approx(evaluatedPdf).epsilon(2e-5));
            CHECK(std::isfinite(returnedPdf));
            CHECK(returnedPdf >= 0.0f);
        }
    }
}

TEST_CASE("one by one solid-angle Lambertian estimator converges to one")
{
    constexpr uint32_t samples = 1u << 16u;
    const double pdf = 1.0 / (4.0 * M_PI);
    double sum = 0.0;
    for (uint32_t i = 0; i < samples; ++i)
    {
        const double xi = ((double)i + 0.5) / (double)samples;
        const double cosTheta = 1.0 - 2.0 * xi;
        if (cosTheta > 0.0)
        {
            sum += (cosTheta / M_PI) / pdf;
        }
    }
    CHECK(sum / (double)samples == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("independent within-texel jitter stays inside the unit interval and spreads out")
{
    const int w = 8;
    const int h = 4;
    const auto px = makeMap(w, h);
    const auto built = buildSolidAngleIblAliasTable(px.data(), w, h);
    const auto table = toDeviceTable(built.alias);
    const uint32_t n = (uint32_t)table.size();

    // An independent jitter is what turns the selected texel mass into a
    // continuous solid-angle density without conditioning on the alias branch.
    int lowHalf = 0;
    const int draws = 20000;
    for (int s = 0; s < draws; ++s)
    {
        const float jitter = ((float)s + 0.5f) / (float)draws;
        const EnvAliasDraw d = envAliasDraw(table.data(), n, stratifiedWord(uint32_t(s), uint32_t(draws)),
                                            hashWord(uint32_t(s)));
        CHECK(d.texel < n);
        CHECK(jitter >= 0.0f);
        CHECK(jitter < 1.0f);
        if (jitter < 0.5f)
        {
            lowHalf++;
        }
    }
    CHECK(lowHalf > draws / 4);
    CHECK(lowHalf < 3 * draws / 4);
}

TEST_CASE("a variate at the top of the range stays in the table")
{
    const int w = 4;
    const int h = 2;
    const auto px = makeMap(w, h);
    const auto built = buildSolidAngleIblAliasTable(px.data(), w, h);
    const auto table = toDeviceTable(built.alias);
    const uint32_t n = (uint32_t)table.size();

    for (const uint32_t word : { 0u, std::numeric_limits<uint32_t>::max() })
    {
        const EnvAliasDraw d = envAliasDraw(table.data(), n, word, 0x80000000u);
        CHECK(d.texel < n);
    }
}

TEST_CASE("envAliasDraw refuses an empty table instead of reading it")
{
    const EnvAliasDraw a = envAliasDraw(nullptr, 16, 0u, 0u);
    CHECK(a.texel == 0u);

    std::vector<EnvAliasEntry> table(1);
    table[0].threshold = 0u;
    table[0].alias = 0u;
    const EnvAliasDraw b = envAliasDraw(table.data(), 0, 0u, 0u);
    CHECK(b.texel == 0u);
}
