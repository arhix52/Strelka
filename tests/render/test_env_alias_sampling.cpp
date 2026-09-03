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

double texelLuminance(const std::vector<float>& px, int w, int x, int y)
{
    const size_t i = ((size_t)y * w + x) * 4;
    return 0.2126 * px[i + 0] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
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
        out[i].prob = src[i].prob;
        out[i].alias = src[i].alias;
    }
    return out;
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
        const float xi = ((float)s + 0.5f) / (float)draws;
        const EnvAliasDraw d = envAliasDraw(table.data(), n, xi);
        REQUIRE(d.texel < n);
        hits[d.texel]++;
    }

    double totalWeight = 0.0;
    std::vector<double> weight(n, 0.0);
    for (int y = 0; y < h; ++y)
    {
        const double theta0 = (double)y / (double)h * M_PI;
        const double theta1 = (double)(y + 1) / (double)h * M_PI;
        const double solidAngle = (2.0 * M_PI / (double)w) * (std::cos(theta0) - std::cos(theta1));
        for (int x = 0; x < w; ++x)
        {
            const size_t i = (size_t)y * w + x;
            weight[i] = texelLuminance(px, w, x, y) * solidAngle;
            totalWeight += weight[i];
        }
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        const double expected = weight[i] / totalWeight;
        const double got = (double)hits[i] / (double)draws;
        CHECK(got == doctest::Approx(expected).epsilon(0.02).scale(1.0 / (double)n));
    }
}

TEST_CASE("a zero-luminance texel is never drawn")
{
    const int w = 16;
    const int h = 8;
    const auto px = makeMap(w, h);
    const auto built = buildSolidAngleIblAliasTable(px.data(), w, h);
    const auto table = toDeviceTable(built.alias);
    const uint32_t n = (uint32_t)table.size();

    const int draws = 100000;
    for (int s = 0; s < draws; ++s)
    {
        const float xi = ((float)s + 0.5f) / (float)draws;
        const EnvAliasDraw d = envAliasDraw(table.data(), n, xi);
        const int y = (int)(d.texel / (uint32_t)w);
        CHECK(y != h / 2);
    }
}

TEST_CASE("envPdfScale turns texel luminance into a density that integrates to one")
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
            integral += texelLuminance(px, w, x, y) * (double)built.envPdfScale * dOmega;
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
            const float xi0 = uniform(rng);
            const float xi1 = uniform(rng);
            const EnvAliasDraw draw = envAliasDraw(table.data(), (uint32_t)table.size(), xi0);
            const uint32_t x = draw.texel % (uint32_t)w;
            const uint32_t y = draw.texel / (uint32_t)w;
            const float u = ((float)x + draw.frac) / (float)w;
            const float v = envSampleSolidAngleV((int)y, h, xi1);
            const float3 direction = envUVToDir(make_float2(u, v), 0.63f);
            const float2 evaluatedUv = dirToEnvUV(direction, 0.63f);
            const int evaluatedX = std::clamp((int)(evaluatedUv.x * (float)w), 0, w - 1);
            const int evaluatedY = std::clamp((int)(evaluatedUv.y * (float)h), 0, h - 1);
            const float returnedPdf = (float)(texelLuminance(px, w, (int)x, (int)y) * (double)built.envPdfScale);
            const float evaluatedPdf = (float)(texelLuminance(px, w, evaluatedX, evaluatedY) * (double)built.envPdfScale);
            INFO("sample=", sample, " texel=", x, ",", y, " frac=", draw.frac, " uv=", u, ",", v,
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

TEST_CASE("the leftover variate stays inside the unit interval and spreads out")
{
    const int w = 8;
    const int h = 4;
    const auto px = makeMap(w, h);
    const auto built = buildSolidAngleIblAliasTable(px.data(), w, h);
    const auto table = toDeviceTable(built.alias);
    const uint32_t n = (uint32_t)table.size();

    // The jitter is what stops the sampler returning only texel centres, so it
    // has to be a usable variate and not a constant.
    int lowHalf = 0;
    const int draws = 20000;
    for (int s = 0; s < draws; ++s)
    {
        const float xi = ((float)s + 0.5f) / (float)draws;
        const EnvAliasDraw d = envAliasDraw(table.data(), n, xi);
        CHECK(d.frac >= 0.0f);
        CHECK(d.frac < 1.0f);
        if (d.frac < 0.5f)
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

    // 1 - 2^-24 is the largest float below one, and n * it rounds to n.
    for (const float xi : { 0.0f, 0.99999994f, 1.0f })
    {
        const EnvAliasDraw d = envAliasDraw(table.data(), n, xi);
        CHECK(d.texel < n);
    }
}

TEST_CASE("envAliasDraw refuses an empty table instead of reading it")
{
    const EnvAliasDraw a = envAliasDraw(nullptr, 16, 0.5f);
    CHECK(a.texel == 0u);
    CHECK(a.frac == 0.0f);

    std::vector<EnvAliasEntry> table(1);
    table[0].prob = 1.0f;
    table[0].alias = 0u;
    const EnvAliasDraw b = envAliasDraw(table.data(), 0, 0.5f);
    CHECK(b.texel == 0u);
    CHECK(b.frac == 0.0f);
}
