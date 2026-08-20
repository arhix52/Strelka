#include <doctest/doctest.h>

#include <env_map_math.h>
#include <ibl_alias_table.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

// ============================================================================
// test_env_map_math.cpp -- the equirectangular parametrisation and the density
// built on it.
//
// These four functions used to exist twice, in common/env_light.h for CUDA and
// metal/env_light_metal.h for Metal, with a comment on each asking that the two
// stay identical and nothing able to check it: the suite could compile neither
// header. They are one file now, and this is that check.
//
// Two properties matter and they are different in kind:
//
//   1. The parametrisation has to be invertible, because the sampler picks a
//      texel, turns (u, v) into a direction, and the MIS weight turns that
//      direction back into a texel. If the round trip lands on a neighbour, the
//      two halves of the estimate are dividing by densities read from different
//      pixels.
//
//   2. The density has to integrate to one. envTexelPdf() is luminance times a
//      single precomputed constant, which only works because the sin(theta) in
//      the texel's solid angle cancels the sin(theta) in its sampling weight.
//      That cancellation is the whole design, and it is one line away from
//      being wrong in a way no image would show.
// ============================================================================

namespace
{

double angleBetween(float3 a, float3 b)
{
    const double d = std::max(-1.0, std::min(1.0, double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z));
    return std::acos(d);
}

/// The exact solid angle of texel row `y` in a map of `h` rows, per column.
double texelSolidAngle(int y, int w, int h)
{
    const double thetaTop = M_PI * double(y) / double(h);
    const double thetaBottom = M_PI * double(y + 1) / double(h);
    return (2.0 * M_PI / double(w)) * (std::cos(thetaTop) - std::cos(thetaBottom));
}

/// Build a map, then integrate the density the shaders would report for it over
/// the texels' true solid angles.
double integrateReportedPdf(int w, int h, const std::function<float(double, double)>& sky)
{
    std::vector<float> px(size_t(w) * size_t(h) * 4, 0.0f);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            const double theta = M_PI * (double(y) + 0.5) / double(h);
            const double phi = 2.0 * M_PI * (double(x) + 0.5) / double(w);
            const float v = sky(theta, phi);
            const size_t i = (size_t(y) * size_t(w) + size_t(x)) * 4;
            px[i + 0] = px[i + 1] = px[i + 2] = v;
            px[i + 3] = 1.0f;
        }
    }

    const oka::metal::IblAliasTableResult table = oka::metal::buildIblAliasTable(px.data(), w, h);

    double total = 0.0;
    for (int y = 0; y < h; ++y)
    {
        const double dOmega = texelSolidAngle(y, w, h);
        for (int x = 0; x < w; ++x)
        {
            const size_t i = (size_t(y) * size_t(w) + size_t(x)) * 4;
            const float3 radiance = make_float3(px[i + 0], px[i + 1], px[i + 2]);
            total += double(envTexelPdf(radiance, table.envPdfScale)) * dOmega;
        }
    }
    return total;
}

} // namespace

TEST_CASE("uv and direction are inverses of each other")
{
    std::mt19937 rng(0xE7Fu);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);

    for (float rotation : { 0.0f, 0.7f, -2.3f, 6.28f })
    {
        CAPTURE(rotation);
        double worst = 0.0;
        for (int i = 0; i < 20000; ++i)
        {
            const float2 uv = make_float2(U(rng), U(rng));
            const float3 dir = envUVToDir(uv, rotation);
            REQUIRE(std::abs(length(dir) - 1.0f) < 1e-4f);

            const float2 back = dirToEnvUV(dir, rotation);
            // Compare through the direction rather than the uv: u is degenerate
            // at the poles, where every azimuth is the same point.
            const float3 again = envUVToDir(back, rotation);
            worst = std::max(worst, angleBetween(dir, again));
        }
        CHECK(worst < 1e-3);
    }
}

TEST_CASE("rotating the map rotates the direction about Y and nothing else")
{
    const float2 uv = make_float2(0.31f, 0.42f);
    const float3 unrotated = envUVToDir(uv, 0.0f);
    const float rotation = 0.9f;
    const float3 rotated = envUVToDir(uv, rotation);

    // The elevation is untouched...
    CHECK(rotated.y == doctest::Approx(unrotated.y).epsilon(1e-5));
    // ...and the azimuth has moved by exactly the rotation.
    const double before = std::atan2(double(unrotated.x), double(unrotated.z));
    const double after = std::atan2(double(rotated.x), double(rotated.z));
    double delta = after - before;
    while (delta < -M_PI)
        delta += 2.0 * M_PI;
    while (delta > M_PI)
        delta -= 2.0 * M_PI;
    CHECK(delta == doctest::Approx(double(rotation)).epsilon(1e-4));
}

TEST_CASE("the poles map to the ends of the v range")
{
    const float2 up = dirToEnvUV(make_float3(0.0f, 1.0f, 0.0f), 0.0f);
    const float2 down = dirToEnvUV(make_float3(0.0f, -1.0f, 0.0f), 0.0f);
    CHECK(up.y == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(down.y == doctest::Approx(1.0f).epsilon(1e-6));

    // And a direction slightly off the axis stays inside the range, so the
    // texel clamp in envMapPdf() is a guard and not the thing keeping it legal.
    const float2 nearlyUp = dirToEnvUV(make_float3(1e-7f, 1.0f, 0.0f), 0.0f);
    CHECK(nearlyUp.y >= 0.0f);
    CHECK(nearlyUp.y <= 1.0f);
}

// ---------------------------------------------------------------------------
// The density
// ---------------------------------------------------------------------------
TEST_CASE("the reported density integrates to one over the sphere")
{
    // Deliberately including a sun at the zenith: that is where the row's
    // sin(theta) is smallest and where a mistake in the cancellation would show
    // up first.
    struct Sky
    {
        const char* name;
        int w;
        int h;
        std::function<float(double, double)> f;
    };
    const Sky skies[] = {
        { "uniform", 256, 128, [](double, double) { return 1.0f; } },
        { "uniform, larger", 1024, 512, [](double, double) { return 1.0f; } },
        { "gradient in theta", 256, 128, [](double th, double) { return float(0.1 + std::cos(th) * 0.5 + 0.5); } },
        { "sun at the horizon", 512, 256,
          [](double th, double ph) { return float(0.05 + (std::hypot(th - M_PI / 2, ph - M_PI) < 0.05 ? 500.0 : 0.0)); } },
        { "sun at the zenith", 512, 256, [](double th, double) { return float(0.05 + (th < 0.05 ? 500.0 : 0.0)); } },
    };

    for (const Sky& sky : skies)
    {
        CAPTURE(sky.name);
        // Not a Monte Carlo estimate -- an exact sum over texels of a piecewise
        // constant density, so the tolerance is the discretisation of the row's
        // sin(theta) and nothing else.
        CHECK(integrateReportedPdf(sky.w, sky.h, sky.f) == doctest::Approx(1.0).epsilon(1e-3));
    }
}

TEST_CASE("luminance matches the weight the host builds the table from")
{
    // buildIblAliasTable() weights texels by 0.2126/0.7152/0.0722 and the shader
    // divides by the result. Different coefficients on either side would be a
    // density for a map that was never sampled.
    const float3 c = make_float3(0.3f, 0.6f, 0.1f);
    CHECK(envLuminance(c) == doctest::Approx(0.2126f * 0.3f + 0.7152f * 0.6f + 0.0722f * 0.1f));
    CHECK(envTexelPdf(c, 2.5f) == doctest::Approx(envLuminance(c) * 2.5f));
    CHECK(envTexelPdf(make_float3(0.0f), 2.5f) == 0.0f);
}

// ---------------------------------------------------------------------------
// The round trip the two halves of the MIS estimate depend on
// ---------------------------------------------------------------------------
TEST_CASE("a jittered texel sample lands back in the texel it was drawn from")
{
    // sampleEnvMap() picks texel (x, y), jitters inside it and returns a
    // direction; envMapPdf() takes that direction and looks the texel up again.
    // They have to agree, or the light half divides by one texel's density while
    // the BSDF half weighs against another's.
    std::mt19937 rng(0x5A17u);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);

    for (int size : { 256, 1024 })
    {
        for (float rotation : { 0.0f, 1.1f })
        {
            CAPTURE(size);
            CAPTURE(rotation);
            const int w = size;
            const int h = size / 2;
            long mismatch = 0;
            const long trials = 200000;
            for (long i = 0; i < trials; ++i)
            {
                const uint32_t texel = uint32_t(U(rng) * float(w * h));
                const uint32_t x = texel % uint32_t(w);
                const uint32_t y = texel / uint32_t(w);

                const float u = (float(x) + U(rng)) / float(w);
                const float v = (float(y) + U(rng)) / float(h);
                const float3 dir = envUVToDir(make_float2(u, v), rotation);

                const float2 back = dirToEnvUV(dir, rotation);
                const uint32_t bx = uint32_t(std::clamp(int(back.x * float(w)), 0, w - 1));
                const uint32_t by = uint32_t(std::clamp(int(back.y * float(h)), 0, h - 1));
                if (bx != x || by != y)
                {
                    ++mismatch;
                }
            }
            // What survives is float precision at the texel boundary, where
            // acos/atan2 round across the edge. Measured at 0.001% for 256x128
            // and 0.02% at 4096x2048; anything above a tenth of a percent is a
            // parametrisation that no longer inverts.
            CHECK(double(mismatch) / double(trials) < 1e-3);
        }
    }
}
