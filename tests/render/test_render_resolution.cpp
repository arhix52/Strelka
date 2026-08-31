#include <doctest/doctest.h>

#include <host/render_resolution.h>

using namespace oka::render_resolution;

TEST_CASE("path trace resolution matches preview when upscale is disabled")
{
    const Resolution resolution = resolve(960, 540, false, 0.25f);

    CHECK(resolution.outputWidth == 960);
    CHECK(resolution.outputHeight == 540);
    CHECK(resolution.pathTraceWidth == 960);
    CHECK(resolution.pathTraceHeight == 540);
    CHECK(resolution.appliedScale == doctest::Approx(1.0f));
    CHECK_FALSE(resolution.upscaling);
}

TEST_CASE("path trace resolution scales inside fixed preview")
{
    const Resolution half = resolve(960, 540, true, 0.5f);
    const Resolution quarter = resolve(960, 540, true, 0.25f);

    CHECK(half.pathTraceWidth == 480);
    CHECK(half.pathTraceHeight == 270);
    CHECK(quarter.pathTraceWidth == 240);
    CHECK(quarter.pathTraceHeight == 135);
    CHECK(half.outputWidth == quarter.outputWidth);
    CHECK(half.outputHeight == quarter.outputHeight);
}

TEST_CASE("path trace scale clamps and odd dimensions match renderer truncation")
{
    const Resolution low = resolve(801, 601, true, 0.1f);
    const Resolution high = resolve(801, 601, true, 2.0f);

    CHECK(low.appliedScale == doctest::Approx(0.25f));
    CHECK(low.pathTraceWidth == 200);
    CHECK(low.pathTraceHeight == 150);
    CHECK(high.appliedScale == doctest::Approx(1.0f));
    CHECK(high.pathTraceWidth == 801);
    CHECK(high.pathTraceHeight == 601);
    CHECK_FALSE(high.upscaling);
}

TEST_CASE("resolution never reaches zero")
{
    const Resolution resolution = resolve(0, 0, true, 0.25f);

    CHECK(resolution.outputWidth == 1);
    CHECK(resolution.outputHeight == 1);
    CHECK(resolution.pathTraceWidth == 1);
    CHECK(resolution.pathTraceHeight == 1);
}

TEST_CASE("supported temporal denoiser scale stays temporal")
{
    const Resolution resolution = resolve(960, 540, true, 0.5f);
    const DenoiserPolicy policy = resolveDenoiserPolicy(true, resolution, 2.0f);

    CHECK(policy.useDenoiser);
    CHECK_FALSE(policy.useSpatialFallback);
    CHECK(policy.lowestSupportedScale == doctest::Approx(0.5f));
}

TEST_CASE("unsupported denoiser scale falls back without raising PT resolution")
{
    const Resolution resolution = resolve(960, 540, true, 0.25f);
    const DenoiserPolicy policy = resolveDenoiserPolicy(true, resolution, 3.0f);

    CHECK_FALSE(policy.useDenoiser);
    CHECK(policy.useSpatialFallback);
    CHECK(resolution.pathTraceWidth == 240);
    CHECK(resolution.pathTraceHeight == 135);
    CHECK(resolution.appliedScale == doctest::Approx(0.25f));
}

TEST_CASE("disabled denoiser ignores MetalFX scale range")
{
    const Resolution resolution = resolve(960, 540, true, 0.25f);
    const DenoiserPolicy policy = resolveDenoiserPolicy(false, resolution, 3.0f);

    CHECK_FALSE(policy.useDenoiser);
    CHECK_FALSE(policy.useSpatialFallback);
}

TEST_CASE("integer truncation cannot exceed the denoiser scale range")
{
    const Resolution resolution = resolve(515, 387, true, 1.0f / 3.0f);
    const DenoiserPolicy policy = resolveDenoiserPolicy(true, resolution, 3.0f);

    CHECK(resolution.pathTraceWidth == 171);
    CHECK(resolution.pathTraceHeight == 129);
    CHECK(policy.useSpatialFallback);
}
