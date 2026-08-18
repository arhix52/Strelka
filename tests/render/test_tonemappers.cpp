#include <doctest/doctest.h>

#include <strelka/render/buffer.h>
#include <tonemappers.h>

namespace
{
void checkColor(const oka::tonemap::float3& actual, const oka::tonemap::float3& expected)
{
    CHECK(actual.x == doctest::Approx(expected.x));
    CHECK(actual.y == doctest::Approx(expected.y));
    CHECK(actual.z == doctest::Approx(expected.z));
}
} // namespace

TEST_CASE("EDR tone mappers retain their SDR curves at unit headroom")
{
    const oka::tonemap::float3 color = oka::tonemap::make_float3(0.25f, 1.0f, 8.0f);

    checkColor(oka::tonemap::reinhard(color, 1.0f), oka::tonemap::reinhard(color));
    checkColor(oka::tonemap::ACESFitted(color, 1.0f), oka::tonemap::ACESFitted(color));
    checkColor(oka::tonemap::ACESFilm(color, 1.0f), oka::tonemap::ACESFilm(color));
}

TEST_CASE("EDR tone mappers move the shoulder to display headroom")
{
    const float headroom = 4.0f;
    const oka::tonemap::float3 color = oka::tonemap::make_float3(0.25f, 1.0f, 8.0f);
    const oka::tonemap::float3 scaledColor = color * headroom;

    checkColor(oka::tonemap::reinhard(scaledColor, headroom), oka::tonemap::reinhard(color) * headroom);
    checkColor(oka::tonemap::ACESFitted(scaledColor, headroom), oka::tonemap::ACESFitted(color) * headroom);
    checkColor(oka::tonemap::ACESFilm(scaledColor, headroom), oka::tonemap::ACESFilm(color) * headroom);
}

TEST_CASE("extended sRGB transfer preserves EDR values")
{
    CHECK(oka::tonemap::gammaFloat(1.0f, 2.4f) == doctest::Approx(1.0f));
    CHECK(oka::tonemap::gammaFloat(4.0f, 2.4f) > 1.0f);
}

TEST_CASE("presentation metadata defaults describe an identity linear handoff")
{
    const oka::PresentationMetadata metadata{};
    const oka::ImageBuffer image{};

    CHECK(metadata.content == oka::PresentationContent::SceneLinear);
    CHECK(metadata.exposure[0] == doctest::Approx(1.0f));
    CHECK(metadata.exposure[1] == doctest::Approx(1.0f));
    CHECK(metadata.exposure[2] == doctest::Approx(1.0f));
    CHECK(metadata.maxOutput == doctest::Approx(1.0f));
    CHECK(metadata.gamma == doctest::Approx(0.0f));
    CHECK(metadata.tonemapper == 0u);
    CHECK(image.frameSerial == 0u);
    CHECK(oka::shouldApplyPresentationTransform(metadata));
}

TEST_CASE("debug presentation content explicitly bypasses display transforms")
{
    oka::PresentationMetadata metadata{};

    metadata.exposure[0] = 4.0f;
    metadata.maxOutput = 2.0f;
    metadata.gamma = 2.4f;
    metadata.tonemapper = 2u;
    metadata.content = oka::PresentationContent::DebugDisplayLinear;

    CHECK(metadata.content == oka::PresentationContent::DebugDisplayLinear);
    CHECK_FALSE(oka::shouldApplyPresentationTransform(metadata));
}

TEST_CASE("presentation frame serial transforms each published frame once")
{
    const uint64_t transformedFrameSerial = 41;

    CHECK_FALSE(oka::shouldTransformFrame(0, transformedFrameSerial));
    CHECK_FALSE(
        oka::shouldTransformFrame(transformedFrameSerial, transformedFrameSerial));
    CHECK(oka::shouldTransformFrame(42, transformedFrameSerial));
}
