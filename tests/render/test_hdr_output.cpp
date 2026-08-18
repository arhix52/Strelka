#include <doctest/doctest.h>

#include <hdr_output.h>

namespace
{
void checkColor(const oka::hdr_output::HdrFloat3& actual,
                const oka::hdr_output::HdrFloat3& expected,
                float epsilon = 0.000001f)
{
    CHECK(actual.x == doctest::Approx(expected.x).epsilon(epsilon));
    CHECK(actual.y == doctest::Approx(expected.y).epsilon(epsilon));
    CHECK(actual.z == doctest::Approx(expected.z).epsilon(epsilon));
}
} // namespace

TEST_CASE("Rec.709 to Rec.2020 preserves neutral colors")
{
    const oka::hdr_output::HdrFloat3 white = oka::tonemap::make_float3(1.0f, 1.0f, 1.0f);

    checkColor(oka::hdr_output::rec709ToRec2020(white), white);
}

TEST_CASE("Rec.709 primaries map to Rec.2020")
{
    const oka::hdr_output::HdrFloat3 red = oka::tonemap::make_float3(1.0f, 0.0f, 0.0f);
    const oka::hdr_output::HdrFloat3 green = oka::tonemap::make_float3(0.0f, 1.0f, 0.0f);
    const oka::hdr_output::HdrFloat3 blue = oka::tonemap::make_float3(0.0f, 0.0f, 1.0f);

    checkColor(oka::hdr_output::rec709ToRec2020(red), oka::tonemap::make_float3(0.6274040f, 0.0690970f, 0.0163916f));
    checkColor(oka::hdr_output::rec709ToRec2020(green), oka::tonemap::make_float3(0.3292820f, 0.9195400f, 0.0880132f));
    checkColor(oka::hdr_output::rec709ToRec2020(blue), oka::tonemap::make_float3(0.0433136f, 0.0113612f, 0.8955950f));
}

TEST_CASE("ST 2084 encodes absolute luminance reference points")
{
    CHECK(oka::hdr_output::st2084EncodeNits(0.0f) == doctest::Approx(0.0f).epsilon(0.000001f));
    CHECK(oka::hdr_output::st2084EncodeNits(100.0f) == doctest::Approx(0.508078f).epsilon(0.00001f));
    CHECK(oka::hdr_output::st2084EncodeNits(1000.0f) == doctest::Approx(0.751827f).epsilon(0.00001f));
    CHECK(oka::hdr_output::st2084EncodeNits(10000.0f) == doctest::Approx(1.0f).epsilon(0.000001f));
    CHECK(oka::hdr_output::st2084EncodeNits(20000.0f) == doctest::Approx(1.0f).epsilon(0.000001f));
    CHECK(oka::hdr_output::st2084EncodeNits(-1.0f) == doctest::Approx(0.0f).epsilon(0.000001f));
}

TEST_CASE("paper white and peak define linear display headroom")
{
    CHECK(oka::hdr_output::displayHeadroom(203.0f, 1000.0f) ==
          doctest::Approx(1000.0f / 203.0f));
    CHECK(oka::hdr_output::linearDisplayToNits(1.0f, 203.0f, 1000.0f) ==
          doctest::Approx(203.0f));
    CHECK(oka::hdr_output::linearDisplayToNits(
              1000.0f / 203.0f, 203.0f, 1000.0f) ==
          doctest::Approx(1000.0f));
    CHECK(oka::hdr_output::linearDisplayToNits(8.0f, 203.0f, 1000.0f) ==
          doctest::Approx(1000.0f));
}

TEST_CASE("invalid HDR luminance settings retain finite headroom")
{
    CHECK(oka::hdr_output::displayHeadroom(0.0f, -10.0f) ==
          doctest::Approx(1.0f));
    CHECK(oka::hdr_output::linearDisplayToNits(-1.0f, 203.0f, 1000.0f) ==
          doctest::Approx(0.0f));
}
