#include <doctest/doctest.h>

#include <strelka/render/buffer.h>
#include <tonemappers.h>

#include <algorithm>

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

TEST_CASE("EDR headroom leaves shadows and midtones on the SDR curve")
{
    // The defect this pins down: scaling both of the curve's axes to the display
    // peak rebuilt the whole tone, and ACES at 3.54x took middle grey from 0.106
    // to 0.050 -- picking an HDR display mode darkened the picture by a stop and
    // a half. Everything the SDR curve maps below the knee must come back
    // untouched, at any headroom.
    for (const float headroom : { 1.5f, 3.54f, 16.0f })
    {
        for (const float level : { 0.02f, 0.05f, 0.18f, 0.35f })
        {
            const oka::tonemap::float3 color = oka::tonemap::make_float3(level, level, level);

            CAPTURE(headroom);
            CAPTURE(level);
            checkColor(oka::tonemap::reinhard(color, headroom), oka::tonemap::reinhard(color));
            checkColor(oka::tonemap::ACESFitted(color, headroom), oka::tonemap::ACESFitted(color));
            checkColor(oka::tonemap::ACESFilm(color, headroom), oka::tonemap::ACESFilm(color));
        }
    }
}

TEST_CASE("EDR headroom lifts what the SDR curve was clipping")
{
    const float headroom = 4.0f;
    const oka::tonemap::float3 bright = oka::tonemap::make_float3(8.0f, 8.0f, 8.0f);

    // Above white the curve has to actually use the range, or the display mode
    // is a no-op with extra steps.
    CHECK(oka::tonemap::reinhard(bright, headroom).x > oka::tonemap::reinhard(bright).x * 1.5f);
    CHECK(oka::tonemap::ACESFitted(bright, headroom).x > oka::tonemap::ACESFitted(bright).x * 1.5f);
    CHECK(oka::tonemap::ACESFilm(bright, headroom).x > oka::tonemap::ACESFilm(bright).x * 1.5f);
}

TEST_CASE("EDR headroom never returns less than the SDR curve")
{
    // The first attempt at this blended toward `f(x / headroom) * headroom`, which
    // is above the SDR curve only for a curve concave through the origin. ACES has
    // an S-curve toe, so at 16x that reference falls *below* the SDR curve from a
    // third of a stop under white to half a stop over it, and the blend darkened
    // exactly the pixels it was supposed to be lifting -- a mean lift of x0.97 on
    // the iso_bathroom frame. More headroom must never mean a darker pixel.
    for (const float headroom : { 1.5f, 3.54f, 16.0f })
    {
        for (int step = 0; step <= 400; ++step)
        {
            const float level = float(step) * 0.05f;
            const oka::tonemap::float3 color = oka::tonemap::make_float3(level, level, level);

            CAPTURE(headroom);
            CAPTURE(level);
            CHECK(oka::tonemap::reinhard(color, headroom).x >= oka::tonemap::reinhard(color).x);
            CHECK(oka::tonemap::ACESFitted(color, headroom).x >= oka::tonemap::ACESFitted(color).x);
            CHECK(oka::tonemap::ACESFilm(color, headroom).x >= oka::tonemap::ACESFilm(color).x);
        }
    }
}

TEST_CASE("EDR headroom is a ceiling the curve approaches but never passes")
{
    const float headroom = 4.0f;
    // Far past anything the shoulder still resolves; the result must land on the
    // display peak rather than run off past what the compositor can show.
    const oka::tonemap::float3 huge = oka::tonemap::make_float3(1.0e4f, 1.0e4f, 1.0e4f);

    CHECK(oka::tonemap::reinhard(huge, headroom).x <= doctest::Approx(headroom));
    CHECK(oka::tonemap::ACESFitted(huge, headroom).x == doctest::Approx(headroom).epsilon(0.02));
    CHECK(oka::tonemap::ACESFilm(huge, headroom).x == doctest::Approx(headroom).epsilon(0.02));
}

TEST_CASE("EDR headroom respects the peak a saturated pixel can already reach")
{
    // Found on a Cornell box frame, not on a grey ramp. Reinhard divides by the
    // pixel's luminance, so a channel far brighter than that luminance comes out
    // above 1 from the SDR curve alone -- thousands of pixels on that frame. A
    // lift budgeted from white rather than from where the curve actually landed
    // pushed them past the display peak, where the window server clips per
    // channel and shifts the hue on the way.
    const oka::tonemap::float3 saturated = oka::tonemap::make_float3(40.0f, 0.5f, 0.5f);

    for (const float headroom : { 1.5f, 3.54f, 16.0f })
    {
        const oka::tonemap::float3 sdr = oka::tonemap::reinhard(saturated);
        const oka::tonemap::float3 hdr = oka::tonemap::reinhard(saturated, headroom);

        CAPTURE(headroom);
        // The SDR curve overshoots on its own here, which is the whole point of
        // the case; pulling that back would be darker than SDR, so the bound is
        // whichever of the two is higher.
        CHECK(sdr.x > 1.0f);
        CHECK(hdr.x >= sdr.x);
        CHECK(hdr.x <= std::max(headroom, sdr.x) + 1e-4f);
    }
}

TEST_CASE("EDR headroom keeps every curve monotone across the knee")
{
    // The knee joins two different curves, so the join is where an inversion
    // would hide: a brighter pixel coming out darker reads as a hard ring around
    // every light source.
    const float headroom = 3.54f;
    float prevReinhard = -1.0f;
    float prevFitted = -1.0f;
    float prevFilm = -1.0f;

    for (int step = 0; step <= 2000; ++step)
    {
        const float level = float(step) * 0.01f;
        const oka::tonemap::float3 color = oka::tonemap::make_float3(level, level, level);
        const float r = oka::tonemap::reinhard(color, headroom).x;
        const float a = oka::tonemap::ACESFitted(color, headroom).x;
        const float f = oka::tonemap::ACESFilm(color, headroom).x;

        CAPTURE(level);
        CHECK(r >= prevReinhard);
        CHECK(a >= prevFitted);
        CHECK(f >= prevFilm);
        prevReinhard = r;
        prevFitted = a;
        prevFilm = f;
    }
}

TEST_CASE("extended sRGB transfer preserves EDR values")
{
    CHECK(oka::tonemap::gammaFloat(1.0f, 2.4f) == doctest::Approx(1.0f));
    CHECK(oka::tonemap::gammaFloat(4.0f, 2.4f) > 1.0f);
}

TEST_CASE("sRGB transfer round trips spatial scaler pixels back to display linear")
{
    for (const float value : { 0.0f, 0.0031308f, 0.18f, 1.0f, 4.0f })
    {
        const float encoded = oka::tonemap::gammaFloat(value, 2.4f);
        CHECK(oka::tonemap::inverseGammaFloat(encoded, 2.4f) == doctest::Approx(value).epsilon(1e-5));
    }
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
    CHECK(metadata.sourceWidth == 0u);
    CHECK(metadata.sourceHeight == 0u);
    CHECK(metadata.resampling == oka::PresentationResampling::None);
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
