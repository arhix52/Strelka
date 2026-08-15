#include <doctest/doctest.h>

#include "optix_denoise_plan.h"

#include <initializer_list>
#include <limits>

using oka::DenoiseModelKind;
using oka::denoiseBufferLayout;
using oka::denoisePlan;

namespace guides = oka::guides;

TEST_CASE("nothing asked for means nothing configured")
{
    const auto plan = denoisePlan(false, false, 0, 0, 512, 512);
    CHECK(plan.kind == DenoiseModelKind::eNone);
    CHECK_FALSE(plan.enabled());
    CHECK_FALSE(plan.writeAov);
    // The path tracer still renders at the caller's resolution.
    CHECK(plan.renderWidth == 512u);
    CHECK(plan.renderHeight == 512u);
}

TEST_CASE("a guide debug view produces guides without a denoiser")
{
    // Debug 0..2 are the beauty and the two single-hit views; 3 upward are the
    // guides themselves, and looking at one has to make it exist.
    CHECK_FALSE(denoisePlan(false, false, 0, 2, 64, 64).writeAov);
    for (uint32_t debugMode = 3; debugMode <= 10; ++debugMode)
    {
        const auto plan = denoisePlan(false, false, 0, debugMode, 64, 64);
        CHECK(plan.writeAov);
        CHECK(plan.kind == DenoiseModelKind::eNone);
    }
}

TEST_CASE("denoise selects the single-image model and upscale_mode makes it temporal")
{
    const auto spatial = denoisePlan(true, false, 0, 0, 512, 512);
    CHECK(spatial.kind == DenoiseModelKind::eAov);
    CHECK_FALSE(spatial.temporal);
    CHECK(spatial.writeAov);
    CHECK(spatial.renderWidth == 512u);

    const auto temporal = denoisePlan(true, false, 1, 0, 512, 512);
    CHECK(temporal.kind == DenoiseModelKind::eTemporalAov);
    CHECK(temporal.temporal);
}

TEST_CASE("upscaling halves the render resolution and keeps the output resolution")
{
    const auto plan = denoisePlan(false, true, 0, 0, 800, 600);
    CHECK(plan.kind == DenoiseModelKind::eUpscale2x);
    CHECK(plan.upscale);
    CHECK(plan.renderWidth == 400u);
    CHECK(plan.renderHeight == 300u);
    CHECK(plan.outputWidth == 800u);
    CHECK(plan.outputHeight == 600u);
    // Upscaling implies the network runs, so guides are needed either way.
    CHECK(plan.writeAov);

    CHECK(denoisePlan(true, true, 1, 0, 800, 600).kind == DenoiseModelKind::eTemporalUpscale2x);
}

TEST_CASE("an odd output size declines to upscale rather than change the frame size")
{
    // 2x of floor(801/2) is 800, which is not what the caller asked for. Denoise
    // still runs when it was asked for; the resolution is left alone.
    const auto declined = denoisePlan(true, true, 0, 0, 801, 600);
    CHECK(declined.kind == DenoiseModelKind::eAov);
    CHECK_FALSE(declined.upscale);
    CHECK(declined.renderWidth == 801u);

    const auto nothing = denoisePlan(false, true, 0, 0, 801, 600);
    CHECK(nothing.kind == DenoiseModelKind::eNone);
    CHECK(nothing.renderWidth == 801u);
}

TEST_CASE("buffer layout follows the render resolution, not the output one")
{
    const auto plan = denoisePlan(true, true, 0, 0, 512, 512);
    const auto layout = denoiseBufferLayout(plan, 64);
    CHECK(layout.renderPixels == 256u * 256u);
    CHECK(layout.outputPixels == 512u * 512u);
    CHECK(layout.aovBytes == (size_t)256 * 256 * 64);
    CHECK(layout.colorBytes == (size_t)256 * 256 * 16);
    CHECK(layout.albedoBytes == layout.colorBytes);
    CHECK(layout.normalBytes == layout.colorBytes);
    CHECK(layout.flowBytes == (size_t)256 * 256 * 8);
    // Flow trustworthiness is one float per pixel, not four.
    CHECK(layout.flowTrustBytes == (size_t)256 * 256 * 4);
    // The denoised image is the one the caller receives, so it is full size.
    CHECK(layout.denoisedBytes == (size_t)512 * 512 * 16);
}

TEST_CASE("the guide walk stops at the first surface it can describe")
{
    // A mirror: too smooth to describe, so the walk goes on.
    CHECK_FALSE(guides::guideWorthy(false, 0, 0.0f));
    CHECK_FALSE(guides::guideWorthy(false, 0, guides::kGuideRoughnessFloor));
    // Anything rougher than the floor is the answer.
    CHECK(guides::guideWorthy(false, 0, 0.2f));
    // With guidePrimaryHit the camera-visible surface is the answer whatever it
    // is made of, and nothing past it ever is.
    CHECK(guides::guideWorthy(true, 0, 0.0f));
    CHECK_FALSE(guides::guideWorthy(true, 1, 1.0f));
}

TEST_CASE("the walk is bounded, and never overwrites a record it already wrote")
{
    // Smooth all the way down: at the last-chance depth the record is written
    // anyway, because imperfect guides beat none at all.
    CHECK_FALSE(guides::shouldWriteGuide(true, false, false, 0, 0.0f));
    CHECK_FALSE(guides::shouldWriteGuide(true, false, false, 1, 0.0f));
    CHECK(guides::shouldWriteGuide(true, false, false, 2, 0.0f));
    // Already written: the bounce after the first describable surface must not
    // replace it.
    CHECK_FALSE(guides::shouldWriteGuide(true, true, false, 2, 0.0f));
    // Not producing guides this frame.
    CHECK_FALSE(guides::shouldWriteGuide(false, false, false, 0, 1.0f));
    // guidePrimaryHit still writes at depth 0 and is bounded the same way.
    CHECK(guides::shouldWriteGuide(true, false, true, 0, 0.0f));
    CHECK_FALSE(guides::shouldWriteGuide(true, false, true, 1, 1.0f));
}

TEST_CASE("reactive marks deferred guides and nothing else")
{
    // The guide came from the camera-visible surface: the history is good.
    CHECK(guides::reactiveFor(0) == doctest::Approx(0.0f));
    // The guide came from somewhere the camera cannot see directly.
    CHECK(guides::reactiveFor(1) == doctest::Approx(1.0f));
    CHECK(guides::reactiveFor(2) == doctest::Approx(1.0f));
}

TEST_CASE("the background depth sentinel matches the depth convention")
{
    // Device depth has a finite far plane, so the sky has to be the far value
    // rather than a large number, or it reads as nearer than the geometry.
    CHECK(guides::backgroundDepth(guides::kDepthDevice) == doctest::Approx(0.0f));
    CHECK(guides::backgroundDepth(guides::kDepthViewZ) > 1e6f);
    CHECK(guides::backgroundDepth(guides::kDepthRadial) > 1e6f);
}

TEST_CASE("a still camera produces no motion")
{
    // A point that projects back to exactly where the ray went through.
    const auto m = guides::screenMotion(0.0f, 0.0f, 1.0f, 256.0f, 256.0f, 512, 512);
    CHECK(m.x == doctest::Approx(0.0f));
    CHECK(m.y == doctest::Approx(0.0f));
}

TEST_CASE("motion is measured in pixels, y down")
{
    // Clip (0.5, 0.5, w=1) is NDC (0.5, 0.5): three quarters across, and a
    // quarter down from the top because NDC +y is up.
    const auto m = guides::screenMotion(0.5f, 0.5f, 1.0f, 256.0f, 256.0f, 512, 512);
    CHECK(m.x == doctest::Approx(384.0f - 256.0f));
    CHECK(m.y == doctest::Approx(128.0f - 256.0f));
}

TEST_CASE("a degenerate reprojection reports no motion rather than a huge one")
{
    // w at or below zero is a point on or behind the previous camera's plane.
    // Dividing by it does not give a large motion vector, it gives a meaningless
    // one, and the denoiser would then fetch history from those coordinates.
    for (float w : { 0.0f, -1.0f, 1e-9f })
    {
        const auto m = guides::screenMotion(1000.0f, 1000.0f, w, 256.0f, 256.0f, 512, 512);
        CHECK(m.x == doctest::Approx(0.0f));
        CHECK(m.y == doctest::Approx(0.0f));
    }
}

TEST_CASE("motion is clamped to something that can still be reprojected")
{
    // Further than the frame is across and the history lookup lands outside the
    // image. Clamped rather than zeroed, so a fast object still drags its
    // history the right way.
    const auto m = guides::screenMotion(1e6f, -1e6f, 1.0f, 256.0f, 256.0f, 512, 512);
    CHECK(m.x == doctest::Approx(1024.0f));
    CHECK(m.y == doctest::Approx(1024.0f));
    CHECK(m.x <= 1024.0f);
}

TEST_CASE("the firefly clamp scales rather than drops, and is off at zero")
{
    // Off.
    CHECK(guides::fireflyScale(1000.0f, 0.0f) == doctest::Approx(1.0f));
    // Under the ceiling: untouched, so an ordinary highlight keeps its energy.
    CHECK(guides::fireflyScale(4.0f, 8.0f) == doctest::Approx(1.0f));
    // Over it: scaled to exactly the ceiling, which keeps the pixel's hue.
    CHECK(guides::fireflyScale(80.0f, 8.0f) == doctest::Approx(0.1f));
    CHECK(80.0f * guides::fireflyScale(80.0f, 8.0f) == doctest::Approx(8.0f));
    // A NaN luminance must not turn the colour into a NaN as well.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(guides::fireflyScale(nan, 8.0f) == doctest::Approx(1.0f));
}
