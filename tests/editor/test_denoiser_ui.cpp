#include <doctest/doctest.h>

#include "editor_denoiser_ui.h"

#include <string>

using oka::Render;
using oka::editor_denoiser::appliedScale;
using oka::editor_denoiser::hasDenoiser;
using oka::editor_denoiser::modeAt;
using oka::editor_denoiser::modeIndexFromSettings;
using oka::editor_denoiser::resolution;
using oka::editor_denoiser::settingsMatchMode;
using oka::editor_denoiser::shouldUpscale;
using oka::editor_denoiser::Ui;

namespace
{
// The indices the panel's combo shows, per backend. Spelled out here rather than
// searched for, so a reordering of either table fails a test instead of quietly
// changing what the numbers below mean.
constexpr int kOff = 0;
constexpr int kMetalSpatial = 1;
constexpr int kMetalTemporalDenoise = 2;
constexpr int kOptixDenoise = 1;
constexpr int kOptixDenoiseUpscale = 2;

Ui metalFx()
{
    return oka::editor_denoiser::uiFor(Render::DenoiserKind::eMetalFx);
}
Ui optix()
{
    return oka::editor_denoiser::uiFor(Render::DenoiserKind::eOptixAi);
}
} // namespace

TEST_CASE("each backend names its own denoiser")
{
    CHECK(std::string(metalFx().title) == "MetalFX");
    CHECK(std::string(optix().title) == "OptiX denoiser");
    CHECK(hasDenoiser(metalFx()));
    CHECK(hasDenoiser(optix()));
    // A backend with no denoiser offers no modes, and the panel says so rather
    // than showing another backend's list.
    CHECK_FALSE(hasDenoiser(oka::editor_denoiser::uiFor(Render::DenoiserKind::eNone)));
}

TEST_CASE("the two backends do not offer the same modes")
{
    CHECK(std::string(modeAt(metalFx(), kMetalSpatial).label) == "Spatial upscale");
    CHECK(std::string(modeAt(metalFx(), kMetalTemporalDenoise).label) == "Temporal denoise");
    CHECK(std::string(modeAt(optix(), kOptixDenoise).label) == "Denoise");
    CHECK(std::string(modeAt(optix(), kOptixDenoiseUpscale).label) == "Denoise + 2x upscale");

    // MetalFX scales without denoising; OptiX has no such path -- every mode of
    // its plan that upscales also runs the network.
    CHECK_FALSE(modeAt(metalFx(), kMetalSpatial).denoise);
    CHECK(modeAt(metalFx(), kMetalSpatial).upscale);
    CHECK(modeAt(optix(), kOptixDenoiseUpscale).denoise);
}

TEST_CASE("spatial MetalFX selection survives one-to-one scale")
{
    CHECK_FALSE(shouldUpscale(metalFx(), kMetalSpatial, 1.0f));
    CHECK(shouldUpscale(metalFx(), kMetalSpatial, 0.5f));
    // ... and is still the selected mode afterwards, which is what
    // settingsMatchMode is there to protect: at 1:1 it wrote enableUpscale=false,
    // the same thing Off writes.
    CHECK(settingsMatchMode(metalFx(), kMetalSpatial, false, false, 1.0f));
    CHECK(settingsMatchMode(metalFx(), kMetalSpatial, false, true, 0.5f));
}

TEST_CASE("off mode never enables upscaling")
{
    CHECK_FALSE(shouldUpscale(metalFx(), kOff, 0.25f));
    CHECK_FALSE(shouldUpscale(metalFx(), kOff, 1.0f));
    CHECK_FALSE(shouldUpscale(optix(), kOff, 0.25f));
}

TEST_CASE("the OptiX upscaling mode ignores the render scale")
{
    // Its model is 2x and only 2x: the enable bit is all the plan reads, so a
    // scale slider next to it would move nothing. 0.25 and 1.00 alike upscale.
    CHECK(shouldUpscale(optix(), kOptixDenoiseUpscale, 0.25f));
    CHECK(shouldUpscale(optix(), kOptixDenoiseUpscale, 1.0f));
    CHECK(appliedScale(optix(), kOptixDenoiseUpscale, 0.25f) == doctest::Approx(0.5f));
    CHECK(appliedScale(optix(), kOptixDenoiseUpscale, 1.0f) == doctest::Approx(0.5f));
    // Denoising without upscaling traces at full resolution on either backend.
    CHECK(appliedScale(optix(), kOptixDenoise, 0.25f) == doctest::Approx(1.0f));
    CHECK(appliedScale(metalFx(), kMetalTemporalDenoise, 0.5f) == doctest::Approx(0.5f));
}

TEST_CASE("the mode combo initializes from renderer settings")
{
    CHECK(modeIndexFromSettings(metalFx(), false, false) == kOff);
    CHECK(modeIndexFromSettings(metalFx(), false, true) == kMetalSpatial);
    CHECK(modeIndexFromSettings(metalFx(), true, true) == kMetalTemporalDenoise);
    // Denoising without upscaling is not a MetalFX mode of its own -- that is
    // "Temporal denoise" with the scale at 1:1, which is what runs.
    CHECK(modeIndexFromSettings(metalFx(), true, false) == kMetalTemporalDenoise);

    CHECK(modeIndexFromSettings(optix(), false, false) == kOff);
    CHECK(modeIndexFromSettings(optix(), true, false) == kOptixDenoise);
    CHECK(modeIndexFromSettings(optix(), true, true) == kOptixDenoiseUpscale);
    // `render.upscale` alone: the OptiX plan upscales and denoises anyway, so the
    // mode that says so is the honest one to show.
    CHECK(modeIndexFromSettings(optix(), false, true) == kOptixDenoiseUpscale);
}

TEST_CASE("settings moved from outside the panel re-derive the mode")
{
    // A benchmark driver turning denoising on behind the combo's back.
    CHECK_FALSE(settingsMatchMode(metalFx(), kOff, true, true, 0.5f));
    CHECK_FALSE(settingsMatchMode(optix(), kOff, true, false, 1.0f));
    CHECK(settingsMatchMode(optix(), kOptixDenoise, true, false, 1.0f));
    CHECK(settingsMatchMode(optix(), kOptixDenoiseUpscale, true, true, 1.0f));
}

TEST_CASE("the resolution readout follows each backend's own rule")
{
    // MetalFX: whatever fraction it is given.
    const oka::editor_denoiser::Resolution metalRes = resolution(metalFx(), kMetalTemporalDenoise, 0.75f, 1920, 1080);
    CHECK(metalRes.pathTraceWidth == 1440);
    CHECK(metalRes.pathTraceHeight == 810);
    CHECK(metalRes.upscaling);

    // OptiX: exactly half, whatever the slider last held.
    const oka::editor_denoiser::Resolution optixRes = resolution(optix(), kOptixDenoiseUpscale, 0.75f, 1920, 1080);
    CHECK(optixRes.pathTraceWidth == 960);
    CHECK(optixRes.pathTraceHeight == 540);
    CHECK(optixRes.upscaling);

    // An odd output dimension: the 2x model would return an image a pixel short,
    // so denoisePlan declines to upscale at all and the readout has to say the
    // tracer is running at full size -- otherwise it reports half a frame that
    // never happens.
    const oka::editor_denoiser::Resolution oddRes = resolution(optix(), kOptixDenoiseUpscale, 0.5f, 1921, 1080);
    CHECK(oddRes.pathTraceWidth == 1921);
    CHECK(oddRes.pathTraceHeight == 1080);
    CHECK_FALSE(oddRes.upscaling);

    // Off traces at the output resolution on both.
    CHECK(resolution(optix(), kOff, 0.25f, 800, 600).pathTraceWidth == 800);
    CHECK(resolution(metalFx(), kOff, 0.25f, 800, 600).pathTraceWidth == 800);
}
