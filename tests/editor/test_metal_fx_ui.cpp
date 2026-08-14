#include <doctest/doctest.h>

#include "editor_metal_fx.h"

using oka::editor_metal_fx::Mode;
using oka::editor_metal_fx::modeFromSettings;
using oka::editor_metal_fx::shouldUpscale;

TEST_CASE("spatial MetalFX selection survives one-to-one scale")
{
    Mode selectedMode = Mode::Spatial;

    CHECK_FALSE(shouldUpscale(selectedMode, 1.0f));
    CHECK(selectedMode == Mode::Spatial);
    CHECK(shouldUpscale(selectedMode, 0.5f));
}

TEST_CASE("MetalFX UI initializes from renderer settings")
{
    CHECK(modeFromSettings(false, false) == Mode::Off);
    CHECK(modeFromSettings(false, true) == Mode::Spatial);
    CHECK(modeFromSettings(true, false) == Mode::TemporalDenoise);
    CHECK(modeFromSettings(true, true) == Mode::TemporalDenoise);
}

TEST_CASE("off mode never enables upscaling")
{
    CHECK_FALSE(shouldUpscale(Mode::Off, 0.25f));
    CHECK_FALSE(shouldUpscale(Mode::Off, 1.0f));
}
