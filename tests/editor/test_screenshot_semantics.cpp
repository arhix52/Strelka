#include <doctest/doctest.h>

#include "editor_screenshot.h"

using oka::editor_screenshot::Source;
using oka::editor_screenshot::encodeSrgb;
using oka::editor_screenshot::sourceForExtension;

TEST_CASE("screenshot formats select a compatible source")
{
    CHECK(sourceForExtension(".png") == Source::DisplayReferredSdr);
    CHECK(sourceForExtension(".png", true) == Source::DisplayReferredSdr);
    CHECK(sourceForExtension(".exr", false) == Source::SceneLinear);
    CHECK(sourceForExtension(".exr", true) == Source::DisplayReferredHdr);
    CHECK(sourceForExtension(".unknown") == Source::SceneLinear);
}

TEST_CASE("PNG screenshot transfer encodes display-linear RGB as sRGB")
{
    CHECK(encodeSrgb(0.0f) == doctest::Approx(0.0f));
    CHECK(encodeSrgb(0.0031308f) == doctest::Approx(0.0404499f));
    CHECK(encodeSrgb(0.18f) == doctest::Approx(0.461356f));
    CHECK(encodeSrgb(1.0f) == doctest::Approx(1.0f));
    CHECK(encodeSrgb(4.0f) == doctest::Approx(1.0f));
}
