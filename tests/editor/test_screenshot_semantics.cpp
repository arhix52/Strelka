#include <doctest/doctest.h>

#include "editor_screenshot.h"

using oka::editor_screenshot::Source;
using oka::editor_screenshot::sourceForExtension;

TEST_CASE("preview screenshots choose the source that matches their format")
{
    CHECK(sourceForExtension(".exr") == Source::LinearPreview);
    CHECK(sourceForExtension(".png") == Source::DisplayPreview);
    CHECK(sourceForExtension(".unknown") == Source::LinearPreview);
}
