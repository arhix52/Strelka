#include <doctest/doctest.h>

#include "editor_document.h"

using namespace oka::editor_document;

TEST_CASE("formatWindowTitle marks dirty and empty documents")
{
    CHECK(formatWindowTitle(false, "", 1.5f, 10) == "Strelka (empty) — [1.5 ms] [10 spp]");
    CHECK(formatWindowTitle(true, "", 2.0f, 1) == "Strelka * (empty) — [2.0 ms] [1 spp]");
    CHECK(formatWindowTitle(true, "/tmp/scenes/cornell_box.glb", 0.5f, 64) ==
          "Strelka * cornell_box.glb — [0.5 ms] [64 spp]");
    CHECK(formatWindowTitle(false, "/tmp/scenes/cornell_box.glb", 12.34f, 0) ==
          "Strelka cornell_box.glb — [12.3 ms] [0 spp]");
}

TEST_CASE("restorePathAfterFailedLoad keeps the previous document path")
{
    CHECK(restorePathAfterFailedLoad("") == "");
    CHECK(restorePathAfterFailedLoad("/a/b.glb") == "/a/b.glb");
}

TEST_CASE("selectMainCameraIndexAfterLoad picks the framed Main camera")
{
    CHECK(selectMainCameraIndexAfterLoad(0) == 0);
    CHECK(selectMainCameraIndexAfterLoad(1) == 0);
    CHECK(selectMainCameraIndexAfterLoad(3) == 2);
}

TEST_CASE("clampCameraIndex stays inside the camera list")
{
    CHECK(clampCameraIndex(0, 0) == 0);
    CHECK(clampCameraIndex(-1, 4) == 0);
    CHECK(clampCameraIndex(99, 4) == 3);
    CHECK(clampCameraIndex(2, 4) == 2);
}
