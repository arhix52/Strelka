#include <doctest/doctest.h>

#include "editor_document.h"

#include <filesystem>
#include <fstream>

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

TEST_CASE("selectCameraIndexAfterLoad prefers the camera the scene authored")
{
    // No cameras at all: nothing to select but index 0.
    CHECK(selectCameraIndexAfterLoad(0, 0) == 0);
    // Scene brought none, so only the appended "Main" exists.
    CHECK(selectCameraIndexAfterLoad(0, 1) == 0);
    // Two authored cameras plus Main: open on the first authored one, which is what
    // the renderer draws and what a click has to be traced through.
    CHECK(selectCameraIndexAfterLoad(2, 3) == 0);
}

TEST_CASE("clampCameraIndex stays inside the camera list")
{
    CHECK(clampCameraIndex(0, 0) == 0);
    CHECK(clampCameraIndex(-1, 4) == 0);
    CHECK(clampCameraIndex(99, 4) == 3);
    CHECK(clampCameraIndex(2, 4) == 2);
}

TEST_CASE("pushRecentScene moves a reopened file to the front and caps the list")
{
    std::vector<std::string> recent;
    pushRecentScene(recent, "/tmp/a.glb", 3);
    pushRecentScene(recent, "/tmp/b.glb", 3);
    pushRecentScene(recent, "/tmp/c.glb", 3);
    REQUIRE(recent.size() == 3);
    CHECK(recent[0] == normalizeRecentPath("/tmp/c.glb"));
    CHECK(recent[1] == normalizeRecentPath("/tmp/b.glb"));
    CHECK(recent[2] == normalizeRecentPath("/tmp/a.glb"));

    // Reopening b promotes it and does not grow past the cap.
    pushRecentScene(recent, "/tmp/b.glb", 3);
    REQUIRE(recent.size() == 3);
    CHECK(recent[0] == normalizeRecentPath("/tmp/b.glb"));
    CHECK(recent[1] == normalizeRecentPath("/tmp/c.glb"));
    CHECK(recent[2] == normalizeRecentPath("/tmp/a.glb"));

    pushRecentScene(recent, "/tmp/d.glb", 3);
    REQUIRE(recent.size() == 3);
    CHECK(recent[0] == normalizeRecentPath("/tmp/d.glb"));
    CHECK(recent.back() == normalizeRecentPath("/tmp/c.glb"));
}

TEST_CASE("pushRecentScene ignores an empty document")
{
    std::vector<std::string> recent = { normalizeRecentPath("/tmp/a.glb") };
    pushRecentScene(recent, "");
    REQUIRE(recent.size() == 1);
    CHECK(recent[0] == normalizeRecentPath("/tmp/a.glb"));
}

TEST_CASE("recent scenes round-trip through a text file without reversing order")
{
    const auto dir = std::filesystem::temp_directory_path() / "strelka_recent_test";
    std::filesystem::create_directories(dir);
    const auto file = dir / "recent_scenes.txt";

    std::vector<std::string> written = {
        normalizeRecentPath("/tmp/newest.glb"),
        normalizeRecentPath("/tmp/middle.glb"),
        normalizeRecentPath("/tmp/oldest.glb"),
    };
    REQUIRE(saveRecentScenes(file, written));

    const std::vector<std::string> loaded = loadRecentScenes(file, 10);
    REQUIRE(loaded.size() == 3);
    CHECK(loaded[0] == written[0]);
    CHECK(loaded[1] == written[1]);
    CHECK(loaded[2] == written[2]);

    // Capacity trims the tail, not the head: the oldest entries fall off.
    const std::vector<std::string> trimmed = loadRecentScenes(file, 2);
    REQUIRE(trimmed.size() == 2);
    CHECK(trimmed[0] == written[0]);
    CHECK(trimmed[1] == written[1]);

    std::filesystem::remove_all(dir);
}

TEST_CASE("loadRecentScenes skips blanks and keeps the first of a duplicate")
{
    const auto dir = std::filesystem::temp_directory_path() / "strelka_recent_dup_test";
    std::filesystem::create_directories(dir);
    const auto file = dir / "recent_scenes.txt";
    {
        std::ofstream out(file);
        out << normalizeRecentPath("/tmp/a.glb") << "\n\n";
        out << normalizeRecentPath("/tmp/b.glb") << "\n";
        out << normalizeRecentPath("/tmp/a.glb") << "\n";
    }
    const std::vector<std::string> loaded = loadRecentScenes(file, 10);
    REQUIRE(loaded.size() == 2);
    CHECK(loaded[0] == normalizeRecentPath("/tmp/a.glb"));
    CHECK(loaded[1] == normalizeRecentPath("/tmp/b.glb"));
    std::filesystem::remove_all(dir);
}
