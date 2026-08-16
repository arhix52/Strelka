#include <doctest/doctest.h>

#include <camera_dump.h>

#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// The camera dump is an input, not a report.
//
// Its whole value is that the block it prints can be pasted into a .toml and
// rendered by StrelkaCLI to the same frame the viewport was showing. Two things
// can quietly destroy that and neither is visible by eye: a key that the CLI
// does not parse, and a coordinate printed to too few digits.
//
// The key names below are checked against HeadlessApp::parseConfig(); if that
// function is renamed a key, this file is the thing that should fail.
// ---------------------------------------------------------------------------

namespace
{

oka::CameraDumpState sample_state()
{
    oka::CameraDumpState s;
    s.scenePath = "/home/ilya/strelka_assets/pine_scene/polyhaven_pine_fir_forest.gltf";
    s.cameraIndex = 0;
    s.width = 1280;
    s.height = 720;
    s.position[0] = 23.1552734f;
    s.position[1] = 0.921875f;
    s.position[2] = 47.5601234f;
    s.target[0] = 22.1552734f;
    s.target[1] = 0.821875f;
    s.target[2] = 46.5601234f;
    s.fov = 39.5977783f;
    s.spp = 256;
    s.maxDepth = 4;
    s.samplerType = 2;
    s.tonemapperType = 2;
    return s;
}

bool contains(const std::string& haystack, const std::string& needle)
{
    return haystack.find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("the dump emits the sections and keys the CLI parses")
{
    const std::string d = oka::formatCameraDump(sample_state());

    CHECK(contains(d, "[scene]"));
    CHECK(contains(d, "path = "));
    CHECK(contains(d, "[output]"));
    CHECK(contains(d, "width = 1280"));
    CHECK(contains(d, "height = 720"));
    CHECK(contains(d, "[render]"));
    CHECK(contains(d, "spp = 256"));
    CHECK(contains(d, "max_depth = 4"));
    CHECK(contains(d, "spp_per_launch = "));
    CHECK(contains(d, "texture_downscale = "));
    CHECK(contains(d, "debug = "));
    CHECK(contains(d, "[camera]"));
    CHECK(contains(d, "index = 0"));
    CHECK(contains(d, "position = ["));
    CHECK(contains(d, "target = ["));
    CHECK(contains(d, "fov = "));
    CHECK(contains(d, "[tonemap]"));
    CHECK(contains(d, "exposure_iso = "));
    CHECK(contains(d, "exposure_fstop = "));
    CHECK(contains(d, "exposure_shutter = "));
}

TEST_CASE("the scene path is what the dump was given")
{
    const std::string d = oka::formatCameraDump(sample_state());
    CHECK(contains(d, "\"/home/ilya/strelka_assets/pine_scene/polyhaven_pine_fir_forest.gltf\""));
}

TEST_CASE("coordinates keep enough digits to land on the same pixel")
{
    // The failure this guards is silent: a position rounded to six digits at
    // forest scale moves the camera by more than a pixel, and the re-render
    // looks like the right frame while being a different one.
    const std::string d = oka::formatCameraDump(sample_state());
    // Tied to the formatter rather than to a literal, so the assertion is
    // "the dump carries this exact float" and not "the dump spells it the way
    // it happened to on the day this was written".
    CHECK(contains(d, oka::cameraDumpFloat(23.1552734f)));
    CHECK(contains(d, oka::cameraDumpFloat(47.5601234f)));
    CHECK(contains(d, oka::cameraDumpFloat(39.5977783f)));
    // And the spelling really is long: these do not survive six digits.
    CHECK(oka::cameraDumpFloat(23.1552734f).size() >= 9);
}

TEST_CASE("a coordinate near zero does not print as a wall of zeroes")
{
    oka::CameraDumpState s = sample_state();
    s.position[0] = 0.000012345f;
    const std::string d = oka::formatCameraDump(s);
    CHECK(contains(d, "1.2345e-05"));
}

TEST_CASE("sampler and tonemapper are spelled the way the CLI parses them")
{
    // HeadlessApp::parseSamplerName's order is not the obvious one -- halton is
    // 0 and sobol is 2 -- so an off-by-one here renders the report with a
    // different sampler than the one that showed the defect.
    CHECK(std::string(oka::cameraDumpSamplerName(0)) == "halton");
    CHECK(std::string(oka::cameraDumpSamplerName(1)) == "pcg");
    CHECK(std::string(oka::cameraDumpSamplerName(2)) == "sobol");
    CHECK(std::string(oka::cameraDumpSamplerName(3)) == "sobol_bn");
    CHECK(std::string(oka::cameraDumpSamplerName(4)) == "hybrid");

    CHECK(std::string(oka::cameraDumpTonemapName(0)) == "none");
    CHECK(std::string(oka::cameraDumpTonemapName(1)) == "reinhard");
    CHECK(std::string(oka::cameraDumpTonemapName(2)) == "aces");
    CHECK(std::string(oka::cameraDumpTonemapName(3)) == "filmic");
}

TEST_CASE("booleans are TOML booleans, not C++ ones")
{
    oka::CameraDumpState s = sample_state();
    s.denoise = true;
    s.upscale = false;
    const std::string d = oka::formatCameraDump(s);
    CHECK(contains(d, "denoise = true"));
    CHECK(contains(d, "upscale = false"));
    CHECK_FALSE(contains(d, "denoise = 1"));
}

TEST_CASE("the pose the CLI cannot carry is printed rather than dropped")
{
    // position/target go through glm::lookAt against world up, so a rolled
    // viewport is not reproduced by them. Saying so in the block is the
    // difference between a known limitation and a mystery.
    oka::CameraDumpState s = sample_state();
    s.orientation[0] = 0.1f;
    s.orientation[3] = 0.99f;
    const std::string d = oka::formatCameraDump(s);
    CHECK(contains(d, "orientation"));
    CHECK(contains(d, "0.99"));
    CHECK(contains(d, "lookAt"));
}

TEST_CASE("an orthographic viewport says the block cannot reproduce it")
{
    // There is no camera.projection key. A perspective config emitted for an
    // orthographic viewport frames something else entirely, and would be read
    // as a renderer bug rather than a dump one.
    oka::CameraDumpState s = sample_state();
    s.orthographic = true;
    s.xmag = 3.5f;
    const std::string d = oka::formatCameraDump(s);
    CHECK(contains(d, "ORTHOGRAPHIC"));
    CHECK(contains(d, "3.5"));
    CHECK(contains(d, "camera.index"));
}

TEST_CASE("depth of field is reported, with where it actually lives")
{
    oka::CameraDumpState s = sample_state();
    s.useDof = true;
    s.focalDistance = 12.25f;
    s.fStopDof = 1.8f;
    const std::string d = oka::formatCameraDump(s);
    CHECK(contains(d, "12.25"));
    CHECK(contains(d, "1.8"));
    CHECK(contains(d, "_camera.json"));

    // And is silent when it is off, so the block stays short enough to read.
    CHECK_FALSE(contains(oka::formatCameraDump(sample_state()), "depth of field"));
}

TEST_CASE("every printed float reads back as the float that went in")
{
    // The property the shortest-round-trip spelling exists for. Shortening is
    // only allowed while it is lossless -- an f-stop may print as "1.8", but a
    // camera position may not lose its ninth digit to the same rule.
    const float values[] = { 1.8f,         23.1552734f, 47.5601234f, 0.921875f, 39.5977783f,
                             0.000012345f, -0.1f,       1e7f,        3.5f,      100.0f,
                             0.0f,         -47.5601234f };
    for (float v : values)
    {
        const std::string s = oka::cameraDumpFloat(v);
        CAPTURE(v);
        CAPTURE(s);
        CHECK(std::strtof(s.c_str(), nullptr) == v);
    }
}

TEST_CASE("a value that is exact at few digits is printed at few digits")
{
    CHECK(oka::cameraDumpFloat(1.8f) == "1.8");
    CHECK(oka::cameraDumpFloat(0.5f) == "0.5");
    CHECK(oka::cameraDumpFloat(100.0f) == "100");
    // Never scientific inside the range camera numbers live in, however few
    // digits would round-trip: "1e+02" is not a position anyone reads.
    CHECK(oka::cameraDumpFloat(1e7f).find('e') == std::string::npos);
    CHECK(oka::cameraDumpFloat(0.0001f).find('e') == std::string::npos);
    // Outside it, scientific is the readable one.
    CHECK(oka::cameraDumpFloat(0.000012345f).find('e') != std::string::npos);
}
