#include <doctest/doctest.h>

#include <camera_dump.h>

#include <cstdlib>
#include <string>

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
    CHECK(contains(d, "projection = \"perspective\""));
    CHECK(contains(d, "position = ["));
    CHECK(contains(d, "target = ["));
    CHECK(contains(d, "up = ["));
    CHECK(contains(d, "orientation = ["));
    CHECK(contains(d, "fov = "));
    CHECK(contains(d, "xmag = "));
    CHECK(contains(d, "ymag = "));
    CHECK(contains(d, "znear = "));
    CHECK(contains(d, "zfar = "));
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
    CHECK(std::string(oka::cameraDumpReconstructionFilterName(0)) == "box");
    CHECK(std::string(oka::cameraDumpReconstructionFilterName(1)) == "mitchell");
    CHECK(std::string(oka::cameraDumpReconstructionFilterName(2)) == "tent");
    CHECK(std::string(oka::cameraDumpReconstructionFilterName(3)) == "lanczos2");

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

TEST_CASE("the exact rolled pose is emitted as CLI input")
{
    oka::CameraDumpState s = sample_state();
    s.orientation[0] = 0.1f;
    s.orientation[3] = 0.99f;
    const std::string d = oka::formatCameraDump(s);
    CHECK(contains(d, "orientation = [0.1, 0, 0, 0.99]"));
    CHECK_FALSE(contains(d, "lookAt"));
}

TEST_CASE("an orthographic viewport emits its live film extent")
{
    oka::CameraDumpState s = sample_state();
    s.orthographic = true;
    s.xmag = 3.5f;
    s.ymag = 1.25f;
    const std::string d = oka::formatCameraDump(s);
    CHECK(contains(d, "projection = \"orthographic\""));
    CHECK(contains(d, "xmag = 3.5"));
    CHECK(contains(d, "ymag = 1.25"));
}

TEST_CASE("depth of field is emitted as CLI input only when enabled")
{
    oka::CameraDumpState s = sample_state();
    s.useDof = true;
    s.focalDistance = 12.25f;
    s.fStopDof = 1.8f;
    s.focalLengthMm = 42.5f;
    const std::string d = oka::formatCameraDump(s);
    CHECK(contains(d, "focal_distance = 12.25"));
    CHECK(contains(d, "fstop = 1.8"));
    CHECK(contains(d, "focal_length_mm = 42.5"));

    CHECK_FALSE(contains(oka::formatCameraDump(sample_state()), "focal_distance"));
}

TEST_CASE("every printed float reads back as the float that went in")
{
    // The property the shortest-round-trip spelling exists for. Shortening is
    // only allowed while it is lossless -- an f-stop may print as "1.8", but a
    // camera position may not lose its ninth digit to the same rule.
    const float values[] = { 1.8f,  23.1552734f, 47.5601234f, 0.921875f, 39.5977783f, 0.000012345f,
                             -0.1f, 1e7f,        3.5f,        100.0f,    0.0f,        -47.5601234f };
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
