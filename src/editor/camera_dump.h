#ifndef STRELKA_EDITOR_CAMERA_DUMP_H
#define STRELKA_EDITOR_CAMERA_DUMP_H

// ============================================================================
// camera_dump.h -- the viewport's state, printed as something that renders it.
//
// A defect that only shows from one angle is a defect nobody else can look at.
// Describing the angle in prose does not survive the trip: "the rock, from
// slightly above" is not a camera, and a screenshot cannot be re-rendered with
// the debug view changed or the sample count raised.
//
// So the dump is not a report, it is an input. Every key it emits is one
// HeadlessApp::parseConfig() already reads, in the section it reads it from, so
// the block can be pasted into a .toml and handed to StrelkaCLI -c unchanged.
// What the CLI cannot express rides along as comments rather than being
// silently dropped -- the orientation quaternion above all, because the CLI
// rebuilds the view with glm::lookAt against world up, which is the same camera
// only while the viewport has no roll on it.
//
// Deliberately free of ImGui, the settings map and the scene: values in, string
// out, so tests/editor/test_camera_dump.cpp can hold it to the one property
// that matters -- that what comes out is the camera that went in, to enough
// digits to land on the same pixel.
// ============================================================================

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

namespace oka
{

/// Everything the dump prints. Plain scalars so the formatter can be tested
/// without a scene, a GPU or a settings map.
struct CameraDumpState
{
    std::string scenePath;
    int cameraIndex = 0;

    uint32_t width = 0;
    uint32_t height = 0;

    float position[3] = { 0.0f, 0.0f, 0.0f };
    /// position + forward, i.e. what the CLI's `target` means.
    float target[3] = { 0.0f, 0.0f, -1.0f };
    float up[3] = { 0.0f, 1.0f, 0.0f };
    /// x, y, z, w -- the pose the CLI cannot take, printed so the roll is not
    /// lost silently.
    float orientation[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

    bool orthographic = false;
    float fov = 45.0f;
    float xmag = 1.0f;
    float ymag = 1.0f;
    float znear = 0.1f;
    float zfar = 1000.0f;

    bool useDof = false;
    float focalDistance = 10.0f;
    float fStopDof = 2.8f;

    uint32_t spp = 0;
    uint32_t sppPerLaunch = 1;
    uint32_t maxDepth = 4;
    uint32_t samplerType = 0;
    uint32_t debugView = 0;
    bool denoise = false;
    bool upscale = false;
    uint32_t textureDownscale = 1;

    uint32_t tonemapperType = 0;
    float gamma = 2.4f;
    float filmIso = 100.0f;
    float fStop = 4.0f;
    float shutterSpeed = 100.0f;
};

/// `render.sampler` takes a name, not the enum the editor carries.
///
/// The order is HeadlessApp::parseSamplerName()'s, and it is not the obvious
/// one -- halton is 0, sobol is 2 -- so this is written against that function
/// and asserted against its spelling in the test.
inline const char* cameraDumpSamplerName(uint32_t samplerType)
{
    switch (samplerType)
    {
    case 0:
        return "halton";
    case 1:
        return "pcg";
    case 2:
        return "sobol";
    case 3:
        return "sobol_bn";
    case 4:
        return "hybrid";
    default:
        return "sobol";
    }
}

/// `tonemap.type` likewise.
inline const char* cameraDumpTonemapName(uint32_t tonemapperType)
{
    switch (tonemapperType)
    {
    case 0:
        return "none";
    case 1:
        return "reinhard";
    case 2:
        return "aces";
    case 3:
        return "filmic";
    default:
        return "none";
    }
}

/// The shortest spelling of `v` that reads back as exactly `v`.
///
/// Both halves matter. Nine significant digits always round-trip a float, and
/// printing nine of them unconditionally is what the whole point of the dump
/// needs -- a position rounded to six digits at forest scale moves the camera
/// by more than a pixel, and the re-render then looks like the right frame
/// while being a different one. But it also spells an f-stop of 1.8 as
/// "1.79999995", and a block full of that is one a reader stops trusting.
///
/// So: the fewest digits that still survive the trip back through strtof.
/// std::to_chars would say the same thing in one call, but its floating-point
/// overloads are the ones libc++ was last to ship, and only the macOS build is
/// in CI to notice.
///
/// `defaultfloat` rather than `fixed`, so a coordinate near zero comes out as
/// an exponent rather than a screenful of zeroes.
inline std::string cameraDumpFloat(float v)
{
    // `defaultfloat` goes scientific as soon as the exponent reaches the
    // precision, so the shortest round-trip of 100 is "1e+02" -- true, and not
    // what anyone wants to read in a camera position. Inside the range these
    // numbers live in, keep raising the precision until the plain spelling
    // appears; outside it (a coordinate at 1e-5) scientific is the readable one.
    const float magnitude = (v < 0.0f) ? -v : v;
    const bool plainRange = (v == 0.0f) || (magnitude >= 1e-4f && magnitude < 1e9f);
    for (int precision = 1; precision < 9; ++precision)
    {
        std::ostringstream os;
        os << std::defaultfloat << std::setprecision(precision) << v;
        const std::string candidate = os.str();
        if (plainRange && candidate.find('e') != std::string::npos)
        {
            continue;
        }
        if (std::strtof(candidate.c_str(), nullptr) == v)
        {
            return candidate;
        }
    }
    std::ostringstream os;
    os << std::defaultfloat << std::setprecision(9) << v;
    return os.str();
}

inline std::string cameraDumpVec3(const float v[3])
{
    return "[" + cameraDumpFloat(v[0]) + ", " + cameraDumpFloat(v[1]) + ", " + cameraDumpFloat(v[2]) + "]";
}

/// The viewport as a config file StrelkaCLI can render.
inline std::string formatCameraDump(const CameraDumpState& s)
{
    std::ostringstream os;
    os << "# --- Strelka camera dump ---------------------------------------\n"
       << "# Paste into a .toml and render it with:\n"
       << "#     StrelkaCLI -c <that file>\n";

    if (s.orthographic)
    {
        // There is no `camera.projection` key, so an orthographic viewport
        // cannot be reproduced by the block below at all. Saying so beats
        // emitting a perspective config that quietly frames something else.
        os << "# NOTE: this camera is ORTHOGRAPHIC (xmag " << cameraDumpFloat(s.xmag) << ", ymag "
           << cameraDumpFloat(s.ymag) << ").\n"
           << "#       The CLI has no key for that -- render it through camera.index instead.\n";
    }
    os << "# orientation (x, y, z, w) = " << cameraDumpFloat(s.orientation[0]) << ", "
       << cameraDumpFloat(s.orientation[1]) << ", " << cameraDumpFloat(s.orientation[2]) << ", "
       << cameraDumpFloat(s.orientation[3]) << "\n"
       << "#       up = " << cameraDumpVec3(s.up) << "\n"
       << "#       The CLI rebuilds the view with lookAt against world up, so any roll\n"
       << "#       in the viewport is not carried by position/target alone.\n"
       << "# znear/zfar = " << cameraDumpFloat(s.znear) << " / " << cameraDumpFloat(s.zfar) << "\n";
    if (s.useDof)
    {
        os << "# depth of field is ON: focal distance " << cameraDumpFloat(s.focalDistance) << ", f/"
           << cameraDumpFloat(s.fStopDof) << " -- carried by <stem>_camera.json, not by this block.\n";
    }

    os << "\n[scene]\npath = \"" << s.scenePath << "\"\n"
       << "\n[output]\npath = \"dump.exr\"\n"
       << "width = " << s.width << "\n"
       << "height = " << s.height << "\n"
       << "\n[render]\n"
       << "integrator = \"pt\"\n"
       << "spp = " << s.spp << "\n"
       << "spp_per_launch = " << s.sppPerLaunch << "\n"
       << "max_depth = " << s.maxDepth << "\n"
       << "sampler = \"" << cameraDumpSamplerName(s.samplerType) << "\"\n"
       << "texture_downscale = " << s.textureDownscale << "\n"
       << "denoise = " << (s.denoise ? "true" : "false") << "\n"
       << "upscale = " << (s.upscale ? "true" : "false") << "\n"
       << "debug = " << s.debugView << "\n"
       << "\n[camera]\n"
       << "index = " << s.cameraIndex << "\n"
       << "position = " << cameraDumpVec3(s.position) << "\n"
       << "target = " << cameraDumpVec3(s.target) << "\n"
       << "fov = " << cameraDumpFloat(s.fov) << "\n"
       << "\n[tonemap]\n"
       << "type = \"" << cameraDumpTonemapName(s.tonemapperType) << "\"\n"
       << "gamma = " << cameraDumpFloat(s.gamma) << "\n"
       << "exposure_iso = " << cameraDumpFloat(s.filmIso) << "\n"
       << "exposure_fstop = " << cameraDumpFloat(s.fStop) << "\n"
       << "exposure_shutter = " << cameraDumpFloat(s.shutterSpeed) << "\n"
       << "# --- end camera dump -------------------------------------------\n";
    return os.str();
}

} // namespace oka

#endif // STRELKA_EDITOR_CAMERA_DUMP_H
