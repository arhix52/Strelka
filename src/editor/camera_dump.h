#ifndef STRELKA_EDITOR_CAMERA_DUMP_H
#define STRELKA_EDITOR_CAMERA_DUMP_H

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
    /// x, y, z, w.
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
    float focalLengthMm = 50.0f;

    uint32_t spp = 0;
    uint32_t sppPerLaunch = 1;
    uint32_t maxDepth = 4;
    uint32_t samplerType = 0;
    uint32_t reconstructionFilter = 0;
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

inline const char* cameraDumpReconstructionFilterName(uint32_t filter)
{
    return filter == 1u ? "mitchell" :
           filter == 2u ? "tent" :
           filter == 3u ? "lanczos2" :
           filter == 4u ? "gaussian" :
           filter == 5u ? "blackman-harris" :
                          "box";
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
    case 4:
        return "agx";
    default:
        return "none";
    }
}

inline std::string cameraDumpFloat(float v)
{
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

inline std::string cameraDumpVec4(const float v[4])
{
    return "[" + cameraDumpFloat(v[0]) + ", " + cameraDumpFloat(v[1]) + ", " + cameraDumpFloat(v[2]) + ", " +
           cameraDumpFloat(v[3]) + "]";
}

/// The viewport as a config file StrelkaCLI can render.
inline std::string formatCameraDump(const CameraDumpState& s)
{
    std::ostringstream os;
    os << "# --- Strelka camera dump ---------------------------------------\n"
       << "# Paste into a .toml and render it with:\n"
       << "#     StrelkaCLI -c <that file>\n";

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
       << "reconstruction_filter = \"" << cameraDumpReconstructionFilterName(s.reconstructionFilter) << "\"\n"
       << "texture_downscale = " << s.textureDownscale << "\n"
       << "denoise = " << (s.denoise ? "true" : "false") << "\n"
       << "upscale = " << (s.upscale ? "true" : "false") << "\n"
       << "debug = " << s.debugView << "\n"
       << "\n[camera]\n"
       << "index = " << s.cameraIndex << "\n"
       << "projection = \"" << (s.orthographic ? "orthographic" : "perspective") << "\"\n"
       << "position = " << cameraDumpVec3(s.position) << "\n"
       << "target = " << cameraDumpVec3(s.target) << "\n"
       << "up = " << cameraDumpVec3(s.up) << "\n"
       << "orientation = " << cameraDumpVec4(s.orientation) << "\n"
       << "fov = " << cameraDumpFloat(s.fov) << "\n"
       << "xmag = " << cameraDumpFloat(s.xmag) << "\n"
       << "ymag = " << cameraDumpFloat(s.ymag) << "\n"
       << "znear = " << cameraDumpFloat(s.znear) << "\n"
       << "zfar = " << cameraDumpFloat(s.zfar) << "\n";
    if (s.useDof)
    {
        os << "focal_distance = " << cameraDumpFloat(s.focalDistance) << "\n"
           << "fstop = " << cameraDumpFloat(s.fStopDof) << "\n"
           << "focal_length_mm = " << cameraDumpFloat(s.focalLengthMm) << "\n";
    }
    os << "\n[tonemap]\n"
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
