#include <log.h>
#include <logmanager.h>

#include <cxxopts.hpp>
#include <toml++/toml.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "HeadlessApp.h"

int main(int argc, const char* argv[])
{
    // Before anything can touch Metal. The capture layer is inserted when the
    // device is created and reads this then; setting it afterwards leaves
    // startCapture failing with "Capture layer is not inserted".
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--capture" || std::string(argv[i]).rfind("--capture=", 0) == 0)
        {
            // The one write to the environment in the tree, and it happens on the
            // first statement of main(): no other thread exists yet for setenv to
            // race, which is the whole of what the check is about.
            // NOLINTNEXTLINE(concurrency-mt-unsafe)
            setenv("MTL_CAPTURE_ENABLED", "1", 1);
            break;
        }
    }

    const oka::Logmanager loggerManager;

    // clang-format off
    cxxopts::Options options("StrelkaCLI", "Headless path-tracing renderer");
    options.add_options()
        ("c,config",     "TOML config file",               cxxopts::value<std::string>()->default_value(""))
        ("s,scene",      "Scene file (.gltf/.glb)",        cxxopts::value<std::string>()->default_value(""))
        ("o,output",     "Output image path (.exr/.png)",   cxxopts::value<std::string>()->default_value(""))
        ("w,width",      "Render width",                    cxxopts::value<uint32_t>())
        ("height",       "Render height",                   cxxopts::value<uint32_t>())
        ("spp",          "Samples per pixel",               cxxopts::value<uint32_t>())
        ("depth",        "Max ray depth",                   cxxopts::value<uint32_t>())
        ("sss-iterations", "Extra subsurface walk iterations (0..256)",
                                                            cxxopts::value<uint32_t>())
        ("exposure-iso", "Film ISO; overrides the scene's own exposure",
                                                            cxxopts::value<float>())
        ("clamp",        "Clamp each indirect path's contribution (0 = off)",
                                                            cxxopts::value<float>())
        ("sampler",      "Sampler: halton, pcg, sobol, sobol_bn, hybrid", cxxopts::value<std::string>())
        ("bn-switch",    "Hybrid: spp before switching blue-noise -> Sobol", cxxopts::value<uint32_t>())
        ("capture",      "Capture one steady-state frame to a .gputrace for Xcode (as large as the scene on the device)", cxxopts::value<std::string>())
        ("camera",       "Camera index",                    cxxopts::value<int>())
        ("frame-node",    "Frame scene node like editor F",  cxxopts::value<uint32_t>())
        ("frame-instance", "Instance used to disambiguate an instanced node",
                                                            cxxopts::value<uint32_t>())
        ("animation-time", "Normalised animation time in [0,1] for every clip",
                                                            cxxopts::value<float>())
        ("tonemap",      "Tonemap: none, reinhard, aces, filmic", cxxopts::value<std::string>())
        ("h,help",       "Print usage");
    // clang-format on

    options.parse_positional({ "scene" });
    options.positional_help("<scene_path>");

    cxxopts::ParseResult result;
    try
    {
        result = options.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        STRELKA_FATAL("{}", e.what());
        return 1;
    }

    if (result.count("help"))
    {
        std::cout << options.help() << '\n';
        return 0;
    }

    oka::RenderConfig cfg;

    const std::string configPath = result["config"].as<std::string>();
    if (!configPath.empty())
    {
        if (!std::filesystem::exists(configPath))
        {
            STRELKA_FATAL("Config file not found: {}", configPath);
            return 1;
        }
        try
        {
            cfg = oka::parseTomlConfig(configPath);
        }
        catch (const toml::parse_error& e)
        {
            STRELKA_FATAL("TOML parse error: {}", e.what());
            return 1;
        }
    }

    const std::string sceneArg = result["scene"].as<std::string>();
    if (!sceneArg.empty())
    {
        cfg.scenePath = sceneArg;
    }

    const std::string outputArg = result["output"].as<std::string>();
    if (!outputArg.empty())
    {
        cfg.outputPath = outputArg;
    }

    if (result.count("width"))
    {
        cfg.width = result["width"].as<uint32_t>();
    }
    if (result.count("height"))
    {
        cfg.height = result["height"].as<uint32_t>();
    }
    if (result.count("spp"))
    {
        cfg.spp = result["spp"].as<uint32_t>();
    }
    if (result.count("depth"))
    {
        cfg.maxDepth = result["depth"].as<uint32_t>();
    }
    if (result.count("sss-iterations"))
    {
        cfg.subsurfaceIterations = std::min(result["sss-iterations"].as<uint32_t>(), 256u);
    }
    if (result.count("exposure-iso"))
    {
        cfg.filmIso = result["exposure-iso"].as<float>();
        cfg.exposureOverridden = true;
    }
    if (result.count("clamp"))
    {
        cfg.clampIndirect = result["clamp"].as<float>();
    }
    if (result.count("camera"))
    {
        cfg.cameraIndex = result["camera"].as<int>();
    }
    if (result.count("frame-node"))
    {
        cfg.frameNode = result["frame-node"].as<uint32_t>();
    }
    if (result.count("frame-instance"))
    {
        cfg.frameInstance = result["frame-instance"].as<uint32_t>();
    }
    if (result.count("animation-time"))
    {
        cfg.animationTime = std::clamp(result["animation-time"].as<float>(), 0.0f, 1.0f);
    }
    if (result.count("bn-switch"))
    {
        cfg.blueNoiseSwitchSpp = result["bn-switch"].as<uint32_t>();
    }

    try
    {
        if (result.count("sampler"))
        {
            cfg.samplerType = oka::parseSamplerName(result["sampler"].as<std::string>());
        }
        if (result.count("capture"))
        {
            cfg.capturePath = result["capture"].as<std::string>();
        }
        if (result.count("tonemap"))
        {
            cfg.tonemapType = oka::parseTonemapName(result["tonemap"].as<std::string>());
        }
    }
    catch (const std::exception& e)
    {
        STRELKA_FATAL("{}", e.what());
        return 1;
    }

    if (cfg.scenePath.empty())
    {
        STRELKA_FATAL("No scene file specified. Use -s <path> or set scene.path in config.");
        return 1;
    }
    if (!std::filesystem::exists(cfg.scenePath))
    {
        STRELKA_FATAL("Scene file not found: {}", cfg.scenePath);
        return 1;
    }

    STRELKA_INFO("StrelkaCLI: scene={}, output={}, {}x{}, {} spp", cfg.scenePath, cfg.outputPath, cfg.width, cfg.height,
                 cfg.spp);

    oka::HeadlessApp app(cfg);
    return app.run();
}
