#include <log.h>
#include <logmanager.h>
#include <cxxopts.hpp>
#include <toml++/toml.hpp>
#include <filesystem>

#include "HeadlessApp.h"

static uint32_t parseIntegratorArg(const std::string& s)
{
    if (s == "pt")   return 0;
    if (s == "bdpt") return 1;
    if (s == "vcm")  return 2;
    throw std::runtime_error("Unknown integrator: " + s);
}

static uint32_t parseSamplerArg(const std::string& s)
{
    if (s == "halton") return 0;
    if (s == "pcg")    return 1;
    if (s == "sobol")  return 2;
    throw std::runtime_error("Unknown sampler: " + s);
}

static uint32_t parseTonemapArg(const std::string& s)
{
    if (s == "none")     return 0;
    if (s == "reinhard") return 1;
    if (s == "aces")     return 2;
    if (s == "filmic")   return 3;
    throw std::runtime_error("Unknown tonemap: " + s);
}

int main(int argc, const char* argv[])
{
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
        ("integrator",   "Integrator: pt, bdpt, vcm",      cxxopts::value<std::string>())
        ("depth",        "Max ray depth",                   cxxopts::value<uint32_t>())
        ("sampler",      "Sampler: halton, pcg, sobol",     cxxopts::value<std::string>())
        ("camera",       "Camera index",                    cxxopts::value<int>())
        ("tonemap",      "Tonemap: none, reinhard, aces, filmic", cxxopts::value<std::string>())
        ("h,help",       "Print usage");
    // clang-format on

    options.parse_positional({"scene"});
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
        fprintf(stdout, "%s\n", options.help().c_str());
        return 0;
    }

    // Start with defaults, optionally load TOML, then apply CLI overrides
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

    // CLI overrides
    const std::string sceneArg = result["scene"].as<std::string>();
    if (!sceneArg.empty())
        cfg.scenePath = sceneArg;

    const std::string outputArg = result["output"].as<std::string>();
    if (!outputArg.empty())
        cfg.outputPath = outputArg;

    if (result.count("width"))
        cfg.width = result["width"].as<uint32_t>();
    if (result.count("height"))
        cfg.height = result["height"].as<uint32_t>();
    if (result.count("spp"))
        cfg.spp = result["spp"].as<uint32_t>();
    if (result.count("depth"))
        cfg.maxDepth = result["depth"].as<uint32_t>();
    if (result.count("camera"))
        cfg.cameraIndex = result["camera"].as<int>();

    try
    {
        if (result.count("integrator"))
            cfg.integrator = parseIntegratorArg(result["integrator"].as<std::string>());
        if (result.count("sampler"))
            cfg.samplerType = parseSamplerArg(result["sampler"].as<std::string>());
        if (result.count("tonemap"))
            cfg.tonemapType = parseTonemapArg(result["tonemap"].as<std::string>());
    }
    catch (const std::exception& e)
    {
        STRELKA_FATAL("{}", e.what());
        return 1;
    }

    // Validate
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

    STRELKA_INFO("StrelkaCLI: scene={}, output={}, {}x{}, {} spp",
                 cfg.scenePath, cfg.outputPath, cfg.width, cfg.height, cfg.spp);

    oka::HeadlessApp app(cfg);
    return app.run();
}
