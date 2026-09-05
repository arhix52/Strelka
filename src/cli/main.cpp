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
        if (std::string(argv[i]) == "--capture" || std::string(argv[i]).starts_with("--capture="))
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
        ("restir-di",    "Enable ReSTIR DI",                 cxxopts::value<bool>()->implicit_value("true"))
        ("restir-candidates", "ReSTIR initial candidates",   cxxopts::value<uint32_t>())
        ("restir-temporal", "Enable ReSTIR temporal reuse",  cxxopts::value<bool>()->implicit_value("true"))
        ("restir-spatial", "Enable ReSTIR spatial reuse",    cxxopts::value<bool>()->implicit_value("true"))
        ("restir-neighbors", "ReSTIR spatial neighbors",     cxxopts::value<uint32_t>())
        ("restir-max-age", "ReSTIR reservoir maximum age",   cxxopts::value<uint32_t>())
        ("restir-debug", "ReSTIR debug mode",                cxxopts::value<uint32_t>())
        ("restir-bias-correction", "ReSTIR bias correction: off, basic, raytraced-diagnostic",
         cxxopts::value<std::string>())
        ("restir-initial-visibility", "Diagnostic initial visibility: off, selected, candidates",
         cxxopts::value<std::string>())
        ("profile-stages", "Report per-stage GPU timings",   cxxopts::value<bool>()->implicit_value("true"))
        ("audit-render-work", "Print debug render-work counters as JSON", cxxopts::value<bool>()->implicit_value("true"))
        ("audit-frames", "Audited frames for moving-light harness", cxxopts::value<uint32_t>())
        ("audit-moving-lights", "Add moving analytic lights for render-work audit", cxxopts::value<uint32_t>())
        ("audit-motion-sequence", "Moving-light sequence: 0 smooth, 1 camera/abrupt, 2 add/delete, 3 local-many",
         cxxopts::value<uint32_t>())
        ("audit-freeze", "Refine final moving-light frame to --spp", cxxopts::value<bool>()->implicit_value("true"))
        ("audit-moving-node", "Move one emissive scene node", cxxopts::value<uint32_t>())
        ("audit-frame-prefix", "Write each audited motion frame as PREFIX-NN.exr", cxxopts::value<std::string>())
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
        catch (const std::exception& e)
        {
            STRELKA_FATAL("Config error: {}", e.what());
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
    if (result.count("restir-di"))
    {
        cfg.restirDIEnabled = result["restir-di"].as<bool>();
    }
    if (result.count("restir-candidates"))
    {
        cfg.initialCandidateCount = std::clamp(result["restir-candidates"].as<uint32_t>(), 1u, 64u);
    }
    if (result.count("restir-temporal"))
    {
        cfg.temporalReuseEnabled = result["restir-temporal"].as<bool>();
    }
    if (result.count("restir-spatial"))
    {
        cfg.spatialReuseEnabled = result["restir-spatial"].as<bool>();
    }
    if (result.count("restir-neighbors"))
    {
        cfg.spatialNeighborCount = std::min(result["restir-neighbors"].as<uint32_t>(), 16u);
    }
    if (result.count("restir-max-age"))
    {
        cfg.reservoirMaxAge = std::min(result["restir-max-age"].as<uint32_t>(), 255u);
    }
    if (result.count("restir-debug"))
    {
        cfg.restirDebugMode = std::min(result["restir-debug"].as<uint32_t>(), 2u);
    }
    if (result.count("restir-bias-correction"))
    {
        const std::string mode = result["restir-bias-correction"].as<std::string>();
        if (mode != "off" && mode != "basic" && mode != "raytraced-diagnostic")
        {
            STRELKA_FATAL("--restir-bias-correction must be off, basic, or raytraced-diagnostic");
            return 1;
        }
        cfg.restirBiasCorrection = mode == "raytraced-diagnostic" ? 2u : mode == "basic" ? 1u : 0u;
    }
    if (result.count("restir-initial-visibility"))
    {
        const std::string mode = result["restir-initial-visibility"].as<std::string>();
        if (mode != "off" && mode != "selected" && mode != "candidates")
        {
            STRELKA_FATAL("--restir-initial-visibility must be off, selected, or candidates");
            return 1;
        }
        cfg.restirInitialVisibility = mode == "candidates" ? 2u : mode == "selected" ? 1u : 0u;
    }
    if (result.count("profile-stages"))
    {
        cfg.profileStages = result["profile-stages"].as<bool>();
    }
    if (result.count("audit-render-work"))
    {
        cfg.auditRenderWork = result["audit-render-work"].as<bool>();
    }
    if (result.count("audit-frames"))
        cfg.auditFrames = std::min(result["audit-frames"].as<uint32_t>(), 1024u);
    if (result.count("audit-moving-lights"))
        cfg.auditMovingLights = std::min(result["audit-moving-lights"].as<uint32_t>(), 4096u);
    if (result.count("audit-motion-sequence"))
        cfg.auditMotionSequence = std::min(result["audit-motion-sequence"].as<uint32_t>(), 3u);
    if (result.count("audit-freeze"))
        cfg.auditFreeze = result["audit-freeze"].as<bool>();
    if (result.count("audit-moving-node"))
        cfg.auditMovingNode = result["audit-moving-node"].as<uint32_t>();
    if (result.count("audit-frame-prefix"))
        cfg.auditFramePrefix = result["audit-frame-prefix"].as<std::string>();
#ifdef NDEBUG
    if (cfg.auditRenderWork)
    {
        STRELKA_FATAL("--audit-render-work is available only in Debug builds");
        return 1;
    }
#endif
    if (cfg.auditFrames != 0u && cfg.auditMovingLights == 0u && !cfg.auditMovingNode && !cfg.auditRenderWork)
    {
        STRELKA_FATAL("--audit-frames without moving lights requires --audit-render-work");
        return 1;
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
