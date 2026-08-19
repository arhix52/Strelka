#include <log.h>
#include <logmanager.h>
#include <cxxopts.hpp>
#include <algorithm>
#include <filesystem>

#include "EditorApp.h"

int main(int argc, const char* argv[])
{
    const oka::Logmanager loggerManager;
    cxxopts::Options options("Strelka -s <Scene path>", "commands");

    // clang-format off
    options.add_options()
        ("s, scene", "scene path", cxxopts::value<std::string>()->default_value(""))
        ("h, help", "Print usage")("t, spp_total", "spp total", cxxopts::value<int32_t>()->default_value("64"))
        ("f, spp_subframe", "spp subframe", cxxopts::value<int32_t>()->default_value("1"))
        ("c, need_screenshot", "Screenshot after spp total", cxxopts::value<bool>()->default_value("false"))
        ("v, validation", "Enable Validation", cxxopts::value<bool>()->default_value("false"));
    // clang-format on
    options.parse_positional({ "s" });
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        STRELKA_INFO("{}", options.help());
        return 0;
    }

    // A scene on the command line is optional. Without one -- or with one that is
    // not there -- the editor comes up on its empty document, which is a state it
    // already supports: File > Open loads into it, and the window is up either way
    // rather than the process exiting before anything is drawn.
    std::string sceneFile(result["s"].as<std::string>());
    if (!sceneFile.empty() && !std::filesystem::exists(sceneFile))
    {
        STRELKA_ERROR("Specified scene file: {} doesn't exist; starting with an empty scene", sceneFile.c_str());
        sceneFile.clear();
    }
    std::string resourceSearchPath;
    if (!sceneFile.empty())
    {
        resourceSearchPath = std::filesystem::path(sceneFile).parent_path().string();
    }
    STRELKA_DEBUG("Resource search path {}", resourceSearchPath);

    oka::EditorApp editor(sceneFile, resourceSearchPath);

    // These were parsed and then dropped on the floor: --need_screenshot in
    // particular advertised a batch capture the editor never performed.
    editor.setBatchCapture(static_cast<uint32_t>(std::max(result["t"].as<int32_t>(), 0)),
                           static_cast<uint32_t>(std::max(result["f"].as<int32_t>(), 0)),
                           result["c"].as<bool>());

    editor.run();

    return 0;
}
