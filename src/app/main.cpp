#include <display/Display.h>

#include <render/common.h>
#include <render/buffer.h>
#include <render/render.h>

#include <sceneloader/gltfloader.h>

#include <log.h>
#include <logmanager.h>
#include <cxxopts.hpp>
#include <filesystem>
#include <memory>

#include <Editor.h>

#include <Params.h>

int main(int argc, const char* argv[])
{
    const oka::Logmanager loggerManager;
    cxxopts::Options options("Strelka -s <Scene path>", "commands");

    // clang-format off
    options.add_options()
        ("s, scene", "scene path", cxxopts::value<std::string>()->default_value(""))
        ("i, iteration", "Iteration to capture", cxxopts::value<int32_t>()->default_value("-1"))
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

    // check params
    const std::string sceneFile(result["s"].as<std::string>());
    if (sceneFile.empty())
    {
        STRELKA_FATAL("Specify scene file name");
        return 1;
    }
    if (!std::filesystem::exists(sceneFile))
    {
        STRELKA_FATAL("Specified scene file: {} doesn't exist", sceneFile.c_str());
        return -1;
    }
    const std::filesystem::path sceneFilePath = { sceneFile.c_str() };
    const std::string resourceSearchPath = sceneFilePath.parent_path().string();
    STRELKA_DEBUG("Resource search path {}", resourceSearchPath);

    Params::sceneFile = sceneFile;
    Params::resourceSearchPath = resourceSearchPath;

    oka::Editor editor;

    editor.run();

    return 0;
}
