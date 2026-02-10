#pragma once

#include <strelka/render/render.h>
#include <strelka/scene/scene.h>
#include <strelka/sceneloader/gltfloader.h>

#include <glm/glm.hpp>

#include <memory>
#include <optional>
#include <string>
#include <cstdint>

namespace oka
{

struct RenderConfig
{
    // scene
    std::string scenePath;

    // output
    std::string outputPath = "output.exr";
    uint32_t width = 1024;
    uint32_t height = 768;

    // render
    uint32_t integrator = 0;   // 0=PT, 1=BDPT, 2=VCM
    uint32_t spp = 256;
    uint32_t sppPerLaunch = 1;
    uint32_t maxDepth = 8;
    uint32_t samplerType = 0;  // 0=Halton, 1=PCG, 2=Sobol

    // camera
    int cameraIndex = 0;
    std::optional<glm::vec3> cameraPosition;
    std::optional<glm::vec3> cameraTarget;
    std::optional<float> cameraFov;

    // tonemap
    uint32_t tonemapType = 2;  // 0=None, 1=Reinhard, 2=ACES, 3=Filmic
    float gamma = 2.4f;
    float filmIso = 100.0f;
    float fStop = 4.0f;
    float shutterSpeed = 100.0f;
};

/// Parse a TOML config file into a RenderConfig, starting from defaults.
RenderConfig parseTomlConfig(const std::string& tomlPath);

class HeadlessApp
{
public:
    explicit HeadlessApp(const RenderConfig& config);
    ~HeadlessApp() = default;

    int run(); // returns 0 on success

private:
    void populateSettings();
    void saveOutput(Buffer* buf);
    void printProgress(uint32_t currentSpp, uint32_t totalSpp, double lastRenderMs);

    RenderConfig m_config;
    std::unique_ptr<SettingsManager> m_settings;
    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<SharedContext> m_sharedCtx;
    std::unique_ptr<Render> m_render;
    std::unique_ptr<GltfLoader> m_loader;
};

} // namespace oka
