#pragma once

#include <strelka/render/render.h>
#include <strelka/scene/scene.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace oka
{

struct RenderConfig
{
    std::string scenePath;

    std::string outputPath = "output.exr";
    uint32_t width = 1024;
    uint32_t height = 768;

    // Reserved for bdpt/vcm when those land; Metal wavefront currently runs PT only.
    // Kept so harness tomls with integrator = "pt"|"bdpt"|"vcm" still parse.
    uint32_t integrator = 0;
    uint32_t spp = 256;
    uint32_t sppPerLaunch = 1;
    uint32_t maxDepth = 8;
    // 0=Halton, 1=PCG, 2=Sobol, 3=Sobol+BN, 4=Hybrid (BN→Sobol)
    uint32_t samplerType = 0;
    // 0 = glTF (-ln(C)/d), 1 = Cycles ((1-C)/d)
    uint32_t volumeModel = 0;
    // Longest side a texture is allowed on load; 0 = no limit.
    uint32_t textureMaxDim = 0;
    uint32_t blueNoiseSwitchSpp = 16;

    int cameraIndex = 0;
    std::optional<glm::vec3> cameraPosition;
    std::optional<glm::vec3> cameraTarget;
    std::optional<float> cameraFov;

    // 0=None, 1=Reinhard, 2=ACES, 3=Filmic
    uint32_t tonemapType = 2;
    float gamma = 2.4f;
    float filmIso = 100.0f;
    float fStop = 4.0f;
    float shutterSpeed = 100.0f;
};

/// Parse named enum values used by both TOML and CLI flags.
/// Throws std::invalid_argument on unknown names.
uint32_t parseSamplerName(const std::string& name);
uint32_t parseTonemapName(const std::string& name);

/// Parse a TOML config file into a RenderConfig, starting from defaults.
RenderConfig parseTomlConfig(const std::string& tomlPath);

class HeadlessApp
{
public:
    explicit HeadlessApp(const RenderConfig& config);
    ~HeadlessApp() = default;

    int run();

private:
    void populateSettings();
    void saveOutput(Buffer* buf);
    void printProgress(uint32_t currentSpp, uint32_t totalSpp, double lastRenderMs);

    RenderConfig m_config;
    std::unique_ptr<SettingsManager> m_settings;
    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<SharedContext> m_sharedCtx;
    std::unique_ptr<Render> m_render;
};

} // namespace oka
