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
    // MetalFX denoising. Off by default: it is a temporal filter and a still
    // frame gives it one frame to work with, so whether it helps is a question
    // to be measured per scene rather than assumed.
    bool denoise = false;
    // Per-stage GPU timestamps, resolved from a counter sample buffer, with a
    // per-bounce breakdown. Costs a sample either side of every dispatch, so it
    // is off unless asked for.
    bool profileStages = false;
    std::string capturePath; // --capture: one steady-state frame to a .gputrace
    // Render below the output resolution and let MetalFX scale up. Off by
    // default because it changes what the image *is*, which an offline render
    // should not do silently; for a realtime budget it is the largest lever
    // there is, since cost is per traced pixel.
    bool upscale = false;
    float upscaleFactor = 0.5f;
    // Metal 4 submission layer. The default: wavefront tracing, both MetalFX
    // scalers and the guide resolve all build and run through it, and it matches
    // Metal 3 to within the renderer's own run-to-run spread. A frame falls back
    // on its own where no Metal 4 path exists -- the denoiser, whose Metal 4
    // constructor asserts inside MPSGraph, and the megakernel.
    uint32_t metal4 = 1;
    // Reorder each bounce's queue by ray origin before traversing it.
    bool sortRays = false;
    // Ray-cone texture level of detail. Off by default -- not because it costs
    // anything, but because it changes the image and buys no time, so turning it
    // on is a decision about filtering rather than about performance. See the
    // note in wavefront.metal for the measurement.
    bool textureLod = false;
    // 0 = spatial scaler, 1 = temporal scaler (ignored when denoise is on).
    uint32_t upscaleMode = 0;
    uint32_t risCandidates = 1; // 1 = plain next-event estimation
    uint32_t estimatorMode = 0; // 0 = NEE + MIS, 1 = BSDF sampling only
    bool sharc = false;
    uint32_t sharcDepth = 1;
    uint32_t sharcMinSamples = 8;
    float sharcBaseSize = 4.0f; // voxel width in pixels
    // 0 = glTF (-ln(C)/d), 1 = Cycles ((1-C)/d)
    uint32_t volumeModel = 0;
    // Longest side a texture is allowed on load; 0 = no limit.
    uint32_t textureMaxDim = 0;
    // Divide every texture's dimensions by this on load; 1 = full size.
    uint32_t textureDownscale = 1;
    // Debug visualisation; 0 renders normally.
    uint32_t debugMode = 0;
    uint32_t blueNoiseSwitchSpp = 16;
    // Upper bound on one indirect path's contribution; 0 = unclamped, which is
    // the default because clamping is a bias the caller has to ask for.
    float clampIndirect = 0.0f;

    int cameraIndex = 0;
    std::optional<glm::vec3> cameraPosition;
    std::optional<glm::vec3> cameraTarget;
    std::optional<float> cameraFov;

    // 0=None, 1=Reinhard, 2=ACES, 3=Filmic
    uint32_t tonemapType = 2;
    /// Whether the caller stated an exposure. A scene can carry its own in the
    /// light sidecar, and it should win over these defaults -- but not over a
    /// value the caller asked for, which is the only way to pin exposure for a
    /// measurement (see tools/feature_tests, which needs exactly 1.0).
    bool exposureOverridden = false;
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
