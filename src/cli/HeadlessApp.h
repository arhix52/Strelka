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
    // Extra wavefront iterations reserved for subsurface random walks. The
    // conservative default preserves the reference image; performance runs can
    // measure a shorter tail explicitly.
    uint32_t subsurfaceIterations = 64;
    // 0=Halton, 1=PCG, 2=Sobol, 3=Sobol+BN, 4=Hybrid (BN→Sobol)
    //
    // Sobol, not the blue-noise variants the editor defaults to: a headless
    // render is a still frame at a few hundred samples, which is past the count
    // where a toroidal shift stops paying for itself.
    //
    // Not Halton, which this used to be. Its dimensions are told apart only by
    // an offset into a table of 32 bases, and a depth-8 path draws 117, so
    // dimensions 32 apart are one sequence read from two places. Measured on
    // vespa at 320x240: its post-filter error is 38% above Sobol's at 128 spp
    // -- roughly twice the samples for the same picture -- and worse than the
    // plain PCG white noise at every count, with a convergence slope that
    // stalls near zero and then jumps as the correlated dimensions come apart.
    uint32_t samplerType = 2;
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
    // Reorder each bounce's queue by ray origin before traversing it.
    bool sortRays = false;
    // Ray-cone texture level of detail. Off by default -- not because it costs
    // anything, but because it changes the image and buys no time, so turning it
    // on is a decision about filtering rather than about performance. See the
    // note in wavefront.metal for the measurement.
    bool textureLod = false;
    // Take the denoiser's material guides at the primary hit instead of walking
    // to the first rough surface. See Uniforms::guidePrimaryHit.
    bool guidePrimaryHit = false;
    /// Luminance ceiling on what the denoiser is handed, in exposed units; 0
    /// disables it. Configurable because it is a truncation, and a scene bright
    /// enough to be clipped by it measures the clamp rather than the denoiser --
    /// which is what a mirror facing a light does. See docs/open-defects.md
    /// entry 7.
    float denoiseFireflyClamp = 8.0f;
    // 0 = spatial scaler, 1 = temporal scaler (ignored when denoise is on).
    uint32_t upscaleMode = 0;
    uint32_t risCandidates = 1; // 1 = plain next-event estimation
    uint32_t estimatorMode = 0; // 0 = NEE + MIS, 1 = BSDF sampling only
    /// Accumulate the diffuse/specular split of the first event into two extra
    /// images. Off because nothing reads them: the raygen wrote four scattered
    /// records per pixel per launch for an output no caller ever asked the
    /// backend for. See docs/open-perf.md.
    bool splitAov = false;
    bool sharc = false;
    uint32_t sharcCapacity = 1u << 22;
    uint32_t sharcDepth = 1;
    uint32_t sharcMinSamples = 8;
    /// Accumulated samples after which cache reads stop; 0 never stops. The
    /// cache's error is correlated and therefore a floor, so past the crossover
    /// it is the only thing keeping the render from converging. See
    /// Params::sharcReadMaxSubframe for the measurement.
    uint32_t sharcReadFrames = 128;
    /// Sample count a Metal voxel needs before a query may read it. The legacy
    /// `sharc_min_samples` sets both backends; this overrides it for Metal,
    /// where 1 is the upstream `> 0` threshold.
    uint32_t sharcMetalMinSamples = 1;
    float sharcBaseSize = 4.0f; // voxel width in pixels
    /// Frames the cache averages a voxel over. Larger is quieter and slower to
    /// notice that the lighting changed; the resolve pass clamps it to the SDK's
    /// bounds. See src/shaders/optix/sharc_resolve.h.
    uint32_t sharcAccumFrames = 32;
    /// Frames an entry survives with nothing deposited into it before its slot
    /// goes back to the table. This is what lets the cache outlive a moving
    /// camera; evicting too eagerly costs more in re-insertion than it frees.
    uint32_t sharcStaleFrames = 64;
    /// The window and lifetime, in frames, of the short-clock entries that hold
    /// what lights marked `responsive` in the scene deliver. Short on purpose.
    uint32_t sharcResponsiveFrames = 4;
    /// Lets a headless A/B turn the split off on a scene that has responsive
    /// lights, which is the only way to measure what it costs and buys.
    bool sharcResponsiveLighting = true;
    /// Metal's own responsive switch, and not the same setting: its compact key
    /// has no spare bit for a per-light tag, so the companion entries hold the
    /// whole lighting signal and come out of the configured capacity. Off by
    /// default for that reason. See docs/sharc-metal.md.
    bool sharcMetalResponsive = false;
    /// World-space voxel scale: larger values make smaller voxels. Metal only;
    /// OptiX sizes its grid from sharcBaseSize in pixels.
    float sharcSceneScale = 50.0f;
    float sharcRoughnessThreshold = 0.4f;
    /// Quantization factor for the atomic radiance accumulator. Reduce it if the
    /// diagnostics report 31-32 occupied radiance bits.
    float sharcRadianceScale = 1000.0f;
    /// One update path per this many pixels, squared. 5 traces about 4%.
    uint32_t sharcUpdateDownscale = 5;
    /// Vertices an update path keeps behind it for back-propagation.
    uint32_t sharcPropagationDepth = 2;
    uint32_t sharcDebug = 0;
    int32_t sharcLevelBias = 0;
    bool sharcMaterialDemodulation = true;
    bool sharcSeparateEmissive = true;
    bool sharcDirectional = false;
    bool sharcCacheResampling = true;
    bool sharcBlendAdjacentLevels = true;
    bool sharcFadeAcceleration = false;
    /// Resolve alpha cutouts in the traversal hardware where the answer is
    /// uniform, and enter the shader only where it is not. Off by default: it is
    /// an acceleration, and one that has to be measured on a machine that can
    /// build it before it becomes anybody's default.
    bool opacityMicromaps = false;
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
    // Frame this scene node exactly like the editor's F command. The optional
    // instance disambiguates EXT_mesh_gpu_instancing while keeping all sibling
    // primitives at that placement in the bounds.
    std::optional<uint32_t> frameNode;
    std::optional<uint32_t> frameInstance;
    std::optional<glm::vec3> cameraPosition;
    std::optional<glm::vec3> cameraTarget;
    std::optional<float> cameraFov;
    // Normalised time in every clip: 0 = start, 1 = end. Unset leaves each
    // animation at its start. A value other than the loader's initial current
    // is what makes the first frame dirty the skeleton and run skinning — a
    // still at t=start never does, which is why the validation scene pins 0.5.
    std::optional<float> animationTime;

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

/// StrelkaCLI returns this when the renderer could not start. GitHub-hosted
/// macOS runners have no GPU passthrough, so the workflow treats this as a skip
/// rather than a failed render.
inline constexpr int kExitRendererUnavailable = 3;

class HeadlessApp
{
public:
    explicit HeadlessApp(RenderConfig config);
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
