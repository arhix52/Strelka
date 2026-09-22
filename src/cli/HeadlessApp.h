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
    uint32_t sppPerLaunch = 16;
    // Periodically publish the accumulated image through an atomic replacement
    // of <stem>.checkpoint.<ext>. Boundaries are observed between launches so
    // checkpointing does not break the requested render batch size.
    uint32_t checkpointSpp = 0;
    uint32_t maxDepth = 8;
    // Extra wavefront iterations reserved for subsurface random walks. The
    // conservative default preserves the reference image; performance runs can
    // measure a shorter tail explicitly.
    uint32_t subsurfaceIterations = 64;
    uint32_t samplerType = 4;
    // 0=box, 1=Mitchell, 2=tent, 3=Lanczos 2, 4=Gaussian, 5=Blackman-Harris.
    uint32_t reconstructionFilter = 0;
    // MetalFX denoising. Off by default: it is a temporal filter and a still
    // frame gives it one frame to work with, so whether it helps is a question
    // to be measured per scene rather than assumed.
    bool denoise = false;
    // Per-stage GPU timestamps, resolved from a counter sample buffer, with a
    // per-bounce breakdown. Costs a sample either side of every dispatch, so it
    // is off unless asked for.
    bool profileStages = false;
    uint32_t animationFrames = 0;
    float animationFps = 60.0f;
    bool auditRenderWork = false;
    uint32_t auditFrames = 0;
    uint32_t auditMovingLights = 0;
    uint32_t auditMotionSequence = 0;
    bool auditFreeze = false;
    std::optional<uint32_t> auditMovingNode;
    std::string auditFramePrefix;
    std::string capturePath; // --capture: one steady-state frame to a .gputrace
    bool upscale = false;
    float upscaleFactor = 0.5f;
    bool textureLod = true;
    float textureLodBias = 0.0f;
    // Take the denoiser's material guides at the primary hit instead of walking
    // to the first rough surface. See Uniforms::guidePrimaryHit.
    bool guidePrimaryHit = false;
    float denoiseFireflyClamp = 8.0f;
    // 0 = spatial scaler, 1 = temporal scaler (ignored when denoise is on).
    uint32_t upscaleMode = 0;
    uint32_t risCandidates = 1; // 1 = plain next-event estimation
    bool restirDIEnabled = false;
    uint32_t initialCandidateCount = 2;
    bool temporalReuseEnabled = true;
    bool spatialReuseEnabled = true;
    uint32_t spatialNeighborCount = 2;
    uint32_t reservoirMaxAge = 20;
    uint32_t restirDebugMode = 0;
    uint32_t restirBiasCorrection = 0; // 0 = off, 1 = basic, 2 = ray-traced diagnostic
    uint32_t restirInitialVisibility = 0; // 0 = off, 1 = selected initial sample
    uint32_t restirFinalVisibilityReuse = 0; // 0 = off, 1 = conservative
    uint32_t restirFinalVisibilityMaxAge = 4;
    uint32_t estimatorMode = 0; // 0 = NEE + MIS, 1 = BSDF sampling only
    bool splitAov = false;
    bool sharc = false;
    uint32_t sharcCapacity = 1u << 22;
    uint32_t sharcDepth = 1;
    uint32_t sharcMinSamples = 8;
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
    bool sharcMetalResponsive = false;
    float sharcRoughnessThreshold = 0.4f;
    /// Quantization factor for the atomic radiance accumulator. Reduce it if the
    /// diagnostics report 31-32 occupied radiance bits.
    float sharcRadianceScale = 1000.0f;
    /// One update path per this many pixels, squared. 5 traces about 4%.
    uint32_t sharcUpdateDownscale = 5;
    /// Vertices an update path keeps behind it for back-propagation.
    uint32_t sharcPropagationDepth = 2;
    uint32_t sharcDebug = 0;
    int32_t sharcLevelBias = 16;
    bool sharcMaterialDemodulation = true;
    bool sharcSeparateEmissive = true;
    bool sharcDirectional = false;
    bool sharcCacheResampling = true;
    bool sharcBlendAdjacentLevels = true;
    bool sharcFadeAcceleration = false;
    bool opacityMicromaps = false;
    // 0 = glTF (-ln(C)/d), 1 = Cycles ((1-C)/d)
    uint32_t volumeModel = 0;
    uint32_t materialModel = 0;
    // Longest side a texture is allowed on load; 0 = no limit.
    uint32_t textureMaxDim = 0;
    // Divide every texture's dimensions by this on load; 1 = full size.
    uint32_t textureDownscale = 1;
    // Lossy block compression is an explicit memory/quality trade-off.
    bool textureCompress = false;
    // Debug visualisation; 0 renders normally.
    uint32_t debugMode = 0;
    uint32_t blueNoiseSwitchSpp = 4;
    // Upper bound on one indirect path's contribution; 0 = unclamped, which is
    // the default because clamping is a bias the caller has to ask for.
    float clampIndirect = 0.0f;
    // Separate primary/direct ceiling: Corona exposes highlight clamping apart
    // from MSI, and glossy NEE outliers happen before the indirect bound applies.
    float clampDirect = 0.0f;

    int cameraIndex = 0;
    // Frame this scene node exactly like the editor's F command. The optional
    // instance disambiguates EXT_mesh_gpu_instancing while keeping all sibling
    // primitives at that placement in the bounds.
    std::optional<uint32_t> frameNode;
    std::optional<uint32_t> frameInstance;
    std::optional<glm::vec3> cameraPosition;
    std::optional<glm::vec3> cameraTarget;
    std::optional<glm::vec3> cameraUp;
    std::optional<glm::quat> cameraOrientation;
    std::optional<Camera::ProjectionType> cameraProjection;
    std::optional<float> cameraFov;
    std::optional<float> cameraXMag;
    std::optional<float> cameraYMag;
    std::optional<float> cameraNear;
    std::optional<float> cameraFar;
    // Depth of field, off unless the config states a focal distance: an offline
    // still is graded against a pinhole everywhere else in this tree.
    std::optional<float> cameraFocalDistance;
    float cameraFStopDof = 2.8f;
    float cameraFocalLengthMm = 50.0f;
    std::optional<float> animationTime;

    // 0=None, 1=Reinhard, 2=ACES, 3=Filmic, 4=AgX
    uint32_t tonemapType = 2;
    bool exposureOverridden = false;
    float gamma = 2.4f;
    float filmIso = 100.0f;
    float fStop = 4.0f;
    float shutterSpeed = 100.0f;
};

/// Parse named enum values used by both TOML and CLI flags.
/// Throws std::invalid_argument on unknown names.
uint32_t parseSamplerName(const std::string& name);
uint32_t parseReconstructionFilterName(const std::string& name);
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
    bool saveOutput(Buffer* buf, const std::string& path = {});
    bool saveCheckpoint(Buffer* buf, uint32_t accumulatedSpp);
    void printProgress(uint32_t current, uint32_t total, double lastItemMs, const char* unit = "spp");

    RenderConfig m_config;
    std::unique_ptr<SettingsManager> m_settings;
    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<SharedContext> m_sharedCtx;
    std::unique_ptr<Render> m_render;
};

} // namespace oka
