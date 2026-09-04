#include "EditorApp.h"

#include "editor_camera_exposure.h"
#include "editor_denoiser_ui.h"
#include "editor_frame_budget.h"
#include "editor_camera_framing.h"
#include "editor_document.h"
#include "editor_screenshot.h"
#include "camera_dump.h"
#include "strelka_version.h"

#include "imgui_impl_glfw.h"
#include "imgui_internal.h" // DockBuilder / window settings lookup
#include "ImGuiFileDialog.h"

#include <strelka/sceneloader/sceneserializer.h>
#include <strelka/display/output_policy.h>
#include <env.h>
#include <log.h>
#include <paths.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <cmath>
#include <ctime>
#include <optional>
#include <vector>
#include <unistd.h>

#include <tinyexr.h>
#include <stb_image_write.h>
#include <cstring>
#include <numbers>

namespace oka
{
namespace
{
constexpr double kInteractiveLoadTimeoutSec = 300.0;

std::optional<std::string> environmentValue(const char* name)
{
    // Environment overrides are immutable after startup. Copy the value at the
    // read boundary so no caller retains getenv's process-global storage.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* value = std::getenv(name);
    return value != nullptr ? std::optional<std::string>(value) : std::nullopt;
}
} // namespace

EditorApp::EditorApp(const std::string& sceneFile, const std::string& resourceSearchPath)
    : m_resourceSearchPath(resourceSearchPath)
{
    m_settingsManager = std::make_unique<SettingsManager>();

    m_scene = std::make_unique<Scene>();
    m_display = std::unique_ptr<Display>(createDisplay());
    m_render = std::unique_ptr<Render>(RenderFactory::createRender());
    m_sharedCtx = std::make_unique<SharedContext>();

    m_sceneLoader = std::make_unique<GltfLoader>();

    m_render->setScene(m_scene.get());
    m_render->setSettingsManager(m_settingsManager.get());
    m_render->setSharedContext(m_sharedCtx.get());
    m_render->setLoadProgress(&m_loadProgress);
    m_sceneLoader->setProgress(&m_loadProgress);

    // Window first, scene second.
    //
    // The scene used to be parsed here, before the window existed, so the five
    // seconds a large scene takes were five seconds of an application that had
    // not drawn anything and could not be closed. Render::init() is only the
    // device and the pipelines -- some twenty milliseconds -- so there is nothing
    // stopping the window from coming up first and the scene arriving into it.
    loadSettings();
    m_render->init();
#ifdef __APPLE__
    m_display->setNativeDevice(m_render->getNativeDevicePtr());
    // Display creates its own command queue for independent frame pacing
#endif
    // Vulkan selects the physical device matching the renderer's active CUDA
    // device, so the renderer identity must be available during display init.
    m_display->setRender(m_render.get());
    m_display->init(1024, 768, m_settingsManager.get());
    m_display->setResizeHandler(this);

    // A camera has to exist before a scene does: the main loop reads
    // getCamera(m_selectedCamera) on every iteration, including the ones that
    // draw nothing but the progress bar. It is replaced with one framed to the
    // scene's bounds when the load lands.
    oka::Camera camera;
    camera.name = "Main";
    camera.fov = 45.0f;
    camera.position = glm::float3(0.0f, 0.0f, 5.0f);
    camera.mOrientation = glm::quat(glm::vec3(0, 0, 0));
    camera.updateViewMatrix();
    m_scene->addCamera(camera);
    setCameraDetached(false);
    m_cameraController = std::make_unique<CameraController>(m_scene->getCamera(m_selectedCamera), true);
    m_display->setInputHandler(m_cameraController.get());

    // Same directory as imgui.ini: next to the binary, so the working set
    // survives a rebuild and does not depend on which directory launched us.
    m_recentScenes = editor_document::loadRecentScenes(getExecutableDir() / "recent_scenes.txt");

    // Keep m_sceneFile empty until a load succeeds, so a failed startup open
    // restores to an empty document instead of pointing Save at a never-loaded path.
    beginSceneLoad(sceneFile, resourceSearchPath);
}

EditorApp::~EditorApp()
{
    if (m_isLoading && m_loadingFuture.valid())
    {
        m_loadProgress.cancel();
        m_loadingFuture.wait();
    }
}

void EditorApp::showAlert(const std::string& message)
{
    m_alertMessage = message;
    m_alertOpen = true;
    m_alertOffersRendererRestart = false;
}

void EditorApp::drawAlertModal()
{
    if (m_alertOpen)
    {
        ImGui::OpenPopup("EditorAlert");
    }
    if (ImGui::BeginPopupModal("EditorAlert", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::TextWrapped("%s", m_alertMessage.c_str());
        if (m_alertOffersRendererRestart && ImGui::Button("Restart renderer at 0.25 PT scale", ImVec2(260, 0)))
        {
            m_rendererRestartRequested = true;
            m_alertOpen = false;
            m_alertOffersRendererRestart = false;
            ImGui::CloseCurrentPopup();
        }
        if (m_alertOffersRendererRestart)
        {
            ImGui::SameLine();
        }
        if (ImGui::Button("OK", ImVec2(120, 0)))
        {
            m_alertOpen = false;
            m_alertOffersRendererRestart = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void EditorApp::drawAboutModal()
{
    if (m_aboutOpen)
    {
        ImGui::OpenPopup("About Strelka");
        m_aboutOpen = false;
    }
    if (ImGui::BeginPopupModal("About Strelka", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("Strelka");
        ImGui::TextDisabled("GPU path tracer for macOS (Metal)");
        ImGui::Spacing();
        ImGui::Text("Version %s", STRELKA_VERSION);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        // Everything a bug report needs, one click from the menu that opened
        // this rather than the user retyping a version string by hand.
        if (ImGui::Button("Copy Version Info", ImVec2(160, 0)))
        {
            const std::string info = fmt::format("Strelka {} (macOS)", STRELKA_VERSION);
            ImGui::SetClipboardText(info.c_str());
        }
        ImGui::SameLine();
        if (ImGui::Button("Close", ImVec2(120, 0)))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void EditorApp::drawFrameBudgetModal()
{
    if (m_frameBudgetConfirmOpen)
    {
        ImGui::OpenPopup("FrameBudgetWarning");
    }
    if (ImGui::BeginPopupModal("FrameBudgetWarning", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::TextWrapped(
            "This preview is estimated at %.0f ms per PT frame, above the %.0f ms "
            "interactive budget.",
            m_pendingPredictedGpuMs, editor_frame_budget::kInteractiveBudgetMs);
        ImGui::TextWrapped("Lower PT scale to keep the editor responsive?");

        const std::string recommendedLabel = fmt::format("Use {:.2f} PT scale", m_pendingRecommendedScale);
        if (ImGui::Button(recommendedLabel.c_str(), ImVec2(170, 0)))
        {
            m_settingsManager->setAs<bool>("render/pt/denoise", false);
            m_settingsManager->setAs<uint32_t>("render/pt/upscaleMode", 0);
            m_settingsManager->setAs<float>("render/pt/upscaleFactor", m_pendingRecommendedScale);
            m_settingsManager->setAs<bool>("render/pt/enableUpscale", true);
            // Which entry of the denoiser combo that is depends on the backend;
            // let the panel work it back out from the settings just written.
            mDenoiseModeInitialized = false;
            applyPreviewResolution(m_pendingPreviewWidth, m_pendingPreviewHeight);
            m_frameBudgetConfirmOpen = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Apply anyway", ImVec2(120, 0)))
        {
            applyPreviewResolution(m_pendingPreviewWidth, m_pendingPreviewHeight);
            m_frameBudgetConfirmOpen = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(90, 0)))
        {
            m_frameBudgetConfirmOpen = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void EditorApp::ensureValidCameraSelection()
{
    const uint32_t count = m_scene ? m_scene->getCameraCount() : 0;
    m_selectedCamera = editor_document::clampCameraIndex(m_selectedCamera, count);

    // The renderer takes the camera it draws from the settings, while picking, the
    // selection box and the gizmo all read m_selectedCamera. Only the camera combo
    // used to write the setting, so loading a scene that carries its own cameras
    // left the frame coming from the scene's first camera while every click was
    // traced through the fitted "Main" one: in a scene whose first camera is
    // orthographic nothing the user clicked was where the UI thought it was.
    if (m_settingsManager && count > 0)
    {
        const uint32_t selected = static_cast<uint32_t>(m_selectedCamera);
        if (m_settingsManager->getAs<uint32_t>("render/selectedCamera") != selected)
        {
            m_settingsManager->setAs<uint32_t>("render/selectedCamera", selected);
        }
    }
}

void EditorApp::handleDeviceError()
{
    if (!m_render || !m_render->deviceError())
    {
        return;
    }
    m_renderSubmissionsBlocked = true;
    if (m_deviceErrorLatched)
    {
        return;
    }
    m_deviceErrorLatched = true;
    m_frameBudgetConfirmOpen = false;
    STRELKA_INFO("ACTION device_error");
    STRELKA_ERROR("GPU device error — render submissions stopped");
    // Print the viewport that did it, unasked. A GPU timeout is reported by the
    // backend as a chunk index and a bounce range, which says what the renderer
    // was doing but not what it was looking at -- and the ones seen so far only
    // reproduce from one angle, which no chunk index recovers. The dump is a
    // .toml StrelkaCLI reads, so a pasted crash log carries its own repro
    // instead of a description of where the camera roughly was.
    dumpCameraSettings();
    showAlert(
        "GPU device error.\nRender submissions have been stopped.\n"
        "You can rebuild the renderer at a safe PT scale or keep the last good frame.");
    m_alertOffersRendererRestart = true;
}

void EditorApp::restoreDocumentAfterFailedLoad(const char* reason)
{
    const std::string attempted = m_attemptedSceneFile.empty() ? m_sceneFile : m_attemptedSceneFile;
    m_sceneFile = editor_document::restorePathAfterFailedLoad(m_sceneFileBeforeLoad);
    m_documentDirty = m_documentDirtyBeforeLoad;
    m_undoStack = std::move(m_undoStackBeforeLoad);
    m_redoStack = std::move(m_redoStackBeforeLoad);
    m_undoStackBeforeLoad.clear();
    m_redoStackBeforeLoad.clear();

    const bool cancelled = m_loadProgress.isCancelled() || std::strcmp(reason, "cancel") == 0;
    if (cancelled)
    {
        STRELKA_INFO("ACTION open_cancel path={}", attempted);
        STRELKA_ERROR("Scene open cancelled: {}", attempted);
        showAlert(fmt::format("Open cancelled:\n{}", attempted));
    }
    else
    {
        STRELKA_INFO("ACTION open_fail path={} reason={}", attempted, reason);
        STRELKA_ERROR("Scene open failed ({}): {}", reason, attempted);
        showAlert(fmt::format("Failed to open scene:\n{}\n({})", attempted, reason));
    }
}

void EditorApp::rememberRecentScene(const std::string& sceneFile)
{
    if (sceneFile.empty())
    {
        return;
    }
    editor_document::pushRecentScene(m_recentScenes, sceneFile);
    persistRecentScenes();
}

void EditorApp::persistRecentScenes()
{
    if (!editor_document::saveRecentScenes(getExecutableDir() / "recent_scenes.txt", m_recentScenes))
    {
        STRELKA_WARNING("Could not write recent scenes list");
    }
}

void EditorApp::beginSceneLoad(const std::string& sceneFile, const std::string& resourceSearchPath)
{
    m_settingsManager->setAs<std::string>("resource/searchPath", resourceSearchPath);
    m_pendingResourcePath = resourceSearchPath;
    m_loadProgress.reset();

    if (sceneFile.empty())
    {
        // No document to read. The editor keeps the scene it was constructed with
        // -- one camera, no geometry -- and renders it; starting a load of "" would
        // only produce a loader error and a progress bar for nothing.
        m_sceneFile.clear();
        m_isLoading = false;
        clearSelection();
        m_documentDirty = false;
        m_undoStack.clear();
        m_redoStack.clear();
        STRELKA_INFO("ACTION open_ok path=(empty)");
        return;
    }

    m_sceneFileBeforeLoad = m_sceneFile;
    m_documentDirtyBeforeLoad = m_documentDirty;
    m_undoStackBeforeLoad = m_undoStack;
    m_redoStackBeforeLoad = m_redoStack;
    m_attemptedSceneFile = sceneFile;
    m_sceneFile = sceneFile;
    m_loadStartedAt = std::chrono::steady_clock::now();

    STRELKA_INFO("ACTION open_begin path={}", sceneFile);

    // The loader is captured raw because it outlives every load: ~EditorApp
    // cancels and joins before any member is destroyed.
    auto loader = m_sceneLoader.get();
    m_loadingFuture = std::async(std::launch::async, [loader, sceneFile]() -> std::unique_ptr<Scene> {
        try
        {
            auto scene = std::make_unique<Scene>();
            if (loader->loadGltf(sceneFile, *scene))
            {
                return scene;
            }
        }
        catch (const std::exception& e)
        {
            // std::async surfaces exceptions from future::get on the UI thread;
            // treat OOM / loader throws as a failed load instead.
            STRELKA_ERROR("Scene load of '{}' threw: {}", sceneFile, e.what());
        }
        catch (...)
        {
            STRELKA_ERROR("Scene load of '{}' threw an unknown exception", sceneFile);
        }
        // Cancelled, malformed, or threw. The partial scene is destroyed here.
        return nullptr;
    });
    m_isLoading = true;
    clearSelection();
    m_documentDirty = false;
    m_undoStack.clear();
    m_redoStack.clear();
}

void EditorApp::framebufferResize(int newWidth, int newHeight)
{
    // Preview resolution is fixed by Render Settings. Window and dockspace
    // resizes only change how ImGui presents that texture.
    (void)newWidth;
    (void)newHeight;
}

void EditorApp::applyPreviewResolution(uint32_t width, uint32_t height)
{
    width = std::clamp(width, editor_viewport::kMinPreviewDimension, editor_viewport::kMaxPreviewDimension);
    height = std::clamp(height, editor_viewport::kMinPreviewDimension, editor_viewport::kMaxPreviewDimension);
    if (m_settingsManager->getAs<uint32_t>("render/width") == width &&
        m_settingsManager->getAs<uint32_t>("render/height") == height)
    {
        return;
    }
    m_settingsManager->setAs<uint32_t>("render/width", width);
    m_settingsManager->setAs<uint32_t>("render/height", height);
    m_sharedCtx->mSubframeIndex = 0;
    m_render->resetTemporalHistory();
}

void EditorApp::requestPreviewResolution(uint32_t width, uint32_t height)
{
    const uint32_t currentWidth = m_settingsManager->getAs<uint32_t>("render/width");
    const uint32_t currentHeight = m_settingsManager->getAs<uint32_t>("render/height");
    if (currentWidth == width && currentHeight == height)
    {
        return;
    }

    // The fraction of the output the tracer is really launching at, which is not
    // the upscale factor on a backend whose upscaler has one fixed ratio: OptiX
    // reads the enable bit and takes exactly half. Dividing a measured frame time
    // by the wrong pixel count mispredicts the next one by that ratio squared.
    const bool enableUpscale = m_settingsManager->getAs<bool>("render/pt/enableUpscale");
    const editor_denoiser::Ui fx = editor_denoiser::uiFor(m_render->denoiserKind());
    const int denoiseMode =
        editor_denoiser::modeIndexFromSettings(fx, m_settingsManager->getAs<bool>("render/pt/denoise"), enableUpscale);
    const float tracedScale =
        editor_denoiser::appliedScale(fx, denoiseMode, m_settingsManager->getAs<float>("render/pt/upscaleFactor"));
    const editor_frame_budget::RenderSettingsSnapshot current{ currentWidth, currentHeight, tracedScale < 1.0f,
                                                               tracedScale };
    const editor_frame_budget::FrameSample sample =
        editor_frame_budget::sampleFrom(m_render->getLastRenderTimeMs(), current);
    const editor_frame_budget::RenderSettingsSnapshot proposed{ width, height, tracedScale < 1.0f, tracedScale };
    const editor_frame_budget::Assessment assessment = editor_frame_budget::assess(sample, proposed);
    if (!assessment.exceedsBudget)
    {
        applyPreviewResolution(width, height);
        return;
    }

    m_pendingPreviewWidth = width;
    m_pendingPreviewHeight = height;
    m_pendingPredictedGpuMs = assessment.predictedGpuTimeMs;
    m_pendingRecommendedScale = editor_frame_budget::recommendedScale(sample, width, height);
    m_frameBudgetConfirmOpen = true;
}

glm::vec3 EditorApp::computeSceneFitPosition(float fovDegrees) const
{
    const auto& vertices = m_scene->getVertices();
    if (vertices.empty())
        return { 0, 0, -10 };

    // Compute AABB
    glm::vec3 aabbMin(std::numeric_limits<float>::max());
    glm::vec3 aabbMax(std::numeric_limits<float>::lowest());
    for (const auto& v : vertices)
    {
        aabbMin = glm::min(aabbMin, v.pos);
        aabbMax = glm::max(aabbMax, v.pos);
    }

    const glm::vec3 center = (aabbMin + aabbMax) * 0.5f;
    float radius = glm::length(aabbMax - center);
    if (radius < 1e-6f)
        radius = 1.0f;

    // Distance so the bounding sphere fits in the vertical FOV
    const float halfFovRad = glm::radians(fovDegrees * 0.5f);
    const float distance = radius / std::tan(halfFovRad);

    // Position camera along -Z looking at center (worldForward = (0,0,-1))
    return center + glm::vec3(0.0f, 0.0f, distance);
}


// Measure the scene's own brightness and expose for it, once.
//
// The radiance buffer is linear and pre-tonemap, which is the only place the
// question can be asked: after the tone curve every scene looks like it has
// roughly the range the curve has. Middle grey is the target because that is
// what a photographer's meter aims at, and it is what makes an unknown scene
// arrive looking neither black nor blown out.
//
// Set through cm2_factor with the film speed at zero, which is the tonemapper's
// own arbitrary-units mode -- the alternative, solving back to an f-stop, states
// a photographic setting the scene never had.
void EditorApp::applyAutoExposure(oka::Buffer* buf)
{
    if (!buf)
    {
        return;
    }
    const float* px = static_cast<const float*>(buf->getHostPointer());
    const size_t count = buf->getHostDataSize() / sizeof(float);
    if (!px || count < 4)
    {
        return;
    }
    // Metering lives in editor_camera_exposure.h so it can be tested against a
    // synthetic frame: the failure it guards is a meter that swings by stops
    // when only the framing changed, which no rendered check would notice.
    size_t n = 0;
    size_t total = 0;
    const double mean = oka::editor_camera_exposure::meteredMeanLuminance(px, count, n, total);
    if (n == 0)
    {
        return;
    }
    // A frame that is genuinely black -- lights off, camera in a wall -- has
    // nothing to expose for, and dividing by it would produce an absurd factor
    // that the next frame cannot recover from.
    if (!(mean > 1e-8))
    {
        return;
    }
    constexpr double kMiddleGrey = 0.18;
    const double factor = kMiddleGrey / mean;
    m_settingsManager->setAs<float>("render/post/tonemapper/filmIso", 0.0f);
    m_settingsManager->setAs<float>("render/post/tonemapper/cm2_factor", (float)factor);
    m_autoExposurePending = false;
    // {:.4g}, not {:.1f}: the factor is middle grey over the scene mean, so on
    // any scene brighter than 0.18 it is well below 1 and a single decimal place
    // printed every one of them as "x0.0". Stops alongside it, because that is
    // the unit the exposure controls in the UI are in.
    // The metered coverage is in the line because it is what distinguishes a
    // dim scene from a mostly empty frame, and those want opposite answers.
    STRELKA_INFO(
        "Auto exposure: scene mean luminance {:.5f} over the {:.1f}% of the frame that caught light, "
        "exposure x{:.4g} ({:+.2f} EV); no exposure in the light sidecar",
        mean, total > 0 ? 100.0 * double(n) / double(total) : 0.0, factor, std::log2(factor));
}

// Exposure comes from the scene when the scene says, and is measured from the
// first frame when it does not.
//
// Has to run after loadSettings(), which writes the photographic defaults
// unconditionally and would otherwise put them back over whatever the scene
// asked for -- that ordering is why the pine forest opened black.
//
// A glTF camera carries a projection and nothing else, so a file cannot state
// how bright it is meant to look. Those defaults are a real daylight setting --
// ISO 100, f/4, 1/100 s -- and against a scene authored in normalised units,
// which is most of them, they land about 1600x under: the pine forest arrives
// with two suns at irradiance 5 and 1 and an environment at intensity 1, and
// renders as black. That reads as a broken renderer.
void EditorApp::applySceneExposure()
{
    if (const auto& exposure = m_scene->getExposure(); exposure.has_value())
    {
        m_settingsManager->setAs<float>("render/post/tonemapper/filmIso", exposure->filmIso);
        m_settingsManager->setAs<float>("render/post/tonemapper/fStop", exposure->fStop);
        m_settingsManager->setAs<float>("render/post/tonemapper/shutterSpeed", exposure->shutterSpeed);
        m_settingsManager->setAs<float>("render/post/tonemapper/cm2_factor", exposure->cm2Factor);
        m_autoExposurePending = false;
        STRELKA_INFO("Exposure from scene: ISO {:.0f}, f/{:.1f}, 1/{:.0f} s, x{:.2f}", exposure->filmIso,
                     exposure->fStop, exposure->shutterSpeed, exposure->cm2Factor);
    }
    else
    {
        m_autoExposurePending = true;
        // A new scene has its own frames to settle over.
        m_framesSinceSceneReady = 0;
    }
}

void EditorApp::loadSettings()
{
    STRELKA_DEBUG("Resource search path {}", m_resourceSearchPath);

    const uint32_t imageWidth = editor_viewport::kDefaultPreviewWidth;
    const uint32_t imageHeight = editor_viewport::kDefaultPreviewHeight;

    seedCommonRenderSettings(*m_settingsManager);
    m_settingsManager->setAs<uint32_t>("render/width", imageWidth);
    m_settingsManager->setAs<uint32_t>("render/height", imageHeight);
    m_settingsManager->setAs<uint32_t>("render/pt/depth", 8);
    m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", 256);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    // stratified sampling, 3 -
    // optimized stratified sampling
    m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", 1); // 0 - None, 1 - Reinhard, 2 - ACES, 3 - Filmic
    m_settingsManager->setAs<uint32_t>("render/pt/debug", 0); // 0 - none, 1 - normals
    m_settingsManager->setAs<float>("render/cameraSpeed", 1.0f);
    // Gamepad. Editor-only, so deliberately not mirrored into
    // HeadlessApp::populateSettings(): a headless render has no one holding a
    // controller, and a key that exists in both places is a key that has to be
    // kept in step in both places.
    //
    // The pad itself needs no enabling -- it is detected and used when present.
    // What is here is the tuning a hand can disagree with: `enabled` exists to
    // turn a pad off without unplugging it (a controller left on a desk with a
    // sticky stick would otherwise keep restarting accumulation), and the rest
    // is the feel. Defaults come from gamepad::Config, which is where the
    // measurements behind them are written down.
    m_settingsManager->setAs<bool>("editor/gamepad/enabled", true);
    m_settingsManager->setAs<bool>("editor/gamepad/invertLookY", false);
    m_settingsManager->setAs<float>("editor/gamepad/lookSpeed", gamepad::Config{}.lookSpeed);
    m_settingsManager->setAs<float>("editor/gamepad/deadzone", gamepad::Config{}.deadzone);
    m_settingsManager->setAs<float>("render/pt/upscaleFactor", 0.5f);
    // Preview resolution itself bounds interactive work. MetalFX remains an
    // explicit quality/performance choice inside that fixed output.
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
    // Sobol with a blue-noise screen-space error distribution, handing over to
    // plain per-pixel Owen scrambling once the frame has enough samples that the
    // spectrum of the error matters less than how fast it shrinks.
    //
    // Not Halton, which was the default and is the reason 1024 samples did not
    // look like four times 256: its dimensions are told apart only by an offset
    // into a table of 32 bases, so any two that are 32 apart -- and a depth-8
    // path uses 117 -- are the same sequence read from two places. Measured on
    // vespa, its error at 1024 spp was no lower than at 512.
    m_settingsManager->setAs<uint32_t>("render/pt/samplerType", 4);
    // Measured crossover on vespa: blue noise wins on post-filter error up to
    // 4 samples (-12% at 1 spp, -7% at 2, -4% at 4) and loses past it (+13% at
    // 8, +9% at 16, +40% at 32), because a toroidal shift is a weaker
    // randomisation than a scramble once there are enough samples for that to
    // matter. Past the crossover the mask does not merely converge slower, it
    // makes the error *less* blue than the plain scramble does -- lp/raw rises
    // from 0.28 to 0.36 -- so there is nothing left to trade for.
    //
    // This was 16, from the same measurement taken before the Owen scramble
    // used Vegdahl's LK hash. The old hash left the blue-noise sampler's
    // dimensions correlated (they share a screen-wide seed and differ only in
    // its low bits), which held that sampler back at the counts where it was
    // still nominally winning. Fixing the hash moved the crossover to 4.
    //
    // Four is also where it matters: accumulation restarts whenever the camera
    // moves, so navigating the scene means looking at 1-4 spp frames.
    m_settingsManager->setAs<uint32_t>("render/pt/blueNoiseSwitchSpp", 4);
    m_settingsManager->setAs<uint32_t>("render/selectedCamera", 0);
    m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
    m_settingsManager->setAs<float>("render/animation/speed", 1.0f);
    // Wavefront by default: bit-identical output, 2.6x faster at depth 8. The
    // megakernel stays selectable so any change can still be A/B'd against it.
    m_settingsManager->setAs<uint32_t>("render/pt/profileStages", 0);
    // Honour STRELKA_STAGES in interactive runs, not only in a benchmark. The
    // GPU-failure path tells the reader to "reproduce with STRELKA_STAGES=1",
    // and until this line the variable was read in runBenchmark() alone -- so
    // following that advice on the editor changed nothing and the log said
    // "stage diagnosis disabled" a second time. Advice a diagnostic prints has
    // to work in the mode that printed it.
    if (envFlag("STRELKA_STAGES"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/profileStages", 1);
    }
    m_settingsManager->setAs<uint32_t>("render/pt/subsurfaceIterations", 64);
    // Four power-sampled candidates cut one-spp log error by 20% on kids room
    // and 5% on bathroom; eight has little left to win for twice the work.
    m_settingsManager->setAs<uint32_t>("render/pt/risCandidates", 4u);
    m_settingsManager->setAs<uint32_t>("render/pt/writeAov", 0);
    m_settingsManager->setAs<bool>("render/pt/denoise", false);
    // The MetalFX denoiser compiles a large graph synchronously. The interactive
    // renderer prepares it during startup so selecting the default-off mode does
    // not put that work on a display frame; headless runs opt in only when their
    // configuration actually enables denoising.
    m_settingsManager->setAs<bool>("render/pt/prewarmDenoiser", true);
    // Luminance ceiling for the denoiser's colour input, in exposed units: a
    // single unbounded sample gets smeared over many frames by a temporal filter.
    // 0 disables it.
    m_settingsManager->setAs<float>("render/pt/denoiseFireflyClamp", 8.0f);
    m_settingsManager->setAs<float>("render/pt/clampIndirect", 0.0f);
    if (envFlag("STRELKA_DENOISE"))
    {
        m_settingsManager->setAs<bool>("render/pt/denoise", envBool("STRELKA_DENOISE", false));
    }
    m_settingsManager->setAs<uint32_t>("render/pt/sortRays", 0);
    m_settingsManager->setAs<uint32_t>("render/pt/textureLod", 0);
    m_settingsManager->setAs<uint32_t>("render/pt/guidePrimaryHit", 0);
    // 0 = the single-image model, 1 = the temporally stable one. Off by default:
    // a temporal model reusing a history it has no motion vectors for produces a
    // smear. Overridable from the environment because it is the switch the
    // denoise audit has to flip to measure reprojection at all -- with no way to
    // turn it on headlessly, the motion vectors it grades feed nothing, which is
    // how they stayed identically zero through several audit runs.
    m_settingsManager->setAs<uint32_t>("render/pt/upscaleMode", envUint("STRELKA_UPSCALE_MODE", 0) != 0 ? 1u : 0u);
    if (envFlag("STRELKA_UPSCALE"))
    {
        const float f = envFloat("STRELKA_UPSCALE", 1.0f);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", f > 0.0f && f < 1.0f);
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", f);
    }
    if (envFlag("STRELKA_AOV"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/writeAov", envUint("STRELKA_AOV", 0));
    }
    if (envFlag("STRELKA_DEBUG_VIEW"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/debug", envUint("STRELKA_DEBUG_VIEW", 0));
    }
    // Static-geometry traversal. Forcing it off makes the wavefront traverse the
    // same structure the megakernel does, which is what the bit-identity check
    // needs: the two intersector types round intersection distances differently.
    if (envFlag("STRELKA_STATIC"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/staticTraversal", envUint("STRELKA_STATIC", 1));
    }
    m_settingsManager->setAs<uint32_t>("render/validate/estimatorMode", 0);
    // Absorption convention for transmissive media: 0 = glTF, 1 = Cycles.
    m_settingsManager->setAs<uint32_t>("render/material/volumeModel", 0);
    // 0 = glTF metallic-roughness, 1 = OpenPBR Surface. Seeded here as well as in
    // HeadlessApp because SettingsManager::getAs asserts on a key nobody set.
    m_settingsManager->setAs<uint32_t>("render/material/model", 0);
    m_settingsManager->setAs<uint32_t>("render/texture/maxDimension", 0);
    m_settingsManager->setAs<uint32_t>("render/texture/downscale", 1);
    // The diffuse/specular split of the first event, accumulated into two extra
    // images. Off by default: nothing in either app reads them back, and writing
    // them costs four scattered records per pixel per launch. See
    // docs/open-perf.md.
    m_settingsManager->setAs<bool>("render/pt/splitAov", false);
    // Spatially hashed radiance cache. Off by default: it trades a little
    // bias for a large cut in path length, which is a choice a scene makes.
    m_settingsManager->setAs<bool>("render/pt/sharc", false);
    m_settingsManager->setAs<uint32_t>("render/pt/sharcCapacity", 1u << 22);
    m_settingsManager->setAs<uint32_t>("render/pt/sharcMinSamples", 8);
    m_settingsManager->setAs<uint32_t>("render/pt/sharcMetalMinSamples", 1);
    m_settingsManager->setAs<uint32_t>("render/pt/sharcDepth", 1);
    // Samples after which the cache stops being read. It accelerates the frames
    // you are moving the camera through and then gets out of the way, because
    // past this point its own error is the larger of the two. 0 = never stop.
    m_settingsManager->setAs<uint32_t>("render/pt/sharcReadFrames", 128);
    // How many pixels wide a cache voxel should be at any distance.
    m_settingsManager->setAs<float>("render/pt/sharcVoxelPixels", 4.0f);
    // The temporal window, in frames, and how long an entry survives with
    // nothing deposited into it. Together these are what let the table outlive a
    // camera movement instead of being cleared by it; see sharc_resolve.h.
    m_settingsManager->setAs<uint32_t>("render/pt/sharcAccumFrames", 32);
    m_settingsManager->setAs<uint32_t>("render/pt/sharcStaleFrames", 64);
    // Responsive lighting: the short window a light marked `responsive` in the
    // scene is cached on. On whenever such a light exists -- the setting can
    // only turn it off, which is what makes it an A/B rather than a switch
    // somebody has to find.
    m_settingsManager->setAs<bool>("render/pt/sharcResponsiveLighting", true);
    m_settingsManager->setAs<uint32_t>("render/pt/sharcResponsiveFrames", 4);
    // Metal's own responsive switch, off by default: its compact key has no
    // spare bit for a per-light tag, so the companion entries hold the whole
    // lighting signal and come out of the configured capacity.
    m_settingsManager->setAs<bool>("render/pt/sharcMetalResponsive", false);
    m_settingsManager->setAs<float>("render/pt/sharcRoughnessThreshold", 0.4f);
    m_settingsManager->setAs<float>("render/pt/sharcRadianceScale", 1000.0f);
    m_settingsManager->setAs<uint32_t>("render/pt/sharcUpdateDownscale", 5u);
    m_settingsManager->setAs<uint32_t>("render/pt/sharcPropagationDepth", 2u);
    m_settingsManager->setAs<uint32_t>("render/pt/sharcDebug", 0u);
    // The compact key stores unsigned 5-bit LODs. Offset the exponent into the
    // middle of that range so sub-unit indoor distances do not all clamp to LOD 1.
    m_settingsManager->setAs<uint32_t>("render/pt/sharcLevelBias", 16u);
    m_settingsManager->setAs<bool>("render/pt/sharcMaterialDemodulation", true);
    m_settingsManager->setAs<bool>("render/pt/sharcSeparateEmissive", true);
    m_settingsManager->setAs<bool>("render/pt/sharcDirectional", false);
    m_settingsManager->setAs<bool>("render/pt/sharcCacheResampling", true);
    m_settingsManager->setAs<bool>("render/pt/sharcBlendAdjacentLevels", true);
    m_settingsManager->setAs<bool>("render/pt/sharcFadeAcceleration", false);
    // How often a scene that is still loading is republished. Every snapshot
    // restarts convergence, so this trades latency against the noise the load
    // finishes with; twice a second reads as continuous without doing that
    // often enough to matter.
    m_settingsManager->setAs<float>("render/stream/publishIntervalMs", 500.0f);
    // The editor picks against the host arrays, so it keeps them.
    m_settingsManager->setAs<bool>("scene/releaseHostGeometry", false);
    m_settingsManager->setAs<std::string>("resource/searchPath", m_resourceSearchPath);
    // Postprocessing settings:
    m_settingsManager->setAs<uint32_t>("render/post/outputMode",
                                       static_cast<uint32_t>(display_output::OutputMode::Auto));
    m_settingsManager->setAs<float>("render/post/paperWhiteNits", 203.0f);
    m_settingsManager->setAs<float>("render/post/peakNits", 1000.0f);
    m_settingsManager->setAs<bool>("display/vrr/enabled", true);
    // Metal presentation. Seeded on every platform because SettingsManager does
    // not insert on read -- a missing key logs an error and asserts -- and the
    // panel picks its branch from the display, not from an #ifdef here.
    // 0 headroom means "whatever the display grants"; 0 fps means "the display's
    // own rate". Both are the neutral choice, so the defaults change nothing
    // until the user touches them.
    m_settingsManager->setAs<float>("display/edr/headroomLimit", 0.0f);
    m_settingsManager->setAs<bool>("display/vsync/enabled", true);
    m_settingsManager->setAs<bool>("display/present/tripleBuffering", true);
    m_settingsManager->setAs<float>("display/present/fpsLimit", 0.0f);
    m_settingsManager->setAs<float>("render/post/tonemapper/filmIso", 100.0f);
    m_settingsManager->setAs<float>("render/post/tonemapper/cm2_factor", 1.0f);
    m_settingsManager->setAs<float>("render/post/tonemapper/fStop", 4.0f);
    m_settingsManager->setAs<float>("render/post/tonemapper/shutterSpeed", 100.0f);
    // Editor UI: one f-stop drives DOF blur and photographic exposure unless the
    // user unticks "Link DOF aperture to exposure".
    m_settingsManager->setAs<bool>("render/post/tonemapper/linkDofFStop", true);

    m_settingsManager->setAs<float>("render/post/gamma", 2.4f); // 0.0f - off

    loadAnimSettings();
}

void EditorApp::setBatchCapture(uint32_t sppTotal, uint32_t sppSubframe, bool screenshotOnComplete)
{
    if (sppTotal > 0)
    {
        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", sppTotal);
        m_batchSppTotal = sppTotal;
    }
    else
    {
        m_batchSppTotal = m_settingsManager->getAs<uint32_t>("render/pt/sppTotal");
    }
    if (sppSubframe > 0)
    {
        m_settingsManager->setAs<uint32_t>("render/pt/spp", sppSubframe);
    }
    m_batchScreenshotArmed = screenshotOnComplete;
}

void EditorApp::loadAnimSettings()
{
    // Erase all previous per-animation settings to avoid leaking keys from old scenes
    m_settingsManager->eraseByPrefix("render/animation/anim");

    for (size_t i = 0; i < m_scene->getAnimations().size(); ++i)
    {
        m_settingsManager->setAs<bool>(animationStateKey(i), false);
        m_settingsManager->setAs<float>(animationTimeKey(i), m_scene->getAnimations()[i].start);
    }
}

void EditorApp::initializeRendererForCurrentScene()
{
    m_render = std::unique_ptr<Render>(RenderFactory::createRender());
    m_render->setSettingsManager(m_settingsManager.get());
    m_render->setSharedContext(m_sharedCtx.get());
    m_render->setScene(m_scene.get());
    m_render->setLoadProgress(&m_loadProgress);
    m_render->init();
    m_display->setRender(m_render.get());
}

void EditorApp::restartRendererAtSafeScale()
{
    if (!m_render || m_isLoading)
    {
        return;
    }

    STRELKA_INFO("ACTION renderer_restart scale=0.25");
    m_renderSubmissionsBlocked = true;
    m_settingsManager->setAs<bool>("render/pt/denoise", false);
    m_settingsManager->setAs<uint32_t>("render/pt/upscaleMode", 0);
    m_settingsManager->setAs<float>("render/pt/upscaleFactor", 0.25f);
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", true);
    mDenoiseModeInitialized = false;

    // Drop every raw pointer and retained frame before destroying the failed
    // renderer. The commit feedback has already completed, so the destructor can
    // drain the unaffected Metal 3 queue and release the old submission domain.
    m_display->resetFrame();
    m_display->setRender(nullptr);
    m_render.reset();

    m_sharedCtx = std::make_unique<SharedContext>();
    initializeRendererForCurrentScene();
    mPresentedPreviewWidth = 0;
    mPresentedPreviewHeight = 0;
    m_framesSinceSceneReady = 0;
    m_lastExposureFrameSeen = static_cast<size_t>(-1);
    m_deviceErrorLatched = false;
    m_renderSubmissionsBlocked = false;
    m_rendererRestartRequested = false;
    m_render->resetTemporalHistory();
    STRELKA_INFO("ACTION renderer_restart_ok scale=0.25");
}

void EditorApp::checkLoadingComplete()
{
    if (!m_isLoading)
        return;
    if (m_loadingFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;
    if (m_render->isRenderBusy())
        return;

    auto new_scene = m_loadingFuture.get();
    m_isLoading = false;

    if (!new_scene)
    {
        const char* reason = m_loadProgress.isCancelled() ? "cancel" : "loader";
        restoreDocumentAfterFailedLoad(reason);
        return;
    }

    // Tear the old renderer down *before* the scene and shared context it points
    // at are replaced. ~MetalRender drains the GPU and waits for in-flight
    // completion handlers; running that after the Scene/SharedContext it holds
    // raw pointers to have already been freed is a use-after-free waiting for
    // the right timing.
    m_display->resetFrame();
    // Clear the display's raw Render* before destroying the object it points at.
    m_display->setRender(nullptr);
    m_render.reset();

    m_scene = std::move(new_scene);

    const uint32_t authoredCameraCount = m_scene->getCameraCount();

    oka::Camera camera;
    camera.name = "Main";
    camera.fov = 45.0f;
    camera.position = computeSceneFitPosition(camera.fov);
    camera.mOrientation = glm::quat(glm::vec3(0, 0, 0));
    camera.updateViewMatrix();
    m_scene->addCamera(camera);

    m_selectedCamera = editor_document::selectCameraIndexAfterLoad(authoredCameraCount, m_scene->getCameraCount());
    setCameraDetached(false);

    loadAnimSettings();

    m_sharedCtx = std::make_unique<SharedContext>();

    initializeRendererForCurrentScene();

    m_resourceSearchPath = m_pendingResourcePath;
    m_documentDirty = false;
    m_undoStackBeforeLoad.clear();
    m_redoStackBeforeLoad.clear();
    m_deviceErrorLatched = false;
    m_renderSubmissionsBlocked = false;

    ensureValidCameraSelection();
    m_cameraController->setCamera(m_scene->getCamera(m_selectedCamera));
    m_render->resetTemporalHistory(); // new scene, new everything
    m_display->setInputHandler(m_cameraController.get());

    // Every scene brings its own exposure, so this belongs here rather than in
    // startup: before, a scene opened through File -> Open kept the previous
    // one's exposure and there was no way to tell from the picture whether that
    // was the scene's intent.
    applySceneExposure();

    rememberRecentScene(m_sceneFile);
    STRELKA_INFO("ACTION open_ok path={}", m_sceneFile);
}

// GPU timing harness (STRELKA_BENCH=<frames>).
//
// Reports the median, not the mean: the first frames warm caches and the
// occasional frame is stretched by an unrelated compositor stall, and both would
// move a mean by more than the effect sizes being measured here. Accumulation is
// off so every frame does the full amount of work.
void EditorApp::runBenchmark()
{
    const uint32_t frames = std::max(4u, envUint("STRELKA_BENCH", 4));
    const uint32_t warmup = std::max(4u, frames / 4);
    const bool residencyStress = envFlag("STRELKA_RESIDENCY_STRESS");

    if (envFlag("STRELKA_BENCH_W") || envFlag("STRELKA_BENCH_H"))
    {
        applyPreviewResolution(envUint("STRELKA_BENCH_W", m_settingsManager->getAs<uint32_t>("render/width")),
                               envUint("STRELKA_BENCH_H", m_settingsManager->getAs<uint32_t>("render/height")));
    }
    if (envFlag("STRELKA_BENCH_SCALE"))
    {
        const float scale = std::clamp(envFloat("STRELKA_BENCH_SCALE", 1.0f), 0.25f, 1.0f);
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", scale);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", scale < 1.0f);
    }
    if (envFlag("STRELKA_BENCH_DENOISE"))
    {
        m_settingsManager->setAs<bool>("render/pt/denoise", envUint("STRELKA_BENCH_DENOISE", 0) != 0);
    }
    if (envFlag("STRELKA_BENCH_FRAME_NODE"))
    {
        m_selectedNodeId = envUint("STRELKA_BENCH_FRAME_NODE", kInvalidIndex);
        m_selectedInstanceId = envUint("STRELKA_BENCH_FRAME_INSTANCE", kInvalidIndex);
        frameSelectionInView();
    }

    m_settingsManager->setAs<bool>("render/pt/enableAcc", false);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    m_settingsManager->setAs<uint32_t>("render/pt/risCandidates", envUint("STRELKA_BENCH_NEE_CANDIDATES", 1u));
    m_settingsManager->setAs<bool>(
        "render/pt/restirDIEnabled",
        envUint("STRELKA_RESTIR_DI", m_settingsManager->getAs<bool>("render/pt/restirDIEnabled") ? 1u : 0u) != 0u);
    m_settingsManager->setAs<uint32_t>(
        "render/pt/initialCandidateCount",
        envUint("STRELKA_RESTIR_CANDIDATES", m_settingsManager->getAs<uint32_t>("render/pt/initialCandidateCount")));
    m_settingsManager->setAs<bool>(
        "render/pt/temporalReuseEnabled",
        envUint("STRELKA_RESTIR_TEMPORAL", m_settingsManager->getAs<bool>("render/pt/temporalReuseEnabled") ? 1u : 0u) !=
            0u);
    m_settingsManager->setAs<bool>(
        "render/pt/spatialReuseEnabled",
        envUint("STRELKA_RESTIR_SPATIAL", m_settingsManager->getAs<bool>("render/pt/spatialReuseEnabled") ? 1u : 0u) !=
            0u);
    m_settingsManager->setAs<uint32_t>(
        "render/pt/spatialNeighborCount",
        envUint("STRELKA_RESTIR_NEIGHBORS", m_settingsManager->getAs<uint32_t>("render/pt/spatialNeighborCount")));
    m_settingsManager->setAs<uint32_t>(
        "render/pt/restirBiasCorrection",
        std::min(envUint("STRELKA_RESTIR_BIAS_CORRECTION",
                         m_settingsManager->getAs<uint32_t>("render/pt/restirBiasCorrection")),
                 1u));
    const bool requestedRestir = m_settingsManager->getAs<bool>("render/pt/restirDIEnabled");
    // One submission per frame, so the number is the tracer's cost and not the
    // inter-band gaps of the responsiveness split.
    if (envFlag("STRELKA_REF_DEPTH"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/depth", envUint("STRELKA_REF_DEPTH", 4));
    }
    if (envFlag("STRELKA_STAGES"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/profileStages", 1);
    }

    // Playback changes the workload qualitatively — deforming geometry needs
    // two-keyframe acceleration structures — so it needs its own measurement.
    const bool play = envFlag("STRELKA_PLAY");
    const bool orbitCamera = envFlag("STRELKA_BENCH_ORBIT");
    oka::Camera& camera = m_scene->getCamera(m_selectedCamera);
    const glm::quat initialCameraOrientation = camera.mOrientation;
    auto setPlaying = [&](bool on) {
        for (size_t i = 0; i < m_scene->getAnimations().size(); ++i)
        {
            m_settingsManager->setAs<bool>(animationStateKey(i), on);
        }
    };

    // One block of frames in one playback state. Returns median GPU and median
    // wall time, or {-1,-1} if the window closed before anything was measured.
    struct Block
    {
        double gpu = -1.0;
        double wall = -1.0;
        double gpuMin = -1.0;
        double gpuMax = -1.0;
    };
    auto measure = [&](bool playing) -> Block {
        setPlaying(playing);
        std::vector<double> samples, wall;
        samples.reserve(frames);
        wall.reserve(frames);
        auto prevFrame = std::chrono::high_resolution_clock::now();
        double last = -1.0;
        for (uint32_t i = 0; i < warmup + frames && !m_display->windowShouldClose();)
        {
            m_display->pollEvents();
            if (m_render->deviceError())
            {
                STRELKA_ERROR("BENCH aborted after GPU device error");
                break;
            }
            if (playing)
            {
                playAnimations(1.0 / 60.0);
            }
            if (orbitCamera)
            {
                camera.mOrientation =
                    initialCameraOrientation * glm::angleAxis(-0.0005f * float(i + 1u), glm::vec3(0.0f, 1.0f, 0.0f));
                camera.updateViewMatrix();
            }
            if (!m_isLoading)
            {
                m_render->triggerRenderIfIdle();
            }
            const double t = m_render->getLastRenderTimeMs();
            if (t > 0.0 && t != last)
            {
                last = t;
                const auto now = std::chrono::high_resolution_clock::now();
                if (i >= warmup)
                {
                    samples.push_back(t);
                    wall.push_back(std::chrono::duration<double, std::milli>(now - prevFrame).count());
                }
                prevFrame = now;
                ++i;
                if (residencyStress)
                {
                    const uint32_t phase = i % 4;
                    const bool temporal = phase == 1;
                    const bool spatial = phase == 2 || phase == 3;
                    const float scale = phase == 0 ? 1.0f : 0.5f;
                    m_settingsManager->setAs<bool>("render/pt/denoise", temporal);
                    m_settingsManager->setAs<float>("render/pt/upscaleFactor", scale);
                    m_settingsManager->setAs<bool>("render/pt/enableUpscale", temporal || spatial);
                    applyPreviewResolution(phase == 3 ? 1280u : 1920u, phase == 3 ? 720u : 1080u);
                }
            }
            usleep(200);
        }
        std::ranges::sort(samples);
        std::ranges::sort(wall);
        Block b;
        if (!samples.empty())
        {
            b.gpu = samples[samples.size() / 2];
            b.gpuMin = samples.front();
            b.gpuMax = samples.back();
            b.wall = wall.empty() ? 0.0 : wall[wall.size() / 2];
        }
        return b;
    };

    if (envFlag("STRELKA_BENCH_RESTIR_COMPARE"))
    {
        m_settingsManager->setAs<bool>("render/pt/restirDIEnabled", false);
        const Block nee = measure(false);
        m_settingsManager->setAs<bool>("render/pt/restirDIEnabled", requestedRestir);
        const Block restir = measure(false);
        STRELKA_INFO("BENCH  NEE={:.2f} ms ReSTIR={:.2f} ms ratio={:.3f}x", nee.gpu, restir.gpu, restir.gpu / nee.gpu);
        return;
    }

    // Both states in one process, alternating.
    //
    // The GPU is shared with the compositor and whatever else is on screen, and
    // that load drifts: eight consecutive runs of the *identical* configuration
    // measured here spanned 28.7 to 59.5 ms. Comparing a play process against a
    // static process therefore reports the drift between two moments as if it
    // were the cost of playback. Alternating the two states inside one process
    // and pairing them keeps both halves of each pair in the same conditions.
    if (envFlag("STRELKA_BENCH_PAIRS"))
    {
        const uint32_t pairs = std::max(1u, envUint("STRELKA_BENCH_PAIRS", 3));
        std::vector<double> ratios, staticGpu, playGpu;
        for (uint32_t p = 0; p < pairs && !m_display->windowShouldClose(); ++p)
        {
            const Block s = measure(false);
            const Block d = measure(true);
            if (s.gpu <= 0.0 || d.gpu <= 0.0)
            {
                continue;
            }
            staticGpu.push_back(s.gpu);
            playGpu.push_back(d.gpu);
            ratios.push_back(d.gpu / s.gpu);
            // Spread as well as median: a tight band means the difference is the
            // configuration, and a bimodal one means the process latched into a
            // power state and neither median means anything.
            STRELKA_INFO(
                "BENCH  pair {}: static gpu={:.2f} [{:.0f}..{:.0f}] wall={:.2f} | "
                "play gpu={:.2f} [{:.0f}..{:.0f}] wall={:.2f} | ratio={:.2f}x",
                p, s.gpu, s.gpuMin, s.gpuMax, s.wall, d.gpu, d.gpuMin, d.gpuMax, d.wall, d.gpu / s.gpu);
        }
        if (ratios.empty())
        {
            STRELKA_INFO("BENCH  no pairs measured");
            return;
        }
        std::ranges::sort(ratios);
        std::ranges::sort(staticGpu);
        std::ranges::sort(playGpu);
        STRELKA_INFO("BENCH  PAIRED static={:.2f} ms  play={:.2f} ms  ratio={:.2f}x  (median of {} pairs)",
                     staticGpu[staticGpu.size() / 2], playGpu[playGpu.size() / 2], ratios[ratios.size() / 2],
                     ratios.size());
        return;
    }

    const Block b = measure(play);
    if (b.gpu <= 0.0)
    {
        STRELKA_INFO("BENCH  no frames measured");
        return;
    }
    STRELKA_INFO("BENCH  tracer={} depth={} frames={}  median={:.2f} ms  min={:.2f}  max={:.2f}  wall={:.2f} ms", 1u,
                 m_settingsManager->getAs<uint32_t>("render/pt/depth"), frames, b.gpu, b.gpuMin, b.gpuMax, b.wall);
}

// Jitter-sign measurement (STRELKA_JITTER_TEST=<frames>).
//
// The MetalFX header documents jitterOffsetX in two sentences that imply
// opposite signs, and our own camera flips Y, so the correct value is measured
// rather than argued. A wrong sign does not shift the converged image -- the
// Halton sequence is zero-mean -- it makes each frame's samples land where the
// scaler believes the opposite subpixel position is. The result is a
// reconstruction that never settles: it swims.
//
// So: static camera, accumulation off, one sample per frame, and the denoiser
// left to do all the work. Two numbers per configuration, both over the tail of
// the run so the history is full:
//
//   swim  -- mean |I(t) - I(t-1)| relative to mean intensity. With a static
//            camera a settled reconstruction changes only by the noise it has
//            not yet filtered, so lower is better and the differences are large.
//   sharp -- mean gradient magnitude, relative to mean intensity. A mismatched
//            jitter averages neighbouring subpixel positions together, which is
//            blur, so higher is better.
void EditorApp::runJitterTest()
{
    const uint32_t frames = std::max(16u, envUint("STRELKA_JITTER_TEST", 16));
    m_settingsManager->setAs<bool>("render/pt/enableAcc", false);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    m_settingsManager->setAs<bool>("render/pt/denoise", true);
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", true);
    if (envFlag("STRELKA_UPSCALE"))
    {
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", envFloat("STRELKA_UPSCALE", 0.5f));
    }
    if (envFlag("STRELKA_REF_DEPTH"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/depth", envUint("STRELKA_REF_DEPTH", 4));
    }

    auto capture = [&](std::vector<float>& out, uint32_t& w, uint32_t& h) {
        const double before = m_render->getLastRenderTimeMs();
        for (int i = 0; i < 4000; ++i)
        {
            m_display->pollEvents();
            m_render->triggerRenderIfIdle();
            if (m_render->getLastRenderTimeMs() != before && m_render->getReadyBuffer())
            {
                break;
            }
            usleep(300);
        }
        usleep(4000); // let the denoise and tonemap that follow the trace land
        return m_render->readDisplayTexture(out, w, h);
    };

    // Ground truth: the same view at native resolution, converged, with neither
    // scaler nor denoiser in the path. Both images come out of the display
    // texture at output resolution, so they are directly comparable -- and this is
    // the metric that separates a jitter sign from a noise-versus-blur tradeoff,
    // because a misregistered reconstruction is further from the truth no matter
    // which side of that tradeoff it sits on.
    std::vector<float> truth;
    uint32_t tw = 0, th = 0;
    {
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
        m_settingsManager->setAs<bool>("render/pt/denoise", false);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
        m_sharedCtx->mSubframeIndex = 0;
        while (m_sharedCtx->mSubframeIndex < 512 && !m_display->windowShouldClose())
        {
            m_display->pollEvents();
            m_render->triggerRenderIfIdle();
            usleep(300);
        }
        usleep(200000);
        m_render->readDisplayTexture(truth, tw, th);
        STRELKA_INFO("JITTER truth {}x{} spp={}", tw, th, (uint32_t)m_sharedCtx->mSubframeIndex);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", true);
        m_settingsManager->setAs<bool>("render/pt/denoise", true);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", false);
    }

    const char* const names[4] = { "(+x,+y)", "(-x,+y)", "(+x,-y)", "(-x,-y)" };
    for (uint32_t sign = 0; sign < 4; ++sign)
    {
        m_settingsManager->setAs<uint32_t>("render/pt/jitterSign", sign);
        m_render->resetTemporalHistory();
        m_sharedCtx->mSubframeIndex = 0;

        std::vector<float> prev, cur;
        uint32_t w = 0, h = 0;
        double swimSum = 0.0, lumSum = 0.0;
        uint32_t swimCount = 0;
        const uint32_t tail = frames / 2;
        for (uint32_t f = 0; f < frames && !m_display->windowShouldClose(); ++f)
        {
            if (!capture(cur, w, h) || cur.empty())
            {
                continue;
            }
            if (f >= tail && prev.size() == cur.size())
            {
                double d = 0.0, l = 0.0;
                for (size_t i = 0; i < cur.size(); i += 4)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        d += std::abs((double)cur[i + k] - (double)prev[i + k]);
                        l += (double)cur[i + k];
                    }
                }
                swimSum += d;
                lumSum += l;
                ++swimCount;
            }
            prev = cur;
        }

        // Sharpness of the final frame: forward differences, both axes.
        double grad = 0.0, mean = 0.0;
        if (w > 1 && h > 1 && !cur.empty())
        {
            for (uint32_t y = 0; y + 1 < h; ++y)
            {
                for (uint32_t x = 0; x + 1 < w; ++x)
                {
                    const size_t i = ((size_t)y * w + x) * 4;
                    for (int k = 0; k < 3; ++k)
                    {
                        grad += std::abs((double)cur[i + 4 + k] - (double)cur[i + k]) +
                                std::abs((double)cur[i + (size_t)w * 4 + k] - (double)cur[i + k]);
                        mean += (double)cur[i + k];
                    }
                }
            }
        }
        // Distance to the truth, and the shift at which that distance is
        // smallest. A minimum away from (0,0) is a reconstruction that sits in the
        // wrong place -- which is what a jitter sign error looks like when the
        // sequence happens not to average out.
        double bestRmse = -1.0;
        int bestDx = 0, bestDy = 0;
        if (!truth.empty() && tw == w && th == h && !cur.empty())
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    double se = 0.0;
                    size_t n = 0;
                    for (uint32_t y = 1; y + 1 < h; ++y)
                    {
                        for (uint32_t x = 1; x + 1 < w; ++x)
                        {
                            const size_t a = ((size_t)y * w + x) * 4;
                            const size_t b = ((size_t)(y + dy) * w + (x + dx)) * 4;
                            for (int k = 0; k < 3; ++k)
                            {
                                const double d = (double)cur[a + k] - (double)truth[b + k];
                                se += d * d;
                                ++n;
                            }
                        }
                    }
                    const double rmse = n ? std::sqrt(se / (double)n) : 0.0;
                    if (bestRmse < 0.0 || rmse < bestRmse)
                    {
                        bestRmse = rmse;
                        bestDx = dx;
                        bestDy = dy;
                    }
                }
            }
        }
        const double swim = (swimCount && lumSum > 0.0) ? swimSum / lumSum : -1.0;
        const double sharp = (mean > 0.0) ? grad / mean : -1.0;
        STRELKA_INFO("JITTER sign={} {}  rmse={:.5f} at shift({},{})  swim={:.5f}  sharp={:.5f}  ({}x{})", sign,
                     names[sign], bestRmse, bestDx, bestDy, swim, sharp, w, h);
    }
}

// --- Denoiser audit (STRELKA_DENOISE_AUDIT=<dir>) ---------------------------
//
// The denoiser's inputs are conventions, not derivations: which way depth grows,
// which space normals are in, which sign a jitter carries, where a motion vector
// points. Every one of them is wrong in a way that still produces a plausible
// picture, so "it looks fine" is not evidence. This measures them.
//
// Two kinds of check, and the first is the one that matters:
//
//   * Guide invariants -- read the exact textures MetalFX consumes and assert
//     things that must hold regardless of taste. A motion vector must be zero
//     when nothing moves and non-zero where something did; a normal must be unit
//     length; an albedo must be in [0,1]. These need no reference image, and they
//     name the broken input instead of reporting that the picture is worse.
//
//   * Reconstruction error -- RMSE against a converged reference, taken at the
//     best of nine one-pixel shifts. The shift is the interesting half: a
//     reconstruction that scores best somewhere other than (0,0) is sitting in
//     the wrong place, which is what a sign error looks like once the noise is
//     gone, and a plain RMSE hides it inside the noise floor.
namespace
{

struct AuditImage
{
    std::vector<float> px;
    uint32_t w = 0, h = 0;
    bool valid() const
    {
        return !px.empty() && w > 1 && h > 1 && px.size() == (size_t)w * h * 4;
    }
};

double auditMean(const AuditImage& img)
{
    double sum = 0.0;
    for (size_t i = 0; i < img.px.size(); i += 4)
        sum += img.px[i] + img.px[i + 1] + img.px[i + 2];
    const size_t channels = img.px.size() / 4 * 3;
    return img.px.empty() ? 0.0 : sum / (double)channels;
}

// RMSE at the best of nine one-pixel shifts, plus which shift won.
double auditRmse(const AuditImage& a, const AuditImage& b, int& bestDx, int& bestDy)
{
    bestDx = bestDy = 0;
    if (!a.valid() || !b.valid() || a.w != b.w || a.h != b.h)
        return -1.0;
    double best = -1.0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            double se = 0.0;
            size_t n = 0;
            for (uint32_t y = 1; y + 1 < a.h; ++y)
            {
                for (uint32_t x = 1; x + 1 < a.w; ++x)
                {
                    const size_t ia = ((size_t)y * a.w + x) * 4;
                    const size_t ib = ((size_t)(y + dy) * a.w + (x + dx)) * 4;
                    for (int k = 0; k < 3; ++k)
                    {
                        const double d = (double)a.px[ia + k] - (double)b.px[ib + k];
                        se += d * d;
                        ++n;
                    }
                }
            }
            const double rmse = n ? std::sqrt(se / (double)n) : 0.0;
            if (best < 0.0 || rmse < best)
            {
                best = rmse;
                bestDx = dx;
                bestDy = dy;
            }
        }
    }
    return best;
}

// Mean absolute frame-to-frame change, relative to intensity. With a still
// camera and a still scene a settled reconstruction only changes by the noise it
// has not filtered yet, so lower is better and the spread between right and
// wrong is large.
double auditSwim(const AuditImage& a, const AuditImage& b)
{
    if (!a.valid() || !b.valid() || a.px.size() != b.px.size())
        return -1.0;
    double d = 0.0, l = 0.0;
    for (size_t i = 0; i < a.px.size(); i += 4)
    {
        for (int k = 0; k < 3; ++k)
        {
            d += std::abs((double)a.px[i + k] - (double)b.px[i + k]);
            l += (double)a.px[i + k];
        }
    }
    return l > 0.0 ? d / l : -1.0;
}

// Mean forward-difference gradient, relative to intensity. A reconstruction that
// averages the wrong subpixel positions together is blurred, and blur shows here
// before it shows anywhere else.
double auditSharp(const AuditImage& img)
{
    if (!img.valid())
        return -1.0;
    double grad = 0.0, mean = 0.0;
    for (uint32_t y = 0; y + 1 < img.h; ++y)
    {
        for (uint32_t x = 0; x + 1 < img.w; ++x)
        {
            const size_t i = ((size_t)y * img.w + x) * 4;
            for (int k = 0; k < 3; ++k)
            {
                grad += std::abs((double)img.px[i + 4 + k] - (double)img.px[i + k]) +
                        std::abs((double)img.px[i + (size_t)img.w * 4 + k] - (double)img.px[i + k]);
                mean += (double)img.px[i + k];
            }
        }
    }
    return mean > 0.0 ? grad / mean : -1.0;
}

bool auditFinite(const AuditImage& img)
{
    for (const float v : img.px)
    {
        if (!std::isfinite(v))
            return false;
    }
    return true;
}

// Per-channel means, for telling "the light got dimmer" from "the light changed
// colour".
glm::float3 auditChannelMean(const AuditImage& img)
{
    glm::double3 sum(0.0);
    for (size_t i = 0; i < img.px.size(); i += 4)
    {
        sum.x += img.px[i];
        sum.y += img.px[i + 1];
        sum.z += img.px[i + 2];
    }
    const size_t pixelCount = img.px.size() / 4;
    const double pixels = img.px.empty() ? 1.0 : (double)pixelCount;
    return { sum / pixels };
}

} // namespace

// --- Light plumbing audit (STRELKA_LIGHT_AUDIT=[dir]) -----------------------
//
// Editing a light has to change the picture. Between the panel and the pixels
// sit the baked GPU light, a dirty bit, a buffer upload, a shader variant keyed
// on whether the scene has lights at all, and an accumulation reset -- and any
// one of them dropping the edit looks from the outside exactly like the renderer
// ignoring it. So this measures the picture: it makes the same
// Scene::setLight() call the panels make and then reads back the frame the
// screen shows.
void EditorApp::runLightAudit()
{
    const std::optional<std::string> outDir = environmentValue("STRELKA_LIGHT_AUDIT");
    const bool saveImages = outDir.has_value() && !outDir->empty() && outDir->find('/') != std::string::npos;
    const uint32_t refSpp = envUint("STRELKA_AUDIT_SPP", 32);
    const uint32_t auditW = envUint("STRELKA_AUDIT_W", 512);
    const uint32_t auditH = envUint("STRELKA_AUDIT_H", 384);
    const double budgetSec = envDouble("STRELKA_AUDIT_BUDGET", 300.0);
    const double stepTimeoutSec = envDouble("STRELKA_AUDIT_STEP_SEC", 20.0);

    const auto auditStart = std::chrono::steady_clock::now();
    auto outOfTime = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - auditStart).count() > budgetSec ||
               m_display->windowShouldClose();
    };
    auto report = [&](const std::string& line) { STRELKA_INFO("{}", line); };

    m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    m_settingsManager->setAs<bool>("render/pt/denoise", false);
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
    m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", refSpp);
    m_settingsManager->setAs<uint32_t>("render/width", auditW);
    m_settingsManager->setAs<uint32_t>("render/height", auditH);

    auto step = [&]() -> bool {
        const size_t target = m_sharedCtx->mFrameNumber + 1;
        bool submitted = false;
        auto phaseStart = std::chrono::steady_clock::now();
        while (!m_display->windowShouldClose())
        {
            m_display->pollEvents();
            if (!submitted)
            {
                m_render->triggerRenderIfIdle();
                if (m_sharedCtx->mFrameNumber >= target)
                {
                    submitted = true;
                    phaseStart = std::chrono::steady_clock::now();
                }
            }
            if (submitted && !m_render->isRenderBusy())
                return true;
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - phaseStart).count() > stepTimeoutSec)
            {
                report(fmt::format("LIGHTAUDIT WARN frame did not {} within {:.0f}s", submitted ? "complete" : "submit",
                                   stepTimeoutSec));
                return false;
            }
            usleep(200);
        }
        return false;
    };

    auto converge = [&](AuditImage& img) {
        m_sharedCtx->mSubframeIndex = 0;
        m_render->resetTemporalHistory();
        uint32_t guard = refSpp * 4 + 16;
        while (m_sharedCtx->mSubframeIndex < refSpp && guard-- > 0 && !outOfTime())
        {
            if (!step())
                break;
        }
        // Two spare frames, not one: the display texture is double buffered and
        // what mReadyIndex points at trails the trace by a frame, so a single
        // extra step can still hand back the picture of the state before this one.
        step();
        step();
        return m_render->readDisplayTexture(img.px, img.w, img.h);
    };

    auto save = [&](const std::string& name, const AuditImage& img) {
        if (!saveImages || !img.valid())
            return;
        const char* err = nullptr;
        SaveEXR(img.px.data(), (int)img.w, (int)img.h, 4, 0, (*outDir + "/" + name + ".exr").c_str(), &err);
        if (err)
            FreeEXRErrorMessage(err);
    };

    const std::vector<Scene::UniformLightDesc> original = m_scene->getLightsDesc();
    report(fmt::format("LIGHTAUDIT lights={} analytic={} tracerMode={} envMap={} spp={} {}x{}", original.size(),
                       m_settingsManager->getAs<bool>("render/validate/analyticLights"), 1u,
                       m_scene->getEnvLight().has_value(), refSpp, auditW, auditH));
    for (size_t i = 0; i < original.size(); ++i)
    {
        const Scene::UniformLightDesc& d = original[i];
        const glm::float4 baked = m_scene->getLights()[i].color;
        report(
            fmt::format("LIGHTAUDIT light{} type={} intensity={:.1f} color=({:.3f},{:.3f},{:.3f}) "
                        "baked=({:.1f},{:.1f},{:.1f}) pos=({:.3f},{:.3f},{:.3f})",
                        i, d.type, d.intensity, d.color.x, d.color.y, d.color.z, baked.x, baked.y, baked.z,
                        d.position.x, d.position.y, d.position.z));
    }
    // The GPU record, not the description: the sampling routines read these
    // points and this normal, and a light that emits nothing usually has
    // something degenerate in here.
    for (size_t i = 0; i < original.size(); ++i)
    {
        const Scene::Light& l = m_scene->getLights()[i];
        report(
            fmt::format("LIGHTAUDIT light{} gpu p0=({:.3f},{:.3f},{:.3f}) p1=({:.3f},{:.3f},{:.3f}) "
                        "p2=({:.3f},{:.3f},{:.3f}) p3=({:.3f},{:.3f},{:.3f}) n=({:.3f},{:.3f},{:.3f})",
                        i, l.points[0].x, l.points[0].y, l.points[0].z, l.points[1].x, l.points[1].y, l.points[1].z,
                        l.points[2].x, l.points[2].y, l.points[2].z, l.points[3].x, l.points[3].y, l.points[3].z,
                        l.normal.x, l.normal.y, l.normal.z));
    }
    if (original.empty())
    {
        report("LIGHTAUDIT no analytic lights in scene, nothing to measure");
        return;
    }

    AuditImage base;
    if (!converge(base) || !base.valid())
    {
        report("LIGHTAUDIT FAIL could not read the displayed frame");
        return;
    }
    const double baseMean = auditMean(base);
    report(fmt::format("LIGHTAUDIT base mean={:.5f}", baseMean));
    save("light_base", base);

    // One light at a time: a scene lit by several of them can lose one and still
    // look lit, so the per-light delta is what says the edit arrived.
    for (size_t i = 0; i < original.size() && !outOfTime(); ++i)
    {
        Scene::UniformLightDesc dark = original[i];
        dark.intensity = 0.0f;
        m_scene->setLight((uint32_t)i, dark);
        const glm::float4 baked = m_scene->getLights()[i].color;

        AuditImage img;
        converge(img);
        const double mean = auditMean(img);
        int dx = 0, dy = 0;
        const double rmse = auditRmse(base, img, dx, dy);
        report(fmt::format(
            "LIGHTAUDIT light{} intensity=0 baked=({:.1f},{:.1f},{:.1f}) mean={:.5f} delta={:+.5f} ({:+.1f}%) rmse_vs_base={:.5f}",
            i, baked.x, baked.y, baked.z, mean, mean - baseMean,
            baseMean > 0.0 ? 100.0 * (mean - baseMean) / baseMean : 0.0, rmse));
        save(fmt::format("light{}_off", i), img);

        m_scene->setLight((uint32_t)i, original[i]);
    }

    // Everything off. With no env light there is nothing left to illuminate the
    // scene, so anything but a black frame means the edits are not reaching the
    // GPU at all.
    if (!outOfTime())
    {
        for (size_t i = 0; i < original.size(); ++i)
        {
            Scene::UniformLightDesc dark = original[i];
            dark.intensity = 0.0f;
            m_scene->setLight((uint32_t)i, dark);
        }
        AuditImage img;
        converge(img);
        const double mean = auditMean(img);
        int dx = 0, dy = 0;
        report(fmt::format("LIGHTAUDIT all off mean={:.5f} ({:.2f}% of base) rmse_vs_base={:.5f}", mean,
                           baseMean > 0.0 ? 100.0 * mean / baseMean : 0.0, auditRmse(base, img, dx, dy)));
        save("light_all_off", img);
    }

    // Back to the start: an edit that cannot be undone is as broken as one that
    // never arrives.
    if (!outOfTime())
    {
        for (size_t i = 0; i < original.size(); ++i)
        {
            m_scene->setLight((uint32_t)i, original[i]);
        }
        AuditImage img;
        converge(img);
        const double mean = auditMean(img);
        report(fmt::format("LIGHTAUDIT restored mean={:.5f} (base {:.5f}, {:+.2f}%)", mean, baseMean,
                           baseMean > 0.0 ? 100.0 * (mean - baseMean) / baseMean : 0.0));
        save("light_restored", img);
    }

    // Colour, size and position each reach the GPU by a different route: colour
    // multiplies the baked radiance, the other two rebuild the light's geometry
    // and its instance transform.
    //
    // Saturated against nearly saturated, because the two must agree. A test that
    // only sets pure red cannot tell "the colour arrived" from "the light was
    // dropped for having a zero channel", which is what the shader used to do.
    if (!outOfTime())
    {
        Scene::UniformLightDesc red = original[0];
        red.color = glm::float3(1.0f, 0.0f, 0.0f);
        m_scene->setLight(0, red);
        AuditImage pure;
        converge(pure);
        const glm::float3 pureRgb = auditChannelMean(pure);
        save("light0_red", pure);

        Scene::UniformLightDesc almost = original[0];
        almost.color = glm::float3(1.0f, 0.02f, 0.02f);
        m_scene->setLight(0, almost);
        AuditImage nearly;
        converge(nearly);
        const glm::float3 nearlyRgb = auditChannelMean(nearly);
        save("light0_almost_red", nearly);

        report(
            fmt::format("LIGHTAUDIT light0 red=({:.5f},{:.5f},{:.5f}) almost_red=({:.5f},{:.5f},{:.5f}) "
                        "red_r/almost_r={:.3f}",
                        pureRgb.x, pureRgb.y, pureRgb.z, nearlyRgb.x, nearlyRgb.y, nearlyRgb.z,
                        nearlyRgb.x > 0.0f ? pureRgb.x / nearlyRgb.x : 0.0f));
        m_scene->setLight(0, original[0]);
    }

    if (!outOfTime())
    {
        Scene::UniformLightDesc moved = original[0];
        moved.position += glm::float3(0.0f, 2.0f, 0.0f);
        m_scene->setLight(0, moved);
        AuditImage img;
        converge(img);
        int dx = 0, dy = 0;
        const double rmse = auditRmse(base, img, dx, dy);
        report(fmt::format("LIGHTAUDIT light0 moved +2y mean={:.5f} rmse_vs_base={:.5f}", auditMean(img), rmse));
        save("light0_moved", img);
        m_scene->setLight(0, original[0]);
    }

    if (!outOfTime() && original[0].type == LIGHT_TYPE_RECT)
    {
        Scene::UniformLightDesc big = original[0];
        big.width *= 3.0f;
        big.height *= 3.0f;
        m_scene->setLight(0, big);
        AuditImage img;
        converge(img);
        report(fmt::format(
            "LIGHTAUDIT light0 3x size mean={:.5f} delta={:+.5f}", auditMean(img), auditMean(img) - baseMean));
        save("light0_big", img);
        m_scene->setLight(0, original[0]);
    }

    // NEE against BSDF sampling alone, per light type. The two are independent
    // unbiased estimators of the same integral, so at convergence they have to
    // produce the same image -- and they only do if the light's sampling routine
    // and the pdf its MIS weight uses describe the same shape the light's
    // geometry has. A light type the sampler does not handle shows up here: NEE
    // contributes nothing for it, or contributes with the wrong weight.
    //
    // Needs samples to mean anything. BSDF sampling finds a small bright light by
    // accident, so at the default spp it is nowhere near converged and reads 7%
    // dark on scenes where the two agree to within 1% at STRELKA_AUDIT_SPP=400.
    for (size_t i = 0; i < original.size() && !outOfTime(); ++i)
    {
        for (size_t j = 0; j < original.size(); ++j)
        {
            Scene::UniformLightDesc d = original[j];
            if (j != i)
                d.intensity = 0.0f;
            m_scene->setLight((uint32_t)j, d);
        }

        m_settingsManager->setAs<uint32_t>("render/validate/estimatorMode", 0);
        AuditImage nee;
        converge(nee);
        const double neeMean = auditMean(nee);

        m_settingsManager->setAs<uint32_t>("render/validate/estimatorMode", 1);
        AuditImage bsdf;
        converge(bsdf);
        const double bsdfMean = auditMean(bsdf);

        m_settingsManager->setAs<uint32_t>("render/validate/estimatorMode", 0);
        report(fmt::format("LIGHTAUDIT light{} type={} alone nee={:.5f} bsdfOnly={:.5f} diff={:+.2f}%", i,
                           original[i].type, neeMean, bsdfMean,
                           bsdfMean > 0.0 ? 100.0 * (neeMean - bsdfMean) / bsdfMean : 0.0));
        save(fmt::format("light{}_nee", i), nee);
        save(fmt::format("light{}_bsdf", i), bsdf);
    }

    for (size_t i = 0; i < original.size(); ++i)
    {
        m_scene->setLight((uint32_t)i, original[i]);
    }
    report("LIGHTAUDIT done");
}

// Paused motion blur (STRELKA_PAUSE_BLUR=1).
//
// Pausing playback mid-clip should freeze a frame *of the film*, and a frame of
// the film has motion blur in it; the estimator then keeps refining that frame.
// Two separate claims, so two separate measurements, and the interesting one is
// easy to miss: an image that converges proves only that the renderer is still
// running, not that the blur survived the pause. A crisp still converges too.
//
//   present  -- the held frame against the same pose rendered with motion blur
//               off. Blur smears the moving parts, so it lowers the gradient
//               magnitude; the no-blur render of the identical pose is the
//               control that says how sharp the frame would be without it.
//   converge -- the displayed mean over the hold, and the RMSE between the first
//               held frame and the last. Refining a frozen frame moves it toward
//               its own mean; it must not empty it or keep moving forever.
void EditorApp::runPauseBlurCheck()
{
    const uint32_t playFrames = envUint("STRELKA_PAUSE_BLUR_PLAY", 90);
    const uint32_t holdFrames = envUint("STRELKA_PAUSE_BLUR_HOLD", 40);
    const double stepTimeoutSec = envDouble("STRELKA_AUDIT_STEP_SEC", 10.0);
    auto report = [&](const std::string& line) { STRELKA_INFO("{}", line); };

    if (m_scene->getAnimations().empty())
    {
        report("PAUSEBLUR skipped: the scene has no animations");
        return;
    }

    m_settingsManager->setAs<uint32_t>("render/width", 512);
    m_settingsManager->setAs<uint32_t>("render/height", 384);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
    m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", 4096);
    m_settingsManager->setAs<bool>("render/pt/denoise", false);
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);

    auto step = [&]() -> bool {
        const size_t target = m_sharedCtx->mFrameNumber + 1;
        bool submitted = false;
        auto phaseStart = std::chrono::steady_clock::now();
        while (!m_display->windowShouldClose())
        {
            m_display->pollEvents();
            if (!submitted)
            {
                m_render->triggerRenderIfIdle();
                if (m_sharedCtx->mFrameNumber >= target)
                {
                    submitted = true;
                    phaseStart = std::chrono::steady_clock::now();
                }
            }
            if (submitted && !m_render->isRenderBusy())
            {
                return true;
            }
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - phaseStart).count() > stepTimeoutSec)
            {
                report("PAUSEBLUR WARN frame did not land in time");
                return false;
            }
            usleep(200);
        }
        return false;
    };
    auto shown = [&](AuditImage& img) { return m_render->readDisplayTexture(img.px, img.w, img.h); };
    auto setPlaying = [&](bool on) {
        for (size_t i = 0; i < m_scene->getAnimations().size(); ++i)
        {
            m_settingsManager->setAs<bool>(animationStateKey(i), on);
        }
    };

    // --- play, then pause and hold, with motion blur on -------------------
    m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
    m_render->resetTemporalHistory();
    for (int i = 0; i < 4; ++i)
    {
        step();
    }
    setPlaying(true);
    for (uint32_t i = 0; i < playFrames; ++i)
    {
        playAnimations(1.0f / 60.0f);
        if (!step())
        {
            break;
        }
    }
    setPlaying(false);

    // The pose the hold is sitting on. Everything below has to be compared at
    // this same time or the control is measuring a different frame.
    std::vector<float> heldTimes;
    for (auto& anim : m_scene->getAnimations())
    {
        heldTimes.push_back(anim.current);
    }

    // Convergence is the *rate* the picture is still changing, not how far it
    // has come. The first held frame carries one sample and the last carries
    // hundreds, so the distance between them is large precisely when the
    // estimator is working; comparing those two would call a converging frame
    // unstable. What has to shrink is the step between consecutive frames.
    AuditImage firstHeld, lastHeld, prevHeld;
    double meanFirst = -1.0, meanLast = -1.0;
    double earlyStep = -1.0, lateStep = -1.0;
    uint32_t heldSeen = 0;
    std::string trace;
    for (uint32_t i = 0; i < holdFrames; ++i)
    {
        if (!step())
        {
            break;
        }
        AuditImage img;
        if (!shown(img) || !img.valid())
        {
            continue;
        }
        const double m = auditMean(img);
        if (meanFirst < 0.0)
        {
            meanFirst = m;
            firstHeld = img;
        }
        meanLast = m;
        if (i % 8 == 0)
        {
            trace += fmt::format("{:.4f} ", m);
        }
        if (prevHeld.valid())
        {
            int sdx = 0, sdy = 0;
            const double stepRmse = auditRmse(prevHeld, img, sdx, sdy);
            // Second consecutive pair as the early reference: the very first
            // step still carries the reset of the accumulator behind it.
            if (heldSeen == 2)
            {
                earlyStep = stepRmse;
            }
            lateStep = stepRmse;
        }
        ++heldSeen;
        prevHeld = img;
        lastHeld = std::move(img);
    }
    const double sharpBlur = auditSharp(lastHeld);

    // --- the control: same pose, motion blur off --------------------------
    m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    for (size_t i = 0; i < m_scene->getAnimations().size() && i < heldTimes.size(); ++i)
    {
        m_settingsManager->setAs<float>(animationTimeKey(i), heldTimes[i]);
    }
    m_render->resetTemporalHistory();
    AuditImage crisp;
    for (uint32_t i = 0; i < holdFrames; ++i)
    {
        if (!step())
        {
            break;
        }
        AuditImage img;
        if (shown(img) && img.valid())
        {
            crisp = std::move(img);
        }
    }
    const double sharpCrisp = auditSharp(crisp);

    report(fmt::format("PAUSEBLUR held mean over the pause: {}", trace));
    report(fmt::format(
        "PAUSEBLUR mean first={:.4f} last={:.4f} ({:+.1f}%)  frame-to-frame step "
        "early={:.5f} late={:.5f}",
        meanFirst, meanLast, meanFirst > 0.0 ? 100.0 * (meanLast - meanFirst) / meanFirst : 0.0, earlyStep, lateStep));
    report(fmt::format("PAUSEBLUR sharpness held(blur)={:.5f}  same pose(no blur)={:.5f}  ({:+.1f}%)", sharpBlur,
                       sharpCrisp, sharpCrisp > 0.0 ? 100.0 * (sharpBlur - sharpCrisp) / sharpCrisp : 0.0));

    if (meanFirst > 0.0 && meanLast > 0.0 && meanLast < meanFirst * 0.75)
    {
        report(fmt::format("PAUSEBLUR FAIL the held frame faded ({:.4f} -> {:.4f})", meanFirst, meanLast));
    }
    if (earlyStep > 0.0 && lateStep > 0.0 && lateStep >= earlyStep)
    {
        report(
            fmt::format("PAUSEBLUR FAIL the held frame is not converging: the step between "
                        "consecutive frames did not shrink ({:.5f} -> {:.5f})",
                        earlyStep, lateStep));
    }
    if (sharpBlur > 0.0 && sharpCrisp > 0.0 && sharpBlur >= sharpCrisp)
    {
        report(
            fmt::format("PAUSEBLUR FAIL the held frame is no softer than the unblurred pose "
                        "({:.5f} vs {:.5f}) -- the blur did not survive the pause",
                        sharpBlur, sharpCrisp));
    }
    report("PAUSEBLUR done");
}

void EditorApp::runDenoiseAudit()
{
    const std::optional<std::string> outDir = environmentValue("STRELKA_DENOISE_AUDIT");
    const bool saveImages = outDir.has_value() && !outDir->empty() && outDir->find('/') != std::string::npos;
    // The reference is bounded by samples, not by time: past a couple of hundred
    // the estimator has converged and the renderer is only re-showing the same
    // picture, so waiting longer buys nothing.
    const uint32_t refSpp = envUint("STRELKA_AUDIT_SPP", 256);
    const uint32_t frames = envUint("STRELKA_AUDIT_FRAMES", 12);
    // A quarter of the editor's default pixel count. Every question here is about
    // whether a guide means what the denoiser thinks it means, and none of them
    // need a big image -- while the whole audit has to fit in the time a person
    // is willing to sit in front of it.
    const uint32_t auditW = envUint("STRELKA_AUDIT_W", 512);
    const uint32_t auditH = envUint("STRELKA_AUDIT_H", 384);
    const float upscale = envFloat("STRELKA_UPSCALE", 0.5f);
    // Wall-clock budgets. The audit drives the renderer through states it is not
    // known to survive -- that is the point of it -- so it must not be able to
    // wait forever on a frame that is never coming, and it must report whatever
    // it already measured when it runs out of time rather than nothing at all.
    const double budgetSec = envDouble("STRELKA_AUDIT_BUDGET", 90.0);
    const double stepTimeoutSec = envDouble("STRELKA_AUDIT_STEP_SEC", 10.0);
    const auto auditStart = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - auditStart).count(); };
    // Distinguished on purpose: a closed window means the run was abandoned and
    // its numbers are partial, which is a different statement from having spent
    // the budget.
    auto outOfTime = [&]() { return elapsed() > budgetSec || m_display->windowShouldClose(); };

    // The logger is flushed per line (flush_on(trace) in Logmanager), which is
    // what a harness whose job is to walk the renderer into states that might
    // abort needs: a number that only exists in a buffer nobody flushed is a
    // number that was never measured.
    auto report = [&](const std::string& line) { STRELKA_INFO("{}", line); };

    // Deterministic conditions for everything below. Motion blur off: it makes
    // the shutter, and therefore the pose the frame is rendered at, depend on
    // playback state, and the audit drives time by hand.
    m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    m_settingsManager->setAs<float>("render/pt/upscaleFactor", upscale);
    m_settingsManager->setAs<uint32_t>("render/width", auditW);
    m_settingsManager->setAs<uint32_t>("render/height", auditH);
    if (envFlag("STRELKA_RESTIR_DI"))
    {
        m_settingsManager->setAs<bool>("render/pt/restirDIEnabled", true);
    }

    auto& camera = m_scene->getCamera(m_selectedCamera);

    // --- frame pump ---------------------------------------------------------
    // One frame, submitted and landed. isRenderBusy() is what makes this exact:
    // sleeping a guessed interval instead would occasionally read the frame
    // before it, and every number here would carry that as noise.
    // Two phases, timed separately. triggerRenderIfIdle() encodes the frame on
    // this thread, and the first frame of a new configuration pays for MetalFX
    // building its network synchronously -- seconds, legitimately. Timing that
    // against a "did the frame land" budget reports a failure for work that was
    // proceeding normally, and every measurement taken afterwards is then of a
    // stale texture. Only the wait for the GPU is held to the timeout.
    auto step = [&]() -> bool {
        const size_t target = m_sharedCtx->mFrameNumber + 1;
        bool submitted = false;
        auto phaseStart = std::chrono::steady_clock::now();
        while (!m_display->windowShouldClose())
        {
            m_display->pollEvents();
            // Only until this frame is away. Triggering again on every iteration
            // starts the next frame the instant this one lands, so the renderer
            // is never observed idle and every read below sees whichever frame
            // happened to be finished -- not the one whose inputs were just set.
            if (!submitted)
            {
                m_render->triggerRenderIfIdle();
                if (m_sharedCtx->mFrameNumber >= target)
                {
                    submitted = true;
                    phaseStart = std::chrono::steady_clock::now();
                }
            }
            if (submitted && !m_render->isRenderBusy())
                return true;
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - phaseStart).count() > stepTimeoutSec)
            {
                report(fmt::format("AUDIT WARN frame did not {} within {:.0f}s (frame={} lastGpu={:.1f}ms)",
                                   submitted ? "complete" : "submit", stepTimeoutSec, m_sharedCtx->mFrameNumber,
                                   m_render->getLastRenderTimeMs()));
                return false;
            }
            usleep(200);
        }
        return false;
    };
    auto shown = [&](AuditImage& img) { return m_render->readDisplayTexture(img.px, img.w, img.h); };
    auto guide = [&](Render::Guide g, AuditImage& img) { return m_render->readGuideTexture(g, img.px, img.w, img.h); };

    auto setDenoise = [&](bool on) {
        m_settingsManager->setAs<bool>("render/pt/denoise", on);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", on);
        // MetalFX owns temporal reconstruction while this branch is active.
        // Keeping the PT accumulator enabled here used to hide frame-stream
        // defects by feeding its converged mean to the denoiser.
        m_settingsManager->setAs<bool>("render/pt/enableAcc", !on);
        m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", refSpp);
    };

    // Converged, at native resolution, with neither scaler nor denoiser in the
    // path: the picture the reconstruction is trying to be.
    auto converge = [&](AuditImage& img) {
        setDenoise(false);
        m_sharedCtx->mSubframeIndex = 0;
        m_render->resetTemporalHistory();
        // Bounded by samples *and* by frames: the estimator restarts its
        // accumulation on any change it considers material, and a reference that
        // keeps being restarted would otherwise never finish.
        uint32_t guard = refSpp * 4 + 16;
        while (m_sharedCtx->mSubframeIndex < refSpp && guard-- > 0 && !outOfTime())
        {
            if (!step())
                break;
        }
        step(); // one more so the tonemap of the final accumulation lands
        return shown(img);
    };

    // EXR for measuring, PNG for looking at. Every image here comes from the
    // display texture, which is already tone mapped, so the PNG is a clamp rather
    // than a second tone curve -- and without one there is nothing on this machine
    // that can open the results.
    auto save = [&](const char* name, const AuditImage& img) {
        if (!saveImages || !img.valid())
            return;
        const char* err = nullptr;
        SaveEXR(img.px.data(), (int)img.w, (int)img.h, 4, 0, (*outDir + "/" + name + ".exr").c_str(), &err);
        if (err)
            FreeEXRErrorMessage(err);
        const size_t pixelCount = (size_t)img.w * img.h;
        std::vector<uint8_t> bytes(pixelCount * 4);
        for (size_t i = 0; i < pixelCount * 4; ++i)
        {
            bytes[i] = (uint8_t)std::lround(std::clamp(img.px[i], 0.0f, 1.0f) * 255.0f);
        }
        stbi_write_png((*outDir + "/" + name + ".png").c_str(), (int)img.w, (int)img.h, 4, bytes.data(), (int)img.w * 4);
    };

    // Yaw orbit about whatever the scene's own camera is already looking at.
    //
    // Deriving the pose from the vertex bounds instead looks reasonable and is
    // wrong: getVertices() is object space, so on any scene that places its
    // meshes with a node transform the camera ends up pointing at nothing. The
    // distance to the subject is therefore measured -- one frame, median of the
    // finite depths -- rather than assumed, which works on any scene without
    // knowing anything about it.
    const glm::vec3 startPos = camera.position;
    const glm::quat startOrient = camera.mOrientation;
    const glm::vec3 startFront = glm::conjugate(startOrient) * glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 orbitTarget = startPos + startFront;
    auto setOrbit = [&](float theta) {
        // view = rot(mOrientation) * translate(-p) and getFront() conjugates, so
        // the stored quaternion is the inverse of the camera's world rotation.
        const glm::quat yaw = glm::angleAxis(theta, glm::vec3(0.0f, 1.0f, 0.0f));
        camera.position = orbitTarget + yaw * (startPos - orbitTarget);
        camera.mOrientation = startOrient * glm::angleAxis(-theta, glm::vec3(0.0f, 1.0f, 0.0f));
        camera.updateViewMatrix();
    };

    const size_t animCount = m_scene->getAnimations().size();
    auto setAnimTime = [&](float t01) {
        for (size_t i = 0; i < animCount; ++i)
        {
            const auto& a = m_scene->getAnimations()[i];
            m_settingsManager->setAs<float>(animationTimeKey(i), a.start + (a.end - a.start) * t01);
        }
    };

    report(fmt::format("AUDIT scene animations={} {}x{} refSpp={} frames={} upscale={:.2f} budget={:.0f}s", animCount,
                       auditW, auditH, refSpp, frames, upscale, budgetSec));

    // === Guide invariants ===================================================
    setOrbit(0.0f);
    setAnimTime(0.0f);
    setDenoise(true);
    m_render->resetTemporalHistory();
    for (int i = 0; i < 4; ++i)
        step();

    // Calibrate the orbit distance from what the camera can actually see.
    {
        AuditImage d;
        std::vector<float> finite;
        if (guide(Render::Guide::Depth, d))
        {
            for (size_t i = 0; i < d.px.size(); i += 4)
            {
                if (d.px[i] > 0.0f && d.px[i] < 1e6f)
                    finite.push_back(d.px[i]);
            }
        }
        if (finite.size() > d.px.size() / 40) // at least 10% of the frame is geometry
        {
            std::ranges::sort(finite);
            orbitTarget = startPos + startFront * finite[finite.size() / 2];
        }
        const size_t depthPixels = d.px.size() / 4;
        report(fmt::format("AUDIT orbit target at {:.2f} units ({:.1f}% of frame is geometry)",
                           glm::length(orbitTarget - startPos),
                           d.px.empty() ? 0.0 : 100.0 * (double)finite.size() / (double)depthPixels));
        setOrbit(0.0f);
        step();
    }

    // === Play, then stop ====================================================
    //
    // Stopping playback should freeze a frame of the film and go on refining it.
    // What it must not do is lose the character. The plain estimator folds new
    // samples into its accumulation buffer, while MetalFX carries temporal
    // history of its per-frame stream; either can fade a stale pose over several
    // frames. That is why this measures the trend across the hold rather than
    // the first frame after it, and why the two-second play matters: held from
    // the start, the pose keyframes are identical and nothing under test is even
    // reachable.
    if (animCount > 0)
    {
        for (int denoiseOn = 0; denoiseOn < 2 && !outOfTime(); ++denoiseOn)
        {
            setOrbit(0.0f);
            setAnimTime(0.0f);
            setDenoise(denoiseOn != 0);
            m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
            m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", refSpp);
            m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
            m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
            m_render->resetTemporalHistory();
            for (int i = 0; i < 4 && !outOfTime(); ++i)
            {
                step();
            }

            for (size_t i = 0; i < animCount; ++i)
            {
                m_settingsManager->setAs<bool>(animationStateKey(i), true);
            }
            for (int i = 0; i < 120 && !outOfTime(); ++i)
            {
                playAnimations(1.0f / 60.0f);
                if (!step())
                {
                    break;
                }
            }
            AuditImage playing;
            shown(playing);
            save(denoiseOn ? "stop_denoise_playing" : "stop_plain_playing", playing);
            for (size_t i = 0; i < animCount; ++i)
            {
                m_settingsManager->setAs<bool>(animationStateKey(i), false);
            }

            // The hold. Nothing is touched from here on: not the time, not the
            // camera, not a setting.
            const float extentAtStop = m_render->skinnedGeometryExtent();
            AuditImage firstHeld, lastHeld;
            double meanFirst = -1.0;
            double meanLast = -1.0;
            std::string trace;
            for (int i = 0; i < 60 && !outOfTime(); ++i)
            {
                if (!step())
                {
                    break;
                }
                AuditImage img;
                if (!shown(img) || !img.valid())
                {
                    continue;
                }
                const double m = auditMean(img);
                if (meanFirst < 0.0)
                {
                    meanFirst = m;
                    firstHeld = img;
                }
                meanLast = m;
                if (i % 10 == 0)
                {
                    trace += fmt::format("{:.3f} ", m);
                }
                lastHeld = std::move(img);
            }
            const float extentHeld = m_render->skinnedGeometryExtent();
            int hdx = 0, hdy = 0;
            const double drift =
                (firstHeld.valid() && lastHeld.valid()) ? auditRmse(firstHeld, lastHeld, hdx, hdy) : -1.0;
            const char* label = denoiseOn ? "stop denoise" : "stop plain  ";
            report(fmt::format("AUDIT {} held mean across the stop: {}", label, trace));
            report(fmt::format(
                "AUDIT {} mean first={:.4f} last={:.4f} ({:+.1f}%)  drift={:.5f}  "
                "skinned extent {:.3f} -> {:.3f}",
                label, meanFirst, meanLast, meanFirst > 0.0 ? 100.0 * (meanLast - meanFirst) / meanFirst : 0.0, drift,
                extentAtStop, extentHeld));
            save(denoiseOn ? "stop_denoise_first" : "stop_plain_first", firstHeld);
            save(denoiseOn ? "stop_denoise_last" : "stop_plain_last", lastHeld);
            // Refining a frozen frame moves the estimate towards its own mean. It
            // does not empty the frame.
            if (meanFirst > 0.0 && meanLast > 0.0 && meanLast < meanFirst * 0.75)
            {
                report(fmt::format("AUDIT FAIL {} the held frame faded ({:.4f} -> {:.4f})", label, meanFirst, meanLast));
            }
            if (extentAtStop > 1e-3f && extentHeld < extentAtStop * 0.25f)
            {
                report(fmt::format("AUDIT FAIL {} skinned geometry collapsed while held ({:.3f} -> {:.3f})", label,
                                   extentAtStop, extentHeld));
            }
        }
        m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    }

    auto reportGuide = [&](const char* label, Render::Guide g, float lo, float hi, int channels) {
        AuditImage img;
        if (!guide(g, img))
        {
            report(fmt::format("AUDIT FAIL guide {:14s} UNAVAILABLE", label));
            return;
        }
        size_t bad = 0, nonFinite = 0;
        float mn = std::numeric_limits<float>::max(), mx = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < img.px.size(); i += 4)
        {
            for (int k = 0; k < channels; ++k)
            {
                const float v = img.px[i + k];
                if (!std::isfinite(v))
                {
                    ++nonFinite;
                    continue;
                }
                mn = std::min(mn, v);
                mx = std::max(mx, v);
                if (v < lo || v > hi)
                    ++bad;
            }
        }
        report(fmt::format("AUDIT guide {:14s} {}x{}  range[{:.4f},{:.4f}]  outside[{:.2f},{:.2f}]={}  nonfinite={}",
                           label, img.w, img.h, mn, mx, lo, hi, bad, nonFinite));
    };
    reportGuide("color", Render::Guide::Color, 0.0f, 1e9f, 3);
    reportGuide("depth", Render::Guide::Depth, 0.0f, 1e9f, 1);
    reportGuide("diffuseAlbedo", Render::Guide::DiffuseAlbedo, 0.0f, 1.0f, 3);
    reportGuide("specAlbedo", Render::Guide::SpecularAlbedo, 0.0f, 1.0f, 3);
    reportGuide("roughness", Render::Guide::Roughness, 0.0f, 1.0f, 1);
    reportGuide("specHitDistance", Render::Guide::SpecularHitDistance, 0.0f, 1e9f, 1);
    reportGuide("reactive", Render::Guide::Reactive, 0.0f, 1.0f, 1);
    // How much of the frame the mask tells the denoiser to treat as unreliable.
    // A range of [0,1] says nothing about this: a mask that is 1 everywhere and
    // one that is 1 on a handful of pixels report the same range, and the first
    // means no temporal accumulation happens at all.
    {
        AuditImage r;
        if (guide(Render::Guide::Reactive, r) && !r.px.empty())
        {
            double sum = 0.0;
            size_t high = 0, n = 0;
            for (size_t i = 0; i < r.px.size(); i += 4)
            {
                sum += r.px[i];
                if (r.px[i] > 0.5f)
                    ++high;
                ++n;
            }
            report(fmt::format("AUDIT guide reactive       mean={:.3f}  above 0.5={:.1f}% of the frame",
                               n ? sum / (double)n : 0.0, n ? 100.0 * (double)high / (double)n : 0.0));
        }
    }
    {
        AuditImage n;
        if (guide(Render::Guide::Normal, n))
        {
            size_t offUnit = 0;
            for (size_t i = 0; i < n.px.size(); i += 4)
            {
                const double len = std::sqrt((double)n.px[i] * n.px[i] + (double)n.px[i + 1] * n.px[i + 1] +
                                             (double)n.px[i + 2] * n.px[i + 2]);
                if (len < 0.9 || len > 1.1)
                    ++offUnit;
            }
            report(fmt::format("AUDIT guide {:14s} non-unit={} / {}", "normal", offUnit, n.px.size() / 4));
        }
    }

    // Motion vectors, the three cases that decide whether the denoiser can
    // reproject anything at all.
    auto motionStats = [&](const char* label, size_t& nonZero, size_t& total) {
        AuditImage mv;
        nonZero = total = 0;
        if (!guide(Render::Guide::Motion, mv))
        {
            report(fmt::format("AUDIT FAIL motion {:22s} UNAVAILABLE", label));
            return AuditImage{};
        }
        double maxMag = 0.0;
        size_t maxIdx = 0;
        // Bucketed, because "max" alone cannot tell one broken pixel from a
        // broken frame -- and the two want completely different investigations.
        size_t over1 = 0, over10 = 0, over1000 = 0;
        for (size_t i = 0; i < mv.px.size(); i += 4)
        {
            const double m = std::abs((double)mv.px[i]) + std::abs((double)mv.px[i + 1]);
            if (m > maxMag)
            {
                maxMag = m;
                maxIdx = i / 4;
            }
            if (m > 0.05)
                ++nonZero;
            if (m > 1.0)
                ++over1;
            if (m > 10.0)
                ++over10;
            if (m > 1000.0)
                ++over1000;
            ++total;
        }
        const double pct = total ? 100.0 / (double)total : 0.0;
        report(
            fmt::format("AUDIT motion {:22s} nonzero={:.1f}%  >1px={:.1f}%  >10px={:.1f}%  >1000px={:.2f}%  "
                        "max={:.2f} px at ({},{}) of {}x{}",
                        label, (double)nonZero * pct, (double)over1 * pct, (double)over10 * pct, (double)over1000 * pct,
                        maxMag, mv.w ? maxIdx % mv.w : 0, mv.w ? maxIdx / mv.w : 0, mv.w, mv.h));
        return mv;
    };

    size_t nz = 0, tot = 0;
    motionStats("still camera+scene", nz, tot);
    if (nz * 200 > tot) // more than 0.5% moving when nothing does
        report("AUDIT FAIL motion vectors non-zero with nothing moving");

    // Camera moved: everything on screen must move, background included.
    AuditImage beforeCam, afterCam;
    guide(Render::Guide::Color, beforeCam);
    setOrbit(0.05f);
    step();
    motionStats("camera moved", nz, tot);
    if (nz * 4 < tot * 3) // fewer than 75% of pixels carrying motion
        report("AUDIT FAIL camera motion missing from motion vectors (background?)");

    // Scene moved, camera still: the pixels whose shading changed are exactly the
    // ones that must carry a motion vector. Counting the ones that changed and
    // did not is a ghosting predictor that needs no reference image.
    if (animCount > 0)
    {
        setOrbit(0.0f);
        setAnimTime(0.0f);
        m_render->resetTemporalHistory();
        for (int i = 0; i < 3; ++i)
            step();
        AuditImage a0;
        guide(Render::Guide::Depth, a0);
        // Control: one frame with nothing changed at all. The primary hit still
        // lands somewhere slightly different every frame because the jitter moves
        // it, so "depth differs" on its own marks the whole image as moving. The
        // control measures that floor and the test clears it.
        step();
        AuditImage ctrl;
        guide(Render::Guide::Depth, ctrl);
        // The 99th percentile, not the maximum: a pixel on a silhouette flips
        // between the surface and the background as the jitter moves it, and that
        // one pixel would otherwise set a threshold nothing can clear.
        std::vector<double> floorDeltas;
        if (a0.valid() && ctrl.valid())
        {
            for (size_t i = 0; i < ctrl.px.size(); i += 4)
            {
                const double d0 = (double)a0.px[i], d1 = (double)ctrl.px[i];
                if (d0 <= 0.0 || d0 >= 1e6 || d1 <= 0.0 || d1 >= 1e6)
                    continue;
                floorDeltas.push_back(std::abs(d1 - d0) / d0);
            }
            std::ranges::sort(floorDeltas);
        }
        const double floorDelta =
            floorDeltas.empty() ? 0.0 : floorDeltas[static_cast<size_t>(static_cast<double>(floorDeltas.size()) * 0.99)];
        const double movedThreshold = std::max(4.0 * floorDelta, 0.01);

        // A step the renderer still calls playback. Animation time is normalised
        // here, so what this costs in seconds depends on the clip -- and a jump
        // past 5% of one is a cut, which drops the previous pose on purpose and
        // hands back a frame with no motion vectors anywhere. At 0.08 of a
        // 35-second clip this test was measuring the cut path.
        setAnimTime(0.01f);
        step();
        AuditImage a1, mv;
        guide(Render::Guide::Depth, a1);
        guide(Render::Guide::Motion, mv);
        // Surface pixels, not changed ones. The mesh deforms mostly across the
        // view rather than along it, so a depth threshold selects the silhouette
        // -- and half of the silhouette is sky in one of the two frames, where a
        // still camera correctly writes no motion at all. Counting those reported
        // 100% of a deforming character as missing its motion vectors on every
        // run, on a renderer whose vectors were fine.
        size_t surface = 0, withMv = 0, silhouette = 0, deformed = 0;
        double sumMv = 0.0, maxMv = 0.0;
        if (a0.valid() && a1.valid() && a1.px.size() == mv.px.size())
        {
            for (size_t i = 0; i < a1.px.size(); i += 4)
            {
                // Depth, not colour: colour also changes with the noise seed.
                const double d0 = (double)a0.px[i], d1 = (double)a1.px[i];
                const bool geo0 = d0 > 0.0 && d0 < 1e6;
                const bool geo1 = d1 > 0.0 && d1 < 1e6;
                if (geo0 != geo1)
                {
                    ++silhouette; // the mesh moved on or off this pixel
                    continue;
                }
                if (!geo1)
                    continue; // background in both
                ++surface;
                if (std::abs(d1 - d0) / d0 >= movedThreshold)
                    ++deformed;
                const double m = std::abs((double)mv.px[i]) + std::abs((double)mv.px[i + 1]);
                sumMv += m;
                maxMv = std::max(maxMv, m);
                if (m > 0.25)
                    ++withMv;
            }
        }
        report(fmt::format(
            "AUDIT motion {:22s} jitter floor={:.4f}, threshold={:.4f}", "scene moved", floorDelta, movedThreshold));
        report(fmt::format("AUDIT motion {:22s} surface={} px, silhouette moved={} px, depth moved={} px",
                           "scene moved", surface, silhouette, deformed));
        report(fmt::format("AUDIT motion {:22s} with motion vector={:.1f}%  mean={:.2f} px  max={:.2f} px",
                           "scene moved", surface ? 100.0 * (double)withMv / (double)surface : 0.0,
                           surface ? sumMv / (double)surface : 0.0, maxMv));
        // Nothing to carry is not the same as failing to carry it: an animation
        // that happens to hold still over this step cannot be measured.
        if (surface && silhouette == 0 && deformed == 0)
            report("AUDIT motion       scene moved            the pose did not change, nothing to measure");
        else if (surface && withMv * 2 < surface)
            report("AUDIT FAIL deforming geometry has no motion vectors");
    }

    // === S1 static camera ===================================================
    auto reconstruct = [&](const char* label, float thetaFrom, float thetaTo, float animFrom, float animTo,
                           AuditImage& last) {
        setDenoise(true);
        setOrbit(thetaFrom);
        setAnimTime(animFrom);
        m_render->resetTemporalHistory();
        // Fill the history before measuring anything. A temporal reconstruction
        // is only itself once it has one; measuring from the reset instead scores
        // whatever state the previous scenario happened to leave behind, which
        // moved this number by an order of magnitude between runs.
        for (uint32_t w = 0; w < 8 && !outOfTime(); ++w)
        {
            if (!step())
                break;
        }
        AuditImage prev;
        double swimSum = 0.0;
        double firstMean = -1.0;
        uint32_t swimCount = 0;
        for (uint32_t f = 0; f < frames && !outOfTime(); ++f)
        {
            const float u = frames > 1 ? (float)f / (float)(frames - 1) : 1.0f;
            setOrbit(thetaFrom + (thetaTo - thetaFrom) * u);
            setAnimTime(animFrom + (animTo - animFrom) * u);
            if (!step())
                break;
            AuditImage cur;
            if (!shown(cur))
                continue;
            if (firstMean < 0.0)
                firstMean = auditMean(cur);
            if (f >= frames / 2 && prev.valid())
            {
                const double s = auditSwim(prev, cur);
                if (s >= 0.0)
                {
                    swimSum += s;
                    ++swimCount;
                }
            }
            prev = std::move(cur);
        }
        last = prev;
        // Whether the picture was already wrong or drifted there matters: the
        // first tells you an input is wrong, the second that the accumulation is.
        report(fmt::format("AUDIT {:14s} mean over run: first={:.4f} last={:.4f}", label, firstMean, auditMean(prev)));
        return swimCount ? swimSum / swimCount : -1.0;
    };

    // === Convention sweep (STRELKA_AUDIT_SWEEP=1) ===========================
    //
    // Jitter sign and depth encoding are conventions MetalFX does not document
    // unambiguously, and both fail the same way: the picture still looks like the
    // scene, it just never resolves. A still camera is the sharpest test there is
    // -- with nothing moving, a correct reconstruction converges to the reference
    // and a mismatched one averages neighbouring subpixel positions forever -- so
    // the sweep scores every combination against one converged reference.
    if (envFlag("STRELKA_AUDIT_SWEEP"))
    {
        setOrbit(0.0f);
        setAnimTime(0.0f);
        AuditImage truth;
        converge(truth);
        const double truthSharp = auditSharp(truth);
        report(fmt::format("AUDIT sweep reference sharp={:.5f}", truthSharp));

        for (uint32_t depthMode = 0; depthMode < 3 && !outOfTime(); ++depthMode)
        {
            for (uint32_t sign = 0; sign < 4 && !outOfTime(); ++sign)
            {
                m_settingsManager->setAs<uint32_t>("render/pt/denoiseDepthMode", depthMode);
                m_settingsManager->setAs<uint32_t>("render/pt/jitterSign", sign);
                AuditImage last;
                const double swim = reconstruct("sweep", 0.0f, 0.0f, 0.0f, 0.0f, last);
                int dx = 0, dy = 0;
                const double rmse = auditRmse(last, truth, dx, dy);
                static const char* const kDepthNames[3] = { "device", "viewZ", "radial" };
                static const char* const kSignNames[4] = { "(+x,+y)", "(-x,+y)", "(+x,-y)", "(-x,-y)" };
                report(fmt::format("AUDIT sweep depth={:6s} jitter={:7s} rmse={:.5f} shift({},{}) swim={:.5f} sharp={:.5f}",
                                   kDepthNames[depthMode], kSignNames[sign], rmse, dx, dy, swim, auditSharp(last)));
            }
        }
        m_settingsManager->setAs<uint32_t>("render/pt/denoiseDepthMode", 0);
        m_settingsManager->setAs<uint32_t>("render/pt/jitterSign", 3);
    }

    struct Scenario
    {
        const char* name;
        float thetaFrom, thetaTo;
        float animFrom, animTo;
    };
    const float orbitSpan = 0.35f; // radians over the whole run
    std::vector<Scenario> scenarios = {
        { "S1 static", 0.0f, 0.0f, 0.0f, 0.0f },
        { "S2 orbit", -orbitSpan, orbitSpan, 0.0f, 0.0f },
    };
    if (animCount > 0)
    {
        scenarios.push_back({ "S3 animation", 0.0f, 0.0f, 0.0f, 0.25f });
        scenarios.push_back({ "S4 anim+orbit", -orbitSpan, orbitSpan, 0.0f, 0.25f });
    }

    for (const Scenario& s : scenarios)
    {
        if (outOfTime())
        {
            report(fmt::format("AUDIT WARN out of budget, skipping {} onwards", s.name));
            break;
        }
        AuditImage last;
        const double swim = reconstruct(s.name, s.thetaFrom, s.thetaTo, s.animFrom, s.animTo, last);

        // Reference at the pose and time the run ended on.
        setOrbit(s.thetaTo);
        setAnimTime(s.animTo);
        AuditImage truth;
        converge(truth);

        int dx = 0, dy = 0;
        const double rmse = auditRmse(last, truth, dx, dy);
        // Quantified, not just flagged: a best-shift away from centre only means
        // something if staying at centre actually costs more error.
        double rmse00 = -1.0;
        {
            double se = 0.0;
            size_t n = 0;
            if (last.valid() && truth.valid() && last.px.size() == truth.px.size())
            {
                for (uint32_t y = 1; y + 1 < last.h; ++y)
                    for (uint32_t x = 1; x + 1 < last.w; ++x)
                        for (int k = 0; k < 3; ++k)
                        {
                            const size_t i = ((size_t)y * last.w + x) * 4 + k;
                            const double d = (double)last.px[i] - (double)truth.px[i];
                            se += d * d;
                            ++n;
                        }
                rmse00 = n ? std::sqrt(se / (double)n) : -1.0;
            }
        }
        report(fmt::format("AUDIT {:14s} rmse@centre={:.5f} (best {:.5f} at shift {},{})", s.name, rmse00, rmse, dx, dy));
        report(fmt::format(
            "AUDIT {:14s} rmse={:.5f} at shift({},{})  swim={:.5f}  sharp={:.5f}/{:.5f}  mean={:.4f}/{:.4f}", s.name,
            rmse, dx, dy, swim, auditSharp(last), auditSharp(truth), auditMean(last), auditMean(truth)));
        // Only a real misregistration if centring costs more than a few percent.
        if (rmse >= 0.0 && (dx != 0 || dy != 0) && rmse00 > rmse * 1.05)
            report(fmt::format("AUDIT FAIL {} reconstruction is misregistered by ({},{}), centre costs {:.1f}%", s.name,
                               dx, dy, 100.0 * (rmse00 / rmse - 1.0)));
        // Uniformly darker and partly black are different faults with the same
        // mean, and only one of them is an exposure problem.
        if (last.valid() && truth.valid() && last.px.size() == truth.px.size())
        {
            std::vector<double> ratios;
            size_t nearBlack = 0, lit = 0;
            for (size_t i = 0; i < last.px.size(); i += 4)
            {
                const double t = (truth.px[i] + truth.px[i + 1] + truth.px[i + 2]) / 3.0;
                const double d = (last.px[i] + last.px[i + 1] + last.px[i + 2]) / 3.0;
                if (t <= 1e-4)
                    continue;
                ++lit;
                if (d < 0.1 * t)
                    ++nearBlack;
                ratios.push_back(d / t);
            }
            std::ranges::sort(ratios);
            const double median = ratios.empty() ? 0.0 : ratios[ratios.size() / 2];
            const double blackShare = lit ? 100.0 * (double)nearBlack / (double)lit : 0.0;
            report(fmt::format("AUDIT {:14s} vs truth: median ratio={:.4f}  pixels below 10% of truth={:.1f}%", s.name,
                               median, blackShare));
            // Loud, because this was the number that knew. A reconstruction that
            // lands at half the reference's brightness with a sixth of the frame
            // near black is not a denoise quality question, it is a broken input
            // -- and it sat in the log as a report line for as long as MetalFX
            // was being handed radiance with no exposure to read it by.
            if (!ratios.empty() && (median < 0.75 || median > 1.33 || blackShare > 5.0))
                report(
                    fmt::format("AUDIT FAIL {} does not reproduce the reference's brightness "
                                "(median ratio {:.2f}, {:.1f}% near black)",
                                s.name, median, blackShare));
        }
        save((std::string(s.name).substr(0, 2) + "_denoised").c_str(), last);
        save((std::string(s.name).substr(0, 2) + "_truth").c_str(), truth);
    }

    // === Playback ===========================================================
    //
    // Pressing play is not the same code path as scrubbing time: it sets the
    // animation state, which turns on motion blur, which swaps the scene onto
    // two-keyframe acceleration structures. Everything above drives time by hand
    // with motion blur off and never touches that.
    //
    // Two different measurements, because "the mesh is gone" has two different
    // causes that look identical on screen: geometry coverage says whether rays
    // still hit anything (is it in the acceleration structure?) and the displayed
    // image says whether what they hit is visible (is it being shaded and shown?).
    // Coverage alone was measured first and reported nothing wrong, which was the
    // wrong question.
    if (animCount > 0)
    {
        for (int denoiseOn = 1; denoiseOn >= 0; --denoiseOn)
        {
            setOrbit(0.0f);
            setAnimTime(0.0f);
            // Guides are only written when denoising, so coverage can only be read
            // in that mode; the displayed image can be read in both.
            setDenoise(denoiseOn != 0);
            m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
            m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
            m_render->resetTemporalHistory();
            for (int i = 0; i < 4 && !outOfTime(); ++i)
                step();

            auto coverage = [&]() {
                AuditImage d;
                if (!guide(Render::Guide::Depth, d) || d.px.empty())
                    return -1.0;
                size_t hits = 0, total = 0;
                for (size_t i = 0; i < d.px.size(); i += 4)
                {
                    ++total;
                    if (d.px[i] > 0.0f && d.px[i] < 1e6f)
                        ++hits;
                }
                return total ? 100.0 * (double)hits / (double)total : -1.0;
            };
            AuditImage still;
            shown(still);
            const double meanBefore = auditMean(still);
            const double covBefore = coverage();
            const float extentBefore = m_render->skinnedGeometryExtent();

            for (size_t i = 0; i < animCount; ++i)
            {
                m_settingsManager->setAs<bool>(animationStateKey(i), true);
            }

            // Driven through the editor's own playAnimations() at a realistic
            // frame time, not by setting normalised times. That is the only way
            // to exercise what actually happens on Play: animations of different
            // lengths drifting apart and wrapping independently.
            const int playFrames = 300;
            double worstCov = 1e9, worstMean = 1e9;
            float worstExtent = 1e9f;
            std::string trace;
            for (int i = 0; i < playFrames && !outOfTime(); ++i)
            {
                playAnimations(1.0f / 60.0f);
                if (!step())
                    break;
                AuditImage img;
                if (shown(img) && img.valid())
                {
                    const double m = auditMean(img);
                    if (m < worstMean)
                    {
                        worstMean = m;
                    }
                    if (i % 20 == 0)
                    {
                        trace += fmt::format("{:.2f} ", m);
                    }
                }
                const double c = coverage();
                if (c >= 0.0)
                    worstCov = std::min(worstCov, c);
                const float e = m_render->skinnedGeometryExtent();
                if (e >= 0.0f)
                    worstExtent = std::min(worstExtent, e);
            }
            for (size_t i = 0; i < animCount; ++i)
            {
                m_settingsManager->setAs<bool>(animationStateKey(i), false);
            }

            // Is the character actually moving? A stable picture proves nothing
            // if the animation never reached the geometry -- that would look
            // exactly like "nothing broke".
            AuditImage endFrame;
            shown(endFrame);
            int mdx = 0, mdy = 0;
            const double movedBy = auditRmse(still, endFrame, mdx, mdy);
            const char* label = denoiseOn ? "playback denoise" : "playback plain  ";
            report(fmt::format("AUDIT {} image changed over playback by rmse={:.5f}", label, movedBy));
            report(fmt::format("AUDIT {} displayed mean over playback: {}", label, trace));
            report(fmt::format("AUDIT {} mean before={:.4f} worst={:.4f}   coverage before={:.1f}% worst={:.1f}%",
                               label, meanBefore, worstMean >= 1e9 ? -1.0 : worstMean, covBefore,
                               worstCov >= 1e9 ? -1.0 : worstCov));
            // The character itself, not the picture. A skinned mesh collapsing to
            // a point is invisible to every whole-frame metric.
            if (extentBefore >= 0.0f)
            {
                const float worst = worstExtent >= 1e9f ? extentBefore : worstExtent;
                report(fmt::format(
                    "AUDIT {} skinned geometry extent before={:.3f} worst={:.3f}", label, extentBefore, worst));
                // Absolute, not relative to the start of this scenario: by the
                // time playback runs the character may already have collapsed in
                // an earlier phase, and a ratio against zero notices nothing.
                if (worst < 1e-3f || extentBefore < 1e-3f)
                    report(fmt::format("AUDIT FAIL {} skinned geometry collapsed to a point", label));
                else if (worst < extentBefore * 0.25f)
                    report(fmt::format(
                        "AUDIT FAIL {} skinned geometry collapsed ({:.3f} -> {:.3f})", label, extentBefore, worst));
            }
            if (meanBefore > 0.0 && worstMean < 1e9 && worstMean < meanBefore * 0.6)
                report(fmt::format("AUDIT FAIL {} picture collapsed during playback ({:.4f} -> {:.4f})", label,
                                   meanBefore, worstMean));
        }
        m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    }

    // === Motion vector correctness ==========================================
    //
    // Everything above only asked whether motion vectors exist and are non-zero.
    // That is not the question. A vector is right when following it lands on the
    // same surface one frame earlier, and a wrong vector is non-zero too.
    //
    // Each axis is tested on its own, by displacing the camera along it. An
    // orbiting camera moves the image horizontally, so a test driven by one leaves
    // the vertical sign unexercised -- and a sign fixed on one axis while the
    // other is untested is a coin flip, not a fix.
    {
        setOrbit(0.0f);
        setAnimTime(0.10f);
        setDenoise(true);
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", 1.0f);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
        // Reverse-Z device depth is intentionally concentrated near one; almost
        // any neighbouring pixel falls within the relative tolerance below and
        // makes every candidate vector look correct. Linear view depth gives
        // this correspondence test enough separation to distinguish the signs.
        m_settingsManager->setAs<uint32_t>("render/pt/denoiseDepthMode", 1);
        m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);

        const glm::vec3 basePos = camera.position;
        const float nudge = 0.02f * glm::length(orbitTarget - startPos);
        const glm::vec3 right = glm::conjugate(camera.mOrientation) * glm::vec3(1.0f, 0.0f, 0.0f);
        const glm::vec3 up = glm::conjugate(camera.mOrientation) * glm::vec3(0.0f, 1.0f, 0.0f);

        for (int axis = 0; axis < 2 && !outOfTime(); ++axis)
        {
            camera.position = basePos;
            camera.updateViewMatrix();
            m_render->resetTemporalHistory();
            for (int i = 0; i < 4 && !outOfTime(); ++i)
                step();
            AuditImage d0;
            guide(Render::Guide::Depth, d0);

            camera.position = basePos + (axis == 0 ? right : up) * nudge;
            camera.updateViewMatrix();
            step();
            AuditImage d1, mv;
            guide(Render::Guide::Depth, d1);
            guide(Render::Guide::Motion, mv);

            size_t moved = 0, hitWithout = 0, hitVariant[4] = { 0, 0, 0, 0 };
            if (d0.valid() && d1.valid() && mv.valid() && d0.px.size() == d1.px.size())
            {
                const int w = (int)d1.w, h = (int)d1.h;
                auto depthAt = [&](const AuditImage& img, int x, int y) {
                    x = std::clamp(x, 0, w - 1);
                    y = std::clamp(y, 0, h - 1);
                    return (double)img.px[((size_t)y * w + x) * 4];
                };
                for (int y = 0; y < h; ++y)
                {
                    for (int x = 0; x < w; ++x)
                    {
                        const size_t p = (size_t)y * w + x;
                        const double now = d1.px[p * 4];
                        if (now <= 0.0 || now >= 1e6)
                            continue;
                        const double mx = mv.px[p * 4], my = mv.px[p * 4 + 1];
                        if (std::abs(mx) + std::abs(my) < 0.5)
                            continue;
                        ++moved;
                        auto agrees = [&](double d) { return d > 0.0 && d < 1e6 && std::abs(d - now) / now < 0.02; };
                        const double sx[4] = { mx, -mx, mx, -mx };
                        const double sy[4] = { my, my, -my, -my };
                        for (int v = 0; v < 4; ++v)
                        {
                            if (agrees(depthAt(d0, x + (int)std::lround(sx[v]), y + (int)std::lround(sy[v]))))
                                ++hitVariant[v];
                        }
                        if (agrees(depthAt(d0, x, y)))
                            ++hitWithout;
                    }
                }
            }
            const double pct = moved ? 100.0 / (double)moved : 0.0;
            report(
                fmt::format("AUDIT motioncheck camera {} : n={} ignoring={:.1f}%  (+x,+y)={:.1f}%  "
                            "(-x,+y)={:.1f}%  (+x,-y)={:.1f}%  (-x,-y)={:.1f}%",
                            axis == 0 ? "right" : "up   ", moved, static_cast<double>(hitWithout) * pct,
                            static_cast<double>(hitVariant[0]) * pct, static_cast<double>(hitVariant[1]) * pct,
                            static_cast<double>(hitVariant[2]) * pct, static_cast<double>(hitVariant[3]) * pct));
            if (moved > 100 && hitVariant[0] <= hitWithout)
                report(
                    fmt::format("AUDIT FAIL motion vectors ({} move) are no better than assuming "
                                "nothing moved",
                                axis == 0 ? "horizontal" : "vertical"));
        }
        camera.position = basePos;
        camera.updateViewMatrix();
        m_settingsManager->setAs<uint32_t>("render/pt/denoiseDepthMode", 0);
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", upscale);
    }

    // === Denoise on moving geometry =========================================
    //
    // A whole-frame RMSE cannot answer this. The animated character occupies a
    // small share of the pixels, so it can ghost badly while the number barely
    // moves -- the same blind spot that let a completely absent robot pass. So
    // the error is measured separately over the pixels that actually move and
    // over the ones that do not, and the interesting quantity is the ratio.
    //
    // The mask comes from two converged references at different times, i.e. from
    // ground truth, not from the thing under test.
    if (animCount > 0)
    {
        setOrbit(0.0f);
        // Motion blur changes what a motion vector even means: the traced hit sits
        // at a random time inside the shutter while the previous pose is the last
        // frame's shutter close, so every sample's vector spans a slightly
        // different interval. The editor has it on during playback, so it has to
        // be measurable here.
        const bool mbOn = envFlag("STRELKA_AUDIT_MB");
        m_settingsManager->setAs<bool>("render/enableMotionBlur", mbOn);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", mbOn);

        const float tA = 0.10f, tB = 0.16f;
        std::vector<uint8_t> surface;
        setAnimTime(tA);
        AuditImage truthA;
        converge(truthA);
        setAnimTime(tB);
        AuditImage truthB;
        converge(truthB);

        // Geometry, not lighting. A mask built from how the *picture* changes
        // between two animation times also catches the character's shadow and the
        // light it bounces, which are not the animated mesh and denoise
        // differently. The depth guide says where there is a surface at all.
        {
            m_settingsManager->setAs<float>("render/pt/upscaleFactor", 1.0f);
            m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
            setDenoise(true);
            m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
            setAnimTime(tA);
            for (int i = 0; i < 3 && !outOfTime(); ++i)
                step();
            AuditImage dA;
            guide(Render::Guide::Depth, dA);
            setAnimTime(tB);
            for (int i = 0; i < 3 && !outOfTime(); ++i)
                step();
            AuditImage dB;
            guide(Render::Guide::Depth, dB);
            if (dA.valid() && dB.valid() && dA.px.size() == dB.px.size())
            {
                surface.assign(dA.px.size() / 4, 0);
                for (size_t i = 0, p = 0; i < dA.px.size(); i += 4, ++p)
                {
                    // Sky in either frame is excluded from everything below. A
                    // pixel the mesh moved off carries the background's guides,
                    // and on this scene the character is 5% of the frame, so
                    // admitting the sky put its zero albedo, zero depth and zero
                    // motion into every per-pixel number the section reports.
                    const double a = dA.px[i], b = dB.px[i];
                    surface[p] = (a > 0.0 && a < 1e6 && b > 0.0 && b < 1e6) ? 1 : 0;
                }
            }
            m_settingsManager->setAs<float>("render/pt/upscaleFactor", upscale);
        }

        // Self-calibrating, and only over pixels that show a surface in both
        // poses: the strongest tenth of the change there is the geometry that
        // actually moved, the weakest half is the part of the model that stayed
        // put. An absolute threshold marks most of the frame instead, because
        // converged references still carry a little noise and the moving
        // character relights everything around it. Ranking the whole frame is no
        // better -- the character is a twentieth of it, so the top decile of the
        // *frame* is mostly sky.
        std::vector<uint8_t> moving;
        size_t movingCount = 0;
        if (truthA.valid() && truthB.valid() && truthA.px.size() == truthB.px.size())
        {
            const size_t pixels = truthA.px.size() / 4;
            std::vector<double> change(pixels, 0.0);
            for (size_t i = 0, p = 0; i < truthA.px.size(); i += 4, ++p)
            {
                for (int k = 0; k < 3; ++k)
                    change[p] += std::abs((double)truthA.px[i + k] - (double)truthB.px[i + k]);
            }
            const bool haveSurface = surface.size() == pixels;
            std::vector<double> sorted;
            sorted.reserve(pixels);
            for (size_t p = 0; p < pixels; ++p)
            {
                if (!haveSurface || surface[p])
                    sorted.push_back(change[p]);
            }
            std::ranges::sort(sorted);
            moving.assign(pixels, 2); // 2 = neither, excluded from both measures
            if (!sorted.empty())
            {
                const double hi = sorted[static_cast<size_t>(static_cast<double>(sorted.size()) * 0.90)];
                const double lo = sorted[static_cast<size_t>(static_cast<double>(sorted.size()) * 0.50)];
                for (size_t p = 0; p < pixels; ++p)
                {
                    if (haveSurface && !surface[p])
                        continue;
                    if (change[p] >= hi)
                    {
                        moving[p] = 1;
                        ++movingCount;
                    }
                    else if (change[p] <= lo)
                    {
                        moving[p] = 0;
                    }
                }
            }
        }

        // The user's report is that the skinned mesh stays noisy even when nothing
        // is animating, so the headline case is a *held* time, not a moving one.
        // With the shutter open the geometry is still sampled at a random instant
        // every frame, which a 1-spp temporal reconstruction has no way to
        // reproject -- accumulation resolves it, the denoiser cannot.
        const bool holdTime = envFlag("STRELKA_AUDIT_HOLD");
        AuditImage last;
        const double swim =
            holdTime ? reconstruct("held", 0.0f, 0.0f, tB, tB, last) : reconstruct("moving", 0.0f, 0.0f, tA, tB, last);

        auto maskedRmse = [&](const AuditImage& img, bool wantMoving) {
            if (!img.valid() || !truthB.valid() || img.px.size() != truthB.px.size() || moving.empty())
                return -1.0;
            double se = 0.0;
            size_t n = 0;
            for (size_t i = 0, p = 0; i < img.px.size(); i += 4, ++p)
            {
                if (moving[p] != (wantMoving ? 1 : 0))
                    continue;
                for (int k = 0; k < 3; ++k)
                {
                    const double d = (double)img.px[i + k] - (double)truthB.px[i + k];
                    se += d * d;
                    ++n;
                }
            }
            return n ? std::sqrt(se / (double)n) : -1.0;
        };
        // What the denoiser is actually told about those pixels. Guides are read
        // at 1:1 so the mask, which came from display-resolution references, maps
        // straight onto them.
        {
            m_settingsManager->setAs<float>("render/pt/upscaleFactor", 1.0f);
            m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
            setDenoise(true);
            m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
            m_render->resetTemporalHistory();
            for (int i = 0; i < 4 && !outOfTime(); ++i)
                step();

            auto guideStats = [&](const char* name, Render::Guide g, int channels) {
                AuditImage img;
                if (!guide(g, img) || img.px.size() / 4 != moving.size())
                {
                    report(fmt::format("AUDIT guide-on-mesh {:14s} unavailable or size mismatch", name));
                    return;
                }
                double sumMesh = 0.0, sumRest = 0.0, magMesh = 0.0, magRest = 0.0;
                size_t nMesh = 0, nRest = 0;
                for (size_t p = 0; p < moving.size(); ++p)
                {
                    double len = 0.0, mag = 0.0;
                    for (int k = 0; k < channels; ++k)
                    {
                        const double v = img.px[p * 4 + k];
                        len += v * v;
                        mag += std::abs(v);
                    }
                    len = std::sqrt(len);
                    if (moving[p] == 1)
                    {
                        sumMesh += len;
                        magMesh += mag;
                        ++nMesh;
                    }
                    else if (moving[p] == 0)
                    {
                        sumRest += len;
                        magRest += mag;
                        ++nRest;
                    }
                }
                report(
                    fmt::format("AUDIT guide-on-mesh {:14s} mesh len={:.4f} sum|v|={:.4f} | rest len={:.4f} "
                                "sum|v|={:.4f}",
                                name, nMesh ? sumMesh / static_cast<double>(nMesh) : -1.0,
                                nMesh ? magMesh / static_cast<double>(nMesh) : -1.0,
                                nRest ? sumRest / static_cast<double>(nRest) : -1.0,
                                nRest ? magRest / static_cast<double>(nRest) : -1.0));
            };
            guideStats("normal", Render::Guide::Normal, 3);
            guideStats("diffuseAlbedo", Render::Guide::DiffuseAlbedo, 3);
            guideStats("specAlbedo", Render::Guide::SpecularAlbedo, 3);
            guideStats("roughness", Render::Guide::Roughness, 1);
            guideStats("depth", Render::Guide::Depth, 1);
            guideStats("motion", Render::Guide::Motion, 2);
            m_settingsManager->setAs<float>("render/pt/upscaleFactor", upscale);
        }

        const double rmseMoving = maskedRmse(last, true);
        const double rmseStatic = maskedRmse(last, false);

        // The denoiser's input, for scale. "Denoising does not work here" means
        // the output is no better than the noisy frame it was given, and without
        // this number the errors above have nothing to be compared against.
        setDenoise(false);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", false);
        m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
        setAnimTime(tB);
        m_render->resetTemporalHistory();
        step();
        AuditImage raw;
        shown(raw);
        const double rawMoving = maskedRmse(raw, true);
        const double rawStatic = maskedRmse(raw, false);
        report(fmt::format("AUDIT moving    motionBlur={} timeHeld={} {:.1f}% of pixels move   swim={:.5f}", (int)mbOn,
                           (int)holdTime, moving.empty() ? 0.0 : 100.0 * (double)movingCount / (double)moving.size(),
                           swim));
        report(fmt::format("AUDIT moving    on MOVING pixels: raw={:.5f} denoised={:.5f}  improvement x{:.2f}",
                           rawMoving, rmseMoving, rmseMoving > 0.0 ? rawMoving / rmseMoving : -1.0));
        report(fmt::format("AUDIT moving    on STATIC pixels: raw={:.5f} denoised={:.5f}  improvement x{:.2f}",
                           rawStatic, rmseStatic, rmseStatic > 0.0 ? rawStatic / rmseStatic : -1.0));
        // Denoising that leaves moving geometry no better than its own noisy
        // input is not denoising it at all.
        if (rawMoving > 0.0 && rmseMoving > 0.0 && rawMoving / rmseMoving < 1.2)
            report("AUDIT FAIL denoiser gives no improvement on moving geometry");
    }

    // === Pause mid-animation =================================================
    //
    // Pausing should freeze a frame of the film and go on refining it, and a
    // frame of the film has motion blur in it. The motion structures were tied to
    // playback rather than to whether the shutter actually spans two poses, so a
    // few frames after the pause the blur was dropped and a crisp still appeared.
    if (animCount > 0)
    {
        setOrbit(0.0f);
        setAnimTime(0.0f);
        setDenoise(true);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", refSpp);
        m_settingsManager->setAs<bool>("render/pt/denoisePlaybackMotionBlur", true);
        m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
        m_render->resetTemporalHistory();

        for (size_t i = 0; i < animCount; ++i)
        {
            m_settingsManager->setAs<bool>(animationStateKey(i), true);
        }
        for (int i = 0; i < 20 && !outOfTime(); ++i)
        {
            playAnimations(1.0f / 60.0f);
            if (!step())
                break;
        }
        const bool motionWhilePlaying = m_render->motionGeometryActive();

        // Pause: stop advancing time, keep rendering.
        for (size_t i = 0; i < animCount; ++i)
        {
            m_settingsManager->setAs<bool>(animationStateKey(i), false);
        }
        const size_t subframeAtPause = m_sharedCtx->mSubframeIndex;
        bool motionHeld = true;
        AuditImage pausedDepthFirst;
        for (int i = 0; i < 30 && !outOfTime(); ++i)
        {
            if (!step())
                break;
            if (i == 0)
                guide(Render::Guide::Depth, pausedDepthFirst);
            motionHeld = motionHeld && m_render->motionGeometryActive();
        }
        const size_t subframeAfter = m_sharedCtx->mSubframeIndex;
        AuditImage pausedDepthLast;
        guide(Render::Guide::Depth, pausedDepthLast);
        const double pausedGuideDelta = auditSwim(pausedDepthFirst, pausedDepthLast);

        report(
            fmt::format("AUDIT pause     motion geometry while playing={} held across pause={}  "
                        "accumulation {} -> {} guideDelta={:.7f}",
                        (int)motionWhilePlaying, (int)motionHeld, subframeAtPause, subframeAfter, pausedGuideDelta));
        if (motionWhilePlaying && !motionHeld)
            report("AUDIT FAIL pause dropped motion blur instead of refining the blurred frame");
        if (subframeAfter <= subframeAtPause)
            report("AUDIT FAIL pause stopped the estimator converging");
        if (pausedGuideDelta > 1e-6)
            report("AUDIT FAIL paused canonical depth guide is not stable");

        m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
        m_settingsManager->setAs<bool>("render/pt/denoisePlaybackMotionBlur", false);
    }

    // Frame-to-frame instability with nothing touched at all: no camera, no time,
    // no settings. Deliberately does not reset the animation clock, so whatever
    // state playback left behind -- including two pose keyframes that differ -- is
    // what gets measured.
    auto reconstructHeld = [&](const char* label, AuditImage& last) {
        m_render->resetTemporalHistory();
        for (int w = 0; w < 8 && !outOfTime(); ++w)
        {
            if (!step())
                break;
        }
        AuditImage prev;
        double swimSum = 0.0;
        uint32_t swimCount = 0;
        for (uint32_t f = 0; f < frames && !outOfTime(); ++f)
        {
            if (!step())
                break;
            AuditImage cur;
            if (!shown(cur))
                continue;
            if (prev.valid())
            {
                const double sw = auditSwim(prev, cur);
                if (sw >= 0.0)
                {
                    swimSum += sw;
                    ++swimCount;
                }
            }
            prev = std::move(cur);
        }
        last = prev;
        (void)label;
        return swimCount ? swimSum / swimCount : -1.0;
    };

    // === Shutter versus denoiser ============================================
    //
    // A stochastic shutter puts the geometry at a different instant every frame,
    // which a one-sample temporal reconstruction cannot follow. The symptom is a
    // still frame that shakes, and it is invisible to error-against-a-reference:
    // the shake averages out, so RMSE barely moves. Frame-to-frame instability is
    // the measure, and the test is that turning the shutter on does not change it.
    if (animCount > 0)
    {
        setOrbit(0.0f);
        setAnimTime(0.12f);
        double swimOff = -1.0, swimOn = -1.0;
        for (int mb = 0; mb < 2 && !outOfTime(); ++mb)
        {
            m_settingsManager->setAs<bool>("render/enableMotionBlur", mb != 0);
            m_settingsManager->setAs<bool>("render/isMotionBlurVisible", mb != 0);
            m_settingsManager->setAs<bool>("render/pt/denoisePlaybackMotionBlur", mb != 0);
            // Play first, then hold. Holding from the start leaves both pose
            // keyframes identical, so the shutter has nothing to smear between and
            // the very defect being tested cannot appear -- which is exactly the
            // difference between "no shake at scene start" and "shake after
            // playing".
            setDenoise(true);
            for (size_t a = 0; a < animCount; ++a)
            {
                m_settingsManager->setAs<bool>(animationStateKey(a), true);
            }
            for (int i = 0; i < 12 && !outOfTime(); ++i)
            {
                playAnimations(1.0f / 60.0f);
                if (!step())
                    break;
            }
            for (size_t a = 0; a < animCount; ++a)
            {
                m_settingsManager->setAs<bool>(animationStateKey(a), false);
            }
            AuditImage last;
            const double swim = reconstructHeld(mb ? "shutter on" : "shutter off", last);
            (mb ? swimOn : swimOff) = swim;
        }
        m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
        m_settingsManager->setAs<bool>("render/pt/denoisePlaybackMotionBlur", false);
        report(fmt::format("AUDIT shutter   held frame instability: shutter off={:.5f} on={:.5f} ratio={:.1f}", swimOff,
                           swimOn, swimOff > 0.0 ? swimOn / swimOff : -1.0));
        if (swimOff > 0.0 && swimOn > swimOff * 3.0)
            report("AUDIT FAIL the shutter is shaking the denoised frame");
    }

    // === Denoising past the sample limit ====================================
    //
    // The sample cap stops the estimator. It must not stop the denoiser: a
    // temporal filter that is no longer fed has nothing to show, and with a
    // reduced-resolution render there is not even a display-sized image to fall
    // back on -- the screen simply empties.
    {
        setOrbit(0.0f);
        setAnimTime(0.0f);
        setDenoise(true);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", 4);
        m_render->resetTemporalHistory();
        // Every frame, not one of them. The display is double buffered, so a path
        // that stops writing updates only one of the pair: the picture then
        // alternates between the last good frame and whatever the other buffer
        // holds. A single sample lands on the good one half the time and reports
        // nothing wrong.
        double worstMean = 1e9;
        for (int i = 0; i < 24 && !outOfTime(); ++i)
        {
            if (!step())
                break;
            if (i < 8)
                continue; // let it cross the cap first
            AuditImage img;
            if (shown(img) && img.valid())
                worstMean = std::min(worstMean, auditMean(img));
        }
        const double meanAfterLimit = worstMean >= 1e9 ? -1.0 : worstMean;
        report(
            fmt::format("AUDIT sppcap    denoised image past a 4-sample cap: dimmest of 16 frames "
                        "mean={:.4f}",
                        meanAfterLimit));
        if (meanAfterLimit >= 0.0 && meanAfterLimit <= 1e-3)
            report("AUDIT FAIL denoised image vanished once the sample cap was reached");
        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", refSpp);
    }

    // === S5 mode stress =====================================================
    // Every switch a user can reach, in the combinations that share state: a
    // scaler bound to one resolution, guide textures sized for another, a tracer
    // that writes no guides at all.
    setOrbit(0.0f);
    setAnimTime(0.0f);
    const uint32_t baseW = auditW;
    const uint32_t baseH = auditH;
    struct Mode
    {
        const char* name;
        bool denoise, upscale;
        float factor;
        uint32_t tracer, w, h;
    };
    const Mode modes[] = {
        { "denoise 0.50", true, true, 0.50f, 1, baseW, baseH },
        { "denoise 0.25", true, true, 0.25f, 1, baseW, baseH },
        { "denoise 0.75", true, true, 0.75f, 1, baseW, baseH },
        { "denoise 1.00", true, false, 1.00f, 1, baseW, baseH },
        // Odd sizes: the render resolution rounds to the same value while the
        // output does not, which is the case a guide-texture cache keyed on the
        // render size alone gets wrong.
        { "denoise odd+1", true, true, 0.50f, 1, baseW + 1, baseH + 1 },
        { "denoise odd+3", true, true, 0.50f, 1, baseW + 3, baseH + 3 },
        { "upscale only", false, true, 0.50f, 1, baseW, baseH },
        { "plain wavefront", false, false, 1.00f, 1, baseW, baseH },
        // The megakernel modes went with the megakernel. They were still listed
        // here and still failing -- STALE and BLACK, every run -- which is worse
        // than not testing them: an audit with permanent failures in it stops
        // being something anyone reads.
    };
    AuditImage prevModeImage;
    std::string prevModeKey;
    for (const Mode& m : modes)
    {
        if (outOfTime())
        {
            report(fmt::format("AUDIT WARN out of budget, skipping mode {} onwards", m.name));
            break;
        }
        // Named before it is entered: if a combination aborts the process, the
        // last line printed is the one that names the combination that did it.
        report(fmt::format("AUDIT mode  {:22s} entering ({}x{} tracer={} denoise={} upscale={} factor={:.2f})", m.name,
                           m.w, m.h, m.tracer, (int)m.denoise, (int)m.upscale, m.factor));
        m_settingsManager->setAs<uint32_t>("render/width", m.w);
        m_settingsManager->setAs<uint32_t>("render/height", m.h);
        m_settingsManager->setAs<bool>("render/pt/denoise", m.denoise);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", m.upscale);
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", m.factor);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", false);
        m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
        m_render->resetTemporalHistory();

        bool stepped = true;
        for (int i = 0; i < 4 && stepped; ++i)
            stepped = step();

        AuditImage img;
        const bool got = shown(img) && img.valid();
        const double mean = got ? auditMean(img) : 0.0;
        const bool finite = got && auditFinite(img);
        // Identical to the previous mode's picture means this mode wrote nothing
        // and the display is still showing the last one. That reads as a pass on
        // every other check, and it is exactly what a path that forgets to encode
        // its scaler looks like.
        // Compared only against a mode that should look different -- and what
        // "different" means is the backend's answer, not the request's. OptiX's
        // upscaling model has exactly one ratio, so 0.25, 0.50 and 0.75 all trace
        // at half and hand back byte-identical frames; keyed on the requested
        // factor that read as "this mode wrote nothing" and put two permanent
        // failures in an audit whose value depends on having none.
        const editor_denoiser::Ui fx = editor_denoiser::uiFor(m_render->denoiserKind());
        const int fxMode = editor_denoiser::modeIndexFromSettings(fx, m.denoise, m.upscale);
        const std::string modeKey =
            fmt::format("{}|{}|{:.2f}|{}x{}", m.tracer, (int)editor_denoiser::modeAt(fx, fxMode).denoise,
                        editor_denoiser::appliedScale(fx, fxMode, m.factor), m.w, m.h);
        const bool stale = got && prevModeImage.valid() && modeKey != prevModeKey &&
                           prevModeImage.px.size() == img.px.size() &&
                           std::memcmp(prevModeImage.px.data(), img.px.data(), img.px.size() * sizeof(float)) == 0;
        prevModeKey = modeKey;
        const char* verdict = !stepped     ? "NO FRAME" :
                              !got         ? "NO IMAGE" :
                              !finite      ? "NON-FINITE" :
                              mean <= 1e-6 ? "BLACK" :
                              stale        ? "STALE (mode wrote nothing)" :
                                             "ok";
        if (got)
            prevModeImage = img;
        if (std::strcmp(verdict, "ok") == 0)
        {
            report(fmt::format("AUDIT mode  {:22s} {}x{} mean={:.4f}  ok", m.name, img.w, img.h, mean));
        }
        else
        {
            report(fmt::format("AUDIT FAIL mode {:22s} {}", m.name, verdict));
        }
    }
    m_settingsManager->setAs<uint32_t>("render/width", baseW);
    m_settingsManager->setAs<uint32_t>("render/height", baseH);

    report(fmt::format("AUDIT done in {:.0f}s", elapsed()));
    m_display->requestClose();
}

// Reference capture / estimator self-consistency check (STRELKA_REF=<dir>).
void EditorApp::runReferenceCapture()
{
    const std::optional<std::string> outDir = environmentValue("STRELKA_REF");
    const uint32_t spp = envUint("STRELKA_REF_SPP", 512);

    struct C
    {
        const char* name;
        uint32_t estimator;
        bool analyticLights;
    };
    const C cases[] = {
        { "nee", 0, true },
        { "bsdf_only", 1, true },
        { "nee_envonly", 0, false },
        { "bsdf_envonly", 1, false },
    };

    // Linear output: the tone curve is irrelevant for comparing estimators and
    // would compress exactly the differences we are looking for.
    m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", 0);
    m_settingsManager->setAs<float>("render/post/gamma", 0.0f);
    m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
    m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", spp);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    // Full resolution unless asked otherwise: an upscaled capture is not what the
    // estimators are being compared at, and the buffer would hold a smaller image
    // than the EXR claims. The display EXR alongside it is the upscaled one.
    if (!envFlag("STRELKA_UPSCALE"))
    {
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
    }
    // Same harness for both tracers, so the wavefront rewrite can be checked
    // against the megakernel's recorded numbers without touching anything else.
    if (envFlag("STRELKA_REF_DEPTH"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/depth", envUint("STRELKA_REF_DEPTH", 4));
    }

    std::vector<std::vector<float>> images;
    for (const C& c : cases)
    {
        m_settingsManager->setAs<uint32_t>("render/validate/estimatorMode", c.estimator);
        m_settingsManager->setAs<bool>("render/validate/analyticLights", c.analyticLights);
        m_sharedCtx->mSubframeIndex = 0;

        while (m_sharedCtx->mSubframeIndex < spp && !m_display->windowShouldClose())
        {
            m_display->pollEvents();
            m_render->triggerRenderIfIdle();
            usleep(300);
        }
        // Let the last submission land.
        for (int i = 0; i < 2000 && m_render->getReadyBuffer() == nullptr; ++i)
            usleep(500);
        usleep(200000);

        oka::Buffer* rb = m_render->getReadyBuffer();
        std::vector<float> img;
        double meanLum = 0.0;
        if (rb)
        {
            const float* px = static_cast<const float*>(rb->getHostPointer());
            const size_t n = (size_t)rb->width() * rb->height() * 4;
            img.assign(px, px + n);
            for (size_t i = 0; i < n; i += 4)
                meanLum += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
            const size_t pixelCount = n / 4;
            meanLum /= (double)pixelCount;
            if (outDir.has_value())
            {
                saveScreenshot(rb, *outDir + "/" + c.name + ".exr");
            }
            // ...and what the screen actually shows, which after MetalFX is a
            // different image at a different resolution.
            std::vector<float> shown;
            uint32_t sw = 0, sh = 0;
            if (outDir.has_value() && m_render->readDisplayTexture(shown, sw, sh))
            {
                const char* err = nullptr;
                SaveEXR(shown.data(), (int)sw, (int)sh, 4, 0, (*outDir + "/" + c.name + "_display.exr").c_str(), &err);
            }
        }
        images.push_back(std::move(img));
        STRELKA_INFO("REF  {:14s} spp={} meanLum={:.6f}", c.name, (uint32_t)m_sharedCtx->mSubframeIndex, meanLum);
    }

    auto compare = [&](const char* label, size_t a, size_t b) {
        if (images[a].empty() || images[b].empty() || images[a].size() != images[b].size())
            return;
        double se = 0.0, refEnergy = 0.0;
        size_t n = 0;
        double lumA = 0.0, lumB = 0.0;
        for (size_t i = 0; i < images[a].size(); i += 4)
        {
            for (int k = 0; k < 3; ++k)
            {
                const double d = images[a][i + k] - images[b][i + k];
                se += d * d;
                refEnergy += (double)images[a][i + k] * images[a][i + k];
            }
            lumA += 0.2126 * images[a][i] + 0.7152 * images[a][i + 1] + 0.0722 * images[a][i + 2];
            lumB += 0.2126 * images[b][i] + 0.7152 * images[b][i + 1] + 0.0722 * images[b][i + 2];
            ++n;
        }
        const double rmse = sqrt(se / (double)(n * 3));
        const double rel = refEnergy > 0 ? sqrt(se / refEnergy) : 0.0;
        STRELKA_INFO("REF  {:28s} RMSE={:.6f}  relative={:.3f}%  meanLum {:.6f} vs {:.6f}  bias={:+.2f}%", label, rmse,
                     100.0 * rel, lumA / n, lumB / n, 100.0 * (lumB / lumA - 1.0));
    };
    compare("NEE vs BSDF-only (all)", 0, 1);
    compare("NEE vs BSDF-only (env only)", 2, 3);

    STRELKA_INFO("REF done");
    m_display->requestClose();
}

// ---------------------------------------------------------------------------
// Convergence sweep (STRELKA_CONV=<dir|1>).
//
// The question is whether the estimator's error actually falls as 1/sqrt(N).
// Eyeballing two renders cannot answer it: the difference between 256 and 1024
// samples is one stop of noise, which is easy to miss on a tonemapped image and
// impossible to miss in a number.
//
// Two metrics, because each has a blind spot:
//
//   * half-split noise. Snapshots at N and 2N give the mean of the first N
//     samples and of all 2N, so the second half's mean is 2*I(2N) - I(N). The
//     two halves are independent estimates of the same integral, so the RMS of
//     their difference over sqrt(2) is the error of an N-sample estimate --
//     with no reference image and therefore no reference bias. It does assume
//     the halves are independent, which for a QMC sequence they are not, so it
//     slightly *over*states QMC error. That is the safe direction.
//
//   * RMSE against the deepest snapshot. Sees any error, including the
//     correlated kind a half-split misses, but its own noise floor is the
//     reference's, so it flattens out near the end of the sweep by
//     construction. Only the first half of the table is meaningful.
//
// A healthy sampler halves each metric per 4x samples (slope 0.5 per doubling
// on a log2 scale). A sampler that has stopped bringing new information shows a
// slope heading to 0.
// ---------------------------------------------------------------------------
void EditorApp::runConvergenceSweep()
{
    const std::optional<std::string> outDir = environmentValue("STRELKA_CONV");
    const bool saveImages = outDir.has_value() && outDir->find('/') != std::string::npos;
    const uint32_t maxSpp = envUint("STRELKA_CONV_MAX", 1024);
    const uint32_t firstSpp = envUint("STRELKA_CONV_MIN", 16);
    // Small on purpose. Every metric here is an average over pixels, so a
    // quarter-size image gives the same answer with four times less waiting --
    // and the whole sweep has to fit inside one run.
    const uint32_t convW = envUint("STRELKA_CONV_W", 320);
    const uint32_t convH = envUint("STRELKA_CONV_H", 240);
    // Samples per launch. Bigger is faster (fewer command buffers for the same
    // sample count) and must divide the checkpoints, so it is a power of two.
    const uint32_t sppPerLaunch = envUint("STRELKA_CONV_STEP", 8);
    const double budgetSec = envDouble("STRELKA_CONV_BUDGET", 600.0);
    const std::optional<std::string> samplersEnvValue = environmentValue("STRELKA_CONV_SAMPLERS");
    const std::string samplersEnv = samplersEnvValue.value_or("0,1,2");

    const auto startTime = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count(); };
    auto outOfTime = [&]() { return elapsed() > budgetSec || m_display->windowShouldClose(); };
    auto report = [&](const std::string& line) { STRELKA_INFO("{}", line); };

    // Linear output and no reconstruction: a tone curve compresses exactly the
    // bright noise this is measuring, and a temporal upscaler would be the thing
    // under test instead of the estimator.
    m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", 0);
    m_settingsManager->setAs<float>("render/post/gamma", 0.0f);
    m_settingsManager->setAs<bool>("render/pt/denoise", false);
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
    m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", sppPerLaunch);
    m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", maxSpp);
    m_settingsManager->setAs<uint32_t>("render/width", convW);
    m_settingsManager->setAs<uint32_t>("render/height", convH);
    if (envFlag("STRELKA_CONV_DEPTH"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/depth", envUint("STRELKA_CONV_DEPTH", 4));
    }

    std::vector<uint32_t> checkpoints;
    for (uint32_t n = firstSpp; n <= maxSpp; n *= 2)
    {
        checkpoints.push_back(n);
    }

    auto step = [&]() -> bool {
        const size_t target = m_sharedCtx->mFrameNumber + 1;
        bool submitted = false;
        while (!m_display->windowShouldClose())
        {
            m_display->pollEvents();
            if (!submitted)
            {
                m_render->triggerRenderIfIdle();
                submitted = m_sharedCtx->mFrameNumber >= target;
            }
            if (submitted && !m_render->isRenderBusy())
                return true;
            if (outOfTime())
                return false;
            usleep(200);
        }
        return false;
    };

    // The tonemap of an accumulation lands one frame after the samples do, so
    // every snapshot is taken after an extra pumped frame.
    auto snapshot = [&](std::vector<float>& out) -> bool {
        step();
        oka::Buffer* rb = m_render->getReadyBuffer();
        if (!rb)
            return false;
        const float* px = static_cast<const float*>(rb->getHostPointer());
        const size_t n = (size_t)rb->width() * rb->height() * 4;
        if (n == 0)
            return false;
        out.assign(px, px + n);
        return true;
    };

    // RMS over the three colour channels, and the same relative to the mean
    // level of `scale` -- an absolute RMSE says nothing without knowing how
    // bright the picture is.
    auto rms = [](const std::vector<float>& a, const std::vector<float>& b, double weight) {
        double se = 0.0;
        size_t n = 0;
        for (size_t i = 0; i + 3 < a.size() && i + 3 < b.size(); i += 4)
        {
            for (int k = 0; k < 3; ++k)
            {
                const double d = ((double)a[i + k] - (double)b[i + k]) * weight;
                se += d * d;
                ++n;
            }
        }
        return n ? std::sqrt(se / (double)n) : -1.0;
    };
    auto mean = [](const std::vector<float>& a) {
        double s = 0.0;
        size_t n = 0;
        for (size_t i = 0; i + 3 < a.size(); i += 4)
        {
            s += 0.2126 * a[i] + 0.7152 * a[i + 1] + 0.0722 * a[i + 2];
            ++n;
        }
        return n ? s / (double)n : 0.0;
    };
    // A mean luminance is dominated by whichever pixels happen to hold the
    // brightest fireflies, so two samplers can disagree on it by percent while
    // agreeing everywhere a person would look. The median says which of those
    // two situations this is.
    auto median = [](const std::vector<float>& a) {
        std::vector<double> l;
        l.reserve(a.size() / 4);
        for (size_t i = 0; i + 3 < a.size(); i += 4)
            l.push_back(0.2126 * a[i] + 0.7152 * a[i + 1] + 0.0722 * a[i + 2]);
        if (l.empty())
            return 0.0;
        const auto middle = static_cast<std::vector<double>::difference_type>(l.size() / 2);
        std::nth_element(l.begin(), l.begin() + middle, l.end());
        return l[l.size() / 2];
    };

    // Error left after a small blur. A raw RMSE counts every frequency the same,
    // which is exactly the assumption a blue-noise sampler is built to violate:
    // it does not remove error, it moves error to high frequencies, where the
    // eye and any reconstruction filter throw it away. Two samplers with the
    // same RMSE and different spectra look very different, and only this number
    // says so.
    auto lowPassRms = [&](const std::vector<float>& a, const std::vector<float>& b, uint32_t w, uint32_t h,
                          double weight) {
        if (a.size() != b.size() || (size_t)w * h * 4 != a.size())
            return -1.0;
        // 5-tap binomial, separable, applied to the error field.
        static const double k[5] = { 1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16 };
        std::vector<double> err((size_t)w * h * 3), tmp((size_t)w * h * 3);
        for (size_t i = 0, j = 0; i + 3 < a.size(); i += 4, j += 3)
            for (int c = 0; c < 3; ++c)
                err[j + c] = ((double)a[i + c] - (double)b[i + c]) * weight;
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    double s = 0.0;
                    for (int t = -2; t <= 2; ++t)
                    {
                        const uint32_t xx = (uint32_t)std::clamp((int)x + t, 0, (int)w - 1);
                        s += k[t + 2] * err[((size_t)y * w + xx) * 3 + c];
                    }
                    tmp[((size_t)y * w + x) * 3 + c] = s;
                }
        double se = 0.0;
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    double s = 0.0;
                    for (int t = -2; t <= 2; ++t)
                    {
                        const uint32_t yy = (uint32_t)std::clamp((int)y + t, 0, (int)h - 1);
                        s += k[t + 2] * tmp[((size_t)yy * w + x) * 3 + c];
                    }
                    se += s * s;
                }
        return std::sqrt(se / (double)((size_t)w * h * 3));
    };

    static const char* const kSamplerNames[] = { "Halton", "PCG", "Sobol", "SobolBN", "Hybrid" };
    // Deepest snapshot per sampler, kept for the cross-check at the end.
    std::vector<float> deepest[5];
    uint32_t deepestSpp[5] = { 0, 0, 0, 0, 0 };

    report(fmt::format("CONV {}x{} spp {}..{} step={} depth={} tracer={}", convW, convH, firstSpp, maxSpp, sppPerLaunch,
                       m_settingsManager->getAs<uint32_t>("render/pt/depth"), 1u));

    // A comma-separated list, so it is parsed here rather than through envUint.
    for (const char* p = samplersEnv.c_str(); p != nullptr && *p != '\0';)
    {
        // strtol's out-parameter is char**, so end cannot be const char*.
        // NOLINTNEXTLINE(misc-const-correctness)
        char* end = nullptr;
        const long parsed = std::strtol(p, &end, 10);
        if (end == p)
        {
            STRELKA_WARNING("STRELKA_CONV_SAMPLERS='{}' is not a comma-separated list of ids", samplersEnv);
            break;
        }
        p = end;
        if (*p == ',')
            ++p;
        if (parsed < 0 || parsed > 4 || outOfTime())
            continue;
        const uint32_t samplerType = static_cast<uint32_t>(parsed);

        m_settingsManager->setAs<uint32_t>("render/pt/samplerType", samplerType);
        m_sharedCtx->mSubframeIndex = 0;
        m_render->resetTemporalHistory();

        std::vector<std::vector<float>> snaps(checkpoints.size());
        const auto samplerStart = std::chrono::steady_clock::now();
        for (size_t c = 0; c < checkpoints.size(); ++c)
        {
            // The renderer refuses to launch once the total is reached, so the
            // cap has to rise ahead of each target rather than being set once.
            m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", checkpoints[c]);
            uint32_t guard = checkpoints[c] / sppPerLaunch + 32;
            while (m_sharedCtx->mSubframeIndex < checkpoints[c] && guard-- > 0 && !outOfTime())
            {
                if (!step())
                    break;
            }
            if (!snapshot(snaps[c]))
            {
                report(fmt::format("CONV {} spp={} no frame", kSamplerNames[samplerType], checkpoints[c]));
                break;
            }
            if (m_sharedCtx->mSubframeIndex != checkpoints[c])
            {
                report(fmt::format("CONV WARN {} wanted spp={} got {}", kSamplerNames[samplerType], checkpoints[c],
                                   (uint32_t)m_sharedCtx->mSubframeIndex));
            }
            if (saveImages)
            {
                const char* err = nullptr;
                SaveEXR(snaps[c].data(), (int)convW, (int)convH, 4, 0,
                        fmt::format("{}/{}_{:05d}.exr", *outDir, kSamplerNames[samplerType], checkpoints[c]).c_str(),
                        &err);
            }
        }
        const double sweepSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - samplerStart).count();

        // Everything is reported relative to the image's own mean level, so the
        // three samplers -- and different scenes -- are on one scale.
        size_t last = 0;
        while (last + 1 < snaps.size() && !snaps[last + 1].empty())
            ++last;
        if (snaps[last].empty())
            continue;
        const double lvl = mean(snaps[last]);
        const double w = lvl > 1e-9 ? 1.0 / lvl : 1.0;

        deepest[samplerType] = snaps[last];
        deepestSpp[samplerType] = checkpoints[last];

        report(fmt::format("CONV --- sampler={} ({}) mean={:.5f} median={:.5f} sweep={:.1f}s ---", samplerType,
                           kSamplerNames[samplerType], lvl, median(snaps[last]), sweepSec));
        report("CONV   spp   halfSplitNoise  slope    rmseVsDeepest  slope    lowPassErr   lp/raw");
        double prevNoise = -1.0, prevRef = -1.0;
        for (size_t c = 0; c < snaps.size(); ++c)
        {
            if (snaps[c].empty())
                break;
            double noise = -1.0;
            if (c + 1 < snaps.size() && !snaps[c + 1].empty())
            {
                // Second half's mean: 2*I(2N) - I(N).
                std::vector<float> secondHalf(snaps[c].size());
                for (size_t i = 0; i < secondHalf.size(); ++i)
                {
                    secondHalf[i] = 2.0f * snaps[c + 1][i] - snaps[c][i];
                }
                noise = rms(snaps[c], secondHalf, w) / std::numbers::sqrt2;
            }
            const double ref = c < last ? rms(snaps[c], snaps[last], w) : -1.0;
            // Same error field as rmseVsDeepest, after a blur -- what survives a
            // reconstruction filter, which is what a person actually sees.
            const double lp = c < last ? lowPassRms(snaps[c], snaps[last], convW, convH, w) : -1.0;
            auto slope = [](double prev, double cur) { return (prev > 0.0 && cur > 0.0) ? std::log2(prev / cur) : 0.0; };
            report(fmt::format("CONV {:6d}   {:12.5f}  {:+5.2f}    {:12.5f}  {:+5.2f}    {:10.5f}  {:6.3f}",
                               checkpoints[c], noise, slope(prevNoise, noise), ref, slope(prevRef, ref), lp,
                               (ref > 0.0 && lp > 0.0) ? lp / ref : 0.0));
            prevNoise = noise;
            prevRef = ref;
        }
    }

    // Cross-check. The samplers estimate the same integral, so at the deepest
    // sample count they must agree to within their own noise. A disagreement
    // larger than that is bias in one of them, and no amount of extra samples
    // will remove it -- which is a different problem from converging slowly and
    // has to be told apart from it.
    for (uint32_t a = 0; a < 5; ++a)
    {
        for (uint32_t b = a + 1; b < 5; ++b)
        {
            if (deepest[a].empty() || deepest[b].empty() || deepest[a].size() != deepest[b].size())
                continue;
            const double lvl = mean(deepest[a]);
            const double w = lvl > 1e-9 ? 1.0 / lvl : 1.0;
            report(fmt::format(
                "CONV agree {:6s} vs {:6s} @spp {}/{}: rmse={:.5f}  mean {:.4f}/{:.4f} "
                "({:+.2f}%)  median {:.4f}/{:.4f} ({:+.2f}%)",
                kSamplerNames[a], kSamplerNames[b], deepestSpp[a], deepestSpp[b], rms(deepest[a], deepest[b], w),
                mean(deepest[a]), mean(deepest[b]), 100.0 * (mean(deepest[b]) / mean(deepest[a]) - 1.0),
                median(deepest[a]), median(deepest[b]), 100.0 * (median(deepest[b]) / median(deepest[a]) - 1.0)));
        }
    }

    report("CONV done");
    m_display->requestClose();
}

// Block until the scene is on the GPU.
//
// Every harness below expects a loaded scene, and since the load moved off the
// main thread there is nothing that guarantees one by the time run() is called.
// It silently did not: the denoise audit reported every guide as empty and
// "0.0% of frame is geometry", because it measured an empty scene.
void EditorApp::waitForSceneLoad()
{
    while (m_isLoading || (m_render && m_render->isBuildingScene()))
    {
        if (m_display)
        {
            m_display->pollEvents();
            if (m_display->windowShouldClose())
            {
                if (m_isLoading)
                {
                    m_loadProgress.cancel();
                }
                return;
            }
        }
        checkLoadingComplete();
        if (!m_isLoading && m_render)
        {
            // What advances the GPU-side build, one stage per call.
            m_render->triggerRenderIfIdle();
        }
        // Sleep on both parse and GPU-build spins so harness waits cannot peg a core.
        usleep(1000);
    }
}

void EditorApp::run()
{
    // Harnesses need a fully loaded scene before measuring; interactive use
    // enters the main loop immediately and shows the loading overlay (same as
    // File -> Open) so the window stays responsive during the initial open.
    const bool harness = envFlag("STRELKA_CONV") || envFlag("STRELKA_REF") || envFlag("STRELKA_JITTER_TEST") ||
                         envFlag("STRELKA_DENOISE_AUDIT") || envFlag("STRELKA_LIGHT_AUDIT") ||
                         envFlag("STRELKA_BENCH") || envFlag("STRELKA_PAUSE_BLUR");
    if (harness)
    {
        waitForSceneLoad();
    }
    if (envFlag("STRELKA_CONV"))
    {
        runConvergenceSweep();
        return;
    }
    if (envFlag("STRELKA_REF"))
    {
        runReferenceCapture();
        return;
    }
    if (envFlag("STRELKA_JITTER_TEST"))
    {
        runJitterTest();
        return;
    }
    if (envFlag("STRELKA_DENOISE_AUDIT"))
    {
        runDenoiseAudit();
        return;
    }
    if (envFlag("STRELKA_PAUSE_BLUR"))
    {
        runPauseBlurCheck();
        return;
    }
    if (envFlag("STRELKA_LIGHT_AUDIT"))
    {
        runLightAudit();
        return;
    }
    if (envFlag("STRELKA_BENCH"))
    {
        runBenchmark();
        return;
    }
    auto prevTime = std::chrono::high_resolution_clock::now();

    while (!m_display->windowShouldClose())
    {
        m_display->pollEvents();

        auto currentTime = std::chrono::high_resolution_clock::now();
        const double deltaTime = std::chrono::duration<double>(currentTime - prevTime).count();

        const auto cameraSpeed = m_settingsManager->getAs<float>("render/cameraSpeed");

        // Before update(), so the stick's contribution is in the queue that
        // update() drains this frame rather than next. The pad shares the mouse's
        // speed setting deliberately: two numbers for one notion of "how fast
        // does the camera fly" is how they end up disagreeing.
        // WantTextInput is last frame's, which is what it has to be here: this
        // runs before NewFrame(). One frame of lag on "a text field has focus"
        // is not something a hand can produce.
        if (m_settingsManager->getAs<bool>("editor/gamepad/enabled") &&
            gamepad::cameraOwnsPad(m_display->getGamepadState(), ImGui::GetIO().WantTextInput))
        {
            gamepad::Config padConfig;
            padConfig.moveSpeed = cameraSpeed;
            padConfig.invertLookY = m_settingsManager->getAs<bool>("editor/gamepad/invertLookY");
            padConfig.lookSpeed = m_settingsManager->getAs<float>("editor/gamepad/lookSpeed");
            padConfig.deadzone = m_settingsManager->getAs<float>("editor/gamepad/deadzone");
            // Clamped like CameraController does: a frame that took a second was
            // a scene load, and integrating the stick across it would teleport
            // the camera by however long the pause happened to be.
            const gamepad::CameraInput padInput = gamepad::mapToCamera(
                m_display->getGamepadState(), padConfig, std::min(static_cast<float>(deltaTime), 0.1f));
            m_cameraController->applyGamepad(padInput);
        }

        m_cameraController->update(deltaTime, cameraSpeed);
        prevTime = currentTime;

        playAnimations(static_cast<float>(deltaTime));

        // Consumed every frame so a gesture cannot be acted on twice.
        const bool userMovedCamera = m_cameraController->consumeUserMovedCamera();

        ensureValidCameraSelection();
        auto& selectedCam = m_scene->getCamera(m_selectedCamera);
        if (selectedCam.node != -1 && !m_cameraDetached)
        {
            // GLTF camera in animation mode
            if (userMovedCamera)
            {
                setCameraDetached(true);
            }
            else
            {
                auto& ctrlCam = m_cameraController->getCamera();
                ctrlCam.position = selectedCam.position;
                ctrlCam.mOrientation = selectedCam.mOrientation;
                ctrlCam.updateViewMatrix();
            }
        }

        if (selectedCam.node == -1 || m_cameraDetached)
        {
            auto& ctrlCam = m_cameraController->getCamera();
            selectedCam.position = ctrlCam.position;
            selectedCam.mOrientation = ctrlCam.mOrientation;
            // An orthographic camera zooms by its film extents rather than its
            // pose, and updateAspectRatio below rebuilds the projection from the
            // extents of *this* camera -- so a sync that carried only the pose
            // would hand the renderer the zoom the user just left behind.
            selectedCam.xmag = ctrlCam.xmag;
            selectedCam.ymag = ctrlCam.ymag;
            selectedCam.matrices = ctrlCam.matrices;
            selectedCam.updated = ctrlCam.updated;
            selectedCam.isDirty = ctrlCam.isDirty;
        }

        // The renderer refreshes the projection of the camera it draws with, but
        // the UI reads the same camera earlier in the frame: picking and the gizmo
        // run before the first render has happened, and with an unset projection
        // both fail without a trace (an inf pick ray, a gizmo that never draws).
        const uint32_t renderWidth = m_settingsManager->getAs<uint32_t>("render/width");
        const uint32_t renderHeight = m_settingsManager->getAs<uint32_t>("render/height");
        if (renderHeight != 0)
        {
            const float aspect = (float)renderWidth / (float)renderHeight;
            selectedCam.updateAspectRatio(aspect);
            m_cameraController->getCamera().updateAspectRatio(aspect);
        }

        if (m_isLoading)
        {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - m_loadStartedAt).count();
            if (elapsed > kInteractiveLoadTimeoutSec)
            {
                STRELKA_INFO("ACTION open_timeout path={} elapsed_s={:.0f}", m_attemptedSceneFile, elapsed);
                STRELKA_ERROR("Scene open timed out after {:.0f}s: {}", elapsed, m_attemptedSceneFile);
                m_loadProgress.cancel();
            }
        }

        checkLoadingComplete();
        handleDeviceError();
        if (m_rendererRestartRequested)
        {
            restartRendererAtSafeScale();
        }

        // getMaxEDR() crosses into AppKit; the value only changes when the window
        // moves between displays, so poll it a few times a second instead of
        // every frame.
        if (std::chrono::duration<double>(currentTime - m_lastEdrQuery).count() > 0.25)
        {
            m_lastEdrQuery = currentTime;
            m_settingsManager->setAs<float>("render/post/tonemapper/maxEDR", m_display->getMaxEDR());
        }

        // Display: always runs at vsync, independent of render
        m_display->onBeginFrame();

        const Render::ReadyFrame readyFrame = m_render->getReadyFrame();
        oka::Buffer* readyBuf = readyFrame.buffer;
        // Not the first frame that happens to be ready. Exposure is measured
        // once and then multiplies every pixel for the rest of the session, so
        // measuring it off an arbitrary frame makes the whole picture depend on
        // load timing: the same scene opened twice came out at x1.8, x1.9 and
        // x2.1 here, which is plainly visible when two saved images are
        // compared. Wait until the scene has finished building -- a frame drawn
        // part way through it is missing geometry that has not been handed to
        // the tracer yet -- and until enough samples have landed that the mean
        // is a property of the scene rather than of the noise.
        //
        // Counted here rather than read from the accumulator. mSubframeIndex is
        // the accumulator's own counter: it returns to zero on every camera move
        // and never leaves zero at all when accumulation is switched off, so
        // gating on it means a scene that is being flown through, or that has
        // accumulation off, never gets an exposure and renders the whole session
        // at the default one.
        // Counted on the renderer's frame number, not on passes through this
        // loop. The UI runs at vsync and a traced frame takes many times longer,
        // so counting loop iterations counts the same finished frame over and
        // over -- and the frame still standing right after the build is one of
        // the partial ones published during it. Measured off an environment-only
        // frame that way, the pine forest read 0.596 instead of 0.089 and the
        // scene came out six times too dark.
        static constexpr uint32_t kExposureSettleFrames = 8;
        const bool sceneReady = !m_render->isBuildingScene() && !m_isLoading;
        if (readyBuf && sceneReady && m_sharedCtx->mFrameNumber != m_lastExposureFrameSeen)
        {
            m_lastExposureFrameSeen = m_sharedCtx->mFrameNumber;
            if (m_framesSinceSceneReady < kExposureSettleFrames)
            {
                ++m_framesSinceSceneReady;
            }
        }
        if (readyBuf && m_autoExposurePending && sceneReady && m_framesSinceSceneReady >= kExposureSettleFrames)
        {
            applyAutoExposure(readyBuf);
        }
        if (readyBuf)
        {
            oka::ImageBuffer outputImage;
            outputImage.deviceData = readyBuf->getDevicePointer();
            outputImage.height = readyBuf->height();
            outputImage.width = readyBuf->width();
            outputImage.pixel_format = oka::BufferFormat::FLOAT4;
            // Metal hands over the tonemapped texture directly; the buffer is
            // still there and still linear, which is what a screenshot wants.
            outputImage.deviceTexture = readyFrame.texture;
            outputImage.frameSerial = readyFrame.frameSerial;
            outputImage.presentation = readyFrame.presentation;
            outputImage.dataSize = (size_t)readyBuf->width() * readyBuf->height() * readyBuf->getElementSize();
            m_display->drawFrame(outputImage);
            // Viewport layout must describe the exact texture adopted above,
            // even if another render slot completes before ImGui is encoded.
            mPresentedPreviewWidth = readyBuf->width();
            mPresentedPreviewHeight = readyBuf->height();
        }

        // Match the Metal backend: when minimised / no drawable, skip ImGui so
        // we do not call NewFrame without ImGui_ImplMetal_NewFrame.
        if (m_display->isFrameValid())
        {
            drawUI();
        }

        // --need_screenshot: the sample cap has been reached, so the estimator has
        // stopped and this frame is the finished one. Queued rather than written
        // here so it takes the same path as the menu's screenshot, including the
        // display readback.
        if (m_batchScreenshotArmed && m_pendingScreenshotPath.empty() && !m_isLoading && !m_render->isRenderBusy() &&
            m_batchSppTotal > 0 && m_sharedCtx->mSubframeIndex >= m_batchSppTotal)
        {
            // PNG, not EXR: StrelkaCLI already writes scene-linear radiance, and
            // what this flag is for is the editor's own picture -- exposure, tone
            // curve and all -- in something that can just be looked at.
            const std::filesystem::path scene(m_sceneFile);
            const std::string stem = scene.has_stem() ? scene.stem().string() : "strelka";
            m_batchScreenshotArmed = false;
            m_batchScreenshotPending = true;
            m_pendingScreenshotPath = fmt::format("{}_{}spp.png", stem, m_batchSppTotal);
            STRELKA_INFO("Batch capture: {} spp reached, writing {}", m_batchSppTotal, m_pendingScreenshotPath);
        }

        // Process pending screenshot save
        if (!m_pendingScreenshotPath.empty() && readyBuf)
        {
            saveScreenshot(readyBuf, m_pendingScreenshotPath);
            m_pendingScreenshotPath.clear();
            // A batch run has nothing left to do once the file is on disk, and
            // leaving the window up would make it look like it had hung.
            if (m_batchScreenshotPending)
            {
                m_batchScreenshotPending = false;
                m_display->requestClose();
            }
        }

        m_display->drawUI();
        m_display->onEndFrame();

        // Enqueue the next render pass only after this frame's presentation work
        // has been committed. The renderer runs on its own command queue, but the
        // GPU still executes submissions roughly in arrival order — submitting a
        // multi-second path-trace batch first would push the compositor's work
        // behind it and stall nextDrawable() on the following frame.
        //
        // Skipped while a scene is being parsed: the renderer still points at the
        // outgoing scene, or at the empty one the window came up with. It is
        // deliberately *not* skipped while the GPU build runs -- that build
        // advances one stage per call to this, and gating it here would stop it
        // before it started.
        if (!m_isLoading && !m_renderSubmissionsBlocked)
        {
            m_render->triggerRenderIfIdle();
        }

        // Window titles go through AppKit; only the dirty flag and scene path can
        // change it now, so once a second is more than enough.
        if (std::chrono::duration<double>(currentTime - m_lastTitleUpdate).count() > 1.0)
        {
            m_lastTitleUpdate = currentTime;
            const std::string title = editor_document::formatWindowTitle(m_documentDirty, m_sceneFile);
            m_display->setWindowTitle(title.c_str());
        }
    }

    STRELKA_INFO("ACTION exit path={}", m_sceneFile.empty() ? "(empty)" : m_sceneFile);
}

void EditorApp::playAnimations(const float deltaTime)
{
    const float speed = m_settingsManager->getAs<float>("render/animation/speed");
    const auto& animations = m_scene->getAnimations();
    for (size_t i = 0; i < animations.size(); ++i)
    {
        if (!m_settingsManager->getAs<bool>(animationStateKey(i)))
        {
            continue;
        }
        const std::string timeKey = animationTimeKey(i);
        float currAnimTime = m_settingsManager->getAs<float>(timeKey);

        const float currAnimStart = animations[i].start;
        const float currAnimEnd = animations[i].end;

        currAnimTime += deltaTime * speed;
        if (currAnimTime > currAnimEnd)
            currAnimTime -= (currAnimEnd - currAnimStart);
        if (currAnimTime < currAnimStart)
            currAnimTime = currAnimStart;
        m_settingsManager->setAs<float>(timeKey, currAnimTime);
    }
}

void EditorApp::saveScreenshot(Buffer* buf, const std::string& path)
{
    auto dotPos = path.find_last_of('.');
    std::string ext = (dotPos != std::string::npos) ? path.substr(dotPos) : "";
    uint32_t w = buf->width();
    uint32_t h = buf->height();
    const float* data = static_cast<const float*>(buf->getHostPointer());
    std::vector<float> displayPixels;
    bool displayReadbackAvailable = false;

    const editor_screenshot::Source source =
        editor_screenshot::sourceForExtension(ext, m_screenshotDisplayReferred);
    if (source != editor_screenshot::Source::SceneLinear)
    {
        const bool hdr = source == editor_screenshot::Source::DisplayReferredHdr;
        uint32_t displayWidth = 0;
        uint32_t displayHeight = 0;
        displayReadbackAvailable =
            hdr ? m_render->readDisplayTextureHdr(displayPixels, displayWidth, displayHeight)
                : m_render->readDisplayTextureSdr(displayPixels, displayWidth, displayHeight);
        if (displayReadbackAvailable && !displayPixels.empty())
        {
            w = displayWidth;
            h = displayHeight;
            data = displayPixels.data();
        }
        else
        {
            STRELKA_INFO("ACTION screenshot path={} ok=false", path);
            STRELKA_ERROR("Display-referred pixels are unavailable for this screenshot");
            showAlert("Display-referred pixels are unavailable for this screenshot");
            return;
        }
    }

    if (ext == ".exr")
    {
        const char* err = nullptr;
        const int ret = SaveEXR(data, static_cast<int>(w), static_cast<int>(h), 4, 0, path.c_str(), &err);
        if (ret != TINYEXR_SUCCESS)
        {
            STRELKA_INFO("ACTION screenshot path={} ok=false", path);
            STRELKA_ERROR("Failed to save EXR: {}", err ? err : "unknown");
            if (err)
                FreeEXRErrorMessage(err);
            showAlert(fmt::format("Failed to save screenshot:\n{}", path));
        }
        else
        {
            STRELKA_INFO("ACTION screenshot path={} ok=true", path);
            STRELKA_INFO("Screenshot saved: {}", path);
        }
    }
    else if (ext == ".png")
    {
        const size_t pixelCount = (size_t)w * h;
        std::vector<uint8_t> pixels(pixelCount * 4);
        for (size_t i = 0; i < pixelCount; ++i)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                const float v =
                    c < 3
                        ? editor_screenshot::encodeSrgb(data[i * 4 + c])
                        : std::clamp(data[i * 4 + c], 0.0f, 1.0f);
                pixels[i * 4 + c] = static_cast<uint8_t>(std::lround(v * 255.0f));
            }
        }
        const int ret = stbi_write_png(
            path.c_str(), static_cast<int>(w), static_cast<int>(h), 4, pixels.data(), static_cast<int>(w * 4));
        if (!ret)
        {
            STRELKA_INFO("ACTION screenshot path={} ok=false", path);
            STRELKA_ERROR("Failed to save PNG: {}", path);
            showAlert(fmt::format("Failed to save screenshot:\n{}", path));
        }
        else
        {
            STRELKA_INFO("ACTION screenshot path={} ok=true", path);
            STRELKA_INFO("Screenshot saved: {}", path);
        }
    }
    else
    {
        STRELKA_INFO("ACTION screenshot path={} ok=false", path);
        STRELKA_ERROR("Unsupported screenshot format: {}", ext);
        showAlert(fmt::format("Unsupported screenshot format: {}", ext));
    }
}

void EditorApp::setCameraDetached(bool detached)
{
    m_cameraDetached = detached;

    // Exactly one camera can be under manual control, and clearing all of them
    // first means a camera left detached and then switched away from does not
    // stay frozen out of playback.
    for (uint32_t i = 0; i < (uint32_t)m_scene->getCameraCount(); ++i)
    {
        m_scene->getCamera(i).manualControl = false;
    }
    if (detached && m_selectedCamera >= 0 && (size_t)m_selectedCamera < m_scene->getCameraCount())
    {
        m_scene->getCamera(m_selectedCamera).manualControl = true;
    }
}

bool EditorApp::computeSelectionWorldBounds(glm::float3& outMin, glm::float3& outMax)
{
    const std::vector<Scene::Node>& nodes = m_scene->getNodes();
    glm::float3 localMin(0.0f);
    glm::float3 localMax(0.0f);
    glm::mat4 worldFromLocal(1.0f);

    // Same priority as the selection overlay: a node union first (so a multi-
    // primitive mesh frames as one object), then a lone instance, then a light.
    if (m_selectedNodeId != kInvalidIndex && m_selectedNodeId < nodes.size() &&
        !nodes[m_selectedNodeId].instanceIds.empty())
    {
        if (computeNodeBounds(nodes[m_selectedNodeId], localMin, localMax, worldFromLocal))
        {
            editor_camera_framing::worldAabbFromLocalBox(localMin, localMax, worldFromLocal, outMin, outMax);
            return true;
        }
    }

    const std::vector<Instance>& instances = m_scene->getInstances();
    if (m_selectedInstanceId != kInvalidIndex && m_selectedInstanceId < instances.size() &&
        m_scene->computeInstanceBounds(m_selectedInstanceId, localMin, localMax))
    {
        editor_camera_framing::worldAabbFromLocalBox(
            localMin, localMax, instances[m_selectedInstanceId].transform, outMin, outMax);
        return true;
    }

    if (m_selectedLightId != kInvalidIndex && m_selectedLightId < m_scene->getLightsDesc().size())
    {
        // Lights have no mesh AABB. Frame a box around the emitter so a rect /
        // disc lands in view at a readable size and a point light is still
        // findable — 25 cm floor so a zero-radius punctual is not a singularity.
        const Scene::UniformLightDesc& light = m_scene->getLightsDesc()[m_selectedLightId];
        const float radius = std::max({ 0.25f, light.width * 0.5f, light.height * 0.5f, light.radius });
        outMin = light.position - glm::float3(radius);
        outMax = light.position + glm::float3(radius);
        return true;
    }

    return false;
}

void EditorApp::dumpCameraSettings()
{
    if (!m_scene || m_scene->getCameraCount() == 0)
    {
        STRELKA_WARNING("Nothing to dump: no scene camera");
        return;
    }

    oka::Camera& cam = m_scene->getCamera(m_selectedCamera);

    oka::CameraDumpState s;
    s.scenePath = m_sceneFile;
    s.cameraIndex = static_cast<int>(m_selectedCamera);
    s.width = m_settingsManager->getAs<uint32_t>("render/width");
    s.height = m_settingsManager->getAs<uint32_t>("render/height");

    // A target one unit down the view axis. The CLI turns position and target
    // back into a view matrix, and any distance along the same ray gives the
    // same one, so the nearest is also the one that keeps the most digits.
    const glm::float3 front = cam.getFront();
    const glm::float3 target = cam.position + front;
    const glm::float3 up = cam.getUp();
    for (int i = 0; i < 3; ++i)
    {
        s.position[i] = cam.position[i];
        s.target[i] = target[i];
        s.up[i] = up[i];
    }
    s.orientation[0] = cam.mOrientation.x;
    s.orientation[1] = cam.mOrientation.y;
    s.orientation[2] = cam.mOrientation.z;
    s.orientation[3] = cam.mOrientation.w;

    s.orthographic = (cam.projection == oka::Camera::ProjectionType::orthographic);
    s.fov = cam.fov;
    s.xmag = cam.xmag;
    s.ymag = cam.ymag;
    s.znear = cam.znear;
    s.zfar = cam.zfar;
    s.useDof = cam.useDof;
    s.focalDistance = cam.focalDistance;
    s.fStopDof = cam.fStopDof;

    s.spp = m_settingsManager->getAs<uint32_t>("render/pt/sppTotal");
    s.sppPerLaunch = m_settingsManager->getAs<uint32_t>("render/pt/spp");
    s.maxDepth = m_settingsManager->getAs<uint32_t>("render/pt/depth");
    s.samplerType = m_settingsManager->getAs<uint32_t>("render/pt/samplerType");
    s.debugView = m_settingsManager->getAs<uint32_t>("render/pt/debug");
    s.denoise = m_settingsManager->getAs<bool>("render/pt/denoise");
    s.upscale = m_settingsManager->getAs<bool>("render/pt/enableUpscale");
    s.textureDownscale = m_settingsManager->getAs<uint32_t>("render/texture/downscale");

    s.tonemapperType = m_settingsManager->getAs<uint32_t>("render/pt/tonemapperType");
    s.gamma = m_settingsManager->getAs<float>("render/post/gamma");
    s.filmIso = m_settingsManager->getAs<float>("render/post/tonemapper/filmIso");
    s.fStop = m_settingsManager->getAs<float>("render/post/tonemapper/fStop");
    s.shutterSpeed = m_settingsManager->getAs<float>("render/post/tonemapper/shutterSpeed");

    const std::string dump = oka::formatCameraDump(s);
    // Straight to stdout as well as to the log: the point is to be copied out of
    // a terminal, and the log's prefixes land on every line of the block.
    // Return values discarded deliberately, and said so: there is nothing to do
    // if writing a debug dump to a terminal fails, and cert-err33-c is an error
    // in this tree.
    (void)std::fputs(dump.c_str(), stdout);
    (void)std::fflush(stdout);
    ImGui::SetClipboardText(dump.c_str());

    STRELKA_INFO("ACTION dump_camera camera={} pos=[{} {} {}] fov={}", m_selectedCamera, cam.position.x, cam.position.y,
                 cam.position.z, cam.fov);
}

void EditorApp::frameSelectionInView()
{
    glm::float3 worldMin(0.0f);
    glm::float3 worldMax(0.0f);
    if (!computeSelectionWorldBounds(worldMin, worldMax))
    {
        return;
    }

    const uint32_t renderWidth = m_settingsManager->getAs<uint32_t>("render/width");
    const uint32_t renderHeight = m_settingsManager->getAs<uint32_t>("render/height");
    if (renderHeight == 0)
    {
        return;
    }
    const float aspect = (float)renderWidth / (float)renderHeight;

    // Framing is the user driving the camera, same as WASD: take a glTF camera
    // over so animation does not pose the frame back on the next tick.
    setCameraDetached(true);

    // Framing replaces the pose outright, so anything the user had queued -- half
    // a drag, a movement key still coasting -- must not land on top of the new one
    // a frame later and slide the frame the user just asked for.
    m_cameraController->clearPendingInput();

    Camera& ctrlCam = m_cameraController->getCamera();
    editor_camera_framing::frameCamera(ctrlCam, worldMin, worldMax, aspect);
    ctrlCam.updateAspectRatio(aspect);

    Camera& selectedCam = m_scene->getCamera(m_selectedCamera);
    selectedCam.position = ctrlCam.position;
    selectedCam.mOrientation = ctrlCam.mOrientation;
    selectedCam.xmag = ctrlCam.xmag;
    selectedCam.ymag = ctrlCam.ymag;
    selectedCam.authoredAspect = ctrlCam.authoredAspect;
    selectedCam.matrices = ctrlCam.matrices;
    selectedCam.updateAspectRatio(aspect);

    if (m_sharedCtx)
    {
        m_sharedCtx->mSubframeIndex = 0;
    }
    // Unlike continuous navigation, framing replaces the whole pose in one
    // operation. Mark that cut explicitly instead of asking the renderer to
    // guess from how far the camera happened to move.
    if (m_render)
    {
        m_render->resetTemporalHistory();
    }

    STRELKA_INFO("ACTION frame_selection camera={} projection={}", m_selectedCamera,
                 selectedCam.projection == Camera::ProjectionType::orthographic ? "ortho" : "persp");
}

void EditorApp::clearSelection()
{
    m_selectedNodeId = kInvalidIndex;
    m_selectedInstanceId = kInvalidIndex;
    m_selectedLightId = kInvalidIndex;
    m_selectedMaterialId = kInvalidIndex;
}

void EditorApp::markDocumentDirty()
{
    m_documentDirty = true;
}

void EditorApp::pushUndoLight(uint32_t lightId)
{
    if (lightId >= m_scene->getLightsDesc().size())
        return;
    UndoState s;
    s.kind = UndoState::Kind::Light;
    s.id = lightId;
    s.light = m_scene->getLightsDesc()[lightId];
    m_undoStack.push_back(s);
    m_redoStack.clear();
}

void EditorApp::pushUndoNode(uint32_t nodeId)
{
    if (nodeId >= m_scene->getNodes().size())
        return;
    const auto& n = m_scene->getNodes()[nodeId];
    UndoState s;
    s.kind = UndoState::Kind::Node;
    s.id = nodeId;
    s.translation = n.translation;
    s.rotation = n.rotation;
    s.scale = n.scale;
    m_undoStack.push_back(s);
    m_redoStack.clear();
}

void EditorApp::pushUndoMaterial(uint32_t materialId)
{
    if (materialId >= m_scene->getMaterials().size())
        return;
    UndoState s;
    s.kind = UndoState::Kind::Material;
    s.id = materialId;
    s.material = m_scene->getMaterials()[materialId];
    m_undoStack.push_back(s);
    m_redoStack.clear();
}

void EditorApp::undo()
{
    if (m_undoStack.empty())
        return;
    UndoState cur = m_undoStack.back();
    m_undoStack.pop_back();
    UndoState redo = cur;
    if (cur.kind == UndoState::Kind::Light && cur.id < m_scene->getLightsDesc().size())
    {
        redo.light = m_scene->getLightsDesc()[cur.id];
        m_scene->setLight(cur.id, cur.light);
    }
    else if (cur.kind == UndoState::Kind::Node && cur.id < m_scene->getNodes().size())
    {
        const auto& n = m_scene->getNodes()[cur.id];
        redo.translation = n.translation;
        redo.rotation = n.rotation;
        redo.scale = n.scale;
        m_scene->setNodeLocalTransform(cur.id, cur.translation, cur.rotation, cur.scale);
    }
    else if (cur.kind == UndoState::Kind::Material && cur.id < m_scene->getMaterials().size())
    {
        redo.material = m_scene->getMaterials()[cur.id];
        m_scene->setMaterial(cur.id, cur.material);
    }
    else
    {
        return;
    }
    m_redoStack.push_back(redo);
    markDocumentDirty();
    STRELKA_INFO("ACTION undo kind={} id={}", static_cast<int>(cur.kind), cur.id);
}

void EditorApp::redo()
{
    if (m_redoStack.empty())
        return;
    UndoState cur = m_redoStack.back();
    m_redoStack.pop_back();
    UndoState undo = cur;
    if (cur.kind == UndoState::Kind::Light && cur.id < m_scene->getLightsDesc().size())
    {
        undo.light = m_scene->getLightsDesc()[cur.id];
        m_scene->setLight(cur.id, cur.light);
    }
    else if (cur.kind == UndoState::Kind::Node && cur.id < m_scene->getNodes().size())
    {
        const auto& n = m_scene->getNodes()[cur.id];
        undo.translation = n.translation;
        undo.rotation = n.rotation;
        undo.scale = n.scale;
        m_scene->setNodeLocalTransform(cur.id, cur.translation, cur.rotation, cur.scale);
    }
    else if (cur.kind == UndoState::Kind::Material && cur.id < m_scene->getMaterials().size())
    {
        undo.material = m_scene->getMaterials()[cur.id];
        m_scene->setMaterial(cur.id, cur.material);
    }
    else
    {
        return;
    }
    m_undoStack.push_back(undo);
    markDocumentDirty();
    STRELKA_INFO("ACTION redo kind={} id={}", static_cast<int>(cur.kind), cur.id);
}

void EditorApp::applySelectionFromPick(const Scene::PickHit& hit)
{
    const uint32_t prevNode = m_selectedNodeId;
    const uint32_t prevInstance = m_selectedInstanceId;
    const uint32_t prevLight = m_selectedLightId;
    const uint32_t prevMaterial = m_selectedMaterialId;

    clearSelection();
    if (!hit.hit)
    {
        if (prevNode != kInvalidIndex || prevInstance != kInvalidIndex || prevLight != kInvalidIndex)
        {
            STRELKA_INFO("ACTION select clear");
        }
        return;
    }
    m_selectedInstanceId = hit.instanceId;
    m_selectedNodeId = hit.nodeId;
    m_selectedLightId = hit.lightId;
    m_outlinerScrollToSelection = true;
    if (hit.instanceId < m_scene->getInstances().size())
        m_selectedMaterialId = m_scene->getInstances()[hit.instanceId].mMaterialId;

    if (m_selectedNodeId != prevNode || m_selectedInstanceId != prevInstance || m_selectedLightId != prevLight ||
        m_selectedMaterialId != prevMaterial)
    {
        STRELKA_INFO("ACTION select node={} instance={} light={} material={}", m_selectedNodeId, m_selectedInstanceId,
                     m_selectedLightId, m_selectedMaterialId);
    }
}

bool EditorApp::saveDocument(bool saveAs)
{
    if (saveAs || m_sceneFile.empty())
    {
        IGFD::FileDialogConfig config;
        config.path = m_resourceSearchPath.empty() ? "." : m_resourceSearchPath;
        ImGuiFileDialog::Instance()->OpenDialog("SaveSceneDlgKey", "Save Scene As", ".gltf,.glb", config);
        m_pendingSaveAs = true;
        STRELKA_INFO("ACTION save_as path={}", m_sceneFile.empty() ? "(empty)" : m_sceneFile);
        return true;
    }

    const bool okGltf = saveGltf(*m_scene, m_sceneFile);
    const bool okLights = saveLightsJson(*m_scene, m_sceneFile);
    if (okGltf && okLights)
    {
        m_documentDirty = false;
        STRELKA_INFO("ACTION save_ok path={}", m_sceneFile);
        STRELKA_INFO("Saved scene: {}", m_sceneFile);
        return true;
    }
    STRELKA_INFO("ACTION save_fail path={} gltf={} lights={}", m_sceneFile, okGltf, okLights);
    STRELKA_ERROR("Failed to save scene: {} (gltf={}, lights={})", m_sceneFile, okGltf, okLights);
    showAlert(fmt::format("Failed to save scene:\n{}", m_sceneFile));
    return false;
}

void EditorApp::buildDefaultDockLayout(ImGuiID dockspaceId)
{
    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspaceId, ImGui::GetMainViewport()->WorkSize);

    ImGuiID center = dockspaceId;
    const ImGuiID left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.20f, nullptr, &center);
    // 0.28 clipped the Render Settings tab bar and Capabilities text.
    const ImGuiID right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.32f, nullptr, &center);
    ImGuiID leftTop = left;
    const ImGuiID leftBottom = ImGui::DockBuilderSplitNode(leftTop, ImGuiDir_Down, 0.35f, nullptr, &leftTop);
    ImGuiID rightTop = right;
    const ImGuiID rightBottom = ImGui::DockBuilderSplitNode(rightTop, ImGuiDir_Down, 0.45f, nullptr, &rightTop);

    ImGui::DockBuilderDockWindow("Outliner", leftTop);
    ImGui::DockBuilderDockWindow("Animations", leftBottom);
    // Memory: not docked, hidden by default, floats when opened from Debug menu.
    ImGui::DockBuilderDockWindow("Viewport", center);
    ImGui::DockBuilderDockWindow("Render Settings:", rightTop);
    ImGui::DockBuilderDockWindow("Camera:", rightBottom);
    ImGui::DockBuilderDockWindow("Properties", rightBottom);
    ImGui::DockBuilderDockWindow("Materials", rightBottom);
    ImGui::DockBuilderFinish(dockspaceId);

    STRELKA_INFO("Editor layout rebuilt from defaults");
}

void EditorApp::drawUI()
{
    ImGui_ImplGlfw_NewFrame();
    // After the backend, before NewFrame: the backend looks for the pad in slot
    // 0 and finds nothing here, so without this ImGui gets no gamepad at all.
    gamepad::feedImGui(m_display->getGamepadState(), ImGui::GetIO());
    ImGui::NewFrame();

    // Set before BeginFrame as well as in the viewport: the flag is global state
    // that everything drawn this frame reads.
    ensureValidCameraSelection();
    const bool orthographicCamera =
        m_scene && m_scene->getCameraCount() > 0 &&
        m_scene->getCamera(static_cast<uint32_t>(m_selectedCamera)).projection == Camera::ProjectionType::orthographic;
    ImGuizmo::SetOrthographic(orthographicCamera);
    ImGuizmo::BeginFrame();

    m_cameraController->setGizmoBlocksInput(ImGuizmo::IsOver() || ImGuizmo::IsUsing());

    const ImGuiIO& io = ImGui::GetIO();

    // Hotkeys
    if (!io.WantTextInput)
    {
        // Gizmo W/E/R only with a selection and the viewport hovered — otherwise
        // camera WASD (and E for down) would fight the gizmo bindings.
        const bool gizmoHotkeys =
            m_display->isViewPortHovered() && (m_selectedNodeId != kInvalidIndex || m_selectedLightId != kInvalidIndex);
        if (gizmoHotkeys)
        {
            if (ImGui::IsKeyPressed(ImGuiKey_W))
                m_gizmoOperation = ImGuizmo::TRANSLATE;
            if (ImGui::IsKeyPressed(ImGuiKey_E))
                m_gizmoOperation = ImGuizmo::ROTATE;
            if (ImGui::IsKeyPressed(ImGuiKey_R))
                m_gizmoOperation = ImGuizmo::SCALE;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape))
        {
            if (m_selectedNodeId != kInvalidIndex || m_selectedLightId != kInvalidIndex ||
                m_selectedInstanceId != kInvalidIndex)
            {
                STRELKA_INFO("ACTION select clear");
            }
            clearSelection();
        }
        // Frame Selection (F): Blender/Maya convention. Not viewport-gated — framing
        // from the outliner after a click is the usual path, and F does not collide
        // with WASD or the gizmo bindings.
        const bool canFrameSelection = m_selectedNodeId != kInvalidIndex || m_selectedInstanceId != kInvalidIndex ||
                                       m_selectedLightId != kInvalidIndex;
        if (canFrameSelection && ImGui::IsKeyPressed(ImGuiKey_F))
            frameSelectionInView();
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S))
            saveDocument(io.KeyShift);
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z))
            undo();
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Y))
            redo();
    }

    const ImGuiID dockspaceId = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    // A layout saved by an older build is missing a panel added since (floats
    // it) or carries a stale Pos/Size with DockId 0 (same effect). Checked once
    // against every window buildDefaultDockLayout places -- not every frame,
    // since FindWindowSettingsByID reads the ini snapshot, which only catches
    // up with a fresh rebuild on ImGui's autosave timer, not immediately.
    if (!m_startupLayoutChecked)
    {
        m_startupLayoutChecked = true;
        static const char* const kDockedWindows[] = { "Outliner", "Animations", "Viewport",
                                                       "Render Settings:", "Camera:", "Properties", "Materials" };
        for (const char* name : kDockedWindows)
        {
            const ImGuiWindowSettings* windowSettings = ImGui::FindWindowSettingsByID(ImHashStr(name));
            if (windowSettings == nullptr || windowSettings->DockId == 0)
            {
                m_layoutRebuildPending = true;
                break;
            }
        }
    }
    if (m_layoutRebuildPending)
    {
        m_layoutRebuildPending = false;
        buildDefaultDockLayout(dockspaceId);
    }

    // --- Main menu bar ---
    ImGui::BeginMainMenuBar();
    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("Open File", nullptr, false, !m_isLoading))
        {
            IGFD::FileDialogConfig config;
            config.path = ".";
            ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".gltf,.glb", config);
        }
        if (ImGui::BeginMenu("Open Recent", !m_isLoading))
        {
            if (m_recentScenes.empty())
            {
                ImGui::MenuItem("(empty)", nullptr, false, false);
            }
            else
            {
                for (size_t i = 0; i < m_recentScenes.size(); ++i)
                {
                    const std::string& path = m_recentScenes[i];
                    std::error_code ec;
                    const bool exists = std::filesystem::exists(path, ec) && !ec;
                    const std::string label = std::filesystem::path(path).filename().string();
                    // PushID so two scenes that share a basename do not collide
                    // in ImGui's id stack and steal each other's clicks.
                    ImGui::PushID(static_cast<int>(i));
                    if (ImGui::MenuItem(label.c_str(), nullptr, false, exists))
                    {
                        beginSceneLoad(path, std::filesystem::path(path).parent_path().string());
                    }
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    {
                        ImGui::SetTooltip("%s", path.c_str());
                    }
                    ImGui::PopID();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Clear Recent"))
                {
                    m_recentScenes.clear();
                    persistRecentScenes();
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Save", "Ctrl+S", false, !m_isLoading && !m_sceneFile.empty()))
            saveDocument(false);
        if (ImGui::MenuItem("Save As...", "Ctrl+Shift+S", false, !m_isLoading))
            saveDocument(true);
        ImGui::Separator();
        if (ImGui::MenuItem("Exit"))
        {
            STRELKA_INFO("ACTION exit path={}", m_sceneFile.empty() ? "(empty)" : m_sceneFile);
            m_display->requestClose();
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Edit"))
    {
        if (ImGui::MenuItem("Undo", "Ctrl+Z", false, !m_undoStack.empty()))
            undo();
        if (ImGui::MenuItem("Redo", "Ctrl+Y", false, !m_redoStack.empty()))
            redo();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View"))
    {
        const bool canFrameSelection = m_selectedNodeId != kInvalidIndex || m_selectedInstanceId != kInvalidIndex ||
                                       m_selectedLightId != kInvalidIndex;
        if (ImGui::MenuItem("Frame Selection", "F", false, canFrameSelection && !m_isLoading))
            frameSelectionInView();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Debug"))
    {
        if (ImGui::MenuItem("Dump camera settings", nullptr, false, !m_isLoading))
            dumpCameraSettings();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        {
            ImGui::SetTooltip(
                "Print this viewport to the console as a StrelkaCLI .toml,\n"
                "and copy it to the clipboard, so the exact frame can be\n"
                "re-rendered headlessly.");
        }
        ImGui::MenuItem("Memory", nullptr, &m_showMemory);
        ImGui::MenuItem("Render Debug", nullptr, &m_showRenderDebug);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Window"))
    {
        ImGui::MenuItem("Outliner", nullptr, &m_showOutliner);
        ImGui::MenuItem("Properties", nullptr, &m_showProperties);
        ImGui::MenuItem("Materials", nullptr, &m_showMaterials);
        ImGui::Separator();
        if (ImGui::MenuItem("Reset Layout"))
        {
            m_layoutRebuildPending = true;
            m_showOutliner = m_showProperties = m_showMaterials = true;
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help"))
    {
        if (ImGui::MenuItem("About Strelka"))
        {
            m_aboutOpen = true;
        }
        ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();

    // --- File dialog handling ---
    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            const std::string sceneFile = ImGuiFileDialog::Instance()->GetFilePathName();
            const std::string resourceSearchPath = ImGuiFileDialog::Instance()->GetCurrentPath();
            STRELKA_DEBUG("Resource search path {}", resourceSearchPath);
            beginSceneLoad(sceneFile, resourceSearchPath);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display("SaveSceneDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            m_sceneFile = ImGuiFileDialog::Instance()->GetFilePathName();
            m_resourceSearchPath = ImGuiFileDialog::Instance()->GetCurrentPath();
            m_settingsManager->setAs<std::string>("resource/searchPath", m_resourceSearchPath);
            saveDocument(false);
        }
        ImGuiFileDialog::Instance()->Close();
        m_pendingSaveAs = false;
    }

    // --- Save screenshot dialog handling ---
    if (ImGuiFileDialog::Instance()->Display("SaveScreenshotDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            m_pendingScreenshotPath = ImGuiFileDialog::Instance()->GetFilePathName();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    drawLoadingOverlay();
    drawAlertModal();
    drawAboutModal();
    drawFrameBudgetModal();

    // --- Panel draw calls (implementations in panels/*.cpp) ---
    drawViewportPanel();
    drawRenderSettingsPanel();
    drawCameraPanel();
    if (m_showMemory)
    {
        drawMemoryPanel();
    }
    if (m_showRenderDebug)
    {
        drawRenderDebugPanel();
    }
    drawAnimationPanel();
    if (m_showOutliner)
        drawOutlinerPanel();
    if (m_showProperties)
        drawPropertyPanel();
    if (m_showMaterials)
        drawMaterialPanel();

    // Rendering
    ImGui::Render();
}

} // namespace oka
