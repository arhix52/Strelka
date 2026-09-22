#include "EditorApp.h"

#include <application_paths.h>

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
#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

#include <tinyexr.h>
#include <stb_image_write.h>

namespace oka
{
namespace
{
constexpr double kInteractiveLoadTimeoutSec = 300.0;
} // namespace

EditorApp::EditorApp(const std::string& sceneFile, const std::string& resourceSearchPath)
    :
#ifdef __APPLE__
      m_initialMetalRendererUnused(!sceneFile.empty()),
#endif
      m_resourceSearchPath(resourceSearchPath)
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

    loadSettings();
    if (!sceneFile.empty())
    {
        beginSceneLoad(sceneFile, resourceSearchPath);
    }
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

    m_recentScenes = editor_document::loadRecentScenes(applicationSupportDirectory() / "recent_scenes.txt");

    if (sceneFile.empty())
    {
        beginSceneLoad(sceneFile, resourceSearchPath);
    }
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
        ImGui::Text("Version %s (%s)", STRELKA_MARKETING_VERSION, STRELKA_BUILD_NUMBER);
        ImGui::TextDisabled("Revision %s", STRELKA_VERSION);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        // Everything a bug report needs, one click from the menu that opened
        // this rather than the user retyping a version string by hand.
        if (ImGui::Button("Copy Version Info", ImVec2(160, 0)))
        {
            const std::string info = fmt::format("Strelka {} ({}, revision {}; macOS)", STRELKA_MARKETING_VERSION,
                                                 STRELKA_BUILD_NUMBER, STRELKA_VERSION);
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
    dumpCameraSettings();
    showAlert(
        "GPU device error.\nRender submissions have been stopped.\n"
        "You can rebuild the renderer at a safe PT scale or keep the last good frame.");
    m_alertOffersRendererRestart = true;
}

void EditorApp::restoreDocumentAfterFailedLoad(const char* reason)
{
    const std::string attempted = m_attemptedSceneFile.empty() ? m_sceneFile : m_attemptedSceneFile;
    m_sceneFile = m_sceneFileBeforeLoad;
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
    const std::filesystem::path supportDirectory = applicationSupportDirectory();
    std::error_code ec;
    std::filesystem::create_directories(supportDirectory, ec);
    if (ec || !editor_document::saveRecentScenes(supportDirectory / "recent_scenes.txt", m_recentScenes))
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
    STRELKA_INFO(
        "Auto exposure: scene mean luminance {:.5f} over the {:.1f}% of the frame that caught light, "
        "exposure x{:.4g} ({:+.2f} EV); no exposure in the light sidecar",
        mean, total > 0 ? 100.0 * double(n) / double(total) : 0.0, factor, std::log2(factor));
}

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

void EditorApp::applyScenePresentation()
{
    if (const auto& presentation = m_scene->getPresentation(); presentation.has_value())
    {
        m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", presentation->tonemapperType);
        m_settingsManager->setAs<float>("render/post/gamma", presentation->gamma);
        m_settingsManager->setAs<uint32_t>("render/material/model", presentation->materialModel);
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
    m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", 1); // 0=None, 1=Reinhard, 2=ACES, 3=Filmic, 4=AgX
    m_settingsManager->setAs<uint32_t>("render/pt/debug", 0); // 0 - none, 1 - normals
    m_settingsManager->setAs<float>("render/cameraSpeed", 1.0f);
    m_settingsManager->setAs<bool>("editor/gamepad/enabled", true);
    m_settingsManager->setAs<bool>("editor/gamepad/invertLookY", false);
    m_settingsManager->setAs<float>("editor/gamepad/lookSpeed", gamepad::Config{}.lookSpeed);
    m_settingsManager->setAs<float>("editor/gamepad/deadzone", gamepad::Config{}.deadzone);
    m_settingsManager->setAs<float>("render/pt/upscaleFactor", 0.5f);
    // Preview resolution itself bounds interactive work. MetalFX remains an
    // explicit quality/performance choice inside that fixed output.
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
    m_settingsManager->setAs<uint32_t>("render/pt/samplerType", 4);
    m_settingsManager->setAs<uint32_t>("render/pt/blueNoiseSwitchSpp", 4);
    m_settingsManager->setAs<uint32_t>("render/selectedCamera", 0);
    m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
    m_settingsManager->setAs<float>("render/animation/speed", 1.0f);
    // Wavefront by default: bit-identical output, 2.6x faster at depth 8. The
    // megakernel stays selectable so any change can still be A/B'd against it.
    m_settingsManager->setAs<uint32_t>("render/pt/profileStages", 0);
    // Honour the stage-profiling override in interactive runs too.
    if (envFlag("STRELKA_STAGES"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/profileStages", 1);
    }
    // Dense, nearly white SSS paths often consume the whole walk budget. Keep
    // interactive frames responsive; the Quality panel can raise this when a
    // close-up needs the longer tail. Headless/final rendering retains 64.
    m_settingsManager->setAs<uint32_t>("render/pt/subsurfaceIterations", 16);
    // One candidate has the best equal-time convergence and keeps interactive
    // navigation responsive. Higher fixed-SPP quality remains available under
    // Advanced > Direct lighting.
    m_settingsManager->setAs<uint32_t>("render/pt/risCandidates", 1u);
    m_settingsManager->setAs<uint32_t>("render/pt/writeAov", 0);
    m_settingsManager->setAs<bool>("render/pt/denoise", false);
    m_settingsManager->setAs<bool>("render/pt/prewarmDenoiser", true);
    // Luminance ceiling for the denoiser's colour input, in exposed units: a
    // single unbounded sample gets smeared over many frames by a temporal filter.
    // 0 disables it.
    m_settingsManager->setAs<float>("render/pt/denoiseFireflyClamp", 8.0f);
    m_settingsManager->setAs<float>("render/pt/clampIndirect", 0.0f);
    m_settingsManager->setAs<float>("render/pt/clampDirect", 0.0f);
    if (envFlag("STRELKA_DENOISE"))
    {
        m_settingsManager->setAs<bool>("render/pt/denoise", envBool("STRELKA_DENOISE", false));
    }
    m_settingsManager->setAs<uint32_t>("render/pt/textureLod", 1);
    m_settingsManager->setAs<uint32_t>("render/pt/guidePrimaryHit", 0);
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
    m_settingsManager->setAs<float>("render/stream/publishIntervalMs", 500.0f);
    // The editor picks against the host arrays, so it keeps them.
    m_settingsManager->setAs<bool>("scene/releaseHostGeometry", false);
    m_settingsManager->setAs<std::string>("resource/searchPath", m_resourceSearchPath);
    // Postprocessing settings:
    m_settingsManager->setAs<uint32_t>("render/post/outputMode", static_cast<uint32_t>(display_output::OutputMode::Auto));
    m_settingsManager->setAs<float>("render/post/paperWhiteNits", 203.0f);
    m_settingsManager->setAs<float>("render/post/peakNits", 1000.0f);
    m_settingsManager->setAs<bool>("display/vrr/enabled", true);
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

void EditorApp::initializeRendererForCurrentScene(bool reuseExisting)
{
    bool sceneHasMotion = !m_scene->getAnimations().empty();
    size_t skeletalMeshCount = 0;
    for (const oka::Mesh& mesh : m_scene->getMeshes())
    {
        sceneHasMotion = sceneHasMotion || mesh.isSkeletal;
        skeletalMeshCount += mesh.isSkeletal ? 1 : 0;
    }
    const bool interactiveMotionBlur = sceneHasMotion && skeletalMeshCount <= 64;
    m_settingsManager->setAs<bool>("render/enableMotionBlur", interactiveMotionBlur);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", interactiveMotionBlur);

    if (reuseExisting && m_render)
    {
        m_render->setSettingsManager(m_settingsManager.get());
        m_render->setSharedContext(m_sharedCtx.get());
        m_render->setScene(m_scene.get());
        m_render->setLoadProgress(&m_loadProgress);
        m_display->setRender(m_render.get());
        return;
    }

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
#ifdef __APPLE__
        // The main loop is now free to render the empty/previous document, so a
        // later Open can no longer assume this renderer has never submitted.
        m_initialMetalRendererUnused = false;
#endif
        const char* reason = m_loadProgress.isCancelled() ? "cancel" : "loader";
        restoreDocumentAfterFailedLoad(reason);
        return;
    }

    bool reuseInitialRenderer = false;
#ifdef __APPLE__
    reuseInitialRenderer = m_initialMetalRendererUnused;
    m_initialMetalRendererUnused = false;
#endif

    m_display->resetFrame();
    // Clear the display's raw Render* before destroying the object it points at.
    if (!reuseInitialRenderer)
    {
        m_display->setRender(nullptr);
        m_render.reset();
    }

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

    applyScenePresentation();
    initializeRendererForCurrentScene(reuseInitialRenderer);

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

    applySceneExposure();

    rememberRecentScene(m_sceneFile);
    STRELKA_INFO("ACTION open_ok path={}", m_sceneFile);
}

void EditorApp::run()
{
    auto prevTime = std::chrono::high_resolution_clock::now();

    while (!m_display->windowShouldClose())
    {
        m_display->pollEvents();

        auto currentTime = std::chrono::high_resolution_clock::now();
        const double deltaTime = std::chrono::duration<double>(currentTime - prevTime).count();
        const auto cameraSpeed = m_settingsManager->getAs<float>("render/cameraSpeed");

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
            selectedCam.xmag = ctrlCam.xmag;
            selectedCam.ymag = ctrlCam.ymag;
            selectedCam.matrices = ctrlCam.matrices;
            selectedCam.updated = ctrlCam.updated;
            selectedCam.isDirty = ctrlCam.isDirty;
        }

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
        const float duration = currAnimEnd - currAnimStart;
        if (duration > 0.0f)
        {
            float offset = std::fmod(currAnimTime - currAnimStart + deltaTime * speed, duration);
            if (offset < 0.0f)
            {
                offset += duration;
            }
            currAnimTime = currAnimStart + offset;
        }
        else
        {
            currAnimTime = currAnimStart;
        }
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

    const editor_screenshot::Source source = editor_screenshot::sourceForExtension(ext, m_screenshotDisplayReferred);
    if (source != editor_screenshot::Source::SceneLinear)
    {
        const bool hdr = source == editor_screenshot::Source::DisplayReferredHdr;
        uint32_t displayWidth = 0;
        uint32_t displayHeight = 0;
        displayReadbackAvailable = hdr ? m_render->readDisplayTextureHdr(displayPixels, displayWidth, displayHeight) :
                                         m_render->readDisplayTextureSdr(displayPixels, displayWidth, displayHeight);
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
                    c < 3 ? editor_screenshot::encodeSrgb(data[i * 4 + c]) : std::clamp(data[i * 4 + c], 0.0f, 1.0f);
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
    s.focalLengthMm = cam.focalLengthMm;

    s.spp = m_settingsManager->getAs<uint32_t>("render/pt/sppTotal");
    s.sppPerLaunch = m_settingsManager->getAs<uint32_t>("render/pt/spp");
    s.maxDepth = m_settingsManager->getAs<uint32_t>("render/pt/depth");
    s.samplerType = m_settingsManager->getAs<uint32_t>("render/pt/samplerType");
    s.reconstructionFilter = m_settingsManager->getAs<uint32_t>("render/pt/reconstructionFilter");
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
    uint32_t selectedNode = hit.nodeId;

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
    if (selectedNode < m_scene->getNodes().size())
    {
        const Scene::Node& selected = m_scene->getNodes()[selectedNode];
        if (selected.skin >= 0 && selected.parent >= 0)
        {
            selectedNode = static_cast<uint32_t>(selected.parent);
        }
    }
    m_selectedNodeId = selectedNode;
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

    if (!m_startupLayoutChecked)
    {
        m_startupLayoutChecked = true;
        static const char* const kDockedWindows[] = { "Outliner", "Animations", "Viewport", "Render Settings:",
                                                      "Camera:",  "Properties", "Materials" };
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
        ImGui::Separator();
        const bool hasSelection = m_selectedNodeId != kInvalidIndex || m_selectedInstanceId != kInvalidIndex ||
                                  m_selectedLightId != kInvalidIndex;
        if (ImGui::MenuItem("Deselect All", "Esc", false, hasSelection))
        {
            STRELKA_INFO("ACTION select clear");
            clearSelection();
        }
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
