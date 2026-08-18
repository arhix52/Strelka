#include "../EditorApp.h"
#include "../editor_camera_exposure.h"
#include "../editor_denoiser_ui.h"
#include "../editor_frame_budget.h"

#include <math.h>
#include <strelka/display/output_policy.h>

#include "imgui.h"
#include "ImGuiFileDialog.h"

#include <cfloat>
#include <ctime>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace oka
{
namespace
{
void drawDisplayOutputSettings(SettingsManager& settings, const Display& display)
{
    const char* const outputModeItems[] = { "Auto", "HDR10", "SDR" };
    display_output::DisplayCapabilities capabilities;
    uint32_t storedMode = 0;
    int outputMode = 0;
    int n = 0;
    bool isSelected = false;
    bool vrrEnabled = false;
    float paperWhite = NAN;
    float peakNits = NAN;
    const char *vrrStatus = nullptr;

    if (!ImGui::TreeNodeEx("Display output", ImGuiTreeNodeFlags_DefaultOpen))
    {
        return;
    }

    storedMode = settings.getAs<uint32_t>("render/post/outputMode");
    outputMode = static_cast<int>(std::min(storedMode, static_cast<uint32_t>(display_output::OutputMode::SDR)));
    if (ImGui::BeginCombo("Dynamic range", outputModeItems[outputMode]))
    {
        for (n = 0; n < IM_ARRAYSIZE(outputModeItems); ++n)
        {
            isSelected = outputMode == n;
            if (ImGui::Selectable(outputModeItems[n], isSelected))
            {
                outputMode = n;
                settings.setAs<uint32_t>("render/post/outputMode", static_cast<uint32_t>(n));
            }
            if (isSelected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    paperWhite = settings.getAs<float>("render/post/paperWhiteNits");
    peakNits = settings.getAs<float>("render/post/peakNits");
    ImGui::BeginDisabled(outputMode == static_cast<int>(display_output::OutputMode::SDR));
    if (ImGui::DragFloat("Paper white", &paperWhite, 1.0f, 80.0f, 500.0f, "%.0f nits"))
    {
        peakNits = std::max(peakNits, paperWhite);
        settings.setAs<float>("render/post/paperWhiteNits", paperWhite);
        settings.setAs<float>("render/post/peakNits", peakNits);
    }
    if (ImGui::DragFloat("HDR peak", &peakNits, 10.0f, paperWhite, 10000.0f, "%.0f nits",
                         ImGuiSliderFlags_Logarithmic))
    {
        settings.setAs<float>("render/post/peakNits", peakNits);
    }
    ImGui::EndDisabled();

    vrrEnabled = settings.getAs<bool>("display/vrr/enabled");
    if (ImGui::Checkbox("Variable refresh rate", &vrrEnabled))
    {
        settings.setAs<bool>("display/vrr/enabled", vrrEnabled);
    }

    capabilities = display.getOutputCapabilities();
    vrrStatus = display_output::vrrStatusName(capabilities.vrrStatus);
    ImGui::SeparatorText("Capabilities");
    ImGui::TextDisabled(
        "HDR10: %s  |  selected: %s  |  metadata: %s",
        capabilities.output.hdr10 ? "supported" : "unavailable",
        capabilities.output.hdrSelected ? "yes" : "no",
        capabilities.output.hdrMetadata ? "supported" : "unavailable");
    ImGui::TextDisabled(
        "Present modes: FIFO%s%s%s",
        capabilities.present.fifoRelaxed ? ", FIFO_RELAXED" : "",
        capabilities.present.mailbox ? ", MAILBOX" : "",
        capabilities.present.immediate ? ", IMMEDIATE" : "");
    ImGui::TextDisabled(
        "Present wait: %s  |  present ID: %s  |  timing: %s",
        capabilities.presentWait ? "available" : "unavailable",
        capabilities.presentId ? "available" : "unavailable",
        capabilities.displayTiming ? "available" : "unavailable");
    ImGui::TextDisabled(
        "VRR: %s  |  current: %.3f Hz; Vulkan FIFO baseline",
        vrrStatus, capabilities.currentRefreshRateHz);
    if (capabilities.minRefreshRateHz > 0.0f &&
        capabilities.maxRefreshRateHz > 0.0f)
    {
        ImGui::TextDisabled(
            "Compositor VRR range: %.3f-%.3f Hz",
            capabilities.minRefreshRateHz, capabilities.maxRefreshRateHz);
    }
    ImGui::TreePop();
}
} // namespace

void EditorApp::drawRenderSettingsPanel()
{
    ImGui::Begin("Render Settings:");

    drawDisplayOutputSettings(*m_settingsManager, *m_display);

    {
        bool analytic = m_settingsManager->getAs<bool>("render/validate/analyticLights");
        if (ImGui::Checkbox("Analytic Lights", &analytic))
        {
            m_settingsManager->setAs<bool>("render/validate/analyticLights", analytic);
            m_sharedCtx->mSubframeIndex = 0;
        }
    }

    // Must match DebugMode in ShaderTypes.h, in order.
    const char* const debugViewOptions[] = { "None",
                                             "Normals",
                                             "Motion Blur",
                                             "AOV: diffuse",
                                             "AOV: specular",
                                             "AOV: normal",
                                             "AOV: roughness",
                                             "AOV: depth",
                                             "AOV: motion",
                                             "AOV: reactive",
                                             "AOV: spec hit distance",
                                             "Cache: voxel grid",
                                             "Cache: radiance",
                                             "Cache: occupancy",
                                             "Cache: bounce count" };
    // What each cache view is for, because none of them is self-explanatory and
    // all four exist to answer a specific question about a specific knob.
    const char* const debugViewHelp[] = {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        "A stable colour per cache voxel, at the first surface each camera ray reaches.\n"
        "This is how the voxel size below gets chosen -- it needs no cache allocated,\n"
        "so dial it in here before switching the cache on. Voxels should be small\n"
        "against the features you want the cache to keep apart, and large enough that\n"
        "many paths land in each.",
        "What the cache would answer at the first surface, shown directly instead of\n"
        "through the bounces that normally stand between a lookup and the pixel.\n"
        "Tonemapped like the beauty render, so the two can be compared side by side.\n"
        "Black means that voxel is missing or has not resolved yet.",
        "One block per table entry: green has resolved radiance, amber is inserted but\n"
        "has nothing to answer with yet, black is free. Around 10-20% occupied with a\n"
        "static camera is healthy. A table that is mostly amber is being evicted before\n"
        "it ever resolves -- raise the entry count or the stale-frame threshold.",
        "How deep paths actually went: blue none, green one, yellow two, red three or\n"
        "more. Comparing this with the cache off and on is the direct measurement of\n"
        "what the cache buys, and the only one that says *where*. A heatmap that does\n"
        "not cool when the cache is switched on is a scene that is not being cached,\n"
        "whatever the frame time says."
    };
    static_assert(IM_ARRAYSIZE(debugViewOptions) == IM_ARRAYSIZE(debugViewHelp),
                  "every debug view needs a help slot, even an empty one");
    static int currentDebugViewOption = 0;
    if (ImGui::BeginCombo("Debug view", debugViewOptions[currentDebugViewOption]))
    {
        for (int n = 0; n < IM_ARRAYSIZE(debugViewOptions); n++)
        {
            const bool is_selected = (currentDebugViewOption == n);
            if (ImGui::Selectable(debugViewOptions[n], is_selected))
            {
                if (currentDebugViewOption != n)
                {
                    currentDebugViewOption = n;
                    m_settingsManager->setAs<uint32_t>("render/pt/debug", currentDebugViewOption);
                }
            }
            if (debugViewHelp[n] != nullptr && ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(debugViewHelp[n]);
                ImGui::EndTooltip();
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    // Camera selection
    {
        const auto& cameras = m_scene->getCameras();
        const int cameraCount = static_cast<int>(cameras.size());
        if (cameraCount > 0)
        {
            const char* previewName = cameras[m_selectedCamera].name.c_str();
            if (ImGui::BeginCombo("Camera", previewName))
            {
                for (int n = 0; n < cameraCount; n++)
                {
                    const bool is_selected = (m_selectedCamera == n);
                    if (ImGui::Selectable(cameras[n].name.c_str(), is_selected))
                    {
                        if (m_selectedCamera != n)
                        {
                            m_selectedCamera = n;
                            setCameraDetached(false);
                            m_cameraController->setCamera(m_scene->getCamera(m_selectedCamera));
                            m_sharedCtx->mSubframeIndex = 0;
                            // A different camera is a cut: nothing in the previous
                            // frame reprojects into this one. The renderer cannot
                            // see this -- the jump may be small in world space.
                            m_render->resetTemporalHistory();
                            m_settingsManager->setAs<uint32_t>("render/selectedCamera", m_selectedCamera);
                        }
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            // Show re-attach button for GLTF cameras that have been detached
            auto& selectedCam = cameras[m_selectedCamera];
            if (selectedCam.node != -1 && m_cameraDetached)
            {
                ImGui::SameLine();
                if (ImGui::Button("Re-attach"))
                {
                    setCameraDetached(false);
                    m_sharedCtx->mSubframeIndex = 0;
                    m_render->resetTemporalHistory();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted("Resume following GLTF camera animation");
                    ImGui::EndTooltip();
                }
            }
        }
    }

    // Camera lens / DOF — physical units (mm, m, f-stop).
    if (ImGui::TreeNode("Camera / lens"))
    {
        oka::Camera& cam = m_scene->getCamera(m_selectedCamera);
        bool changed = false;
        const bool linkDof = m_settingsManager->getAs<bool>("render/post/tonemapper/linkDofFStop");
        float exposureFStop = m_settingsManager->getAs<float>("render/post/tonemapper/fStop");
        // Keep the lens f-stop aligned with exposure whenever the link is on so
        // lensRadius and the path tracer see the same N the exposure panel edits.
        if (linkDof)
        {
            cam.fStopDof = exposureFStop;
        }

        changed |= ImGui::DragFloat("Focal length", &cam.focalLengthMm, 0.5f, 1.0f, 500.0f, "%.1f mm",
                                    ImGuiSliderFlags_Logarithmic);
        changed |= ImGui::DragFloat("Sensor width", &cam.sensorWidth, 0.1f, 1.0f, 100.0f, "%.1f mm");
        changed |= ImGui::DragFloat("Sensor height", &cam.sensorHeight, 0.1f, 1.0f, 100.0f, "%.1f mm");
        ImGui::TextDisabled("Vertical FOV %.1f deg (from lens + sensor)",
                            editor_camera_exposure::verticalFovDegrees(cam.focalLengthMm, cam.sensorHeight));

        if (ImGui::Checkbox("Enable DOF", &cam.useDof))
            changed = true;

        if (cam.useDof)
        {
            if (ImGui::SliderFloat("Focus distance", &cam.focalDistance, 0.1f, 1000.0f, "%.2f m",
                                   ImGuiSliderFlags_Logarithmic))
                changed = true;

            float dofFStop = linkDof ? exposureFStop : cam.fStopDof;
            if (ImGui::DragFloat(linkDof ? "F-stop (linked)" : "DOF f-stop", &dofFStop, 0.05f, 0.7f, 32.0f, "f/%.1f"))
            {
                changed = true;
                if (linkDof)
                {
                    exposureFStop = dofFStop;
                    cam.fStopDof = dofFStop;
                    m_settingsManager->setAs<float>("render/post/tonemapper/fStop", exposureFStop);
                }
                else
                {
                    cam.fStopDof = dofFStop;
                }
            }
            ImGui::TextDisabled("Lens radius %.4f m",
                                editor_camera_exposure::lensRadiusMetres(cam.focalLengthMm, cam.fStopDof));

            if (ImGui::SliderInt("Aperture blades", &cam.apertureBlades, 0, 8))
                changed = true;
            ImGui::SameLine();
            ImGui::TextDisabled("(0 = circular)");

            float bladeDeg = editor_camera_exposure::degreesFromRadians(cam.bladeRotation);
            if (ImGui::DragFloat("Blade rotation", &bladeDeg, 1.0f, 0.0f, 360.0f, "%.0f deg"))
            {
                cam.bladeRotation = editor_camera_exposure::radiansFromDegrees(bladeDeg);
                changed = true;
            }
            if (ImGui::DragFloat("Anamorphic ratio", &cam.anamorphicRatio, 0.01f, 0.25f, 4.0f, "%.2f"))
                changed = true;
        }

        if (ImGui::DragFloat("Shift X", &cam.shiftX, 0.01f, -2.0f, 2.0f, "%.3f (sensor frac)"))
            changed = true;
        if (ImGui::DragFloat("Shift Y", &cam.shiftY, 0.01f, -2.0f, 2.0f, "%.3f (sensor frac)"))
            changed = true;

        if (changed)
            m_sharedCtx->mSubframeIndex = 0;

        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Preview Resolution", ImGuiTreeNodeFlags_DefaultOpen))
    {
        uint32_t previewWidth = m_settingsManager->getAs<uint32_t>("render/width");
        uint32_t previewHeight = m_settingsManager->getAs<uint32_t>("render/height");
        const int preset = editor_viewport::findPreset(previewWidth, previewHeight);
        static bool customSelected = false;
        int selectedPreset =
            !customSelected && preset >= 0 ? preset : static_cast<int>(editor_viewport::kPreviewPresets.size());
        const char* const presetNames[] = {
            editor_viewport::kPreviewPresets[0].label,
            editor_viewport::kPreviewPresets[1].label,
            editor_viewport::kPreviewPresets[2].label,
            editor_viewport::kPreviewPresets[3].label,
            "Custom",
        };
        if (ImGui::Combo("Preset", &selectedPreset, presetNames, IM_ARRAYSIZE(presetNames)))
        {
            customSelected = selectedPreset == static_cast<int>(editor_viewport::kPreviewPresets.size());
            if (!customSelected && selectedPreset >= 0 &&
                selectedPreset < static_cast<int>(editor_viewport::kPreviewPresets.size()))
            {
                const editor_viewport::PreviewPreset& selected = editor_viewport::kPreviewPresets[selectedPreset];
                requestPreviewResolution(selected.width, selected.height);
                previewWidth = selected.width;
                previewHeight = selected.height;
            }
        }

        if (customSelected)
        {
            static bool lockAspect = true;
            int customWidth = static_cast<int>(previewWidth);
            int customHeight = static_cast<int>(previewHeight);
            const float aspect =
                previewHeight > 0 ? static_cast<float>(previewWidth) / static_cast<float>(previewHeight) : 1.0f;
            if (ImGui::InputInt("Width", &customWidth))
            {
                const uint32_t width = editor_viewport::clampPreviewDimension(customWidth);
                const uint32_t height =
                    lockAspect
                        ? editor_viewport::clampPreviewDimension(
                              static_cast<int>(std::lround(static_cast<float>(width) / aspect)))
                               : previewHeight;
                requestPreviewResolution(width, height);
            }
            if (ImGui::InputInt("Height", &customHeight))
            {
                const uint32_t height = editor_viewport::clampPreviewDimension(customHeight);
                const uint32_t width =
                    lockAspect
                        ? editor_viewport::clampPreviewDimension(
                              static_cast<int>(std::lround(static_cast<float>(height) * aspect)))
                               : previewWidth;
                requestPreviewResolution(width, height);
            }
            ImGui::Checkbox("Lock aspect ratio", &lockAspect);
            ImGui::SameLine();
            if (ImGui::Button("Swap"))
            {
                requestPreviewResolution(previewHeight, previewWidth);
            }
        }

        ImGui::TextDisabled("Viewport resize changes presentation only");
        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Path Tracer"))
    {
        const char* rectlightSamplingMethodItems[] = { "Uniform", "Advanced" };
        static int currentRectlightSamplingMethodItemId = 0;
        if (ImGui::BeginCombo("Rect Light Sampling", rectlightSamplingMethodItems[currentRectlightSamplingMethodItemId]))
        {
            for (const auto& item : rectlightSamplingMethodItems)
            {
                const bool is_selected =
                    (item == rectlightSamplingMethodItems[currentRectlightSamplingMethodItemId]);
                if (ImGui::Selectable(item, is_selected))
                {
                    currentRectlightSamplingMethodItemId =
                        static_cast<int>(&item - rectlightSamplingMethodItems);
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            m_settingsManager->setAs<uint32_t>("render/pt/rectLightSamplingMethod", currentRectlightSamplingMethodItemId);
            ImGui::EndCombo();
        }

        const char* samplerTypeItems[] = { "Halton", "PCG", "Sobol (Owen)", "Sobol + blue noise",
                                           "Hybrid (blue noise -> Sobol)" };
        // Read back rather than remembered in a static: the default is set in
        // loadSettings, and a static starting at zero showed "Halton" no matter
        // what was actually running.
        int currentSamplerTypeId =
            (int)std::min(m_settingsManager->getAs<uint32_t>("render/pt/samplerType"), 4u);
        if (ImGui::BeginCombo("Sampler", samplerTypeItems[currentSamplerTypeId]))
        {
            for (const auto& item : samplerTypeItems)
            {
                const bool is_selected = (item == samplerTypeItems[currentSamplerTypeId]);
                if (ImGui::Selectable(item, is_selected))
                {
                    currentSamplerTypeId = (int)(&item - samplerTypeItems);
                    m_settingsManager->setAs<uint32_t>("render/pt/samplerType", currentSamplerTypeId);
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (currentSamplerTypeId == 0)
        {
            ImGui::TextDisabled("Halton aliases its bases every 32 dimensions; error stalls past ~512 spp.");
        }
        if (currentSamplerTypeId == 4)
        {
            auto bnSwitch = m_settingsManager->getAs<uint32_t>("render/pt/blueNoiseSwitchSpp");
            if (ImGui::SliderInt("Blue-noise samples", (int*)&bnSwitch, 0, 256))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/blueNoiseSwitchSpp", bnSwitch);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Samples drawn from the blue-noise sequence before handing over to\n"
                                  "per-pixel scrambling. Blue noise looks cleaner at low sample counts;\n"
                                  "scrambling converges faster past a few dozen.");
            }
        }

        // One choice, not two checkboxes -- and this backend's choices, not the
        // other backend's.
        //
        // Denoising and upscaling are alternatives on both: MetalFX sends a frame
        // through the spatial scaler or the temporal denoised one, and the OptiX
        // plan has upscaling imply denoising because nothing in it scales without
        // also running the network. As separate toggles they offered four states,
        // two of which meant the same thing and none of which said so.
        //
        // Which states exist, what they are called, whether the render scale is a
        // slider or follows from the mode, and what the fallback warning means
        // are all the backend's answer to give -- see editor_denoiser_ui.h. This
        // panel used to hard-code MetalFX's answers and show them over OptiX,
        // where the name was wrong and the scale slider did nothing.
        const editor_denoiser::Ui fx = editor_denoiser::uiFor(m_render->denoiserKind());
        const float requestedScale = m_settingsManager->getAs<float>("render/pt/upscaleFactor");
        const bool denoiseSetting = m_settingsManager->getAs<bool>("render/pt/denoise");
        const bool upscaleSetting = m_settingsManager->getAs<bool>("render/pt/enableUpscale");
        // Keep the remembered index and what is actually running in step, every
        // frame rather than once. A mode index outliving the list it indexed is
        // how the combo came to show one thing while the renderer ran another: it
        // displayed whatever sat at that index, and the next click picked
        // something nobody asked for. The settings move without the panel too --
        // the frame-budget button, the benchmark drivers, STRELKA_DENOISE.
        if (fx.modeCount > 0 &&
            (!mDenoiseModeInitialized || mDenoiseModeIndex >= fx.modeCount ||
             !editor_denoiser::settingsMatchMode(fx, mDenoiseModeIndex, denoiseSetting, upscaleSetting,
                                                 requestedScale)))
        {
            mDenoiseModeIndex = editor_denoiser::modeIndexFromSettings(fx, denoiseSetting, upscaleSetting);
            mDenoiseModeInitialized = true;
        }

        if (!editor_denoiser::hasDenoiser(fx))
        {
            ImGui::TextDisabled("This backend has no denoiser");
        }
        else
        {
            if (ImGui::BeginCombo(fx.title, editor_denoiser::modeAt(fx, mDenoiseModeIndex).label))
            {
                for (int n = 0; n < fx.modeCount; n++)
                {
                    const bool is_selected = (mDenoiseModeIndex == n);
                    if (ImGui::Selectable(fx.modes[n].label, is_selected) && mDenoiseModeIndex != n)
                    {
                        mDenoiseModeIndex = n;
                        m_settingsManager->setAs<bool>("render/pt/denoise", fx.modes[n].denoise);
                        // The denoiser is a scaler too: it needs the reduced-
                        // resolution render whenever the mode asks for one, and
                        // nothing else does.
                        m_settingsManager->setAs<bool>(
                            "render/pt/enableUpscale", editor_denoiser::shouldUpscale(fx, n, requestedScale));
                        m_render->resetTemporalHistory();
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            const editor_denoiser::Mode fxMode = editor_denoiser::modeAt(fx, mDenoiseModeIndex);
            const bool denoiserOn = fxMode.denoise || fxMode.upscale;
            if (denoiserOn && fx.modeHint != nullptr)
            {
                ImGui::TextDisabled("%s", fx.modeHint);
            }

            // Temporal is a property of the network on OptiX rather than a mode of
            // its own: both the denoise-only and the 2x model have a temporal
            // variant, so it is one switch instead of a doubled list.
            if (fx.temporalToggle && denoiserOn)
            {
                bool temporal = m_settingsManager->getAs<uint32_t>("render/pt/upscaleMode") == 1u;
                if (ImGui::Checkbox("Temporal", &temporal))
                {
                    m_settingsManager->setAs<uint32_t>("render/pt/upscaleMode", temporal ? 1u : 0u);
                    m_render->resetTemporalHistory();
                }
                ImGui::SameLine();
                ImGui::BeginDisabled();
                ImGui::TextUnformatted(temporal ? "(reprojects the previous frame)" : "(each frame denoised alone)");
                ImGui::EndDisabled();
            }

            if (fx.playbackMotionBlurToggle && fxMode.denoise)
            {
                bool playbackBlur =
                    m_settingsManager->getAs<bool>(
                        "render/pt/denoisePlaybackMotionBlur");
                if (ImGui::Checkbox("Path-traced playback blur", &playbackBlur))
                {
                    m_settingsManager->setAs<bool>(
                        "render/pt/denoisePlaybackMotionBlur", playbackBlur);
                    m_render->resetTemporalHistory();
                }
                ImGui::SameLine();
                ImGui::BeginDisabled();
                ImGui::TextUnformatted(
                    playbackBlur ? "(uses SPP per frame)" : "(stable shutter-close guides)");
                ImGui::EndDisabled();
            }

            if (denoiserOn && fx.freeRenderScale)
            {
                float factor = requestedScale;
                if (ImGui::SliderFloat("PT scale inside preview", &factor, 0.25f, 1.0f, "%.2f"))
                {
                    m_settingsManager->setAs<float>("render/pt/upscaleFactor", factor);
                    m_settingsManager->setAs<bool>(
                        "render/pt/enableUpscale", editor_denoiser::shouldUpscale(fx, mDenoiseModeIndex, factor));
                    m_render->resetTemporalHistory();
                }
                ImGui::SameLine();
                ImGui::BeginDisabled();
                const char* const scaleStatus = factor < 1.0f
                                                    ? "(rendering below display resolution)"
                                                    : (fxMode.denoise ? "(denoising at 1:1)"
                                                                      : "(inactive at 1:1; lower scale to enable)");
                ImGui::TextUnformatted(scaleStatus);
                ImGui::EndDisabled();
            }
        }

        const uint32_t displayWidth = m_settingsManager->getAs<uint32_t>("render/width");
        const uint32_t displayHeight = m_settingsManager->getAs<uint32_t>("render/height");
        const editor_denoiser::Resolution previewRes =
            editor_denoiser::resolution(fx, mDenoiseModeIndex, requestedScale, displayWidth, displayHeight);
        ImGui::TextDisabled("PT internal: %u x %u", previewRes.pathTraceWidth, previewRes.pathTraceHeight);
        ImGui::TextDisabled("Preview output: %u x %u", previewRes.outputWidth, previewRes.outputHeight);
        if (m_render->denoiserFallbackActive())
        {
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f), "%s", fx.fallbackMessage);
        }
        const double lastGpuMs = m_render->getLastRenderTimeMs();
        if (lastGpuMs > editor_frame_budget::kInteractiveBudgetMs)
        {
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f),
                               "Last PT frame: %.0f ms (interactive budget: %.0f ms)", lastGpuMs,
                               editor_frame_budget::kInteractiveBudgetMs);
            // The scale the frame was *actually* traced at, not the one the
            // slider holds: the budget divides a measured GPU time by a pixel
            // count, and on a fixed-ratio backend the slider is not that count.
            const float tracedScale = editor_denoiser::appliedScale(fx, mDenoiseModeIndex, requestedScale);
            const editor_frame_budget::RenderSettingsSnapshot current{
                displayWidth,
                displayHeight,
                tracedScale < 1.0f,
                tracedScale,
            };
            const editor_frame_budget::FrameSample sample = editor_frame_budget::sampleFrom(lastGpuMs, current);
            if (fx.freeRenderScale)
            {
                const float suggestedScale = editor_frame_budget::recommendedScale(sample, displayWidth, displayHeight);
                // Scaling, not denoising: the cheapest way to buy frame time, and
                // the one that does not depend on a history the camera is about to
                // invalidate anyway.
                const int scalingMode = editor_denoiser::modeIndexFromSettings(fx, false, true);
                const std::string label = fmt::format("Lower PT scale to {:.2f}", suggestedScale);
                if (ImGui::Button(label.c_str()))
                {
                    m_settingsManager->setAs<bool>("render/pt/denoise", fx.modes[scalingMode].denoise);
                    m_settingsManager->setAs<uint32_t>("render/pt/upscaleMode", 0);
                    m_settingsManager->setAs<float>("render/pt/upscaleFactor", suggestedScale);
                    m_settingsManager->setAs<bool>(
                        "render/pt/enableUpscale", editor_denoiser::shouldUpscale(fx, scalingMode, suggestedScale));
                    mDenoiseModeIndex = scalingMode;
                    mDenoiseModeInitialized = true;
                    m_render->resetTemporalHistory();
                }
            }
            else if (editor_denoiser::hasDenoiser(fx))
            {
                // Nothing to lower: this backend's only lever is its fixed ratio,
                // so the offer is to switch it on rather than to pick a number.
                const int scalingMode = editor_denoiser::modeIndexFromSettings(fx, true, true);
                const std::string label = fmt::format("Switch to \"{}\"", fx.modes[scalingMode].label);
                ImGui::BeginDisabled(mDenoiseModeIndex == scalingMode);
                if (ImGui::Button(label.c_str()))
                {
                    m_settingsManager->setAs<bool>("render/pt/denoise", fx.modes[scalingMode].denoise);
                    m_settingsManager->setAs<bool>(
                        "render/pt/enableUpscale", editor_denoiser::shouldUpscale(fx, scalingMode, requestedScale));
                    mDenoiseModeIndex = scalingMode;
                    mDenoiseModeInitialized = true;
                    m_render->resetTemporalHistory();
                }
                ImGui::EndDisabled();
            }
        }

        auto maxDepth = m_settingsManager->getAs<uint32_t>("render/pt/depth");
        if (ImGui::SliderInt("Max Depth", (int*)&maxDepth, 1, 16))
        {
            m_settingsManager->setAs<uint32_t>("render/pt/depth", maxDepth);
        }

        auto sppSubframe = m_settingsManager->getAs<uint32_t>("render/pt/spp");
        if (ImGui::SliderInt("SPP per frame", (int*)&sppSubframe, 1, 32))
        {
            m_settingsManager->setAs<uint32_t>("render/pt/spp", sppSubframe);
        }

        auto sppTotal = m_settingsManager->getAs<uint32_t>("render/pt/sppTotal");
        if (ImGui::SliderInt("Accumulation SPP limit", (int*)&sppTotal, 1, 10000))
        {
            m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", sppTotal);
        }

        bool accumulationEnabled = m_settingsManager->getAs<bool>("render/pt/enableAcc");
        if (ImGui::Checkbox("Accumulate while still", &accumulationEnabled))
        {
            m_settingsManager->setAs<bool>("render/pt/enableAcc", accumulationEnabled);
        }

        ImGui::TreePop();
    }

    // --- Radiance cache ----------------------------------------------------
    //
    // Everything here changes what the image is, not just how fast it arrives,
    // so every control restarts accumulation. Counting occupancy is a pass over
    // the whole table, so it is asked for only while this node is open -- which
    // is what the setting outside the `if` turns back off again.
    {
        const bool cachePanelOpen = ImGui::TreeNode("Radiance cache (SHaRC)");
        m_settingsManager->setAs<bool>("render/pt/sharcReportOccupancy", cachePanelOpen);
        if (cachePanelOpen)
        {
            auto restart = [this]() { m_sharedCtx->mSubframeIndex = 0; };

            bool cacheEnabled = m_settingsManager->getAs<bool>("render/pt/sharc");
            if (ImGui::Checkbox("Enable", &cacheEnabled))
            {
                m_settingsManager->setAs<bool>("render/pt/sharc", cacheEnabled);
                restart();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(
                    "Let a path stop after a few bounces and read what the rest of it\n"
                    "would have gathered, averaged over every path that has passed\n"
                    "through the same place. Trades a little bias for path length.");
                ImGui::EndTooltip();
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset cache"))
            {
                // Consumed by the renderer on the next frame. Worth having as a
                // button: the cache deliberately survives camera movement now,
                // so this is the only way to see a scene cached from nothing.
                m_settingsManager->setAs<bool>("render/pt/sharcReset", true);
                restart();
            }

            uint32_t entriesUsed = 0;
            uint32_t capacity = 0;
            if (m_render != nullptr && m_render->radianceCacheOccupancy(entriesUsed, capacity) && capacity > 0)
            {
                const float occupancy = 100.0f * (float)entriesUsed / (float)capacity;
                ImGui::Text("Occupancy: %.1f%%  (%u / %u entries)", (double)occupancy, entriesUsed, capacity);
                // The SDK's own reading of this number, which is the only thing
                // that makes it actionable.
                if (occupancy > 60.0f)
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f),
                                       "Table is crowded -- raise entries, or evict sooner.");
                }
            }
            else if (cacheEnabled)
            {
                ImGui::TextDisabled("Occupancy: not reported by this backend");
            }

            // Entries, as an exponent: the table is masked rather than divided
            // into, so it has to be a power of two, and a free-typed number
            // would only be rounded down behind the user's back.
            uint32_t entries = m_settingsManager->getAs<uint32_t>("render/pt/sharcCapacity");
            int exponent = 22;
            while ((1u << exponent) > entries && exponent > 16)
            {
                --exponent;
            }
            if (ImGui::SliderInt("Entries (2^n)", &exponent, 16, 25))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/sharcCapacity", 1u << exponent);
                restart();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::Text("%u entries, %.0f MB.\nMore entries means fewer probe runs that come back full.",
                            1u << exponent, (double)(1u << exponent) * 40.0 / 1e6);
                ImGui::EndTooltip();
            }

            float voxelPixels = m_settingsManager->getAs<float>("render/pt/sharcVoxelPixels");
            if (ImGui::SliderFloat("Voxel size (px)", &voxelPixels, 1.0f, 32.0f, "%.1f"))
            {
                m_settingsManager->setAs<float>("render/pt/sharcVoxelPixels", voxelPixels);
                restart();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(
                    "How many pixels wide a voxel is, at any distance -- the size follows\n"
                    "the distance to the camera, so one number means the same thing in a\n"
                    "room and in a forest. Use the 'Cache: voxel grid' debug view to set it.");
                ImGui::EndTooltip();
            }

            uint32_t firstBounce = m_settingsManager->getAs<uint32_t>("render/pt/sharcDepth");
            if (ImGui::SliderInt("First cached bounce", (int*)&firstBounce, 0, 8))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/sharcDepth", firstBounce);
                restart();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(
                    "Bounces before this are always traced. The camera ray and the first\n"
                    "bounce carry the detail a voxel average would blur, so reading the\n"
                    "cache too early shows up as flat, blotchy indirect light.");
                ImGui::EndTooltip();
            }

            uint32_t readFrames = m_settingsManager->getAs<uint32_t>("render/pt/sharcReadFrames");
            if (ImGui::SliderInt("Read until (samples)", (int*)&readFrames, 0, 1024))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/sharcReadFrames", readFrames);
                restart();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(
                    "Stop reading the cache once this many samples have accumulated;\n"
                    "0 never stops. Deposits carry on either way, so the cache is warm\n"
                    "the moment the camera moves again.\n\n"
                    "This is not a taste setting. The cache's error is one value per\n"
                    "voxel held across a temporal window, so it is correlated in space\n"
                    "and time and does not average away -- it is a floor, while plain\n"
                    "path tracing keeps converging past it. Measured on the isometric\n"
                    "bathroom against a 4096-spp reference, structured error only:\n"
                    "  16 spp   0.154 without, 0.098 with   -- cache half the error\n"
                    "  64 spp   0.063 without, 0.039 with   -- cache half the error\n"
                    " 256 spp   0.022 without, 0.019 with   -- level\n"
                    "1024 spp   0.006 without, 0.012 with   -- cache twice the error\n\n"
                    "So the cache is both the faster and the better image while you are\n"
                    "moving, and the thing in the way once you stop.");
                ImGui::EndTooltip();
            }

            uint32_t minSamples = m_settingsManager->getAs<uint32_t>("render/pt/sharcMinSamples");
            if (ImGui::SliderInt("Min samples to read", (int*)&minSamples, 1, 256))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/sharcMinSamples", minSamples);
                restart();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(
                    "How much a voxel has to have seen before a path will believe it.\n"
                    "Too low and the cache spreads one path's noise over a region.");
                ImGui::EndTooltip();
            }

            uint32_t accumFrames = m_settingsManager->getAs<uint32_t>("render/pt/sharcAccumFrames");
            if (ImGui::SliderInt("Temporal window (frames)", (int*)&accumFrames, 1, 256))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/sharcAccumFrames", accumFrames);
                restart();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(
                    "How many frames a voxel averages over. Larger is quieter and slower\n"
                    "to notice that the lighting changed -- a light switched on takes\n"
                    "roughly this many frames to appear in the cache.");
                ImGui::EndTooltip();
            }

            uint32_t staleFrames = m_settingsManager->getAs<uint32_t>("render/pt/sharcStaleFrames");
            if (ImGui::SliderInt("Evict after (frames)", (int*)&staleFrames, 8, 512))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/sharcStaleFrames", staleFrames);
                restart();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(
                    "How long an entry survives with nothing deposited into it. This is\n"
                    "what lets the cache outlive a moving camera instead of being thrown\n"
                    "away by it. Evicting too eagerly costs more in re-insertion than the\n"
                    "slots are worth, so small values are clamped.");
                ImGui::EndTooltip();
            }

            // Responsive lighting. The controls are shown whether or not the
            // scene has a responsive light, because the answer to "why is this
            // doing nothing" is on the light's own panel and a control that is
            // not there cannot say so.
            ImGui::SeparatorText("Responsive lighting");
            bool responsiveEnabled = m_settingsManager->getAs<bool>("render/pt/sharcResponsiveLighting");
            if (ImGui::Checkbox("Enable##sharcResponsive", &responsiveEnabled))
            {
                m_settingsManager->setAs<bool>("render/pt/sharcResponsiveLighting", responsiveEnabled);
                restart();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(
                    "Cache lights marked Responsive -- on the light's own panel -- in a\n"
                    "second entry per voxel with a much shorter window, so they can change\n"
                    "faster than the rest of the cache follows. Does nothing unless some\n"
                    "light is marked; turning it off here is how you A/B a scene that has\n"
                    "one.");
                ImGui::EndTooltip();
            }

            uint32_t responsiveFrames = m_settingsManager->getAs<uint32_t>("render/pt/sharcResponsiveFrames");
            if (ImGui::SliderInt("Responsive window (frames)", (int*)&responsiveFrames, 1, 64))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/sharcResponsiveFrames", responsiveFrames);
                restart();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(
                    "The window responsive entries average over, and how long they survive\n"
                    "unvisited. Both, because they are the same trade: short enough to\n"
                    "follow the light, long enough not to be noise. Well below the window\n"
                    "above, or there is no point having two.");
                ImGui::EndTooltip();
            }

            ImGui::TreePop();
        }
    }

    if (ImGui::Button("Save Preview Screenshot"))
    {
        // Generate default filename with timestamp
        const std::time_t now = std::time(nullptr);
        std::tm localTime{};
        const std::tm* tm = localtime_r(&now, &localTime);
        // localtime() returns null for a clock it cannot convert, and strftime()
        // returns 0 when the result would not fit; either way the dialog still
        // needs a name to open with.
        std::string defaultName = "screenshot.exr";
        char stamp[64];
        if (tm != nullptr && std::strftime(stamp, sizeof(stamp), "screenshot_%Y%m%d_%H%M%S.exr", tm) != 0)
        {
            defaultName = stamp;
        }

        IGFD::FileDialogConfig config{};
        config.path = ".";
        config.fileName = defaultName;
        ImGuiFileDialog::Instance()->OpenDialog(
            "SaveScreenshotDlgKey", "Save Screenshot", ".exr,.png", config);
    }

    auto cameraSpeed = m_settingsManager->getAs<float>("render/cameraSpeed");
    ImGui::InputFloat("Camera Speed", (float*)&cameraSpeed, 0.5);
    m_settingsManager->setAs<float>("render/cameraSpeed", cameraSpeed);

    // Exposure, the camera side of the tone curve. The renderer computes
    //     film speed  > 0 : cm2_factor * iso / (shutter * fstop^2) / 100
    //     film speed == 0 : cm2_factor
    // so a zero film speed is the arbitrary-units mode, which is what a scene lit
    // in normalised rather than photometric units wants -- and what the light
    // sidecar writes. Both forms are editable here because the sidecar can carry
    // either, and a scene that opens too dark is otherwise unexplainable from
    // inside the editor.
    if (ImGui::TreeNode("Photographic exposure"))
    {
        float iso = m_settingsManager->getAs<float>("render/post/tonemapper/filmIso");
        float fStop = m_settingsManager->getAs<float>("render/post/tonemapper/fStop");
        float shutter = m_settingsManager->getAs<float>("render/post/tonemapper/shutterSpeed");
        float cm2 = m_settingsManager->getAs<float>("render/post/tonemapper/cm2_factor");
        bool linkDof = m_settingsManager->getAs<bool>("render/post/tonemapper/linkDofFStop");
        bool changed = false;

        int mode = iso > 0.0f ? 0 : 1;
        const char* const modeItems[] = { "Photographic", "Multiplier" };
        if (ImGui::Combo("Mode", &mode, modeItems, 2))
        {
            editor_camera_exposure::carryExposureAcrossModeSwitch(mode == 1, iso, fStop, shutter, cm2);
            changed = true;
        }

        if (ImGui::Checkbox("Link DOF aperture to exposure", &linkDof))
        {
            m_settingsManager->setAs<bool>("render/post/tonemapper/linkDofFStop", linkDof);
            if (linkDof)
            {
                // Keep film brightness: push exposure f-stop into the lens.
                m_scene->getCamera(m_selectedCamera).fStopDof = fStop;
                m_sharedCtx->mSubframeIndex = 0;
            }
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("When on, one f-stop drives both DOF blur and photographic exposure.\n"
                              "Untick to blur the lens without changing film brightness.");
        }

        if (mode == 0)
        {
            changed |= ImGui::DragFloat("Film ISO", &iso, 1.0f, 1.0f, 25600.0f, "%.0f",
                                        ImGuiSliderFlags_Logarithmic);
            if (ImGui::DragFloat(linkDof ? "Aperture (linked)" : "Exposure f-stop", &fStop, 0.05f, 0.7f, 32.0f,
                                 "f/%.1f"))
            {
                changed = true;
                if (linkDof)
                {
                    m_scene->getCamera(m_selectedCamera).fStopDof = fStop;
                }
            }
            changed |= ImGui::DragFloat("Shutter", &shutter, 1.0f, 1.0f, 8000.0f, "1/%.0f s",
                                        ImGuiSliderFlags_Logarithmic);
            changed |= ImGui::DragFloat("cd/m^2 factor", &cm2, 0.01f, 0.0001f, 100000.0f, "%.4f",
                                        ImGuiSliderFlags_Logarithmic);
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Photometric scale (candela per square metre factor).\n"
                                  "Not a generic exposure multiplier — use Mode=Multiplier for that.");
            }
            const float linear = editor_camera_exposure::photographicLinearScale(iso, fStop, shutter, cm2);
            const float ev = editor_camera_exposure::ev100(iso, fStop, shutter);
            ImGui::TextDisabled("EV100 %.2f  |  Linear radiance x%.4f", ev, linear);
        }
        else
        {
            changed |= ImGui::DragFloat("Linear multiplier", &cm2, 0.01f, 0.0001f, 100000.0f, "x%.4f",
                                        ImGuiSliderFlags_Logarithmic);
            ImGui::TextDisabled("Linear radiance x%.4f (photographic controls off)", cm2);
        }

        if (ImGui::Button("Auto-expose"))
        {
            m_autoExposurePending = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(meters frame → Multiplier mode)");

        if (changed)
        {
            m_settingsManager->setAs<float>("render/post/tonemapper/filmIso", iso);
            m_settingsManager->setAs<float>("render/post/tonemapper/fStop", fStop);
            m_settingsManager->setAs<float>("render/post/tonemapper/shutterSpeed", shutter);
            m_settingsManager->setAs<float>("render/post/tonemapper/cm2_factor", cm2);
            m_sharedCtx->mSubframeIndex = 0;
        }

        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Display tonemap"))
    {
        const char* const tonemapItems[] = { "None", "Reinhard", "ACES", "Filmic" };
        int currentTonemapItemId = (int)std::min(m_settingsManager->getAs<uint32_t>("render/pt/tonemapperType"), 3u);
        if (ImGui::BeginCombo("Operator", tonemapItems[currentTonemapItemId]))
        {
            for (int n = 0; n < IM_ARRAYSIZE(tonemapItems); n++)
            {
                const bool is_selected = (currentTonemapItemId == n);
                if (ImGui::Selectable(tonemapItems[n], is_selected))
                {
                    currentTonemapItemId = n;
                    m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", (uint32_t)n);
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        auto gamma = m_settingsManager->getAs<float>("render/post/gamma");
        if (ImGui::DragFloat("Gamma", &gamma, 0.05f, 0.0f, 5.0f, "%.2f"))
        {
            m_settingsManager->setAs<float>("render/post/gamma", gamma);
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("0 = off; default 2.4 is an sRGB-like transfer, not a pure power.");
        }

        const float maxEdr = m_settingsManager->getAs<float>("render/post/tonemapper/maxEDR");
        ImGui::TextDisabled("Display max EDR %.2f (tone-map shoulder follows screen headroom)", maxEdr);

        ImGui::TreePop();
    }

    auto materialRayTmin = m_settingsManager->getAs<float>("render/pt/dev/materialRayTmin");
    ImGui::InputFloat("Material ray T min", (float*)&materialRayTmin, 0.1);
    m_settingsManager->setAs<float>("render/pt/dev/materialRayTmin", materialRayTmin);
    auto shadowRayTmin = m_settingsManager->getAs<float>("render/pt/dev/shadowRayTmin");
    ImGui::InputFloat("Shadow ray T min", (float*)&shadowRayTmin, 0.1);
    m_settingsManager->setAs<float>("render/pt/dev/shadowRayTmin", shadowRayTmin);

    ImGui::End();
}

// The scene load, while it is happening.
//
// Shown for the parse and for the GPU build alike: to the user those are one
// wait, and the fact that one runs on a worker and the other a stage per frame
// on the main loop is not something the window should expose.
void EditorApp::drawLoadingOverlay()
{
    const bool building = m_render && m_render->isBuildingScene();
    if (!m_isLoading && !building)
    {
        return;
    }

    // Weighted by measured cost on a large scene (the pine forest, seconds), not
    // by stage count: evenly divided, the bar would spend 40% of the wait in one
    // sixth of its length, which reads as stuck rather than slow. Indexed by
    // Stage, which is declared in run order for this to be meaningful.
    static constexpr float kStageWeights[(size_t)LoadProgress::Stage::Count] = {
        0.00f, // Idle
        1.12f, // Reading
        0.73f, // Parsing
        0.14f, // Geometry
        1.07f, // Textures
        1.95f, // Structures
        0.04f, // Environment
        0.00f, // Done
    };
    static const char* const kStageNames[(size_t)LoadProgress::Stage::Count] = {
        "Starting", "Reading file",   "Parsing scene", "Uploading geometry",
        "Loading textures", "Building acceleration structures", "Environment", "Finishing",
    };

    const uint32_t stage =
        std::min(m_loadProgress.stage.load(std::memory_order_acquire), (uint32_t)LoadProgress::Stage::Done);
    const uint32_t done = m_loadProgress.done.load(std::memory_order_relaxed);
    const uint32_t total = m_loadProgress.total.load(std::memory_order_relaxed);

    float totalWeight = 0.0f;
    for (const float w : kStageWeights)
    {
        totalWeight += w;
    }
    float before = 0.0f;
    for (uint32_t i = 0; i < stage; ++i)
    {
        before += kStageWeights[i];
    }
    // A stage that cannot say how much work it holds contributes nothing beyond
    // its starting point, rather than pretending to be complete.
    const float within = total > 0 ? std::min(1.0f, (float)done / (float)total) : 0.0f;
    const float fraction = totalWeight > 0.0f ? (before + kStageWeights[stage] * within) / totalWeight : 0.0f;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(460.0f, 0.0f), ImGuiCond_Always);
    ImGui::Begin("##Loading", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing);

    ImGui::TextUnformatted(std::filesystem::path(m_sceneFile).filename().string().c_str());
    ImGui::Spacing();

    const std::string label =
        total > 0 ? fmt::format("{}  {}/{}", kStageNames[stage], done, total) : std::string(kStageNames[stage]);
    ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 0.0f), label.c_str());

    // Only the parse can be abandoned. The GPU build hands out buffers and
    // acceleration structures that the renderer is already holding, so stopping
    // halfway would leave it in a state nothing else knows how to describe --
    // and it is the shorter half of the wait anyway.
    ImGui::Spacing();
    ImGui::BeginDisabled(!m_isLoading || m_loadProgress.isCancelled());
    if (ImGui::Button("Cancel"))
    {
        m_loadProgress.cancel();
    }
    ImGui::EndDisabled();
    if (m_loadProgress.isCancelled())
    {
        ImGui::SameLine();
        ImGui::TextDisabled("cancelling...");
    }

    ImGui::End();
}

} // namespace oka
