#include "../EditorApp.h"
#include "../editor_camera_exposure.h"

#include "imgui.h"
#include "ImGuiFileDialog.h"

#include <cfloat>
#include <ctime>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace oka
{

void EditorApp::drawRenderSettingsPanel()
{
    ImGui::Begin("Render Settings:");

    {
        bool analytic = m_settingsManager->getAs<bool>("render/validate/analyticLights");
        if (ImGui::Checkbox("Analytic Lights", &analytic))
        {
            m_settingsManager->setAs<bool>("render/validate/analyticLights", analytic);
            m_sharedCtx->mSubframeIndex = 0;
        }
    }

    // Must match DebugMode in ShaderTypes.h, in order.
    const char* debugViewOptions[] = { "None",           "Normals",        "Motion Blur",
                                       "AOV: diffuse",   "AOV: specular",  "AOV: normal",
                                       "AOV: roughness", "AOV: depth",     "AOV: motion",
                                       "AOV: reactive",  "AOV: spec hit distance" };
    static int currentDebugViewOption = 0;
    if (ImGui::BeginCombo("Debug view", debugViewOptions[currentDebugViewOption]))
    {
        for (int n = 0; n < IM_ARRAYSIZE(debugViewOptions); n++)
        {
            bool is_selected = (currentDebugViewOption == n);
            if (ImGui::Selectable(debugViewOptions[n], is_selected))
            {
                if (currentDebugViewOption != n)
                {
                    currentDebugViewOption = n;
                    m_settingsManager->setAs<uint32_t>("render/pt/debug", currentDebugViewOption);
                }
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
        int cameraCount = (int)cameras.size();
        if (cameraCount > 0)
        {
            const char* previewName = cameras[m_selectedCamera].name.c_str();
            if (ImGui::BeginCombo("Camera", previewName))
            {
                for (int n = 0; n < cameraCount; n++)
                {
                    bool is_selected = (m_selectedCamera == n);
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
        bool linkDof = m_settingsManager->getAs<bool>("render/post/tonemapper/linkDofFStop");
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

    if (ImGui::TreeNode("Path Tracer"))
    {
        const char* rectlightSamplingMethodItems[] = { "Uniform", "Advanced" };
        static int currentRectlightSamplingMethodItemId = 0;
        if (ImGui::BeginCombo("Rect Light Sampling", rectlightSamplingMethodItems[currentRectlightSamplingMethodItemId]))
        {
            for (const auto& item : rectlightSamplingMethodItems)
            {
                bool is_selected = (item == rectlightSamplingMethodItems[currentRectlightSamplingMethodItemId]);
                if (ImGui::Selectable(item, is_selected))
                {
                    currentRectlightSamplingMethodItemId = &item - rectlightSamplingMethodItems;
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
                bool is_selected = (item == samplerTypeItems[currentSamplerTypeId]);
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

        // One choice, not two checkboxes.
        //
        // The two effects are alternatives -- MetalFX has a spatial scaler and a
        // temporal denoised scaler, and a frame goes through one or the other --
        // but as separate toggles they offered four states, two of which meant the
        // same thing and none of which said so. The render scale stays a separate
        // control because it applies to both: at 1.00 the denoiser only denoises.
        //
        // Always available now: the wavefront tracer writes the guides the
        // denoiser reads, and it is the only tracer.
        const bool denoiseAvailable = true;
        // Keep the setting and the list in step. Selecting the megakernel while
        // denoising left the mode index pointing past the end of a now shorter
        // list -- the combo showed whatever happened to be there, the next click
        // picked something the user did not ask for, and the setting stayed on
        // while the renderer ignored it. Turning it off here means the control and
        // the renderer always agree about what is running.
        if (!denoiseAvailable && m_settingsManager->getAs<bool>("render/pt/denoise"))
        {
            m_settingsManager->setAs<bool>("render/pt/denoise", false);
            m_render->resetTemporalHistory();
        }
        const bool denoiseSetting = m_settingsManager->getAs<bool>("render/pt/denoise");
        const bool upscaleSetting = m_settingsManager->getAs<bool>("render/pt/enableUpscale");
        int fxMode = denoiseSetting ? 2 : (upscaleSetting ? 1 : 0);
        const char* fxItems[] = { "Off", "Spatial upscale", "Temporal denoise" };
        const int fxItemCount = denoiseAvailable ? 3 : 2;
        if (ImGui::Combo("MetalFX", &fxMode, fxItems, fxItemCount))
        {
            const bool wantDenoise = (fxMode == 2);
            m_settingsManager->setAs<bool>("render/pt/denoise", wantDenoise);
            // The denoiser is a scaler too: it needs the reduced-resolution render
            // whenever the scale asks for one, and nothing else does.
            const float factor = m_settingsManager->getAs<float>("render/pt/upscaleFactor");
            m_settingsManager->setAs<bool>("render/pt/enableUpscale", fxMode != 0 && factor < 1.0f);
            m_render->resetTemporalHistory();
        }
        if (!denoiseAvailable)
        {
            ImGui::SameLine();
            ImGui::BeginDisabled();
            ImGui::TextUnformatted("(denoise needs the wavefront tracer)");
            ImGui::EndDisabled();
        }

        if (fxMode == 2)
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

        if (fxMode != 0)
        {
            auto factor = m_settingsManager->getAs<float>("render/pt/upscaleFactor");
            if (ImGui::SliderFloat("Render scale", &factor, 0.25f, 1.0f, "%.2f"))
            {
                m_settingsManager->setAs<float>("render/pt/upscaleFactor", factor);
                m_settingsManager->setAs<bool>("render/pt/enableUpscale", factor < 1.0f);
                m_render->resetTemporalHistory();
            }
            ImGui::SameLine();
            ImGui::BeginDisabled();
            ImGui::TextUnformatted(
                factor < 1.0f ? "(rendering below display resolution)" : "(1:1)");
            ImGui::EndDisabled();
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

    if (ImGui::Button("Save Screenshot"))
    {
        // Generate default filename with timestamp
        const std::time_t now = std::time(nullptr);
        const std::tm* tm = std::localtime(&now);
        // localtime() returns null for a clock it cannot convert, and strftime()
        // returns 0 when the result would not fit; either way the dialog still
        // needs a name to open with.
        std::string defaultName = "screenshot.exr";
        char stamp[64];
        if (tm != nullptr && std::strftime(stamp, sizeof(stamp), "screenshot_%Y%m%d_%H%M%S.exr", tm) != 0)
        {
            defaultName = stamp;
        }

        IGFD::FileDialogConfig config;
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
        const char* modeItems[] = { "Photographic", "Multiplier" };
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
        const char* tonemapItems[] = { "None", "Reinhard", "ACES", "Filmic" };
        int currentTonemapItemId = (int)std::min(m_settingsManager->getAs<uint32_t>("render/pt/tonemapperType"), 3u);
        if (ImGui::BeginCombo("Operator", tonemapItems[currentTonemapItemId]))
        {
            for (int n = 0; n < IM_ARRAYSIZE(tonemapItems); n++)
            {
                bool is_selected = (currentTonemapItemId == n);
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
        ImGui::TextDisabled("Display max EDR %.2f (from screen, Metal path does not scale by it)", maxEdr);

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
    static const char* kStageNames[(size_t)LoadProgress::Stage::Count] = {
        "Starting", "Reading file",   "Parsing scene", "Uploading geometry",
        "Loading textures", "Building acceleration structures", "Environment", "Finishing",
    };

    const uint32_t stage =
        std::min(m_loadProgress.stage.load(std::memory_order_acquire), (uint32_t)LoadProgress::Stage::Done);
    const uint32_t done = m_loadProgress.done.load(std::memory_order_relaxed);
    const uint32_t total = m_loadProgress.total.load(std::memory_order_relaxed);

    float totalWeight = 0.0f;
    for (float w : kStageWeights)
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
