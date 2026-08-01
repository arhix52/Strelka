#include "../EditorApp.h"

#include "imgui.h"
#include "ImGuiFileDialog.h"

#include <ctime>

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
    const char* debugViewOptions[] = { "None",          "Normals",         "Motion Blur",
                                       "AOV: diffuse",  "AOV: specular",   "AOV: normal",
                                       "AOV: roughness", "AOV: depth",     "AOV: motion" };
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
                            m_cameraDetached = false;
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
                    m_cameraDetached = false;
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

    // Camera DOF / Lens controls
    if (ImGui::TreeNode("Camera / DOF"))
    {
        oka::Camera& cam = m_scene->getCamera(m_selectedCamera);
        bool changed = false;

        if (ImGui::Checkbox("Enable DOF", &cam.useDof))
            changed = true;

        if (cam.useDof)
        {
            if (ImGui::SliderFloat("Focus Distance", &cam.focalDistance, 0.1f, 1000.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
                changed = true;
            if (ImGui::SliderFloat("F-Stop", &cam.fStopDof, 1.0f, 22.0f, "%.1f"))
                changed = true;
            if (ImGui::SliderInt("Aperture Blades", &cam.apertureBlades, 0, 8))
                changed = true;
            if (ImGui::SliderFloat("Blade Rotation", &cam.bladeRotation, 0.0f, 6.2832f, "%.2f"))
                changed = true;
            if (ImGui::SliderFloat("Anamorphic Ratio", &cam.anamorphicRatio, 0.25f, 4.0f, "%.2f"))
                changed = true;
        }

        if (ImGui::InputFloat("Shift X", &cam.shiftX, 0.01f))
            changed = true;
        if (ImGui::InputFloat("Shift Y", &cam.shiftY, 0.01f))
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

        const char* tracerItems[] = { "Megakernel", "Wavefront" };
        auto tracerMode = m_settingsManager->getAs<uint32_t>("render/pt/tracerMode");
        if (ImGui::Combo("Tracer", (int*)&tracerMode, tracerItems, IM_ARRAYSIZE(tracerItems)))
        {
            m_settingsManager->setAs<uint32_t>("render/pt/tracerMode", tracerMode);
        }

        // One choice, not two checkboxes.
        //
        // The two effects are alternatives -- MetalFX has a spatial scaler and a
        // temporal denoised scaler, and a frame goes through one or the other --
        // but as separate toggles they offered four states, two of which meant the
        // same thing and none of which said so. The render scale stays a separate
        // control because it applies to both: at 1.00 the denoiser only denoises.
        //
        // Only the wavefront tracer writes the guides the denoiser reads, so on
        // the megakernel that entry is unavailable rather than silently ignored.
        const bool denoiseAvailable = tracerMode == 1;
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
        std::time_t now = std::time(nullptr);
        std::tm* tm = std::localtime(&now);
        char defaultName[64];
        std::strftime(defaultName, sizeof(defaultName), "screenshot_%Y%m%d_%H%M%S.exr", tm);

        IGFD::FileDialogConfig config;
        config.path = ".";
        config.fileName = defaultName;
        ImGuiFileDialog::Instance()->OpenDialog(
            "SaveScreenshotDlgKey", "Save Screenshot", ".exr,.png", config);
    }

    auto cameraSpeed = m_settingsManager->getAs<float>("render/cameraSpeed");
    ImGui::InputFloat("Camera Speed", (float*)&cameraSpeed, 0.5);
    m_settingsManager->setAs<float>("render/cameraSpeed", cameraSpeed);

    const char* tonemapItems[] = { "None", "Reinhard", "ACES", "Filmic" };
    static int currentTonemapItemId = 1;
    if (ImGui::BeginCombo("Tonemap", tonemapItems[currentTonemapItemId]))
    {
        for (int n = 0; n < IM_ARRAYSIZE(tonemapItems); n++)
        {
            bool is_selected = (currentTonemapItemId == n);
            if (ImGui::Selectable(tonemapItems[n], is_selected))
            {
                currentTonemapItemId = n;
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", currentTonemapItemId);

    auto gamma = m_settingsManager->getAs<float>("render/post/gamma");
    ImGui::InputFloat("Gamma", (float*)&gamma, 0.5);
    m_settingsManager->setAs<float>("render/post/gamma", gamma);

    auto materialRayTmin = m_settingsManager->getAs<float>("render/pt/dev/materialRayTmin");
    ImGui::InputFloat("Material ray T min", (float*)&materialRayTmin, 0.1);
    m_settingsManager->setAs<float>("render/pt/dev/materialRayTmin", materialRayTmin);
    auto shadowRayTmin = m_settingsManager->getAs<float>("render/pt/dev/shadowRayTmin");
    ImGui::InputFloat("Shadow ray T min", (float*)&shadowRayTmin, 0.1);
    m_settingsManager->setAs<float>("render/pt/dev/shadowRayTmin", shadowRayTmin);

    ImGui::End();
}

} // namespace oka
