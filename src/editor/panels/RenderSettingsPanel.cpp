#include "../EditorApp.h"

#include "imgui.h"
#include "ImGuiFileDialog.h"

#include <ctime>

namespace oka
{

void EditorApp::drawRenderSettingsPanel()
{
    ImGui::Begin("Render Settings:");

    const char* debugViewOptions[] = { "None", "Normals", "Motion Blur", "Diffuse AOV", "Specular AOV" };
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
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Resume following GLTF camera animation");
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

        const char* samplerTypeItems[] = { "Halton", "PCG" };
        static int currentSamplerTypeId = 0;
        if (ImGui::BeginCombo("Sampler", samplerTypeItems[currentSamplerTypeId]))
        {
            for (const auto& item : samplerTypeItems)
            {
                bool is_selected = (item == samplerTypeItems[currentSamplerTypeId]);
                if (ImGui::Selectable(item, is_selected))
                {
                    currentSamplerTypeId = &item - samplerTypeItems;
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            m_settingsManager->setAs<uint32_t>("render/pt/samplerType", currentSamplerTypeId);
            ImGui::EndCombo();
        }

        auto maxDepth = m_settingsManager->getAs<uint32_t>("render/pt/depth");
        if (ImGui::SliderInt("Max Depth", (int*)&maxDepth, 1, 16))
        {
            m_settingsManager->setAs<uint32_t>("render/pt/depth", maxDepth);
        }

        auto sppTotal = m_settingsManager->getAs<uint32_t>("render/pt/sppTotal");
        if (ImGui::SliderInt("SPP Total", (int*)&sppTotal, 1, 10000))
        {
            m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", sppTotal);
        }

        auto sppSubframe = m_settingsManager->getAs<uint32_t>("render/pt/spp");
        if (ImGui::SliderInt("SPP Subframe", (int*)&sppSubframe, 1, 32))
        {
            m_settingsManager->setAs<uint32_t>("render/pt/spp", sppSubframe);
        }

        bool accumulationEnabled = m_settingsManager->getAs<bool>("render/pt/enableAcc");
        if (ImGui::Checkbox("Enable Path Tracer Acc", &accumulationEnabled))
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
