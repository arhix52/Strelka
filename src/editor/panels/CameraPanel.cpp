#include "../EditorApp.h"
#include "../editor_camera_exposure.h"

#include "imgui.h"

namespace oka
{

/// Camera selection, lens/DOF, photographic exposure and navigation speed.
///
/// Was part of Render Settings; moved out because a camera is a scene object a
/// user picks and tunes, not a renderer preference -- and it made that panel's
/// scroll twice as long as it needed to be.
void EditorApp::drawCameraPanel()
{
    ImGui::Begin("Camera:");

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
    if (ImGui::TreeNodeEx("Lens", ImGuiTreeNodeFlags_DefaultOpen))
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

        changed |= ImGui::DragFloat(
            "Focal length", &cam.focalLengthMm, 0.5f, 1.0f, 500.0f, "%.1f mm", ImGuiSliderFlags_Logarithmic);
        changed |= ImGui::DragFloat("Sensor width", &cam.sensorWidth, 0.1f, 1.0f, 100.0f, "%.1f mm");
        changed |= ImGui::DragFloat("Sensor height", &cam.sensorHeight, 0.1f, 1.0f, 100.0f, "%.1f mm");
        ImGui::TextDisabled("Vertical FOV %.1f deg (from lens + sensor)",
                            editor_camera_exposure::verticalFovDegrees(cam.focalLengthMm, cam.sensorHeight));

        if (ImGui::Checkbox("Enable DOF", &cam.useDof))
            changed = true;

        if (cam.useDof)
        {
            if (ImGui::SliderFloat(
                    "Focus distance", &cam.focalDistance, 0.1f, 1000.0f, "%.2f m", ImGuiSliderFlags_Logarithmic))
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
            ImGui::TextDisabled(
                "Lens radius %.4f m", editor_camera_exposure::lensRadiusMetres(cam.focalLengthMm, cam.fStopDof));

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

    // Exposure, the camera side of the tone curve. The renderer computes
    //     film speed  > 0 : cm2_factor * iso / (shutter * fstop^2) / 100
    //     film speed == 0 : cm2_factor
    // so a zero film speed is the arbitrary-units mode, which is what a scene lit
    // in normalised rather than photometric units wants -- and what the light
    // sidecar writes. Both forms are editable here because the sidecar can carry
    // either, and a scene that opens too dark is otherwise unexplainable from
    // inside the editor.
    if (ImGui::TreeNode("Exposure"))
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
            ImGui::SetTooltip(
                "When on, one f-stop drives both DOF blur and photographic exposure.\n"
                "Untick to blur the lens without changing film brightness.");
        }

        if (mode == 0)
        {
            changed |= ImGui::DragFloat("Film ISO", &iso, 1.0f, 1.0f, 25600.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
            if (ImGui::DragFloat(linkDof ? "Aperture (linked)" : "Exposure f-stop", &fStop, 0.05f, 0.7f, 32.0f, "f/%.1f"))
            {
                changed = true;
                if (linkDof)
                {
                    m_scene->getCamera(m_selectedCamera).fStopDof = fStop;
                }
            }
            changed |=
                ImGui::DragFloat("Shutter", &shutter, 1.0f, 1.0f, 8000.0f, "1/%.0f s", ImGuiSliderFlags_Logarithmic);
            changed |=
                ImGui::DragFloat("cd/m^2 factor", &cm2, 0.01f, 0.0001f, 100000.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip(
                    "Photometric scale (candela per square metre factor).\n"
                    "Not a generic exposure multiplier — use Mode=Multiplier for that.");
            }
            const float linear = editor_camera_exposure::photographicLinearScale(iso, fStop, shutter, cm2);
            const float ev = editor_camera_exposure::ev100(iso, fStop, shutter);
            ImGui::TextDisabled("EV100 %.2f  |  Linear radiance x%.4f", ev, linear);
        }
        else
        {
            changed |= ImGui::DragFloat(
                "Linear multiplier", &cm2, 0.01f, 0.0001f, 100000.0f, "x%.4f", ImGuiSliderFlags_Logarithmic);
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

    if (ImGui::TreeNode("Navigation"))
    {
        auto cameraSpeed = m_settingsManager->getAs<float>("render/cameraSpeed");
        ImGui::InputFloat("Camera Speed", (float*)&cameraSpeed, 0.5);
        m_settingsManager->setAs<float>("render/cameraSpeed", cameraSpeed);

        drawGamepadSettings();

        ImGui::TreePop();
    }

    ImGui::End();
}

} // namespace oka
