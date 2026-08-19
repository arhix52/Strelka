#include "../EditorApp.h"

#include "imgui.h"
#include "ImGuizmo.h"
#include "ImGuiFileDialog.h"

#include <strelka/scene/light_desc.h>
#include <strelka/sceneloader/iesloader.h>

#include <glm/gtc/type_ptr.hpp>

#include <cmath>

namespace oka
{

void EditorApp::showGizmo(Camera& cam, float* matrix, ImGuizmo::OPERATION operation)
{
    glm::float4x4 cameraView = cam.matrices.view;
    glm::float4x4 cameraProjection = cam.matrices.perspective;
    ImGuizmo::Manipulate(
        glm::value_ptr(cameraView), glm::value_ptr(cameraProjection), operation, m_gizmoMode, matrix);
}

void EditorApp::drawPropertyPanel()
{
    if (!ImGui::Begin("Properties", &m_showProperties))
    {
        ImGui::End();
        return;
    }

    if (ImGui::RadioButton("Translate", m_gizmoOperation == ImGuizmo::TRANSLATE))
        m_gizmoOperation = ImGuizmo::TRANSLATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Rotate", m_gizmoOperation == ImGuizmo::ROTATE))
        m_gizmoOperation = ImGuizmo::ROTATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Scale", m_gizmoOperation == ImGuizmo::SCALE))
        m_gizmoOperation = ImGuizmo::SCALE;

    if (ImGui::RadioButton("Local", m_gizmoMode == ImGuizmo::LOCAL))
        m_gizmoMode = ImGuizmo::LOCAL;
    ImGui::SameLine();
    if (ImGui::RadioButton("World", m_gizmoMode == ImGuizmo::WORLD))
        m_gizmoMode = ImGuizmo::WORLD;

    if (m_selectedLightId != (uint32_t)-1 && m_selectedLightId < m_scene->getLightsDesc().size())
    {
        Scene::UniformLightDesc desc = m_scene->getLightsDesc()[m_selectedLightId];
        ImGui::SeparatorText(desc.name.empty() ? "Light" : desc.name.c_str());
        ImGui::Text("id %u · %s", m_selectedLightId, lightTypeName(desc.type));

        bool changed = false;
        changed |= ImGui::Checkbox("Enabled", &desc.enabled);

        // Type picker. Changing type keeps shared fields and fills sensible
        // defaults for the ones the new shape needs.
        const char* const typeItems[] = { "Area (Rect)", "Disc", "Sphere", "Sun / Distant", "Point", "Spot", "Projector" };
        const int typeValues[] = { LIGHT_TYPE_RECT,  LIGHT_TYPE_DISC, LIGHT_TYPE_SPHERE,   LIGHT_TYPE_DISTANT,
                                   LIGHT_TYPE_POINT, LIGHT_TYPE_SPOT, LIGHT_TYPE_PROJECTOR };
        const int typeCount = IM_ARRAYSIZE(typeItems);
        int typeIdx = 0;
        for (int i = 0; i < typeCount; ++i)
        {
            if (typeValues[i] == desc.type)
                typeIdx = i;
        }
        if (ImGui::Combo("Type", &typeIdx, typeItems, typeCount))
        {
            const int prev = desc.type;
            desc.type = typeValues[typeIdx];
            if (lightTypeIsPunctual(desc.type))
            {
                if (desc.intensityUnit == LIGHT_UNIT_RADIANCE)
                    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
            }
            else if (desc.type == LIGHT_TYPE_DISTANT)
            {
                if (desc.intensityUnit == LIGHT_UNIT_INTENSITY)
                    desc.intensityUnit = LIGHT_UNIT_IRRADIANCE;
                if (desc.halfAngle <= 0.0f)
                    desc.halfAngle = 0.53f * 0.5f * (float(M_PI) / 180.0f);
            }
            else if (lightTypeIsPunctual(prev))
            {
                if (desc.intensityUnit == LIGHT_UNIT_INTENSITY)
                    desc.intensityUnit = LIGHT_UNIT_POWER;
            }
            changed = true;
        }

        changed |= ImGui::ColorEdit3("Color", &desc.color.x);

        // Blender-style units: the label follows the unit so a Power of 10 reads
        // as Watts, not as an abstract multiplier.
        const char* const unitItems[] = { "Radiance", "Power (W)", "Intensity (cd)", "Irradiance (W/m²)" };
        int unitIdx = desc.intensityUnit;
        if (unitIdx < 0 || unitIdx > 3)
            unitIdx = 0;
        if (ImGui::Combo("Unit", &unitIdx, unitItems, 4))
        {
            desc.intensityUnit = unitIdx;
            changed = true;
        }
        const char* intensityLabel = unitItems[unitIdx];
        changed |= ImGui::DragFloat(intensityLabel, &desc.intensity, 1.0f, 0.0f, 1.0e9f, "%.3f");

        if (desc.type != LIGHT_TYPE_DISTANT)
        {
            changed |= ImGui::DragFloat3("Position", &desc.position.x, 0.05f);
        }
        changed |= ImGui::DragFloat3("Rotation", &desc.orientation.x, 0.5f);

        if (desc.type == LIGHT_TYPE_RECT)
        {
            float wh[2] = { desc.width, desc.height };
            if (ImGui::DragFloat2("Size (W×H)", wh, 0.05f, 0.005f))
            {
                desc.width = wh[0];
                desc.height = wh[1];
                changed = true;
            }
        }
        else if (desc.type == LIGHT_TYPE_DISC || desc.type == LIGHT_TYPE_SPHERE)
        {
            changed |= ImGui::DragFloat("Radius", &desc.radius, 0.05f, 0.001f);
        }
        else if (desc.type == LIGHT_TYPE_POINT)
        {
            changed |= ImGui::DragFloat("Soft Size", &desc.radius, 0.01f, 0.0f);
            changed |= ImGui::DragFloat("Range", &desc.range, 0.1f, 0.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = infinite (no KHR range window)");
        }
        else if (desc.type == LIGHT_TYPE_SPOT)
        {
            float outerDeg = desc.outerConeAngle * (180.0f / float(M_PI));
            if (ImGui::DragFloat("Spot Size", &outerDeg, 0.5f, 0.1f, 90.0f))
            {
                desc.outerConeAngle = outerDeg * (float(M_PI) / 180.0f);
                desc.innerConeAngle = std::min(desc.innerConeAngle, desc.outerConeAngle);
                changed = true;
            }
            // Blender's "Blend" is how soft the penumbra is: 0 = hard, 1 = inner at 0.
            float blend = (desc.outerConeAngle > 1e-4f)
                              ? (1.0f - desc.innerConeAngle / desc.outerConeAngle)
                              : 0.0f;
            if (ImGui::DragFloat("Blend", &blend, 0.01f, 0.0f, 1.0f))
            {
                desc.innerConeAngle = desc.outerConeAngle * (1.0f - blend);
                changed = true;
            }
            changed |= ImGui::DragFloat("Soft Size", &desc.radius, 0.01f, 0.0f);
            changed |= ImGui::DragFloat("Range", &desc.range, 0.1f, 0.0f);
        }
        else if (desc.type == LIGHT_TYPE_PROJECTOR)
        {
            // The full horizontal field of view, the way a projector is
            // specified. The desc stores half of it, in the outer-cone field the
            // GPU light already has -- see Scene::UniformLightDesc.
            float fovDeg = desc.outerConeAngle * 2.0f * (180.0f / float(M_PI));
            if (ImGui::DragFloat("Field of View", &fovDeg, 0.5f, 1.0f, 178.0f, "%.1f deg"))
            {
                desc.outerConeAngle = fovDeg * 0.5f * (float(M_PI) / 180.0f);
                changed = true;
            }
            // Throw ratio is how a projector is actually sold -- "1.5:1" means
            // the image is as wide as the throw distance divided by 1.5 -- so it
            // is worth showing even though nothing stores it.
            const float tanHalf = std::tan(std::max(desc.outerConeAngle, 1e-4f));
            ImGui::SameLine();
            ImGui::TextDisabled("(throw %.2f:1)", 0.5f / std::max(tanHalf, 1e-4f));

            changed |= ImGui::DragFloat("Aspect (W:H)", &desc.projectorAspect, 0.01f, 0.1f, 10.0f, "%.3f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Shape of the thrown frame. 1.778 = 16:9, 1.333 = 4:3");
            changed |= ImGui::DragFloat("Edge Softness", &desc.projectorEdgeSoftness, 0.01f, 0.0f, 1.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = the crisp border a focused projector has");
            changed |= ImGui::DragFloat("Soft Size", &desc.radius, 0.01f, 0.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Lamp size. Above zero the image blurs, like a lens out of focus");
            changed |= ImGui::DragFloat("Range", &desc.range, 0.1f, 0.0f);

            ImGui::SeparatorText("Projected Image");
            ImGui::TextWrapped(
                "%s", desc.projectorImagePath.empty() ? "(none - plain white frame)" : desc.projectorImagePath.c_str());
            if (ImGui::Button("Choose Image..."))
            {
                IGFD::FileDialogConfig config;
                config.path = m_scene->getSceneDir().empty() ? "." : m_scene->getSceneDir();
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ProjectorImageDlgKey", "Open Image",
                    "Images{.png,.jpg,.jpeg,.tga,.bmp,.hdr,.exr,.PNG,.JPG,.JPEG,.TGA,.BMP,.HDR,.EXR}", config);
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear Image") && (!desc.projectorImagePath.empty() || desc.projectorImage >= 0))
            {
                desc.projectorImagePath.clear();
                desc.projectorImage = -1;
                changed = true;
            }
        }
        else if (desc.type == LIGHT_TYPE_DISTANT)
        {
            float angleDeg = desc.halfAngle * 2.0f * (180.0f / float(M_PI));
            if (ImGui::DragFloat("Angular Diameter", &angleDeg, 0.05f, 0.01f, 20.0f))
            {
                desc.halfAngle = angleDeg * 0.5f * (float(M_PI) / 180.0f);
                changed = true;
            }
        }

        // IES profile — only meaningful for point/spot. A projector's angular
        // profile is its image, and Scene::updateLight drops the IES slot for
        // one, so offering the picker there would do nothing.
        if (desc.type == LIGHT_TYPE_POINT || desc.type == LIGHT_TYPE_SPOT)
        {
            ImGui::SeparatorText("IES Profile");
            ImGui::TextWrapped("%s", desc.iesPath.empty() ? "(none — isotropic)" : desc.iesPath.c_str());
            if (ImGui::Button("Load IES…"))
            {
                IGFD::FileDialogConfig config;
                config.path = m_scene->getSceneDir().empty() ? "." : m_scene->getSceneDir();
                ImGuiFileDialog::Instance()->OpenDialog("LoadIesDlgKey", "Open IES", ".ies,.IES", config);
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear IES") && (!desc.iesPath.empty() || desc.iesProfile >= 0))
            {
                desc.iesPath.clear();
                desc.iesProfile = -1;
                changed = true;
            }
        }

        // Radiance cache. Only meaningful while the cache is on, and harmless
        // otherwise, so it is shown either way rather than appearing and
        // vanishing with a render setting on another panel.
        ImGui::SeparatorText("Radiance cache");
        if (ImGui::Checkbox("Responsive", &desc.responsive))
        {
            changed = true;
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(
                "Cache this light's contribution on a short clock, separately from the\n"
                "rest of the scene's. For a light that changes faster than the cache's\n"
                "temporal window can follow -- swung, animated, switched on mid-shot --\n"
                "which otherwise lags visibly behind the render.\n\n"
                "Costs a second cache entry per voxel this light reaches and a second\n"
                "deposit per path, so it is worth setting only on the lights that need\n"
                "it. Nothing in the scene has to be responsive, and a scene with none\n"
                "pays nothing at all.");
            ImGui::EndTooltip();
        }

        if (changed)
        {
            pushUndoLight(m_selectedLightId);
            desc.intensity = glm::max(desc.intensity, 0.0f);
            m_scene->setLight(m_selectedLightId, desc);
            markDocumentDirty();
        }
    }
    else if (m_selectedNodeId != (uint32_t)-1 && m_selectedNodeId < m_scene->getNodes().size())
    {
        const Scene::Node& node = m_scene->getNodes()[m_selectedNodeId];
        ImGui::SeparatorText("Node");
        ImGui::Text("%s", node.name.empty() ? "(unnamed)" : node.name.c_str());

        glm::float3 translation = node.translation;
        glm::quat rotation = node.rotation;
        glm::float3 scale = node.scale;
        glm::float3 eulerDegrees = glm::degrees(glm::eulerAngles(rotation));

        bool changed = false;
        changed |= ImGui::DragFloat3("Translation", &translation.x, 0.05f);
        if (ImGui::DragFloat3("Rotation", &eulerDegrees.x, 0.5f))
        {
            rotation = glm::quat(glm::radians(eulerDegrees));
            changed = true;
        }
        changed |= ImGui::DragFloat3("Scale", &scale.x, 0.05f, 0.001f);

        if (changed)
        {
            pushUndoNode(m_selectedNodeId);
            m_scene->setNodeLocalTransform(m_selectedNodeId, translation, rotation, scale);
            markDocumentDirty();
        }
    }
    else
    {
        ImGui::TextUnformatted("Nothing selected. Click in the viewport or Outliner.");
    }

    ImGui::End();

    if (ImGuiFileDialog::Instance()->Display("ProjectorImageDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk() && m_selectedLightId != (uint32_t)-1 &&
            m_selectedLightId < m_scene->getLightsDesc().size())
        {
            Scene::UniformLightDesc desc = m_scene->getLightsDesc()[m_selectedLightId];
            pushUndoLight(m_selectedLightId);
            desc.projectorImagePath = ImGuiFileDialog::Instance()->GetFilePathName();
            // Registered with the scene, not decoded here: the renderer walks
            // Scene::getProjectorImages() whenever the light set changes and
            // uploads what it finds, so picking a file is the same kind of edit
            // as moving the light.
            desc.projectorImage = m_scene->addProjectorImage(desc.projectorImagePath);
            m_scene->setLight(m_selectedLightId, desc);
            markDocumentDirty();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display("LoadIesDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk() && m_selectedLightId != (uint32_t)-1 &&
            m_selectedLightId < m_scene->getLightsDesc().size())
        {
            Scene::UniformLightDesc desc = m_scene->getLightsDesc()[m_selectedLightId];
            const std::string path = ImGuiFileDialog::Instance()->GetFilePathName();
            Scene::IesProfile profile;
            if (loadIesProfile(path, profile))
            {
                pushUndoLight(m_selectedLightId);
                desc.iesPath = path;
                desc.iesProfile = m_scene->addIesProfile(std::move(profile));
                // An IES file is already in candela; keep the intensity as a
                // multiplier on top of the profile.
                if (desc.intensityUnit == LIGHT_UNIT_RADIANCE)
                    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
                m_scene->setLight(m_selectedLightId, desc);
                markDocumentDirty();
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }
}

} // namespace oka
