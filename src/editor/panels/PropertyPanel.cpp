#include "../EditorApp.h"

#include "imgui.h"
#include "ImGuizmo.h"

#include <glm/gtc/type_ptr.hpp>

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
        ImGui::SeparatorText("Light");
        ImGui::Text("Light id: %u  type: %d", m_selectedLightId, desc.type);

        bool changed = false;
        changed |= ImGui::DragFloat3("Position", &desc.position.x, 0.05f);
        changed |= ImGui::DragFloat3("Orientation", &desc.orientation.x, 0.5f);
        if (desc.type == 0)
        {
            float wh[2] = { desc.width, desc.height };
            if (ImGui::DragFloat2("Width/Height", wh, 0.05f, 0.005f))
            {
                desc.width = wh[0];
                desc.height = wh[1];
                changed = true;
            }
        }
        else if (desc.type == 1 || desc.type == 2)
        {
            changed |= ImGui::DragFloat("Radius", &desc.radius, 0.05f, 0.001f);
        }
        changed |= ImGui::ColorEdit3("Color", &desc.color.x);
        changed |= ImGui::DragFloat("Intensity", &desc.intensity, 1.0f, 0.0f);

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
}

} // namespace oka
