#include "../EditorApp.h"

#include "imgui.h"
#include "ImGuizmo.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace oka
{

void EditorApp::showGizmo(Camera& cam, float* matrix, ImGuizmo::OPERATION operation)
{
    static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::LOCAL);
    glm::float4x4 cameraView = cam.matrices.view;
    glm::float4x4 cameraProjection = cam.matrices.perspective;
    ImGuizmo::Manipulate(
        glm::value_ptr(cameraView), glm::value_ptr(cameraProjection), operation, mCurrentGizmoMode, matrix);
}

void EditorApp::drawPropertyPanel(uint32_t lightId)
{
    static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::TRANSLATE);

    Camera& cam = m_scene->getCamera(m_selectedCamera);
    glm::float3 camPos = cam.getPosition();

    std::vector<Scene::UniformLightDesc>& lightDescs = m_scene->getLightsDesc();
    Scene::UniformLightDesc& currLightDesc = lightDescs[lightId];

    if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
        mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
        mCurrentGizmoOperation = ImGuizmo::ROTATE;

    ImGui::Text("Rectangle light");
    ImGui::Spacing();
    ImGui::AlignTextToFramePadding();
    ImGui::DragFloat3("Position", &currLightDesc.position.x);
    ImGui::Spacing();
    ImGui::DragFloat3("Orientation", &currLightDesc.orientation.x);
    ImGui::Spacing();
    float width_height[2] = { currLightDesc.width, currLightDesc.height };
    ImGui::DragFloat2("Width/Height", width_height, 0.1f, 0.005f);
    ImGui::Spacing();
    ImGui::ColorEdit3("Color", &currLightDesc.color.x);
    ImGui::DragFloat("Intensity", &currLightDesc.intensity, 1.0f, 1.0f);
    currLightDesc.intensity = glm::clamp(currLightDesc.intensity, 1.0f, std::numeric_limits<float>::max());
    currLightDesc.width = glm::clamp(width_height[0], 0.005f, std::numeric_limits<float>::max());
    currLightDesc.height = glm::clamp(width_height[1], 0.005f, std::numeric_limits<float>::max());

    ImGuizmo::SetID(lightId);

    const glm::float4x4 translationMatrix = glm::translate(glm::float4x4(1.0f), currLightDesc.position);
    glm::quat rotation = glm::quat(glm::radians(currLightDesc.orientation));
    const glm::float4x4 rotationMatrix{ rotation };
    glm::float3 scale = { currLightDesc.width, currLightDesc.height, 1.0f };
    const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), scale);

    glm::float4x4 lightXform = translationMatrix * rotationMatrix * scaleMatrix;

    showGizmo(cam, &lightXform[0][0], mCurrentGizmoOperation);

    float matrixTranslation[3], matrixRotation[3], matrixScale[3];
    ImGuizmo::DecomposeMatrixToComponents(&lightXform[0][0], matrixTranslation, matrixRotation, matrixScale);

    currLightDesc.position = glm::float3(matrixTranslation[0], matrixTranslation[1], matrixTranslation[2]);
    currLightDesc.orientation = glm::float3(matrixRotation[0], matrixRotation[1], matrixRotation[2]);

    Scene::UniformLightDesc desc{};
    desc.position = currLightDesc.position;
    desc.orientation = currLightDesc.orientation;
    desc.width = currLightDesc.width;
    desc.height = currLightDesc.height;
    desc.color = currLightDesc.color;
    desc.intensity = currLightDesc.intensity;
    m_scene->updateLight(lightId, desc);
    m_scene->updateInstanceTransform(m_scene->mLightIdToInstanceId[lightId], lightXform);
}

} // namespace oka
