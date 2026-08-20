#include "../EditorApp.h"

#include "imgui.h"

#include <algorithm>
#include <cstring>

namespace oka
{
namespace
{

// ImGui::InputText edits a fixed char buffer in place, so the path has to be
// copied into one. Truncates instead of overrunning: a path longer than the
// field is the field's problem.
template <size_t N>
void toEditBuffer(const std::string& src, char (&dst)[N])
{
    const size_t n = std::min(src.size(), N - 1);
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

} // namespace

void EditorApp::drawMaterialPanel()
{
    if (!ImGui::Begin("Materials", &m_showMaterials))
    {
        ImGui::End();
        return;
    }

    if (m_selectedInstanceId != kInvalidIndex && m_selectedInstanceId < m_scene->getInstances().size())
    {
        const auto& inst = m_scene->getInstances()[m_selectedInstanceId];
        if (inst.type == Instance::Type::eMesh)
            m_selectedMaterialId = inst.mMaterialId;
    }

    auto& materials = m_scene->getMaterials();
    if (materials.empty())
    {
        ImGui::TextUnformatted("No materials in scene.");
        ImGui::End();
        return;
    }

    if (m_selectedMaterialId >= materials.size())
        m_selectedMaterialId = 0;

    if (ImGui::BeginCombo("Material", materials[m_selectedMaterialId].name.c_str()))
    {
        for (uint32_t i = 0; i < materials.size(); ++i)
        {
            const bool selected = (i == m_selectedMaterialId);
            const char* name = materials[i].name.empty() ? "(unnamed)" : materials[i].name.c_str();
            if (ImGui::Selectable(name, selected))
                m_selectedMaterialId = i;
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    Scene::MaterialDescription desc = materials[m_selectedMaterialId];
    bool changed = false;
    changed |= ImGui::ColorEdit3("Base Color", &desc.params.base_color.x);
    changed |= ImGui::DragFloat("Metallic", &desc.params.metallic, 0.01f, 0.0f, 1.0f);
    changed |= ImGui::DragFloat("Roughness", &desc.params.roughness, 0.01f, 0.0f, 1.0f);
    changed |= ImGui::ColorEdit3("Emission", &desc.params.emission.x);
    changed |= ImGui::DragFloat("Emission Strength", &desc.params.emission_strength, 0.1f, 0.0f, 1000.0f);
    changed |= ImGui::DragFloat("IOR", &desc.params.ior, 0.01f, 1.0f, 3.0f);

    char pathBuf[512];
    toEditBuffer(desc.baseColorTexPath, pathBuf);
    if (ImGui::InputText("Base Color Tex", pathBuf, sizeof(pathBuf)))
    {
        desc.baseColorTexPath = pathBuf;
        changed = true;
    }

    {
        ImGui::SeparatorText("Environment");
        Scene::EnvLightDesc env = m_scene->getEnvLight().value_or(Scene::EnvLightDesc{});
        bool envChanged = false;
        char envPath[512];
        toEditBuffer(env.texturePath, envPath);
        if (ImGui::InputText("HDR path", envPath, sizeof(envPath)))
        {
            env.texturePath = envPath;
            envChanged = true;
        }
        envChanged |= ImGui::DragFloat("Env Intensity", &env.intensity, 0.05f, 0.0f, 100.0f);
        envChanged |= ImGui::ColorEdit3("Env Color", &env.color.x);
        envChanged |= ImGui::DragFloat("Env Rotation Y", &env.rotationY, 0.5f);
        if (envChanged)
        {
            m_scene->setEnvLight(env);
            markDocumentDirty();
        }
    }

    if (changed)
    {
        pushUndoMaterial(m_selectedMaterialId);
        m_scene->setMaterial(m_selectedMaterialId, desc);
        markDocumentDirty();
    }

    ImGui::End();
}

} // namespace oka
