#include "../EditorApp.h"

#include "imgui.h"

#include <string>

namespace oka
{

namespace
{

// A node picked in the viewport can sit deep inside a collapsed subtree, so the
// path from the root has to be opened before the tree is walked.
bool nodeIsAncestorOf(const Scene& scene, int candidate, uint32_t nodeId)
{
    int walk = (int)nodeId;
    while (walk >= 0 && walk < (int)scene.getNodes().size())
    {
        if (walk == candidate)
        {
            return true;
        }
        walk = scene.getNodes()[walk].parent;
    }
    return false;
}

bool subtreeMatchesFilter(const Scene& scene, int nodeId, const ImGuiTextFilter& filter)
{
    if (nodeId < 0 || nodeId >= (int)scene.getNodes().size())
    {
        return false;
    }
    const Scene::Node& node = scene.getNodes()[nodeId];
    if (filter.PassFilter(node.name.c_str()))
    {
        return true;
    }
    for (int child : node.children)
    {
        if (subtreeMatchesFilter(scene, child, filter))
        {
            return true;
        }
    }
    return false;
}

} // namespace

void EditorApp::drawNodeRecursive(int nodeId, const ImGuiTextFilter& filter)
{
    const Scene& scene = *m_scene;
    if (nodeId < 0 || nodeId >= (int)scene.getNodes().size())
    {
        return;
    }
    if (filter.IsActive() && !subtreeMatchesFilter(scene, nodeId, filter))
    {
        return;
    }

    const Scene::Node& node = scene.getNodes()[nodeId];
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (node.children.empty())
    {
        flags |= ImGuiTreeNodeFlags_Leaf;
    }
    const bool isSelected = (uint32_t)nodeId == m_selectedNodeId;
    if (isSelected)
    {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    // Keep the selected node reachable: open its ancestors, scroll to it once.
    if (m_selectedNodeId != (uint32_t)-1 && !node.children.empty() &&
        nodeIsAncestorOf(scene, nodeId, m_selectedNodeId) && !isSelected)
    {
        ImGui::SetNextItemOpen(true);
    }

    const char* label = node.name.empty() ? "(unnamed)" : node.name.c_str();
    const bool open = ImGui::TreeNodeEx((void*)(intptr_t)nodeId, flags, "%s%s", label,
                                        node.instanceIds.empty() ? "" : " [mesh]");
    if (isSelected && m_outlinerScrollToSelection)
    {
        ImGui::SetScrollHereY(0.5f);
        m_outlinerScrollToSelection = false;
    }
    if (ImGui::IsItemClicked())
    {
        m_selectedNodeId = (uint32_t)nodeId;
        m_selectedLightId = (uint32_t)-1;
        m_selectedInstanceId = node.instanceIds.empty() ? (uint32_t)-1 : node.instanceIds.front();
        m_selectedMaterialId = (uint32_t)-1;
        if (m_selectedInstanceId != (uint32_t)-1 && m_selectedInstanceId < scene.getInstances().size())
        {
            const Instance& inst = scene.getInstances()[m_selectedInstanceId];
            m_selectedMaterialId = inst.mMaterialId;
            if (inst.type == Instance::Type::eLight)
            {
                m_selectedLightId = inst.mLightId;
            }
        }
    }
    if (open)
    {
        for (int child : node.children)
        {
            drawNodeRecursive(child, filter);
        }
        ImGui::TreePop();
    }
}

void EditorApp::drawOutlinerPanel()
{
    if (!ImGui::Begin("Outliner", &m_showOutliner))
    {
        ImGui::End();
        return;
    }

    static ImGuiTextFilter filter;
    filter.Draw("##filter", -1.0f);

    const auto& lights = m_scene->getLightsDesc();
    if (ImGui::TreeNodeEx("Lights", ImGuiTreeNodeFlags_DefaultOpen, "Lights (%zu)", lights.size()))
    {
        for (uint32_t i = 0; i < lights.size(); ++i)
        {
            static const char* kTypeNames[] = { "rect", "disc", "sphere", "distant" };
            const char* typeName = lights[i].type >= 0 && lights[i].type < 4 ? kTypeNames[lights[i].type] : "unknown";
            char label[64];
            snprintf(label, sizeof(label), "Light %u (%s)", i, typeName);
            if (ImGui::Selectable(label, m_selectedLightId == i))
            {
                clearSelection();
                m_selectedLightId = i;
                m_selectedInstanceId = m_scene->getLightInstanceId(i);
            }
        }
        ImGui::TreePop();
    }

    const auto& nodes = m_scene->getNodes();
    if (ImGui::TreeNodeEx("Scene Graph", ImGuiTreeNodeFlags_DefaultOpen, "Scene Graph (%zu nodes)", nodes.size()))
    {
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            if (nodes[i].parent == -1)
            {
                drawNodeRecursive((int)i, filter);
            }
        }
        ImGui::TreePop();
    }

    ImGui::End();
}

} // namespace oka
