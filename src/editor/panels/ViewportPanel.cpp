#include "../EditorApp.h"

#include <cstring>

#include "imgui.h"
#include "ImGuizmo.h"

#include <cmath>
#include <limits>

#include <strelka/scene/transform.h>

#include <glm/gtc/matrix_transform.hpp>

namespace oka
{

// Ray through a point of the rendered image, matching generateCameraRay in the
// shaders: same NDC mapping, same Y flip, same clip -> view -> world chain, so a
// CPU pick lands on whatever the user sees under the cursor.
Scene::PickHit EditorApp::pickAtScreenPos(const ImVec2& screenPos)
{
    const float width = m_viewportRectMax.x - m_viewportRectMin.x;
    const float height = m_viewportRectMax.y - m_viewportRectMin.y;
    if (width <= 0.0f || height <= 0.0f)
    {
        return {};
    }

    const float u = (screenPos.x - m_viewportRectMin.x) / width;
    const float v = (screenPos.y - m_viewportRectMin.y) / height;
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f)
    {
        return {};
    }

    glm::float3 origin, dir;
    generatePickRay(m_scene->getCamera(m_selectedCamera), glm::float2(u, v), origin, dir);

    return m_scene->pick(origin, dir);
}

// Trim a clip-space segment to the part in front of the eye. Without this a box
// the camera sits inside of, or one that merely pokes past the near plane,
// projects garbage or disappears entirely.
static bool clipSegmentToEye(glm::float4& a, glm::float4& b)
{
    constexpr float kMinW = 1e-4f;
    const bool aFront = a.w > kMinW;
    const bool bFront = b.w > kMinW;
    if (!aFront && !bFront)
    {
        return false;
    }
    if (aFront && bFront)
    {
        return true;
    }
    const float t = (kMinW - a.w) / (b.w - a.w);
    const glm::float4 crossing = a + (b - a) * t;
    if (aFront)
    {
        b = crossing;
    }
    else
    {
        a = crossing;
    }
    return true;
}

// Wireframe box around geometry, given bounds and the transform that maps them to
// world. DrawCubes() would only ever draw a unit cube at the instance origin,
// which for anything but a unit-sized mesh sits in the wrong place or inside the
// geometry.
void EditorApp::drawBoundsWireframe(const glm::float3& bbMin,
                                    const glm::float3& bbMax,
                                    const glm::mat4& worldFromLocal,
                                    Camera& cam)
{
    // Bounds are expressed in the space the transform maps to world, so the box
    // follows the geometry's own orientation instead of being an inflated
    // world-axis-aligned hull.
    const glm::mat4 clipFromLocal = cam.matrices.perspective * cam.matrices.view * worldFromLocal;
    const float width = m_viewportRectMax.x - m_viewportRectMin.x;
    const float height = m_viewportRectMax.y - m_viewportRectMin.y;

    glm::float4 clip[8];
    for (int i = 0; i < 8; ++i)
    {
        const glm::float3 corner((i & 1) ? bbMax.x : bbMin.x, (i & 2) ? bbMax.y : bbMin.y,
                                 (i & 4) ? bbMax.z : bbMin.z);
        clip[i] = clipFromLocal * glm::float4(corner, 1.0f);
    }

    const auto toScreen = [&](const glm::float4& c) {
        return ImVec2(m_viewportRectMin.x + (c.x / c.w * 0.5f + 0.5f) * width,
                      m_viewportRectMin.y + (0.5f - c.y / c.w * 0.5f) * height);
    };

    static const int edges[12][2] = { { 0, 1 }, { 1, 3 }, { 3, 2 }, { 2, 0 }, { 4, 5 }, { 5, 7 },
                                      { 7, 6 }, { 6, 4 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } };
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const ImU32 color = IM_COL32(255, 170, 40, 220);
    for (const auto& e : edges)
    {
        glm::float4 a = clip[e[0]];
        glm::float4 b = clip[e[1]];
        if (!clipSegmentToEye(a, b))
        {
            continue;
        }
        drawList->AddLine(toScreen(a), toScreen(b), color, 1.5f);
    }
}

// Box enclosing every instance of a node, in the space of the first one's
// transform. A glTF mesh with several primitives becomes one instance per
// primitive -- the BrainStem figure alone has 59 -- so boxing only the instance
// under the cursor outlines a fragment of what the user thinks is selected, and
// boxing each of them separately is a cage, not a highlight.
bool EditorApp::computeNodeBounds(const Scene::Node& node,
                                 glm::float3& outMin,
                                 glm::float3& outMax,
                                 glm::mat4& outWorldFromLocal)
{
    const std::vector<Instance>& instances = m_scene->getInstances();
    bool any = false;
    glm::mat4 fromRef(1.0f);
    outMin = glm::float3(std::numeric_limits<float>::max());
    outMax = glm::float3(std::numeric_limits<float>::lowest());

    // Only the placements that share the picked one's transform.
    //
    // The union used to run over every instance of the node, which is right for
    // the case it was written for -- a glTF mesh split into primitives by
    // material, all at one transform. EXT_mesh_gpu_instancing breaks that: one
    // node there carries up to a million placements scattered across the scene,
    // so the union was a box around the whole forest rather than around the tree
    // under the cursor, and it cost a pass over every placement to draw. Sibling
    // primitives of the clicked placement still share its transform, so they are
    // still boxed together.
    const bool haveSelected = m_selectedInstanceId != (uint32_t)-1 && m_selectedInstanceId < instances.size();
    const glm::mat4* selectedXform = haveSelected ? &instances[m_selectedInstanceId].transform : nullptr;

    for (const uint32_t instId : node.instanceIds)
    {
        if (instId >= instances.size())
        {
            continue;
        }
        if (selectedXform && memcmp(selectedXform, &instances[instId].transform, sizeof(glm::mat4)) != 0)
        {
            continue;
        }
        glm::float3 instMin(0.0f);
        glm::float3 instMax(0.0f);
        if (!m_scene->computeInstanceBounds(instId, instMin, instMax))
        {
            continue;
        }
        if (!any)
        {
            outWorldFromLocal = instances[instId].transform;
            fromRef = glm::inverse(outWorldFromLocal);
        }
        // Primitives of one node share its transform, but going through the
        // reference space keeps the union honest if they ever stop doing so.
        const glm::mat4 refFromInst = fromRef * instances[instId].transform;
        for (int i = 0; i < 8; ++i)
        {
            const glm::float3 corner((i & 1) ? instMax.x : instMin.x, (i & 2) ? instMax.y : instMin.y,
                                     (i & 4) ? instMax.z : instMin.z);
            const glm::float3 inRef = glm::float3(refFromInst * glm::float4(corner, 1.0f));
            outMin = glm::min(outMin, inRef);
            outMax = glm::max(outMax, inRef);
        }
        any = true;
    }
    return any;
}

void EditorApp::drawSelectionOverlay(Camera& cam)
{
    const std::vector<Scene::Node>& nodes = m_scene->getNodes();
    glm::float3 bbMin(0.0f);
    glm::float3 bbMax(0.0f);
    glm::mat4 worldFromLocal(1.0f);

    if (m_selectedNodeId != (uint32_t)-1 && m_selectedNodeId < nodes.size() &&
        !nodes[m_selectedNodeId].instanceIds.empty())
    {
        if (computeNodeBounds(nodes[m_selectedNodeId], bbMin, bbMax, worldFromLocal))
        {
            drawBoundsWireframe(bbMin, bbMax, worldFromLocal, cam);
        }
        return;
    }

    const std::vector<Instance>& instances = m_scene->getInstances();
    if (m_selectedInstanceId != (uint32_t)-1 && m_selectedInstanceId < instances.size() &&
        m_scene->computeInstanceBounds(m_selectedInstanceId, bbMin, bbMax))
    {
        drawBoundsWireframe(bbMin, bbMax, instances[m_selectedInstanceId].transform, cam);
    }
}

// The gizmo is drawn from inside the Viewport panel: SetDrawlist()/SetRect() and
// Manipulate() then share one window scope, and the gizmo keeps working when the
// Properties panel is closed or its dock tab is inactive.
void EditorApp::drawSelectionGizmo(Camera& cam)
{
    const bool hasLight = m_selectedLightId != (uint32_t)-1 && m_selectedLightId < m_scene->getLightsDesc().size();
    const bool hasNode = m_selectedNodeId != (uint32_t)-1 && m_selectedNodeId < m_scene->getNodes().size();
    if (!hasLight && !hasNode)
    {
        return;
    }

    if (hasLight)
    {
        Scene::UniformLightDesc desc = m_scene->getLightsDesc()[m_selectedLightId];
        const glm::float4x4 translationMatrix = glm::translate(glm::float4x4(1.0f), desc.position);
        const glm::float4x4 rotationMatrix{ glm::quat(glm::radians(desc.orientation)) };
        glm::float4x4 lightXform = translationMatrix * rotationMatrix;

        ImGuizmo::PushID((int)m_selectedLightId);
        const bool wasUsing = ImGuizmo::IsUsing();
        showGizmo(cam, &lightXform[0][0], m_gizmoOperation);
        if (ImGuizmo::IsUsing())
        {
            if (!wasUsing)
            {
                pushUndoLight(m_selectedLightId);
            }
            float translation[3], rotation[3], scale[3];
            ImGuizmo::DecomposeMatrixToComponents(&lightXform[0][0], translation, rotation, scale);
            desc.position = glm::float3(translation[0], translation[1], translation[2]);
            desc.orientation = glm::float3(rotation[0], rotation[1], rotation[2]);
            m_scene->setLight(m_selectedLightId, desc);
            markDocumentDirty();
        }
        ImGuizmo::PopID();
        return;
    }

    const Scene::Node& node = m_scene->getNodes()[m_selectedNodeId];
    glm::mat4 world = m_scene->getGlobalTransforms()[m_selectedNodeId];
    ImGuizmo::PushID((int)m_selectedNodeId + 1000);
    const bool wasUsing = ImGuizmo::IsUsing();
    showGizmo(cam, &world[0][0], m_gizmoOperation);
    if (ImGuizmo::IsUsing())
    {
        if (!wasUsing)
        {
            pushUndoNode(m_selectedNodeId);
        }
        glm::mat4 parentWorld(1.0f);
        if (node.parent >= 0)
        {
            parentWorld = m_scene->getGlobalTransforms()[node.parent];
        }
        const glm::mat4 local = glm::inverse(parentWorld) * world;
        glm::float3 translation, scale;
        glm::quat rotation;
        decomposeTrs(local, translation, rotation, scale);
        m_scene->setNodeLocalTransform(m_selectedNodeId, translation, rotation, scale);
        markDocumentDirty();
    }
    ImGuizmo::PopID();
}

void EditorApp::drawViewportPanel()
{
    static bool mIsHoveredViewport = false;
    bool thisFrameHovered = false;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    if (ImGui::Begin("Viewport"))
    {
        const ImVec2 availableSize = ImGui::GetContentRegionAvail();

        auto calculateAspectRatioSize = [](ImVec2 availableSize, int fixedWidth, int fixedHeight) {
            float aspectRatio = static_cast<float>(fixedWidth) / static_cast<float>(fixedHeight);
            float width = availableSize.x;
            float height = availableSize.x / aspectRatio;
            if (height > availableSize.y)
            {
                height = availableSize.y;
                width = height * aspectRatio;
            }
            return ImVec2(width, height);
        };

        auto calculateVerticalPadding = [](ImVec2 availableSize, float renderedHeight) {
            return (availableSize.y - renderedHeight) / 2.0f;
        };

        const uint32_t renderW = m_settingsManager->getAs<uint32_t>("render/width");
        const uint32_t renderH = m_settingsManager->getAs<uint32_t>("render/height");
        ImVec2 viewportSize = calculateAspectRatioSize(availableSize, renderW, renderH);
        float verticalPadding = calculateVerticalPadding(availableSize, viewportSize.y);
        const float horizontalPadding = (availableSize.x - viewportSize.x) * 0.5f;

        ImGui::PushStyleVar(ImGuiStyleVar_ImageBorderSize, 0.0f);

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + verticalPadding);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + horizontalPadding);

        // The frame is shown through items that claim no ID. ImGuizmo starts a
        // drag only while ImGui reports nothing hovered and nothing active, so an
        // ImageButton spanning the viewport leaves the handles drawn but dead:
        // the cursor is always over it whenever it is over a handle.
        void* viewportTexture = m_display->getDisplayNativeTexure();
        const ImVec2 topLeft = ImGui::GetCursorScreenPos();
        if (viewportTexture != nullptr)
        {
            ImGui::Image((ImTextureID)viewportTexture, viewportSize);
        }
        else
        {
            ImGui::GetWindowDrawList()->AddRectFilled(
                topLeft, ImVec2(topLeft.x + viewportSize.x, topLeft.y + viewportSize.y),
                ImGui::GetColorU32(ImGuiCol_FrameBg));
            ImGui::Dummy(viewportSize);
        }

        m_viewportRectMin = ImGui::GetItemRectMin();
        m_viewportRectMax = ImGui::GetItemRectMax();
        const bool itemHovered = ImGui::IsItemHovered();

        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(m_viewportRectMin.x, m_viewportRectMin.y, m_viewportRectMax.x - m_viewportRectMin.x,
                          m_viewportRectMax.y - m_viewportRectMin.y);

        Camera& cam = m_scene->getCamera(m_selectedCamera);
        drawSelectionOverlay(cam);
        drawSelectionGizmo(cam);

        ImGui::PopStyleVar();

        if (itemHovered)
        {
            m_display->setViewPortHovered(true);
            m_cameraController->setViewportHovered(true);
            thisFrameHovered = true;

            // Select on release, and only when the cursor stayed put: pressing LMB
            // also starts a camera dolly, and a selection change in the middle of
            // navigating the scene is never what the user meant.
            const bool gizmoBusy = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !gizmoBusy)
            {
                const ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                if (std::abs(drag.x) < 4.0f && std::abs(drag.y) < 4.0f)
                {
                    applySelectionFromPick(pickAtScreenPos(ImGui::GetIO().MousePos));
                }
            }
        }

        // Selection readout: the only in-viewport confirmation that a click landed.
        ImGui::SetCursorScreenPos(ImVec2(m_viewportRectMin.x + 8.0f, m_viewportRectMin.y + 8.0f));
        if (m_selectedLightId != (uint32_t)-1)
        {
            ImGui::Text("Selected: light %u", m_selectedLightId);
        }
        else if (m_selectedNodeId != (uint32_t)-1 && m_selectedNodeId < m_scene->getNodes().size())
        {
            const Scene::Node& node = m_scene->getNodes()[m_selectedNodeId];
            ImGui::Text("Selected: %s", node.name.empty() ? "(unnamed)" : node.name.c_str());
        }
        else
        {
            ImGui::TextDisabled("Click an object to select");
        }
    }

    if (mIsHoveredViewport && !thisFrameHovered)
    {
        m_display->setViewPortHovered(false);
        m_cameraController->setViewportHovered(false);
    }
    mIsHoveredViewport = thisFrameHovered;

    ImGui::End();
    ImGui::PopStyleVar();
}

} // namespace oka
