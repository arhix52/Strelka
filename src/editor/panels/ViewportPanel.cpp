#include "../EditorApp.h"

#include "../editor_overlay.h"

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
    const std::optional<editor_viewport::Point> uv =
        editor_viewport::screenToImageUv({ screenPos.x, screenPos.y }, m_viewportLayout);
    if (!uv)
    {
        return {};
    }

    Camera camera = m_scene->getCamera(m_selectedCamera);
    if (mPresentedPreviewHeight > 0)
    {
        camera.updateAspectRatio(static_cast<float>(mPresentedPreviewWidth) / static_cast<float>(mPresentedPreviewHeight));
    }
    glm::float3 origin, dir;
    generatePickRay(camera, glm::float2(uv->x, uv->y), origin, dir);

    return m_scene->pick(origin, dir);
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
    const glm::mat4 viewFromLocal = cam.matrices.view * worldFromLocal;
    const glm::mat4 clipFromLocal = cam.matrices.perspective * viewFromLocal;
    const glm::float2 rectMin(m_viewportRectMin.x, m_viewportRectMin.y);
    const glm::float2 rectSize(m_viewportRectMax.x - m_viewportRectMin.x, m_viewportRectMax.y - m_viewportRectMin.y);

    glm::float4 clip[8];
    float viewZ[8];
    for (int i = 0; i < 8; ++i)
    {
        const glm::float3 corner((i & 1) ? bbMax.x : bbMin.x, (i & 2) ? bbMax.y : bbMin.y, (i & 4) ? bbMax.z : bbMin.z);
        const glm::float4 local(corner, 1.0f);
        clip[i] = clipFromLocal * local;
        viewZ[i] = (viewFromLocal * local).z;
    }

    static const int edges[12][2] = { { 0, 1 }, { 1, 3 }, { 3, 2 }, { 2, 0 }, { 4, 5 }, { 5, 7 },
                                      { 7, 6 }, { 6, 4 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } };
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const ImU32 color = IM_COL32(255, 170, 40, 220);
    for (const auto& e : edges)
    {
        glm::float4 a = clip[e[0]];
        glm::float4 b = clip[e[1]];
        if (!editor_overlay::trimSegmentToNearPlane(a, b, viewZ[e[0]], viewZ[e[1]], cam.znear))
        {
            continue;
        }
        glm::float2 pa(0.0f), pb(0.0f);
        if (!editor_overlay::clipToScreen(a, rectMin, rectSize, pa) ||
            !editor_overlay::clipToScreen(b, rectMin, rectSize, pb))
        {
            continue;
        }
        drawList->AddLine(ImVec2(pa.x, pa.y), ImVec2(pb.x, pb.y), color, 1.5f);
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
    const bool haveSelected = m_selectedInstanceId != kInvalidIndex && m_selectedInstanceId < instances.size();
    const glm::mat4* selectedXform = haveSelected ? &instances[m_selectedInstanceId].transform : nullptr;

    for (const uint32_t instId : node.instanceIds)
    {
        if (instId >= instances.size())
        {
            continue;
        }
        if (selectedXform && *selectedXform != instances[instId].transform)
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
            const glm::float3 corner(
                (i & 1) ? instMax.x : instMin.x, (i & 2) ? instMax.y : instMin.y, (i & 4) ? instMax.z : instMin.z);
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

    if (m_selectedNodeId != kInvalidIndex && m_selectedNodeId < nodes.size() &&
        !nodes[m_selectedNodeId].instanceIds.empty())
    {
        if (computeNodeBounds(nodes[m_selectedNodeId], bbMin, bbMax, worldFromLocal))
        {
            drawBoundsWireframe(bbMin, bbMax, worldFromLocal, cam);
        }
        return;
    }

    const std::vector<Instance>& instances = m_scene->getInstances();
    if (m_selectedInstanceId != kInvalidIndex && m_selectedInstanceId < instances.size() &&
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
    const bool hasLight = m_selectedLightId != kInvalidIndex && m_selectedLightId < m_scene->getLightsDesc().size();
    const bool hasNode = m_selectedNodeId != kInvalidIndex && m_selectedNodeId < m_scene->getNodes().size();
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
    if (ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
    {
        const ImVec2 availableSize = ImGui::GetContentRegionAvail();
        const ImVec2 panelMin = ImGui::GetCursorScreenPos();
        const ImVec2 panelMax(panelMin.x + availableSize.x, panelMin.y + availableSize.y);
        const ImVec2 framebufferScale = ImGui::GetIO().DisplayFramebufferScale;
        const uint32_t renderWidth = m_settingsManager->getAs<uint32_t>("render/width");
        const uint32_t renderHeight = m_settingsManager->getAs<uint32_t>("render/height");
        const uint32_t presentationWidth = mPresentedPreviewWidth > 0 ? mPresentedPreviewWidth : renderWidth;
        const uint32_t presentationHeight = mPresentedPreviewHeight > 0 ? mPresentedPreviewHeight : renderHeight;
        const editor_viewport::Rect panelRect = {
            { panelMin.x, panelMin.y },
            { panelMax.x, panelMax.y },
        };
        m_viewportLayout =
            editor_viewport::computeLayout(panelRect, presentationWidth, presentationHeight,
                                           { framebufferScale.x, framebufferScale.y }, m_viewportPresentation);
        m_viewportRectMin = ImVec2(m_viewportLayout.imageRect.min.x, m_viewportLayout.imageRect.min.y);
        m_viewportRectMax = ImVec2(m_viewportLayout.imageRect.max.x, m_viewportLayout.imageRect.max.y);
        const ImVec2 viewportSize(m_viewportLayout.imageRect.width(), m_viewportLayout.imageRect.height());

        // The panel is deliberately black outside the image so Fit mode reads as
        // a camera frame rather than as empty editor chrome.
        ImGui::GetWindowDrawList()->AddRectFilled(panelMin, panelMax, IM_COL32_BLACK);

        ImGui::PushStyleVar(ImGuiStyleVar_ImageBorderSize, 0.0f);
        ImGui::SetCursorScreenPos(m_viewportRectMin);

        // The frame is shown through items that claim no ID. ImGuizmo starts a
        // drag only while ImGui reports nothing hovered and nothing active, so an
        // ImageButton spanning the viewport leaves the handles drawn but dead:
        // the cursor is always over it whenever it is over a handle.
        const void* viewportTexture = m_display->getDisplayNativeTexure();
        const ImVec2 topLeft = ImGui::GetCursorScreenPos();
        if (viewportTexture != nullptr)
        {
            ImGui::Image((ImTextureID)viewportTexture, viewportSize);
        }
        else
        {
            ImGui::GetWindowDrawList()->AddRectFilled(topLeft,
                                                      ImVec2(topLeft.x + viewportSize.x, topLeft.y + viewportSize.y),
                                                      ImGui::GetColorU32(ImGuiCol_FrameBg));
            ImGui::Dummy(viewportSize);
        }

        bool itemHovered = ImGui::IsItemHovered();

        Camera cam = m_scene->getCamera(m_selectedCamera);
        if (presentationHeight > 0)
        {
            cam.updateAspectRatio(static_cast<float>(presentationWidth) / static_cast<float>(presentationHeight));
        }

        // ImGuizmo derives the facing of the rotation rings from the projection,
        // and drops the whole gizmo when it reads the object as behind the eye.
        // Both of those assume perspective, so an orthographic camera needs to say
        // so or its gizmo comes out mirrored or missing.
        ImGuizmo::SetOrthographic(cam.projection == Camera::ProjectionType::orthographic);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(m_viewportRectMin.x, m_viewportRectMin.y, m_viewportRectMax.x - m_viewportRectMin.x,
                          m_viewportRectMax.y - m_viewportRectMin.y);

        // Selection geometry belongs to the rendered camera image. In Fit mode
        // the surrounding pixels are letterbox bars, while Fill and 1:1 can crop
        // the image at the panel edge; neither area should receive outline lines.
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->PushClipRect(ImVec2(m_viewportLayout.visibleRect.min.x, m_viewportLayout.visibleRect.min.y),
                               ImVec2(m_viewportLayout.visibleRect.max.x, m_viewportLayout.visibleRect.max.y), true);
        drawSelectionOverlay(cam);
        drawList->PopClipRect();
        drawSelectionGizmo(cam);

        ImGui::PopStyleVar();

        // Presentation affects only this quad; it never reallocates renderer
        // resources or resets accumulation.
        const char* const modeNames[] = { "Fit", "1:1", "Fill" };
        ImGui::SetCursorScreenPos(ImVec2(panelMax.x - 112.0f, panelMin.y + 8.0f));
        ImGui::SetNextItemWidth(104.0f);
        int presentation = static_cast<int>(m_viewportPresentation);
        if (ImGui::Combo("##viewportPresentation", &presentation, modeNames, IM_ARRAYSIZE(modeNames)))
        {
            m_viewportPresentation = static_cast<editor_viewport::PresentationMode>(presentation);
        }
        if (ImGui::IsItemHovered() || ImGui::IsItemActive())
        {
            itemHovered = false;
        }

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
        ImGui::SetCursorScreenPos(ImVec2(panelMin.x + 8.0f, panelMin.y + 8.0f));
        if (m_selectedLightId != kInvalidIndex)
        {
            ImGui::Text("Selected: light %u", m_selectedLightId);
        }
        else if (m_selectedNodeId != kInvalidIndex && m_selectedNodeId < m_scene->getNodes().size())
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
