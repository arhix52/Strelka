#include "../EditorApp.h"

#include "imgui.h"
#include "ImGuizmo.h"

namespace oka
{

void EditorApp::drawViewportPanel()
{
    static bool mIsHoveredViewport = false;
    bool thisFrameHovered = false;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    if (ImGui::Begin("Viewport"))
    {
        const ImVec2 availableSize = ImGui::GetContentRegionAvail();
        const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;

        auto calculateAspectRatioSize = [](ImVec2 availableSize, int fixedWidth, int fixedHeight)
        {
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

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + verticalPadding);
        ImGui::ImageButton(m_display->getDisplayNativeTexure(), viewportSize);

        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y + verticalPadding, viewportSize.x, viewportSize.y);

        ImGui::PopStyleVar();

        if (ImGui::IsItemHovered())
        {
            if (ImGui::IsKeyDown(ImGuiKey_Space))
            {
                // picking code, selecting
            }
            m_display->setViewPortHovered(true);
            thisFrameHovered = true;
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
