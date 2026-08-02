#pragma once

#include "imgui.h"

namespace oka::test
{

/// An ImGui context with no backend, sized like the editor's default window.
///
/// Enough for input and hit-testing behaviour: NewFrame/Render run, items get
/// their rects, and mouse events go in through the normal event queue.
struct HeadlessImGui
{
    HeadlessImGui()
    {
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(1024.0f, 768.0f);
        io.DeltaTime = 1.0f / 60.0f;
        io.Fonts->AddFontDefault();
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    }

    ~HeadlessImGui()
    {
        ImGui::DestroyContext();
    }

    HeadlessImGui(const HeadlessImGui&) = delete;
    HeadlessImGui& operator=(const HeadlessImGui&) = delete;
};

} // namespace oka::test
