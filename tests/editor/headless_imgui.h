#pragma once

#include "imgui.h"

namespace oka::test
{

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
