#pragma once

#include "imgui.h"

#include <cstdint>

namespace oka::imgui_style
{

inline ImVec4 rgb(uint32_t hex, float alpha = 1.0f)
{
    constexpr float kByteToFloat = 1.0f / 255.0f;
    return { static_cast<float>((hex >> 16u) & 0xffu) * kByteToFloat,
             static_cast<float>((hex >> 8u) & 0xffu) * kByteToFloat,
             static_cast<float>(hex & 0xffu) * kByteToFloat,
             alpha };
}

inline void loadEditorFont()
{
    constexpr float kFontSize = 15.0f;
#if defined(__APPLE__)
    const char* const candidates[] = {
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    };
#elif defined(_WIN32)
    const char* const candidates[] = {
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    };
#else
    const char* const candidates[] = {
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    };
#endif

    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig config;
    config.Flags |= ImFontFlags_NoLoadError;

    ImFont* font = nullptr;
    for (const char* candidate : candidates)
    {
        font = io.Fonts->AddFontFromFileTTF(candidate, kFontSize, &config, io.Fonts->GetGlyphRangesCyrillic());
        if (font != nullptr)
        {
            break;
        }
    }
    if (font == nullptr)
    {
        config.SizePixels = kFontSize;
        font = io.Fonts->AddFontDefaultVector(&config);
    }

    io.FontDefault = font;
    io.ConfigDpiScaleFonts = true;
}

inline void applyGraphiteBlue()
{
    loadEditorFont();

    ImGuiStyle& style = ImGui::GetStyle();

    style.WindowPadding = ImVec2(10.0f, 10.0f);
    style.FramePadding = ImVec2(8.0f, 5.0f);
    style.CellPadding = ImVec2(8.0f, 5.0f);
    style.ItemSpacing = ImVec2(8.0f, 6.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    style.IndentSpacing = 18.0f;
    style.ScrollbarSize = 14.0f;
    style.GrabMinSize = 10.0f;

    // Restrained rounding keeps dense editor panels precise rather than making
    // them read like a touch interface.
    style.WindowRounding = 4.0f;
    style.ChildRounding = 4.0f;
    style.PopupRounding = 5.0f;
    style.FrameRounding = 4.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 4.0f;

    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.TabBorderSize = 0.0f;
    style.TabBarBorderSize = 1.0f;
    style.TabBarOverlineSize = 2.0f;
    style.SeparatorTextBorderSize = 1.0f;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text] = rgb(0xe6e9ef);
    colors[ImGuiCol_TextDisabled] = rgb(0x8d96a5);
    colors[ImGuiCol_WindowBg] = rgb(0x101216);
    colors[ImGuiCol_ChildBg] = rgb(0x15181d);
    colors[ImGuiCol_PopupBg] = rgb(0x171a20, 0.98f);
    colors[ImGuiCol_Border] = rgb(0x343a45);
    colors[ImGuiCol_BorderShadow] = rgb(0x000000, 0.0f);

    colors[ImGuiCol_FrameBg] = rgb(0x20252d);
    colors[ImGuiCol_FrameBgHovered] = rgb(0x29313b);
    colors[ImGuiCol_FrameBgActive] = rgb(0x334052);
    colors[ImGuiCol_TitleBg] = rgb(0x12151a);
    colors[ImGuiCol_TitleBgActive] = rgb(0x171c24);
    colors[ImGuiCol_TitleBgCollapsed] = rgb(0x12151a);
    colors[ImGuiCol_MenuBarBg] = rgb(0x15181d);

    colors[ImGuiCol_ScrollbarBg] = rgb(0x111419);
    colors[ImGuiCol_ScrollbarGrab] = rgb(0x343b47);
    colors[ImGuiCol_ScrollbarGrabHovered] = rgb(0x414b5a);
    colors[ImGuiCol_ScrollbarGrabActive] = rgb(0x526176);
    colors[ImGuiCol_CheckMark] = rgb(0x4c8dff);
    colors[ImGuiCol_SliderGrab] = rgb(0x4c8dff);
    colors[ImGuiCol_SliderGrabActive] = rgb(0x78a9ff);

    colors[ImGuiCol_Button] = rgb(0x252b34);
    colors[ImGuiCol_ButtonHovered] = rgb(0x2e3a4a);
    colors[ImGuiCol_ButtonActive] = rgb(0x3b64a0);
    colors[ImGuiCol_Header] = rgb(0x263140);
    colors[ImGuiCol_HeaderHovered] = rgb(0x30425a);
    colors[ImGuiCol_HeaderActive] = rgb(0x3b577a);
    colors[ImGuiCol_Separator] = rgb(0x343a45);
    colors[ImGuiCol_SeparatorHovered] = rgb(0x4c8dff, 0.75f);
    colors[ImGuiCol_SeparatorActive] = rgb(0x4c8dff);

    colors[ImGuiCol_ResizeGrip] = rgb(0x4c8dff, 0.20f);
    colors[ImGuiCol_ResizeGripHovered] = rgb(0x4c8dff, 0.55f);
    colors[ImGuiCol_ResizeGripActive] = rgb(0x4c8dff, 0.85f);
    colors[ImGuiCol_Tab] = rgb(0x171b22);
    colors[ImGuiCol_TabHovered] = rgb(0x2e3a4a);
    colors[ImGuiCol_TabSelected] = rgb(0x202b3a);
    colors[ImGuiCol_TabSelectedOverline] = rgb(0x4c8dff);
    colors[ImGuiCol_TabDimmed] = rgb(0x14171c);
    colors[ImGuiCol_TabDimmedSelected] = rgb(0x1b2028);
    colors[ImGuiCol_TabDimmedSelectedOverline] = rgb(0x4c8dff, 0.45f);
    colors[ImGuiCol_DockingPreview] = rgb(0x4c8dff, 0.55f);
    colors[ImGuiCol_DockingEmptyBg] = rgb(0x0d0f13);

    colors[ImGuiCol_PlotLines] = rgb(0x78a9ff);
    colors[ImGuiCol_PlotLinesHovered] = rgb(0xa7c5ff);
    colors[ImGuiCol_PlotHistogram] = rgb(0x4c8dff);
    colors[ImGuiCol_PlotHistogramHovered] = rgb(0x78a9ff);
    colors[ImGuiCol_TableHeaderBg] = rgb(0x1d232c);
    colors[ImGuiCol_TableBorderStrong] = rgb(0x343a45);
    colors[ImGuiCol_TableBorderLight] = rgb(0x282e37);
    colors[ImGuiCol_TableRowBg] = rgb(0x000000, 0.0f);
    colors[ImGuiCol_TableRowBgAlt] = rgb(0xffffff, 0.025f);

    colors[ImGuiCol_TextSelectedBg] = rgb(0x4c8dff, 0.35f);
    colors[ImGuiCol_DragDropTarget] = rgb(0x78a9ff);
    colors[ImGuiCol_NavCursor] = rgb(0x78a9ff);
    colors[ImGuiCol_NavWindowingHighlight] = rgb(0xffffff, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = rgb(0x080a0d, 0.55f);
    colors[ImGuiCol_ModalWindowDimBg] = rgb(0x080a0d, 0.65f);
}

} // namespace oka::imgui_style
