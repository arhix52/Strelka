#include <doctest/doctest.h>

#include "headless_imgui.h"

using oka::test::HeadlessImGui;

namespace
{

// The viewport frame is submitted as an item with no ID so the gizmo can grab the
// mouse. Everything else the viewport does keys off that item being hovered:
// picking fires on release, and camera input is gated on it, so the hover state
// has to survive a button being held down.
struct HoverTrace
{
    bool onPress = false;
    bool whileHeld = false;
    bool onRelease = false;
};

enum class ViewportItem
{
    image,
    imageButton
};

HoverTrace traceHoverThroughClick(ViewportItem item)
{
    HeadlessImGui ctx;

    const ImVec2 viewportSize(800.0f, 600.0f);
    const ImVec2 inside(100.0f, 100.0f);

    HoverTrace trace;
    struct Step
    {
        bool down;
        bool* record;
    };
    const Step steps[] = {
        { false, nullptr }, { true, &trace.onPress }, { true, &trace.whileHeld }, { false, &trace.onRelease }
    };

    for (const Step& step : steps)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.AddMousePosEvent(inside.x, inside.y);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, step.down);

        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoTitleBar);
        ImGui::SetCursorScreenPos(ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_ImageBorderSize, 0.0f);
        if (item == ViewportItem::image)
        {
            ImGui::Image((ImTextureID)1, viewportSize);
        }
        else
        {
            ImGui::ImageButton("##viewport", (ImTextureID)1, viewportSize);
        }
        ImGui::PopStyleVar(2);

        const bool hovered = ImGui::IsItemHovered();
        if (step.record != nullptr)
        {
            *step.record = hovered;
        }
        ImGui::End();
        ImGui::Render();
    }

    return trace;
}

} // namespace

// A click on an item with no ID makes ImGui claim the window's move ID, which is
// exactly the state that would otherwise report the item as no longer hovered.
TEST_CASE("the viewport frame stays hovered while the mouse is held")
{
    const HoverTrace trace = traceHoverThroughClick(ViewportItem::image);

    CHECK(trace.onPress);
    CHECK(trace.whileHeld);
    CHECK(trace.onRelease);
}

// The same trace for the interactive item the viewport used to be drawn with, so
// the two behaviours sit side by side: it keeps hover, and it is what killed the
// gizmo handles.
TEST_CASE("an interactive viewport frame also stays hovered")
{
    const HoverTrace trace = traceHoverThroughClick(ViewportItem::imageButton);

    CHECK(trace.onPress);
    CHECK(trace.whileHeld);
    CHECK(trace.onRelease);
}
