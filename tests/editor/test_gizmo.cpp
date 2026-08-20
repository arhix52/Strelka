#include <doctest/doctest.h>

#include <strelka/scene/camera.h>

#include "headless_imgui.h"
#include "ImGuizmo.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/geometric.hpp>

#include <cmath>

using namespace oka;
using oka::test::HeadlessImGui;

namespace
{

// How the viewport frame is submitted before the gizmo runs. The editor draws it
// with an item that claims no ID for a reason this fixture is built to pin down.
enum class ViewportItem
{
    image,
    imageButton
};

struct DragResult
{
    glm::float4x4 model{ 1.0f };
    bool grabbed = false;
};

// One mouse gesture over the gizmo: press at `grabOffset` from the projected
// origin, drag by `dragBy`, release. The target sits at the origin facing the
// camera, so the gizmo centre is the middle of the viewport rect.
DragResult dragGizmo(ViewportItem item,
                     ImGuizmo::OPERATION operation,
                     ImVec2 grabOffset,
                     ImVec2 dragBy,
                     const glm::float4x4& startModel = glm::float4x4(1.0f),
                     glm::float3 cameraPos = glm::float3(0.0f, 0.0f, 5.0f))
{
    HeadlessImGui ctx;

    Camera cam;
    cam.position = cameraPos;
    cam.setPerspective(45.0f, 1024.0f / 768.0f, 0.1f, 1000.0f);
    cam.updateViewMatrix();

    DragResult result;
    result.model = startModel;

    const ImVec2 viewportSize(1024.0f, 768.0f);
    const ImVec2 grab(viewportSize.x * 0.5f + grabOffset.x, viewportSize.y * 0.5f + grabOffset.y);

    struct Step
    {
        ImVec2 mouse;
        bool down;
    };
    // Hover first: ImGui derives a click from the previous frame's button state,
    // so a press on frame one would never read as one.
    const Step steps[] = {
        { grab, false },
        { grab, true },
        { ImVec2(grab.x + dragBy.x, grab.y + dragBy.y), true },
        { ImVec2(grab.x + dragBy.x, grab.y + dragBy.y), false },
    };

    for (const Step& step : steps)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.AddMousePosEvent(step.mouse.x, step.mouse.y);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, step.down);

        ImGui::NewFrame();
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::BeginFrame();

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

        const ImVec2 itemMin = ImGui::GetItemRectMin();
        const ImVec2 itemMax = ImGui::GetItemRectMax();
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(itemMin.x, itemMin.y, itemMax.x - itemMin.x, itemMax.y - itemMin.y);
        ImGuizmo::Manipulate(glm::value_ptr(cam.matrices.view), glm::value_ptr(cam.matrices.perspective), operation,
                             ImGuizmo::WORLD, &result.model[0][0]);
        result.grabbed = result.grabbed || ImGuizmo::IsUsing();
        ImGui::End();

        ImGui::Render();
    }

    return result;
}

// Handle geometry is internal to ImGuizmo and scales with the projection, so the
// grab point is found by walking outwards from the centre along +x instead of
// being hardcoded to a size the library is free to change.
DragResult dragHandleAlongX(ImGuizmo::OPERATION operation, ImVec2 dragBy)
{
    for (int offsetPixels = 10; offsetPixels < 400; offsetPixels += 5)
    {
        const float offset = static_cast<float>(offsetPixels);
        DragResult result = dragGizmo(ViewportItem::image, operation, ImVec2(offset, 0.0f), dragBy);
        if (result.grabbed)
        {
            return result;
        }
    }
    return {};
}

} // namespace

// The bug this pins: dragging a handle did nothing in the editor, because the
// viewport frame was an ImageButton. ImGuizmo::CanActivate() requires ImGui to
// report no hovered and no active item, so the handles drew and never grabbed.
TEST_CASE("dragging the gizmo moves the target")
{
    const DragResult moved = dragGizmo(ViewportItem::image, ImGuizmo::TRANSLATE, ImVec2(0.0f, 0.0f), ImVec2(80.0f, 0.0f));

    CHECK(moved.grabbed);
    INFO("translation=(" << moved.model[3][0] << "," << moved.model[3][1] << "," << moved.model[3][2] << ")");
    CHECK(moved.model[3][0] > 0.0f);
    CHECK(glm::length(glm::float3(moved.model[3])) > 0.01f);
}

TEST_CASE("an interactive viewport item makes the gizmo handles dead")
{
    const DragResult unmoved =
        dragGizmo(ViewportItem::imageButton, ImGuizmo::TRANSLATE, ImVec2(0.0f, 0.0f), ImVec2(80.0f, 0.0f));

    CHECK_FALSE(unmoved.grabbed);
    CHECK(unmoved.model[3][0] == doctest::Approx(0.0f));
    CHECK(unmoved.model[3][1] == doctest::Approx(0.0f));
    CHECK(unmoved.model[3][2] == doctest::Approx(0.0f));
}

// A glTF export that authored its model in centimetres and scaled the node down
// (the vespa asset uses 0.003) is the ordinary case, not an exotic one.
TEST_CASE("dragging a heavily scaled target keeps the matrix finite")
{
    const glm::float3 position(1.2437f, -0.07f, -0.0804f);
    const glm::float4x4 model =
        glm::translate(glm::float4x4(1.0f), position) * glm::scale(glm::float4x4(1.0f), glm::float3(0.003f));

    const DragResult moved = dragGizmo(ViewportItem::image, ImGuizmo::TRANSLATE, ImVec2(0.0f, 0.0f),
                                       ImVec2(80.0f, 0.0f), model, position + glm::float3(0.0f, 0.0f, 5.0f));

    REQUIRE(moved.grabbed);
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            INFO("column " << col << " row " << row);
            REQUIRE(std::isfinite(moved.model[col][row]));
        }
    }
    // The scale has to survive a translate untouched, or the object jumps in size
    // the moment it is dragged.
    CHECK(glm::length(glm::float3(moved.model[0])) == doctest::Approx(0.003f).epsilon(0.01f));
    CHECK(moved.model[3][0] > position.x);
}

TEST_CASE("dragging a rotate handle turns the target")
{
    const DragResult rotated = dragHandleAlongX(ImGuizmo::ROTATE, ImVec2(0.0f, 60.0f));

    REQUIRE(rotated.grabbed);
    // A rotation touches the basis and leaves the position where it was.
    const glm::float3 axisX(rotated.model[0]);
    CHECK(glm::length(glm::float3(rotated.model[3])) == doctest::Approx(0.0f));
    CHECK(glm::length(axisX) == doctest::Approx(1.0f).epsilon(0.01f));
    CHECK(glm::length(axisX - glm::float3(1.0f, 0.0f, 0.0f)) > 0.01f);
}

TEST_CASE("dragging a scale handle resizes the target")
{
    const DragResult scaled = dragHandleAlongX(ImGuizmo::SCALE, ImVec2(60.0f, 0.0f));

    REQUIRE(scaled.grabbed);
    const float scaleX = glm::length(glm::float3(scaled.model[0]));
    INFO("scaleX=" << scaleX);
    CHECK(scaleX > 1.01f);
    CHECK(glm::length(glm::float3(scaled.model[3])) == doctest::Approx(0.0f));
}

TEST_CASE("ImGuizmo draws with editor call order (dockspace, manipulate from another window)")
{
    HeadlessImGui ctx;
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    Camera cam;
    cam.position = glm::float3(0.0f, 0.0f, 5.0f);
    cam.setPerspective(45.0f, 1024.0f / 768.0f, 0.1f, 1000.0f);
    cam.updateViewMatrix();

    glm::float4x4 model(1.0f);

    // Two frames: docking needs a settled layout, and hover state needs history.
    int vtxBefore = 0, vtxAfter = 0;
    for (int frame = 0; frame < 2; ++frame)
    {
        ImGui::NewFrame();
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::BeginFrame();
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

        const ImDrawList* viewportList = nullptr;
        ImGui::Begin("Viewport");
        viewportList = ImGui::GetWindowDrawList();
        ImGui::InvisibleButton("##viewport", ImVec2(600, 400));
        const ImVec2 itemMin = ImGui::GetItemRectMin();
        const ImVec2 itemMax = ImGui::GetItemRectMax();
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(itemMin.x, itemMin.y, itemMax.x - itemMin.x, itemMax.y - itemMin.y);
        ImGui::End();

        vtxBefore = viewportList->VtxBuffer.Size;
        ImGui::Begin("Properties");
        ImGuizmo::Manipulate(glm::value_ptr(cam.matrices.view), glm::value_ptr(cam.matrices.perspective),
                             ImGuizmo::TRANSLATE, ImGuizmo::WORLD, &model[0][0]);
        ImGui::End();
        vtxAfter = viewportList->VtxBuffer.Size;

        ImGui::Render();
    }

    INFO("vtxBefore=" << vtxBefore << " vtxAfter=" << vtxAfter);
    CHECK(vtxAfter > vtxBefore);
}

// ImGuizmo bails out without any warning when the target projects behind the
// camera, which is also what a stale or never-built view matrix looks like: the
// gizmo just never shows up. Pinning the behaviour keeps that failure mode
// documented next to the positive case.
TEST_CASE("ImGuizmo draws nothing when the target is behind the camera")
{
    HeadlessImGui ctx;

    Camera cam;
    cam.position = glm::float3(0.0f, 0.0f, 5.0f);
    cam.setPerspective(45.0f, 1024.0f / 768.0f, 0.1f, 1000.0f);
    cam.updateViewMatrix();

    // Camera looks down -z from z = 5, so z = 20 is behind it.
    glm::float4x4 model = glm::translate(glm::float4x4(1.0f), glm::float3(0.0f, 0.0f, 20.0f));

    ImGui::NewFrame();
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::BeginFrame();

    ImGui::Begin("Viewport");
    const ImDrawList* viewportList = ImGui::GetWindowDrawList();
    ImGuizmo::SetDrawlist();
    ImGuizmo::SetRect(0.0f, 0.0f, 1024.0f, 768.0f);
    ImGui::End();

    const int vtxBefore = viewportList->VtxBuffer.Size;
    ImGuizmo::Manipulate(glm::value_ptr(cam.matrices.view), glm::value_ptr(cam.matrices.perspective),
                         ImGuizmo::TRANSLATE, ImGuizmo::WORLD, &model[0][0]);
    const int vtxAfter = viewportList->VtxBuffer.Size;
    ImGui::Render();

    CHECK(vtxAfter == vtxBefore);
}
