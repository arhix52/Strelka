#include <doctest/doctest.h>

#include <strelka/scene/camera.h>

#include "imgui.h"
#include "ImGuizmo.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace oka;

namespace
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
};

} // namespace

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

        ImDrawList* viewportList = nullptr;
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
    ImDrawList* viewportList = ImGui::GetWindowDrawList();
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
