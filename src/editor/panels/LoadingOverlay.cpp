#include "../EditorApp.h"

#include "imgui.h"

#include <algorithm>
#include <cfloat>
#include <filesystem>

namespace oka
{

// The parse and GPU build are one wait from the user's point of view, even
// though one runs on a worker and the other advances a stage per frame.
void EditorApp::drawLoadingOverlay()
{
    const bool building = m_render && m_render->isBuildingScene();
    if (!m_isLoading && !building)
    {
        return;
    }

    // Weighted by measured cost on a large scene so the bar does not appear
    // stuck during acceleration-structure construction.
    static constexpr float kStageWeights[(size_t)LoadProgress::Stage::Count] = {
        0.00f, // Idle
        1.12f, // Reading
        0.73f, // Parsing
        0.14f, // Geometry
        1.07f, // Textures
        1.95f, // Structures
        0.04f, // Environment
        0.00f, // Done
    };
    static const char* const kStageNames[(size_t)LoadProgress::Stage::Count] = {
        "Starting",           "Reading file",     "Parsing scene",
        "Uploading geometry", "Loading textures", "Building acceleration structures",
        "Environment",        "Finishing",
    };

    const uint32_t stage =
        std::min(m_loadProgress.stage.load(std::memory_order_acquire), (uint32_t)LoadProgress::Stage::Done);
    const uint32_t done = m_loadProgress.done.load(std::memory_order_relaxed);
    const uint32_t total = m_loadProgress.total.load(std::memory_order_relaxed);

    float totalWeight = 0.0f;
    for (const float weight : kStageWeights)
    {
        totalWeight += weight;
    }
    float before = 0.0f;
    for (uint32_t i = 0; i < stage; ++i)
    {
        before += kStageWeights[i];
    }
    const float within = total > 0 ? std::min(1.0f, (float)done / (float)total) : 0.0f;
    const float fraction = totalWeight > 0.0f ? (before + kStageWeights[stage] * within) / totalWeight : 0.0f;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(460.0f, 0.0f), ImGuiCond_Always);
    ImGui::Begin("##Loading", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing);

    ImGui::TextUnformatted(std::filesystem::path(m_sceneFile).filename().string().c_str());
    ImGui::Spacing();

    const std::string label =
        total > 0 ? fmt::format("{}  {}/{}", kStageNames[stage], done, total) : std::string(kStageNames[stage]);
    ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 0.0f), label.c_str());

    // GPU resources are already owned once the build starts, so only the parse
    // stage can be cancelled safely.
    ImGui::Spacing();
    ImGui::BeginDisabled(!m_isLoading || m_loadProgress.isCancelled());
    if (ImGui::Button("Cancel"))
    {
        m_loadProgress.cancel();
    }
    ImGui::EndDisabled();
    if (m_loadProgress.isCancelled())
    {
        ImGui::SameLine();
        ImGui::TextDisabled("cancelling...");
    }

    ImGui::End();
}

} // namespace oka
