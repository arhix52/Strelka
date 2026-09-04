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
    // Compiling the shader pipeline outlives the scene build and produces no
    // image at all while it runs -- OptiX JITs its module the first time it sees
    // a specialisation, which on a cold cache is ten seconds of black viewport.
    // Without this the overlay closed at the end of the build and left the user
    // looking at nothing, with no way to tell it apart from a hang.
    const double compileMs = m_render ? m_render->pipelineCompileElapsedMs() : -1.0;
    const bool compiling = compileMs >= 0.0;
    if (!m_isLoading && !building && !compiling)
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

    // The pipeline also compiles with no scene open -- at startup, on a cold
    // cache -- and there the file name is empty and the bar sits at zero for the
    // whole wait. A blank title over a stalled bar reads as a hang, which is the
    // opposite of what this overlay is for, so neither is drawn unless a scene
    // is genuinely being loaded.
    const bool loading = m_isLoading || building;
    if (loading)
    {
        ImGui::TextUnformatted(std::filesystem::path(m_sceneFile).filename().string().c_str());
        ImGui::Spacing();

        const std::string label =
            total > 0 ? fmt::format("{}  {}/{}", kStageNames[stage], done, total) : std::string(kStageNames[stage]);
        ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 0.0f), label.c_str());
    }

    if (compiling)
    {
        // No bar: OptiX reports no progress during a module compile, and a bar
        // that does not move is a worse lie than no bar. The elapsed seconds are
        // the whole message -- they are what says this is working rather than
        // stuck.
        ImGui::Spacing();
        ImGui::TextUnformatted(loading ? "Compiling shaders for this scene" : "Compiling shaders");
        ImGui::SameLine();
        ImGui::TextDisabled("%.0f s", compileMs / 1000.0);
        ImGui::TextDisabled("First time only -- the result is cached on disk.");
    }

    // GPU resources are already owned once the build starts, so only the parse
    // stage can be cancelled safely -- and a compile with no scene behind it has
    // nothing to cancel at all, so the button is not drawn rather than drawn
    // dead.
    if (loading)
    {
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
    }

    ImGui::End();
}

} // namespace oka
