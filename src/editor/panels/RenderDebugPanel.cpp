#include "../EditorApp.h"

#include "imgui.h"

namespace oka
{

void EditorApp::drawRenderDebugPanel()
{
    if (!ImGui::Begin("Render Debug", &m_showRenderDebug))
    {
        ImGui::End();
        return;
    }

    bool analytic = m_settingsManager->getAs<bool>("render/validate/analyticLights");
    if (ImGui::Checkbox("Analytic lights", &analytic))
    {
        m_settingsManager->setAs<bool>("render/validate/analyticLights", analytic);
        m_sharedCtx->mSubframeIndex = 0;
    }

    // Must match DebugMode in ShaderTypes.h, in order.
    const char* const debugViewOptions[] = { "None",
                                             "Normals",
                                             "Motion Blur",
                                             "AOV: diffuse",
                                             "AOV: specular",
                                             "AOV: normal",
                                             "AOV: roughness",
                                             "AOV: depth",
                                             "AOV: motion",
                                             "AOV: reactive",
                                             "AOV: spec hit distance",
                                             "Cache: voxel grid",
                                             "Cache: radiance",
                                             "Cache: occupancy",
                                             "Cache: bounce count" };
    const char* const debugViewHelp[] = { nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          "A stable colour per cache voxel, at the first surface each camera ray reaches.\n"
                                          "Use this to choose a voxel size before switching the cache on.",
                                          "The cache answer at the first surface, shown directly and tonemapped like\n"
                                          "the beauty render. Black means that voxel is missing or unresolved.",
                                          "Green entries have resolved radiance, amber entries are inserted but unresolved,\n"
                                          "and black entries are free. Around 10-20% occupied is healthy.",
                                          "Path depth: blue none, green one, yellow two, red three or more.\n"
                                          "Compare cache off and on to see where it shortens paths." };
    static_assert(IM_ARRAYSIZE(debugViewOptions) == IM_ARRAYSIZE(debugViewHelp));

    const int debugViewOptionCount = IM_ARRAYSIZE(debugViewOptions);
    const uint32_t requestedDebugView = m_settingsManager->getAs<uint32_t>("render/pt/debug");
    const int currentDebugViewOption =
        requestedDebugView < static_cast<uint32_t>(debugViewOptionCount) ? static_cast<int>(requestedDebugView) : 0;
    if (ImGui::BeginCombo("Debug view", debugViewOptions[currentDebugViewOption]))
    {
        for (int n = 0; n < debugViewOptionCount; ++n)
        {
            const bool selected = currentDebugViewOption == n;
            if (ImGui::Selectable(debugViewOptions[n], selected) && !selected)
            {
                m_settingsManager->setAs<uint32_t>("render/pt/debug", static_cast<uint32_t>(n));
                m_sharedCtx->mSubframeIndex = 0;
                m_render->resetTemporalHistory();
            }
            if (debugViewHelp[n] != nullptr && ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("%s", debugViewHelp[n]);
            }
            if (selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    ImGui::SeparatorText("Ray offsets");

    float materialRayTmin = m_settingsManager->getAs<float>("render/pt/dev/materialRayTmin");
    if (ImGui::InputFloat("Material ray T min", &materialRayTmin, 0.1f))
    {
        m_settingsManager->setAs<float>("render/pt/dev/materialRayTmin", materialRayTmin);
    }

    float shadowRayTmin = m_settingsManager->getAs<float>("render/pt/dev/shadowRayTmin");
    if (ImGui::InputFloat("Shadow ray T min", &shadowRayTmin, 0.1f))
    {
        m_settingsManager->setAs<float>("render/pt/dev/shadowRayTmin", shadowRayTmin);
    }

    ImGui::End();
}

} // namespace oka
