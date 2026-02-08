#include "../EditorApp.h"

#include "imgui.h"

namespace oka
{

void EditorApp::drawAnimationPanel()
{
    if (ImGui::Begin("Animations"))
    {
        const auto& animations = m_scene->getAnimations();

        if (!animations.empty())
        {
            // Playback controls
            if (ImGui::CollapsingHeader("Playback", ImGuiTreeNodeFlags_DefaultOpen))
            {
                bool anyPlaying = false;
                for (int i = 0; i < (int)animations.size(); ++i)
                {
                    std::string key = "render/animation/anim" + std::to_string(i) + "/state";
                    anyPlaying |= m_settingsManager->getAs<bool>(key.c_str());
                }

                constexpr float frameDuration = 1.0f / 24.0f;

                if (ImGui::Button("|<"))
                {
                    for (int i = 0; i < (int)animations.size(); ++i)
                    {
                        std::string key = "render/animation/anim" + std::to_string(i) + "/time";
                        m_settingsManager->setAs<float>(key.c_str(), animations[i].start);
                    }
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset to start");

                ImGui::SameLine();

                if (ImGui::Button("<"))
                {
                    for (int i = 0; i < (int)animations.size(); ++i)
                    {
                        std::string timeKey = "render/animation/anim" + std::to_string(i) + "/time";
                        float t = m_settingsManager->getAs<float>(timeKey.c_str());
                        t = std::max(t - frameDuration, animations[i].start);
                        m_settingsManager->setAs<float>(timeKey.c_str(), t);
                    }
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Previous frame (1/24s)");

                ImGui::SameLine();

                if (anyPlaying)
                {
                    if (ImGui::Button("Pause"))
                    {
                        for (int i = 0; i < (int)animations.size(); ++i)
                        {
                            std::string key = "render/animation/anim" + std::to_string(i) + "/state";
                            m_settingsManager->setAs<bool>(key.c_str(), false);
                        }
                    }
                }
                else
                {
                    if (ImGui::Button(" Play"))
                    {
                        for (int i = 0; i < (int)animations.size(); ++i)
                        {
                            std::string key = "render/animation/anim" + std::to_string(i) + "/state";
                            m_settingsManager->setAs<bool>(key.c_str(), true);
                        }
                    }
                }

                ImGui::SameLine();

                if (ImGui::Button(">"))
                {
                    for (int i = 0; i < (int)animations.size(); ++i)
                    {
                        std::string timeKey = "render/animation/anim" + std::to_string(i) + "/time";
                        float t = m_settingsManager->getAs<float>(timeKey.c_str());
                        t = std::min(t + frameDuration, animations[i].end);
                        m_settingsManager->setAs<float>(timeKey.c_str(), t);
                    }
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next frame (1/24s)");

                ImGui::SameLine();

                if (ImGui::Button(">|"))
                {
                    for (int i = 0; i < (int)animations.size(); ++i)
                    {
                        std::string key = "render/animation/anim" + std::to_string(i) + "/time";
                        m_settingsManager->setAs<float>(key.c_str(), animations[i].end);
                    }
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Jump to end");

                float speed = m_settingsManager->getAs<float>("render/animation/speed");
                ImGui::SliderFloat("Speed", &speed, 0.0f, 5.0f, "%.2fx");
                m_settingsManager->setAs<float>("render/animation/speed", speed);
            }

            // Motion blur
            if (ImGui::CollapsingHeader("Motion Blur"))
            {
                bool enableMotionBlur = m_settingsManager->getAs<bool>("render/enableMotionBlur");
                if (ImGui::Checkbox("Enable Motion Blur", &enableMotionBlur))
                {
                    m_settingsManager->setAs<bool>("render/enableMotionBlur", enableMotionBlur);
                }
                if (enableMotionBlur)
                {
                    bool isMotionBlurVisible = m_settingsManager->getAs<bool>("render/isMotionBlurVisible");
                    if (ImGui::Checkbox("Show Motion Blur Effect", &isMotionBlurVisible))
                    {
                        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", isMotionBlurVisible);
                    }

                    bool enableCameraMotionBlur = m_settingsManager->getAs<bool>("render/enableCameraMotionBlur");
                    if (ImGui::Checkbox("Camera Motion Blur", &enableCameraMotionBlur))
                    {
                        m_settingsManager->setAs<bool>("render/enableCameraMotionBlur", enableCameraMotionBlur);
                    }

                    float shutterTime = m_settingsManager->getAs<float>("render/motionBlur/shutterTime");
                    if (ImGui::SliderFloat("Shutter Duration (s)", &shutterTime, 0.001f, 1.0f, "%.3f"))
                    {
                        m_settingsManager->setAs<float>("render/motionBlur/shutterTime", shutterTime);
                    }

                    const char* shutterModes[] = { "Centered", "Leading", "Trailing" };
                    int shutterMode = (int)m_settingsManager->getAs<uint32_t>("render/motionBlur/shutterMode");
                    if (ImGui::Combo("Shutter Mode", &shutterMode, shutterModes, IM_ARRAYSIZE(shutterModes)))
                    {
                        m_settingsManager->setAs<uint32_t>("render/motionBlur/shutterMode", (uint32_t)shutterMode);
                    }
                }
            }

            // Per-animation controls
            if (ImGui::CollapsingHeader("Clips", ImGuiTreeNodeFlags_DefaultOpen))
            {
                for (int i = 0; i < (int)animations.size(); ++i)
                {
                    ImGui::PushID(i);
                    std::string checkboxNameStr = "render/animation/anim" + std::to_string(i) + "/state";
                    std::string scrollNameStr = "render/animation/anim" + std::to_string(i) + "/time";

                    bool currAnimEnable = m_settingsManager->getAs<bool>(checkboxNameStr.c_str());
                    ImGui::Checkbox(animations[i].name.c_str(), &currAnimEnable);
                    m_settingsManager->setAs<bool>(checkboxNameStr.c_str(), currAnimEnable);

                    float currAnimTime = m_settingsManager->getAs<float>(scrollNameStr.c_str());
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                    ImGui::SliderFloat("##time", &currAnimTime, animations[i].start, animations[i].end, "%.3f s");
                    m_settingsManager->setAs<float>(scrollNameStr.c_str(), currAnimTime);

                    ImGui::PopID();
                }
            }
        }
        else
        {
            ImGui::TextDisabled("No animations in scene");
        }

        ImGui::End();
    }
}

} // namespace oka
