#include "../EditorApp.h"

#include "imgui.h"

namespace oka
{

void EditorApp::drawAnimationPanel()
{
    // ImGui::End() must be called for every Begin(), including the collapsed /
    // clipped case where Begin() returns false — otherwise the window stack is
    // left unbalanced and ImGui asserts on the next frame. Collapsing the
    // Animations panel used to be enough to trip this.
    const bool visible = ImGui::Begin("Animations");
    if (visible)
    {
        const auto& animations = m_scene->getAnimations();

        if (!animations.empty())
        {
            // Playback controls
            if (ImGui::CollapsingHeader("Playback", ImGuiTreeNodeFlags_DefaultOpen))
            {
                bool anyPlaying = false;
                for (size_t i = 0; i < animations.size(); ++i)
                {
                    anyPlaying |= m_settingsManager->getAs<bool>(animationStateKey(i));
                }

                constexpr float frameDuration = 1.0f / 24.0f;

                if (ImGui::Button("|<"))
                {
                    for (size_t i = 0; i < animations.size(); ++i)
                    {
                        m_settingsManager->setAs<float>(animationTimeKey(i), animations[i].start);
                    }
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset to start");

                ImGui::SameLine();

                if (ImGui::Button("<"))
                {
                    for (size_t i = 0; i < animations.size(); ++i)
                    {
                        const std::string timeKey = animationTimeKey(i);
                        const float t = m_settingsManager->getAs<float>(timeKey);
                        m_settingsManager->setAs<float>(timeKey, std::max(t - frameDuration, animations[i].start));
                    }
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Previous frame (1/24s)");

                ImGui::SameLine();

                if (anyPlaying)
                {
                    if (ImGui::Button("Pause"))
                    {
                        for (size_t i = 0; i < animations.size(); ++i)
                        {
                            m_settingsManager->setAs<bool>(animationStateKey(i), false);
                        }
                    }
                }
                else
                {
                    if (ImGui::Button(" Play"))
                    {
                        for (size_t i = 0; i < animations.size(); ++i)
                        {
                            m_settingsManager->setAs<bool>(animationStateKey(i), true);
                        }
                    }
                }

                ImGui::SameLine();

                if (ImGui::Button(">"))
                {
                    for (size_t i = 0; i < animations.size(); ++i)
                    {
                        const std::string timeKey = animationTimeKey(i);
                        const float t = m_settingsManager->getAs<float>(timeKey);
                        m_settingsManager->setAs<float>(timeKey, std::min(t + frameDuration, animations[i].end));
                    }
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next frame (1/24s)");

                ImGui::SameLine();

                if (ImGui::Button(">|"))
                {
                    for (size_t i = 0; i < animations.size(); ++i)
                    {
                        m_settingsManager->setAs<float>(animationTimeKey(i), animations[i].end);
                    }
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Jump to end");

                float speed = m_settingsManager->getAs<float>("render/animation/speed");
                ImGui::SliderFloat("Speed", &speed, 0.0f, 5.0f, "%.2fx");
                m_settingsManager->setAs<float>("render/animation/speed", speed);

                // Pausing partway through a clip with motion blur on leaves a
                // motion-blurred still that keeps accumulating samples. That takes
                // a while to converge, so say so -- otherwise the slow clean-up of
                // the frozen frame reads as the render having stalled.
                if (!anyPlaying && m_settingsManager->getAs<bool>("render/enableMotionBlur") &&
                    m_settingsManager->getAs<bool>("render/isMotionBlurVisible"))
                {
                    bool midClip = false;
                    for (size_t i = 0; i < animations.size(); ++i)
                    {
                        const float t = m_settingsManager->getAs<float>(animationTimeKey(i));
                        if (t > animations[i].start + 1e-4f && t < animations[i].end - 1e-4f)
                        {
                            midClip = true;
                            break;
                        }
                    }
                    if (midClip)
                    {
                        ImGui::TextDisabled("Paused: refining motion-blurred frame (%u samples)",
                                            (uint32_t)m_sharedCtx->mSubframeIndex);
                    }
                }
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
                for (size_t i = 0; i < animations.size(); ++i)
                {
                    ImGui::PushID((int)i);

                    const std::string stateKey = animationStateKey(i);
                    bool currAnimEnable = m_settingsManager->getAs<bool>(stateKey);
                    ImGui::Checkbox(animations[i].name.c_str(), &currAnimEnable);
                    m_settingsManager->setAs<bool>(stateKey, currAnimEnable);

                    const std::string timeKey = animationTimeKey(i);
                    float currAnimTime = m_settingsManager->getAs<float>(timeKey);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                    ImGui::SliderFloat("##time", &currAnimTime, animations[i].start, animations[i].end, "%.3f s");
                    m_settingsManager->setAs<float>(timeKey, currAnimTime);

                    ImGui::PopID();
                }
            }
        }
        else
        {
            ImGui::TextDisabled("No animations in scene");
        }
    }
    ImGui::End();
}

} // namespace oka
