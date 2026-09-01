#include "../EditorApp.h"

#include "imgui.h"

namespace oka
{

void EditorApp::drawGamepadSettings()
{
    const GamepadState& pad = m_display->getGamepadState();

    if (!ImGui::TreeNode("Gamepad"))
    {
        return;
    }

    if (pad.connected)
    {
        ImGui::TextUnformatted(pad.name.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("(slot %d)", pad.slot);
    }
    else
    {
        ImGui::TextDisabled("No controller detected");
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip(
                "A controller is detected automatically. If one is connected and\n"
                "this still says no, the platform has no mapping for its axes.");
        }
    }

    bool enabled = m_settingsManager->getAs<bool>("editor/gamepad/enabled");
    if (ImGui::Checkbox("Enabled", &enabled))
    {
        m_settingsManager->setAs<bool>("editor/gamepad/enabled", enabled);
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Disable a drifting controller without unplugging it.");
    }

    bool invert = m_settingsManager->getAs<bool>("editor/gamepad/invertLookY");
    if (ImGui::Checkbox("Invert look Y", &invert))
    {
        m_settingsManager->setAs<bool>("editor/gamepad/invertLookY", invert);
    }

    float lookSpeed = m_settingsManager->getAs<float>("editor/gamepad/lookSpeed");
    if (ImGui::SliderFloat("Look speed", &lookSpeed, 100.0f, 3000.0f, "%.0f"))
    {
        m_settingsManager->setAs<float>("editor/gamepad/lookSpeed", lookSpeed);
    }

    float deadzone = m_settingsManager->getAs<float>("editor/gamepad/deadzone");
    if (ImGui::SliderFloat("Deadzone", &deadzone, 0.0f, 0.5f, "%.3f"))
    {
        m_settingsManager->setAs<float>("editor/gamepad/deadzone", deadzone);
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Increase this if the camera drifts while the sticks are at rest.");
    }

    if (pad.connected && ImGui::TreeNode("Live input"))
    {
        ImGui::Text("Left  stick  %+.3f %+.3f", pad.leftX, pad.leftY);
        ImGui::Text("Right stick  %+.3f %+.3f", pad.rightX, pad.rightY);
        ImGui::Text("Triggers     L2 %.3f  R2 %.3f", pad.leftTrigger, pad.rightTrigger);
        ImGui::Text("Speed scale  x%.2f",
                    gamepad::speedScaleFromTriggers(pad.leftTrigger, pad.rightTrigger, gamepad::Config{}));

        struct Named
        {
            oka::GamepadState::Button button;
            const char* label;
        };
        static const Named kButtons[] = {
            { oka::GamepadState::a, "Cross" },       { oka::GamepadState::b, "Circle" },
            { oka::GamepadState::x, "Square" },      { oka::GamepadState::y, "Triangle" },
            { oka::GamepadState::leftBumper, "L1" }, { oka::GamepadState::rightBumper, "R1" },
            { oka::GamepadState::leftThumb, "L3" },  { oka::GamepadState::rightThumb, "R3" },
            { oka::GamepadState::dpadUp, "Up" },     { oka::GamepadState::dpadDown, "Down" },
            { oka::GamepadState::dpadLeft, "Left" }, { oka::GamepadState::dpadRight, "Right" },
            { oka::GamepadState::back, "Share" },    { oka::GamepadState::start, "Options" },
            { oka::GamepadState::guide, "PS" },
        };
        int column = 0;
        for (const Named& entry : kButtons)
        {
            if (column++ % 4 != 0)
            {
                ImGui::SameLine(static_cast<float>(column % 4) * 90.0f);
            }
            if (pad.pressed(entry.button))
            {
                ImGui::TextUnformatted(entry.label);
            }
            else
            {
                ImGui::TextDisabled("%s", entry.label);
            }
        }
        ImGui::TreePop();
    }

    ImGui::Separator();
    ImGui::TextDisabled("Left stick   move        Right stick  look");
    ImGui::TextDisabled("L1 / R1      down / up   L2 / R2      slower / faster");
    ImGui::TextDisabled("D-pad        navigate the UI");

    ImGui::TreePop();
}

} // namespace oka
