#pragma once

#include <strelka/display/gamepad.h>

#include <GLFW/glfw3.h>

#include <log.h>

namespace oka::glfw_gamepad
{

inline int findGamepadSlot()
{
    for (int slot = GLFW_JOYSTICK_1; slot <= GLFW_JOYSTICK_LAST; ++slot)
    {
        if (glfwJoystickPresent(slot) == GLFW_TRUE && glfwJoystickIsGamepad(slot) == GLFW_TRUE)
        {
            return slot;
        }
    }
    return -1;
}

inline bool read(int slot, GamepadState& out)
{
    GLFWgamepadstate raw{};
    if (slot < 0 || glfwGetGamepadState(slot, &raw) != GLFW_TRUE)
    {
        out = GamepadState{};
        return false;
    }

    out.connected = true;
    out.slot = slot;
    const char* name = glfwGetGamepadName(slot);
    out.name = (name != nullptr) ? name : "gamepad";

    out.leftX = raw.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
    out.leftY = raw.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
    out.rightX = raw.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
    out.rightY = raw.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];
    // GLFW reports triggers on [-1, 1], released at -1. Measured on a DualSense:
    // both read exactly -1.000 at rest. GamepadState promises [0, 1].
    out.leftTrigger = (raw.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] + 1.0f) * 0.5f;
    out.rightTrigger = (raw.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.0f) * 0.5f;

    static_assert(static_cast<int>(GamepadState::count) == GLFW_GAMEPAD_BUTTON_LAST + 1,
                  "GamepadState::Button mirrors the SDL mapping order GLFW uses");
    for (int i = 0; i < static_cast<int>(GamepadState::count); ++i)
    {
        out.buttons[i] = raw.buttons[i] == GLFW_PRESS;
    }
    return true;
}

inline bool poll(GamepadState& state)
{
    const bool wasConnected = state.connected;
    const std::string previousName = state.name;

    int slot = state.connected ? state.slot : -1;
    if (slot < 0 || glfwJoystickPresent(slot) != GLFW_TRUE || glfwJoystickIsGamepad(slot) != GLFW_TRUE)
    {
        slot = findGamepadSlot();
    }

    const bool nowConnected = read(slot, state);

    if (nowConnected && !wasConnected)
    {
        STRELKA_INFO("ACTION gamepad_connected slot={} name=\"{}\"", state.slot, state.name);
    }
    else if (!nowConnected && wasConnected)
    {
        STRELKA_INFO("ACTION gamepad_disconnected name=\"{}\"", previousName);
    }
    return nowConnected;
}

} // namespace oka::glfw_gamepad
