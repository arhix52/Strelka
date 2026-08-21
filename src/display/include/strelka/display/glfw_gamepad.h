#pragma once

#include <strelka/display/gamepad.h>

#include <GLFW/glfw3.h>

#include <log.h>

namespace oka::glfw_gamepad
{

/// Finding and reading a gamepad through GLFW, once, for both windowing
/// backends. Metal and Vulkan each own their own GLFWwindow but the joystick API
/// is per-process and window-independent, so a copy per backend would be two
/// answers to one question -- and the copy without the slot-scan below is the
/// one that would look correct and never see the pad.

/// The first connected joystick GLFW has a game controller mapping for, or -1.
///
/// A scan, not GLFW_JOYSTICK_1: joystick slots are handed out to anything the
/// platform exposes as one, and on a desk with a Keychron K8 Pro the keyboard's
/// system-control HID collection takes slot 0 while the DualSense lands in slot
/// 1. Reading slot 0 finds a device with no mapping, no axes worth the name and
/// no way to tell that something went wrong.
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

/// Read `slot` into `out`, normalising as GamepadState documents.
///
/// Returns false and leaves `out` disconnected if the pad went away between the
/// scan and the read, which is a frame that happens every time somebody pulls
/// the cable.
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

/// Rescan when the pad we were using goes away, and log the transitions.
///
/// Held by the display rather than by the editor so that a headless or
/// non-GLFW display simply never calls it. `previous` is the state from last
/// frame, used only to decide whether anything is worth saying.
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
