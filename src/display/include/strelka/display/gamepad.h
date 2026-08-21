#pragma once

#include <cstdint>
#include <string>

namespace oka
{

/// One gamepad, as the platform reports it, in units the rest of the editor can
/// use without knowing which platform that was.
///
/// Normalised on the way in rather than at every reader: GLFW hands back
/// triggers on [-1, 1] with -1 meaning "released", which is neither what a
/// caller expects nor what the axis is called, and getting that wrong reads as a
/// gamepad that is permanently on the brake. Triggers here are [0, 1], 0
/// released. Sticks stay [-1, 1] with +Y down, which is the sign convention the
/// mouse delta already uses in CameraController.
struct GamepadState
{
    /// Buttons, in the order the SDL game controller mapping defines them --
    /// which is what GLFW's GLFW_GAMEPAD_BUTTON_* are, so the two agree by
    /// construction. Named for the Xbox layout because that is what the mapping
    /// database is written in; the PlayStation face buttons land as
    /// cross=a, circle=b, square=x, triangle=y.
    enum Button : uint8_t
    {
        a = 0,
        b,
        x,
        y,
        leftBumper,
        rightBumper,
        back,
        start,
        guide,
        leftThumb,
        rightThumb,
        dpadUp,
        dpadRight,
        dpadDown,
        dpadLeft,
        count
    };

    /// False means every other field is stale and must not be acted on.
    bool connected = false;

    /// What the mapping database calls this pad ("Sony DualSense" for a PS5
    /// controller). Shown in the UI and logged on connect, so that a pad which
    /// binds oddly can be identified from a log rather than guessed at.
    std::string name;

    /// The platform's index for the pad. Kept because it is the only thing that
    /// distinguishes two identical controllers, and because it is *not*
    /// necessarily 0: on a machine with a keyboard that reports a system-control
    /// HID collection, slot 0 is the keyboard and the pad is slot 1.
    int slot = -1;

    float leftX = 0.0f;
    float leftY = 0.0f;
    float rightX = 0.0f;
    float rightY = 0.0f;
    /// [0, 1], 0 released. See the note on the struct.
    float leftTrigger = 0.0f;
    float rightTrigger = 0.0f;

    bool buttons[Button::count] = {};

    bool pressed(Button which) const
    {
        return connected && buttons[which];
    }
};

} // namespace oka
