#pragma once

#include <cstdint>
#include <string>

namespace oka
{

struct GamepadState
{
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
