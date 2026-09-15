#pragma once

#include <strelka/display/gamepad.h>
#include <strelka/scene/glm_wrapper.hpp>

#include <glm/geometric.hpp>

#include "imgui.h"

#include <algorithm>
#include <cmath>

namespace oka::gamepad
{

struct Config
{
    float deadzone = 0.12f;

    float responseExponent = 2.0f;

    float lookSpeed = 900.0f;

    /// Multiplied into the editor's own camera speed setting, so the gamepad
    /// stays consistent with the keyboard rather than needing its own number.
    float moveSpeed = 1.0f;

    float triggerBoost = 5.0f;
    float triggerBrake = 0.2f;

    bool invertLookY = false;
};

/// What one frame of gamepad input asks of the camera. Deltas, not state: this
/// is added to what the mouse and keyboard already queued, so a user with a hand
/// on each gets the sum rather than one overwriting the other.
struct CameraInput
{
    glm::float3 translate{ 0.0f };

    float zoom = 0.0f;

    float worldUp = 0.0f;

    /// Look delta in raw-mouse-travel units, matching mPendingLookX/Y.
    float lookX = 0.0f;
    float lookY = 0.0f;

    /// Speed multiplier the triggers asked for, already folded into `translate`.
    /// Reported so the UI can show it.
    float speedScale = 1.0f;

    /// Whether the user actually moved anything. Drives camera takeover and
    /// accumulation restart, so it must be false for a pad merely being plugged
    /// in and lying still -- otherwise the render never converges.
    bool active = false;
};

inline float shapeAxis(float raw, const Config& cfg)
{
    const float magnitude = std::fabs(raw);
    if (magnitude <= cfg.deadzone)
    {
        return 0.0f;
    }
    const float span = 1.0f - cfg.deadzone;
    const float rescaled = std::min((magnitude - cfg.deadzone) / (span > 0.0f ? span : 1.0f), 1.0f);
    const float curved = std::pow(rescaled, cfg.responseExponent);
    return (raw < 0.0f) ? -curved : curved;
}

struct Stick
{
    float x = 0.0f;
    float y = 0.0f;
};

inline Stick shapeStick(float rawX, float rawY, const Config& cfg)
{
    const float magnitude = std::sqrt(rawX * rawX + rawY * rawY);
    if (magnitude <= cfg.deadzone || magnitude <= 0.0f)
    {
        return {};
    }
    const float clamped = std::min(magnitude, 1.0f);
    const float span = 1.0f - cfg.deadzone;
    const float rescaled = std::min((clamped - cfg.deadzone) / (span > 0.0f ? span : 1.0f), 1.0f);
    const float curved = std::pow(rescaled, cfg.responseExponent);
    // Direction from the *raw* vector, magnitude from the curve: shaping the
    // components separately would rotate the direction the user pushed.
    return { rawX / magnitude * curved, rawY / magnitude * curved };
}

inline float speedScaleFromTriggers(float leftTrigger, float rightTrigger, const Config& cfg)
{
    const float boost = std::pow(cfg.triggerBoost, std::clamp(rightTrigger, 0.0f, 1.0f));
    const float brake = std::pow(cfg.triggerBrake, std::clamp(leftTrigger, 0.0f, 1.0f));
    return boost * brake;
}

template <typename ImGuiIoT>
void feedImGui(const GamepadState& pad, ImGuiIoT& io)
{
    if (!pad.connected)
    {
        return;
    }
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;

    const auto button = [&io](ImGuiKey key, bool down) { io.AddKeyEvent(key, down); };
    // ImGui maps an axis onto [v0, v1] and calls it pressed past 0.10.
    const auto analog = [&io](ImGuiKey key, float raw, float v0, float v1) {
        const float v = std::clamp((raw - v0) / (v1 - v0), 0.0f, 1.0f);
        io.AddKeyAnalogEvent(key, v > 0.10f, v);
    };

    button(ImGuiKey_GamepadStart, pad.pressed(GamepadState::start));
    button(ImGuiKey_GamepadBack, pad.pressed(GamepadState::back));
    button(ImGuiKey_GamepadFaceLeft, pad.pressed(GamepadState::x)); // Square
    button(ImGuiKey_GamepadFaceRight, pad.pressed(GamepadState::b)); // Circle
    button(ImGuiKey_GamepadFaceUp, pad.pressed(GamepadState::y)); // Triangle
    button(ImGuiKey_GamepadFaceDown, pad.pressed(GamepadState::a)); // Cross
    button(ImGuiKey_GamepadDpadLeft, pad.pressed(GamepadState::dpadLeft));
    button(ImGuiKey_GamepadDpadRight, pad.pressed(GamepadState::dpadRight));
    button(ImGuiKey_GamepadDpadUp, pad.pressed(GamepadState::dpadUp));
    button(ImGuiKey_GamepadDpadDown, pad.pressed(GamepadState::dpadDown));
    button(ImGuiKey_GamepadL1, pad.pressed(GamepadState::leftBumper));
    button(ImGuiKey_GamepadR1, pad.pressed(GamepadState::rightBumper));
    button(ImGuiKey_GamepadL3, pad.pressed(GamepadState::leftThumb));
    button(ImGuiKey_GamepadR3, pad.pressed(GamepadState::rightThumb));

    // GamepadState already normalised the triggers to [0, 1]; ImGui's own
    // version maps GLFW's raw [-0.75, 1] because it reads them unconverted.
    analog(ImGuiKey_GamepadL2, pad.leftTrigger, 0.125f, 1.0f);
    analog(ImGuiKey_GamepadR2, pad.rightTrigger, 0.125f, 1.0f);

    analog(ImGuiKey_GamepadLStickLeft, pad.leftX, -0.25f, -1.0f);
    analog(ImGuiKey_GamepadLStickRight, pad.leftX, 0.25f, 1.0f);
    analog(ImGuiKey_GamepadLStickUp, pad.leftY, -0.25f, -1.0f);
    analog(ImGuiKey_GamepadLStickDown, pad.leftY, 0.25f, 1.0f);
    analog(ImGuiKey_GamepadRStickLeft, pad.rightX, -0.25f, -1.0f);
    analog(ImGuiKey_GamepadRStickRight, pad.rightX, 0.25f, 1.0f);
    analog(ImGuiKey_GamepadRStickUp, pad.rightY, -0.25f, -1.0f);
    analog(ImGuiKey_GamepadRStickDown, pad.rightY, 0.25f, 1.0f);
}

inline bool cameraOwnsPad(const GamepadState& pad, bool imguiWantsTextInput)
{
    return pad.connected && !imguiWantsTextInput && !pad.pressed(GamepadState::x);
}

/// Map a frame of pad state to camera input. `dt` is the frame's wall clock in
/// seconds, already clamped by the caller.
inline CameraInput mapToCamera(const GamepadState& pad, const Config& cfg, float dt)
{
    CameraInput out;
    if (!pad.connected || dt <= 0.0f)
    {
        return out;
    }

    out.speedScale = speedScaleFromTriggers(pad.leftTrigger, pad.rightTrigger, cfg);

    const Stick move = shapeStick(pad.leftX, pad.leftY, cfg);
    const Stick look = shapeStick(pad.rightX, pad.rightY, cfg);

    // Left stick. GLFW reports +Y as *down*, so pushing the stick away from you
    // is a negative y -- and forward is a negative z in this frame, so the two
    // sign conventions cancel and the axis is carried straight across.
    out.translate.x = move.x;
    out.translate.z = move.y;

    // Shoulders lift and lower, matching Q/E on the keyboard. Both held is a
    // deliberate zero rather than whichever branch ran last.
    const float up = pad.pressed(GamepadState::rightBumper) ? 1.0f : 0.0f;
    const float down = pad.pressed(GamepadState::leftBumper) ? 1.0f : 0.0f;
    out.worldUp = up - down;

    // Per-axis, not from `move`: see CameraInput::zoom.
    out.zoom = shapeAxis(pad.leftY, cfg);

    const float scale = cfg.moveSpeed * out.speedScale * dt;
    if (glm::dot(out.translate, out.translate) > 0.0f || out.worldUp != 0.0f || out.zoom != 0.0f)
    {
        out.translate *= scale;
        out.worldUp *= scale;
        out.zoom *= scale;
        out.active = true;
    }

    if (look.x != 0.0f || look.y != 0.0f)
    {
        out.lookX = look.x * cfg.lookSpeed * dt;
        out.lookY = look.y * cfg.lookSpeed * dt * (cfg.invertLookY ? -1.0f : 1.0f);
        out.active = true;
    }

    return out;
}

} // namespace oka::gamepad
