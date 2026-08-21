#pragma once

#include <strelka/display/gamepad.h>
#include <strelka/scene/glm_wrapper.hpp>

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>

namespace oka::gamepad
{

/// Turning a gamepad's sticks into the same units CameraController already
/// accepts from the mouse and the movement keys.
///
/// Pure, and separate from the polling, for the reason the rest of the renderer
/// splits things this way: the interesting part is the deadzone and the response
/// curve, both of which are tuning nobody can check by looking at it, and
/// neither needs a window, a GPU or a controller to test.

struct Config
{
    /// Below this the stick is treated as centred.
    ///
    /// Measured, not chosen: a DualSense at rest on a desk reports up to 0.043
    /// on the right stick's Y axis and 0.027 on its X. Anything at or under that
    /// makes the camera drift on its own, which reads as the renderer being
    /// broken rather than as the pad needing a deadzone -- and it never settles,
    /// so accumulation restarts every frame and the image never converges. 0.12
    /// clears the measured drift with room for a pad that has seen more use.
    float deadzone = 0.12f;

    /// Applied to the deadzone-corrected magnitude. Above 1 it buys fine control
    /// near centre at the cost of the top of the range; 2.0 is the usual choice
    /// for a look stick and it matters more here than in a game, because the
    /// thing being aimed is a camera somebody is trying to park.
    float responseExponent = 2.0f;

    /// Degrees of look per second at full stick deflection. In the same units
    /// CameraController::mPendingLookX takes, which are raw mouse pixels scaled
    /// by Camera::rotationSpeed -- so this is "pixels per second of equivalent
    /// mouse travel", and 900 is a comfortable ~180 degrees/s at the default
    /// rotation speed of 0.025.
    float lookSpeed = 900.0f;

    /// Multiplied into the editor's own camera speed setting, so the gamepad
    /// stays consistent with the keyboard rather than needing its own number.
    float moveSpeed = 1.0f;

    /// What the analogue triggers do to the speed. R2 fully pressed multiplies
    /// by boost, L2 fully pressed multiplies by brake, and both together cancel
    /// -- which is a state a hand can reach by accident, so it is defined rather
    /// than left to whichever term is applied last.
    float triggerBoost = 5.0f;
    float triggerBrake = 0.2f;

    bool invertLookY = false;
};

/// What one frame of gamepad input asks of the camera. Deltas, not state: this
/// is added to what the mouse and keyboard already queued, so a user with a hand
/// on each gets the sum rather than one overwriting the other.
struct CameraInput
{
    /// Camera-space translation for this frame, in world units.
    ///
    /// Camera::translate's frame, which is *not* the frame Camera::mMoveInput
    /// uses: translate() adds `conjugate(orientation) * delta` to the position
    /// and getFront() is `conjugate(orientation) * (0,0,-1)`, so **forward is
    /// -z here** while the movement keys spell forward as mMoveInput.z = +1.
    /// The two conventions sit four lines apart in camera.cpp and swapping them
    /// gives a camera that flies backwards, which is worth writing down once.
    glm::float3 translate{ 0.0f };

    /// Lift along *world* up, kept out of `translate` because that is what the
    /// keys it mirrors do: Camera::update moves Q/E along getWorldUp(), so a
    /// shoulder button folded into the camera-space y would climb at an angle
    /// whenever the camera was pitched, and only then -- which is exactly the
    /// kind of difference that gets reported as "the pad feels wrong" without
    /// anyone being able to say how.
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

/// One axis, deadzoned and curved. Returns 0 inside the deadzone and reaches
/// +/-1 at full deflection, so the usable range is not shortened by the
/// deadzone -- a stick that only reached 0.88 after a 0.12 cut is the classic
/// way a pad ends up feeling weaker than the keyboard.
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

/// The same, for a stick taken as a whole.
///
/// Radial rather than per-axis: a per-axis deadzone leaves a cross-shaped dead
/// region, so a stick pushed diagonally at low force moves on one axis only and
/// the camera crabs sideways when the user asked for a diagonal. The magnitude
/// is also clamped to 1 before shaping, because the hardware reports slightly
/// past 1.0 in the corners and an unclamped diagonal is faster than a cardinal.
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

/// The trigger pair as one speed multiplier.
///
/// Geometric rather than additive, so brake and boost compose the way a hand
/// expects: both fully pressed lands on 1.0 (boost * brake would not), and
/// either alone reaches its own extreme.
inline float speedScaleFromTriggers(float leftTrigger, float rightTrigger, const Config& cfg)
{
    const float boost = std::pow(cfg.triggerBoost, std::clamp(rightTrigger, 0.0f, 1.0f));
    const float brake = std::pow(cfg.triggerBrake, std::clamp(leftTrigger, 0.0f, 1.0f));
    return boost * brake;
}

/// Whether the camera may read the pad this frame, or whether the UI has it.
///
/// One controller, two consumers, and the division is Dear ImGui's own rather
/// than something invented here. From its key table:
///
///   * D-pad          -- navigate and tweak widgets
///   * Cross / Circle -- activate / cancel
///   * Square         -- "Toggle Menu. Hold for Windowing mode"
///   * L/R sticks     -- "[Analog] Move Window (in Windowing mode)"
///   * L1 / R1        -- "Tweak Slower / Focus Previous (in Windowing mode)"
///
/// So ImGui wants the sticks and the shoulders exactly while windowing mode is
/// held, and nothing else the camera uses at any other time. The camera yields
/// then, and while a text field is taking input, and owns the pad otherwise --
/// which is the state the editor is in essentially all of the time.
///
/// Deliberately *not* gated on io.NavActive. That reads "a window is focused and
/// does not opt out of nav", which in a docked editor is true from the first
/// frame onward; gating on it would mean the pad never flew the camera at all
/// and the whole feature would look like it had failed to detect the hardware.
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

    const float scale = cfg.moveSpeed * out.speedScale * dt;
    if (glm::dot(out.translate, out.translate) > 0.0f || out.worldUp != 0.0f)
    {
        out.translate *= scale;
        out.worldUp *= scale;
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
