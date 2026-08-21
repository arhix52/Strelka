#include <doctest/doctest.h>

#include "gamepad_camera.h"

#include <cmath>

using oka::GamepadState;
namespace gp = oka::gamepad;

namespace
{

/// A pad reporting exactly what a DualSense reports lying still on a desk,
/// measured through glfwGetGamepadState with GLFW 3.5.1's built-in mapping.
/// These are the numbers the deadzone default exists to clear.
GamepadState restingDualSense()
{
    GamepadState pad;
    pad.connected = true;
    pad.slot = 1;
    pad.name = "Sony DualSense";
    pad.leftX = 0.004f;
    pad.leftY = 0.012f;
    pad.rightX = 0.027f;
    pad.rightY = 0.043f;
    // GLFW's -1 at rest, after glfw_gamepad::read normalises it.
    pad.leftTrigger = 0.0f;
    pad.rightTrigger = 0.0f;
    return pad;
}

constexpr float kDt = 1.0f / 60.0f;

} // namespace

TEST_CASE("a pad lying still asks for nothing")
{
    // The whole point of the deadzone. A pad that reports its rest drift as
    // camera motion keeps mUserMovedCamera true forever, which restarts
    // accumulation every frame -- the image never converges and the cause looks
    // like the renderer rather than like an idle controller.
    const gp::CameraInput out = gp::mapToCamera(restingDualSense(), gp::Config{}, kDt);

    CHECK(out.active == false);
    CHECK(out.lookX == 0.0f);
    CHECK(out.lookY == 0.0f);
    CHECK(out.translate.x == 0.0f);
    CHECK(out.translate.y == 0.0f);
    CHECK(out.translate.z == 0.0f);
    CHECK(out.worldUp == 0.0f);
}

TEST_CASE("a disconnected pad asks for nothing whatever its axes say")
{
    GamepadState pad = restingDualSense();
    pad.connected = false;
    pad.leftY = -1.0f;
    pad.rightX = 1.0f;

    const gp::CameraInput out = gp::mapToCamera(pad, gp::Config{}, kDt);
    CHECK(out.active == false);
    CHECK(out.translate.z == 0.0f);
    CHECK(out.lookX == 0.0f);
}

TEST_CASE("pushing the left stick away from the user flies forward")
{
    // Two sign conventions meet here and they are four lines apart in
    // camera.cpp: GLFW reports stick +Y as down, and Camera::translate's forward
    // is -z (it applies conjugate(orientation), and getFront() is
    // conjugate(orientation) * (0,0,-1)). Getting either wrong flies backwards.
    GamepadState pad = restingDualSense();
    pad.leftY = -1.0f; // stick pushed away

    const gp::CameraInput out = gp::mapToCamera(pad, gp::Config{}, kDt);
    CHECK(out.active == true);
    CHECK(out.translate.z < 0.0f);

    pad.leftY = 1.0f; // pulled back
    CHECK(gp::mapToCamera(pad, gp::Config{}, kDt).translate.z > 0.0f);
}

TEST_CASE("pushing the left stick right strafes right")
{
    GamepadState pad = restingDualSense();
    pad.leftX = 1.0f;
    CHECK(gp::mapToCamera(pad, gp::Config{}, kDt).translate.x > 0.0f);
    pad.leftX = -1.0f;
    CHECK(gp::mapToCamera(pad, gp::Config{}, kDt).translate.x < 0.0f);
}

TEST_CASE("the right stick looks in the same direction the mouse does")
{
    // CameraController queues mouse look as mPendingLookX += -(oldX - newX), so
    // moving the mouse right is a positive lookX and moving it down a positive
    // lookY. A pad that disagreed would invert relative to the mouse for a user
    // holding both, which is the failure this pins.
    GamepadState pad = restingDualSense();
    pad.rightX = 1.0f;
    CHECK(gp::mapToCamera(pad, gp::Config{}, kDt).lookX > 0.0f);

    pad = restingDualSense();
    pad.rightY = 1.0f; // stick pulled toward the user; GLFW +Y is down
    CHECK(gp::mapToCamera(pad, gp::Config{}, kDt).lookY > 0.0f);
}

TEST_CASE("invertLookY flips only the vertical")
{
    GamepadState pad = restingDualSense();
    pad.rightX = 0.8f;
    pad.rightY = 0.8f;

    gp::Config inverted;
    inverted.invertLookY = true;

    const gp::CameraInput plain = gp::mapToCamera(pad, gp::Config{}, kDt);
    const gp::CameraInput flipped = gp::mapToCamera(pad, inverted, kDt);

    CHECK(plain.lookX == doctest::Approx(flipped.lookX));
    CHECK(plain.lookY == doctest::Approx(-flipped.lookY));
}

TEST_CASE("the shoulder buttons lift and lower, and cancel when both are held")
{
    GamepadState pad = restingDualSense();

    pad.buttons[GamepadState::rightBumper] = true;
    const gp::CameraInput up = gp::mapToCamera(pad, gp::Config{}, kDt);
    CHECK(up.active == true);
    CHECK(up.worldUp > 0.0f);

    pad.buttons[GamepadState::rightBumper] = false;
    pad.buttons[GamepadState::leftBumper] = true;
    CHECK(gp::mapToCamera(pad, gp::Config{}, kDt).worldUp < 0.0f);

    // A hand can reach this by accident, so it is defined rather than left to
    // whichever term was applied last.
    pad.buttons[GamepadState::rightBumper] = true;
    const gp::CameraInput both = gp::mapToCamera(pad, gp::Config{}, kDt);
    CHECK(both.worldUp == 0.0f);
    CHECK(both.active == false);
}

TEST_CASE("the deadzone does not cost the top of the stick's range")
{
    // A stick that only reached 1 - deadzone after the cut is the classic way a
    // pad ends up feeling weaker than the keyboard for the same full deflection.
    GamepadState pad = restingDualSense();
    pad.leftX = 1.0f;
    pad.leftY = 0.0f;

    gp::Config cfg;
    cfg.responseExponent = 1.0f; // linear, so the magnitude is readable
    cfg.moveSpeed = 1.0f;

    const gp::CameraInput out = gp::mapToCamera(pad, cfg, 1.0f);
    CHECK(out.translate.x == doctest::Approx(1.0f));
}

TEST_CASE("the response curve trades the top of the range for control near centre")
{
    GamepadState pad = restingDualSense();
    pad.leftX = 0.5f;
    pad.leftY = 0.0f;

    gp::Config linear;
    linear.responseExponent = 1.0f;
    gp::Config curved;
    curved.responseExponent = 2.0f;

    const float linearX = gp::mapToCamera(pad, linear, 1.0f).translate.x;
    const float curvedX = gp::mapToCamera(pad, curved, 1.0f).translate.x;
    CHECK(curvedX < linearX);
    CHECK(curvedX > 0.0f);
}

TEST_CASE("the deadzone is radial, so a diagonal push stays diagonal")
{
    // A per-axis deadzone leaves a cross-shaped dead region: a gentle diagonal
    // survives on one axis only and the camera crabs sideways when the user
    // asked to go diagonally.
    GamepadState pad = restingDualSense();
    pad.leftX = 0.6f;
    pad.leftY = -0.6f;

    gp::Config cfg;
    cfg.moveSpeed = 1.0f;

    const gp::CameraInput out = gp::mapToCamera(pad, cfg, 1.0f);
    // Forward is -z and the stick's -y is forward, so the two magnitudes match
    // and the direction the user pushed is preserved.
    CHECK(std::fabs(out.translate.x) == doctest::Approx(std::fabs(out.translate.z)));
    CHECK(out.translate.x > 0.0f);
    CHECK(out.translate.z < 0.0f);
}

TEST_CASE("a diagonal is not faster than a cardinal")
{
    // The hardware reports slightly past 1.0 in the corners; an unclamped
    // diagonal would outrun a straight push.
    GamepadState pad = restingDualSense();
    gp::Config cfg;
    cfg.responseExponent = 1.0f;
    cfg.moveSpeed = 1.0f;

    pad.leftX = 1.0f;
    pad.leftY = 0.0f;
    const gp::CameraInput cardinal = gp::mapToCamera(pad, cfg, 1.0f);
    const float cardinalSpeed = glm::length(cardinal.translate);

    pad.leftX = 1.0f;
    pad.leftY = -1.0f; // both axes railed, magnitude sqrt(2)
    const gp::CameraInput diagonal = gp::mapToCamera(pad, cfg, 1.0f);
    CHECK(glm::length(diagonal.translate) <= doctest::Approx(cardinalSpeed));
}

TEST_CASE("triggers scale speed and cancel each other when both are pressed")
{
    gp::Config cfg;

    CHECK(gp::speedScaleFromTriggers(0.0f, 0.0f, cfg) == doctest::Approx(1.0f));
    CHECK(gp::speedScaleFromTriggers(0.0f, 1.0f, cfg) == doctest::Approx(cfg.triggerBoost));
    CHECK(gp::speedScaleFromTriggers(1.0f, 0.0f, cfg) == doctest::Approx(cfg.triggerBrake));

    // Geometric, not additive: both railed lands back on 1.0 rather than on
    // whichever term happened to be applied second.
    CHECK(gp::speedScaleFromTriggers(1.0f, 1.0f, cfg) ==
          doctest::Approx(cfg.triggerBoost * cfg.triggerBrake));

    // Analogue, not a switch: half-pressed is between the ends.
    const float half = gp::speedScaleFromTriggers(0.0f, 0.5f, cfg);
    CHECK(half > 1.0f);
    CHECK(half < cfg.triggerBoost);
}

TEST_CASE("the boost trigger scales movement but not look")
{
    // Look speed is an angular rate the hand has calibrated to; scaling it with
    // the throttle would make aiming impossible exactly when the camera is
    // moving fastest.
    GamepadState pad = restingDualSense();
    pad.leftY = -1.0f;
    pad.rightX = 1.0f;

    const gp::CameraInput plain = gp::mapToCamera(pad, gp::Config{}, kDt);
    pad.rightTrigger = 1.0f;
    const gp::CameraInput boosted = gp::mapToCamera(pad, gp::Config{}, kDt);

    CHECK(boosted.translate.z == doctest::Approx(plain.translate.z * gp::Config{}.triggerBoost));
    CHECK(boosted.lookX == doctest::Approx(plain.lookX));
}

TEST_CASE("a trigger alone does not move the camera")
{
    // Holding the throttle with the sticks centred multiplies zero. It must not
    // read as input, or resting a finger on R2 restarts accumulation forever.
    GamepadState pad = restingDualSense();
    pad.rightTrigger = 1.0f;

    const gp::CameraInput out = gp::mapToCamera(pad, gp::Config{}, kDt);
    CHECK(out.active == false);
    CHECK(out.speedScale == doctest::Approx(gp::Config{}.triggerBoost));
}

TEST_CASE("input scales with the frame time, so speed does not depend on frame rate")
{
    // A path tracer's frame time swings by a factor of several between the frame
    // that restarts accumulation and the ones after it. Per-frame input would
    // make the camera's speed a function of how expensive the scene is.
    GamepadState pad = restingDualSense();
    pad.leftY = -1.0f;
    pad.rightX = 1.0f;

    const gp::CameraInput slow = gp::mapToCamera(pad, gp::Config{}, 1.0f / 30.0f);
    const gp::CameraInput fast = gp::mapToCamera(pad, gp::Config{}, 1.0f / 60.0f);

    CHECK(slow.translate.z == doctest::Approx(fast.translate.z * 2.0f));
    CHECK(slow.lookX == doctest::Approx(fast.lookX * 2.0f));
}

TEST_CASE("a zero-length frame asks for nothing rather than dividing by it")
{
    GamepadState pad = restingDualSense();
    pad.leftY = -1.0f;

    const gp::CameraInput out = gp::mapToCamera(pad, gp::Config{}, 0.0f);
    CHECK(out.active == false);
    CHECK(out.translate.z == 0.0f);
}

TEST_CASE("PlayStation face buttons land where the SDL mapping puts them")
{
    // The mapping database is written in Xbox names, so cross is 'a' and circle
    // is 'b'. Recorded because a reader holding a DualSense has no other way to
    // know which enumerator their thumb is on.
    CHECK(static_cast<int>(GamepadState::a) == 0);
    CHECK(static_cast<int>(GamepadState::b) == 1);
    CHECK(static_cast<int>(GamepadState::x) == 2);
    CHECK(static_cast<int>(GamepadState::y) == 3);
    CHECK(static_cast<int>(GamepadState::count) == 15);
}

TEST_CASE("pressed() is false on a disconnected pad whatever the button array holds")
{
    GamepadState pad;
    pad.buttons[GamepadState::a] = true;
    CHECK(pad.connected == false);
    CHECK(pad.pressed(GamepadState::a) == false);

    pad.connected = true;
    CHECK(pad.pressed(GamepadState::a) == true);
}

TEST_CASE("the UI takes the pad only while windowing mode is held or text is being typed")
{
    // ImGui and the camera share one controller. ImGui's own key table says the
    // sticks and shoulders are its only "in Windowing mode", which is Square
    // held; the D-pad and face buttons it uses at all times are not bound here.
    GamepadState pad = restingDualSense();

    CHECK(gp::cameraOwnsPad(pad, /*imguiWantsTextInput=*/false) == true);

    // Square held: ImGui is moving windows with the sticks.
    pad.buttons[GamepadState::x] = true;
    CHECK(gp::cameraOwnsPad(pad, false) == false);
    pad.buttons[GamepadState::x] = false;

    // A text field has focus.
    CHECK(gp::cameraOwnsPad(pad, true) == false);

    // Nothing plugged in.
    pad.connected = false;
    CHECK(gp::cameraOwnsPad(pad, false) == false);
}

TEST_CASE("the D-pad and face buttons stay ImGui's")
{
    // Pinning the split the other way round: holding anything ImGui navigates
    // with must not stop the camera, or a user tabbing through a panel with the
    // D-pad would find the view frozen for no visible reason.
    GamepadState pad = restingDualSense();
    for (const GamepadState::Button held : { GamepadState::dpadUp, GamepadState::dpadDown,
                                             GamepadState::dpadLeft, GamepadState::dpadRight,
                                             GamepadState::a, GamepadState::b, GamepadState::y })
    {
        pad.buttons[held] = true;
        CHECK(gp::cameraOwnsPad(pad, false) == true);
        pad.buttons[held] = false;
    }
}

// ---------------------------------------------------------------------------
// Against the real camera.
//
// Everything above tests the mapping in isolation, which cannot catch the one
// thing most likely to be wrong: the sign conventions between the pad, this
// mapping, and Camera::translate. Those are settled by three separate pieces of
// code, and the failure mode -- a camera that flies backwards, or climbs when
// asked to descend -- is invisible in a unit test of any one of them.
// ---------------------------------------------------------------------------

#include "../../src/editor/CameraController.h"

namespace
{

/// Run the controller far enough for the smoothing filter to have paid out
/// essentially all of a queued gesture. The filter is exponential, so this is
/// "close enough to have a direction", not "exactly there".
void settle(oka::CameraController& controller)
{
    for (int i = 0; i < 240; ++i)
    {
        controller.applyPendingInput(1.0f / 60.0f);
    }
}

oka::CameraController makeController()
{
    oka::Camera cam;
    cam.position = glm::float3(0.0f, 0.0f, 0.0f);
    cam.type = oka::Camera::CameraType::firstperson;
    oka::CameraController controller(cam, /*isYup=*/true);
    controller.getCamera().position = glm::float3(0.0f, 0.0f, 0.0f);
    controller.updateViewMatrix();
    return controller;
}

} // namespace

TEST_CASE("the left stick pushed away actually moves the camera along its forward axis")
{
    oka::CameraController controller = makeController();
    const glm::float3 front = controller.getCamera().getFront();
    const glm::float3 start = controller.getCamera().position;

    GamepadState pad = restingDualSense();
    pad.leftY = -1.0f; // pushed away from the user

    gp::Config cfg;
    cfg.moveSpeed = 10.0f;
    controller.applyGamepad(gp::mapToCamera(pad, cfg, 1.0f / 60.0f));
    settle(controller);

    const glm::float3 moved = controller.getCamera().position - start;
    CHECK(glm::length(moved) > 0.0f);
    // Same direction as the camera looks, not the opposite one.
    CHECK(glm::dot(glm::normalize(moved), front) > 0.99f);
}

TEST_CASE("the left stick pushed right moves the camera along its right axis")
{
    oka::CameraController controller = makeController();
    const glm::float3 right = controller.getCamera().getRight();
    const glm::float3 start = controller.getCamera().position;

    GamepadState pad = restingDualSense();
    pad.leftX = 1.0f;

    gp::Config cfg;
    cfg.moveSpeed = 10.0f;
    controller.applyGamepad(gp::mapToCamera(pad, cfg, 1.0f / 60.0f));
    settle(controller);

    const glm::float3 moved = controller.getCamera().position - start;
    CHECK(glm::length(moved) > 0.0f);
    CHECK(glm::dot(glm::normalize(moved), right) > 0.99f);
}

TEST_CASE("R1 lifts along world up even when the camera is pitched")
{
    // The point of keeping worldUp out of the camera-space queue. A shoulder
    // button folded into the camera's own y would climb at an angle whenever the
    // camera was pitched -- and only then, which is how it would survive review.
    oka::CameraController controller = makeController();
    // Pitch well off the horizon before lifting.
    controller.getCamera().rotate(0.0f, 40.0f);
    controller.updateViewMatrix();

    const glm::float3 worldUp = controller.getCamera().getWorldUp();
    const glm::float3 start = controller.getCamera().position;

    GamepadState pad = restingDualSense();
    pad.buttons[GamepadState::rightBumper] = true;

    gp::Config cfg;
    cfg.moveSpeed = 10.0f;
    controller.applyGamepad(gp::mapToCamera(pad, cfg, 1.0f / 60.0f));
    settle(controller);

    const glm::float3 moved = controller.getCamera().position - start;
    CHECK(glm::length(moved) > 0.0f);
    CHECK(glm::dot(glm::normalize(moved), glm::normalize(worldUp)) > 0.99f);
}

TEST_CASE("L1 lowers along world up")
{
    oka::CameraController controller = makeController();
    const glm::float3 worldUp = controller.getCamera().getWorldUp();
    const glm::float3 start = controller.getCamera().position;

    GamepadState pad = restingDualSense();
    pad.buttons[GamepadState::leftBumper] = true;

    gp::Config cfg;
    cfg.moveSpeed = 10.0f;
    controller.applyGamepad(gp::mapToCamera(pad, cfg, 1.0f / 60.0f));
    settle(controller);

    const glm::float3 moved = controller.getCamera().position - start;
    CHECK(glm::dot(glm::normalize(moved), glm::normalize(worldUp)) < -0.99f);
}

TEST_CASE("an idle pad does not take the camera over from an animated glTF camera")
{
    // consumeUserMovedCamera() is what detaches the editor camera from a glTF
    // one during playback. A pad reporting its rest drift as motion would detach
    // it the moment the editor started, with nobody touching anything.
    oka::CameraController controller = makeController();
    (void)controller.consumeUserMovedCamera();

    controller.applyGamepad(gp::mapToCamera(restingDualSense(), gp::Config{}, 1.0f / 60.0f));
    CHECK(controller.consumeUserMovedCamera() == false);

    GamepadState pushed = restingDualSense();
    pushed.leftY = -1.0f;
    controller.applyGamepad(gp::mapToCamera(pushed, gp::Config{}, 1.0f / 60.0f));
    CHECK(controller.consumeUserMovedCamera() == true);
}

TEST_CASE("the gizmo blocks the pad the way it blocks the mouse")
{
    // While a gizmo has the drag, the camera must hold still, or the object and
    // the view move together and the user cannot place anything.
    oka::CameraController controller = makeController();
    const glm::float3 start = controller.getCamera().position;
    controller.setGizmoBlocksInput(true);

    GamepadState pad = restingDualSense();
    pad.leftY = -1.0f;
    pad.rightX = 1.0f;

    gp::Config cfg;
    cfg.moveSpeed = 10.0f;
    controller.applyGamepad(gp::mapToCamera(pad, cfg, 1.0f / 60.0f));
    settle(controller);

    CHECK(controller.getCamera().position == start);
    CHECK(controller.consumeUserMovedCamera() == false);
}

TEST_CASE("stick and mouse compose rather than overwrite")
{
    // Both queue into the same pending input, so a user with a hand on each gets
    // the sum. Checked because "the last one wins" is the shape this would take
    // if the pad were given its own path to the camera.
    oka::CameraController controller = makeController();

    controller.handleMouseMoveCallback(100.0, 100.0);
    controller.mouseButtonCallback(GLFW_MOUSE_BUTTON_RIGHT, GLFW_PRESS, 0, true);
    controller.handleMouseMoveCallback(140.0, 100.0); // drag right: +lookX

    GamepadState pad = restingDualSense();
    pad.rightX = 1.0f; // stick right: +lookX too

    const glm::quat before = controller.getCamera().mOrientation;
    controller.applyGamepad(gp::mapToCamera(pad, gp::Config{}, 1.0f / 60.0f));
    settle(controller);

    // Turned further than the mouse alone would have.
    oka::CameraController mouseOnly = makeController();
    mouseOnly.handleMouseMoveCallback(100.0, 100.0);
    mouseOnly.mouseButtonCallback(GLFW_MOUSE_BUTTON_RIGHT, GLFW_PRESS, 0, true);
    mouseOnly.handleMouseMoveCallback(140.0, 100.0);
    settle(mouseOnly);

    const float bothYaw = glm::angle(glm::normalize(before * glm::conjugate(controller.getCamera().mOrientation)));
    const float mouseYaw =
        glm::angle(glm::normalize(before * glm::conjugate(mouseOnly.getCamera().mOrientation)));
    CHECK(bothYaw > mouseYaw);
}
