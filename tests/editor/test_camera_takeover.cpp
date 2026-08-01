#include <doctest/doctest.h>

#include "../../src/editor/CameraController.h"

using namespace oka;

namespace
{

// The controller reads deltas against its last cursor position, so a gesture has
// to start from a known one.
void placeCursor(CameraController& controller, float x, float y)
{
    controller.handleMouseMoveCallback(x, y);
}

void press(CameraController& controller, int button)
{
    controller.mouseButtonCallback(button, GLFW_PRESS, 0, true);
}

void release(CameraController& controller, int button)
{
    controller.mouseButtonCallback(button, GLFW_RELEASE, 0, true);
}

} // namespace

TEST_CASE("a selection click does not take the camera over")
{
    Camera cam;
    CameraController controller(cam, true);
    placeCursor(controller, 100.0f, 100.0f);
    CHECK_FALSE(controller.consumeUserMovedCamera());

    // Click in place: this is how the viewport picks an object, and it shares the
    // left button with the camera dolly.
    press(controller, GLFW_MOUSE_BUTTON_LEFT);
    release(controller, GLFW_MOUSE_BUTTON_LEFT);
    CHECK_FALSE(controller.consumeUserMovedCamera());

    // A click with the unsteady hand every click has: still a click, and picking
    // treats it as one too.
    press(controller, GLFW_MOUSE_BUTTON_LEFT);
    placeCursor(controller, 101.0f, 100.0f);
    placeCursor(controller, 101.0f, 101.0f);
    release(controller, GLFW_MOUSE_BUTTON_LEFT);
    CHECK_FALSE(controller.consumeUserMovedCamera());
}

TEST_CASE("dragging takes the camera over")
{
    Camera cam;
    CameraController controller(cam, true);
    placeCursor(controller, 100.0f, 100.0f);

    press(controller, GLFW_MOUSE_BUTTON_LEFT);
    placeCursor(controller, 100.0f, 140.0f);
    release(controller, GLFW_MOUSE_BUTTON_LEFT);
    CHECK(controller.consumeUserMovedCamera());
    // Consuming clears it: one gesture, one takeover.
    CHECK_FALSE(controller.consumeUserMovedCamera());

    press(controller, GLFW_MOUSE_BUTTON_RIGHT);
    placeCursor(controller, 160.0f, 140.0f);
    release(controller, GLFW_MOUSE_BUTTON_RIGHT);
    CHECK(controller.consumeUserMovedCamera());
}

TEST_CASE("cursor motion with no button held is not a takeover")
{
    Camera cam;
    CameraController controller(cam, true);
    placeCursor(controller, 100.0f, 100.0f);

    placeCursor(controller, 400.0f, 300.0f);
    placeCursor(controller, 10.0f, 20.0f);
    CHECK_FALSE(controller.consumeUserMovedCamera());
}

TEST_CASE("movement keys take the camera over")
{
    Camera cam;
    CameraController controller(cam, true);

    controller.keyCallback(GLFW_KEY_W, 0, GLFW_PRESS, 0);
    controller.update(0.016, 1.0f);
    CHECK(controller.consumeUserMovedCamera());

    controller.keyCallback(GLFW_KEY_W, 0, GLFW_RELEASE, 0);
    controller.update(0.016, 1.0f);
    CHECK_FALSE(controller.consumeUserMovedCamera());

    controller.keyCallback(GLFW_KEY_LEFT, 0, GLFW_PRESS, 0);
    controller.update(0.016, 1.0f);
    CHECK(controller.consumeUserMovedCamera());
}

TEST_CASE("a click the gizmo swallowed is not a takeover")
{
    Camera cam;
    CameraController controller(cam, true);
    placeCursor(controller, 100.0f, 100.0f);

    controller.setGizmoBlocksInput(true);
    press(controller, GLFW_MOUSE_BUTTON_LEFT);
    placeCursor(controller, 160.0f, 140.0f);
    release(controller, GLFW_MOUSE_BUTTON_LEFT);
    CHECK_FALSE(controller.consumeUserMovedCamera());
}
