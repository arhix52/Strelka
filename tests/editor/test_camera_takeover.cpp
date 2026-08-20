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

    // The key coming up does not stop the camera dead -- movement is smoothed, so
    // it glides for a couple of tenths of a second, and that glide is still the
    // user's motion. Only once it has settled is the camera nobody's.
    controller.keyCallback(GLFW_KEY_W, 0, GLFW_RELEASE, 0);
    controller.update(0.016, 1.0f);
    CHECK(controller.consumeUserMovedCamera());

    for (int i = 0; i < 100; ++i)
    {
        controller.update(0.016, 1.0f);
        controller.consumeUserMovedCamera();
    }
    controller.update(0.016, 1.0f);
    CHECK_FALSE(controller.consumeUserMovedCamera());

    controller.keyCallback(GLFW_KEY_LEFT, 0, GLFW_PRESS, 0);
    controller.update(0.016, 1.0f);
    CHECK(controller.consumeUserMovedCamera());
}

// The wheel used to reach nothing at all -- Display::scrollCallback asserted its
// window and returned -- which left an orthographic camera with no zoom control of
// any kind, since its framing is xmag/ymag rather than its position.
TEST_CASE("the wheel zooms an orthographic camera")
{
    Camera cam;
    cam.setOrthographic(0.45f, 0.45f, 0.001f, 1000.0f);
    CameraController controller(cam, true);

    const float startMag = controller.getCamera().xmag;
    controller.scrollCallback(0.0, 1.0);
    CHECK(controller.getCamera().xmag < startMag);
    // A zoom is the user driving the camera, so it has to take a glTF camera over
    // exactly as a movement key does -- otherwise animation poses it back and the
    // zoom the user asked for is the one thing that does not survive the frame.
    CHECK(controller.consumeUserMovedCamera());

    controller.scrollCallback(0.0, -1.0);
    CHECK(controller.getCamera().xmag == doctest::Approx(startMag).epsilon(1e-5));

    // Trackpads send fractional deltas in a stream; they compose the same way.
    controller.scrollCallback(0.0, 0.25);
    controller.scrollCallback(0.0, 0.75);
    const float trackpad = controller.getCamera().xmag;
    controller.scrollCallback(0.0, -1.0);
    CHECK(trackpad < startMag);
    CHECK(controller.getCamera().xmag == doctest::Approx(startMag).epsilon(1e-5));
}

TEST_CASE("the wheel leaves a perspective camera and a gizmo drag alone")
{
    Camera cam;
    CameraController controller(cam, true);
    const float startMag = controller.getCamera().xmag;
    const glm::float3 startPos = controller.getCamera().position;

    controller.scrollCallback(0.0, 1.0);
    CHECK(controller.getCamera().xmag == startMag);
    CHECK(controller.getCamera().position == startPos);
    CHECK_FALSE(controller.consumeUserMovedCamera());

    Camera ortho;
    ortho.setOrthographic(0.45f, 0.45f, 0.001f, 1000.0f);
    CameraController orthoController(ortho, true);
    orthoController.setGizmoBlocksInput(true);
    orthoController.scrollCallback(0.0, 1.0);
    CHECK(orthoController.getCamera().xmag == doctest::Approx(0.45f).epsilon(1e-5));
    CHECK_FALSE(orthoController.consumeUserMovedCamera());
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
