#include <doctest/doctest.h>

#include "../../src/editor/CameraController.h"

#include <glm/gtx/quaternion.hpp>

using namespace oka;

namespace
{

// One second of input at a fixed frame time, so a smoothed control can be
// compared against the unsmoothed distance it is meant to converge on.
void hold(CameraController& controller, int frames, float dt = 1.0f / 60.0f)
{
    for (int i = 0; i < frames; ++i)
    {
        controller.update(dt, 1.0f);
    }
}

float yawDegrees(const Camera& cam)
{
    return glm::degrees(glm::yaw(cam.mOrientation));
}

} // namespace

// What this pins: the camera used to take deltaTime * speed as a step every frame,
// and a path tracer's deltaTime is not steady -- the frame that restarts
// accumulation costs several times what the next one does. Equal input, unequal
// step, motion that visibly stutters. Smoothing ramps the *input*, so a spike in
// one frame's dt is spread across the ones after it.
TEST_CASE("held movement ramps up instead of starting at full speed")
{
    Camera cam;
    CameraController controller(cam, true);

    const glm::float3 start = controller.getCamera().position;
    controller.keyCallback(GLFW_KEY_W, 0, GLFW_PRESS, 0);

    hold(controller, 1);
    const float firstStep = glm::length(controller.getCamera().position - start);
    // A frame's worth at full speed; the first frame must be a fraction of it.
    const float unsmoothedStep = (1.0f / 60.0f) * 1.0f;
    CHECK(firstStep > 0.0f);
    CHECK(firstStep < unsmoothedStep * 0.5f);

    // ...and settle onto the full rate, so the smoothing is a ramp and not a
    // permanent tax on how fast the camera flies.
    hold(controller, 60);
    const glm::float3 before = controller.getCamera().position;
    hold(controller, 1);
    const float steadyStep = glm::length(controller.getCamera().position - before);
    CHECK(steadyStep == doctest::Approx(unsmoothedStep).epsilon(0.02));
}

TEST_CASE("movement coasts to a stop after the key comes up")
{
    Camera cam;
    CameraController controller(cam, true);

    controller.keyCallback(GLFW_KEY_W, 0, GLFW_PRESS, 0);
    hold(controller, 60);
    controller.keyCallback(GLFW_KEY_W, 0, GLFW_RELEASE, 0);

    const glm::float3 atRelease = controller.getCamera().position;
    hold(controller, 1);
    CHECK(controller.getCamera().position != atRelease);

    // The coast is bounded: it dies out rather than drifting forever, which would
    // keep restarting accumulation on a camera nobody is touching.
    hold(controller, 120);
    const glm::float3 settled = controller.getCamera().position;
    hold(controller, 10);
    CHECK(controller.getCamera().position == settled);
    CHECK_FALSE(controller.getCamera().isSettling());
}

// Mouse look is filtered by queueing, not by damping: the queue drains completely,
// so a gesture turns the camera by exactly as much as it always did -- it just
// arrives over a few frames instead of in whichever frame the callbacks landed in.
TEST_CASE("mouse look is spread over frames without changing how far it turns")
{
    Camera reference;
    // The same rotation the drag below asks for, applied in one go.
    reference.rotationSpeed = 0.025f;
    reference.rotate(60.0f, 0.0f);

    Camera cam;
    CameraController controller(cam, true);
    controller.handleMouseMoveCallback(100.0, 100.0);
    controller.mouseButtonCallback(GLFW_MOUSE_BUTTON_RIGHT, GLFW_PRESS, 0, true);
    controller.handleMouseMoveCallback(160.0, 100.0);

    // Nothing has reached the camera yet: the callback only queues.
    CHECK(yawDegrees(controller.getCamera()) == doctest::Approx(0.0f));

    hold(controller, 1);
    const float afterOneFrame = std::abs(yawDegrees(controller.getCamera()));
    CHECK(afterOneFrame > 0.0f);
    CHECK(afterOneFrame < std::abs(yawDegrees(reference)) * 0.75f);

    hold(controller, 120);
    CHECK(yawDegrees(controller.getCamera()) == doctest::Approx(yawDegrees(reference)).epsilon(1e-3));
}

// A frame that took a second is a hitch -- a scene load, a window drag, a
// breakpoint -- not the renderer running slowly. Integrating it moves the camera
// by however long the pause happened to be, which is a teleport, not a frame.
TEST_CASE("a hitch does not teleport the camera")
{
    Camera cam;
    CameraController controller(cam, true);
    controller.keyCallback(GLFW_KEY_W, 0, GLFW_PRESS, 0);

    const glm::float3 start = controller.getCamera().position;
    controller.update(5.0, 1.0f);
    const float moved = glm::length(controller.getCamera().position - start);
    // At most one clamped frame at full speed, and in fact less: the ramp has
    // only had that one frame to rise.
    CHECK(moved <= 0.1f);
}
