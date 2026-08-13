#include <doctest/doctest.h>

#include "editor_camera_framing.h"

#include <cmath>

using namespace oka;
using namespace oka::editor_camera_framing;

namespace
{

Camera perspectiveLookingDownZ()
{
    Camera cam;
    cam.setPerspective(60.0f, 1.0f, 0.1f, 1000.0f);
    cam.position = glm::float3(0.0f, 0.0f, 10.0f);
    cam.mOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    cam.updateViewMatrix();
    return cam;
}

Camera orthographicLookingDownZ()
{
    Camera cam;
    cam.setOrthographic(2.0f, 2.0f, 0.1f, 1000.0f);
    cam.position = glm::float3(0.0f, 0.0f, 10.0f);
    cam.mOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    cam.updateViewMatrix();
    return cam;
}

} // namespace

TEST_CASE("worldAabbFromLocalBox transforms an oriented box into a world AABB")
{
    // Unit cube centred at the origin, translated to (10, 0, 0) and scaled 2x on Y.
    glm::mat4 xform(1.0f);
    xform = glm::translate(xform, glm::float3(10.0f, 0.0f, 0.0f));
    xform = glm::scale(xform, glm::float3(1.0f, 2.0f, 1.0f));

    glm::float3 wMin, wMax;
    worldAabbFromLocalBox(glm::float3(-0.5f), glm::float3(0.5f), xform, wMin, wMax);

    CHECK(wMin.x == doctest::Approx(-0.5f + 10.0f));
    CHECK(wMax.x == doctest::Approx(0.5f + 10.0f));
    CHECK(wMin.y == doctest::Approx(-1.0f));
    CHECK(wMax.y == doctest::Approx(1.0f));
    CHECK(wMin.z == doctest::Approx(-0.5f));
    CHECK(wMax.z == doctest::Approx(0.5f));
}

TEST_CASE("perspectiveFitDistance keeps the box inside the vertical and horizontal FOV")
{
    // Square film, 90 deg vertical: tan(45)=1, so a half-extent of 1 fits at
    // distance 1 before padding. Padding of 1 leaves that number alone.
    const float d = perspectiveFitDistance(/*halfW*/ 1.0f, /*halfH*/ 1.0f, /*halfD*/ 0.0f,
                                           /*fov*/ 90.0f, /*aspect*/ 1.0f, /*padding*/ 1.0f);
    CHECK(d == doctest::Approx(1.0f).epsilon(1e-4));

    // A wider box must push the camera back by the horizontal FOV.
    const float wide = perspectiveFitDistance(2.0f, 1.0f, 0.0f, 90.0f, 1.0f, 1.0f);
    CHECK(wide == doctest::Approx(2.0f).epsilon(1e-4));

    // A deep box cannot sit closer than its own half-depth, or the near face is
    // behind the film.
    const float deep = perspectiveFitDistance(0.1f, 0.1f, 5.0f, 90.0f, 1.0f, 1.0f);
    CHECK(deep == doctest::Approx(5.0f).epsilon(1e-4));
}

TEST_CASE("orthographicFitExtents survive magForAspect without cropping")
{
    float xmag = 0.0f;
    float ymag = 0.0f;

    // Object taller than the landscape viewport: the held horizontal extent has
    // to grow so the derived vertical still covers the object.
    orthographicFitExtents(/*halfW*/ 1.0f, /*halfH*/ 2.0f, /*aspect*/ 2.0f, xmag, ymag, 1.0f);
    CHECK(xmag == doctest::Approx(4.0f).epsilon(1e-4)); // max(1, 2*2)
    CHECK(ymag == doctest::Approx(2.0f).epsilon(1e-4));

    Camera cam;
    cam.setOrthographic(xmag, ymag, 0.1f, 1000.0f);
    float halfW = 0.0f;
    float halfH = 0.0f;
    cam.magForAspect(2.0f, halfW, halfH);
    CHECK(halfW >= 1.0f);
    CHECK(halfH >= 2.0f);

    // Portrait viewport, object wider than tall.
    orthographicFitExtents(2.0f, 1.0f, 0.5f, xmag, ymag, 1.0f);
    cam.setOrthographic(xmag, ymag, 0.1f, 1000.0f);
    cam.magForAspect(0.5f, halfW, halfH);
    CHECK(halfW >= 2.0f);
    CHECK(halfH >= 1.0f);
}

TEST_CASE("frameCamera on a perspective camera dollies along the view axis")
{
    Camera cam = perspectiveLookingDownZ();
    const glm::quat orientation = cam.mOrientation;
    const glm::float3 worldMin(-1.0f, -1.0f, -1.0f);
    const glm::float3 worldMax(1.0f, 1.0f, 1.0f);

    frameCamera(cam, worldMin, worldMax, /*aspect*/ 1.0f, /*padding*/ 1.0f);

    CHECK(cam.mOrientation == orientation);
    // Looking down -Z: the camera should sit on +Z, in front of the box centre.
    CHECK(cam.position.x == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(cam.position.y == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(cam.position.z > 1.0f);

    // The projected half-extents of the unit cube on a camera looking down -Z
    // are 1 in X and Y; at 60 deg FOV that distance is 1/tan(30) ≈ 1.732, and
    // half-depth 1 forces at least that — the FOV term wins here.
    const float expected = 1.0f / std::tan(glm::radians(30.0f));
    CHECK(cam.position.z == doctest::Approx(expected).epsilon(1e-3));
}

TEST_CASE("frameCamera on an orthographic camera resizes the film, not the distance alone")
{
    Camera cam = orthographicLookingDownZ();
    const float startXmag = cam.xmag;
    const glm::float3 worldMin(-0.5f, -0.25f, -0.5f);
    const glm::float3 worldMax(0.5f, 0.25f, 0.5f);

    frameCamera(cam, worldMin, worldMax, /*aspect*/ 1.0f, /*padding*/ 1.0f);

    // Square aspect, object half-extents 0.5 x 0.25 → film half 0.5.
    CHECK(cam.xmag == doctest::Approx(0.5f).epsilon(1e-4));
    CHECK(cam.ymag == doctest::Approx(0.5f).epsilon(1e-4));
    CHECK(cam.xmag != startXmag);

    // Film sits just in front of the near face (z = +0.5 of the box), not at the
    // far pre-frame distance of 10 — otherwise "frame" would leave the box as a
    // speck on an unchanged film.
    CHECK(cam.position.z == doctest::Approx(0.5f).epsilon(1e-3));
    CHECK(cam.position.x == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(cam.position.y == doctest::Approx(0.0f).epsilon(1e-4));
}

TEST_CASE("frameCamera padding leaves a margin around the selection")
{
    Camera cam = orthographicLookingDownZ();
    const glm::float3 worldMin(-1.0f, -1.0f, -1.0f);
    const glm::float3 worldMax(1.0f, 1.0f, 1.0f);

    frameCamera(cam, worldMin, worldMax, 1.0f, /*padding*/ 1.25f);
    CHECK(cam.xmag == doctest::Approx(1.25f).epsilon(1e-4));
    CHECK(cam.ymag == doctest::Approx(1.25f).epsilon(1e-4));
}
