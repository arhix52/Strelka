#include <doctest/doctest.h>

#include <strelka/scene/camera.h>

#include <glm/gtc/matrix_transform.hpp>

using namespace oka;

namespace
{

Camera makeYawedCamera(float yawDegrees)
{
    Camera cam;
    cam.position = glm::float3(2.0f, 1.0f, 3.0f);
    cam.mOrientation = glm::quat(glm::vec3(0.0f, glm::radians(yawDegrees), 0.0f));
    cam.setPerspective(45.0f, 4.0f / 3.0f, 0.1f, 1000.0f);
    cam.updateViewMatrix();
    return cam;
}

} // namespace

// The shaders reconstruct the camera basis from viewToWorld to place the lens
// sample and to measure the focus distance (see generateCameraRay in
// shading_common.h). glm and Metal both store columns, so the basis is the matrix
// *columns*; reading the rows instead transposes the rotation, and the depth of
// field that comes out of it is not merely skewed -- past 45 degrees of yaw the
// forward axis it derives is orthogonal to the ray, the focus distance divides by
// zero, and the aperture stops doing anything at all.
TEST_CASE("Camera basis lives in the columns of viewToWorld")
{
    for (const float yaw : { 0.0f, 30.0f, 45.0f, 90.0f, 145.0f })
    {
        const Camera cam = makeYawedCamera(yaw);
        const glm::mat4 viewToWorld = glm::inverse(cam.matrices.view);

        const glm::float3 right(viewToWorld[0]);
        const glm::float3 up(viewToWorld[1]);
        const glm::float3 backward(viewToWorld[2]);
        const glm::float3 eye(viewToWorld[3]);

        CHECK(glm::length(right - cam.getRight()) == doctest::Approx(0.0f).epsilon(1e-4));
        CHECK(glm::length(up - cam.getUp()) == doctest::Approx(0.0f).epsilon(1e-4));
        CHECK(glm::length(-backward - cam.getFront()) == doctest::Approx(0.0f).epsilon(1e-4));
        CHECK(glm::length(eye - cam.position) == doctest::Approx(0.0f).epsilon(1e-4));
    }
}

// What the transposed read produced, stated as a fact rather than a warning: a
// rotated camera's rows are a different basis, and a ray down the view axis is
// perpendicular to the forward axis they yield at 45 degrees. That zero is what
// silently switched depth of field off.
TEST_CASE("Rows of viewToWorld are not the camera basis for a rotated camera")
{
    const Camera cam = makeYawedCamera(45.0f);
    const glm::mat4 viewToWorld = glm::inverse(cam.matrices.view);

    const glm::float3 forwardFromColumns = -glm::float3(viewToWorld[2]);
    const glm::float3 forwardFromRows = -glm::float3(viewToWorld[0][2], viewToWorld[1][2], viewToWorld[2][2]);

    CHECK(glm::dot(forwardFromColumns, cam.getFront()) == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(glm::dot(forwardFromRows, cam.getFront()) == doctest::Approx(0.0f).epsilon(1e-4));
}
