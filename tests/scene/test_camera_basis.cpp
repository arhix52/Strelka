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

TEST_CASE("Rows of viewToWorld are not the camera basis for a rotated camera")
{
    const Camera cam = makeYawedCamera(45.0f);
    const glm::mat4 viewToWorld = glm::inverse(cam.matrices.view);

    const glm::float3 forwardFromColumns = -glm::float3(viewToWorld[2]);
    const glm::float3 forwardFromRows = -glm::float3(viewToWorld[0][2], viewToWorld[1][2], viewToWorld[2][2]);

    CHECK(glm::dot(forwardFromColumns, cam.getFront()) == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(glm::dot(forwardFromRows, cam.getFront()) == doctest::Approx(0.0f).epsilon(1e-4));
}
