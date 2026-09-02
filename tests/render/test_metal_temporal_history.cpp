#include <doctest/doctest.h>

#include "temporal_history_policy.h"

using namespace oka;

TEST_CASE("perspective camera motion preserves temporal history")
{
    Camera camera;
    camera.setPerspective(50.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
    camera.updateViewMatrix();
    const Camera::Matrices previous = camera.matrices;

    SUBCASE("an accelerating sub-millimetre translation is still motion")
    {
        camera.position += glm::vec3(0.0001f, 0.0f, 0.0f);
        camera.updateViewMatrix();
        CHECK_FALSE(metal::temporal_history::projectionChanged(camera.matrices, previous));
    }
}

TEST_CASE("a projection change invalidates temporal history")
{
    Camera camera;
    camera.setPerspective(50.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
    const Camera::Matrices previous = camera.matrices;

    camera.setPerspective(35.0f, 16.0f / 9.0f, 0.1f, 1000.0f);

    CHECK(metal::temporal_history::projectionChanged(camera.matrices, previous));
}

TEST_CASE("MetalFX depth convention follows the camera projection")
{
    Camera camera;
    camera.setPerspective(45.0f, 1.0f, 0.1f, 100.0f);
    const glm::vec4 perspectiveNear = camera.matrices.perspective * glm::vec4(0.0f, 0.0f, -0.1f, 1.0f);
    const glm::vec4 perspectiveFar = camera.matrices.perspective * glm::vec4(0.0f, 0.0f, -100.0f, 1.0f);
    CHECK(perspectiveNear.z / perspectiveNear.w == doctest::Approx(0.0f));
    CHECK(perspectiveFar.z / perspectiveFar.w == doctest::Approx(1.0f));
    camera.setOrthographic(1.0f, 1.0f, 0.1f, 100.0f);
    const glm::vec4 orthoNear = camera.matrices.perspective * glm::vec4(0.0f, 0.0f, -0.1f, 1.0f);
    const glm::vec4 orthoFar = camera.matrices.perspective * glm::vec4(0.0f, 0.0f, -100.0f, 1.0f);
    CHECK(orthoNear.z / orthoNear.w == doctest::Approx(1.0f));
    CHECK(orthoFar.z / orthoFar.w == doctest::Approx(0.0f));
}
