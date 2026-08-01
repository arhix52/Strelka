#include <doctest/doctest.h>

#include <strelka/scene/camera.h>
#include <strelka/scene/scene.h>

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

using namespace oka;

namespace
{

uint32_t addQuad(Scene& scene, const glm::float3& center, float halfSize)
{
    if (scene.getMaterials().empty())
    {
        Scene::MaterialDescription mat{};
        mat.name = "default";
        scene.addMaterial(mat);
    }

    std::vector<Scene::Vertex> vb(4);
    vb[0].pos = glm::float3(-halfSize, -halfSize, 0.0f);
    vb[1].pos = glm::float3(halfSize, -halfSize, 0.0f);
    vb[2].pos = glm::float3(halfSize, halfSize, 0.0f);
    vb[3].pos = glm::float3(-halfSize, halfSize, 0.0f);
    std::vector<uint32_t> ib = { 0, 1, 2, 0, 2, 3 };
    const uint32_t meshId = scene.createMesh(vb, ib);
    return scene.createInstance(Instance::Type::eMesh, meshId, 0, glm::translate(glm::mat4(1.0f), center));
}

Camera makeCamera(float fovDegrees, float aspect)
{
    Camera cam;
    cam.position = glm::float3(0.0f, 0.0f, 5.0f);
    cam.mOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    cam.setPerspective(fovDegrees, aspect, 0.1f, 1000.0f);
    cam.updateViewMatrix();
    return cam;
}

} // namespace

TEST_CASE("Pick ray leaves the camera along the view direction")
{
    const Camera cam = makeCamera(60.0f, 16.0f / 9.0f);

    glm::float3 origin, dir;
    generatePickRay(cam, glm::float2(0.5f, 0.5f), origin, dir);

    CHECK(origin.x == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(origin.y == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(origin.z == doctest::Approx(5.0f).epsilon(1e-5));
    CHECK(dir.x == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(dir.y == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(dir.z == doctest::Approx(-1.0f).epsilon(1e-4));
}

// The editor reads the camera for picking and gizmos before the first frame has
// been rendered, and the renderer is what normally fills in the projection. A
// non-finite ray there misses everything and reports nothing, so the camera has
// to stay usable even when nobody configured it yet.
TEST_CASE("Pick ray stays finite for a camera whose projection was never set")
{
    Camera cam;
    cam.updateViewMatrix();

    glm::float3 origin, dir;
    generatePickRay(cam, glm::float2(0.5f, 0.5f), origin, dir);

    CHECK(std::isfinite(origin.x));
    CHECK(std::isfinite(origin.y));
    CHECK(std::isfinite(origin.z));
    CHECK(std::isfinite(dir.x));
    CHECK(std::isfinite(dir.y));
    CHECK(std::isfinite(dir.z));
    CHECK(glm::length(dir) == doctest::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("Pick ray spans exactly the camera frustum, y up on screen")
{
    const float fov = 60.0f;
    const float aspect = 16.0f / 9.0f;
    const Camera cam = makeCamera(fov, aspect);
    const float tanHalf = std::tan(glm::radians(fov) * 0.5f);

    glm::float3 origin, dir;

    // uv.y == 0 is the top of the image, which is +y in world space here.
    generatePickRay(cam, glm::float2(0.5f, 0.0f), origin, dir);
    CHECK(dir.y > 0.0f);
    CHECK(dir.y / -dir.z == doctest::Approx(tanHalf).epsilon(1e-3));

    generatePickRay(cam, glm::float2(0.5f, 1.0f), origin, dir);
    CHECK(dir.y < 0.0f);

    // Horizontal half-angle widens with the aspect ratio.
    generatePickRay(cam, glm::float2(1.0f, 0.5f), origin, dir);
    CHECK(dir.x > 0.0f);
    CHECK(dir.x / -dir.z == doctest::Approx(tanHalf * aspect).epsilon(1e-3));

    generatePickRay(cam, glm::float2(0.0f, 0.5f), origin, dir);
    CHECK(dir.x < 0.0f);
}

TEST_CASE("Screen space pick selects the object under the cursor in every quadrant")
{
    Scene scene;
    // Four quads on the z = 0 plane, one per screen quadrant as seen from +z.
    const uint32_t topLeft = addQuad(scene, glm::float3(-1.5f, 1.5f, 0.0f), 0.6f);
    const uint32_t topRight = addQuad(scene, glm::float3(1.5f, 1.5f, 0.0f), 0.6f);
    const uint32_t bottomLeft = addQuad(scene, glm::float3(-1.5f, -1.5f, 0.0f), 0.6f);
    const uint32_t bottomRight = addQuad(scene, glm::float3(1.5f, -1.5f, 0.0f), 0.6f);

    const Camera cam = makeCamera(60.0f, 1.0f);

    auto pickAt = [&](float u, float v) {
        glm::float3 origin, dir;
        generatePickRay(cam, glm::float2(u, v), origin, dir);
        return scene.pick(origin, dir);
    };

    // Quads sit at +-1.5 in a plane 5 units away: with a 60 degree vertical fov
    // the half extent at that depth is 5 * tan(30) ~ 2.9, so each sample lands on
    // the centre of one quad -- which is also the diagonal shared by its two
    // triangles, so this doubles as a watertightness check.
    const float offset = 1.5f / (2.0f * 5.0f * std::tan(glm::radians(30.0f)));

    const Scene::PickHit tl = pickAt(0.5f - offset, 0.5f - offset);
    REQUIRE(tl.hit);
    CHECK(tl.instanceId == topLeft);

    const Scene::PickHit tr = pickAt(0.5f + offset, 0.5f - offset);
    REQUIRE(tr.hit);
    CHECK(tr.instanceId == topRight);

    const Scene::PickHit bl = pickAt(0.5f - offset, 0.5f + offset);
    REQUIRE(bl.hit);
    CHECK(bl.instanceId == bottomLeft);

    const Scene::PickHit br = pickAt(0.5f + offset, 0.5f + offset);
    REQUIRE(br.hit);
    CHECK(br.instanceId == bottomRight);

    // Dead centre falls between all four quads.
    CHECK_FALSE(pickAt(0.5f, 0.5f).hit);
}
