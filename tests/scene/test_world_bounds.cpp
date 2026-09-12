#include <doctest/doctest.h>

#include <strelka/scene/scene.h>

#include <glm/gtc/matrix_transform.hpp>

#include <vector>

using namespace oka;

namespace
{

// A unit quad in the XY plane, so a translated instance's world box is that
// translation plus [-1, 1] on x and y.
uint32_t addQuad(Scene& scene)
{
    const glm::float3 positions[4] = { { -1, -1, 0 }, { 1, -1, 0 }, { 1, 1, 0 }, { -1, 1, 0 } };
    std::vector<Scene::Vertex> vb(4);
    for (int i = 0; i < 4; ++i)
    {
        vb[i].pos = positions[i];
    }
    const std::vector<uint32_t> ib = { 0, 1, 2, 0, 2, 3 };
    return scene.createMesh(vb, ib);
}

} // namespace

TEST_CASE("worldBounds is empty until the scene has geometry")
{
    Scene scene;
    glm::float3 lo(0.0f);
    glm::float3 hi(0.0f);
    CHECK_FALSE(scene.worldBounds(lo, hi));
}

TEST_CASE("worldBounds unions every instance")
{
    Scene scene;
    Scene::MaterialDescription mat{};
    mat.name = "default";
    const uint32_t matId = scene.addMaterial(mat);
    const uint32_t meshId = addQuad(scene);

    scene.createInstance(Instance::Type::eMesh, meshId, matId, glm::mat4(1.0f));
    scene.createInstance(Instance::Type::eMesh, meshId, matId,
                         glm::translate(glm::mat4(1.0f), glm::float3(10.0f, 0.0f, 0.0f)));

    glm::float3 lo(0.0f);
    glm::float3 hi(0.0f);
    REQUIRE(scene.worldBounds(lo, hi));
    CHECK(lo.x == doctest::Approx(-1.0f));
    CHECK(hi.x == doctest::Approx(11.0f));
    CHECK(lo.y == doctest::Approx(-1.0f));
    CHECK(hi.y == doctest::Approx(1.0f));
}

// The property the subsurface walk depends on: the scene's diagonal is a real
// upper bound on any distance inside it, so a free flight longer than this one
// cannot have stayed within a bounded medium. See docs/open-defects.md #15.
TEST_CASE("worldBounds diagonal bounds every distance inside the scene")
{
    Scene scene;
    Scene::MaterialDescription mat{};
    mat.name = "default";
    const uint32_t matId = scene.addMaterial(mat);
    const uint32_t meshId = addQuad(scene);

    scene.createInstance(Instance::Type::eMesh, meshId, matId,
                         glm::translate(glm::mat4(1.0f), glm::float3(-4.0f, -3.0f, 0.0f)));
    scene.createInstance(Instance::Type::eMesh, meshId, matId,
                         glm::translate(glm::mat4(1.0f), glm::float3(4.0f, 3.0f, 0.0f)));

    glm::float3 lo(0.0f);
    glm::float3 hi(0.0f);
    REQUIRE(scene.worldBounds(lo, hi));
    const float extent = glm::length(hi - lo);

    // Corner to opposite corner is the longest segment the box contains.
    CHECK(extent == doctest::Approx(glm::length(glm::float3(10.0f, 8.0f, 0.0f))));
    CHECK(glm::length(hi - lo) >= glm::length(glm::float3(hi.x - lo.x, 0.0f, 0.0f)));
}

TEST_CASE("worldBounds follows a transform change")
{
    Scene scene;
    Scene::MaterialDescription mat{};
    mat.name = "default";
    const uint32_t matId = scene.addMaterial(mat);
    const uint32_t meshId = addQuad(scene);
    const uint32_t instId = scene.createInstance(Instance::Type::eMesh, meshId, matId, glm::mat4(1.0f));

    glm::float3 lo(0.0f);
    glm::float3 hi(0.0f);
    REQUIRE(scene.worldBounds(lo, hi));
    CHECK(hi.x == doctest::Approx(1.0f));

    // Cached against the transform generation, so a move has to invalidate it --
    // a stale extent would silently re-bound the walk to the wrong world.
    scene.updateInstanceTransform(instId, glm::translate(glm::mat4(1.0f), glm::float3(5.0f, 0.0f, 0.0f)));
    REQUIRE(scene.worldBounds(lo, hi));
    CHECK(hi.x == doctest::Approx(6.0f));
}

TEST_CASE("worldBounds follows a new instance")
{
    Scene scene;
    Scene::MaterialDescription mat{};
    mat.name = "default";
    const uint32_t matId = scene.addMaterial(mat);
    const uint32_t meshId = addQuad(scene);
    scene.createInstance(Instance::Type::eMesh, meshId, matId, glm::mat4(1.0f));

    glm::float3 lo(0.0f);
    glm::float3 hi(0.0f);
    REQUIRE(scene.worldBounds(lo, hi));
    CHECK(hi.x == doctest::Approx(1.0f));

    // The reduction is cached on the transform generation, and adding an
    // instance does not move it: an instance that arrives without a transform
    // edit has to invalidate the cache on the count alone, or the scene extent
    // keeps describing the scene as it was one object ago.
    scene.createInstance(Instance::Type::eMesh, meshId, matId,
                         glm::translate(glm::mat4(1.0f), glm::float3(9.0f, 0.0f, 0.0f)));
    REQUIRE(scene.worldBounds(lo, hi));
    CHECK(hi.x == doctest::Approx(10.0f));
}

TEST_CASE("cached mesh bounds survive host geometry release")
{
    Scene scene;
    const uint32_t meshId = addQuad(scene);
    glm::float3 lo(0.0f);
    glm::float3 hi(0.0f);

    REQUIRE(scene.meshBounds(meshId, lo, hi));
    scene.releaseHostGeometry();

    REQUIRE(scene.meshBounds(meshId, lo, hi));
    CHECK(lo == glm::float3(-1.0f, -1.0f, 0.0f));
    CHECK(hi == glm::float3(1.0f, 1.0f, 0.0f));
}
