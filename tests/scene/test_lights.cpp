#include <doctest/doctest.h>

#include <strelka/scene/scene.h>
#include <light_types.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

using namespace oka;

namespace
{

Scene::UniformLightDesc rectDesc()
{
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_RECT;
    desc.position = glm::float3(0.0f, 2.0f, 0.0f);
    desc.orientation = glm::float3(0.0f);
    desc.color = glm::float3(1.0f);
    desc.intensity = 100.0f;
    desc.width = 0.4f;
    desc.height = 0.4f;
    return desc;
}

Scene::UniformLightDesc discDesc()
{
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_DISC;
    desc.position = glm::float3(0.0f, 2.0f, 0.0f);
    desc.orientation = glm::float3(0.0f);
    desc.color = glm::float3(1.0f);
    desc.intensity = 100.0f;
    desc.radius = 0.2f;
    return desc;
}

// What the shader derives for a rect light: the negated cross product of the
// edges spanned by the stored corners.
glm::float3 rectNormalFromPoints(const Scene::Light& l)
{
    const glm::float3 e1 = glm::float3(l.points[1]) - glm::float3(l.points[0]);
    const glm::float3 e2 = glm::float3(l.points[3]) - glm::float3(l.points[0]);
    return -glm::normalize(glm::cross(e1, e2));
}

} // namespace

TEST_CASE("a light's GPU record has no uninitialized fields")
{
    // Every light is memcpy'd to the GPU whole, including the fields its own type
    // never writes.
    Scene::Light fresh;
    for (int i = 0; i < 4; ++i)
    {
        CHECK(fresh.points[i] == glm::float4(0.0f));
    }
    CHECK(fresh.normal == glm::float4(0.0f));
    CHECK(fresh.halfAngle == 0.0f);

    Scene scene;
    const uint32_t id = scene.createLight(rectDesc());
    const Scene::Light& baked = scene.getLights()[id];
    // A rect light leaves the normal alone; it must still be a finite number.
    CHECK(std::isfinite(baked.normal.x));
    CHECK(std::isfinite(baked.normal.y));
    CHECK(std::isfinite(baked.normal.z));
    CHECK(glm::length(glm::float3(baked.normal)) < 1e3f);
}

TEST_CASE("intensity and colour end up multiplied into the GPU light")
{
    Scene scene;
    Scene::UniformLightDesc desc = rectDesc();
    desc.intensity = 250.0f;
    desc.color = glm::float3(1.0f, 0.5f, 0.0f);
    const uint32_t id = scene.createLight(desc);

    const glm::float4 baked = scene.getLights()[id].color;
    CHECK(baked.x == doctest::Approx(250.0f));
    CHECK(baked.y == doctest::Approx(125.0f));
    // A saturated channel stays zero rather than being clamped up, and the shader
    // has to treat the light as emitting all the same.
    CHECK(baked.z == doctest::Approx(0.0f));

    desc.intensity = 0.0f;
    scene.setLight(id, desc);
    CHECK(scene.getLights()[id].color == glm::float4(0.0f));
    CHECK(any(scene.peekChanges() & ChangeBits::Lights));
}

// The bug: getTransform() scaled every light by (width, height, 1), and a disc
// light has no width. Its in-plane axes came out zero -- so the sampler had no
// disc to sample -- and its mesh instance was squashed flat, which made the light
// invisible as well.
TEST_CASE("a disc light's frame survives having no width")
{
    Scene scene;
    const Scene::UniformLightDesc desc = discDesc();
    const uint32_t id = scene.createLight(desc);
    const Scene::Light& l = scene.getLights()[id];

    CHECK(l.type == LIGHT_TYPE_DISC);
    CHECK(l.points[0].x == doctest::Approx(desc.radius));
    CHECK(glm::float3(l.points[1]) == desc.position);

    const glm::float3 axisX(l.points[2]);
    const glm::float3 axisY(l.points[3]);
    REQUIRE(glm::length(axisX) > 0.0f);
    REQUIRE(glm::length(axisY) > 0.0f);
    // Perpendicular, and both in the plane the normal describes: anything else is
    // not a disc.
    CHECK(glm::dot(glm::normalize(axisX), glm::normalize(axisY)) == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(glm::dot(glm::normalize(axisX), glm::float3(l.normal)) == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(glm::length(glm::float3(l.normal)) == doctest::Approx(1.0f).epsilon(1e-4));

    // The light's own geometry has to be as big as the light: a disc mesh is unit
    // sized, so the instance carries the radius.
    const uint32_t instId = scene.getLightInstanceId(id);
    REQUIRE(instId != (uint32_t)-1);
    const glm::float4x4 xform = scene.getInstances()[instId].transform;
    CHECK(glm::length(glm::float3(xform[0])) == doctest::Approx(desc.radius));
    CHECK(glm::length(glm::float3(xform[1])) == doctest::Approx(desc.radius));
}

TEST_CASE("a sphere light's mesh is scaled to its radius")
{
    Scene scene;
    Scene::UniformLightDesc desc = discDesc();
    desc.type = LIGHT_TYPE_SPHERE;
    desc.radius = 0.35f;
    const uint32_t id = scene.createLight(desc);

    const uint32_t instId = scene.getLightInstanceId(id);
    REQUIRE(instId != (uint32_t)-1);
    const glm::float4x4 xform = scene.getInstances()[instId].transform;
    CHECK(glm::length(glm::float3(xform[0])) == doctest::Approx(0.35f));
    CHECK(scene.getLights()[id].points[0].x == doctest::Approx(0.35f));
}

// A disc that emits along +Z while the rect and distant lights emit along -Z
// faces away from whatever its orientation was aimed at, so it lights nothing.
TEST_CASE("disc and rect lights emit the same way for the same orientation")
{
    Scene scene;
    const uint32_t rectId = scene.createLight(rectDesc());
    const uint32_t discId = scene.createLight(discDesc());

    const glm::float3 rectNormal = rectNormalFromPoints(scene.getLights()[rectId]);
    const glm::float3 discNormal = glm::normalize(glm::float3(scene.getLights()[discId].normal));

    CHECK(glm::length(rectNormal - discNormal) == doctest::Approx(0.0f).epsilon(1e-3));

    // And the same holds once they are turned: an orientation must move both by
    // the same amount.
    Scene::UniformLightDesc turnedRect = rectDesc();
    turnedRect.orientation = glm::float3(35.0f, -20.0f, 10.0f);
    Scene::UniformLightDesc turnedDisc = discDesc();
    turnedDisc.orientation = turnedRect.orientation;
    scene.setLight(rectId, turnedRect);
    scene.setLight(discId, turnedDisc);

    const glm::float3 turnedRectNormal = rectNormalFromPoints(scene.getLights()[rectId]);
    const glm::float3 turnedDiscNormal = glm::normalize(glm::float3(scene.getLights()[discId].normal));
    CHECK(glm::length(turnedRectNormal - turnedDiscNormal) == doctest::Approx(0.0f).epsilon(1e-3));
    // The turn actually moved them, or the check above proves nothing.
    CHECK(glm::length(turnedRectNormal - rectNormal) > 0.1f);
}

TEST_CASE("distant lights emit along their orientation too")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_DISTANT;
    desc.orientation = glm::float3(0.0f);
    desc.color = glm::float3(1.0f);
    desc.intensity = 1.0f;
    desc.halfAngle = 0.01f;
    const uint32_t id = scene.createLight(desc);

    const glm::float3 normal = glm::float3(scene.getLights()[id].normal);
    CHECK(glm::length(normal) == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(normal.z == doctest::Approx(-1.0f).epsilon(1e-4));
}

TEST_CASE("editing a rect light moves its geometry with it")
{
    Scene scene;
    Scene::UniformLightDesc desc = rectDesc();
    const uint32_t id = scene.createLight(desc);
    const uint32_t instId = scene.getLightInstanceId(id);
    REQUIRE(instId != (uint32_t)-1);

    desc.position = glm::float3(1.0f, 3.0f, -2.0f);
    desc.width = 1.2f;
    desc.height = 0.8f;
    scene.setLight(id, desc);

    const glm::float4x4 xform = scene.getInstances()[instId].transform;
    CHECK(glm::float3(xform[3]) == desc.position);
    CHECK(glm::length(glm::float3(xform[0])) == doctest::Approx(1.2f));
    CHECK(glm::length(glm::float3(xform[1])) == doctest::Approx(0.8f));

    // The sampled shape has to follow the geometry, not just the instance.
    const Scene::Light& l = scene.getLights()[id];
    const glm::float3 e1 = glm::float3(l.points[1]) - glm::float3(l.points[0]);
    const glm::float3 e2 = glm::float3(l.points[3]) - glm::float3(l.points[0]);
    CHECK(glm::length(e1) == doctest::Approx(1.2f));
    CHECK(glm::length(e2) == doctest::Approx(0.8f));
    CHECK(glm::length(glm::cross(e1, e2)) == doctest::Approx(1.2f * 0.8f));
}
