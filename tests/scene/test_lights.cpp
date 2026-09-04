#include <doctest/doctest.h>

#include <strelka/scene/scene.h>
#include <analytic_light.h>
#include <light_types.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <numbers>

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

bool accelerationStructureTransformIsSafe(const glm::float4x4& transform)
{
    for (int column = 0; column < 4; ++column)
    {
        for (int row = 0; row < 4; ++row)
        {
            if (!std::isfinite(transform[column][row]))
            {
                return false;
            }
        }
    }
    return std::abs(glm::determinant(glm::dmat3(transform))) > 0.0;
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
    const Scene::Light fresh;
    for (auto point : fresh.points)
    {
        CHECK(point == glm::float4(0.0f));
    }
    CHECK(fresh.normal == glm::float4(0.0f));
    CHECK(fresh.halfAngle == 0.0f);

    Scene scene;
    const uint32_t id = scene.createLight(rectDesc());
    const Scene::Light& baked = scene.getLights()[id];
    // Rect sampling uses this packed inverse-transpose normal.
    CHECK(std::isfinite(baked.normal.x));
    CHECK(std::isfinite(baked.normal.y));
    CHECK(std::isfinite(baked.normal.z));
    CHECK(glm::length(glm::float3(baked.normal)) == doctest::Approx(1.0f));
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
    CHECK(glm::float3(scene.getLights()[id].color) == glm::float3(0.0f));
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

TEST_CASE("a sphere light record preserves every affine analytic axis")
{
    Scene scene;
    Scene::UniformLightDesc desc = discDesc();
    desc.type = LIGHT_TYPE_SPHERE;
    desc.radius = 0.5f;
    desc.useXform = true;
    desc.xform = glm::scale(glm::mat4(1.0f), glm::vec3(-2.0f, 3.0f, 4.0f));
    const uint32_t id = scene.createLight(desc);

    const Scene::Light& light = scene.getLights()[id];
    CHECK(glm::vec3(light.points[0]) == glm::vec3(-1.0f, 0.0f, 0.0f));
    CHECK(glm::vec3(light.points[2]) == glm::vec3(0.0f, 1.5f, 0.0f));
    CHECK(glm::vec3(light.points[3]) == glm::vec3(0.0f, 0.0f, 2.0f));
}

TEST_CASE("power units use the transformed analytic surface area")
{
    const auto emittedPower = [](const Scene::Light& light) {
        float area = 0.0f;
        if (light.type == LIGHT_TYPE_RECT)
        {
            const glm::vec3 edgeX = glm::vec3(light.points[1] - light.points[0]);
            const glm::vec3 edgeY = glm::vec3(light.points[3] - light.points[0]);
            area = finiteVectorLength(glm::cross(edgeX, edgeY));
        }
        else if (light.type == LIGHT_TYPE_DISC)
        {
            area = analyticDiscArea(glm::vec3(light.points[2]), glm::vec3(light.points[3]));
        }
        else
        {
            area = analyticEllipsoidSurfaceArea(glm::vec3(light.points[0]), glm::vec3(light.points[2]),
                                                 glm::vec3(light.points[3]));
        }
        return std::numbers::pi_v<float> * area * light.color.x;
    };

    for (const int type : { LIGHT_TYPE_RECT, LIGHT_TYPE_DISC, LIGHT_TYPE_SPHERE })
    {
        Scene scene;
        Scene::UniformLightDesc desc = rectDesc();
        desc.type = type;
        desc.intensityUnit = LIGHT_UNIT_POWER;
        desc.intensity = 600.0f;
        desc.width = 2.0f;
        desc.height = 3.0f;
        desc.radius = 0.5f;
        desc.useXform = true;
        desc.xform = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 1.0f));
        const Scene::Light& light = scene.getLights()[scene.createLight(desc)];

        CAPTURE(type);
        CHECK(emittedPower(light) == doctest::Approx(desc.intensity).epsilon(2e-4));
    }

    Scene radianceScene;
    Scene::UniformLightDesc radiance = rectDesc();
    radiance.intensityUnit = LIGHT_UNIT_RADIANCE;
    radiance.useXform = true;
    radiance.xform = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 1.0f));
    const Scene::Light& light = radianceScene.getLights()[radianceScene.createLight(radiance)];
    CHECK(light.color.x == doctest::Approx(radiance.intensity));
}

TEST_CASE("a sheared mirrored disc uses its inverse-transpose emission normal")
{
    Scene scene;
    Scene::UniformLightDesc desc = discDesc();
    desc.useXform = true;
    desc.xform = glm::scale(glm::mat4(1.0f), glm::vec3(-2.0f, 3.0f, 1.0f));
    desc.xform[2][0] = 1.0f; // shear local Z into world X; the disc itself remains in Z=0
    const uint32_t id = scene.createLight(desc);

    const glm::vec3 normal(scene.getLights()[id].normal);
    CHECK(normal.x == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(normal.y == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(normal.z == doctest::Approx(-1.0f).epsilon(1e-5));
}

TEST_CASE("a rectangle mirrored through its plane keeps inverse-transpose orientation")
{
    Scene identityScene;
    Scene::UniformLightDesc identity = rectDesc();
    identity.useXform = true;
    identity.xform = glm::mat4(1.0f);
    const Scene::Light& ordinary = identityScene.getLights()[identityScene.createLight(identity)];

    Scene mirroredScene;
    Scene::UniformLightDesc mirrored = identity;
    mirrored.xform = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, -1.0f));
    const Scene::Light& reflected = mirroredScene.getLights()[mirroredScene.createLight(mirrored)];

    CHECK(glm::vec3(ordinary.normal) == glm::vec3(0.0f, 0.0f, -1.0f));
    CHECK(glm::vec3(reflected.normal) == glm::vec3(0.0f, 0.0f, 1.0f));
    for (size_t i = 0; i < 4; ++i)
    {
        CHECK(ordinary.points[i] == reflected.points[i]);
    }
    // Mutation: reconstructing orientation from the coplanar points cannot
    // distinguish these two transforms.
    CHECK(rectNormalFromPoints(ordinary) == rectNormalFromPoints(reflected));
}

TEST_CASE("singular transforms produce finite invalid directional-light records")
{
    for (const int type : { LIGHT_TYPE_SPOT, LIGHT_TYPE_PROJECTOR, LIGHT_TYPE_DISTANT })
    {
        Scene scene;
        Scene::UniformLightDesc desc = discDesc();
        desc.type = type;
        desc.useXform = true;
        desc.xform = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, 0.0f));
        desc.outerConeAngle = 0.4f;
        desc.halfAngle = 0.1f;
        const Scene::Light& light = scene.getLights()[scene.createLight(desc)];
        CHECK(glm::vec3(light.normal) == glm::vec3(0.0f));
        CHECK(std::isfinite(light.normal.x));
        CHECK(std::isfinite(light.normal.y));
        CHECK(std::isfinite(light.normal.z));
    }
}

TEST_CASE("disc normal ignores a large irrelevant normal-axis scale")
{
    Scene scene;
    Scene::UniformLightDesc desc = discDesc();
    desc.radius = 0.2f;
    desc.useXform = true;
    desc.xform = glm::scale(glm::mat4(1.0f), glm::vec3(100.0f, 100.0f, 3e38f));
    const Scene::Light& light = scene.getLights()[scene.createLight(desc)];
    CHECK(glm::vec3(light.normal) == glm::vec3(0.0f, 0.0f, -1.0f));
    CHECK(analyticDiscArea(glm::vec3(light.points[2]), glm::vec3(light.points[3])) > 0.0f);

    const AnalyticLightIntersection hit = intersectAnalyticDisc(
        glm::vec3(light.points[1]) - 2.0f * glm::vec3(light.normal), glm::vec3(light.normal), 0.0f, 10.0f,
        glm::vec3(light.points[1]), glm::vec3(light.points[2]), glm::vec3(light.points[3]), glm::vec3(light.normal));
    CHECK(hit.hit);
}

TEST_CASE("analytic light visibility is packed for manual traversal")
{
    Scene scene;
    Scene::UniformLightDesc visible = discDesc();
    visible.visibleToCamera = true;
    const uint32_t visibleId = scene.createLight(visible);
    CHECK(uint32_t(scene.getLights()[visibleId].normal.w) ==
          (STRELKA_ANALYTIC_LIGHT_CAMERA_BIT | STRELKA_ANALYTIC_LIGHT_SECONDARY_BIT));

    Scene::UniformLightDesc hidden = visible;
    hidden.visibleToCamera = false;
    const uint32_t hiddenId = scene.createLight(hidden);
    CHECK(uint32_t(scene.getLights()[hiddenId].normal.w) == STRELKA_ANALYTIC_LIGHT_SECONDARY_BIT);

    Scene::UniformLightDesc disabled = visible;
    disabled.enabled = false;
    const uint32_t disabledId = scene.createLight(disabled);
    CHECK(scene.getLights()[disabledId].normal.w == 0.0f);
}

TEST_CASE("CPU picking intersects smooth transformed analytic lights")
{
    for (const int type : { LIGHT_TYPE_DISC, LIGHT_TYPE_SPHERE, LIGHT_TYPE_POINT, LIGHT_TYPE_SPOT, LIGHT_TYPE_PROJECTOR })
    {
        Scene scene;
        Scene::UniformLightDesc desc = discDesc();
        desc.type = type;
        desc.radius = 0.5f;
        desc.useXform = true;
        desc.xform = glm::scale(glm::mat4(1.0f), glm::vec3(-2.0f, 3.0f, 4.0f));
        desc.xform[2][0] = 0.25f;
        const uint32_t lightId = scene.createLight(desc);
        const Scene::Light& light = scene.getLights()[lightId];
        const glm::vec3 center(light.points[1]);
        glm::vec3 target;
        glm::vec3 origin;
        if (type == LIGHT_TYPE_DISC)
        {
            const float angle = std::numbers::pi_v<float> / 16.0f;
            // At the midpoint of a 16-gon edge, r=0.99 lies on the smooth disc
            // but outside the editor proxy (whose radius there is cos(pi/16)).
            target = center + 0.99f * (std::cos(angle) * glm::vec3(light.points[2]) +
                                       std::sin(angle) * glm::vec3(light.points[3]));
            origin = target - 3.0f * glm::vec3(light.normal);
        }
        else if (type == LIGHT_TYPE_SPHERE)
        {
            const glm::vec3 q = glm::normalize(glm::vec3(0.31f, 0.47f, 0.73f));
            const glm::vec3 radial =
                q.x * glm::vec3(light.points[0]) + q.y * glm::vec3(light.points[2]) + q.z * glm::vec3(light.points[3]);
            target = center + radial;
            origin = center + 2.0f * radial;
        }
        else
        {
            // Soft punctual lights sample a world-space sphere. Their editor
            // proxy is an affine sphere for point and a disc for spot/projector,
            // neither of which is the radiometric surface under this transform.
            const glm::vec3 q = glm::normalize(glm::vec3(0.31f, 0.47f, 0.73f));
            const glm::vec3 radial = desc.radius * q;
            target = center + radial;
            origin = center + 2.0f * radial;
        }
        const Scene::PickHit hit = scene.pick(origin, glm::normalize(target - origin));
        REQUIRE(hit.hit);
        CHECK(hit.lightId == lightId);
        CHECK(glm::length(hit.position - target) < 1e-4f);
    }
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

TEST_CASE("a dome survives scene packing as an infinite light")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_DOME;
    desc.color = glm::float3(0.25f, 0.5f, 1.0f);
    desc.intensity = 3.0f;
    const uint32_t id = scene.createLight(desc);

    const Scene::Light& light = scene.getLights()[id];
    CHECK(light.type == LIGHT_TYPE_DOME);
    CHECK(glm::float3(light.color) == desc.color * desc.intensity);
    CHECK(scene.getLightInstanceId(id) == uint32_t(-1));
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

TEST_CASE("changing a light type replaces or creates the matching proxy topology")
{
    Scene scene;
    Scene::UniformLightDesc desc = discDesc();
    desc.type = LIGHT_TYPE_SPHERE;
    const uint32_t id = scene.createLight(desc);
    const uint32_t instanceId = scene.getLightInstanceId(id);
    REQUIRE(instanceId != kInvalidIndex);
    const uint32_t sphereMeshId = scene.getInstances()[instanceId].mMeshId;

    desc = rectDesc();
    scene.setLight(id, desc);
    const uint32_t rectMeshId = scene.getInstances()[instanceId].mMeshId;
    CHECK(rectMeshId != sphereMeshId);
    CHECK(scene.getMeshes()[rectMeshId].mCount == 6u);
    CHECK(any(scene.peekChanges() & ChangeBits::Geometry));

    Scene infiniteScene;
    Scene::UniformLightDesc distant = desc;
    distant.type = LIGHT_TYPE_DISTANT;
    const uint32_t distantId = infiniteScene.createLight(distant);
    REQUIRE(infiniteScene.getLightInstanceId(distantId) == kInvalidIndex);
    infiniteScene.setLight(distantId, desc);
    const uint32_t createdInstanceId = infiniteScene.getLightInstanceId(distantId);
    REQUIRE(createdInstanceId != kInvalidIndex);
    CHECK(infiniteScene.getMeshes()[infiniteScene.getInstances()[createdInstanceId].mMeshId].mCount == 6u);

    const size_t proxyMeshCount = infiniteScene.getMeshes().size();
    for (int i = 0; i < 100; ++i)
    {
        desc.width += 0.01f;
        infiniteScene.setLight(distantId, desc);
    }
    CHECK(infiniteScene.getMeshes().size() == proxyMeshCount);

    // Mutation: updating only the transform leaves the original sphere mesh.
    CHECK(sphereMeshId != rectMeshId);
}

TEST_CASE("changing a finite light to infinite deactivates its editor proxy")
{
    Scene scene;
    const Scene::UniformLightDesc finite = rectDesc();
    const uint32_t lightId = scene.createLight(finite);
    const uint32_t proxyId = scene.getLightInstanceId(lightId);
    REQUIRE(proxyId != kInvalidIndex);

    Scene::UniformLightDesc infinite = finite;
    infinite.type = LIGHT_TYPE_DISTANT;
    infinite.halfAngle = 0.05f;
    scene.setLight(lightId, infinite);
    CHECK(scene.getLightInstanceId(lightId) == kInvalidIndex);
    CHECK_FALSE(scene.pick(glm::float3(0.0f, 2.0f, 2.0f), glm::float3(0.0f, 0.0f, -1.0f)).hit);

    infinite.type = LIGHT_TYPE_DOME;
    scene.setLight(lightId, infinite);
    CHECK(scene.getLightInstanceId(lightId) == kInvalidIndex);
    CHECK_FALSE(scene.pick(glm::float3(0.0f, 2.0f, 2.0f), glm::float3(0.0f, 0.0f, -1.0f)).hit);

    // Returning to a finite type reuses the cached editor instance instead of
    // accumulating an orphan proxy for every type toggle.
    scene.setLight(lightId, finite);
    CHECK(scene.getLightInstanceId(lightId) == proxyId);
    const Scene::PickHit restored = scene.pick(glm::float3(0.0f, 2.0f, 2.0f), glm::float3(0.0f, 0.0f, -1.0f));
    REQUIRE(restored.hit);
    CHECK(restored.lightId == lightId);
}

TEST_CASE("invalid analytic lights keep safe acceleration structure transforms")
{
    Scene created;
    Scene::UniformLightDesc invalid = rectDesc();
    invalid.useXform = true;
    invalid.xform = glm::float4x4(1.0f);
    invalid.xform[3][0] = std::numeric_limits<float>::infinity();
    const uint32_t createdLight = created.createLight(invalid);
    const uint32_t createdProxy = created.getLightInstanceId(createdLight);
    REQUIRE(createdProxy != kInvalidIndex);
    CHECK(accelerationStructureTransformIsSafe(created.getInstances()[createdProxy].transform));

    Scene edited;
    const uint32_t editedLight = edited.createLight(rectDesc());
    const uint32_t editedProxy = edited.getLightInstanceId(editedLight);
    REQUIRE(editedProxy != kInvalidIndex);
    invalid.xform = glm::float4x4(1.0f);
    invalid.xform[0][0] = 0.0f;
    edited.setLight(editedLight, invalid);
    CHECK(accelerationStructureTransformIsSafe(edited.getInstances()[editedProxy].transform));

    // Mutation: the authored matrices themselves remain invalid; the check is
    // sensitive to accidentally publishing either raw transform to a TLAS.
    CHECK_FALSE(accelerationStructureTransformIsSafe(invalid.xform));
}

TEST_CASE("headless light edits do not recreate released proxy geometry")
{
    Scene scene;
    Scene::UniformLightDesc desc = rectDesc();
    desc.type = LIGHT_TYPE_DISTANT;
    const uint32_t id = scene.createLight(desc);
    REQUIRE(scene.getLightInstanceId(id) == kInvalidIndex);

    scene.consumeChanges();
    scene.releaseHostGeometry();
    const size_t meshCount = scene.getMeshes().size();

    desc = rectDesc();
    scene.setLight(id, desc);

    const ChangeBits changes = scene.peekChanges();
    CHECK(any(changes & ChangeBits::Lights));
    CHECK_FALSE(any(changes & ChangeBits::Geometry));
    CHECK_FALSE(any(changes & ChangeBits::Transforms));
    CHECK(scene.getLightInstanceId(id) == kInvalidIndex);
    CHECK(scene.getMeshes().size() == meshCount);
    CHECK(scene.getVertices().empty());
    CHECK(scene.getIndices().empty());

    // Headless rendering uses the packed analytic primitive, not an editor
    // tessellation. A topology-free edit must still leave the exact sampled
    // surface intersectable.
    const Scene::Light& light = scene.getLights()[id];
    const glm::float3 center = 0.25f * (glm::float3(light.points[0]) + glm::float3(light.points[1]) +
                                        glm::float3(light.points[2]) + glm::float3(light.points[3]));
    const glm::float3 normal(light.normal);
    const AnalyticLightIntersection hit = intersectAnalyticLightSurface(
        light.type, glm::float3(light.points[0]), glm::float3(light.points[1]), glm::float3(light.points[2]),
        glm::float3(light.points[3]), normal, center - 2.0f * normal, normal, 0.0f, 10.0f);
    CHECK(hit.hit);
    CHECK(hit.areaPdf > 0.0f);

    Scene addedScene;
    addedScene.releaseHostGeometry();
    const uint32_t addedId = addedScene.createLight(rectDesc());
    const ChangeBits addedChanges = addedScene.peekChanges();
    CHECK(any(addedChanges & ChangeBits::Lights));
    CHECK_FALSE(any(addedChanges & ChangeBits::Geometry));
    CHECK(addedScene.getLightInstanceId(addedId) == kInvalidIndex);
    CHECK(addedScene.getMeshes().empty());
    CHECK(addedScene.getVertices().empty());
    CHECK(addedScene.getIndices().empty());
}

TEST_CASE("a rect or disc light carries its area density rather than rebuilding it")
{
    // The density is a constant of the light, and the shading path used to
    // rebuild it per next-event draw out of exponent-decomposed compensated
    // arithmetic. It now travels in pad0, which those two types write and never
    // read -- so this is what says the number packed there is the number the
    // device would have computed. Both halves of the MIS estimate read it, so a
    // wrong value does not cancel: it biases.
    Scene scene;

    Scene::UniformLightDesc rect{};
    rect.type = LIGHT_TYPE_RECT;
    rect.intensityUnit = LIGHT_UNIT_INTENSITY;
    rect.intensity = 5.0f;
    rect.color = glm::float3(1.0f);
    rect.width = 0.7f;
    rect.height = 0.3f;
    const Scene::Light& packedRect = scene.getLights()[scene.createLight(rect)];
    REQUIRE(packedRect.type == LIGHT_TYPE_RECT);
    CHECK(packedRect.pad0 ==
          doctest::Approx(inverseFiniteCrossLength(glm::float3(packedRect.points[1] - packedRect.points[0]),
                                                    glm::float3(packedRect.points[3] - packedRect.points[0]))));
    // 1 / area, and the area is what was authored.
    CHECK(packedRect.pad0 == doctest::Approx(1.0f / (0.7f * 0.3f)).epsilon(1e-4));

    Scene::UniformLightDesc disc{};
    disc.type = LIGHT_TYPE_DISC;
    disc.intensityUnit = LIGHT_UNIT_INTENSITY;
    disc.intensity = 5.0f;
    disc.color = glm::float3(1.0f);
    disc.radius = 0.25f;
    const Scene::Light& packedDisc = scene.getLights()[scene.createLight(disc)];
    REQUIRE(packedDisc.type == LIGHT_TYPE_DISC);
    CHECK(packedDisc.pad0 == doctest::Approx(analyticDiscAreaPdf(glm::float3(packedDisc.points[2]),
                                                                 glm::float3(packedDisc.points[3]))));
    CHECK(packedDisc.pad0 == doctest::Approx(1.0f / (std::numbers::pi_v<float> * 0.25f * 0.25f)).epsilon(1e-4));
}
