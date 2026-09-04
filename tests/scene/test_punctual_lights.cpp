#include <doctest/doctest.h>

#include <strelka/scene/scene.h>
#include <strelka/scene/light_desc.h>
#include <strelka/sceneloader/sceneserializer.h>
#include <strelka/sceneloader/iesloader.h>
#include <host/light_selection.h>
#include <light_types.h>

#include <fstream>
#include <filesystem>
#include <cmath>
#include <limits>
#include <numbers>

using namespace oka;
namespace fs = std::filesystem;

namespace
{

bool packedLightIsFinite(const Scene::Light& light)
{
    auto finiteVector = [](const glm::float4& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) && std::isfinite(value.w);
    };
    for (const glm::float4& point : light.points)
    {
        if (!finiteVector(point))
        {
            return false;
        }
    }
    return finiteVector(light.color) && finiteVector(light.normal) && std::isfinite(light.halfAngle) &&
           std::isfinite(light.pad0) && std::isfinite(light.pad1);
}

} // namespace

TEST_CASE("point light bakes candela into radiant intensity")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_POINT;
    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
    desc.intensity = 40.0f;
    desc.color = glm::float3(1.0f, 0.5f, 0.25f);
    desc.position = glm::float3(1.0f, 2.0f, 3.0f);
    const uint32_t id = scene.createLight(desc);

    const Scene::Light& gpu = scene.getLights()[id];
    CHECK(gpu.type == LIGHT_TYPE_POINT);
    CHECK(gpu.color.x == doctest::Approx(40.0f));
    CHECK(gpu.color.y == doctest::Approx(20.0f));
    CHECK(gpu.color.z == doctest::Approx(10.0f));
    CHECK(glm::float3(gpu.points[1]) == desc.position);
}

TEST_CASE("spot light stores cone angles and emits along -Z")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_SPOT;
    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
    desc.intensity = 10.0f;
    desc.color = glm::float3(1.0f);
    desc.orientation = glm::float3(0.0f);
    desc.outerConeAngle = std::numbers::pi_v<float> / 4.0f;
    desc.innerConeAngle = std::numbers::pi_v<float> / 8.0f;
    const uint32_t id = scene.createLight(desc);

    const Scene::Light& gpu = scene.getLights()[id];
    CHECK(gpu.type == LIGHT_TYPE_SPOT);
    CHECK(gpu.halfAngle == doctest::Approx(desc.outerConeAngle));
    CHECK(gpu.pad0 == doctest::Approx(desc.innerConeAngle));
    CHECK(glm::normalize(glm::float3(gpu.normal)).z == doctest::Approx(-1.0f).epsilon(1e-4));
}

TEST_CASE("projector light packs its frame, its image and its axes")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_PROJECTOR;
    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
    desc.intensity = 1000.0f;
    desc.color = glm::float3(1.0f);
    desc.position = glm::float3(0.0f, 2.0f, 3.0f);
    desc.orientation = glm::float3(0.0f);
    desc.outerConeAngle = 0.35f; // half the horizontal field of view
    desc.projectorAspect = 16.0f / 9.0f;
    desc.projectorEdgeSoftness = 0.1f;
    desc.projectorImage = scene.addProjectorImage("slides/beach.png");
    const uint32_t id = scene.createLight(desc);

    const Scene::Light& gpu = scene.getLights()[id];
    CHECK(gpu.type == LIGHT_TYPE_PROJECTOR);
    CHECK(gpu.halfAngle == doctest::Approx(desc.outerConeAngle));
    CHECK(gpu.pad0 == doctest::Approx(desc.projectorEdgeSoftness));
    CHECK(gpu.points[0].z == doctest::Approx(0.0f)); // first image in the table
    CHECK(gpu.points[0].w == doctest::Approx(16.0f / 9.0f));
    CHECK(glm::float3(gpu.points[1]) == desc.position);
    // Emission along -Z with the frame's right and up axes beside it, which is
    // the basis the shader rebuilds to find a direction's place in the image.
    CHECK(glm::normalize(glm::float3(gpu.normal)).z == doctest::Approx(-1.0f).epsilon(1e-4));
    CHECK(glm::normalize(glm::float3(gpu.points[2])).x == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(glm::normalize(glm::float3(gpu.points[3])).y == doctest::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("projector image slots stay inside the registered texture table")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_PROJECTOR;
    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
    desc.intensity = 1.0f;
    desc.projectorImage = 0;
    const uint32_t id = scene.createLight(desc);

    // No texture exists yet. The packed record must request the white fallback,
    // not leave either backend an out-of-bounds bindless-table index.
    CHECK(scene.getLights()[id].points[0].z == -1.0f);

    REQUIRE(scene.addProjectorImage("slides/first.hdr") == 0);
    scene.setLight(id, desc);
    CHECK(scene.getLights()[id].points[0].z == 0.0f);

    desc.projectorImage = std::numeric_limits<int32_t>::max();
    scene.setLight(id, desc);
    CHECK(scene.getLights()[id].points[0].z == -1.0f);

    desc.projectorImage = -2;
    scene.setLight(id, desc);
    CHECK(scene.getLights()[id].points[0].z == -1.0f);
}

TEST_CASE("IES profile slots stay inside the registered table")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_POINT;
    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
    desc.iesProfile = 0;
    const uint32_t id = scene.createLight(desc);
    CHECK(scene.getLights()[id].points[0].y == -1.0f);

    Scene::IesProfile profile;
    profile.path = "profiles/first.ies";
    REQUIRE(scene.addIesProfile(profile) == 0);
    scene.setLight(id, desc);
    CHECK(scene.getLights()[id].points[0].y == 0.0f);

    desc.iesProfile = std::numeric_limits<int32_t>::max();
    scene.setLight(id, desc);
    CHECK(scene.getLights()[id].points[0].y == -1.0f);

    desc.iesProfile = -2;
    scene.setLight(id, desc);
    CHECK(scene.getLights()[id].points[0].y == -1.0f);
}

TEST_CASE("punctual packing rejects non-finite positions and collapsed profile frames")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_POINT;
    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
    desc.intensity = 10.0f;
    desc.color = glm::float3(1.0f);
    const uint32_t id = scene.createLight(desc);

    desc.useXform = true;
    desc.xform = glm::float4x4(1.0f);
    desc.xform[3][0] = std::numeric_limits<float>::infinity();
    scene.setLight(id, desc);
    CHECK(glm::float3(scene.getLights()[id].color) == glm::float3(0.0f));
    CHECK(scene.getLights()[id].normal.w == 0.0f);

    desc.type = LIGHT_TYPE_PROJECTOR;
    desc.outerConeAngle = 0.4f;
    desc.projectorAspect = 1.0f;
    desc.xform = glm::float4x4(1.0f);
    desc.xform[0][0] = 0.0f;
    scene.setLight(id, desc);
    CHECK(glm::float3(scene.getLights()[id].color) == glm::float3(0.0f));
    CHECK(scene.getLights()[id].normal.w == 0.0f);

    desc.type = LIGHT_TYPE_POINT;
    Scene::IesProfile profile;
    profile.path = "profiles/frame-validation.ies";
    REQUIRE(scene.addIesProfile(profile) == 0);
    desc.iesProfile = 0;
    scene.setLight(id, desc);
    CHECK(glm::float3(scene.getLights()[id].color) == glm::float3(0.0f));
    CHECK(scene.getLights()[id].normal.w == 0.0f);
}

TEST_CASE("non-finite light scalars produce one finite disabled record")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_PROJECTOR;
    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
    desc.intensity = 7.0f;
    desc.color = glm::float3(1.0f);
    desc.outerConeAngle = 0.4f;
    desc.projectorAspect = std::numeric_limits<float>::quiet_NaN();
    const uint32_t id = scene.createLight(desc);

    auto checkDisabled = [&]() {
        const Scene::Light& light = scene.getLights()[id];
        CHECK(packedLightIsFinite(light));
        CHECK(glm::float3(light.color) == glm::float3(0.0f));
        CHECK(light.normal.w == 0.0f);
        CHECK(oka::metal::analyticLightPower(light) == 0.0);
    };
    checkDisabled();

    desc.projectorAspect = 1.0f;
    desc.outerConeAngle = std::numeric_limits<float>::infinity();
    scene.setLight(id, desc);
    checkDisabled();

    desc.type = LIGHT_TYPE_POINT;
    desc.outerConeAngle = 0.4f;
    desc.radius = std::numeric_limits<float>::infinity();
    scene.setLight(id, desc);
    checkDisabled();

    desc.radius = 0.0f;
    desc.range = std::numeric_limits<float>::quiet_NaN();
    scene.setLight(id, desc);
    checkDisabled();

    desc.range = 0.0f;
    desc.intensity = std::numeric_limits<float>::quiet_NaN();
    scene.setLight(id, desc);
    checkDisabled();
}

TEST_CASE("sheared and mirrored projector transforms pack an orthonormal oriented frame")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_PROJECTOR;
    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
    desc.intensity = 10.0f;
    desc.color = glm::float3(1.0f);
    desc.outerConeAngle = 0.4f;
    desc.projectorAspect = 1.0f;
    desc.useXform = true;
    desc.xform = glm::float4x4(1.0f);
    desc.xform[1][0] = 1.0f;
    const uint32_t id = scene.createLight(desc);

    const auto checkFrame = [&](const Scene::Light& light) {
        const glm::float3 x(light.points[2]);
        const glm::float3 y(light.points[3]);
        const glm::float3 z(light.normal);
        CHECK(glm::length(x) == doctest::Approx(1.0f));
        CHECK(glm::length(y) == doctest::Approx(1.0f));
        CHECK(glm::length(z) == doctest::Approx(1.0f));
        CHECK(glm::dot(x, y) == doctest::Approx(0.0f).scale(1.0f).epsilon(1e-6));
        CHECK(glm::dot(x, z) == doctest::Approx(0.0f).scale(1.0f).epsilon(1e-6));
        CHECK(glm::dot(y, z) == doctest::Approx(0.0f).scale(1.0f).epsilon(1e-6));
    };
    checkFrame(scene.getLights()[id]);

    desc.xform = glm::float4x4(1.0f);
    desc.xform[0][0] = -1.0f;
    scene.setLight(id, desc);
    checkFrame(scene.getLights()[id]);
    CHECK(glm::float3(scene.getLights()[id].points[2]).x == doctest::Approx(-1.0f));
    CHECK(glm::float3(scene.getLights()[id].points[3]).y == doctest::Approx(1.0f));
    CHECK(glm::float3(scene.getLights()[id].normal).z == doctest::Approx(-1.0f));
}

TEST_CASE("a projector carries no IES profile, even one left over from a spot")
{
    // The image is the angular profile. Applying a table as well would shape the
    // beam twice, so the slot is cleared on the way to the GPU rather than left
    // holding whatever the desc still remembers from before the type changed.
    Scene scene;
    Scene::IesProfile profile;
    profile.path = "left-over.ies";
    profile.verticalAngles = { 0.0f, 90.0f };
    profile.horizontalAngles = { 0.0f, 360.0f };
    profile.candela = { 1.0f, 1.0f, 1.0f, 1.0f };
    profile.maxCandela = 1.0f;

    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_SPOT;
    desc.intensityUnit = LIGHT_UNIT_INTENSITY;
    desc.intensity = 10.0f;
    desc.color = glm::float3(1.0f);
    desc.iesProfile = scene.addIesProfile(profile);
    const uint32_t id = scene.createLight(desc);
    CHECK(scene.getLights()[id].points[0].y == doctest::Approx(0.0f));

    desc.type = LIGHT_TYPE_PROJECTOR;
    scene.setLight(id, desc);
    CHECK(scene.getLights()[id].points[0].y == doctest::Approx(-1.0f));
}

TEST_CASE("the projector image table reuses an entry for the same file")
{
    Scene scene;
    const int32_t first = scene.addProjectorImage("slides/beach.png");
    const int32_t again = scene.addProjectorImage("slides/beach.png");
    const int32_t other = scene.addProjectorImage("slides/city.png");
    CHECK(first == again);
    CHECK(other != first);
    CHECK(scene.getProjectorImages().size() == 2);
}

TEST_CASE("disabling a light zeroes its GPU contribution")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_RECT;
    desc.width = 1.0f;
    desc.height = 1.0f;
    desc.intensity = 100.0f;
    desc.color = glm::float3(1.0f);
    const uint32_t id = scene.createLight(desc);
    CHECK(glm::length(glm::float3(scene.getLights()[id].color)) > 0.0f);

    desc.enabled = false;
    scene.setLight(id, desc);
    CHECK(glm::float3(scene.getLights()[id].color) == glm::float3(0.0f));
    CHECK(any(scene.peekChanges() & ChangeBits::Lights));
}

TEST_CASE("power unit converts area lights to Lambertian radiance")
{
    // Φ =  π A  →  L = 1 for a unit-white rect of area 1.
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_RECT;
    desc.intensityUnit = LIGHT_UNIT_POWER;
    desc.width = 1.0f;
    desc.height = 1.0f;
    desc.intensity = std::numbers::pi_v<float>;
    desc.color = glm::float3(1.0f);
    const uint32_t id = scene.createLight(desc);
    CHECK(scene.getLights()[id].color.x == doctest::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("light JSON round-trips point, spot, enabled and unit")
{
    Scene scene;
    Scene::UniformLightDesc spot{};
    spot.type = LIGHT_TYPE_SPOT;
    spot.name = "key";
    spot.enabled = false;
    spot.intensityUnit = LIGHT_UNIT_INTENSITY;
    spot.intensity = 55.0f;
    spot.color = glm::float3(1.0f, 0.8f, 0.6f);
    spot.position = glm::float3(0.5f, 1.0f, -2.0f);
    spot.orientation = glm::float3(10.0f, 20.0f, 0.0f);
    spot.innerConeAngle = 0.2f;
    spot.outerConeAngle = 0.5f;
    spot.range = 12.0f;
    scene.createLight(spot);

    Scene::UniformLightDesc point{};
    point.type = LIGHT_TYPE_POINT;
    point.intensityUnit = LIGHT_UNIT_POWER;
    point.intensity = 8.0f;
    point.color = glm::float3(1.0f);
    point.position = glm::float3(0.0f);
    point.radius = 0.05f;
    scene.createLight(point);

    const fs::path tmp = fs::temp_directory_path() / "strelka_punctual.gltf";
    REQUIRE(saveLightsJson(scene, tmp.string()));

    Scene loaded;
    const fs::path jsonPath = fs::temp_directory_path() / "strelka_punctual_light.json";
    REQUIRE(loadLightsJson(loaded, jsonPath.string()));
    REQUIRE(loaded.getLightsDesc().size() == 2);

    const auto& s = loaded.getLightsDesc()[0];
    CHECK(s.type == LIGHT_TYPE_SPOT);
    CHECK(s.name == "key");
    CHECK_FALSE(s.enabled);
    CHECK(s.intensityUnit == LIGHT_UNIT_INTENSITY);
    CHECK(s.intensity == doctest::Approx(55.0f));
    CHECK(s.outerConeAngle == doctest::Approx(0.5f).epsilon(1e-4));
    CHECK(s.range == doctest::Approx(12.0f));

    const auto& p = loaded.getLightsDesc()[1];
    CHECK(p.type == LIGHT_TYPE_POINT);
    CHECK(p.intensityUnit == LIGHT_UNIT_POWER);
    CHECK(p.radius == doctest::Approx(0.05f));

    fs::remove(jsonPath);
}

TEST_CASE("IES loader reads a minimal LM-63 file")
{
    const fs::path path = fs::temp_directory_path() / "strelka_minimal.ies";
    {
        std::ofstream out(path);
        out << "IESNA:LM-63-2002\n";
        out << "TILT=NONE\n";
        // lamps, lumens, multiplier, nV, nH, phototype, units, w, l, h, ballast, unused, watts
        out << "1 1000 1.0 3 1 1 1 0 0 0 1 1 10\n";
        out << "0.0 45.0 90.0\n"; // vertical
        out << "0.0\n"; // horizontal
        out << "100.0 50.0 10.0\n"; // candela
    }

    Scene::IesProfile profile;
    REQUIRE(loadIesProfile(path.string(), profile));
    CHECK(profile.verticalAngles.size() == 3);
    // The single tabulated plane of a rotationally symmetric file is unfolded to
    // 0 and 360 at load time, so the cubic interpolation has a column on either
    // side of every direction. See unfoldIesAzimuth().
    CHECK(profile.horizontalAngles.size() == 2);
    CHECK(profile.candela.size() == 6);
    CHECK(profile.maxCandela == doctest::Approx(100.0f));

    // Along -Z (nadir) → vertical 0° → 100 cd.
    CHECK(sampleIesCandela(profile, glm::float3(0.0f, 0.0f, -1.0f)) == doctest::Approx(100.0f));

    fs::remove(path);
}
