#include <doctest/doctest.h>

#include <strelka/scene/scene.h>
#include <strelka/scene/light_desc.h>
#include <strelka/sceneloader/sceneserializer.h>
#include <strelka/sceneloader/iesloader.h>
#include <light_types.h>

#include <fstream>
#include <filesystem>
#include <cmath>

using namespace oka;
namespace fs = std::filesystem;

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
    desc.outerConeAngle = float(M_PI) / 4.0f;
    desc.innerConeAngle = float(M_PI) / 8.0f;
    const uint32_t id = scene.createLight(desc);

    const Scene::Light& gpu = scene.getLights()[id];
    CHECK(gpu.type == LIGHT_TYPE_SPOT);
    CHECK(gpu.halfAngle == doctest::Approx(desc.outerConeAngle));
    CHECK(gpu.pad0 == doctest::Approx(desc.innerConeAngle));
    CHECK(glm::normalize(glm::float3(gpu.normal)).z == doctest::Approx(-1.0f).epsilon(1e-4));
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
    desc.intensity = float(M_PI);
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
    CHECK(profile.horizontalAngles.size() == 1);
    CHECK(profile.candela.size() == 3);
    CHECK(profile.maxCandela == doctest::Approx(100.0f));

    // Along -Z (nadir) → vertical 0° → 100 cd.
    CHECK(sampleIesCandela(profile, glm::float3(0.0f, 0.0f, -1.0f)) == doctest::Approx(100.0f));

    fs::remove(path);
}
