#include <doctest/doctest.h>

#include <strelka/scene/scene.h>
#include <strelka/sceneloader/sceneserializer.h>
#include <light_types.h>

#include <fstream>
#include <filesystem>
#include <cmath>

using namespace oka;
namespace fs = std::filesystem;

TEST_CASE("Light JSON round-trip preserves desc fields")
{
    Scene scene;
    Scene::UniformLightDesc rect{};
    rect.type = LIGHT_TYPE_RECT;
    rect.position = glm::float3(1.5f, 2.5f, 3.5f);
    rect.orientation = glm::float3(10.0f, 20.0f, 30.0f);
    rect.width = 2.0f;
    rect.height = 0.5f;
    rect.color = glm::float3(0.9f, 0.8f, 0.7f);
    rect.intensity = 123.0f;
    scene.createLight(rect);

    Scene::UniformLightDesc distant{};
    distant.type = LIGHT_TYPE_DISTANT;
    distant.orientation = glm::float3(-45.0f, 15.0f, 0.0f);
    distant.halfAngle = 0.53f * 0.5f * (float(M_PI) / 180.0f);
    distant.color = glm::float3(1.0f);
    distant.intensity = 50000.0f;
    scene.createLight(distant);

    Scene::EnvLightDesc env{};
    env.texturePath = "hdr/studio.exr";
    env.intensity = 1.25f;
    env.color = glm::float3(0.9f, 0.95f, 1.0f);
    env.rotationY = 42.0f;
    scene.setEnvLight(env);

    const fs::path tmp = fs::temp_directory_path() / "strelka_test_scene.gltf";
    REQUIRE(saveLightsJson(scene, tmp.string()));

    Scene loaded;
    const fs::path jsonPath = fs::temp_directory_path() / "strelka_test_scene_light.json";
    REQUIRE(loadLightsJson(loaded, jsonPath.string()));
    REQUIRE(loaded.getLightsDesc().size() == 2);

    const auto& r = loaded.getLightsDesc()[0];
    CHECK(r.type == LIGHT_TYPE_RECT);
    CHECK(r.position.x == doctest::Approx(1.5f));
    CHECK(r.width == doctest::Approx(2.0f));
    CHECK(r.height == doctest::Approx(0.5f));
    CHECK(r.intensity == doctest::Approx(123.0f));
    CHECK(r.orientation.y == doctest::Approx(20.0f));

    const auto& d = loaded.getLightsDesc()[1];
    CHECK(d.type == LIGHT_TYPE_DISTANT);
    CHECK(d.intensity == doctest::Approx(50000.0f));
    CHECK(d.halfAngle == doctest::Approx(distant.halfAngle).epsilon(1e-4));

    REQUIRE(loaded.getEnvLight().has_value());
    CHECK(loaded.getEnvLight()->texturePath == "hdr/studio.exr");
    CHECK(loaded.getEnvLight()->intensity == doctest::Approx(1.25f));
    CHECK(loaded.getEnvLight()->rotationY == doctest::Approx(42.0f));

    fs::remove(jsonPath);
}

TEST_CASE("saveGltf does not embed lights; lights stay in sidecar")
{
    Scene scene;
    Scene::MaterialDescription mat{};
    mat.name = "m";
    mat.params.base_color = glm::float3(1, 0, 0);
    scene.addMaterial(mat);

    std::vector<Scene::Vertex> vb(3);
    vb[0].pos = glm::float3(0, 0, 0);
    vb[1].pos = glm::float3(1, 0, 0);
    vb[2].pos = glm::float3(0, 1, 0);
    std::vector<uint32_t> ib = { 0, 1, 2 };
    const uint32_t meshId = scene.createMesh(vb, ib);
    scene.createInstance(Instance::Type::eMesh, meshId, 0, glm::mat4(1.0f));

    Scene::UniformLightDesc light{};
    light.type = LIGHT_TYPE_DISTANT;
    light.intensity = 10.0f;
    light.color = glm::float3(1.0f);
    scene.createLight(light);

    const fs::path gltfPath = fs::temp_directory_path() / "strelka_minimal.gltf";
    REQUIRE(saveGltf(scene, gltfPath.string()));
    REQUIRE(saveLightsJson(scene, gltfPath.string()));

    // glTF ASCII should not mention analytic light types as Strelka lights
    std::ifstream in(gltfPath);
    REQUIRE(in);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(content.find("\"meshes\"") != std::string::npos);
    // KHR_lights_punctual not used
    CHECK(content.find("KHR_lights_punctual") == std::string::npos);

    const fs::path lightPath = fs::temp_directory_path() / "strelka_minimal_light.json";
    REQUIRE(fs::exists(lightPath));

    fs::remove(gltfPath);
    fs::remove(lightPath);
    // tinygltf may write .bin
    fs::remove(fs::temp_directory_path() / "strelka_minimal.bin");
}
