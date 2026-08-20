#include <doctest/doctest.h>

#include <strelka/scene/scene.h>
#include <strelka/sceneloader/sceneserializer.h>
#include <light_types.h>

#include <fstream>
#include <filesystem>
#include <cmath>
#include <numbers>

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
    // A light that lights the scene without being in frame. Written only when
    // false, so this also pins that the sidecar stays quiet about the default.
    rect.visibleToCamera = false;
    // Cached on the radiance cache's short clock. Written only when true, so
    // this pins both halves of that: it survives the round trip, and the light
    // below -- which never asks for it -- comes back false rather than picking
    // up whatever the previous entry set.
    rect.responsive = true;
    scene.createLight(rect);

    Scene::UniformLightDesc distant{};
    distant.type = LIGHT_TYPE_DISTANT;
    distant.orientation = glm::float3(-45.0f, 15.0f, 0.0f);
    distant.halfAngle = 0.53f * 0.5f * (std::numbers::pi_v<float> / 180.0f);
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
    CHECK(r.visibleToCamera == false);
    CHECK(r.responsive == true);

    const auto& d = loaded.getLightsDesc()[1];
    CHECK(d.type == LIGHT_TYPE_DISTANT);
    CHECK(d.intensity == doctest::Approx(50000.0f));
    CHECK(d.halfAngle == doctest::Approx(distant.halfAngle).epsilon(1e-4));
    // Absent from the JSON entirely; the loader has to default it to visible.
    CHECK(d.visibleToCamera == true);
    CHECK(d.responsive == false);

    // Bound once rather than re-fetched: getEnvLight() returns by value, so
    // each `->` was a fresh optional the has_value() above had never seen --
    // which is what bugprone-unchecked-optional-access was reporting, and it
    // was right that the guard did not guard these three.
    // Bound once rather than re-fetched: getEnvLight() returns by value, so each
    // `loaded.getEnvLight()->` was a fresh optional that the REQUIRE had never
    // seen. The `if` is not redundant with the REQUIRE either -- REQUIRE is a
    // macro the analyser cannot read as a guard, so without the branch every
    // access below is an unchecked one. REQUIRE still owns the failure message;
    // the branch only tells the analyser what REQUIRE already guarantees.
    const auto loadedEnv = loaded.getEnvLight();
    REQUIRE(loadedEnv.has_value());
    if (loadedEnv.has_value())
    {
        const Scene::EnvLightDesc& readBack = *loadedEnv;
        CHECK(readBack.texturePath == "hdr/studio.exr");
        CHECK(readBack.intensity == doctest::Approx(1.25f));
        CHECK(readBack.rotationY == doctest::Approx(42.0f));
    }

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
    const std::vector<uint32_t> ib = { 0, 1, 2 };
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
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
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

TEST_CASE("a projector round-trips its frame and its image through the sidecar")
{
    Scene scene;
    Scene::UniformLightDesc projector{};
    projector.type = LIGHT_TYPE_PROJECTOR;
    projector.name = "beamer";
    projector.intensityUnit = LIGHT_UNIT_POWER;
    projector.intensity = 250.0f;
    projector.color = glm::float3(1.0f);
    projector.position = glm::float3(0.0f, 1.8f, 4.0f);
    projector.orientation = glm::float3(0.0f, 180.0f, 0.0f);
    // 40 degrees of horizontal field, stored as its half angle.
    projector.outerConeAngle = 20.0f * (std::numbers::pi_v<float> / 180.0f);
    projector.projectorAspect = 16.0f / 9.0f;
    projector.projectorEdgeSoftness = 0.05f;
    projector.projectorImagePath = "slides/beach.png";
    projector.range = 20.0f;
    scene.createLight(projector);

    const fs::path tmp = fs::temp_directory_path() / "strelka_projector.gltf";
    REQUIRE(saveLightsJson(scene, tmp.string()));

    Scene loaded;
    const fs::path jsonPath = fs::temp_directory_path() / "strelka_projector_light.json";
    REQUIRE(loadLightsJson(loaded, jsonPath.string()));
    REQUIRE(loaded.getLightsDesc().size() == 1);

    const auto& p = loaded.getLightsDesc()[0];
    CHECK(p.type == LIGHT_TYPE_PROJECTOR);
    CHECK(p.name == "beamer");
    CHECK(p.intensityUnit == LIGHT_UNIT_POWER);
    CHECK(p.intensity == doctest::Approx(250.0f));
    CHECK(p.outerConeAngle == doctest::Approx(20.0f * (std::numbers::pi_v<float> / 180.0f)).epsilon(1e-4));
    CHECK(p.projectorAspect == doctest::Approx(16.0f / 9.0f));
    CHECK(p.projectorEdgeSoftness == doctest::Approx(0.05f));
    CHECK(p.range == doctest::Approx(20.0f));

    // The path is resolved against the sidecar's own directory and registered in
    // the scene's image table, which is what the renderer uploads from and what
    // the light's points[0].z indexes.
    CHECK(fs::path(p.projectorImagePath).is_absolute());
    CHECK(fs::path(p.projectorImagePath).filename() == "beach.png");
    CHECK(p.projectorImage == 0);
    REQUIRE(loaded.getProjectorImages().size() == 1);
    CHECK(loaded.getProjectorImages()[0] == p.projectorImagePath);
    CHECK(loaded.getLights()[0].points[0].z == doctest::Approx(0.0f));

    fs::remove(jsonPath);
}

TEST_CASE("a sidecar names a projector's field of view, not half of it")
{
    // The one place the projector and the spot disagree about what an angle in
    // the file means, so it is worth stating outright: a spot writes its outer
    // *half* angle and a projector writes the full horizontal field, because
    // nobody describes a beamer by half its throw angle.
    Scene scene;
    Scene::UniformLightDesc projector{};
    projector.type = LIGHT_TYPE_PROJECTOR;
    projector.intensity = 1.0f;
    projector.color = glm::float3(1.0f);
    projector.outerConeAngle = 30.0f * (std::numbers::pi_v<float> / 180.0f);
    scene.createLight(projector);

    const fs::path tmp = fs::temp_directory_path() / "strelka_projector_fov.gltf";
    REQUIRE(saveLightsJson(scene, tmp.string()));

    const fs::path jsonPath = fs::temp_directory_path() / "strelka_projector_fov_light.json";
    std::ifstream in(jsonPath);
    REQUIRE(in);
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(content.find("\"fov\"") != std::string::npos);
    CHECK(content.find("\"projector\"") != std::string::npos);
    // A projector has no cone, so the spot's half-angle keys must not appear at
    // all -- finding both would mean two ways to say the same thing.
    CHECK(content.find("outerConeAngle") == std::string::npos);

    Scene loaded;
    REQUIRE(loadLightsJson(loaded, jsonPath.string()));
    REQUIRE(loaded.getLightsDesc().size() == 1);
    CHECK(loaded.getLightsDesc()[0].outerConeAngle ==
          doctest::Approx(30.0f * (std::numbers::pi_v<float> / 180.0f)).epsilon(1e-4));

    fs::remove(jsonPath);
}
