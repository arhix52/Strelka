
#include <doctest/doctest.h>

#include <strelka/material/openpbr/openpbr_params.h>
#include <strelka/sceneloader/gltfloader.h>
#include <strelka/sceneloader/material_sidecar.h>

#include <cmath>
#include <filesystem>
#include <fstream>

using nlohmann::json;

TEST_CASE("glTF transmission without a volume is thin-walled")
{
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "strelka_transmission_thickness.gltf";
    {
        std::ofstream out(path);
        out << R"({
            "asset":{"version":"2.0"},
            "extensionsUsed":["KHR_materials_transmission","KHR_materials_volume"],
            "materials":[
                {"name":"thin","extensions":{"KHR_materials_transmission":{"transmissionFactor":1}}},
                {"name":"solid","extensions":{"KHR_materials_transmission":{"transmissionFactor":1},
                                                   "KHR_materials_volume":{"thicknessFactor":1}}},
                {"name":"opaque"},
                {"name":"shadow","extensions":{"STRELKA_materials_shadow_transparent":{}}},
                {"name":"backplate","emissiveFactor":[1,1,1],
                 "normalTexture":{"index":0},"emissiveTexture":{"index":0},
                 "pbrMetallicRoughness":{"baseColorTexture":{"index":0},
                                           "metallicRoughnessTexture":{"index":0}},"extensions":{
                    "KHR_materials_emissive_strength":{"emissiveStrength":7}}}
            ],
            "images":[{"uri":"texture.png"}],"textures":[{"source":0}],
            "scenes":[{"nodes":[]}],"scene":0
        })";
    }

    oka::Scene scene;
    oka::GltfLoader loader;
    REQUIRE(loader.loadGltf(path.string(), scene));
    REQUIRE(scene.getMaterials().size() == 5);
    CHECK(scene.getMaterials()[0].params.thin_walled == 1u);
    CHECK(scene.getMaterials()[1].params.thin_walled == 0u);
    CHECK(scene.getMaterials()[2].params.thin_walled == 0u);
    CHECK(scene.getMaterials()[3].params.alpha_mode == ALPHA_MODE_SHADOW_TRANSPARENT);
    CHECK(scene.getMaterials()[4].params.emission_strength == doctest::Approx(7.0f));
    CHECK(scene.getMaterials()[4].openpbrTexPaths[OPENPBR_TEX_BASE_COLOR] == "texture.png");
    CHECK(scene.getMaterials()[4].openpbrTexPaths[OPENPBR_TEX_SPECULAR_ROUGHNESS] == "texture.png");
    CHECK(scene.getMaterials()[4].openpbrTexPaths[OPENPBR_TEX_GEOMETRY_NORMAL] == "texture.png");
    CHECK(scene.getMaterials()[4].openpbrTexPaths[OPENPBR_TEX_EMISSION_COLOR] == "texture.png");
    std::filesystem::remove(path);
}

TEST_CASE("an omitted key keeps the OpenPBR default rather than going to zero")
{
    OpenPBRParams p = openpbr_make_default_params();
    const json j = json::parse(R"({ "base_color": [0.1, 0.2, 0.3] })");
    CHECK(oka::materialsidecar::parseOpenPBR(j, p) == 0);

    CHECK(p.base_color.r == doctest::Approx(0.1f));
    CHECK(p.base_color.g == doctest::Approx(0.2f));
    CHECK(p.base_color.b == doctest::Approx(0.3f));

    // Everything else must still be the spec default. These four are the ones
    // whose zero is degenerate rather than merely dark.
    CHECK(p.specular_ior == doctest::Approx(1.5f));
    CHECK(p.coat_ior == doctest::Approx(1.6f));
    CHECK(p.coat_darkening == doctest::Approx(1.0f));
    CHECK(p.specular_anisotropy_rotation_cos == doctest::Approx(1.0f));
    CHECK(p.base_weight == doctest::Approx(1.0f));
    CHECK(p.specular_roughness == doctest::Approx(0.3f));
}

TEST_CASE("an OpenPBR sidecar keeps the glTF normal scale")
{
    const std::filesystem::path stem =
        std::filesystem::temp_directory_path() / "strelka_sidecar_normal_scale";
    const std::filesystem::path gltf = stem.string() + ".gltf";
    const std::filesystem::path sidecar = stem.string() + "_openpbr.json";
    {
        std::ofstream out(gltf);
        out << R"({"asset":{"version":"2.0"},"materials":[{"name":"leather","normalTexture":{"index":0,"scale":0.15}}],"images":[{"uri":"normal.png"}],"textures":[{"source":0}],"scenes":[{"nodes":[]}],"scene":0})";
    }
    {
        std::ofstream out(sidecar);
        out << R"({"version":1,"materials":[{"gltfMaterial":"leather","openpbr":{},"textures":{"geometry_normal":"normal.png"}}]})";
    }

    oka::Scene scene;
    oka::GltfLoader loader;
    REQUIRE(loader.loadGltf(gltf.string(), scene));
    REQUIRE(scene.getMaterials().size() == 1);
    CHECK(scene.getMaterials()[0].openpbr.texture_normal_scale == doctest::Approx(0.15f));
    std::filesystem::remove(gltf);
    std::filesystem::remove(sidecar);
}

TEST_CASE("an OpenPBR sidecar can select packed glTF roughness")
{
    OpenPBRParams p = openpbr_make_default_params();
    std::array<std::string, MAX_OPENPBR_TEXTURES> paths{};
    const json j = json::parse(
        R"({"specular_roughness":{"path":"metallic_roughness.png","channel":"g"}})");

    CHECK(oka::materialsidecar::parseTextures(j, paths, p) == 0);
    CHECK(paths[OPENPBR_TEX_SPECULAR_ROUGHNESS] == "metallic_roughness.png");
    CHECK((p.texture_scalar_flags & OPENPBR_ROUGHNESS_CHANNEL_MASK) == 1u);
}

TEST_CASE("the parameters glTF cannot reach are settable here")
{
    // The reason the format exists. Each of these is left at its default by
    // openpbr_from_gltf.h because no glTF extension carries it.
    OpenPBRParams p = openpbr_make_default_params();
    const json j = json::parse(R"({
        "base_diffuse_roughness": 0.9,
        "coat_darkening": 0.4,
        "coat_roughness_anisotropy": 0.25,
        "fuzz_weight": 1.0,
        "fuzz_roughness": 0.3,
        "transmission_dispersion_scale": 1.0,
        "transmission_dispersion_abbe_number": 35.0,
        "transmission_scatter": [0.1, 0.2, 0.3],
        "subsurface_radius_scale": [1.0, 0.4, 0.2],
        "geometry_thin_walled": true
    })");
    CHECK(oka::materialsidecar::parseOpenPBR(j, p) == 0);

    CHECK(p.base_diffuse_roughness == doctest::Approx(0.9f));
    CHECK(p.coat_darkening == doctest::Approx(0.4f));
    CHECK(p.coat_roughness_anisotropy == doctest::Approx(0.25f));
    CHECK(p.fuzz_weight == doctest::Approx(1.0f));
    CHECK(p.fuzz_roughness == doctest::Approx(0.3f));
    CHECK(p.transmission_dispersion_scale == doctest::Approx(1.0f));
    CHECK(p.transmission_dispersion_abbe_number == doctest::Approx(35.0f));
    CHECK(p.transmission_scatter.b == doctest::Approx(0.3f));
    CHECK(p.subsurface_radius_scale.g == doctest::Approx(0.4f));
    CHECK(p.geometry_thin_walled == 1u);
}

TEST_CASE("rotations are written as angles and stored as a direction")
{
    OpenPBRParams p = openpbr_make_default_params();
    const json j = json::parse(R"({ "specular_anisotropy_rotation": 1.5707963,
                                    "coat_anisotropy_rotation": 3.1415927 })");
    CHECK(oka::materialsidecar::parseOpenPBR(j, p) == 0);

    CHECK(p.specular_anisotropy_rotation_cos == doctest::Approx(0.0f).epsilon(1e-5f));
    CHECK(p.specular_anisotropy_rotation_sin == doctest::Approx(1.0f));
    CHECK(p.coat_anisotropy_rotation_cos == doctest::Approx(-1.0f));
    CHECK(p.coat_anisotropy_rotation_sin == doctest::Approx(0.0f).epsilon(1e-5f));

    // The pair must stay a unit direction, or the frame it rotates is scaled.
    const float r2 = p.specular_anisotropy_rotation_cos * p.specular_anisotropy_rotation_cos +
                     p.specular_anisotropy_rotation_sin * p.specular_anisotropy_rotation_sin;
    CHECK(r2 == doctest::Approx(1.0f));
}

TEST_CASE("a misspelt key is reported, not swallowed")
{
    // Without this the format's worst failure is indistinguishable from the
    // renderer ignoring the file.
    OpenPBRParams p = openpbr_make_default_params();
    const json j = json::parse(R"({ "base_colour": [1,0,0], "coat_wieght": 1.0, "base_color": [0,1,0] })");

    CHECK(oka::materialsidecar::parseOpenPBR(j, p) == 2);
    // The correctly spelled neighbour still applies.
    CHECK(p.base_color.g == doctest::Approx(1.0f));
    // And the misspelt ones changed nothing.
    CHECK(p.coat_weight == doctest::Approx(0.0f));
}

TEST_CASE("a colour given as a scalar is rejected rather than half-applied")
{
    OpenPBRParams p = openpbr_make_default_params();
    const OpenPBRColor before = p.base_color;
    const json j = json::parse(R"({ "base_color": 0.5 })");

    // Recognised key, so it is not counted as unknown -- but the value is not a
    // colour, and the previous one has to survive intact.
    CHECK(oka::materialsidecar::parseOpenPBR(j, p) == 0);
    CHECK(p.base_color.r == doctest::Approx(before.r));
    CHECK(p.base_color.g == doctest::Approx(before.g));
    CHECK(p.base_color.b == doctest::Approx(before.b));
}

TEST_CASE("the shipped validation scene's sidecar parses with no unknown keys")
{
    // The scene exists to exercise the OpenPBR path; a typo in it would quietly
    // turn it back into a plain diffuse box and the row would still pass.
    const std::string path = std::string(STRELKA_TEST_ASSETS_DIR) + "/openpbr/openpbr_cornell_openpbr.json";
    std::ifstream file(path);
    REQUIRE(file.is_open());

    json doc;
    file >> doc;
    REQUIRE(doc.contains("materials"));
    CHECK(doc["materials"].size() == 3);

    int authored = 0;
    for (const auto& entry : doc["materials"])
    {
        REQUIRE(entry.contains("openpbr"));
        OpenPBRParams p = openpbr_make_default_params();
        CHECK(oka::materialsidecar::parseOpenPBR(entry["openpbr"], p) == 0);
        ++authored;
    }
    CHECK(authored == 3);
}
