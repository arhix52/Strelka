// ============================================================================
// test_material_sidecar.cpp
//
// <stem>_openpbr.json is the only way to author an OpenPBR material for now,
// so it is also the only route by which coat_darkening, a fuzz layer, dispersion
// or a per-channel subsurface radius can reach the renderer at all -- glTF can
// express none of them, and openpbr_from_gltf.h therefore never moves them off
// their defaults.
//
// The failure mode a JSON format has is silence. A misspelt key, a colour given
// as a scalar, a material name that matches nothing: each of those does exactly
// nothing and looks from the outside like the renderer ignoring the file. So
// what is pinned here is mostly that the loader is *not* silent, plus the two
// places where the file's spelling and the runtime's differ:
//
//   * rotations. The spec and the file use an angle; OpenPBRParams stores its
//     cosine and sine, so that a filtered value cannot wrap through the
//     discontinuity. The conversion happens on load.
//   * defaults. An omitted key must leave the OpenPBR 1.1.1 default in place,
//     not zero -- a zeroed block is a material with no refractive index and a
//     degenerate anisotropy frame.
// ============================================================================

#include <doctest/doctest.h>

#include <strelka/material/openpbr/openpbr_params.h>
#include <strelka/sceneloader/material_sidecar.h>

#include <cmath>

using nlohmann::json;

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
