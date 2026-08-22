// ============================================================================
// test_openpbr_from_gltf.cpp
//
// openpbr_from_gltf.h re-spells a glTF material as an OpenPBR one. Most of it is
// field-to-field and would be dull to test; what is tested here is the handful
// of places where the two models disagree about how to *store* the same physical
// quantity, because those are the ones that fail silently.
//
// Three of them, and each has a characteristic wrong-looking result rather than
// an error:
//
//   * specular_weight. gltfloader.cpp halves KHR_materials_specular's
//     specularFactor on the way in -- glTF's default 1.0 is stored as 0.5 -- and
//     OpenPBR's specular_weight means the unhalved thing. Forget to undo it and
//     every converted material quietly loses half its specular reflection, which
//     reads as "OpenPBR is duller" rather than as a conversion bug.
//   * emission. A colour times a multiplier becomes a level times a tint. The
//     product is the radiance, and it is the product that has to survive.
//   * subsurface radius. A per-channel vector becomes a scalar length times a
//     normalised scale, so that the longest channel keeps its world units.
//
// The last case here is the one that matters for validation strategy: a material
// with no coat, no fuzz, no transmission and no subsurface must convert to an
// OpenPBR block that is *only* base and specular. That is the configuration the
// degenerate-case cross-check against standard_pbr renders, and if conversion
// leaves a stray lobe switched on, the cross-check measures that instead of the
// thing it was written for.
// ============================================================================

#include <doctest/doctest.h>

// material_params.h deliberately does not pull in material_math.h (it would
// clash with CUDA's vector_types.h on host), so make_float3 has to come in
// separately here.
#include <strelka/material/material_math.h>
#include <strelka/material/material_params.h>
#include <strelka/material/openpbr/openpbr_from_gltf.h>
#include <strelka/material/openpbr/openpbr_params.h>

#include <cmath>
#include <limits>

namespace
{

/// A glTF material as the loader leaves it for a plain metallic-roughness
/// surface: gltfloader.cpp writes specular 0.5 for the extension's default 1.0.
MaterialParams plainGltfMaterial()
{
    MaterialParams p = {};
    p.base_color = make_float3(0.8f, 0.4f, 0.2f);
    p.metallic = 0.0f;
    p.roughness = 0.4f;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.specular_color = make_float3(1.0f, 1.0f, 1.0f);
    p.base_color_alpha = 1.0f;
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;
    p.uv_scale_x = 1.0f;
    p.uv_scale_y = 1.0f;
    p.clearcoat_ior = 1.5f;
    return p;
}

} // namespace

TEST_CASE("the specular halving is undone, or every material loses half its highlight")
{
    MaterialParams p = plainGltfMaterial();

    // glTF default specularFactor 1.0, stored halved.
    p.specular = 0.5f;
    CHECK(openpbr_from_material_params(p).specular_weight == doctest::Approx(1.0f));

    // A material that really did ask for half.
    p.specular = 0.25f;
    CHECK(openpbr_from_material_params(p).specular_weight == doctest::Approx(0.5f));

    // And it must not run past the top of the range.
    p.specular = 0.9f;
    CHECK(openpbr_from_material_params(p).specular_weight == doctest::Approx(1.0f));
}

TEST_CASE("emission survives the change of parameterisation as a product")
{
    MaterialParams p = plainGltfMaterial();
    p.emission = make_float3(0.9f, 0.35f, 0.1f);
    p.emission_strength = 12.0f;

    const OpenPBRParams o = openpbr_from_material_params(p);

    // level * tint must reproduce colour * strength, channel by channel: that
    // product is the radiance the emitter puts into the scene.
    CHECK(o.emission_luminance * o.emission_color.r == doctest::Approx(0.9f * 12.0f));
    CHECK(o.emission_luminance * o.emission_color.g == doctest::Approx(0.35f * 12.0f));
    CHECK(o.emission_luminance * o.emission_color.b == doctest::Approx(0.1f * 12.0f));

    SUBCASE("a black emitter has no hue to preserve and must not divide by it")
    {
        p.emission = make_float3(0.0f, 0.0f, 0.0f);
        p.emission_strength = 5.0f;
        const OpenPBRParams z = openpbr_from_material_params(p);
        CHECK(z.emission_luminance == 0.0f);
        CHECK(std::isfinite(z.emission_color.r));
        CHECK(z.emission_color.r == 1.0f);
    }
}

TEST_CASE("subsurface colour is the authored albedo, not the inverted one")
{
    // OpenPBR's interior volume runs its own van de Hulst inversion on this
    // input, so it has to receive the colour the DCC authored. Strelka's own
    // walk wants the single-scattering albedo instead and keeps it on
    // diffuse_transmission_color, which makes the two fields easy to confuse --
    // and confusing them inverts twice, which whitens a saturated medium
    // rather than shifting it slightly.
    MaterialParams p = plainGltfMaterial();
    p.subsurface = 1.0f;
    p.subsurface_radius = make_float3(6.75f, 1.66f, 0.33f);
    p.diffuse_transmission_color = make_float3(0.811f, 0.679f, 0.991f); // already inverted
    p.subsurface_reference = make_float3(0.352f, 0.240f, 0.800f);       // what the author picked

    const OpenPBRParams o = openpbr_from_material_params(p);
    CHECK(o.subsurface_color.r == doctest::Approx(0.352f));
    CHECK(o.subsurface_color.g == doctest::Approx(0.240f));
    CHECK(o.subsurface_color.b == doctest::Approx(0.800f));

    SUBCASE("a diffuse-transmission material has no reference and keeps its colour")
    {
        // KHR_materials_diffuse_transmission states an authored colour directly,
        // so nothing inverted it and there is nothing to undo.
        p.subsurface_reference = make_float3(0.0f, 0.0f, 0.0f);
        const OpenPBRParams d = openpbr_from_material_params(p);
        CHECK(d.subsurface_color.r == doctest::Approx(0.811f));
        CHECK(d.subsurface_color.g == doctest::Approx(0.679f));
        CHECK(d.subsurface_color.b == doctest::Approx(0.991f));
    }
}

TEST_CASE("subsurface radius splits into a length and a normalised scale")
{
    MaterialParams p = plainGltfMaterial();
    p.subsurface = 1.0f;
    p.subsurface_radius = make_float3(1.0f, 0.5f, 0.25f);

    const OpenPBRParams o = openpbr_from_material_params(p);

    // The product is the mean free path per channel and has to come back.
    CHECK(o.subsurface_radius * o.subsurface_radius_scale.r == doctest::Approx(1.0f));
    CHECK(o.subsurface_radius * o.subsurface_radius_scale.g == doctest::Approx(0.5f));
    CHECK(o.subsurface_radius * o.subsurface_radius_scale.b == doctest::Approx(0.25f));
    // The scalar carries the world-space length, so the largest channel is 1.
    CHECK(o.subsurface_radius == doctest::Approx(1.0f));

    SUBCASE("a zero radius must not divide by itself")
    {
        p.subsurface_radius = make_float3(0.0f, 0.0f, 0.0f);
        const OpenPBRParams z = openpbr_from_material_params(p);
        CHECK(std::isfinite(z.subsurface_radius));
        CHECK(std::isfinite(z.subsurface_radius_scale.r));
        // Left at the spec default rather than degenerate.
        CHECK(z.subsurface_radius == doctest::Approx(1.0f));
    }
}

TEST_CASE("glTF volume absorption becomes a transmission depth, and its absence becomes none")
{
    MaterialParams p = plainGltfMaterial();
    p.transmission = 1.0f;

    SUBCASE("no KHR_materials_volume -> no medium")
    {
        p.attenuation_distance = 0.0f;
        const OpenPBRParams o = openpbr_from_material_params(p);
        CHECK(o.transmission_weight == doctest::Approx(1.0f));
        // depth 0 is OpenPBR's "no volume", matching glTF's absent extension.
        CHECK(o.transmission_depth == 0.0f);
        CHECK(o.transmission_color.r == 1.0f);
    }
    SUBCASE("with a volume -> depth and tint carried across")
    {
        p.attenuation_distance = 2.5f;
        p.attenuation_color = make_float3(0.2f, 0.7f, 0.9f);
        const OpenPBRParams o = openpbr_from_material_params(p);
        CHECK(o.transmission_depth == doctest::Approx(2.5f));
        CHECK(o.transmission_color.r == doctest::Approx(0.2f));
        CHECK(o.transmission_color.b == doctest::Approx(0.9f));
    }
    SUBCASE("an infinite attenuation distance is glTF's 'no volume' too")
    {
        p.attenuation_distance = std::numeric_limits<float>::infinity();
        CHECK(openpbr_from_material_params(p).transmission_depth == 0.0f);
    }
}

TEST_CASE("a plain metallic-roughness material converts with no extra lobe switched on")
{
    // This is the precondition of the degenerate-case cross-check against
    // standard_pbr. If any of these is non-zero, that comparison stops measuring
    // the base and specular lobes and starts measuring whatever leaked in.
    const OpenPBRParams o = openpbr_from_material_params(plainGltfMaterial());

    CHECK(o.coat_weight == 0.0f);
    CHECK(o.fuzz_weight == 0.0f);
    CHECK(o.transmission_weight == 0.0f);
    CHECK(o.subsurface_weight == 0.0f);
    CHECK(o.thin_film_weight == 0.0f);
    CHECK(o.emission_luminance == 0.0f);
    CHECK(o.specular_roughness_anisotropy == 0.0f);
    CHECK(o.base_diffuse_roughness == 0.0f);

    // And the parts that must be live are.
    CHECK(o.base_weight == doctest::Approx(1.0f));
    CHECK(o.base_color.r == doctest::Approx(0.8f));
    CHECK(o.specular_weight == doctest::Approx(1.0f));
    CHECK(o.specular_roughness == doctest::Approx(0.4f));
    CHECK(o.specular_ior == doctest::Approx(1.5f));

    // Rotation stored as a direction, not an angle: (1,0) is "no rotation".
    CHECK(o.specular_anisotropy_rotation_cos == doctest::Approx(1.0f));
    CHECK(o.specular_anisotropy_rotation_sin == doctest::Approx(0.0f));
    // A zero UV scale would collapse every lookup to one texel.
    CHECK(o.uv_scale_x == doctest::Approx(1.0f));
    CHECK(o.uv_scale_y == doctest::Approx(1.0f));
}

TEST_CASE("anisotropy rotation is carried as a direction, not an angle")
{
    MaterialParams p = plainGltfMaterial();
    p.anisotropy = 0.75f;
    p.anisotropy_rotation = 3.14159265358979323846f * 0.5f;

    const OpenPBRParams o = openpbr_from_material_params(p);
    CHECK(o.specular_roughness_anisotropy == doctest::Approx(0.75f));
    CHECK(o.specular_anisotropy_rotation_cos == doctest::Approx(0.0f).epsilon(1e-6f));
    CHECK(o.specular_anisotropy_rotation_sin == doctest::Approx(1.0f));

    SUBCASE("a negative anisotropy strength is a magnitude here")
    {
        p.anisotropy = -0.5f;
        CHECK(openpbr_from_material_params(p).specular_roughness_anisotropy == doctest::Approx(0.5f));
    }
}
