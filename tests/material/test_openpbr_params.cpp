
#include <doctest/doctest.h>

#include <strelka/material/openpbr/openpbr_params.h>
#include <strelka/material/openpbr/openpbr_bridge.h>

#include <cstddef>

namespace
{

SurfaceInteraction identitySurface()
{
    SurfaceInteraction si = {};
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 1.0f, 0.0f);
    si.shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    si.geometry_normal = si.shading_normal;
    si.wo = make_float3(0.0f, 0.0f, 1.0f);
    si.exterior_ior = 1.0f;
    si.front_face = true;
    return si;
}

} // namespace

TEST_CASE("OpenPBRParams layout is the one every backend was told to expect")
{
    // Total size and OpenPBRColor's size are static_asserted in the header on
    // each compiler; repeat them here so a host-only reader sees the contract.
    CHECK(sizeof(OpenPBRParams) == 272);
    CHECK(sizeof(OpenPBRColor) == 12);
    CHECK(alignof(OpenPBRColor) == 4);

    // 16-byte rows. Every colour must start one, or Metal's reader drifts.
    CHECK(offsetof(OpenPBRParams, base_color) == 0);
    CHECK(offsetof(OpenPBRParams, specular_color) == 32);
    CHECK(offsetof(OpenPBRParams, coat_color) == 64);
    CHECK(offsetof(OpenPBRParams, fuzz_color) == 112);
    CHECK(offsetof(OpenPBRParams, transmission_color) == 128);
    CHECK(offsetof(OpenPBRParams, transmission_scatter) == 144);
    CHECK(offsetof(OpenPBRParams, subsurface_color) == 176);
    CHECK(offsetof(OpenPBRParams, subsurface_radius_scale) == 192);
    CHECK(offsetof(OpenPBRParams, emission_color) == 208);
    CHECK(offsetof(OpenPBRParams, texture_mask) == 236);

    // Nothing may be added past the padding without moving it.
    CHECK(offsetof(OpenPBRParams, _pad) == 260);
}

TEST_CASE("texture slot ids are the ABI the loader and both backends share")
{
    // Append-only: a reordering silently rebinds every map in every .mtlx
    // already exported, and nothing downstream can detect it.
    CHECK(OPENPBR_TEX_BASE_COLOR == 0);
    CHECK(OPENPBR_TEX_SPECULAR_ROUGHNESS == 2);
    CHECK(OPENPBR_TEX_GEOMETRY_NORMAL == 13);
    CHECK(OPENPBR_TEX_GEOMETRY_OPACITY == 15);
    // Appended when real content asked; the earlier sixteen kept their numbers.
    CHECK(OPENPBR_TEX_SUBSURFACE_WEIGHT == 16);
    CHECK(OPENPBR_TEX_FUZZ_COLOR == 18);
    CHECK(MAX_OPENPBR_TEXTURES == 19);
    // The mask has one bit per slot and must fit the word that carries it.
    CHECK(MAX_OPENPBR_TEXTURES <= 32);
}

TEST_CASE("OpenPBR specialization keeps only scene features")
{
    OpenPBRParams p = openpbr_make_default_params();
    CHECK(openpbr_features(p) == 0u);

    p.coat_weight = 1.0f;
    CHECK(openpbr_features(p) == OPENPBR_FEATURE_SHEEN_AND_COAT);
    p.coat_weight = 0.0f;
    p.transmission_dispersion_scale = 1.0f;
    CHECK(openpbr_features(p) == OPENPBR_FEATURE_DISPERSION);
    p.transmission_dispersion_scale = 0.0f;
    p.texture_mask = 1u << OPENPBR_TEX_SUBSURFACE_WEIGHT;
    CHECK(openpbr_features(p) == OPENPBR_FEATURE_TRANSLUCENCY);
    p.texture_mask = 1u << OPENPBR_TEX_SUBSURFACE_RADIUS;
    CHECK(openpbr_features(p) == 0u);
    p.texture_mask = 1u << OPENPBR_TEX_BASE_METALNESS;
    CHECK(openpbr_features(p) == OPENPBR_FEATURE_METALLIC);
    CHECK(openpbr_base_only(openpbr_features(p)));
    p.transmission_weight = 1.0f;
    CHECK_FALSE(openpbr_base_only(openpbr_features(p)));
}

TEST_CASE("openpbr_make_default_params matches the vendored spec defaults")
{
    const OpenPBRParams p = openpbr_make_default_params();
    const OpenPBR_ResolvedInputs ref = openpbr_make_default_resolved_inputs();
    const OpenPBR_ResolvedInputs got = openpbr_resolve_inputs(p, identitySurface());

    const auto checkColor = [](float3 a, float3 b) {
        CHECK(a.x == b.x);
        CHECK(a.y == b.y);
        CHECK(a.z == b.z);
    };
#define CHECK_SCALAR(field) CHECK(got.field == ref.field)
#define CHECK_COLOR(field) checkColor(got.field, ref.field)

    CHECK_SCALAR(base_weight);
    CHECK_COLOR(base_color);
    CHECK_SCALAR(base_diffuse_roughness);
    CHECK_SCALAR(base_metalness);

    CHECK_SCALAR(subsurface_weight);
    CHECK_COLOR(subsurface_color);
    CHECK_SCALAR(subsurface_radius);
    CHECK_COLOR(subsurface_radius_scale);
    CHECK_SCALAR(subsurface_scatter_anisotropy);

    CHECK_SCALAR(specular_weight);
    CHECK_COLOR(specular_color);
    CHECK_SCALAR(specular_roughness);
    CHECK_SCALAR(specular_roughness_anisotropy);
    CHECK_SCALAR(specular_ior);
    CHECK(got.specular_anisotropy_rotation_cos_sin.x == ref.specular_anisotropy_rotation_cos_sin.x);
    CHECK(got.specular_anisotropy_rotation_cos_sin.y == ref.specular_anisotropy_rotation_cos_sin.y);

    CHECK_SCALAR(coat_weight);
    CHECK_COLOR(coat_color);
    CHECK_SCALAR(coat_roughness);
    CHECK_SCALAR(coat_roughness_anisotropy);
    CHECK_SCALAR(coat_ior);
    CHECK_SCALAR(coat_darkening);
    CHECK(got.coat_anisotropy_rotation_cos_sin.x == ref.coat_anisotropy_rotation_cos_sin.x);
    CHECK(got.coat_anisotropy_rotation_cos_sin.y == ref.coat_anisotropy_rotation_cos_sin.y);

    CHECK_SCALAR(fuzz_weight);
    CHECK_COLOR(fuzz_color);
    CHECK_SCALAR(fuzz_roughness);

    CHECK_SCALAR(transmission_weight);
    CHECK_COLOR(transmission_color);
    CHECK_SCALAR(transmission_depth);
    CHECK_COLOR(transmission_scatter);
    CHECK_SCALAR(transmission_scatter_anisotropy);
    CHECK_SCALAR(transmission_dispersion_scale);
    CHECK_SCALAR(transmission_dispersion_abbe_number);

    CHECK_SCALAR(thin_film_weight);
    CHECK_SCALAR(thin_film_thickness);
    CHECK_SCALAR(thin_film_ior);

    CHECK_SCALAR(emission_luminance);
    CHECK_COLOR(emission_color);

    CHECK_SCALAR(geometry_opacity);
    CHECK(got.geometry_thin_walled == ref.geometry_thin_walled);

#undef CHECK_COLOR
#undef CHECK_SCALAR
}

TEST_CASE("a zeroed OpenPBRParams is not a usable material")
{
    // Stated as a test because it is the mistake a caller makes once, and
    // because the failure is a black or NaN surface rather than an error.
    const OpenPBRParams zeroed = {};
    const OpenPBRParams defaults = openpbr_make_default_params();

    CHECK(zeroed.specular_ior == 0.0f); // a refractive index of zero
    CHECK(zeroed.coat_darkening == 0.0f); // spec says 1.0
    CHECK(zeroed.uv_scale_x == 0.0f); // collapses every lookup to one texel
    CHECK(zeroed.specular_anisotropy_rotation_cos == 0.0f); // (0,0) is not a rotation

    CHECK(defaults.specular_ior == 1.5f);
    CHECK(defaults.coat_darkening == 1.0f);
    CHECK(defaults.uv_scale_x == 1.0f);
    CHECK(defaults.uv_scale_y == 1.0f);
    CHECK(defaults.specular_anisotropy_rotation_cos == 1.0f);
    CHECK(defaults.coat_anisotropy_rotation_cos == 1.0f);
    CHECK(defaults.texture_mask == 0u);
}
