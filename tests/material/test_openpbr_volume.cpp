
#include <doctest/doctest.h>

#include <strelka/material/openpbr/openpbr_bridge.h>
#include <strelka/material/openpbr/openpbr_params.h>

#include <cmath>

TEST_CASE("subsurface extinction is the reciprocal of the per-channel mean free path")
{
    OpenPBRParams p = openpbr_make_default_params();
    p.subsurface_weight = 1.0f;
    p.subsurface_radius = 1.0f;
    p.subsurface_radius_scale = OpenPBRColor{ 1.0f, 0.5f, 0.25f };
    p.subsurface_color = OpenPBRColor{ 0.8f, 0.8f, 0.8f };

    const OpenPBR_HomogeneousVolume v = openpbr_interior_volume(p);

    CHECK(v.extinction_coefficient.x == doctest::Approx(1.0f).epsilon(1e-3f));
    CHECK(v.extinction_coefficient.y == doctest::Approx(2.0f).epsilon(1e-3f));
    CHECK(v.extinction_coefficient.z == doctest::Approx(4.0f).epsilon(1e-3f));
    // Highly scattering: this is the medium a random walk is for.
    CHECK(v.albedo.x > 0.9f);
    CHECK(openpbr_has_interior_medium(p));

    SUBCASE("the radius scales the whole thing, until the clamp")
    {
        // The Open Chess Set's own figure: subsurface_scale 0.003.
        p.subsurface_radius = 0.003f;
        const OpenPBR_HomogeneousVolume s = openpbr_interior_volume(p);
        CHECK(s.extinction_coefficient.x == doctest::Approx(1.0f / 0.003f).epsilon(1e-3f));
        CHECK(s.extinction_coefficient.y == doctest::Approx(1.0f / (0.003f * 0.5f)).epsilon(1e-3f));

        CHECK(s.extinction_coefficient.z == doctest::Approx(1000.0f).epsilon(1e-3f));
        CHECK(s.extinction_coefficient.z < 1.0f / (0.003f * 0.25f));
    }
}

TEST_CASE("transmission extinction is Beer-Lambert over the transmission depth")
{
    OpenPBRParams p = openpbr_make_default_params();
    p.transmission_weight = 1.0f;
    p.transmission_depth = 2.5f;
    p.transmission_color = OpenPBRColor{ 0.2f, 0.7f, 0.9f };

    const OpenPBR_HomogeneousVolume v = openpbr_interior_volume(p);

    CHECK(v.extinction_coefficient.x == doctest::Approx(-std::log(0.2f) / 2.5f).epsilon(1e-3f));
    CHECK(v.extinction_coefficient.y == doctest::Approx(-std::log(0.7f) / 2.5f).epsilon(1e-3f));
    CHECK(v.extinction_coefficient.z == doctest::Approx(-std::log(0.9f) / 2.5f).epsilon(1e-3f));

    // Pure absorption: nothing scatters unless transmission_scatter asks for it.
    // This is the one that says the existing Beer-Lambert path and this volume
    // are describing the same physics, and so must not both be applied.
    CHECK(v.albedo.x == doctest::Approx(0.0f));
    CHECK(v.albedo.y == doctest::Approx(0.0f));
    CHECK(v.albedo.z == doctest::Approx(0.0f));
    CHECK(openpbr_has_interior_medium(p));
}

TEST_CASE("the two configurations that must enclose nothing")
{
    SUBCASE("a thin-walled surface has no interior")
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.transmission_weight = 1.0f;
        p.transmission_depth = 2.5f;
        p.subsurface_weight = 1.0f;
        p.geometry_thin_walled = 1u;

        CHECK_FALSE(openpbr_has_interior_medium(p));
        const OpenPBR_HomogeneousVolume v = openpbr_interior_volume(p);
        CHECK(v.extinction_coefficient.x == doctest::Approx(0.0f));
    }
    SUBCASE("transmission with no depth is a surface tint, not a medium")
    {
        // What a glTF material converts to: KHR_materials_transmission alone,
        // with no KHR_materials_volume behind it.
        OpenPBRParams p = openpbr_make_default_params();
        p.transmission_weight = 1.0f;
        p.transmission_depth = 0.0f;

        CHECK_FALSE(openpbr_has_interior_medium(p));
        CHECK(openpbr_interior_volume(p).extinction_coefficient.x == doctest::Approx(0.0f));
    }
    SUBCASE("a plain opaque material")
    {
        CHECK_FALSE(openpbr_has_interior_medium(openpbr_make_default_params()));
    }
}

TEST_CASE("scatter anisotropy reaches the phase function")
{
    OpenPBRParams p = openpbr_make_default_params();
    p.subsurface_weight = 1.0f;
    p.subsurface_radius = 1.0f;
    p.subsurface_scatter_anisotropy = 0.6f;
    CHECK(openpbr_interior_volume(p).anisotropy == doctest::Approx(0.6f));
}

TEST_CASE("a dark subsurface colour makes a dark medium")
{
    auto albedoFor = [](float c) {
        OpenPBRParams p = openpbr_make_default_params();
        p.subsurface_weight = 1.0f;
        // The chess set's own scale, and a radius tint as dark as its marble.
        p.subsurface_radius = 0.003f;
        p.subsurface_radius_scale = OpenPBRColor{ c, c, c };
        p.subsurface_color = OpenPBRColor{ c, c, c };
        return openpbr_interior_volume(p).albedo.x;
    };

    CHECK(albedoFor(0.0f) < 1e-3f);
    CHECK(albedoFor(0.05f) < 0.3f);
    CHECK(albedoFor(0.05f) < albedoFor(0.2f));
    CHECK(albedoFor(0.2f) < albedoFor(0.8f));
    // Marble at 0.8 is the case the walk is for: nearly every extinction event
    // scatters rather than absorbs.
    CHECK(albedoFor(0.8f) > 0.9f);
}

TEST_CASE("the subsurface entry lobe carries no colour of its own")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    si.geometry_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 1.0f, 0.0f);
    si.wo = normalize(make_float3(0.3f, 0.0f, 1.0f));
    si.front_face = true;
    si.ior = 1.5f;
    si.exterior_ior = 1.0f;

    auto prepareFor = [&](float c) {
        OpenPBRParams p = openpbr_make_default_params();
        p.subsurface_weight = 1.0f;
        p.subsurface_radius = 0.003f;
        p.subsurface_color = OpenPBRColor{ c, c, c };
        p.subsurface_radius_scale = OpenPBRColor{ c, c, c };
        p.base_color = OpenPBRColor{ c, c, c };
        return openpbr_prepare_at(p, si, make_float3(1.0f, 1.0f, 1.0f));
    };

    const OpenPBR_PreparedBsdf dark = prepareFor(0.05f);
    const OpenPBR_PreparedBsdf light = prepareFor(1.0f);

    const float4 xi = make_float4(0.3f, 0.4f, 0.5f, 0.6f);
    const BsdfSampleResult sDark = openpbr_bsdf_sample(dark, xi);
    const BsdfSampleResult sLight = openpbr_bsdf_sample(light, xi);

    CHECK((sDark.event_type & BSDF_EVENT_TRANSMISSION) != 0u);
    CHECK(sDark.event_type == sLight.event_type);
    CHECK(sDark.bsdf_over_pdf.x == doctest::Approx(sLight.bsdf_over_pdf.x));
    CHECK(sDark.bsdf_over_pdf.x == doctest::Approx(1.0f).epsilon(0.01f));
}
