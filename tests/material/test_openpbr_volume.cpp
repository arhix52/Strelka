// ============================================================================
// test_openpbr_volume.cpp
//
// The interior medium an OpenPBR material encloses: what fills a closed surface
// once subsurface scattering and transmission are blended, which the
// specification defines jointly rather than as two volumes.
//
// Worth pinning separately from the BSDF because the integrator consumes it
// through a different route. openpbr_interior_volume() is called from the
// wavefront tracer's `extend` stage, which has a ray and a medium id and no
// shading point at all -- no hit, no basis, no view direction. That works only
// because Adobe's staged initialisation lets the volume be derived on its own,
// and if a later version started reading the geometry basis there, the extend
// stage would be handing it an identity frame without knowing.
//
// The numbers below are checked against the closed forms rather than against
// whatever the library returned when this was written:
//
//   subsurface   extinction = 1 / (radius * radius_scale)   per channel
//   transmission extinction = -ln(transmission_color) / transmission_depth
//
// and the two cases that must produce *no* medium, because they are the ones a
// caller gets wrong: a thin-walled surface has no interior, and transmission
// with no depth is a tint on the surface lobe rather than a volume.
// ============================================================================

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

        // The blue channel would be 1/(0.003 * 0.25) = 1333, and is 1000
        // instead: openpbr_clamp_input_distance floors the mean free path at
        // OpenPBR_MinDistance = 1e-3, which caps extinction at 1000.
        //
        // Pinned rather than worked around. A clamp on the mean free path is
        // exactly the kind of constant that moves in a library bump, and it
        // would shift the look of every dense subsurface material -- the chess
        // pieces among them -- with nothing else to notice.
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
