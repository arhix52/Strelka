// ============================================================================
// test_standard_pbr_furnace.cpp
//
// A white furnace on the opaque half of standard_pbr: integrate bsdf * cos over
// the hemisphere with a white base and no metal. A surface cannot return more
// light than falls on it, so the directional albedo must not exceed one.
//
// It used to, and by enough to see: the diffuse lobe was summed with the
// specular one rather than layered under it, so a rough dielectric returned
// about 4% too much at normal incidence and 23% at grazing -- entry 15 of
// docs/open-defects.md, which this file measured and now guards.
//
// What named the lobe was the one configuration that was already right. At
// `specular = 0` the material was exactly energy-conserving head on, which is
// why the ladder's `00_calibration` sphere -- authored
// `KHR_materials_specular: {specularFactor: 0}` -- matched Cycles while the
// stage it sits on did not. The defect had hidden behind a compensating error:
// that scene read 1.010 overall while its sphere was 4.5% dark and its stage
// 3.5% bright, and only the frame mean was ever recorded.
//
// The grazing rows are a second, smaller story and are still open: even at
// `specular = 0` the albedo climbs above one as the view goes grazing, worst at
// low roughness. That is Schlick's (1-F0)(1-cos)^5 tail, which reaches one
// whatever the interface is, and it is why `ggx_specular_albedo()` drops the
// split-sum's B term -- subtracting it would drain a material whose specular
// weight is zero. Pinned separately so a fix to either can be graded.
// ============================================================================

#include <doctest/doctest.h>

#include <strelka/material/bsdf.h>

#include <cmath>

namespace
{

/// Integrate bsdf * cos(theta_l) over the upper hemisphere for one view angle.
///
/// A 400x400 grid rather than a Monte Carlo estimate: the quantity is smooth,
/// the integrand is cheap, and a deterministic sum is what lets the thresholds
/// below be tight enough to catch a 1% move.
double directionalAlbedo(float roughness, float specular, float base, float cosV)
{
    SurfaceInteraction si = {};
    si.material_type = MATERIAL_TYPE_STANDARD_PBR;
    si.albedo = make_float3(base, base, base);
    si.roughness = roughness;
    si.metallic = 0.0f;
    si.specular = specular;
    si.specular_color = make_float3(1.0f, 1.0f, 1.0f);
    si.ior = 1.5f;
    si.exterior_ior = 1.0f;
    si.shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    si.geometry_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.front_face = true;

    const float sinV = std::sqrt(std::max(0.0f, 1.0f - cosV * cosV));
    si.wo = make_float3(sinV, 0.0f, cosV);

    constexpr int kTheta = 400;
    constexpr int kPhi = 400;
    double sum = 0.0;
    for (int i = 0; i < kTheta; ++i)
    {
        const double theta = (i + 0.5) * (M_PI / 2) / kTheta;
        for (int j = 0; j < kPhi; ++j)
        {
            const double phi = (j + 0.5) * (2 * M_PI) / kPhi;
            const float3 wi = make_float3(static_cast<float>(std::sin(theta) * std::cos(phi)),
                                          static_cast<float>(std::sin(theta) * std::sin(phi)),
                                          static_cast<float>(std::cos(theta)));
            sum += bsdf_eval(si, wi).bsdf.x * std::cos(theta) * std::sin(theta);
        }
    }
    return sum * (M_PI / 2) / kTheta * (2 * M_PI) / kPhi;
}

} // namespace

TEST_CASE("the specular lobe is layered over the base rather than added to it")
{
    // Head on, where Schlick's grazing tail is zero and conservation is therefore
    // exactly testable. Before specular_base_scale() these read 1.042, 1.074 and
    // 1.080.
    for (const float roughness : { 0.85f, 0.50f, 0.20f })
    {
        CHECK(directionalAlbedo(roughness, 1.0f, 1.0f, 1.0f) == doctest::Approx(1.0).epsilon(0.001));
    }
}

TEST_CASE("layering the base costs nothing where there is no lobe to layer under")
{
    // The configuration that was already right before the fix, and so both the
    // control for the case above and a guard on the fix itself: the specular
    // weight at zero has to leave the material untouched. An earlier attempt
    // scaled by (1 - Fresnel) instead of by the lobe's own albedo and took 2.4%
    // from a material with no specular lobe at all.
    for (const float roughness : { 0.85f, 0.50f, 0.20f })
    {
        CHECK(directionalAlbedo(roughness, 0.0f, 1.0f, 1.0f) == doctest::Approx(1.0).epsilon(0.001));
    }
}

TEST_CASE("the specular weight no longer decides whether energy is conserved")
{
    // The split that named the defect: with it fixed, turning the lobe on and off
    // moves the albedo by well under a percent at every angle, instead of by 4-23%.
    for (const float roughness : { 0.85f, 0.50f, 0.20f })
    {
        for (const float cosV : { 1.0f, 0.7f, 0.3f })
        {
            const double on = directionalAlbedo(roughness, 1.0f, 1.0f, cosV);
            const double off = directionalAlbedo(roughness, 0.0f, 1.0f, cosV);
            CHECK(on == doctest::Approx(off).epsilon(0.01));
        }
    }
}

TEST_CASE("a grazing view creates energy even with the specular lobe switched off")
{
    // Separate from the above and smaller: the single-scattering GGX/Smith
    // masking leaves the base itself over unity as the view goes grazing, worst
    // where the lobe is tightest. Recorded so it is not mistaken for the
    // coupling defect, and so a fix to one is not credited with the other.
    CHECK(directionalAlbedo(0.85f, 0.0f, 1.0f, 0.3f) == doctest::Approx(1.015).epsilon(0.002));
    CHECK(directionalAlbedo(0.50f, 0.0f, 1.0f, 0.3f) == doctest::Approx(1.065).epsilon(0.002));
    CHECK(directionalAlbedo(0.20f, 0.0f, 1.0f, 0.3f) == doctest::Approx(1.161).epsilon(0.002));
}
