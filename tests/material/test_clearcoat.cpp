
#include <doctest/doctest.h>

#include <strelka/material/material_math.h>
#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_params.h>
#include <strelka/material/surface_interaction.h>
#include <strelka/material/sampling.h>
#include <strelka/material/fresnel.h>
#include <strelka/material/microfacet.h>
#include <strelka/material/bsdf.h>

#include <cmath>
#include <cstdint>
#include <numbers>

namespace
{

struct Lcg
{
    std::uint32_t state;
    explicit Lcg(std::uint32_t seed) : state(seed | 1u) {}
    float next()
    {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>((state >> 8) & 0xFFFFFFu) / static_cast<float>(0x1000000);
    }
};

MaterialParams ceramic_params(float coat, float coatIor)
{
    MaterialParams p = {};
    p.base_color = make_float3(0.9f, 0.9f, 0.9f);
    p.metallic = 0.0f;
    p.roughness = 0.35f;
    p.ior = 1.6f;
    p.specular = 0.5f;
    p.transmission = 0.0f;
    p.clearcoat = coat;
    p.clearcoat_roughness = 0.05f;
    p.clearcoat_ior = coatIor;
    p.anisotropy = 0.0f;
    p.normal_scale = 1.0f;
    p.occlusion_strength = 1.0f;
    p.alpha_cutoff = 0.5f;
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;
    p.base_color_alpha = 1.0f;
    p.attenuation_distance = 1e30f;
    p.attenuation_color = make_float3(1.0f);
    p.uv_scale_x = 1.0f;
    p.uv_scale_y = 1.0f;
    return p;
}

SurfaceInteraction make_si(const MaterialParams& p, float3 wo)
{
    SurfaceInteraction si = {};
    si.position = make_float3(0.0f);
    si.geometry_normal = make_float3(0.0f, 0.0f, 1.0f);
    si.shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 1.0f, 0.0f);
    si.wo = wo;
    si.uv = make_float2(0.0f, 0.0f);
    si.front_face = wo.z > 0.0f;
    bsdf_init(si, p, nullptr);
    return si;
}

float3 dir_at(float deg)
{
    const float r = deg * std::numbers::pi_v<float> / 180.0f;
    return make_float3(std::sin(r), 0.0f, std::cos(r));
}

float3 integrate_albedo(const MaterialParams& p, float3 wo, int samples, std::uint32_t seed)
{
    float3 total = make_float3(0.0f);
    const SurfaceInteraction si = make_si(p, wo);
    Lcg rng(seed);
    for (int i = 0; i < samples; ++i)
    {
        const BsdfSampleResult s =
            bsdf_sample(si, make_float4(rng.next(), rng.next(), rng.next(), rng.next()));
        if (s.event_type == BSDF_EVENT_ABSORB || s.pdf <= 0.0f)
            continue;
        // bsdf_over_pdf already carries the cosine; see bsdf_types.h.
        total = total + s.bsdf_over_pdf;
    }
    const float inv = 1.0f / static_cast<float>(samples);
    return make_float3(total.x * inv, total.y * inv, total.z * inv);
}

} // namespace

TEST_CASE("clearcoat: weight 0 changes nothing")
{
    const MaterialParams bare = ceramic_params(0.0f, 1.5f);
    const MaterialParams zeroed = ceramic_params(0.0f, 2.0f); // IOR set, weight not

    for (const float deg : { 5.0f, 40.0f, 80.0f })
    {
        const SurfaceInteraction a = make_si(bare, dir_at(deg));
        const SurfaceInteraction b = make_si(zeroed, dir_at(deg));
        const float3 wi = dir_at(-deg + 30.0f);
        const BsdfEvalResult ea = bsdf_eval(a, wi);
        const BsdfEvalResult eb = bsdf_eval(b, wi);
        CHECK(ea.bsdf.x == doctest::Approx(eb.bsdf.x));
        CHECK(ea.pdf == doctest::Approx(eb.pdf));
    }
}

TEST_CASE("clearcoat: a higher IOR reflects more, and by less than its own ratio")
{
    const float3 wo = dir_at(20.0f);
    const float3 wi = dir_at(-20.0f);
    const BsdfEvalResult lacquer = bsdf_eval(make_si(ceramic_params(1.0f, 1.5f), wo), wi);
    const BsdfEvalResult glaze = bsdf_eval(make_si(ceramic_params(1.0f, 2.0f), wo), wi);

    REQUIRE(lacquer.bsdf.x > 0.0f);
    CHECK(glaze.bsdf.x > lacquer.bsdf.x);

    const float f0Ratio = f0_from_ior(2.0f) / f0_from_ior(1.5f);
    CHECK(f0Ratio == doctest::Approx(2.78f).epsilon(0.05));
    // Strictly less: the extra reflection is taken out of the base, not added on
    // top of it. Equality here would be the un-layered coat this test exists for.
    CHECK(glaze.bsdf.x / lacquer.bsdf.x < f0Ratio);
}

TEST_CASE("clearcoat: an unset IOR is the extension's lacquer, not no coat")
{
    // MaterialParams is zero-initialised throughout, and an IOR of 0 clamps to 1,
    // whose F0 is exactly zero. Without a default the layer would stop behaving
    // like a coat with nothing to say it had.
    const MaterialParams unset = ceramic_params(1.0f, 0.0f);
    const MaterialParams lacquer = ceramic_params(1.0f, 1.5f);

    const float3 wo = dir_at(20.0f);
    const float3 wi = dir_at(-20.0f);
    const BsdfEvalResult a = bsdf_eval(make_si(unset, wo), wi);
    const BsdfEvalResult b = bsdf_eval(make_si(lacquer, wo), wi);
    CHECK(a.bsdf.x == doctest::Approx(b.bsdf.x));
    CHECK(a.bsdf.x > 0.0f);
}

TEST_CASE("clearcoat: does not manufacture energy")
{
    for (const float ior : { 1.5f, 2.0f })
    {
        for (const float deg : { 15.0f, 45.0f })
        {
            const float3 albedo =
                integrate_albedo(ceramic_params(1.0f, ior), dir_at(deg), 20000, 11u);
            CHECK(albedo.x <= 1.0f);
            CHECK(albedo.y <= 1.0f);
            CHECK(albedo.z <= 1.0f);
            CHECK(albedo.x > 0.0f);
        }
        const float3 coated =
            integrate_albedo(ceramic_params(1.0f, ior), dir_at(75.0f), 20000, 11u);
        const float3 bare =
            integrate_albedo(ceramic_params(0.0f, ior), dir_at(75.0f), 20000, 11u);
        CHECK(coated.x <= bare.x + 0.05f);
    }
}

TEST_CASE("clearcoat: underside bounces return energy that scales with IOR")
{
    MaterialParams lacquer = ceramic_params(1.0f, 1.5f);
    MaterialParams glaze = ceramic_params(1.0f, 2.0f);
    lacquer.clearcoat_roughness = 0.05f;
    glaze.clearcoat_roughness = 0.05f;

    const float3 wo = dir_at(25.0f);
    const float3 wi = dir_at(70.0f); // reflection of wo is -25, so this is off-peak
    const float baseLacquer = bsdf_eval(make_si(lacquer, wo), wi).bsdf.x;
    const float baseGlaze = bsdf_eval(make_si(glaze, wo), wi).bsdf.x;
    REQUIRE(baseLacquer > 0.0f);
    REQUIRE(baseGlaze > 0.0f);

    auto single = [](float ior, float nDotV, float nDotL) {
        const float f0 = f0_from_ior(ior);
        const float Fl = f0 + (1.0f - f0) * std::pow(1.0f - nDotL, 5.0f);
        const float Fv = f0 + (1.0f - f0) * std::pow(1.0f - nDotV, 5.0f);
        return (1.0f - Fl) * (1.0f - Fv);
    };
    const float nDotV = std::cos(25.0f * std::numbers::pi_v<float> / 180.0f);
    const float nDotL = std::cos(70.0f * std::numbers::pi_v<float> / 180.0f);
    // Lambert * albedo / pi, times the single-scatter scale -- what eval would
    // report with the series left out.
    const float lambert = 0.9f * 0.318309886f;
    const float floorLacquer = single(1.5f, nDotV, nDotL) * lambert;
    const float floorGlaze = single(2.0f, nDotV, nDotL) * lambert;

    CHECK(baseLacquer > floorLacquer * 1.01f);
    CHECK(baseGlaze > floorGlaze * 1.01f);
    // Stronger coat => larger F_avg => larger relative lift over the floor.
    CHECK((baseGlaze / floorGlaze) > (baseLacquer / floorLacquer));
}

TEST_CASE("clearcoat: sample and eval agree")
{
    MaterialParams glaze = ceramic_params(1.0f, 2.0f);
    glaze.clearcoat_roughness = 0.25f;
    for (const float deg : { 20.0f, 55.0f })
    {
        const SurfaceInteraction si = make_si(glaze, dir_at(deg));
        Lcg rng(0xC0A7u + static_cast<std::uint32_t>(deg));
        int checked = 0;
        for (int i = 0; i < 600 && checked < 60; ++i)
        {
            const BsdfSampleResult s =
                bsdf_sample(si, make_float4(rng.next(), rng.next(), rng.next(), rng.next()));
            if (s.event_type == BSDF_EVENT_ABSORB || s.pdf <= 0.0f)
                continue;
            if ((s.event_type & BSDF_EVENT_SPECULAR) != 0)
                continue; // delta lobes have no density to compare against

            const BsdfEvalResult e = bsdf_eval(si, s.wi);
            const float cosL = std::fabs(dot(s.wi, si.shading_normal));
            const float expected = e.bsdf.x * cosL / std::max(e.pdf, 1e-10f);
            CHECK(s.pdf == doctest::Approx(e.pdf).epsilon(0.02));
            CHECK(s.bsdf_over_pdf.x == doctest::Approx(expected).epsilon(0.02));
            ++checked;
        }
        CHECK(checked > 0);
    }
}
