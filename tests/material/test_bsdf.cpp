#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <strelka/material/material_math.h>
#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_params.h>
#include <strelka/material/surface_interaction.h>
#include <strelka/material/sampling.h>
#include <strelka/material/fresnel.h>
#include <strelka/material/microfacet.h>
#include <strelka/material/bsdf.h>

#include <algorithm>
#include <cmath>

static SurfaceInteraction make_test_si()
{
    SurfaceInteraction si = {};
    si.position = make_float3(0, 0, 0);
    si.shading_normal = make_float3(0, 1, 0);
    si.geometry_normal = make_float3(0, 1, 0);
    si.tangent = make_float3(1, 0, 0);
    si.bitangent = make_float3(0, 0, 1);
    si.uv = make_float2(0, 0);
    si.wo = make_float3(0, 1, 0); // looking straight down at surface
    si.front_face = true;
    return si;
}

static MaterialParams make_diffuse_params()
{
    MaterialParams p = {};
    p.material_type = MATERIAL_TYPE_DIFFUSE;
    p.base_color = make_float3(0.8f, 0.2f, 0.1f);
    p.roughness = 1.0f;
    p.metallic = 0.0f;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.specular_color = make_float3(1.0f);
    p.transmission = 0.0f;
    p.clearcoat = 0.0f;
    p.clearcoat_roughness = 0.0f;
    p.anisotropy = 0.0f;
    p.emission = make_float3(0.0f);
    p.emission_strength = 0.0f;
    p.normal_scale = 1.0f;
    p.occlusion_strength = 1.0f;
    p.alpha_cutoff = 0.5f;
    p.base_color_tex = -1;
    p.metallic_roughness_tex = -1;
    p.normal_tex = -1;
    p.emission_tex = -1;
    p.occlusion_tex = -1;
    p.transmission_tex = -1;
    p.thin_walled = 0;
    return p;
}

// --- material_math tests ---

TEST_CASE("make_float3 constructs correctly")
{
    float3 v = make_float3(1.0f, 2.0f, 3.0f);
    CHECK(v.x == doctest::Approx(1.0f));
    CHECK(v.y == doctest::Approx(2.0f));
    CHECK(v.z == doctest::Approx(3.0f));
}

TEST_CASE("reflect_dir gives correct reflection")
{
    float3 incident = make_float3(1, -1, 0);
    float3 normal = make_float3(0, 1, 0);
    float3 r = reflect_dir(incident, normal);
    CHECK(r.x == doctest::Approx(1.0f));
    CHECK(r.y == doctest::Approx(1.0f));
    CHECK(r.z == doctest::Approx(0.0f));
}

// --- fresnel tests ---

TEST_CASE("fresnel_schlick at normal incidence returns F0")
{
    float3 F0 = make_float3(0.04f);
    float3 F = fresnel_schlick(F0, 1.0f);
    CHECK(F.x == doctest::Approx(0.04f));
    CHECK(F.y == doctest::Approx(0.04f));
    CHECK(F.z == doctest::Approx(0.04f));
}

TEST_CASE("fresnel_schlick at grazing angle approaches 1")
{
    float3 F0 = make_float3(0.04f);
    float3 F = fresnel_schlick(F0, 0.0f);
    CHECK(F.x == doctest::Approx(1.0f));
}

TEST_CASE("fresnel_dielectric total internal reflection")
{
    // Going from glass (ior=1.5) to air (eta = 1.5)
    // At angle > critical angle, should get TIR
    float F = fresnel_dielectric(0.1f, 1.5f);
    CHECK(F == doctest::Approx(1.0f));
}

// --- sampling tests ---

TEST_CASE("cosine_hemisphere_sample produces valid directions")
{
    float3 dir = cosine_hemisphere_sample(0.5f, 0.5f);
    // z should be positive (upper hemisphere)
    CHECK(dir.z >= 0.0f);
    // should be approximately unit length
    float len = glm::length(dir);
    CHECK(len == doctest::Approx(1.0f).epsilon(0.01f));
}

TEST_CASE("cosine_hemisphere_pdf is positive for positive cos_theta")
{
    float pdf = cosine_hemisphere_pdf(0.5f);
    CHECK(pdf > 0.0f);
    CHECK(pdf == doctest::Approx(0.5f * M_1_PI_F));
}

// --- microfacet tests ---

TEST_CASE("ggx_ndf peaks at normal direction")
{
    float alpha = 0.5f;
    float NdotH = 1.0f; // half-vector aligned with normal
    float D = ggx_ndf(alpha, NdotH);
    CHECK(D > 0.0f);
}

// --- diffuse BSDF tests ---

TEST_CASE("diffuse_sample produces valid result")
{
    MaterialParams p = make_diffuse_params();
    SurfaceInteraction si = make_test_si();
    bsdf_init(si, p, nullptr);

    BsdfSampleResult r = diffuse_sample(si, 0.5f, 0.5f);

    CHECK(r.pdf > 0.0f);
    CHECK((r.event_type & BSDF_EVENT_DIFFUSE) != 0);
    CHECK((r.event_type & BSDF_EVENT_REFLECTION) != 0);
    // bsdf_over_pdf should be approximately base_color for Lambert
    CHECK(r.bsdf_over_pdf.x == doctest::Approx(p.base_color.x).epsilon(0.01f));
}

TEST_CASE("diffuse_eval returns zero for directions below surface")
{
    MaterialParams p = make_diffuse_params();
    SurfaceInteraction si = make_test_si();
    bsdf_init(si, p, nullptr);
    float3 wi = make_float3(0, -1, 0); // below surface

    BsdfEvalResult r = diffuse_eval(si, wi);
    CHECK(r.pdf == doctest::Approx(0.0f));
}

TEST_CASE("diffuse_eval returns positive for directions above surface")
{
    MaterialParams p = make_diffuse_params();
    SurfaceInteraction si = make_test_si();
    bsdf_init(si, p, nullptr);
    float3 wi = make_float3(0, 1, 0); // above surface

    BsdfEvalResult r = diffuse_eval(si, wi);
    CHECK(r.pdf > 0.0f);
    CHECK(r.bsdf.x > 0.0f);
}

// --- bsdf dispatch tests ---

TEST_CASE("bsdf_init resolves material parameters")
{
    MaterialParams p = make_diffuse_params();
    SurfaceInteraction si = make_test_si();

    bsdf_init(si, p, nullptr);

    CHECK(si.material_type == p.material_type);
    CHECK(si.albedo.x == doctest::Approx(p.base_color.x));
    // roughness should be clamped to minimum
    CHECK(si.roughness >= 0.0001f);
}

TEST_CASE("bsdf_sample dispatches to correct lobe")
{
    MaterialParams p = make_diffuse_params();
    SurfaceInteraction si = make_test_si();
    bsdf_init(si, p, nullptr);

    float4 xi = make_float4(0.3f, 0.7f, 0.5f, 0.5f);
    BsdfSampleResult r = bsdf_sample(si, xi);

    CHECK((r.event_type & BSDF_EVENT_DIFFUSE) != 0);
}

TEST_CASE("emission is resolved by bsdf_init")
{
    MaterialParams p = {};
    p.material_type = MATERIAL_TYPE_DIFFUSE;
    p.base_color = make_float3(0.0f);
    p.emission = make_float3(10.0f);
    p.emission_strength = 100.0f;
    p.roughness = 1.0f;
    p.base_color_tex = -1;
    p.metallic_roughness_tex = -1;
    p.normal_tex = -1;
    p.emission_tex = -1;
    p.occlusion_tex = -1;
    p.transmission_tex = -1;

    SurfaceInteraction si = make_test_si();
    bsdf_init(si, p, nullptr);

    // Emission = emission_color * emission_strength * texture(1,1,1)
    CHECK(si.emission.x == doctest::Approx(1000.0f));
    CHECK(si.emission.y == doctest::Approx(1000.0f));
    CHECK(si.emission.z == doctest::Approx(1000.0f));
}

// --- conductor tests ---

TEST_CASE("conductor_sample produces reflection")
{
    MaterialParams p = {};
    p.material_type = MATERIAL_TYPE_CONDUCTOR;
    p.base_color = make_float3(0.95f, 0.64f, 0.54f); // copper-like F0
    p.roughness = 0.3f;
    p.metallic = 1.0f;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.base_color_tex = -1;
    p.metallic_roughness_tex = -1;
    p.normal_tex = -1;
    p.emission_tex = -1;
    p.occlusion_tex = -1;
    p.transmission_tex = -1;

    SurfaceInteraction si = make_test_si();
    bsdf_init(si, p, nullptr);

    float4 xi = make_float4(0.5f, 0.5f, 0.5f, 0.5f);
    BsdfSampleResult r = bsdf_sample(si, xi);

    if (r.event_type != BSDF_EVENT_ABSORB)
    {
        CHECK((r.event_type & BSDF_EVENT_REFLECTION) != 0);
        CHECK(r.pdf > 0.0f);
    }
}

// --- standard PBR tests ---

TEST_CASE("standard_pbr_sample produces valid result")
{
    MaterialParams p = {};
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;
    p.base_color = make_float3(0.8f, 0.1f, 0.1f);
    p.roughness = 0.5f;
    p.metallic = 0.0f;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.transmission = 0.0f;
    p.base_color_tex = -1;
    p.metallic_roughness_tex = -1;
    p.normal_tex = -1;
    p.emission_tex = -1;
    p.occlusion_tex = -1;
    p.transmission_tex = -1;

    SurfaceInteraction si = make_test_si();
    bsdf_init(si, p, nullptr);

    float4 xi = make_float4(0.3f, 0.7f, 0.2f, 0.1f);
    BsdfSampleResult r = bsdf_sample(si, xi);

    CHECK(r.event_type != BSDF_EVENT_ABSORB);
    CHECK(r.pdf > 0.0f);
}

TEST_CASE("standard_pbr_eval returns positive for valid directions")
{
    MaterialParams p = {};
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;
    p.base_color = make_float3(0.8f, 0.1f, 0.1f);
    p.roughness = 0.5f;
    p.metallic = 0.0f;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.transmission = 0.0f;
    p.base_color_tex = -1;
    p.metallic_roughness_tex = -1;
    p.normal_tex = -1;
    p.emission_tex = -1;
    p.occlusion_tex = -1;
    p.transmission_tex = -1;

    SurfaceInteraction si = make_test_si();
    bsdf_init(si, p, nullptr);

    float3 wi = glm::normalize(make_float3(0.3f, 1.0f, 0.2f));
    BsdfEvalResult r = bsdf_eval(si, wi);

    CHECK(r.pdf > 0.0f);
}

// --- energy conservation ---

TEST_CASE("diffuse bsdf_over_pdf bounded")
{
    MaterialParams p = make_diffuse_params();
    SurfaceInteraction si = make_test_si();
    bsdf_init(si, p, nullptr);

    // Sample many directions and check throughput doesn't explode
    for (int i = 0; i < 100; ++i)
    {
        float u1 = (float)i / 100.0f;
        float u2 = (float)(i * 7 % 100) / 100.0f;
        float4 xi = make_float4(u1, u2, 0.5f, 0.5f);
        BsdfSampleResult r = bsdf_sample(si, xi);

        if (r.event_type != BSDF_EVENT_ABSORB)
        {
            // bsdf_over_pdf should not be negative
            CHECK(r.bsdf_over_pdf.x >= 0.0f);
            CHECK(r.bsdf_over_pdf.y >= 0.0f);
            CHECK(r.bsdf_over_pdf.z >= 0.0f);
            // bsdf_over_pdf should be bounded (energy conservation)
            CHECK(r.bsdf_over_pdf.x <= 2.0f); // allow some margin
        }
    }
}

// ---------------------------------------------------------------------------
// Sample / eval consistency
//
// Multiple importance sampling weighs two strategies by each other's density,
// so bsdf_eval() has to report exactly the density bsdf_sample() draws from, and
// exactly the same f(wo, wi). If they disagree the weights no longer sum to one
// and the estimator is biased — invisibly, because each strategy on its own
// still looks plausible. This is checked pointwise at the sampled direction,
// which is far sharper than any histogram test.
// ---------------------------------------------------------------------------
namespace
{
struct Lcg
{
    uint32_t s = 123456789u;
    float next()
    {
        s = s * 1664525u + 1013904223u;
        return (float)((s >> 8) & 0xFFFFFF) / (float)0x1000000;
    }
};

SurfaceInteraction si_with(uint32_t materialType, float roughness, float metallic,
                           float transmission, float clearcoat, float wo_tilt)
{
    MaterialParams p = make_diffuse_params();
    p.material_type = materialType;
    p.roughness = roughness;
    p.metallic = metallic;
    p.transmission = transmission;
    p.clearcoat = clearcoat;
    p.base_color = make_float3(0.8f, 0.6f, 0.4f);

    SurfaceInteraction si = make_test_si();
    const float s = std::sin(wo_tilt);
    si.wo = normalize(make_float3(s, std::cos(wo_tilt), 0.0f));
    bsdf_init(si, p, nullptr);
    si.exterior_ior = 1.0f;
    return si;
}
} // namespace

TEST_CASE("bsdf_eval reports the density bsdf_sample draws from")
{
    const uint32_t types[] = { MATERIAL_TYPE_DIFFUSE, MATERIAL_TYPE_CONDUCTOR, MATERIAL_TYPE_STANDARD_PBR };
    Lcg rng;
    for (uint32_t type : types)
    {
        for (float roughness : { 0.15f, 0.45f, 1.0f })
        {
            for (float metallic : { 0.0f, 1.0f })
            {
                for (float tilt : { 0.1f, 0.7f, 1.2f })
                {
                    SurfaceInteraction si = si_with(type, roughness, metallic, 0.0f, 0.0f, tilt);
                    int compared = 0;
                    double worstPdf = 0.0, worstF = 0.0;
                    for (int i = 0; i < 4000; ++i)
                    {
                        const float4 xi =
                            make_float4(rng.next(), rng.next(), rng.next(), rng.next());
                        BsdfSampleResult s = bsdf_sample(si, xi);
                        if (s.event_type == BSDF_EVENT_ABSORB ||
                            (s.event_type & BSDF_EVENT_SPECULAR) != 0 || s.pdf <= 1e-4f)
                        {
                            continue; // delta lobes have no density to compare
                        }
                        BsdfEvalResult e = bsdf_eval(si, s.wi);
                        if (e.pdf <= 1e-4f)
                        {
                            continue;
                        }
                        ++compared;
                        worstPdf = std::max(worstPdf, (double)std::fabs(e.pdf - s.pdf) / s.pdf);

                        // And the throughput each route produces for that
                        // direction must agree, once both are in the same cosine
                        // convention -- see the note on the result structs in
                        // bsdf_types.h. This is precisely the quantity multiple
                        // importance sampling assumes is the same on both paths.
                        const float cosWi = std::fabs(dot(si.shading_normal, s.wi));
                        const float3 fromSample = s.bsdf_over_pdf;
                        const float3 fromEval = e.bsdf * (cosWi / e.pdf);
                        const float denom =
                            std::max({ fromEval.x, fromEval.y, fromEval.z, 1e-4f });
                        worstF = std::max(worstF, (double)length(fromSample - fromEval) / denom);
                    }
                    if (compared > 100)
                    {
                        CAPTURE(type);
                        CAPTURE(roughness);
                        CAPTURE(metallic);
                        CAPTURE(tilt);
                        CHECK(worstPdf < 0.02);
                        CHECK(worstF < 0.02);
                    }
                }
            }
        }
    }
}
