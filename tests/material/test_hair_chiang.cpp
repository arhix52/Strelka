#include <doctest/doctest.h>

#include <strelka/material/bsdf.h>
#include <strelka/material/material_params.h>
#include <strelka/material/surface_interaction.h>

#include "../support/sampling.h"

#include <cmath>

using oka::test::stratum;

namespace
{

SurfaceInteraction hair_si(float tilt_deg, float roughness = 0.3f)
{
    SurfaceInteraction si{};
    const float a = tilt_deg * (float)M_PI / 180.0f;
    // View in the XZ plane; strand runs along +Y (tangent).
    si.wo = safe_normalize(make_float3(std::sin(a), 0.0f, std::cos(a)));
    si.tangent = make_float3(0.0f, 1.0f, 0.0f);
    si.shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    si.geometry_normal = si.shading_normal;
    si.bitangent = make_float3(1.0f, 0.0f, 0.0f);
    si.albedo = make_float3(0.4f, 0.22f, 0.1f);
    si.roughness = roughness;
    si.anisotropy = roughness; // radial
    si.ior = 1.55f;
    si.clearcoat = 0.0f;
    si.material_type = MATERIAL_TYPE_HAIR;
    si.exterior_ior = 1.0f;
    return si;
}

} // namespace

TEST_CASE("hair Chiang samples produce a finite glossy lobe")
{
    SurfaceInteraction si = hair_si(25.0f, 0.35f);
    int ok = 0;
    float3 mean = make_float3(0.0f);
    for (int i = 0; i < 1024; ++i)
    {
        const float u1 = stratum((i * 3) % 1024, 1024);
        const float u2 = stratum((i * 7) % 1024, 1024);
        const float u3 = stratum((i * 11) % 1024, 1024);
        BsdfSampleResult s = bsdf_sample(si, make_float4(u1, u2, u3, 0.0f));
        if (s.event_type == BSDF_EVENT_ABSORB)
            continue;
        CHECK(s.pdf > 0.0f);
        CHECK(std::isfinite(s.bsdf_over_pdf.x));
        CHECK(std::isfinite(s.bsdf_over_pdf.y));
        CHECK(std::isfinite(s.bsdf_over_pdf.z));
        CHECK((s.event_type & BSDF_EVENT_GLOSSY) != 0);
        mean = mean + s.wi;
        ++ok;
    }
    REQUIRE(ok > 200);
    mean = mean * (1.0f / (float)ok);
    CHECK(std::isfinite(mean.x));
}

TEST_CASE("hair Chiang sample and eval agree on the sampled direction")
{
    SurfaceInteraction si = hair_si(20.0f, 0.3f);
    int checked = 0;
    for (int i = 0; i < 800 && checked < 40; ++i)
    {
        const float u1 = stratum((i * 5) % 800, 800);
        const float u2 = stratum((i * 11) % 800, 800);
        const float u3 = stratum((i * 17) % 800, 800);
        BsdfSampleResult s = bsdf_sample(si, make_float4(u1, u2, u3, 0.0f));
        if (s.event_type == BSDF_EVENT_ABSORB || s.pdf < 1e-6f)
            continue;

        BsdfEvalResult e = bsdf_eval(si, s.wi);
        CAPTURE(s.pdf);
        CAPTURE(e.pdf);
        CHECK(e.pdf == doctest::Approx(s.pdf).epsilon(0.08));

        const float cos_n = std::fabs(dot(si.shading_normal, s.wi));
        const float expected = e.bsdf.x * cos_n / std::max(e.pdf, 1e-10f);
        CHECK(s.bsdf_over_pdf.x == doctest::Approx(expected).epsilon(0.12));
        ++checked;
    }
    CHECK(checked > 0);
}

TEST_CASE("hair Chiang is brighter in transmission than a dielectric cylinder")
{
    // The defect: a rough dielectric at the same colour and roughness under-
    // counts the light that crosses the fibre. Chiang's TT/TRT lobes should
    // put more energy into the forward hemisphere.
    SurfaceInteraction hair = hair_si(15.0f, 0.4f);

    SurfaceInteraction cyl = hair;
    cyl.material_type = MATERIAL_TYPE_STANDARD_PBR;
    cyl.metallic = 0.0f;
    cyl.transmission = 0.0f;
    cyl.specular = 0.5f;
    cyl.specular_color = make_float3(1.0f);

    float hair_fwd = 0.0f, cyl_fwd = 0.0f;
    const float3 forward = make_float3(0.0f) - hair.wo;
    for (int i = 0; i < 2048; ++i)
    {
        const float u1 = stratum((i * 3) % 2048, 2048);
        const float u2 = stratum((i * 7) % 2048, 2048);
        const float u3 = stratum((i * 13) % 2048, 2048);
        const float u4 = stratum((i * 19) % 2048, 2048);

        BsdfSampleResult hs = bsdf_sample(hair, make_float4(u1, u2, u3, u4));
        if (hs.event_type != BSDF_EVENT_ABSORB && dot(hs.wi, forward) > 0.0f)
            hair_fwd += luminance(hs.bsdf_over_pdf);

        BsdfSampleResult cs = bsdf_sample(cyl, make_float4(u1, u2, u3, u4));
        if (cs.event_type != BSDF_EVENT_ABSORB && dot(cs.wi, forward) > 0.0f)
            cyl_fwd += luminance(cs.bsdf_over_pdf);
    }
    // Not a hard ratio -- just that the hair lobe actually transmits forward
    // energy the cylinder was swallowing.
    CHECK(hair_fwd > cyl_fwd * 1.2f);
}

TEST_CASE("hair Chiang scatters over the whole sphere, not a hemisphere")
{
    // Why this is worth asserting: a renderer that treats hair as a surface BRDF
    // will test the shading hemisphere before connecting to a light, and will
    // offset a transmitted bounce into the strand's interior. Both are wrong here
    // and neither is visible in an aggregate image metric -- they cancel, one
    // darkening the groom and the other adding a second whole-fibre event on the
    // far wall. What makes them wrong is the property below: most of this lobe's
    // energy leaves on the far side of the shading normal, because TT is the
    // dominant term of a bright fibre. See docs/open-defects.md entry 3.
    SurfaceInteraction si = hair_si(20.0f, 0.35f);
    si.albedo = make_float3(0.42f, 0.22f, 0.10f);

    float below = 0.0f, above = 0.0f;
    int below_n = 0;
    const int kSamples = 4096;
    for (int i = 0; i < kSamples; ++i)
    {
        const float u1 = stratum((i * 3) % kSamples, kSamples);
        const float u2 = stratum((i * 7) % kSamples, kSamples);
        const float u3 = stratum((i * 13) % kSamples, kSamples);
        BsdfSampleResult s = bsdf_sample(si, make_float4(u1, u2, u3, 0.0f));
        if (s.event_type == BSDF_EVENT_ABSORB)
            continue;
        const float w = luminance(s.bsdf_over_pdf);
        if (dot(si.shading_normal, s.wi) < 0.0f)
        {
            below += w;
            ++below_n;
        }
        else
        {
            above += w;
        }
    }
    REQUIRE(below_n > 100);
    CHECK(below > above);

    // And eval() has to agree with sample() there: a direction on the far side is
    // a real one to ask about, not a zero. Its pdf being positive is what lets
    // next-event estimation reach it and what makes the MIS weights add to one.
    int nonzero = 0;
    for (int i = 0; i < 64; ++i)
    {
        const float phi = (float)(2.0 * M_PI * (i + 0.5) / 64.0);
        // Around the fibre, on the far side of the shading normal.
        const float3 wi = safe_normalize(make_float3(0.35f * std::cos(phi),
                                                     0.35f * std::sin(phi), -1.0f));
        REQUIRE(dot(si.shading_normal, wi) < 0.0f);
        BsdfEvalResult e = bsdf_eval(si, wi);
        if (e.pdf > 0.0f && luminance(e.bsdf) > 0.0f)
            ++nonzero;
    }
    CHECK(nonzero == 64);
}
