// ============================================================================
// test_diffuse_transmission.cpp
//
// KHR_materials_diffuse_transmission: light that enters a surface and leaves
// diffusely on the far side. A leaf, not a pane of glass -- which is why it is a
// separate lobe from `transmission`, whose specular refraction through an
// interface would make a pine needle read as a shard.
//
// The spec defines the result as mix(diffuse_brdf, diffuse_btdf, weight), so the
// lobe *splits* the diffuse response rather than adding to it. That is the
// invariant most easily lost: it would be natural to add a transmitted lobe
// alongside the reflected one, and the material would then be brighter than the
// light falling on it -- visible as a canopy that glows rather than one that is
// backlit, which is a difference nobody catches by eye.
//
// Pinned here:
//   1. weight 0 leaves the material bit-identical to before the lobe existed
//   2. weight 1 sends the whole diffuse response to the far side and nothing back
//   3. reflected + transmitted never exceeds what arrived, at any weight
//   4. sample and eval agree, on the far side too -- MIS weighs each strategy by
//      the other's density, and next-event estimation through a canopy is
//      exactly where this lobe earns its keep
//   5. a back-face hit still scatters: a cutout leaf is hit from both sides
//      constantly, and a lobe that returns nothing there makes foliage opaque
//      from behind
// ============================================================================

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

MaterialParams leaf_params(float weight)
{
    MaterialParams p = {};
    p.base_color = make_float3(0.13f, 0.31f, 0.09f);
    p.metallic = 0.0f;
    p.roughness = 0.6f;
    p.ior = 1.45f;
    p.specular = 0.5f;
    p.specular_color = make_float3(1.0f);
    p.transmission = 0.0f;
    p.clearcoat = 0.0f;
    p.clearcoat_roughness = 0.3f;
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
    // Deliberately not grey and not the base colour: a leaf transmits greener
    // than it reflects, and a slip that reads one where it means the other would
    // hide behind a grey.
    p.diffuse_transmission_color = make_float3(0.21f, 0.47f, 0.11f);
    p.diffuse_transmission = weight;
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

// Hemispherical response, split by which side of the surface it left on.
struct Response
{
    float3 reflected;
    float3 transmitted;
};

Response integrate(const MaterialParams& p, float3 wo, int samples, std::uint32_t seed)
{
    Response r{ make_float3(0.0f), make_float3(0.0f) };
    SurfaceInteraction si = make_si(p, wo);
    Lcg rng(seed);
    for (int i = 0; i < samples; ++i)
    {
        const BsdfSampleResult s =
            bsdf_sample(si, make_float4(rng.next(), rng.next(), rng.next(), rng.next()));
        if (s.event_type == BSDF_EVENT_ABSORB || s.pdf <= 0.0f)
            continue;
        // bsdf_over_pdf already carries the cosine; see bsdf_types.h.
        if (dot(s.wi, si.shading_normal) * dot(wo, si.shading_normal) > 0.0f)
            r.reflected = r.reflected + s.bsdf_over_pdf;
        else
            r.transmitted = r.transmitted + s.bsdf_over_pdf;
    }
    const float inv = 1.0f / static_cast<float>(samples);
    r.reflected = r.reflected * inv;
    r.transmitted = r.transmitted * inv;
    return r;
}

} // namespace

TEST_CASE("weight 0 leaves the material exactly as it was")
{
    const float3 wo = glm::normalize(make_float3(0.3f, 0.1f, 0.9f));
    MaterialParams off = leaf_params(0.0f);
    // Also blank the colour, so a lobe that leaks at weight 0 cannot hide behind
    // a tint that happens to resemble the albedo.
    off.diffuse_transmission_color = make_float3(0.0f);

    const Response r = integrate(off, wo, 4096, 12345u);
    CHECK(r.transmitted.x == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(r.transmitted.y == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(r.transmitted.z == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(r.reflected.y > 0.05f); // the diffuse lobe is untouched and still there
}

TEST_CASE("weight 1 sends the diffuse response through and keeps none of it")
{
    const float3 wo = glm::normalize(make_float3(0.0f, 0.0f, 1.0f));
    const Response r = integrate(leaf_params(1.0f), wo, 8192, 777u);

    CHECK(r.transmitted.y > 0.1f);
    // What returns is specular only: at weight 1 no diffuse reflection remains,
    // and the leaf's green is on the far side.
    CHECK(r.reflected.y < r.transmitted.y * 0.5f);
    // The transmitted tint is the transmission colour, not the albedo. Those
    // differ by roughly 1.5x in green here, which a mixed-up field would show.
    CHECK(r.transmitted.y / fmaxf(r.transmitted.x, 1e-6f) ==
          doctest::Approx(0.47f / 0.21f).epsilon(0.15));
}

TEST_CASE("turning the lobe up does not add energy, only move it")
{
    // The failure this guards against is additive rather than split: a canopy
    // that transmits without giving up the matching reflection is brighter than
    // the sky behind it.
    //
    // Measured against the material's own weight-0 response rather than against
    // 1.0, because a white glTF surface already sits a few percent over -- the
    // diffuse lobe does not subtract the specular Fresnel and the GGX
    // multiple-scattering compensation adds a little more. That offset is a
    // property of the standard model and predates this lobe; asserting on it
    // here would be testing the wrong thing and would move whenever the
    // compensation fit is retuned.
    const float weights[] = { 0.25f, 0.5f, 0.75f, 1.0f };
    const float3 directions[] = { glm::normalize(make_float3(0.0f, 0.0f, 1.0f)),
                                  glm::normalize(make_float3(0.6f, 0.0f, 0.8f)),
                                  glm::normalize(make_float3(0.9f, 0.2f, 0.35f)) };
    for (float3 wo : directions)
    {
        MaterialParams base = leaf_params(0.0f);
        // White on both sides: a gain is then unambiguous rather than hidden
        // under a dark albedo.
        base.base_color = make_float3(1.0f);
        base.diffuse_transmission_color = make_float3(1.0f);
        const Response r0 = integrate(base, wo, 8192, 4242u);
        const float3 t0 = r0.reflected + r0.transmitted;

        for (float weight : weights)
        {
            CAPTURE(weight);
            CAPTURE(wo.z);
            MaterialParams p = base;
            p.diffuse_transmission = weight;
            const Response r = integrate(p, wo, 8192, 4242u);
            const float3 total = r.reflected + r.transmitted;
            CHECK(total.x <= t0.x * 1.01f);
            CHECK(total.y <= t0.y * 1.01f);
            CHECK(total.z <= t0.z * 1.01f);
            // And it does move: at weight 1 essentially all of the diffuse
            // response is on the far side.
            if (weight >= 1.0f)
            {
                CHECK(r.transmitted.y > 0.5f * t0.y);
            }
        }
    }
}

TEST_CASE("sample and eval agree on the far side")
{
    const float3 wo = glm::normalize(make_float3(0.25f, -0.15f, 0.95f));
    MaterialParams p = leaf_params(0.6f);
    SurfaceInteraction si = make_si(p, wo);
    Lcg rng(31337u);

    int checked = 0;
    for (int i = 0; i < 3000; ++i)
    {
        const BsdfSampleResult s =
            bsdf_sample(si, make_float4(rng.next(), rng.next(), rng.next(), rng.next()));
        if (s.event_type != BSDF_EVENT_DIFFUSE_TRANSMISSION || s.pdf <= 0.0f)
            continue;
        ++checked;

        const BsdfEvalResult e = bsdf_eval(si, s.wi);
        CAPTURE(i);
        // A direction the sampler produces with zero claimed density gets an
        // infinite MIS weight, so this is the sharpest of the checks.
        CHECK(e.pdf > 0.0f);
        CHECK(e.pdf == doctest::Approx(s.pdf).epsilon(1e-3));

        // bsdf_over_pdf carries the cosine, bsdf does not.
        const float ndotl = fabsf(dot(s.wi, si.shading_normal));
        CHECK(s.bsdf_over_pdf.y * s.pdf == doctest::Approx(e.bsdf.y * ndotl).epsilon(1e-3));
    }
    CHECK(checked > 100); // the lobe has to actually be reachable
}

TEST_CASE("a back-face hit still scatters")
{
    // A cutout leaf is a single sheet: rays arrive from behind as often as from
    // in front. With no specular transmission to fall back on, this lobe is the
    // only thing that can answer, and returning nothing makes foliage opaque
    // from one side -- which reads as a shadowing bug, not a BSDF one.
    const float3 wo = glm::normalize(make_float3(0.2f, 0.1f, -0.97f)); // below the surface
    MaterialParams p = leaf_params(0.8f);
    SurfaceInteraction si = make_si(p, wo);
    Lcg rng(99u);

    int scattered = 0;
    float3 sum = make_float3(0.0f);
    for (int i = 0; i < 2000; ++i)
    {
        const BsdfSampleResult s =
            bsdf_sample(si, make_float4(rng.next(), rng.next(), rng.next(), rng.next()));
        if (s.event_type == BSDF_EVENT_ABSORB || s.pdf <= 0.0f)
            continue;
        ++scattered;
        sum = sum + s.bsdf_over_pdf;
        // It left on the far side from where it came, which for a back-face hit
        // means the front.
        CHECK(dot(s.wi, si.shading_normal) > 0.0f);
    }
    CHECK(scattered > 1500);
    CHECK(sum.y / static_cast<float>(scattered) > 0.05f);
}
