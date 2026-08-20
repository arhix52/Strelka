// ============================================================================
// test_sheen.cpp
//
// KHR_materials_sheen: the retroreflective fabric layer. Its whole reason to
// exist is what happens at grazing angles -- a GGX lobe falls off exactly where
// a towel or a rug gets brighter, so fabric rendered with roughness alone reads
// as plastic no matter which roughness it is given.
//
// The layer is implemented on top of the existing lobes rather than beside them:
// sheen shares the cosine-sampled lobe with diffuse, because the Charlie
// distribution has no cheap invertible sampling routine. Sharing is what makes
// the pdf stay a single cosine term, and it is also what could quietly break --
// the shared lobe has to remain reachable on a fabric whose albedo is black, or
// the sheen is simply never sampled and the material renders as if the extension
// were not there.
//
// Pinned here:
//   1. sheen 0 leaves the material bit-identical to before the lobe existed
//   2. sheen brightens grazing angles far more than facing ones -- the property
//      the lobe was added for, stated as a measurement rather than a look
//   3. sample and eval agree, since MIS weighs each strategy by the other's
//      density and a term present in one and missing from the other blends two
//      different BRDFs
//   4. a black-albedo fabric still scatters: the shared lobe must stay reachable
//   5. the layer does not manufacture energy
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

MaterialParams cloth_params(float sheen, float sheenRoughness = 0.3f)
{
    MaterialParams p = {};
    p.base_color = make_float3(0.8f, 0.8f, 0.8f);
    p.metallic = 0.0f;
    p.roughness = 0.5f;
    p.ior = 1.3f;
    p.specular = 0.1f;
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
    p.sheen = sheen;
    p.sheen_roughness = sheenRoughness;
    p.sheen_color = make_float3(1.0f);
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

// A direction at `deg` from the normal, in the XZ plane.
float3 dir_at(float deg)
{
    const float r = deg * 3.14159265358979f / 180.0f;
    return make_float3(std::sin(r), 0.0f, std::cos(r));
}

// Directional albedo: what fraction of the light arriving from wo leaves again.
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

TEST_CASE("sheen: weight 0 changes nothing")
{
    // The regression that matters most: every scene without fabric has to render
    // exactly as it did before the lobe was added.
    const MaterialParams without = cloth_params(0.0f);
    MaterialParams zeroed = cloth_params(0.0f);
    zeroed.sheen_color = make_float3(0.2f, 0.7f, 0.4f); // colour set, weight not
    zeroed.sheen_roughness = 0.9f;

    for (const float deg : { 5.0f, 35.0f, 65.0f, 85.0f })
    {
        const float3 wo = dir_at(deg);
        const SurfaceInteraction a = make_si(without, wo);
        const SurfaceInteraction b = make_si(zeroed, wo);
        const float3 wi = dir_at(-deg + 40.0f);

        const BsdfEvalResult ea = bsdf_eval(a, wi);
        const BsdfEvalResult eb = bsdf_eval(b, wi);
        CHECK(ea.bsdf.x == doctest::Approx(eb.bsdf.x));
        CHECK(ea.bsdf.y == doctest::Approx(eb.bsdf.y));
        CHECK(ea.bsdf.z == doctest::Approx(eb.bsdf.z));
        CHECK(ea.pdf == doctest::Approx(eb.pdf));
    }
}

TEST_CASE("sheen: brightens grazing angles more than facing ones")
{
    // The reason the lobe exists. Measured as the ratio of the sheen material's
    // response to the same material without it, at two angles: a term that
    // merely added a constant would move both equally.
    const MaterialParams plain = cloth_params(0.0f);
    const MaterialParams fabric = cloth_params(1.0f, 0.3f);

    const float3 facing = dir_at(10.0f);
    const float3 grazing = dir_at(80.0f);

    // Retroreflection: fabric is brightest looking back along the incoming
    // direction, which is where a GGX lobe has nothing to say.
    const BsdfEvalResult plainFacing = bsdf_eval(make_si(plain, facing), facing);
    const BsdfEvalResult sheenFacing = bsdf_eval(make_si(fabric, facing), facing);
    const BsdfEvalResult plainGrazing = bsdf_eval(make_si(plain, grazing), grazing);
    const BsdfEvalResult sheenGrazing = bsdf_eval(make_si(fabric, grazing), grazing);

    REQUIRE(plainFacing.bsdf.x > 0.0f);
    REQUIRE(plainGrazing.bsdf.x > 0.0f);

    const float gainFacing = sheenFacing.bsdf.x / plainFacing.bsdf.x;
    const float gainGrazing = sheenGrazing.bsdf.x / plainGrazing.bsdf.x;

    // Head-on, Charlie has almost nothing to add -- the distribution peaks at
    // grazing -- while the base is scaled down by what the layer above it took.
    // So the facing response should sit just under where it started, not above:
    // a sheen that brightens a surface seen flat-on is a sheen that is really
    // just a diffuse boost.
    CHECK(gainFacing == doctest::Approx(1.0f).epsilon(0.02));
    CHECK(gainGrazing > 2.0f);
}

TEST_CASE("sheen: sample and eval agree")
{
    // sample() evaluates every lobe at the direction it drew so MIS has a
    // consistent density. If sheen is added on one path and not the other, the
    // two describe different materials and the weights blend them.
    const MaterialParams fabric = cloth_params(1.0f, 0.4f);

    for (const float deg : { 15.0f, 45.0f, 75.0f })
    {
        const SurfaceInteraction si = make_si(fabric, dir_at(deg));
        Lcg rng(0xC10Du + static_cast<std::uint32_t>(deg));
        int checked = 0;
        for (int i = 0; i < 400 && checked < 60; ++i)
        {
            const BsdfSampleResult s =
                bsdf_sample(si, make_float4(rng.next(), rng.next(), rng.next(), rng.next()));
            if (s.event_type == BSDF_EVENT_ABSORB || s.pdf <= 0.0f)
                continue;
            if ((s.event_type & BSDF_EVENT_SPECULAR) != 0)
                continue; // delta lobes have no density to compare against

            const BsdfEvalResult e = bsdf_eval(si, s.wi);
            const float cosL = std::fabs(dot(s.wi, si.shading_normal));
            // sample returns f * cos / pdf; eval returns f and the same pdf.
            const float expected = e.bsdf.x * cosL / std::max(e.pdf, 1e-10f);
            CHECK(s.pdf == doctest::Approx(e.pdf).epsilon(0.02));
            CHECK(s.bsdf_over_pdf.x == doctest::Approx(expected).epsilon(0.02));
            ++checked;
        }
        CHECK(checked > 0);
    }
}

TEST_CASE("sheen: a black fabric still scatters")
{
    // Sheen shares the diffuse lobe's cosine sampling, and the share is weighted
    // by the diffuse weight -- which is zero for a black base colour. Without the
    // max in pbr_lobe_weights the lobe is then never selected and the sheen is
    // silently absent on exactly the materials it is most visible on.
    MaterialParams fabric = cloth_params(1.0f, 0.3f);
    fabric.base_color = make_float3(0.0f, 0.0f, 0.0f);

    const SurfaceInteraction si = make_si(fabric, dir_at(70.0f));
    Lcg rng(99u);
    float3 total = make_float3(0.0f);
    int diffuseSamples = 0;
    for (int i = 0; i < 2000; ++i)
    {
        const BsdfSampleResult s =
            bsdf_sample(si, make_float4(rng.next(), rng.next(), rng.next(), rng.next()));
        if (s.event_type == BSDF_EVENT_ABSORB || s.pdf <= 0.0f)
            continue;
        if ((s.event_type & BSDF_EVENT_DIFFUSE) != 0)
            ++diffuseSamples;
        total = total + s.bsdf_over_pdf;
    }
    CHECK(diffuseSamples > 0);
    CHECK(total.x > 0.0f);
}

TEST_CASE("sheen: does not manufacture energy")
{
    // The layer is additive over the base, like the existing clearcoat, so it is
    // not energy-exact -- but it must not push a fabric past the light that fell
    // on it, which is what a missing visibility term or a mis-normalised Charlie
    // distribution would do.
    for (const float deg : { 20.0f, 50.0f, 78.0f })
    {
        const float3 wo = dir_at(deg);
        const float3 albedo = integrate_albedo(cloth_params(1.0f, 0.3f), wo, 20000, 7u);
        CHECK(albedo.x <= 1.0f);
        CHECK(albedo.y <= 1.0f);
        CHECK(albedo.z <= 1.0f);
        CHECK(albedo.x > 0.0f);
    }
}
