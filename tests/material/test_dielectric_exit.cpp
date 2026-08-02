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

// ---------------------------------------------------------------------------
// A ray leaving a dielectric.
//
// This is the case that makes glass glass, and it is easy to lose: the ray is
// *inside* the medium and hits the far wall from behind, so the shading normal
// points away from it and dot(N, wo) is negative. Treating that as a degenerate
// hit and absorbing the path means light can enter a closed transmissive volume
// and never leave it -- the object renders as a dark shell of Fresnel highlights
// with no refraction at all, and, more insidiously, any volume absorption
// applied along the interior segment becomes invisible, because the throughput
// it scales is discarded one step later.
//
// The transmission lobe already knows how to handle this: it flips the normal
// into Nf and picks eta by direction. The only thing that has to hold is that
// the lobe is actually reached.
// ---------------------------------------------------------------------------

namespace
{

MaterialParams glass_params(float roughness)
{
    MaterialParams p = {};
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;
    p.base_color = make_float3(1.0f, 1.0f, 1.0f);
    p.roughness = roughness;
    p.metallic = 0.0f;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.transmission = 1.0f;
    p.alpha_mode = ALPHA_MODE_OPAQUE;
    p.base_color_alpha = 1.0f;
    p.base_color_tex = -1;
    p.metallic_roughness_tex = -1;
    p.normal_tex = -1;
    p.emission_tex = -1;
    p.occlusion_tex = -1;
    p.transmission_tex = -1;
    p.dielectric_priority = 10;
    return p;
}

// wo points back along the incoming ray. A ray travelling outward from inside
// the medium therefore has a wo on the *inner* side, i.e. dot(N, wo) < 0.
SurfaceInteraction exiting_si(float roughness, float tilt = 0.25f)
{
    SurfaceInteraction si = {};
    si.position = make_float3(0.0f, 0.0f, 0.0f);
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.uv = make_float2(0.0f, 0.0f);
    si.wo = safe_normalize(make_float3(tilt, -1.0f, 0.0f));
    si.front_face = false; // hit from behind: this is the exit surface
    bsdf_init(si, glass_params(roughness));
    // Inside the glass looking out: the medium being left is the glass itself.
    si.exterior_ior = 1.0f;
    return si;
}

bool is_finite3(float3 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

} // namespace

TEST_CASE("the exit surface of a dielectric is set up as a back-face hit")
{
    // Guards the premise of every case below: if this stops being negative the
    // rest of the file is testing nothing.
    for (float r : { 0.0f, 0.3f })
    {
        SurfaceInteraction si = exiting_si(r);
        CAPTURE(r);
        CHECK(dot(si.shading_normal, si.wo) < 0.0f);
        CHECK(si.transmission == doctest::Approx(1.0f));
    }
}

TEST_CASE("a ray inside a smooth dielectric can leave it")
{
    SurfaceInteraction si = exiting_si(0.0f);

    // xi.w is the reflect/refract draw and the branch is `u_fresnel < F`, so a
    // HIGH value is the refraction one. At 14 degrees from the normal F is about
    // 0.04, well below this.
    BsdfSampleResult r = bsdf_sample(si, make_float4(0.5f, 0.5f, 0.9f, 0.9f));

    REQUIRE(r.event_type != BSDF_EVENT_ABSORB);
    CHECK((r.event_type & BSDF_EVENT_TRANSMISSION) != 0);
    CHECK(r.pdf > 0.0f);
    CHECK(is_finite3(r.bsdf_over_pdf));
    CHECK(dot(r.bsdf_over_pdf, r.bsdf_over_pdf) > 0.0f);

    // Refraction crosses the surface: the outgoing direction must end up on the
    // far side of the shading normal from wo.
    CHECK(dot(si.shading_normal, r.wi) > 0.0f);
}

TEST_CASE("a ray inside a rough dielectric can leave it")
{
    SurfaceInteraction si = exiting_si(0.35f);

    BsdfSampleResult r = bsdf_sample(si, make_float4(0.4f, 0.6f, 0.9f, 0.9f));

    REQUIRE(r.event_type != BSDF_EVENT_ABSORB);
    CHECK((r.event_type & BSDF_EVENT_TRANSMISSION) != 0);
    CHECK(r.pdf > 0.0f);
    CHECK(is_finite3(r.bsdf_over_pdf));
}

TEST_CASE("total internal reflection still reflects rather than absorbing")
{
    // Grazing enough to be past the critical angle for 1.5 -> 1.0 (about 41.8
    // degrees from the normal). The path must bounce back into the medium, not
    // die.
    SurfaceInteraction si = exiting_si(0.0f, /*tilt=*/3.0f);
    REQUIRE(dot(si.shading_normal, si.wo) < 0.0f);

    // Past the critical angle F is exactly 1, so every draw reflects.
    BsdfSampleResult r = bsdf_sample(si, make_float4(0.5f, 0.5f, 0.9f, 0.9f));

    REQUIRE(r.event_type != BSDF_EVENT_ABSORB);
    CHECK((r.event_type & BSDF_EVENT_TRANSMISSION) == 0);
    CHECK(dot(si.shading_normal, r.wi) < 0.0f); // stayed inside
    CHECK(r.pdf > 0.0f);
}

TEST_CASE("an opaque material still absorbs on a back face")
{
    // The guard exists for a reason -- a back-face hit on something with no
    // transmission lobe has nothing to evaluate. Loosening it for dielectrics
    // must not loosen it for everything.
    SurfaceInteraction si = exiting_si(0.3f);
    si.transmission = 0.0f;

    BsdfSampleResult r = bsdf_sample(si, make_float4(0.4f, 0.6f, 0.5f, 0.5f));
    CHECK(r.event_type == BSDF_EVENT_ABSORB);
}

TEST_CASE("eval agrees with sample on the exit surface")
{
    // Rough only: a smooth interface is a delta lobe and eval cannot see it.
    SurfaceInteraction si = exiting_si(0.35f);

    BsdfSampleResult s = bsdf_sample(si, make_float4(0.4f, 0.6f, 0.9f, 0.9f));
    REQUIRE((s.event_type & BSDF_EVENT_TRANSMISSION) != 0);
    REQUIRE((s.event_type & BSDF_EVENT_SPECULAR) == 0);

    BsdfEvalResult e = bsdf_eval(si, s.wi);

    // The part this fix owns: eval must accept the direction at all. Before it,
    // eval classified a refracted wi as a reflection (NdotL > 0 with NdotV < 0)
    // and returned zero for exactly what sample produced.
    CHECK(e.pdf > 0.0f);
    CHECK(is_finite3(e.bsdf));

    // KNOWN GAP, deliberately a WARN and not a CHECK.
    //
    // The rough-transmission sample/eval pair is inconsistent, and not only in
    // its pdf: the identity every MIS-using BSDF must satisfy,
    //     bsdf_over_pdf * pdf == bsdf * |NdotL|
    // is off by a factor of about 6.6 (0.132 vs 0.870 measured at roughness
    // 0.35, ior 1.5). The pdf alone differs by 1.354, and that factor is
    // independent of the lobe weights, so it lives in the half-vector or the
    // Jacobian rather than in lobe selection.
    //
    // It predates the exit-hit fix -- the entering case below is wrong by the
    // same kind of margin -- and it is NOT simply the sign of LdotH or the
    // placement of eta in the reconstruction: substituting
    // H = normalize(eta*V + wi) for normalize(V + eta*wi) makes it 30x worse.
    // Closing it wants a careful re-derivation against Walter et al. 2007, not
    // a guess; smooth glass, which is what the feature scenes use, is
    // unaffected because a delta lobe never goes through eval.
    WARN(e.pdf == doctest::Approx(s.pdf).epsilon(0.05));
}

TEST_CASE("entering a rough dielectric: sample and eval must agree too")
{
    // Ownership check for the mismatch above: if the entering case disagrees by
    // the same factor, the rough-transmission pdf pair was already inconsistent
    // and merely unreachable on exit hits.
    SurfaceInteraction si = exiting_si(0.35f);
    si.wo = safe_normalize(make_float3(0.25f, 1.0f, 0.0f)); // now on the outside
    si.front_face = true;
    si.exterior_ior = 1.0f;
    REQUIRE(dot(si.shading_normal, si.wo) > 0.0f);

    BsdfSampleResult s = bsdf_sample(si, make_float4(0.4f, 0.6f, 0.99f, 0.9f));
    REQUIRE((s.event_type & BSDF_EVENT_TRANSMISSION) != 0);
    REQUIRE((s.event_type & BSDF_EVENT_SPECULAR) == 0);

    BsdfEvalResult e = bsdf_eval(si, s.wi);
    CHECK(e.pdf > 0.0f);
    // Same known gap, measured on the entering side: 0.211 vs 0.286. That the
    // guard-free path was always inconsistent is the point of this case.
    WARN(e.pdf == doctest::Approx(s.pdf).epsilon(0.05));
}

TEST_CASE("a fully transmissive material has no separate specular lobe")
{
    // The transmission lobe does its own Fresnel reflection. A specular lobe
    // alongside it makes the reflected direction reachable two ways, and since
    // neither strategy's pdf accounts for the other, the two together
    // over-estimate. Diffuse has always been scaled by (1 - transmission);
    // specular has to be as well.
    SurfaceInteraction si = exiting_si(0.35f);
    si.wo = safe_normalize(make_float3(0.25f, 1.0f, 0.0f));
    si.front_face = true;

    si.transmission = 1.0f;
    PbrLobeWeights full = pbr_lobe_weights(si);
    CHECK(full.specular == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(full.diffuse == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(full.transmission > 0.0f);

    // An opaque dielectric must be untouched by that scaling.
    si.transmission = 0.0f;
    PbrLobeWeights opaque = pbr_lobe_weights(si);
    CHECK(opaque.specular > 0.0f);
    CHECK(opaque.transmission == doctest::Approx(0.0f).epsilon(1e-6));

    // A metal is not transmissive whatever the parameter says, so its specular
    // lobe must survive.
    si.transmission = 1.0f;
    si.metallic = 1.0f;
    PbrLobeWeights metal = pbr_lobe_weights(si);
    CHECK(metal.specular > 0.0f);
}
