#include <doctest/doctest.h>

#include <strelka/material/material_math.h>
#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_params.h>
#include <strelka/material/surface_interaction.h>
#include <strelka/material/sampling.h>
#include <strelka/material/fresnel.h>
#include <strelka/material/microfacet.h>
#include <strelka/material/shading_frame.h>
#include <strelka/material/bsdf.h>

#include <shading/nee_pairing.h>

#include <cmath>

// ---------------------------------------------------------------------------
// An opaque surface hit from behind.
//
// Nothing culls a back face here, so this hit is ordinary: a leaf card seen
// from its underside, or a rock whose decimated shell has the winding inverted.
// Neither has an interior to be inside of, so the far side is the near side.
//
// Two other things arrive as dot(N, wo) < 0 and neither is this, which is what
// most of the cases below are pinning down:
//
//   * a ray leaving a dielectric -- test_dielectric_exit.cpp owns that one, and
//     the transmission lobe answers it by flipping the normal itself;
//   * a triangle facing the camera whose normal map tipped the shading normal
//     past the viewer. That is a different defect with a different fix, and
//     flipping it costs 06_normalmap 1.8% against Cycles.
//
// What the flip is worth: on the pine forest 40331 primary hits, 13.2% of the
// frame, were absorbing with the shading normal below the geometric horizon,
// and 99.7% of them were geometric back faces. They came out exactly black
// rather than merely dark, because the closest-hit program terminates on absorb
// above next-event estimation, so the pixel lost its direct lighting too.
// ---------------------------------------------------------------------------

namespace
{

MaterialParams opaque_params(float roughness)
{
    MaterialParams p = {};
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;
    p.base_color = make_float3(0.6f, 0.55f, 0.5f);
    p.roughness = roughness;
    p.metallic = 0.0f;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.transmission = 0.0f;
    p.diffuse_transmission = 0.0f;
    p.alpha_mode = ALPHA_MODE_OPAQUE;
    p.base_color_alpha = 1.0f;
    p.base_color_tex = -1;
    p.metallic_roughness_tex = -1;
    p.normal_tex = -1;
    p.emission_tex = -1;
    p.occlusion_tex = -1;
    p.transmission_tex = -1;
    p.dielectric_priority = 0;
    return p;
}

/// The shading normal points up; `wo` comes from below. Both sides of the same
/// opaque surface, so the two differ only in which one the viewer is on.
SurfaceInteraction back_hit(float roughness, float tilt = 0.3f)
{
    SurfaceInteraction si = {};
    si.position = make_float3(0.0f, 0.0f, 0.0f);
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.uv = make_float2(0.0f, 0.0f);
    si.wo = safe_normalize(make_float3(tilt, -1.0f, 0.0f));
    si.front_face = false;
    bsdf_init(si, opaque_params(roughness));
    si.exterior_ior = 1.0f;
    return si;
}

SurfaceInteraction front_hit(float roughness, float tilt = 0.3f)
{
    SurfaceInteraction si = back_hit(roughness, tilt);
    si.wo = safe_normalize(make_float3(tilt, 1.0f, 0.0f));
    si.front_face = true;
    return si;
}

bool is_finite3(float3 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

} // namespace

TEST_CASE("the predicate flips an opaque back hit and nothing else")
{
    // The case at hand: geometry hit from behind, opaque.
    CHECK(opaqueBackHitFlipsFrame(false, -0.7f, 0.0f, 0.0f));

    // A front hit is already the right way round.
    CHECK_FALSE(opaqueBackHitFlipsFrame(true, 0.7f, 0.0f, 0.0f));

    // The distinction the ladder paid for: a triangle facing the camera whose
    // normal map tipped the shading normal past the horizon is NOT this defect.
    // Flipping it shades a surface pointing away from the light that lights it,
    // and 06_normalmap goes from 1.061 to 1.079 against Cycles.
    CHECK_FALSE(opaqueBackHitFlipsFrame(true, -0.7f, 0.0f, 0.0f));

    // A ray leaving glass. Flipping here would take the hit away from the
    // transmission lobe, which is the one thing that must not happen.
    CHECK_FALSE(opaqueBackHitFlipsFrame(false, -0.7f, 1.0f, 0.0f));

    // A leaf lit through its own thickness answers with its own lobe.
    CHECK_FALSE(opaqueBackHitFlipsFrame(false, -0.7f, 0.0f, 0.5f));

    // Exactly edge-on is not a hit any lobe can carry, but it must not be left
    // on the side that absorbs.
    CHECK(opaqueBackHitFlipsFrame(false, 0.0f, 0.0f, 0.0f));
}

TEST_CASE("the back-face premise holds")
{
    // If this stops being negative every case below is testing nothing.
    for (float r : { 0.05f, 0.4f, 0.9f })
    {
        SurfaceInteraction si = back_hit(r);
        CAPTURE(r);
        CHECK(dot(si.shading_normal, si.wo) < 0.0f);
        CHECK(si.transmission == doctest::Approx(0.0f));
        CHECK(si.diffuse_transmission == doctest::Approx(0.0f));
    }
}

TEST_CASE("an opaque surface hit from behind scatters instead of absorbing")
{
    for (float r : { 0.05f, 0.4f, 0.9f })
    {
        CAPTURE(r);
        SurfaceInteraction si = back_hit(r);
        BsdfSampleResult s = bsdf_sample(si, make_float4(0.4f, 0.6f, 0.3f, 0.2f));

        REQUIRE(s.event_type != BSDF_EVENT_ABSORB);
        CHECK(s.pdf > 0.0f);
        CHECK(is_finite3(s.bsdf_over_pdf));
        CHECK(dot(s.bsdf_over_pdf, s.bsdf_over_pdf) > 0.0f);

        // It scattered back out the side it was hit from, which is the side the
        // viewer is on -- not through the surface.
        CHECK(dot(si.shading_normal, s.wi) < 0.0f);
    }
}

TEST_CASE("light reaching the lit side of a back hit is evaluated, not discarded")
{
    SurfaceInteraction si = back_hit(0.4f);

    // A light below the surface, on the same side as the viewer.
    const float3 wi = safe_normalize(make_float3(-0.4f, -1.0f, 0.2f));
    BsdfEvalResult e = bsdf_eval(si, wi);

    CHECK(e.pdf > 0.0f);
    CHECK(is_finite3(e.bsdf));
    CHECK(dot(e.bsdf, e.bsdf) > 0.0f);
}

TEST_CASE("eval accepts what sample produced, from behind as from in front")
{
    // The property that makes MIS legal: the two describe one BRDF. A direction
    // sampling can return must be a direction eval scores, or the estimate
    // blends two different materials.
    for (float r : { 0.15f, 0.5f, 0.85f })
    {
        for (int i = 0; i < 8; ++i)
        {
            const float u = (i + 0.5f) / 8.0f;
            CAPTURE(r);
            CAPTURE(u);

            SurfaceInteraction si = back_hit(r);
            BsdfSampleResult s = bsdf_sample(si, make_float4(u, 1.0f - u, 0.3f, 0.2f));
            if (s.event_type == BSDF_EVENT_ABSORB || (s.event_type & BSDF_EVENT_SPECULAR) != 0)
            {
                continue; // a rejected draw, or a delta lobe eval cannot score
            }

            BsdfEvalResult e = bsdf_eval(si, s.wi);
            CHECK(e.pdf > 0.0f);
            CHECK(dot(e.bsdf, e.bsdf) > 0.0f);
        }
    }
}

TEST_CASE("the two sides of one opaque surface shade the same")
{
    // The whole claim of the flip: an opaque surface seen from behind is the
    // same surface seen from the front. Mirroring the viewer and the light
    // through the surface has to give the same value back.
    for (float r : { 0.2f, 0.6f })
    {
        CAPTURE(r);
        const float3 wiFront = safe_normalize(make_float3(-0.4f, 1.0f, 0.2f));
        const float3 wiBack = safe_normalize(make_float3(-0.4f, -1.0f, 0.2f));

        BsdfEvalResult front = bsdf_eval(front_hit(r), wiFront);
        BsdfEvalResult back = bsdf_eval(back_hit(r), wiBack);

        REQUIRE(front.pdf > 0.0f);
        REQUIRE(back.pdf > 0.0f);
        CHECK(back.bsdf.x == doctest::Approx(front.bsdf.x).epsilon(1e-4));
        CHECK(back.bsdf.y == doctest::Approx(front.bsdf.y).epsilon(1e-4));
        CHECK(back.bsdf.z == doctest::Approx(front.bsdf.z).epsilon(1e-4));
        CHECK(back.pdf == doctest::Approx(front.pdf).epsilon(1e-4));
    }
}

TEST_CASE("a ray leaving a dielectric is still routed to the transmission lobe")
{
    // The regression this fix could most easily cause: glass has to keep
    // reading a back hit as an exit rather than as an opaque underside.
    MaterialParams glass = opaque_params(0.0f);
    glass.transmission = 1.0f;

    SurfaceInteraction si = back_hit(0.0f);
    bsdf_init(si, glass);
    si.exterior_ior = 1.0f;

    REQUIRE(dot(si.shading_normal, si.wo) < 0.0f);
    CHECK_FALSE(opaqueBackHitFlipsFrame(si.front_face, dot(si.shading_normal, si.wo), si.transmission,
                                        si.diffuse_transmission));

    BsdfSampleResult s = bsdf_sample(si, make_float4(0.5f, 0.5f, 0.9f, 0.9f));
    REQUIRE(s.event_type != BSDF_EVENT_ABSORB);
    CHECK((s.event_type & BSDF_EVENT_TRANSMISSION) != 0);
    // Refraction crosses the surface; the flip would have kept it on this side.
    CHECK(dot(si.shading_normal, s.wi) > 0.0f);
}

TEST_CASE("the shaded frame is what both halves of the estimate must be told")
{
    // Unflipped cases pass the geometry straight through.
    {
        ShadedFrame f = shadedFrame(true, 0.7f, 0.0f, 0.0f);
        CHECK(f.frontFace);
        CHECK(f.normalSign == doctest::Approx(1.0f));
    }
    {
        // A dielectric exit stays a back face: the transmission lobe owns it.
        ShadedFrame f = shadedFrame(false, -0.7f, 1.0f, 0.0f);
        CHECK_FALSE(f.frontFace);
        CHECK(f.normalSign == doctest::Approx(1.0f));
    }
    {
        // A leaf lit through its own thickness, likewise.
        ShadedFrame f = shadedFrame(false, -0.7f, 0.0f, 0.5f);
        CHECK_FALSE(f.frontFace);
        CHECK(f.normalSign == doctest::Approx(1.0f));
    }
    {
        // The flipped case: shaded as a front face, with the normal negated.
        ShadedFrame f = shadedFrame(false, -0.7f, 0.0f, 0.0f);
        CHECK(f.frontFace);
        CHECK(f.normalSign == doctest::Approx(-1.0f));
    }
}

TEST_CASE("a flipped hit offers and pairs over the same hemisphere it scatters into")
{
    // The property that keeps MIS legal after the flip: the directions
    // next-event estimation offers, the directions the bounce is weighted over,
    // and the directions the BSDF actually samples all have to be one set. The
    // failure this guards is the one that withholds the weight from a bounce the
    // estimate did offer, which counts the light about twice.
    SurfaceInteraction si = back_hit(0.4f);
    const ShadedFrame f = shadedFrame(si.front_face, dot(si.shading_normal, si.wo), si.transmission,
                                      si.diffuse_transmission);
    REQUIRE(f.normalSign == doctest::Approx(-1.0f));

    for (int i = 0; i < 16; ++i)
    {
        const float u = (i + 0.5f) / 16.0f;
        CAPTURE(u);
        BsdfSampleResult s = bsdf_sample(si, make_float4(u, 1.0f - u, 0.3f, 0.2f));
        if (s.event_type == BSDF_EVENT_ABSORB || (s.event_type & BSDF_EVENT_SPECULAR) != 0)
        {
            continue;
        }
        const float nDotDir = f.normalSign * dot(si.shading_normal, s.wi);

        // Sampled into the hemisphere the shaded frame calls "above".
        CHECK(nDotDir > 0.0f);
        // So the estimate would have offered it, and the bounce pairs with it.
        CHECK(neeProposesDirection(false, f.frontFace, nDotDir));
        CHECK(neePairsWithBounce(true, false, f.frontFace, nDotDir));
    }
}
