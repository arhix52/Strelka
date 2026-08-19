#include <doctest/doctest.h>

#include <strelka/material/material_math.h>
#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_params.h>
#include <strelka/material/surface_interaction.h>
#include <strelka/material/sampling.h>
#include <strelka/material/fresnel.h>
#include <strelka/material/microfacet.h>
#include <strelka/material/bsdf.h>

#include <nee_pairing.h>

#include <cmath>
#include <cstdint>

// ============================================================================
// test_smooth_lobe.cpp -- what decides whether a vertex makes a next-event
// estimate.
//
// bsdf_has_smooth_lobe() answers "is there anything here a light connection
// could reach", and neeRunsAtVertex() turns that into the decision. The reason
// both exist is that the integrators used to ask a different question: they drew
// a BSDF sample first and gated next-event estimation on whether *that draw*
// came back non-delta.
//
// That is a coin flip belonging to the other half of the estimate. On a material
// with both a delta lobe and a smooth one -- a clearcoat over a diffuse base is
// the ordinary case, and glTF's default coat roughness is 0 -- the smooth lobe's
// direct lighting was then delivered only on the draws where the delta lobe lost
// the lobe selection, and the rest of it was simply lost. The cases below
// measure how often that happens, so the gap between "the material has a smooth
// lobe" and "this draw produced one" is a number in the test rather than an
// argument in a comment.
// ============================================================================

namespace
{

struct FixedSeedSampler
{
    std::uint32_t state;

    explicit FixedSeedSampler(std::uint32_t seed) : state(seed | 1u)
    {
    }

    float next()
    {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>((state >> 8) & 0xFFFFFFu) / static_cast<float>(0x1000000);
    }

    float4 next4()
    {
        const float a = next();
        const float b = next();
        const float c = next();
        const float d = next();
        return make_float4(a, b, c, d);
    }
};

struct Material
{
    unsigned int type = MATERIAL_TYPE_STANDARD_PBR;
    float3 baseColor = make_float3(0.18f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float transmission = 0.0f;
    float diffuseTransmission = 0.0f;
    float clearcoat = 0.0f;
    float clearcoatRoughness = 0.3f;
    int thinWalled = 0;
};

SurfaceInteraction makeSi(const Material& m, float viewTilt = 0.4f)
{
    MaterialParams p = {};
    p.material_type = m.type;
    p.base_color = m.baseColor;
    p.roughness = m.roughness;
    p.metallic = m.metallic;
    p.transmission = m.transmission;
    p.diffuse_transmission = m.diffuseTransmission;
    p.clearcoat = m.clearcoat;
    p.clearcoat_roughness = m.clearcoatRoughness;
    p.thin_walled = m.thinWalled;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.specular_color = make_float3(1.0f);
    p.normal_scale = 1.0f;
    p.occlusion_strength = 1.0f;
    p.alpha_cutoff = 0.5f;
    p.base_color_tex = -1;
    p.metallic_roughness_tex = -1;
    p.normal_tex = -1;
    p.emission_tex = -1;
    p.occlusion_tex = -1;
    p.transmission_tex = -1;

    SurfaceInteraction si = {};
    si.position = make_float3(0.0f);
    si.geometry_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.uv = make_float2(0.0f, 0.0f);
    si.wo = safe_normalize(make_float3(std::sin(viewTilt), std::cos(viewTilt), 0.0f));
    si.front_face = true;
    bsdf_init(si, p, nullptr);
    si.exterior_ior = 1.0f;
    return si;
}

/// The fraction of draws that come back flagged BSDF_EVENT_SPECULAR -- i.e. the
/// fraction of vertices at which the old, sample-gated rule skipped next-event
/// estimation entirely.
double specularDrawFraction(const SurfaceInteraction& si, int samples = 100000)
{
    FixedSeedSampler rng(0x85EBCA6Bu);
    int live = 0;
    int specular = 0;
    for (int i = 0; i < samples; ++i)
    {
        const BsdfSampleResult s = bsdf_sample(si, rng.next4());
        if (s.event_type == BSDF_EVENT_ABSORB)
        {
            continue;
        }
        ++live;
        if ((s.event_type & BSDF_EVENT_SPECULAR) != 0)
        {
            ++specular;
        }
    }
    return live > 0 ? double(specular) / double(live) : 0.0;
}

/// How much of bsdf_eval()'s density a light connection could actually reach.
///
/// The integral over the sphere, by uniform sampling. Deliberately not "is there
/// a direction with a non-zero pdf": a near-delta GGX lobe at alpha 1e-8 still
/// returns a positive-but-denormal density far from its peak, so that question
/// answers yes for a perfect mirror and means nothing. What next-event
/// estimation can deliver is mass, and a delta lobe's mass is unreachable by any
/// sampling the connection does.
double reachableDensityMass(const SurfaceInteraction& si, int samples = 400000)
{
    FixedSeedSampler rng(0x27D4EB2Fu);
    double sum = 0.0;
    for (int i = 0; i < samples; ++i)
    {
        const float cosTheta = 1.0f - 2.0f * rng.next();
        const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
        const float phi = 2.0f * float(M_PI_F) * rng.next();
        const float3 wi = make_float3(sinTheta * std::cos(phi), cosTheta, sinTheta * std::sin(phi));
        sum += double(bsdf_eval(si, wi).pdf);
    }
    return sum / double(samples) * 4.0 * double(M_PI_F);
}

} // namespace

TEST_CASE("a diffuse base under a mirror-smooth coat still has a lobe to connect to")
{
    Material m;
    m.clearcoat = 1.0f;
    m.clearcoatRoughness = 0.0f; // the glTF default
    m.roughness = 0.5f;
    const SurfaceInteraction si = makeSi(m);

    // The material's answer, which is what the integrators now ask.
    CHECK(bsdf_has_smooth_lobe(si));
    CHECK(neeRunsAtVertex(/*neeEnabled=*/true, /*hasEmitter=*/true, bsdf_has_smooth_lobe(si)));

    // And the draw's answer, which is what they used to ask. Over half the
    // vertices took the delta coat and skipped next-event estimation on a
    // surface whose diffuse base needs it.
    const double specular = specularDrawFraction(si);
    CHECK(specular > 0.4);
    CHECK(specular < 0.7);
}

TEST_CASE("a dark lacquered paint loses even more of its direct light to the old rule")
{
    Material m;
    m.baseColor = make_float3(0.05f);
    m.roughness = 0.4f;
    m.clearcoat = 1.0f;
    m.clearcoatRoughness = 0.0f;
    const SurfaceInteraction si = makeSi(m);

    CHECK(bsdf_has_smooth_lobe(si));
    // The darker the base, the smaller its lobe weight and the more often the
    // coat wins the selection -- so the sample-gated rule threw away more.
    CHECK(specularDrawFraction(si) > 0.6);
}

TEST_CASE("polished plastic keeps its diffuse lobe below the delta threshold")
{
    Material m;
    m.roughness = 0.02f; // alpha 4e-4, under BSDF_DELTA_ALPHA
    const SurfaceInteraction si = makeSi(m);

    CHECK(bsdf_has_smooth_lobe(si));
    CHECK(specularDrawFraction(si) > 0.1);
}

TEST_CASE("a perfect mirror has no lobe for a light connection")
{
    Material m;
    m.metallic = 1.0f;
    m.roughness = 0.0f;
    m.baseColor = make_float3(1.0f);
    const SurfaceInteraction si = makeSi(m);

    // Nothing to connect to, and asking for a connection would cost a shadow ray
    // that bsdf_eval() then values at zero. The bounce keeps the whole
    // contribution through the specular exemption at the light hit.
    CHECK_FALSE(bsdf_has_smooth_lobe(si));
    CHECK_FALSE(neeRunsAtVertex(true, true, bsdf_has_smooth_lobe(si)));
    CHECK(specularDrawFraction(si) == doctest::Approx(1.0));
}

TEST_CASE("smooth glass has no lobe either, rough glass does")
{
    Material smooth;
    smooth.transmission = 1.0f;
    smooth.roughness = 0.0f;
    CHECK_FALSE(bsdf_has_smooth_lobe(makeSi(smooth)));

    Material rough = smooth;
    rough.roughness = 0.3f;
    CHECK(bsdf_has_smooth_lobe(makeSi(rough)));

    // A thin wall raises the transmission lobe's alpha, so a wall that is smooth
    // on its reflection side can still be rough on its transmission side. The
    // predicate errs towards true for that reason -- an unnecessary connection
    // costs a shadow ray, a missing one costs light.
    Material thin = smooth;
    thin.thinWalled = 1;
    thin.roughness = 0.05f;
    CHECK(bsdf_has_smooth_lobe(makeSi(thin)));
}

TEST_CASE("a leaf lit from behind has its diffuse transmission lobe")
{
    Material m;
    m.diffuseTransmission = 0.8f;
    m.roughness = 0.0f; // the reflection side is a mirror; the leaf is not
    CHECK(bsdf_has_smooth_lobe(makeSi(m)));
}

TEST_CASE("the material types the loader does not emit answer too")
{
    Material diffuse;
    diffuse.type = MATERIAL_TYPE_DIFFUSE;
    diffuse.roughness = 0.0f;
    CHECK(bsdf_has_smooth_lobe(makeSi(diffuse))); // Lambert is never delta

    Material hair;
    hair.type = MATERIAL_TYPE_HAIR;
    hair.roughness = 0.0f;
    CHECK(bsdf_has_smooth_lobe(makeSi(hair))); // Chiang floors its roughness

    Material conductor;
    conductor.type = MATERIAL_TYPE_CONDUCTOR;
    conductor.roughness = 0.0f;
    CHECK_FALSE(bsdf_has_smooth_lobe(makeSi(conductor)));
    conductor.roughness = 0.3f;
    CHECK(bsdf_has_smooth_lobe(makeSi(conductor)));
}

TEST_CASE("bsdf_has_smooth_lobe agrees with what bsdf_eval will actually report")
{
    // The predicate is a claim about bsdf_eval(). Saying yes when there is
    // nothing to reach buys a wasted shadow ray; saying no when there is loses
    // that light for good, so that is the direction asserted here. The measure
    // is how much density mass a connection can reach, not whether some
    // direction returns a non-zero float -- see reachableDensityMass().
    const Material materials[] = {
        Material{}, // plain diffuse-ish
        Material{ MATERIAL_TYPE_STANDARD_PBR, make_float3(0.18f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0 },
        Material{ MATERIAL_TYPE_STANDARD_PBR, make_float3(0.18f), 0.3f, 1.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0 },
        Material{ MATERIAL_TYPE_STANDARD_PBR, make_float3(0.5f), 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.3f, 0 },
        Material{ MATERIAL_TYPE_STANDARD_PBR, make_float3(0.5f), 0.4f, 0.0f, 1.0f, 0.0f, 0.0f, 0.3f, 0 },
        Material{ MATERIAL_TYPE_STANDARD_PBR, make_float3(0.5f), 0.5f, 0.0f, 0.0f, 0.7f, 0.0f, 0.3f, 0 },
        Material{ MATERIAL_TYPE_STANDARD_PBR, make_float3(0.18f), 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0 },
        Material{ MATERIAL_TYPE_STANDARD_PBR, make_float3(0.18f), 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.4f, 0 },
    };

    for (const Material& m : materials)
    {
        CAPTURE(m.roughness);
        CAPTURE(m.metallic);
        CAPTURE(m.transmission);
        CAPTURE(m.clearcoat);
        CAPTURE(m.clearcoatRoughness);
        const SurfaceInteraction si = makeSi(m);
        const bool claimed = bsdf_has_smooth_lobe(si);
        const double mass = reachableDensityMass(si);
        CAPTURE(mass);
        if (mass > 0.05)
        {
            CHECK(claimed);
        }
        if (!claimed)
        {
            // Nothing meaningful was left on the table by declining to connect.
            CHECK(mass < 0.05);
        }
    }
}

TEST_CASE("a perfect mirror's density is all in the delta lobe a connection cannot reach")
{
    Material mirror;
    mirror.metallic = 1.0f;
    mirror.roughness = 0.0f;
    mirror.baseColor = make_float3(1.0f);
    const SurfaceInteraction si = makeSi(mirror);

    // A near-delta GGX lobe does return a positive density away from its peak --
    // denormally small, but positive -- so "some direction has a pdf" is not the
    // question. The mass is what a light connection could deliver, and there is
    // none of it.
    // Not exactly zero: at alpha 1e-8 the GGX tails still integrate to about
    // half a percent of the lobe. That is the residue a connection could reach,
    // and it is far below the noise any shadow ray spent on it would carry.
    CHECK_FALSE(bsdf_has_smooth_lobe(si));
    CHECK(reachableDensityMass(si) < 0.01);
}
