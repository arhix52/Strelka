// ============================================================================
// test_sample_eval_consistency.cpp
//
// bsdf_sample() and bsdf_eval() are two descriptions of the same BRDF, and
// multiple importance sampling assumes they agree exactly. MIS weighs each
// strategy by the *other* strategy's density: the BSDF-sampling estimator
// divides by a light-sampling pdf, and next-event estimation divides by
// bsdf_eval()'s pdf. If the two routines drift apart, each one still looks
// plausible on its own -- the sampled image converges, the NEE image converges
// -- but their weighted sum no longer sums to one and the combined estimator is
// quietly biased.
//
// The concrete way this happens in this codebase: standard_pbr.h builds f_spec
// in FOUR separate places (the diffuse, specular, transmission and clearcoat
// branches of standard_pbr_sample(), plus standard_pbr_eval()). The
// multiple-scattering energy compensation had to be pasted into every one of
// them. Applying it to three out of four is a one-line omission that no image
// review would catch, because a rough metal simply stays a bit dark in a way
// that looks like "GGX loses energy" -- which is exactly what the compensation
// was added to fix. These tests fail loudly instead.
//
// The invariants pinned here, at directions actually produced by the sampler
// (a pointwise check, far sharper than any chi-square histogram test):
//
//   1. bsdf_eval(wi).pdf == sampleResult.pdf
//   2. bsdf_over_pdf * pdf == bsdf * |NdotL|   (the two cosine conventions,
//      see the comment on the result structs in bsdf_types.h)
//   3. bsdf_pdf(wi) == bsdf_eval(wi).pdf
//   4. bsdf_eval(wi).pdf > 0 for every direction the sampler can produce --
//      a direction with zero claimed density gets an infinite MIS weight
//   5. nothing anywhere on the grid is NaN, infinite or negative
//
// BSDF_EVENT_SPECULAR samples are skipped throughout: a delta lobe has no
// density with respect to solid angle, bsdf_sample() reports a discrete
// probability in the pdf field instead, and bsdf_eval() correctly returns zero.
// Comparing the two is meaningless rather than a failure.
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

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace
{

// ---------------------------------------------------------------------------
// A fixed-seed LCG. Deterministic on purpose: a flaky BSDF test is a test
// people learn to re-run rather than read.
// ---------------------------------------------------------------------------
struct FixedSeedSampler
{
    std::uint32_t state;

    explicit FixedSeedSampler(std::uint32_t seed) : state(seed | 1u) {}

    float next()
    {
        state = state * 1664525u + 1013904223u;
        // Use the high bits; the low bits of an LCG are famously poor.
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

MaterialParams default_params()
{
    MaterialParams p = {};
    p.base_color          = make_float3(0.82f, 0.61f, 0.43f); // non-grey: catches per-channel slips
    p.metallic            = 0.0f;
    p.roughness           = 0.5f;
    p.ior                 = 1.5f;
    p.specular            = 0.5f;
    p.specular_color      = make_float3(1.0f);
    p.transmission        = 0.0f;
    p.clearcoat           = 0.0f;
    p.clearcoat_roughness = 0.3f;
    p.anisotropy          = 0.0f;
    p.emission            = make_float3(0.0f);
    p.emission_strength   = 0.0f;
    p.normal_scale        = 1.0f;
    p.occlusion_strength  = 1.0f;
    p.alpha_cutoff        = 0.5f;
    p.material_type       = MATERIAL_TYPE_STANDARD_PBR;
    p.base_color_tex          = -1;
    p.metallic_roughness_tex  = -1;
    p.normal_tex              = -1;
    p.emission_tex            = -1;
    p.occlusion_tex           = -1;
    p.transmission_tex        = -1;
    p.dielectric_priority = 0;
    p.thin_walled         = 0;
    return p;
}

// One point on the grid. Surface frame is fixed (N = +Y); the view direction is
// tilted away from the normal by `viewTilt` radians in the XY plane, so
// viewTilt = 0 is normal incidence and 1.3 rad is ~75 degrees off normal.
struct GridPoint
{
    unsigned int materialType = MATERIAL_TYPE_STANDARD_PBR;
    float roughness           = 0.5f;
    float metallic            = 0.0f;
    float transmission        = 0.0f;
    float clearcoat           = 0.0f;
    float clearcoatRoughness  = 0.3f;
    float viewTilt            = 0.4f;
};

SurfaceInteraction make_si(const GridPoint& g)
{
    MaterialParams p = default_params();
    p.material_type       = g.materialType;
    p.roughness           = g.roughness;
    p.metallic            = g.metallic;
    p.transmission        = g.transmission;
    p.clearcoat           = g.clearcoat;
    p.clearcoat_roughness = g.clearcoatRoughness;

    SurfaceInteraction si = {};
    si.position        = make_float3(0.0f, 0.0f, 0.0f);
    si.geometry_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.shading_normal  = make_float3(0.0f, 1.0f, 0.0f);
    si.tangent         = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent       = make_float3(0.0f, 0.0f, 1.0f);
    si.uv              = make_float2(0.0f, 0.0f);
    si.wo              = glm::normalize(make_float3(std::sin(g.viewTilt), std::cos(g.viewTilt), 0.0f));
    si.front_face      = true;

    bsdf_init(si, p, nullptr);
    si.exterior_ior = 1.0f; // ray travelling through air
    return si;
}

// Worst-case relative disagreements accumulated over one grid point.
struct Disagreement
{
    double worstPdf     = 0.0; // |eval.pdf - sample.pdf| / sample.pdf
    double worstValue   = 0.0; // ||bsdf_over_pdf*pdf - bsdf*|NdotL||| / scale
    double worstPdfFunc = 0.0; // |bsdf_pdf() - eval.pdf| / eval.pdf
    int compared        = 0;   // non-specular samples actually checked
    int zeroEvalPdf     = 0;   // sampler produced a direction eval calls impossible
    int specularSkipped = 0;
    int absorbed        = 0;
    int transmissionSkipped = 0;
    int nonFinite       = 0;   // NaN or infinity in a pdf or a colour
    int negative        = 0;   // negative component in a pdf or a colour
};

double max_component(float3 v)
{
    return std::max({ static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z) });
}

bool is_finite(float3 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool is_negative(float3 v)
{
    return v.x < 0.0f || v.y < 0.0f || v.z < 0.0f;
}

// Draw `samples` directions from bsdf_sample() and cross-check each one against
// bsdf_eval() / bsdf_pdf(). `skipTransmission` drops refraction events (see the
// dielectric test case for why they are excluded rather than asserted on).
Disagreement cross_check(const GridPoint& g, int samples, std::uint32_t seed, bool skipTransmission)
{
    const SurfaceInteraction si = make_si(g);
    FixedSeedSampler rng(seed);
    Disagreement d;

    for (int i = 0; i < samples; ++i)
    {
        const BsdfSampleResult s = bsdf_sample(si, rng.next4());

        if (s.event_type == BSDF_EVENT_ABSORB)
        {
            ++d.absorbed;
            continue;
        }
        // A delta lobe carries a discrete probability in `pdf`, not a density;
        // bsdf_eval() rightly reports zero for it. Nothing to compare.
        if ((s.event_type & BSDF_EVENT_SPECULAR) != 0)
        {
            ++d.specularSkipped;
            continue;
        }
        if (skipTransmission && (s.event_type & BSDF_EVENT_TRANSMISSION) != 0)
        {
            ++d.transmissionSkipped;
            continue;
        }

        const BsdfEvalResult e = bsdf_eval(si, s.wi);

        // Invariant 4: a direction the sampler can return must have a density.
        if (!(e.pdf > 0.0f))
        {
            ++d.zeroEvalPdf;
            continue;
        }
        ++d.compared;

        // Invariant 1.
        d.worstPdf = std::max(d.worstPdf, std::fabs(e.pdf - s.pdf) / static_cast<double>(s.pdf));

        // Invariant 2. bsdf_sample() folds |cos| into bsdf_over_pdf and
        // bsdf_eval() does not, so put both sides in the same convention before
        // comparing -- see bsdf_types.h. Comparing the raw f's instead looks
        // like a failure of exactly the cosine, which is a false alarm.
        const float  cosWi      = std::fabs(dot(si.shading_normal, s.wi));
        const float3 fromSample = s.bsdf_over_pdf * s.pdf;
        const float3 fromEval   = e.bsdf * cosWi;
        const double scale      = std::max({ max_component(fromSample), max_component(fromEval), 1e-6 });
        d.worstValue = std::max(d.worstValue, static_cast<double>(length(fromSample - fromEval)) / scale);

        // Invariant 3.
        const float pdfOnly = bsdf_pdf(si, s.wi);
        d.worstPdfFunc =
            std::max(d.worstPdfFunc, std::fabs(pdfOnly - e.pdf) / static_cast<double>(e.pdf));

        // Invariant 5. Counted rather than REQUIREd per sample: one assertion
        // per grid point keeps the doctest report readable, and the counter
        // still pins the failure to a specific parameter combination.
        if (!std::isfinite(s.pdf) || !std::isfinite(e.pdf) || !is_finite(s.bsdf_over_pdf) ||
            !is_finite(e.bsdf))
        {
            ++d.nonFinite;
        }
        if (is_negative(s.bsdf_over_pdf) || is_negative(e.bsdf))
        {
            ++d.negative;
        }
    }

    return d;
}

// Tolerances. The disagreements that survive are pure float noise: the half
// vector is rebuilt from wi in eval() rather than reused from the sampler, and
// GGX amplifies that error as alpha shrinks. Measured worst case over the grids
// below is 1.3e-3 for the pdf and 2.2e-4 for the value, so 1% leaves the better
// part of an order of magnitude of headroom while still catching a dropped
// energy-compensation factor (>= 10% at roughness 0.6).
//
// Roughness stays at or above 0.15 for the same reason: at roughness 0.05 the
// lobe is so peaked that rebuilding H costs 6% in the pdf, which says nothing
// about whether the two routines implement the same BRDF.
constexpr double kPdfTolerance   = 0.01;
constexpr double kValueTolerance = 0.01;

const float kRoughnessGrid[] = { 0.15f, 0.3f, 0.6f, 1.0f };
const float kViewTiltGrid[]  = { 0.0f, 0.4f, 0.8f, 1.3f };

void check_grid_point(const GridPoint& g, const Disagreement& d)
{
    CAPTURE(g.materialType);
    CAPTURE(g.roughness);
    CAPTURE(g.metallic);
    CAPTURE(g.transmission);
    CAPTURE(g.clearcoat);
    CAPTURE(g.viewTilt);
    CAPTURE(d.compared);
    CAPTURE(d.zeroEvalPdf);

    // Guard against a vacuous pass: if the sampler stopped producing usable
    // directions the CHECKs below would all trivially hold.
    CHECK(d.compared > 0);
    CHECK(d.zeroEvalPdf == 0);
    CHECK(d.nonFinite == 0);
    CHECK(d.negative == 0);
    CHECK(d.worstPdf < kPdfTolerance);
    CHECK(d.worstValue < kValueTolerance);
    CHECK(d.worstPdfFunc < 1e-5);
}

} // namespace

// ---------------------------------------------------------------------------
// The three opaque material types, over roughness x metallic x view angle.
//
// MATERIAL_TYPE_STANDARD_PBR is the one that matters in practice: the glTF
// loader tags every opaque material with it, so this is the path every test
// scene in tools/feature_tests/ actually runs through.
// ---------------------------------------------------------------------------
TEST_CASE("bsdf_sample and bsdf_eval describe the same BRDF (opaque materials)")
{
    const unsigned int types[] = { MATERIAL_TYPE_DIFFUSE, MATERIAL_TYPE_CONDUCTOR,
                                   MATERIAL_TYPE_STANDARD_PBR };

    std::uint32_t seed = 0x9E3779B9u;
    for (unsigned int type : types)
    {
        for (float roughness : kRoughnessGrid)
        {
            for (float metallic : { 0.0f, 0.5f, 1.0f })
            {
                for (float tilt : kViewTiltGrid)
                {
                    GridPoint g;
                    g.materialType = type;
                    g.roughness    = roughness;
                    g.metallic     = metallic;
                    g.viewTilt     = tilt;

                    seed += 0x9E3779B9u;
                    const Disagreement d = cross_check(g, 2000, seed, /*skipTransmission=*/false);
                    check_grid_point(g, d);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The clearcoat layer adds a fourth f_spec site and a fourth term to the
// combined pdf, in a branch that only runs when clearcoat > 0 -- so it is
// exactly the kind of code that can rot unnoticed. Clearcoat roughness is kept
// well away from the delta threshold (alpha < 0.001, i.e. clearcoat_roughness
// below ~0.032) so the layer produces a real density to compare.
// ---------------------------------------------------------------------------
TEST_CASE("bsdf_sample and bsdf_eval describe the same BRDF (clearcoat layer)")
{
    std::uint32_t seed = 0x85EBCA6Bu;
    for (float roughness : kRoughnessGrid)
    {
        for (float metallic : { 0.0f, 1.0f })
        {
            for (float tilt : kViewTiltGrid)
            {
                GridPoint g;
                g.materialType      = MATERIAL_TYPE_STANDARD_PBR;
                g.roughness         = roughness;
                g.metallic          = metallic;
                g.clearcoat         = 1.0f;
                g.clearcoatRoughness = 0.3f;
                g.viewTilt          = tilt;

                seed += 0x9E3779B9u;
                const Disagreement d = cross_check(g, 2000, seed, /*skipTransmission=*/false);
                check_grid_point(g, d);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The rough dielectric, both lobes.
//
// Refraction used to be excluded here, because sample and eval disagreed on it
// by orders of magnitude rather than by float noise. Two separate mistakes, both
// of them in the change of variables from the half vector to the outgoing
// direction, and both now fixed in microfacet.h:
//
//   - the half vector was rebuilt as normalize(V + eta * wi) when Walter et al.
//     2007 build it from eta_i * V + eta_t * wi, which in this file's eta
//     convention is V + wi / eta. That is a half vector about 0.2 rad away from
//     the one the sampler actually bent around;
//   - the density used ggx_vndf_pdf(), which is already divided by the 4 * VdotH
//     that turns a half-vector density into a *reflected direction* density.
//     A refraction needs the half-vector density itself, so the reported pdf was
//     short by roughly a factor of four on top of everything else.
//
// Together they put the integral of the reported pdf over the sphere at 0.24
// where the sampler produces a non-delta event 0.96 of the time. The
// normalisation case at the bottom of this file is what pins that down; the
// grid here pins the pointwise agreement.
// ---------------------------------------------------------------------------
TEST_CASE("bsdf_sample and bsdf_eval describe the same BRDF (dielectric, both lobes)")
{
    std::uint32_t seed = 0xC2B2AE35u;
    for (float roughness : kRoughnessGrid)
    {
        for (float tilt : kViewTiltGrid)
        {
            GridPoint g;
            g.materialType = MATERIAL_TYPE_DIELECTRIC;
            g.roughness    = roughness;
            g.transmission = 1.0f;
            g.viewTilt     = tilt;

            seed += 0x9E3779B9u;
            const Disagreement d = cross_check(g, 4000, seed, /*skipTransmission=*/false);
            check_grid_point(g, d);
        }
    }
}

// ---------------------------------------------------------------------------
// A transmissive STANDARD_PBR material.
//
// This is the gap that mattered in practice, and it was a gap in the grid rather
// than in the invariants: the glTF loader tags every material
// MATERIAL_TYPE_STANDARD_PBR (gltfloader.cpp), including anything carrying
// KHR_materials_transmission, so real glass never goes through dielectric.h at
// all -- and the grids above hold transmission at zero.
//
// What was hiding there: standard_pbr_sample() produces reflections from inside
// the transmission lobe through its Fresnel coin flip, and standard_pbr_eval()
// left that term out of both f and the pdf. On transmission = 1, every single
// non-delta reflection the sampler produced was reported by eval as pdf 0 --
// 8194 out of 8194 at roughness 0.15. A direction the sampler produces and eval
// calls impossible is not a small inconsistency: next-event estimation cannot
// see a rough glass reflection at all (it needs eval's pdf to be positive),
// while the light hit still deducts a MIS share for it. The share is deducted
// and never delivered.
//
// Partial transmission was worse in a quieter way, because eval returned a
// plausible non-zero number that simply was not sample's: up to 17000x apart on
// the directions the two do share.
// ---------------------------------------------------------------------------
TEST_CASE("bsdf_sample and bsdf_eval describe the same BRDF (transmissive standard_pbr)")
{
    std::uint32_t seed = 0x165667B1u;
    for (float transmission : { 0.25f, 0.5f, 0.75f, 1.0f })
    {
        for (float roughness : kRoughnessGrid)
        {
            for (float tilt : kViewTiltGrid)
            {
                GridPoint g;
                g.materialType = MATERIAL_TYPE_STANDARD_PBR;
                g.roughness = roughness;
                g.transmission = transmission;
                g.viewTilt = tilt;

                seed += 0x9E3779B9u;
                const Disagreement d = cross_check(g, 4000, seed, /*skipTransmission=*/false);
                check_grid_point(g, d);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Normalisation.
//
// The grid above is pointwise: it compares the two routines to each other at
// directions the sampler produced. Two routines can agree perfectly and both be
// wrong, and that is exactly what the refraction lobe did -- sample and eval
// applied the same mistaken change of variables, so a pointwise check with the
// refraction events skipped saw nothing.
//
// A density has an absolute property no amount of agreement can supply: it has
// to integrate to one over the sphere, or to the probability of the events it
// describes when some of them are delta. That is what this measures, and it is
// the case that would have caught both refraction mistakes on its own.
// ---------------------------------------------------------------------------
TEST_CASE("bsdf_eval's density integrates to the probability of a non-delta event")
{
    struct Point
    {
        unsigned int type;
        float roughness;
        float transmission;
        float tilt;
    };
    const Point points[] = {
        { MATERIAL_TYPE_STANDARD_PBR, 0.3f, 0.0f, 0.0f },  { MATERIAL_TYPE_STANDARD_PBR, 0.3f, 0.0f, 0.9f },
        { MATERIAL_TYPE_STANDARD_PBR, 0.2f, 1.0f, 0.0f },  { MATERIAL_TYPE_STANDARD_PBR, 0.2f, 1.0f, 0.9f },
        { MATERIAL_TYPE_STANDARD_PBR, 0.4f, 0.5f, 0.4f },  { MATERIAL_TYPE_DIELECTRIC, 0.2f, 1.0f, 0.4f },
        { MATERIAL_TYPE_DIELECTRIC, 0.4f, 1.0f, 0.9f },
    };

    for (const Point& p : points)
    {
        CAPTURE(p.type);
        CAPTURE(p.roughness);
        CAPTURE(p.transmission);
        CAPTURE(p.tilt);

        GridPoint g;
        g.materialType = p.type;
        g.roughness = p.roughness;
        g.transmission = p.transmission;
        g.viewTilt = p.tilt;
        const SurfaceInteraction si = make_si(g);

        // How often the sampler produces an event that has a density at all.
        FixedSeedSampler drawRng(0x2545F491u);
        int nonDelta = 0;
        const int drawCount = 200000;
        for (int i = 0; i < drawCount; ++i)
        {
            const BsdfSampleResult s = bsdf_sample(si, drawRng.next4());
            if (s.event_type != BSDF_EVENT_ABSORB && (s.event_type & BSDF_EVENT_SPECULAR) == 0)
            {
                ++nonDelta;
            }
        }
        const double eventProbability = double(nonDelta) / double(drawCount);

        // The integral of the reported density over the whole sphere, by uniform
        // sampling. Slow to converge because the lobes are peaked, hence the
        // loose tolerance -- but "0.24 where it should be 0.96" is not a
        // tolerance question.
        FixedSeedSampler intRng(0x27220A95u);
        double integral = 0.0;
        const int integralSamples = 4000000;
        for (int i = 0; i < integralSamples; ++i)
        {
            const float cosTheta = 1.0f - 2.0f * intRng.next();
            const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
            const float phi = 2.0f * float(M_PI_F) * intRng.next();
            const float3 wi =
                make_float3(sinTheta * std::cos(phi), cosTheta, sinTheta * std::sin(phi));
            const float pdf = bsdf_eval(si, wi).pdf;
            REQUIRE(std::isfinite(pdf));
            REQUIRE(pdf >= 0.0f);
            integral += double(pdf);
        }
        integral = integral / double(integralSamples) * 4.0 * double(M_PI_F);

        CHECK(integral == doctest::Approx(eventProbability).epsilon(0.08));
    }
}

// ---------------------------------------------------------------------------
// Nothing on the grid -- including the parameter combinations excluded from the
// equality checks above, and including degenerate roughness and near-grazing
// view angles -- may produce a NaN, an infinity or a negative value. A single
// NaN in a BSDF poisons its pixel for the whole render, and a negative bsdf
// silently subtracts light.
// ---------------------------------------------------------------------------
TEST_CASE("no NaNs, infinities or negative values anywhere on the material grid")
{
    std::uint32_t seed = 0x27D4EB2Fu;
    int evaluated = 0;

    for (unsigned int type : { MATERIAL_TYPE_DIFFUSE, MATERIAL_TYPE_CONDUCTOR,
                               MATERIAL_TYPE_DIELECTRIC, MATERIAL_TYPE_STANDARD_PBR })
    {
        for (float roughness : { 0.001f, 0.05f, 0.15f, 0.5f, 1.0f })
        {
            for (float metallic : { 0.0f, 1.0f })
            {
                for (float transmission : { 0.0f, 1.0f })
                {
                    for (float clearcoat : { 0.0f, 1.0f })
                    {
                        // Includes 89.4 degrees off normal, where the shading
                        // frame is nearly edge-on and the divisions by NdotV
                        // are at their most fragile.
                        for (float tilt : { 0.0f, 0.6f, 1.2f, 1.56f })
                        {
                            GridPoint g;
                            g.materialType = type;
                            g.roughness    = roughness;
                            g.metallic     = metallic;
                            g.transmission = transmission;
                            g.clearcoat    = clearcoat;
                            g.viewTilt     = tilt;

                            const SurfaceInteraction si = make_si(g);
                            seed += 0x9E3779B9u;
                            FixedSeedSampler rng(seed);

                            int nonFinite = 0;
                            int negative  = 0;

                            for (int i = 0; i < 400; ++i)
                            {
                                const BsdfSampleResult s = bsdf_sample(si, rng.next4());
                                const float3 wi =
                                    (s.event_type == BSDF_EVENT_ABSORB) ? si.wo : s.wi;
                                const BsdfEvalResult e = bsdf_eval(si, wi);
                                const float pdfOnly    = bsdf_pdf(si, wi);

                                if (!std::isfinite(s.pdf) || !is_finite(s.bsdf_over_pdf) ||
                                    !is_finite(wi) || !std::isfinite(e.pdf) ||
                                    !is_finite(e.bsdf) || !std::isfinite(pdfOnly))
                                {
                                    ++nonFinite;
                                }
                                if (s.pdf < 0.0f || is_negative(s.bsdf_over_pdf) ||
                                    e.pdf < 0.0f || is_negative(e.bsdf) || pdfOnly < 0.0f)
                                {
                                    ++negative;
                                }
                                ++evaluated;
                            }

                            CAPTURE(type);
                            CAPTURE(roughness);
                            CAPTURE(metallic);
                            CAPTURE(transmission);
                            CAPTURE(clearcoat);
                            CAPTURE(tilt);
                            CHECK(nonFinite == 0);
                            CHECK(negative == 0);
                        }
                    }
                }
            }
        }
    }

    CHECK(evaluated > 0);
}
