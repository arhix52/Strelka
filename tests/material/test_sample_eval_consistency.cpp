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
#include <limits>

namespace
{

// ---------------------------------------------------------------------------
// A fixed-seed LCG. Deterministic on purpose: a flaky BSDF test is a test
// people learn to re-run rather than read.
// ---------------------------------------------------------------------------
struct FixedSeedSampler
{
    std::uint32_t state;

    explicit FixedSeedSampler(std::uint32_t seed) : state(seed | 1u)
    {
    }

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
    p.base_color = make_float3(0.82f, 0.61f, 0.43f); // non-grey: catches per-channel slips
    p.metallic = 0.0f;
    p.roughness = 0.5f;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.specular_color = make_float3(1.0f);
    p.transmission = 0.0f;
    p.clearcoat = 0.0f;
    p.clearcoat_roughness = 0.3f;
    p.anisotropy = 0.0f;
    p.emission = make_float3(0.0f);
    p.emission_strength = 0.0f;
    p.normal_scale = 1.0f;
    p.occlusion_strength = 1.0f;
    p.alpha_cutoff = 0.5f;
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;
    p.base_color_tex = -1;
    p.metallic_roughness_tex = -1;
    p.normal_tex = -1;
    p.emission_tex = -1;
    p.occlusion_tex = -1;
    p.transmission_tex = -1;
    p.dielectric_priority = 0;
    p.thin_walled = 0;
    return p;
}

// One point on the grid. Surface frame is fixed (N = +Y); the view direction is
// tilted away from the normal by `viewTilt` radians in the XY plane, so
// viewTilt = 0 is normal incidence and 1.3 rad is ~75 degrees off normal.
struct GridPoint
{
    unsigned int materialType = MATERIAL_TYPE_STANDARD_PBR;
    float roughness = 0.5f;
    float metallic = 0.0f;
    float transmission = 0.0f;
    float clearcoat = 0.0f;
    float clearcoatRoughness = 0.3f;
    float viewTilt = 0.4f;
};

SurfaceInteraction make_si(const GridPoint& g)
{
    MaterialParams p = default_params();
    p.material_type = g.materialType;
    p.roughness = g.roughness;
    p.metallic = g.metallic;
    p.transmission = g.transmission;
    p.clearcoat = g.clearcoat;
    p.clearcoat_roughness = g.clearcoatRoughness;

    SurfaceInteraction si = {};
    si.position = make_float3(0.0f, 0.0f, 0.0f);
    si.geometry_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.uv = make_float2(0.0f, 0.0f);
    si.wo = glm::normalize(make_float3(std::sin(g.viewTilt), std::cos(g.viewTilt), 0.0f));
    si.front_face = true;

    bsdf_init(si, p, nullptr);
    si.exterior_ior = 1.0f; // ray travelling through air
    return si;
}

// Worst-case relative disagreements accumulated over one grid point.
struct Disagreement
{
    double worstPdf = 0.0; // |eval.pdf - sample.pdf| / sample.pdf
    double worstValue = 0.0; // ||bsdf_over_pdf*pdf - bsdf*|NdotL||| / scale
    double worstPdfFunc = 0.0; // |bsdf_pdf() - eval.pdf| / eval.pdf
    int compared = 0; // non-specular samples actually checked
    int zeroEvalPdf = 0; // sampler produced a direction eval calls impossible
    int specularSkipped = 0;
    int absorbed = 0;
    int transmissionSkipped = 0;
    int nonFinite = 0; // NaN or infinity in a pdf or a colour
    int negative = 0; // negative component in a pdf or a colour
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

struct Double3
{
    double x;
    double y;
    double z;
};

Double3 to_double3(float3 v)
{
    return { static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z) };
}

Double3 operator+(Double3 a, Double3 b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

Double3 operator/(Double3 v, double s)
{
    return { v.x / s, v.y / s, v.z / s };
}

Double3 operator-(Double3 v)
{
    return { -v.x, -v.y, -v.z };
}

double dot_double(Double3 a, Double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Double3 normalize_double(Double3 v)
{
    const double lengthSquared = dot_double(v, v);
    if (!(lengthSquared > 0.0))
    {
        return { 0.0, 0.0, 1.0 };
    }
    return v / std::sqrt(lengthSquared);
}

double luminance_double(float3 c)
{
    return 0.2126 * static_cast<double>(c.x) + 0.7152 * static_cast<double>(c.y) + 0.0722 * static_cast<double>(c.z);
}

double fresnel_dielectric_double(double cosThetaI, double eta)
{
    if (cosThetaI < 0.0)
    {
        eta = 1.0 / eta;
        cosThetaI = -cosThetaI;
    }
    const double sinThetaTSquared = eta * eta * (1.0 - cosThetaI * cosThetaI);
    if (sinThetaTSquared > 1.0)
    {
        return 1.0;
    }
    const double cosThetaT = std::sqrt(std::max(0.0, 1.0 - sinThetaTSquared));
    const double rs = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    const double rp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return 0.5 * (rs * rs + rp * rp);
}

double schlick_double(double f0, double cosine)
{
    const double t = 1.0 - std::clamp(cosine, 0.0, 1.0);
    return f0 + (1.0 - f0) * t * t * t * t * t;
}

double ggx_energy_term_double(double roughness, double nDotV)
{
    const double g1 = 1.0 - nDotV;
    const double g4 = g1 * g1 * g1 * g1;
    const double p0 = 0.154250 + roughness * (-1.181688 + roughness * (2.959942 + roughness * 0.325514));
    const double p1 = -2.939542 + roughness * (17.241920 + roughness * (-23.443382 + roughness * 7.088613));
    const double p2 = 9.297826 + roughness * (-38.077586 + roughness * (44.542460 - roughness * 15.902794));
    return std::max(0.0, roughness * roughness * (p0 + p1 * g1 + p2 * g4));
}

// Independent double-precision oracle for the lower-hemisphere density of the
// solid, isotropic mixed-transmission case below. It deliberately does not call
// any production PDF helper: this is the sum of a cosine BTDF proposal and a
// Walter/Heitz GGX refraction proposal, including both lobe-selection PMFs.
double mixed_transmission_pdf_oracle(const SurfaceInteraction& si, float3 wiFloat)
{
    const Double3 n = to_double3(si.shading_normal);
    const Double3 v = to_double3(si.wo);
    const Double3 wi = to_double3(wiFloat);
    const double nDotV = dot_double(n, v);
    const double nDotL = dot_double(n, wi);
    if (!(nDotV > 0.0) || !(nDotL < 0.0) || si.thin_walled)
    {
        return 0.0;
    }

    const double metallic = static_cast<double>(si.metallic);
    const double transmission = static_cast<double>(si.transmission);
    const double diffuseTransmission = std::clamp(static_cast<double>(si.diffuse_transmission), 0.0, 1.0);
    const double dielectricWeight = 1.0 - metallic;
    const double diffuseBase = dielectricWeight * (1.0 - transmission);
    const double diffuseWeight = diffuseBase * (1.0 - diffuseTransmission) * luminance_double(si.albedo);
    const double diffuseTransmissionWeight =
        diffuseBase * diffuseTransmission * luminance_double(si.diffuse_transmission_color);
    const double f0 = std::pow((static_cast<double>(si.ior) - 1.0) / (static_cast<double>(si.ior) + 1.0), 2.0);
    const double specularLuminance =
        (1.0 - metallic) * f0 * luminance_double(si.specular_color) + metallic * luminance_double(si.albedo);
    const double specularWeight = std::max(specularLuminance, 0.04) * (1.0 - transmission * dielectricWeight);
    const double transmissionWeight = dielectricWeight * transmission;
    const double clearcoatWeight = static_cast<double>(si.clearcoat) *
                                   std::max(std::pow((std::max(static_cast<double>(si.clearcoat_ior), 1.0) - 1.0) /
                                                         (std::max(static_cast<double>(si.clearcoat_ior), 1.0) + 1.0),
                                                     2.0),
                                            0.04) *
                                   6.0;
    const double total =
        diffuseWeight + diffuseTransmissionWeight + specularWeight + transmissionWeight + clearcoatWeight;
    if (!(total > 0.0))
    {
        return 0.0;
    }

    const double diffusePdf = (diffuseTransmissionWeight / total) * (-nDotL) / M_PI;

    const double roughness = std::max(static_cast<double>(si.roughness), 1.0e-4);
    const double alpha = roughness * roughness;
    if (alpha < static_cast<double>(BSDF_DELTA_ALPHA) || !(transmissionWeight > 0.0))
    {
        return diffusePdf;
    }

    const double eta = static_cast<double>(si.exterior_ior) / static_cast<double>(si.ior);
    Double3 h = normalize_double(v + wi / eta);
    if (dot_double(n, h) < 0.0)
    {
        h = -h;
    }
    const double nDotH = dot_double(n, h);
    const double vDotH = dot_double(v, h);
    const double lDotH = dot_double(wi, h);
    if (!(nDotH > 0.0) || !(vDotH > 0.0))
    {
        return diffusePdf;
    }

    const double alphaSquared = alpha * alpha;
    const double ndfDenominator = nDotH * nDotH * (alphaSquared - 1.0) + 1.0;
    const double d = alphaSquared / (M_PI * ndfDenominator * ndfDenominator);
    const double g1 = 2.0 * nDotV / (nDotV + std::sqrt(alphaSquared + (1.0 - alphaSquared) * nDotV * nDotV));
    const double halfVectorPdf = d * g1 * vDotH / nDotV;
    const double jacobianDenominator = eta * vDotH + lDotH;
    const double dHalfDWi = std::abs(lDotH) / (jacobianDenominator * jacobianDenominator);
    const double fresnel = fresnel_dielectric_double(vDotH, eta);
    const double refractionPdf = (transmissionWeight / total) * (1.0 - fresnel) * halfVectorPdf * dHalfDWi;
    return diffusePdf + refractionPdf;
}

SurfaceInteraction mixed_transmission_si(float roughness = 0.35f,
                                         float transmission = 0.5f,
                                         float diffuseTransmission = 0.5f,
                                         float viewTilt = 0.2914568f,
                                         bool exiting = false,
                                         bool thinWalled = false)
{
    SurfaceInteraction si = {};
    si.geometry_normal = make_float3(0.0f, 0.0f, 1.0f);
    si.shading_normal = si.geometry_normal;
    si.bump_normal = si.geometry_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 1.0f, 0.0f);
    const float viewZ = std::cos(viewTilt) * (exiting ? -1.0f : 1.0f);
    si.wo = normalize(make_float3(std::sin(viewTilt), 0.0f, viewZ));
    si.albedo = make_float3(0.7f, 0.8f, 0.6f);
    si.diffuse_transmission_color = make_float3(0.4f, 0.8f, 0.3f);
    si.roughness = roughness;
    si.ior = 1.5f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.transmission = transmission;
    si.diffuse_transmission = diffuseTransmission;
    si.clearcoat_ior = 1.5f;
    si.material_type = MATERIAL_TYPE_STANDARD_PBR;
    si.front_face = !exiting;
    si.thin_walled = thinWalled;
    return si;
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
        const float cosWi = std::fabs(dot(si.shading_normal, s.wi));
        const float3 fromSample = s.bsdf_over_pdf * s.pdf;
        const float3 fromEval = e.bsdf * cosWi;
        const double scale = std::max({ max_component(fromSample), max_component(fromEval), 1e-6 });
        d.worstValue = std::max(d.worstValue, static_cast<double>(length(fromSample - fromEval)) / scale);

        // Invariant 3.
        const float pdfOnly = bsdf_pdf(si, s.wi);
        d.worstPdfFunc = std::max(d.worstPdfFunc, std::fabs(pdfOnly - e.pdf) / static_cast<double>(e.pdf));

        // Invariant 5. Counted rather than REQUIREd per sample: one assertion
        // per grid point keeps the doctest report readable, and the counter
        // still pins the failure to a specific parameter combination.
        if (!std::isfinite(s.pdf) || !std::isfinite(e.pdf) || !is_finite(s.bsdf_over_pdf) || !is_finite(e.bsdf))
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
constexpr double kPdfTolerance = 0.01;
constexpr double kValueTolerance = 0.01;

const float kRoughnessGrid[] = { 0.15f, 0.3f, 0.6f, 1.0f };
const float kViewTiltGrid[] = { 0.0f, 0.4f, 0.8f, 1.3f };

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
    const unsigned int types[] = { MATERIAL_TYPE_DIFFUSE, MATERIAL_TYPE_CONDUCTOR, MATERIAL_TYPE_STANDARD_PBR };

    std::uint32_t seed = 0x9E3779B9u;
    for (const unsigned int type : types)
    {
        for (const float roughness : kRoughnessGrid)
        {
            for (const float metallic : { 0.0f, 0.5f, 1.0f })
            {
                for (const float tilt : kViewTiltGrid)
                {
                    GridPoint g;
                    g.materialType = type;
                    g.roughness = roughness;
                    g.metallic = metallic;
                    g.viewTilt = tilt;

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
    for (const float roughness : kRoughnessGrid)
    {
        for (const float metallic : { 0.0f, 1.0f })
        {
            for (const float tilt : kViewTiltGrid)
            {
                GridPoint g;
                g.materialType = MATERIAL_TYPE_STANDARD_PBR;
                g.roughness = roughness;
                g.metallic = metallic;
                g.clearcoat = 1.0f;
                g.clearcoatRoughness = 0.3f;
                g.viewTilt = tilt;

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
    for (const float roughness : kRoughnessGrid)
    {
        for (const float tilt : kViewTiltGrid)
        {
            GridPoint g;
            g.materialType = MATERIAL_TYPE_DIELECTRIC;
            g.roughness = roughness;
            g.transmission = 1.0f;
            g.viewTilt = tilt;

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
    for (const float transmission : { 0.25f, 0.5f, 0.75f, 1.0f })
    {
        for (const float roughness : kRoughnessGrid)
        {
            for (const float tilt : kViewTiltGrid)
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

TEST_CASE("mixed diffuse and specular transmission returns the full marginal PDF")
{
    const SurfaceInteraction si = mixed_transmission_si();
    FixedSeedSampler rng(0xd1ffu);
    int compared = 0;
    int evalMismatches = 0;
    int oracleMismatches = 0;

    for (int i = 0; i < 50000; ++i)
    {
        const BsdfSampleResult sample = bsdf_sample(si, rng.next4());
        if (sample.event_type == BSDF_EVENT_ABSORB || (sample.event_type & BSDF_EVENT_SPECULAR) != 0 ||
            dot(si.shading_normal, sample.wi) >= 0.0f)
        {
            continue;
        }

        const BsdfEvalResult evaluated = bsdf_eval(si, sample.wi);
        const double oraclePdf = mixed_transmission_pdf_oracle(si, sample.wi);
        const double evalScale =
            std::max({ static_cast<double>(sample.pdf), static_cast<double>(evaluated.pdf), 1.0e-8 });
        const double oracleScale = std::max({ static_cast<double>(sample.pdf), oraclePdf, 1.0e-8 });
        ++compared;
        if (std::abs(static_cast<double>(sample.pdf) - static_cast<double>(evaluated.pdf)) > 1.0e-3 * evalScale)
        {
            ++evalMismatches;
        }
        if (std::abs(static_cast<double>(sample.pdf) - oraclePdf) > 1.0e-3 * oracleScale)
        {
            ++oracleMismatches;
        }
    }

    CAPTURE(compared);
    CAPTURE(evalMismatches);
    CAPTURE(oracleMismatches);
    CHECK(compared > 10000);
    CHECK(evalMismatches == 0);
    CHECK(oracleMismatches == 0);
}

TEST_CASE("splitting an identical continuous lobe leaves a mixture estimator unchanged")
{
    const double otherWeight = 0.3;
    const double splitWeightA = 0.2;
    const double splitWeightB = 0.5;
    const double otherPdf = 0.11;
    const double duplicatedPdf = 0.73;
    const double unsplitPdf = otherWeight * otherPdf + (splitWeightA + splitWeightB) * duplicatedPdf;
    const double splitPdf = otherWeight * otherPdf + splitWeightA * duplicatedPdf + splitWeightB * duplicatedPdf;
    const double numerator = 0.42;

    CHECK(splitPdf == doctest::Approx(unsplitPdf).epsilon(1.0e-15));
    CHECK(numerator / splitPdf == doctest::Approx(numerator / unsplitPdf).epsilon(1.0e-15));
}

TEST_CASE("sub-threshold GGX lobes are exact delta atoms")
{
    GridPoint conductorPoint;
    conductorPoint.materialType = MATERIAL_TYPE_CONDUCTOR;
    conductorPoint.roughness = 0.02f;
    conductorPoint.metallic = 1.0f;
    conductorPoint.viewTilt = 0.31f;
    const SurfaceInteraction conductor = make_si(conductorPoint);
    const float3 mirror = reflect_dir(-conductor.wo, conductor.shading_normal);

    const BsdfSampleResult conductorA = conductor_sample(conductor, 0.01f, 0.02f);
    const BsdfSampleResult conductorB = conductor_sample(conductor, 0.91f, 0.72f);
    CHECK((conductorA.event_type & BSDF_EVENT_SPECULAR_REFLECTION) != 0u);
    CHECK((conductorB.event_type & BSDF_EVENT_SPECULAR_REFLECTION) != 0u);
    CHECK(length(conductorA.wi - mirror) < 1.0e-6f);
    CHECK(length(conductorB.wi - mirror) < 1.0e-6f);
    CHECK(conductorA.pdf == doctest::Approx(1.0f));
    CHECK(conductorB.pdf == doctest::Approx(1.0f));
    CHECK(conductor_eval(conductor, mirror).pdf == 0.0f);
    CHECK(conductor_pdf(conductor, mirror) == 0.0f);

    // Frozen mutation: the former implementation still VNDF-sampled H and only
    // changed the event label. These two supposed atoms are visibly different.
    float3 T, B;
    build_onb(conductor.shading_normal, T, B);
    const float3 localV = world_to_local(conductor.wo, T, B, conductor.shading_normal);
    const float alpha = alpha_from_roughness(conductor.roughness);
    const float3 oldA = reflect_dir(
        -conductor.wo, local_to_world(ggx_vndf_sample(localV, alpha, 0.01f, 0.02f), T, B, conductor.shading_normal));
    const float3 oldB = reflect_dir(
        -conductor.wo, local_to_world(ggx_vndf_sample(localV, alpha, 0.91f, 0.72f), T, B, conductor.shading_normal));
    CHECK(length(oldA - oldB) > 1.0e-4f);

    GridPoint metalPoint = conductorPoint;
    metalPoint.materialType = MATERIAL_TYPE_STANDARD_PBR;
    const SurfaceInteraction metal = make_si(metalPoint);
    const float3 metalMirror = reflect_dir(-metal.wo, metal.shading_normal);
    const BsdfSampleResult metalA = standard_pbr_sample(metal, 0.01f, 0.02f, 0.5f, 0.5f);
    const BsdfSampleResult metalB = standard_pbr_sample(metal, 0.91f, 0.72f, 0.5f, 0.5f);
    CHECK((metalA.event_type & BSDF_EVENT_SPECULAR_REFLECTION) != 0u);
    CHECK((metalB.event_type & BSDF_EVENT_SPECULAR_REFLECTION) != 0u);
    CHECK(length(metalA.wi - metalMirror) < 1.0e-6f);
    CHECK(length(metalB.wi - metalMirror) < 1.0e-6f);
    CHECK(metalA.pdf == doctest::Approx(1.0f));
    CHECK(metalB.pdf == doctest::Approx(1.0f));
    CHECK(standard_pbr_eval(metal, metalMirror).pdf == 0.0f);
}

TEST_CASE("coincident Standard PBR delta lobes return their full marginal mass")
{
    SurfaceInteraction si = mixed_transmission_si(0.02f, 0.5f, 0.0f, 0.4f);
    si.clearcoat = 1.0f;
    si.clearcoat_roughness = 0.02f;

    const PbrLobeWeights w = pbr_lobe_weights(si);
    const float pDiffuse = w.diffuse / w.total;
    const float pSpecular = w.specular / w.total;
    const float pTransmission = w.transmission / w.total;
    const float pClearcoat = w.clearcoat / w.total;
    REQUIRE(pSpecular > 0.0f);
    REQUIRE(pTransmission > 0.0f);
    REQUIRE(pClearcoat > 0.0f);

    const float f = fresnel_dielectric(dot(si.shading_normal, si.wo), si.exterior_ior / si.ior);
    const double expectedMass = static_cast<double>(pSpecular) + static_cast<double>(pClearcoat) +
                                static_cast<double>(pTransmission) * static_cast<double>(f);
    const float specularDraw = pDiffuse + 0.5f * pSpecular;
    const float transmissionDraw = pDiffuse + pSpecular + 0.5f * pTransmission;
    const float clearcoatDraw = pDiffuse + pSpecular + pTransmission + 0.5f * pClearcoat;
    const BsdfSampleResult fromSpecular = standard_pbr_sample(si, 0.17f, 0.63f, specularDraw, 0.0f);
    const BsdfSampleResult fromTransmission = standard_pbr_sample(si, 0.81f, 0.29f, transmissionDraw, 0.0f);
    const BsdfSampleResult fromClearcoat = standard_pbr_sample(si, 0.46f, 0.94f, clearcoatDraw, 0.0f);
    const float3 mirror = reflect_dir(-si.wo, si.shading_normal);

    for (const BsdfSampleResult& sample : { fromSpecular, fromTransmission, fromClearcoat })
    {
        CHECK((sample.event_type & BSDF_EVENT_SPECULAR_REFLECTION) != 0u);
        CHECK(length(sample.wi - mirror) < 1.0e-6f);
        CHECK(static_cast<double>(sample.pdf) == doctest::Approx(expectedMass).epsilon(2.0e-6));
        CHECK(is_finite(sample.bsdf_over_pdf));
        CHECK_FALSE(is_negative(sample.bsdf_over_pdf));
    }
    CHECK(length(fromSpecular.bsdf_over_pdf - fromTransmission.bsdf_over_pdf) < 1.0e-6f);
    CHECK(length(fromSpecular.bsdf_over_pdf - fromClearcoat.bsdf_over_pdf) < 1.0e-6f);

    // Independent double oracle for the physical coefficient carried by that
    // atom. Continuous diffuse at the same coordinate is a different measure.
    const double nDotV = static_cast<double>(dot(si.shading_normal, si.wo));
    const double f0 = 0.04;
    const double coatF = schlick_double(f0, nDotV);
    const double coatSingle = (1.0 - coatF) * (1.0 - coatF);
    const double coatCeiling = 1.0 - coatF;
    const double energyTerm = ggx_energy_term_double(si.roughness, nDotV);
    const double albedo[] = { si.albedo.x, si.albedo.y, si.albedo.z };
    const double sampled[] = { fromSpecular.bsdf_over_pdf.x, fromSpecular.bsdf_over_pdf.y, fromSpecular.bsdf_over_pdf.z };
    for (int channel = 0; channel < 3; ++channel)
    {
        const double baseAttenuation =
            std::min(coatSingle / (1.0 - coatF * std::clamp(albedo[channel], 0.0, 1.0)), coatCeiling);
        const double base = schlick_double(f0, nDotV) * 0.5 * (1.0 + f0 * energyTerm) * baseAttenuation;
        const double transmission = albedo[channel] * fresnel_dielectric_double(nDotV, 1.0 / 1.5) * 0.5;
        const double coat = coatF;
        CHECK(sampled[channel] == doctest::Approx((base + transmission + coat) / expectedMass).epsilon(2.0e-5));
    }
}

TEST_CASE("tiny Standard PBR probabilities keep support without throughput floors")
{
    const SurfaceInteraction si = mixed_transmission_si(0.02f, 1.0e-7f, 0.0f, 0.4f);
    const PbrLobeWeights w = pbr_lobe_weights(si);
    const float pDiffuse = w.diffuse / w.total;
    const float pSpecular = w.specular / w.total;
    const float pTransmission = w.transmission / w.total;
    REQUIRE(pTransmission > 0.0f);

    const float uLobe = pDiffuse + pSpecular + 0.5f * pTransmission;
    const BsdfSampleResult sample = standard_pbr_sample(si, 0.2f, 0.7f, uLobe, 0.9f);
    REQUIRE((sample.event_type & BSDF_EVENT_SPECULAR_TRANSMISSION) != 0u);

    const double cosV = static_cast<double>(dot(si.shading_normal, si.wo));
    const double eta = static_cast<double>(si.exterior_ior) / static_cast<double>(si.ior);
    const double fresnel = fresnel_dielectric_double(cosV, eta);
    const double physicalWeight = static_cast<double>(si.transmission) * (1.0 - static_cast<double>(si.metallic));
    const double expectedNumerator = physicalWeight * (1.0 - fresnel) * eta * eta;
    const float3 numerator = sample.bsdf_over_pdf * sample.pdf;
    CHECK(static_cast<double>(numerator.x / si.albedo.x) / expectedNumerator == doctest::Approx(1.0).epsilon(2.0e-5));
    CHECK(sample.pdf > 0.0f);
    CHECK(std::isfinite(sample.pdf));

    SurfaceInteraction coat = mixed_transmission_si(0.5f, 0.0f, 0.0f, 0.4f);
    coat.metallic = 1.0f;
    coat.clearcoat = 1.0e-10f;
    coat.clearcoat_roughness = 0.5f;
    const PbrLobeWeights coatWeights = pbr_lobe_weights(coat);
    const float coatStart =
        (coatWeights.diffuse + coatWeights.diffuse_transmission + coatWeights.specular + coatWeights.transmission) /
        coatWeights.total;
    const std::uint32_t firstLatticePoint = static_cast<std::uint32_t>(std::ceil(coatStart * 8388608.0f));
    REQUIRE(firstLatticePoint < 8388608u);
    const float latticeU = static_cast<float>(firstLatticePoint) / 8388608.0f;
    const BsdfSampleResult coatSample = standard_pbr_sample(coat, 0.3f, 0.8f, latticeU, 0.5f);
    CHECK(coatWeights.clearcoat > 0.0f);
    CHECK((coatSample.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
}

TEST_CASE("Standard PBR proposal masses are exact on the production lattice")
{
    SurfaceInteraction si = mixed_transmission_si(0.47f, 0.31f, 0.19f, 0.37f);
    si.clearcoat = 0.23f;
    si.sheen = 0.17f;
    const PbrLobeWeights weights = pbr_lobe_weights(si);
    const PbrLobeProbabilities p = pbr_lobe_probabilities(weights, false);

    CHECK(p.diffuse + p.diffuseTransmission + p.specular + p.transmission + p.clearcoat == 1.0f);
    CHECK(p.cdfDiffuse == discreteFloatLatticeCount(p.diffuse));
    CHECK(p.cdfDiffuseTransmission - p.cdfDiffuse == discreteFloatLatticeCount(p.diffuseTransmission));
    CHECK(p.cdfSpecular - p.cdfDiffuseTransmission == discreteFloatLatticeCount(p.specular));
    CHECK(p.cdfTransmission - p.cdfSpecular == discreteFloatLatticeCount(p.transmission));
    CHECK(STRELKA_FLOAT_LATTICE_STATES - p.cdfTransmission == discreteFloatLatticeCount(p.clearcoat));

    const PbrLobeProbabilities exiting = pbr_lobe_probabilities(weights, true);
    CHECK(exiting.diffuse == 0.0f);
    CHECK(exiting.specular == 0.0f);
    CHECK(exiting.clearcoat == 0.0f);
    CHECK(exiting.diffuseTransmission + exiting.transmission == 1.0f);

    constexpr float physicalFresnel = 0.1234567f;
    const float represented = pbr_fresnel_proposal(physicalFresnel);
    const uint32_t count = discreteFloatLatticeCount(physicalFresnel);
    CHECK(represented == float(count) * 0x1p-23f);
    CHECK(discreteFloatLatticeBernoulli(count - 1u, represented));
    CHECK_FALSE(discreteFloatLatticeBernoulli(count, represented));
}

TEST_CASE("Standard PBR returns a sub-1e-10 marginal without flooring it")
{
    SurfaceInteraction si = mixed_transmission_si(0.02f, 1.0f, 0.0f, 1.0f);
    si.albedo = make_float3(0.8f);
    si.clearcoat = 5.0e-7f;
    si.clearcoat_roughness = std::sqrt(BSDF_DELTA_ALPHA);

    const float latticeMax = static_cast<float>(8388607u) / 8388608.0f;
    const BsdfSampleResult sample = standard_pbr_sample(si, 0x1.fff8ccp-1f, 0x1.6668c4p-1f, latticeMax, 0.5f);
    REQUIRE((sample.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    const BsdfEvalResult evaluated = standard_pbr_eval(si, sample.wi);
    CHECK(evaluated.pdf > 0.0f);
    CHECK(evaluated.pdf < 2.0e-10f);
    CHECK(sample.pdf == evaluated.pdf);
    const float expected = evaluated.bsdf.x * dot(si.shading_normal, sample.wi) / evaluated.pdf;
    CHECK(sample.bsdf_over_pdf.x == doctest::Approx(expected).epsilon(2.0e-5));
}

TEST_CASE("sheen retains a cosine proposal under full interface transmission")
{
    SurfaceInteraction si = mixed_transmission_si(0.02f, 1.0f, 0.0f, 0.4f);
    si.sheen = 1.0f;
    si.sheen_color = make_float3(1.0f);
    si.sheen_roughness = 0.5f;
    const PbrLobeWeights w = pbr_lobe_weights(si);
    CHECK(w.diffuse > 0.0f);
    CHECK(bsdf_has_smooth_lobe(si));

    const float pDiffuse = w.diffuse / w.total;
    REQUIRE(pDiffuse > 0.0f);
    const BsdfSampleResult sample = standard_pbr_sample(si, 0.3f, 0.7f, 0.5f * pDiffuse, 0.5f);
    REQUIRE((sample.event_type & BSDF_EVENT_DIFFUSE_REFLECTION) != 0u);
    const BsdfEvalResult evaluated = standard_pbr_eval(si, sample.wi);
    CHECK(evaluated.pdf > 0.0f);
    CHECK(sample.pdf == doctest::Approx(evaluated.pdf).epsilon(kPdfTolerance));
    CHECK(is_finite(sample.bsdf_over_pdf));
}

TEST_CASE("GGX density is not numerically broadened at the continuous threshold")
{
    const double alpha = static_cast<double>(BSDF_DELTA_ALPHA);
    const double oracle = 1.0 / (M_PI * alpha * alpha);
    CHECK(static_cast<double>(ggx_ndf(BSDF_DELTA_ALPHA, 1.0f)) == doctest::Approx(oracle).epsilon(2.0e-4));
    CHECK(static_cast<double>(ggx_ndf_aniso(BSDF_DELTA_ALPHA, BSDF_DELTA_ALPHA, make_float3(0.0f, 0.0f, 1.0f))) ==
          doctest::Approx(oracle).epsilon(2.0e-4));
}

TEST_CASE("thin transmission uses its widened roughness classification")
{
    SurfaceInteraction si = mixed_transmission_si(0.03f, 1.0f, 0.0f, 0.0f, false, true);
    si.ior = 2.5f;
    const float baseAlpha = alpha_from_roughness(si.roughness);
    const float transmissionAlpha = thin_glass_transmission_alpha(baseAlpha, si.ior / si.exterior_ior);
    REQUIRE(baseAlpha < BSDF_DELTA_ALPHA);
    REQUIRE(transmissionAlpha >= BSDF_DELTA_ALPHA);
    REQUIRE(bsdf_has_smooth_lobe(si));

    const BsdfSampleResult sample = standard_pbr_sample(si, 0.2f, 0.7f, 0.5f, 0.5f);
    REQUIRE((sample.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    const BsdfEvalResult evaluated = standard_pbr_eval(si, sample.wi);
    CHECK(evaluated.pdf > 0.0f);
    CHECK(sample.pdf == evaluated.pdf);
    CHECK(is_finite(sample.bsdf_over_pdf));
}

TEST_CASE("thin Standard PBR exposes its rough Fresnel reflection to MIS")
{
    const SurfaceInteraction si = mixed_transmission_si(0.033f, 1.0f, 0.0f, 0.31f, false, true);
    const float alpha = alpha_from_roughness(si.roughness);
    const float alphaTransmission = thin_glass_transmission_alpha(alpha, si.ior / si.exterior_ior);
    REQUIRE(alpha >= BSDF_DELTA_ALPHA);
    REQUIRE(alphaTransmission < BSDF_DELTA_ALPHA);

    const BsdfSampleResult reflected = standard_pbr_sample(si, 0.2f, 0.7f, 0.5f, 0.0f);
    REQUIRE((reflected.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    const BsdfEvalResult evaluated = standard_pbr_eval(si, reflected.wi);
    CHECK(reflected.pdf == evaluated.pdf);
    CHECK(reflected.pdf > 0.0f);
    CHECK(bsdf_has_smooth_lobe(si));
}

TEST_CASE("anisotropic Standard PBR is delta only when both GGX axes are delta")
{
    GridPoint point;
    point.roughness = 0.03f;
    point.metallic = 1.0f;
    SurfaceInteraction si = make_si(point);
    si.anisotropy = 1.0f;

    float ax = 0.0f;
    float ay = 0.0f;
    anisotropic_alpha(si.roughness, si.anisotropy, ax, ay);
    REQUIRE(alpha_from_roughness(si.roughness) < BSDF_DELTA_ALPHA);
    REQUIRE(ax > BSDF_DELTA_ALPHA);
    REQUIRE(ay < BSDF_DELTA_ALPHA);

    const BsdfSampleResult a = standard_pbr_sample(si, 0.2f, 0.3f, 0.5f, 0.5f);
    const BsdfSampleResult b = standard_pbr_sample(si, 0.8f, 0.7f, 0.5f, 0.5f);
    REQUIRE((a.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    REQUIRE((b.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    CHECK(length(a.wi - b.wi) > 1.0e-5f);
    CHECK(a.pdf == standard_pbr_eval(si, a.wi).pdf);
    CHECK(b.pdf == standard_pbr_eval(si, b.wi).pdf);
    CHECK(bsdf_has_smooth_lobe(si));
}

TEST_CASE("an index-matched rough interface is a transmission atom")
{
    for (const unsigned int materialType : { MATERIAL_TYPE_STANDARD_PBR, MATERIAL_TYPE_DIELECTRIC })
    {
        GridPoint point;
        point.materialType = materialType;
        point.roughness = 0.5f;
        point.transmission = 1.0f;
        point.viewTilt = 0.31f;
        SurfaceInteraction si = make_si(point);
        si.ior = 1.0f;
        si.exterior_ior = 1.0f;
        const float3 expected = -si.wo;

        for (const float2 u : { make_float2(0.01f, 0.02f), make_float2(0.2f, 0.7f), make_float2(0.91f, 0.72f) })
        {
            const BsdfSampleResult sample = materialType == MATERIAL_TYPE_STANDARD_PBR ?
                                                standard_pbr_sample(si, u.x, u.y, 0.5f, 0.5f) :
                                                dielectric_sample(si, u.x, u.y, 0.5f);
            CAPTURE(materialType);
            CHECK((sample.event_type & BSDF_EVENT_SPECULAR_TRANSMISSION) != 0u);
            CHECK(length(sample.wi - expected) < 1.0e-6f);
            CHECK(sample.pdf == doctest::Approx(1.0f));
            CHECK(length(sample.bsdf_over_pdf - si.albedo) < 1.0e-6f);
        }
        CHECK_FALSE(bsdf_has_smooth_lobe(si));
        CHECK(bsdf_eval(si, expected).pdf == 0.0f);
    }
}

TEST_CASE("grazing GGX sample and evaluation use the same stable quotient")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.wo = normalize(make_float3(std::sqrt(1.0f - 1.0e-12f), 1.0e-6f, 0.0f));
    si.albedo = make_float3(0.8f);
    si.roughness = 0.5f;
    si.material_type = MATERIAL_TYPE_CONDUCTOR;

    const BsdfSampleResult sample = conductor_sample(si, 0x1.ffffe4p-1f, 0x1.ec718cp-1f);
    REQUIRE((sample.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    const BsdfEvalResult evaluated = conductor_eval(si, sample.wi);
    const float lhs = sample.bsdf_over_pdf.x * sample.pdf;
    const float rhs = evaluated.bsdf.x * dot(si.shading_normal, sample.wi);
    CHECK(lhs == doctest::Approx(rhs).epsilon(2.0e-5));
    CHECK(std::isfinite(lhs));
    CHECK(std::isfinite(rhs));

    const float smallest = std::numeric_limits<float>::denorm_min();
    CHECK(std::isfinite(ggx_smith_g2(0.25f, smallest, smallest)));
}

TEST_CASE("thin dielectric pass-through is a discrete event")
{
    GridPoint point;
    point.materialType = MATERIAL_TYPE_DIELECTRIC;
    point.roughness = 0.5f;
    point.transmission = 1.0f;
    point.viewTilt = 0.31f;
    SurfaceInteraction si = make_si(point);
    si.thin_walled = true;
    const float expectedMass = 1.0f - fresnel_dielectric(dot(si.shading_normal, si.wo), si.exterior_ior / si.ior);

    for (const float2 u : { make_float2(0.01f, 0.02f), make_float2(0.2f, 0.7f), make_float2(0.91f, 0.72f) })
    {
        const BsdfSampleResult sample = dielectric_sample(si, u.x, u.y, 0.5f);
        CHECK((sample.event_type & BSDF_EVENT_SPECULAR_TRANSMISSION) != 0u);
        CHECK(length(sample.wi + si.wo) < 1.0e-6f);
        CHECK(sample.pdf == doctest::Approx(expectedMass));
        CHECK(length(sample.bsdf_over_pdf - si.albedo) < 1.0e-6f);
    }
    CHECK(dielectric_eval(si, -si.wo).pdf == 0.0f);
}

TEST_CASE("standalone thin dielectric uses the entering interface on both faces")
{
    GridPoint point;
    point.materialType = MATERIAL_TYPE_DIELECTRIC;
    point.roughness = 0.02f;
    point.transmission = 1.0f;
    point.viewTilt = static_cast<float>(M_PI / 3.0);
    SurfaceInteraction si = make_si(point);
    si.thin_walled = true;
    si.wo.y = -si.wo.y;
    si.front_face = false;

    const float enteringEta = si.exterior_ior / si.ior;
    const float expectedFresnel = fresnel_dielectric(0.5f, enteringEta);
    REQUIRE(expectedFresnel < 0.5f);
    const BsdfSampleResult sample = dielectric_sample(si, 0.2f, 0.7f, 0.5f);
    REQUIRE((sample.event_type & BSDF_EVENT_SPECULAR_TRANSMISSION) != 0u);
    CHECK(length(sample.wi + si.wo) < 1.0e-7f);
    CHECK(sample.pdf == doctest::Approx(1.0f - expectedFresnel).epsilon(2.0e-6));
    CHECK(length(sample.bsdf_over_pdf - si.albedo) < 1.0e-7f);
}

TEST_CASE("iridescent transmission reflection has proposal support")
{
    SurfaceInteraction rough = mixed_transmission_si(0.5f, 1.0f, 0.0f, 0.31f);
    rough.ior = 1.0f;
    rough.exterior_ior = 1.0f;
    rough.albedo = make_float3(1.0f);
    rough.iridescence = 1.0f;
    rough.iridescence_ior = 1.3f;
    rough.iridescence_thickness = 500.0f;

    const float3 mirror = reflect_dir(-rough.wo, rough.shading_normal);
    const BsdfEvalResult evaluated = standard_pbr_eval(rough, mirror);
    CHECK(max_component(evaluated.bsdf) > 0.0);
    CHECK(evaluated.pdf > 0.0f);
    CHECK(bsdf_has_smooth_lobe(rough));

    const BsdfSampleResult reflection = standard_pbr_sample(rough, 0.2f, 0.7f, 0.5f, 0.0f);
    CHECK((reflection.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    CHECK(reflection.pdf > 0.0f);

    SurfaceInteraction smooth = rough;
    smooth.roughness = 0.02f;
    const BsdfSampleResult atom = standard_pbr_sample(smooth, 0.2f, 0.7f, 0.5f, 0.0f);
    CHECK((atom.event_type & BSDF_EVENT_SPECULAR_REFLECTION) != 0u);
    CHECK(atom.pdf > 0.0f);
    CHECK(max_component(atom.bsdf_over_pdf * atom.pdf) > 0.0);
}

TEST_CASE("iridescence cannot reopen transmission under total internal reflection")
{
    SurfaceInteraction rough = mixed_transmission_si(0.5f, 1.0f, 0.0f, std::acos(0.5f), true);
    rough.iridescence = 1.0f;
    rough.iridescence_ior = 1.3f;
    rough.iridescence_thickness = 500.0f;
    const float eta = rough.ior / rough.exterior_ior;
    CHECK(transmission_fresnel(rough, 0.5f, eta).x == 1.0f);
    CHECK(transmission_fresnel(rough, 0.5f, eta).y == 1.0f);
    CHECK(transmission_fresnel(rough, 0.5f, eta).z == 1.0f);

    const BsdfSampleResult roughReflect = standard_pbr_sample(rough, 0.2f, 0.7f, 0.5f, 0.0f);
    const BsdfSampleResult roughFallbackMutation =
        standard_pbr_sample(rough, 0.2f, 0.7f, 0.5f, static_cast<float>(8388607u) / 8388608.0f);
    CHECK((roughReflect.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    CHECK((roughFallbackMutation.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    CHECK(length(roughReflect.wi - roughFallbackMutation.wi) < 1.0e-6f);
    CHECK(roughReflect.pdf == roughFallbackMutation.pdf);

    SurfaceInteraction smooth = rough;
    smooth.roughness = 0.02f;
    const BsdfSampleResult atom =
        standard_pbr_sample(smooth, 0.2f, 0.7f, 0.5f, static_cast<float>(8388607u) / 8388608.0f);
    CHECK((atom.event_type & BSDF_EVENT_SPECULAR_REFLECTION) != 0u);
    CHECK(atom.pdf == doctest::Approx(1.0f));
}

TEST_CASE("rough dielectric rejects a refracted endpoint on the reflection hemisphere")
{
    GridPoint point;
    point.materialType = MATERIAL_TYPE_DIELECTRIC;
    point.roughness = 0.5f;
    point.transmission = 1.0f;
    SurfaceInteraction si = make_si(point);
    si.wo = make_float3(std::sqrt(1.0f - 0.036f * 0.036f), -0.036f, 0.0f);
    si.front_face = false;

    const BsdfSampleResult sample = dielectric_sample(si, 0.911663413f, 0.778671265f, 0.158887744f);
    CHECK(sample.event_type == BSDF_EVENT_ABSORB);
    CHECK(sample.pdf == 0.0f);
    CHECK(length(sample.bsdf_over_pdf) == 0.0f);
}

TEST_CASE("rough dielectric sample uses the evaluated grazing endpoint")
{
    GridPoint point;
    point.materialType = MATERIAL_TYPE_DIELECTRIC;
    point.roughness = 0.5f;
    point.transmission = 1.0f;
    SurfaceInteraction si = make_si(point);
    si.wo = make_float3(std::sqrt(1.0f - 0.272f * 0.272f), -0.272f, 0.0f);
    si.front_face = false;
    si.albedo = make_float3(0.8f);

    const BsdfSampleResult sample = dielectric_sample(si, 0.655616164f, 0.763962150f, 0.490510464f);
    REQUIRE((sample.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    const BsdfEvalResult evaluated = dielectric_eval(si, sample.wi);
    CHECK(sample.pdf == evaluated.pdf);
    const float cosine = fabsf(dot(si.shading_normal, sample.wi));
    CHECK(sample.bsdf_over_pdf.x * sample.pdf == doctest::Approx(evaluated.bsdf.x * cosine).epsilon(2.0e-5));
    CHECK(std::isfinite(sample.pdf));
    CHECK(is_finite(sample.bsdf_over_pdf));
}

TEST_CASE("microfacet transmission rejects rounded endpoints on the wrong macro hemisphere")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(std::sqrt(1.0f - 0.282f * 0.282f), -0.282f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.5f;
    si.transmission = 1.0f;
    si.ior = 1.5f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = false;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;

    const float u1 = 0.684159279f;
    const float u2 = 0.631801605f;
    const float uFresnel = 0.993292689f;
    const BsdfSampleResult dielectric = dielectric_sample(si, u1, u2, uFresnel);
    CHECK(dielectric.event_type == BSDF_EVENT_ABSORB);
    CHECK(dielectric.pdf == 0.0f);

    si.material_type = MATERIAL_TYPE_STANDARD_PBR;
    const BsdfSampleResult standard = standard_pbr_sample(si, u1, u2, 0.5f, uFresnel);
    CHECK(standard.event_type == BSDF_EVENT_ABSORB);
    CHECK(standard.pdf == 0.0f);
}

TEST_CASE("near-index-matched rough transmission remains continuous")
{
    CHECK(refraction_is_delta(1.0f, 1.0f));
    CHECK_FALSE(refraction_is_delta(1.00100005f, 1.0f));
    CHECK_FALSE(refraction_is_delta(1.01f, 1.0f));

    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(0.866025388f, -0.5f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.5f;
    si.transmission = 1.0f;
    si.ior = 0x1.000002p0f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = false;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;

    const float u1 = 0.964016438f;
    const float u2 = 0.829291344f;
    const float uFresnel = 0.483169198f;
    const float eta = si.ior / si.exterior_ior;
    const BsdfSampleResult dielectric = dielectric_sample(si, u1, u2, uFresnel);
    REQUIRE((dielectric.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    const BsdfEvalResult dielectricEval = dielectric_eval(si, dielectric.wi);
    CHECK(dielectric.pdf == dielectricEval.pdf);
    CHECK(dielectric.pdf > 0.0f);
    CHECK(dielectric.bsdf_over_pdf.x * dielectric.pdf ==
          doctest::Approx(dielectricEval.bsdf.x * fabsf(dot(si.shading_normal, dielectric.wi))).epsilon(2.0e-5));
    CHECK(bsdf_has_smooth_lobe(si));

    // Mutation: separately rounding wi/eta before adding V loses most of the
    // cancellation residual that identifies the sampled half vector.
    const float3 Nf = -si.shading_normal;
    float3 oldHalf = safe_normalize(si.wo + dielectric.wi / eta);
    if (dot(Nf, oldHalf) < 0.0f)
        oldHalf = -oldHalf;
    const float3 recoveredHalf = refraction_half_vector(si.wo, dielectric.wi, eta, Nf);
    CHECK(length(oldHalf - recoveredHalf) > 1.0e-4f);

    si.material_type = MATERIAL_TYPE_STANDARD_PBR;
    const BsdfSampleResult standard = standard_pbr_sample(si, u1, u2, 0.5f, uFresnel);
    REQUIRE((standard.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    const BsdfEvalResult standardEval = standard_pbr_eval(si, standard.wi);
    CHECK(standard.pdf == standardEval.pdf);
    CHECK(standard.pdf > 0.0f);
    CHECK(standard.bsdf_over_pdf.x * standard.pdf ==
          doctest::Approx(standardEval.bsdf.x * fabsf(dot(si.shading_normal, standard.wi))).epsilon(2.0e-5));
    CHECK(bsdf_has_smooth_lobe(si));

    SurfaceInteraction thin = si;
    thin.material_type = MATERIAL_TYPE_DIELECTRIC;
    thin.thin_walled = true;
    const float thinFresnel = fresnel_dielectric(0.5f, thin.exterior_ior / thin.ior);
    REQUIRE(thinFresnel > 0.0f);
    const BsdfSampleResult thinReflection = dielectric_sample(thin, 0.2f, 0.7f, 0.0f);
    REQUIRE((thinReflection.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    const BsdfEvalResult thinEval = dielectric_eval(thin, thinReflection.wi);
    CHECK(thinReflection.pdf == thinEval.pdf);
    CHECK(thinReflection.pdf > 0.0f);
    CHECK(bsdf_has_smooth_lobe(thin));
}

TEST_CASE("above-threshold refraction keeps its rounded endpoint in continuous support")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(0.866025388f, 0.5f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.5f;
    si.transmission = 1.0f;
    si.ior = 1.00279999f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = true;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;
    REQUIRE_FALSE(refraction_is_delta(si.ior, si.exterior_ior));

    const BsdfSampleResult sampled = dielectric_sample(si, 0.999999642f, 0.746976733f, 0.810415268f);
    REQUIRE((sampled.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    const BsdfEvalResult evaluated = dielectric_eval(si, sampled.wi);
    CHECK(sampled.pdf == evaluated.pdf);
    CHECK(sampled.pdf > 0.0f);
    CHECK(is_finite(sampled.bsdf_over_pdf));

    // Mutation: accepting endpoint absorption here silently removes a finite
    // VNDF/Fresnel cell without renormalizing the remaining proposal.
    CHECK(sampled.event_type != BSDF_EVENT_ABSORB);
}

TEST_CASE("refraction endpoint repair does not fabricate distant support")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(std::sqrt(0.75f), 0.5f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 1.0f;
    si.transmission = 1.0f;
    si.ior = 1.5f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = true;
    const float3 unsupported = make_float3(0.2f, -std::sqrt(0.96f), 0.0f);

    si.material_type = MATERIAL_TYPE_DIELECTRIC;
    CHECK(dielectric_eval(si, unsupported).pdf == 0.0f);
    si.material_type = MATERIAL_TYPE_STANDARD_PBR;
    CHECK(standard_pbr_eval(si, unsupported).pdf == 0.0f);

    const float eta = si.exterior_ior / si.ior;
    const float3 raw = normalizeFiniteVectorOrZero(eta * si.wo + unsupported);
    REQUIRE(dot(si.shading_normal, raw) < -0.6f);
    // Mutation: unconditional tangent-plane projection turns this large support
    // violation into H=(1,0,0) and produces a positive BTDF density.
    const float3 projected = normalizeFiniteVectorOrZero(raw - dot(si.shading_normal, raw) * si.shading_normal);
    CHECK(dot(si.wo, projected) > 0.0f);
}

TEST_CASE("rounded near-match refraction keeps its Fresnel-side support")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(std::sqrt(1.0f - 0.001f * 0.001f), -0.001f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.0320000015f;
    si.transmission = 1.0f;
    si.ior = std::nextafter(1.0f, 2.0f);
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = false;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;

    const BsdfSampleResult sampled = dielectric_sample(si, 0.33584404f, 0.208683968f, 0.107509494f);
    REQUIRE((sampled.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    const BsdfEvalResult evaluated = dielectric_eval(si, sampled.wi);
    CHECK(sampled.pdf == evaluated.pdf);
    CHECK(sampled.pdf > 0.0f);
    CHECK(is_finite(sampled.bsdf_over_pdf));

    // Mutation: the unconstrained inverse lands below the critical V.H even
    // though this float direction cell contains the sampled valid refraction.
    const float eta = si.ior / si.exterior_ior;
    const float3 inverse = normalizeFiniteVectorOrZero(eta * si.wo + sampled.wi);
    CHECK(fresnel_dielectric(fabsf(dot(si.wo, inverse)), eta) == 1.0f);
}

TEST_CASE("rounded entering near-match keeps its grazing Fresnel support")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(1.0f, 9.99999935e-39f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.0320000015f;
    si.transmission = 1.0f;
    si.ior = std::nextafter(1.0f, 2.0f);
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = true;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;

    const BsdfSampleResult sampled = dielectric_sample(si, 0.998710155f, 0.313050032f, 0.90280354f);
    REQUIRE((sampled.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    const BsdfEvalResult evaluated = dielectric_eval(si, sampled.wi);
    CHECK(sampled.pdf == evaluated.pdf);
    CHECK(sampled.pdf > 0.0f);
    CHECK(is_finite(sampled.bsdf_over_pdf));
}

TEST_CASE("near-match refraction Jacobian agrees with a double cancellation oracle")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(1.0f, 1.53052426e-7f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.977256656f;
    si.transmission = 1.0f;
    si.ior = 1.00000048f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = true;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;

    const BsdfSampleResult sampled = dielectric_sample(si, 0.22552526f, 0.712052941f, 0.253172755f);
    REQUIRE((sampled.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);

    const double eta = static_cast<double>(si.exterior_ior / si.ior);
    double hx = eta * static_cast<double>(si.wo.x) + static_cast<double>(sampled.wi.x);
    double hy = eta * static_cast<double>(si.wo.y) + static_cast<double>(sampled.wi.y);
    double hz = eta * static_cast<double>(si.wo.z) + static_cast<double>(sampled.wi.z);
    const double hLength = std::sqrt(hx * hx + hy * hy + hz * hz);
    hx /= hLength;
    hy /= hLength;
    hz /= hLength;
    if (hy < 0.0)
    {
        hx = -hx;
        hy = -hy;
        hz = -hz;
    }
    const double vDotH = static_cast<double>(si.wo.x) * hx + static_cast<double>(si.wo.y) * hy;
    const double lDotH = static_cast<double>(sampled.wi.x) * hx + static_cast<double>(sampled.wi.y) * hy +
                         static_cast<double>(sampled.wi.z) * hz;
    const double alpha = static_cast<double>(si.roughness) * static_cast<double>(si.roughness);
    const double alphaSquared = alpha * alpha;
    const double denominator = hx * hx + hz * hz + alphaSquared * hy * hy;
    const double D = alphaSquared / (M_PI * denominator * denominator);
    const double nDotV = static_cast<double>(si.wo.y);
    const double root = std::sqrt(alphaSquared + (1.0 - alphaSquared) * nDotV * nDotV);
    const double g1 = 2.0 * nDotV / (nDotV + root);
    const double pdfH = D * g1 * vDotH / nDotV;
    const double jacobian = std::abs(lDotH) / std::pow(eta * vDotH + lDotH, 2.0);
    const double oracle = (1.0 - fresnel_dielectric_double(vDotH, eta)) * pdfH * jacobian;
    CHECK(static_cast<double>(sampled.pdf) == doctest::Approx(oracle).epsilon(5.0e-4));

    const float naiveDenominator =
        (si.exterior_ior / si.ior) *
            dot(si.wo, refraction_half_vector(si.wo, sampled.wi, si.exterior_ior / si.ior, si.shading_normal)) +
        dot(sampled.wi, refraction_half_vector(si.wo, sampled.wi, si.exterior_ior / si.ior, si.shading_normal));
    const float etaFloat = si.exterior_ior / si.ior;
    const float compensatedDenominator =
        finiteVectorLength(make_float3(fmaf(etaFloat, si.wo.x, sampled.wi.x), fmaf(etaFloat, si.wo.y, sampled.wi.y),
                                       fmaf(etaFloat, si.wo.z, sampled.wi.z)));
    CHECK(fabsf(naiveDenominator - compensatedDenominator) > 1.0e-8f);
}

TEST_CASE("near-critical dielectric Fresnel agrees with a double cancellation oracle")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(0.99994725f, -0.0102710156f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.169755638f;
    si.transmission = 1.0f;
    si.ior = 1.00001323f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = false;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;

    const BsdfSampleResult sampled = dielectric_sample(si, 0.974763274f, 0.141315222f, 0.961037874f);
    REQUIRE((sampled.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    REQUIRE(sampled.pdf > 0.0f);

    const float eta = si.ior / si.exterior_ior;
    const float3 halfVector = refraction_half_vector(si.wo, sampled.wi, eta, -si.shading_normal);
    const float vDotH = dot(si.wo, halfVector);
    const float actualFresnel = fresnel_dielectric(vDotH, eta);
    const double oracleFresnel = fresnel_dielectric_double(static_cast<double>(vDotH), static_cast<double>(eta));
    CHECK(static_cast<double>(actualFresnel) == doctest::Approx(oracleFresnel).epsilon(1.0e-5));

    const double alpha = static_cast<double>(si.roughness) * static_cast<double>(si.roughness);
    const double alphaSquared = alpha * alpha;
    const double nDotH = std::abs(static_cast<double>(halfVector.y));
    const double ndfDenominator = static_cast<double>(halfVector.x) * static_cast<double>(halfVector.x) +
                                  static_cast<double>(halfVector.z) * static_cast<double>(halfVector.z) +
                                  alphaSquared * nDotH * nDotH;
    const double D = alphaSquared / (M_PI * ndfDenominator * ndfDenominator);
    const double nDotV = std::abs(static_cast<double>(si.wo.y));
    const double vDotHDouble = static_cast<double>(vDotH);
    const double g1Root = std::sqrt(alphaSquared + (1.0 - alphaSquared) * nDotV * nDotV);
    const double g1 = 2.0 * nDotV / (nDotV + g1Root);
    const double pdfH = D * g1 * vDotHDouble / nDotV;
    const double lDotH = static_cast<double>(dot(sampled.wi, halfVector));
    const double residualLength = std::sqrt(
        std::pow(static_cast<double>(eta) * static_cast<double>(si.wo.x) + static_cast<double>(sampled.wi.x), 2.0) +
        std::pow(static_cast<double>(eta) * static_cast<double>(si.wo.y) + static_cast<double>(sampled.wi.y), 2.0) +
        std::pow(static_cast<double>(eta) * static_cast<double>(si.wo.z) + static_cast<double>(sampled.wi.z), 2.0));
    const double jacobian = std::abs(lDotH) / (residualLength * residualLength);
    const double oraclePdf = (1.0 - oracleFresnel) * pdfH * jacobian;
    CHECK(std::abs(static_cast<double>(sampled.pdf) / oraclePdf - 1.0) < 5.0e-4);
}

TEST_CASE("near-critical inverse half vector preserves its double-precision view cosine")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(0.622382998f, -0.782712817f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.344393492f;
    si.transmission = 1.0f;
    si.ior = 1.40776122f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = false;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;

    const BsdfSampleResult sampled = dielectric_sample(si, 0.73514235f, 0.116987467f, 0.999999881f);
    REQUIRE((sampled.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    REQUIRE(sampled.pdf > 0.0f);

    const double eta = static_cast<double>(si.ior / si.exterior_ior);
    double hx = eta * static_cast<double>(si.wo.x) + static_cast<double>(sampled.wi.x);
    double hy = eta * static_cast<double>(si.wo.y) + static_cast<double>(sampled.wi.y);
    double hz = eta * static_cast<double>(si.wo.z) + static_cast<double>(sampled.wi.z);
    const double residualLength = std::sqrt(hx * hx + hy * hy + hz * hz);
    hx /= residualLength;
    hy /= residualLength;
    hz /= residualLength;
    if (-hy < 0.0)
    {
        hx = -hx;
        hy = -hy;
        hz = -hz;
    }

    const double vDotH = static_cast<double>(si.wo.x) * hx + static_cast<double>(si.wo.y) * hy;
    const double lDotH = static_cast<double>(sampled.wi.x) * hx + static_cast<double>(sampled.wi.y) * hy +
                         static_cast<double>(sampled.wi.z) * hz;
    const double alpha = static_cast<double>(si.roughness) * static_cast<double>(si.roughness);
    const double alphaSquared = alpha * alpha;
    const double nDotH = -hy;
    const double ndfDenominator = hx * hx + hz * hz + alphaSquared * nDotH * nDotH;
    const double D = alphaSquared / (M_PI * ndfDenominator * ndfDenominator);
    const double nDotV = std::abs(static_cast<double>(si.wo.y));
    const double g1Root = std::sqrt(alphaSquared + (1.0 - alphaSquared) * nDotV * nDotV);
    const double g1 = 2.0 * nDotV / (nDotV + g1Root);
    const double pdfH = D * g1 * vDotH / nDotV;
    const double jacobian = std::abs(lDotH) / (residualLength * residualLength);
    const double oraclePdf = (1.0 - fresnel_dielectric_double(vDotH, eta)) * pdfH * jacobian;
    CHECK(std::abs(static_cast<double>(sampled.pdf) / oraclePdf - 1.0) < 5.0e-4);

    const float3 roundedHalf = refraction_half_vector(si.wo, sampled.wi, static_cast<float>(eta), -si.shading_normal);
    const double roundedVdotH = static_cast<double>(dot(si.wo, roundedHalf));
    const double roundedTransmission =
        1.0 - static_cast<double>(fresnel_dielectric(static_cast<float>(roundedVdotH), static_cast<float>(eta)));
    const double oracleTransmission = 1.0 - fresnel_dielectric_double(vDotH, eta);
    CHECK(std::abs(roundedTransmission / oracleTransmission - 1.0) > 0.01);
}

TEST_CASE("near-critical Fresnel and refraction use the same Snell discriminant")
{
    const float cosine = 0.204175785f;
    const float eta = 1.02151906f;
    const float fresnel = fresnel_dielectric(cosine, eta);
    REQUIRE(fresnel < 0.999f);

    const float3 normal = make_float3(0.0f, -1.0f, 0.0f);
    const float3 view = make_float3(std::sqrt((1.0f - cosine) * (1.0f + cosine)), -cosine, 0.0f);
    float3 refracted = make_float3(0.0f);
    CHECK(refract_dir(-view, normal, eta, refracted));

    SurfaceInteraction si = {};
    si.shading_normal = -normal;
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = view;
    si.albedo = make_float3(0.8f);
    si.roughness = 0.02f;
    si.transmission = 1.0f;
    si.ior = eta;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = false;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;

    const BsdfSampleResult sampled = dielectric_sample(si, 0.2f, 0.7f, 0.999f);
    CHECK((sampled.event_type & BSDF_EVENT_SPECULAR_TRANSMISSION) != 0u);
    CHECK(sampled.pdf == doctest::Approx(1.0f - fresnel));

    // Exact-sign counterexample for the former two-float evaluation of
    // 1-eta^2(1-c^2): the true positive remainder is only 2.80e-16, but its
    // transmission mass still spans 3.79 values of the 23-bit RNG lattice.
    const float adjacentEta = 1.0111535787582397f;
    const float adjacentCosine = 0.14811962842941284f;
    const double exactCosineSquared =
        1.0 - static_cast<double>(adjacentEta) * static_cast<double>(adjacentEta) *
                  (1.0 - static_cast<double>(adjacentCosine) * static_cast<double>(adjacentCosine));
    REQUIRE(exactCosineSquared > 0.0);
    CHECK(1.0f - fresnel_dielectric(adjacentCosine, adjacentEta) > 3.0f / 8388608.0f);
    const float3 adjacentView =
        make_float3(std::sqrt((1.0f - adjacentCosine) * (1.0f + adjacentCosine)), -adjacentCosine, 0.0f);
    CHECK(refract_dir(-adjacentView, normal, adjacentEta, adjacentCosine, refracted));

    const float rootEta = 1.0000027418136597f;
    const float rootCosine = 0.0023417097982f;
    const double exactRootSquared = 1.0 - static_cast<double>(rootEta) * static_cast<double>(rootEta) *
                                              (1.0 - static_cast<double>(rootCosine) * static_cast<double>(rootCosine));
    REQUIRE(exactRootSquared > 0.0);
    const float3 rootView = make_float3(std::sqrt((1.0f - rootCosine) * (1.0f + rootCosine)), -rootCosine, 0.0f);
    REQUIRE(refract_dir(-rootView, normal, rootEta, rootCosine, refracted));
    CHECK(std::abs(static_cast<double>(std::abs(refracted.y)) / std::sqrt(exactRootSquared) - 1.0) < 5.0e-4);
}

TEST_CASE("adjacent non-equal indices keep the same microfacet measure")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(std::sqrt(1.0f - 0.05f * 0.05f), -0.05f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.5f;
    si.transmission = 1.0f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = false;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;

    SurfaceInteraction belowSi = si;
    belowSi.ior = 1.00276208f;
    const BsdfSampleResult below = dielectric_sample(belowSi, 0.2f, 0.7f, 0.5f);
    SurfaceInteraction aboveSi = si;
    aboveSi.ior = 1.00276220f;
    const BsdfSampleResult above = dielectric_sample(aboveSi, 0.2f, 0.7f, 0.5f);

    REQUIRE((below.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    REQUIRE((above.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    CHECK(length(below.wi - above.wi) < 1.0e-5f);
    CHECK(below.pdf == dielectric_eval(belowSi, below.wi).pdf);
    CHECK(above.pdf == dielectric_eval(aboveSi, above.wi).pdf);
}

TEST_CASE("non-equal nested indices classify refraction identically from either side")
{
    GridPoint point;
    point.materialType = MATERIAL_TYPE_DIELECTRIC;
    point.roughness = 0.5f;
    point.transmission = 1.0f;
    point.viewTilt = 0.2f;
    SurfaceInteraction entering = make_si(point);
    entering.exterior_ior = 1.2358373403549194f;
    entering.ior = 1.239250898361206f;

    SurfaceInteraction exiting = entering;
    exiting.wo.y = -exiting.wo.y;
    exiting.front_face = false;

    const BsdfSampleResult fromExterior = dielectric_sample(entering, 0.2f, 0.7f, 0.5f);
    const BsdfSampleResult fromInterior = dielectric_sample(exiting, 0.2f, 0.7f, 0.5f);
    REQUIRE((fromExterior.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    REQUIRE((fromInterior.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    CHECK(fromExterior.pdf == dielectric_eval(entering, fromExterior.wi).pdf);
    CHECK(fromInterior.pdf == dielectric_eval(exiting, fromInterior.wi).pdf);
}

TEST_CASE("GGX grazing limits remain positive and finite")
{
    const float q = std::numeric_limits<float>::denorm_min();
    CHECK(ggx_smith_g1(0.25f, q) > 0.0f);
    CHECK(ggx_smith_g2(0.25f, q, q) > 0.0f);
    CHECK(ggx_vndf_pdf(0.25f, 1.0f, q, 1.0f) > 0.0f);

    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.wo = make_float3(1.0f, q, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.5f;
    const BsdfEvalResult evaluated = conductor_eval(si, make_float3(-1.0f, q, 0.0f));
    CHECK(evaluated.pdf > 0.0f);
    CHECK(is_finite(evaluated.bsdf));
    CHECK_FALSE(is_negative(evaluated.bsdf));

    const float normalMin = std::numeric_limits<float>::min();
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(1.0f, normalMin, 0.0f);
    si.albedo = make_float3(1.0f);
    si.roughness = 1.0f;
    si.metallic = 0.0f;
    si.transmission = 0.5f;
    si.ior = 1.5f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat = 1.0f;
    si.clearcoat_roughness = 0.5f;
    si.clearcoat_ior = 1.5f;
    si.front_face = true;
    const BsdfEvalResult standard = standard_pbr_eval(si, make_float3(-1.0f, normalMin, 0.0f));
    CHECK(standard.pdf > 0.0f);
    CHECK(is_finite(standard.bsdf));
    CHECK_FALSE(is_negative(standard.bsdf));
}

TEST_CASE("exit-side reflection keeps a scale-safe oriented half-vector")
{
    const float q = std::numeric_limits<float>::min();
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(1.0f, -q, 0.0f);
    si.albedo = make_float3(1.0f);
    si.roughness = 1.0f;
    si.transmission = 1.0f;
    si.ior = 1.5f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = false;
    const float3 reflected = make_float3(-1.0f, -q, 0.0f);
    const float expectedPdf = 0.5f * M_1_PI_F;

    si.material_type = MATERIAL_TYPE_DIELECTRIC;
    const BsdfEvalResult dielectric = dielectric_eval(si, reflected);
    CHECK(dielectric.pdf == doctest::Approx(expectedPdf).epsilon(2.0e-6));
    CHECK(is_finite(dielectric.bsdf));

    si.material_type = MATERIAL_TYPE_STANDARD_PBR;
    const BsdfEvalResult standard = standard_pbr_eval(si, reflected);
    CHECK(standard.pdf == doctest::Approx(expectedPdf).epsilon(2.0e-6));
    CHECK(is_finite(standard.bsdf));

    // Mutation: the generic normalizer's fixed +Y fallback points away from
    // the exit frame and erases this otherwise valid TIR reflection.
    const float3 oldHalf = safe_normalize(si.wo + reflected);
    CHECK(dot(-si.shading_normal, oldHalf) < 0.0f);
}

TEST_CASE("near-matched continuous transmission preserves microfacet total internal reflection")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = si.shading_normal;
    si.bump_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.wo = make_float3(std::sqrt(1.0f - 0.01f * 0.01f), -0.01f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.5f;
    si.transmission = 1.0f;
    si.ior = 1.001f;
    si.exterior_ior = 1.0f;
    si.specular = 0.5f;
    si.specular_color = make_float3(1.0f);
    si.clearcoat_ior = 1.5f;
    si.front_face = false;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;
    REQUIRE(fresnel_dielectric(0.01f, si.ior / si.exterior_ior) == 1.0f);

    const BsdfSampleResult dielectric = dielectric_sample(si, 0.0f, 0.7f, 0.0f);
    REQUIRE((dielectric.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    const BsdfEvalResult dielectricEval = dielectric_eval(si, dielectric.wi);
    CHECK(dielectric.pdf == dielectricEval.pdf);
    CHECK(dielectric.pdf > 0.0f);

    si.material_type = MATERIAL_TYPE_STANDARD_PBR;
    const BsdfSampleResult standard = standard_pbr_sample(si, 0.0f, 0.7f, 0.5f, 0.0f);
    REQUIRE((standard.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    const BsdfEvalResult standardEval = standard_pbr_eval(si, standard.wi);
    CHECK(standard.pdf == standardEval.pdf);
    CHECK(standard.pdf > 0.0f);

    si.ior = 1.0f;
    si.exterior_ior = 1.0f;
    si.wo = make_float3(1.0f, 0.0f, 0.0f);
    si.material_type = MATERIAL_TYPE_DIELECTRIC;
    const BsdfSampleResult grazingDielectric = dielectric_sample(si, 0.2f, 0.7f, 0.5f);
    CHECK((grazingDielectric.event_type & BSDF_EVENT_SPECULAR_TRANSMISSION) != 0u);
    CHECK(grazingDielectric.pdf == 1.0f);
    CHECK(is_finite(grazingDielectric.bsdf_over_pdf));

    si.material_type = MATERIAL_TYPE_STANDARD_PBR;
    const BsdfSampleResult grazingMatch = standard_pbr_sample(si, 0.2f, 0.7f, 0.5f, 0.5f);
    CHECK((grazingMatch.event_type & BSDF_EVENT_SPECULAR_TRANSMISSION) != 0u);
    CHECK(grazingMatch.pdf == 1.0f);
    CHECK(std::isfinite(grazingMatch.pdf));
    CHECK(is_finite(grazingMatch.bsdf_over_pdf));
}

TEST_CASE("isotropic GGX density uses the returned half-vector tangent components")
{
    SurfaceInteraction si = {};
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.wo = make_float3(0.6f, 0.8f, 0.0f);
    si.albedo = make_float3(0.8f);
    si.roughness = 0.032f;
    si.material_type = MATERIAL_TYPE_CONDUCTOR;

    const BsdfSampleResult sample = conductor_sample(si, 0.19466269f, 0.343412042f);
    REQUIRE((sample.event_type & BSDF_EVENT_GLOSSY_REFLECTION) != 0u);
    const BsdfEvalResult evaluated = conductor_eval(si, sample.wi);
    CHECK(sample.pdf == evaluated.pdf);
    CHECK(sample.bsdf_over_pdf.x * sample.pdf ==
          doctest::Approx(evaluated.bsdf.x * dot(si.shading_normal, sample.wi)).epsilon(2.0e-5));

    // Independent double oracle from the actual returned direction. Near H=N,
    // tangent components retain the angle while 1-dot(N,H)^2 loses it.
    double hx = static_cast<double>(si.wo.x) + static_cast<double>(sample.wi.x);
    double hy = static_cast<double>(si.wo.y) + static_cast<double>(sample.wi.y);
    double hz = static_cast<double>(si.wo.z) + static_cast<double>(sample.wi.z);
    const double hLength = std::sqrt(hx * hx + hy * hy + hz * hz);
    hx /= hLength;
    hy /= hLength;
    hz /= hLength;
    const double alpha = static_cast<double>(si.roughness) * static_cast<double>(si.roughness);
    const double a2 = alpha * alpha;
    const double sin2 = hx * hx + hz * hz;
    const double denom = sin2 + a2 * hy * hy;
    const double D = a2 / (M_PI * denom * denom);
    const double nDotV = static_cast<double>(si.wo.y);
    const double root = std::sqrt(a2 + (1.0 - a2) * nDotV * nDotV);
    const double oraclePdf = D / (2.0 * (nDotV + root));
    CHECK(static_cast<double>(sample.pdf) == doctest::Approx(oraclePdf).epsilon(3.0e-4));

    const float3 returnedH = safe_normalize(si.wo + sample.wi);
    const float scalarComplementPdf = ggx_vndf_pdf(alpha_from_roughness(si.roughness), dot(si.shading_normal, returnedH),
                                                   dot(si.shading_normal, si.wo), dot(si.wo, returnedH));
    CHECK(fabsf(scalarComplementPdf - sample.pdf) / sample.pdf > 0.1f);

    si.wo = make_float3(0.6f, -0.8f, 0.0f);
    si.ior = 1.5f;
    si.exterior_ior = 1.0f;
    si.material_type = MATERIAL_TYPE_DIELECTRIC;
    const BsdfSampleResult transmission = dielectric_sample(si, 0.19466269f, 0.343412042f, 0.230215192f);
    REQUIRE((transmission.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) != 0u);
    const BsdfEvalResult transmissionEval = dielectric_eval(si, transmission.wi);
    CHECK(transmission.pdf == transmissionEval.pdf);

    const double eta = 1.5;
    hx = static_cast<double>(si.wo.x) + static_cast<double>(transmission.wi.x) / eta;
    hy = static_cast<double>(si.wo.y) + static_cast<double>(transmission.wi.y) / eta;
    hz = static_cast<double>(si.wo.z) + static_cast<double>(transmission.wi.z) / eta;
    const double transmissionHLength = std::sqrt(hx * hx + hy * hy + hz * hz);
    hx /= transmissionHLength;
    hy /= transmissionHLength;
    hz /= transmissionHLength;
    if (hy > 0.0)
    {
        hx = -hx;
        hy = -hy;
        hz = -hz;
    }
    const double transmissionSin2 = hx * hx + hz * hz;
    const double transmissionDenom = transmissionSin2 + a2 * hy * hy;
    const double transmissionD = a2 / (M_PI * transmissionDenom * transmissionDenom);
    const double vDotH = static_cast<double>(si.wo.x) * hx + static_cast<double>(si.wo.y) * hy;
    const double lDotH = static_cast<double>(transmission.wi.x) * hx + static_cast<double>(transmission.wi.y) * hy +
                         static_cast<double>(transmission.wi.z) * hz;
    const double transmissionRoot = std::sqrt(a2 + (1.0 - a2) * 0.8 * 0.8);
    const double pdfH = transmissionD * (2.0 * vDotH / (0.8 + transmissionRoot));
    const double jacobian = std::abs(lDotH) / std::pow(eta * vDotH + lDotH, 2.0);
    const double transmissionOracle = (1.0 - fresnel_dielectric_double(vDotH, eta)) * pdfH * jacobian;
    CHECK(static_cast<double>(transmission.pdf) == doctest::Approx(transmissionOracle).epsilon(5.0e-4));

    const float3 transmissionH = refraction_half_vector(si.wo, transmission.wi, 1.5f, -si.shading_normal);
    const float scalarTransmissionPdf =
        (1.0f - fresnel_dielectric(dot(si.wo, transmissionH), 1.5f)) *
        ggx_vndf_pdf_half(alpha_from_roughness(si.roughness), dot(-si.shading_normal, transmissionH), 0.8f,
                          dot(si.wo, transmissionH)) *
        refraction_jacobian(1.5f, dot(si.wo, transmissionH), dot(transmission.wi, transmissionH));
    float worstScalarMutation = fabsf(scalarTransmissionPdf - transmission.pdf) / transmission.pdf;
    FixedSeedSampler mutationRng(0x4c957f2du);
    for (int i = 0; i < 4096; ++i)
    {
        const BsdfSampleResult candidate = dielectric_sample(si, mutationRng.next(), mutationRng.next(), 0.9f);
        if ((candidate.event_type & BSDF_EVENT_GLOSSY_TRANSMISSION) == 0u)
            continue;
        const float3 candidateH = refraction_half_vector(si.wo, candidate.wi, 1.5f, -si.shading_normal);
        const float candidateVdotH = dot(si.wo, candidateH);
        const float scalarPdf = (1.0f - fresnel_dielectric(candidateVdotH, 1.5f)) *
                                ggx_vndf_pdf_half(alpha_from_roughness(si.roughness),
                                                  dot(-si.shading_normal, candidateH), 0.8f, candidateVdotH) *
                                refraction_jacobian(1.5f, candidateVdotH, dot(candidate.wi, candidateH));
        worstScalarMutation =
            fmaxf(worstScalarMutation, fabsf(scalarPdf - candidate.pdf) / fmaxf(candidate.pdf, 1.0e-30f));
    }
    CHECK(worstScalarMutation > 0.2f);
}

TEST_CASE("mixed transmission is finite and marginal across domains and roughness limits")
{
    FixedSeedSampler rng(0x7f4a7c15u);
    int continuous = 0;
    int delta = 0;
    int diffuseTransmission = 0;
    int interfaceTransmission = 0;

    for (const bool exiting : { false, true })
    {
        for (const bool thinWalled : { false, true })
        {
            for (const float roughness : { 0.0f, 0.031f, 0.032f, 0.35f, 1.0f })
            {
                for (const float transmission : { 0.0f, 0.5f, 1.0f })
                {
                    for (const float diffuseWeight : { 0.0f, 0.5f, 1.0f })
                    {
                        for (const float viewTilt : { 0.0f, 0.8f, 1.5f })
                        {
                            const SurfaceInteraction si = mixed_transmission_si(
                                roughness, transmission, diffuseWeight, viewTilt, exiting, thinWalled);
                            for (int i = 0; i < 300; ++i)
                            {
                                const BsdfSampleResult sample = bsdf_sample(si, rng.next4());
                                CAPTURE(exiting);
                                CAPTURE(thinWalled);
                                CAPTURE(roughness);
                                CAPTURE(transmission);
                                CAPTURE(diffuseWeight);
                                CAPTURE(viewTilt);
                                CAPTURE(i);

                                CHECK(std::isfinite(sample.pdf));
                                CHECK(sample.pdf >= 0.0f);
                                CHECK(is_finite(sample.bsdf_over_pdf));
                                CHECK_FALSE(is_negative(sample.bsdf_over_pdf));
                                if (sample.event_type == BSDF_EVENT_ABSORB)
                                {
                                    continue;
                                }
                                CHECK(is_finite(sample.wi));
                                if ((sample.event_type & BSDF_EVENT_DIFFUSE_TRANSMISSION) != 0u)
                                {
                                    ++diffuseTransmission;
                                }
                                if ((sample.event_type &
                                     (BSDF_EVENT_GLOSSY_TRANSMISSION | BSDF_EVENT_SPECULAR_TRANSMISSION)) != 0u)
                                {
                                    ++interfaceTransmission;
                                }
                                if ((sample.event_type & BSDF_EVENT_SPECULAR) != 0u)
                                {
                                    ++delta;
                                    // A delta sample carries probability mass. eval() may still
                                    // see an overlapping diffuse density at that exact direction,
                                    // but the two measures must not be compared or summed.
                                    CHECK(sample.pdf > 0.0f);
                                    continue;
                                }

                                ++continuous;
                                const BsdfEvalResult evaluated = bsdf_eval(si, sample.wi);
                                CHECK(evaluated.pdf > 0.0f);
                                CHECK(std::isfinite(evaluated.pdf));
                                CHECK(evaluated.pdf == doctest::Approx(sample.pdf).epsilon(kPdfTolerance));
                                const float absoluteCosine = std::fabs(dot(si.shading_normal, sample.wi));
                                const float3 fromSample = sample.bsdf_over_pdf * sample.pdf;
                                const float3 fromEval = evaluated.bsdf * absoluteCosine;
                                const double scale =
                                    std::max({ max_component(fromSample), max_component(fromEval), 1.0e-6 });
                                CHECK(static_cast<double>(length(fromSample - fromEval)) / scale < kValueTolerance);
                            }
                        }
                    }
                }
            }
        }
    }

    CHECK(continuous > 1000);
    CHECK(delta > 1000);
    CHECK(diffuseTransmission > 1000);
    CHECK(interfaceTransmission > 1000);
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
        { MATERIAL_TYPE_STANDARD_PBR, 0.3f, 0.0f, 0.0f }, { MATERIAL_TYPE_STANDARD_PBR, 0.3f, 0.0f, 0.9f },
        { MATERIAL_TYPE_STANDARD_PBR, 0.2f, 1.0f, 0.0f }, { MATERIAL_TYPE_STANDARD_PBR, 0.2f, 1.0f, 0.9f },
        { MATERIAL_TYPE_STANDARD_PBR, 0.4f, 0.5f, 0.4f }, { MATERIAL_TYPE_DIELECTRIC, 0.2f, 1.0f, 0.4f },
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
            const float3 wi = make_float3(sinTheta * std::cos(phi), cosTheta, sinTheta * std::sin(phi));
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

    for (unsigned int type :
         { MATERIAL_TYPE_DIFFUSE, MATERIAL_TYPE_CONDUCTOR, MATERIAL_TYPE_DIELECTRIC, MATERIAL_TYPE_STANDARD_PBR })
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
                            g.roughness = roughness;
                            g.metallic = metallic;
                            g.transmission = transmission;
                            g.clearcoat = clearcoat;
                            g.viewTilt = tilt;

                            const SurfaceInteraction si = make_si(g);
                            seed += 0x9E3779B9u;
                            FixedSeedSampler rng(seed);

                            int nonFinite = 0;
                            int negative = 0;

                            for (int i = 0; i < 400; ++i)
                            {
                                const BsdfSampleResult s = bsdf_sample(si, rng.next4());
                                const float3 wi = (s.event_type == BSDF_EVENT_ABSORB) ? si.wo : s.wi;
                                const BsdfEvalResult e = bsdf_eval(si, wi);
                                const float pdfOnly = bsdf_pdf(si, wi);

                                if (!std::isfinite(s.pdf) || !is_finite(s.bsdf_over_pdf) || !is_finite(wi) ||
                                    !std::isfinite(e.pdf) || !is_finite(e.bsdf) || !std::isfinite(pdfOnly))
                                {
                                    ++nonFinite;
                                }
                                if (s.pdf < 0.0f || is_negative(s.bsdf_over_pdf) || e.pdf < 0.0f ||
                                    is_negative(e.bsdf) || pdfOnly < 0.0f)
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
