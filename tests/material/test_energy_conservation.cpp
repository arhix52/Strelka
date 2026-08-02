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

// ---------------------------------------------------------------------------
// White furnace test for the GGX lobes.
//
// Put a perfectly white conductor (F0 = 1, so every Fresnel evaluation returns
// 1) under a uniform white environment and the surface must return exactly the
// light it receives: no more -- that would be energy created out of nothing --
// and no less, since nothing absorbs. The quantity being measured is the
// directional albedo
//
//     E(wo) = integral over the hemisphere of f(wo, wi) * |cos(theta_i)| dwi
//
// which the sampling routines hand over directly: the Monte Carlo mean of
// bsdf_over_pdf IS that integral, because bsdf_over_pdf = f * cos / pdf and the
// directions are drawn from pdf. Samples that come back BSDF_EVENT_ABSORB (a
// VNDF half-vector that reflects below the horizon) count as zero, which is
// what makes the test bite -- that is precisely the energy single-scattering
// GGX throws away.
//
// This is the test that would have caught the defect fixed in
// ggx_energy_term(): before multiple-scattering compensation a white metal
// viewed at NdotV = 0.5 returned 0.451 of its incident energy at roughness 1.0
// and 0.999 at roughness 0.1 -- a loss that is invisible where these lobes are
// usually eyeballed and dominant where they are not. Stubbing the compensation
// back out turns 93 of the assertions below red.
//
// The sampler is a fixed-seed jittered stratified grid, never rand(): the same
// numbers every run, so a threshold that passes today cannot start flickering
// tomorrow.
// ---------------------------------------------------------------------------

namespace
{

// Fixed-seed LCG, same generator as tests/material/test_bsdf.cpp.
struct Lcg
{
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    float next()
    {
        s = s * 1664525u + 1013904223u;
        return (float)((s >> 8) & 0xFFFFFF) / (float)0x1000000;
    }
};

// A white metal: base_color = 1 and metallic = 1, so gltf_f0() and the
// conductor's own F0 both come out at (1,1,1) and Fresnel is identically 1.
SurfaceInteraction whiteMetal(unsigned int materialType, float roughness, float NdotV)
{
    MaterialParams p = {};
    p.material_type = materialType;
    p.base_color = make_float3(1.0f, 1.0f, 1.0f);
    p.roughness = roughness;
    p.metallic = 1.0f;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.specular_tint = 0.0f;
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

    SurfaceInteraction si = {};
    si.position = make_float3(0.0f);
    si.shading_normal = make_float3(0, 1, 0);
    si.geometry_normal = make_float3(0, 1, 0);
    si.tangent = make_float3(1, 0, 0);
    si.bitangent = make_float3(0, 0, 1);
    si.uv = make_float2(0, 0);

    const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - NdotV * NdotV));
    si.wo = normalize(make_float3(sinTheta, NdotV, 0.0f));
    si.front_face = true;

    bsdf_init(si, p, nullptr);
    si.exterior_ior = 1.0f;
    return si;
}

// Number of strata per axis; total samples per configuration is kSqrtSpp^2.
constexpr int kSqrtSpp = 64;

enum class Lobe
{
    Conductor,
    StandardPbr
};

// Monte Carlo estimate of the directional albedo, averaged over RGB (the
// material is white, so the three channels agree to the last bit; averaging
// just keeps the reported number scalar).
double directionalAlbedo(Lobe lobe, float roughness, float NdotV)
{
    const unsigned int type =
        (lobe == Lobe::Conductor) ? MATERIAL_TYPE_CONDUCTOR : MATERIAL_TYPE_STANDARD_PBR;
    const SurfaceInteraction si = whiteMetal(type, roughness, NdotV);

    // Seed derived from the configuration so neighbouring roughness values do
    // not share the same jitter pattern, but is still fixed run to run.
    Lcg rng(2463534242u + (uint32_t)(roughness * 1024.0f) * 9781u +
            (uint32_t)(NdotV * 1024.0f) * 6151u);

    double acc = 0.0;
    for (int i = 0; i < kSqrtSpp; ++i)
    {
        for (int j = 0; j < kSqrtSpp; ++j)
        {
            const float u1 = ((float)i + rng.next()) / (float)kSqrtSpp;
            const float u2 = ((float)j + rng.next()) / (float)kSqrtSpp;

            BsdfSampleResult r;
            if (lobe == Lobe::Conductor)
            {
                r = conductor_sample(si, u1, u2);
            }
            else
            {
                // metallic = 1 zeroes the diffuse and transmission weights, so
                // lobe selection lands on specular for any u_lobe; u_fresnel is
                // unused by the reflection path.
                r = standard_pbr_sample(si, u1, u2, 0.5f, 0.5f);
            }

            if (r.event_type == BSDF_EVENT_ABSORB)
            {
                continue; // energy lost to masking -- counts as zero
            }
            acc += (double)(r.bsdf_over_pdf.x + r.bsdf_over_pdf.y + r.bsdf_over_pdf.z) / 3.0;
        }
    }
    return acc / (double)(kSqrtSpp * kSqrtSpp);
}

const float kRoughnesses[] = { 0.0f, 0.05f, 0.1f, 0.2f, 0.3f, 0.4f,
                               0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };

// Grazing (0.1) through normal incidence (1.0).
const float kViewCosines[] = { 0.1f, 0.25f, 0.5f, 0.75f, 0.95f, 1.0f };

} // namespace

// ---------------------------------------------------------------------------
// The upper bound: a white furnace may never brighten.
//
// The compensation factor is a fit, so it overshoots slightly in places; the
// tolerance is the fit's own worst-case error (~2.4% at grazing incidence),
// not licence for the lobe to invent energy. Anything past this is a real leak.
// ---------------------------------------------------------------------------
TEST_CASE("white furnace: GGX directional albedo never exceeds one")
{
    const double kUpperBound = 1.03;

    for (Lobe lobe : { Lobe::Conductor, Lobe::StandardPbr })
    {
        for (float NdotV : kViewCosines)
        {
            for (float roughness : kRoughnesses)
            {
                const double E = directionalAlbedo(lobe, roughness, NdotV);
                CAPTURE((int)lobe);
                CAPTURE(NdotV);
                CAPTURE(roughness);
                CHECK(E <= kUpperBound);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The lower bound: this is the regression guard.
//
// Without ggx_energy_compensation() the single-scattering lobe returns only E
// of the light it should. Measured at NdotV = 0.5: 0.971 at roughness 0.3,
// 0.857 at 0.5, 0.704 at 0.7, 0.451 at 1.0 -- so this assertion fails at every
// roughness from 0.4 up the moment the compensation is removed.
//
// The compensated lobe clears the bound with room to spare: the worst case
// over the whole grid is 0.977, at grazing incidence where the fit is weakest.
// ---------------------------------------------------------------------------
TEST_CASE("white furnace: GGX keeps at least 94% of its energy at every roughness")
{
    const double kLowerBound = 0.94;

    for (Lobe lobe : { Lobe::Conductor, Lobe::StandardPbr })
    {
        for (float NdotV : kViewCosines)
        {
            for (float roughness : kRoughnesses)
            {
                const double E = directionalAlbedo(lobe, roughness, NdotV);
                CAPTURE((int)lobe);
                CAPTURE(NdotV);
                CAPTURE(roughness);
                CHECK(E >= kLowerBound);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The two entry points must agree. standard_pbr_sample() with metallic = 1
// reduces to exactly the conductor lobe, and a divergence here means the
// compensation was applied at one call site and forgotten at another -- the
// failure mode the four-call-site comment in standard_pbr.h warns about.
// ---------------------------------------------------------------------------
TEST_CASE("white furnace: standard_pbr at metallic=1 matches the conductor lobe")
{
    for (float NdotV : kViewCosines)
    {
        for (float roughness : kRoughnesses)
        {
            const double conductor = directionalAlbedo(Lobe::Conductor, roughness, NdotV);
            const double pbr = directionalAlbedo(Lobe::StandardPbr, roughness, NdotV);
            CAPTURE(NdotV);
            CAPTURE(roughness);
            CHECK(pbr == doctest::Approx(conductor).epsilon(0.01));
        }
    }
}

// ---------------------------------------------------------------------------
// Properties of the fitted term itself, independent of any integration.
// ---------------------------------------------------------------------------

TEST_CASE("ggx_energy_term vanishes at roughness zero")
{
    for (float NdotV : kViewCosines)
    {
        CAPTURE(NdotV);
        CHECK(ggx_energy_term(0.0f, NdotV) == doctest::Approx(0.0f));
    }
    // A mirror has nothing to compensate for: the multiplier is exactly 1
    // whatever F0 is.
    const float3 factor = ggx_energy_compensation(make_float3(1.0f), 0.0f, 0.5f);
    CHECK(factor.x == doctest::Approx(1.0f));
    CHECK(factor.y == doctest::Approx(1.0f));
    CHECK(factor.z == doctest::Approx(1.0f));
}

TEST_CASE("ggx_energy_compensation leaves a black lobe alone")
{
    // F0 = 0 reflects nothing, so there is no multiply-scattered energy to
    // restore; the factor must stay at 1 no matter how rough the surface is.
    for (float roughness : kRoughnesses)
    {
        const float3 factor = ggx_energy_compensation(make_float3(0.0f), roughness, 0.5f);
        CAPTURE(roughness);
        CHECK(factor.x == doctest::Approx(1.0f));
        CHECK(factor.y == doctest::Approx(1.0f));
        CHECK(factor.z == doctest::Approx(1.0f));
    }
}

TEST_CASE("ggx_energy_term is non-decreasing in roughness")
{
    // Rougher microsurfaces scatter more times before escaping, so the energy
    // to put back can only grow. The tolerance absorbs a ~1e-4 wobble the
    // polynomial fit has near roughness 0.56 at grazing incidence; it is two
    // orders of magnitude below the term's own value there.
    const float kWobble = 1e-3f;

    for (float NdotV : kViewCosines)
    {
        float previous = ggx_energy_term(0.0f, NdotV);
        float worstDrop = 0.0f;
        for (int i = 1; i <= 200; ++i)
        {
            const float roughness = (float)i / 200.0f;
            const float t = ggx_energy_term(roughness, NdotV);
            worstDrop = std::max(worstDrop, previous - t);
            previous = t;
        }
        CAPTURE(NdotV);
        CHECK(worstDrop <= kWobble);
        // And it must actually rise: a term that is flat at zero everywhere
        // would satisfy monotonicity while compensating for nothing.
        CHECK(ggx_energy_term(1.0f, NdotV) > 0.3f);
    }
}

// ---------------------------------------------------------------------------
// The fit against its reference.
//
// Reference values come from a VNDF-sampled computation of the single-
// scattering directional albedo E = mean of G2/G1, quoted as the compensation
// factor 1/E at NdotV = 0.5. ggx_energy_term() is (1/E - 1), so the factor the
// shader applies at F0 = 1 is 1 + ggx_energy_term(). Re-fitting the polynomial
// is fine; drifting away from the reference it was fitted to is not.
// ---------------------------------------------------------------------------
TEST_CASE("ggx_energy_term reproduces the VNDF reference at NdotV = 0.5")
{
    struct Ref
    {
        float roughness;
        double invE;
    };
    const Ref refs[] = {
        { 0.51f, 1.177 },
        { 0.76f, 1.519 },
        { 1.00f, 2.215 },
    };

    for (const Ref& ref : refs)
    {
        const double fitted = 1.0 + (double)ggx_energy_term(ref.roughness, 0.5f);
        CAPTURE(ref.roughness);
        CHECK(fitted == doctest::Approx(ref.invE).epsilon(0.03));
    }
}
