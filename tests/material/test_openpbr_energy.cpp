// ============================================================================
// test_openpbr_energy.cpp
//
// A white furnace test for the OpenPBR path: put a surface in a uniform unit
// radiance field and ask how much of it comes back. The answer is the
// directional albedo,
//
//     rho(wo) = integral over the sphere of  f(wo, wi) * |cos(N, wi)|  dwi
//
// and for a passive material it cannot exceed 1. Exceeding it is not a subtle
// error -- an albedo of 1.05 compounds to 1.6x over eight bounces, and shows up
// as an interior that will not stop getting brighter the longer it renders.
//
// This complements test_openpbr_consistency.cpp rather than repeating it. That
// file checks sample and eval against each other pointwise; a bridge that
// dropped an entire lobe would pass it, because the lobe would be missing
// consistently from both sides. This file integrates, so a missing lobe shows
// up as albedo that is too low, and a double-counted one as albedo too high.
//
// Two estimators, on purpose:
//
//   rho_sampled     the mean of bsdf_over_pdf over importance-sampled
//                   directions. Exact in expectation, and the only one of the
//                   two that can measure a delta lobe -- a smooth coat has no
//                   density with respect to solid angle, so quadrature simply
//                   misses it.
//   rho_quadrature  a stratified sum of bsdf * |cos| * dwi over the sphere.
//                   Knows nothing about the sampler, so agreement between the
//                   two is evidence that neither is lying. Only meaningful for
//                   rough materials, so it is only asked of those.
//
// Both are deterministic: directions come from a radical-inverse sequence, not
// a random stream, so a failure reproduces exactly rather than on average.
//
// The bound is stated as rho <= 1 + tolerance, where the tolerance is Monte
// Carlo error at the sample count used, not a claim that a few percent of extra
// energy would be acceptable.
// ============================================================================

#include <doctest/doctest.h>

#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_math.h>
#include <strelka/material/surface_interaction.h>
#include <strelka/material/openpbr/openpbr_bridge.h>
#include <strelka/material/openpbr/openpbr_params.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace
{

// Van der Corput radical inverse. Deterministic, well distributed, and it makes
// the sample count the only knob that changes a result.
float radicalInverse(unsigned int bits, unsigned int base)
{
    float result = 0.0f;
    const float invBase = 1.0f / static_cast<float>(base);
    float f = invBase;
    while (bits > 0u)
    {
        result += static_cast<float>(bits % base) * f;
        bits /= base;
        f *= invBase;
    }
    return result;
}

SurfaceInteraction surfaceLookingAt(float3 wo)
{
    SurfaceInteraction si = {};
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 1.0f, 0.0f);
    si.shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    si.geometry_normal = si.shading_normal;
    si.wo = normalize(wo);
    si.exterior_ior = 1.0f;
    si.front_face = true;
    si.material_type = MATERIAL_TYPE_OPENPBR;
    return si;
}

constexpr int kSampledCount = 8192;

// Mean of bsdf_over_pdf over importance-sampled directions == the integral.
// Returned per channel: an achromatic average would hide a lobe that only
// over-brightens one channel, which is what a mis-mapped colour looks like.
float3 albedoSampled(const OpenPBR_PreparedBsdf& prepared)
{
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < kSampledCount; ++i)
    {
        const unsigned int n = static_cast<unsigned int>(i);
        const float4 xi = make_float4(radicalInverse(n, 2), radicalInverse(n, 3), radicalInverse(n, 5), 0.5f);
        const BsdfSampleResult s = openpbr_bsdf_sample(prepared, xi);
        if (s.event_type == BSDF_EVENT_ABSORB)
        {
            continue; // contributes zero, which is already what sum holds
        }
        sum = sum + s.bsdf_over_pdf;
    }
    return sum / static_cast<float>(kSampledCount);
}

// Stratified sum of f * |cos| over the whole sphere. Rough materials only.
float3 albedoQuadrature(const OpenPBR_PreparedBsdf& prepared, const SurfaceInteraction& si)
{
    constexpr int kTheta = 96;
    constexpr int kPhi = 192;
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int t = 0; t < kTheta; ++t)
    {
        // Uniform in cos(theta) over the full sphere: [-1, 1].
        const float cosTheta = -1.0f + 2.0f * (static_cast<float>(t) + 0.5f) / static_cast<float>(kTheta);
        const float sinTheta = std::sqrt(std::fmax(0.0f, 1.0f - cosTheta * cosTheta));
        for (int p = 0; p < kPhi; ++p)
        {
            const float phi = 2.0f * 3.14159265358979323846f * (static_cast<float>(p) + 0.5f) / static_cast<float>(kPhi);
            const float3 wi = make_float3(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
            const BsdfEvalResult e = openpbr_bsdf_eval(prepared, si, wi);
            sum = sum + e.bsdf * std::fabs(cosTheta);
        }
    }
    // Solid angle of one cell: 4*pi / (kTheta * kPhi), uniform by construction.
    const float dOmega = 4.0f * 3.14159265358979323846f / static_cast<float>(kTheta * kPhi);
    return sum * dOmega;
}

float maxComponent(float3 v)
{
    return std::fmax(v.x, std::fmax(v.y, v.z));
}

struct NamedMaterial
{
    std::string name;
    OpenPBRParams params;
    bool rough; // safe to cross-check with quadrature
};

std::vector<NamedMaterial> furnaceLadder()
{
    std::vector<NamedMaterial> out;
    auto add = [&out](const char* n, OpenPBRParams p, bool rough) { out.push_back({ n, p, rough }); };

    for (const float roughness : { 0.1f, 0.35f, 0.7f, 1.0f })
    {
        {
            OpenPBRParams p = openpbr_make_default_params();
            p.base_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
            p.specular_roughness = roughness;
            add("white dielectric", p, true);
        }
        {
            // The classic case energy compensation exists for: a rough metal
            // without it loses a large fraction of its energy at high roughness.
            OpenPBRParams p = openpbr_make_default_params();
            p.base_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
            p.base_metalness = 1.0f;
            p.specular_roughness = roughness;
            add("white metal", p, true);
        }
        {
            OpenPBRParams p = openpbr_make_default_params();
            p.base_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
            p.specular_roughness = roughness;
            p.coat_weight = 1.0f;
            p.coat_roughness = roughness;
            add("coat over white base", p, true);
        }
        {
            OpenPBRParams p = openpbr_make_default_params();
            p.base_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
            p.specular_roughness = roughness;
            p.fuzz_weight = 1.0f;
            add("fuzz over white base", p, true);
        }
        {
            OpenPBRParams p = openpbr_make_default_params();
            p.transmission_weight = 1.0f;
            p.specular_roughness = roughness;
            add("transmissive", p, true);
        }
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.base_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
        p.specular_roughness = 0.0f;
        p.coat_weight = 1.0f;
        p.coat_roughness = 0.0f;
        add("smooth coat over smooth base (delta)", p, false);
    }
    return out;
}

// A function rather than a namespace-scope array: see the note in
// test_openpbr_consistency.cpp -- glm::vec3's constructor is not constexpr here.
std::array<float3, 3> views()
{
    return { make_float3(0.0f, 0.0f, 1.0f), make_float3(0.55f, 0.0f, 0.83f), make_float3(0.96f, 0.0f, 0.28f) };
}

} // namespace

TEST_CASE("openpbr white furnace: no material returns more energy than it receives")
{
    // The hard invariant. 1.5% covers Monte Carlo error at kSampledCount; it is
    // not a licence for the model to gain energy.
    constexpr float kTolerance = 0.015f;

    for (const NamedMaterial& material : furnaceLadder())
    {
        CAPTURE(material.name);
        for (const float3 view : views())
        {
            const SurfaceInteraction si = surfaceLookingAt(view);
            const OpenPBR_PreparedBsdf prepared = openpbr_prepare_at(material.params, si, make_float3(1.0f));
            const float3 rho = albedoSampled(prepared);
            CAPTURE(si.wo.z);
            CAPTURE(rho.x);
            CHECK(std::isfinite(rho.x));
            CHECK(rho.x >= 0.0f);
            CHECK(maxComponent(rho) <= 1.0f + kTolerance);
        }
    }
}

TEST_CASE("openpbr white furnace: a lossless white surface returns nearly all of it")
{
    // The other half of the bound. Without this a bridge that returned zero for
    // every lobe would pass the test above with room to spare.
    //
    // Not asked of the coat or fuzz: with OPENPBR_RECIPROCAL_COAT_AND_FUZZ = 0
    // the layering is deliberately non-reciprocal and does not conserve energy
    // exactly, which openpbr_settings.h documents as the chosen trade.
    for (const float roughness : { 0.1f, 0.35f, 0.7f, 1.0f })
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.base_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
        p.base_metalness = 1.0f;
        p.specular_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
        p.specular_roughness = roughness;

        CAPTURE(roughness);
        for (const float3 view : views())
        {
            const SurfaceInteraction si = surfaceLookingAt(view);
            const OpenPBR_PreparedBsdf prepared = openpbr_prepare_at(p, si, make_float3(1.0f));
            const float3 rho = albedoSampled(prepared);
            CAPTURE(si.wo.z);
            CAPTURE(rho.x);
            // A white conductor with multiple-scattering compensation should be
            // near-lossless at every roughness. The floor is what catches a lobe
            // that quietly stopped contributing.
            CHECK(rho.x > 0.9f);
            CHECK(rho.x <= 1.0f + 0.015f);
        }
    }
}

TEST_CASE("openpbr white furnace: sampling and evaluation integrate to the same albedo")
{
    // Two estimators that share no code path beyond the BSDF itself. Agreement
    // is what says the sampler's density really is the one eval reports --
    // integrated rather than pointwise, so it catches a lobe whose weight and
    // pdf are wrong by the same factor.
    for (const NamedMaterial& material : furnaceLadder())
    {
        if (!material.rough)
        {
            continue; // quadrature cannot see a delta lobe
        }
        if (material.params.specular_roughness < 0.3f)
        {
            continue; // near-delta: the quadrature grid cannot resolve the peak
        }
        CAPTURE(material.name);
        CAPTURE(material.params.specular_roughness);

        const SurfaceInteraction si = surfaceLookingAt(make_float3(0.55f, 0.0f, 0.83f));
        const OpenPBR_PreparedBsdf prepared = openpbr_prepare_at(material.params, si, make_float3(1.0f));
        const float3 sampled = albedoSampled(prepared);
        const float3 quadrature = albedoQuadrature(prepared, si);

        CAPTURE(sampled.x);
        CAPTURE(quadrature.x);
        // 4% absorbs both estimators' error at these resolutions. A missing or
        // doubled lobe is a far larger discrepancy than that.
        CHECK(sampled.x == doctest::Approx(quadrature.x).epsilon(0.04f));
    }
}
