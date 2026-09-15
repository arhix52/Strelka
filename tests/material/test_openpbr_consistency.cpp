
#include <doctest/doctest.h>

#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_math.h>
#include <strelka/material/surface_interaction.h>
#include <strelka/material/openpbr/openpbr_bridge.h>
#include <strelka/material/openpbr/openpbr_params.h>

#include "../support/sampling.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace
{

using oka::test::stratum;

bool finite3(float3 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool nonNegative3(float3 v)
{
    return v.x >= 0.0f && v.y >= 0.0f && v.z >= 0.0f;
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

struct NamedMaterial
{
    std::string name;
    OpenPBRParams params;
};

// One entry per lobe the bridge has to carry. Roughnesses are kept away from
// zero on purpose: this file tests the non-delta path, and the delta path is
// skipped by the BSDF_EVENT_SPECULAR guard below rather than tested here.
std::vector<NamedMaterial> lobeLadder()
{
    std::vector<NamedMaterial> out;

    auto add = [&out](const char* name, OpenPBRParams p) { out.push_back({ name, p }); };

    {
        OpenPBRParams p = openpbr_make_default_params();
        p.specular_roughness = 0.6f;
        add("diffuse base", p);
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.base_metalness = 1.0f;
        p.specular_roughness = 0.35f;
        add("rough metal", p);
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.base_metalness = 1.0f;
        p.specular_roughness = 0.35f;
        p.specular_roughness_anisotropy = 0.8f;
        add("anisotropic metal", p);
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.specular_roughness = 0.4f;
        p.coat_weight = 1.0f;
        p.coat_roughness = 0.25f;
        add("rough coat over diffuse", p);
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.specular_roughness = 0.5f;
        p.fuzz_weight = 1.0f;
        p.fuzz_roughness = 0.4f;
        add("fuzz", p);
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.transmission_weight = 1.0f;
        p.specular_roughness = 0.35f;
        add("rough transmission", p);
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.transmission_weight = 1.0f;
        p.specular_roughness = 0.35f;
        p.geometry_thin_walled = 1u;
        add("thin-walled transmission", p);
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.subsurface_weight = 1.0f;
        p.specular_roughness = 0.5f;
        add("subsurface", p);
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.specular_roughness = 0.4f;
        p.base_diffuse_roughness = 0.9f;
        add("rough diffuse (Oren-Nayar-like)", p);
    }
    {
        // Everything at once: the combination is what a real .mtlx produces and
        // what a lobe-by-lobe bridge gets wrong.
        OpenPBRParams p = openpbr_make_default_params();
        p.specular_roughness = 0.4f;
        p.base_metalness = 0.3f;
        p.coat_weight = 0.7f;
        p.coat_roughness = 0.3f;
        p.fuzz_weight = 0.4f;
        p.transmission_weight = 0.3f;
        p.subsurface_weight = 0.2f;
        p.thin_film_weight = 0.5f;
        add("all lobes", p);
    }

    return out;
}

std::array<float3, 2> views()
{
    return { make_float3(0.15f, 0.05f, 1.0f), make_float3(0.94f, 0.10f, 0.32f) };
}

constexpr int kStrata = 6;

} // namespace

TEST_CASE("openpbr bridge: bsdf_over_pdf * pdf == bsdf * |NdotL|")
{
    // The load-bearing one. This is the invariant that fails if the cosine is
    // not divided back out in openpbr_bsdf_eval().
    for (const NamedMaterial& material : lobeLadder())
    {
        CAPTURE(material.name);
        for (const float3 view : views())
        {
            const SurfaceInteraction si = surfaceLookingAt(view);
            const OpenPBR_PreparedBsdf prepared = openpbr_prepare_at(material.params, si, make_float3(1.0f));

            for (int i = 0; i < kStrata; ++i)
            {
                for (int j = 0; j < kStrata; ++j)
                {
                    for (int k = 0; k < kStrata; ++k)
                    {
                        const float4 xi =
                            make_float4(stratum(i, kStrata), stratum(j, kStrata), stratum(k, kStrata), 0.5f);
                        const BsdfSampleResult s = openpbr_bsdf_sample(prepared, xi);
                        if (s.event_type == BSDF_EVENT_ABSORB || (s.event_type & BSDF_EVENT_SPECULAR) != 0u)
                        {
                            continue;
                        }
                        REQUIRE(s.pdf > 0.0f);

                        const BsdfEvalResult e = openpbr_bsdf_eval(prepared, si, s.wi);
                        const float cosI = std::fabs(dot(si.shading_normal, s.wi));
                        if (cosI < 1e-4f)
                        {
                            continue; // f is zero here; the identity is 0 == 0
                        }

                        CAPTURE(i);
                        CAPTURE(j);
                        CAPTURE(k);
                        // Both sides are f * |cos|, reached by the two different
                        // routes the bridge is responsible for.
                        const float lhs = s.bsdf_over_pdf.x * s.pdf;
                        const float rhs = e.bsdf.x * cosI;
                        CHECK(lhs == doctest::Approx(rhs).epsilon(2e-3f));
                    }
                }
            }
        }
    }
}

TEST_CASE("openpbr bridge: eval and pdf agree with the sampler's own pdf")
{
    for (const NamedMaterial& material : lobeLadder())
    {
        CAPTURE(material.name);
        for (const float3 view : views())
        {
            const SurfaceInteraction si = surfaceLookingAt(view);
            const OpenPBR_PreparedBsdf prepared = openpbr_prepare_at(material.params, si, make_float3(1.0f));

            for (int i = 0; i < kStrata; ++i)
            {
                for (int j = 0; j < kStrata; ++j)
                {
                    const float4 xi = make_float4(stratum(i, kStrata), stratum(j, kStrata), 0.5f, 0.5f);
                    const BsdfSampleResult s = openpbr_bsdf_sample(prepared, xi);
                    if (s.event_type == BSDF_EVENT_ABSORB || (s.event_type & BSDF_EVENT_SPECULAR) != 0u)
                    {
                        continue;
                    }

                    const BsdfEvalResult e = openpbr_bsdf_eval(prepared, si, s.wi);
                    CAPTURE(i);
                    CAPTURE(j);

                    // A direction the sampler produced but whose claimed density
                    // is zero would take an infinite MIS weight.
                    CHECK(e.pdf > 0.0f);
                    CHECK(e.pdf == doctest::Approx(s.pdf).epsilon(2e-3f));
                    CHECK(openpbr_bsdf_pdf(prepared, s.wi) == doctest::Approx(e.pdf).epsilon(1e-5f));
                }
            }
        }
    }
}

TEST_CASE("openpbr bridge: nothing is NaN, infinite or negative")
{
    for (const NamedMaterial& material : lobeLadder())
    {
        CAPTURE(material.name);
        for (const float3 view : views())
        {
            const SurfaceInteraction si = surfaceLookingAt(view);
            const OpenPBR_PreparedBsdf prepared = openpbr_prepare_at(material.params, si, make_float3(1.0f));

            for (int i = 0; i < kStrata; ++i)
            {
                for (int j = 0; j < kStrata; ++j)
                {
                    for (int k = 0; k < kStrata; ++k)
                    {
                        const float4 xi =
                            make_float4(stratum(i, kStrata), stratum(j, kStrata), stratum(k, kStrata), 0.5f);
                        const BsdfSampleResult s = openpbr_bsdf_sample(prepared, xi);
                        CAPTURE(i);
                        CAPTURE(j);
                        CAPTURE(k);

                        CHECK(std::isfinite(s.pdf));
                        CHECK(s.pdf >= 0.0f);
                        CHECK(finite3(s.bsdf_over_pdf));
                        CHECK(nonNegative3(s.bsdf_over_pdf));

                        if (s.event_type == BSDF_EVENT_ABSORB)
                        {
                            continue;
                        }
                        // A sampled direction must be a unit vector, or the ray
                        // it becomes has the wrong length and every distance
                        // downstream is wrong.
                        CHECK(finite3(s.wi));
                        CHECK(length(s.wi) == doctest::Approx(1.0f).epsilon(1e-3f));

                        const BsdfEvalResult e = openpbr_bsdf_eval(prepared, si, s.wi);
                        CHECK(std::isfinite(e.pdf));
                        CHECK(e.pdf >= 0.0f);
                        CHECK(finite3(e.bsdf));
                        CHECK(nonNegative3(e.bsdf));
                    }
                }
            }
        }
    }
}

TEST_CASE("openpbr bridge: a sampled event is exactly one of the six kinds")
{
    // The lobe-flag relabelling in openpbr_lobe_to_event() is a hand-written
    // mapping between two bitmasks; this is what stops it returning a
    // combination the integrator's branches do not expect.
    for (const NamedMaterial& material : lobeLadder())
    {
        CAPTURE(material.name);
        const SurfaceInteraction si = surfaceLookingAt(views()[0]);
        const OpenPBR_PreparedBsdf prepared = openpbr_prepare_at(material.params, si, make_float3(1.0f));

        for (int i = 0; i < kStrata; ++i)
        {
            for (int j = 0; j < kStrata; ++j)
            {
                const float4 xi = make_float4(stratum(i, kStrata), stratum(j, kStrata), 0.5f, 0.5f);
                const unsigned int e = openpbr_bsdf_sample(prepared, xi).event_type;
                if (e == BSDF_EVENT_ABSORB)
                {
                    continue;
                }
                CAPTURE(e);
                // Exactly one reflection-or-transmission bit, and nothing outside
                // the set the integrator knows.
                CHECK((e & ~static_cast<unsigned int>(BSDF_EVENT_ALL)) == 0u);
                const bool reflect = (e & BSDF_EVENT_REFLECTION) != 0u;
                const bool transmit = (e & BSDF_EVENT_TRANSMISSION) != 0u;
                CHECK(reflect != transmit);
            }
        }
    }
}
