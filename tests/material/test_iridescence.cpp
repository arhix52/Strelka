// ============================================================================
// test_iridescence.cpp
//
// KHR_materials_iridescence: a film thinner than a wavelength over the specular
// lobe. Light reflects off both of its faces, the two paths interfere, and which
// wavelengths survive depends on the film's optical thickness -- so a soap
// bubble is coloured without anything about it being coloured, and the colour
// turns as you move around it.
//
// That last part is the whole test. A thin film is easy to fake with a fixed
// tint, and a fixed tint passes every check except the one that matters: the hue
// has to depend on the angle and on the thickness, because the quantity that
// sets it is the optical path difference and nothing else.
//
// Pinned here:
//   1. factor 0 leaves the material bit-identical to before the film existed
//   2. a grey F0 under a film reflects a colour -- the film, not the base, is
//      what is being seen
//   3. the hue turns with the viewing angle, and with the thickness
//   4. the reflectance stays a reflectance: within [0, 1] at every angle
//   5. sample and eval agree
// ============================================================================

#include <doctest/doctest.h>

#include <strelka/material/material_math.h>
#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_params.h>
#include <strelka/material/surface_interaction.h>
#include <strelka/material/sampling.h>
#include <strelka/material/fresnel.h>
#include <strelka/material/iridescence.h>
#include <strelka/material/microfacet.h>
#include <strelka/material/bsdf.h>

#include <algorithm>
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

MaterialParams bubble_params(float factor, float thicknessNm)
{
    MaterialParams p = {};
    p.base_color = make_float3(0.02f, 0.02f, 0.02f);
    p.metallic = 0.0f;
    p.roughness = 0.15f;
    p.ior = 1.6f;
    p.specular = 0.5f;
    p.specular_color = make_float3(1.0f);
    p.transmission = 0.0f;
    p.clearcoat = 0.0f;
    p.clearcoat_roughness = 0.1f;
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
    p.iridescence = factor;
    p.iridescence_ior = 1.4f;
    p.iridescence_thickness = thicknessNm;
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

float3 dir_at(float deg)
{
    const float r = deg * 3.14159265358979f / 180.0f;
    return make_float3(std::sin(r), 0.0f, std::cos(r));
}

// How far a colour is from grey, as a fraction of its own brightness. A film
// that is working produces a number well away from zero; a base with none
// produces exactly zero for a grey F0.
float chroma(float3 c)
{
    const float mx = std::max({ c.x, c.y, c.z });
    const float mn = std::min({ c.x, c.y, c.z });
    return (mx > 1e-6f) ? (mx - mn) / mx : 0.0f;
}

} // namespace

TEST_CASE("iridescence: factor 0 changes nothing")
{
    const MaterialParams bare = bubble_params(0.0f, 400.0f);
    const MaterialParams zeroed = bubble_params(0.0f, 900.0f); // thickness set, weight not

    for (const float deg : { 10.0f, 45.0f, 80.0f })
    {
        const SurfaceInteraction a = make_si(bare, dir_at(deg));
        const SurfaceInteraction b = make_si(zeroed, dir_at(deg));
        const float3 wi = dir_at(-deg);
        const BsdfEvalResult ea = bsdf_eval(a, wi);
        const BsdfEvalResult eb = bsdf_eval(b, wi);
        CHECK(ea.bsdf.x == doctest::Approx(eb.bsdf.x));
        CHECK(ea.bsdf.y == doctest::Approx(eb.bsdf.y));
        CHECK(ea.bsdf.z == doctest::Approx(eb.bsdf.z));
    }
}

TEST_CASE("iridescence: a grey base reflects a colour")
{
    // The base F0 here is achromatic, so any colour in the result came from the
    // film. Taken at the Fresnel level rather than through the whole BSDF, which
    // would fold in the base colour and blur the point.
    const float3 greyF0 = make_float3(0.04f, 0.04f, 0.04f);
    const float3 plain = fresnel_schlick(greyF0, 0.7f);
    const float3 filmed = iridescence_fresnel(1.0f, 1.4f, 0.7f, 400.0f, greyF0);

    CHECK(chroma(plain) == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(chroma(filmed) > 0.1f);
}

TEST_CASE("iridescence: the hue turns with angle and with thickness")
{
    // The signature of interference, and what separates it from a tint. Both
    // knobs move the same quantity -- the optical path difference, 2 n d cos t --
    // so both have to move the colour.
    const float3 greyF0 = make_float3(0.04f, 0.04f, 0.04f);

    const float3 headOn = iridescence_fresnel(1.0f, 1.4f, 0.98f, 400.0f, greyF0);
    const float3 grazing = iridescence_fresnel(1.0f, 1.4f, 0.30f, 400.0f, greyF0);
    // Compared as ratios between channels, so a change in overall brightness --
    // which Fresnel produces on its own at grazing angles -- does not count as a
    // change in hue.
    const float headOnRB = headOn.x / std::max(headOn.z, 1e-6f);
    const float grazingRB = grazing.x / std::max(grazing.z, 1e-6f);
    CHECK(std::fabs(headOnRB - grazingRB) > 0.1f);

    const float3 thin = iridescence_fresnel(1.0f, 1.4f, 0.7f, 250.0f, greyF0);
    const float3 thick = iridescence_fresnel(1.0f, 1.4f, 0.7f, 750.0f, greyF0);
    const float thinRB = thin.x / std::max(thin.z, 1e-6f);
    const float thickRB = thick.x / std::max(thick.z, 1e-6f);
    CHECK(std::fabs(thinRB - thickRB) > 0.1f);
}

TEST_CASE("iridescence: a zero-thickness film is no film")
{
    // The extension fades the film's IOR to the outside medium as the thickness
    // goes to zero, so that a material with the extension present but nothing
    // authored does not shift the Fresnel it sits on.
    const float3 greyF0 = make_float3(0.04f, 0.04f, 0.04f);
    const float3 plain = fresnel_schlick(greyF0, 0.7f);
    const float3 none = iridescence_fresnel(1.0f, 1.4f, 0.7f, 0.0f, greyF0);
    CHECK(none.x == doctest::Approx(plain.x).epsilon(0.02));
    CHECK(none.y == doctest::Approx(plain.y).epsilon(0.02));
    CHECK(none.z == doctest::Approx(plain.z).epsilon(0.02));
}

TEST_CASE("iridescence: the reflectance stays a reflectance")
{
    // The Airy summation is a series of signed terms projected through a fitted
    // colour-matching curve, so nothing about its construction keeps it in range.
    const float3 greyF0 = make_float3(0.04f, 0.04f, 0.04f);
    for (int i = 0; i <= 20; ++i)
    {
        const float cosTheta = 0.02f + 0.98f * (float)i / 20.0f;
        for (const float thickness : { 100.0f, 300.0f, 550.0f, 800.0f, 1200.0f })
        {
            const float3 f = iridescence_fresnel(1.0f, 1.4f, cosTheta, thickness, greyF0);
            CHECK(f.x >= 0.0f);
            CHECK(f.y >= 0.0f);
            CHECK(f.z >= 0.0f);
            CHECK(f.x <= 1.0f);
            CHECK(f.y <= 1.0f);
            CHECK(f.z <= 1.0f);
        }
    }
}

TEST_CASE("iridescence: sample and eval agree")
{
    const MaterialParams bubble = bubble_params(1.0f, 400.0f);
    for (const float deg : { 20.0f, 60.0f })
    {
        const SurfaceInteraction si = make_si(bubble, dir_at(deg));
        Lcg rng(0x121Du + static_cast<std::uint32_t>(deg));
        int checked = 0;
        for (int i = 0; i < 600 && checked < 60; ++i)
        {
            const BsdfSampleResult s =
                bsdf_sample(si, make_float4(rng.next(), rng.next(), rng.next(), rng.next()));
            if (s.event_type == BSDF_EVENT_ABSORB || s.pdf <= 0.0f)
                continue;
            if ((s.event_type & BSDF_EVENT_SPECULAR) != 0)
                continue;

            const BsdfEvalResult e = bsdf_eval(si, s.wi);
            const float cosL = std::fabs(dot(s.wi, si.shading_normal));
            const float expected = e.bsdf.x * cosL / std::max(e.pdf, 1e-10f);
            CHECK(s.pdf == doctest::Approx(e.pdf).epsilon(0.02));
            CHECK(s.bsdf_over_pdf.x == doctest::Approx(expected).epsilon(0.02));
            ++checked;
        }
        CHECK(checked > 0);
    }
}
