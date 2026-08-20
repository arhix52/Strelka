#include <doctest/doctest.h>

#include <strelka/material/material_math.h>
#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_params.h>
#include <strelka/material/surface_interaction.h>
#include <strelka/material/sampling.h>
#include <strelka/material/fresnel.h>
#include <strelka/material/microfacet.h>
#include <strelka/material/bsdf.h>

#include "../support/sampling.h"

#include <cmath>
#include <numbers>

using oka::test::stratum;

// ---------------------------------------------------------------------------
// A thin-walled surface hit from its far side.
//
// A soap bubble is a closed sphere of film with air on both sides. A ray that
// passes through the front wall crosses the inside and meets the far wall from
// behind, so the shading normal points away from it -- geometrically identical
// to a ray leaving solid glass, and physically nothing like it. There is no
// medium being left: the far wall is another air-to-film interface, and its
// index ratio is the entering one.
//
// Deriving the ratio from the side instead made the far wall dense-to-thin,
// where everything past the critical angle reflects with probability 1. At IOR
// 1.6 the critical angle is 38.7 degrees, and the incidence angle at radius r
// on a sphere is asin(r / R), so the entire annulus outside r / R = 1 / 1.6 =
// 0.625 reflected every ray that reached it. Reflected, never absorbed -- so
// Russian roulette never ended the path and maxDepth did, after the ray had
// bounced between the two walls carrying full throughput and returning nothing.
//
// The rendered symptom was a black ring covering the outer 37.5% of every
// bubble in the Isometric Bathroom scene, which is what that arithmetic says it
// should be. tools/iso_bathroom/bubble_profile.py is what measured it: the
// luminance across a bubble sat at 0.94 of the wall behind it out to r / R =
// 0.6 and fell to 0.18 beyond it, and the break landed in the bin holding
// 0.625.
//
// What is pinned here:
//   1. the far wall really is a back-face hit, or the rest of the file is
//      testing nothing
//   2. it transmits past the solid critical angle, at every angle up to grazing
//   3. its reflectance follows the entering-side Fresnel, not the exiting one
//   4. neither wall creates or destroys energy
//   5. solid glass still total-internally-reflects, i.e. the exemption did not
//      leak into the case the critical angle is real for
// ---------------------------------------------------------------------------

namespace
{

constexpr float kIor = 1.6f;

MaterialParams film_params(float roughness, bool thin)
{
    MaterialParams p = {};
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;
    p.base_color = make_float3(1.0f, 1.0f, 1.0f);
    p.roughness = roughness;
    p.metallic = 0.0f;
    p.ior = kIor;
    p.specular = 0.5f;
    p.transmission = 1.0f;
    p.thin_walled = thin ? 1u : 0u;
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

// The wall as seen by a ray arriving at `degrees` from the normal. `front` is
// the near wall, hit from outside; otherwise it is the far wall, hit from
// within the bubble, which is the case the defect lived in.
SurfaceInteraction wall_si(float degrees, bool front, float roughness = 0.0f, bool thin = true)
{
    const float th = degrees * std::numbers::pi_v<float> / 180.0f;
    const float c = std::cos(th) * (front ? 1.0f : -1.0f);
    const float s = std::sin(th);

    SurfaceInteraction si = {};
    si.position = make_float3(0.0f, 0.0f, 0.0f);
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.uv = make_float2(0.0f, 0.0f);
    si.wo = safe_normalize(make_float3(s, c, 0.0f));
    si.front_face = front;
    bsdf_init(si, film_params(roughness, thin));
    si.exterior_ior = 1.0f;
    return si;
}

// Fraction of draws that leave through the wall rather than reflecting off it,
// and the mean throughput multiplier over the same draws. The lobe decides with
// xi.w, so sweeping it is sweeping the Fresnel coin.
void sweep(const SurfaceInteraction& si, float& transmitted, float& mean_throughput)
{
    const int N = 4096;
    int through = 0;
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
    {
        // Offset off the ends of the interval: the draw is compared against F
        // with a strict inequality, and 0 or 1 exactly would test the boundary
        // rather than the distribution.
        const float u = stratum(i, N);
        const BsdfSampleResult r = bsdf_sample(si, make_float4(0.5f, 0.5f, 0.25f, u));
        if ((r.event_type & BSDF_EVENT_TRANSMISSION) != 0)
        {
            ++through;
        }
        sum += (r.bsdf_over_pdf.x + r.bsdf_over_pdf.y + r.bsdf_over_pdf.z) / 3.0;
    }
    transmitted = (float)through / (float)N;
    mean_throughput = (float)(sum / N);
}

} // namespace

TEST_CASE("the far wall of a bubble is set up as a back-face hit")
{
    // The premise. If this stops being negative the rest of the file passes for
    // the wrong reason.
    for (float deg : { 10.0f, 50.0f, 80.0f })
    {
        const SurfaceInteraction si = wall_si(deg, /*front=*/false);
        CAPTURE(deg);
        CHECK(dot(si.shading_normal, si.wo) < 0.0f);
        CHECK(si.thin_walled == 1u);
        CHECK(si.transmission == doctest::Approx(1.0f));
    }
}

TEST_CASE("a thin wall transmits past the critical angle of the solid it is made of")
{
    // asin(1 / 1.6) = 38.68 degrees. Everything below used to reflect with
    // probability 1 on the far wall.
    const float critical = std::asin(1.0f / kIor) * 180.0f / std::numbers::pi_v<float>;
    REQUIRE(critical == doctest::Approx(38.68f).epsilon(0.01));

    for (float deg : { 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 88.0f })
    {
        const SurfaceInteraction si = wall_si(deg, /*front=*/false);
        float transmitted = 0.0f, throughput = 0.0f;
        sweep(si, transmitted, throughput);
        CAPTURE(deg);
        CHECK(transmitted > 0.0f);
    }
}

TEST_CASE("a thin wall reflects by the entering-side Fresnel from either side")
{
    // Both walls are air-to-film, so both reflect the same fraction at the same
    // angle -- which is the whole content of "thin-walled". The exiting-side
    // ratio would put the far wall at 1.0 for every angle past 38.7 degrees.
    for (float deg : { 0.0f, 20.0f, 40.0f, 60.0f, 80.0f })
    {
        const float cos_i = std::cos(deg * std::numbers::pi_v<float> / 180.0f);
        const float expect = fresnel_dielectric(cos_i, 1.0f / kIor);

        float front_t = 0.0f, front_e = 0.0f, far_t = 0.0f, far_e = 0.0f;
        sweep(wall_si(deg, /*front=*/true), front_t, front_e);
        sweep(wall_si(deg, /*front=*/false), far_t, far_e);

        CAPTURE(deg);
        // The near wall shares its draw with the clearcoat lobe selection, so
        // only the far wall -- where the transmission lobe is taken with
        // probability 1 -- reads the coin directly.
        CHECK(1.0f - far_t == doctest::Approx(expect).epsilon(0.02));
        CHECK(far_t >= front_t);
    }
}

TEST_CASE("neither wall of a bubble creates or destroys energy")
{
    for (float deg : { 0.0f, 30.0f, 60.0f, 85.0f })
    {
        for (bool front : { true, false })
        {
            float transmitted = 0.0f, throughput = 0.0f;
            sweep(wall_si(deg, front), transmitted, throughput);
            CAPTURE(deg);
            CAPTURE(front);
            // The near wall carries the clearcoat lobe as well and is checked
            // only for the ceiling; the far wall is the transmission lobe alone
            // and has to come out at exactly one.
            CHECK(throughput <= doctest::Approx(1.0f).epsilon(0.02));
            if (!front)
            {
                CHECK(throughput == doctest::Approx(1.0f).epsilon(0.02));
            }
        }
    }
}

TEST_CASE("a smooth thin wall transmits as a delta, and says so")
{
    // Rough thin walls are a real GGX lobe (OpenPBR / Cycles); only the smooth
    // case -- and the transmission roughness that collapses to smooth -- is a
    // delta at exactly -V.
    for (float rough : { 0.0f, 0.001f })
    {
        const SurfaceInteraction si = wall_si(35.0f, /*front=*/true, rough);
        int transmitted = 0;
        for (int i = 0; i < 512; ++i)
        {
            const float u = stratum(i, 512);
            const BsdfSampleResult r = bsdf_sample(si, make_float4(0.3f, 0.7f, 0.25f, u));
            if ((r.event_type & BSDF_EVENT_TRANSMISSION) == 0) continue;
            ++transmitted;
            CAPTURE(rough);
            const float3 d = r.wi - (make_float3(0.0f) - si.wo);
            CHECK(dot(d, d) == doctest::Approx(0.0f).epsilon(1e-6));
            CHECK(r.event_type == BSDF_EVENT_SPECULAR_TRANSMISSION);
        }
        REQUIRE(transmitted > 0);
    }
}

TEST_CASE("a rough thin wall blurs transmission around -V")
{
    // Every sample used to land on exactly -V. With the mirrored-GGX lobe the
    // mean stays on -V and the variance grows with roughness -- that is the
    // blur a frosted sheet has to have.
    const SurfaceInteraction si = wall_si(20.0f, /*front=*/true, /*roughness=*/0.4f);
    float3 mean = make_float3(0.0f);
    float var = 0.0f;
    int transmitted = 0;
    const float3 through = make_float3(0.0f) - si.wo;
    for (int i = 0; i < 2048; ++i)
    {
        const float u = stratum(i, 2048);
        const float u1 = stratum((i * 7) % 2048, 2048);
        const float u2 = stratum((i * 13) % 2048, 2048);
        const BsdfSampleResult r = bsdf_sample(si, make_float4(u1, u2, 0.25f, u));
        if ((r.event_type & BSDF_EVENT_TRANSMISSION) == 0) continue;
        ++transmitted;
        CHECK(r.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION);
        mean = mean + r.wi;
        const float3 d = r.wi - through;
        var += dot(d, d);
    }
    REQUIRE(transmitted > 100);
    mean = mean * (1.0f / (float)transmitted);
    var /= (float)transmitted;
    // Mean within ~10 degrees of straight through.
    CHECK(dot(safe_normalize(mean), through) > 0.98f);
    CHECK(var > 0.01f); // not a delta
}

TEST_CASE("a smooth thin wall cannot be evaluated in transmission")
{
    // eval() on a delta returns zero; a light connection through a smooth wall
    // has nothing to land on. Rough walls are evaluable -- see the next case.
    const SurfaceInteraction si = wall_si(35.0f, /*front=*/true, 0.0f);
    for (float tilt : { 0.0f, 0.15f, 0.4f })
    {
        const float3 wi = safe_normalize(make_float3(-si.wo.x + tilt, tilt, -si.wo.z));
        const BsdfEvalResult e = bsdf_eval(si, wi);
        CAPTURE(tilt);
        CHECK(dot(e.bsdf, e.bsdf) == doctest::Approx(0.0f));
    }
}

TEST_CASE("a rough thin wall is evaluable in transmission and agrees with sample")
{
    const SurfaceInteraction si = wall_si(25.0f, /*front=*/true, 0.35f);
    int checked = 0;
    for (int i = 0; i < 800 && checked < 40; ++i)
    {
        const float u = stratum(i, 800);
        const float u1 = stratum((i * 3) % 800, 800);
        const float u2 = stratum((i * 11) % 800, 800);
        BsdfSampleResult s = bsdf_sample(si, make_float4(u1, u2, 0.25f, u));
        if ((s.event_type & BSDF_EVENT_TRANSMISSION) == 0) continue;
        if ((s.event_type & BSDF_EVENT_SPECULAR) != 0) continue;

        BsdfEvalResult e = bsdf_eval(si, s.wi);
        CAPTURE(s.pdf);
        CAPTURE(e.pdf);
        CHECK(e.pdf == doctest::Approx(s.pdf).epsilon(0.05));
        // bsdf_over_pdf * pdf ~= bsdf * |cos| for the reflection-mapped lobe;
        // compare the throughput sample reported against eval's bsdf/pdf.
        const float3 wi_r = s.wi - si.shading_normal * (2.0f * dot(s.wi, si.shading_normal));
        const float cosL = std::fabs(dot(wi_r, si.shading_normal));
        const float expected = e.bsdf.x * cosL / std::max(e.pdf, 1e-10f);
        CHECK(s.bsdf_over_pdf.x == doctest::Approx(expected).epsilon(0.08));
        ++checked;
    }
    CHECK(checked > 0);
}

TEST_CASE("solid glass still total-internally-reflects")
{
    // The exemption is for materials with no interior. A solid sphere has one,
    // and past the critical angle its far wall must keep the light in -- this is
    // what makes glass look like glass, and it is the case the thin-walled
    // branch must not have leaked into.
    for (float deg : { 45.0f, 60.0f, 80.0f })
    {
        const SurfaceInteraction si = wall_si(deg, /*front=*/false, 0.0f, /*thin=*/false);
        float transmitted = 0.0f, throughput = 0.0f;
        sweep(si, transmitted, throughput);
        CAPTURE(deg);
        CHECK(transmitted == doctest::Approx(0.0f));
    }

    // And below it, the same surface lets light out.
    const SurfaceInteraction si = wall_si(20.0f, /*front=*/false, 0.0f, /*thin=*/false);
    float transmitted = 0.0f, throughput = 0.0f;
    sweep(si, transmitted, throughput);
    CHECK(transmitted > 0.9f);
}
