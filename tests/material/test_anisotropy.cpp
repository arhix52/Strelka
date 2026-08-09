// ============================================================================
// test_anisotropy.cpp -- anisotropic GGX: the helper, and the wiring that
// is missing around it.
//
// Scene 05_anisotropy is the worst of the "geometry is right, shading is not"
// failures in the Cycles harness (rel 0.1030). The reason is not a subtle
// numeric one: anisotropy is simply not connected. anisotropic_alpha() exists
// in microfacet.h and is called by nothing; SurfaceInteraction::anisotropy is
// filled in by bsdf_init() and read by no BxDF; si.tangent / si.bitangent never
// reach a distribution function at all. Every "anisotropic" material in the
// harness therefore renders as a plain isotropic GGX.
//
// This file is in two halves:
//   * the first half pins down anisotropic_alpha() itself, which is correct
//     today and must stay correct once it is finally called;
//   * the second half is marked RED and fails on purpose. It states what the
//     BSDF is supposed to do with si.anisotropy, so that wiring anisotropy into
//     standard_pbr.h turns the suite green instead of silently changing pixels.
//
// No define of DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN here: test_bsdf.cpp owns
// main() for the whole unit_tests binary.
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

namespace
{

// A shading frame with an explicit, right-handed tangent basis:
//   N = +Y, T = +X, B = +Z, and B == cross(N, T).
// The view direction looks straight down the normal, which is what makes the
// symmetry argument in the second half of the file exact.
SurfaceInteraction make_aniso_si(float roughness, float metallic, float anisotropy)
{
    MaterialParams p = {};
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;
    p.base_color = make_float3(0.9f, 0.9f, 0.9f);
    p.roughness = roughness;
    p.metallic = metallic;
    p.ior = 1.5f;
    p.specular = 0.5f;
    p.specular_color = make_float3(1.0f);
    p.transmission = 0.0f;
    p.clearcoat = 0.0f;
    p.clearcoat_roughness = 0.0f;
    p.anisotropy = anisotropy;
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
    si.position = make_float3(0.0f, 0.0f, 0.0f);
    si.shading_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.geometry_normal = make_float3(0.0f, 1.0f, 0.0f);
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 0.0f, 1.0f);
    si.uv = make_float2(0.0f, 0.0f);
    si.wo = make_float3(0.0f, 1.0f, 0.0f);
    si.front_face = true;

    bsdf_init(si, p, nullptr);
    si.exterior_ior = 1.0f;
    return si;
}

// Direction at polar angle theta from N, leaning toward the tangent.
float3 dir_along_tangent(const SurfaceInteraction& si, float theta)
{
    return normalize(std::sin(theta) * si.tangent + std::cos(theta) * si.shading_normal);
}

// Same polar angle, leaning toward the bitangent instead. This is the tangent
// direction reflected across the N-B plane and then across the N-T plane -- i.e.
// the pair only differ in *which* tangent axis they are tilted along, so an
// isotropic lobe cannot tell them apart and an anisotropic one must.
float3 dir_along_bitangent(const SurfaceInteraction& si, float theta)
{
    return normalize(std::sin(theta) * si.bitangent + std::cos(theta) * si.shading_normal);
}

float luminance_of(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

float rel_gap(float a, float b)
{
    const float denom = std::max({ std::fabs(a), std::fabs(b), 1e-8f });
    return std::fabs(a - b) / denom;
}

} // namespace

// ===========================================================================
// Part 1 -- anisotropic_alpha() in isolation. GREEN today.
// ===========================================================================

TEST_CASE("anisotropic_alpha at anisotropy 0 degenerates to the isotropic alpha")
{
    for (float roughness : { 0.05f, 0.2f, 0.5f, 0.8f, 1.0f })
    {
        CAPTURE(roughness);
        float ax = -1.0f, ay = -1.0f;
        anisotropic_alpha(roughness, 0.0f, ax, ay);

        const float iso = alpha_from_roughness(roughness);
        CHECK(ax == doctest::Approx(iso));
        CHECK(ay == doctest::Approx(iso));
        CHECK(ax == doctest::Approx(ay));
    }
}

TEST_CASE("anisotropic_alpha separates the two axes monotonically")
{
    const float roughness = 0.4f;
    const float iso = alpha_from_roughness(roughness);

    float prev_x = iso, prev_y = iso;
    for (float aniso : { 0.2f, 0.4f, 0.6f, 0.8f, 0.9f, 0.99f })
    {
        CAPTURE(aniso);
        float ax = 0.0f, ay = 0.0f;
        anisotropic_alpha(roughness, aniso, ax, ay);

        // Positive anisotropy stretches the lobe along the tangent: alpha_x
        // grows, alpha_y shrinks. This is the convention the BSDF has to honour.
        CHECK(ax > prev_x);
        CHECK(ay < prev_y);
        CHECK(ax > iso);
        CHECK(ay < iso);
        prev_x = ax;
        prev_y = ay;
    }

    // A strongly anisotropic lobe is visibly elongated, not a rounding artefact.
    float ax = 0.0f, ay = 0.0f;
    anisotropic_alpha(roughness, 0.9f, ax, ay);
    CHECK(ax / ay > 4.0f);
}

TEST_CASE("anisotropic_alpha is energy-preserving: alpha_x * alpha_y == alpha^2")
{
    // aspect enters as a reciprocal pair (alpha/aspect, alpha*aspect), so the
    // geometric mean of the two axes is the isotropic alpha for every
    // anisotropy. Without this the lobe would gain or lose energy as the slider
    // moves, which is exactly the artefact the harness would report as a
    // brightness ratio drift on 05_anisotropy.
    for (float roughness : { 0.15f, 0.35f, 0.6f, 1.0f })
    {
        const float iso = alpha_from_roughness(roughness);
        for (float aniso : { -0.9f, -0.5f, 0.0f, 0.5f, 0.9f, 0.99f })
        {
            CAPTURE(roughness);
            CAPTURE(aniso);
            float ax = 0.0f, ay = 0.0f;
            anisotropic_alpha(roughness, aniso, ax, ay);
            CHECK(ax * ay == doctest::Approx(iso * iso).epsilon(1e-3));
            CHECK(std::sqrt(ax * ay) == doctest::Approx(iso).epsilon(1e-3));
        }
    }
}

TEST_CASE("anisotropic_alpha keeps both axes above ROUGHNESS_MIN")
{
    // The narrow axis is the one that collapses: at high anisotropy it is
    // multiplied by an aspect well below 1, and at roughness 0 it would be
    // exactly 0 and divide by zero inside ggx_ndf.
    for (float roughness : { 0.0f, 1e-6f, 0.001f, 0.01f, 0.5f, 1.0f })
    {
        for (float aniso : { 0.0f, 0.9f, 0.99f, 1.0f })
        {
            CAPTURE(roughness);
            CAPTURE(aniso);
            float ax = 0.0f, ay = 0.0f;
            anisotropic_alpha(roughness, aniso, ax, ay);
            CHECK(ax >= ROUGHNESS_MIN);
            CHECK(ay >= ROUGHNESS_MIN);
            CHECK(std::isfinite(ax));
            CHECK(std::isfinite(ay));
        }
    }
}

TEST_CASE("anisotropic_alpha flips the long axis for negative anisotropy")
{
    // The field is documented as [-1, 1]; the negative half must mirror the
    // positive half rather than clamp or produce a NaN aspect.
    const float roughness = 0.4f;
    float px = 0.0f, py = 0.0f;
    float nx = 0.0f, ny = 0.0f;
    anisotropic_alpha(roughness, 0.7f, px, py);
    anisotropic_alpha(roughness, -0.7f, nx, ny);

    CHECK(px > py);
    CHECK(nx < ny);
    CHECK(std::isfinite(nx));
    CHECK(std::isfinite(ny));
}

// ===========================================================================
// Part 2 -- the BSDF's response to si.anisotropy.
//
// *** RED: every CHECK below this line fails today. ***
//
// They stay red until anisotropy is wired into standard_pbr.h: the specular
// lobe must build a tangent-frame anisotropic GGX from anisotropic_alpha(),
// si.tangent and si.bitangent instead of calling ggx_ndf/ggx_smith_g2/
// ggx_vndf_pdf with the single isotropic alpha_from_roughness(si.roughness).
// Right now si.anisotropy is copied into the SurfaceInteraction by bsdf_init()
// and then read by nobody, so the two directions below evaluate bit-identically
// no matter what the slider says.
//
// These are CHECKs, not REQUIREs, so the run reports the missing behaviour by
// name and keeps going rather than aborting the binary.
// ===========================================================================

TEST_CASE("bsdf_init carries anisotropy into the SurfaceInteraction")
{
    // The plumbing up to the BSDF boundary does work -- this one is green, and
    // it is what makes the failures below unambiguously a BxDF-side defect
    // rather than a lost parameter.
    SurfaceInteraction si = make_aniso_si(0.25f, 1.0f, 0.9f);
    CHECK(si.anisotropy == doctest::Approx(0.9f));
}

TEST_CASE("isotropic control: tangent and bitangent directions match at anisotropy 0")
{
    // With wo == N, a direction tilted toward T and the same direction tilted
    // toward B are related by a 90-degree rotation about the normal. An
    // isotropic lobe is invariant under that rotation, so these must agree
    // exactly. GREEN today (trivially, since anisotropy is ignored), but it is
    // the control that stops a future anisotropy patch from perturbing
    // isotropic materials.
    SurfaceInteraction si = make_aniso_si(0.25f, 1.0f, 0.0f);

    for (float theta : { 0.15f, 0.35f, 0.5f, 0.8f })
    {
        CAPTURE(theta);
        BsdfEvalResult t = bsdf_eval(si, dir_along_tangent(si, theta));
        BsdfEvalResult b = bsdf_eval(si, dir_along_bitangent(si, theta));

        CHECK(luminance_of(t.bsdf) == doctest::Approx(luminance_of(b.bsdf)).epsilon(1e-4));
        CHECK(t.pdf == doctest::Approx(b.pdf).epsilon(1e-4));
    }
}

TEST_CASE("RED: anisotropy 0.9 must split the specular lobe along tangent vs bitangent")
{
    // Same two directions, same everything, only si.anisotropy differs from the
    // control above. A tangent-frame GGX with alpha_x / alpha_y about 4.8:1
    // spreads energy along T and starves it along B, so off the specular peak
    // the two evaluations must differ by far more than noise.
    SurfaceInteraction si = make_aniso_si(0.25f, 1.0f, 0.9f);
    REQUIRE(si.anisotropy == doctest::Approx(0.9f));

    for (float theta : { 0.35f, 0.5f, 0.8f })
    {
        CAPTURE(theta);
        BsdfEvalResult t = bsdf_eval(si, dir_along_tangent(si, theta));
        BsdfEvalResult b = bsdf_eval(si, dir_along_bitangent(si, theta));

        const float lt = luminance_of(t.bsdf);
        const float lb = luminance_of(b.bsdf);
        CAPTURE(lt);
        CAPTURE(lb);

        // FAILS TODAY: the two are bit-identical, so the gap is 0.
        CHECK(rel_gap(lt, lb) > 0.1f);

        // And the direction of the split is fixed by anisotropic_alpha()'s own
        // convention: alpha_x > alpha_y for positive anisotropy, so the lobe is
        // the broad one along the tangent.
        CHECK(lt > lb);
    }
}

TEST_CASE("RED: the anisotropic sampling density must follow the anisotropic lobe")
{
    // A wired-up implementation has to move the VNDF density with the
    // distribution, not just the value of f. If only the NDF is made
    // anisotropic and ggx_vndf_pdf() is left isotropic, MIS weights and the
    // sample/eval consistency check in test_bsdf.cpp go quietly wrong.
    SurfaceInteraction si = make_aniso_si(0.25f, 1.0f, 0.9f);

    BsdfEvalResult t = bsdf_eval(si, dir_along_tangent(si, 0.5f));
    BsdfEvalResult b = bsdf_eval(si, dir_along_bitangent(si, 0.5f));
    REQUIRE(t.pdf > 0.0f);
    REQUIRE(b.pdf > 0.0f);

    // FAILS TODAY: identical densities.
    CHECK(rel_gap(t.pdf, b.pdf) > 0.1f);
    CHECK(t.pdf > b.pdf);
}

TEST_CASE("RED: flipping the sign of anisotropy must swap the two axes")
{
    // +a stretched along T should be the same lobe as -a stretched along B.
    // This pins the sign convention end to end, so the tangent frame cannot be
    // wired in transposed and still pass the magnitude test above.
    SurfaceInteraction pos = make_aniso_si(0.25f, 1.0f, 0.8f);
    SurfaceInteraction neg = make_aniso_si(0.25f, 1.0f, -0.8f);

    const float theta = 0.5f;
    const float pos_t = luminance_of(bsdf_eval(pos, dir_along_tangent(pos, theta)).bsdf);
    const float pos_b = luminance_of(bsdf_eval(pos, dir_along_bitangent(pos, theta)).bsdf);
    const float neg_t = luminance_of(bsdf_eval(neg, dir_along_tangent(neg, theta)).bsdf);
    const float neg_b = luminance_of(bsdf_eval(neg, dir_along_bitangent(neg, theta)).bsdf);

    // The mirrored pair must match...
    CHECK(pos_t == doctest::Approx(neg_b).epsilon(1e-3));
    CHECK(pos_b == doctest::Approx(neg_t).epsilon(1e-3));

    // ...and must not be the degenerate "everything is equal" match that the
    // current isotropic code produces. FAILS TODAY on this line.
    CHECK(rel_gap(pos_t, pos_b) > 0.1f);
}

TEST_CASE("RED: rotating the tangent frame must rotate the specular lobe with it")
{
    // The strongest statement of the defect: si.tangent / si.bitangent are
    // never read, so today the lobe is completely insensitive to the frame.
    // Swapping T and B is a 90-degree rotation of the frame about N; with
    // anisotropy held fixed, the value along the world direction +X has to
    // follow the frame.
    SurfaceInteraction a = make_aniso_si(0.25f, 1.0f, 0.9f);

    SurfaceInteraction rotated = a;
    rotated.tangent = a.bitangent;
    rotated.bitangent = -a.tangent;

    const float3 wi = normalize(std::sin(0.5f) * make_float3(1.0f, 0.0f, 0.0f) +
                                std::cos(0.5f) * make_float3(0.0f, 1.0f, 0.0f));

    const float la = luminance_of(bsdf_eval(a, wi).bsdf);
    const float lr = luminance_of(bsdf_eval(rotated, wi).bsdf);
    CAPTURE(la);
    CAPTURE(lr);

    // FAILS TODAY: the tangent frame has no effect at all, so la == lr.
    CHECK(rel_gap(la, lr) > 0.1f);
}
