#include <doctest/doctest.h>

#include <strelka/material/material_math.h>
#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_params.h>
#include <strelka/material/surface_interaction.h>
#include <strelka/material/sampling.h>
#include <strelka/material/fresnel.h>
#include <strelka/material/microfacet.h>
#include <strelka/material/valid_reflection.h>
#include <strelka/material/bsdf.h>

#include <cmath>

// ---------------------------------------------------------------------------
// A normal map that outruns the geometry.
//
// At a grazing angle the perturbed normal can end up pointing away from the
// viewer on a triangle squarely facing the camera. Nothing downstream can shade
// that: standard_pbr reads dot(N, wo) <= 0 as a dielectric exit, an opaque
// material has no transmission lobe to answer with, and the sample comes back
// BSDF_EVENT_ABSORB -- which the closest-hit program turns into an exactly black
// pixel, because it terminates above next-event estimation.
//
// The correction is Cycles': rotate the normal the smallest amount that puts
// the mirror direction back above the surface. The other half of Cycles' answer
// is that the correction is for the glossy closures only -- its diffuse closure
// keeps the map's own normal and evaluates to nothing behind the view ray --
// and the second block of cases below is that half.
// ---------------------------------------------------------------------------

namespace
{

float3 norm(float x, float y, float z)
{
    return safe_normalize(make_float3(x, y, z));
}

/// Mirror `wo` about `n`, which is what the correction exists to keep valid.
float3 reflect_about(float3 n, float3 wo)
{
    return 2.0f * dot(n, wo) * n - wo;
}

bool is_unit(float3 v)
{
    return std::fabs(length(v) - 1.0f) < 1e-4f;
}

MaterialParams rough_opaque()
{
    MaterialParams p = {};
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;
    p.base_color = make_float3(0.6f, 0.55f, 0.5f);
    p.roughness = 0.4f;
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
    return p;
}

/// A front-facing triangle seen at a grazing angle, with the shading normal
/// already corrected the way initSurfaceInteraction does it.
SurfaceInteraction corrected_hit(bool suppressDiffuse)
{
    const float3 ng = norm(0.0f, 1.0f, 0.0f);
    const float3 wo = norm(0.97f, 0.24f, 0.0f);
    const float3 tipped = norm(-0.9f, 0.35f, 0.0f);

    SurfaceInteraction si = {};
    si.position = make_float3(0.0f);
    si.geometry_normal = ng;
    si.tangent = make_float3(0.0f, 0.0f, 1.0f);
    si.bitangent = make_float3(1.0f, 0.0f, 0.0f);
    si.uv = make_float2(0.0f, 0.0f);
    si.wo = wo;
    si.front_face = true;
    si.shading_normal = ensureValidSpecularReflection(ng, wo, tipped);
    si.diffuse_faces_away = suppressDiffuse;
    bsdf_init(si, rough_opaque());
    si.shading_normal = ensureValidSpecularReflection(ng, wo, tipped);
    si.diffuse_faces_away = suppressDiffuse;
    si.exterior_ior = 1.0f;
    return si;
}

} // namespace

TEST_CASE("a normal that already reflects above the surface is not touched")
{
    // The common case, and the one that must cost nothing: almost every shading
    // point in a frame is here, and a correction applied to it would be a
    // normal map quietly flattened.
    const float3 ng = norm(0.0f, 1.0f, 0.0f);
    const float3 wo = norm(0.3f, 1.0f, 0.1f);

    for (const float3 n : { norm(0.0f, 1.0f, 0.0f), norm(0.2f, 1.0f, 0.0f), norm(-0.3f, 1.0f, 0.25f) })
    {
        const float3 fixed_n = ensureValidSpecularReflection(ng, wo, n);
        CHECK(fixed_n.x == doctest::Approx(n.x).epsilon(1e-5));
        CHECK(fixed_n.y == doctest::Approx(n.y).epsilon(1e-5));
        CHECK(fixed_n.z == doctest::Approx(n.z).epsilon(1e-5));
    }
}

TEST_CASE("a normal tipped past the viewer is brought back in front of it")
{
    // The reported defect, in one assertion: dot(N, wo) must not stay negative,
    // because that is the value standard_pbr turns into an absorbed sample.
    const float3 ng = norm(0.0f, 1.0f, 0.0f);
    const float3 wo = norm(0.98f, 0.2f, 0.0f); // grazing, as the rock was
    const float3 tipped = norm(-0.95f, 0.3f, 0.0f); // map leaning away from wo

    REQUIRE(dot(tipped, wo) < 0.0f);

    const float3 fixed_n = ensureValidSpecularReflection(ng, wo, tipped);
    CHECK(is_unit(fixed_n));
    CHECK(dot(fixed_n, wo) > 0.0f);
}

TEST_CASE("the corrected reflection clears the geometric surface")
{
    // Swept rather than sampled at one angle: the failure is a function of how
    // grazing the view is and how far the map leans, and it appears in a band
    // rather than at a point.
    const float3 ng = norm(0.0f, 1.0f, 0.0f);
    for (int vi = 1; vi <= 9; ++vi)
    {
        const float elevation = (float)vi * 0.1f;
        const float3 wo = norm(1.0f, elevation, 0.0f);
        for (int ni = -9; ni <= 9; ++ni)
        {
            const float lean = (float)ni * 0.1f;
            const float3 n = norm(lean, std::sqrt(std::fmax(0.02f, 1.0f - lean * lean)), 0.0f);
            CAPTURE(elevation);
            CAPTURE(lean);

            const float3 fixed_n = ensureValidSpecularReflection(ng, wo, n);
            REQUIRE(is_unit(fixed_n));

            const float3 r = reflect_about(fixed_n, wo);
            CHECK(dot(ng, r) >= -1e-4f);
        }
    }
}

TEST_CASE("the correction is the smallest one that works")
{
    // A correction that overshoots is a normal map replaced by the geometry,
    // which is the cheap fix this one exists to avoid: the result must stay on
    // the leaning side of the geometric normal rather than snapping onto it.
    const float3 ng = norm(0.0f, 1.0f, 0.0f);
    const float3 wo = norm(0.97f, 0.24f, 0.0f);
    const float3 tipped = norm(-0.8f, 0.6f, 0.0f);

    const float3 fixed_n = ensureValidSpecularReflection(ng, wo, tipped);

    CHECK(fixed_n.x < 0.0f);           // still leaning the way the map asked
    CHECK(fixed_n.x > tipped.x);       // just not as far
    CHECK(dot(fixed_n, ng) > dot(tipped, ng)); // moved toward the geometry, not past it
}

TEST_CASE("the result is always finite and unit length")
{
    // The degenerate corners: exactly tangent views, normals on the horizon,
    // and a normal opposite the geometry. Any NaN here reaches the display
    // buffer, and the two debug views that skip tonemapping have no guard.
    const float3 ng = norm(0.0f, 1.0f, 0.0f);
    const float3 cases[][2] = {
        { norm(1.0f, 0.0001f, 0.0f), norm(-1.0f, 0.0001f, 0.0f) },
        { norm(1.0f, 0.0f, 0.0f), norm(0.0f, 1.0f, 0.0f) },
        { norm(0.0f, 1.0f, 0.0f), norm(0.0f, -1.0f, 0.0f) },
        { norm(0.5f, 0.5f, 0.5f), norm(-0.5f, -0.5f, -0.5f) },
        { norm(0.001f, 1.0f, 0.001f), norm(1.0f, 0.0f, 0.0f) },
    };
    for (const auto& c : cases)
    {
        const float3 fixed_n = ensureValidSpecularReflection(ng, c[0], c[1]);
        CHECK(std::isfinite(fixed_n.x));
        CHECK(std::isfinite(fixed_n.y));
        CHECK(std::isfinite(fixed_n.z));
        CHECK(is_unit(fixed_n));
    }
}

// ---------------------------------------------------------------------------
// The second half: which lobes the correction is for.
// ---------------------------------------------------------------------------

TEST_CASE("a corrected hit scatters instead of absorbing")
{
    // What the user sees: the pixel is no longer black.
    const SurfaceInteraction si = corrected_hit(/*suppressDiffuse=*/true);
    REQUIRE(dot(si.shading_normal, si.wo) > 0.0f);

    const BsdfSampleResult s = bsdf_sample(si, make_float4(0.4f, 0.6f, 0.3f, 0.2f));
    CHECK(s.event_type != BSDF_EVENT_ABSORB);
    CHECK(s.pdf > 0.0f);
}

TEST_CASE("the corrected hit is lit by its highlight, not by its whole surface")
{
    // The half of Cycles' answer that the ladder pays for. Correcting the normal
    // for the diffuse lobe as well lights the entire surface rather than only
    // the highlight, and took 06_normalmap from 1.061 to 1.091 against Cycles.
    const SurfaceInteraction lit = corrected_hit(/*suppressDiffuse=*/false);
    const SurfaceInteraction specularOnly = corrected_hit(/*suppressDiffuse=*/true);

    const float3 wi = norm(0.2f, 1.0f, 0.0f);
    const BsdfEvalResult a = bsdf_eval(lit, wi);
    const BsdfEvalResult b = bsdf_eval(specularOnly, wi);

    // Both answer -- the surface is not black either way ...
    CHECK(dot(a.bsdf, a.bsdf) > 0.0f);
    // ... and the suppressed one answers with strictly less.
    CHECK(luminance(b.bsdf) < luminance(a.bsdf));
}

TEST_CASE("a hit that needed no correction is not suppressed")
{
    // The no-op guarantee for the rest of the frame: diffuse_faces_away is
    // false, which is also what zero-initialisation gives.
    SurfaceInteraction si = {};
    CHECK_FALSE(si.diffuse_faces_away);

    si.geometry_normal = norm(0.0f, 1.0f, 0.0f);
    si.shading_normal = norm(0.1f, 1.0f, 0.0f);
    si.tangent = make_float3(0.0f, 0.0f, 1.0f);
    si.bitangent = make_float3(1.0f, 0.0f, 0.0f);
    si.wo = norm(0.2f, 1.0f, 0.0f);
    si.front_face = true;
    bsdf_init(si, rough_opaque());
    si.exterior_ior = 1.0f;

    const BsdfEvalResult e = bsdf_eval(si, norm(-0.2f, 1.0f, 0.1f));
    CHECK(e.pdf > 0.0f);
    CHECK(dot(e.bsdf, e.bsdf) > 0.0f);
}
