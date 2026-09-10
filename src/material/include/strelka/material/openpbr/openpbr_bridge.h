#ifndef STRELKA_MATERIAL_OPENPBR_BRIDGE_H
#define STRELKA_MATERIAL_OPENPBR_BRIDGE_H

// ============================================================================
// openpbr_bridge.h -- Adobe's OpenPBR BSDF, in Strelka's terms
// ============================================================================
//
// Everything that knows about *both* conventions lives here, so that the two
// shading kernels can call OpenPBR the same way they call standard_pbr and the
// vendored library stays untouched.
//
// The conversion that matters, and the reason this file is not a typedef:
//
//   Adobe's openpbr_eval() returns  f * |cos(N, wi)|
//   Adobe's openpbr_sample() weight returns  f * |cos(N, wi)| / pdf
//
//   Strelka's BsdfSampleResult::bsdf_over_pdf is  f * |cos| / pdf   -- same
//   Strelka's BsdfEvalResult::bsdf         is  f                    -- NOT same
//
// (bsdf_types.h states both conventions; the pairing above was then checked
// numerically -- weight == eval/pdf holds to six digits for every sampled
// direction.) So the sample path passes the weight through untouched and the
// eval path must divide the cosine back out. Getting that wrong does not break
// anything visibly: it scales direct lighting by cos, which reads as "the new
// material is a bit dark at grazing angles" rather than as a bug, and it would
// desynchronise MIS against a pdf that is still correct.
//
// The other asymmetry is who samples textures. Metal resolves them outside the
// material library (shading_common.h owns the bindless handles), CUDA resolves
// them inside it. So this header splits the work: openpbr_apply_textures()
// folds maps into a thread-local copy of the parameters and has one overload
// per backend exactly like bsdf_init does, and openpbr_resolve_inputs() is
// texture-free and identical everywhere.

#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_math.h>
#include <strelka/material/microfacet.h>
#include <strelka/material/surface_interaction.h>
#include <strelka/material/texture_sample.h>
#include <strelka/material/openpbr/openpbr_params.h>
#include <strelka/material/openpbr/openpbr_shim.h>

// Wavelengths the three render channels stand for, in nanometres. Only
// dispersion reads them (transmission_dispersion_scale > 0); everything else is
// wavelength-agnostic. sRGB primaries.
#define OPENPBR_RGB_WAVELENGTHS_NM make_float3(600.0f, 550.0f, 450.0f)

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

DEVICE_FUNC float3 openpbr_color_to_float3(OpenPBRColor c)
{
    return make_float3(c.r, c.g, c.b);
}

DEVICE_FUNC bool openpbr_has_texture(const THREAD_REF OpenPBRParams& p, unsigned int slot)
{
    return (p.texture_mask & (1u << slot)) != 0u;
}

// OpenPBR's lobe flags -> Strelka's BsdfEventType. Both are bitmasks over the
// same three-by-two space (diffuse/glossy/specular x reflect/transmit), so this
// is a relabelling rather than a decision.
DEVICE_FUNC unsigned int openpbr_lobe_to_event(OpenPBR_BsdfLobeType lobe)
{
    const bool transmit = (lobe & OpenPBR_BsdfLobeTypeTransmission) != 0u;
    if ((lobe & OpenPBR_BsdfLobeTypeDiffuse) != 0u)
    {
        return transmit ? BSDF_EVENT_DIFFUSE_TRANSMISSION : BSDF_EVENT_DIFFUSE_REFLECTION;
    }
    if ((lobe & OpenPBR_BsdfLobeTypeSpecular) != 0u)
    {
        return transmit ? BSDF_EVENT_SPECULAR_TRANSMISSION : BSDF_EVENT_SPECULAR_REFLECTION;
    }
    if ((lobe & OpenPBR_BsdfLobeTypeGlossy) != 0u)
    {
        return transmit ? BSDF_EVENT_GLOSSY_TRANSMISSION : BSDF_EVENT_GLOSSY_REFLECTION;
    }
    return BSDF_EVENT_ABSORB;
}

// Whether the material has any lobe with a density -- i.e. whether next-event
// estimation has a second strategy to weigh against at this vertex.
//
// Same contract and the same asymmetry as bsdf_has_smooth_lobe(): erring towards
// true costs one connection that finds pdf 0 and is discarded, erring towards
// false is direct light that never arrives. So each lobe is admitted on the
// weakest test that could let it contribute.
DEVICE_FUNC bool openpbr_has_smooth_lobe(const THREAD_REF OpenPBRParams& p)
{
    const float dielectric = 1.0f - p.base_metalness;

    // Diffuse, subsurface and fuzz are never delta.
    if (p.base_weight * dielectric * (1.0f - p.transmission_weight) > 0.0f)
    {
        return true;
    }
    if (p.subsurface_weight * dielectric > 0.0f)
    {
        return true;
    }
    if (p.fuzz_weight > 0.0f)
    {
        return true;
    }

    // What is left is microfacet, and only its roughness decides.
    if (alpha_from_roughness(p.specular_roughness) >= BSDF_DELTA_ALPHA)
    {
        return true;
    }
    if (p.coat_weight > 0.0f && alpha_from_roughness(p.coat_roughness) >= BSDF_DELTA_ALPHA)
    {
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Parameters -> OpenPBR_ResolvedInputs
// ---------------------------------------------------------------------------
//
// Texture-free: whatever maps the material has must already be folded into `p`
// by openpbr_apply_textures(). The two geometry bases are built here rather than
// stored, because they are a property of the hit, not of the material.

DEVICE_FUNC OpenPBR_ResolvedInputs openpbr_resolve_inputs(const THREAD_REF OpenPBRParams& p,
                                                          const THREAD_REF SurfaceInteraction& si)
{
    OpenPBR_ResolvedInputs in = openpbr_make_default_resolved_inputs();

    in.base_weight = p.base_weight;
    in.base_color = openpbr_color_to_float3(p.base_color);
    in.base_diffuse_roughness = p.base_diffuse_roughness;
    in.base_metalness = p.base_metalness;

    in.subsurface_weight = p.subsurface_weight;
    in.subsurface_color = openpbr_color_to_float3(p.subsurface_color);
    in.subsurface_radius = p.subsurface_radius;
    in.subsurface_radius_scale = openpbr_color_to_float3(p.subsurface_radius_scale);
    in.subsurface_scatter_anisotropy = p.subsurface_scatter_anisotropy;

    in.specular_weight = p.specular_weight;
    in.specular_color = openpbr_color_to_float3(p.specular_color);
    in.specular_roughness = p.specular_roughness;
    in.specular_roughness_anisotropy = p.specular_roughness_anisotropy;
    in.specular_ior = p.specular_ior;
    in.specular_anisotropy_rotation_cos_sin =
        make_float2(p.specular_anisotropy_rotation_cos, p.specular_anisotropy_rotation_sin);

    in.coat_weight = p.coat_weight;
    in.coat_color = openpbr_color_to_float3(p.coat_color);
    in.coat_roughness = p.coat_roughness;
    in.coat_roughness_anisotropy = p.coat_roughness_anisotropy;
    in.coat_ior = p.coat_ior;
    in.coat_darkening = p.coat_darkening;
    in.coat_anisotropy_rotation_cos_sin = make_float2(p.coat_anisotropy_rotation_cos, p.coat_anisotropy_rotation_sin);

    in.fuzz_weight = p.fuzz_weight;
    in.fuzz_color = openpbr_color_to_float3(p.fuzz_color);
    in.fuzz_roughness = p.fuzz_roughness;

    in.transmission_weight = p.transmission_weight;
    in.transmission_color = openpbr_color_to_float3(p.transmission_color);
    in.transmission_depth = p.transmission_depth;
    in.transmission_scatter = openpbr_color_to_float3(p.transmission_scatter);
    in.transmission_scatter_anisotropy = p.transmission_scatter_anisotropy;
    in.transmission_dispersion_scale = p.transmission_dispersion_scale;
    in.transmission_dispersion_abbe_number = p.transmission_dispersion_abbe_number;

    in.thin_film_weight = p.thin_film_weight;
    in.thin_film_thickness = p.thin_film_thickness;
    in.thin_film_ior = p.thin_film_ior;

    in.emission_luminance = p.emission_luminance;
    in.emission_color = openpbr_color_to_float3(p.emission_color);

    in.geometry_opacity = p.geometry_opacity;
    in.geometry_thin_walled = (p.geometry_thin_walled != 0u);

    // openpbr_make_basis orthonormalises with modified Gram-Schmidt and derives
    // handedness from the bitangent, which is what a mirrored UV chart needs;
    // passing the interpolated tangent frame straight in is therefore safe even
    // when it is neither orthogonal nor normalised.
    //
    // The *normal* is the exception, and it is asserted rather than fixed up:
    // openpbr_make_basis requires |n| within 1e-6 of 1 and copies it into the
    // basis unchanged. Nothing upstream promises that tightly. OptiX builds the
    // shading normal with `normalize()` under --use_fast_math, whose reciprocal
    // square root is approximate, and both backends may then pass it through
    // ensureValidSpecularReflection(), which rotates it and does not renormalise.
    // The residue is around 1e-7 and occasionally over the bound.
    //
    // On OptiX that is not a rounding difference but a stopped render: the CUDA
    // interop maps OPENPBR_ASSERT to assert(), a failure aborts the whole launch
    // with cudaErrorAssert, and the chess set trips it within 64 samples. On
    // Metal the same input skews the basis silently. Normalising once here fixes
    // both, and costs one rsqrt per shading point.
    in.geometry_basis = openpbr_make_basis(safe_normalize(si.shading_normal), si.tangent, si.bitangent);
    in.geometry_coat_basis = in.geometry_basis;

    return in;
}

// ---------------------------------------------------------------------------
// The interior medium, without preparing the lobes
// ---------------------------------------------------------------------------
//
// What fills a closed OpenPBR surface: subsurface scattering and transmission
// blended into one homogeneous medium, which is how the specification defines
// them jointly rather than as two separate volumes.
//
// Separate from openpbr_prepare_at() because the integrator needs it where no
// shading point exists. A wavefront tracer samples the next free flight in its
// `extend` stage, which has a ray and a medium id and nothing else -- no hit, no
// basis, no view direction. Adobe's staged initialisation is built for exactly
// that case: openpbr_prepare_volume() reads only the eleven volume parameters
// (checked -- it touches no basis and no LUT), so this costs a fraction of a
// full prepare and can run per scattering event.
//
// Recomputed per event rather than carried in the path state, which is what the
// specification asks for ("recompute it at each surface interaction") and what
// MediumPathState's own note argues for: the parameters are per material and the
// table fits in cache, while carrying sigma_t, albedo and g per pixel would cost
// tens of megabytes.
DEVICE_FUNC OpenPBR_HomogeneousVolume openpbr_interior_volume(const THREAD_REF OpenPBRParams& p)
{
    OpenPBR_ResolvedInputs in = openpbr_make_default_resolved_inputs();

    in.subsurface_weight = p.subsurface_weight;
    in.subsurface_color = openpbr_color_to_float3(p.subsurface_color);
    in.subsurface_radius = p.subsurface_radius;
    in.subsurface_radius_scale = openpbr_color_to_float3(p.subsurface_radius_scale);
    in.subsurface_scatter_anisotropy = p.subsurface_scatter_anisotropy;

    in.transmission_weight = p.transmission_weight;
    in.transmission_color = openpbr_color_to_float3(p.transmission_color);
    in.transmission_depth = p.transmission_depth;
    in.transmission_scatter = openpbr_color_to_float3(p.transmission_scatter);
    in.transmission_scatter_anisotropy = p.transmission_scatter_anisotropy;

    in.geometry_thin_walled = (p.geometry_thin_walled != 0u);

    OpenPBR_VolumeDerivedProps derived;
    OpenPBR_PreparedBsdf prepared;
    // The lobes are not prepared, so only `prepared.volume` may be read: the
    // rest of that struct is uninitialised here by design.
    openpbr_prepare_volume(in, derived, prepared, true);
    return prepared.volume;
}

/// Whether a material's interior is a medium a path should enter at all.
///
/// A thin-walled surface has no interior -- the specification says to apply the
/// volume only to segments inside a non-thin-walled object -- and a material
/// with neither subsurface weight nor a transmission depth encloses nothing.
DEVICE_FUNC bool openpbr_has_interior_medium(const THREAD_REF OpenPBRParams& p)
{
    if (p.geometry_thin_walled != 0u)
    {
        return false;
    }
    return p.subsurface_weight > 0.0f || (p.transmission_weight > 0.0f && p.transmission_depth > 0.0f);
}

// ---------------------------------------------------------------------------
// Preparation
// ---------------------------------------------------------------------------
//
// Called once per shading point. Not folded into the three entry points below
// on purpose: openpbr_prepare() builds the whole lobe stack, and sample + eval +
// pdf can all run at one vertex, so preparing inside each would do that work
// three times.

DEVICE_FUNC OpenPBR_PreparedBsdf openpbr_prepare_at(const THREAD_REF OpenPBRParams& p,
                                                    const THREAD_REF SurfaceInteraction& si,
                                                    float3 path_throughput)
{
    const OpenPBR_ResolvedInputs in = openpbr_resolve_inputs(p, si);
    const float exterior_ior = (si.exterior_ior > 0.0f) ? si.exterior_ior : 1.0f;
    return openpbr_prepare(in, path_throughput, OPENPBR_RGB_WAVELENGTHS_NM, exterior_ior, si.wo);
}

// ---------------------------------------------------------------------------
// The three entry points, in Strelka's result types
// ---------------------------------------------------------------------------

// Takes no SurfaceInteraction: openpbr_prepare() already cached the view
// direction and both bases, and sampling needs nothing else from the hit.
// Passing si anyway for symmetry with openpbr_bsdf_eval() would be an unused
// parameter, which this build treats as an error.
DEVICE_FUNC BsdfSampleResult openpbr_bsdf_sample(const THREAD_REF OpenPBR_PreparedBsdf& prepared, float4 xi)
{
    BsdfSampleResult r;
    r.wi = make_float3(0.0f, 0.0f, 0.0f);
    r.pdf = 0.0f;
    r.bsdf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
    r.event_type = BSDF_EVENT_ABSORB;

    // Initialised although openpbr_sample writes all four: it only does so on
    // the paths where it succeeds, and a reader cannot tell that from here.
    // `vec3`, not float3: openpbr_sample takes it by non-const reference, and on
    // CUDA `vec3` is a distinct type from float3 (see openpbr_shim.h). It is an
    // alias for float3 on Metal and for glm::vec3 on the host, so this is the one
    // spelling that binds on all three.
    vec3 wi = make_float3(0.0f, 0.0f, 0.0f);
    OpenPBR_DiffuseSpecular weight = openpbr_make_zero_diffuse_specular();
    float pdf = 0.0f;
    OpenPBR_BsdfLobeType lobe = OpenPBR_BsdfLobeTypeNone;
    // openpbr_sample takes three uniforms; Strelka hands out four because
    // standard_pbr spends one of them choosing a lobe. The unused one is xi.w.
    openpbr_sample(prepared, make_float3(xi.x, xi.y, xi.z), wi, weight, pdf, lobe);

    // The library's own contract: a sample is only meaningful when pdf > 0.
    if (!(pdf > 0.0f))
    {
        return r;
    }

    r.wi = wi;
    r.pdf = pdf;
    // Already f * |cos| / pdf -- Strelka's convention exactly. See the header note.
    r.bsdf_over_pdf = weight.diffuse + weight.specular;
    r.event_type = openpbr_lobe_to_event(lobe);
    return r;
}

DEVICE_FUNC BsdfEvalResult openpbr_bsdf_eval(const THREAD_REF OpenPBR_PreparedBsdf& prepared,
                                             const THREAD_REF SurfaceInteraction& si,
                                             float3 wi)
{
    BsdfEvalResult r;
    r.bsdf = make_float3(0.0f, 0.0f, 0.0f);
    r.pdf = 0.0f;

    const OpenPBR_DiffuseSpecular e = openpbr_eval(prepared, wi);
    const float3 f_cos = e.diffuse + e.specular;

    // Divide the cosine back out: Strelka's BsdfEvalResult::bsdf excludes it.
    // Guarded because wi can be exactly in the tangent plane, where f is already
    // zero and the quotient would be 0/0.
    const float cos_i = fabsf(dot(si.shading_normal, wi));
    if (cos_i > 1e-6f)
    {
        r.bsdf = f_cos / cos_i;
    }
    r.pdf = openpbr_pdf(prepared, wi);
    return r;
}

DEVICE_FUNC float openpbr_bsdf_pdf(const THREAD_REF OpenPBR_PreparedBsdf& prepared, float3 wi)
{
    return openpbr_pdf(prepared, wi);
}

#if defined(__METAL_VERSION__)
// The base Metal shade queue excludes coat, fuzz, thin film, transmission and
// subsurface materials. With those lobes absent, retaining the enclosing
// Fuzz->Coat->Aggregate tree keeps hundreds of bytes live for fields that are
// provably never read. Preserve the upstream lobe implementations and their
// exact prepared values, but retain only the three active base lobes.
struct OpenPBR_BasePreparedBsdf
{
    OpenPBR_ComprehensiveMicrofacetReflectionTransmissionLobe specular_lobe;
    OpenPBR_MetalMicrofacetMultipleScatteringLobe metal_mms_lobe;
    OpenPBR_EnergyConservingRoughDiffuseLobe diffuse_lobe;
    float specular_weight;
    float metal_mms_weight;
    float diffuse_weight;
    vec3 view_direction;
};
static_assert(sizeof(OpenPBR_PreparedBsdf) == 752, "Unexpected packed Metal OpenPBR layout");
static_assert(sizeof(OpenPBR_BasePreparedBsdf) == 328, "Metal base OpenPBR state grew unexpectedly");

DEVICE_FUNC OpenPBR_BasePreparedBsdf openpbr_compact_base(const THREAD_REF OpenPBR_PreparedBsdf& prepared)
{
    const THREAD_REF OpenPBR_AggregateLobe& base = prepared.fuzz_lobe.coating_lobe.base_lobe;
    OpenPBR_BasePreparedBsdf result;
    result.specular_lobe = base.specular_lobe;
    result.metal_mms_lobe = base.metal_mms_lobe;
    result.diffuse_lobe = base.diffuse_lobe;
    result.specular_weight = base.lobe_weights[OpenPBR_SpecularLobeIndex];
    result.metal_mms_weight = base.lobe_weights[OpenPBR_MetalMMSLobeIndex];
    result.diffuse_weight = base.lobe_weights[OpenPBR_DiffuseLobeIndex];
    result.view_direction = prepared.view_direction;
    return result;
}

DEVICE_FUNC OpenPBR_DiffuseSpecular openpbr_base_value(const THREAD_REF OpenPBR_BasePreparedBsdf& prepared, float3 wi)
{
    OpenPBR_DiffuseSpecular result = openpbr_calculate_lobe_value(prepared.specular_lobe, prepared.view_direction, wi);
    result = openpbr_add_diffuse_specular(
        result, openpbr_calculate_lobe_value(prepared.metal_mms_lobe, prepared.view_direction, wi));
    return openpbr_add_diffuse_specular(
        result, openpbr_calculate_lobe_value(prepared.diffuse_lobe, prepared.view_direction, wi));
}

DEVICE_FUNC float openpbr_base_pdf(const THREAD_REF OpenPBR_BasePreparedBsdf& prepared, float3 wi)
{
    float sum =
        prepared.specular_weight * openpbr_calculate_lobe_pdf(prepared.specular_lobe, prepared.view_direction, wi);
    sum += prepared.metal_mms_weight * openpbr_calculate_lobe_pdf(prepared.metal_mms_lobe, prepared.view_direction, wi);
    sum += prepared.diffuse_weight * openpbr_calculate_lobe_pdf(prepared.diffuse_lobe, prepared.view_direction, wi);
    const float total = prepared.specular_weight + prepared.metal_mms_weight + prepared.diffuse_weight;
    return total > 0.0f ? sum / total : 0.0f;
}

DEVICE_FUNC BsdfEvalResult openpbr_bsdf_eval(const THREAD_REF OpenPBR_BasePreparedBsdf& prepared,
                                             const THREAD_REF SurfaceInteraction& si,
                                             float3 wi)
{
    BsdfEvalResult result;
    const OpenPBR_DiffuseSpecular value = openpbr_base_value(prepared, wi);
    const float3 f_cos = value.diffuse + value.specular;
    const float cos_i = fabsf(dot(si.shading_normal, wi));
    result.bsdf = cos_i > 1e-6f ? f_cos / cos_i : float3(0.0f);
    result.pdf = openpbr_base_pdf(prepared, wi);
    return result;
}

DEVICE_FUNC BsdfSampleResult openpbr_bsdf_sample(const THREAD_REF OpenPBR_BasePreparedBsdf& prepared, float4 xi)
{
    BsdfSampleResult result;
    result.wi = float3(0.0f);
    result.pdf = 0.0f;
    result.bsdf_over_pdf = float3(0.0f);
    result.event_type = BSDF_EVENT_ABSORB;

    const float total = prepared.specular_weight + prepared.metal_mms_weight + prepared.diffuse_weight;
    if (!(total > OpenPBR_FloatMin))
    {
        return result;
    }

    float selector = xi.x * total;
    float selectedWeight;
    uint selectedLobe;
    if (selector < prepared.specular_weight)
    {
        selectedWeight = prepared.specular_weight;
        selectedLobe = 0u;
    }
    else if (selector < prepared.specular_weight + prepared.metal_mms_weight)
    {
        selector -= prepared.specular_weight;
        selectedWeight = prepared.metal_mms_weight;
        selectedLobe = 1u;
    }
    else
    {
        selector -= prepared.specular_weight + prepared.metal_mms_weight;
        selectedWeight = prepared.diffuse_weight;
        selectedLobe = 2u;
    }
    selector /= selectedWeight;
    openpbr_clamp_remapped_random_number(selector);

    vec3 wi = float3(0.0f);
    OpenPBR_DiffuseSpecular weight = openpbr_make_zero_diffuse_specular();
    float pdf = 0.0f;
    OpenPBR_BsdfLobeType sampledType = OpenPBR_BsdfLobeTypeNone;
    const vec3 rand = float3(selector, xi.y, xi.z);
    bool valid;
    if (selectedLobe == 0u)
    {
        valid = openpbr_sample_lobe(prepared.specular_lobe, rand, prepared.view_direction, wi, weight, pdf, sampledType);
    }
    else if (selectedLobe == 1u)
    {
        valid = openpbr_sample_lobe(prepared.metal_mms_lobe, rand, prepared.view_direction, wi, weight, pdf, sampledType);
    }
    else
    {
        valid = openpbr_sample_lobe(prepared.diffuse_lobe, rand, prepared.view_direction, wi, weight, pdf, sampledType);
    }
    if (!valid)
    {
        return result;
    }

    if (!bool(sampledType & OpenPBR_BsdfLobeTypeSpecular))
    {
        OpenPBR_DiffuseSpecular bsdfCos = openpbr_scale_diffuse_specular(weight, pdf);
        pdf *= selectedWeight;
        if (selectedLobe != 0u)
        {
            bsdfCos = openpbr_add_diffuse_specular(
                bsdfCos, openpbr_calculate_lobe_value(prepared.specular_lobe, prepared.view_direction, wi));
            pdf += prepared.specular_weight *
                   openpbr_calculate_lobe_pdf(prepared.specular_lobe, prepared.view_direction, wi);
        }
        if (selectedLobe != 1u)
        {
            bsdfCos = openpbr_add_diffuse_specular(
                bsdfCos, openpbr_calculate_lobe_value(prepared.metal_mms_lobe, prepared.view_direction, wi));
            pdf += prepared.metal_mms_weight *
                   openpbr_calculate_lobe_pdf(prepared.metal_mms_lobe, prepared.view_direction, wi);
        }
        if (selectedLobe != 2u)
        {
            bsdfCos = openpbr_add_diffuse_specular(
                bsdfCos, openpbr_calculate_lobe_value(prepared.diffuse_lobe, prepared.view_direction, wi));
            pdf +=
                prepared.diffuse_weight * openpbr_calculate_lobe_pdf(prepared.diffuse_lobe, prepared.view_direction, wi);
        }
        pdf /= total;
        weight = openpbr_scale_diffuse_specular(bsdfCos, 1.0f / pdf);
    }
    else
    {
        weight = openpbr_scale_diffuse_specular(weight, total / selectedWeight);
    }

    result.wi = wi;
    result.pdf = pdf;
    result.bsdf_over_pdf = weight.diffuse + weight.specular;
    result.event_type = openpbr_lobe_to_event(sampledType);
    return result;
}
#endif

#endif // STRELKA_MATERIAL_OPENPBR_BRIDGE_H
