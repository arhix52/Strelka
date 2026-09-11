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

#if defined(__METAL_VERSION__)
// Surface shading never reads PreparedBsdf::volume or ::emission. Prepare only
// the derived values consumed by the lobes, then the lobes themselves. Passing
// volumes_enabled=false preserves the thin-wall/transmission-tint algebra but
// avoids constructing the interior volume; emission is accumulated by Strelka
// before BSDF preparation and does not belong in this live state either.
DEVICE_FUNC OpenPBR_PreparedBsdf openpbr_prepare_surface_at(const THREAD_REF OpenPBRParams& p,
                                                            const THREAD_REF SurfaceInteraction& si,
                                                            float3 path_throughput)
{
    const OpenPBR_ResolvedInputs in = openpbr_resolve_inputs(p, si);
    OpenPBR_VolumeDerivedProps volumeDerived;
    OpenPBR_PreparedBsdf prepared;
    openpbr_prepare_volume(in, volumeDerived, prepared, false);
    const float exteriorIor = si.exterior_ior > 0.0f ? si.exterior_ior : 1.0f;
    openpbr_prepare_lobes(in, volumeDerived, prepared, path_throughput, OPENPBR_RGB_WAVELENGTHS_NM, exteriorIor, si.wo);
    return prepared;
}
#endif

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
// provably never read. Retain the upstream prepared values and math, but only
// the three active base lobes and the reflection-only specular fields.
struct OpenPBR_BaseParams
{
    OpenPBRColor base_color;
    float base_weight;
    float base_diffuse_roughness;
    float base_metalness;
    float specular_weight;
    float specular_roughness;
    OpenPBRColor specular_color;
    float specular_roughness_anisotropy;
    float specular_ior;
    float specular_anisotropy_rotation_cos;
    float specular_anisotropy_rotation_sin;
};

DEVICE_FUNC bool openpbr_has_smooth_lobe(const THREAD_REF OpenPBR_BaseParams& p)
{
    if (p.base_weight * (1.0f - p.base_metalness) > 0.0f)
    {
        return true;
    }
    return alpha_from_roughness(p.specular_roughness) >= BSDF_DELTA_ALPHA;
}

struct OpenPBR_BaseMicrofacetDistribution
{
    half2 alpha;
    half3 tangent;
    half3 bitangent;
    half3 normal;
    half isotropic_alpha;
};

// OpenPBR colour parameters are normalized. Store the long-lived prepared
// colours as UNORM16 rather than half: this is both smaller than float3 and
// about 64x more precise near one than binary16. Packing two channels into one
// uint also makes the reduced live state explicit to the Metal compiler.
DEVICE_FUNC uint openpbr_pack_base_unorm2(float2 value)
{
    const ushort2 quantized = ushort2(round(saturate(value) * 65535.0f));
    return as_type<uint>(quantized);
}

DEVICE_FUNC float2 openpbr_unpack_base_unorm2(uint value)
{
    return float2(as_type<ushort2>(value)) * (1.0f / 65535.0f);
}

DEVICE_FUNC uint2 openpbr_pack_base_color(float3 value)
{
    return uint2(openpbr_pack_base_unorm2(value.xy), openpbr_pack_base_unorm2(float2(value.z, 0.0f)));
}

DEVICE_FUNC float3 openpbr_unpack_base_color(uint2 value)
{
    return float3(openpbr_unpack_base_unorm2(value.x), openpbr_unpack_base_unorm2(value.y).x);
}

// Lobe weights are consumed only after division by their sum: they are a
// categorical distribution, not radiometric values. Store its two cumulative
// thresholds instead of three unbounded floats. The first threshold keeps 16
// bits, the second 15 bits plus a validity bit; both are much more precise over
// [0,1] than half near one and have no reciprocal singularity at zero.
DEVICE_FUNC uint openpbr_pack_base_lobe_thresholds(float specular, float metalMms, float diffuse)
{
    const float total = specular + metalMms + diffuse;
    if (!(total > OpenPBR_FloatMin))
    {
        return 0u;
    }
    const float inverseTotal = 1.0f / total;
    const uint specularThreshold = uint(round(saturate(specular * inverseTotal) * 65535.0f));
    const uint metalThreshold = uint(round(saturate((specular + metalMms) * inverseTotal) * 32767.0f));
    return 0x80000000u | (metalThreshold << 16u) | specularThreshold;
}

DEVICE_FUNC bool openpbr_unpack_base_lobe_weights(uint packed, THREAD_REF float3& weights)
{
    if ((packed & 0x80000000u) == 0u)
    {
        weights = float3(0.0f);
        return false;
    }
    const float specular = float(packed & 0xffffu) * (1.0f / 65535.0f);
    const float specularAndMetal = max(float((packed >> 16u) & 0x7fffu) * (1.0f / 32767.0f), specular);
    weights = float3(specular, specularAndMetal - specular, 1.0f - specularAndMetal);
    return true;
}

struct OpenPBR_BaseSpecularLobe
{
    OpenPBR_BaseMicrofacetDistribution microfacet_distr;
    half eta_t_over_eta_i_for_opaque_part;
    uint2 specular_color;
    uint2 f0_for_metal;
    half dielectric_amount;
    half metal_amount;
};

struct OpenPBR_BaseMetalMmsLobe
{
    uint2 scale;
    float energy_complement_idotn;
};

struct OpenPBR_BaseDiffuseLobe
{
    uint2 diffuse_albedo;
    float diffuse_roughness;
    float cached_specular_energy_compensation;
};

struct OpenPBR_BasePreparedBsdf
{
    OpenPBR_BaseSpecularLobe specular_lobe;
    OpenPBR_BaseMetalMmsLobe metal_mms_lobe;
    OpenPBR_BaseDiffuseLobe diffuse_lobe;
    uint lobe_thresholds;
};
static_assert(sizeof(OpenPBR_PreparedBsdf) == 752, "Unexpected packed Metal OpenPBR layout");
static_assert(sizeof(OpenPBR_BaseParams) == 60, "Metal base OpenPBR inputs grew unexpectedly");
static_assert(sizeof(OpenPBR_BasePreparedBsdf) <= 128, "Metal base OpenPBR state grew unexpectedly");

DEVICE_FUNC OpenPBR_AnisotropicGGXSmithVNDFMicrofacetDistribution
openpbr_expand_base_distribution(const THREAD_REF OpenPBR_BaseMicrofacetDistribution& compact)
{
    OpenPBR_AnisotropicGGXSmithVNDFMicrofacetDistribution result;
    result.alpha = vec2(compact.alpha);
    result.basis_ff.t = vec3(compact.tangent);
    result.basis_ff.b = vec3(compact.bitangent);
    result.basis_ff.n = vec3(compact.normal);
    result.isotropic_alpha = float(compact.isotropic_alpha);
    return result;
}

DEVICE_FUNC OpenPBR_BaseSpecularLobe
openpbr_compact_base_specular(const THREAD_REF OpenPBR_ComprehensiveMicrofacetReflectionTransmissionLobe& lobe)
{
    OpenPBR_BaseSpecularLobe result;
    result.microfacet_distr.alpha = half2(lobe.microfacet_distr.alpha);
    result.microfacet_distr.tangent = half3(lobe.microfacet_distr.basis_ff.t);
    result.microfacet_distr.bitangent = half3(lobe.microfacet_distr.basis_ff.b);
    result.microfacet_distr.normal = half3(lobe.microfacet_distr.basis_ff.n);
    result.microfacet_distr.isotropic_alpha = half(lobe.microfacet_distr.isotropic_alpha);
    result.eta_t_over_eta_i_for_opaque_part = lobe.refl_trans_coeff.eta_t_over_eta_i_for_opaque_part.x;
    const vec3 specularColor = lobe.refl_trans_coeff.f82_tint_for_metal;
    result.specular_color = openpbr_pack_base_color(specularColor);
    result.f0_for_metal = openpbr_pack_base_color(lobe.refl_trans_coeff.f0_for_metal);
    const float specularMax = max(specularColor.x, max(specularColor.y, specularColor.z));
    const vec3 reflectionScale = lobe.refl_trans_coeff.scale_for_reflection_for_opaque_part;
    const float scaleMax = max(reflectionScale.x, max(reflectionScale.y, reflectionScale.z));
    result.dielectric_amount = specularMax > OpenPBR_FloatMin ? scaleMax / specularMax : 0.0f;
    result.metal_amount = lobe.refl_trans_coeff.metal_amount;
    return result;
}

DEVICE_FUNC OpenPBR_BaseMetalMmsLobe
openpbr_compact_base_metal_mms(const THREAD_REF OpenPBR_MetalMicrofacetMultipleScatteringLobe& lobe)
{
    OpenPBR_BaseMetalMmsLobe result;
    result.scale = openpbr_pack_base_color(lobe.scale);
    result.energy_complement_idotn = lobe.energy_complement_idotn;
    return result;
}

DEVICE_FUNC OpenPBR_BaseDiffuseLobe
openpbr_compact_base_diffuse(const THREAD_REF OpenPBR_EnergyConservingRoughDiffuseLobe& lobe)
{
    OpenPBR_BaseDiffuseLobe result;
    result.diffuse_albedo = openpbr_pack_base_color(lobe.diffuse_albedo);
    result.diffuse_roughness = lobe.diffuse_roughness;
    result.cached_specular_energy_compensation = lobe.cached_specular_energy_compensation;
    return result;
}

DEVICE_FUNC OpenPBR_BasePreparedBsdf openpbr_compact_base(const THREAD_REF OpenPBR_PreparedBsdf& prepared)
{
    const THREAD_REF OpenPBR_AggregateLobe& base = prepared.fuzz_lobe.coating_lobe.base_lobe;
    OpenPBR_BasePreparedBsdf result;
    result.specular_lobe = openpbr_compact_base_specular(base.specular_lobe);
    result.metal_mms_lobe = openpbr_compact_base_metal_mms(base.metal_mms_lobe);
    result.diffuse_lobe = openpbr_compact_base_diffuse(base.diffuse_lobe);
    result.lobe_thresholds = openpbr_pack_base_lobe_thresholds(base.lobe_weights[OpenPBR_SpecularLobeIndex],
                                                               base.lobe_weights[OpenPBR_MetalMMSLobeIndex],
                                                               base.lobe_weights[OpenPBR_DiffuseLobeIndex]);
    return result;
}

// Prepare the reflection-only base directly. Building OpenPBR_PreparedBsdf and
// compacting it afterwards preserves the final 168-byte value, but still makes
// the compiler materialise the 752-byte layered tree at the peak of this hot
// path. The base bucket guarantees that coat, fuzz, thin film, transmission and
// subsurface are absent, so the upstream equations reduce to these three lobes.
DEVICE_FUNC OpenPBR_BasePreparedBsdf openpbr_prepare_base_at(const THREAD_REF OpenPBR_BaseParams& p,
                                                             const THREAD_REF SurfaceInteraction& si,
                                                             float3 path_throughput)
{
    OpenPBR_BasePreparedBsdf result;

    OpenPBR_Basis basis = openpbr_make_basis(safe_normalize(si.shading_normal), si.tangent, si.bitangent);
    openpbr_apply_anisotropy_rotation(
        basis, make_float2(p.specular_anisotropy_rotation_cos, p.specular_anisotropy_rotation_sin));
    if (dot(si.wo, basis.n) < 0.0f)
    {
        openpbr_invert_basis(basis);
    }
    const float3 normalFf = basis.n;
    const float idotn = dot(si.wo, normalFf);

    const float exteriorIor = si.exterior_ior > 0.0f ? si.exterior_ior : 1.0f;
    const float relativeIor = openpbr_apply_specular_weight_to_ior(p.specular_ior / exteriorIor, p.specular_weight);
    const float specularAlpha = openpbr_square(p.specular_roughness);
    const float2 anisotropicAlpha =
        openpbr_compute_anisotropic_alpha(specularAlpha, p.specular_roughness_anisotropy, false, 1.0e-6f);

    const float dielectric = 1.0f - p.base_metalness;
    const float darkenedMetal = p.base_metalness * p.specular_weight;
    const float3 weightedBaseColor = openpbr_color_to_float3(p.base_color) * p.base_weight;
    const float3 specularColor = openpbr_color_to_float3(p.specular_color);
    float3 metalAverageFresnel = float3(1.0f);
    if (OPENPBR_GET_SPECIALIZATION_CONSTANT(EnableMetallic))
    {
        metalAverageFresnel = openpbr_metal_average_fresnel_with_f82_tint(weightedBaseColor, specularColor);
    }

    result.specular_lobe.microfacet_distr.alpha = half2(anisotropicAlpha);
    result.specular_lobe.microfacet_distr.tangent = half3(basis.t);
    result.specular_lobe.microfacet_distr.bitangent = half3(basis.b);
    result.specular_lobe.microfacet_distr.normal = half3(basis.n);
    result.specular_lobe.microfacet_distr.isotropic_alpha = half(specularAlpha);
    result.specular_lobe.eta_t_over_eta_i_for_opaque_part = relativeIor;
    result.specular_lobe.specular_color = openpbr_pack_base_color(specularColor);
    result.specular_lobe.f0_for_metal = openpbr_pack_base_color(weightedBaseColor);
    result.specular_lobe.dielectric_amount = dielectric;
    result.specular_lobe.metal_amount = darkenedMetal;

    float3 specularEstimate = specularColor * dielectric * openpbr_fresnel_rgb(float3(relativeIor), idotn, false);
    if (OPENPBR_GET_SPECIALIZATION_CONSTANT(EnableMetallic))
    {
        specularEstimate += darkenedMetal * openpbr_metal_schlick_with_f82_tint(weightedBaseColor, specularColor, idotn);
    }
    const float specularWeight = openpbr_max_component_of_throughput_weighted_color(path_throughput, specularEstimate);

    const float3 metalMmsScale = openpbr_square(metalAverageFresnel) * darkenedMetal;
    result.metal_mms_lobe.scale = openpbr_pack_base_color(metalMmsScale);
    result.metal_mms_lobe.energy_complement_idotn =
        specularAlpha < OpenPBR_MinAlphaWithVisibleEnergyLoss ?
            0.0f :
            openpbr_look_up_ideal_metal_energy_complement(specularAlpha, idotn);
    const float metalMmsWeight = specularAlpha < OpenPBR_MinAlphaWithVisibleEnergyLoss ?
                                     0.0f :
                                     openpbr_max_component_of_throughput_weighted_color(path_throughput, metalMmsScale) *
                                         result.metal_mms_lobe.energy_complement_idotn;

    const float3 diffuseAlbedo = weightedBaseColor * dielectric;
    result.diffuse_lobe.diffuse_albedo = openpbr_pack_base_color(diffuseAlbedo);
    result.diffuse_lobe.diffuse_roughness = p.base_diffuse_roughness;
    const float untintedNumerator =
        openpbr_look_up_opaque_dielectric_energy_complement(relativeIor, specularAlpha, idotn);
    const float untintedDenominator =
        openpbr_look_up_opaque_dielectric_average_energy_complement(relativeIor, specularAlpha);
    result.diffuse_lobe.cached_specular_energy_compensation =
        untintedNumerator / openpbr_clamp_average_energy_complement_above_zero(untintedDenominator);
    const float diffuseWeight = openpbr_max_component_of_throughput_weighted_color(
        path_throughput, diffuseAlbedo * result.diffuse_lobe.cached_specular_energy_compensation);
    result.lobe_thresholds = openpbr_pack_base_lobe_thresholds(specularWeight, metalMmsWeight, diffuseWeight);

    return result;
}

DEVICE_FUNC vec3 openpbr_base_reflection_coefficient(const THREAD_REF OpenPBR_BaseSpecularLobe& lobe, float idoth)
{
    const vec3 specularColor = openpbr_unpack_base_color(lobe.specular_color);
    vec3 result = specularColor * float(lobe.dielectric_amount) *
                  openpbr_fresnel_rgb(vec3(float(lobe.eta_t_over_eta_i_for_opaque_part)), idoth, false);
    if (OPENPBR_GET_SPECIALIZATION_CONSTANT(EnableMetallic))
    {
        result += float(lobe.metal_amount) * openpbr_metal_schlick_with_f82_tint(
                                                 openpbr_unpack_base_color(lobe.f0_for_metal), specularColor, idoth);
    }
    return result;
}

DEVICE_FUNC OpenPBR_DiffuseSpecular openpbr_calculate_lobe_value(const THREAD_REF OpenPBR_BaseSpecularLobe& lobe,
                                                                 float3 view_direction,
                                                                 float3 light_direction)
{
    const OpenPBR_AnisotropicGGXSmithVNDFMicrofacetDistribution distribution =
        openpbr_expand_base_distribution(lobe.microfacet_distr);
    const float3 normalFf = distribution.basis_ff.n;
    const float idotn = dot(normalFf, view_direction);
    const float odotn = dot(normalFf, light_direction);
    if (idotn * odotn <= 0.0f)
    {
        return openpbr_make_zero_diffuse_specular();
    }

    const vec3 half_vector = openpbr_fast_normalize(view_direction + light_direction);
    const float idoth = dot(view_direction, half_vector);
    const float D = openpbr_eval_ggx(distribution, half_vector, normalFf);
    const vec3 F = openpbr_base_reflection_coefficient(lobe, abs(idoth));
    const float G = openpbr_eval_smith_g2(distribution, view_direction, light_direction, idotn, odotn);
    return openpbr_make_diffuse_specular_from_specular(G * D * (1.0f / (4.0f * idotn)) * F);
}

DEVICE_FUNC float openpbr_calculate_lobe_pdf(const THREAD_REF OpenPBR_BaseSpecularLobe& lobe,
                                             float3 view_direction,
                                             float3 light_direction)
{
    const OpenPBR_AnisotropicGGXSmithVNDFMicrofacetDistribution distribution =
        openpbr_expand_base_distribution(lobe.microfacet_distr);
    const float3 normalFf = distribution.basis_ff.n;
    const float idotn = dot(normalFf, view_direction);
    const float odotn = dot(normalFf, light_direction);
    if (idotn * odotn <= 0.0f)
    {
        return 0.0f;
    }

    const vec3 half_vector = openpbr_fast_normalize(view_direction + light_direction);
    const float idoth = dot(view_direction, half_vector);
    const float odoth = dot(light_direction, half_vector);
    if (idoth * odoth < 0.0f)
    {
        return 0.0f;
    }
    const float D = openpbr_eval_ggx(distribution, half_vector, normalFf);
    const float G = openpbr_eval_smith_g1(distribution, view_direction, idotn);
    return G * D * idoth / (4.0f * odoth * idotn);
}

DEVICE_FUNC bool openpbr_sample_lobe(const THREAD_REF OpenPBR_BaseSpecularLobe& lobe,
                                     vec3 rand,
                                     vec3 view_direction,
                                     THREAD_REF vec3& light_direction,
                                     THREAD_REF OpenPBR_DiffuseSpecular& weight,
                                     THREAD_REF float& pdf,
                                     THREAD_REF OpenPBR_BsdfLobeType& sampled_type)
{
    const OpenPBR_AnisotropicGGXSmithVNDFMicrofacetDistribution distribution =
        openpbr_expand_base_distribution(lobe.microfacet_distr);
    const float3 normalFf = distribution.basis_ff.n;
    const float idotn = dot(view_direction, normalFf);
    const vec3 half_vector = openpbr_sample_ggx_smith_vndf(distribution, view_direction, normalFf, rand.xy);
    const float idoth = dot(view_direction, half_vector);
    if (idoth < 0.0f)
    {
        openpbr_clear_lobe_sampling_output(light_direction, weight, pdf, sampled_type);
        return false;
    }

    light_direction = openpbr_fast_normalize(-view_direction + half_vector * (2.0f * idoth));
    const float odotn = dot(normalFf, light_direction);
    if (idotn * odotn <= 0.0f)
    {
        openpbr_clear_lobe_sampling_output(light_direction, weight, pdf, sampled_type);
        return false;
    }

    const float odoth = dot(light_direction, half_vector);
    const vec3 F = openpbr_base_reflection_coefficient(lobe, abs(idoth));
    const float GShadowing = openpbr_eval_smith_g1(distribution, light_direction, odotn);
    weight = openpbr_make_diffuse_specular_from_specular(F * GShadowing);
    const float D = openpbr_eval_ggx(distribution, half_vector, normalFf);
    if (D <= 0.0f)
    {
        openpbr_clear_lobe_sampling_output(light_direction, weight, pdf, sampled_type);
        return false;
    }
    const float GMasking = openpbr_eval_smith_g1(distribution, view_direction, idotn);
    pdf = D * GMasking * idoth / (4.0f * odoth * idotn);
    sampled_type = OpenPBR_BsdfLobeTypeGlossy | OpenPBR_BsdfLobeTypeReflection;
    return true;
}

DEVICE_FUNC OpenPBR_MetalMicrofacetMultipleScatteringLobe
openpbr_expand_base_lobe(const THREAD_REF OpenPBR_BaseMetalMmsLobe& lobe, float3 normalFf, float specularAlpha)
{
    OpenPBR_MetalMicrofacetMultipleScatteringLobe result;
    result.normal_ff = normalFf;
    result.alpha = specularAlpha;
    result.scale = openpbr_unpack_base_color(lobe.scale);
    result.energy_complement_idotn = lobe.energy_complement_idotn;
    return result;
}

DEVICE_FUNC OpenPBR_EnergyConservingRoughDiffuseLobe openpbr_expand_base_lobe(
    const THREAD_REF OpenPBR_BaseDiffuseLobe& lobe, float3 normalFf, float specularAlpha, float relativeIor)
{
    OpenPBR_EnergyConservingRoughDiffuseLobe result;
    result.normal_ff = normalFf;
    result.diffuse_albedo = openpbr_unpack_base_color(lobe.diffuse_albedo);
    result.diffuse_roughness = lobe.diffuse_roughness;
    result.specular_alpha = specularAlpha;
    result.specular_eta_t_over_eta_i = relativeIor;
    result.cached_specular_energy_compensation = lobe.cached_specular_energy_compensation;
    return result;
}

#    define STRELKA_COMMA ,
#    define STRELKA_OPENPBR_BASE_LOBE_WRAPPERS(CompactType, FullType, ExtraDecl, ExtraArgs)                             \
        DEVICE_FUNC OpenPBR_DiffuseSpecular openpbr_calculate_lobe_value(                                               \
            const THREAD_REF CompactType& lobe, float3 viewDirection, float3 lightDirection, float3 normalFf ExtraDecl) \
        {                                                                                                               \
            const FullType expanded = openpbr_expand_base_lobe(lobe, normalFf ExtraArgs);                               \
            return openpbr_calculate_lobe_value(expanded, viewDirection, lightDirection);                               \
        }                                                                                                               \
        DEVICE_FUNC float openpbr_calculate_lobe_pdf(const THREAD_REF CompactType& lobe, float3 viewDirection,          \
                                                     float3 lightDirection, float3 normalFf ExtraDecl)                  \
        {                                                                                                               \
            const FullType expanded = openpbr_expand_base_lobe(lobe, normalFf ExtraArgs);                               \
            return openpbr_calculate_lobe_pdf(expanded, viewDirection, lightDirection);                                 \
        }                                                                                                               \
        DEVICE_FUNC bool openpbr_sample_lobe(const THREAD_REF CompactType& lobe, vec3 rand, vec3 viewDirection,         \
                                             THREAD_REF vec3& lightDirection,                                           \
                                             THREAD_REF OpenPBR_DiffuseSpecular& weight, THREAD_REF float& pdf,         \
                                             THREAD_REF OpenPBR_BsdfLobeType& sampledType, float3 normalFf ExtraDecl)   \
        {                                                                                                               \
            const FullType expanded = openpbr_expand_base_lobe(lobe, normalFf ExtraArgs);                               \
            return openpbr_sample_lobe(expanded, rand, viewDirection, lightDirection, weight, pdf, sampledType);        \
        }

STRELKA_OPENPBR_BASE_LOBE_WRAPPERS(OpenPBR_BaseMetalMmsLobe,
                                   OpenPBR_MetalMicrofacetMultipleScatteringLobe,
                                   STRELKA_COMMA float specularAlpha,
                                   STRELKA_COMMA specularAlpha)
STRELKA_OPENPBR_BASE_LOBE_WRAPPERS(OpenPBR_BaseDiffuseLobe,
                                   OpenPBR_EnergyConservingRoughDiffuseLobe,
                                   STRELKA_COMMA float specularAlpha STRELKA_COMMA float relativeIor,
                                   STRELKA_COMMA specularAlpha STRELKA_COMMA relativeIor)
#    undef STRELKA_OPENPBR_BASE_LOBE_WRAPPERS
#    undef STRELKA_COMMA

DEVICE_FUNC OpenPBR_DiffuseSpecular openpbr_base_value(const THREAD_REF OpenPBR_BasePreparedBsdf& prepared,
                                                       float3 viewDirection,
                                                       float3 wi)
{
    const float3 normalFf = float3(prepared.specular_lobe.microfacet_distr.normal);
    const float specularAlpha = float(prepared.specular_lobe.microfacet_distr.isotropic_alpha);
    const float relativeIor = float(prepared.specular_lobe.eta_t_over_eta_i_for_opaque_part);
    OpenPBR_DiffuseSpecular result = openpbr_calculate_lobe_value(prepared.specular_lobe, viewDirection, wi);
    result = openpbr_add_diffuse_specular(
        result, openpbr_calculate_lobe_value(prepared.metal_mms_lobe, viewDirection, wi, normalFf, specularAlpha));
    return openpbr_add_diffuse_specular(result, openpbr_calculate_lobe_value(prepared.diffuse_lobe, viewDirection, wi,
                                                                             normalFf, specularAlpha, relativeIor));
}

DEVICE_FUNC float openpbr_base_pdf(const THREAD_REF OpenPBR_BasePreparedBsdf& prepared, float3 viewDirection, float3 wi)
{
    float3 weights;
    if (!openpbr_unpack_base_lobe_weights(prepared.lobe_thresholds, weights))
    {
        return 0.0f;
    }
    const float3 normalFf = float3(prepared.specular_lobe.microfacet_distr.normal);
    const float specularAlpha = float(prepared.specular_lobe.microfacet_distr.isotropic_alpha);
    const float relativeIor = float(prepared.specular_lobe.eta_t_over_eta_i_for_opaque_part);
    float sum = weights.x * openpbr_calculate_lobe_pdf(prepared.specular_lobe, viewDirection, wi);
    sum += weights.y * openpbr_calculate_lobe_pdf(prepared.metal_mms_lobe, viewDirection, wi, normalFf, specularAlpha);
    sum += weights.z *
           openpbr_calculate_lobe_pdf(prepared.diffuse_lobe, viewDirection, wi, normalFf, specularAlpha, relativeIor);
    return sum;
}

DEVICE_FUNC BsdfEvalResult openpbr_bsdf_eval(const THREAD_REF OpenPBR_BasePreparedBsdf& prepared,
                                             const THREAD_REF SurfaceInteraction& si,
                                             float3 wi)
{
    BsdfEvalResult result;
    const OpenPBR_DiffuseSpecular value = openpbr_base_value(prepared, si.wo, wi);
    const float3 f_cos = value.diffuse + value.specular;
    const float cos_i = fabsf(dot(si.shading_normal, wi));
    result.bsdf = cos_i > 1e-6f ? f_cos / cos_i : float3(0.0f);
    result.pdf = openpbr_base_pdf(prepared, si.wo, wi);
    return result;
}

DEVICE_FUNC BsdfSampleResult openpbr_bsdf_sample(const THREAD_REF OpenPBR_BasePreparedBsdf& prepared,
                                                 float3 viewDirection,
                                                 float4 xi)
{
    BsdfSampleResult result;
    result.wi = float3(0.0f);
    result.pdf = 0.0f;
    result.bsdf_over_pdf = float3(0.0f);
    result.event_type = BSDF_EVENT_ABSORB;

    float3 lobeWeights;
    if (!openpbr_unpack_base_lobe_weights(prepared.lobe_thresholds, lobeWeights))
    {
        return result;
    }

    const float specularAndMetal = lobeWeights.x + lobeWeights.y;
    float selector = xi.x;
    float selectedWeight;
    uint selectedLobe;
    if (selector < lobeWeights.x)
    {
        selectedWeight = lobeWeights.x;
        selectedLobe = 0u;
    }
    else if (selector < specularAndMetal)
    {
        selector -= lobeWeights.x;
        selectedWeight = lobeWeights.y;
        selectedLobe = 1u;
    }
    else
    {
        selector -= specularAndMetal;
        selectedWeight = lobeWeights.z;
        selectedLobe = 2u;
    }
    selector /= selectedWeight;
    openpbr_clamp_remapped_random_number(selector);

    vec3 wi = float3(0.0f);
    OpenPBR_DiffuseSpecular weight = openpbr_make_zero_diffuse_specular();
    float pdf = 0.0f;
    OpenPBR_BsdfLobeType sampledType = OpenPBR_BsdfLobeTypeNone;
    const vec3 rand = float3(selector, xi.y, xi.z);
    const float3 normalFf = float3(prepared.specular_lobe.microfacet_distr.normal);
    const float specularAlpha = float(prepared.specular_lobe.microfacet_distr.isotropic_alpha);
    const float relativeIor = float(prepared.specular_lobe.eta_t_over_eta_i_for_opaque_part);
    bool valid;
    if (selectedLobe == 0u)
    {
        valid = openpbr_sample_lobe(prepared.specular_lobe, rand, viewDirection, wi, weight, pdf, sampledType);
    }
    else if (selectedLobe == 1u)
    {
        valid = openpbr_sample_lobe(
            prepared.metal_mms_lobe, rand, viewDirection, wi, weight, pdf, sampledType, normalFf, specularAlpha);
    }
    else
    {
        valid = openpbr_sample_lobe(prepared.diffuse_lobe, rand, viewDirection, wi, weight, pdf, sampledType, normalFf,
                                    specularAlpha, relativeIor);
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
                bsdfCos, openpbr_calculate_lobe_value(prepared.specular_lobe, viewDirection, wi));
            pdf += lobeWeights.x * openpbr_calculate_lobe_pdf(prepared.specular_lobe, viewDirection, wi);
        }
        if (selectedLobe != 1u)
        {
            bsdfCos = openpbr_add_diffuse_specular(
                bsdfCos,
                openpbr_calculate_lobe_value(prepared.metal_mms_lobe, viewDirection, wi, normalFf, specularAlpha));
            pdf += lobeWeights.y *
                   openpbr_calculate_lobe_pdf(prepared.metal_mms_lobe, viewDirection, wi, normalFf, specularAlpha);
        }
        if (selectedLobe != 2u)
        {
            bsdfCos = openpbr_add_diffuse_specular(
                bsdfCos, openpbr_calculate_lobe_value(
                             prepared.diffuse_lobe, viewDirection, wi, normalFf, specularAlpha, relativeIor));
            pdf += lobeWeights.z * openpbr_calculate_lobe_pdf(
                                       prepared.diffuse_lobe, viewDirection, wi, normalFf, specularAlpha, relativeIor);
        }
        weight = openpbr_scale_diffuse_specular(bsdfCos, 1.0f / pdf);
    }
    else
    {
        weight = openpbr_scale_diffuse_specular(weight, 1.0f / selectedWeight);
    }

    result.wi = wi;
    result.pdf = pdf;
    result.bsdf_over_pdf = weight.diffuse + weight.specular;
    result.event_type = openpbr_lobe_to_event(sampledType);
    return result;
}
#endif

#endif // STRELKA_MATERIAL_OPENPBR_BRIDGE_H
