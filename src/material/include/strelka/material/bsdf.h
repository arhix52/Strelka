#ifndef STRELKA_BSDF_H
#define STRELKA_BSDF_H

// ============================================================================
// bsdf.h -- Top-level BSDF dispatch
//
// Provides four entry points for the path tracer:
//   bsdf_init()   -- Resolve textures into SurfaceInteraction fields
//   bsdf_sample() -- Importance-sample an incoming direction
//   bsdf_eval()   -- Evaluate f(wo, wi) and the sampling PDF
//   bsdf_pdf()    -- Return only the PDF for a given direction pair
//
// The material_type field in SurfaceInteraction selects the appropriate BxDF.
//
// MATERIAL_TYPE_OPENPBR is deliberately absent from every switch below, and
// lands on the standard_pbr `default:` arm. Two reasons, both intentional:
//
//   - OpenPBR's parameters are not in SurfaceInteraction. They are a separate
//     OpenPBRParams block, and its evaluator wants a prepared state built once
//     per hit rather than rebuilt inside sample, eval and pdf separately. So the
//     shading kernels call openpbr/openpbr_bridge.h directly, and this header
//     stays unaware of it -- which also keeps openpbr.h's ~264 KB of lookup
//     tables out of every translation unit that only wants a Lambert lobe.
//
//   - Falling through to standard_pbr is the wanted behaviour when the OpenPBR
//     path is compiled out (see the feature bit in
//     src/render/metal/integrator_features.h): the material shades as a plain
//     PBR surface instead of going black or NaN.
// ============================================================================

#include "material_math.h"
#include "bsdf_types.h"
#include "material_params.h"
#include "surface_interaction.h"
#include "sampling.h"
#include "fresnel.h"
#include "microfacet.h"
#include "texture_sample.h"

// BxDF implementations
#include "bxdfs/diffuse.h"
#include "bxdfs/conductor.h"
#include "bxdfs/dielectric.h"
#include "bxdfs/standard_pbr.h"
#include "bxdfs/hair_chiang.h"

// ---------------------------------------------------------------------------
// bsdf_init -- Resolve material parameters and textures into the
//              SurfaceInteraction.  Call this once per hit before sampling
//              or evaluation.
//
// CUDA overload: textures is a cudaTextureObject_t* array
// CPU overload:  textures can be nullptr (stubs return white)
// Metal: caller resolves textures before calling; this function still clamps
//        and prepares derived values.
// ---------------------------------------------------------------------------

#if defined(__CUDA_ARCH__)
DEVICE_FUNC void bsdf_init(SurfaceInteraction& si, const MaterialParams& params, const cudaTextureObject_t* textures)
{
    // -- Base color --
    float4 base_tex = texture_sample_2d(textures, params.base_color_tex, si.uv);
    si.albedo = make_float3(
        params.base_color.x * base_tex.x, params.base_color.y * base_tex.y, params.base_color.z * base_tex.z);

    // -- Metallic / Roughness (glTF packs: G = roughness, B = metallic) --
    float4 mr_tex = texture_sample_2d(textures, params.metallic_roughness_tex, si.uv);
    si.roughness = fmaxf(params.roughness * mr_tex.y, 0.0001f);
    si.metallic = saturate(params.metallic * mr_tex.z);

    // -- Emission --
    float4 em_tex = texture_sample_2d(textures, params.emission_tex, si.uv);
    si.emission = make_float3(params.emission.x * em_tex.x * params.emission_strength,
                              params.emission.y * em_tex.y * params.emission_strength,
                              params.emission.z * em_tex.z * params.emission_strength);

    // -- Other parameters --
    si.ior = params.ior;
    si.transmission = params.transmission;
    si.clearcoat = params.clearcoat;
    si.clearcoat_roughness = fmaxf(params.clearcoat_roughness, 0.0001f);
    si.anisotropy = params.anisotropy;
    si.specular = params.specular;
    // White when unset. Every other field here is happy with a zero-initialised
    // MaterialParams; a zero tint is a black highlight, so this one is not.
    si.specular_color = (params.specular_color.x + params.specular_color.y + params.specular_color.z) > 0.0f ?
                            params.specular_color :
                            make_float3(1.0f);
    si.diffuse_transmission = params.diffuse_transmission;
    si.diffuse_transmission_color = params.diffuse_transmission_color;
    si.sheen = params.sheen;
    si.sheen_roughness = params.sheen_roughness;
    si.sheen_color = params.sheen_color;
    si.subsurface = params.subsurface;
    si.subsurface_radius = params.subsurface_radius;
    si.subsurface_anisotropy = params.subsurface_anisotropy;
    si.subsurface_reference = params.subsurface_reference;
    // 1.5, the extension's clear lacquer, for anything below it. A
    // zero-initialised MaterialParams would otherwise give the coat an F0 of
    // zero, and the layer would vanish without a word.
    si.clearcoat_ior = (params.clearcoat_ior >= 1.0f) ? params.clearcoat_ior : 1.5f;
    si.iridescence = params.iridescence;
    si.iridescence_ior = (params.iridescence_ior >= 1.0f) ? params.iridescence_ior : 1.3f;
    si.iridescence_thickness = params.iridescence_thickness;
    si.material_type = params.material_type;
    si.thin_walled = params.thin_walled;
    si.dielectric_priority = params.dielectric_priority;
    si.exterior_ior = 1.0f; // default: air; overridden by IOR stack
}
#elif defined(__METAL_VERSION__)
// Metal: textures are resolved externally; this overload takes no texture arg.
DEVICE_FUNC void bsdf_init(THREAD_REF SurfaceInteraction& si, const THREAD_REF MaterialParams& params)
{
    // Assume si.albedo, si.emission are pre-filled from texture sampling.
    // Clamp and finalize derived values.
    si.roughness = fmax(params.roughness, 0.0001f);
    si.metallic = saturate(params.metallic);
    si.ior = params.ior;
    si.transmission = params.transmission;
    si.clearcoat = params.clearcoat;
    si.clearcoat_roughness = fmax(params.clearcoat_roughness, 0.0001f);
    si.anisotropy = params.anisotropy;
    si.specular = params.specular;
    // White when unset. Every other field here is happy with a zero-initialised
    // MaterialParams; a zero tint is a black highlight, so this one is not.
    si.specular_color = (params.specular_color.x + params.specular_color.y + params.specular_color.z) > 0.0f ?
                            params.specular_color :
                            make_float3(1.0f);
    si.diffuse_transmission = params.diffuse_transmission;
    si.diffuse_transmission_color = params.diffuse_transmission_color;
    si.sheen = params.sheen;
    si.sheen_roughness = params.sheen_roughness;
    si.sheen_color = params.sheen_color;
    si.subsurface = params.subsurface;
    si.subsurface_radius = params.subsurface_radius;
    si.subsurface_anisotropy = params.subsurface_anisotropy;
    si.subsurface_reference = params.subsurface_reference;
    // 1.5, the extension's clear lacquer, for anything below it. A
    // zero-initialised MaterialParams would otherwise give the coat an F0 of
    // zero, and the layer would vanish without a word.
    si.clearcoat_ior = (params.clearcoat_ior >= 1.0f) ? params.clearcoat_ior : 1.5f;
    si.iridescence = params.iridescence;
    si.iridescence_ior = (params.iridescence_ior >= 1.0f) ? params.iridescence_ior : 1.3f;
    si.iridescence_thickness = params.iridescence_thickness;
    si.material_type = params.material_type;
    si.thin_walled = params.thin_walled;
    si.dielectric_priority = params.dielectric_priority;
    si.exterior_ior = 1.0f; // default: air; overridden by IOR stack
}
#else
// CPU
DEVICE_FUNC void bsdf_init(SurfaceInteraction& si, const MaterialParams& params, const void* textures = nullptr)
{
    // Texture stubs return (1,1,1,1) so factor multiplication is harmless.
    const float4 base_tex = texture_sample_2d(textures, params.base_color_tex, si.uv);
    si.albedo = make_float3(
        params.base_color.x * base_tex.x, params.base_color.y * base_tex.y, params.base_color.z * base_tex.z);

    si.roughness = fmaxf(params.roughness, 0.0001f);
    si.metallic = saturate(params.metallic);

    const float4 em_tex = texture_sample_2d(textures, params.emission_tex, si.uv);
    si.emission = make_float3(params.emission.x * em_tex.x * params.emission_strength,
                              params.emission.y * em_tex.y * params.emission_strength,
                              params.emission.z * em_tex.z * params.emission_strength);

    si.ior = params.ior;
    si.transmission = params.transmission;
    si.clearcoat = params.clearcoat;
    si.clearcoat_roughness = fmaxf(params.clearcoat_roughness, 0.0001f);
    si.anisotropy = params.anisotropy;
    si.specular = params.specular;
    // White when unset. Every other field here is happy with a zero-initialised
    // MaterialParams; a zero tint is a black highlight, so this one is not.
    si.specular_color = (params.specular_color.x + params.specular_color.y + params.specular_color.z) > 0.0f ?
                            params.specular_color :
                            make_float3(1.0f);
    si.diffuse_transmission = params.diffuse_transmission;
    si.diffuse_transmission_color = params.diffuse_transmission_color;
    si.sheen = params.sheen;
    si.sheen_roughness = params.sheen_roughness;
    si.sheen_color = params.sheen_color;
    si.subsurface = params.subsurface;
    si.subsurface_radius = params.subsurface_radius;
    si.subsurface_anisotropy = params.subsurface_anisotropy;
    si.subsurface_reference = params.subsurface_reference;
    // 1.5, the extension's clear lacquer, for anything below it. A
    // zero-initialised MaterialParams would otherwise give the coat an F0 of
    // zero, and the layer would vanish without a word.
    si.clearcoat_ior = (params.clearcoat_ior >= 1.0f) ? params.clearcoat_ior : 1.5f;
    si.iridescence = params.iridescence;
    si.iridescence_ior = (params.iridescence_ior >= 1.0f) ? params.iridescence_ior : 1.3f;
    si.iridescence_thickness = params.iridescence_thickness;
    si.material_type = params.material_type;
    si.thin_walled = params.thin_walled;
    si.dielectric_priority = params.dielectric_priority;
    si.exterior_ior = 1.0f; // default: air; overridden by IOR stack
}
#endif

// ---------------------------------------------------------------------------
// bsdf_sample -- Importance-sample the BSDF
//
// xi = float4 of uniform random numbers:
//   xi.x, xi.y  -- microfacet / hemisphere sampling
//   xi.z         -- lobe selection  (standard_pbr)
//   xi.w         -- Fresnel coin-flip (dielectric, transmission)
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfSampleResult bsdf_sample(const THREAD_REF SurfaceInteraction& si, float4 xi)
{
    switch (si.material_type)
    {
    case MATERIAL_TYPE_DIFFUSE:
        return diffuse_sample(si, xi.x, xi.y);

    case MATERIAL_TYPE_CONDUCTOR:
        return conductor_sample(si, xi.x, xi.y);

    case MATERIAL_TYPE_DIELECTRIC:
        return dielectric_sample(si, xi.x, xi.y, xi.z);

    case MATERIAL_TYPE_HAIR:
        return hair_chiang_sample(si, xi.x, xi.y, xi.z);

    case MATERIAL_TYPE_STANDARD_PBR:
    default:
        return standard_pbr_sample(si, xi.x, xi.y, xi.z, xi.w);
    }
}

// ---------------------------------------------------------------------------
// bsdf_eval -- Evaluate f(wo, wi) and return the sampling PDF
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfEvalResult bsdf_eval(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    switch (si.material_type)
    {
    case MATERIAL_TYPE_DIFFUSE:
        return diffuse_eval(si, wi);

    case MATERIAL_TYPE_CONDUCTOR:
        return conductor_eval(si, wi);

    case MATERIAL_TYPE_DIELECTRIC:
        return dielectric_eval(si, wi);

    case MATERIAL_TYPE_HAIR:
        return hair_chiang_eval(si, wi);

    case MATERIAL_TYPE_STANDARD_PBR:
    default:
        return standard_pbr_eval(si, wi);
    }
}

// ---------------------------------------------------------------------------
// bsdf_has_smooth_lobe -- is there anything here for a light connection to reach?
//
// True when the material carries at least one lobe with a density with respect
// to solid angle, so that bsdf_eval() can return a non-zero pdf and next-event
// estimation has a second strategy to be weighed against.
//
// The integrators call this to decide whether to run next-event estimation at a
// vertex. They used to ask the *sample* instead -- "did this draw come back
// non-delta" -- which makes the decision depend on a coin flip the light
// connection has nothing to do with, and loses the smooth lobe's direct light in
// proportion to how often the delta lobe wins the draw. See neeRunsAtVertex() in
// shaders/common/nee_pairing.h for the measurements.
//
// Erring towards true is free and erring towards false is not: an unnecessary
// connection is evaluated, finds bsdf_eval().pdf == 0 and is discarded, whereas
// a missing one is light that is never delivered. The transmission lobe is
// therefore admitted when either its own alpha or the thin-walled one is above
// the delta threshold.
// ---------------------------------------------------------------------------
DEVICE_FUNC bool bsdf_has_smooth_lobe(const THREAD_REF SurfaceInteraction& si)
{
    const float alpha = alpha_from_roughness(si.roughness);

    switch (si.material_type)
    {
    // Lambertian is never delta, and the Chiang lobes are driven by
    // longitudinal/azimuthal roughness that hair_chiang.h floors well above the
    // delta threshold, so a strand always has a density.
    case MATERIAL_TYPE_DIFFUSE:
    case MATERIAL_TYPE_HAIR:
        return true;

    // A single GGX lobe, smooth or not.
    case MATERIAL_TYPE_CONDUCTOR:
        return alpha >= BSDF_DELTA_ALPHA;

    case MATERIAL_TYPE_DIELECTRIC:
        if (alpha < BSDF_DELTA_ALPHA)
            return false;
        {
            const bool entering = dot(si.shading_normal, si.wo) > 0.0f;
            const float eta = (entering || si.thin_walled) ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);
            if (si.thin_walled || refraction_is_delta(si.ior, si.exterior_ior))
                return fresnel_dielectric(fabsf(dot(si.shading_normal, si.wo)), eta) > 0.0f;
            return true;
        }

    case MATERIAL_TYPE_STANDARD_PBR:
    default:
        break;
    }

    const PbrLobeWeights w = pbr_lobe_weights(si);
    const float alphaCoat = alpha_from_roughness(si.clearcoat_roughness);
    const float alphaThin = thin_glass_transmission_alpha(alpha, fmaxf(si.ior / fmaxf(si.exterior_ior, 1e-4f), 1.0f));
    float alphaX = 0.0f;
    float alphaY = 0.0f;
    anisotropic_alpha(si.roughness, si.anisotropy, alphaX, alphaY);
    const bool roughBaseSpecular = !anisotropic_ggx_is_delta(alphaX, alphaY);
    const bool roughTransmissionBase = alpha >= BSDF_DELTA_ALPHA;
    const bool entering = dot(si.shading_normal, si.wo) > 0.0f;
    const float eta = (entering || si.thin_walled) ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);
    const bool deltaTransmission = !si.thin_walled && refraction_is_delta(si.ior, si.exterior_ior);
    const float splitProbability =
        transmission_fresnel_probability(transmission_fresnel(si, fabsf(dot(si.shading_normal, si.wo)), eta));
    const bool roughTransmission = si.thin_walled ?
                                       ((roughTransmissionBase && splitProbability > 0.0f) ||
                                        (alphaThin >= BSDF_DELTA_ALPHA && splitProbability < 1.0f)) :
                                       (roughTransmissionBase && (!deltaTransmission || splitProbability > 0.0f));

    return (w.diffuse > 0.0f) || (w.diffuse_transmission > 0.0f) || (w.specular > 0.0f && roughBaseSpecular) ||
           (w.transmission > 0.0f && roughTransmission) || (w.clearcoat > 0.0f && alphaCoat >= BSDF_DELTA_ALPHA);
}

// ---------------------------------------------------------------------------
// bsdf_pdf -- Return only the PDF for a given (wo, wi) pair
// ---------------------------------------------------------------------------
DEVICE_FUNC float bsdf_pdf(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    switch (si.material_type)
    {
    case MATERIAL_TYPE_DIFFUSE:
        return diffuse_pdf(si, wi);

    case MATERIAL_TYPE_CONDUCTOR:
        return conductor_pdf(si, wi);

    case MATERIAL_TYPE_DIELECTRIC:
        return dielectric_pdf(si, wi);

    case MATERIAL_TYPE_HAIR:
        return hair_chiang_pdf(si, wi);

    case MATERIAL_TYPE_STANDARD_PBR:
    default:
        return standard_pbr_pdf(si, wi);
    }
}

#endif // STRELKA_BSDF_H
