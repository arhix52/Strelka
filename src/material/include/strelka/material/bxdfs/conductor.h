#ifndef STRELKA_BXDF_CONDUCTOR_H
#define STRELKA_BXDF_CONDUCTOR_H

// ============================================================================
// bxdfs/conductor.h -- Metallic (conductor) GGX microfacet reflection BRDF
//
// Uses the Cook-Torrance model:
//   f(wo, wi) = D(H) * G2(wo, wi) * F(wo, H) / (4 * NdotV * NdotL)
//
// Sampling is done via VNDF (visible normal distribution function).
// ============================================================================

#include "../material_math.h"
#include "../bsdf_types.h"
#include "../surface_interaction.h"
#include "../sampling.h"
#include "../fresnel.h"
#include "../microfacet.h"

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfSampleResult conductor_sample(const THREAD_REF SurfaceInteraction& si,
                                              float u1, float u2)
{
    BsdfSampleResult result;

    const float3 N = si.shading_normal;
    const float3 V = si.wo;

    const float NdotV = dot(N, V);
    if (NdotV <= 0.0f)
    {
        result.bsdf_over_pdf = make_float3(0.0f);
        result.pdf           = 0.0f;
        result.event_type    = BSDF_EVENT_ABSORB;
        return result;
    }

    const float alpha = alpha_from_roughness(si.roughness);

    // Build local frame
    float3 T, B;
    build_onb(N, T, B);

    // Transform V to local space for VNDF sampling
    const float3 V_local = world_to_local(V, T, B, N);

    // Sample half-vector via VNDF
    const float3 H_local = ggx_vndf_sample(V_local, alpha, u1, u2);
    const float3 H = local_to_world(H_local, T, B, N);

    // Reflect
    const float VdotH = dot(V, H);
    if (VdotH <= 0.0f)
    {
        result.bsdf_over_pdf = make_float3(0.0f);
        result.pdf           = 0.0f;
        result.event_type    = BSDF_EVENT_ABSORB;
        return result;
    }
    result.wi = reflect_dir(-V, H);

    const float NdotL = dot(N, result.wi);
    if (NdotL <= 0.0f)
    {
        result.bsdf_over_pdf = make_float3(0.0f);
        result.pdf           = 0.0f;
        result.event_type    = BSDF_EVENT_ABSORB;
        return result;
    }

    const float NdotH = dot(N, H);

    // Fresnel -- for conductors, F0 = base_color (tinted metal)
    const float3 F = fresnel_schlick(si.albedo, VdotH);

    // Smith G2 (height-correlated)
    const float G2 = ggx_smith_g2(alpha, NdotV, NdotL);

    // G1 for the outgoing direction (needed for VNDF PDF cancellation)
    const float G1 = ggx_smith_g1(alpha, NdotV);

    // bsdf_over_pdf = F * G2 / G1  (VNDF sampling simplification), then the
    // multiple-scattering energy the single-scatter lobe drops. The pdf is
    // deliberately left alone: compensation rescales the BRDF, not the density
    // it was sampled from.
    const float3 ms = ggx_energy_compensation(si.albedo, si.roughness, NdotV);
    result.bsdf_over_pdf = F * (G2 / (G1 + 1e-10f)) * ms;

    // PDF in solid-angle measure
    result.pdf = ggx_vndf_pdf(alpha, NdotH, NdotV, VdotH);

    result.event_type = (alpha < BSDF_DELTA_ALPHA)
                      ? BSDF_EVENT_SPECULAR_REFLECTION
                      : BSDF_EVENT_GLOSSY_REFLECTION;

    return result;
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfEvalResult conductor_eval(const THREAD_REF SurfaceInteraction& si,
                                          float3 wi)
{
    BsdfEvalResult result;

    const float3 N = si.shading_normal;
    const float3 V = si.wo;

    const float NdotV = dot(N, V);
    const float NdotL = dot(N, wi);
    if (NdotV <= 0.0f || NdotL <= 0.0f)
    {
        result.bsdf = make_float3(0.0f);
        result.pdf  = 0.0f;
        return result;
    }

    const float alpha = alpha_from_roughness(si.roughness);

    const float3 H = safe_normalize(V + wi);
    const float NdotH = dot(N, H);
    const float VdotH = dot(V, H);

    if (NdotH <= 0.0f || VdotH <= 0.0f)
    {
        result.bsdf = make_float3(0.0f);
        result.pdf  = 0.0f;
        return result;
    }

    const float D = ggx_ndf(alpha, NdotH);
    const float G2 = ggx_smith_g2(alpha, NdotV, NdotL);
    const float3 F = fresnel_schlick(si.albedo, VdotH);

    // Cook-Torrance: D * G2 * F / (4 * NdotV * NdotL), plus the multiple
    // scattering the single-scatter lobe drops. Must match conductor_sample()
    // exactly or MIS blends two different BRDFs.
    const float3 ms = ggx_energy_compensation(si.albedo, si.roughness, NdotV);
    result.bsdf = F * (D * G2 / (4.0f * NdotV * NdotL + 1e-10f)) * ms;
    result.pdf  = ggx_vndf_pdf(alpha, NdotH, NdotV, VdotH);

    return result;
}

// ---------------------------------------------------------------------------
// PDF only
// ---------------------------------------------------------------------------
DEVICE_FUNC float conductor_pdf(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    const float3 N = si.shading_normal;
    const float3 V = si.wo;

    const float NdotV = dot(N, V);
    const float NdotL = dot(N, wi);
    if (NdotV <= 0.0f || NdotL <= 0.0f)
        return 0.0f;

    const float alpha = alpha_from_roughness(si.roughness);
    const float3 H = safe_normalize(V + wi);
    const float NdotH = dot(N, H);
    const float VdotH = dot(V, H);

    if (NdotH <= 0.0f || VdotH <= 0.0f)
        return 0.0f;

    return ggx_vndf_pdf(alpha, NdotH, NdotV, VdotH);
}

#endif // STRELKA_BXDF_CONDUCTOR_H
