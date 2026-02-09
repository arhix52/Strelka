#ifndef STRELKA_BXDF_DIELECTRIC_H
#define STRELKA_BXDF_DIELECTRIC_H

// ============================================================================
// bxdfs/dielectric.h -- Glass / dielectric BSDF (reflection + refraction)
//
// Smooth or rough dielectric using the GGX microfacet model.
// Handles both thin-walled and solid (volumetric) modes.
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
DEVICE_FUNC BsdfSampleResult dielectric_sample(const THREAD_REF SurfaceInteraction& si,
                                               float u1, float u2, float u3)
{
    BsdfSampleResult result;

    float3 N = si.shading_normal;
    float3 V = si.wo;

    float NdotV = dot(N, V);
    bool entering = NdotV > 0.0f;

    // Flip normal to face the incoming ray
    float3 Nf = entering ? N : -N;
    float NdotV_abs = fabsf(NdotV);

    // IOR ratio: exterior / interior (supports nested dielectrics via exterior_ior)
    float eta = entering ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);

    float alpha = alpha_from_roughness(si.roughness);
    bool is_smooth = (alpha < 0.001f);

    // -- Sample microfacet half-vector (or use normal for smooth case) ------
    float3 H;
    if (is_smooth)
    {
        H = Nf;
    }
    else
    {
        float3 T, B;
        build_onb(Nf, T, B);
        float3 V_local = world_to_local(V, T, B, Nf);
        float3 H_local = ggx_vndf_sample(V_local, alpha, u1, u2);
        H = local_to_world(H_local, T, B, Nf);
    }

    float VdotH = dot(V, H);
    if (VdotH <= 0.0f)
    {
        result.bsdf_over_pdf = make_float3(0.0f);
        result.pdf           = 0.0f;
        result.event_type    = BSDF_EVENT_ABSORB;
        return result;
    }

    // -- Fresnel ------------------------------------------------------------
    float F = fresnel_dielectric(VdotH, eta);

    // -- Choose reflection or refraction ------------------------------------
    bool do_reflect = (u3 < F);

    if (do_reflect)
    {
        // Reflection
        result.wi = reflect_dir(-V, H);

        float NdotL = dot(Nf, result.wi);
        if (NdotL <= 0.0f)
        {
            result.bsdf_over_pdf = make_float3(0.0f);
            result.pdf           = 0.0f;
            result.event_type    = BSDF_EVENT_ABSORB;
            return result;
        }

        if (is_smooth)
        {
            // Delta reflection: bsdf_over_pdf = F * albedo / F = albedo
            result.bsdf_over_pdf = si.albedo;
            result.pdf           = F; // discrete: not a true density
            result.event_type    = BSDF_EVENT_SPECULAR_REFLECTION;
        }
        else
        {
            float NdotH = dot(Nf, H);
            float G2    = ggx_smith_g2(alpha, NdotV_abs, NdotL);
            float G1    = ggx_smith_g1(alpha, NdotV_abs);

            result.bsdf_over_pdf = si.albedo * (G2 / (G1 + 1e-10f));
            result.pdf           = F * ggx_vndf_pdf(alpha, NdotH, NdotV_abs, VdotH);
            result.event_type    = BSDF_EVENT_GLOSSY_REFLECTION;
        }
    }
    else
    {
        // Refraction
        float3 wi_refracted;
        bool valid = refract_dir(-V, H, eta, wi_refracted);
        if (!valid)
        {
            // Total internal reflection fallback -- should be rare if F is correct
            result.wi            = reflect_dir(-V, H);
            result.bsdf_over_pdf = si.albedo;
            result.pdf           = 1.0f;
            result.event_type    = BSDF_EVENT_SPECULAR_REFLECTION;
            return result;
        }

        result.wi = safe_normalize(wi_refracted);

        // For thin-walled surfaces, negate refraction to simulate double interface
        if (si.thin_walled)
        {
            result.wi = reflect_dir(-V, Nf);
            // Thin-walled acts like a perfect pass-through with tint
            result.wi = -V; // Continue straight through
            result.wi = safe_normalize(result.wi);
        }

        float NdotL = fabsf(dot(Nf, result.wi));
        if (NdotL <= 0.0f && !si.thin_walled)
        {
            // Check that refracted ray is on the other side
            // (for non-thin-walled, NdotL should be < 0 in face-normal sense)
        }

        if (is_smooth)
        {
            // The non-symmetry correction factor eta^2 for BTDF importance sampling
            float factor = si.thin_walled ? 1.0f : (eta * eta);
            result.bsdf_over_pdf = si.albedo * factor;
            result.pdf           = (1.0f - F); // discrete
            result.event_type    = BSDF_EVENT_SPECULAR_TRANSMISSION;
        }
        else
        {
            float NdotH  = fabsf(dot(Nf, H));
            float LdotH  = fabsf(dot(result.wi, H));
            float G2     = ggx_smith_g2(alpha, NdotV_abs, fmaxf(NdotL, 0.001f));
            float G1     = ggx_smith_g1(alpha, NdotV_abs);

            float factor = si.thin_walled ? 1.0f : (eta * eta);
            result.bsdf_over_pdf = si.albedo * factor * (G2 / (G1 + 1e-10f));

            // BTDF PDF includes the Jacobian |dH/dwi| for refraction
            float denom = (VdotH + eta * LdotH);
            float dwh_dwi = (eta * eta * LdotH) / (denom * denom + 1e-10f);
            float vndf_p  = ggx_vndf_pdf(alpha, NdotH, NdotV_abs, VdotH);
            result.pdf    = (1.0f - F) * vndf_p * fabsf(dwh_dwi);
            result.event_type = BSDF_EVENT_GLOSSY_TRANSMISSION;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Evaluate  (for rough dielectrics; smooth dielectrics are delta and return 0)
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfEvalResult dielectric_eval(const THREAD_REF SurfaceInteraction& si,
                                           float3 wi)
{
    BsdfEvalResult result;
    result.bsdf = make_float3(0.0f);
    result.pdf  = 0.0f;

    float alpha = alpha_from_roughness(si.roughness);
    if (alpha < 0.001f)
        return result; // Delta distribution -- cannot evaluate

    float3 N = si.shading_normal;
    float3 V = si.wo;

    float NdotV = dot(N, V);
    float NdotL = dot(N, wi);
    bool entering   = NdotV > 0.0f;
    float3 Nf       = entering ? N : -N;
    float NdotV_abs = fabsf(NdotV);
    float eta       = entering ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);

    bool is_reflection = (NdotL * NdotV > 0.0f); // same hemisphere

    if (is_reflection)
    {
        // Reflection lobe
        float NdotL_abs = fabsf(NdotL);
        float3 H     = safe_normalize(V + wi);
        float NdotH  = dot(Nf, H);
        float VdotH  = dot(V, H);

        if (NdotH <= 0.0f || VdotH <= 0.0f)
            return result;

        float F  = fresnel_dielectric(VdotH, eta);
        float D  = ggx_ndf(alpha, NdotH);
        float G2 = ggx_smith_g2(alpha, NdotV_abs, NdotL_abs);

        result.bsdf = si.albedo * (F * D * G2 / (4.0f * NdotV_abs * NdotL_abs + 1e-10f));
        result.pdf  = F * ggx_vndf_pdf(alpha, NdotH, NdotV_abs, VdotH);
    }
    else
    {
        // Transmission lobe
        float NdotL_abs = fabsf(NdotL);

        // Half-vector for refraction
        float3 H = safe_normalize(V + eta * wi);
        // Ensure H points to the same side as Nf
        if (dot(Nf, H) < 0.0f) H = -H;

        float NdotH = dot(Nf, H);
        float VdotH = dot(V, H);
        float LdotH = dot(wi, H);

        if (NdotH <= 0.0f || VdotH <= 0.0f)
            return result;

        float F  = fresnel_dielectric(VdotH, eta);
        float D  = ggx_ndf(alpha, NdotH);
        float G2 = ggx_smith_g2(alpha, NdotV_abs, NdotL_abs);

        float denom   = (VdotH + eta * LdotH);
        float factor  = fabsf(VdotH * LdotH) / (NdotV_abs * NdotL_abs + 1e-10f);
        float btdf    = (1.0f - F) * D * G2 * eta * eta * factor / (denom * denom + 1e-10f);
        result.bsdf   = si.albedo * fmaxf(btdf, 0.0f);

        float dwh_dwi = (eta * eta * fabsf(LdotH)) / (denom * denom + 1e-10f);
        float vndf_p  = ggx_vndf_pdf(alpha, NdotH, NdotV_abs, VdotH);
        result.pdf    = (1.0f - F) * vndf_p * dwh_dwi;
    }

    return result;
}

// ---------------------------------------------------------------------------
// PDF only
// ---------------------------------------------------------------------------
DEVICE_FUNC float dielectric_pdf(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    BsdfEvalResult r = dielectric_eval(si, wi);
    return r.pdf;
}

// ---------------------------------------------------------------------------
// Reverse PDF: p(wo | wi) -- for BDPT
// For smooth dielectric: delta -> return 0
// For rough: reflection symmetric (swap V<->wi), transmission inverts eta
// ---------------------------------------------------------------------------
DEVICE_FUNC float dielectric_pdf_reverse(const THREAD_REF SurfaceInteraction& si, float3 wo_light)
{
    float alpha = alpha_from_roughness(si.roughness);
    if (alpha < 0.001f)
        return 0.0f; // Delta distribution

    float3 N = si.shading_normal;
    float3 wi = si.wo; // "view" direction in forward path = wi for reverse
    float3 wo = wo_light;

    float NdotWi = dot(N, wi);
    float NdotWo = dot(N, wo);
    bool entering   = NdotWi > 0.0f;
    float3 Nf       = entering ? N : -N;
    float eta_fwd    = entering ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);
    float eta_rev    = 1.0f / eta_fwd; // inverted for reverse

    bool is_reflection = (NdotWo * NdotWi > 0.0f);

    if (is_reflection)
    {
        float NdotWo_abs = fabsf(NdotWo);
        float3 H     = safe_normalize(wi + wo);
        float NdotH  = dot(Nf, H);
        float WidotH = dot(wi, H);

        if (NdotH <= 0.0f || WidotH <= 0.0f)
            return 0.0f;

        float F = fresnel_dielectric(WidotH, eta_fwd);
        // VNDF PDF with wi as view direction
        return F * ggx_vndf_pdf(alpha, NdotH, NdotWo_abs, fabsf(dot(wo, H)));
    }
    else
    {
        // Transmission: use inverted eta for reverse half-vector
        float NdotWo_abs = fabsf(NdotWo);
        float3 H = safe_normalize(wi + eta_rev * wo);
        if (dot(Nf, H) < 0.0f) H = -H;

        float NdotH  = dot(Nf, H);
        float WidotH = dot(wi, H);
        float WodotH = dot(wo, H);

        if (NdotH <= 0.0f || WidotH <= 0.0f)
            return 0.0f;

        float F_rev = fresnel_dielectric(fabsf(WodotH), eta_rev);
        float denom = (fabsf(WodotH) + eta_rev * fabsf(WidotH));
        float dwh_dwo = (eta_rev * eta_rev * fabsf(WidotH)) / (denom * denom + 1e-10f);
        float vndf_p  = ggx_vndf_pdf(alpha, NdotH, NdotWo_abs, fabsf(WodotH));
        return (1.0f - F_rev) * vndf_p * dwh_dwo;
    }
}

#endif // STRELKA_BXDF_DIELECTRIC_H
