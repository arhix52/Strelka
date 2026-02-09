#ifndef STRELKA_BXDF_STANDARD_PBR_H
#define STRELKA_BXDF_STANDARD_PBR_H

// ============================================================================
// bxdfs/standard_pbr.h -- glTF metallic-roughness PBR material
//
// Implements the standard PBR model as described in the glTF 2.0 spec with
// the following lobes:
//
//   1. Diffuse (Lambertian) -- weighted by (1 - metallic) * (1 - transmission)
//   2. Specular reflection (GGX Cook-Torrance)
//   3. Transmission (refraction through rough/smooth dielectric)
//   4. Clearcoat (additional GGX layer with fixed IOR = 1.5)
//
// Lobe selection is done stochastically: we choose one lobe proportional to
// its approximate weight and divide out the selection probability.
// ============================================================================

#include "../material_math.h"
#include "../bsdf_types.h"
#include "../surface_interaction.h"
#include "../sampling.h"
#include "../fresnel.h"
#include "../microfacet.h"

// ---------------------------------------------------------------------------
// Internal: compute lobe weights for stochastic lobe selection
// ---------------------------------------------------------------------------
struct PbrLobeWeights
{
    float diffuse;
    float specular;
    float transmission;
    float clearcoat;
    float total;
};

DEVICE_FUNC PbrLobeWeights pbr_lobe_weights(const THREAD_REF SurfaceInteraction& si)
{
    PbrLobeWeights w;

    float dielectric_weight = 1.0f - si.metallic;

    w.diffuse      = dielectric_weight * (1.0f - si.transmission) * luminance(si.albedo);
    w.diffuse      = fmaxf(w.diffuse, 0.0f);

    // For specular, use the approximate Fresnel reflectance at normal incidence
    float f0_scalar = f0_from_ior(si.ior);
    float spec_lum  = mix(f0_scalar, luminance(si.albedo), si.metallic);
    w.specular      = fmaxf(spec_lum, 0.04f); // always give specular a chance

    w.transmission  = dielectric_weight * si.transmission;
    w.transmission  = fmaxf(w.transmission, 0.0f);

    w.clearcoat     = si.clearcoat * 0.25f; // fixed F0 ~ 0.04, attenuated
    w.clearcoat     = fmaxf(w.clearcoat, 0.0f);

    w.total = w.diffuse + w.specular + w.transmission + w.clearcoat;
    if (w.total < 1e-10f)
    {
        w.total    = 1.0f;
        w.specular = 1.0f; // fallback to specular
    }

    return w;
}

// ---------------------------------------------------------------------------
// Sample
//
// u1, u2: uniform random for microfacet / hemisphere sampling
// u_lobe: uniform random for lobe selection
// u_fresnel: uniform random for dielectric reflect/refract choice
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfSampleResult standard_pbr_sample(const THREAD_REF SurfaceInteraction& si,
                                                 float u1, float u2,
                                                 float u_lobe, float u_fresnel)
{
    BsdfSampleResult result;
    result.bsdf_over_pdf = make_float3(0.0f);
    result.pdf           = 0.0f;
    result.event_type    = BSDF_EVENT_ABSORB;

    PbrLobeWeights w = pbr_lobe_weights(si);
    float inv_total  = 1.0f / w.total;

    // Normalize weights to probabilities
    float p_diffuse      = w.diffuse      * inv_total;
    float p_specular     = w.specular     * inv_total;
    float p_transmission = w.transmission * inv_total;
    // p_clearcoat = 1 - p_diffuse - p_specular - p_transmission

    float3 N = si.shading_normal;
    float3 V = si.wo;
    float NdotV = dot(N, V);
    if (NdotV <= 0.0f) return result;

    float alpha    = alpha_from_roughness(si.roughness);
    float alpha_cc = alpha_from_roughness(si.clearcoat_roughness);

    // Build tangent frame
    float3 T, B;
    build_onb(N, T, B);

    // F0 for the specular lobe (mix between dielectric F0 and base color for metals)
    float3 F0 = gltf_f0(si.ior, si.specular, si.specular_tint, si.albedo, si.metallic);

    // -----------------------------------------------------------------------
    // Lobe selection
    // -----------------------------------------------------------------------
    float cdf = p_diffuse;
    if (u_lobe < cdf)
    {
        // ===== DIFFUSE LOBE ===============================================
        float3 wi_local = cosine_hemisphere_sample(u1, u2);
        result.wi = local_to_world(wi_local, T, B, N);

        float NdotL = dot(N, result.wi);
        if (NdotL <= 0.0f) return result;

        // Evaluate all lobes for the sampled direction (MIS)
        // Diffuse contribution
        float3 f_diffuse = si.albedo * M_1_PI_F * (1.0f - si.metallic) * (1.0f - si.transmission);

        // Specular contribution
        float3 H     = safe_normalize(V + result.wi);
        float NdotH  = fmaxf(dot(N, H), 0.0f);
        float VdotH  = fmaxf(dot(V, H), 0.0f);
        float D      = ggx_ndf(alpha, NdotH);
        float G2     = ggx_smith_g2(alpha, NdotV, NdotL);
        float3 F     = fresnel_schlick(F0, VdotH);
        float3 f_spec = F * (D * G2 / (4.0f * NdotV * NdotL + 1e-10f));

        // Clearcoat contribution
        float3 f_cc   = make_float3(0.0f);
        float pdf_cc  = 0.0f;
        if (si.clearcoat > 0.0f)
        {
            float D_cc    = ggx_ndf(alpha_cc, NdotH);
            float G2_cc   = ggx_smith_g2(alpha_cc, NdotV, NdotL);
            float F_cc    = fresnel_schlick_scalar(0.04f, VdotH);
            float cc_brdf = D_cc * G2_cc * F_cc / (4.0f * NdotV * NdotL + 1e-10f);
            f_cc          = make_float3(si.clearcoat * cc_brdf);
            pdf_cc        = ggx_vndf_pdf(alpha_cc, NdotH, NdotV, VdotH);
        }

        float3 f_total = f_diffuse + f_spec + f_cc;

        // Combined PDF
        float pdf_diffuse = cosine_hemisphere_pdf(NdotL);
        float pdf_spec    = ggx_vndf_pdf(alpha, NdotH, NdotV, VdotH);
        float p_cc        = w.clearcoat * inv_total;
        float combined_pdf = p_diffuse * pdf_diffuse
                           + p_specular * pdf_spec
                           + p_cc * pdf_cc;
        // Transmission does not contribute to reflection hemisphere
        combined_pdf = fmaxf(combined_pdf, 1e-10f);

        result.bsdf_over_pdf = f_total * NdotL / combined_pdf;
        result.pdf           = combined_pdf;
        result.event_type    = BSDF_EVENT_DIFFUSE_REFLECTION;
    }
    else if (u_lobe < (cdf += p_specular))
    {
        // ===== SPECULAR LOBE ==============================================
        float3 V_local = world_to_local(V, T, B, N);
        float3 H_local = ggx_vndf_sample(V_local, alpha, u1, u2);
        float3 H       = local_to_world(H_local, T, B, N);
        float VdotH    = dot(V, H);
        if (VdotH <= 0.0f) return result;

        result.wi = reflect_dir(-V, H);
        float NdotL = dot(N, result.wi);
        if (NdotL <= 0.0f) return result;

        float NdotH = dot(N, H);

        // Evaluate all lobes
        float3 F     = fresnel_schlick(F0, VdotH);
        float D      = ggx_ndf(alpha, NdotH);
        float G2     = ggx_smith_g2(alpha, NdotV, NdotL);
        float3 f_spec = F * (D * G2 / (4.0f * NdotV * NdotL + 1e-10f));

        float3 f_diffuse = si.albedo * M_1_PI_F * (1.0f - si.metallic) * (1.0f - si.transmission);

        float3 f_cc  = make_float3(0.0f);
        float pdf_cc = 0.0f;
        if (si.clearcoat > 0.0f)
        {
            float D_cc    = ggx_ndf(alpha_cc, NdotH);
            float G2_cc   = ggx_smith_g2(alpha_cc, NdotV, NdotL);
            float F_cc    = fresnel_schlick_scalar(0.04f, VdotH);
            float cc_brdf = D_cc * G2_cc * F_cc / (4.0f * NdotV * NdotL + 1e-10f);
            f_cc          = make_float3(si.clearcoat * cc_brdf);
            pdf_cc        = ggx_vndf_pdf(alpha_cc, NdotH, NdotV, VdotH);
        }

        float3 f_total = f_diffuse + f_spec + f_cc;

        float pdf_diffuse = cosine_hemisphere_pdf(NdotL);
        float pdf_spec    = ggx_vndf_pdf(alpha, NdotH, NdotV, VdotH);
        float p_cc        = w.clearcoat * inv_total;
        float combined_pdf = p_diffuse * pdf_diffuse
                           + p_specular * pdf_spec
                           + p_cc * pdf_cc;
        combined_pdf = fmaxf(combined_pdf, 1e-10f);

        result.bsdf_over_pdf = f_total * NdotL / combined_pdf;
        result.pdf           = combined_pdf;
        result.event_type    = (alpha < 0.001f)
                             ? BSDF_EVENT_SPECULAR_REFLECTION
                             : BSDF_EVENT_GLOSSY_REFLECTION;
    }
    else if (u_lobe < (cdf += p_transmission))
    {
        // ===== TRANSMISSION LOBE ==========================================
        bool entering = NdotV > 0.0f;
        float3 Nf     = entering ? N : -N;
        float eta     = entering ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);
        bool is_smooth = (alpha < 0.001f);

        float3 H;
        if (is_smooth)
        {
            H = Nf;
        }
        else
        {
            float3 V_local = world_to_local(V, T, B, Nf);
            float3 H_local = ggx_vndf_sample(V_local, alpha, u1, u2);
            H = local_to_world(H_local, T, B, Nf);
        }

        float VdotH = dot(V, H);
        if (VdotH <= 0.0f) return result;

        float F_val = fresnel_dielectric(VdotH, eta);

        if (u_fresnel < F_val)
        {
            // Specular reflection within transmission lobe
            result.wi = reflect_dir(-V, H);
            float NdotL = dot(Nf, result.wi);
            if (NdotL <= 0.0f) return result;

            if (is_smooth)
            {
                result.bsdf_over_pdf = si.albedo;
                result.pdf           = p_transmission * F_val;
                result.event_type    = BSDF_EVENT_SPECULAR_REFLECTION;
            }
            else
            {
                float NdotH  = dot(Nf, H);
                float G2     = ggx_smith_g2(alpha, fabsf(NdotV), NdotL);
                float G1     = ggx_smith_g1(alpha, fabsf(NdotV));
                result.bsdf_over_pdf = si.albedo * (G2 / (G1 + 1e-10f));
                result.pdf   = p_transmission * F_val
                             * ggx_vndf_pdf(alpha, NdotH, fabsf(NdotV), VdotH);
                result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
            }
        }
        else
        {
            // Refraction
            float3 wi_refracted;
            bool valid = refract_dir(-V, H, eta, wi_refracted);
            if (!valid)
            {
                // Total internal reflection
                result.wi            = reflect_dir(-V, H);
                result.bsdf_over_pdf = si.albedo;
                result.pdf           = p_transmission;
                result.event_type    = BSDF_EVENT_SPECULAR_REFLECTION;
                return result;
            }

            result.wi = safe_normalize(wi_refracted);

            if (si.thin_walled)
            {
                // Thin-walled: pass through without bending
                result.wi = safe_normalize(-V);
            }

            if (is_smooth)
            {
                float factor = si.thin_walled ? 1.0f : (eta * eta);
                result.bsdf_over_pdf = si.albedo * factor;
                result.pdf           = p_transmission * (1.0f - F_val);
                result.event_type    = BSDF_EVENT_SPECULAR_TRANSMISSION;
            }
            else
            {
                float NdotH   = fabsf(dot(Nf, H));
                float NdotL   = fabsf(dot(Nf, result.wi));
                float LdotH   = fabsf(dot(result.wi, H));
                float G2      = ggx_smith_g2(alpha, fabsf(NdotV), fmaxf(NdotL, 0.001f));
                float G1      = ggx_smith_g1(alpha, fabsf(NdotV));
                float factor  = si.thin_walled ? 1.0f : (eta * eta);

                result.bsdf_over_pdf = si.albedo * factor * (G2 / (G1 + 1e-10f));

                float denom   = (VdotH + eta * LdotH);
                float dwh_dwi = (eta * eta * LdotH) / (denom * denom + 1e-10f);
                float vndf_p  = ggx_vndf_pdf(alpha, NdotH, fabsf(NdotV), VdotH);
                result.pdf    = p_transmission * (1.0f - F_val) * vndf_p * fabsf(dwh_dwi);
                result.event_type = BSDF_EVENT_GLOSSY_TRANSMISSION;
            }
        }
    }
    else
    {
        // ===== CLEARCOAT LOBE =============================================
        float3 V_local = world_to_local(V, T, B, N);
        float3 H_local = ggx_vndf_sample(V_local, alpha_cc, u1, u2);
        float3 H       = local_to_world(H_local, T, B, N);
        float VdotH    = dot(V, H);
        if (VdotH <= 0.0f) return result;

        result.wi = reflect_dir(-V, H);
        float NdotL = dot(N, result.wi);
        if (NdotL <= 0.0f) return result;

        float NdotH = dot(N, H);

        // Evaluate all lobes at this direction for proper MIS weighting
        float3 F      = fresnel_schlick(F0, VdotH);
        float D       = ggx_ndf(alpha, NdotH);
        float G2_main = ggx_smith_g2(alpha, NdotV, NdotL);
        float3 f_spec = F * (D * G2_main / (4.0f * NdotV * NdotL + 1e-10f));

        float3 f_diffuse = si.albedo * M_1_PI_F * (1.0f - si.metallic) * (1.0f - si.transmission);

        float D_cc    = ggx_ndf(alpha_cc, NdotH);
        float G2_cc   = ggx_smith_g2(alpha_cc, NdotV, NdotL);
        float F_cc    = fresnel_schlick_scalar(0.04f, VdotH);
        float cc_brdf = D_cc * G2_cc * F_cc / (4.0f * NdotV * NdotL + 1e-10f);
        float3 f_cc   = make_float3(si.clearcoat * cc_brdf);

        float3 f_total = f_diffuse + f_spec + f_cc;

        float pdf_diffuse = cosine_hemisphere_pdf(NdotL);
        float pdf_spec    = ggx_vndf_pdf(alpha, NdotH, NdotV, VdotH);
        float pdf_cc      = ggx_vndf_pdf(alpha_cc, NdotH, NdotV, VdotH);
        float p_cc        = w.clearcoat * inv_total;
        float combined_pdf = p_diffuse * pdf_diffuse
                           + p_specular * pdf_spec
                           + p_cc * pdf_cc;
        combined_pdf = fmaxf(combined_pdf, 1e-10f);

        result.bsdf_over_pdf = f_total * NdotL / combined_pdf;
        result.pdf           = combined_pdf;
        result.event_type    = (alpha_cc < 0.001f)
                             ? BSDF_EVENT_SPECULAR_REFLECTION
                             : BSDF_EVENT_GLOSSY_REFLECTION;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfEvalResult standard_pbr_eval(const THREAD_REF SurfaceInteraction& si,
                                             float3 wi)
{
    BsdfEvalResult result;
    result.bsdf = make_float3(0.0f);
    result.pdf  = 0.0f;

    float3 N = si.shading_normal;
    float3 V = si.wo;

    float NdotV = dot(N, V);
    float NdotL = dot(N, wi);

    if (NdotV <= 0.0f) return result;

    float alpha    = alpha_from_roughness(si.roughness);
    float alpha_cc = alpha_from_roughness(si.clearcoat_roughness);

    PbrLobeWeights w = pbr_lobe_weights(si);
    float inv_total  = 1.0f / w.total;
    float p_diffuse      = w.diffuse      * inv_total;
    float p_specular     = w.specular     * inv_total;
    float p_transmission = w.transmission * inv_total;
    float p_clearcoat    = w.clearcoat    * inv_total;

    float3 F0 = gltf_f0(si.ior, si.specular, si.specular_tint, si.albedo, si.metallic);

    bool is_reflection = (NdotL > 0.0f);

    if (is_reflection)
    {
        // ---- Reflection hemisphere ----------------------------------------
        float3 H     = safe_normalize(V + wi);
        float NdotH  = fmaxf(dot(N, H), 0.0f);
        float VdotH  = fmaxf(dot(V, H), 0.0f);

        if (NdotH <= 0.0f || VdotH <= 0.0f)
            return result;

        // Diffuse
        float3 f_diffuse = si.albedo * M_1_PI_F * (1.0f - si.metallic) * (1.0f - si.transmission);

        // Specular
        float3 F      = fresnel_schlick(F0, VdotH);
        float  D      = ggx_ndf(alpha, NdotH);
        float  G2     = ggx_smith_g2(alpha, NdotV, NdotL);
        float3 f_spec = F * (D * G2 / (4.0f * NdotV * NdotL + 1e-10f));

        // Clearcoat
        float3 f_cc  = make_float3(0.0f);
        float pdf_cc = 0.0f;
        if (si.clearcoat > 0.0f)
        {
            float D_cc    = ggx_ndf(alpha_cc, NdotH);
            float G2_cc   = ggx_smith_g2(alpha_cc, NdotV, NdotL);
            float F_cc    = fresnel_schlick_scalar(0.04f, VdotH);
            float cc_brdf = D_cc * G2_cc * F_cc / (4.0f * NdotV * NdotL + 1e-10f);
            f_cc          = make_float3(si.clearcoat * cc_brdf);
            pdf_cc        = ggx_vndf_pdf(alpha_cc, NdotH, NdotV, VdotH);
        }

        result.bsdf = f_diffuse + f_spec + f_cc;

        float pdf_diffuse = cosine_hemisphere_pdf(NdotL);
        float pdf_spec    = ggx_vndf_pdf(alpha, NdotH, NdotV, VdotH);
        result.pdf = p_diffuse * pdf_diffuse
                   + p_specular * pdf_spec
                   + p_clearcoat * pdf_cc;
    }
    else
    {
        // ---- Transmission hemisphere --------------------------------------
        if (alpha < 0.001f)
            return result; // Smooth transmission is delta -- cannot eval

        if (si.transmission <= 0.0f)
            return result;

        bool entering   = NdotV > 0.0f;
        float3 Nf       = entering ? N : -N;
        float NdotV_abs = fabsf(NdotV);
        float NdotL_abs = fabsf(NdotL);
        float eta       = entering ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);

        float3 H = safe_normalize(V + eta * wi);
        if (dot(Nf, H) < 0.0f) H = -H;

        float NdotH = dot(Nf, H);
        float VdotH = dot(V, H);
        float LdotH = dot(wi, H);

        if (NdotH <= 0.0f || VdotH <= 0.0f)
            return result;

        float F_val   = fresnel_dielectric(VdotH, eta);
        float D       = ggx_ndf(alpha, NdotH);
        float G2      = ggx_smith_g2(alpha, NdotV_abs, NdotL_abs);

        float denom   = (VdotH + eta * LdotH);
        float factor  = fabsf(VdotH * LdotH) / (NdotV_abs * NdotL_abs + 1e-10f);
        float btdf    = (1.0f - F_val) * D * G2 * eta * eta * factor / (denom * denom + 1e-10f);

        // Weight by transmission and dielectric fraction
        result.bsdf = si.albedo * fmaxf(btdf, 0.0f) * (1.0f - si.metallic) * si.transmission;

        float dwh_dwi = (eta * eta * fabsf(LdotH)) / (denom * denom + 1e-10f);
        float vndf_p  = ggx_vndf_pdf(alpha, NdotH, NdotV_abs, VdotH);
        result.pdf    = p_transmission * (1.0f - F_val) * vndf_p * dwh_dwi;
    }

    return result;
}

// ---------------------------------------------------------------------------
// PDF only
// ---------------------------------------------------------------------------
DEVICE_FUNC float standard_pbr_pdf(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    BsdfEvalResult r = standard_pbr_eval(si, wi);
    return r.pdf;
}

#endif // STRELKA_BXDF_STANDARD_PBR_H
