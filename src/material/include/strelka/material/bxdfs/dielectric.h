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
#include <discrete_sampling.h>

// Continuous samples are finalized from the rounded direction that is actually
// returned to the path tracer.  Near the critical angle, reconstructing the
// half vector from that endpoint is better conditioned than carrying a second,
// pre-rounding BSDF/PDF formula in the sampler.
DEVICE_FUNC BsdfEvalResult dielectric_eval(const THREAD_REF SurfaceInteraction& si, float3 wi);

DEVICE_FUNC bool dielectric_finish_continuous_sample(const THREAD_REF SurfaceInteraction& si,
                                                     float absoluteCosine,
                                                     THREAD_REF BsdfSampleResult& result)
{
    const BsdfEvalResult evaluated = dielectric_eval(si, result.wi);
    if (!(evaluated.pdf > 0.0f))
    {
        result.bsdf_over_pdf = make_float3(0.0f);
        result.pdf = 0.0f;
        result.event_type = BSDF_EVENT_ABSORB;
        return false;
    }
    result.pdf = evaluated.pdf;
    result.bsdf_over_pdf = evaluated.bsdf * (absoluteCosine / evaluated.pdf);
    return true;
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfSampleResult
dielectric_sample(const THREAD_REF SurfaceInteraction& si, float u1, float u2, unsigned int fresnelWord)
{
    BsdfSampleResult result;

    const float3 N = si.shading_normal;
    const float3 V = si.wo;

    const float NdotV = dot(N, V);
    const bool entering = NdotV > 0.0f;

    // Flip normal to face the incoming ray
    const float3 Nf = entering ? N : -N;
    const float NdotV_abs = fabsf(NdotV);

    // IOR ratio: exterior / interior (supports nested dielectrics via exterior_ior)
    const float eta = (entering || si.thin_walled) ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);

    const float alpha = alpha_from_roughness(si.roughness);
    const bool is_smooth = (alpha < BSDF_DELTA_ALPHA);

    const bool deltaTransmission = refraction_is_delta(si.ior, si.exterior_ior);

    // A thin sheet and an exactly index-matched interface split Fresnel at the
    // macroscopic interface. Their transmitted direction is exactly -V, so it
    // is a discrete atom; the reflected side remains a VNDF when rough.
    if (si.thin_walled || deltaTransmission)
    {
        const float fresnel = fresnel_dielectric(NdotV_abs, eta);
        const float proposalFresnel = discreteFloatLatticeProbability(fresnel);
        if (!discreteFloatLatticeBernoulli(fresnelWord, fresnel))
        {
            result.wi = -V;
            result.pdf = 1.0f - proposalFresnel;
            result.bsdf_over_pdf = si.albedo * ((1.0f - fresnel) / result.pdf) *
                                   (si.thin_walled ? 1.0f : eta * eta);
            result.event_type = BSDF_EVENT_SPECULAR_TRANSMISSION;
            return result;
        }

        float3 H = Nf;
        if (!is_smooth)
        {
            float3 T, B;
            build_onb(Nf, T, B);
            H = local_to_world(ggx_vndf_sample(world_to_local(V, T, B, Nf), alpha, u1, u2), T, B, Nf);
        }
        const float VdotH = dot(V, H);
        result.wi = reflect_dir(-V, H);
        const float NdotL = dot(Nf, result.wi);
        if (!(VdotH > 0.0f) || !(NdotL > 0.0f))
        {
            result.bsdf_over_pdf = make_float3(0.0f);
            result.pdf = 0.0f;
            result.event_type = BSDF_EVENT_ABSORB;
            return result;
        }
        if (is_smooth)
        {
            result.pdf = proposalFresnel;
            result.bsdf_over_pdf = si.albedo * (fresnel / result.pdf);
            result.event_type = BSDF_EVENT_SPECULAR_REFLECTION;
            return result;
        }
        if (!dielectric_finish_continuous_sample(si, NdotL, result))
            return result;
        result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
        return result;
    }

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
        const float3 V_local = world_to_local(V, T, B, Nf);
        const float3 H_local = ggx_vndf_sample(V_local, alpha, u1, u2);
        H = local_to_world(H_local, T, B, Nf);
    }

    const float VdotH = dot(V, H);
    if (VdotH <= 0.0f)
    {
        result.bsdf_over_pdf = make_float3(0.0f);
        result.pdf = 0.0f;
        result.event_type = BSDF_EVENT_ABSORB;
        return result;
    }

    // -- Fresnel ------------------------------------------------------------
    const float F = fresnel_dielectric(VdotH, eta);
    const float proposalFresnel = discreteFloatLatticeProbability(F);

    // -- Choose reflection or refraction ------------------------------------
    const bool do_reflect = discreteFloatLatticeBernoulli(fresnelWord, F);

    if (do_reflect)
    {
        // Reflection
        result.wi = reflect_dir(-V, H);

        const float NdotL = dot(Nf, result.wi);
        if (NdotL <= 0.0f)
        {
            result.bsdf_over_pdf = make_float3(0.0f);
            result.pdf = 0.0f;
            result.event_type = BSDF_EVENT_ABSORB;
            return result;
        }

        if (is_smooth)
        {
            result.pdf = proposalFresnel; // discrete: not a true density
            result.bsdf_over_pdf = si.albedo * (F / result.pdf);
            result.event_type = BSDF_EVENT_SPECULAR_REFLECTION;
        }
        else
        {
            if (!dielectric_finish_continuous_sample(si, NdotL, result))
                return result;
            result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
        }
    }
    else
    {
        // Refraction
        float3 wi_refracted;
        const bool valid = refract_dir(-V, H, eta, VdotH, wi_refracted);
        if (!valid)
        {
            // Fresnel uses the same TIR predicate, so this is only a numerical
            // fallback. Preserve the measure of the sampled half-vector: a
            // rough H is a continuous glossy reflection, not a delta event.
            result.wi = reflect_dir(-V, H);
            if (is_smooth)
            {
                result.bsdf_over_pdf = si.albedo;
                result.pdf = 1.0f;
                result.event_type = BSDF_EVENT_SPECULAR_REFLECTION;
            }
            else
            {
                const float NdotLReflection = dot(Nf, result.wi);
                if (!(NdotLReflection > 0.0f))
                {
                    result.bsdf_over_pdf = make_float3(0.0f);
                    result.pdf = 0.0f;
                    result.event_type = BSDF_EVENT_ABSORB;
                    return result;
                }
                if (!dielectric_finish_continuous_sample(si, NdotLReflection, result))
                    return result;
                result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
            }
            return result;
        }

        result.wi = safe_normalize(wi_refracted);

        const float signedNdotL = dot(Nf, result.wi);
        if (!(signedNdotL < 0.0f))
        {
            result.bsdf_over_pdf = make_float3(0.0f);
            result.pdf = 0.0f;
            result.event_type = BSDF_EVENT_ABSORB;
            return result;
        }
        const float NdotL = -signedNdotL;

        if (is_smooth)
        {
            // The non-symmetry correction factor eta^2 for BTDF importance sampling
            const float factor = eta * eta;
            result.pdf = (1.0f - proposalFresnel); // discrete
            result.bsdf_over_pdf = si.albedo * (((1.0f - F) / result.pdf) * factor);
            result.event_type = BSDF_EVENT_SPECULAR_TRANSMISSION;
        }
        else
        {
            if (!dielectric_finish_continuous_sample(si, NdotL, result))
                return result;
            result.event_type = BSDF_EVENT_GLOSSY_TRANSMISSION;
        }
    }

    return result;
}

DEVICE_FUNC BsdfSampleResult dielectric_sample(const THREAD_REF SurfaceInteraction& si, float u1, float u2, float u3)
{
    return dielectric_sample(si, u1, u2, discreteFloatLatticeWord(u3));
}

// ---------------------------------------------------------------------------
// Evaluate  (for rough dielectrics; smooth dielectrics are delta and return 0)
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfEvalResult dielectric_eval(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    BsdfEvalResult result;
    result.bsdf = make_float3(0.0f);
    result.pdf = 0.0f;

    const float alpha = alpha_from_roughness(si.roughness);
    if (alpha < BSDF_DELTA_ALPHA || si.ior == si.exterior_ior)
        return result; // Delta distribution -- cannot evaluate

    const float3 N = si.shading_normal;
    const float3 V = si.wo;

    const float NdotV = dot(N, V);
    const float NdotL = dot(N, wi);
    const bool entering = NdotV > 0.0f;
    const float3 Nf = entering ? N : -N;
    const float NdotV_abs = fabsf(NdotV);
    const float eta = (entering || si.thin_walled) ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);

    const bool deltaTransmission = !si.thin_walled && refraction_is_delta(si.ior, si.exterior_ior);

    if (!((NdotV > 0.0f) || (NdotV < 0.0f)) || !((NdotL > 0.0f) || (NdotL < 0.0f)))
        return result;
    const bool is_reflection = ((NdotL > 0.0f) == (NdotV > 0.0f));

    if (is_reflection)
    {
        // Reflection lobe
        const float NdotL_abs = fabsf(NdotL);
        const float3 H = reflection_half_vector(V, wi, Nf);
        const float NdotH = dot(Nf, H);
        const float VdotH = dot(V, H);

        if (NdotH <= 0.0f || VdotH <= 0.0f)
            return result;

        const float F = fresnel_dielectric((si.thin_walled || deltaTransmission) ? NdotV_abs : VdotH, eta);
        const float proposalFresnel = discreteFloatLatticeProbability(F);
        const float shape = saturating_nonnegative_product(F, ggx_ndf_visibility(alpha, Nf, H, NdotV_abs, NdotL_abs));
        result.bsdf = make_float3(saturating_nonnegative_product(si.albedo.x, shape),
                                  saturating_nonnegative_product(si.albedo.y, shape),
                                  saturating_nonnegative_product(si.albedo.z, shape));
        result.pdf = proposalFresnel * ggx_vndf_pdf(alpha, Nf, H, NdotV_abs, VdotH);
    }
    else
    {
        if (si.thin_walled || deltaTransmission)
            return result; // exact pass-through atom

        // Transmission lobe
        const float NdotL_abs = fabsf(NdotL);

        // eta_i * V + eta_t * wi, normalised, oriented to Nf's side -- see
        // refraction_half_vector().
        CompensatedFloat robustVdotH = compensatedSum(0.0f, 0.0f);
        const float3 H = refraction_half_vector(V, wi, eta, Nf, robustVdotH);

        const float NdotH = dot(Nf, H);
        const float VdotH = saturate(compensatedValue(robustVdotH));
        const float LdotH = dot(wi, H);

        if (!(NdotH >= 0.0f) || VdotH <= 0.0f)
            return result;

        const float F = fresnel_dielectric(robustVdotH, eta);
        const float proposalFresnel = discreteFloatLatticeProbability(F);
        const float denom = refraction_residual_length(V, wi, eta);
        const float denomSquared = denom * denom;
        if (!(denomSquared > 0.0f))
            return result;
        float scale = saturating_nonnegative_product(1.0f - F, fabsf(VdotH));
        const float etaSquared = saturating_nonnegative_product(eta, eta);
        scale = saturating_nonnegative_product(scale, saturating_nonnegative_product(4.0f, etaSquared));
        scale = saturating_nonnegative_product(scale, refraction_jacobian(V, wi, eta, LdotH));
        const float btdf = saturating_nonnegative_product(scale, ggx_ndf_visibility(alpha, Nf, H, NdotV_abs, NdotL_abs));
        result.bsdf = make_float3(saturating_nonnegative_product(si.albedo.x, btdf),
                                  saturating_nonnegative_product(si.albedo.y, btdf),
                                  saturating_nonnegative_product(si.albedo.z, btdf));

        // The same pair dielectric_sample() applies; see the note there.
        const float dwh_dwi = refraction_jacobian(V, wi, eta, LdotH);
        const float pdf_h = ggx_vndf_pdf_half(alpha, Nf, H, NdotV_abs, VdotH);
        result.pdf = saturating_nonnegative_product(
            saturating_nonnegative_product(1.0f - proposalFresnel, pdf_h), dwh_dwi);
    }

    return result;
}

// ---------------------------------------------------------------------------
// PDF only
// ---------------------------------------------------------------------------
DEVICE_FUNC float dielectric_pdf(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    const BsdfEvalResult r = dielectric_eval(si, wi);
    return r.pdf;
}

#endif // STRELKA_BXDF_DIELECTRIC_H
