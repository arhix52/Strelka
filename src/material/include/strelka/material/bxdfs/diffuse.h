#ifndef STRELKA_BXDF_DIFFUSE_H
#define STRELKA_BXDF_DIFFUSE_H

// ============================================================================
// bxdfs/diffuse.h -- Lambertian diffuse BSDF
// ============================================================================

#include "../material_math.h"
#include "../bsdf_types.h"
#include "../surface_interaction.h"
#include "../sampling.h"

// ---------------------------------------------------------------------------
// Lambertian diffuse: f(wo, wi) = albedo / pi
// ---------------------------------------------------------------------------

DEVICE_FUNC BsdfSampleResult diffuse_sample(const THREAD_REF SurfaceInteraction& si,
                                            float u1, float u2)
{
    BsdfSampleResult result;

    // Build tangent frame from shading normal
    float3 T, B;
    build_onb(si.shading_normal, T, B);

    // Cosine-weighted hemisphere sample in local space
    const float3 wi_local = cosine_hemisphere_sample(u1, u2);

    // Transform to world space
    result.wi  = local_to_world(wi_local, T, B, si.shading_normal);
    result.pdf = cosine_hemisphere_pdf(wi_local.z);

    if (result.pdf < 1e-10f)
    {
        result.bsdf_over_pdf = make_float3(0.0f);
        result.event_type    = BSDF_EVENT_ABSORB;
        return result;
    }

    // Lambert: f = albedo / pi
    // bsdf_over_pdf = (albedo / pi) * cos_theta / pdf
    //               = (albedo / pi) * cos_theta / (cos_theta / pi)
    //               = albedo
    result.bsdf_over_pdf = si.albedo;
    result.event_type    = BSDF_EVENT_DIFFUSE_REFLECTION;

    return result;
}

DEVICE_FUNC BsdfEvalResult diffuse_eval(const THREAD_REF SurfaceInteraction& si,
                                        float3 wi)
{
    BsdfEvalResult result;

    const float cos_theta = dot(si.shading_normal, wi);
    if (cos_theta <= 0.0f)
    {
        result.bsdf = make_float3(0.0f);
        result.pdf  = 0.0f;
        return result;
    }

    result.bsdf = si.albedo * M_1_PI_F;
    result.pdf  = cosine_hemisphere_pdf(cos_theta);

    return result;
}

DEVICE_FUNC float diffuse_pdf(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    const float cos_theta = dot(si.shading_normal, wi);
    return cosine_hemisphere_pdf(fmaxf(cos_theta, 0.0f));
}

#endif // STRELKA_BXDF_DIFFUSE_H
