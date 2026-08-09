#ifndef STRELKA_FRESNEL_H
#define STRELKA_FRESNEL_H

// ============================================================================
// fresnel.h -- Fresnel reflectance functions
// ============================================================================

#include "material_math.h"

// ---------------------------------------------------------------------------
// Schlick approximation for conductor Fresnel
//   F0 = reflectance at normal incidence
//   cos_theta = dot(N, V) -- clamped to [0, 1]
// ---------------------------------------------------------------------------
DEVICE_FUNC float3 fresnel_schlick(float3 F0, float cos_theta)
{
    float t  = 1.0f - saturate(cos_theta);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return F0 + (make_float3(1.0f) - F0) * t5;
}

// ---------------------------------------------------------------------------
// Schlick with a roughness-dependent maximum (used by some PBR models
// to attenuate Fresnel at grazing angles for rough surfaces)
//   F0   = reflectance at normal incidence
//   F90  = reflectance at grazing angle (often 1.0)
// ---------------------------------------------------------------------------
DEVICE_FUNC float3 fresnel_schlick_roughness(float3 F0, float cos_theta, float roughness)
{
    float t  = 1.0f - saturate(cos_theta);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    float3 F_max = make_float3(fmaxf(1.0f - roughness, F0.x),
                               fmaxf(1.0f - roughness, F0.y),
                               fmaxf(1.0f - roughness, F0.z));
    return F0 + (F_max - F0) * t5;
}

// ---------------------------------------------------------------------------
// Scalar Schlick (for clearcoat, etc.)
// ---------------------------------------------------------------------------
DEVICE_FUNC float fresnel_schlick_scalar(float F0, float cos_theta)
{
    float t  = 1.0f - saturate(cos_theta);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return F0 + (1.0f - F0) * t5;
}

// ---------------------------------------------------------------------------
// Exact dielectric Fresnel reflectance
//   cos_theta_i = cosine of incident angle (positive = same side as normal)
//   eta         = ratio of IORs (exterior / interior)
//   Returns reflectance in [0, 1].  Total internal reflection returns 1.
// ---------------------------------------------------------------------------
DEVICE_FUNC float fresnel_dielectric(float cos_theta_i, float eta)
{
    // Ensure cos_theta_i is positive (flip if needed)
    if (cos_theta_i < 0.0f)
    {
        eta = 1.0f / eta;
        cos_theta_i = -cos_theta_i;
    }

    float sin2_t = eta * eta * (1.0f - cos_theta_i * cos_theta_i);
    if (sin2_t > 1.0f)
        return 1.0f; // total internal reflection

    float cos_theta_t = sqrtf(fmaxf(0.0f, 1.0f - sin2_t));

    float r_s = (eta * cos_theta_i - cos_theta_t)
              / (eta * cos_theta_i + cos_theta_t);
    float r_p = (cos_theta_i - eta * cos_theta_t)
              / (cos_theta_i + eta * cos_theta_t);

    return 0.5f * (r_s * r_s + r_p * r_p);
}

// ---------------------------------------------------------------------------
// F0 from IOR (for dielectric materials)
// ---------------------------------------------------------------------------
DEVICE_FUNC float f0_from_ior(float ior)
{
    float r = (ior - 1.0f) / (ior + 1.0f);
    return r * r;
}

// ---------------------------------------------------------------------------
// F0 from IOR ratio (for nested dielectrics)
//   n1 = exterior IOR, n2 = interior IOR
// ---------------------------------------------------------------------------
DEVICE_FUNC float f0_from_ior_ratio(float n1, float n2)
{
    float r = (n2 - n1) / (n2 + n1);
    return r * r;
}

// ---------------------------------------------------------------------------
// F0 for the glTF specular extension
//   ior             = index of refraction
//   specular        = specular level [0, 1], i.e. specularFactor / 2
//   specular_color  = KHR_materials_specular specularColorFactor
//   base_color      = albedo
//   metallic        = metallic weight [0, 1]
//
// specular_color is an independent multiplier on the dielectric F0, not a weight
// blending it toward the base colour. The Disney-style tint this used to take
// was a different parameter wearing the same name: it made a red plastic's
// highlight red, where the extension makes a gold-tinted varnish's highlight
// gold over any base at all. The loader hardcoded it to 0, so nothing ever
// exercised the difference.
// ---------------------------------------------------------------------------
DEVICE_FUNC float3 gltf_f0(float ior, float specular, float3 specular_color,
                            float3 base_color, float metallic)
{
    const float dielectric_f0 = f0_from_ior(ior) * 2.0f * specular;
    // Clamped per the extension: F0 is a reflectance and a tint above 1 would
    // make a dielectric reflect more than it receives.
    float3 F0_dielectric = specular_color * dielectric_f0;
    F0_dielectric.x = fminf(F0_dielectric.x, 1.0f);
    F0_dielectric.y = fminf(F0_dielectric.y, 1.0f);
    F0_dielectric.z = fminf(F0_dielectric.z, 1.0f);
    return mix(F0_dielectric, base_color, metallic);
}

#endif // STRELKA_FRESNEL_H
