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
    const float t = 1.0f - saturate(cos_theta);
    const float t2 = t * t;
    const float t5 = t2 * t2 * t;
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
    const float t = 1.0f - saturate(cos_theta);
    const float t2 = t * t;
    const float t5 = t2 * t2 * t;
    const float3 F_max =
        make_float3(fmaxf(1.0f - roughness, F0.x), fmaxf(1.0f - roughness, F0.y), fmaxf(1.0f - roughness, F0.z));
    return F0 + (F_max - F0) * t5;
}

// ---------------------------------------------------------------------------
// Scalar Schlick (for clearcoat, etc.)
// ---------------------------------------------------------------------------
DEVICE_FUNC float fresnel_schlick_scalar(float F0, float cos_theta)
{
    const float t = 1.0f - saturate(cos_theta);
    const float t2 = t * t;
    const float t5 = t2 * t2 * t;
    return F0 + (1.0f - F0) * t5;
}

// ---------------------------------------------------------------------------
// Exact dielectric Fresnel reflectance
//   cos_theta_i = cosine of incident angle (positive = same side as normal)
//   eta         = ratio of IORs (exterior / interior)
//   Returns reflectance in [0, 1].  Total internal reflection returns 1.
// ---------------------------------------------------------------------------
DEVICE_FUNC float fresnel_dielectric(CompensatedFloat cos_theta_i, float eta)
{
    // Identical media have no interface. At exactly grazing incidence the
    // generic formula becomes 0/0 even though its physical limit is zero.
    if (eta == 1.0f)
        return 0.0f;
    if (!(eta > 0.0f) || !(eta <= 3.402823466e38f))
        return 1.0f;

    // Ensure cos_theta_i is positive (flip if needed)
    float cosineValue = compensatedValue(cos_theta_i);
    if (cosineValue < 0.0f)
    {
        eta = 1.0f / eta;
        cos_theta_i = negateCompensated(cos_theta_i);
        cosineValue = -cosineValue;
    }

    if (!(cosineValue > 0.0f))
        cos_theta_i = compensatedSum(0.0f, 0.0f);
    else if (cosineValue >= 1.0f)
        cos_theta_i = compensatedSum(1.0f, 0.0f);

    // Near the critical angle, eta^2 * (1 - cos^2(theta_i)) can differ by
    // several ulps from one even though the remaining transmitted cosine is
    // much smaller than either term. Preserve those low parts through the
    // subtraction and square root; this value controls both the Fresnel branch
    // mass and the continuous transmission density.
    const CompensatedFloat cos2_t = dielectricTransmittedCosineSquared(cos_theta_i, eta);
    if (!(compensatedValue(cos2_t) > 0.0f))
        return 1.0f; // total internal reflection

    const CompensatedFloat cos_theta_t = sqrtCompensated(cos2_t);
    const CompensatedFloat eta_cos_theta_i = scaleCompensated(cos_theta_i, eta);
    const CompensatedFloat eta_cos_theta_t = scaleCompensated(cos_theta_t, eta);

    const float r_s = compensatedValue(divideCompensated(
        addCompensated(eta_cos_theta_i, negateCompensated(cos_theta_t)), addCompensated(eta_cos_theta_i, cos_theta_t)));
    const float r_p = compensatedValue(divideCompensated(
        addCompensated(cos_theta_i, negateCompensated(eta_cos_theta_t)), addCompensated(cos_theta_i, eta_cos_theta_t)));

    const CompensatedFloat reflectance = addCompensated(compensatedProduct(r_s, r_s), compensatedProduct(r_p, r_p));
    return saturate(0.5f * compensatedValue(reflectance));
}

DEVICE_FUNC float fresnel_dielectric(float cos_theta_i, float eta)
{
    return fresnel_dielectric(compensatedSum(cos_theta_i, 0.0f), eta);
}

// ---------------------------------------------------------------------------
// F0 from IOR (for dielectric materials)
// ---------------------------------------------------------------------------
DEVICE_FUNC float f0_from_ior(float ior)
{
    const float r = (ior - 1.0f) / (ior + 1.0f);
    return r * r;
}

// ---------------------------------------------------------------------------
// F0 from IOR ratio (for nested dielectrics)
//   n1 = exterior IOR, n2 = interior IOR
// ---------------------------------------------------------------------------
DEVICE_FUNC float f0_from_ior_ratio(float n1, float n2)
{
    const float r = (n2 - n1) / (n2 + n1);
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
DEVICE_FUNC float3 gltf_f0(float ior, float specular, float3 specular_color, float3 base_color, float metallic)
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
