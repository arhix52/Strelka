#ifndef STRELKA_FRESNEL_H
#define STRELKA_FRESNEL_H

// ============================================================================
// fresnel.h -- Fresnel reflectance functions
// ============================================================================

#include "material_math.h"

DEVICE_FUNC float3 fresnel_schlick(float3 F0, float cos_theta)
{
    const float t = 1.0f - saturate(cos_theta);
    const float t2 = t * t;
    const float t5 = t2 * t2 * t;
    return F0 + (make_float3(1.0f) - F0) * t5;
}

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

DEVICE_FUNC float fresnel_dielectric(InterfaceCosine cos_theta_i, float eta)
{
    // Identical media have no interface. At exactly grazing incidence the
    // generic formula becomes 0/0 even though its physical limit is zero.
    if (eta == 1.0f)
        return 0.0f;
    if (!(eta > 0.0f) || !(eta <= 3.402823466e38f))
        return 1.0f;

    // Ensure cos_theta_i is positive (flip if needed)
    float cosineValue = interfaceCosineValue(cos_theta_i);
    if (cosineValue < 0.0f)
    {
        eta = 1.0f / eta;
#if defined(STRELKA_FAST_FINITE_GPU_MATH) && STRELKA_FAST_FINITE_GPU_MATH
        cos_theta_i = -cos_theta_i;
#else
        cos_theta_i = negateCompensated(cos_theta_i);
#endif
        cosineValue = -cosineValue;
    }

    if (!(cosineValue > 0.0f))
        cos_theta_i = makeInterfaceCosine(0.0f);
    else if (cosineValue >= 1.0f)
        cos_theta_i = makeInterfaceCosine(1.0f);

#if defined(STRELKA_FAST_FINITE_GPU_MATH) && STRELKA_FAST_FINITE_GPU_MATH
    const float cos2_t = fmaf(eta * eta, cos_theta_i * cos_theta_i - 1.0f, 1.0f);
    if (!(cos2_t > 0.0f))
        return 1.0f;

    const float cos_theta_t = sqrtf(cos2_t);
    const float eta_cos_theta_i = eta * cos_theta_i;
    const float eta_cos_theta_t = eta * cos_theta_t;
    const float r_s = (eta_cos_theta_i - cos_theta_t) / (eta_cos_theta_i + cos_theta_t);
    const float r_p = (cos_theta_i - eta_cos_theta_t) / (cos_theta_i + eta_cos_theta_t);
    return saturate(0.5f * fmaf(r_s, r_s, r_p * r_p));
#else
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
#endif
}

#if !defined(STRELKA_FAST_FINITE_GPU_MATH) || !STRELKA_FAST_FINITE_GPU_MATH
DEVICE_FUNC float fresnel_dielectric(float cos_theta_i, float eta)
{
    return fresnel_dielectric(makeInterfaceCosine(cos_theta_i), eta);
}
#endif

// ---------------------------------------------------------------------------
// F0 from IOR (for dielectric materials)
// ---------------------------------------------------------------------------
DEVICE_FUNC float f0_from_ior(float ior)
{
    const float r = (ior - 1.0f) / (ior + 1.0f);
    return r * r;
}

DEVICE_FUNC float f0_from_ior_ratio(float n1, float n2)
{
    const float r = (n2 - n1) / (n2 + n1);
    return r * r;
}

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
