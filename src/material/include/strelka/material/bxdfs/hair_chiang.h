#ifndef STRELKA_BXDF_HAIR_CHIANG_H
#define STRELKA_BXDF_HAIR_CHIANG_H

// ============================================================================
// bxdfs/hair_chiang.h -- Chiang et al. 2016 hair / fur BSDF
//
// Port of Cycles' bsdf_principled_hair_chiang.h (Apache-2.0 / Blender Foundation)
// into Strelka's three-compiler material headers. R / TT / TRT / TRRT+ lobes with
// longitudinal (Mp) and azimuthal (Np) factors. What a rough dielectric cylinder
// cannot do -- light entering a strand and leaving through another path -- is
// exactly what these lobes carry, and why the kids-bedroom monster came out 42%
// darker than the reference with geometry alone.
//
// Parameter mapping onto MaterialParams (no struct growth):
//   albedo      -> reflectance colour (Direct Coloring -> sigma_a)
//   roughness   -> longitudinal roughness
//   anisotropy  -> radial roughness (0 falls back to roughness)
//   clearcoat   -> coat weight (scales primary roughness)
//   ior         -> eta (default 1.55 keratin)
// Cuticle tilt is fixed at 2 degrees, Cycles' Principled Hair default.
// ============================================================================

#include "../material_math.h"
#include "../bsdf_types.h"
#include "../surface_interaction.h"
#include "../fresnel.h"

#if defined(__METAL_VERSION__)
#define hair_sinhf(x) metal::sinh(x)
#define hair_asinf(x) metal::asin(x)
#define hair_atan2f(y, x) metal::atan2(y, x)
#define hair_floorf(x) metal::floor(x)
#define hair_copysignf(a, b) metal::copysign(a, b)
#else
#define hair_sinhf(x) sinhf(x)
#define hair_asinf(x) asinf(x)
#define hair_atan2f(y, x) atan2f(y, x)
#define hair_floorf(x) floorf(x)
#define hair_copysignf(a, b) copysignf(a, b)
#endif

#ifndef M_2PI_F
#define M_2PI_F (2.0f * M_PI_F)
#endif
#ifndef M_1_2PI_F
#define M_1_2PI_F (0.5f * M_1_PI_F)
#endif
#ifndef M_SQRT_PI_8_F
#define M_SQRT_PI_8_F 0.6266570686577501f
#endif
#ifndef M_LN_2PI_F
#define M_LN_2PI_F 1.8378770664093453f
#endif

// Cuticle tilt, radians. Cycles Principled Hair default Offset.
#define HAIR_CUTICLE_ALPHA 0.034906585f

DEVICE_FUNC float hair_cos_from_sin(float s)
{
    return sqrtf(fmaxf(0.0f, 1.0f - s * s));
}

DEVICE_FUNC float hair_sin_from_cos(float c)
{
    return sqrtf(fmaxf(0.0f, 1.0f - c * c));
}

DEVICE_FUNC float hair_safe_asinf(float x)
{
    return hair_asinf(clamp(x, -1.0f, 1.0f));
}

DEVICE_FUNC float hair_safe_divide(float a, float b)
{
    return (fabsf(b) > 1e-20f) ? (a / b) : 0.0f;
}

DEVICE_FUNC float3 hair_safe_divide3(float3 a, float3 b)
{
    return make_float3(hair_safe_divide(a.x, b.x), hair_safe_divide(a.y, b.y),
                       hair_safe_divide(a.z, b.z));
}

DEVICE_FUNC float hair_pow20(float x)
{
    const float x2 = x * x;
    const float x4 = x2 * x2;
    const float x8 = x4 * x4;
    return x8 * x8 * x4;
}

DEVICE_FUNC float hair_pow22(float x)
{
    return hair_pow20(x) * x * x;
}

DEVICE_FUNC float hair_delta_phi(int p, float gamma_o, float gamma_t)
{
    return 2.0f * (float)p * gamma_t - 2.0f * gamma_o + (float)p * M_PI_F;
}

DEVICE_FUNC float hair_wrap_angle(float a)
{
    return (a + M_PI_F) - M_2PI_F * hair_floorf((a + M_PI_F) / M_2PI_F) - M_PI_F;
}

DEVICE_FUNC float hair_logistic(float x, float s)
{
    const float v = expf(-fabsf(x) / s);
    return v / (s * sqr(1.0f + v));
}

DEVICE_FUNC float hair_logistic_cdf(float x, float s)
{
    const float arg = -x / s;
    if (arg > 88.0f)
        return 0.0f;
    return 1.0f / (1.0f + expf(arg));
}

DEVICE_FUNC float hair_bessel_I0(float x)
{
    x = sqr(x);
    float val = 1.0f + 0.25f * x;
    float pow_x_2i = sqr(x);
    float i_fac_2 = 1.0f;
    float pow_4_i = 16.0f;
    for (int i = 2; i < 10; ++i)
    {
        i_fac_2 *= (float)(i * i);
        const float newval = val + pow_x_2i / (pow_4_i * i_fac_2);
        if (val == newval)
            return val;
        val = newval;
        pow_x_2i *= x;
        pow_4_i *= 4.0f;
    }
    return val;
}

DEVICE_FUNC float hair_log_bessel_I0(float x)
{
    if (x > 12.0f)
        return x + 0.5f * (1.0f / (8.0f * x) - M_LN_2PI_F - logf(x));
    return logf(hair_bessel_I0(x));
}

DEVICE_FUNC float hair_trimmed_logistic(float x, float s)
{
    const float scaling_fac = 1.0f - 2.0f * hair_logistic_cdf(-M_PI_F, s);
    return hair_safe_divide(hair_logistic(x, s), scaling_fac);
}

DEVICE_FUNC float hair_sample_trimmed_logistic(float u, float s)
{
    const float cdf_minuspi = hair_logistic_cdf(-M_PI_F, s);
    const float x = -s * logf(1.0f / (u * (1.0f - 2.0f * cdf_minuspi) + cdf_minuspi) - 1.0f);
    return clamp(x, -M_PI_F, M_PI_F);
}

DEVICE_FUNC float hair_azimuthal_scattering(float phi, int p, float s, float gamma_o, float gamma_t)
{
    const float phi_o = hair_wrap_angle(phi - hair_delta_phi(p, gamma_o, gamma_t));
    return hair_trimmed_logistic(phi_o, s);
}

DEVICE_FUNC float hair_longitudinal_scattering(float sin_theta_i,
                                               float cos_theta_i,
                                               float sin_theta_o,
                                               float cos_theta_o,
                                               float v)
{
    const float inv_v = 1.0f / v;
    const float cos_arg = cos_theta_i * cos_theta_o * inv_v;
    const float sin_arg = sin_theta_i * sin_theta_o * inv_v;
    if (v <= 0.1f)
    {
        const float i0 = hair_log_bessel_I0(cos_arg);
        return expf(i0 - sin_arg - inv_v + 0.6931f + logf(0.5f * inv_v));
    }
    const float i0 = hair_bessel_I0(cos_arg);
    return (expf(-sin_arg) * i0) / (hair_sinhf(inv_v) * 2.0f * v);
}

// Direct Coloring: reflectance colour -> absorption (Cycles).
DEVICE_FUNC float3 hair_sigma_from_reflectance(float3 color, float radial_roughness)
{
    const float x = radial_roughness;
    const float roughness_fac =
        (((((0.245f * x) + 5.574f) * x - 10.73f) * x + 2.532f) * x - 0.215f) * x + 5.969f;
    const float3 c = make_float3(fmaxf(color.x, 1e-5f), fmaxf(color.y, 1e-5f), fmaxf(color.z, 1e-5f));
    const float3 sigma = make_float3(logf(c.x), logf(c.y), logf(c.z)) * (1.0f / roughness_fac);
    return make_float3(sigma.x * sigma.x, sigma.y * sigma.y, sigma.z * sigma.z);
}

DEVICE_FUNC void hair_attenuation(float f,
                                  float3 T,
                                  THREAD_REF float3* Ap,
                                  THREAD_REF float* Ap_energy)
{
    Ap[0] = make_float3(f);
    Ap_energy[0] = f;

    float3 col = sqr(1.0f - f) * T;
    Ap[1] = col;
    Ap_energy[1] = luminance(col);

    col = col * T * f;
    Ap[2] = col;
    Ap_energy[2] = luminance(col);

    const float3 denom = make_float3(1.0f) - T * f;
    col = col * hair_safe_divide3(T * f, denom);
    Ap[3] = col;
    Ap_energy[3] = luminance(col);

    const float tot = Ap_energy[0] + Ap_energy[1] + Ap_energy[2] + Ap_energy[3];
    const float fac = hair_safe_divide(1.0f, tot);
    Ap_energy[0] *= fac;
    Ap_energy[1] *= fac;
    Ap_energy[2] *= fac;
    Ap_energy[3] *= fac;
}

DEVICE_FUNC void hair_alpha_angles(float sin_theta_o,
                                   float cos_theta_o,
                                   float alpha,
                                   THREAD_REF float* angles)
{
    const float sin_1alpha = sinf(alpha);
    const float cos_1alpha = hair_cos_from_sin(sin_1alpha);
    const float sin_2alpha = 2.0f * sin_1alpha * cos_1alpha;
    const float cos_2alpha = sqr(cos_1alpha) - sqr(sin_1alpha);
    const float sin_4alpha = 2.0f * sin_2alpha * cos_2alpha;
    const float cos_4alpha = sqr(cos_2alpha) - sqr(sin_2alpha);

    angles[0] = sin_theta_o * cos_2alpha - cos_theta_o * sin_2alpha;
    angles[1] = fabsf(cos_theta_o * cos_2alpha + sin_theta_o * sin_2alpha);
    angles[2] = sin_theta_o * cos_1alpha + cos_theta_o * sin_1alpha;
    angles[3] = fabsf(cos_theta_o * cos_1alpha - sin_theta_o * sin_1alpha);
    angles[4] = sin_theta_o * cos_4alpha + cos_theta_o * sin_4alpha;
    angles[5] = fabsf(cos_theta_o * cos_4alpha - sin_theta_o * sin_4alpha);
}

struct HairChiangParams
{
    float3 sigma;
    float v;
    float s;
    float m0_roughness;
    float eta;
    float alpha;
    float h;
    float3 X; // strand tangent
    float3 Y; // secondary axis (from setup)
    float3 Z;
};

// Build the Chiang local frame and remap roughness the way Cycles does in
// bsdf_hair_chiang_setup. X is the strand tangent (si.tangent).
DEVICE_FUNC HairChiangParams hair_chiang_prepare(const THREAD_REF SurfaceInteraction& si)
{
    HairChiangParams p;

    float rough_u = clamp(si.roughness, 0.001f, 1.0f);
    float rough_v = (fabsf(si.anisotropy) > 1e-4f) ? clamp(fabsf(si.anisotropy), 0.001f, 1.0f)
                                                     : rough_u;
    float coat = saturate(si.clearcoat);
    float m0 = clamp((1.0f - coat) * rough_u, 0.001f, 1.0f);

    // Map roughness -> variance / scale (Chiang 2016 via Cycles).
    p.v = sqr(0.726f * rough_u + 0.812f * sqr(rough_u) + 3.700f * hair_pow20(rough_u));
    p.s = (0.265f * rough_v + 1.194f * sqr(rough_v) + 5.372f * hair_pow22(rough_v)) * M_SQRT_PI_8_F;
    p.m0_roughness = sqr(0.726f * m0 + 0.812f * sqr(m0) + 3.700f * hair_pow20(m0));

    p.eta = fmaxf(si.ior, 1.01f);
    p.alpha = -HAIR_CUTICLE_ALPHA;
    p.sigma = hair_sigma_from_reflectance(si.albedo, rough_v);

    p.X = safe_normalize(si.tangent);
    // Secondary axis from the view, as Cycles does -- keeps the azimuthal
    // origin aligned with the incident ray rather than an arbitrary binormal.
    const float3 wi = si.wo;
    p.Y = safe_normalize(cross(p.X, wi));
    if (dot(p.Y, p.Y) < 1e-8f)
        p.Y = safe_normalize(si.bitangent);
    p.Z = safe_normalize(cross(p.X, p.Y));

    // Offset within the fibre cross-section. Round-curve hits give a radial
    // geometric normal; this is the sine of the angle between Ng and Z.
    p.h = clamp(dot(cross(si.geometry_normal, p.X), p.Z), -1.0f, 1.0f);
    return p;
}

DEVICE_FUNC float3 hair_to_local(float3 w, float3 X, float3 Y, float3 Z)
{
    // Hair frame: X = strand tangent (sin θ along X, matching pbrt / Chiang).
    return make_float3(dot(w, X), dot(w, Y), dot(w, Z));
}

DEVICE_FUNC float3 hair_to_world(float3 w, float3 X, float3 Y, float3 Z)
{
    return X * w.x + Y * w.y + Z * w.z;
}

DEVICE_FUNC void hair_eval_lobes(const THREAD_REF HairChiangParams& p,
                                 float sin_theta_o,
                                 float cos_theta_o,
                                 float phi_o,
                                 float sin_theta_i,
                                 float cos_theta_i,
                                 float phi_i,
                                 THREAD_REF float3& F,
                                 THREAD_REF float& F_energy)
{
    const float sin_theta_t = sin_theta_o / p.eta;
    const float cos_theta_t = hair_cos_from_sin(sin_theta_t);

    const float sin_gamma_o = p.h;
    const float cos_gamma_o = hair_cos_from_sin(sin_gamma_o);
    const float gamma_o = hair_safe_asinf(sin_gamma_o);

    const float denom_g = sqr(p.eta) - sqr(sin_theta_o);
    const float sin_gamma_t =
        sin_gamma_o * cos_theta_o / sqrtf(fmaxf(denom_g, 1e-8f));
    const float cos_gamma_t = hair_cos_from_sin(sin_gamma_t);
    const float gamma_t = hair_safe_asinf(sin_gamma_t);

    const float3 T =
        make_float3(expf(-p.sigma.x * (2.0f * cos_gamma_t / fmaxf(cos_theta_t, 1e-4f))),
                    expf(-p.sigma.y * (2.0f * cos_gamma_t / fmaxf(cos_theta_t, 1e-4f))),
                    expf(-p.sigma.z * (2.0f * cos_gamma_t / fmaxf(cos_theta_t, 1e-4f))));

    // eta is hair IOR; our Fresnel wants exterior/interior = 1/eta.
    const float F0 =
        fresnel_dielectric(cos_theta_o * cos_gamma_o, 1.0f / p.eta);

    float3 Ap[4];
    float Ap_energy[4];
    hair_attenuation(F0, T, Ap, Ap_energy);

    const float phi = phi_i - phi_o;
    float angles[6];
    hair_alpha_angles(sin_theta_o, cos_theta_o, p.alpha, angles);

    F = make_float3(0.0f);
    F_energy = 0.0f;

    for (int i = 0; i < 3; ++i)
    {
        const float v_lobe = (i == 0) ? p.m0_roughness : (i == 1) ? 0.25f * p.v : 4.0f * p.v;
        const float Mp = hair_longitudinal_scattering(
            sin_theta_i, cos_theta_i, angles[2 * i], angles[2 * i + 1], v_lobe);
        const float Np = hair_azimuthal_scattering(phi, i, p.s, gamma_o, gamma_t);
        F = F + Ap[i] * (Mp * Np);
        F_energy += Ap_energy[i] * Mp * Np;
    }

    {
        const float Mp = hair_longitudinal_scattering(
            sin_theta_i, cos_theta_i, sin_theta_o, cos_theta_o, 4.0f * p.v);
        const float Np = M_1_2PI_F;
        F = F + Ap[3] * (Mp * Np);
        F_energy += Ap_energy[3] * Mp * Np;
    }
}

DEVICE_FUNC BsdfSampleResult hair_chiang_sample(const THREAD_REF SurfaceInteraction& si,
                                                float u1,
                                                float u2,
                                                float u3)
{
    BsdfSampleResult result;
    result.bsdf_over_pdf = make_float3(0.0f);
    result.pdf = 0.0f;
    result.event_type = BSDF_EVENT_ABSORB;

    HairChiangParams p = hair_chiang_prepare(si);
    const float3 local_O = hair_to_local(si.wo, p.X, p.Y, p.Z);
    const float sin_theta_o = local_O.x;
    const float cos_theta_o = hair_cos_from_sin(sin_theta_o);
    const float phi_o = hair_atan2f(local_O.z, local_O.y);

    const float sin_theta_t = sin_theta_o / p.eta;
    const float cos_theta_t = hair_cos_from_sin(sin_theta_t);
    const float sin_gamma_o = p.h;
    const float cos_gamma_o = hair_cos_from_sin(sin_gamma_o);
    const float gamma_o = hair_safe_asinf(sin_gamma_o);
    const float denom_g = sqr(p.eta) - sqr(sin_theta_o);
    const float sin_gamma_t =
        sin_gamma_o * cos_theta_o / sqrtf(fmaxf(denom_g, 1e-8f));
    const float cos_gamma_t = hair_cos_from_sin(sin_gamma_t);
    const float gamma_t = hair_safe_asinf(sin_gamma_t);

    const float3 T =
        make_float3(expf(-p.sigma.x * (2.0f * cos_gamma_t / fmaxf(cos_theta_t, 1e-4f))),
                    expf(-p.sigma.y * (2.0f * cos_gamma_t / fmaxf(cos_theta_t, 1e-4f))),
                    expf(-p.sigma.z * (2.0f * cos_gamma_t / fmaxf(cos_theta_t, 1e-4f))));
    const float F0 = fresnel_dielectric(cos_theta_o * cos_gamma_o, 1.0f / p.eta);

    float3 Ap[4];
    float Ap_energy[4];
    hair_attenuation(F0, T, Ap, Ap_energy);

    float rz = u3;
    int lobe = 0;
    for (; lobe < 3; ++lobe)
    {
        if (rz < Ap_energy[lobe])
            break;
        rz -= Ap_energy[lobe];
    }
    rz = hair_safe_divide(rz, Ap_energy[lobe]);

    float v = p.v;
    if (lobe == 1)
        v *= 0.25f;
    if (lobe >= 2)
        v *= 4.0f;

    float angles[6];
    hair_alpha_angles(sin_theta_o, cos_theta_o, p.alpha, angles);
    float sin_theta_o_tilted = sin_theta_o;
    float cos_theta_o_tilted = cos_theta_o;
    if (lobe < 3)
    {
        sin_theta_o_tilted = angles[2 * lobe];
        cos_theta_o_tilted = angles[2 * lobe + 1];
    }

    rz = fmaxf(rz, 1e-5f);
    const float fac = 1.0f + v * logf(rz + (1.0f - rz) * expf(-2.0f / v));
    const float sin_theta_i =
        -fac * sin_theta_o_tilted +
        hair_sin_from_cos(fac) * cosf(M_2PI_F * u2) * cos_theta_o_tilted;
    const float cos_theta_i = hair_cos_from_sin(sin_theta_i);

    float phi;
    if (lobe < 3)
        phi = hair_delta_phi(lobe, gamma_o, gamma_t) + hair_sample_trimmed_logistic(u1, p.s);
    else
        phi = M_2PI_F * u1;
    const float phi_i = phi_o + phi;

    float3 F;
    float F_energy;
    hair_eval_lobes(p, sin_theta_o, cos_theta_o, phi_o, sin_theta_i, cos_theta_i, phi_i, F,
                    F_energy);
    if (!(F_energy > 1e-10f))
        return result;

    // pbrt / Chiang local direction: (sin θ, cos θ cos φ, cos θ sin φ).
    const float3 local_I =
        make_float3(sin_theta_i, cos_theta_i * cosf(phi_i), cos_theta_i * sinf(phi_i));
    result.wi = safe_normalize(hair_to_world(local_I, p.X, p.Y, p.Z));
    result.pdf = F_energy;
    // Hair is not a surface BRDF with an |n·wi| factor in its definition.
    // Strelka's NEE multiplies cos back in against eval(), so sample() reports
    // F/pdf directly and eval() divides F by |n·wi| to keep the identity.
    result.bsdf_over_pdf = F * (1.0f / F_energy);
    result.event_type = (lobe == 0) ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;
    return result;
}

DEVICE_FUNC BsdfEvalResult hair_chiang_eval(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    BsdfEvalResult result;
    result.bsdf = make_float3(0.0f);
    result.pdf = 0.0f;

    HairChiangParams p = hair_chiang_prepare(si);
    const float3 local_O = hair_to_local(si.wo, p.X, p.Y, p.Z);
    const float3 local_I = hair_to_local(wi, p.X, p.Y, p.Z);

    const float sin_theta_o = local_O.x;
    const float cos_theta_o = hair_cos_from_sin(sin_theta_o);
    const float phi_o = hair_atan2f(local_O.z, local_O.y);
    const float sin_theta_i = local_I.x;
    const float cos_theta_i = hair_cos_from_sin(sin_theta_i);
    const float phi_i = hair_atan2f(local_I.z, local_I.y);

    float3 F;
    float F_energy;
    hair_eval_lobes(p, sin_theta_o, cos_theta_o, phi_o, sin_theta_i, cos_theta_i, phi_i, F,
                    F_energy);
    if (!(F_energy > 0.0f))
        return result;

    const float cos_n = fabsf(dot(si.shading_normal, wi));
    result.bsdf = F * (1.0f / fmaxf(cos_n, 1e-4f));
    result.pdf = F_energy;
    return result;
}

DEVICE_FUNC float hair_chiang_pdf(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    return hair_chiang_eval(si, wi).pdf;
}

#endif // STRELKA_BXDF_HAIR_CHIANG_H
