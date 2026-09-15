#ifndef STRELKA_IRIDESCENCE_H
#define STRELKA_IRIDESCENCE_H

#include "material_math.h"
#include "fresnel.h"

DEVICE_FUNC float3 iridescence_f0_to_ior(float3 f0)
{
    // Inverse of ((n - 1) / (n + 1))^2, for an air interface.
    const float3 s = make_float3(sqrtf(f0.x), sqrtf(f0.y), sqrtf(f0.z));
    return make_float3((1.0f + s.x) / fmaxf(1.0f - s.x, 1e-4f),
                       (1.0f + s.y) / fmaxf(1.0f - s.y, 1e-4f),
                       (1.0f + s.z) / fmaxf(1.0f - s.z, 1e-4f));
}

DEVICE_FUNC float iridescence_ior_to_f0(float transmitted, float incident)
{
    const float r = (transmitted - incident) / (transmitted + incident);
    return r * r;
}

DEVICE_FUNC float3 iridescence_ior_to_f0(float3 transmitted, float incident)
{
    return make_float3(iridescence_ior_to_f0(transmitted.x, incident),
                       iridescence_ior_to_f0(transmitted.y, incident),
                       iridescence_ior_to_f0(transmitted.z, incident));
}

DEVICE_FUNC float3 iridescence_sensitivity(float opd, float3 shift)
{
    const float phase = 2.0f * M_PI_F * opd * 1.0e-9f;

    const float3 val = make_float3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
    const float3 pos = make_float3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
    const float3 var = make_float3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);

    const float phase2 = phase * phase;
    float3 xyz = make_float3(
        val.x * sqrtf(2.0f * M_PI_F * var.x) * cosf(pos.x * phase + shift.x) * expf(-phase2 * var.x),
        val.y * sqrtf(2.0f * M_PI_F * var.y) * cosf(pos.y * phase + shift.y) * expf(-phase2 * var.y),
        val.z * sqrtf(2.0f * M_PI_F * var.z) * cosf(pos.z * phase + shift.z) * expf(-phase2 * var.z));
    xyz.x += 9.7470e-14f * sqrtf(2.0f * M_PI_F * 4.5282e+09f) *
             cosf(2.2399e+06f * phase + shift.x) * expf(-4.5282e+09f * phase2);
    xyz = xyz * (1.0f / 1.0685e-7f);

    // XYZ -> linear sRGB (Rec.709 primaries, D65).
    return make_float3(3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
                       -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
                       0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
}

DEVICE_FUNC float3 iridescence_fresnel(float outside_ior, float film_ior, float cos_theta1,
                                       float thickness, float3 base_f0)
{
    if (thickness <= 0.0f)
    {
        return fresnel_schlick(base_f0, cos_theta1);
    }

    const float t = saturate(thickness / 0.03f);
    const float smooth_t = t * t * (3.0f - 2.0f * t);
    const float eta2 = mix(outside_ior, film_ior, smooth_t);

    const float ratio = outside_ior / fmaxf(eta2, 1e-4f);
    const float sin2 = ratio * ratio * (1.0f - cos_theta1 * cos_theta1);
    const float cos2sq = 1.0f - sin2;
    if (cos2sq < 0.0f)
    {
        return make_float3(1.0f); // total internal reflection in the film
    }
    const float cos_theta2 = sqrtf(cos2sq);

    // Outer interface.
    const float r0 = iridescence_ior_to_f0(eta2, outside_ior);
    const float r12 = fresnel_schlick_scalar(r0, cos_theta1);
    const float t121 = 1.0f - r12;
    const float phi12 = (eta2 < outside_ior) ? M_PI_F : 0.0f;
    const float phi21 = M_PI_F - phi12;

    // Inner interface, against whatever the base would have reflected.
    const float3 base_ior = iridescence_f0_to_ior(make_float3(fminf(base_f0.x, 0.9999f),
                                                              fminf(base_f0.y, 0.9999f),
                                                              fminf(base_f0.z, 0.9999f)));
    const float3 r1 = iridescence_ior_to_f0(base_ior, eta2);
    const float3 r23 = fresnel_schlick(r1, cos_theta2);
    const float3 phi23 = make_float3((base_ior.x < eta2) ? M_PI_F : 0.0f,
                                     (base_ior.y < eta2) ? M_PI_F : 0.0f,
                                     (base_ior.z < eta2) ? M_PI_F : 0.0f);

    const float opd = 2.0f * eta2 * thickness * cos_theta2;
    const float3 phi = make_float3(phi21 + phi23.x, phi21 + phi23.y, phi21 + phi23.z);

    const float3 r123 = make_float3(fminf(fmaxf(r12 * r23.x, 1e-5f), 0.9999f),
                                    fminf(fmaxf(r12 * r23.y, 1e-5f), 0.9999f),
                                    fminf(fmaxf(r12 * r23.z, 1e-5f), 0.9999f));
    const float3 sqrt_r123 = make_float3(sqrtf(r123.x), sqrtf(r123.y), sqrtf(r123.z));
    const float3 rs = make_float3(t121 * t121 * r23.x / (1.0f - r123.x),
                                  t121 * t121 * r23.y / (1.0f - r123.y),
                                  t121 * t121 * r23.z / (1.0f - r123.z));

    // The m = 0 term is the incoherent sum; the m > 0 terms are the interference
    // fringes, and two of them is where the series stops being visible.
    float3 result = make_float3(r12 + rs.x, r12 + rs.y, r12 + rs.z);
    float3 cm = make_float3(rs.x - t121, rs.y - t121, rs.z - t121);
    for (int m = 1; m <= 2; ++m)
    {
        cm = cm * sqrt_r123;
        const float3 sm = iridescence_sensitivity((float)m * opd,
                                                  make_float3((float)m * phi.x, (float)m * phi.y,
                                                              (float)m * phi.z)) *
                          2.0f;
        result = result + cm * sm;
    }

    return make_float3(fmaxf(result.x, 0.0f), fmaxf(result.y, 0.0f), fmaxf(result.z, 0.0f));
}

#endif // STRELKA_IRIDESCENCE_H
