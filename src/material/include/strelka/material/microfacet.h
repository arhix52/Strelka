#ifndef STRELKA_MICROFACET_H
#define STRELKA_MICROFACET_H

// ============================================================================
// microfacet.h -- GGX (Trowbridge-Reitz) microfacet distribution and sampling
//
// All directions are in the local shading frame where Z = surface normal.
// Roughness "alpha" is the squared roughness (alpha = roughness^2) unless
// noted otherwise.
// ============================================================================

#include "material_math.h"

// ---------------------------------------------------------------------------
// Minimum roughness to avoid singularities
// ---------------------------------------------------------------------------
#define ROUGHNESS_MIN 0.0001f

// ---------------------------------------------------------------------------
// Clamp and square roughness
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Charlie sheen -- Estevez & Kulla, "Production Friendly Microfacet Sheen BRDF",
// which is the distribution KHR_materials_sheen is specified against.
//
// The whole point of it is retroreflection at grazing angles. A GGX lobe falls
// off exactly where fabric gets brighter, which is why a towel or a rug rendered
// with roughness alone reads as plastic no matter what roughness it is given.
// ---------------------------------------------------------------------------
DEVICE_FUNC float sheen_d_charlie(float alpha, float n_dot_h)
{
    alpha = fmaxf(alpha, 1e-3f);
    const float inv_alpha = 1.0f / alpha;
    const float cos2h = n_dot_h * n_dot_h;
    // Clamped away from zero: sin2h == 0 at normal incidence and the exponent is
    // negative for alpha < 1, so the unclamped form is an infinity on the one
    // direction every flat-on surface is sampled at.
    const float sin2h = fmaxf(1.0f - cos2h, 1e-7f);
    return (2.0f + inv_alpha) * powf(sin2h, inv_alpha * 0.5f) * (0.5f * M_1_PI_F);
}

// Ashikhmin's visibility term, as in the glTF spec's reference implementation.
// Not height-correlated Smith: Charlie has no matching Smith term, and this is
// what the extension is defined against.
DEVICE_FUNC float sheen_v_ashikhmin(float n_dot_l, float n_dot_v)
{
    return 1.0f / (4.0f * (n_dot_l + n_dot_v - n_dot_l * n_dot_v) + 1e-7f);
}

DEVICE_FUNC float alpha_from_roughness(float roughness)
{
    float r = fmaxf(roughness, ROUGHNESS_MIN);
    return r * r;
}

// ---------------------------------------------------------------------------
// Multiple-scattering energy compensation
//
// A single-scattering GGX lobe only carries the light that leaves the
// microsurface after one bounce. The rest -- everything that hits a second
// microfacet -- is discarded, and the loss grows with roughness: measured
// against Cycles, a white metal keeps 95% of its energy at roughness 0.33 but
// only 47% at roughness 1.0. Conductors show it starkly because they have no
// diffuse lobe to hide it.
//
// Turquin's compensation restores it multiplicatively:
//     f_ms = f_ss * (1 + F0 * (1/E - 1))
// where E is the directional albedo of the single-scattering lobe with F = 1.
// At F0 = 1 the factor is 1/E (all the energy comes back); at F0 = 0 it is 1.
//
// ggx_energy_term() is that (1/E - 1), fitted to a VNDF-sampled reference
// (E = mean of G2/G1) over roughness and cos(theta) in [0,1]. RMS error on the
// resulting factor is 1%, worst case 2.2% away from extreme grazing. The fit
// dips very slightly negative where the true term is already ~0, hence the
// clamp.
// ---------------------------------------------------------------------------
DEVICE_FUNC float ggx_energy_term(float roughness, float NdotV)
{
    const float r  = roughness;
    const float g1 = 1.0f - NdotV;
    const float g4 = g1 * g1 * g1 * g1;

    const float p0 = 0.154250f  + r * (-1.181688f  + r * (  2.959942f + r *   0.325514f));
    const float p1 = -2.939542f + r * (17.241920f  + r * (-23.443382f + r *   7.088613f));
    const float p2 = 9.297826f  + r * (-38.077586f + r * ( 44.542460f + r * -15.902794f));

    return fmaxf(0.0f, r * r * (p0 + p1 * g1 + p2 * g4));
}

// Convenience wrapper: the full multiplier for a lobe whose normal-incidence
// reflectance is F0.
DEVICE_FUNC float3 ggx_energy_compensation(float3 F0, float roughness, float NdotV)
{
    // Component-wise rather than vector arithmetic: float3 + float is not
    // spelled the same way on CUDA, Metal and GLM.
    const float t = ggx_energy_term(roughness, NdotV);
    return make_float3(1.0f + F0.x * t, 1.0f + F0.y * t, 1.0f + F0.z * t);
}

// ---------------------------------------------------------------------------
// GGX (Trowbridge-Reitz) Normal Distribution Function
//   alpha  = roughness^2
//   NdotH  = dot(N, H)
// ---------------------------------------------------------------------------
DEVICE_FUNC float ggx_ndf(float alpha, float NdotH)
{
    float a2    = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * denom * denom + 1e-10f);
}

// ---------------------------------------------------------------------------
// Smith G1 for GGX (height-correlated)
//   alpha  = roughness^2
//   NdotV  = abs(dot(N, V))
// ---------------------------------------------------------------------------
DEVICE_FUNC float ggx_smith_g1(float alpha, float NdotV)
{
    float a2    = alpha * alpha;
    float NdotV2 = NdotV * NdotV;
    return 2.0f * NdotV / (NdotV + sqrtf(a2 + (1.0f - a2) * NdotV2) + 1e-10f);
}

// ---------------------------------------------------------------------------
// Smith G2 height-correlated masking-shadowing for GGX
//   alpha  = roughness^2
//   NdotV  = abs(dot(N, V))
//   NdotL  = abs(dot(N, L))
// ---------------------------------------------------------------------------
DEVICE_FUNC float ggx_smith_g2(float alpha, float NdotV, float NdotL)
{
    float a2 = alpha * alpha;
    float ggx_v = NdotL * sqrtf(a2 + (1.0f - a2) * NdotV * NdotV);
    float ggx_l = NdotV * sqrtf(a2 + (1.0f - a2) * NdotL * NdotL);
    return 2.0f * NdotV * NdotL / (ggx_v + ggx_l + 1e-10f);
}

// ---------------------------------------------------------------------------
// Smith G2 / (4 * NdotV * NdotL) -- the "visibility" term V used in many
// rendering equations (combined masking-shadowing divided by the denominator
// of the Cook-Torrance specular BRDF).
// ---------------------------------------------------------------------------
DEVICE_FUNC float ggx_smith_visibility(float alpha, float NdotV, float NdotL)
{
    float a2 = alpha * alpha;
    float ggx_v = NdotL * sqrtf(a2 + (1.0f - a2) * NdotV * NdotV);
    float ggx_l = NdotV * sqrtf(a2 + (1.0f - a2) * NdotL * NdotL);
    return 0.5f / (ggx_v + ggx_l + 1e-10f);
}

// ---------------------------------------------------------------------------
// GGX VNDF (Visible Normal Distribution Function) sampling
//   Heitz 2018 -- "Sampling the GGX Distribution of Visible Normals"
//
//   wo_local = outgoing direction in local space (Z = up)
//   alpha    = roughness^2
//   u1, u2   = uniform random numbers in [0,1)
//   Returns: sampled half-vector in local space
// ---------------------------------------------------------------------------
DEVICE_FUNC float3 ggx_vndf_sample(float3 wo_local, float alpha, float u1, float u2)
{
    // 1. Stretch wo
    float3 Vh = safe_normalize(make_float3(alpha * wo_local.x,
                                           alpha * wo_local.y,
                                           wo_local.z));

    // 2. Build orthonormal basis around Vh
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 1e-7f
        ? make_float3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq)
        : make_float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);

    // 3. Parameterize projected area (hemisphere cap)
    float r   = sqrtf(u1);
    float phi = 2.0f * M_PI_F * u2;
    float t1  = r * cosf(phi);
    float t2  = r * sinf(phi);
    float s   = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

    // 4. Reproject onto hemisphere
    float3 Nh = t1 * T1 + t2 * T2
              + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // 5. Unstretch
    float3 H = safe_normalize(make_float3(alpha * Nh.x,
                                          alpha * Nh.y,
                                          fmaxf(0.0f, Nh.z)));
    return H;
}

// ---------------------------------------------------------------------------
// PDF of the VNDF sample (in terms of the half-vector H)
//   D_visible(H) = G1(wo) * D(H) * max(dot(wo, H), 0) / (NdotV)
//   But we need the PDF with respect to the reflected direction wi, which
//   introduces a Jacobian of 1 / (4 * dot(wo, H)).
//
//   alpha   = roughness^2
//   NdotH   = dot(N, H)
//   NdotV   = dot(N, wo)     (clamped positive)
//   VdotH   = dot(wo, H)     (clamped positive)
// ---------------------------------------------------------------------------
DEVICE_FUNC float ggx_vndf_pdf(float alpha, float NdotH, float NdotV, float VdotH)
{
    float D  = ggx_ndf(alpha, NdotH);
    float G1 = ggx_smith_g1(alpha, NdotV);
    return D * G1 * fmaxf(VdotH, 0.0f) / (NdotV + 1e-10f) / (4.0f * VdotH + 1e-10f);
}

// ---------------------------------------------------------------------------
// Anisotropic GGX
//
// The two axes are aligned with the surface's tangent frame, so unlike the
// isotropic form these take vectors in that frame rather than scalars: an
// anisotropic lobe is not a function of the angle to the normal alone.
//
// Every one of them reduces exactly to its isotropic counterpart when
// alpha_x == alpha_y, which is what lets the specular sites call only these and
// still render an isotropic material identically to before.
// ---------------------------------------------------------------------------
// The aspect ratio is driven by |anisotropy| and the sign only chooses which
// axis is the long one. Feeding a signed value straight into sqrt(1 - 0.9a)
// would make -a a *differently* elongated lobe rather than the same lobe turned
// 90 degrees, which is what the sign is supposed to mean; glTF sidesteps this by
// keeping anisotropyStrength in [0,1] and putting direction in a separate
// rotation, but nothing stops a caller passing a negative value.
DEVICE_FUNC void anisotropic_alpha(float roughness, float anisotropy,
                                   THREAD_REF float& alpha_x, THREAD_REF float& alpha_y)
{
    float r2 = roughness * roughness;
    float a = anisotropy < 0.0f ? -anisotropy : anisotropy;
    float aspect = sqrtf(1.0f - 0.9f * a);
    float long_axis  = fmaxf(r2 / aspect, ROUGHNESS_MIN);
    float short_axis = fmaxf(r2 * aspect, ROUGHNESS_MIN);
    alpha_x = anisotropy < 0.0f ? short_axis : long_axis;
    alpha_y = anisotropy < 0.0f ? long_axis : short_axis;
}

// D(H), with H in the tangent frame (x along T, y along B, z along N).
DEVICE_FUNC float ggx_ndf_aniso(float ax, float ay, float3 H)
{
    const float hx = H.x / ax;
    const float hy = H.y / ay;
    const float d  = hx * hx + hy * hy + H.z * H.z;
    return 1.0f / (M_PI_F * ax * ay * d * d + 1e-10f);
}

// Smith lambda for anisotropic GGX, w in the tangent frame.
DEVICE_FUNC float ggx_smith_lambda_aniso(float ax, float ay, float3 w)
{
    const float wz2 = w.z * w.z;
    if (wz2 >= 1.0f - 1e-7f)
        return 0.0f;
    const float a2 = (ax * ax * w.x * w.x + ay * ay * w.y * w.y) / (wz2 + 1e-10f);
    return 0.5f * (sqrtf(1.0f + a2) - 1.0f);
}

DEVICE_FUNC float ggx_smith_g1_aniso(float ax, float ay, float3 V)
{
    return 1.0f / (1.0f + ggx_smith_lambda_aniso(ax, ay, V));
}

DEVICE_FUNC float ggx_smith_g2_aniso(float ax, float ay, float3 V, float3 L)
{
    return 1.0f / (1.0f + ggx_smith_lambda_aniso(ax, ay, V) + ggx_smith_lambda_aniso(ax, ay, L));
}

// Heitz 2018, with the stretch applied per axis instead of uniformly.
DEVICE_FUNC float3 ggx_vndf_sample_aniso(float3 wo_local, float ax, float ay, float u1, float u2)
{
    float3 Vh = safe_normalize(make_float3(ax * wo_local.x, ay * wo_local.y, wo_local.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 1e-7f ? make_float3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq)
                              : make_float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);

    float r   = sqrtf(u1);
    float phi = 2.0f * M_PI_F * u2;
    float t1  = r * cosf(phi);
    float t2  = r * sinf(phi);
    float s   = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    return safe_normalize(make_float3(ax * Nh.x, ay * Nh.y, fmaxf(0.0f, Nh.z)));
}

// Density of ggx_vndf_sample_aniso with respect to the reflected direction.
DEVICE_FUNC float ggx_vndf_pdf_aniso(float ax, float ay, float3 H, float3 V)
{
    const float VdotH = dot(V, H);
    if (VdotH <= 0.0f || V.z <= 0.0f)
        return 0.0f;
    const float D  = ggx_ndf_aniso(ax, ay, H);
    const float G1 = ggx_smith_g1_aniso(ax, ay, V);
    return D * G1 / (4.0f * V.z + 1e-10f);
}

#endif // STRELKA_MICROFACET_H
