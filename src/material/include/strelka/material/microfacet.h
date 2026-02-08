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
DEVICE_FUNC float alpha_from_roughness(float roughness)
{
    float r = fmaxf(roughness, ROUGHNESS_MIN);
    return r * r;
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
// Anisotropic GGX helpers (for future use)
// ---------------------------------------------------------------------------
DEVICE_FUNC void anisotropic_alpha(float roughness, float anisotropy,
                                   float& alpha_x, float& alpha_y)
{
    float r2 = roughness * roughness;
    float aspect = sqrtf(1.0f - 0.9f * anisotropy);
    alpha_x = fmaxf(r2 / aspect, ROUGHNESS_MIN);
    alpha_y = fmaxf(r2 * aspect, ROUGHNESS_MIN);
}

#endif // STRELKA_MICROFACET_H
