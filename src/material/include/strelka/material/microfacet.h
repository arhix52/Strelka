#ifndef STRELKA_MICROFACET_H
#define STRELKA_MICROFACET_H

#include "material_math.h"
#include "fresnel.h"
// build_onb / world_to_local, for the subsurface entry frame. sampling.h includes
// only material_math.h, so this does not cycle.
#include "sampling.h"

// ---------------------------------------------------------------------------
// Minimum roughness to avoid singularities
// ---------------------------------------------------------------------------
#define ROUGHNESS_MIN 0.0001f

DEVICE_FUNC bool refraction_is_delta(float interiorIor, float exteriorIor)
{
    return interiorIor == exteriorIor;
}

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
    const float r = fmaxf(roughness, ROUGHNESS_MIN);
    return r * r;
}

DEVICE_FUNC float ggx_energy_term(float roughness, float NdotV)
{
    const float r = roughness;
    const float g1 = 1.0f - NdotV;
    const float g4 = g1 * g1 * g1 * g1;

    const float p0 = 0.154250f + r * (-1.181688f + r * (2.959942f + r * 0.325514f));
    const float p1 = -2.939542f + r * (17.241920f + r * (-23.443382f + r * 7.088613f));
    const float p2 = 9.297826f + r * (-38.077586f + r * (44.542460f + r * -15.902794f));

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

DEVICE_FUNC float3 ggx_specular_albedo(float3 F0, float roughness, float NdotV)
{
    const float t = ggx_energy_term(roughness, NdotV);
    const float singleScatter = 1.0f / (1.0f + t);
    // Component-wise for the reason ggx_energy_compensation() is: float3 + float
    // is not spelled the same way on CUDA, Metal and GLM.
    return make_float3(fminf(F0.x * singleScatter * (1.0f + F0.x * t), 1.0f),
                       fminf(F0.y * singleScatter * (1.0f + F0.y * t), 1.0f),
                       fminf(F0.z * singleScatter * (1.0f + F0.z * t), 1.0f));
}

DEVICE_FUNC float ggx_ndf_from_cos_sin(float alpha, float NdotH, float sinThetaSquared)
{
    const float a2 = alpha * alpha;
    const float cos2 = NdotH * NdotH;
    const float denom = fmaxf(sinThetaSquared, 0.0f) + a2 * cos2;
    return (denom > 0.0f) ? (a2 / (M_PI_F * denom * denom)) : 0.0f;
}

DEVICE_FUNC float ggx_ndf(float alpha, float NdotH)
{
    return ggx_ndf_from_cos_sin(alpha, NdotH, (1.0f - NdotH) * (1.0f + NdotH));
}

// A rounded normalized H does not satisfy 1-dot(N,H)^2 accurately near N.
// Its tangent components do remain accurate, so production call sites use this
// full-vector form instead of reconstructing sin^2(theta) by subtraction.
DEVICE_FUNC float ggx_ndf(float alpha, float3 N, float3 H)
{
    const float3 tangent = cross(N, H);
    return ggx_ndf_from_cos_sin(alpha, dot(N, H), dot(tangent, tangent));
}

DEVICE_FUNC float ggx_smith_g1(float alpha, float NdotV)
{
    if (!(NdotV > 0.0f))
        return 0.0f;
    const float a2 = alpha * alpha;
    const float NdotV2 = NdotV * NdotV;
    const float root = sqrtf(a2 + (1.0f - a2) * NdotV2);
    return 2.0f * NdotV / (NdotV + root);
}

DEVICE_FUNC float ggx_smith_g2(float alpha, float NdotV, float NdotL)
{
    if (!(NdotV > 0.0f) || !(NdotL > 0.0f))
        return 0.0f;
    const float a2 = alpha * alpha;
    const float rootV = sqrtf(a2 + (1.0f - a2) * NdotV * NdotV);
    const float rootL = sqrtf(a2 + (1.0f - a2) * NdotL * NdotL);
    const float scale = fmaxf(NdotV, NdotL);
    const float scaledV = NdotV / scale;
    const float scaledL = NdotL / scale;
    return (2.0f * scaledV * NdotL) / (scaledL * rootV + scaledV * rootL);
}

DEVICE_FUNC float ggx_smith_visibility(float alpha, float NdotV, float NdotL)
{
    if (!(NdotV > 0.0f) || !(NdotL > 0.0f))
        return 0.0f;
    const float a2 = alpha * alpha;
    const float ggx_v = NdotL * sqrtf(a2 + (1.0f - a2) * NdotV * NdotV);
    const float ggx_l = NdotV * sqrtf(a2 + (1.0f - a2) * NdotL * NdotL);
    const float denominator = ggx_v + ggx_l;
    return fminf(0.5f / denominator, 3.402823466e+38f);
}

// A finite float can represent the limiting microfacet value at grazing only
// up to FLT_MAX.  Multiplying a separately saturated visibility by D can still
// overflow, so keep the complete D*G2/(4 cosV cosL) shape finite as one unit.
DEVICE_FUNC float saturating_nonnegative_product(float a, float b)
{
    const float maxFloat = 3.402823466e+38f;
#if defined(STRELKA_FAST_FINITE_GPU_MATH)
    if (STRELKA_FAST_FINITE_GPU_MATH)
    {
        return fminf(a * b, maxFloat);
    }
#endif
    if (!(a >= 0.0f) || !(a <= maxFloat) || !(b >= 0.0f) || !(b <= maxFloat))
        return 0.0f;
    if (!(a > 0.0f) || !(b > 0.0f))
        return 0.0f;
    return a > maxFloat / b ? maxFloat : a * b;
}

DEVICE_FUNC float saturating_nonnegative_sum(float a, float b)
{
    const float maxFloat = 3.402823466e+38f;
#if defined(STRELKA_FAST_FINITE_GPU_MATH)
    if (STRELKA_FAST_FINITE_GPU_MATH)
    {
        return fminf(a + b, maxFloat);
    }
#endif
    if (!(a >= 0.0f) || !(a <= maxFloat) || !(b >= 0.0f) || !(b <= maxFloat))
        return 0.0f;
    return a > maxFloat - b ? maxFloat : a + b;
}

DEVICE_FUNC float ggx_ndf_visibility(float alpha, float NdotH, float NdotV, float NdotL)
{
    return saturating_nonnegative_product(ggx_ndf(alpha, NdotH), ggx_smith_visibility(alpha, NdotV, NdotL));
}

DEVICE_FUNC float ggx_ndf_visibility(float alpha, float3 N, float3 H, float NdotV, float NdotL)
{
    return saturating_nonnegative_product(ggx_ndf(alpha, N, H), ggx_smith_visibility(alpha, NdotV, NdotL));
}

DEVICE_FUNC float3 ggx_vndf_sample(float3 wo_local, float alpha, float u1, float u2)
{
    // 1. Stretch wo
    const float3 Vh = safe_normalize(make_float3(alpha * wo_local.x, alpha * wo_local.y, wo_local.z));

    // 2. Build orthonormal basis around Vh
    const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    const float3 T1 = lensq > 1e-7f ? make_float3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq) : make_float3(1.0f, 0.0f, 0.0f);
    const float3 T2 = cross(Vh, T1);

    // 3. Parameterize projected area (hemisphere cap)
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PI_F * u2;
    const float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    const float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

    // 4. Reproject onto hemisphere
    const float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // 5. Unstretch
    const float3 H = safe_normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
    return H;
}

DEVICE_FUNC float ggx_vndf_pdf(float alpha, float NdotH, float NdotV, float VdotH)
{
    if (!(NdotV > 0.0f) || !(VdotH > 0.0f) || !(NdotH > 0.0f))
        return 0.0f;
    const float a2 = alpha * alpha;
    const float root = sqrtf(a2 + (1.0f - a2) * NdotV * NdotV);
    // Substitute G1=2*NdotV/(NdotV+root) before dividing by NdotV.
    // This keeps the removable cosine from underflowing at grazing incidence.
    return ggx_ndf(alpha, NdotH) / (2.0f * (NdotV + root));
}

DEVICE_FUNC float ggx_vndf_pdf(float alpha, float3 N, float3 H, float NdotV, float VdotH)
{
    if (!(NdotV > 0.0f) || !(VdotH > 0.0f) || !(dot(N, H) > 0.0f))
        return 0.0f;
    const float a2 = alpha * alpha;
    const float root = sqrtf(a2 + (1.0f - a2) * NdotV * NdotV);
    return ggx_ndf(alpha, N, H) / (2.0f * (NdotV + root));
}

DEVICE_FUNC float ggx_vndf_pdf_half(float alpha, float NdotH, float NdotV, float VdotH)
{
    if (!(NdotV > 0.0f) || !(VdotH > 0.0f) || !(NdotH >= 0.0f))
        return 0.0f;
    const float a2 = alpha * alpha;
    const float root = sqrtf(a2 + (1.0f - a2) * NdotV * NdotV);
    return ggx_ndf(alpha, NdotH) * (2.0f * VdotH / (NdotV + root));
}

DEVICE_FUNC float ggx_vndf_pdf_half(float alpha, float3 N, float3 H, float NdotV, float VdotH)
{
    const float NdotH = dot(N, H);
    if (!(NdotV > 0.0f) || !(VdotH > 0.0f) || !(NdotH >= 0.0f))
        return 0.0f;
    const float a2 = alpha * alpha;
    const float root = sqrtf(a2 + (1.0f - a2) * NdotV * NdotV);
    return ggx_ndf(alpha, N, H) * (2.0f * VdotH / (NdotV + root));
}

DEVICE_FUNC float3 subsurface_entry_direction(float3 wo, float3 n, float u1, float u2)
{
    const float eta = 1.0f / 1.4f;

    float3 T;
    float3 B;
    build_onb(n, T, B);
    const float3 woLocal = world_to_local(wo, T, B, n);
    if (woLocal.z <= 0.0f)
    {
        return make_float3(0.0f);
    }

    const float3 h = local_to_world(ggx_vndf_sample(woLocal, 1.0f, u1, u2), T, B, n);

    const float cosHI = fmaxf(dot(h, wo), 0.0f);
    const float k = 1.0f - eta * eta * (1.0f - cosHI * cosHI);
    const float cosT = sqrtf(fmaxf(k, 0.0f));
    return safe_normalize(-eta * wo + (eta * cosHI - cosT) * h);
}

DEVICE_FUNC float3 refraction_residual(float3 V, float3 wt, float eta)
{
    const float etaSafe = fmaxf(eta, 1e-6f);
    return make_float3(fmaf(etaSafe, V.x, wt.x), fmaf(etaSafe, V.y, wt.y), fmaf(etaSafe, V.z, wt.z));
}

#if defined(STRELKA_FAST_FINITE_GPU_MATH) && STRELKA_FAST_FINITE_GPU_MATH

DEVICE_FUNC float refraction_residual_length(float3 V, float3 wt, float eta)
{
    return length(refraction_residual(V, wt, eta));
}

DEVICE_FUNC float3 refraction_half_vector(float3 V, float3 wt, float eta, float3 Nf, THREAD_REF InterfaceCosine& viewDotHalf)
{
    const float3 residual = refraction_residual(V, wt, eta);
    const float lengthSquared = dot(residual, residual);
    if (!(lengthSquared > 0.0f))
    {
        viewDotHalf = saturate(dot(V, Nf));
        return Nf;
    }
    float3 H = residual / sqrtf(lengthSquared);
    if (dot(V, H) < 0.0f)
    {
        H = -H;
    }
    viewDotHalf = saturate(dot(V, H));
    return H;
}

DEVICE_FUNC float3 refraction_half_vector(float3 V, float3 wt, float eta, float3 Nf)
{
    InterfaceCosine viewDotHalf = 0.0f;
    return refraction_half_vector(V, wt, eta, Nf, viewDotHalf);
}

#else

struct RefractionResidualExpansion
{
    CompensatedFloat x;
    CompensatedFloat y;
    CompensatedFloat z;
};

DEVICE_FUNC RefractionResidualExpansion refraction_residual_expansion(float3 V, float3 wt, float eta)
{
    const float etaSafe = fmaxf(eta, 1e-6f);
    RefractionResidualExpansion result{};
    result.x = addCompensated(compensatedProduct(etaSafe, V.x), compensatedSum(wt.x, 0.0f));
    result.y = addCompensated(compensatedProduct(etaSafe, V.y), compensatedSum(wt.y, 0.0f));
    result.z = addCompensated(compensatedProduct(etaSafe, V.z), compensatedSum(wt.z, 0.0f));
    return result;
}

DEVICE_FUNC void refraction_residual_metrics(
    float3 V, float3 wt, float eta, THREAD_REF float& residualLength, THREAD_REF CompensatedFloat& signedVdotH)
{
#    if defined(STRELKA_FAST_FINITE_GPU_MATH)
    if (STRELKA_FAST_FINITE_GPU_MATH)
    {
        const float3 residual = refraction_residual(V, wt, eta);
        residualLength = length(residual);
        signedVdotH = compensatedSum(residualLength > 0.0f ? dot(V, residual) / residualLength : 0.0f, 0.0f);
        return;
    }
#    endif
    const RefractionResidualExpansion residual = refraction_residual_expansion(V, wt, eta);
    const float x = compensatedValue(residual.x);
    const float y = compensatedValue(residual.y);
    const float z = compensatedValue(residual.z);
    const float scale = fmaxf(fabsf(x), fmaxf(fabsf(y), fabsf(z)));
    if (!(scale > 0.0f) || !(scale <= 3.402823466e38f))
    {
        residualLength = 0.0f;
        signedVdotH = compensatedSum(0.0f, 0.0f);
        return;
    }

    int exponent = 0;
    decomposeFloatExponent(scale, exponent);
    const CompensatedFloat sx = scaleCompensatedExponent(residual.x, -exponent);
    const CompensatedFloat sy = scaleCompensatedExponent(residual.y, -exponent);
    const CompensatedFloat sz = scaleCompensatedExponent(residual.z, -exponent);
    const CompensatedFloat lengthSquared = addCompensated(
        addCompensated(multiplyCompensated(sx, sx), multiplyCompensated(sy, sy)), multiplyCompensated(sz, sz));
    const CompensatedFloat scaledLength = sqrtCompensated(lengthSquared);
    const float scaledLengthValue = compensatedValue(scaledLength);
    if (!(scaledLengthValue > 0.0f))
    {
        residualLength = 0.0f;
        signedVdotH = compensatedSum(0.0f, 0.0f);
        return;
    }

    residualLength = scaleFloatExponent(scaledLengthValue, exponent);
    const CompensatedFloat viewNumerator =
        addCompensated(addCompensated(scaleCompensated(sx, V.x), scaleCompensated(sy, V.y)), scaleCompensated(sz, V.z));
    signedVdotH = divideCompensated(viewNumerator, scaledLength);
}

DEVICE_FUNC float refraction_residual_length(float3 V, float3 wt, float eta)
{
    float residualLength = 0.0f;
    CompensatedFloat signedVdotH = compensatedSum(0.0f, 0.0f);
    refraction_residual_metrics(V, wt, eta, residualLength, signedVdotH);
    return residualLength;
}

DEVICE_FUNC float3 refraction_half_vector(float3 V, float3 wt, float eta, float3 Nf, THREAD_REF CompensatedFloat& robustVdotH)
{
    const float etaSafe = fmaxf(eta, 1e-6f);
    float3 residual = refraction_residual(V, wt, etaSafe);
#    if defined(STRELKA_FAST_FINITE_GPU_MATH)
    if (STRELKA_FAST_FINITE_GPU_MATH)
    {
        const float lengthSquared = dot(residual, residual);
        if (!(lengthSquared > 0.0f))
        {
            robustVdotH = compensatedSum(saturate(dot(V, Nf)), 0.0f);
            return Nf;
        }
        float3 H = residual / sqrtf(lengthSquared);
        if (dot(V, H) < 0.0f)
        {
            H = -H;
        }
        robustVdotH = compensatedSum(saturate(dot(V, H)), 0.0f);
        return H;
    }
#    endif
    float residualLength = 0.0f;
    CompensatedFloat signedVdotH = compensatedSum(0.0f, 0.0f);
    refraction_residual_metrics(V, wt, etaSafe, residualLength, signedVdotH);
    if (!(residualLength > 0.0f))
    {
        robustVdotH = compensatedSum(saturate(accurateDot(V, Nf)), 0.0f);
        return Nf;
    }

    const float roundoffBound = 9.5367431640625e-7f * (etaSafe * (fabsf(V.x) + fabsf(V.y) + fabsf(V.z)) + fabsf(wt.x) +
                                                       fabsf(wt.y) + fabsf(wt.z));
    if (compensatedValue(signedVdotH) < 0.0f)
    {
        residual = -residual;
        signedVdotH = negateCompensated(signedVdotH);
    }
    const float normalResidual = dot(Nf, residual);
    bool adjustedRepresentative = false;
    if (normalResidual < 0.0f && -normalResidual <= roundoffBound)
    {
        residual = residual - normalResidual * Nf;
        adjustedRepresentative = true;
    }

    float3 H = normalizeFiniteVectorOrZero(residual);
    if (!(dot(H, H) > 0.0f))
    {
        robustVdotH = compensatedSum(saturate(accurateDot(V, Nf)), 0.0f);
        return Nf;
    }
    robustVdotH = adjustedRepresentative ? compensatedSum(saturate(accurateDot(V, H)), 0.0f) : signedVdotH;
    const float robustVdotHValue = saturate(compensatedValue(robustVdotH));
    if (etaSafe != 1.0f && robustVdotHValue > 0.0f && fresnel_dielectric(robustVdotH, etaSafe) == 1.0f)
    {
        const float etaSquared = etaSafe * etaSafe;
        const float angularRoundoff = fminf(1.0f, 2.0f * roundoffBound / residualLength);
        float targetSquared = 0.0f;
        if (etaSafe > 1.0f)
        {
            const float criticalSquared = fmaxf(0.0f, 1.0f - 1.0f / etaSquared);
            targetSquared = fminf(1.0f, fmaxf(criticalSquared, 1.0f - (1.0f - 3.814697265625e-6f) / etaSquared));
        }
        else
        {
            const float transmittedNormalCosine = sqrtf(fmaxf(0.0f, 1.0f - etaSquared));
            const float target = 7.62939453125e-6f * transmittedNormalCosine;
            targetSquared = target * target;
        }
        const float target = sqrtf(targetSquared);
        if (target > robustVdotHValue && target - robustVdotHValue <= angularRoundoff)
        {
            const float3 Vn = normalizeFiniteVectorOrZero(V);
            const float3 tangent = normalizeFiniteVectorOrZero(H - robustVdotHValue * Vn);
            if (dot(Vn, Vn) > 0.0f && dot(tangent, tangent) > 0.0f)
            {
                // Stay a few float roundoff units inside the represented
                // non-unit-Fresnel side so sample and eval make the same branch
                // decision at both critical and exterior grazing limits.
                H = normalizeFiniteVectorOrZero(target * Vn + sqrtf(fmaxf(0.0f, 1.0f - targetSquared)) * tangent);
                robustVdotH = compensatedSum(saturate(accurateDot(V, H)), 0.0f);
            }
        }
    }
    return H;
}

DEVICE_FUNC float3 refraction_half_vector(float3 V, float3 wt, float eta, float3 Nf)
{
    CompensatedFloat robustVdotH = compensatedSum(0.0f, 0.0f);
    return refraction_half_vector(V, wt, eta, Nf, robustVdotH);
}

#endif

// Reflection has H parallel to V+L. Use a scale-safe normalization and orient
// a genuinely collapsed tangent-limit sum to the active side rather than to a
// fixed world axis.
DEVICE_FUNC float3 reflection_half_vector(float3 V, float3 L, float3 Nf)
{
    float3 H = normalizeFiniteVectorOrZero(V + L);
    if (!(dot(H, H) > 0.0f))
        H = Nf;
    return dot(Nf, H) < 0.0f ? -H : H;
}

// eta^2 |wt.h| / (eta_i (V.h) + eta_t (wt.h))^2, with the same normalisation as
// refraction_half_vector: dividing through by eta_i and substituting eta_t/eta_i
// = 1/eta leaves |wt.h| / (eta (V.h) + (wt.h))^2, which is what this returns.
DEVICE_FUNC float refraction_jacobian(float eta, float VdotH, float LdotH)
{
    const float denom = fmaf(eta, VdotH, LdotH);
    const float denomSquared = denom * denom;
    if (!(denomSquared > 0.0f))
        return 3.402823466e+38f;
    return fminf(fabsf(LdotH) / denomSquared, 3.402823466e+38f);
}

DEVICE_FUNC float refraction_jacobian(float3 V, float3 L, float eta, float LdotH)
{
    const float denominator = refraction_residual_length(V, L, eta);
    const float denominatorSquared = denominator * denominator;
    if (!(denominatorSquared > 0.0f))
        return 3.402823466e+38f;
    return fminf(fabsf(LdotH) / denominatorSquared, 3.402823466e+38f);
}

DEVICE_FUNC void anisotropic_alpha(float roughness, float anisotropy, THREAD_REF float& alpha_x, THREAD_REF float& alpha_y)
{
    const float r2 = roughness * roughness;
    const float a = anisotropy < 0.0f ? -anisotropy : anisotropy;
    const float aspect = sqrtf(1.0f - 0.9f * a);
    const float long_axis = fmaxf(r2 / aspect, ROUGHNESS_MIN);
    const float short_axis = fmaxf(r2 * aspect, ROUGHNESS_MIN);
    alpha_x = anisotropy < 0.0f ? short_axis : long_axis;
    alpha_y = anisotropy < 0.0f ? long_axis : short_axis;
}

DEVICE_FUNC bool anisotropic_ggx_is_delta(float alpha_x, float alpha_y)
{
    return alpha_x < BSDF_DELTA_ALPHA && alpha_y < BSDF_DELTA_ALPHA;
}

// D(H), with H in the tangent frame (x along T, y along B, z along N).
DEVICE_FUNC float ggx_ndf_aniso(float ax, float ay, float3 H)
{
    const float hx = H.x / ax;
    const float hy = H.y / ay;
    const float d = hx * hx + hy * hy + H.z * H.z;
    const float denom = M_PI_F * ax * ay * d * d;
    return (denom > 0.0f) ? (1.0f / denom) : 0.0f;
}

// Smith lambda for anisotropic GGX, w in the tangent frame.
DEVICE_FUNC float ggx_smith_lambda_aniso(float ax, float ay, float3 w)
{
    const float wz2 = w.z * w.z;
    if (wz2 >= 1.0f - 1e-7f)
        return 0.0f;
    if (!(wz2 > 0.0f))
        return 1.0e30f;
    const float a2 = (ax * ax * w.x * w.x + ay * ay * w.y * w.y) / wz2;
    return 0.5f * (sqrtf(1.0f + a2) - 1.0f);
}

DEVICE_FUNC float ggx_smith_g1_aniso(float ax, float ay, float3 V)
{
    if (!(V.z > 0.0f))
        return 0.0f;
    const float root = sqrtf(V.z * V.z + ax * ax * V.x * V.x + ay * ay * V.y * V.y);
    return 2.0f * V.z / (V.z + root);
}

DEVICE_FUNC float ggx_smith_g2_aniso(float ax, float ay, float3 V, float3 L)
{
    if (!(V.z > 0.0f) || !(L.z > 0.0f))
        return 0.0f;
    const float rootV = sqrtf(V.z * V.z + ax * ax * V.x * V.x + ay * ay * V.y * V.y);
    const float rootL = sqrtf(L.z * L.z + ax * ax * L.x * L.x + ay * ay * L.y * L.y);
    const float scale = fmaxf(V.z, L.z);
    const float scaledV = V.z / scale;
    const float scaledL = L.z / scale;
    return (2.0f * scaledV * L.z) / (scaledL * rootV + scaledV * rootL);
}

DEVICE_FUNC float ggx_smith_visibility_aniso(float ax, float ay, float3 V, float3 L)
{
    if (!(V.z > 0.0f) || !(L.z > 0.0f))
        return 0.0f;
    const float rootV = sqrtf(V.z * V.z + ax * ax * V.x * V.x + ay * ay * V.y * V.y);
    const float rootL = sqrtf(L.z * L.z + ax * ax * L.x * L.x + ay * ay * L.y * L.y);
    const float denominator = L.z * rootV + V.z * rootL;
    return fminf(0.5f / denominator, 3.402823466e+38f);
}

DEVICE_FUNC float ggx_ndf_visibility_aniso(float ax, float ay, float3 H, float3 V, float3 L)
{
    return saturating_nonnegative_product(ggx_ndf_aniso(ax, ay, H), ggx_smith_visibility_aniso(ax, ay, V, L));
}

// Heitz 2018, with the stretch applied per axis instead of uniformly.
DEVICE_FUNC float3 ggx_vndf_sample_aniso(float3 wo_local, float ax, float ay, float u1, float u2)
{
    const float3 Vh = safe_normalize(make_float3(ax * wo_local.x, ay * wo_local.y, wo_local.z));

    const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    const float3 T1 = lensq > 1e-7f ? make_float3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq) : make_float3(1.0f, 0.0f, 0.0f);
    const float3 T2 = cross(Vh, T1);

    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PI_F * u2;
    const float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    const float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

    const float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    return safe_normalize(make_float3(ax * Nh.x, ay * Nh.y, fmaxf(0.0f, Nh.z)));
}

// Density of ggx_vndf_sample_aniso with respect to the reflected direction.
DEVICE_FUNC float ggx_vndf_pdf_aniso(float ax, float ay, float3 H, float3 V)
{
    const float VdotH = dot(V, H);
    if (VdotH <= 0.0f || V.z <= 0.0f)
        return 0.0f;
    const float root = sqrtf(V.z * V.z + ax * ax * V.x * V.x + ay * ay * V.y * V.y);
    return ggx_ndf_aniso(ax, ay, H) / (2.0f * (V.z + root));
}

#endif // STRELKA_MICROFACET_H
