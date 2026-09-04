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
    // Only an exactly index-matched represented interface collapses every
    // microfacet refraction to -V. Near matches remain continuous: at internal
    // grazing their H-dependent Fresnel can still differ by O(1) from the
    // macroscopic interface, so a numeric eta cutoff is not a valid delta
    // approximation.
    return interiorIor == exteriorIor;
}

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
    const float r = fmaxf(roughness, ROUGHNESS_MIN);
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

// ---------------------------------------------------------------------------
// ggx_specular_albedo -- what the specular lobe takes, so the base can be told.
//
// Derived, not fitted: integrating the lobe over F0 shows it is exactly
// (A * F0 + B) * (1 + F0 * t), the split-sum form times the factor
// ggx_energy_compensation() applies, and A is the single-scatter white albedo
// 1 / (1 + t). Accurate to under 1%; the numbers are in docs/open-defects.md.
//
// B is dropped deliberately. It is what the lobe reflects at F0 = 0 -- Schlick's
// (1-F0)(1-cos)^5 tail, which reaches one at grazing whatever the interface is --
// so subtracting it would take energy from a material whose specular weight is
// zero. It is a defect of its own, still open, and test_standard_pbr_furnace.cpp
// pins the two apart so a fix to either can be graded. B is zero head on.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// GGX (Trowbridge-Reitz) Normal Distribution Function
//   alpha  = roughness^2
//   NdotH  = dot(N, H)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Smith G1 for GGX (height-correlated)
//   alpha  = roughness^2
//   NdotV  = abs(dot(N, V))
// ---------------------------------------------------------------------------
DEVICE_FUNC float ggx_smith_g1(float alpha, float NdotV)
{
    if (!(NdotV > 0.0f))
        return 0.0f;
    const float a2 = alpha * alpha;
    const float NdotV2 = NdotV * NdotV;
    const float root = sqrtf(a2 + (1.0f - a2) * NdotV2);
    return 2.0f * NdotV / (NdotV + root);
}

// ---------------------------------------------------------------------------
// Smith G2 height-correlated masking-shadowing for GGX
//   alpha  = roughness^2
//   NdotV  = abs(dot(N, V))
//   NdotL  = abs(dot(N, L))
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Smith G2 / (4 * NdotV * NdotL) -- the "visibility" term V used in many
// rendering equations (combined masking-shadowing divided by the denominator
// of the Cook-Torrance specular BRDF).
// ---------------------------------------------------------------------------
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
    if (!(a >= 0.0f) || !(a <= maxFloat) || !(b >= 0.0f) || !(b <= maxFloat))
        return 0.0f;
    if (!(a > 0.0f) || !(b > 0.0f))
        return 0.0f;
    return a > maxFloat / b ? maxFloat : a * b;
}

DEVICE_FUNC float saturating_nonnegative_sum(float a, float b)
{
    const float maxFloat = 3.402823466e+38f;
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

// The density of the half vector itself, before any change of variables.
//
// ggx_vndf_pdf() above already divides by the 4 * VdotH that turns a half-vector
// density into a *reflected direction* density, which is what a reflection lobe
// wants and is why it is spelled that way. A refraction lobe needs the other
// Jacobian, so it needs the half-vector density back: multiplying the reflection
// form by 4 * VdotH is exactly that, and cheaper than a second D * G1.
//
// Getting this wrong is not a subtle error. Both transmission lobes used the
// reflection form directly and then applied the refraction Jacobian on top, so
// their reported density was short by a factor of 4 * VdotH -- around four at
// normal incidence. Integrating the pdf over the lower hemisphere gave 0.24
// where the sampler refracts 0.96 of the time.
DEVICE_FUNC float ggx_vndf_pdf_half(float alpha, float NdotH, float NdotV, float VdotH)
{
    // The tangent boundary NdotH==0 has zero continuous measure, but a finite
    // float refraction endpoint can inverse-map exactly onto it. GGX has a
    // finite boundary density, so rejecting that represented cell loses
    // support for near-index-matched interfaces.
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

// The direction a path takes on entering a subsurface medium.
//
// Refraction through the interface, about a GGX microfacet normal -- the port of
// Cycles' subsurface_entry_bounce(). This walk used to enter along the glTF
// diffuse-transmission lobe, a cosine hemisphere, and a path entering a slab of
// thickness d at angle theta crosses it along d / cos(theta), so a cosine entry
// transmits 2 * E3(tau) rather than exp(-tau): measurably steeper, and not even
// exponential. See docs/open-defects.md entry 16 and tools/feature_tests/sss_slab.py,
// which grades this against algebra rather than against a reference.
//
// The microfacet matters as much as the refraction, and a smooth interface was
// tried first and is wrong. Snell alone compresses the cone so hard -- at an index
// of 1.4 even a grazing ray bends to 45.6 degrees -- that every path dives almost
// radially, crosses the whole body and is absorbed instead of turning round near
// the surface. The slab, which only measures what crosses, was correct; the lit
// half of a sphere fell to 0.36 of the reference, because what lights it is
// scattering close to the entry and nothing was landing there.
//
// Neither the index nor the interface roughness is a parameter: the extension
// carries no index, 1.4 is Cycles' skin default and the value the ladder is
// graded against, and the roughness is measured to be 1 rather than the
// material's -- see the note at the sample below. Give either an argument when a
// scene needs to author it.
//
// `wo` points away from the surface toward where the light came from, and `n` is
// the shading normal on that side. Entering from the outside cannot reach total
// internal reflection, since eta < 1 leaves the radicand above 1 - eta^2.
DEVICE_FUNC float3 subsurface_entry_direction(float3 wo, float3 n, float u1, float u2)
{
    const float eta = 1.0f / 1.4f;

    float3 T;
    float3 B;
    build_onb(n, T, B);
    const float3 woLocal = world_to_local(wo, T, B, n);
    if (woLocal.z <= 0.0f)
    {
        // Cycles returns false here and the caller gives up on the bounce. The
        // zero vector is how that is spelled across a function that has to
        // return a direction; the two call sites reject it with the same test
        // they use on the geometric normal.
        return make_float3(0.0f);
    }

    // Fully rough, and a fixed index -- neither of which is what Cycles does.
    //
    // Cycles is explicit: `bssrdf->alpha = sqr(roughness)` and `bssrdf->ior = eta`,
    // the material's own index, with a separate `subsurface_ior` only for its skin
    // walk. Both were ported exactly and measured, and the exact port is 40% dark:
    // `32_subsurface_roughness` reads 0.600, 0.520, 0.576, 0.553, 0.821 across a
    // roughness ramp, and the half-space row's departure from Chandrasekhar goes
    // from 0.0131 to 0.0160. A narrow entry sends every path straight through the
    // body instead of letting it turn round near the surface, and at roughness 0
    // Cycles has the same alpha of zero and does not go dark.
    //
    // So the same formula behaves differently in the two renderers, and what is
    // here is the compensation that measures best rather than the port: alpha 1
    // and index 1.4. Against 1.5, the index the glTF actually carries, the ramp
    // reads 1.125, 1.126, 1.227, 1.108, 1.061 and Chandrasekhar 0.0160; at 1.4 it
    // reads 1.094, 1.087, 1.164, 1.068, 1.031 and 0.0131.
    //
    // That difference is unlocated and it is docs/open-defects.md entry 16. Do not
    // read these two constants as a description of Cycles.
    const float3 h = local_to_world(ggx_vndf_sample(woLocal, 1.0f, u1, u2), T, B, n);

    const float cosHI = fmaxf(dot(h, wo), 0.0f);
    const float k = 1.0f - eta * eta * (1.0f - cosHI * cosHI);
    const float cosT = sqrtf(fmaxf(k, 0.0f));
    return safe_normalize(-eta * wo + (eta * cosHI - cosT) * h);
}

// The half vector a refraction through an interface of relative index `eta`
// bends around, and the Jacobian of the map from it to the outgoing direction.
//
// eta is the shader's convention throughout: the ratio of the medium the view
// vector is in to the medium the transmitted ray enters. Walter et al. 2007
// build the half vector from eta_i * wi + eta_t * wt, which in that convention
// is V + wt / eta -- NOT V + eta * wt, which is what both eval paths were using.
// The difference is a half vector 0.2 radians away from the one the sampler
// bent around, and pdfs three orders of magnitude apart at grazing angles.
DEVICE_FUNC float3 refraction_residual(float3 V, float3 wt, float eta)
{
    const float etaSafe = fmaxf(eta, 1e-6f);
    return make_float3(fmaf(etaSafe, V.x, wt.x), fmaf(etaSafe, V.y, wt.y), fmaf(etaSafe, V.z, wt.z));
}

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
    // Multiplying V + wt/eta by eta gives eta*V + wt. FMA evaluates that
    // cancellation with one rounding instead of first rounding wt/eta and then
    // subtracting two nearly equal unit vectors. This is essential for
    // non-equal indices close to one, where the residual is the half vector.
    const float etaSafe = fmaxf(eta, 1e-6f);
    float3 residual = refraction_residual(V, wt, etaSafe);
    float residualLength = 0.0f;
    CompensatedFloat signedVdotH = compensatedSum(0.0f, 0.0f);
    refraction_residual_metrics(V, wt, etaSafe, residualLength, signedVdotH);
    if (!(residualLength > 0.0f))
    {
        robustVdotH = compensatedSum(saturate(accurateDot(V, Nf)), 0.0f);
        return Nf;
    }

    // A float direction represents the real-valued cell that rounded to it.
    // Bound how far eta*V+wt could move inside that cell. This certificate lets
    // us repair support lost by endpoint rounding without accepting an
    // unrelated direction whose exact inverse is genuinely below the VNDF
    // tangent plane.
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
    // The rounded direction denotes a float cell, not one exact real endpoint.
    // Pick the orientation visible from V, then project a roundoff-sized normal
    // sign violation to the nearest point in the closed VNDF support. Near
    // eta=1, endpoint rounding can otherwise put eta*V+wt on the wrong side of
    // the tangent plane even though the latent sampled H was valid.
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
