#ifndef STRELKA_BXDF_STANDARD_PBR_H
#    define STRELKA_BXDF_STANDARD_PBR_H

#    include "../material_math.h"
#    include "../bsdf_types.h"
#    include "../surface_interaction.h"
#    include "../sampling.h"
#    include "../fresnel.h"
#    include "../microfacet.h"
#    include "../sheen_albedo_lut.h"
#    include "../iridescence.h"
#    include "../shading_frame.h"
#    include <discrete_sampling.h>

// NOLINTBEGIN(cppcoreguidelines-pro-type-member-init, cppcoreguidelines-init-variables)

// ---------------------------------------------------------------------------
// Internal: compute lobe weights for stochastic lobe selection
// ---------------------------------------------------------------------------
struct PbrLobeWeights
{
    float diffuse;
    float diffuse_transmission;
    float specular;
    float transmission;
    float clearcoat;
    float total;
};

DEVICE_FUNC float3 specular_fresnel(const THREAD_REF SurfaceInteraction& si, float3 F0, float v_dot_h)
{
    const float3 base = fresnel_schlick(F0, v_dot_h);
    if (si.iridescence <= 0.0f)
    {
        return base;
    }
    const float3 film = iridescence_fresnel(1.0f, si.iridescence_ior, v_dot_h, si.iridescence_thickness, F0);
    return mix(base, film, saturate(si.iridescence));
}

DEVICE_FUNC float specular_lobe_scale(const THREAD_REF SurfaceInteraction& si)
{
    return 1.0f - si.transmission * (1.0f - si.metallic);
}

DEVICE_FUNC float diffuse_lobe_scale(const THREAD_REF SurfaceInteraction& si)
{
    return si.diffuse_faces_away ? 0.0f : 1.0f;
}

DEVICE_FUNC float3 transmission_fresnel(const THREAD_REF SurfaceInteraction& si, InterfaceCosine v_dot_h, float eta)
{
    const float f = fresnel_dielectric(v_dot_h, eta);
    // A thin-film approximation must not reopen the physically forbidden
    // transmission branch under total internal reflection.
    if (f >= 1.0f)
    {
        return make_float3(1.0f);
    }
    if (si.iridescence <= 0.0f)
    {
        return make_float3(f);
    }
    const float cosine = interfaceCosineValue(v_dot_h);
    const float3 film =
        iridescence_fresnel(1.0f, si.iridescence_ior, fabsf(cosine), si.iridescence_thickness, make_float3(f));
    const float3 result = mix(make_float3(f), film, saturate(si.iridescence));
    return make_float3(saturate(result.x), saturate(result.y), saturate(result.z));
}

#    if !defined(STRELKA_FAST_FINITE_GPU_MATH) || !STRELKA_FAST_FINITE_GPU_MATH
DEVICE_FUNC float3 transmission_fresnel(const THREAD_REF SurfaceInteraction& si, float v_dot_h, float eta)
{
    return transmission_fresnel(si, makeInterfaceCosine(v_dot_h), eta);
}
#    endif

DEVICE_FUNC float transmission_fresnel_probability(float3 filmFresnel)
{
    float probability = saturate(luminance(filmFresnel));
    const float reflected = fmaxf(filmFresnel.x, fmaxf(filmFresnel.y, filmFresnel.z));
    const float transmitted = fmaxf(1.0f - filmFresnel.x, fmaxf(1.0f - filmFresnel.y, 1.0f - filmFresnel.z));
    const float latticeGuard = 4.0f / 8388608.0f;
    if (reflected > 0.0f)
        probability = fmaxf(probability, latticeGuard);
    if (transmitted > 0.0f)
        probability = fminf(probability, 1.0f - latticeGuard);
    return probability;
}

// Charlie sheen evaluated for one direction pair. Zero unless the material
// carries the extension, so every scene without fabric compiles to the same
// work it did before.
DEVICE_FUNC float3 sheen_brdf(const THREAD_REF SurfaceInteraction& si, float n_dot_h, float n_dot_l, float n_dot_v)
{
    if (si.sheen <= 0.0f)
    {
        return make_float3(0.0f);
    }
    const float alpha = alpha_from_roughness(si.sheen_roughness);
    const float e = sheen_albedo(fabsf(n_dot_v), si.sheen_roughness);
    const float norm = (e > 1.0f) ? (1.0f / e) : 1.0f;
    return si.sheen_color * (si.sheen * norm * sheen_d_charlie(alpha, n_dot_h) * sheen_v_ashikhmin(n_dot_l, n_dot_v));
}

DEVICE_FUNC float clearcoat_f0(const THREAD_REF SurfaceInteraction& si)
{
    return f0_from_ior(fmaxf(si.clearcoat_ior, 1.0f));
}

DEVICE_FUNC float3 specular_base_scale(const THREAD_REF SurfaceInteraction& si, const THREAD_REF float3& F0, float n_dot_v)
{
    // ggx_specular_albedo() is already clamped to one and specular_lobe_scale()
    // to [0,1], so the complement cannot go negative and needs no clamp of its own.
    const float3 e = specular_lobe_scale(si) * ggx_specular_albedo(F0, si.roughness, fabsf(n_dot_v));
    return make_float3(1.0f - e.x, 1.0f - e.y, 1.0f - e.z);
}

DEVICE_FUNC float3 clearcoat_base_scale(const THREAD_REF SurfaceInteraction& si, float n_dot_v, float n_dot_l)
{
    if (si.clearcoat <= 0.0f)
    {
        return make_float3(1.0f);
    }
    const float f0 = clearcoat_f0(si);
    const float w = si.clearcoat;
    const float F_L = w * fresnel_schlick_scalar(f0, fabsf(n_dot_l));
    const float F_V = w * fresnel_schlick_scalar(f0, fabsf(n_dot_v));
    const float single = (1.0f - F_L) * (1.0f - F_V);
    const float F_ms = 0.5f * (F_L + F_V);
    const float ceiling = 1.0f - F_ms;

    const float3 rho = make_float3(saturate(si.albedo.x), saturate(si.albedo.y), saturate(si.albedo.z));
    return make_float3(fminf(single / fmaxf(1.0f - F_ms * rho.x, 1e-5f), ceiling),
                       fminf(single / fmaxf(1.0f - F_ms * rho.y, 1e-5f), ceiling),
                       fminf(single / fmaxf(1.0f - F_ms * rho.z, 1e-5f), ceiling));
}

DEVICE_FUNC float thin_glass_transmission_alpha(float alpha, float eta)
{
    eta = fmaxf(eta, 1.0f);
    const float t = (eta - 1.0f) * sqr(eta - 0.5f) / (eta * eta * eta);
    return saturate(alpha * sqrtf(3.4f * t));
}

// Mirror a direction through the macroscopic surface: (x, y, z) -> (x, y, -z)
// in the frame of `n`. Used to turn a reflection sample into a thin-wall
// transmission sample (Cycles / OpenPBR).
DEVICE_FUNC float3 flip_through_surface(float3 w, float3 n)
{
    return w - n * (2.0f * dot(w, n));
}

DEVICE_FUNC float sheen_base_scale(const THREAD_REF SurfaceInteraction& si, float n_dot_v)
{
    if (si.sheen <= 0.0f)
    {
        return 1.0f;
    }
    const float peak = fmaxf(si.sheen_color.x, fmaxf(si.sheen_color.y, si.sheen_color.z));
    const float e = fminf(sheen_albedo(fabsf(n_dot_v), si.sheen_roughness), 1.0f);
    return saturate(1.0f - peak * si.sheen * e);
}

DEVICE_FUNC PbrLobeWeights pbr_lobe_weights(const THREAD_REF SurfaceInteraction& si)
{
    PbrLobeWeights w;

    const float dielectric_weight = 1.0f - si.metallic;

    const float dt = saturate(si.diffuse_transmission);
    const float diffuse_base = dielectric_weight * (1.0f - si.transmission);

    w.diffuse = diffuse_base * (1.0f - dt) * luminance(si.albedo) * diffuse_lobe_scale(si);
    w.diffuse = fmaxf(w.diffuse, 0.0f);

    const float sheen_lum = si.sheen * luminance(si.sheen_color);
    w.diffuse = fmaxf(w.diffuse, sheen_lum * diffuse_lobe_scale(si));

    w.diffuse_transmission = diffuse_base * dt * luminance(si.diffuse_transmission_color);
    w.diffuse_transmission = fmaxf(w.diffuse_transmission, 0.0f);

    const float f0_scalar = f0_from_ior(si.ior);
    const float spec_lum = mix(f0_scalar * luminance(si.specular_color), luminance(si.albedo), si.metallic);
    w.specular = fmaxf(spec_lum, 0.04f) * (1.0f - si.transmission * dielectric_weight);

    w.transmission = dielectric_weight * si.transmission;
    w.transmission = fmaxf(w.transmission, 0.0f);

    // Scaled by the coat's own reflectance rather than by a constant: at IOR 2.0
    // the coat is nearly three times as reflective as the lacquer the old 0.25
    // stood in for, and a lobe selected too rarely is noise, not bias.
    w.clearcoat = si.clearcoat * fmaxf(f0_from_ior(fmaxf(si.clearcoat_ior, 1.0f)), 0.04f) * 6.0f;
    w.clearcoat = fmaxf(w.clearcoat, 0.0f);

    w.total = w.diffuse + w.diffuse_transmission + w.specular + w.transmission + w.clearcoat;
    if (!(w.total > 0.0f))
    {
        w.total = 1.0f;
        w.specular = 1.0f; // fallback to specular
        return w;
    }

    const float minPositiveWeight = w.total * (4.0f / 8388608.0f);
    if (w.diffuse > 0.0f)
        w.diffuse = fmaxf(w.diffuse, minPositiveWeight);
    if (w.diffuse_transmission > 0.0f)
        w.diffuse_transmission = fmaxf(w.diffuse_transmission, minPositiveWeight);
    if (w.specular > 0.0f)
        w.specular = fmaxf(w.specular, minPositiveWeight);
    if (w.transmission > 0.0f)
        w.transmission = fmaxf(w.transmission, minPositiveWeight);
    if (w.clearcoat > 0.0f)
        w.clearcoat = fmaxf(w.clearcoat, minPositiveWeight);
    w.total = w.diffuse + w.diffuse_transmission + w.specular + w.transmission + w.clearcoat;

    return w;
}

struct PbrLobeProbabilities
{
    float diffuse;
    float diffuseTransmission;
    float specular;
    float transmission;
    float clearcoat;
    unsigned int cdfDiffuse;
    unsigned int cdfDiffuseTransmission;
    unsigned int cdfSpecular;
    unsigned int cdfTransmission;
};

DEVICE_FUNC PbrLobeProbabilities pbr_lobe_probabilities(const THREAD_REF PbrLobeWeights& w, bool exiting)
{
    const float weights[5] = { exiting ? 0.0f : w.diffuse,
                               w.diffuse_transmission,
                               exiting ? 0.0f : w.specular,
                               w.transmission,
                               exiting ? 0.0f : w.clearcoat };
    const float total = exiting ? (w.diffuse_transmission + w.transmission) : w.total;
    unsigned int counts[5] = { 0u, 0u, 0u, 0u, 0u };
    unsigned int countSum = 0u;
    unsigned int largest = 0u;
    for (unsigned int i = 0u; i < 5u; ++i)
    {
        if (weights[i] > weights[largest])
        {
            largest = i;
        }
        if (weights[i] > 0.0f)
        {
            counts[i] = discreteFloatLatticeCount(weights[i] / total);
            countSum += counts[i];
        }
    }
    if (countSum < STRELKA_FLOAT_LATTICE_STATES)
    {
        counts[largest] += STRELKA_FLOAT_LATTICE_STATES - countSum;
    }
    else if (countSum > STRELKA_FLOAT_LATTICE_STATES)
    {
        counts[largest] -= countSum - STRELKA_FLOAT_LATTICE_STATES;
    }

    PbrLobeProbabilities p;
    p.diffuse = float(counts[0]) * 0x1p-23f;
    p.diffuseTransmission = float(counts[1]) * 0x1p-23f;
    p.specular = float(counts[2]) * 0x1p-23f;
    p.transmission = float(counts[3]) * 0x1p-23f;
    p.clearcoat = float(counts[4]) * 0x1p-23f;
    p.cdfDiffuse = counts[0];
    p.cdfDiffuseTransmission = p.cdfDiffuse + counts[1];
    p.cdfSpecular = p.cdfDiffuseTransmission + counts[2];
    p.cdfTransmission = p.cdfSpecular + counts[3];
    return p;
}

DEVICE_FUNC float pbr_fresnel_proposal(float physicalProbability)
{
    return discreteFloatLatticeProbability(physicalProbability);
}

struct PbrReflectionTerms
{
    float3 f;
    float pdf;
};

DEVICE_FUNC PbrReflectionTerms pbr_transmission_reflection(const THREAD_REF SurfaceInteraction& si,
                                                           float alpha,
                                                           float eta,
                                                           float3 Nf,
                                                           float3 H,
                                                           float NdotV_abs,
                                                           float NdotL_abs,
                                                           float NdotH,
                                                           float VdotH)
{
    PbrReflectionTerms r;
    r.f = make_float3(0.0f);
    r.pdf = 0.0f;

    const float weight = si.transmission * (1.0f - si.metallic);
    if (!(weight > 0.0f) || alpha < BSDF_DELTA_ALPHA)
    {
        return r;
    }
    if (!(NdotH > 0.0f) || !(VdotH > 0.0f) || !(NdotV_abs > 0.0f) || !(NdotL_abs > 0.0f))
    {
        return r;
    }

    // A thin wall splits Fresnel at the shading normal once, a solid interface
    // splits it at the microfacet. Same rule sample() applies, and reading the
    // interface differently in the two places is what the pdfs then disagree by.
    const bool deltaTransmission = !si.thin_walled && refraction_is_delta(si.ior, si.exterior_ior);
    const float cosSplit = (si.thin_walled || deltaTransmission) ? NdotV_abs : VdotH;
    const float3 F_film = transmission_fresnel(si, cosSplit, eta);
    const float sampleFresnel = pbr_fresnel_proposal(transmission_fresnel_probability(F_film));

    const float shape = ggx_ndf_visibility(alpha, Nf, H, NdotV_abs, NdotL_abs);
    r.f = make_float3(
        saturating_nonnegative_product(
            saturating_nonnegative_product(saturating_nonnegative_product(si.albedo.x, F_film.x), shape), weight),
        saturating_nonnegative_product(
            saturating_nonnegative_product(saturating_nonnegative_product(si.albedo.y, F_film.y), shape), weight),
        saturating_nonnegative_product(
            saturating_nonnegative_product(saturating_nonnegative_product(si.albedo.z, F_film.z), shape), weight));
    r.pdf = sampleFresnel * ggx_vndf_pdf(alpha, Nf, H, NdotV_abs, VdotH);
    return r;
}

DEVICE_FUNC PbrReflectionTerms pbr_reflection_terms(const THREAD_REF SurfaceInteraction& si,
                                                    const THREAD_REF PbrLobeProbabilities& probabilities,
                                                    float pTransEff,
                                                    float3 N,
                                                    float3 T,
                                                    float3 B,
                                                    float3 V,
                                                    float3 L,
                                                    float ax,
                                                    float ay,
                                                    float alpha,
                                                    float alphaCoat,
                                                    float3 F0)
{
    PbrReflectionTerms out;
    out.f = make_float3(0.0f);
    out.pdf = 0.0f;

    const float NdotV = dot(N, V);
    const float NdotL = dot(N, L);
    const bool exiting = NdotV <= 0.0f;
    // Oriented so the microfacet terms below are asked about the side the ray is
    // actually on. On an exit hit only the transmission lobe answers, but it
    // does answer: a ray inside glass can reflect back into it.
    const float3 Nf = exiting ? -N : N;
    const float NdotV_abs = fabsf(NdotV);
    const float NdotL_abs = fabsf(NdotL);

    const float3 H = reflection_half_vector(V, L, Nf);
    const float NdotH = dot(Nf, H);
    const float VdotH = dot(V, H);
    if (!(NdotH > 0.0f) || !(VdotH > 0.0f))
    {
        return out;
    }

    // Entering-side eta on a thin wall, for the reason given in sample().
    const float eta = (!exiting || si.thin_walled) ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);
    const PbrReflectionTerms tr = pbr_transmission_reflection(si, alpha, eta, Nf, H, NdotV_abs, NdotL_abs, NdotH, VdotH);
    out.f = tr.f;
    out.pdf = pTransEff * tr.pdf;

    if (exiting)
    {
        // The reflection lobes are defined against a front-facing frame and are
        // faded out on a transmissive material anyway; sample() skips them here
        // for the same reason.
        return out;
    }

    const float3 Hl = world_to_local(H, T, B, N);
    const float3 Vl = world_to_local(V, T, B, N);
    const float3 Ll = world_to_local(L, T, B, N);

    const float3 f_diffuse = diffuse_lobe_scale(si) * si.albedo * M_1_PI_F * (1.0f - si.metallic) *
                             (1.0f - si.transmission) * (1.0f - saturate(si.diffuse_transmission));

    float3 f_spec = make_float3(0.0f);
    float pdf_spec = 0.0f;
    if (!anisotropic_ggx_is_delta(ax, ay))
    {
        const float3 F = specular_fresnel(si, F0, VdotH);
        const float shape = ggx_ndf_visibility_aniso(ax, ay, Hl, Vl, Ll);
        const float3 compensation = ggx_energy_compensation(F0, si.roughness, NdotV);
        f_spec = make_float3(saturating_nonnegative_product(saturating_nonnegative_product(F.x, shape), compensation.x),
                             saturating_nonnegative_product(saturating_nonnegative_product(F.y, shape), compensation.y),
                             saturating_nonnegative_product(saturating_nonnegative_product(F.z, shape), compensation.z));
        pdf_spec = ggx_vndf_pdf_aniso(ax, ay, Hl, Vl);
    }

    float3 f_cc = make_float3(0.0f);
    float pdf_cc = 0.0f;
    if (si.clearcoat > 0.0f && alphaCoat >= BSDF_DELTA_ALPHA)
    {
        const float F_cc = fresnel_schlick_scalar(clearcoat_f0(si), VdotH);
        const float cc_brdf = saturating_nonnegative_product(F_cc, ggx_ndf_visibility(alphaCoat, N, H, NdotV, NdotL));
        f_cc = make_float3(saturating_nonnegative_product(si.clearcoat, cc_brdf));
        pdf_cc = ggx_vndf_pdf(alphaCoat, N, H, NdotV, VdotH);
    }

    const float3 f_sheen = sheen_brdf(si, NdotH, NdotL, NdotV);

    const float3 baseScale = specular_base_scale(si, F0, NdotV);
    const float3 coatScale = clearcoat_base_scale(si, NdotV, NdotL);
    const float specularScale = specular_lobe_scale(si);
    const float sheenScale = sheen_base_scale(si, NdotV);
    const float baseX = saturating_nonnegative_sum(saturating_nonnegative_product(f_diffuse.x, baseScale.x),
                                                   saturating_nonnegative_product(f_spec.x, specularScale));
    const float baseY = saturating_nonnegative_sum(saturating_nonnegative_product(f_diffuse.y, baseScale.y),
                                                   saturating_nonnegative_product(f_spec.y, specularScale));
    const float baseZ = saturating_nonnegative_sum(saturating_nonnegative_product(f_diffuse.z, baseScale.z),
                                                   saturating_nonnegative_product(f_spec.z, specularScale));
    const float surfaceX = saturating_nonnegative_sum(
        saturating_nonnegative_product(
            saturating_nonnegative_sum(saturating_nonnegative_product(baseX, coatScale.x), f_cc.x), sheenScale),
        f_sheen.x);
    const float surfaceY = saturating_nonnegative_sum(
        saturating_nonnegative_product(
            saturating_nonnegative_sum(saturating_nonnegative_product(baseY, coatScale.y), f_cc.y), sheenScale),
        f_sheen.y);
    const float surfaceZ = saturating_nonnegative_sum(
        saturating_nonnegative_product(
            saturating_nonnegative_sum(saturating_nonnegative_product(baseZ, coatScale.z), f_cc.z), sheenScale),
        f_sheen.z);
    out.f = make_float3(saturating_nonnegative_sum(out.f.x, surfaceX), saturating_nonnegative_sum(out.f.y, surfaceY),
                        saturating_nonnegative_sum(out.f.z, surfaceZ));

    const float p_diffuse = probabilities.diffuse;
    const float p_specular = probabilities.specular;
    const float p_clearcoat = probabilities.clearcoat;
    out.pdf = out.pdf + p_diffuse * cosine_hemisphere_pdf(NdotL) + p_specular * pdf_spec + p_clearcoat * pdf_cc;
    return out;
}

struct PbrDeltaReflectionTerms
{
    float3 numerator;
    float mass;
};

DEVICE_FUNC PbrDeltaReflectionTerms pbr_delta_reflection_terms(const THREAD_REF SurfaceInteraction& si,
                                                               const THREAD_REF PbrLobeProbabilities& probabilities,
                                                               float pTransEff,
                                                               float3 N,
                                                               float3 V,
                                                               bool baseSpecularDelta,
                                                               float alpha,
                                                               float alphaCoat,
                                                               float3 F0)
{
    PbrDeltaReflectionTerms out;
    out.numerator = make_float3(0.0f);
    out.mass = 0.0f;

    const float NdotV = dot(N, V);
    const bool exiting = NdotV <= 0.0f;
    const float NdotVAbs = fabsf(NdotV);

    const float transmissionWeight = si.transmission * (1.0f - si.metallic);
    if (alpha < BSDF_DELTA_ALPHA && transmissionWeight > 0.0f && pTransEff > 0.0f)
    {
        const float eta = (!exiting || si.thin_walled) ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);
        const float3 filmFresnel = transmission_fresnel(si, NdotVAbs, eta);
        out.mass = out.mass + pTransEff *
                                       pbr_fresnel_proposal(transmission_fresnel_probability(filmFresnel));
        out.numerator = out.numerator + si.albedo * filmFresnel * transmissionWeight;
    }

    if (exiting)
        return out;

    const float sheenScale = sheen_base_scale(si, NdotV);
    if (baseSpecularDelta && probabilities.specular > 0.0f)
    {
        const float3 base = specular_fresnel(si, F0, NdotV) * specular_lobe_scale(si) *
                            ggx_energy_compensation(F0, si.roughness, NdotV) * clearcoat_base_scale(si, NdotV, NdotV) *
                            sheenScale;
        out.mass = out.mass + probabilities.specular;
        out.numerator = out.numerator + base;
    }
    if (alphaCoat < BSDF_DELTA_ALPHA && probabilities.clearcoat > 0.0f)
    {
        const float coat = si.clearcoat * fresnel_schlick_scalar(clearcoat_f0(si), NdotV) * sheenScale;
        out.mass = out.mass + probabilities.clearcoat;
        out.numerator = out.numerator + make_float3(coat);
    }
    return out;
}

DEVICE_FUNC bool pbr_finish_delta_reflection(const THREAD_REF SurfaceInteraction& si,
                                             const THREAD_REF PbrLobeProbabilities& probabilities,
                                             float pTransEff,
                                             float3 N,
                                             float3 V,
                                             bool baseSpecularDelta,
                                             float alpha,
                                             float alphaCoat,
                                             float3 F0,
                                             THREAD_REF BsdfSampleResult& result)
{
    const float3 Nf = dot(N, V) > 0.0f ? N : -N;
    result.wi = reflect_dir(-V, Nf);
    const PbrDeltaReflectionTerms terms =
        pbr_delta_reflection_terms(si, probabilities, pTransEff, N, V, baseSpecularDelta, alpha, alphaCoat, F0);
    if (!(terms.mass > 0.0f))
        return false;
    result.pdf = terms.mass;
    result.bsdf_over_pdf = terms.numerator / terms.mass;
    result.event_type = BSDF_EVENT_SPECULAR_REFLECTION;
    return true;
}

DEVICE_FUNC bool pbr_finish_delta_transmission(const THREAD_REF SurfaceInteraction& si,
                                               float pTransEff,
                                               float fresnel,
                                               float3 filmFresnel,
                                               float etaFactor,
                                               THREAD_REF BsdfSampleResult& result)
{
    const float mass = pTransEff * (1.0f - fresnel);
    if (!(mass > 0.0f))
        return false;
    const float transmissionWeight = si.transmission * (1.0f - si.metallic);
    const float3 numerator = si.albedo * (make_float3(1.0f) - filmFresnel) * (etaFactor * transmissionWeight);
    result.pdf = mass;
    result.bsdf_over_pdf = numerator / mass;
    result.event_type = BSDF_EVENT_SPECULAR_TRANSMISSION;
    return true;
}

struct PbrPrepared;
struct PbrPrepared
{
    /// The normal actually shaded with -- flipped when an opaque surface is hit
    /// from behind, which is the one decision both halves have to agree on.
    float3 N;
    float3 T;
    float3 B;
    float3 F0;
    PbrLobeWeights w;
    PbrLobeProbabilities p;
    float NdotV;
    float alpha;
    float alpha_cc;
    float ax;
    float ay;
    /// NdotV <= 0: a ray on its way out of a dielectric. Only the transmission
    /// lobes describe such a hit.
    bool exiting;
};

DEVICE_FUNC PbrPrepared pbr_prepare(const THREAD_REF SurfaceInteraction& si)
{
    PbrPrepared prep;

    prep.N = si.shading_normal;
    const float3 V = si.wo;
    // An opaque surface hit from behind is the same surface seen from the front,
    // and is shaded as such rather than absorbed. See shading_frame.h; both
    // sample() and eval() used to make this call and had to agree about it.
    if (opaqueBackHitFlipsFrame(si.front_face, dot(prep.N, V), si.transmission, si.diffuse_transmission))
    {
        prep.N = -prep.N;
    }
    prep.NdotV = dot(prep.N, V);
    // A ray leaving a dielectric hits the far wall from behind, so the shading
    // normal points away from it. That is not a degenerate hit -- it is how light
    // gets out of a medium.
    prep.exiting = prep.NdotV <= 0.0f;

    prep.alpha = alpha_from_roughness(si.roughness);
    prep.alpha_cc = alpha_from_roughness(si.clearcoat_roughness);

    {
        const float3 Tp = si.tangent - prep.N * dot(prep.N, si.tangent);
        if (dot(Tp, Tp) > 1e-8f)
        {
            prep.T = safe_normalize(Tp);
            prep.B = cross(prep.N, prep.T);
        }
        else
        {
            build_onb(prep.N, prep.T, prep.B);
        }
    }
    // ax == ay when anisotropy is 0, and every *_aniso routine degenerates to
    // its isotropic form there, so isotropic materials are unchanged.
    anisotropic_alpha(si.roughness, si.anisotropy, prep.ax, prep.ay);

    // F0 for the specular lobe (mix between dielectric F0 and base colour for metals)
    prep.F0 = gltf_f0(si.ior, si.specular, si.specular_color, si.albedo, si.metallic);
    prep.w = pbr_lobe_weights(si);
    prep.p = pbr_lobe_probabilities(prep.w, prep.exiting);
    return prep;
}

DEVICE_FUNC PbrPrepared pbr_prepare_for(const THREAD_REF SurfaceInteraction& si)
{
    if (si.material_type != MATERIAL_TYPE_STANDARD_PBR)
    {
        PbrPrepared none = {};
        return none;
    }
    return pbr_prepare(si);
}

DEVICE_FUNC BsdfEvalResult standard_pbr_eval(const THREAD_REF SurfaceInteraction& si, float3 wi);
DEVICE_FUNC BsdfEvalResult
standard_pbr_eval(const THREAD_REF SurfaceInteraction& si, float3 wi, const THREAD_REF PbrPrepared& prep);

DEVICE_FUNC bool pbr_finish_continuous_sample(const THREAD_REF SurfaceInteraction& si,
                                              float absoluteCosine,
                                              THREAD_REF BsdfSampleResult& result,
                                              const THREAD_REF PbrPrepared& prep)
{
    // The same preparation the sample was drawn from. Evaluating through the
    // plain entry point here would prepare the vertex a second time inside every
    // continuous draw, which is most of them.
    const BsdfEvalResult evaluated = standard_pbr_eval(si, result.wi, prep);
    if (!(evaluated.pdf > 0.0f))
    {
        return false;
    }
    result.pdf = evaluated.pdf;
    result.bsdf_over_pdf = evaluated.bsdf * (absoluteCosine / evaluated.pdf);
    return true;
}

DEVICE_FUNC BsdfSampleResult
standard_pbr_sample(const THREAD_REF SurfaceInteraction& si,
                    float u1,
                    float u2,
                    unsigned int lobeWord,
                    unsigned int fresnelWord,
                    const THREAD_REF PbrPrepared& prep)
{
    BsdfSampleResult result;
    result.bsdf_over_pdf = make_float3(0.0f);
    result.pdf = 0.0f;
    result.event_type = BSDF_EVENT_ABSORB;

    // Named locals rather than prep.x at every use: what follows is the body
    // that computed these, unchanged, and renaming inside it is the kind of edit
    // that silently swaps two cosines.
    const PbrLobeWeights w = prep.w;
    const float3 N = prep.N;
    const float3 V = si.wo;
    const float NdotV = prep.NdotV;
    // Only the transmission lobe can describe an exit hit (it flips the normal
    // into Nf and picks eta by direction), so the reflection lobes are skipped
    // rather than evaluated against a back-facing normal.
    const bool exiting = prep.exiting;
    const float exit_transmission_total = w.diffuse_transmission + w.transmission;
    if (exiting && !(exit_transmission_total > 0.0f))
        return result;
    const PbrLobeProbabilities probabilities = prep.p;
    const float p_trans_eff = probabilities.transmission;

    const float alpha = prep.alpha;
    const float alpha_cc = prep.alpha_cc;
    const float3 T = prep.T;
    const float3 B = prep.B;
    const float ax = prep.ax;
    const float ay = prep.ay;
    const bool baseSpecularDelta = anisotropic_ggx_is_delta(ax, ay);
    const float3 F0 = prep.F0;

    if (!exiting && lobeWord < probabilities.cdfDiffuse)
    {
        // ===== DIFFUSE LOBE ===============================================
        const float3 wi_local = cosine_hemisphere_sample(u1, u2);
        result.wi = local_to_world(wi_local, T, B, N);

        const float NdotL = dot(N, result.wi);
        if (NdotL <= 0.0f)
            return result;

        if (!pbr_finish_continuous_sample(si, NdotL, result, prep))
            return result;
        result.event_type = BSDF_EVENT_DIFFUSE_REFLECTION;
    }
    else if (lobeWord < probabilities.cdfDiffuseTransmission)
    {
        const float3 Nt = (NdotV > 0.0f) ? -N : N;
        float3 Tt, Bt;
        build_onb(Nt, Tt, Bt);
        const float3 wi_local = cosine_hemisphere_sample(u1, u2);
        result.wi = local_to_world(wi_local, Tt, Bt, Nt);

        const float NdotL_t = dot(Nt, result.wi);
        if (NdotL_t <= 0.0f)
            return result;

        // Diffuse and rough specular transmission overlap on this hemisphere.
        // The selected component does not own the direction: finish the sample
        // with their full marginal f and solid-angle density.
        if (!pbr_finish_continuous_sample(si, NdotL_t, result, prep))
            return result;
        result.event_type = BSDF_EVENT_DIFFUSE_TRANSMISSION;
    }
    else if (!exiting && lobeWord < probabilities.cdfSpecular)
    {
        // ===== SPECULAR LOBE ==============================================
        if (baseSpecularDelta)
        {
            pbr_finish_delta_reflection(
                si, probabilities, p_trans_eff, N, V, baseSpecularDelta, alpha, alpha_cc, F0, result);
            return result;
        }

        const float3 V_local = world_to_local(V, T, B, N);
        const float3 H_local = ggx_vndf_sample_aniso(V_local, ax, ay, u1, u2);
        const float3 H = local_to_world(H_local, T, B, N);
        const float VdotH = dot(V, H);
        if (VdotH <= 0.0f)
            return result;

        result.wi = reflect_dir(-V, H);
        const float NdotL = dot(N, result.wi);
        if (NdotL <= 0.0f)
            return result;

        if (!pbr_finish_continuous_sample(si, NdotL, result, prep))
            return result;
        result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
    }
    else if (exiting || lobeWord < probabilities.cdfTransmission)
    {
        // ===== TRANSMISSION LOBE ==========================================
        const bool entering = NdotV > 0.0f;
        const float3 Nf = entering ? N : -N;
        const float eta = (entering || si.thin_walled) ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);
        const bool is_smooth = (alpha < BSDF_DELTA_ALPHA);

        if (si.thin_walled)
        {
            const float NdotV_abs = fabsf(NdotV);
            const float3 F_film = transmission_fresnel(si, NdotV_abs, eta);
            const float sampleFresnel = pbr_fresnel_proposal(transmission_fresnel_probability(F_film));

            if (discreteFloatLatticeBernoulli(fresnelWord, sampleFresnel))
            {
                if (is_smooth)
                {
                    pbr_finish_delta_reflection(si, probabilities, p_trans_eff, N, V, baseSpecularDelta, alpha,
                                                alpha_cc, F0, result);
                    return result;
                }

                const float3 V_local = world_to_local(V, T, B, Nf);
                const float3 H_local = ggx_vndf_sample(V_local, alpha, u1, u2);
                const float3 H = local_to_world(H_local, T, B, Nf);
                const float VdotH = dot(V, H);
                if (VdotH <= 0.0f)
                    return result;

                result.wi = reflect_dir(-V, H);
                const float NdotL = dot(Nf, result.wi);
                if (NdotL <= 0.0f)
                    return result;

                if (!pbr_finish_continuous_sample(si, NdotL, result, prep))
                    return result;
                result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
                return result;
            }

            // Transmission: smooth is exactly -V; rough is a GGX reflection of
            // the view mirrored through the surface (Cycles / OpenPBR).
            const float eta_rel = fmaxf(si.ior / fmaxf(si.exterior_ior, 1e-4f), 1.0f);
            const float alpha_t = thin_glass_transmission_alpha(alpha, eta_rel);
            const bool t_smooth = (alpha_t < BSDF_DELTA_ALPHA);

            float3 H_t = Nf;
            if (!t_smooth)
            {
                const float3 V_local_t = world_to_local(V, T, B, Nf);
                const float3 H_local_t = ggx_vndf_sample(V_local_t, alpha_t, u1, u2);
                H_t = local_to_world(H_local_t, T, B, Nf);
                if (dot(V, H_t) <= 0.0f)
                    return result;
            }

            const float3 wi_r = reflect_dir(-V, H_t);
            result.wi = flip_through_surface(wi_r, Nf);

            if (t_smooth)
            {
                pbr_finish_delta_transmission(si, p_trans_eff, sampleFresnel, F_film, 1.0f, result);
            }
            else
            {
                const float NdotL_r = dot(Nf, wi_r);
                if (NdotL_r <= 0.0f)
                    return result;
                if (!pbr_finish_continuous_sample(si, fabsf(dot(Nf, result.wi)), result, prep))
                    return result;
                result.event_type = BSDF_EVENT_GLOSSY_TRANSMISSION;
            }
            return result;
        }

        float3 H;
        if (is_smooth)
        {
            H = Nf;
        }
        else
        {
            const float3 V_local = world_to_local(V, T, B, Nf);
            const float3 H_local = ggx_vndf_sample(V_local, alpha, u1, u2);
            H = local_to_world(H_local, T, B, Nf);
        }

        const float VdotH = dot(V, H);
        if (VdotH <= 0.0f)
            return result;

        const bool deltaTransmission = refraction_is_delta(si.ior, si.exterior_ior);
        const float fresnelCosine = deltaTransmission ? fabsf(NdotV) : VdotH;
        const float3 F_film = transmission_fresnel(si, fresnelCosine, eta);
        const float sampleFresnel = pbr_fresnel_proposal(transmission_fresnel_probability(F_film));

        if (discreteFloatLatticeBernoulli(fresnelWord, sampleFresnel))
        {
            // Specular reflection within transmission lobe
            result.wi = reflect_dir(-V, H);
            const float NdotL = dot(Nf, result.wi);
            if (NdotL <= 0.0f)
                return result;

            if (is_smooth)
            {
                pbr_finish_delta_reflection(si, probabilities, p_trans_eff, N, V, baseSpecularDelta, alpha, alpha_cc,
                                            F0, result);
            }
            else
            {
                // Rough: the same shared evaluation the reflection lobes and
                // eval() use, so this direction has one density rather than the
                // transmission lobe's own share reported as the whole of it.
                if (!pbr_finish_continuous_sample(si, NdotL, result, prep))
                    return result;
                result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
            }
        }
        else
        {
            if (deltaTransmission)
            {
                result.wi = -V;
                pbr_finish_delta_transmission(si, p_trans_eff, sampleFresnel, F_film, eta * eta, result);
                return result;
            }

            // Solid refraction
            float3 wi_refracted;
            const bool valid = refract_dir(-V, H, eta, VdotH, wi_refracted);
            if (!valid)
            {
                // The Fresnel test makes this branch unreachable for exact TIR,
                // but keep the numerical fallback in the same measure as the H
                // that generated it.
                result.wi = reflect_dir(-V, H);
                if (is_smooth)
                {
                    pbr_finish_delta_reflection(si, probabilities, p_trans_eff, N, V, baseSpecularDelta, alpha,
                                                alpha_cc, F0, result);
                }
                else
                {
                    const float NdotL = dot(Nf, result.wi);
                    if (!pbr_finish_continuous_sample(si, NdotL, result, prep))
                        return result;
                    result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
                }
                return result;
            }

            result.wi = safe_normalize(wi_refracted);
            const float signedNdotL = dot(Nf, result.wi);
            if (!(signedNdotL < 0.0f))
                return result;

            if (is_smooth)
            {
                const float factor = eta * eta;
                pbr_finish_delta_transmission(si, p_trans_eff, sampleFresnel, F_film, factor, result);
            }
            else
            {
                if (!pbr_finish_continuous_sample(si, -signedNdotL, result, prep))
                    return result;
                result.event_type = BSDF_EVENT_GLOSSY_TRANSMISSION;
            }
        }
    }
    else
    {
        // ===== CLEARCOAT LOBE =============================================
        if (alpha_cc < BSDF_DELTA_ALPHA)
        {
            pbr_finish_delta_reflection(
                si, probabilities, p_trans_eff, N, V, baseSpecularDelta, alpha, alpha_cc, F0, result);
            return result;
        }

        const float3 V_local = world_to_local(V, T, B, N);
        const float3 H_local = ggx_vndf_sample(V_local, alpha_cc, u1, u2);
        const float3 H = local_to_world(H_local, T, B, N);
        const float VdotH = dot(V, H);
        if (VdotH <= 0.0f)
            return result;

        result.wi = reflect_dir(-V, H);
        const float NdotL = dot(N, result.wi);
        if (NdotL <= 0.0f)
            return result;

        if (!pbr_finish_continuous_sample(si, NdotL, result, prep))
            return result;
        result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
    }

    return result;
}

DEVICE_FUNC BsdfSampleResult standard_pbr_sample(
    const THREAD_REF SurfaceInteraction& si, float u1, float u2, unsigned int lobeWord, unsigned int fresnelWord)
{
    return standard_pbr_sample(si, u1, u2, lobeWord, fresnelWord, pbr_prepare(si));
}

DEVICE_FUNC BsdfSampleResult
standard_pbr_sample(const THREAD_REF SurfaceInteraction& si, float u1, float u2, float uLobe, float uFresnel)
{
    return standard_pbr_sample(
        si, u1, u2, discreteFloatLatticeWord(uLobe), discreteFloatLatticeWord(uFresnel));
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfEvalResult
standard_pbr_eval(const THREAD_REF SurfaceInteraction& si, float3 wi, const THREAD_REF PbrPrepared& prep)
{
    BsdfEvalResult result;
    result.bsdf = make_float3(0.0f);
    result.pdf = 0.0f;

    // The same frame sample() shades in, taken from the same preparation rather
    // than derived a second time -- which is also what guarantees the two cannot
    // drift apart and have MIS blend two different BRDFs.
    const float3 N = prep.N;
    const float3 V = si.wo;

    {
        const float cosNsI = dot(si.bump_normal, V);
        const float cosNsN = dot(si.bump_normal, N);
        const float cosNI = dot(N, V);
        if (cosNsI * cosNsN * cosNI < 0.0f)
        {
            return result;
        }
    }

    const float NdotV = prep.NdotV;
    const float NdotL = dot(N, wi);

    const bool exiting = prep.exiting;

    const float alpha = prep.alpha;
    const float alpha_cc = prep.alpha_cc;

    // Same tangent frame and axis split as standard_pbr_sample(), now by
    // construction: one preparation feeds both, so eval and sample cannot
    // describe two different BRDFs for MIS to blend.
    const float3 T = prep.T;
    const float3 B = prep.B;
    const float ax = prep.ax;
    const float ay = prep.ay;

    const PbrLobeWeights w = prep.w;
    const float exit_transmission_total = w.diffuse_transmission + w.transmission;
    if (exiting && !(exit_transmission_total > 0.0f))
        return result;
    // Same conditional selection PMFs as sample(): on an exit hit the two
    // transmission proposals are renormalized after incompatible reflection
    // lobes are removed.
    const PbrLobeProbabilities probabilities = prep.p;
    const float p_diffuse_tr_eff = probabilities.diffuseTransmission;
    const float p_trans_eff = probabilities.transmission;

    const float3 F0 = prep.F0;

    const bool is_reflection = ((NdotL > 0.0f) == (NdotV > 0.0f));

    if (is_reflection)
    {
        // One shared evaluation, so this cannot drift from what sample()
        // reports for the same direction. p_trans_eff is what sample() would
        // apply on this side.
        const PbrReflectionTerms terms =
            pbr_reflection_terms(si, probabilities, p_trans_eff, N, T, B, V, wi, ax, ay, alpha, alpha_cc, F0);
        result.bsdf = terms.f;
        result.pdf = terms.pdf;
    }
    else
    {
        const float dt = saturate(si.diffuse_transmission);
        if (dt > 0.0f)
        {
            const float3 Nt = (NdotV > 0.0f) ? -N : N;
            const float NdotL_t = dot(Nt, wi);
            if (NdotL_t > 0.0f)
            {
                // The same share the reflected diffuse lobe gives up; see the
                // note at the matching site in standard_pbr_sample().
                result.bsdf = si.diffuse_transmission_color * M_1_PI_F * (1.0f - si.metallic) *
                              (1.0f - si.transmission) * dt * specular_base_scale(si, F0, NdotV);
                result.pdf = p_diffuse_tr_eff * cosine_hemisphere_pdf(NdotL_t);
            }
        }

        // Delta transmission cannot be evaluated. A smooth thin wall is still a
        // delta (exactly -V); a rough one is a GGX lobe about the mirrored view
        // and is evaluated below the same way sample() produces it.
        const float eta_rel_eval = fmaxf(si.ior / fmaxf(si.exterior_ior, 1e-4f), 1.0f);
        const float alpha_t_eval = si.thin_walled ? thin_glass_transmission_alpha(alpha, eta_rel_eval) : alpha;
        if ((si.thin_walled ? alpha_t_eval : alpha) < BSDF_DELTA_ALPHA)
            return result;

        if (si.transmission <= 0.0f)
            return result;

        const bool entering = NdotV > 0.0f;
        const float3 Nf = entering ? N : -N;
        const float NdotV_abs = fabsf(NdotV);
        const float NdotL_abs = fabsf(NdotL);
        // Entering-side eta on a thin wall, for the reason given in sample().
        // MIS weighs this density against the one sample() wrote, so the two
        // have to read the interface the same way or the weights do not sum.
        const float eta = (entering || si.thin_walled) ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);

        if (si.thin_walled)
        {
            // Map wi back to the reflection hemisphere and evaluate the GGX
            // BRDF sample() used, with the transmission roughness. Fresnel is
            // the same macro split sample() used, not a microfacet F.
            const float3 wi_r = flip_through_surface(wi, Nf);
            const float NdotL_r = dot(Nf, wi_r);
            if (NdotL_r <= 0.0f)
                return result;

            const float3 H = reflection_half_vector(V, wi_r, Nf);
            const float NdotH = dot(Nf, H);
            const float VdotH = dot(V, H);
            if (NdotH <= 0.0f || VdotH <= 0.0f)
                return result;

            const float3 F_film = transmission_fresnel(si, NdotV_abs, eta);
            const float sampleFresnel = pbr_fresnel_proposal(transmission_fresnel_probability(F_film));
            const float shape = ggx_ndf_visibility(alpha_t_eval, Nf, H, NdotV_abs, NdotL_r);
            const float3 brdf = make_float3(saturating_nonnegative_product(1.0f - F_film.x, shape),
                                            saturating_nonnegative_product(1.0f - F_film.y, shape),
                                            saturating_nonnegative_product(1.0f - F_film.z, shape));
            const float weight = (1.0f - si.metallic) * si.transmission;
            result.bsdf = make_float3(
                saturating_nonnegative_sum(
                    result.bsdf.x, saturating_nonnegative_product(
                                       saturating_nonnegative_product(si.albedo.x, fmaxf(brdf.x, 0.0f)), weight)),
                saturating_nonnegative_sum(
                    result.bsdf.y, saturating_nonnegative_product(
                                       saturating_nonnegative_product(si.albedo.y, fmaxf(brdf.y, 0.0f)), weight)),
                saturating_nonnegative_sum(
                    result.bsdf.z, saturating_nonnegative_product(
                                       saturating_nonnegative_product(si.albedo.z, fmaxf(brdf.z, 0.0f)), weight)));
            result.pdf =
                result.pdf + p_trans_eff * (1.0f - sampleFresnel) * ggx_vndf_pdf(alpha_t_eval, Nf, H, NdotV_abs, VdotH);
            return result;
        }

        // At equal represented indices every microfacet refracts to exactly
        // -V. That is one atom (possibly alongside a rough iridescent reflection
        // lobe), not a solid-angle density.
        if (refraction_is_delta(si.ior, si.exterior_ior))
            return result;

        // eta_i * V + eta_t * wi, normalised -- see refraction_half_vector().
        InterfaceCosine robustVdotH = makeInterfaceCosine(0.0f);
        const float3 H = refraction_half_vector(V, wi, eta, Nf, robustVdotH);

        const float NdotH = dot(Nf, H);
        const float VdotH = saturate(interfaceCosineValue(robustVdotH));
        const float LdotH = dot(wi, H);

        if (!(NdotH >= 0.0f) || VdotH <= 0.0f)
            return result;

        const float3 F_film = transmission_fresnel(si, robustVdotH, eta);
        const float sampleFresnel = pbr_fresnel_proposal(transmission_fresnel_probability(F_film));
        const float denom = refraction_residual_length(V, wi, eta);
        const float denomSquared = denom * denom;
        if (!(denomSquared > 0.0f))
            return result;
        float transmissionScale = fabsf(VdotH);
        const float etaSquared = saturating_nonnegative_product(eta, eta);
        transmissionScale =
            saturating_nonnegative_product(transmissionScale, saturating_nonnegative_product(4.0f, etaSquared));
        transmissionScale = saturating_nonnegative_product(transmissionScale, refraction_jacobian(V, wi, eta, LdotH));
        const float transmissionShape =
            saturating_nonnegative_product(transmissionScale, ggx_ndf_visibility(alpha, Nf, H, NdotV_abs, NdotL_abs));
        const float3 btdf = make_float3(saturating_nonnegative_product(1.0f - F_film.x, transmissionShape),
                                        saturating_nonnegative_product(1.0f - F_film.y, transmissionShape),
                                        saturating_nonnegative_product(1.0f - F_film.z, transmissionShape));

        // Accumulated, not assigned: a material can be both diffusely and
        // specularly transmissive, and the diffuse term above has already
        // written into the same hemisphere.
        const float3 btdf_pos = make_float3(fmaxf(btdf.x, 0.0f), fmaxf(btdf.y, 0.0f), fmaxf(btdf.z, 0.0f));
        const float weight = (1.0f - si.metallic) * si.transmission;
        result.bsdf = make_float3(
            saturating_nonnegative_sum(
                result.bsdf.x,
                saturating_nonnegative_product(saturating_nonnegative_product(si.albedo.x, btdf_pos.x), weight)),
            saturating_nonnegative_sum(
                result.bsdf.y,
                saturating_nonnegative_product(saturating_nonnegative_product(si.albedo.y, btdf_pos.y), weight)),
            saturating_nonnegative_sum(
                result.bsdf.z,
                saturating_nonnegative_product(saturating_nonnegative_product(si.albedo.z, btdf_pos.z), weight)));

        // The same pair standard_pbr_sample() applies; see the note there.
        const float dwh_dwi = refraction_jacobian(V, wi, eta, LdotH);
        const float pdf_h = ggx_vndf_pdf_half(alpha, Nf, H, NdotV_abs, VdotH);
        const float transmissionPdf = saturating_nonnegative_product(
            saturating_nonnegative_product(saturating_nonnegative_product(p_trans_eff, 1.0f - sampleFresnel), pdf_h),
            dwh_dwi);
        result.pdf = saturating_nonnegative_sum(result.pdf, transmissionPdf);
    }

    return result;
}

// ---------------------------------------------------------------------------
// PDF only
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfEvalResult standard_pbr_eval(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    return standard_pbr_eval(si, wi, pbr_prepare(si));
}

DEVICE_FUNC float standard_pbr_pdf(const THREAD_REF SurfaceInteraction& si, float3 wi, const THREAD_REF PbrPrepared& prep)
{
    const BsdfEvalResult r = standard_pbr_eval(si, wi, prep);
    return r.pdf;
}

DEVICE_FUNC float standard_pbr_pdf(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    const BsdfEvalResult r = standard_pbr_eval(si, wi);
    return r.pdf;
}

#endif // STRELKA_BXDF_STANDARD_PBR_H

// NOLINTEND(cppcoreguidelines-pro-type-member-init, cppcoreguidelines-init-variables)
