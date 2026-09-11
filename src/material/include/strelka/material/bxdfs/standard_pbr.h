#ifndef STRELKA_BXDF_STANDARD_PBR_H
#    define STRELKA_BXDF_STANDARD_PBR_H

// ============================================================================
// bxdfs/standard_pbr.h -- glTF metallic-roughness PBR material
//
// Implements the standard PBR model as described in the glTF 2.0 spec with
// the following lobes:
//
//   1. Diffuse (Lambertian) -- weighted by (1 - metallic) * (1 - transmission)
//   2. Specular reflection (GGX Cook-Torrance)
//   3. Transmission (refraction through rough/smooth dielectric)
//   4. Clearcoat (additional GGX layer with fixed IOR = 1.5)
//
// Lobe selection is done stochastically: we choose one lobe proportional to
// its approximate weight and divide out the selection probability.
// ============================================================================

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
//
// Device-shared header: NVCC and the Metal compiler read this too, and
// clang-tidy only ever sees the host build, so these two suggestions cannot be
// taken here. Initialising the locals means a dead store in a BSDF inner loop --
// they are out-parameters written on the next line -- and the fixer spells the
// initialiser NAN, which needs <math.h>, which Metal rejects outright. Default
// member initialisers do the same to structs that are memcpy'd to the GPU.
// Suppressed rather than left to warn because these repeat in every translation
// unit that includes the header, and 700 lines of unactionable output per build
// is how the handful that matter get skipped.

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

// The specular lobe's Fresnel, with a thin film over it when the material has
// one. Everything the film changes is here: it replaces the reflectance and
// leaves the distribution and the shadowing alone, which is what makes it a
// property of the interface rather than of the microsurface.
DEVICE_FUNC float3 specular_fresnel(const THREAD_REF SurfaceInteraction& si, float3 F0, float v_dot_h)
{
    const float3 base = fresnel_schlick(F0, v_dot_h);
    if (si.iridescence <= 0.0f)
    {
        return base;
    }
    // At the microfacet, not at the shading normal. The glTF reference evaluates
    // the film once per shading point against NdotV; inside a microfacet BRDF the
    // angle the Fresnel is taken at is VdotH, and using anything else makes the
    // film disagree with the lobe it is modifying.
    const float3 film = iridescence_fresnel(1.0f, si.iridescence_ior, v_dot_h, si.iridescence_thickness, F0);
    return mix(base, film, saturate(si.iridescence));
}

// How much of the separate specular lobe survives a transmissive material.
//
// Zero for glass, and the same factor pbr_lobe_weights uses to zero the lobe's
// selection probability. The transmission lobe runs its own Fresnel and produces
// reflection events itself, so a second specular lobe would double-count -- which
// is why the weight was already scaled this way. The *BRDF* was not, and that is
// worse than double-counting: a lobe evaluated into f_total whose selection
// probability is zero is divided by a pdf that does not include it. On a smooth
// coated bubble the coat lobe is the only one selected, and the specular term
// rides along at 1/0.19 of its proper weight.
//
// What that looked like: soap bubbles that glowed instead of being transparent.
// Adding thin-film interference did not cause it, it coloured it -- the same
// over-count had been shipping as a white halo.
DEVICE_FUNC float specular_lobe_scale(const THREAD_REF SurfaceInteraction& si)
{
    return 1.0f - si.transmission * (1.0f - si.metallic);
}

// Whether the diffuse lobe has a response to give at all.
//
// Zero once the normal map has turned the shading normal past the viewer and
// valid_reflection.h has corrected it: the correction exists for the lobes that
// reflect, and the diffuse one is defined by the normal the map asked for, which
// faces away. Cycles reaches the same place from the other side -- it corrects
// its glossy closures only, and its diffuse closure returns nothing for a normal
// behind the view ray.
//
// Exactly one on every hit that needed no correction, so this is a no-op for all
// but a handful of grazing pixels, and the ladder does not move for it.
DEVICE_FUNC float diffuse_lobe_scale(const THREAD_REF SurfaceInteraction& si)
{
    return si.diffuse_faces_away ? 0.0f : 1.0f;
}

// Reflectance at a transmissive interface, coloured when a thin film sits on it.
//
// The film is applied in the specular lobe, and a transmissive material does not
// have one: its reflection is the transmission lobe's own Fresnel coin flip. So
// a soap bubble -- the thing thin-film interference exists to render -- got no
// film at all, and came out with a black rim where the reference has a bright
// iridescent one.
//
// Returned as a colour with the scalar the coin flip uses left alone, so the
// sampling is unchanged and the tint rides on the throughput. With no film the
// colour is that same scalar and both correction factors are exactly one.
DEVICE_FUNC float3 transmission_fresnel(const THREAD_REF SurfaceInteraction& si, CompensatedFloat v_dot_h, float eta)
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
    const float cosine = compensatedValue(v_dot_h);
    const float3 film =
        iridescence_fresnel(1.0f, si.iridescence_ior, fabsf(cosine), si.iridescence_thickness, make_float3(f));
    const float3 result = mix(make_float3(f), film, saturate(si.iridescence));
    return make_float3(saturate(result.x), saturate(result.y), saturate(result.z));
}

DEVICE_FUNC float3 transmission_fresnel(const THREAD_REF SurfaceInteraction& si, float v_dot_h, float eta)
{
    return transmission_fresnel(si, compensatedSum(v_dot_h, 0.0f), eta);
}

// A coloured Fresnel response needs a scalar branch probability, but that
// probability is a proposal rather than the response itself. Luminance gives a
// low-variance split; the endpoint guards preserve both outcomes whenever any
// colour channel has support on the renderer's finite random lattice.
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
    // Normalised by its own directional albedo where that exceeds 1. Ashikhmin's
    // visibility term does not conserve energy -- it reaches 2.78 at low
    // roughness and grazing incidence -- so the raw lobe returns more light than
    // arrived, which is a glowing towel rather than a shiny one.
    const float e = sheen_albedo(fabsf(n_dot_v), si.sheen_roughness);
    const float norm = (e > 1.0f) ? (1.0f / e) : 1.0f;
    return si.sheen_color * (si.sheen * norm * sheen_d_charlie(alpha, n_dot_h) * sheen_v_ashikhmin(n_dot_l, n_dot_v));
}

// The coat's reflectance at normal incidence. KHR_materials_clearcoat fixes this
// at 0.04, i.e. a clear lacquer; a DCC that lets an artist author the coat's IOR
// means something else by a "coat" and the difference is not subtle -- a coat at
// IOR 2.0 reflects 11% head-on rather than 4%.
DEVICE_FUNC float clearcoat_f0(const THREAD_REF SurfaceInteraction& si)
{
    return f0_from_ior(fmaxf(si.clearcoat_ior, 1.0f));
}

// What survives under the coat: enter, bounce on the base, leave -- and the
// geometric series of bounces between the base and the coat's underside.
//
// The coat used to be added on top with nothing taken away, which makes a glazed
// ceramic brighter than the light falling on it. Taking (1-F_L)*(1-F_V) out
// without giving the series back then left scenes/feature_tests/15_clearcoat
// 6% dark at IOR 2.2 while matching at IOR 1.0 -- the signature of a missing
// term that scales with the coat's reflectance.
//
// Per channel, against ½(F_L+F_V). A hemispherical F_avg is larger than F at
// normal incidence, so dividing (1-F0)^2 by (1-F_avg ρ) is what pushed a white
// ceramic to 1.02; the internal average (~0.6 with TIR) pushed it to 2.43.
// Both are the wrong Fresnel for a model that never refracts L and V into the
// coat. The ceiling at (1-F_ms) is what keeps a base whose albedo we have
// under-counted -- specular sits under the coat too -- from climbing past one.
// What survives under the specular layer, the way clearcoat_base_scale() answers
// it for the coat. The lobe used to be summed on top of a full diffuse one, so a
// rough dielectric returned more light than fell on it -- docs/open-defects.md,
// Closed, for the measurements.
//
// Against the layer's directional albedo rather than a Fresnel at one angle:
// Schlick with F0 = 0 has a (1-cos)^5 tail reaching one, so a (1-F) complement
// drains a material that has no specular lobe at all. ggx_specular_albedo() is
// the quantity that goes to zero when the lobe does.
//
// View side only, because the model never refracts L into the layer.
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
    // Twice, because the light crosses the coat twice: in along L and out along
    // V. Scaling by the view-side Fresnel alone -- which is what the glTF sample
    // viewer does -- still let a glazed white ceramic reach 1.06 directional
    // albedo, measured in tests/material/test_clearcoat.cpp.
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

// OpenPBR / Cycles thin-glass transmission roughness. Two refraction events
// widen the lobe; the scale is from Kulla Conty (Imageworks 2017, p.40 -- the
// slides say 3.7, the Cycles port and the algebra say 3.4). `eta` is n_glass /
// n_air, at least one.
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

// How much of the base layer survives under the sheen, per KHR_materials_sheen.
// What the fabric reflected is not available to the lobes beneath it; leaving
// this out is what made an additive sheen measure 1.40 directional albedo on a
// plain white cloth.
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

    // The diffuse lobe splits rather than grows: KHR_materials_diffuse_transmission
    // defines the result as mix(diffuse_brdf, diffuse_btdf, weight), so what goes
    // through is what no longer comes back, and a leaf cannot reflect and
    // transmit its way past the energy that hit it.
    const float dt = saturate(si.diffuse_transmission);
    const float diffuse_base = dielectric_weight * (1.0f - si.transmission);

    w.diffuse = diffuse_base * (1.0f - dt) * luminance(si.albedo) * diffuse_lobe_scale(si);
    w.diffuse = fmaxf(w.diffuse, 0.0f);

    // Sheen rides the cosine-sampled lobe instead of getting one of its own.
    // Charlie has no cheap invertible sampling routine, a cosine hemisphere
    // covers its support, and sharing the selection probability keeps
    // combined_pdf a single cosine term in every branch below. What sharing does
    // require is that the lobe stays reachable on a dark fabric, which the max
    // guarantees; for a material without sheen this is exactly a no-op.
    const float sheen_lum = si.sheen * luminance(si.sheen_color);
    w.diffuse = fmaxf(w.diffuse, sheen_lum * diffuse_lobe_scale(si));

    w.diffuse_transmission = diffuse_base * dt * luminance(si.diffuse_transmission_color);
    w.diffuse_transmission = fmaxf(w.diffuse_transmission, 0.0f);

    // For specular, use the approximate Fresnel reflectance at normal incidence.
    //
    // Scaled by (1 - transmission) for the same reason diffuse is: the
    // transmission lobe runs its own Fresnel and reflects internally, so leaving
    // a separate specular lobe alive on a transmissive material makes the
    // reflected direction reachable by two strategies whose pdfs each ignore the
    // other. Neither over-counts on its own; together they do, and smooth glass
    // came out about 9% too bright.
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

    // The shipped GPU generators return k/2^23. A positive interval narrower
    // than that can disappear entirely after the float CDF additions, leaving a
    // lobe that eval() sees but sample() cannot select. Four lattice cells leave
    // room for those additions to round while changing only the proposal, never
    // the material response. Exact represented categorical masses are unified in
    // finding R; this guard establishes support in the meantime.
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

// Apportion the five proposal weights onto the exact 23-bit categorical
// lattice consumed by production BSDF sampling. Every positive lobe keeps at
// least one state, zero lobes keep none, and the counts sum exactly to the
// lattice size. The float probabilities are exact dyadics and are the same
// values eval() places in the marginal PDF.
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

// ---------------------------------------------------------------------------
// The reflection hemisphere, evaluated once
//
// f and the combined sampling density for a direction on the same side of the
// surface as the view vector. Every lobe that can produce such a direction
// contributes to both, whichever lobe the sampler happened to pick -- that is
// what makes bsdf_sample() and bsdf_eval() two descriptions of one BRDF, and
// what the MIS weights on either side of the estimate assume.
//
// This used to be written out four times: once in each of the diffuse, specular
// and clearcoat branches of standard_pbr_sample(), and once in
// standard_pbr_eval(). tests/material/test_sample_eval_consistency.cpp exists
// because of that duplication. The transmission lobe's Fresnel reflection was
// the copy that never got made: sample() produced those directions from inside
// the transmission branch and reported only that branch's density, while eval()
// left the term out of f and out of the pdf altogether. On glass the two did not
// merely disagree -- eval() returned pdf 0 for 100% of the reflections sample()
// generated, so next-event estimation could not see a rough glass reflection at
// all while the light hit still deducted a MIS share for it. At transmission
// 0.5 the surviving disagreement was up to 17000x on the directions the two do
// share.
// ---------------------------------------------------------------------------
struct PbrReflectionTerms
{
    float3 f;
    float pdf;
};

/// The transmission lobe's Fresnel reflection, as a BRDF and as the density it
/// contributes *within* that lobe (the caller scales by the lobe's selection
/// probability).
///
/// Weighted by transmission * (1 - metallic), which is exactly the complement of
/// specular_lobe_scale(): the separate specular lobe is faded out by the same
/// factor as a transmissive material takes over its own reflection, so the two
/// sum to one reflection rather than double counting.
///
/// A delta interface has no density and is left at zero: sample() reports a
/// discrete probability for it and eval() correctly returns nothing.
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

/// f and the combined pdf for `L`, which must be in the same hemisphere as V.
///
/// `pTransEff` is the transmission lobe's selection probability as the sampler
/// applies it -- 1 on an exit hit, where that lobe is taken unconditionally.
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

// All smooth reflective components land on the same macro mirror direction.
// Their lobe identities are latent variables, not different path-space events,
// so a selected component must return the sum of every compatible discrete mass
// and the sum of their physical atom coefficients.
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

// Forward-declared so every non-delta sample can be finished from the same full
// marginal value and density that the light-sampling side evaluates. This is
// intentionally not used for delta events: their `pdf` field is discrete mass,
// whereas standard_pbr_eval() returns only a solid-angle density.
struct PbrPrepared;
// ---------------------------------------------------------------------------
// Prepare once per vertex
//
// sample() and eval() open with the same twenty-odd lines: flip the frame for an
// opaque back hit, take NdotV, derive the tangent frame, split the roughness
// into ax/ay, weigh the five lobes and apportion them onto the 23-bit
// categorical lattice. A vertex ran that prologue three times or more --
// bsdf_has_smooth_lobe() before next-event estimation, bsdf_eval() for the
// connection, bsdf_sample() for the bounce -- over inputs that cannot have
// changed between them.
//
// OpenPBR has had openpbr_prepare_at() for exactly this reason; this is the same
// thing for the glTF model. The struct is a cache and nothing else: every field
// is what the code that used to compute it computed, which is why the old entry
// points are still here and simply prepare first.
// ---------------------------------------------------------------------------
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

    // Anisotropy is defined relative to the surface's own tangent frame, so the
    // arbitrary azimuthal basis build_onb() derives from N alone will not do:
    // rotate the mesh's UV tangent and the highlight has to rotate with it.
    // Gram-Schmidt against N rather than si.bitangent, which already carries the
    // TANGENT.w handedness and would flip the lobe with it.
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

/// The preparation, or a zeroed one for a material that has no glTF lobe stack.
///
/// A hair or a pure dielectric vertex reads none of these fields, and building
/// them for it would be the cost this cache exists to remove, charged to the
/// materials that cannot use it.
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

// ---------------------------------------------------------------------------
// Sample
//
// u1, u2: uniform random for microfacet / hemisphere sampling
// lobeWord: 23-bit categorical word for lobe selection
// fresnelWord: 23-bit categorical word for dielectric reflect/refract choice
// ---------------------------------------------------------------------------
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
    // On an exit hit the reflective base/coat lobes do not apply, but diffuse
    // and interface transmission can both cross the surface. Renormalize those
    // two compatible proposals rather than assigning the whole discrete choice
    // to one of them. This also makes either proposal probability one when it is
    // the only surviving lobe.
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

    // -----------------------------------------------------------------------
    // Lobe selection
    // -----------------------------------------------------------------------
    // The thresholds are the running sums of the lobe probabilities, in the
    // order the branches test them. Spelled out rather than accumulated inside
    // the conditions: an assignment in an `if` reads as a typo for a comparison,
    // and here it was also load-bearing in a way that is easy to misread, since
    // the increment on a short-circuited branch never happened. It does not have
    // to -- that branch is the one being taken -- and naming each sum says so
    // without the reader having to work it out. Same additions in the same
    // order, so the same floats.
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
        // ===== DIFFUSE TRANSMISSION LOBE ==================================
        //
        // Lambertian about -N: light enters, scatters inside, and leaves on the
        // far side with no memory of where it came from. No Fresnel and no eta,
        // which is what separates this from the specular transmission lobe
        // below -- there is no interface being refracted through.
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
        // Which side of the interface the ray is on decides eta -- except that a
        // thin-walled surface has no side to be on. Its far wall is another film
        // met from the air, not the way out of a dense medium, so the ratio is
        // the entering one whichever way the shading normal points.
        //
        // Reading that far wall as an exit made it dense-to-thin, where
        // everything past the critical angle reflects with probability 1. At IOR
        // 1.6 the critical angle is 38.7 degrees, and on a sphere the incidence
        // angle at radius r is asin(r / R) -- so the whole annulus outside
        // r / R = 1 / 1.6 = 0.625 total-internally-reflected. A ray through the
        // front wall was trapped between the two walls, reflected every time and
        // absorbed none, so Russian roulette never ended it and maxDepth did:
        // the path returned nothing. That is the black ring on the soap bubbles,
        // and it covered the outer 37.5% of each one, which is what the
        // arithmetic above predicts.
        const float eta = (entering || si.thin_walled) ? (si.exterior_ior / si.ior) : (si.ior / si.exterior_ior);
        const bool is_smooth = (alpha < BSDF_DELTA_ALPHA);

        // Thin wall: Cycles / OpenPBR split Fresnel at the shading normal once,
        // then run a reflection lobe at `alpha` and a transmission lobe at the
        // Kulla-Conty-raised alpha. Using a microfacet F here made the coin flip
        // track the reflection distribution instead of the wall, and disagreed
        // with the weight Cycles bakes into its two closures.
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

    // Cycles' opening test in bump_shadowing_term, which applies to evaluation
    // whatever the lobe: when the normal the map asked for and the one that was
    // shaded disagree about which side the viewer is on, there is nothing to
    // evaluate. Such a hit is lit by its bounce alone and takes no next-event
    // estimate, which is exactly why Cycles renders these pixels dimmer than a
    // renderer that just shades them with the corrected normal.
    //
    // A no-op wherever nothing was corrected: bump_normal is shading_normal
    // there, and the product reduces to a square.
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

    // Reflection means wi and wo are on the SAME side of the surface. Testing
    // NdotL alone only worked while NdotV was guaranteed positive; on an exit
    // hit NdotV is negative, so a refracted direction has NdotL > 0 and would be
    // mistaken for a reflection -- eval would then return 0 for exactly the
    // directions sample produces.
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
        // ---- Transmission hemisphere --------------------------------------
        //
        // Diffuse transmission first, and on its own terms: it is not delta at
        // any roughness and needs no interface, so neither of the guards below
        // applies to it. A leaf is the whole reason next-event estimation has
        // anything to connect to on the shadowed side of a canopy.
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
        CompensatedFloat robustVdotH = compensatedSum(0.0f, 0.0f);
        const float3 H = refraction_half_vector(V, wi, eta, Nf, robustVdotH);

        const float NdotH = dot(Nf, H);
        const float VdotH = saturate(compensatedValue(robustVdotH));
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
