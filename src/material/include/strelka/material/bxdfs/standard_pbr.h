#ifndef STRELKA_BXDF_STANDARD_PBR_H
#define STRELKA_BXDF_STANDARD_PBR_H

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

#include "../material_math.h"
#include "../bsdf_types.h"
#include "../surface_interaction.h"
#include "../sampling.h"
#include "../fresnel.h"
#include "../microfacet.h"
#include "../sheen_albedo_lut.h"
#include "../iridescence.h"
#include "../shading_frame.h"

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
DEVICE_FUNC float3 specular_fresnel(const THREAD_REF SurfaceInteraction& si, float3 F0,
                                    float v_dot_h)
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
    const float3 film =
        iridescence_fresnel(1.0f, si.iridescence_ior, v_dot_h, si.iridescence_thickness, F0);
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
DEVICE_FUNC float3 transmission_fresnel(const THREAD_REF SurfaceInteraction& si, float v_dot_h,
                                        float eta)
{
    const float f = fresnel_dielectric(v_dot_h, eta);
    if (si.iridescence <= 0.0f)
    {
        return make_float3(f);
    }
    const float3 film = iridescence_fresnel(1.0f, si.iridescence_ior, fabsf(v_dot_h),
                                            si.iridescence_thickness, make_float3(f));
    return mix(make_float3(f), film, saturate(si.iridescence));
}

// Charlie sheen evaluated for one direction pair. Zero unless the material
// carries the extension, so every scene without fabric compiles to the same
// work it did before.
DEVICE_FUNC float3 sheen_brdf(const THREAD_REF SurfaceInteraction& si,
                              float n_dot_h, float n_dot_l, float n_dot_v)
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
    return si.sheen_color * (si.sheen * norm * sheen_d_charlie(alpha, n_dot_h) *
                             sheen_v_ashikhmin(n_dot_l, n_dot_v));
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
DEVICE_FUNC float3 clearcoat_base_scale(const THREAD_REF SurfaceInteraction& si, float n_dot_v,
                                        float n_dot_l)
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

    float dielectric_weight = 1.0f - si.metallic;

    // The diffuse lobe splits rather than grows: KHR_materials_diffuse_transmission
    // defines the result as mix(diffuse_brdf, diffuse_btdf, weight), so what goes
    // through is what no longer comes back, and a leaf cannot reflect and
    // transmit its way past the energy that hit it.
    const float dt = saturate(si.diffuse_transmission);
    const float diffuse_base = dielectric_weight * (1.0f - si.transmission);

    w.diffuse      = diffuse_base * (1.0f - dt) * luminance(si.albedo) * diffuse_lobe_scale(si);
    w.diffuse      = fmaxf(w.diffuse, 0.0f);

    // Sheen rides the cosine-sampled lobe instead of getting one of its own.
    // Charlie has no cheap invertible sampling routine, a cosine hemisphere
    // covers its support, and sharing the selection probability keeps
    // combined_pdf a single cosine term in every branch below. What sharing does
    // require is that the lobe stays reachable on a dark fabric, which the max
    // guarantees; for a material without sheen this is exactly a no-op.
    const float sheen_lum = si.sheen * luminance(si.sheen_color);
    w.diffuse = fmaxf(w.diffuse, diffuse_base * (1.0f - dt) * sheen_lum * diffuse_lobe_scale(si));

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
    float f0_scalar = f0_from_ior(si.ior);
    float spec_lum  = mix(f0_scalar * luminance(si.specular_color), luminance(si.albedo), si.metallic);
    w.specular      = fmaxf(spec_lum, 0.04f) * (1.0f - si.transmission * dielectric_weight);

    w.transmission  = dielectric_weight * si.transmission;
    w.transmission  = fmaxf(w.transmission, 0.0f);

    // Scaled by the coat's own reflectance rather than by a constant: at IOR 2.0
    // the coat is nearly three times as reflective as the lacquer the old 0.25
    // stood in for, and a lobe selected too rarely is noise, not bias.
    w.clearcoat     = si.clearcoat * fmaxf(f0_from_ior(fmaxf(si.clearcoat_ior, 1.0f)), 0.04f) * 6.0f;
    w.clearcoat     = fmaxf(w.clearcoat, 0.0f);

    w.total = w.diffuse + w.diffuse_transmission + w.specular + w.transmission + w.clearcoat;
    if (w.total < 1e-10f)
    {
        w.total    = 1.0f;
        w.specular = 1.0f; // fallback to specular
    }

    return w;
}

// ---------------------------------------------------------------------------
// Sample
//
// u1, u2: uniform random for microfacet / hemisphere sampling
// u_lobe: uniform random for lobe selection
// u_fresnel: uniform random for dielectric reflect/refract choice
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfSampleResult standard_pbr_sample(const THREAD_REF SurfaceInteraction& si,
                                                 float u1, float u2,
                                                 float u_lobe, float u_fresnel)
{
    BsdfSampleResult result;
    result.bsdf_over_pdf = make_float3(0.0f);
    result.pdf           = 0.0f;
    result.event_type    = BSDF_EVENT_ABSORB;

    PbrLobeWeights w = pbr_lobe_weights(si);
    float inv_total  = 1.0f / w.total;

    // Normalize weights to probabilities
    float p_diffuse      = w.diffuse      * inv_total;
    float p_diffuse_tr   = w.diffuse_transmission * inv_total;
    float p_specular     = w.specular     * inv_total;
    float p_transmission = w.transmission * inv_total;
    // p_clearcoat = 1 - p_diffuse - p_diffuse_tr - p_specular - p_transmission

    float3 N = si.shading_normal;
    float3 V = si.wo;
    // An opaque surface hit from behind is the same surface seen from the front,
    // and is shaded as such rather than absorbed. See shading_frame.h -- the
    // identical call in standard_pbr_eval() is what keeps the two describing one
    // BRDF.
    if (opaqueBackHitFlipsFrame(si.front_face, dot(N, V), si.transmission, si.diffuse_transmission))
    {
        N = -N;
    }
    float NdotV = dot(N, V);
    // A ray leaving a dielectric hits the far wall from behind, so the shading
    // normal points away from it. That is not a degenerate hit -- it is how
    // light gets out of a medium -- and rejecting it meant a closed
    // transmissive volume absorbed everything that entered it.
    //
    // Only the transmission lobe can describe such a hit (it flips the normal
    // into Nf and picks eta by direction), so the reflection lobes are skipped
    // rather than evaluated against a back-facing normal.
    const bool exiting = NdotV <= 0.0f;
    // Diffuse transmission is the one lobe that legitimately answers a back-face
    // hit without an interface to refract through: a leaf lit from behind is
    // seen from the front through its own thickness. A thin cutout card gets hit
    // from both sides constantly, so this is the common case, not a corner one.
    const bool dt_only_exit = exiting && si.transmission <= 0.0f;
    if (dt_only_exit && w.diffuse_transmission <= 0.0f) return result;
    // Exiting takes the transmission lobe with probability 1, so its selection
    // probability must not divide into the pdf.
    const float p_trans_eff = exiting ? 1.0f : p_transmission;

    float alpha    = alpha_from_roughness(si.roughness);
    float alpha_cc = alpha_from_roughness(si.clearcoat_roughness);

    // Anisotropy is defined relative to the surface's own tangent frame, so the
    // arbitrary azimuthal basis build_onb() derives from N alone will not do:
    // rotate the mesh's UV tangent and the highlight has to rotate with it.
    // Gram-Schmidt against N rather than si.bitangent, which already carries the
    // TANGENT.w handedness and would flip the lobe with it.
    float3 T, B;
    {
        float3 Tp = si.tangent - N * dot(N, si.tangent);
        if (dot(Tp, Tp) > 1e-8f)
        {
            T = safe_normalize(Tp);
            B = cross(N, T);
        }
        else
        {
            build_onb(N, T, B);
        }
    }
    // ax == ay when anisotropy is 0, and every *_aniso routine degenerates to
    // its isotropic form there, so isotropic materials are unchanged.
    float ax, ay;
    anisotropic_alpha(si.roughness, si.anisotropy, ax, ay);

    // F0 for the specular lobe (mix between dielectric F0 and base color for metals)
    float3 F0 = gltf_f0(si.ior, si.specular, si.specular_color, si.albedo, si.metallic);

    // -----------------------------------------------------------------------
    // Lobe selection
    // -----------------------------------------------------------------------
    // On an exit hit with nothing but diffuse transmission available, the lobe
    // draw must land there with probability 1 rather than be filtered out by a
    // branch that never runs.
    float cdf = dt_only_exit ? 0.0f : p_diffuse;
    if (!exiting && u_lobe < cdf)
    {
        // ===== DIFFUSE LOBE ===============================================
        float3 wi_local = cosine_hemisphere_sample(u1, u2);
        result.wi = local_to_world(wi_local, T, B, N);

        float NdotL = dot(N, result.wi);
        if (NdotL <= 0.0f) return result;

        // Evaluate all lobes for the sampled direction (MIS)
        // Diffuse contribution
        float3 f_diffuse = diffuse_lobe_scale(si) * si.albedo * M_1_PI_F * (1.0f - si.metallic) * (1.0f - si.transmission) *
                           (1.0f - saturate(si.diffuse_transmission));

        // Specular contribution
        float3 H     = safe_normalize(V + result.wi);
        float NdotH  = fmaxf(dot(N, H), 0.0f);
        float VdotH  = fmaxf(dot(V, H), 0.0f);
        const float3 Hl_s = world_to_local(H, T, B, N);
        const float3 Vl_s = world_to_local(V, T, B, N);
        const float3 Ll_s = world_to_local(result.wi, T, B, N);
        float D      = ggx_ndf_aniso(ax, ay, Hl_s);
        float G2     = ggx_smith_g2_aniso(ax, ay, Vl_s, Ll_s);
        float3 F     = specular_fresnel(si, F0, VdotH);
        // Multiple-scattering compensation. The single-scatter lobe drops every
        // bounce after the first, which costs a rough metal half its energy; see
        // ggx_energy_term(). Applied identically in all four places f_spec is
        // built, or the sample and eval paths would disagree and MIS would blend
        // two different BRDFs.
        float3 f_spec = F * (D * G2 / (4.0f * NdotV * NdotL + 1e-10f)) *
                        ggx_energy_compensation(F0, si.roughness, NdotV);

        // Clearcoat contribution
        float3 f_cc   = make_float3(0.0f);
        float pdf_cc  = 0.0f;
        if (si.clearcoat > 0.0f)
        {
            float D_cc    = ggx_ndf(alpha_cc, NdotH);
            float G2_cc   = ggx_smith_g2(alpha_cc, NdotV, NdotL);
            float F_cc    = fresnel_schlick_scalar(clearcoat_f0(si), VdotH);
            float cc_brdf = D_cc * G2_cc * F_cc / (4.0f * NdotV * NdotL + 1e-10f);
            f_cc          = make_float3(si.clearcoat * cc_brdf);
            pdf_cc        = ggx_vndf_pdf(alpha_cc, NdotH, NdotV, VdotH);
        }

        float3 f_sheen = sheen_brdf(si, NdotH, NdotL, NdotV);
        float3 f_total = ((f_diffuse + f_spec * specular_lobe_scale(si)) * clearcoat_base_scale(si, NdotV, NdotL) + f_cc) *
                             sheen_base_scale(si, NdotV) +
                         f_sheen;

        // Combined PDF
        float pdf_diffuse = cosine_hemisphere_pdf(NdotL);
        float pdf_spec    = ggx_vndf_pdf_aniso(ax, ay, world_to_local(H, T, B, N),
                                               world_to_local(V, T, B, N));
        float p_cc        = w.clearcoat * inv_total;
        float combined_pdf = p_diffuse * pdf_diffuse
                           + p_specular * pdf_spec
                           + p_cc * pdf_cc;
        // Transmission does not contribute to reflection hemisphere
        combined_pdf = fmaxf(combined_pdf, 1e-10f);

        result.bsdf_over_pdf = f_total * NdotL / combined_pdf;
        result.pdf           = combined_pdf;
        result.event_type    = BSDF_EVENT_DIFFUSE_REFLECTION;
    }
    else if (dt_only_exit || u_lobe < (cdf += p_diffuse_tr))
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
        float3 wi_local = cosine_hemisphere_sample(u1, u2);
        result.wi = local_to_world(wi_local, Tt, Bt, Nt);

        const float NdotL_t = dot(Nt, result.wi);
        if (NdotL_t <= 0.0f) return result;

        const float dt = saturate(si.diffuse_transmission);
        const float3 f_dt = si.diffuse_transmission_color * M_1_PI_F *
                            (1.0f - si.metallic) * (1.0f - si.transmission) * dt;

        // Only this lobe reaches the far hemisphere without an interface, so the
        // pdf has no other term to share with -- unless the material is also
        // specularly transmissive, which foliage is not and glass does not do
        // diffusely.
        const float pdf_dt = cosine_hemisphere_pdf(NdotL_t);
        const float p_eff = dt_only_exit ? 1.0f : p_diffuse_tr;
        const float combined_pdf = fmaxf(p_eff * pdf_dt, 1e-10f);

        result.bsdf_over_pdf = f_dt * NdotL_t / combined_pdf;
        result.pdf           = combined_pdf;
        result.event_type    = BSDF_EVENT_DIFFUSE_TRANSMISSION;
    }
    else if (!exiting && u_lobe < (cdf += p_specular))
    {
        // ===== SPECULAR LOBE ==============================================
        float3 V_local = world_to_local(V, T, B, N);
        float3 H_local = ggx_vndf_sample_aniso(V_local, ax, ay, u1, u2);
        float3 H       = local_to_world(H_local, T, B, N);
        float VdotH    = dot(V, H);
        if (VdotH <= 0.0f) return result;

        result.wi = reflect_dir(-V, H);
        float NdotL = dot(N, result.wi);
        if (NdotL <= 0.0f) return result;

        float NdotH = dot(N, H);

        // Evaluate all lobes
        float3 F     = specular_fresnel(si, F0, VdotH);
        const float3 Hl_s = world_to_local(H, T, B, N);
        const float3 Vl_s = world_to_local(V, T, B, N);
        const float3 Ll_s = world_to_local(result.wi, T, B, N);
        float D      = ggx_ndf_aniso(ax, ay, Hl_s);
        float G2     = ggx_smith_g2_aniso(ax, ay, Vl_s, Ll_s);
        float3 f_spec = F * (D * G2 / (4.0f * NdotV * NdotL + 1e-10f)) *
                        ggx_energy_compensation(F0, si.roughness, NdotV);

        float3 f_diffuse = diffuse_lobe_scale(si) * si.albedo * M_1_PI_F * (1.0f - si.metallic) * (1.0f - si.transmission) *
                           (1.0f - saturate(si.diffuse_transmission));

        float3 f_cc  = make_float3(0.0f);
        float pdf_cc = 0.0f;
        if (si.clearcoat > 0.0f)
        {
            float D_cc    = ggx_ndf(alpha_cc, NdotH);
            float G2_cc   = ggx_smith_g2(alpha_cc, NdotV, NdotL);
            float F_cc    = fresnel_schlick_scalar(clearcoat_f0(si), VdotH);
            float cc_brdf = D_cc * G2_cc * F_cc / (4.0f * NdotV * NdotL + 1e-10f);
            f_cc          = make_float3(si.clearcoat * cc_brdf);
            pdf_cc        = ggx_vndf_pdf(alpha_cc, NdotH, NdotV, VdotH);
        }

        float3 f_sheen = sheen_brdf(si, NdotH, NdotL, NdotV);
        float3 f_total = ((f_diffuse + f_spec * specular_lobe_scale(si)) * clearcoat_base_scale(si, NdotV, NdotL) + f_cc) *
                             sheen_base_scale(si, NdotV) +
                         f_sheen;

        float pdf_diffuse = cosine_hemisphere_pdf(NdotL);
        float pdf_spec    = ggx_vndf_pdf_aniso(ax, ay, world_to_local(H, T, B, N),
                                               world_to_local(V, T, B, N));
        float p_cc        = w.clearcoat * inv_total;
        float combined_pdf = p_diffuse * pdf_diffuse
                           + p_specular * pdf_spec
                           + p_cc * pdf_cc;
        combined_pdf = fmaxf(combined_pdf, 1e-10f);

        result.bsdf_over_pdf = f_total * NdotL / combined_pdf;
        result.pdf           = combined_pdf;
        result.event_type    = (alpha < 0.001f)
                             ? BSDF_EVENT_SPECULAR_REFLECTION
                             : BSDF_EVENT_GLOSSY_REFLECTION;
    }
    else if (exiting || u_lobe < (cdf += p_transmission))
    {
        // ===== TRANSMISSION LOBE ==========================================
        bool entering = NdotV > 0.0f;
        float3 Nf     = entering ? N : -N;
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
        float eta     = (entering || si.thin_walled) ? (si.exterior_ior / si.ior)
                                                     : (si.ior / si.exterior_ior);
        bool is_smooth = (alpha < 0.001f);

        // Thin wall: Cycles / OpenPBR split Fresnel at the shading normal once,
        // then run a reflection lobe at `alpha` and a transmission lobe at the
        // Kulla-Conty-raised alpha. Using a microfacet F here made the coin flip
        // track the reflection distribution instead of the wall, and disagreed
        // with the weight Cycles bakes into its two closures.
        if (si.thin_walled)
        {
            const float NdotV_abs = fabsf(NdotV);
            const float F_val = fresnel_dielectric(NdotV_abs, eta);
            const float3 F_film = transmission_fresnel(si, NdotV_abs, eta);
            const float3 reflectTint = F_film / fmaxf(F_val, 1e-4f);
            const float3 refractTint =
                (make_float3(1.0f) - F_film) / fmaxf(1.0f - F_val, 1e-4f);

            if (u_fresnel < F_val)
            {
                float3 H = Nf;
                if (!is_smooth)
                {
                    float3 V_local = world_to_local(V, T, B, Nf);
                    float3 H_local = ggx_vndf_sample(V_local, alpha, u1, u2);
                    H = local_to_world(H_local, T, B, Nf);
                }
                const float VdotH = dot(V, H);
                if (VdotH <= 0.0f)
                    return result;

                result.wi = reflect_dir(-V, H);
                const float NdotL = dot(Nf, result.wi);
                if (NdotL <= 0.0f)
                    return result;

                if (is_smooth)
                {
                    result.bsdf_over_pdf = si.albedo * reflectTint;
                    result.pdf           = p_trans_eff * F_val;
                    result.event_type    = BSDF_EVENT_SPECULAR_REFLECTION;
                }
                else
                {
                    const float NdotH = dot(Nf, H);
                    const float G2 = ggx_smith_g2(alpha, NdotV_abs, NdotL);
                    const float G1 = ggx_smith_g1(alpha, NdotV_abs);
                    result.bsdf_over_pdf = si.albedo * reflectTint * (G2 / (G1 + 1e-10f));
                    result.pdf = p_trans_eff * F_val *
                                 ggx_vndf_pdf(alpha, NdotH, NdotV_abs, VdotH);
                    result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
                }
                return result;
            }

            // Transmission: smooth is exactly -V; rough is a GGX reflection of
            // the view mirrored through the surface (Cycles / OpenPBR).
            const float eta_rel = fmaxf(si.ior / fmaxf(si.exterior_ior, 1e-4f), 1.0f);
            const float alpha_t = thin_glass_transmission_alpha(alpha, eta_rel);
            const bool t_smooth = (alpha_t < 0.001f);

            float3 H_t = Nf;
            if (!t_smooth)
            {
                float3 V_local_t = world_to_local(V, T, B, Nf);
                float3 H_local_t = ggx_vndf_sample(V_local_t, alpha_t, u1, u2);
                H_t = local_to_world(H_local_t, T, B, Nf);
                if (dot(V, H_t) <= 0.0f)
                    return result;
            }

            const float3 wi_r = reflect_dir(-V, H_t);
            result.wi = flip_through_surface(wi_r, Nf);

            if (t_smooth)
            {
                result.bsdf_over_pdf = si.albedo * refractTint;
                result.pdf           = p_trans_eff * (1.0f - F_val);
                result.event_type    = BSDF_EVENT_SPECULAR_TRANSMISSION;
            }
            else
            {
                const float NdotH_t = dot(Nf, H_t);
                const float NdotL_r = dot(Nf, wi_r);
                if (NdotL_r <= 0.0f)
                    return result;
                const float VdotH_t = dot(V, H_t);
                const float G2_t = ggx_smith_g2(alpha_t, NdotV_abs, NdotL_r);
                const float G1_t = ggx_smith_g1(alpha_t, NdotV_abs);
                result.bsdf_over_pdf = si.albedo * refractTint * (G2_t / (G1_t + 1e-10f));
                result.pdf = p_trans_eff * (1.0f - F_val) *
                             ggx_vndf_pdf(alpha_t, NdotH_t, NdotV_abs, VdotH_t);
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
            float3 V_local = world_to_local(V, T, B, Nf);
            float3 H_local = ggx_vndf_sample(V_local, alpha, u1, u2);
            H = local_to_world(H_local, T, B, Nf);
        }

        float VdotH = dot(V, H);
        if (VdotH <= 0.0f) return result;

        const float F_val = fresnel_dielectric(VdotH, eta);
        const float3 F_film = transmission_fresnel(si, VdotH, eta);
        // Expected value F_film with a coin flipped at F_val.
        const float3 reflectTint = F_film / fmaxf(F_val, 1e-4f);
        const float3 refractTint =
            (make_float3(1.0f) - F_film) / fmaxf(1.0f - F_val, 1e-4f);

        if (u_fresnel < F_val)
        {
            // Specular reflection within transmission lobe
            result.wi = reflect_dir(-V, H);
            float NdotL = dot(Nf, result.wi);
            if (NdotL <= 0.0f) return result;

            if (is_smooth)
            {
                result.bsdf_over_pdf = si.albedo * reflectTint;
                result.pdf           = p_trans_eff * F_val;
                result.event_type    = BSDF_EVENT_SPECULAR_REFLECTION;
            }
            else
            {
                float NdotH  = dot(Nf, H);
                float G2     = ggx_smith_g2(alpha, fabsf(NdotV), NdotL);
                float G1     = ggx_smith_g1(alpha, fabsf(NdotV));
                result.bsdf_over_pdf = si.albedo * reflectTint * (G2 / (G1 + 1e-10f));
                result.pdf   = p_trans_eff * F_val
                             * ggx_vndf_pdf(alpha, NdotH, fabsf(NdotV), VdotH);
                result.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
            }
        }
        else
        {
            // Solid refraction
            float3 wi_refracted;
            const bool valid = refract_dir(-V, H, eta, wi_refracted);
            if (!valid)
            {
                // Total internal reflection
                result.wi            = reflect_dir(-V, H);
                result.bsdf_over_pdf = si.albedo;
                result.pdf           = p_trans_eff;
                result.event_type    = BSDF_EVENT_SPECULAR_REFLECTION;
                return result;
            }

            result.wi = safe_normalize(wi_refracted);

            if (is_smooth)
            {
                float factor = eta * eta;
                result.bsdf_over_pdf = si.albedo * refractTint * factor;
                result.pdf           = p_trans_eff * (1.0f - F_val);
                result.event_type    = BSDF_EVENT_SPECULAR_TRANSMISSION;
            }
            else
            {
                float NdotH   = fabsf(dot(Nf, H));
                float NdotL   = fabsf(dot(Nf, result.wi));
                float LdotH   = fabsf(dot(result.wi, H));
                float G2      = ggx_smith_g2(alpha, fabsf(NdotV), fmaxf(NdotL, 0.001f));
                float G1      = ggx_smith_g1(alpha, fabsf(NdotV));
                float factor  = eta * eta;

                result.bsdf_over_pdf = si.albedo * factor * (G2 / (G1 + 1e-10f));

                float denom   = (VdotH + eta * LdotH);
                float dwh_dwi = (eta * eta * LdotH) / (denom * denom + 1e-10f);
                float vndf_p  = ggx_vndf_pdf(alpha, NdotH, fabsf(NdotV), VdotH);
                result.pdf    = p_trans_eff * (1.0f - F_val) * vndf_p * fabsf(dwh_dwi);
                result.event_type = BSDF_EVENT_GLOSSY_TRANSMISSION;
            }
        }
    }
    else
    {
        // ===== CLEARCOAT LOBE =============================================
        float3 V_local = world_to_local(V, T, B, N);
        float3 H_local = ggx_vndf_sample(V_local, alpha_cc, u1, u2);
        float3 H       = local_to_world(H_local, T, B, N);
        float VdotH    = dot(V, H);
        if (VdotH <= 0.0f) return result;

        result.wi = reflect_dir(-V, H);
        float NdotL = dot(N, result.wi);
        if (NdotL <= 0.0f) return result;

        float NdotH = dot(N, H);

        // Evaluate all lobes at this direction for proper MIS weighting
        float3 F      = specular_fresnel(si, F0, VdotH);
        float D       = ggx_ndf_aniso(ax, ay, H_local);
        float G2_main = ggx_smith_g2_aniso(ax, ay, world_to_local(V, T, B, N),
                                           world_to_local(result.wi, T, B, N));
        float3 f_spec = F * (D * G2_main / (4.0f * NdotV * NdotL + 1e-10f)) *
                        ggx_energy_compensation(F0, si.roughness, NdotV);

        float3 f_diffuse = diffuse_lobe_scale(si) * si.albedo * M_1_PI_F * (1.0f - si.metallic) * (1.0f - si.transmission) *
                           (1.0f - saturate(si.diffuse_transmission));

        float D_cc    = ggx_ndf(alpha_cc, NdotH);
        float G2_cc   = ggx_smith_g2(alpha_cc, NdotV, NdotL);
        float F_cc    = fresnel_schlick_scalar(clearcoat_f0(si), VdotH);
        float cc_brdf = D_cc * G2_cc * F_cc / (4.0f * NdotV * NdotL + 1e-10f);
        float3 f_cc   = make_float3(si.clearcoat * cc_brdf);

        float3 f_sheen = sheen_brdf(si, NdotH, NdotL, NdotV);
        float3 f_total = ((f_diffuse + f_spec * specular_lobe_scale(si)) * clearcoat_base_scale(si, NdotV, NdotL) + f_cc) *
                             sheen_base_scale(si, NdotV) +
                         f_sheen;

        float pdf_diffuse = cosine_hemisphere_pdf(NdotL);
        float pdf_spec    = ggx_vndf_pdf_aniso(ax, ay, world_to_local(H, T, B, N),
                                               world_to_local(V, T, B, N));
        float pdf_cc      = ggx_vndf_pdf(alpha_cc, NdotH, NdotV, VdotH);
        float p_cc        = w.clearcoat * inv_total;
        float combined_pdf = p_diffuse * pdf_diffuse
                           + p_specular * pdf_spec
                           + p_cc * pdf_cc;
        combined_pdf = fmaxf(combined_pdf, 1e-10f);

        result.bsdf_over_pdf = f_total * NdotL / combined_pdf;
        result.pdf           = combined_pdf;
        result.event_type    = (alpha_cc < 0.001f)
                             ? BSDF_EVENT_SPECULAR_REFLECTION
                             : BSDF_EVENT_GLOSSY_REFLECTION;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------
DEVICE_FUNC BsdfEvalResult standard_pbr_eval(const THREAD_REF SurfaceInteraction& si,
                                             float3 wi)
{
    BsdfEvalResult result;
    result.bsdf = make_float3(0.0f);
    result.pdf  = 0.0f;

    float3 N = si.shading_normal;
    float3 V = si.wo;
    // Same flip as standard_pbr_sample(), from the same predicate, before either
    // cosine is taken -- so NdotV and NdotL are both measured against the frame
    // that was actually shaded.
    if (opaqueBackHitFlipsFrame(si.front_face, dot(N, V), si.transmission, si.diffuse_transmission))
    {
        N = -N;
    }

    float NdotV = dot(N, V);
    float NdotL = dot(N, wi);

    // Mirror of the guard in standard_pbr_sample(): an exit hit is legitimate for
    // a transmissive material, and eval must accept exactly what sample can
    // produce or MIS blends two different BRDFs.
    const bool exiting = NdotV <= 0.0f;
    if (exiting && si.transmission <= 0.0f) return result;

    float alpha    = alpha_from_roughness(si.roughness);
    float alpha_cc = alpha_from_roughness(si.clearcoat_roughness);

    // Same tangent frame and axis split as standard_pbr_sample(). They must be
    // derived identically or eval and sample describe two different BRDFs and
    // MIS blends them.
    float3 T, B;
    {
        float3 Tp = si.tangent - N * dot(N, si.tangent);
        if (dot(Tp, Tp) > 1e-8f)
        {
            T = safe_normalize(Tp);
            B = cross(N, T);
        }
        else
        {
            build_onb(N, T, B);
        }
    }
    float ax, ay;
    anisotropic_alpha(si.roughness, si.anisotropy, ax, ay);

    PbrLobeWeights w = pbr_lobe_weights(si);
    float inv_total  = 1.0f / w.total;
    float p_diffuse      = w.diffuse      * inv_total;
    float p_diffuse_tr   = w.diffuse_transmission * inv_total;
    float p_specular     = w.specular     * inv_total;
    float p_transmission = w.transmission * inv_total;
    float p_clearcoat    = w.clearcoat    * inv_total;
    // Same reasoning as in sample: on an exit hit the transmission lobe is the
    // only one that can be chosen, so it carries probability 1.
    const float p_trans_eff = exiting ? 1.0f : p_transmission;

    float3 F0 = gltf_f0(si.ior, si.specular, si.specular_color, si.albedo, si.metallic);

    // Reflection means wi and wo are on the SAME side of the surface. Testing
    // NdotL alone only worked while NdotV was guaranteed positive; on an exit
    // hit NdotV is negative, so a refracted direction has NdotL > 0 and would be
    // mistaken for a reflection -- eval would then return 0 for exactly the
    // directions sample produces.
    bool is_reflection = ((NdotL > 0.0f) == (NdotV > 0.0f));

    if (is_reflection)
    {
        // ---- Reflection hemisphere ----------------------------------------
        float3 H     = safe_normalize(V + wi);
        float NdotH  = fmaxf(dot(N, H), 0.0f);
        float VdotH  = fmaxf(dot(V, H), 0.0f);

        if (NdotH <= 0.0f || VdotH <= 0.0f)
            return result;

        // Diffuse
        float3 f_diffuse = diffuse_lobe_scale(si) * si.albedo * M_1_PI_F * (1.0f - si.metallic) * (1.0f - si.transmission) *
                           (1.0f - saturate(si.diffuse_transmission));

        // Specular
        float3 F      = specular_fresnel(si, F0, VdotH);
        float  D      = ggx_ndf_aniso(ax, ay, world_to_local(H, T, B, N));
        float  G2     = ggx_smith_g2_aniso(ax, ay, world_to_local(V, T, B, N),
                                           world_to_local(wi, T, B, N));
        float3 f_spec = F * (D * G2 / (4.0f * NdotV * NdotL + 1e-10f)) *
                        ggx_energy_compensation(F0, si.roughness, NdotV);

        // Clearcoat
        float3 f_cc  = make_float3(0.0f);
        float pdf_cc = 0.0f;
        if (si.clearcoat > 0.0f)
        {
            float D_cc    = ggx_ndf(alpha_cc, NdotH);
            float G2_cc   = ggx_smith_g2(alpha_cc, NdotV, NdotL);
            float F_cc    = fresnel_schlick_scalar(clearcoat_f0(si), VdotH);
            float cc_brdf = D_cc * G2_cc * F_cc / (4.0f * NdotV * NdotL + 1e-10f);
            f_cc          = make_float3(si.clearcoat * cc_brdf);
            pdf_cc        = ggx_vndf_pdf(alpha_cc, NdotH, NdotV, VdotH);
        }

        // Sheen. Added here and in all three reflection branches of sample():
        // the two paths are MIS-weighted against each other, so a term present
        // in one and missing from the other is not a small error, it is two
        // different BRDFs being blended.
        float3 f_sheen = sheen_brdf(si, NdotH, NdotL, NdotV);

        result.bsdf = ((f_diffuse + f_spec * specular_lobe_scale(si)) * clearcoat_base_scale(si, NdotV, NdotL) + f_cc) *
                          sheen_base_scale(si, NdotV) +
                      f_sheen;

        float pdf_diffuse = cosine_hemisphere_pdf(NdotL);
        float pdf_spec    = ggx_vndf_pdf_aniso(ax, ay, world_to_local(H, T, B, N),
                                               world_to_local(V, T, B, N));
        result.pdf = p_diffuse * pdf_diffuse
                   + p_specular * pdf_spec
                   + p_clearcoat * pdf_cc;
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
                result.bsdf = si.diffuse_transmission_color * M_1_PI_F *
                              (1.0f - si.metallic) * (1.0f - si.transmission) * dt;
                result.pdf = p_diffuse_tr * cosine_hemisphere_pdf(NdotL_t);
            }
        }

        // Delta transmission cannot be evaluated. A smooth thin wall is still a
        // delta (exactly -V); a rough one is a GGX lobe about the mirrored view
        // and is evaluated below the same way sample() produces it.
        const float eta_rel_eval = fmaxf(si.ior / fmaxf(si.exterior_ior, 1e-4f), 1.0f);
        const float alpha_t_eval = si.thin_walled ? thin_glass_transmission_alpha(alpha, eta_rel_eval)
                                                  : alpha;
        if (alpha < 0.001f || (si.thin_walled && alpha_t_eval < 0.001f))
            return result;

        if (si.transmission <= 0.0f)
            return result;

        bool entering   = NdotV > 0.0f;
        float3 Nf       = entering ? N : -N;
        float NdotV_abs = fabsf(NdotV);
        float NdotL_abs = fabsf(NdotL);
        // Entering-side eta on a thin wall, for the reason given in sample().
        // MIS weighs this density against the one sample() wrote, so the two
        // have to read the interface the same way or the weights do not sum.
        float eta       = (entering || si.thin_walled) ? (si.exterior_ior / si.ior)
                                                       : (si.ior / si.exterior_ior);

        if (si.thin_walled)
        {
            // Map wi back to the reflection hemisphere and evaluate the GGX
            // BRDF sample() used, with the transmission roughness. Fresnel is
            // the same macro split sample() used, not a microfacet F.
            const float3 wi_r = flip_through_surface(wi, Nf);
            const float NdotL_r = dot(Nf, wi_r);
            if (NdotL_r <= 0.0f)
                return result;

            const float3 H = safe_normalize(V + wi_r);
            const float NdotH = dot(Nf, H);
            const float VdotH = dot(V, H);
            if (NdotH <= 0.0f || VdotH <= 0.0f)
                return result;

            const float F_val = fresnel_dielectric(NdotV_abs, eta);
            const float3 F_film = transmission_fresnel(si, NdotV_abs, eta);
            const float D = ggx_ndf(alpha_t_eval, NdotH);
            const float G2 = ggx_smith_g2(alpha_t_eval, NdotV_abs, NdotL_r);
            const float3 brdf = (make_float3(1.0f) - F_film) *
                                (D * G2 / (4.0f * NdotV_abs * NdotL_r + 1e-10f));
            result.bsdf = result.bsdf +
                          si.albedo * make_float3(fmaxf(brdf.x, 0.0f), fmaxf(brdf.y, 0.0f),
                                                  fmaxf(brdf.z, 0.0f)) *
                              (1.0f - si.metallic) * si.transmission;
            result.pdf = result.pdf + p_trans_eff * (1.0f - F_val) *
                                          ggx_vndf_pdf(alpha_t_eval, NdotH, NdotV_abs, VdotH);
            return result;
        }

        float3 H = safe_normalize(V + eta * wi);
        if (dot(Nf, H) < 0.0f) H = -H;

        float NdotH = dot(Nf, H);
        float VdotH = dot(V, H);
        float LdotH = dot(wi, H);

        if (NdotH <= 0.0f || VdotH <= 0.0f)
            return result;

        float F_val   = fresnel_dielectric(VdotH, eta);
        const float3 F_film = transmission_fresnel(si, VdotH, eta);
        float D       = ggx_ndf(alpha, NdotH);
        float G2      = ggx_smith_g2(alpha, NdotV_abs, NdotL_abs);

        float denom   = (VdotH + eta * LdotH);
        float factor  = fabsf(VdotH * LdotH) / (NdotV_abs * NdotL_abs + 1e-10f);
        const float3 btdf = (make_float3(1.0f) - F_film) * (D * G2 * eta * eta * factor /
                                                            (denom * denom + 1e-10f));

        // Accumulated, not assigned: a material can be both diffusely and
        // specularly transmissive, and the diffuse term above has already
        // written into the same hemisphere.
        const float3 btdf_pos = make_float3(fmaxf(btdf.x, 0.0f), fmaxf(btdf.y, 0.0f),
                                            fmaxf(btdf.z, 0.0f));
        result.bsdf = result.bsdf +
                      si.albedo * btdf_pos * (1.0f - si.metallic) * si.transmission;

        float dwh_dwi = (eta * eta * fabsf(LdotH)) / (denom * denom + 1e-10f);
        float vndf_p  = ggx_vndf_pdf(alpha, NdotH, NdotV_abs, VdotH);
        result.pdf    = result.pdf + p_trans_eff * (1.0f - F_val) * vndf_p * dwh_dwi;
    }

    return result;
}

// ---------------------------------------------------------------------------
// PDF only
// ---------------------------------------------------------------------------
DEVICE_FUNC float standard_pbr_pdf(const THREAD_REF SurfaceInteraction& si, float3 wi)
{
    BsdfEvalResult r = standard_pbr_eval(si, wi);
    return r.pdf;
}

#endif // STRELKA_BXDF_STANDARD_PBR_H
