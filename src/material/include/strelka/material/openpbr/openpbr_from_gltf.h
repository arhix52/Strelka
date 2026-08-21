#ifndef STRELKA_MATERIAL_OPENPBR_FROM_GLTF_H
#define STRELKA_MATERIAL_OPENPBR_FROM_GLTF_H

// ============================================================================
// openpbr_from_gltf.h -- MaterialParams (glTF) expressed as OpenPBRParams
// ============================================================================
//
// Host-only. Two jobs, and it is worth being clear that they are different:
//
//   1. It is how OpenPBR gets brought up at all. A .mtlx loader is a lot of
//      machinery to build before seeing a single pixel; this maps the materials
//      every existing scene already has, so the GPU path, the estimator A/B and
//      the cross-check against standard_pbr all become available first.
//
//   2. It is the cross-check itself. Where the glTF metallic-roughness model and
//      OpenPBR describe the same surface, the two BSDFs must render it the same
//      way. That is the only external check on the OpenPBR integration that does
//      not involve another renderer -- and Cycles cannot serve here, because
//      Blender has no OpenPBR and its Principled is a third model.
//
// So the mappings below are labelled. EXACT means the two specifications agree
// and the degenerate-case test may demand image parity. APPROX means they do
// not, and a difference there is information rather than a bug.
//
// This is *not* on the path any glTF scene takes by default: it runs only when
// the material model is switched over deliberately.
//
// It converts factors, not maps. OpenPBRParams carries no texture and the slot
// table beside it is filled by the authoring routes -- the sidecar and the
// MaterialX loader -- so a material converted here renders with its constants
// and none of its images. Two consequences worth knowing:
//
//   * the A/B this exists for compares two BSDFs on the *untextured* material,
//     which is the sharper comparison anyway: a texture is the same lookup on
//     both sides and only dilutes the difference between the models;
//   * it is measurable. Transcribing iso_bathroom to MaterialX with
//     tools/gltf_to_mtlx.py and rendering both ways puts 2.4% between them;
//     strip the images from the .mtlx and it falls to 1.0%. Half the gap is
//     this, and the rest is what neither transcription maps yet
//     (KHR_texture_transform, STRELKA_materials_medium).
//
// Wiring the maps means deciding what to do about glTF's packed
// metallic-roughness image, whose two channels OpenPBR names as two separate
// inputs -- the shader reads .r from every scalar slot, so the same file in two
// slots would read the wrong channel twice.

#include <strelka/material/material_params.h>
#include <strelka/material/openpbr/openpbr_params.h>

#if !defined(__CUDA_ARCH__) && !defined(__METAL_VERSION__)

#    include <algorithm>
#    include <cmath>

/// Rec. 709 luminance. OpenPBR splits emission into a scalar level and a tint,
/// where MaterialParams carries a colour and a multiplier, so the split has to
/// be made somewhere and this is the standard place to make it.
inline float openpbr_luminance709(float r, float g, float b)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

inline OpenPBRParams openpbr_from_material_params(const MaterialParams& p)
{
    OpenPBRParams o = openpbr_make_default_params();

    // -- Base ---------------------------------------------------------------
    // EXACT. Both models call this the diffuse albedo of a dielectric and the
    // reflectivity of a conductor, and both blend between them on metalness.
    o.base_weight = 1.0f;
    o.base_color = OpenPBRColor{ p.base_color.x, p.base_color.y, p.base_color.z };
    o.base_metalness = std::clamp(p.metallic, 0.0f, 1.0f);
    // EXACT. glTF's diffuse lobe is Lambert, and so is OpenPBR's at roughness 0.
    o.base_diffuse_roughness = 0.0f;

    // -- Specular -----------------------------------------------------------
    // EXACT for the lobe shape: both are GGX with a Smith height-correlated
    // shadowing term, parameterised by perceptual roughness.
    o.specular_roughness = std::clamp(p.roughness, 0.0f, 1.0f);
    o.specular_ior = (p.ior > 0.0f) ? p.ior : 1.5f;
    o.specular_color = OpenPBRColor{ p.specular_color.x, p.specular_color.y, p.specular_color.z };

    // The factor of two is not a fudge. gltfloader.cpp stores
    // KHR_materials_specular's specularFactor halved -- glTF's default of 1.0 is
    // Strelka's 0.5 -- while OpenPBR's specular_weight defaults to 1.0 and means
    // the same thing glTF's unhalved factor does. Undo the halving here or every
    // converted material loses half its specular reflection.
    o.specular_weight = std::clamp(2.0f * p.specular, 0.0f, 1.0f);

    // APPROX. MaterialParams keeps an angle; OpenPBR keeps its cosine and sine,
    // so that a texture-filtered angle cannot wrap through the discontinuity.
    o.specular_roughness_anisotropy = std::clamp(std::fabs(p.anisotropy), 0.0f, 1.0f);
    o.specular_anisotropy_rotation_cos = std::cos(p.anisotropy_rotation);
    o.specular_anisotropy_rotation_sin = std::sin(p.anisotropy_rotation);

    // -- Coat ---------------------------------------------------------------
    // APPROX. KHR_materials_clearcoat is a Fresnel-weighted GGX layer with no
    // absorption and no darkening of what is underneath. OpenPBR's coat has
    // both, and coat_darkening = 1 is the spec default, not "off" -- so a
    // converted clearcoat is legitimately darker underneath than the glTF one.
    o.coat_weight = std::clamp(p.clearcoat, 0.0f, 1.0f);
    o.coat_roughness = std::clamp(p.clearcoat_roughness, 0.0f, 1.0f);
    o.coat_ior = (p.clearcoat_ior > 1.0f) ? p.clearcoat_ior : 1.5f;

    // -- Fuzz / sheen -------------------------------------------------------
    // APPROX. glTF's sheen is a Charlie distribution with an Ashikhmin
    // visibility term; OpenPBR's fuzz is a linearly-transformed-cosine fit of a
    // microflake layer. Same intent, different lobe -- expect a visible
    // difference on fabric and do not read it as a defect.
    o.fuzz_weight = std::clamp(p.sheen, 0.0f, 1.0f);
    o.fuzz_color = OpenPBRColor{ p.sheen_color.x, p.sheen_color.y, p.sheen_color.z };
    o.fuzz_roughness = std::clamp(p.sheen_roughness, 0.0f, 1.0f);

    // -- Transmission -------------------------------------------------------
    // EXACT for the weight. KHR_materials_transmission is the fraction of the
    // dielectric lobe that refracts, which is what transmission_weight means.
    o.transmission_weight = std::clamp(p.transmission, 0.0f, 1.0f);

    // APPROX, and the one worth watching. glTF puts absorption in
    // KHR_materials_volume as an attenuation colour reached over an attenuation
    // distance; OpenPBR uses a transmission colour reached at transmission_depth.
    // The two agree only when the colour is read as the transmittance at that
    // distance, which is how Strelka's volume.h already reads glTF's. A zero or
    // infinite attenuation distance means "no volume" in glTF, and depth 0 means
    // exactly that in OpenPBR too.
    const bool hasVolume = (p.attenuation_distance > 0.0f) && std::isfinite(p.attenuation_distance);
    o.transmission_depth = hasVolume ? p.attenuation_distance : 0.0f;
    o.transmission_color = hasVolume ?
                               OpenPBRColor{ p.attenuation_color.x, p.attenuation_color.y, p.attenuation_color.z } :
                               OpenPBRColor{ 1.0f, 1.0f, 1.0f };

    // -- Subsurface ---------------------------------------------------------
    // APPROX. Strelka drives its random walk from STRELKA_materials_subsurface;
    // OpenPBR expresses the same medium as a weight, a colour and a mean free
    // path with a per-channel scale. The radius is split into a scalar and a
    // normalised scale so that the largest channel keeps its world-space length.
    o.subsurface_weight = std::clamp(p.subsurface, 0.0f, 1.0f);
    o.subsurface_color =
        OpenPBRColor{ p.diffuse_transmission_color.x, p.diffuse_transmission_color.y, p.diffuse_transmission_color.z };
    const float radiusMax = std::max({ p.subsurface_radius.x, p.subsurface_radius.y, p.subsurface_radius.z });
    if (radiusMax > 0.0f)
    {
        o.subsurface_radius = radiusMax;
        o.subsurface_radius_scale = OpenPBRColor{ p.subsurface_radius.x / radiusMax, p.subsurface_radius.y / radiusMax,
                                                  p.subsurface_radius.z / radiusMax };
    }
    o.subsurface_scatter_anisotropy = std::clamp(p.subsurface_anisotropy, -1.0f, 1.0f);

    // -- Thin film ----------------------------------------------------------
    // EXACT in parameterisation: KHR_materials_iridescence is a thin film over
    // the specular interface, given as a weight, an IOR and a thickness in
    // nanometres, which is what OpenPBR's thin_film_* are.
    o.thin_film_weight = std::clamp(p.iridescence, 0.0f, 1.0f);
    o.thin_film_ior = (p.iridescence_ior > 0.0f) ? p.iridescence_ior : 1.4f;
    o.thin_film_thickness = p.iridescence_thickness;

    // -- Emission -----------------------------------------------------------
    // APPROX by construction: MaterialParams has a colour times a multiplier,
    // OpenPBR a level times a tint. Splitting on luminance keeps the product
    // equal, so the radiance a converted emitter puts out is unchanged; only the
    // way it is spelled differs. A black emission colour has no hue to preserve,
    // so the tint stays white and the level goes to zero with it.
    const float er = p.emission.x * p.emission_strength;
    const float eg = p.emission.y * p.emission_strength;
    const float eb = p.emission.z * p.emission_strength;
    const float lum = openpbr_luminance709(er, eg, eb);
    if (lum > 0.0f)
    {
        o.emission_luminance = lum;
        o.emission_color = OpenPBRColor{ er / lum, eg / lum, eb / lum };
    }
    else
    {
        o.emission_luminance = 0.0f;
        o.emission_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
    }

    // -- Geometry -----------------------------------------------------------
    o.geometry_thin_walled = p.thin_walled;
    // Opacity is the host renderer's job in OpenPBR (the BSDF does not consume
    // it), and Strelka already resolves cutouts before shading. Carried so the
    // block is complete, not so the BSDF reads it.
    o.geometry_opacity = std::clamp(p.base_color_alpha, 0.0f, 1.0f);

    // -- Texture transform --------------------------------------------------
    o.uv_offset_x = p.uv_offset_x;
    o.uv_offset_y = p.uv_offset_y;
    o.uv_scale_x = (p.uv_scale_x != 0.0f) ? p.uv_scale_x : 1.0f;
    o.uv_scale_y = (p.uv_scale_y != 0.0f) ? p.uv_scale_y : 1.0f;
    o.uv_rotation = p.uv_rotation;

    return o;
}

#endif // host only

#endif // STRELKA_MATERIAL_OPENPBR_FROM_GLTF_H
