#ifndef STRELKA_MATERIAL_OPENPBR_FROM_GLTF_H
#define STRELKA_MATERIAL_OPENPBR_FROM_GLTF_H

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

    // KHR_materials_specular only scales the dielectric BRDF; it explicitly has
    // no effect on metals. OpenPBR's specular_weight also darkens its metal
    // lobe, so a fully metallic glTF surface must keep the neutral weight.
    o.specular_weight = o.base_metalness >= 1.0f ? 1.0f : std::clamp(2.0f * p.specular, 0.0f, 1.0f);

    // APPROX. MaterialParams keeps an angle; OpenPBR keeps its cosine and sine,
    // so that a texture-filtered angle cannot wrap through the discontinuity.
    o.specular_roughness_anisotropy = std::clamp(std::fabs(p.anisotropy), 0.0f, 1.0f);
    o.specular_anisotropy_rotation_cos = std::cos(p.anisotropy_rotation);
    o.specular_anisotropy_rotation_sin = std::sin(p.anisotropy_rotation);

    o.coat_weight = std::clamp(p.clearcoat, 0.0f, 1.0f);
    o.coat_roughness = std::clamp(p.clearcoat_roughness, 0.0f, 1.0f);
    o.coat_ior = (p.clearcoat_ior > 1.0f) ? p.clearcoat_ior : 1.5f;

    o.fuzz_weight = std::clamp(p.sheen, 0.0f, 1.0f);
    o.fuzz_color = OpenPBRColor{ p.sheen_color.x, p.sheen_color.y, p.sheen_color.z };
    o.fuzz_roughness = std::clamp(p.sheen_roughness, 0.0f, 1.0f);

    // -- Transmission -------------------------------------------------------
    // EXACT for the weight. KHR_materials_transmission is the fraction of the
    // dielectric lobe that refracts, which is what transmission_weight means.
    o.transmission_weight = std::clamp(p.transmission, 0.0f, 1.0f);

    const bool hasVolume = (p.attenuation_distance > 0.0f) && std::isfinite(p.attenuation_distance);
    o.transmission_depth = hasVolume ? p.attenuation_distance : 0.0f;
    o.transmission_color = hasVolume ?
                               OpenPBRColor{ p.attenuation_color.x, p.attenuation_color.y, p.attenuation_color.z } :
                               OpenPBRColor{ 1.0f, 1.0f, 1.0f };

    o.subsurface_weight = std::clamp(p.subsurface, 0.0f, 1.0f);
    const bool hasReference = p.subsurface_reference.x > 0.0f || p.subsurface_reference.y > 0.0f ||
                              p.subsurface_reference.z > 0.0f;
    const float3 subsurfaceColor = hasReference ? p.subsurface_reference : p.diffuse_transmission_color;
    o.subsurface_color = OpenPBRColor{ subsurfaceColor.x, subsurfaceColor.y, subsurfaceColor.z };
    const float radiusMax = std::max({ p.subsurface_radius.x, p.subsurface_radius.y, p.subsurface_radius.z });
    if (radiusMax > 0.0f)
    {
        o.subsurface_radius = radiusMax;
        o.subsurface_radius_scale = OpenPBRColor{ p.subsurface_radius.x / radiusMax, p.subsurface_radius.y / radiusMax,
                                                  p.subsurface_radius.z / radiusMax };
    }
    o.subsurface_scatter_anisotropy = std::clamp(p.subsurface_anisotropy, -1.0f, 1.0f);

    o.thin_film_weight = std::clamp(p.iridescence, 0.0f, 1.0f);
    o.thin_film_ior = (p.iridescence_ior > 0.0f) ? p.iridescence_ior : 1.4f;
    o.thin_film_thickness = p.iridescence_thickness;

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
    o.texture_normal_scale = p.normal_scale;
    o.texture_scalar_flags = 1u | OPENPBR_ROUGHNESS_MULTIPLY | OPENPBR_TEXTURES_GLTF;

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
