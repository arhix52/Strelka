#ifndef STRELKA_OPTIX_SHADING_COMMON_H
#define STRELKA_OPTIX_SHADING_COMMON_H

// Behaviour port of src/shaders/metal/shading_common.h, not a code port.
// CUDA's bsdf_init samples textures, so it receives transformed UVs before this
// layer applies the material properties it cannot resolve itself.

#include <optix.h>

#include <OptixRenderParams.h>

#include <sutil/vec_math.h>
#include <sutil/vec_math_adv.h>

#include <strelka/material/bsdf.h>
#include <strelka/material/blender_procedural.h>
#include <strelka/material/openpbr/openpbr_bridge.h>
#include <strelka/material/normal_filter.h>
#include <strelka/material/valid_reflection.h>
#include <strelka/material/volume.h>

#include "../optix_device_utils.h"
#include "fibre_geometry.h"
#include <nee_pairing.h>
#include "texture_transform.h"

// KHR_materials_volume uses the model selected by `render/material/volumeModel`.

static __forceinline__ __device__ bool scattersThroughFibre(const SurfaceInteraction& si)
{
    return si.material_type == MATERIAL_TYPE_HAIR;
}

static __forceinline__ __device__ bool lightReachesShadingPoint(const SurfaceInteraction& si,
                                                                float3 L)
{
    return neeSurfaceSupportsDirection(scattersThroughFibre(si), si.front_face, dot(si.shading_normal, si.wo),
                                       si.transmission, si.diffuse_transmission, dot(si.shading_normal, L));
}

// The factor that cancels the one hair_chiang_eval() divides by. It has to be the
// same |n.wi| and never a clamp to zero, or the two do not cancel and the fibre's
// far side comes back either black or blown out.
static __forceinline__ __device__ float shadingCosine(const SurfaceInteraction& si, float3 L)
{
    const float c = dot(si.shading_normal, L);
    return neeSurfaceCosine(scattersThroughFibre(si), si.front_face, dot(si.shading_normal, si.wo), si.transmission,
                            si.diffuse_transmission, c);
}

// Where a ray that scattered through a strand has to start. See fibre_geometry.h
// for the chord; the only thing added here is the ray offset, which is CUDA's.
static __forceinline__ __device__ float3 fibreExitOrigin(
    float3 position, float3 tangent, float3 normal, float radius, float3 dir)
{
    const FibreExit e = fibre_exit(position, tangent, normal, radius, dir);
    return offset_ray(e.position, e.normal);
}

// clampPathContribution() -- the firefly bounds for direct and indirect paths -- lives in
// optix_device_utils.h, included above. This file carried an identical copy of it
// until the two streams met; one definition is enough.

static __forceinline__ __device__ float2 openpbr_transform_uv(const OpenPBRParams& p, float2 uv)
{
    // KHR_texture_transform's composition order: scale, then rotate, then
    // translate. Invisible at rotation 0, which is what every exporter writes by
    // default, and wrong everywhere else.
    const float c = cosf(p.uv_rotation);
    const float sn = sinf(p.uv_rotation);
    const float2 k = make_float2(p.uv_scale_x, p.uv_scale_y);
    return make_float2(uv.x * k.x * c - uv.y * k.y * sn, uv.x * k.x * sn + uv.y * k.y * c) +
           make_float2(p.uv_offset_x, p.uv_offset_y);
}

static __forceinline__ __device__ float2 openpbr_transform_uv_direction(const OpenPBRParams& p, float2 d)
{
    const float c = cosf(p.uv_rotation);
    const float sn = sinf(p.uv_rotation);
    return make_float2(d.x * p.uv_scale_x * c - d.y * p.uv_scale_y * sn,
                       d.x * p.uv_scale_x * sn + d.y * p.uv_scale_y * c);
}

static __forceinline__ __device__ float2 layered_transform_uv(const OpenPBRLayeredTextureParams& p,
                                                               unsigned int layer,
                                                               float2 uv,
                                                               bool data)
{
    const float rotation = data ? p.data_uv_rotation[layer] : p.uv_rotation[layer];
    const float scaleX = data ? p.data_uv_scale_x[layer] : p.uv_scale_x[layer];
    const float scaleY = data ? p.data_uv_scale_y[layer] : p.uv_scale_y[layer];
    const float offsetX = data ? p.data_uv_offset_x[layer] : p.uv_offset_x[layer];
    const float offsetY = data ? p.data_uv_offset_y[layer] : p.uv_offset_y[layer];
    const float c = cosf(rotation);
    const float sn = sinf(rotation);
    const float2 centered = uv - make_float2(0.5f);
    return make_float2((centered.x * c - centered.y * sn) * scaleX,
                       (centered.x * sn + centered.y * c) * scaleY) +
           make_float2(0.5f + offsetX, 0.5f + offsetY);
}

static __forceinline__ __device__ float2 layered_transform_direction(const OpenPBRLayeredTextureParams& p,
                                                                      unsigned int layer,
                                                                      float2 d,
                                                                      bool data)
{
    const float rotation = data ? p.data_uv_rotation[layer] : p.uv_rotation[layer];
    const float scaleX = data ? p.data_uv_scale_x[layer] : p.uv_scale_x[layer];
    const float scaleY = data ? p.data_uv_scale_y[layer] : p.uv_scale_y[layer];
    const float c = cosf(rotation);
    const float sn = sinf(rotation);
    return make_float2((d.x * c - d.y * sn) * scaleX,
                       (d.x * sn + d.y * c) * scaleY);
}

static __forceinline__ __device__ float3 layered_adjust_values(float3 rgb,
                                                                float hue,
                                                                float saturation,
                                                                float value,
                                                                float gamma,
                                                                float brightness,
                                                                float contrast)
{
    if (gamma == 0.0f)
        rgb = make_float3(1.0f);
    else if (gamma != 1.0f)
    {
        const float exponent = max_color_correction_exponent(gamma);
        rgb.x = rgb.x > 0.0f ? powf(rgb.x, exponent) : rgb.x;
        rgb.y = rgb.y > 0.0f ? powf(rgb.y, exponent) : rgb.y;
        rgb.z = rgb.z > 0.0f ? powf(rgb.z, exponent) : rgb.z;
    }
    const float hi = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
    const float lo = fminf(rgb.x, fminf(rgb.y, rgb.z));
    const float d = hi - lo;
    float h = 0.0f;
    if (d > 1.0e-8f)
    {
        if (hi == rgb.x)
            h = (rgb.y - rgb.z) / d;
        else if (hi == rgb.y)
            h = 2.0f + (rgb.z - rgb.x) / d;
        else
            h = 4.0f + (rgb.x - rgb.y) / d;
        h = h / 6.0f + hue;
        h -= floorf(h);
    }
    const float s = hi > 0.0f ? saturate((d / hi) * saturation) : 0.0f;
    const float v = hi * value;
    const float sector = h * 6.0f;
    const int i = (int)floorf(sector);
    const float f = sector - floorf(sector);
    const float a = v * (1.0f - s);
    const float b = v * (1.0f - s * f);
    const float c = v * (1.0f - s * (1.0f - f));
    float3 adjusted;
    switch (i % 6)
    {
    case 0: adjusted = make_float3(v, c, a); break;
    case 1: adjusted = make_float3(b, v, a); break;
    case 2: adjusted = make_float3(a, v, c); break;
    case 3: adjusted = make_float3(a, b, v); break;
    case 4: adjusted = make_float3(c, a, v); break;
    default: adjusted = make_float3(v, a, b); break;
    }
    const float contrastScale = 1.0f + contrast;
    const float offset = brightness - contrast * 0.5f;
    adjusted = adjusted * contrastScale + make_float3(offset);
    return make_float3(fmaxf(adjusted.x, 0.0f), fmaxf(adjusted.y, 0.0f), fmaxf(adjusted.z, 0.0f));
}

static __forceinline__ __device__ float3 layered_adjust(float3 rgb,
                                                         const OpenPBRLayeredTextureParams& p,
                                                         bool data,
                                                         unsigned int layer)
{
    return data ?
               layered_adjust_values(rgb, p.data_hue[layer], p.data_saturation[layer], p.data_value[layer],
                                     p.data_gamma[layer], p.data_brightness[layer], p.data_contrast[layer]) :
               layered_adjust_values(rgb, p.color_hue[layer], p.color_saturation[layer], p.color_value[layer],
                                     p.color_gamma[layer], p.color_brightness[layer], p.color_contrast[layer]);
}

static __forceinline__ __device__ float3 layered_sample_source(const OpenPBRTextures& t,
                                                                unsigned int firstSlot,
                                                                unsigned int layer,
                                                                float2 uv,
                                                                float4 textureGradients)
{
    const cudaTextureObject_t tex = t.tex[firstSlot + layer];
    if (tex == 0ull)
        return make_float3(0.0f);
    const bool data = firstSlot == OPENPBR_TEX_LAYER_DATA_0;
    const float2 dx = layered_transform_direction(
        t.layered, layer, make_float2(textureGradients.x, textureGradients.y), data);
    const float2 dy = layered_transform_direction(
        t.layered, layer, make_float2(textureGradients.z, textureGradients.w), data);
    const float4 sample = texture_sample_2d(
        tex, layered_transform_uv(t.layered, layer, uv, data), make_float4(dx.x, dx.y, dy.x, dy.y));
    return make_float3(sample.x, sample.y, sample.z);
}

static __forceinline__ __device__ float3 layered_blend(float3 base,
                                                        float3 layer,
                                                        float factor,
                                                        unsigned int mode)
{
    if (mode == OPENPBR_LAYER_BLEND_MIX)
        return lerp(base, layer, factor);
    if (mode == OPENPBR_LAYER_BLEND_SCREEN)
        return make_float3(1.0f) -
               (make_float3(1.0f - factor) + (make_float3(1.0f) - layer) * factor) *
                   (make_float3(1.0f) - base);
    if (mode == OPENPBR_LAYER_BLEND_OVERLAY)
    {
        float3 result;
        result.x = base.x < 0.5f ? base.x * (1.0f - factor + 2.0f * factor * layer.x) :
                                  1.0f - (1.0f - factor + 2.0f * factor * (1.0f - layer.x)) * (1.0f - base.x);
        result.y = base.y < 0.5f ? base.y * (1.0f - factor + 2.0f * factor * layer.y) :
                                  1.0f - (1.0f - factor + 2.0f * factor * (1.0f - layer.y)) * (1.0f - base.y);
        result.z = base.z < 0.5f ? base.z * (1.0f - factor + 2.0f * factor * layer.z) :
                                  1.0f - (1.0f - factor + 2.0f * factor * (1.0f - layer.z)) * (1.0f - base.z);
        return result;
    }
    return base * (make_float3(1.0f - factor) + layer * factor);
}

static __forceinline__ __device__ float3 layered_sample(const OpenPBRTextures& t,
                                                         unsigned int firstSlot,
                                                         float2 uv,
                                                         float4 textureGradients,
                                                         float facing)
{
    const OpenPBRLayeredTextureParams& p = t.layered;
    const bool data = firstSlot == OPENPBR_TEX_LAYER_DATA_0;
    const bool hasBase = data ? p.data_has_base != 0u : p.color_has_base != 0u;
    float3 result = hasBase ?
                        (data ? make_float3(p.data_base[0], p.data_base[1], p.data_base[2]) :
                                make_float3(p.color_base[0], p.color_base[1], p.color_base[2])) :
                        make_float3(1.0f);
    for (unsigned int layer = 0; layer < p.layer_count; ++layer)
    {
        if (t.tex[firstSlot + layer] == 0ull)
            continue;
        const float3 source = layered_sample_source(t, firstSlot, layer, uv, textureGradients);
        const float3 value = layered_adjust(source, p, data, layer);
        if (layer == 0u && !hasBase)
        {
            result = value;
        }
        else
        {
            if (data)
                result = layered_blend(result, value, p.data_opacity[layer], p.data_blend_mode[layer]);
            else
            {
                float map = 0.0f;
                const unsigned int factorLayer = p.color_factor_data_layer[layer];
                if (factorLayer < 4u && t.tex[OPENPBR_TEX_LAYER_DATA_0 + factorLayer] != 0ull)
                    map = luminance(layered_sample_source(
                        t, OPENPBR_TEX_LAYER_DATA_0, factorLayer, uv, textureGradients));
                const float factor = p.color_factor_base[layer] + p.color_factor_data[layer] * map +
                                     p.color_factor_facing[layer] * facing +
                                     p.color_factor_facing_data[layer] * facing * map;
                result = layered_blend(result, value, factor, p.color_blend_mode[layer]);
            }
        }
    }
    if (!data)
        result = layered_adjust_values(result, p.color_post_hue, p.color_post_saturation,
                                       p.color_post_value, p.color_post_gamma,
                                       p.color_post_brightness, p.color_post_contrast);
    return result;
}

static __forceinline__ __device__ float layered_bump_height(const OpenPBRTextures& t,
                                                             float2 uv,
                                                             float4 textureGradients)
{
    const OpenPBRLayeredTextureParams& p = t.layered;
    float texture = 0.0f;
    if (p.bump_data_layer < 4u && t.tex[OPENPBR_TEX_LAYER_DATA_0 + p.bump_data_layer] != 0ull)
        texture = luminance(layered_adjust(
            layered_sample_source(t, OPENPBR_TEX_LAYER_DATA_0, p.bump_data_layer, uv, textureGradients),
            p, true, p.bump_data_layer));
    else if (p.bump_procedural == OPENPBR_LAYER_PROCEDURAL_NONE)
        texture = luminance(layered_sample(
            t, OPENPBR_TEX_LAYER_DATA_0, uv, textureGradients, 0.0f));
    float procedural = texture;
    if (p.bump_procedural == OPENPBR_LAYER_PROCEDURAL_VORONOI_RIDGE)
        procedural = blender_voronoi_ridge(uv, p.bump_procedural_scale);
    else if (p.bump_procedural == OPENPBR_LAYER_PROCEDURAL_NOISE)
        procedural = blender_noise_fbm(uv, p.bump_procedural_scale, p.bump_procedural_detail,
                                       p.bump_procedural_roughness, p.bump_procedural_lacunarity);
    return lerp(procedural, texture, p.bump_texture_mix) * p.bump_gain;
}

static __forceinline__ __device__ void apply_layered_texture(OpenPBRParams& p,
                                                              const OpenPBRTextures& t,
                                                              SurfaceInteraction& si,
                                                              float2 uv,
                                                              float4 textureGradients,
                                                              float4 bumpGradients,
                                                              float3 dPdx,
                                                              float3 dPdy)
{
    const unsigned int outputs = t.layered.output_mask;
    if (outputs == 0u)
        return;

    const float facing = 1.0f - saturate(fabsf(dot(si.shading_normal, si.wo)));
    const float3 data = layered_sample(t, OPENPBR_TEX_LAYER_DATA_0, uv, textureGradients, facing);
    const float3 color = layered_sample(t, OPENPBR_TEX_LAYER_COLOR_0, uv, textureGradients, facing);
    if ((outputs & OPENPBR_LAYER_OUTPUT_BASE_COLOR) != 0u)
        p.base_color = OpenPBRColor{ saturate(color.x), saturate(color.y), saturate(color.z) };
    if ((outputs & OPENPBR_LAYER_OUTPUT_SPECULAR_COLOR) != 0u)
    {
        const float3 source = t.layered.specular_color_uses_color != 0u ? color : data;
        const float3 base = make_float3(t.layered.specular_color_base[0],
                                        t.layered.specular_color_base[1],
                                        t.layered.specular_color_base[2]);
        const float3 specular = lerp(
            base, saturate(source * t.layered.specular_gain), t.layered.specular_color_mix);
        p.specular_color = OpenPBRColor{ specular.x, specular.y, specular.z };
    }
    if ((outputs & OPENPBR_LAYER_OUTPUT_SPECULAR_WEIGHT) != 0u)
    {
        const float reflectivity = luminance(data) * t.layered.specular_gain;
        p.specular_weight = saturate(p.specular_weight * reflectivity);
    }
    if ((outputs & OPENPBR_LAYER_OUTPUT_SPECULAR_IOR) != 0u)
    {
        const float3 source = t.layered.specular_ior_uses_color != 0u ? color : data;
        const float map = saturate(luminance(source) * t.layered.specular_gain);
        const float reflectivity = lerp(t.layered.specular_ior_base, map, t.layered.specular_ior_mix);
        p.specular_ior = lerp(1.0f, t.layered.specular_ior_authored, reflectivity);
    }
    if ((outputs & OPENPBR_LAYER_OUTPUT_OPACITY) != 0u)
    {
        const unsigned int layer = t.layered.opacity_layer;
        const cudaTextureObject_t tex = t.tex[OPENPBR_TEX_LAYER_DATA_0 + layer];
        if (tex != 0ull)
        {
            const float3 value = layered_sample_source(
                t, OPENPBR_TEX_LAYER_DATA_0, layer, uv, textureGradients);
            p.geometry_opacity = saturate(
                t.layered.opacity_base +
                luminance(value) * t.layered.opacity_mix +
                facing * t.layered.opacity_facing_mix);
        }
    }
    if ((outputs & OPENPBR_LAYER_OUTPUT_ROUGHNESS) != 0u)
    {
        const float height = luminance(data) * t.layered.roughness_gain;
        p.specular_roughness =
            saturate(lerp(t.layered.roughness_base, 1.0f - height, t.layered.roughness_mix));
    }
    if ((outputs & OPENPBR_LAYER_OUTPUT_NORMAL) == 0u)
        return;

    // Match Cycles' Bump node: evaluate height along the actual screen-space
    // differentials, then use those same world-space vectors for the surface
    // gradient.  Axis-aligned UV probes discard both rotation and UV scale.
    constexpr float bumpFilterWidth = 0.1f;
    const float2 dUVdx = make_float2(bumpGradients.x, bumpGradients.y);
    const float2 dUVdy = make_float2(bumpGradients.z, bumpGradients.w);
    // A Blender Color socket connected to a float socket uses scene-linear
    // luminance, not the arithmetic RGB mean (Cycles NODE_CONVERT_CF).
    // Cycles samples the image node itself at level zero for all three bump
    // evaluations; only the coordinate offset carries the ray footprint.
    const float4 bumpTextureGradients = make_float4(0.0f);
    const float h = layered_bump_height(t, uv, bumpTextureGradients);
    const float hx = layered_bump_height(t, uv + dUVdx * bumpFilterWidth, bumpTextureGradients);
    const float hy = layered_bump_height(t, uv + dUVdy * bumpFilterWidth, bumpTextureGradients);
    const float3 rx = cross(dPdy, si.shading_normal);
    const float3 ry = cross(si.shading_normal, dPdx);
    const float det = dot(dPdx, rx);
    if (fabsf(det) <= 1.0e-20f)
        return;
    const float3 surfaceGradient = (hx - h) * rx + (hy - h) * ry;
    si.shading_normal = safe_normalize(
        bumpFilterWidth * fabsf(det) * si.shading_normal -
        t.layered.bump_scale * copysignf(1.0f, det) * surfaceGradient);
    if (dot(si.shading_normal, si.wo) <= 0.0f)
    {
        const float3 facingGeom = dot(si.geometry_normal, si.wo) > 0.0f ? si.geometry_normal : -si.geometry_normal;
        si.shading_normal = ensureValidSpecularReflection(facingGeom, si.wo, si.shading_normal);
        si.diffuse_faces_away = true;
    }
}

static __forceinline__ __device__ void openpbr_apply_textures(OpenPBRParams& p,
                                                              const OpenPBRTextures& textures,
                                                              SurfaceInteraction& si,
                                                              float2 uv,
                                                              float4 textureGradients,
                                                              float4 bumpGradients,
                                                              float3 dPdx,
                                                              float3 dPdy)
{
    const float2 tuv = openpbr_transform_uv(p, uv);
    const float2 dx = openpbr_transform_uv_direction(p, make_float2(textureGradients.x, textureGradients.y));
    const float2 dy = openpbr_transform_uv_direction(p, make_float2(textureGradients.z, textureGradients.w));
    const float4 gradients = make_float4(dx.x, dx.y, dy.x, dy.y);

    // Both halves of the gate in one place: the mask says the material named a
    // file, the handle says one arrived.
    auto bound = [&](unsigned int slot) -> bool
    { return openpbr_has_texture(p, slot) && textures.tex[slot] != 0ull; };
    auto sample = [&](unsigned int slot) -> float4
    { return texture_sample_2d(textures.tex[slot], tuv, gradients); };

    struct ScalarSlot
    {
        unsigned int slot;
        float OpenPBRParams::*field;
    };
    struct ColorSlot
    {
        unsigned int slot;
        OpenPBRColor OpenPBRParams::*field;
    };

    constexpr ScalarSlot kScalarSlots[] = {
        { OPENPBR_TEX_BASE_METALNESS, &OpenPBRParams::base_metalness },
        { OPENPBR_TEX_SPECULAR_ROUGHNESS, &OpenPBRParams::specular_roughness },
        { OPENPBR_TEX_SPECULAR_ANISOTROPY, &OpenPBRParams::specular_roughness_anisotropy },
        { OPENPBR_TEX_COAT_WEIGHT, &OpenPBRParams::coat_weight },
        { OPENPBR_TEX_COAT_ROUGHNESS, &OpenPBRParams::coat_roughness },
        { OPENPBR_TEX_FUZZ_WEIGHT, &OpenPBRParams::fuzz_weight },
        { OPENPBR_TEX_FUZZ_ROUGHNESS, &OpenPBRParams::fuzz_roughness },
        { OPENPBR_TEX_GEOMETRY_OPACITY, &OpenPBRParams::geometry_opacity },
        { OPENPBR_TEX_SUBSURFACE_WEIGHT, &OpenPBRParams::subsurface_weight },
    };
    constexpr ColorSlot kColorSlots[] = {
        { OPENPBR_TEX_BASE_COLOR, &OpenPBRParams::base_color },
        { OPENPBR_TEX_SPECULAR_COLOR, &OpenPBRParams::specular_color },
        { OPENPBR_TEX_COAT_COLOR, &OpenPBRParams::coat_color },
        { OPENPBR_TEX_TRANSMISSION_COLOR, &OpenPBRParams::transmission_color },
        { OPENPBR_TEX_SUBSURFACE_COLOR, &OpenPBRParams::subsurface_color },
        { OPENPBR_TEX_FUZZ_COLOR, &OpenPBRParams::fuzz_color },
        { OPENPBR_TEX_SUBSURFACE_RADIUS, &OpenPBRParams::subsurface_radius_scale },
    };

#pragma unroll
    for (const ScalarSlot& s : kScalarSlots)
    {
        if (bound(s.slot))
        {
            const float4 v = sample(s.slot);
            float scalar = v.x;
            if (s.slot == OPENPBR_TEX_SPECULAR_ROUGHNESS)
            {
                switch (p.texture_scalar_flags & OPENPBR_ROUGHNESS_CHANNEL_MASK)
                {
                case 1u: scalar = v.y; break;
                case 2u: scalar = v.z; break;
                case 3u: scalar = v.w; break;
                default: break;
                }
                if ((p.texture_scalar_flags & OPENPBR_ROUGHNESS_MULTIPLY) != 0u)
                {
                    p.*(s.field) *= scalar;
                    continue;
                }
            }
            p.*(s.field) = scalar;
        }
    }
#pragma unroll
    for (const ColorSlot& c : kColorSlots)
    {
        if (bound(c.slot))
        {
            const float4 v = sample(c.slot);
            if (c.slot == OPENPBR_TEX_BASE_COLOR && (p.texture_scalar_flags & OPENPBR_TEXTURES_GLTF) != 0u)
            {
                const OpenPBRColor factor = p.*(c.field);
                p.*(c.field) = OpenPBRColor{ factor.r * v.x, factor.g * v.y, factor.b * v.z };
            }
            else
            {
                p.*(c.field) = OpenPBRColor{ v.x, v.y, v.z };
            }
        }
    }

    // Layer graphs own their per-source transforms; their coordinates start at
    // the mesh UV, so their derivatives must start there as well.
    apply_layered_texture(p, textures, si, uv, textureGradients, bumpGradients, dPdx, dPdy);

    if (bound(OPENPBR_TEX_EMISSION_COLOR))
    {
        const float4 v = sample(OPENPBR_TEX_EMISSION_COLOR);
        const float3 tint = (p.texture_scalar_flags & OPENPBR_TEXTURES_GLTF) != 0u ?
                                make_float3(p.emission_color.r * v.x, p.emission_color.g * v.y,
                                            p.emission_color.b * v.z) :
                                make_float3(v.x, v.y, v.z);
        p.emission_color = OpenPBRColor{ tint.x, tint.y, tint.z };
        // Emission is consumed from SurfaceInteraction rather than the BSDF
        // parameter block, so update both representations at the same sample.
        si.emission = tint * p.emission_luminance;
    }

    if (bound(OPENPBR_TEX_GEOMETRY_NORMAL))
    {
        const float4 n = sample(OPENPBR_TEX_GEOMETRY_NORMAL);
        const float2 xy = make_float2(n.x * 2.0f - 1.0f, n.y * 2.0f - 1.0f);
        const float z = sqrtf(saturate(1.0f - (xy.x * xy.x + xy.y * xy.y)));
        const float scale = p.texture_normal_scale;
        p.specular_roughness = normal_filter_roughness(p.specular_roughness, n.w, scale);
        p.coat_roughness = normal_filter_roughness(p.coat_roughness, n.w, scale);
        si.shading_normal =
            safe_normalize(si.tangent * (xy.x * scale) + si.bitangent * (xy.y * scale) + si.shading_normal * z);
        si.bump_normal = si.shading_normal;
        // Same grazing-angle correction as the glTF path, from the same header,
        // so the two do not disagree about a surface a map bent past the viewer.
        if (dot(si.shading_normal, si.wo) <= 0.0f)
        {
            const float3 facingGeom =
                (dot(si.geometry_normal, si.wo) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
            si.shading_normal = ensureValidSpecularReflection(facingGeom, si.wo, si.shading_normal);
            si.diffuse_faces_away = true;
        }
    }
}

static __forceinline__ __device__ void initSurfaceInteraction(
    SurfaceInteraction& si,
    const MaterialParams& material,
    const cudaTextureObject_t* textures,
    float3 worldPosition,
    float3 worldNormal,
    float3 geomNormal,
    float3 worldTangent,
    float3 worldBinormal,
    float2 uv,
    float3 rayDir,
    float3 vertexColor = make_float3(1.0f),
    float4 textureGradients = make_float4(0.0f, 0.0f, 0.0f, 0.0f))
{
    si.position = worldPosition;
    si.shading_normal = worldNormal;
    si.geometry_normal = geomNormal;
    si.tangent = worldTangent;
    si.bitangent = worldBinormal;
    si.wo = -rayDir;
    si.front_face = dot(geomNormal, -rayDir) > 0.0f;
    // The closest-hit program does not initialise SurfaceInteraction, so every
    // field must be written on every path.
    si.diffuse_faces_away = false;
    si.bump_normal = si.shading_normal;

    // One transform for every slot of the material -- Blender drives every slot
    // from the same Mapping node, and the loader reads it that way.
    const float2 tuv = apply_texture_transform(uv, material);
    const float2 dx = apply_texture_transform_direction(make_float2(textureGradients.x, textureGradients.y), material);
    const float2 dy = apply_texture_transform_direction(make_float2(textureGradients.z, textureGradients.w), material);
    const float4 gradients = make_float4(dx.x, dx.y, dy.x, dy.y);

    si.uv = tuv;
    bsdf_init(si, material, textures, gradients);
    si.uv = uv;

    // glTF composes base colour as baseColorFactor * baseColorTexture * COLOR_0,
    // all three multiplicative. bsdf_init knows the first two.
    si.albedo *= vertexColor;

    if (material.normal_tex >= 0)
    {
        const float4 nTex = texture_sample_2d(textures[material.normal_tex], tuv, gradients);
        const float2 bumpXY = make_float2(nTex.x * 2.0f - 1.0f, nTex.y * 2.0f - 1.0f);
        // glTF scales X and Y and leaves Z, so Z is rebuilt before the scale.
        const float bumpZ = sqrtf(saturate(1.0f - (bumpXY.x * bumpXY.x + bumpXY.y * bumpXY.y)));
        const float scale = material.normal_scale;
        si.roughness = normal_filter_roughness(si.roughness, nTex.w, scale);
        si.clearcoat_roughness = normal_filter_roughness(si.clearcoat_roughness, nTex.w, scale);
        const float3 bump =
            worldTangent * (bumpXY.x * scale) + worldBinormal * (bumpXY.y * scale) + worldNormal * bumpZ;
        si.shading_normal = safe_normalize(bump);
        si.bump_normal = si.shading_normal;
        if (dot(si.shading_normal, si.wo) <= 0.0f)
        {
            const float3 facingGeom =
                (dot(si.geometry_normal, si.wo) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
            si.shading_normal = ensureValidSpecularReflection(facingGeom, si.wo, si.shading_normal);
            si.diffuse_faces_away = true;
        }
    }

    // Coverage. The base-colour texture's alpha channel is linear even when its
    // RGB is not, so it is read the same way whatever the transfer function.
    float alpha = material.base_color_alpha;
    if (material.base_color_tex >= 0)
    {
        alpha *= tex2D<float4>(textures[material.base_color_tex], tuv.x, tuv.y).w;
    }
    si.opacity = resolve_opacity(material, alpha);
}

#endif // STRELKA_OPTIX_SHADING_COMMON_H
