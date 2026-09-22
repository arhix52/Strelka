#pragma once
// Shared wavefront shading, sampling and light helpers.
// Scheduling stays in wavefront.metal; estimator rules shared with OptiX live in common headers.

#include <metal_stdlib>
#include <simd/simd.h>

#include "random_metal.h"
#include "lights_metal.h"
#include "env_light_metal.h"

#include "ShaderTypes.h"
// The two rules the OptiX closest-hit program applies as well: which directions
// the halves of the MIS estimate share, and when a vertex owes the bounce ray a
// deduction at all. Restating them by hand here is how the backends drifted.
#include <nee_pairing.h>
#include <light_alias_sampling.h>
#include <reconstruction_filter.h>
#include <temporal_reconstruction.h>
#include <strelka/material/ior_stack.h>
#include <strelka/material/blender_procedural.h>
#include <strelka/material/volume.h>
#include <strelka/material/bsdf.h>
#include <strelka/material/normal_filter.h>
#include <strelka/material/valid_reflection.h>

using namespace metal;
using namespace raytracing;

constant bool kFcEnvMap [[function_constant(0)]];
constant bool kFcLights [[function_constant(1)]];
constant bool kFcMotionBlur [[function_constant(2)]];
constant bool kFcDof [[function_constant(3)]];
constant bool kFcDebug [[function_constant(4)]];
constant bool kFcAlpha [[function_constant(5)]];
constant bool kFcFog [[function_constant(6)]];
constant bool kFcSharc [[function_constant(7)]];
constant bool kFcSubsurface [[function_constant(8)]];
constant bool kFcCurves [[function_constant(9)]];
constant bool kFcSharcUpdate [[function_constant(10)]];
constant bool kFcOpenPBR [[function_constant(11)]];
constant bool kFcRenderWorkAudit [[function_constant(12)]];
constant bool kFcRestirRayTracedDiagnostic [[function_constant(13)]];
constant bool kFcRestir [[function_constant(14)]];
constant bool kFcRisOne [[function_constant(15)]];
constant bool kFcAov [[function_constant(16)]];
constant bool kFcAllOpenPBR [[function_constant(17)]];
constant bool kFcAllNativeOpenPBR [[function_constant(19)]];
constant bool kFcOpenPBRSheenAndCoat [[function_constant(20)]];
constant bool kFcOpenPBRDispersion [[function_constant(21)]];
constant bool kFcOpenPBRTranslucency [[function_constant(22)]];
constant bool kFcOpenPBRMetallic [[function_constant(23)]];
// Profiling-only phase cut for the production shade entry points. Zero is the
// real renderer; non-zero values let Xcode report register allocation for the
// exact same prefix with later phases compiled out.
constant uint kFcShadeProbe [[function_constant(24)]];
constant bool kFcEmissiveMeshLights [[function_constant(25)]];
constant bool kFcAllAnalyticLightsRect [[function_constant(26)]];
constant bool kFcUniformRectLightSampling [[function_constant(27)]];
constant bool kFcSplitBaseNee [[function_constant(28)]];
constant bool kFcStochasticAlphaVisibility [[function_constant(29)]];
// Offline compiler probe for attributing extend's register peak. Production
// always sets this true; disabling it without a replacement preparation pass
// would leave the shade payload undefined.
constant bool kFcPrepareSurfaceGeometryInExtend [[function_constant(30)]];
constant bool kFcPrimitiveAlphaData [[function_constant(31)]];
constant bool kFcAllAlphaBlend [[function_constant(33)]];
constant bool kFcAlphaUvIdentity [[function_constant(34)]];
constant bool kFcAlphaBaseColorOne [[function_constant(35)]];
constant bool kFcCompactAlphaMaterials [[function_constant(36)]];
constant bool kFcNearestAlphaTexture [[function_constant(37)]];
constant bool kFcTextureLodCode [[function_constant(38)]];

constant bool SPEC_FOG = is_function_constant_defined(kFcFog) ? kFcFog : false;
constant bool SPEC_STOCHASTIC_ALPHA_VISIBILITY =
    is_function_constant_defined(kFcStochasticAlphaVisibility) ? kFcStochasticAlphaVisibility : true;
constant bool SPEC_ALL_ALPHA_BLEND = is_function_constant_defined(kFcAllAlphaBlend) ? kFcAllAlphaBlend : false;
constant bool SPEC_ALPHA_UV_IDENTITY = is_function_constant_defined(kFcAlphaUvIdentity) ? kFcAlphaUvIdentity : false;
constant bool SPEC_ALPHA_BASE_COLOR_ONE =
    is_function_constant_defined(kFcAlphaBaseColorOne) ? kFcAlphaBaseColorOne : false;
constant bool SPEC_COMPACT_ALPHA_MATERIALS =
    is_function_constant_defined(kFcCompactAlphaMaterials) ? kFcCompactAlphaMaterials : true;
constant bool SPEC_NEAREST_ALPHA_TEXTURE =
    is_function_constant_defined(kFcNearestAlphaTexture) ? kFcNearestAlphaTexture : false;
constant bool SPEC_TEXTURE_LOD_CODE = is_function_constant_defined(kFcTextureLodCode) ? kFcTextureLodCode : true;
constant bool SPEC_SHARC = is_function_constant_defined(kFcSharc) ? kFcSharc : false;
constant bool SPEC_SSS = is_function_constant_defined(kFcSubsurface) ? kFcSubsurface : false;
constant bool SPEC_ENV_MAP = is_function_constant_defined(kFcEnvMap) ? kFcEnvMap : true;
constant bool SPEC_LIGHTS = is_function_constant_defined(kFcLights) ? kFcLights : true;
constant bool SPEC_MOTION_BLUR = is_function_constant_defined(kFcMotionBlur) ? kFcMotionBlur : true;
constant bool SPEC_DOF = is_function_constant_defined(kFcDof) ? kFcDof : true;
constant bool SPEC_DEBUG = is_function_constant_defined(kFcDebug) ? kFcDebug : true;
constant bool SPEC_ALPHA = is_function_constant_defined(kFcAlpha) ? kFcAlpha : true;
constant bool SPEC_CURVES = is_function_constant_defined(kFcCurves) ? kFcCurves : false;
constant bool SPEC_SHARC_UPDATE = is_function_constant_defined(kFcSharcUpdate) ? kFcSharcUpdate : false;
constant bool SPEC_OPENPBR = is_function_constant_defined(kFcOpenPBR) ? kFcOpenPBR : false;
constant bool SPEC_RENDER_WORK_AUDIT = is_function_constant_defined(kFcRenderWorkAudit) ? kFcRenderWorkAudit : false;
constant bool SPEC_RESTIR_RAY_TRACED_DIAGNOSTIC =
    is_function_constant_defined(kFcRestirRayTracedDiagnostic) ? kFcRestirRayTracedDiagnostic : false;
constant bool SPEC_RESTIR = is_function_constant_defined(kFcRestir) ? kFcRestir : false;
constant bool SPEC_RIS_ONE = is_function_constant_defined(kFcRisOne) ? kFcRisOne : false;
constant bool SPEC_AOV = is_function_constant_defined(kFcAov) ? kFcAov : true;
constant bool SPEC_ALL_OPENPBR = is_function_constant_defined(kFcAllOpenPBR) ? kFcAllOpenPBR : false;
constant bool SPEC_ALL_NATIVE_OPENPBR = is_function_constant_defined(kFcAllNativeOpenPBR) ? kFcAllNativeOpenPBR : false;
constant bool SPEC_OPENPBR_SHEEN_AND_COAT =
    is_function_constant_defined(kFcOpenPBRSheenAndCoat) ? kFcOpenPBRSheenAndCoat : true;
constant bool SPEC_OPENPBR_DISPERSION = is_function_constant_defined(kFcOpenPBRDispersion) ? kFcOpenPBRDispersion : true;
constant bool SPEC_OPENPBR_TRANSLUCENCY =
    is_function_constant_defined(kFcOpenPBRTranslucency) ? kFcOpenPBRTranslucency : true;
constant bool SPEC_OPENPBR_METALLIC = is_function_constant_defined(kFcOpenPBRMetallic) ? kFcOpenPBRMetallic : true;
constant uint SPEC_SHADE_PROBE = is_function_constant_defined(kFcShadeProbe) ? kFcShadeProbe : 0u;
constant bool SPEC_EMISSIVE_MESH_LIGHTS =
    is_function_constant_defined(kFcEmissiveMeshLights) ? kFcEmissiveMeshLights : true;
constant bool SPEC_ALL_ANALYTIC_LIGHTS_RECT =
    is_function_constant_defined(kFcAllAnalyticLightsRect) ? kFcAllAnalyticLightsRect : false;
constant bool SPEC_UNIFORM_RECT_LIGHT_SAMPLING =
    is_function_constant_defined(kFcUniformRectLightSampling) ? kFcUniformRectLightSampling : false;
constant bool SPEC_SPLIT_BASE_NEE = is_function_constant_defined(kFcSplitBaseNee) ? kFcSplitBaseNee : false;
constant bool SPEC_PREPARE_SURFACE_GEOMETRY_IN_EXTEND =
    is_function_constant_defined(kFcPrepareSurfaceGeometryInExtend) ? kFcPrepareSurfaceGeometryInExtend : true;
constant bool SPEC_PRIMITIVE_ALPHA_DATA =
    is_function_constant_defined(kFcPrimitiveAlphaData) ? kFcPrimitiveAlphaData : true;

__attribute__((always_inline)) float3 transformDirection(float3 p, float3 axisX, float3 axisY, float3 axisZ)
{
    return axisX * p.x + axisY * p.y + axisZ * p.z;
}

struct FastNormalTransform
{
    float3 cofactorX;
    float3 cofactorY;
    float3 cofactorZ;
    float orientation;
};

static __attribute__((always_inline)) FastNormalTransform makeFastNormalTransform(float3 axisX, float3 axisY, float3 axisZ)
{
    FastNormalTransform result;
    result.cofactorX = cross(axisY, axisZ);
    result.cofactorY = cross(axisZ, axisX);
    result.cofactorZ = cross(axisX, axisY);
    result.orientation = dot(result.cofactorZ, axisZ) < 0.0f ? -1.0f : 1.0f;
    return result;
}

static __attribute__((always_inline)) float3
transformNormalFast(float3 n, float3 cofactorX, float3 cofactorY, float3 cofactorZ, float orientation)
{
    const float3 cofactorNormal = cofactorX * n.x + cofactorY * n.y + cofactorZ * n.z;
    return orientation * normalize(cofactorNormal);
}

// Vertex directions are RGB10A2-unorm. Metal extracts all four fields in one
// operation; the A2 field is metadata and never leaks into z.
static float3 unpackNormal(uint32_t val)
{
    return unpack_unorm10a2_to_float(val).xyz * 2.0f - 1.0f;
}

static bool openpbrHasMap(thread const OpenPBRParams& p, uint slot)
{
    return (p.texture_mask & (1u << slot)) != 0u;
}

template <typename Tex2D>
inline float texLod(Tex2D tex, float lodBase, bool hasLod)
{
    if (!hasLod)
    {
        return 0.0f;
    }
    const float dim = float(tex.get_width() * tex.get_height());
    return max(0.0f, lodBase + 0.5f * log2(max(dim, 1.0f)));
}

static float2 applyOpenPBRTextureTransform(float2 uv, float rotation, float2 scale, float2 offset)
{
    const float c = cos(rotation);
    const float sn = sin(rotation);
    return float2(uv.x * scale.x * c - uv.y * scale.y * sn, uv.x * scale.x * sn + uv.y * scale.y * c) + offset;
}

static float3 adjustLayeredTextureValues(float3 rgb,
                                         float hue,
                                         float saturation,
                                         float value,
                                         float gamma,
                                         float brightness,
                                         float contrast)
{
    if (gamma == 0.0f)
        rgb = float3(1.0f);
    else if (gamma != 1.0f)
    {
        const float exponent = max_color_correction_exponent(gamma);
        rgb = select(rgb, pow(rgb, float3(exponent)), rgb > 0.0f);
    }
    const float hi = max(rgb.x, max(rgb.y, rgb.z));
    const float lo = min(rgb.x, min(rgb.y, rgb.z));
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
        h = fract(h / 6.0f + hue);
    }
    const float s = hi > 0.0f ? saturate((d / hi) * saturation) : 0.0f;
    const float v = hi * value;
    const float sector = h * 6.0f;
    const int i = int(floor(sector));
    const float f = fract(sector);
    const float a = v * (1.0f - s);
    const float b = v * (1.0f - s * f);
    const float c = v * (1.0f - s * (1.0f - f));
    float3 adjusted;
    switch (i % 6)
    {
    case 0: adjusted = float3(v, c, a); break;
    case 1: adjusted = float3(b, v, a); break;
    case 2: adjusted = float3(a, v, c); break;
    case 3: adjusted = float3(a, b, v); break;
    case 4: adjusted = float3(c, a, v); break;
    default: adjusted = float3(v, a, b); break;
    }
    return max(adjusted * (1.0f + contrast) + brightness - contrast * 0.5f, 0.0f);
}

static float3 adjustLayeredTexture(float3 rgb,
                                   device const OpenPBRLayeredTextureParams& p,
                                   bool data,
                                   uint layer)
{
    return data ?
               adjustLayeredTextureValues(rgb, p.data_hue[layer], p.data_saturation[layer], p.data_value[layer],
                                          p.data_gamma[layer], p.data_brightness[layer], p.data_contrast[layer]) :
               adjustLayeredTextureValues(rgb, p.color_hue[layer], p.color_saturation[layer],
                                          p.color_value[layer], p.color_gamma[layer],
                                          p.color_brightness[layer], p.color_contrast[layer]);
}

static float3 sampleLayeredSource(device const OpenPBRTextures& t,
                                  uint firstSlot,
                                  uint layer,
                                  float2 uv,
                                  float lodBase,
                                  bool hasLod)
{
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear, mip_filter::linear, address::repeat);
    const uint slot = firstSlot + layer;
    if (is_null_texture(t.tex[slot]))
        return 0.0f;
    device const OpenPBRLayeredTextureParams& p = t.layered;
    const bool data = firstSlot == OPENPBR_TEX_LAYER_DATA_0;
    const float2 scale = data ? float2(p.data_uv_scale_x[layer], p.data_uv_scale_y[layer]) :
                                float2(p.uv_scale_x[layer], p.uv_scale_y[layer]);
    const float rotation = data ? p.data_uv_rotation[layer] : p.uv_rotation[layer];
    const float2 offset = data ? float2(p.data_uv_offset_x[layer], p.data_uv_offset_y[layer]) :
                                 float2(p.uv_offset_x[layer], p.uv_offset_y[layer]);
    const float c = cos(rotation);
    const float sn = sin(rotation);
    const float2 centered = uv - 0.5f;
    const float2 transformed = float2((centered.x * c - centered.y * sn) * scale.x,
                                       (centered.x * sn + centered.y * c) * scale.y) +
                               float2(0.5f) + offset;
    const float transformedLodBase = hasLod ? lodBase + log2(max(max(abs(scale.x), abs(scale.y)), 1.0e-8f)) : lodBase;
    return t.tex[slot].sample(textureSampler, transformed,
                              level(texLod(t.tex[slot], transformedLodBase, hasLod))).rgb;
}

static float3 blendLayered(float3 base, float3 layer, float factor, uint mode)
{
    if (mode == OPENPBR_LAYER_BLEND_MIX)
        return mix(base, layer, factor);
    if (mode == OPENPBR_LAYER_BLEND_SCREEN)
        return 1.0f - ((1.0f - factor) + (1.0f - layer) * factor) * (1.0f - base);
    if (mode == OPENPBR_LAYER_BLEND_OVERLAY)
        return select(1.0f - ((1.0f - factor) + 2.0f * factor * (1.0f - layer)) * (1.0f - base),
                      base * ((1.0f - factor) + 2.0f * factor * layer), base < 0.5f);
    return base * ((1.0f - factor) + layer * factor);
}

static float3 sampleLayeredTexture(device const OpenPBRTextures& t,
                                   uint firstSlot,
                                   float2 uv,
                                   float lodBase,
                                   bool hasLod,
                                   float facing)
{
    device const OpenPBRLayeredTextureParams& p = t.layered;
    const bool data = firstSlot == OPENPBR_TEX_LAYER_DATA_0;
    const bool hasBase = data ? p.data_has_base != 0u : p.color_has_base != 0u;
    float3 result = hasBase ?
                        (data ? float3(p.data_base[0], p.data_base[1], p.data_base[2]) :
                                float3(p.color_base[0], p.color_base[1], p.color_base[2])) :
                        float3(1.0f);
    for (uint layer = 0; layer < p.layer_count; ++layer)
    {
        if (is_null_texture(t.tex[firstSlot + layer]))
            continue;
        const float3 source = sampleLayeredSource(t, firstSlot, layer, uv, lodBase, hasLod);
        const float3 value = adjustLayeredTexture(source, p, data, layer);
        if (layer == 0u && !hasBase)
        {
            result = value;
        }
        else
        {
            if (data)
                result = blendLayered(result, value, p.data_opacity[layer], p.data_blend_mode[layer]);
            else
            {
                float map = 0.0f;
                const uint factorLayer = p.color_factor_data_layer[layer];
                if (factorLayer < 4u && !is_null_texture(t.tex[OPENPBR_TEX_LAYER_DATA_0 + factorLayer]))
                    map = luminance(sampleLayeredSource(
                        t, OPENPBR_TEX_LAYER_DATA_0, factorLayer, uv, lodBase, hasLod));
                const float factor = p.color_factor_base[layer] + p.color_factor_data[layer] * map +
                                     p.color_factor_facing[layer] * facing +
                                     p.color_factor_facing_data[layer] * facing * map;
                result = blendLayered(result, value, factor, p.color_blend_mode[layer]);
            }
        }
    }
    if (!data)
        result = adjustLayeredTextureValues(result, p.color_post_hue, p.color_post_saturation,
                                            p.color_post_value, p.color_post_gamma,
                                            p.color_post_brightness, p.color_post_contrast);
    return result;
}

static float layeredBumpHeight(device const OpenPBRTextures& t,
                               float2 uv,
                               float lodBase,
                               bool hasLod)
{
    device const OpenPBRLayeredTextureParams& p = t.layered;
    float texture = 0.0f;
    if (p.bump_data_layer < 4u && !is_null_texture(t.tex[OPENPBR_TEX_LAYER_DATA_0 + p.bump_data_layer]))
        texture = luminance(adjustLayeredTexture(
            sampleLayeredSource(t, OPENPBR_TEX_LAYER_DATA_0, p.bump_data_layer, uv, lodBase, hasLod),
            p, true, p.bump_data_layer));
    else if (p.bump_procedural == OPENPBR_LAYER_PROCEDURAL_NONE)
        texture = luminance(sampleLayeredTexture(
            t, OPENPBR_TEX_LAYER_DATA_0, uv, lodBase, hasLod, 0.0f));
    float procedural = texture;
    if (p.bump_procedural == OPENPBR_LAYER_PROCEDURAL_VORONOI_RIDGE)
        procedural = blender_voronoi_ridge(uv, p.bump_procedural_scale);
    else if (p.bump_procedural == OPENPBR_LAYER_PROCEDURAL_NOISE)
        procedural = blender_noise_fbm(uv, p.bump_procedural_scale, p.bump_procedural_detail,
                                       p.bump_procedural_roughness, p.bump_procedural_lacunarity);
    return mix(procedural, texture, p.bump_texture_mix) * p.bump_gain;
}

static void applyOpenPBRLayeredTexture(thread OpenPBRParams& p,
                                       device const OpenPBRTextures& t,
                                       thread SurfaceInteraction& si,
                                       float2 uv,
                                       float lodBase)
{
    device const OpenPBRLayeredTextureParams& graph = t.layered;
    if (graph.output_mask == 0u)
        return;

    const bool hasLod = lodBase > -1e29f;
    const float facing = 1.0f - saturate(abs(dot(si.shading_normal, si.wo)));
    const float3 data = sampleLayeredTexture(t, OPENPBR_TEX_LAYER_DATA_0, uv, lodBase, hasLod, facing);
    const float3 color = sampleLayeredTexture(t, OPENPBR_TEX_LAYER_COLOR_0, uv, lodBase, hasLod, facing);
    if ((graph.output_mask & OPENPBR_LAYER_OUTPUT_BASE_COLOR) != 0u)
        p.base_color = OpenPBRColor{ saturate(color.r), saturate(color.g), saturate(color.b) };
    if ((graph.output_mask & OPENPBR_LAYER_OUTPUT_SPECULAR_COLOR) != 0u)
    {
        const float3 source = graph.specular_color_uses_color != 0u ? color : data;
        const float3 base = float3(graph.specular_color_base[0], graph.specular_color_base[1],
                                   graph.specular_color_base[2]);
        const float3 specular = mix(base, saturate(source * graph.specular_gain),
                                    graph.specular_color_mix);
        p.specular_color = OpenPBRColor{ specular.r, specular.g, specular.b };
    }
    if ((graph.output_mask & OPENPBR_LAYER_OUTPUT_SPECULAR_WEIGHT) != 0u)
    {
        const float reflectivity = luminance(data) * graph.specular_gain;
        p.specular_weight = saturate(p.specular_weight * reflectivity);
    }
    if ((graph.output_mask & OPENPBR_LAYER_OUTPUT_SPECULAR_IOR) != 0u)
    {
        const float3 source = graph.specular_ior_uses_color != 0u ? color : data;
        const float map = saturate(luminance(source) * graph.specular_gain);
        const float reflectivity = mix(graph.specular_ior_base, map, graph.specular_ior_mix);
        p.specular_ior = mix(1.0f, graph.specular_ior_authored, reflectivity);
    }
    if ((graph.output_mask & OPENPBR_LAYER_OUTPUT_OPACITY) != 0u)
    {
        constexpr sampler opacitySampler(mag_filter::linear, min_filter::linear, mip_filter::linear,
                                         address::repeat);
        const uint layer = graph.opacity_layer;
        const uint slot = OPENPBR_TEX_LAYER_DATA_0 + layer;
        if (!is_null_texture(t.tex[slot]))
        {
            const float2 scale = float2(graph.data_uv_scale_x[layer], graph.data_uv_scale_y[layer]);
            const float c = cos(graph.data_uv_rotation[layer]);
            const float sn = sin(graph.data_uv_rotation[layer]);
            const float2 centered = uv - 0.5f;
            const float2 tuv = float2((centered.x * c - centered.y * sn) * scale.x,
                                      (centered.x * sn + centered.y * c) * scale.y) +
                               float2(0.5f, 0.5f) +
                               float2(graph.data_uv_offset_x[layer], graph.data_uv_offset_y[layer]);
            const float3 value = t.tex[slot].sample(opacitySampler, tuv).rgb;
            p.geometry_opacity = saturate(graph.opacity_base + luminance(value) * graph.opacity_mix +
                                          facing * graph.opacity_facing_mix);
        }
    }
    if ((graph.output_mask & OPENPBR_LAYER_OUTPUT_ROUGHNESS) != 0u)
    {
        const float height = luminance(data) * graph.roughness_gain;
        p.specular_roughness = saturate(mix(graph.roughness_base, 1.0f - height, graph.roughness_mix));
    }
    if ((graph.output_mask & OPENPBR_LAYER_OUTPUT_NORMAL) == 0u)
        return;

    constexpr float bumpFilterWidth = 0.1f;
    float step = bumpFilterWidth / 2048.0f;
    if (hasLod && !is_null_texture(t.tex[OPENPBR_TEX_LAYER_DATA_0]))
    {
        step *= exp2(texLod(t.tex[OPENPBR_TEX_LAYER_DATA_0], lodBase, true));
    }
    // Match Cycles' Color-to-Float conversion (scene-linear luminance).
    const float h = layeredBumpHeight(t, uv, lodBase, hasLod);
    const float hx = layeredBumpHeight(t, uv + float2(step, 0.0f), lodBase, hasLod);
    const float hy = layeredBumpHeight(t, uv + float2(0.0f, step), lodBase, hasLod);
    const float dhdu =
        (hx - h) * graph.bump_scale / step;
    const float dhdv =
        (hy - h) * graph.bump_scale / step;
    si.shading_normal = normalize(si.shading_normal - si.tangent * dhdu - si.bitangent * dhdv);
    if (dot(si.shading_normal, si.wo) <= 0.0f)
    {
        const float3 facingGeom = dot(si.geometry_normal, si.wo) > 0.0f ? si.geometry_normal : -si.geometry_normal;
        si.shading_normal = ensureValidSpecularReflection(facingGeom, si.wo, si.shading_normal);
        si.diffuse_faces_away = true;
    }
}

static void applyOpenPBRTextures(thread OpenPBRParams& p,
                                 device const OpenPBRTextures& t,
                                 thread SurfaceInteraction& si,
                                 float2 uv,
                                 float lodBase = -1e30f,
                                 bool uvPretransformed = false)
{
    const float2 tuv = uvPretransformed ?
                           uv :
                           applyOpenPBRTextureTransform(uv, p.uv_rotation, float2(p.uv_scale_x, p.uv_scale_y),
                                                        float2(p.uv_offset_x, p.uv_offset_y));
    const bool hasLod = lodBase > -1e29f;
    constexpr sampler openpbrSampler(mag_filter::linear, min_filter::linear, mip_filter::linear, address::repeat);
#define SAMPLE_OPENPBR_TEXTURE(slot)                                                                                   \
    t.tex[slot].sample(openpbrSampler, tuv, level(texLod(t.tex[slot], lodBase, hasLod)))

    if (openpbrHasMap(p, OPENPBR_TEX_BASE_COLOR) && !is_null_texture(t.tex[OPENPBR_TEX_BASE_COLOR]))
    {
        const float3 v = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_BASE_COLOR).rgb;
        p.base_color = (p.texture_scalar_flags & OPENPBR_TEXTURES_GLTF) != 0u ?
                           OpenPBRColor{ p.base_color.r * v.r, p.base_color.g * v.g, p.base_color.b * v.b } :
                           OpenPBRColor{ v.r, v.g, v.b };
    }
    if (openpbrHasMap(p, OPENPBR_TEX_BASE_METALNESS) && !is_null_texture(t.tex[OPENPBR_TEX_BASE_METALNESS]))
        p.base_metalness = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_BASE_METALNESS).r;
    if (openpbrHasMap(p, OPENPBR_TEX_SPECULAR_ROUGHNESS) && !is_null_texture(t.tex[OPENPBR_TEX_SPECULAR_ROUGHNESS]))
    {
        const float4 v = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_SPECULAR_ROUGHNESS);
        const float roughness = v[p.texture_scalar_flags & OPENPBR_ROUGHNESS_CHANNEL_MASK];
        p.specular_roughness = (p.texture_scalar_flags & OPENPBR_ROUGHNESS_MULTIPLY) != 0u ?
                                   p.specular_roughness * roughness :
                                   roughness;
    }
    if (openpbrHasMap(p, OPENPBR_TEX_SPECULAR_ANISOTROPY) && !is_null_texture(t.tex[OPENPBR_TEX_SPECULAR_ANISOTROPY]))
        p.specular_roughness_anisotropy = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_SPECULAR_ANISOTROPY).r;
    if (openpbrHasMap(p, OPENPBR_TEX_SPECULAR_COLOR) && !is_null_texture(t.tex[OPENPBR_TEX_SPECULAR_COLOR]))
    {
        const float3 v = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_SPECULAR_COLOR).rgb;
        p.specular_color = OpenPBRColor{ v.r, v.g, v.b };
    }
    if (openpbrHasMap(p, OPENPBR_TEX_COAT_WEIGHT) && !is_null_texture(t.tex[OPENPBR_TEX_COAT_WEIGHT]))
        p.coat_weight = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_COAT_WEIGHT).r;
    if (openpbrHasMap(p, OPENPBR_TEX_COAT_ROUGHNESS) && !is_null_texture(t.tex[OPENPBR_TEX_COAT_ROUGHNESS]))
        p.coat_roughness = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_COAT_ROUGHNESS).r;
    if (openpbrHasMap(p, OPENPBR_TEX_COAT_COLOR) && !is_null_texture(t.tex[OPENPBR_TEX_COAT_COLOR]))
    {
        const float3 v = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_COAT_COLOR).rgb;
        p.coat_color = OpenPBRColor{ v.r, v.g, v.b };
    }
    if (openpbrHasMap(p, OPENPBR_TEX_FUZZ_WEIGHT) && !is_null_texture(t.tex[OPENPBR_TEX_FUZZ_WEIGHT]))
        p.fuzz_weight = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_FUZZ_WEIGHT).r;
    if (openpbrHasMap(p, OPENPBR_TEX_FUZZ_ROUGHNESS) && !is_null_texture(t.tex[OPENPBR_TEX_FUZZ_ROUGHNESS]))
        p.fuzz_roughness = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_FUZZ_ROUGHNESS).r;
    if (openpbrHasMap(p, OPENPBR_TEX_TRANSMISSION_COLOR) && !is_null_texture(t.tex[OPENPBR_TEX_TRANSMISSION_COLOR]))
    {
        const float3 v = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_TRANSMISSION_COLOR).rgb;
        p.transmission_color = OpenPBRColor{ v.r, v.g, v.b };
    }
    if (openpbrHasMap(p, OPENPBR_TEX_SUBSURFACE_COLOR) && !is_null_texture(t.tex[OPENPBR_TEX_SUBSURFACE_COLOR]))
    {
        const float3 v = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_SUBSURFACE_COLOR).rgb;
        p.subsurface_color = OpenPBRColor{ v.r, v.g, v.b };
    }
    if (openpbrHasMap(p, OPENPBR_TEX_GEOMETRY_OPACITY) && !is_null_texture(t.tex[OPENPBR_TEX_GEOMETRY_OPACITY]))
        p.geometry_opacity = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_GEOMETRY_OPACITY).r;
    if (openpbrHasMap(p, OPENPBR_TEX_SUBSURFACE_WEIGHT) && !is_null_texture(t.tex[OPENPBR_TEX_SUBSURFACE_WEIGHT]))
        p.subsurface_weight = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_SUBSURFACE_WEIGHT).r;
    if (openpbrHasMap(p, OPENPBR_TEX_FUZZ_COLOR) && !is_null_texture(t.tex[OPENPBR_TEX_FUZZ_COLOR]))
    {
        const float3 v = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_FUZZ_COLOR).rgb;
        p.fuzz_color = OpenPBRColor{ v.r, v.g, v.b };
    }
    if (openpbrHasMap(p, OPENPBR_TEX_SUBSURFACE_RADIUS) && !is_null_texture(t.tex[OPENPBR_TEX_SUBSURFACE_RADIUS]))
    {
        // A per-channel tint on the mean free path. The scalar length stays as
        // authored: a map here says how the three channels differ, not how far
        // light travels, which is what subsurface_radius carries.
        const float3 v = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_SUBSURFACE_RADIUS).rgb;
        p.subsurface_radius_scale = OpenPBRColor{ v.r, v.g, v.b };
    }
    // Preserve nonlinear graph semantics: reconstruct evaluated samples, not
    // independently mipped graph inputs. This matches Cycles Image Texture.
    applyOpenPBRLayeredTexture(p, t, si, uv, -1e30f);
    if (openpbrHasMap(p, OPENPBR_TEX_EMISSION_COLOR) && !is_null_texture(t.tex[OPENPBR_TEX_EMISSION_COLOR]))
    {
        // Emissive-mesh NEE cannot reconstruct this surface ray's cone. Keep
        // both strategies on level zero until they share an explicit footprint.
        constexpr sampler emissionSampler(mag_filter::linear, min_filter::linear, address::repeat);
        const float3 v = t.tex[OPENPBR_TEX_EMISSION_COLOR].sample(emissionSampler, tuv).rgb;
        p.emission_color = (p.texture_scalar_flags & OPENPBR_TEXTURES_GLTF) != 0u ?
                               OpenPBRColor{ p.emission_color.r * v.r, p.emission_color.g * v.g,
                                             p.emission_color.b * v.b } :
                               OpenPBRColor{ v.r, v.g, v.b };
    }
    si.emission = float3(p.emission_color.r, p.emission_color.g, p.emission_color.b) * p.emission_luminance;

    // The normal map, read the same way the glTF one is: Z rebuilt from X and Y
    // because a compressed normal map is BC5 and stores two channels.
    if (openpbrHasMap(p, OPENPBR_TEX_GEOMETRY_NORMAL) && !is_null_texture(t.tex[OPENPBR_TEX_GEOMETRY_NORMAL]))
    {
        const float4 normalSample = SAMPLE_OPENPBR_TEXTURE(OPENPBR_TEX_GEOMETRY_NORMAL);
        const float2 xy = normalSample.xy * 2.0f - 1.0f;
        const float z = sqrt(saturate(1.0f - dot(xy, xy)));
        p.specular_roughness = normal_filter_roughness(p.specular_roughness, normalSample.a, p.texture_normal_scale);
        p.coat_roughness = normal_filter_roughness(p.coat_roughness, normalSample.a, p.texture_normal_scale);
        const float3x3 TBN = float3x3(si.tangent, si.bitangent, si.shading_normal);
        si.shading_normal = normalize(TBN * float3(xy * p.texture_normal_scale, z));
        si.bump_normal = si.shading_normal;
        // Same grazing-angle correction as the glTF path, from the same header,
        // so the two do not disagree about a surface a map bent past the viewer.
        if (dot(si.shading_normal, si.wo) <= 0.0f)
        {
            const float3 facingGeom = (dot(si.geometry_normal, si.wo) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
            si.shading_normal = ensureValidSpecularReflection(facingGeom, si.wo, si.shading_normal);
            si.diffuse_faces_away = true;
        }
    }
#undef SAMPLE_OPENPBR_TEXTURE
}

static float2 applyTextureTransform(float2 uv, device const Material& m)
{
    return float2(dot(uv, float2(m.uv_transform_x)), dot(uv, float2(m.uv_transform_y))) + float2(m.uv_offset);
}

// Coverage of a surface at a given uv. MASK is a binary predicate, BLEND passes
// the alpha through, OPAQUE is always 1 -- so callers only ever see a float in
// [0,1] and never need to branch on the mode themselves.
static float resolveOpacity(device const Material& material, float2 uv, bool uvPretransformed = false)
{
    if (material.alpha_mode == ALPHA_MODE_OPAQUE || material.alpha_mode == ALPHA_MODE_SHADOW_TRANSPARENT)
        return 1.0f;
    // glTF defaults to REPEAT, so transformed UVs must not use Metal's clamp-to-edge default.
    constexpr sampler alphaSampler(mag_filter::linear, min_filter::linear, address::repeat);
    float alpha = material.base_color_alpha;
    if ((material.features & MATERIAL_TEX_BASE_COLOR) != 0u && !is_null_texture(material.baseColorTexture))
    {
        if (!uvPretransformed)
        {
            uv = applyTextureTransform(uv, material);
        }
        // RGBA8Unorm_sRGB puts only RGB through the transfer function, so the
        // alpha channel read here is already linear.
        alpha *= material.baseColorTexture.sample(alphaSampler, uv).a;
    }
    if (material.alpha_mode == ALPHA_MODE_MASK)
        return alpha >= material.alpha_cutoff ? 1.0f : 0.0f;
    return saturate(alpha);
}

// glTF COLOR_0, packed RGBA8 and LINEAR -- it carries no transfer function,
// unlike a base-colour texture, so nothing is decoded here.
static float3 unpackVertexColor(uint32_t val)
{
    return unpack_unorm4x8_to_float(val).rgb;
}

//  valid range of coordinates [-10; 10]
static float2 unpackUV(uint32_t val)
{
    float2 uv;
    uv.y = ((val & 0xffff0000) >> 16) / 16383.99999f * 20.0f - 10.0f;
    uv.x = (val & 0x0000ffff) / 16383.99999f * 20.0f - 10.0f;
    return uv;
}

static float2 unpackUV(uint32_t x, uint32_t y)
{
    return float2(as_type<float>(x), as_type<float>(y));
}

static __attribute__((always_inline)) float3 interpolateAttrib(const float3 attr1,
                                                               const float3 attr2,
                                                               const float3 attr3,
                                                               const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

static __attribute__((always_inline)) float2 interpolateAttrib(const float2 attr1,
                                                               const float2 attr2,
                                                               const float2 attr3,
                                                               const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

// Whether a radiance carries any energy at all. Testing every channel instead
// would drop a saturated light: a pure red one has two zero channels and still
// lights the scene.
static __attribute__((always_inline)) bool emitsLight(const float3 radiance)
{
    return radiance.x > 0.0f || radiance.y > 0.0f || radiance.z > 0.0f;
}

__attribute__((always_inline)) float4x4 lerpMatrix(float4x4 a, float4x4 b, float t)
{
    float4x4 r;
    r[0] = mix(a[0], b[0], t);
    r[1] = mix(a[1], b[1], t);
    r[2] = mix(a[2], b[2], t);
    r[3] = mix(a[3], b[3], t);
    return r;
}

// Concentric disk mapping (Shirley & Chiu 1997)
float2 concentricDiskSample(float u1, float u2)
{
    float2 offset = float2(2.0f * u1 - 1.0f, 2.0f * u2 - 1.0f);
    if (offset.x == 0.0f && offset.y == 0.0f)
        return float2(0.0f, 0.0f);

    float theta, r;
    if (abs(offset.x) > abs(offset.y))
    {
        r = offset.x;
        theta = (M_PI_F / 4.0f) * (offset.y / offset.x);
    }
    else
    {
        r = offset.y;
        theta = (M_PI_F / 2.0f) - (M_PI_F / 4.0f) * (offset.x / offset.y);
    }
    return float2(r * cos(theta), r * sin(theta));
}

// Sample regular polygon aperture (blades >= 3)
float2 samplePolygonAperture(float u1, float u2, int blades)
{
    float sectorAngle = 2.0f * M_PI_F / (float)blades;
    int sector = (int)(u1 * blades);
    if (sector >= blades)
        sector = blades - 1;
    float u = u1 * blades - (float)sector;

    float su = sqrt(u);
    float bary0 = 1.0f - su;
    float bary1 = u2 * su;

    float angle0 = sectorAngle * sector;
    float angle1 = sectorAngle * (sector + 1);

    float x = bary1 * cos(angle0) + (1.0f - bary0 - bary1) * cos(angle1);
    float y = bary1 * sin(angle0) + (1.0f - bary0 - bary1) * sin(angle1);
    return float2(x, y);
}

float2 sampleAperture(thread SamplerState& sampler, const constant Uniforms& params)
{
    const float2 lensSample =
        random2<SampleDimension::eLensU, SampleDimension::eLensV>(sampler, params.samplerType).value;
    const float u1 = lensSample.x;
    const float u2 = lensSample.y;

    float2 p;
    if (params.apertureBlades < 3)
        p = concentricDiskSample(u1, u2);
    else
        p = samplePolygonAperture(u1, u2, params.apertureBlades);

    if (params.bladeRotation != 0.0f)
    {
        float cosR = cos(params.bladeRotation);
        float sinR = sin(params.bladeRotation);
        p = float2(p.x * cosR - p.y * sinR, p.x * sinR + p.y * cosR);
    }

    p.y *= params.anamorphicRatio;
    return p;
}

inline float3 clampPathContribution(float3 radiance, uint depth, float directLimit, float indirectLimit)
{
    const float limit = depth == 0u ? directLimit : indirectLimit;
    if (limit <= 0.0f)
    {
        return radiance;
    }
    const float m = max(radiance.x, max(radiance.y, radiance.z));
    return (m > limit) ? radiance * (limit / m) : radiance;
}

void generateCameraRay(uint2 pixelIndex,
                       thread SamplerState& samplerRnd,
                       thread float3& origin,
                       thread float3& direction,
                       const constant Uniforms& params,
                       float motionTime,
                       thread float& reconstructionWeight)
{
    float2 subpixel_jitter;
    reconstructionWeight = 1.0f;
    if (params.useFrameJitter)
    {
        subpixel_jitter =
            float2(strelkaCameraSampleCoordinate(params.jitterX), strelkaCameraSampleCoordinate(params.jitterY));
    }
    else
    {
        const float2 randomSample =
            random2<SampleDimension::ePixelX, SampleDimension::ePixelY>(samplerRnd, params.samplerType).value;
        const uint filter = (params.textureLodMode & RECONSTRUCTION_FILTER_MASK) >> RECONSTRUCTION_FILTER_SHIFT;
        const ReconstructionFilterSample sampleX = reconstructionFilterSample(filter, randomSample.x);
        const ReconstructionFilterSample sampleY = reconstructionFilterSample(filter, randomSample.y);
        subpixel_jitter = float2(0.5f + sampleX.offset, 0.5f + sampleY.offset);
        reconstructionWeight = sampleX.weight * sampleY.weight;
    }
    float2 pixelPos{ pixelIndex.x + subpixel_jitter.x, params.height - (pixelIndex.y + subpixel_jitter.y) };

    float2 dimension{ (float)params.width, (float)params.height };
    float2 pixelNDC = (pixelPos / dimension) * 2.0f - 1.0f;

    // Lens shift
    pixelNDC.x += params.shiftX * 2.0f;
    pixelNDC.y += params.shiftY * 2.0f;

    // Camera motion retains the matrix path because interpolating two expanded
    // endpoint bases would not equal the existing product of interpolated
    // projection and view matrices.
    float4x4 viewToWorld = params.viewToWorld;
    const bool interpolateCamera = SPEC_MOTION_BLUR && motionTime < 1.0f && params.enableCameraMotionBlur;
    if (interpolateCamera)
    {
        viewToWorld = lerpMatrix(params.prevViewToWorld, params.viewToWorld, motionTime);
    }

    if (params.projectionType == PROJECTION_ORTHOGRAPHIC)
    {
        const float3 filmPos = float3(pixelNDC.x * params.orthoHalfWidth, pixelNDC.y * params.orthoHalfHeight, 0.0f);
        origin = (viewToWorld * float4(filmPos, 1.0f)).xyz;
        direction = normalize((viewToWorld * float4(0.0f, 0.0f, -1.0f, 0.0f)).xyz);
    }
    else
    {
        if (interpolateCamera)
        {
            const float4x4 clipToView = lerpMatrix(params.prevClipToView, params.clipToView, motionTime);
            const float4 viewSpace = clipToView * float4(pixelNDC.x, pixelNDC.y, 1.0f, 1.0f);
            direction = normalize((viewToWorld * float4(viewSpace.xyz, 0.0f)).xyz);
        }
        else
        {
            direction = normalize(float3(params.cameraRayForward) + pixelNDC.x * float3(params.cameraRayRight) +
                                  pixelNDC.y * float3(params.cameraRayUp));
        }
        origin = viewToWorld[3].xyz;
    }

    // Thin lens depth of field
    if (SPEC_DOF && params.useDof && params.lensRadius > 0.0f)
    {
        // The world-space camera basis is stored in the columns of viewToWorld.
        float3 camRight = viewToWorld[0].xyz;
        float3 camUp = viewToWorld[1].xyz;
        float3 camFwd = -viewToWorld[2].xyz;

        float t = params.focalDistance / max(dot(direction, camFwd), 1e-6f);
        float3 focalPoint = origin + direction * t;

        float2 lensSample = sampleAperture(samplerRnd, params) * params.lensRadius;
        origin += camRight * lensSample.x + camUp * lensSample.y;
        direction = normalize(focalPoint - origin);
    }

    origin += direction * params.cameraNear;
}

// Geometry-only half of surface initialisation. Native OpenPBR materials use
// it without ever touching the parallel generic Material record; their maps
// need the frame before their parameter block can be finalised below.
static void initSurfaceGeometry(thread SurfaceInteraction& si,
                                float3 worldPosition,
                                float3 worldNormal,
                                float3 geomNormal,
                                float3 worldTangent,
                                float3 worldBinormal,
                                float2 uv,
                                float3 rayDir)
{
    si.position = worldPosition;
    si.shading_normal = worldNormal;
    si.geometry_normal = geomNormal;
    si.tangent = worldTangent;
    si.bitangent = worldBinormal;
    si.uv = uv;
    si.wo = -rayDir;
    si.front_face = dot(geomNormal, -rayDir) > 0.0f;
    si.diffuse_faces_away = false;
    si.bump_normal = worldNormal;
}

static __attribute__((always_inline)) void initOpenPBRSurfaceMaterial(thread SurfaceInteraction& si,
                                                                      device const OpenPBRParams& p,
                                                                      float3 vertexColor)
{
    // OpenPBR opacity has historically not participated in Metal traversal or
    // shadow coverage. Preserve that contract here; making it coherent across
    // all ray types is a separate correctness change, not part of this fast path.
    si.opacity = 1.0f;
    si.roughness = 0.0001f;
    si.ior = 0.0f;
    si.transmission = 0.0f;
    si.emission = float3(p.emission_color.r, p.emission_color.g, p.emission_color.b) * p.emission_luminance;
    si.diffuse_transmission = 0.0f;
    si.subsurface = saturate(p.subsurface_weight);
    si.thin_walled = p.geometry_thin_walled;
    si.dielectric_priority = 0u;
    si.exterior_ior = 1.0f;

    // SHaRC's current common demodulation path still projects OpenPBR onto the
    // generic base/specular fields. Keep that cold compatibility state only in
    // variants which can actually read it.
    if (SPEC_SHARC || SPEC_SHARC_UPDATE)
    {
        const float3 base = float3(p.base_color.r, p.base_color.g, p.base_color.b);
        si.albedo = base * vertexColor;
        si.metallic = 0.0f;
        si.specular = 0.0f;
        si.specular_color = float3(1.0f);
    }
}

// Fill SurfaceInteraction from hit geometry and sample Material textures
void initSurfaceInteraction(thread SurfaceInteraction& si,
                            const device Material& material,
                            float3 worldPosition,
                            float3 worldNormal,
                            float3 geomNormal,
                            float3 worldTangent,
                            float3 worldBinormal,
                            float2 uv,
                            float3 rayDir,
                            float3 vertexColor = float3(1.0f),
                            float lodBase = -1e30f,
                            bool uvPretransformed = false)
{
    // Keep a non-mip sampler so disabling LOD preserves explicit level-zero filtering; glTF wrapping remains REPEAT.
    constexpr sampler texSampler(mag_filter::linear, min_filter::linear, address::repeat);
    constexpr sampler texSamplerMip(mag_filter::linear, min_filter::linear, mip_filter::linear, address::repeat);

    si.position = worldPosition;
    si.shading_normal = worldNormal;
    si.geometry_normal = geomNormal;
    si.tangent = worldTangent;
    si.bitangent = worldBinormal;
    si.uv = uv;
    // The cone gives texels per world unit; a texture turns that into a level
    // once its own resolution is known. Clamped at zero because a cone narrower
    // than a texel still wants the sharpest mip, not a negative one.
    const bool hasLod = lodBase > -1e29f;
    // One transform for every slot of the material -- see readTextureTransform()
    // in the loader for why that is not a compromise in practice.
    const uint32_t materialFeatures = material.features;
    const bool transformUv = (materialFeatures & MATERIAL_TEXTURE_MASK) != 0u && !uvPretransformed;
    const float2 tuv = transformUv ? applyTextureTransform(uv, material) : uv;
    si.wo = -rayDir;
    si.front_face = dot(geomNormal, -rayDir) > 0.0f;
    // Initialize every field because callers may pass an uninitialized SurfaceInteraction.
    si.diffuse_faces_away = false;
    si.bump_normal = si.shading_normal;

    // Sample base color texture. glTF composes base colour as
    // baseColorFactor * baseColorTexture * COLOR_0, all three multiplicative.
    float3 baseColor = float3(material.base_color) * vertexColor;
    if ((materialFeatures & MATERIAL_TEX_BASE_COLOR) != 0u && !is_null_texture(material.baseColorTexture))
    {
        baseColor *= (hasLod ? material.baseColorTexture.sample(
                                   texSamplerMip, tuv, level(texLod(material.baseColorTexture, lodBase, hasLod))) :
                               material.baseColorTexture.sample(texSampler, tuv))
                         .rgb;
    }
    si.albedo = baseColor;
    si.opacity = resolveOpacity(material, uv, uvPretransformed);

    // Sample metallic-roughness texture (glTF: G = roughness, B = metallic)
    float resolvedRoughness = material.roughness;
    float resolvedClearcoatRoughness = material.clearcoat_roughness;
    float resolvedMetallic = material.metallic;
    if ((materialFeatures & MATERIAL_TEX_METALLIC_ROUGHNESS) != 0u && !is_null_texture(material.metallicRoughnessTexture))
    {
        float4 mrTex =
            (hasLod ? material.metallicRoughnessTexture.sample(
                          texSamplerMip, tuv, level(texLod(material.metallicRoughnessTexture, lodBase, hasLod))) :
                      material.metallicRoughnessTexture.sample(texSampler, tuv));
        resolvedRoughness *= mrTex.g;
        resolvedMetallic *= mrTex.b;
    }

    if ((materialFeatures & MATERIAL_TEX_NORMAL) != 0u && !is_null_texture(material.normalTexture))
    {
        const float4 normalSample = hasLod ? material.normalTexture.sample(
                                                texSamplerMip, tuv, level(texLod(material.normalTexture, lodBase, hasLod))) :
                                            material.normalTexture.sample(texSampler, tuv);
        float2 bumpXY = normalSample.xy * 2.0f - 1.0f;
        // glTF scales X and Y and leaves Z, so Z is rebuilt before the scale.
        const float bumpZ = sqrt(saturate(1.0f - dot(bumpXY, bumpXY)));
        resolvedRoughness = normal_filter_roughness(resolvedRoughness, normalSample.a, material.normal_scale);
        resolvedClearcoatRoughness =
            normal_filter_roughness(resolvedClearcoatRoughness, normalSample.a, material.normal_scale);
        float3 bumpNormal = float3(bumpXY * material.normal_scale, bumpZ);
        float3x3 TBN = float3x3(worldTangent, worldBinormal, worldNormal);
        si.shading_normal = normalize(TBN * bumpNormal);
        si.bump_normal = si.shading_normal;
        if (dot(si.shading_normal, si.wo) <= 0.0f)
        {
            const float3 facingGeom = (dot(si.geometry_normal, si.wo) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
            si.shading_normal = ensureValidSpecularReflection(facingGeom, si.wo, si.shading_normal);
            si.diffuse_faces_away = true;
        }
    }

    // Sample emission texture
    float3 emissionColor = float3(material.emission);
    if ((materialFeatures & MATERIAL_TEX_EMISSION) != 0u && !is_null_texture(material.emissionTexture))
    {
        // Emission is an integrand shared by NEE and BSDF-hit strategies. A
        // strategy-dependent ray-cone mip would make the two evaluate different
        // radiance for the same path-space event, so both use level zero.
        const float4 emTex = material.emissionTexture.sample(texSampler, tuv);
        emissionColor *= emTex.rgb;
    }
    si.emission = emissionColor * material.emission_strength;

    // Fill remaining material parameters for bsdf_init. Zero-initialised because
    // that is the default bsdf_init is written against, and a field added to the
    // struct but forgotten here would otherwise be read as stack garbage.
    MaterialParams matParams = {};
    matParams.roughness = resolvedRoughness;
    matParams.metallic = resolvedMetallic;
    matParams.ior = material.ior;
    matParams.specular = material.specular;
    matParams.specular_color =
        (materialFeatures & MATERIAL_FEATURE_SPECULAR_COLOR) != 0u ? float3(material.specular_color) : float3(1.0f);
    if ((materialFeatures & MATERIAL_FEATURE_TRANSMISSION) != 0u)
    {
        matParams.transmission = material.transmission;
    }
    if ((materialFeatures & MATERIAL_FEATURE_CLEARCOAT) != 0u)
    {
        matParams.clearcoat = material.clearcoat;
        matParams.clearcoat_roughness = resolvedClearcoatRoughness;
        matParams.clearcoat_ior = material.clearcoat_ior;
    }
    if ((materialFeatures & MATERIAL_FEATURE_ANISOTROPY) != 0u)
    {
        matParams.anisotropy = material.anisotropy;
    }
    if ((materialFeatures & MATERIAL_FEATURE_DIFFUSE_TRANSMISSION) != 0u)
    {
        matParams.diffuse_transmission = material.diffuse_transmission;
        matParams.diffuse_transmission_color = float3(material.diffuse_transmission_color);
    }
    if ((materialFeatures & MATERIAL_FEATURE_SHEEN) != 0u)
    {
        matParams.sheen = material.sheen;
        matParams.sheen_roughness = material.sheen_roughness;
        matParams.sheen_color = float3(material.sheen_color);
    }
    if ((materialFeatures & MATERIAL_FEATURE_SUBSURFACE) != 0u)
    {
        // si.subsurface is what gates the random walk in shade(); leaving it
        // out made every subsurface material plain diffuse transmission.
        matParams.subsurface = material.subsurface;
        matParams.subsurface_radius = float3(material.subsurface_radius);
        matParams.subsurface_anisotropy = material.subsurface_anisotropy;
        matParams.subsurface_reference = float3(material.subsurface_reference);
    }
    if ((materialFeatures & MATERIAL_FEATURE_IRIDESCENCE) != 0u)
    {
        matParams.iridescence = material.iridescence;
        matParams.iridescence_ior = material.iridescence_ior;
        matParams.iridescence_thickness = material.iridescence_thickness;
    }
    matParams.material_type = material.material_type;
    matParams.thin_walled = material.thin_walled;
    matParams.dielectric_priority = material.dielectric_priority;

    // bsdf_init (Metal overload) clamps and finalizes derived values
    bsdf_init(si, matParams);
    // Restore texture-resolved values that bsdf_init may have overwritten
    si.roughness = max(resolvedRoughness, 0.0001f);
    si.metallic = saturate(resolvedMetallic);
}

struct LightConnection
{
    float3 radiance; // unoccluded Li times the cosine at the surface
    float3 toLight; // shadow ray direction
    float3 origin; // shadow ray origin
    float3 visibilityTarget; // offset near-side endpoint for traversed mesh emitters
    float pdf;
    float tMax;
    bool needsRay; // false when the connection is degenerate and contributes nothing
    bool hasVisibilityTarget;
    bool isDelta;
    RestirLightSample sample;
};

static LightConnection makeEmptyConnection()
{
    LightConnection c;
    c.radiance = float3(0.0f);
    c.toLight = float3(0.0f);
    c.origin = float3(0.0f);
    c.visibilityTarget = float3(0.0f);
    c.pdf = 0.0f;
    c.tMax = 0.0f;
    c.needsRay = false;
    c.hasVisibilityTarget = false;
    c.isDelta = false;
    c.sample = {};
    return c;
}

static RestirLightSample restirDirectionSample(uint32_t type, uint32_t lightId, float3 direction)
{
    return { restirSampleKey(type, lightId), as_type<uint32_t>(direction.x), as_type<uint32_t>(direction.y),
             as_type<uint32_t>(direction.z) };
}

static RestirLightSample restirAnalyticSample(uint32_t lightId, uint32_t lightType, float2 uv, float3 direction)
{
    return lightType == LIGHT_TYPE_DISTANT ? restirDirectionSample(RESTIR_SAMPLE_ANALYTIC, lightId, direction) :
                                             RestirLightSample{ restirSampleKey(RESTIR_SAMPLE_ANALYTIC, lightId),
                                                                as_type<uint32_t>(uv.x), as_type<uint32_t>(uv.y), 0u };
}

static RestirLightSample restirMeshSample(uint32_t lightId, uint32_t primitiveId, float2 uv)
{
    return { restirSampleKey(RESTIR_SAMPLE_EMISSIVE_TRIANGLE, lightId), primitiveId, as_type<uint32_t>(uv.x),
             as_type<uint32_t>(uv.y) };
}

static float3 restirSampleData3(thread const RestirLightSample& sample)
{
    return float3(as_type<float>(sample.data0), as_type<float>(sample.data1), as_type<float>(sample.data2));
}

static float3 restirDistantTangent(float3 axis)
{
    return abs(axis.x) > abs(axis.y) ? normalize(float3(-axis.z, 0.0f, axis.x)) :
                                       normalize(float3(0.0f, axis.z, -axis.y));
}

static float3 remapRestirDistantDirection(float3 direction,
                                          device const UniformLight& source,
                                          device const UniformLight& destination)
{
    const float3 sourceAxis = -float3(source.normal);
    const float3 destinationAxis = -float3(destination.normal);
    if (all(sourceAxis == destinationAxis) && source.halfAngle == destination.halfAngle)
        return direction;
    const float3 sourceTangent = restirDistantTangent(sourceAxis);
    const float3 sourceBitangent = cross(sourceAxis, sourceTangent);
    const float3 destinationTangent = restirDistantTangent(destinationAxis);
    const float3 destinationBitangent = cross(destinationAxis, destinationTangent);
    const float2 azimuth = float2(dot(direction, sourceTangent), dot(direction, sourceBitangent));
    const float azimuthLength = length(azimuth);
    const float2 unitAzimuth = azimuthLength > 0.0f ? azimuth / azimuthLength : float2(1.0f, 0.0f);
    const float sourceHalfSin = sin(0.5f * distantLightHalfAngle(source.halfAngle));
    const float q = sourceHalfSin > 0.0f ?
                        saturate((1.0f - dot(direction, sourceAxis)) / (2.0f * sourceHalfSin * sourceHalfSin)) :
                        0.0f;
    const float destinationHalfSin = sin(0.5f * distantLightHalfAngle(destination.halfAngle));
    const float cosTheta = 1.0f - 2.0f * q * destinationHalfSin * destinationHalfSin;
    const float sinTheta =
        2.0f * destinationHalfSin * sqrt(max(q * (1.0f - q * destinationHalfSin * destinationHalfSin), 0.0f));
    return normalize((unitAzimuth.x * destinationTangent + unitAzimuth.y * destinationBitangent) * sinTheta +
                     destinationAxis * cosTheta);
}

static uint32_t remapRestirSample(constant Uniforms& uniforms,
                                  device UniformLight* currentLights,
                                  thread const RestirLightSample& source,
                                  bool previousToCurrent,
                                  thread RestirLightSample& mapped)
{
    mapped = source;
    const uint32_t type = restirSampleType(source);
    const uint32_t id = restirSampleLightId(source);
    if (type == RESTIR_SAMPLE_ANALYTIC)
    {
        const uint32_t fromCount = previousToCurrent ? uniforms.previousNumLights : uniforms.numLights;
        const uint32_t toCount = previousToCurrent ? uniforms.numLights : uniforms.previousNumLights;
        if (id >= fromCount)
            return RESTIR_LIGHT_UNMAPPED;
        device const uint32_t* map =
            previousToCurrent ? uniforms.previousToCurrentLight : uniforms.currentToPreviousLight;
        const uint32_t mappedId = map ? map[id] : id;
        if (mappedId == RESTIR_LIGHT_TYPE_CHANGED)
            return RESTIR_LIGHT_TYPE_CHANGED;
        if (mappedId >= toCount)
            return RESTIR_LIGHT_UNMAPPED;
        mapped.typeAndLightId = restirSampleKey(type, mappedId);
        device UniformLight* previousLights = (device UniformLight*)uniforms.previousLights;
        device UniformLight* sourceLights = previousToCurrent ? previousLights : currentLights;
        device UniformLight* destinationLights = previousToCurrent ? currentLights : previousLights;
        device const UniformLight& sourceLight = sourceLights[id];
        device const UniformLight& destinationLight = destinationLights[mappedId];
        if (sourceLight.type == LIGHT_TYPE_DISTANT)
        {
            const float3 direction =
                remapRestirDistantDirection(restirSampleData3(source), sourceLight, destinationLight);
            mapped.data0 = as_type<uint32_t>(direction.x);
            mapped.data1 = as_type<uint32_t>(direction.y);
            mapped.data2 = as_type<uint32_t>(direction.z);
        }
        return mappedId;
    }
    if (type == RESTIR_SAMPLE_ENVIRONMENT)
        return uniforms.restirEnvironmentHistoryValid != 0u ? 0u : RESTIR_LIGHT_UNMAPPED;
    if (type == RESTIR_SAMPLE_EMISSIVE_TRIANGLE)
    {
        const uint32_t fromCount = previousToCurrent ? uniforms.previousNumEmissiveMeshes : uniforms.numEmissiveMeshes;
        const uint32_t toCount = previousToCurrent ? uniforms.numEmissiveMeshes : uniforms.previousNumEmissiveMeshes;
        return uniforms.restirMeshHistoryValid != 0u && id < fromCount && id < toCount ? id : RESTIR_LIGHT_UNMAPPED;
    }
    return RESTIR_LIGHT_UNMAPPED;
}

// Hair TT/TRT lobes transmit across the strand, so fibre lighting must not reject the opposite shading hemisphere.
static inline bool scattersThroughFibre(thread SurfaceInteraction& si)
{
    return si.material_type == MATERIAL_TYPE_HAIR;
}

static inline bool lightReachesShadingPoint(thread SurfaceInteraction& si, float3 L)
{
    return neeSurfaceSupportsDirection(scattersThroughFibre(si), si.front_face, dot(si.shading_normal, si.wo),
                                       si.transmission, si.diffuse_transmission, dot(si.shading_normal, L));
}

// The factor that cancels the one hair_chiang_eval() divides by. It has to be the
// same |n.wi| and never a clamp to zero, or the two do not cancel and the fibre's
// far side comes back either black or blown out.
static inline float shadingCosine(thread SurfaceInteraction& si, float3 L)
{
    return neeSurfaceCosine(scattersThroughFibre(si), si.front_face, dot(si.shading_normal, si.wo), si.transmission,
                            si.diffuse_transmission, dot(si.shading_normal, L));
}

__attribute__((always_inline)) int __float_as_int(float x)
{
    return as_type<int>(x);
}
__attribute__((always_inline)) float __int_as_float(int x)
{
    return as_type<float>(x);
}

static float3 offset_ray(const float3 p, const float3 n)
{
    const float origin = 1.0f / 32.0f;
    const float float_scale = 1.0f / 65536.0f;
    const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = float3(__int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                        __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                        __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                  abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                  abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

static float3 emittedLightRadiance(device const UniformLight& light,
                                   float3 directionFromLight,
                                   float distance,
                                   device const IesGpuBufferHeader* iesBuffer,
                                   int lightType)
{
    float3 radiance = float3(light.color);
    if (lightIsPunctual(lightType))
    {
        const float safeDistance = max(distance, 1e-4f);
        const bool soft = punctualLightIsSoft(light.points[0].x);
        radiance *= rangeWindow(light, safeDistance) *
                    (soft ? sphereRadianceFromIntensity(light.points[0].x) : (1.0f / (safeDistance * safeDistance)));
        const bool hasIes = light.points[0].y >= 0.0f;
        if (lightType == LIGHT_TYPE_PROJECTOR)
        {
            radiance *= projectorEmission(light, directionFromLight);
        }
        else if (hasIes)
        {
            radiance *= sampleIesCandela(iesBuffer, light, directionFromLight);
        }
        else if (lightType == LIGHT_TYPE_SPOT)
        {
            radiance *= spotAttenuation(light, directionFromLight);
        }
    }
    return radiance * areaFalloff(light, distance, lightType);
}

static float3 emittedLightRadiance(device const UniformLight& light,
                                   float3 directionFromLight,
                                   float distance,
                                   device const IesGpuBufferHeader* iesBuffer)
{
    return emittedLightRadiance(light, directionFromLight, distance, iesBuffer, light.type);
}

LightConnection connectLightSample(constant Uniforms& uniforms,
                                   device const UniformLight& light,
                                   uint32_t lightId,
                                   float2 uv,
                                   uint2 retryWords,
                                   RestirLightSample storedSample,
                                   bool reconnecting,
                                   thread SurfaceInteraction& si,
                                   bool volumeEvent,
                                   device const IesGpuBufferHeader* iesBuffer,
                                   float localSelectionPdf,
                                   float analyticSelectionPdf,
                                   float lightSelectionPdf)
{
    const int lightType = SPEC_ALL_ANALYTIC_LIGHTS_RECT ? LIGHT_TYPE_RECT : light.type;
    LightSampleData lightSampleData = {};
    switch (lightType)
    {
    case LIGHT_TYPE_RECT:
        if (SPEC_UNIFORM_RECT_LIGHT_SAMPLING)
        {
            lightSampleData = SampleRectLightUniform(light, uv, si.position);
        }
        else
        {
            lightSampleData = SampleRectLight(light, uv, si.position);
        }
        break;
    case LIGHT_TYPE_DISC:
        lightSampleData = SampleDiscLight(light, uv, si.position);
        break;
    case LIGHT_TYPE_SPHERE:
        lightSampleData = SampleSphereLight(light, uv, si.position);
        break;
    case LIGHT_TYPE_DISTANT: {
        lightSampleData = SampleDistantLight(light, uv, retryWords, si.position);
        if (reconnecting)
        {
            lightSampleData.L = restirSampleData3(storedSample);
            lightSampleData.pointOnLight = lightSampleData.L;
        }
        break;
    }
    case LIGHT_TYPE_DOME:
        lightSampleData = SampleDomeLight(light, uv, si.position);
        break;
    case LIGHT_TYPE_POINT:
    case LIGHT_TYPE_SPOT:
    case LIGHT_TYPE_PROJECTOR:
        // One sampler for all three: a lamp at a point, or the sphere a soft
        // radius turns it into. What differs is the angular profile applied
        // below, not where the sample is taken.
        lightSampleData = SamplePointLight(light, uv, si.position);
        break;
    }

    LightConnection c = makeEmptyConnection();
    c.sample = reconnecting ? storedSample : restirAnalyticSample(lightId, lightType, uv, lightSampleData.L);
    c.toLight = lightSampleData.L;
    const float shapeParameter = lightIsPunctual(lightType) ? light.points[0].x : light.halfAngle;
    c.isDelta = lightIsDeltaForMis(lightType, shapeParameter);

    const float3 Li = emittedLightRadiance(light, -lightSampleData.L, lightSampleData.distToLight, iesBuffer, lightType);

    const bool lit = lightReachesShadingPoint(si, lightSampleData.L);
    const bool facesLight = lightConnectionFacesVertex(lightType, -dot(lightSampleData.L, lightSampleData.normal),
                                                       lightIsPunctual(lightType) ? light.points[0].x : 0.0f);
    const bool facing = (volumeEvent || lit) && facesLight && emitsLight(Li);
    if (facing)
    {
        // The cosine belongs here because bsdf_eval() returns f alone, unlike
        // bsdf_sample()'s bsdf_over_pdf which already carries it. See the note on
        // both result structs in bsdf_types.h.
        c.radiance = volumeEvent ? Li : Li * shadingCosine(si, lightSampleData.L);
        c.origin = volumeEvent ? si.position :
                                 offset_ray(si.position, orientedFaceNormal(si.geometry_normal, lightSampleData.L));
        const LightPdfQuery query = buildLightPdfQuery(light, lightSampleData, lightType);
        c.pdf = marginalLightSolidAnglePdf(query, localSelectionPdf, analyticSelectionPdf, lightSelectionPdf);
        c.tMax = lightSampleData.distToLight;
        c.needsRay = true;
        if (lightUsesAnalyticSurfaceIntersection(lightType, lightIsPunctual(lightType) ? light.points[0].x : 0.0f))
        {
            c.visibilityTarget =
                offset_ray(lightSampleData.pointOnLight, orientedFaceNormal(lightSampleData.normal, -lightSampleData.L));
            c.hasVisibilityTarget = true;
        }
    }
    return c;
}

LightConnection connectLight(constant Uniforms& uniforms,
                             thread SamplerState& samplerRnd,
                             device const UniformLight& light,
                             uint32_t lightId,
                             thread SurfaceInteraction& si,
                             bool volumeEvent,
                             device const IesGpuBufferHeader* iesBuffer,
                             float localSelectionPdf,
                             float analyticSelectionPdf,
                             float lightSelectionPdf)
{
    const RandomSample4 lightRandom =
        random4<SampleDimension::eLightPointX, SampleDimension::eLightPointY, SampleDimension::eLightRetryU,
                SampleDimension::eLightRetryV>(samplerRnd, uniforms.samplerType);
    const float2 uv = float2(lightOpenUnitInterval(lightRandom.value.x), lightOpenUnitInterval(lightRandom.value.y));
    const uint2 retryWords = lightRandom.bits.zw;
    return connectLightSample(uniforms, light, lightId, uv, retryWords, {}, false, si, volumeEvent, iesBuffer,
                              localSelectionPdf, analyticSelectionPdf, lightSelectionPdf);
}

LightConnection connectEnvDirectionAtUv(constant Uniforms& uniforms,
                                        float3 dir,
                                        float2 uv,
                                        float envPdf,
                                        thread SurfaceInteraction& si,
                                        texture2d<float> envMapTexture,
                                        bool volumeEvent)
{
    LightConnection c = makeEmptyConnection();
    c.sample = restirDirectionSample(RESTIR_SAMPLE_ENVIRONMENT, 0u, dir);
    c.toLight = dir;
    c.pdf = envPdf;

    if (envPdf <= 0.0f || (!volumeEvent && !lightReachesShadingPoint(si, dir)))
    {
        return c;
    }

    constexpr sampler envSampler(
        mag_filter::linear, min_filter::linear, s_address::repeat, t_address::clamp_to_edge, coord::normalized);
    const float4 envSample = envMapTexture.sample(envSampler, uv);
    const float3 Li = envSample.xyz * uniforms.envMapIntensity * uniforms.envMapColorTint.xyz;

    c.radiance = volumeEvent ? Li : Li * shadingCosine(si, dir);
    c.origin = offset_ray(si.position, orientedFaceNormal(si.geometry_normal, dir));
    c.tMax = 1e16f;
    c.needsRay = true;
    return c;
}

LightConnection connectEnvDirection(constant Uniforms& uniforms,
                                    float3 dir,
                                    float envPdf,
                                    thread SurfaceInteraction& si,
                                    texture2d<float> envMapTexture,
                                    bool volumeEvent)
{
    return connectEnvDirectionAtUv(
        uniforms, dir, metalDirToEnvUv(dir, uniforms.envMapRotationSinCos), envPdf, si, envMapTexture, volumeEvent);
}

LightConnection connectEnvLight(constant Uniforms& uniforms,
                                thread SamplerState& samplerRnd,
                                thread SurfaceInteraction& si,
                                device const uint2* envAliasTable,
                                texture2d<float> envMapTexture,
                                bool volumeEvent)
{
    const RandomSample4 envRandom =
        random4<SampleDimension::eLightBucket, SampleDimension::eLightAlias, SampleDimension::eLightPointX,
                SampleDimension::eLightPointY>(samplerRnd, uniforms.samplerType);
    const uint2 aliasWords = envRandom.bits.xy;
    const float2 jitter = envRandom.value.zw;

    float envPdf = 0.0f;
    float2 envUv;
    float3 dir = sampleEnvMap(aliasWords, jitter, envAliasTable, uniforms.envPdfTable, uniforms.envRowCosBounds,
                              uniforms.envMapWidth, uniforms.envMapHeight, uniforms.envMapRotationSinCos, envUv, envPdf);

    // The alias sample already owns the exact texture coordinate. Reversing
    // its direction through atan2/asin only to recover that coordinate adds
    // SFU latency and can move a boundary value into the adjacent texel.
    return connectEnvDirectionAtUv(uniforms, dir, envUv, envPdf, si, envMapTexture, volumeEvent);
}

struct EmissiveTriangleGeometry
{
    float3 p0;
    float3 p1;
    float3 p2;
    float2 uv0;
    float2 uv1;
    float2 uv2;
};

template <typename InstancePointer>
static float4x4 emissiveObjectToWorld(InstancePointer instances, uint32_t instanceId)
{
    const auto inst = instances[instanceId];
    return float4x4(
        float4(float3(inst.transformationMatrix[0]), 0.0f), float4(float3(inst.transformationMatrix[1]), 0.0f),
        float4(float3(inst.transformationMatrix[2]), 0.0f), float4(float3(inst.transformationMatrix[3]), 1.0f));
}

static uint32_t geometryTransformIndex(constant Uniforms& uniforms,
                                       uint32_t instanceIndex,
                                       uint32_t geometryEntryIndex,
                                       GeometryEntry entry)
{
    return (entry.flags & GEOM_FLAG_BAKED_TRANSFORM) != 0u ? uniforms.geometryTransformBase + geometryEntryIndex :
                                                             instanceIndex;
}

template <typename InstancePointer>
static float4x4 geometryObjectToWorld(constant Uniforms& uniforms,
                                      InstancePointer instances,
                                      uint32_t instanceIndex,
                                      uint32_t geometryEntryIndex,
                                      GeometryEntry entry)
{
    return emissiveObjectToWorld(instances, geometryTransformIndex(uniforms, instanceIndex, geometryEntryIndex, entry));
}

template <typename InstancePointer>
static EmissiveTriangleGeometry fetchEmissiveTriangle(constant Uniforms& uniforms,
                                                      InstancePointer instances,
                                                      device const char* vertexBuffer,
                                                      device const char* prevVertexBuffer,
                                                      device const uint32_t* indexBuffer,
                                                      device const EmissiveMeshLight& mesh,
                                                      uint32_t primitiveId,
                                                      float motionTime)
{
    constexpr uint32_t stride = 32u;
    constexpr uint32_t uvOffset = 20u;
    EmissiveTriangleGeometry triangle;
    thread float3* points[3] = { &triangle.p0, &triangle.p1, &triangle.p2 };
    thread float2* uvs[3] = { &triangle.uv0, &triangle.uv1, &triangle.uv2 };
    const float4x4 objectToWorld = emissiveObjectToWorld(instances, mesh.transformIndex);
    for (uint32_t k = 0u; k < 3u; ++k)
    {
        const uint32_t vertexId = indexBuffer[mesh.indexOffset + primitiveId * 3u + k] + mesh.vertexOffset;
        device const char* current = vertexBuffer + size_t(vertexId) * stride;
        float3 objectPoint = float3(*(device const packed_float3*)current);
        if (SPEC_MOTION_BLUR && uniforms.enableMotionBlur && motionTime < 1.0f && prevVertexBuffer)
        {
            device const char* previous = prevVertexBuffer + size_t(vertexId) * stride;
            objectPoint = mix(float3(*(device const packed_float3*)previous), objectPoint, motionTime);
        }
        *points[k] = (objectToWorld * float4(objectPoint, 1.0f)).xyz;
        *uvs[k] = unpackUV(*(device const uint32_t*)(current + uvOffset),
                           *(device const uint32_t*)(current + uvOffset + 4u));
    }
    return triangle;
}

static float3 emissiveMeshRadiance(constant Uniforms& uniforms,
                                   device const Material& material,
                                   uint32_t materialId,
                                   float2 uv)
{
    if ((SPEC_ALL_OPENPBR || (SPEC_OPENPBR && material.material_type == MATERIAL_TYPE_OPENPBR)) &&
        uniforms.openpbrParams != nullptr)
    {
        device const OpenPBRParams& p = uniforms.openpbrParams[materialId];
        float3 emission = float3(p.emission_color.r, p.emission_color.g, p.emission_color.b);
        if ((p.texture_mask & (1u << OPENPBR_TEX_EMISSION_COLOR)) != 0u && uniforms.openpbrTextures != nullptr)
        {
            device const OpenPBRTextures& textures = uniforms.openpbrTextures[materialId];
            if (!is_null_texture(textures.tex[OPENPBR_TEX_EMISSION_COLOR]))
            {
                constexpr sampler emissionSampler(mag_filter::linear, min_filter::linear, address::repeat);
                const float c = cos(p.uv_rotation);
                const float s = sin(p.uv_rotation);
                const float2 scale = float2(p.uv_scale_x, p.uv_scale_y);
                const float2 tuv =
                    float2(uv.x * scale.x * c - uv.y * scale.y * s, uv.x * scale.x * s + uv.y * scale.y * c) +
                    float2(p.uv_offset_x, p.uv_offset_y);
                emission = textures.tex[OPENPBR_TEX_EMISSION_COLOR].sample(emissionSampler, tuv).rgb;
            }
        }
        return emission * p.emission_luminance;
    }

    float3 emission = float3(material.emission) * material.emission_strength;
    if ((material.features & MATERIAL_TEX_EMISSION) != 0u && !is_null_texture(material.emissionTexture))
    {
        constexpr sampler emissionSampler(mag_filter::linear, min_filter::linear, address::repeat);
        emission *= material.emissionTexture.sample(emissionSampler, applyTextureTransform(uv, material)).rgb;
    }
    return emission;
}

static uint32_t sampleEmissiveMesh(constant Uniforms& uniforms, thread SamplerState& sampler, uint32_t bucketWord)
{
    const uint32_t bucket = lightAliasBucket(uniforms.numEmissiveMeshes, bucketWord);
    device const EmissiveMeshLight& entry = uniforms.emissiveMeshes[bucket];
    const uint32_t coinWord = randomBits<SampleDimension::eLightAlias>(sampler, uniforms.samplerType);
    return lightAliasSelect(uniforms.numEmissiveMeshes, bucket, coinWord, entry.aliasThreshold, entry.alias);
}

static uint32_t sampleEmissiveTriangleIndex(constant Uniforms& uniforms,
                                            thread SamplerState& sampler,
                                            device const EmissiveMeshLight& mesh)
{
    const uint32_t bucketWord = randomBits<SampleDimension::eTriangleBucket>(sampler, uniforms.samplerType);
    const uint32_t bucket = lightAliasBucket(mesh.triangleCount, bucketWord);
    device const EmissiveTriangleLight& entry = uniforms.emissiveTriangles[mesh.triangleOffset + bucket];
    const uint32_t coinWord = randomBits<SampleDimension::eTriangleAlias>(sampler, uniforms.samplerType);
    return lightAliasSelect(mesh.triangleCount, bucket, coinWord, entry.aliasThreshold, entry.alias);
}

static int findEmissiveMesh(constant Uniforms& uniforms, uint32_t instanceId, uint32_t geometryId)
{
    uint32_t first = 0u;
    uint32_t count = uniforms.numEmissiveMeshes;
    while (count > 0u)
    {
        const uint32_t step = count / 2u;
        const uint32_t middle = first + step;
        if (emissiveMeshKeyLess(uniforms.emissiveMeshes[middle].instanceId, uniforms.emissiveMeshes[middle].geometryId,
                                instanceId, geometryId))
        {
            first = middle + 1u;
            count -= step + 1u;
        }
        else
        {
            count = step;
        }
    }
    if (first < uniforms.numEmissiveMeshes)
    {
        device const EmissiveMeshLight& light = uniforms.emissiveMeshes[first];
        if (light.instanceId == instanceId && light.geometryId == geometryId)
        {
            return int(first);
        }
    }
    return -1;
}

template <typename InstancePointer>
static LightConnection connectEmissiveMeshSample(constant Uniforms& uniforms,
                                                 InstancePointer instances,
                                                 device const char* vertexBuffer,
                                                 device const char* prevVertexBuffer,
                                                 device const uint32_t* indexBuffer,
                                                 device const Material* materials,
                                                 thread SurfaceInteraction& si,
                                                 uint32_t meshId,
                                                 uint32_t primitiveId,
                                                 float2 randomSample,
                                                 float motionTime,
                                                 bool volumeEvent,
                                                 float localSelectionPdf,
                                                 float meshClassPdf,
                                                 uint32_t numEmissiveMeshes)
{
    LightConnection connection = makeEmptyConnection();
    if (meshId >= numEmissiveMeshes)
    {
        return connection;
    }
    device const EmissiveMeshLight& mesh = uniforms.emissiveMeshes[meshId];
    if (primitiveId >= mesh.triangleCount)
    {
        return connection;
    }
    device const EmissiveTriangleLight& triangleEntry = uniforms.emissiveTriangles[mesh.triangleOffset + primitiveId];
    if (!(mesh.selectionPdf > 0.0f) || !(triangleEntry.selectionPdf > 0.0f))
    {
        return connection;
    }
    const EmissiveTriangleGeometry triangle = fetchEmissiveTriangle(
        uniforms, instances, vertexBuffer, prevVertexBuffer, indexBuffer, mesh, primitiveId, motionTime);
    const EmissiveTriangleSample sample = sampleEmissiveTriangle(
        triangle.p0, triangle.p1, triangle.p2, triangle.uv0, triangle.uv1, triangle.uv2, randomSample.x, randomSample.y);
    if (!sample.valid)
    {
        return connection;
    }
    const float3 offset = sample.point - si.position;
    float distance;
    const float3 direction = finiteDirectionAndDistance(offset, distance);
    if (!(distance > 1e-5f))
    {
        return connection;
    }
    device const Material& material = materials[mesh.materialId];
    const float3 emission =
        emissiveMeshRadiance(uniforms, material, mesh.materialId, sample.uv) * resolveOpacity(material, sample.uv);
    if (!emitsLight(emission) || (!volumeEvent && !lightReachesShadingPoint(si, direction)))
    {
        return connection;
    }
    const float selectedMeshPdf = emissiveMeshMarginalSolidAnglePdf(localSelectionPdf, meshClassPdf, mesh.selectionPdf,
                                                                    triangleEntry.selectionPdf, sample.areaPdf,
                                                                    si.position, sample.point, sample.normal);
    if (!(selectedMeshPdf > 0.0f))
    {
        return connection;
    }

    connection.radiance = volumeEvent ? emission : emission * shadingCosine(si, direction);
    connection.toLight = direction;
    connection.origin =
        volumeEvent ? si.position : offset_ray(si.position, orientedFaceNormal(si.geometry_normal, direction));
    connection.visibilityTarget = offset_ray(sample.point, orientedFaceNormal(sample.normal, -direction));
    connection.pdf = selectedMeshPdf;
    connection.tMax = distance;
    connection.needsRay = true;
    connection.hasVisibilityTarget = true;
    connection.isDelta = false;
    connection.sample = restirMeshSample(meshId, primitiveId, randomSample);
    return connection;
}

static LightConnection connectEmissiveMesh(constant Uniforms& uniforms,
                                           constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                           device const char* vertexBuffer,
                                           device const char* prevVertexBuffer,
                                           device const uint32_t* indexBuffer,
                                           device const Material* materials,
                                           thread SamplerState& sampler,
                                           thread SurfaceInteraction& si,
                                           uint32_t meshBucketWord,
                                           float motionTime,
                                           bool volumeEvent,
                                           float localSelectionPdf,
                                           float meshClassPdf)
{
    const uint32_t meshId = sampleEmissiveMesh(uniforms, sampler, meshBucketWord);
    if (meshId >= uniforms.numEmissiveMeshes)
    {
        return makeEmptyConnection();
    }
    const uint32_t primitiveId = sampleEmissiveTriangleIndex(uniforms, sampler, uniforms.emissiveMeshes[meshId]);
    const float2 randomSample =
        random2<SampleDimension::eLightPointX, SampleDimension::eLightPointY>(sampler, uniforms.samplerType).value;
    return connectEmissiveMeshSample(uniforms, instances, vertexBuffer, prevVertexBuffer, indexBuffer, materials, si,
                                     meshId, primitiveId, randomSample, motionTime, volumeEvent, localSelectionPdf,
                                     meshClassPdf, uniforms.numEmissiveMeshes);
}

static EmissiveVisibilitySegment lightVisibilitySegment(thread const LightConnection& connection, float3 shadowOrigin)
{
    if (connection.hasVisibilityTarget)
    {
        return emissiveVisibilitySegment(shadowOrigin, connection.visibilityTarget);
    }
    EmissiveVisibilitySegment segment;
    segment.direction = connection.toLight;
    segment.maxDistance = connection.tMax;
    segment.valid = connection.needsRay && connection.tMax > 0.0f;
    return segment;
}

// Choose a strategy and build the connection. The caller decides when to test
// visibility.
uint32_t sampleAnalyticLight(const uint32_t numLights,
                             device UniformLight* lights,
                             const uint32_t bucketWord,
                             const uint32_t coinWord)
{
    const uint32_t bucket = lightAliasBucket(numLights, bucketWord);
    device const UniformLight& entry = lights[bucket];
    return lightAliasSelect(numLights, bucket, coinWord, entry.selectionAliasThreshold, entry.selectionAlias);
}

float analyticLightSelectionPdf(device const UniformLight& light)
{
    return light.color.w;
}

LightConnection connectToLight(constant Uniforms& uniforms,
                               const uint32_t numLights,
                               device UniformLight* lights,
                               constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                               device const Material* materials,
                               device const char* vertexBuffer,
                               device const char* prevVertexBuffer,
                               device const uint32_t* indexBuffer,
                               float motionTime,
                               thread SamplerState& samplerRnd,
                               thread SurfaceInteraction& si,
                               device const uint2* envAliasTable,
                               texture2d<float> envMapTexture,
                               device const IesGpuBufferHeader* iesBuffer,
                               bool volumeEvent = false)
{
    const bool hasAnalytic = SPEC_LIGHTS && numLights > 0u;
    const bool hasMesh = SPEC_LIGHTS && SPEC_EMISSIVE_MESH_LIGHTS && uniforms.numEmissiveMeshes > 0u;
    const bool hasLocal = hasAnalytic || hasMesh;
    // Keep categorical choices per path. Sharing one word across a SIMD-group
    // preserves the expectation but turns finite-spp error into visible bands.
    const uint32_t emitterWord = randomBits<SampleDimension::eLightId>(samplerRnd, uniforms.samplerType);
    float localSelectionPdf = 1.0f;
    if (SPEC_ENV_MAP && uniforms.hasEnvMap)
    {
        const float envSelectionPdf = uniforms.envMapColorTint.w;
        localSelectionPdf = 1.0f - envSelectionPdf;

        if (!hasLocal || discreteBernoulli(emitterWord, envSelectionPdf))
        {
            LightConnection c = connectEnvLight(uniforms, samplerRnd, si, envAliasTable, envMapTexture, volumeEvent);
            c.pdf *= hasLocal ? envSelectionPdf : 1.0f;
            return c;
        }
    }

    if (!hasLocal)
    {
        return makeEmptyConnection();
    }

    const float meshSelectionPdf = uniforms.meshLightSelectionPdf;
    bool chooseMesh = hasMesh && !hasAnalytic;
    if (hasMesh && hasAnalytic)
    {
        const uint32_t classWord = randomBits<SampleDimension::eLightClass>(samplerRnd, uniforms.samplerType);
        chooseMesh = discreteBernoulli(classWord, meshSelectionPdf);
    }
    if (chooseMesh)
    {
        const uint32_t meshWord = randomBits<SampleDimension::eLightBucket>(samplerRnd, uniforms.samplerType);
        return connectEmissiveMesh(uniforms, instances, vertexBuffer, prevVertexBuffer, indexBuffer, materials,
                                   samplerRnd, si, meshWord, motionTime, volumeEvent, localSelectionPdf,
                                   hasAnalytic ? meshSelectionPdf : 1.0f);
    }

    const float analyticSelectionPdf = hasMesh ? 1.0f - meshSelectionPdf : 1.0f;
    if (!(analyticSelectionPdf > 0.0f))
    {
        return makeEmptyConnection();
    }
    const uint32_t analyticWord = randomBits<SampleDimension::eLightBucket>(samplerRnd, uniforms.samplerType);
    const uint32_t aliasWord = randomBits<SampleDimension::eLightAlias>(samplerRnd, uniforms.samplerType);
    const uint32_t lightId = sampleAnalyticLight(numLights, lights, analyticWord, aliasWord);
    if (lightId >= numLights)
    {
        return makeEmptyConnection();
    }
    return connectLight(uniforms, samplerRnd, lights[lightId], lightId, si, volumeEvent, iesBuffer, localSelectionPdf,
                        analyticSelectionPdf, analyticLightSelectionPdf(lights[lightId]));
}

template <typename InstancePointer>
LightConnection reconnectRestirSampleContext(constant Uniforms& uniforms,
                                             device UniformLight* lights,
                                             uint32_t numLights,
                                             uint32_t numEmissiveMeshes,
                                             float meshLightSelectionPdf,
                                             bool hasEnvMap,
                                             float envSelectionPdf,
                                             InstancePointer instances,
                                             device const Material* materials,
                                             device const char* vertexBuffer,
                                             device const char* prevVertexBuffer,
                                             device const uint32_t* indexBuffer,
                                             float motionTime,
                                             thread SurfaceInteraction& si,
                                             device const uint2* envAliasTable,
                                             texture2d<float> envMapTexture,
                                             device const IesGpuBufferHeader* iesBuffer,
                                             thread const RestirLightSample& sample)
{
    const bool hasAnalytic = SPEC_LIGHTS && numLights > 0u;
    const bool hasMesh = SPEC_LIGHTS && SPEC_EMISSIVE_MESH_LIGHTS && numEmissiveMeshes > 0u;
    const bool hasLocal = hasAnalytic || hasMesh;
    const float localSelectionPdf = SPEC_ENV_MAP && hasEnvMap && hasLocal ? 1.0f - envSelectionPdf : 1.0f;
    const uint32_t sampleType = restirSampleType(sample);
    const uint32_t lightId = restirSampleLightId(sample);
    if (sampleType == RESTIR_SAMPLE_ENVIRONMENT && SPEC_ENV_MAP && hasEnvMap)
    {
        const float3 direction = restirSampleData3(sample);
        const float envPdf = envMapPdf(direction, uniforms.envPdfTable, uniforms.envMapWidth, uniforms.envMapHeight,
                                       uniforms.envMapRotationSinCos) *
                             (hasLocal ? envSelectionPdf : 1.0f);
        return connectEnvDirection(uniforms, direction, envPdf, si, envMapTexture, false);
    }
    if (sampleType == RESTIR_SAMPLE_ANALYTIC && hasAnalytic && lightId < numLights)
    {
        const float analyticClassPdf = hasMesh ? 1.0f - meshLightSelectionPdf : 1.0f;
        const bool distant = lights[lightId].type == LIGHT_TYPE_DISTANT;
        const float2 uv = distant ? float2(0.0f) : float2(as_type<float>(sample.data0), as_type<float>(sample.data1));
        return connectLightSample(uniforms, lights[lightId], lightId, uv, uint2(0u), sample, true, si, false, iesBuffer,
                                  localSelectionPdf, analyticClassPdf, analyticLightSelectionPdf(lights[lightId]));
    }
    if (sampleType == RESTIR_SAMPLE_EMISSIVE_TRIANGLE && hasMesh)
    {
        const float meshClassPdf = hasAnalytic ? meshLightSelectionPdf : 1.0f;
        const float2 uv = float2(as_type<float>(sample.data1), as_type<float>(sample.data2));
        return connectEmissiveMeshSample(uniforms, instances, vertexBuffer, prevVertexBuffer, indexBuffer, materials,
                                         si, lightId, sample.data0, uv, motionTime, false, localSelectionPdf,
                                         meshClassPdf, numEmissiveMeshes);
    }
    return makeEmptyConnection();
}

LightConnection reconnectRestirSample(constant Uniforms& uniforms,
                                      device UniformLight* lights,
                                      constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                      device const Material* materials,
                                      device const char* vertexBuffer,
                                      device const char* prevVertexBuffer,
                                      device const uint32_t* indexBuffer,
                                      float motionTime,
                                      thread SurfaceInteraction& si,
                                      device const uint2* envAliasTable,
                                      texture2d<float> envMapTexture,
                                      device const IesGpuBufferHeader* iesBuffer,
                                      thread const RestirLightSample& sample)
{
    return reconnectRestirSampleContext(uniforms, lights, uniforms.numLights, uniforms.numEmissiveMeshes,
                                        uniforms.meshLightSelectionPdf, uniforms.hasEnvMap != 0u,
                                        uniforms.envMapColorTint.w, instances, materials, vertexBuffer, prevVertexBuffer,
                                        indexBuffer, motionTime, si, envAliasTable, envMapTexture, iesBuffer, sample);
}

static float emissiveMeshHitPdf(constant Uniforms& uniforms,
                                constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                device const char* vertexBuffer,
                                device const char* prevVertexBuffer,
                                device const uint32_t* indexBuffer,
                                uint32_t instanceId,
                                uint32_t geometryId,
                                uint32_t primitiveId,
                                float3 shadingPoint,
                                float3 pointOnLight,
                                float motionTime)
{
    const int meshId = findEmissiveMesh(uniforms, instanceId, geometryId);
    if (meshId < 0)
    {
        return 0.0f;
    }
    device const EmissiveMeshLight& mesh = uniforms.emissiveMeshes[meshId];
    if (primitiveId >= mesh.triangleCount)
    {
        return 0.0f;
    }
    device const EmissiveTriangleLight& triangleEntry = uniforms.emissiveTriangles[mesh.triangleOffset + primitiveId];
    const EmissiveTriangleGeometry triangle = fetchEmissiveTriangle(
        uniforms, instances, vertexBuffer, prevVertexBuffer, indexBuffer, mesh, primitiveId, motionTime);
    const EmissiveTriangleSample geometry = sampleEmissiveTriangle(
        triangle.p0, triangle.p1, triangle.p2, triangle.uv0, triangle.uv1, triangle.uv2, 0.25f, 0.5f);
    if (!geometry.valid)
    {
        return 0.0f;
    }
    const float localSelectionPdf = (SPEC_ENV_MAP && uniforms.hasEnvMap) ? 1.0f - uniforms.envMapColorTint.w : 1.0f;
    const float meshClassPdf = uniforms.numLights > 0u ? uniforms.meshLightSelectionPdf : 1.0f;
    return emissiveMeshMarginalSolidAnglePdf(localSelectionPdf, meshClassPdf, mesh.selectionPdf,
                                             triangleEntry.selectionPdf, geometry.areaPdf, shadingPoint, pointOnLight,
                                             geometry.normal);
}
