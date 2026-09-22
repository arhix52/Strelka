#ifndef STRELKA_MATERIAL_OPENPBR_PARAMS_H
#define STRELKA_MATERIAL_OPENPBR_PARAMS_H

#include <strelka/material/material_params.h>

struct OpenPBRColor
{
    float r;
    float g;
    float b;
};

/// The imported MAX_Color_Correction group feeds Blender's Gamma node with
/// 1 / authoredGamma.  Keep that conversion here so CUDA and Metal cannot
/// silently disagree with the graph they are executing.
#if defined(__CUDACC__)
static __forceinline__ __host__ __device__ float max_color_correction_exponent(float authoredGamma)
#else
inline float max_color_correction_exponent(float authoredGamma)
#endif
{
    return authoredGamma != 0.0f ? 1.0f / authoredGamma : 0.0f;
}

// Texture slots. The order is an ABI shared by the loader and both backends;
// append only. Anything a MaterialX document can drive with an <image> and that
// Strelka can consume per-pixel gets a slot; everything else is a constant.
enum OpenPBRTextureSlot : unsigned int
{
    OPENPBR_TEX_BASE_COLOR = 0,
    OPENPBR_TEX_BASE_METALNESS = 1,
    OPENPBR_TEX_SPECULAR_ROUGHNESS = 2,
    OPENPBR_TEX_SPECULAR_COLOR = 3,
    OPENPBR_TEX_SPECULAR_ANISOTROPY = 4,
    OPENPBR_TEX_COAT_WEIGHT = 5,
    OPENPBR_TEX_COAT_ROUGHNESS = 6,
    OPENPBR_TEX_COAT_COLOR = 7,
    OPENPBR_TEX_FUZZ_WEIGHT = 8,
    OPENPBR_TEX_FUZZ_ROUGHNESS = 9,
    OPENPBR_TEX_EMISSION_COLOR = 10,
    OPENPBR_TEX_TRANSMISSION_COLOR = 11,
    OPENPBR_TEX_SUBSURFACE_COLOR = 12,
    OPENPBR_TEX_GEOMETRY_NORMAL = 13,
    OPENPBR_TEX_GEOMETRY_COAT_NORMAL = 14,
    OPENPBR_TEX_GEOMETRY_OPACITY = 15,
    OPENPBR_TEX_SUBSURFACE_WEIGHT = 16,
    OPENPBR_TEX_SUBSURFACE_RADIUS = 17,
    OPENPBR_TEX_FUZZ_COLOR = 18,
    // A four-layer MaterialX texture graph keeps colour and data views separate:
    // the same JPEG is transfer-decoded for base colour and read verbatim for
    // roughness / height. These are source images, not OpenPBR inputs, so the
    // ordinary slot loop deliberately ignores them.
    OPENPBR_TEX_LAYER_COLOR_0 = 19,
    OPENPBR_TEX_LAYER_COLOR_1 = 20,
    OPENPBR_TEX_LAYER_COLOR_2 = 21,
    OPENPBR_TEX_LAYER_COLOR_3 = 22,
    OPENPBR_TEX_LAYER_DATA_0 = 23,
    OPENPBR_TEX_LAYER_DATA_1 = 24,
    OPENPBR_TEX_LAYER_DATA_2 = 25,
    OPENPBR_TEX_LAYER_DATA_3 = 26,
    MAX_OPENPBR_TEXTURES = 27
};

enum : unsigned int
{
    OPENPBR_LAYER_OUTPUT_BASE_COLOR = 1u << 0,
    OPENPBR_LAYER_OUTPUT_SPECULAR_COLOR = 1u << 1,
    OPENPBR_LAYER_OUTPUT_ROUGHNESS = 1u << 2,
    OPENPBR_LAYER_OUTPUT_NORMAL = 1u << 3,
    OPENPBR_LAYER_OUTPUT_SPECULAR_WEIGHT = 1u << 4,
    OPENPBR_LAYER_OUTPUT_OPACITY = 1u << 5,
    OPENPBR_LAYER_OUTPUT_SPECULAR_IOR = 1u << 6,
};

enum OpenPBRLayerBlendMode : unsigned int
{
    OPENPBR_LAYER_BLEND_MULTIPLY = 0,
    OPENPBR_LAYER_BLEND_MIX = 1,
    OPENPBR_LAYER_BLEND_OVERLAY = 2,
    OPENPBR_LAYER_BLEND_SCREEN = 3,
};

enum OpenPBRLayerProcedural : unsigned int
{
    OPENPBR_LAYER_PROCEDURAL_NONE = 0,
    OPENPBR_LAYER_PROCEDURAL_VORONOI_RIDGE = 1,
    OPENPBR_LAYER_PROCEDURAL_NOISE = 2,
};

/// Compact runtime form of the MaterialX strelka_layered_texture node. It is a
/// deliberately narrow Corona bitmap stack: four repeated images, Multiply
/// blend, independent colour/data placement and correction, and the Output
/// operations used by the source materials. Keeping it beside the texture
/// handles avoids another bindless table in both ray-tracing backends.
struct OpenPBRLayeredTextureParams
{
    // Colour and scalar/height branches may read different bitmaps and use
    // different placements. 3ds Max's Corona materials do this routinely
    // (albedo + a separate bump map), so sharing these transforms silently
    // sampled the right files through the wrong UV footprint.
    float uv_scale_x[4];
    float uv_scale_y[4];
    float uv_offset_x[4];
    float uv_offset_y[4];
    float uv_rotation[4];
    float data_uv_scale_x[4];
    float data_uv_scale_y[4];
    float data_uv_offset_x[4];
    float data_uv_offset_y[4];
    float data_uv_rotation[4];
    float color_opacity[4];
    float data_opacity[4];
    unsigned int data_blend_mode[4];

    // Some Corona Composite stacks start from a literal colour and apply the
    // bitmap layers after it.  A white implicit start is not equivalent for a
    // partial Multiply, and was the reason walls and dark metal came out pale.
    float color_base[3];
    unsigned int color_has_base;
    float data_base[3];
    unsigned int data_has_base;

    // Data layers need the same independent correction as colour layers.
    // Applying layer zero's MAX_Color_Correction to the whole height stack
    // turns low-opacity macro maps into dominant bump detail.
    float data_hue[4];
    float data_saturation[4];
    float data_value[4];
    float data_gamma[4];
    float data_brightness[4];
    float data_contrast[4];

    float specular_gain;
    float roughness_gain;
    float roughness_base;
    float roughness_mix;

    float bump_gain;
    float bump_scale;
    float opacity_base;
    float opacity_mix;

    unsigned int opacity_layer;
    unsigned int layer_count;
    unsigned int output_mask;
    // Corona Falloff / Blender Layer Weight at blend 0.5 is
    // 1 - abs(dot(N, V)). Keeping its coefficient here preserves the
    // view-dependent opacity of fabrics without baking it into camera UVs.
    float opacity_facing_mix;

    // Legacy documents written before reflection tint was represented exactly.
    float specular_ior_base;
    float specular_ior_mix;
    float specular_ior_authored;
    unsigned int specular_ior_uses_color;

    // CoronaLegacy keeps Fresnel IOR independent, then multiplies the
    // reflection by lerp(constant colour, map, map amount) * level.
    float specular_color_base[3];
    float specular_color_mix;
    unsigned int specular_color_uses_color;

    // The imported Composite graph may correct every bitmap independently and
    // drive its factor from Facing and another bitmap.  Keeping these as four
    // straight-line instructions is enough for the source scene and avoids a
    // scene-specific bake or a general-purpose shader VM.
    float color_hue[4];
    float color_saturation[4];
    float color_value[4];
    float color_gamma[4];
    float color_brightness[4];
    float color_contrast[4];
    unsigned int color_blend_mode[4];
    float color_factor_base[4];
    float color_factor_data[4];
    float color_factor_facing[4];
    float color_factor_facing_data[4];
    unsigned int color_factor_data_layer[4];

    float color_post_hue;
    float color_post_saturation;
    float color_post_value;
    float color_post_gamma;
    float color_post_brightness;
    float color_post_contrast;

    // Blender procedural heights used by the two remaining imported graphs.
    // They are evaluated from the original UVs, never raster-baked.
    unsigned int bump_data_layer;
    unsigned int bump_procedural;
    float bump_procedural_scale;
    float bump_procedural_detail;
    float bump_procedural_roughness;
    float bump_procedural_lacunarity;
    float bump_texture_mix;
};
static_assert(sizeof(OpenPBRLayeredTextureParams) == 664, "layered texture ABI changed");

struct OpenPBRParams
{
    // -- Base ---------------------------------------------------------------
    OpenPBRColor base_color; // 12
    float base_weight; //  4  -- 16

    float base_diffuse_roughness; //  4
    float base_metalness; //  4
    float specular_weight; //  4
    float specular_roughness; //  4  -- 32

    // -- Specular -----------------------------------------------------------
    OpenPBRColor specular_color; // 12
    float specular_roughness_anisotropy; //  4  -- 48

    float specular_ior; //  4
    // cos/sin of the anisotropy rotation rather than the angle: an Eclair
    // extension to the spec, so that a texture-filtered angle does not wrap.
    float specular_anisotropy_rotation_cos; //  4
    float specular_anisotropy_rotation_sin; //  4
    float coat_weight; //  4  -- 64

    // -- Coat ---------------------------------------------------------------
    OpenPBRColor coat_color; // 12
    float coat_roughness; //  4  -- 80

    float coat_roughness_anisotropy; //  4
    float coat_ior; //  4
    float coat_darkening; //  4
    float coat_anisotropy_rotation_cos; //  4  -- 96

    float coat_anisotropy_rotation_sin; //  4
    float fuzz_weight; //  4
    float fuzz_roughness; //  4
    float transmission_weight; //  4  -- 112

    // -- Fuzz ---------------------------------------------------------------
    OpenPBRColor fuzz_color; // 12
    float transmission_depth; //  4  -- 128

    // -- Transmission -------------------------------------------------------
    OpenPBRColor transmission_color; // 12
    float transmission_scatter_anisotropy; // 4  -- 144

    OpenPBRColor transmission_scatter; // 12
    float transmission_dispersion_scale; //  4  -- 160

    float transmission_dispersion_abbe_number; // 4
    float subsurface_weight; //  4
    float subsurface_radius; //  4
    float subsurface_scatter_anisotropy; //  4  -- 176

    // -- Subsurface ---------------------------------------------------------
    OpenPBRColor subsurface_color; // 12
    float thin_film_weight; //  4  -- 192

    OpenPBRColor subsurface_radius_scale; // 12
    float thin_film_thickness; //  4  -- 208

    // -- Emission -----------------------------------------------------------
    OpenPBRColor emission_color; // 12
    float thin_film_ior; //  4  -- 224

    float emission_luminance; //  4
    float geometry_opacity; //  4
    unsigned int geometry_thin_walled; //  4
    unsigned int texture_mask; //  4  -- 240

    // -- KHR_texture_transform equivalent, folded from MaterialX place2d ------
    // One transform per material, matching what MaterialParams already does for
    // glTF. Stored as scalars, not a float2: see the header note.
    float uv_offset_x; //  4
    float uv_offset_y; //  4
    float uv_scale_x; //  4
    float uv_scale_y; //  4  -- 256

    float uv_rotation; //  4
    // Low two bits select the glTF roughness channel; the remaining flags keep
    // glTF's texture-times-factor semantics. Native OpenPBR/MaterialX leaves
    // them 0 and treats a connected image as the authored input value.
    unsigned int texture_scalar_flags;
    float texture_normal_scale;
    float _pad; // 4  -- 272
};

enum : unsigned int
{
    OPENPBR_ROUGHNESS_CHANNEL_MASK = 3u,
    OPENPBR_ROUGHNESS_MULTIPLY = 1u << 2,
    OPENPBR_TEXTURES_GLTF = 1u << 3,
};

#if defined(__METAL_VERSION__)
static_assert(sizeof(OpenPBRParams) == 272, "OpenPBRParams must stay 272 bytes (Metal)");
static_assert(sizeof(OpenPBRColor) == 12, "OpenPBRColor must stay 12 bytes (Metal)");
#else
static_assert(sizeof(OpenPBRParams) == 272, "OpenPBRParams must stay 272 bytes (host/CUDA)");
static_assert(sizeof(OpenPBRColor) == 12, "OpenPBRColor must stay 12 bytes (host/CUDA)");
#endif

#if !defined(__CUDA_ARCH__) && !defined(__METAL_VERSION__)
enum OpenPBRFeature : unsigned int
{
    OPENPBR_FEATURE_SHEEN_AND_COAT = 1u << 0,
    OPENPBR_FEATURE_DISPERSION = 1u << 1,
    OPENPBR_FEATURE_TRANSLUCENCY = 1u << 2,
    OPENPBR_FEATURE_METALLIC = 1u << 3
};

inline unsigned int openpbr_features(const OpenPBRParams& p)
{
    constexpr unsigned int layerMaps = (1u << OPENPBR_TEX_COAT_WEIGHT) |
                                       (1u << OPENPBR_TEX_COAT_ROUGHNESS) |
                                       (1u << OPENPBR_TEX_COAT_COLOR) | (1u << OPENPBR_TEX_FUZZ_WEIGHT) |
                                       (1u << OPENPBR_TEX_FUZZ_ROUGHNESS) | (1u << OPENPBR_TEX_FUZZ_COLOR) |
                                       (1u << OPENPBR_TEX_GEOMETRY_COAT_NORMAL);
    constexpr unsigned int subsurfaceWeightMap = 1u << OPENPBR_TEX_SUBSURFACE_WEIGHT;

    return ((p.coat_weight > 0.0f || p.fuzz_weight > 0.0f || p.thin_film_weight > 0.0f ||
             (p.texture_mask & layerMaps) != 0u)
                ? OPENPBR_FEATURE_SHEEN_AND_COAT
                : 0u) |
           (p.transmission_dispersion_scale > 0.0f ? OPENPBR_FEATURE_DISPERSION : 0u) |
           ((p.transmission_weight > 0.0f || p.subsurface_weight > 0.0f ||
             (p.texture_mask & subsurfaceWeightMap) != 0u)
                ? OPENPBR_FEATURE_TRANSLUCENCY
                : 0u) |
           ((p.base_metalness > 0.0f || (p.texture_mask & (1u << OPENPBR_TEX_BASE_METALNESS)) != 0u)
                ? OPENPBR_FEATURE_METALLIC
                : 0u);
}

inline bool openpbr_base_only(unsigned int features)
{
    return (features &
            (OPENPBR_FEATURE_SHEEN_AND_COAT | OPENPBR_FEATURE_DISPERSION | OPENPBR_FEATURE_TRANSLUCENCY)) == 0u;
}

inline OpenPBRParams openpbr_make_default_params()
{
    OpenPBRParams p = {};

    p.base_weight = 1.0f;
    p.base_color = OpenPBRColor{ 0.8f, 0.8f, 0.8f };
    p.base_diffuse_roughness = 0.0f;
    p.base_metalness = 0.0f;

    p.subsurface_weight = 0.0f;
    p.subsurface_color = OpenPBRColor{ 0.8f, 0.8f, 0.8f };
    p.subsurface_radius = 1.0f;
    p.subsurface_radius_scale = OpenPBRColor{ 1.0f, 0.5f, 0.25f };
    p.subsurface_scatter_anisotropy = 0.0f;

    p.specular_weight = 1.0f;
    p.specular_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
    p.specular_roughness = 0.3f;
    p.specular_roughness_anisotropy = 0.0f;
    p.specular_ior = 1.5f;
    p.specular_anisotropy_rotation_cos = 1.0f;
    p.specular_anisotropy_rotation_sin = 0.0f;

    p.coat_weight = 0.0f;
    p.coat_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
    p.coat_roughness = 0.0f;
    p.coat_roughness_anisotropy = 0.0f;
    p.coat_ior = 1.6f;
    p.coat_darkening = 1.0f;
    p.coat_anisotropy_rotation_cos = 1.0f;
    p.coat_anisotropy_rotation_sin = 0.0f;

    p.fuzz_weight = 0.0f;
    p.fuzz_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
    p.fuzz_roughness = 0.5f;

    p.transmission_weight = 0.0f;
    p.transmission_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };
    p.transmission_depth = 0.0f;
    p.transmission_scatter = OpenPBRColor{ 0.0f, 0.0f, 0.0f };
    p.transmission_scatter_anisotropy = 0.0f;
    p.transmission_dispersion_scale = 0.0f;
    p.transmission_dispersion_abbe_number = 20.0f;

    p.thin_film_weight = 0.0f;
    p.thin_film_thickness = 0.5f;
    p.thin_film_ior = 1.4f;

    p.emission_luminance = 0.0f;
    p.emission_color = OpenPBRColor{ 1.0f, 1.0f, 1.0f };

    p.geometry_opacity = 1.0f;
    p.geometry_thin_walled = 0u;
    p.texture_mask = 0u;

    // Identity UV transform, matching KHR_texture_transform's defaults.
    p.uv_offset_x = 0.0f;
    p.uv_offset_y = 0.0f;
    p.uv_scale_x = 1.0f;
    p.uv_scale_y = 1.0f;
    p.uv_rotation = 0.0f;
    p.texture_normal_scale = 1.0f;

    return p;
}
#endif

#endif // STRELKA_MATERIAL_OPENPBR_PARAMS_H
