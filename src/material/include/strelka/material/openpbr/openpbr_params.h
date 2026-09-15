#ifndef STRELKA_MATERIAL_OPENPBR_PARAMS_H
#define STRELKA_MATERIAL_OPENPBR_PARAMS_H

#include <strelka/material/material_params.h>

struct OpenPBRColor
{
    float r;
    float g;
    float b;
};

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
    MAX_OPENPBR_TEXTURES = 19
};

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
    float _pad[3]; // 12  -- 272
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

    return p;
}
#endif

#endif // STRELKA_MATERIAL_OPENPBR_PARAMS_H
