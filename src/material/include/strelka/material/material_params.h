#ifndef STRELKA_MATERIAL_PARAMS_H
#define STRELKA_MATERIAL_PARAMS_H

#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
    // CUDA device code: float3 comes from vector_types.h (via cuda_runtime.h)
#elif defined(__METAL_VERSION__)
    // Metal: float3 is a built-in type
#elif defined(STRELKA_MATERIAL_CUDA_HOST)
    // OptiX backend host code: match the device types it interoperates with.
    #include <vector_types.h>
#else
    // Everything else (tests, loaders, the Metal backend's host side): GLM
    #include <glm/glm.hpp>
    #ifndef STRELKA_MATERIAL_FLOAT_TYPES
    #define STRELKA_MATERIAL_FLOAT_TYPES
    using float2 = glm::vec2;
    using float3 = glm::vec3;
    using float4 = glm::vec4;
    #endif
#endif

enum AlphaMode : unsigned int
{
    ALPHA_MODE_OPAQUE = 0,
    ALPHA_MODE_MASK   = 1,
    ALPHA_MODE_BLEND  = 2,
    ALPHA_MODE_SHADOW_TRANSPARENT = 3,
};

// The values are an ABI the shaders share, written out one by one on purpose;
// COUNT is a sentinel rather than a material, so it has no number of its own.
// NOLINTNEXTLINE(cert-int09-c)
enum MaterialType : unsigned int
{
    MATERIAL_TYPE_DIFFUSE       = 0,
    MATERIAL_TYPE_CONDUCTOR     = 1,
    MATERIAL_TYPE_DIELECTRIC    = 2,
    MATERIAL_TYPE_STANDARD_PBR  = 3,
    MATERIAL_TYPE_HAIR          = 4, // Chiang et al. 2016; curves only in practice
    MATERIAL_TYPE_OPENPBR       = 5,
    MATERIAL_TYPE_COUNT
};

struct MaterialParams
{
    // -- Base color (albedo) ------------------------------------------------
    float3      base_color;             // 12 bytes
    float       metallic;               //  4 bytes  -- total 16

    // -- Roughness / specular -----------------------------------------------
    float       roughness;              //  4 bytes
    float       ior;                    //  4 bytes  (index of refraction)
    float       specular;               //  4 bytes  (specular level, glTF)
    float       _pad_specular;          //  4 bytes  -- total 32 (was specular_tint)

    // -- Transmission / clearcoat -------------------------------------------
    float       transmission;           //  4 bytes
    float       clearcoat;              //  4 bytes
    float       clearcoat_roughness;    //  4 bytes
    float       anisotropy;             //  4 bytes  -- total 48

    // -- Emission -----------------------------------------------------------
    float3      emission;               // 12 bytes
    float       emission_strength;      //  4 bytes  -- total 64

    // -- Normal / opacity ---------------------------------------------------
    float       normal_scale;           //  4 bytes
    float       occlusion_strength;     //  4 bytes
    float       alpha_cutoff;           //  4 bytes
    unsigned int material_type;         //  4 bytes  (MaterialType)  -- total 80

    // -- Texture indices (-1 = no texture) ----------------------------------
    int         base_color_tex;         //  4 bytes
    int         metallic_roughness_tex; //  4 bytes
    int         normal_tex;             //  4 bytes
    int         emission_tex;           //  4 bytes  -- total 96

    int         occlusion_tex;          //  4 bytes
    int         transmission_tex;       //  4 bytes
    unsigned int dielectric_priority;   //  4 bytes  (nested dielectrics)
    // glTF alphaMode. Opacity, not material type: routing MASK/BLEND to a
    // dielectric converter is what used to turn every cutout into glass.
    unsigned int alpha_mode;            //  4 bytes  (AlphaMode)  -- total 112

    // -- Thin-surface flag --------------------------------------------------
    unsigned int thin_walled;           //  4 bytes
    float       base_color_alpha;       //  4 bytes  (baseColorFactor.a)
    float       anisotropy_rotation;    //  4 bytes  (radians, CCW about N)
    float       attenuation_distance;   //  4 bytes  -- total 128

    // -- KHR_materials_volume ------------------------------------------------
    // Colour the medium leaves after attenuation_distance of travel. See
    // volume_extinction() for the two conventions this can be read under.
    float3      attenuation_color;      // 12 bytes
    float       uv_rotation;            //  4 bytes  (radians, KHR_texture_transform)

    float       uv_offset_x;            //  4 bytes
    float       uv_offset_y;            //  4 bytes
    float       uv_scale_x;             //  4 bytes
    float       uv_scale_y;             //  4 bytes  -- total 160

    float3      diffuse_transmission_color; // 12 bytes
    float       diffuse_transmission;       //  4 bytes  -- total 176

    float3      sheen_color;            // 12 bytes
    float       sheen;                  //  4 bytes  -- total 192

    float3      subsurface_radius;      // 12 bytes
    float       sheen_roughness;        //  4 bytes  -- total 208
    float       subsurface;             //  4 bytes  (weight; 0 = no medium)
    float       subsurface_anisotropy;  //  4 bytes  (Henyey-Greenstein g)
    /// KHR_materials_clearcoat has no IOR field, but V-Ray and every DCC that
    /// authors a coat does; 1.5 is the extension's implied lacquer.
    float       clearcoat_ior;          //  4 bytes

    unsigned int medium_flags;          //  4 bytes
    float3       medium_emission;       // 12 bytes  -- total 240

    float3      specular_color;         // 12 bytes

    float3      subsurface_reference;   // 12 bytes
    float       iridescence;            //  4 bytes  -- total 272
    float       iridescence_ior;        //  4 bytes
    float       iridescence_thickness;  //  4 bytes
    float       _pad_irid;              //  4 bytes  -- total 288
};

/// medium_flags bit 0: this material's geometry is the boundary of a medium.
/// A ray crossing it toggles the medium it is in and carries on, unshaded and
/// without spending a bounce.
enum : unsigned int
{
    MEDIUM_FLAG_BOUNDARY = 1u
};

// Static assert equivalent for size (works on all three backends)
// 288 bytes, 16-byte aligned -- fits nicely in SBT / argument buffers.

#endif // STRELKA_MATERIAL_PARAMS_H
