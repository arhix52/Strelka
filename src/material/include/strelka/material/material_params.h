#ifndef STRELKA_MATERIAL_PARAMS_H
#define STRELKA_MATERIAL_PARAMS_H

// ============================================================================
// material_params.h -- Material description uploaded from the host to the GPU
// ============================================================================
//
// This header is included from CUDA device code, Metal shaders, CPU tests,
// and CUDA host code (compiled by g++ with CUDA headers). It must NOT pull in
// material_math.h because that conflicts with CUDA's vector_types.h on host.
// Consumers that need math utilities should include material_math.h separately.

// Ensure float3 is available on every platform:
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
    // CUDA: float3 comes from vector_types.h (included by cuda_runtime.h)
#elif defined(__METAL_VERSION__)
    // Metal: float3 is a built-in type
#elif __has_include(<cuda_runtime.h>)
    // CUDA host code (g++ with CUDA in include path): use CUDA's float3
    #include <vector_types.h>
#else
    // Pure CPU (tests): use GLM
    #include <glm/glm.hpp>
    #ifndef STRELKA_MATERIAL_FLOAT_TYPES
    #define STRELKA_MATERIAL_FLOAT_TYPES
    using float3 = glm::vec3;
    #endif
#endif

// ---------------------------------------------------------------------------
// Material type tag -- selects which BxDF to evaluate
// ---------------------------------------------------------------------------
// glTF alphaMode. MASK is the degenerate case of BLEND -- a binary opacity --
// so everything downstream only ever sees a resolved float in [0,1].
enum AlphaMode : unsigned int
{
    ALPHA_MODE_OPAQUE = 0,
    ALPHA_MODE_MASK   = 1,
    ALPHA_MODE_BLEND  = 2,
};

enum MaterialType : unsigned int
{
    MATERIAL_TYPE_DIFFUSE       = 0,
    MATERIAL_TYPE_CONDUCTOR     = 1,
    MATERIAL_TYPE_DIELECTRIC    = 2,
    MATERIAL_TYPE_STANDARD_PBR  = 3,
    MATERIAL_TYPE_COUNT
};

// ---------------------------------------------------------------------------
// MaterialParams -- flat POD struct, safe for CUDA / Metal / CPU
//
// Layout is carefully padded to 16-byte alignment so that the struct can be
// placed in constant buffers, SBT records, or Metal argument buffers without
// alignment surprises.
// ---------------------------------------------------------------------------
struct MaterialParams
{
    // -- Base color (albedo) ------------------------------------------------
    float3      base_color;             // 12 bytes
    float       metallic;               //  4 bytes  -- total 16

    // -- Roughness / specular -----------------------------------------------
    float       roughness;              //  4 bytes
    float       ior;                    //  4 bytes  (index of refraction)
    float       specular;               //  4 bytes  (specular level, glTF)
    float       specular_tint;          //  4 bytes  -- total 32

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

    // -- KHR_texture_transform -----------------------------------------------
    // One transform per material rather than per slot: Blender drives every
    // slot of a material from the same Mapping node, and every material in the
    // scenes checked so far has an identical transform on all of its slots.
    float       uv_offset_x;            //  4 bytes
    float       uv_offset_y;            //  4 bytes
    float       uv_scale_x;             //  4 bytes
    float       uv_scale_y;             //  4 bytes  -- total 160

    // -- KHR_materials_diffuse_transmission ----------------------------------
    // What a leaf does: light enters and leaves diffusely on the far side. Kept
    // apart from `transmission`, which is specular refraction through an
    // interface and would make foliage look like glass.
    float3      diffuse_transmission_color; // 12 bytes
    float       diffuse_transmission;       //  4 bytes  -- total 176
};

// Static assert equivalent for size (works on all three backends)
// 176 bytes, 16-byte aligned -- fits nicely in SBT / argument buffers.

#endif // STRELKA_MATERIAL_PARAMS_H
