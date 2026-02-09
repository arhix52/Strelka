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
    float       _pad1;                  //  4 bytes  -- total 112

    // -- Thin-surface flag --------------------------------------------------
    unsigned int thin_walled;           //  4 bytes
    float       _pad2;                  //  4 bytes
    float       _pad3;                  //  4 bytes
    float       _pad4;                  //  4 bytes  -- total 128
};

// Static assert equivalent for size (works on all three backends)
// 128 bytes, 16-byte aligned -- fits nicely in SBT / argument buffers.

#endif // STRELKA_MATERIAL_PARAMS_H
