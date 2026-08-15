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

// Ensure float3 is available on every platform.
//
// The host uses GLM whether or not CUDA happens to be installed. This used to
// branch on __has_include(<cuda_runtime.h>) and pull in CUDA's vector_types.h
// for "CUDA host code", which made the *same* host translation unit see two
// different float3: material_math.h's CPU branch typedefs glm::vec3, and then
// this header re-declared it as CUDA's struct. On macOS there is no CUDA header
// to find, so the contradiction was invisible; on Linux it broke every
// translation unit that included both -- all of tests/material, test_scene and
// test_light_json -- with "using typedef-name after struct".
//
// The one host translation unit set that genuinely wants CUDA's float3 is the
// OptiX backend's own .cpp files, which also include sutil and build the launch
// Params out of make_float3. Those opt in explicitly with
// STRELKA_MATERIAL_CUDA_HOST, set on the strelka_render target. Detecting it by
// probing for the header instead meant the choice depended on whether CUDA
// happened to be installed, which is why this was correct on macOS and broken on
// every Linux box.
//
// MaterialParams is copied between host and device as a whole struct (see
// OptiXRender::createOptixMaterials), never field by field, and glm::vec3 and
// CUDA's float3 are both three unpadded floats at alignment 4 -- so both spellings
// produce the layout the GPU reads.
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
    MATERIAL_TYPE_HAIR          = 4, // Chiang et al. 2016; curves only in practice
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

    // -- KHR_materials_sheen -------------------------------------------------
    // A retroreflective fabric layer over the base. `sheen` is the extension's
    // sheenColorFactor collapsed to a weight and `sheen_color` its hue, kept
    // apart so a black sheen colour reads as "off" rather than as "black fuzz".
    float3      sheen_color;            // 12 bytes
    float       sheen;                  //  4 bytes  -- total 192

    // -- STRELKA_materials_subsurface ----------------------------------------
    // A scattering medium bounded by the surface, entered through the diffuse
    // transmission lobe and left by a random walk -- see the SSS block in
    // wavefront.metal. `subsurface_radius` is the mean free path per channel in
    // world units, which is where the colour of wax, marble and skin comes from:
    // red travels furthest, so a thin edge goes red before it goes bright.
    float3      subsurface_radius;      // 12 bytes
    float       sheen_roughness;        //  4 bytes  -- total 208
    float       subsurface;             //  4 bytes  (weight; 0 = no medium)
    float       subsurface_anisotropy;  //  4 bytes  (Henyey-Greenstein g)
    /// KHR_materials_clearcoat has no IOR field, but V-Ray and every DCC that
    /// authors a coat does; 1.5 is the extension's implied lacquer.
    float       clearcoat_ior;          //  4 bytes

    // -- STRELKA_materials_medium --------------------------------------------
    // A participating medium bounded by the geometry carrying this material --
    // V-Ray's EnvironmentFog with a gizmo, and the reason the bath water glows.
    //
    // The medium reuses subsurface_radius (mean free path), the scattering albedo
    // and the anisotropy: a fog volume and a block of wax differ in where light
    // enters, not in what happens inside. What is only a medium's is emission,
    // and the flag saying the surface is a boundary to be crossed rather than a
    // surface to be shaded.
    unsigned int medium_flags;          //  4 bytes
    float3       medium_emission;       // 12 bytes  -- total 240

    // -- KHR_materials_specular ----------------------------------------------
    // specularColorFactor: an independent tint on the dielectric F0. White is
    // the neutral value, so this cannot share the zero-initialised default the
    // rest of the struct relies on -- see the note in bsdf_init.
    float3      specular_color;         // 12 bytes

    // -- KHR_materials_iridescence -------------------------------------------
    // Thin-film interference over the specular lobe. `thickness` is in
    // nanometres and is the extension's iridescenceThicknessMaximum, which is
    // what the spec says to use when there is no thickness texture.
    // The surface albedo `diffuse_transmission_color` was derived from. The walk
    // scales its albedo by how far the resolved surface colour departs from this,
    // which is what lets a textured translucent surface -- marble veining, a
    // printed rubber duck -- carry its texture into the medium. Equal to the
    // resolved albedo when there is no texture, so the ratio is exactly 1 and
    // nothing changes for a flat material.
    float3      subsurface_reference;   // 12 bytes
    float       iridescence;            //  4 bytes  -- total 272
    float       iridescence_ior;        //  4 bytes
    float       iridescence_thickness;  //  4 bytes
    float       _pad_irid;              //  4 bytes  -- total 288
};

/// medium_flags bit 0: this material's geometry is the boundary of a medium.
/// A ray crossing it toggles the medium it is in and carries on, unshaded and
/// without spending a bounce.
#define MEDIUM_FLAG_BOUNDARY 1u

// Static assert equivalent for size (works on all three backends)
// 288 bytes, 16-byte aligned -- fits nicely in SBT / argument buffers.

#endif // STRELKA_MATERIAL_PARAMS_H
