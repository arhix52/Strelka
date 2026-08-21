#ifndef STRELKA_MATERIAL_OPENPBR_PARAMS_H
#define STRELKA_MATERIAL_OPENPBR_PARAMS_H

// ============================================================================
// openpbr_params.h -- the OpenPBR Surface argument block, host to GPU
// ============================================================================
//
// One struct, read verbatim by CUDA device code, Metal shaders, CUDA host code
// and the CPU tests. Unlike MaterialParams there is deliberately **no
// per-backend mirror** of it, and that is the whole point of how it is spelled.
//
// MaterialParams has one, `struct Material` in src/shaders/metal/ShaderTypes.h,
// because MaterialParams uses `float3` -- which is 12 bytes on the host and
// under CUDA, and *16* under Metal. A struct containing it therefore cannot be
// read out of a Metal buffer, so Metal grew a parallel declaration spelled with
// packed_float3 plus a hand-written field-by-field converter
// (MetalMaterials.mm::makeMaterialParams). That converter currently drops nine
// MaterialParams fields on the floor, silently, because nothing checks it.
//
// This struct avoids the mirror by containing no vector type at all. Colours are
// OpenPBRColor -- three bare floats, which every one of the four compilers lays
// out as 12 bytes at alignment 4 -- and the two anisotropy rotations are stored
// as separate scalars for the same reason (Metal's and CUDA's float2 align to 8,
// GLM's vec2 to 4). What the host writes is byte-for-byte what each shader
// reads, and the static_assert at the bottom is what keeps it that way.
//
// Textures are not in here either. They are a parallel table indexed by
// [materialId * MAX_OPENPBR_TEXTURES + slot], resolved by each backend into the
// handle type it uses -- cudaTextureObject_t on OptiX, MTL::ResourceID on Metal
// -- exactly as params.materialTextures already does for the glTF path.
//
// Field names and defaults follow the OpenPBR 1.1.1 specification, so this maps
// one-to-one onto OpenPBR_ResolvedInputs in third_party/openpbr_bsdf. The two
// geometry bases that struct also carries are built at the shading point from
// the hit, not stored here.

#include <strelka/material/material_params.h>

// Three floats, 12 bytes, alignment 4 -- on all four compilers. See the note
// above: this exists so that no backend needs its own copy of the struct below.
//
// This *is* Metal's packed_float3, and the numbers are worth stating because the
// alternative is not obviously worse until it is measured. On an M4 Pro:
//
//     sizeof(packed_float3)              12   align 4
//     sizeof(float3)                     16   align 16
//     sizeof(struct{packed_float3,float})16
//     sizeof(struct{float3,float})       32   <-- the row silently doubles
//
// So packed_float3 is exactly the right tool and `struct Material` in
// src/shaders/metal/ShaderTypes.h correctly uses it. It is simply not reachable
// from here: that header is the Metal backend's, it opens with
// #include <simd/simd.h>, and its non-Metal shim takes a vector_float3 -- none
// of which exists in a CUDA build on Linux or Windows, where this header still
// has to compile. Hence a fourth spelling of the same twelve bytes rather than a
// fifth dependency.
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
    // Appended after real content asked for them, which is the only reason any
    // of these should grow. The Open Chess Set drives subsurface_radius from a
    // map in 13 of its 15 materials and subsurface weight in 4; a fabric wants
    // its fuzz tint the same way velvet states it as a constant.
    OPENPBR_TEX_SUBSURFACE_WEIGHT = 16,
    OPENPBR_TEX_SUBSURFACE_RADIUS = 17,
    OPENPBR_TEX_FUZZ_COLOR = 18,
    MAX_OPENPBR_TEXTURES = 19
};

// The ceiling is 32, not a matter of taste: OpenPBRParams::texture_mask is one
// uint32_t with a bit per slot, and it is what makes the count nearly free --
// see the note on that field. Past 32 the gate would need a second word and
// every shader that reads it would have to change.


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
    // One bit per OpenPBRTextureSlot, derived by the renderer from which slots
    // the material actually names a file for.
    //
    // This is what decides the cost of having slots at all. The handles live in
    // a *different* buffer (OpenPBRTextures, in the backend's own header,
    // because a texture handle has a different type per backend), so testing
    // them directly means streaming that buffer's cache lines at every hit --
    // for every material, including the ones with no maps, which is most of
    // them. The mask is already here, in a struct the shading path has loaded,
    // so `texture_mask == 0` skips the second buffer entirely and a per-slot bit
    // skips a handle load. Adding a slot then costs eight bytes in a buffer that
    // is only touched when some material says it has maps.
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

// The mirror this struct does not have is replaced by this line. If a field is
// added without keeping the 16-byte rows above, the Metal shader and the host
// stop agreeing and nothing else in the build would say so -- Scene::Vertex
// carries the same guard for the same reason.
#if defined(__METAL_VERSION__)
static_assert(sizeof(OpenPBRParams) == 272, "OpenPBRParams must stay 272 bytes (Metal)");
static_assert(sizeof(OpenPBRColor) == 12, "OpenPBRColor must stay 12 bytes (Metal)");
#else
static_assert(sizeof(OpenPBRParams) == 272, "OpenPBRParams must stay 272 bytes (host/CUDA)");
static_assert(sizeof(OpenPBRColor) == 12, "OpenPBRColor must stay 12 bytes (host/CUDA)");
#endif

// ---------------------------------------------------------------------------
// Specification defaults
// ---------------------------------------------------------------------------
//
// Host-side only: device code is handed a filled block and never authors one.
// Keeping it out of the shaders also keeps this header free of material_math.h,
// which material_params.h documents it must not pull in.
//
// These numbers are the OpenPBR 1.1.1 defaults, and they are duplicated from
// openpbr_make_default_resolved_inputs() in third_party/openpbr_bsdf on purpose:
// this header must not depend on the vendored one, because the glTF loader
// includes it and has no business compiling a BSDF. The duplication is not
// trusted -- tests/material/test_openpbr_params.cpp asserts the two agree field
// by field, so the copy cannot drift.
//
// A zero-initialised OpenPBRParams is NOT a valid material: it has zero IOR,
// zero coat_darkening and a zero anisotropy rotation cosine, which is a
// degenerate basis rather than "no rotation". Always start here.
#if !defined(__CUDA_ARCH__) && !defined(__METAL_VERSION__)
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
