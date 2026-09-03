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
#include <strelka/material/openpbr/openpbr_bridge.h>
#include <strelka/material/valid_reflection.h>
#include <strelka/material/volume.h>

#include "../optix_device_utils.h"
#include "fibre_geometry.h"
#include <nee_pairing.h>
#include "texture_transform.h"

// KHR_materials_volume uses the model selected by `render/material/volumeModel`.

// ---------------------------------------------------------------------------
// Fibre semantics.
//
// A strand is not a surface with a lit side and a dark side. Chiang's TT and TRT
// terms describe light that entered one side of the fibre and left the other,
// and on a bright groom TT alone holds about four fifths of the albedo:
// integrated over the sphere the lobe gives R 0.047, TT 0.799, TRT 0.033. Testing
// the shading hemisphere the way a surface does discards every one of those
// connections and leaves the dominant lobe to be found by chance.
//
// All three are identities for every non-hair material, which is why the other
// twenty-seven ladder rows re-render unchanged.
// ---------------------------------------------------------------------------
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

// clampIndirectContribution() -- the firefly bound on indirect paths -- lives in
// optix_device_utils.h, included above. This file carried an identical copy of it
// until the two streams met; one definition is enough.

// ---------------------------------------------------------------------------
// OpenPBR maps
// ---------------------------------------------------------------------------
//
// Behaviour port of applyOpenPBRTextures() in src/shaders/metal/shading_common.h.
// Every decision it records holds here for the same reason, so they are not
// restated: maps *replace* rather than modulate (a MaterialX input is either a
// value or a nodegraph, never both), emission is deliberately left to the
// Material struct, and the normal map rebuilds Z from X and Y.
//
// What differs is only the plumbing. Metal binds nineteen texture handles in a
// per-material struct; OptiX indexes one flat array of cudaTextureObject_t, the
// same shape params.materialTextures already uses, and the wrap mode and
// transfer function live in the texture object rather than in a sampler here.

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

/// Folds an OpenPBR material's maps into a thread-local copy of its parameters.
///
/// `textures` is this material's slice of the global handle array, i.e.
/// `&params.openpbrTextures[materialId * MAX_OPENPBR_TEXTURES]`. A slot with no
/// map holds 0, which `texture_mask` already says; the handle is tested anyway
/// because a texture that failed to load leaves the bit set and the handle null.
static __forceinline__ __device__ void openpbr_apply_textures(OpenPBRParams& p,
                                                              const cudaTextureObject_t* textures,
                                                              SurfaceInteraction& si,
                                                              float2 uv)
{
    const float2 tuv = openpbr_transform_uv(p, uv);

    // Both halves of the gate in one place: the mask says the material named a
    // file, the handle says one arrived.
    auto bound = [&](unsigned int slot) -> bool
    { return openpbr_has_texture(p, slot) && textures[slot] != 0ull; };
    auto sample = [&](unsigned int slot) -> float4
    { return tex2D<float4>(textures[slot], tuv.x, tuv.y); };

    // The sixteen value slots as two tables rather than sixteen near-identical
    // `if` blocks. Both arrays are constexpr and both loop bounds are literals,
    // so this unrolls into exactly the same code the blocks compiled to -- the
    // saving is in what a reader has to check, not in what the GPU runs.
    //
    // Split by what the slot writes, because that is the only thing that ever
    // differed between the blocks: a scalar takes the red channel, a colour takes
    // three. Adding a slot is now one line in one table.
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
    // subsurface_radius_scale is a per-channel tint on the mean free path, which
    // is why it is here and not with the scalars: the scalar length stays as
    // authored, and a map says how the three channels differ rather than how far
    // light travels.
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
            p.*(s.field) = sample(s.slot).x;
        }
    }
#pragma unroll
    for (const ColorSlot& c : kColorSlots)
    {
        if (bound(c.slot))
        {
            const float4 v = sample(c.slot);
            p.*(c.field) = OpenPBRColor{ v.x, v.y, v.z };
        }
    }

    // Emission stays out of this on purpose: the shade path reads it from the
    // Material struct, not from the BSDF, so an emission map would have to be
    // folded there instead. Left unhandled rather than half-handled.

    if (bound(OPENPBR_TEX_GEOMETRY_NORMAL))
    {
        const float4 n = sample(OPENPBR_TEX_GEOMETRY_NORMAL);
        const float2 xy = make_float2(n.x * 2.0f - 1.0f, n.y * 2.0f - 1.0f);
        const float z = sqrtf(saturate(1.0f - (xy.x * xy.x + xy.y * xy.y)));
        si.shading_normal = safe_normalize(si.tangent * xy.x + si.bitangent * xy.y + si.shading_normal * z);
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

// ---------------------------------------------------------------------------
// Fill SurfaceInteraction from hit geometry and sample the material's textures.
//
// OptiX currently has no mipmapped material arrays, so lodBase is unused.
// ---------------------------------------------------------------------------
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
    float3 vertexColor = make_float3(1.0f))
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

    // bsdf_init's CUDA overload samples base colour, metallic-roughness and
    // emission itself, at si.uv. Handing it the transformed coordinate is what
    // makes KHR_texture_transform apply to all three at once; the untransformed
    // one goes back afterwards because that is what the interaction is supposed
    // to carry.
    si.uv = tuv;
    bsdf_init(si, material, textures);
    si.uv = uv;

    // glTF composes base colour as baseColorFactor * baseColorTexture * COLOR_0,
    // all three multiplicative. bsdf_init knows the first two.
    si.albedo *= vertexColor;

    // Normal map. Z comes from X and Y rather than from the texture: a normal map
    // is two-channel once compression is on, and for a unit-length tangent-space
    // normal this is the value that was dropped -- reading it the same way whether
    // or not the texture was compressed keeps the two paths from disagreeing.
    if (material.normal_tex >= 0)
    {
        const float4 nTex = tex2D<float4>(textures[material.normal_tex], tuv.x, tuv.y);
        const float2 bumpXY = make_float2(nTex.x * 2.0f - 1.0f, nTex.y * 2.0f - 1.0f);
        // glTF scales X and Y and leaves Z, so Z is rebuilt before the scale.
        const float bumpZ = sqrtf(saturate(1.0f - (bumpXY.x * bumpXY.x + bumpXY.y * bumpXY.y)));
        const float scale = material.normal_scale;
        const float3 bump =
            worldTangent * (bumpXY.x * scale) + worldBinormal * (bumpXY.y * scale) + worldNormal * bumpZ;
        si.shading_normal = safe_normalize(bump);
        // What the map asked for, kept before anything bends it: the test in
        // standard_pbr_eval compares the two, and comparing against the
        // pre-bump normal instead would reject directions no map ever moved.
        si.bump_normal = si.shading_normal;

        // A grazing normal map can turn the normal past the viewer, where an
        // opaque material has no valid lobe. Correct it where the shading normal
        // is produced so sampling and evaluation use the same normal; disable
        // the diffuse lobe because the correction is for reflective lobes.
        //
        // Against the geometric normal turned to agree with the view ray: the
        // correction is about the surface the reflection has to clear, and on a
        // back-face hit that surface is the side being looked at.
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
