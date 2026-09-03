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
