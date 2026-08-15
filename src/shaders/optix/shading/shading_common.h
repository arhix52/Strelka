#ifndef STRELKA_OPTIX_SHADING_COMMON_H
#define STRELKA_OPTIX_SHADING_COMMON_H

// ============================================================================
// shading_common.h -- the CUDA counterpart of src/shaders/metal/shading_common.h
//
// The Metal backend has owned initSurfaceInteraction -- texture fetch, normal
// mapping, KHR_texture_transform, vertex colour, opacity -- since the material
// system was split out; OptiX had no equivalent and called bsdf_init() directly,
// which does none of those. That is why the OptiX path rendered no normal maps
// at all and ignored uv_offset / uv_scale / uv_rotation although the loader
// uploads them.
//
// This is a port of the *behaviour*, not of the code: Metal resolves textures
// itself and calls a bsdf_init overload that leaves si.albedo alone, while the
// CUDA overload of bsdf_init does its own tex2D fetches. So the order here is
// "hand bsdf_init the transformed uv, then correct the three things it cannot
// know about" rather than "resolve everything, then call bsdf_init".
// ============================================================================

#include <optix.h>

#include <OptixRenderParams.h>

#include <sutil/vec_math.h>
#include <sutil/vec_math_adv.h>

#include <strelka/material/bsdf.h>
#include <strelka/material/volume.h>

#include "../optix_device_utils.h"
#include "fibre_geometry.h"
#include "texture_transform.h"

// Which reading of KHR_materials_volume the OptiX path uses.
//
// volume.h documents the two, and they disagree by a lot -- at an attenuation
// colour of 0.5 the glTF form gives sigma_t = 0.69/d and the Cycles form 0.5/d.
// Metal takes it from `render/material/volumeModel`, which the CLI drives from
// the scene's `volume_model` key; the OptiX launch Params carry no such field, so
// this is pinned to the spec's own reading -- which is also the application's
// default -- until one is added. `10_glass_absorption` and `18_bounded_volume`
// both set `volume_model = "cycles"`, so both currently measure the wrong
// density on this backend; see the hand-off note in the stream report.
static const unsigned int kOptixVolumeModel = VOLUME_MODEL_GLTF;

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
    return scattersThroughFibre(si) || dot(si.shading_normal, L) > 0.0f;
}

// The factor that cancels the one hair_chiang_eval() divides by. It has to be the
// same |n.wi| and never a clamp to zero, or the two do not cancel and the fibre's
// far side comes back either black or blown out.
static __forceinline__ __device__ float shadingCosine(const SurfaceInteraction& si, float3 L)
{
    const float c = dot(si.shading_normal, L);
    return scattersThroughFibre(si) ? fabsf(c) : saturate(c);
}

// Where a ray that scattered through a strand has to start. See fibre_geometry.h
// for the chord; the only thing added here is the ray offset, which is CUDA's.
static __forceinline__ __device__ float3 fibreExitOrigin(
    float3 position, float3 tangent, float3 normal, float radius, float3 dir)
{
    const FibreExit e = fibre_exit(position, tangent, normal, radius, dir);
    return offset_ray(e.position, e.normal);
}

// ---------------------------------------------------------------------------
// Bound what a single indirect path may contribute.
//
// A firefly is a sample with an enormous weight and a tiny probability. Clamping
// trades an unbiased estimator for a biased one, so it is off by default and
// applied only past the first bounce: clamping depth 0 as well would dim every
// directly visible emitter and the environment behind it.
// ---------------------------------------------------------------------------
static __forceinline__ __device__ float3 clampIndirectContribution(float3 radiance,
                                                                   unsigned int depth,
                                                                   float limit)
{
    if (limit <= 0.0f || depth == 0u)
    {
        return radiance;
    }
    const float m = fmaxf(radiance.x, fmaxf(radiance.y, radiance.z));
    return (m > limit) ? radiance * (limit / m) : radiance;
}

// ---------------------------------------------------------------------------
// Fill SurfaceInteraction from hit geometry and sample the material's textures.
//
// `lodBase` is accepted so the signature matches Metal's and so the ray-cone
// work can land here later, but the OptiX backend builds its textures without a
// mipmapped array (see OptiXRender::loadTextureFromFile), so there is no level to
// select and it is currently unused. Silently sampling level 0 is what the
// backend did before this function existed.
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
