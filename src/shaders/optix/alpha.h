#pragma once

// Alpha cutout for the OptiX backend.
//
// glTF has two transparent modes and one mechanism underneath them: MASK is a
// binary predicate on the base-colour alpha, BLEND passes that alpha through.
// Both resolve to a coverage in [0,1], so callers never branch on the mode --
// which is the same shape resolveOpacity() has on Metal, deliberately.
//
// The two ray types spend that coverage differently, again matching Metal:
//
//   * radiance rays test it stochastically at the closest hit and, when the
//     test passes, continue in the same direction unshaded. One random draw
//     covers MASK and BLEND alike.
//   * shadow rays accumulate the product of (1 - opacity) in an any-hit
//     program, so a light seen through two half-transparent leaves arrives at a
//     quarter strength rather than being blocked outright by the first one.
//
// Doing the radiance test at the closest hit rather than in an any-hit program
// is not a shortcut: an any-hit that accepts stochastically is called once per
// candidate the traversal considers, so a path that slips through a canopy pays
// for every leaf in it, and OptiX is free to invoke an any-hit more than once
// for the same intersection -- which a random accept turns into a different
// answer each time.

#include <strelka/material/material_params.h>
#include <strelka/material/texture_sample.h>

#include <random.h> // pcg_hash

#include <sutil/vec_math.h>

/// Coverage of a surface at a uv. 1 for an opaque material, 0 or 1 for MASK,
/// the base-colour alpha for BLEND.
///
/// Opacity uses raw UV; material shading passes transformed UV to bsdf_init.
static __forceinline__ __device__ float resolveOpacity(const MaterialParams& material,
                                                       const cudaTextureObject_t* textures,
                                                       float2 uv)
{
    if (material.alpha_mode == ALPHA_MODE_OPAQUE)
    {
        return 1.0f;
    }
    float alpha = material.base_color_alpha;
    if (material.base_color_tex >= 0)
    {
        // The base-colour texture is uploaded as unsigned char RGBA read as a
        // normalised float, with no sRGB decode applied by the texture unit, so
        // the alpha channel is already linear.
        alpha *= texture_sample_2d(textures, material.base_color_tex, uv).w;
    }
    if (material.alpha_mode == ALPHA_MODE_MASK)
    {
        return alpha >= material.alpha_cutoff ? 1.0f : 0.0f;
    }
    return __saturatef(alpha);
}

/// Use sampler eOpacity for joint stratification with pixel jitter, rotated per
/// cutout layer.
///
/// The per-layer rotation is what makes a stack of cutouts behave like a stack.
/// Passing through does not advance the sampler, so without it two leaves of
/// the same alpha make the same decision and a canopy that should pass a^2 of
/// what reaches it passes a. A rotation rather than a fresh hash, so that each
/// layer on its own stays stratified across samples. Same shape as Metal's.
static __forceinline__ __device__ float opacitySample(SamplerState& sampler,
                                                      uint32_t pixelIndex,
                                                      uint32_t layer)
{
    float u = random<SampleDimension::eOpacity>(sampler);
    if (layer != 0u)
    {
        constexpr float kInv2p32 = 2.3283064365386963e-10f; // 1 / 2^32
        const float rot = (float)pcg_hash(layer * 2654435761u + pixelIndex * 2246822519u) * kInv2p32;
        u += rot;
        u -= floorf(u);
    }
    return u;
}

/// Below this fraction of the light a shadow ray is treated as blocked.
///
/// A fraction of the light rather than a distance or a size, so it means the
/// same thing in any scene. Through a canopy the expensive rays are not the
/// blocked ones -- those stop at the first opaque leaf -- but the ones that keep
/// slipping past cutouts carrying a thousandth of the light and traverse the
/// whole crown to deliver something that rounds to nothing.
///
/// Metal spends this budget as Russian roulette and scales the survivors back
/// up, which is unbiased at a higher threshold. Here it is a plain floor at a
/// value small enough that the bias is far below the noise the estimator
/// already carries, because the OptiX shadow payload is one word and roulette
/// needs the compensation factor carried alongside the transmittance.
#define SHADOW_TRANSMITTANCE_CUTOFF 1e-3f
