#pragma once

#include <strelka/material/material_params.h>
#include <strelka/material/texture_sample.h>

#include <random.h> // pcg_hash

#include <sutil/vec_math.h>

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

static __forceinline__ __device__ float resolveOpacity(const OptixAlphaMaterialData& material,
                                                       cudaTextureObject_t baseColorTexture,
                                                       float2 uv)
{
    float alpha = material.baseColorAlpha;
    if (baseColorTexture)
    {
        alpha *= tex2D<float4>(baseColorTexture, uv.x, uv.y).w;
    }
    if (material.alphaMode == ALPHA_MODE_MASK)
    {
        return alpha >= material.alphaCutoff ? 1.0f : 0.0f;
    }
    return __saturatef(alpha);
}

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

#define SHADOW_TRANSMITTANCE_CUTOFF 1e-3f
