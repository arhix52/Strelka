#ifndef STRELKA_TEXTURE_SAMPLE_H
#define STRELKA_TEXTURE_SAMPLE_H

// ============================================================================
// texture_sample.h -- Platform-specific texture sampling abstractions
//
// Provides a uniform texture_sample_2d() function across CUDA, Metal, and CPU.
// ============================================================================

#include "material_math.h"

// ===========================================================================
// CUDA backend
// ===========================================================================
#if defined(__CUDA_ARCH__)

// On CUDA, textures are identified by cudaTextureObject_t (a 64-bit handle).
// The caller passes a pointer/array of texture objects and an index.

DEVICE_FUNC float4 texture_sample_2d(cudaTextureObject_t tex, float2 uv)
{
    return tex2D<float4>(tex, uv.x, uv.y);
}

// Convenience: sample by index into an array of texture objects
// Returns white (1,1,1,1) if index < 0 (no texture bound).
DEVICE_FUNC float4 texture_sample_2d(const cudaTextureObject_t* textures,
                                     int tex_index, float2 uv)
{
    if (tex_index < 0)
        return make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    return tex2D<float4>(textures[tex_index], uv.x, uv.y);
}

// ===========================================================================
// Metal backend
// ===========================================================================
#elif defined(__METAL_VERSION__)

// On Metal the kernel receives an array of texture2d<float> via an argument
// buffer.  The caller is responsible for passing the texture array and a
// sampler.  We provide a thin wrapper so BxDF code can stay backend-agnostic.

// NOTE: Metal textures must be passed explicitly; these helpers are templates
// so they work with whatever texture/sampler types the shader uses.

template <typename TextureArray, typename Sampler>
inline float4 texture_sample_2d(TextureArray textures, int tex_index,
                                float2 uv, Sampler s)
{
    if (tex_index < 0)
        return float4(1.0f, 1.0f, 1.0f, 1.0f);
    return textures[tex_index].sample(s, uv);
}

// Overload for a single texture2d<float, access::sample>
template <typename Tex2D, typename Sampler>
inline float4 texture_sample_2d_single(Tex2D tex, float2 uv, Sampler s)
{
    return tex.sample(s, uv);
}

// ===========================================================================
// CPU stub (for unit tests and offline previews)
// ===========================================================================
#else

// CPU textures are not available by default.  Return white so that material
// evaluation still runs correctly (texture factor = 1).

struct CpuTexture2D
{
    // Placeholder -- host code can specialize this if needed.
};

DEVICE_FUNC float4 texture_sample_2d(const CpuTexture2D* /*textures*/,
                                     int /*tex_index*/, float2 /*uv*/)
{
    return make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}

// Overload accepting a null pointer (convenience for tests)
DEVICE_FUNC float4 texture_sample_2d(const void* /*textures*/,
                                     int /*tex_index*/, float2 /*uv*/)
{
    return make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}

#endif

#endif // STRELKA_TEXTURE_SAMPLE_H
