#pragma once
// Minimal texture support for CUDA -- MDL-free replacement.
// Only the Texture struct is needed by loadTextureFromFile().

#include <cuda.h>
#include <cuda_runtime.h>

// Custom structure representing a CUDA texture with filtered and unfiltered
// texture objects and size information.
struct Texture
{
    explicit Texture()
        : filtered_object(0)
        , unfiltered_object(0)
        , size(make_uint3(0, 0, 0))
    {}

    explicit Texture(
        cudaTextureObject_t  filtered,
        cudaTextureObject_t  unfiltered,
        uint3                dimensions)
        : filtered_object(filtered)
        , unfiltered_object(unfiltered)
        , size(dimensions)
    {}

    cudaTextureObject_t  filtered_object;    // uses filter mode cudaFilterModeLinear
    cudaTextureObject_t  unfiltered_object;  // uses filter mode cudaFilterModePoint
    uint3                size;               // size of the texture
};
