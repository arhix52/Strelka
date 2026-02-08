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
        cudaTextureObject_t  filtered_object,
        cudaTextureObject_t  unfiltered_object,
        uint3                size)
        : filtered_object(filtered_object)
        , unfiltered_object(unfiltered_object)
        , size(size)
    {}

    cudaTextureObject_t  filtered_object;    // uses filter mode cudaFilterModeLinear
    cudaTextureObject_t  unfiltered_object;  // uses filter mode cudaFilterModePoint
    uint3                size;               // size of the texture
};
