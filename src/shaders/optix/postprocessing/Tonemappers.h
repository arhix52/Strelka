#pragma once
#include <strelka/render/buffer.h>

#include <cuda_runtime_api.h>
#include <sutil/vec_math.h>
#include <stdint.h>

enum class ToneMapperType : uint32_t
{
    eNone = 0,
    eReinhard,
    eACES,
    eFilmic,
};

extern "C" void tonemap(const ToneMapperType type,
                        const float3 exposure,
                        const float maxOutput,
                        const float gamma,
                        float4 *image,
                        const uint32_t width,
                        const uint32_t height);

/// Convert scene-linear float4 pixels directly into a linear RGBA16F Vulkan
/// image. Gamma and output transfer functions intentionally remain the
/// swapchain's responsibility.
extern "C" cudaError_t tonemapToSurfaceLinear(
    const float4 *source,
    cudaSurfaceObject_t destination,
    const uint32_t width,
    const uint32_t height,
    const oka::PresentationMetadata *metadata,
    cudaStream_t stream);
