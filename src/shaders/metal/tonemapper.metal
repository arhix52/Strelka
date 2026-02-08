#include <metal_stdlib>
#include "tonemappers.h"
#include "ShaderTypes.h"

using namespace metal;


kernel void toneMappingComputeShader(
    uint2 tid [[thread_position_in_grid]],
    constant UniformsTonemap& uniforms [[buffer(0)]],
    device float4* buffer [[buffer(1)]]
    )
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
    {
        return;
    }
    const uint32_t linearPixelIndex = tid.y * uniforms.width + tid.x;

    // Fetch the input color
    float4 inputColor = buffer[linearPixelIndex];
    float3 result = inputColor.xyz;
    // Apply exposure only (maxEDR is macOS HDR headroom — irrelevant for SDR/gamma output)
    float3 exposedResult = result * uniforms.exposureValue;
    switch ((ToneMapperType) uniforms.tonemapperType)
    {
    case ToneMapperType::eReinhard:
        result = reinhard(exposedResult);
        break;
    case ToneMapperType::eACES:
        result = ACESFitted(exposedResult);
        break;
    case ToneMapperType::eFilmic:
        result = ACESFilm(exposedResult);
        break;
    case ToneMapperType::eNone:
        result = exposedResult;
        break;
    }

    if (uniforms.gamma > 0.0f)
    {
        result = srgbGamma(result, uniforms.gamma);
    }
    buffer[linearPixelIndex] = float4(result, inputColor.a);
}
