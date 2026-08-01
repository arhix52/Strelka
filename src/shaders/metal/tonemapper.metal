#include <metal_stdlib>
#include "tonemappers.h"
#include "ShaderTypes.h"

using namespace metal;


kernel void toneMappingComputeShader(
    uint2 tid [[thread_position_in_grid]],
    constant UniformsTonemap& uniforms [[buffer(0)]],
    device float4* buffer [[buffer(1)]],
    // The display image goes to a texture rather than back into the buffer, so
    // the buffer keeps the linear radiance the reference capture writes out and
    // MetalFX gets something it can consume without a copy.
    texture2d<float, access::write> displayTexture [[texture(0)]]
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
    displayTexture.write(float4(result, inputColor.a), tid);
}

// Same tone curve, texture in instead of buffer in.
//
// After denoising the frame is a texture at display resolution and the buffer no
// longer holds it, so the input side has to change; the maths below is the same
// and stays in one place by construction, because both kernels call it.
kernel void toneMappingTextureShader(
    uint2 tid [[thread_position_in_grid]],
    constant UniformsTonemap& uniforms [[buffer(0)]],
    texture2d<float, access::read> source [[texture(1)]],
    texture2d<float, access::write> displayTexture [[texture(0)]]
    )
{
    // Display resolution, not render resolution. The denoiser hands back a
    // display-sized texture, and bounding this pass by the render size instead
    // leaves everything outside the top-left corner holding whatever was in the
    // display texture before -- a quarter of the screen live and the rest a stale
    // still, which reads as "working" for as long as that still happens to be
    // roughly right.
    if (tid.x >= uniforms.outWidth || tid.y >= uniforms.outHeight)
    {
        return;
    }
    const float4 inputColor = source.read(tid);
    float3 result = inputColor.xyz;
    const float3 exposedResult = result * uniforms.exposureValue;
    switch ((ToneMapperType) uniforms.tonemapperType)
    {
    case ToneMapperType::eReinhard: result = reinhard(exposedResult); break;
    case ToneMapperType::eACES:     result = ACESFitted(exposedResult); break;
    case ToneMapperType::eFilmic:   result = ACESFilm(exposedResult); break;
    case ToneMapperType::eNone:     result = exposedResult; break;
    }
    if (uniforms.gamma > 0.0f)
    {
        result = srgbGamma(result, uniforms.gamma);
    }
    displayTexture.write(float4(result, inputColor.a), tid);
}
