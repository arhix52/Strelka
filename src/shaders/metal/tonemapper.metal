#include <metal_stdlib>
#include "tonemappers.h"
#include "ShaderTypes.h"

using namespace metal;

float3 toneMapForDisplay(const float3 color, constant UniformsTonemap& uniforms)
{
    float3 result;
    const float maxOutput = max(uniforms.maxEDR, 1.0f);

    switch ((ToneMapperType)uniforms.tonemapperType)
    {
    case ToneMapperType::eReinhard:
        result = reinhard(color, maxOutput);
        break;
    case ToneMapperType::eACES:
        result = ACESFitted(color, maxOutput);
        break;
    case ToneMapperType::eFilmic:
        result = ACESFilm(color, maxOutput);
        break;
    case ToneMapperType::eNone:
        result = color;
        break;
    }

    if (uniforms.gamma > 0.0f)
    {
        result = srgbGamma(result, uniforms.gamma);
    }
    return result;
}

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
    const float3 result = toneMapForDisplay(inputColor.xyz * uniforms.exposureValue, uniforms);
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
    const float3 result = toneMapForDisplay(inputColor.xyz * uniforms.exposureValue, uniforms);
    displayTexture.write(float4(result, inputColor.a), tid);
}

// The denoised frame, back into the buffer the headless writer reads.
//
// It lands in a texture and nothing put it anywhere else, so StrelkaCLI wrote
// the accumulation buffer -- the estimate the denoiser was handed, not the one
// it produced. `--denoise` therefore cost a canonical guide sample, turned on
// frame jitter and the firefly clamp, and delivered no denoising at all to the
// file. Linear in and linear out: the tone curve is the host's, and an EXR must
// keep the radiance.
kernel void denoisedTextureToBuffer(
    uint2 tid [[thread_position_in_grid]],
    constant UniformsTonemap& uniforms [[buffer(0)]],
    device float4* buffer [[buffer(1)]],
    texture2d<float, access::read> source [[texture(0)]]
    )
{
    if (tid.x >= uniforms.outWidth || tid.y >= uniforms.outHeight)
    {
        return;
    }
    buffer[tid.y * uniforms.outWidth + tid.x] = source.read(tid);
}
