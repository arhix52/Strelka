#include "MetalFxContext.h"

#include <log.h>

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

namespace oka
{

static MTLFXSpatialScalerColorProcessingMode toNative(MetalFxContext::ColorMode mode)
{
    switch (mode)
    {
    case MetalFxContext::ColorMode::Linear:
        return MTLFXSpatialScalerColorProcessingModeLinear;
    case MetalFxContext::ColorMode::HDR:
        return MTLFXSpatialScalerColorProcessingModeHDR;
    case MetalFxContext::ColorMode::Perceptual:
    default:
        return MTLFXSpatialScalerColorProcessingModePerceptual;
    }
}

bool MetalFxContext::ensureSpatialScaler(MTL::Device* device,
                                         MTL::PixelFormat colorFormat,
                                         MTL::PixelFormat outputFormat,
                                         uint32_t inputWidth,
                                         uint32_t inputHeight,
                                         uint32_t outputWidth,
                                         uint32_t outputHeight,
                                         ColorMode colorMode,
                                         void* metal4Compiler)
{
    if (mSpatialScaler && colorFormat == mColorFormat && outputFormat == mOutputFormat &&
        inputWidth == mInputWidth && inputHeight == mInputHeight && outputWidth == mOutputWidth &&
        outputHeight == mOutputHeight && colorMode == mColorMode)
    {
        return true;
    }
    release();

    @autoreleasepool
    {
        id<MTLDevice> nativeDevice = (__bridge id<MTLDevice>)device;
        if (!nativeDevice || inputWidth == 0 || inputHeight == 0)
        {
            return false;
        }

        MTLFXSpatialScalerDescriptor* desc = [MTLFXSpatialScalerDescriptor new];
        desc.colorTextureFormat = (MTLPixelFormat)colorFormat;
        desc.outputTextureFormat = (MTLPixelFormat)outputFormat;
        desc.inputWidth = inputWidth;
        desc.inputHeight = inputHeight;
        desc.outputWidth = outputWidth;
        desc.outputHeight = outputHeight;
        desc.colorProcessingMode = toNative(colorMode);

        id<MTLFXSpatialScaler> scaler = [desc newSpatialScalerWithDevice:nativeDevice];
        if (!scaler)
        {
            STRELKA_ERROR("MetalFX spatial scaler unavailable for {}x{} -> {}x{}", inputWidth, inputHeight,
                          outputWidth, outputHeight);
            return false;
        }
        mSpatialScaler = (__bridge_retained void*)scaler;

        if (metal4Compiler)
        {
            id<MTL4FXSpatialScaler> scaler4 =
                [desc newSpatialScalerWithDevice:nativeDevice
                                        compiler:(__bridge id<MTL4Compiler>)metal4Compiler];
            if (scaler4)
            {
                mSpatialScaler4 = (__bridge_retained void*)scaler4;
            }
            else
            {
                STRELKA_WARNING("MetalFX Metal 4 spatial scaler unavailable; falling back to Metal 3");
            }
        }
    }

    mColorFormat = colorFormat;
    mOutputFormat = outputFormat;
    mInputWidth = inputWidth;
    mInputHeight = inputHeight;
    mOutputWidth = outputWidth;
    mOutputHeight = outputHeight;
    mColorMode = colorMode;

    STRELKA_INFO("MetalFX spatial scaler: {}x{} -> {}x{}", inputWidth, inputHeight, outputWidth, outputHeight);
    return true;
}

MTL::TextureUsage MetalFxContext::requiredColorUsage() const
{
    if (!mSpatialScaler)
    {
        return MTL::TextureUsageShaderRead;
    }
    id<MTLFXSpatialScaler> scaler = (__bridge id<MTLFXSpatialScaler>)mSpatialScaler;
    return (MTL::TextureUsage)scaler.colorTextureUsage;
}

MTL::TextureUsage MetalFxContext::requiredOutputUsage() const
{
    if (!mSpatialScaler)
    {
        return MTL::TextureUsageShaderWrite;
    }
    id<MTLFXSpatialScaler> scaler = (__bridge id<MTLFXSpatialScaler>)mSpatialScaler;
    return (MTL::TextureUsage)scaler.outputTextureUsage;
}

void MetalFxContext::encodeSpatial(void* commandBuffer,
                                   bool metal4,
                                   MTL::Texture* colorTexture,
                                   MTL::Texture* outputTexture,
                                   uint32_t inputContentWidth,
                                   uint32_t inputContentHeight)
{
    if (!mSpatialScaler || !commandBuffer || !colorTexture || !outputTexture)
    {
        return;
    }
    id<MTLFXSpatialScaler> scaler = (__bridge id<MTLFXSpatialScaler>)mSpatialScaler;
    scaler.colorTexture = (__bridge id<MTLTexture>)colorTexture;
    scaler.outputTexture = (__bridge id<MTLTexture>)outputTexture;
    // Content size can be smaller than the texture, which is how a dynamic
    // resolution scheme reuses one allocation; we always fill it.
    scaler.inputContentWidth = inputContentWidth;
    scaler.inputContentHeight = inputContentHeight;

    if (metal4)
    {
        // The Metal 4 protocol is a different type taking a different command
        // buffer, so it cannot share the encode call with the Metal 3 one.
        id<MTL4FXSpatialScaler> scaler4 = (__bridge id<MTL4FXSpatialScaler>)mSpatialScaler4;
        if (scaler4)
        {
            [scaler4 encodeToCommandBuffer:(__bridge id<MTL4CommandBuffer>)commandBuffer];
        }
        return;
    }
    [scaler encodeToCommandBuffer:(__bridge id<MTLCommandBuffer>)commandBuffer];
}


// Formats the guide textures are allocated with. Kept next to the descriptor so
// the two cannot drift: MetalFX validates them against the textures at encode
// time and a mismatch is an assertion, not a warning.
static constexpr MTLPixelFormat kColorFormat = MTLPixelFormatRGBA16Float;
static constexpr MTLPixelFormat kDepthFormat = MTLPixelFormatR32Float;
static constexpr MTLPixelFormat kMotionFormat = MTLPixelFormatRG16Float;
static constexpr MTLPixelFormat kAlbedoFormat = MTLPixelFormatRGBA16Float;
static constexpr MTLPixelFormat kNormalFormat = MTLPixelFormatRGBA16Float;
static constexpr MTLPixelFormat kRoughnessFormat = MTLPixelFormatR16Float;

bool MetalFxContext::denoiserSupportsMetal4(MTL::Device* device)
{
    return [MTLFXTemporalDenoisedScalerDescriptor supportsMetal4FX:(__bridge id<MTLDevice>)device];
}

bool MetalFxContext::ensureDenoiser(MTL::Device* device,
                                    uint32_t inputWidth,
                                    uint32_t inputHeight,
                                    uint32_t outputWidth,
                                    uint32_t outputHeight,
                                    void* metal4Compiler)
{
    if (mDenoiser && inputWidth == mDenoiseInputWidth && inputHeight == mDenoiseInputHeight &&
        outputWidth == mDenoiseOutputWidth && outputHeight == mDenoiseOutputHeight)
    {
        return true;
    }
    if (mDenoiser)
    {
        CFRelease(mDenoiser);
        mDenoiser = nullptr;
    }
    if (mDenoiser4)
    {
        CFRelease(mDenoiser4);
        mDenoiser4 = nullptr;
    }

    @autoreleasepool
    {
        id<MTLDevice> nativeDevice = (__bridge id<MTLDevice>)device;
        if (!nativeDevice || inputWidth == 0 || inputHeight == 0)
        {
            return false;
        }

        MTLFXTemporalDenoisedScalerDescriptor* desc = [MTLFXTemporalDenoisedScalerDescriptor new];
        desc.colorTextureFormat = kColorFormat;
        desc.depthTextureFormat = kDepthFormat;
        desc.motionTextureFormat = kMotionFormat;
        desc.diffuseAlbedoTextureFormat = kAlbedoFormat;
        desc.specularAlbedoTextureFormat = kAlbedoFormat;
        desc.normalTextureFormat = kNormalFormat;
        desc.roughnessTextureFormat = kRoughnessFormat;
        desc.outputTextureFormat = kColorFormat;
        desc.inputWidth = inputWidth;
        desc.inputHeight = inputHeight;
        desc.outputWidth = outputWidth;
        desc.outputHeight = outputHeight;
        // The tracer has no exposure texture to offer and its radiance is already
        // absolute, so let MetalFX work the exposure out itself.
        desc.autoExposureEnabled = YES;
        // Block until the graph is built. Asynchronous initialisation returns a
        // scaler whose network is still being assembled, and encoding into it
        // asserts inside MPSGraph rather than failing the creation call.
        desc.requiresSynchronousInitialization = YES;

        id<MTLFXTemporalDenoisedScaler> denoiser = [desc newTemporalDenoisedScalerWithDevice:nativeDevice];
        if (!denoiser)
        {
            STRELKA_ERROR("MetalFX temporal denoiser unavailable for {}x{} -> {}x{}", inputWidth, inputHeight,
                          outputWidth, outputHeight);
            return false;
        }
        mDenoiser = (__bridge_retained void*)denoiser;

        if (metal4Compiler)
        {
            id<MTL4FXTemporalDenoisedScaler> denoiser4 =
                [desc newTemporalDenoisedScalerWithDevice:nativeDevice
                                                 compiler:(__bridge id<MTL4Compiler>)metal4Compiler];
            if (denoiser4)
            {
                mDenoiser4 = (__bridge_retained void*)denoiser4;
            }
        }
    }

    mDenoiseInputWidth = inputWidth;
    mDenoiseInputHeight = inputHeight;
    mDenoiseOutputWidth = outputWidth;
    mDenoiseOutputHeight = outputHeight;
    STRELKA_INFO("MetalFX temporal denoiser: {}x{} -> {}x{}", inputWidth, inputHeight, outputWidth, outputHeight);
    return true;
}

MTL::TextureUsage MetalFxContext::denoiseColorUsage() const
{
    if (!mDenoiser) return MTL::TextureUsageShaderWrite;
    return (MTL::TextureUsage)((__bridge id<MTLFXTemporalDenoisedScaler>)mDenoiser).colorTextureUsage;
}

MTL::TextureUsage MetalFxContext::denoiseGuideUsage() const
{
    if (!mDenoiser) return MTL::TextureUsageShaderWrite;
    id<MTLFXTemporalDenoisedScaler> d = (__bridge id<MTLFXTemporalDenoisedScaler>)mDenoiser;
    // One flag set for every guide: they are all read the same way and taking the
    // union costs nothing.
    return (MTL::TextureUsage)(d.depthTextureUsage | d.motionTextureUsage | d.normalTextureUsage |
                               d.roughnessTextureUsage | d.diffuseAlbedoTextureUsage |
                               d.specularAlbedoTextureUsage);
}

MTL::TextureUsage MetalFxContext::denoiseOutputUsage() const
{
    if (!mDenoiser) return MTL::TextureUsageShaderRead;
    return (MTL::TextureUsage)((__bridge id<MTLFXTemporalDenoisedScaler>)mDenoiser).outputTextureUsage;
}

void MetalFxContext::encodeDenoise(void* commandBuffer, bool metal4, const DenoiseInputs& inputs)
{
    if (!mDenoiser || !commandBuffer || !inputs.color || !inputs.output)
    {
        return;
    }
    id<MTLFXTemporalDenoisedScaler> d = (__bridge id<MTLFXTemporalDenoisedScaler>)mDenoiser;
    d.colorTexture = (__bridge id<MTLTexture>)inputs.color;
    d.depthTexture = (__bridge id<MTLTexture>)inputs.depth;
    d.motionTexture = (__bridge id<MTLTexture>)inputs.motion;
    d.diffuseAlbedoTexture = (__bridge id<MTLTexture>)inputs.diffuseAlbedo;
    d.specularAlbedoTexture = (__bridge id<MTLTexture>)inputs.specularAlbedo;
    d.normalTexture = (__bridge id<MTLTexture>)inputs.normal;
    d.roughnessTexture = (__bridge id<MTLTexture>)inputs.roughness;
    d.outputTexture = (__bridge id<MTLTexture>)inputs.output;
    d.jitterOffsetX = inputs.jitterX;
    d.jitterOffsetY = inputs.jitterY;
    // Our motion vectors are already in pixels and point from the current frame
    // back to the previous one, which is the sign MetalFX expects.
    d.motionVectorScaleX = 1.0f;
    d.motionVectorScaleY = 1.0f;
    // Distance to the camera, growing away from it.
    d.depthReversed = NO;
    d.shouldResetHistory = inputs.resetHistory ? YES : NO;
    // Reprojection uses the camera directly, not just the motion vectors, which
    // is what lets it tell a moving camera from moving geometry.
    simd_float4x4 w2v, v2c;
    memcpy(&w2v, inputs.worldToView, sizeof(w2v));
    memcpy(&v2c, inputs.viewToClip, sizeof(v2c));
    d.worldToViewMatrix = w2v;
    d.viewToClipMatrix = v2c;

    if (metal4)
    {
        id<MTL4FXTemporalDenoisedScaler> d4 = (__bridge id<MTL4FXTemporalDenoisedScaler>)mDenoiser4;
        if (d4)
        {
            [d4 encodeToCommandBuffer:(__bridge id<MTL4CommandBuffer>)commandBuffer];
        }
        return;
    }
    [d encodeToCommandBuffer:(__bridge id<MTLCommandBuffer>)commandBuffer];
}

void MetalFxContext::release()
{
    if (mSpatialScaler)
    {
        CFRelease(mSpatialScaler);
        mSpatialScaler = nullptr;
    }
    if (mSpatialScaler4)
    {
        CFRelease(mSpatialScaler4);
        mSpatialScaler4 = nullptr;
    }
    if (mDenoiser)
    {
        CFRelease(mDenoiser);
        mDenoiser = nullptr;
    }
    if (mDenoiser4)
    {
        CFRelease(mDenoiser4);
        mDenoiser4 = nullptr;
    }
    mDenoiseInputWidth = mDenoiseInputHeight = mDenoiseOutputWidth = mDenoiseOutputHeight = 0;
    mColorFormat = MTL::PixelFormatInvalid;
    mOutputFormat = MTL::PixelFormatInvalid;
    mInputWidth = mInputHeight = mOutputWidth = mOutputHeight = 0;
}

} // namespace oka
