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
    mColorFormat = MTL::PixelFormatInvalid;
    mOutputFormat = MTL::PixelFormatInvalid;
    mInputWidth = mInputHeight = mOutputWidth = mOutputHeight = 0;
}

} // namespace oka
