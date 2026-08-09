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
        mSpatialScaler = (void*)scaler;

        if (metal4Compiler)
        {
            id<MTL4FXSpatialScaler> scaler4 =
                [desc newSpatialScalerWithDevice:nativeDevice
                                        compiler:(__bridge id<MTL4Compiler>)metal4Compiler];
            if (scaler4)
            {
                mSpatialScaler4 = (void*)scaler4;
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
    if (metal4)
    {
        // The Metal 4 protocol is a different type taking a different command
        // buffer, so it cannot share the encode call with the Metal 3 one -- and,
        // less obviously, it is also a different *object*. Setting the textures on
        // the Metal 3 scaler and then encoding the Metal 4 one ran the scaler with
        // no input and no output attached, which is not an error: it simply never
        // wrote the display texture, and the viewport went black the moment
        // upscaling was switched on.
        id<MTL4FXSpatialScaler> scaler4 = (__bridge id<MTL4FXSpatialScaler>)mSpatialScaler4;
        if (scaler4)
        {
            scaler4.colorTexture = (__bridge id<MTLTexture>)colorTexture;
            scaler4.outputTexture = (__bridge id<MTLTexture>)outputTexture;
            scaler4.inputContentWidth = inputContentWidth;
            scaler4.inputContentHeight = inputContentHeight;
            [scaler4 encodeToCommandBuffer:(__bridge id<MTL4CommandBuffer>)commandBuffer];
        }
        return;
    }

    id<MTLFXSpatialScaler> scaler = (__bridge id<MTLFXSpatialScaler>)mSpatialScaler;
    scaler.colorTexture = (__bridge id<MTLTexture>)colorTexture;
    scaler.outputTexture = (__bridge id<MTLTexture>)outputTexture;
    // Content size can be smaller than the texture, which is how a dynamic
    // resolution scheme reuses one allocation; we always fill it.
    scaler.inputContentWidth = inputContentWidth;
    scaler.inputContentHeight = inputContentHeight;
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
static constexpr MTLPixelFormat kSpecularHitDistanceFormat = MTLPixelFormatR16Float;
static constexpr MTLPixelFormat kReactiveFormat = MTLPixelFormatR8Unorm;

void MetalFxContext::denoiserScaleRange(MTL::Device* device, float& minScale, float& maxScale)
{
    id<MTLDevice> native = (__bridge id<MTLDevice>)device;
    minScale = [MTLFXTemporalDenoisedScalerDescriptor supportedInputContentMinScaleForDevice:native];
    maxScale = [MTLFXTemporalDenoisedScalerDescriptor supportedInputContentMaxScaleForDevice:native];
}

bool MetalFxContext::denoiserSupportsMetal4(MTL::Device* device)
{
    return [MTLFXTemporalDenoisedScalerDescriptor supportsMetal4FX:(__bridge id<MTLDevice>)device];
}


bool MetalFxContext::ensureTemporalScaler(MTL::Device* device,
                                          MTL::PixelFormat colorFormat,
                                          MTL::PixelFormat depthFormat,
                                          MTL::PixelFormat motionFormat,
                                          MTL::PixelFormat outputFormat,
                                          uint32_t inputWidth,
                                          uint32_t inputHeight,
                                          uint32_t outputWidth,
                                          uint32_t outputHeight,
                                          void* metal4Compiler)
{
    if (mTemporalScaler && inputWidth == mTemporalInputWidth && inputHeight == mTemporalInputHeight &&
        outputWidth == mTemporalOutputWidth && outputHeight == mTemporalOutputHeight)
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

        MTLFXTemporalScalerDescriptor* desc = [MTLFXTemporalScalerDescriptor new];
        desc.colorTextureFormat = (MTLPixelFormat)colorFormat;
        desc.depthTextureFormat = (MTLPixelFormat)depthFormat;
        desc.motionTextureFormat = (MTLPixelFormat)motionFormat;
        desc.outputTextureFormat = (MTLPixelFormat)outputFormat;
        desc.inputWidth = inputWidth;
        desc.inputHeight = inputHeight;
        desc.outputWidth = outputWidth;
        desc.outputHeight = outputHeight;
        // Off, and it matters.
        //
        // The colour handed over is linear radiance and the tone curve runs after
        // the scaler, so with auto exposure on there are two normalisations in the
        // loop: MetalFX rescales luminance from its own per-frame estimate, and
        // the tonemapper rescales again by the scene's exposure. The scaler's
        // estimate moves whenever the frame's content does, so the history it
        // blends against was normalised differently from the frame being added --
        // which is the one state a temporal filter cannot converge out of, and it
        // reads as noise that never settles.
        //
        // This is what RRS does, where the same MetalFX denoiser converges well:
        // it turns auto exposure off precisely so the scaler's normalisation does
        // not fight the application's own exposure. STRELKA_MFX_AUTOEXPOSURE=1
        // puts it back for comparison.
        desc.autoExposureEnabled = getenv("STRELKA_MFX_AUTOEXPOSURE") ? YES : NO;
        desc.requiresSynchronousInitialization = YES;

        id<MTLFXTemporalScaler> scaler = [desc newTemporalScalerWithDevice:nativeDevice];
        if (!scaler)
        {
            STRELKA_ERROR("MetalFX temporal scaler unavailable for {}x{} -> {}x{}", inputWidth, inputHeight,
                          outputWidth, outputHeight);
            return false;
        }
        mTemporalScaler = (void*)scaler;

        if (metal4Compiler)
        {
            id<MTL4FXTemporalScaler> scaler4 =
                [desc newTemporalScalerWithDevice:nativeDevice
                                         compiler:(__bridge id<MTL4Compiler>)metal4Compiler];
            if (scaler4)
            {
                mTemporalScaler4 = (void*)scaler4;
            }
            else
            {
                STRELKA_WARNING("MetalFX Metal 4 temporal scaler unavailable; falling back to Metal 3");
            }
        }
    }

    mTemporalInputWidth = inputWidth;
    mTemporalInputHeight = inputHeight;
    mTemporalOutputWidth = outputWidth;
    mTemporalOutputHeight = outputHeight;
    STRELKA_INFO("MetalFX temporal scaler: {}x{} -> {}x{}{}", inputWidth, inputHeight, outputWidth, outputHeight,
                 mTemporalScaler4 ? " (metal4)" : "");
    return true;
}

MTL::TextureUsage MetalFxContext::temporalColorUsage() const
{
    if (!mTemporalScaler)
        return MTL::TextureUsageShaderRead;
    return (MTL::TextureUsage)((__bridge id<MTLFXTemporalScaler>)mTemporalScaler).colorTextureUsage;
}

MTL::TextureUsage MetalFxContext::temporalDepthUsage() const
{
    if (!mTemporalScaler)
        return MTL::TextureUsageShaderRead;
    return (MTL::TextureUsage)((__bridge id<MTLFXTemporalScaler>)mTemporalScaler).depthTextureUsage;
}

MTL::TextureUsage MetalFxContext::temporalMotionUsage() const
{
    if (!mTemporalScaler)
        return MTL::TextureUsageShaderRead;
    return (MTL::TextureUsage)((__bridge id<MTLFXTemporalScaler>)mTemporalScaler).motionTextureUsage;
}

MTL::TextureUsage MetalFxContext::temporalOutputUsage() const
{
    if (!mTemporalScaler)
        return MTL::TextureUsageShaderWrite;
    return (MTL::TextureUsage)((__bridge id<MTLFXTemporalScaler>)mTemporalScaler).outputTextureUsage;
}

void MetalFxContext::encodeTemporal(void* commandBuffer, bool metal4, const TemporalInputs& inputs)
{
    if (!commandBuffer || !inputs.color || !inputs.output)
    {
        return;
    }
    if (metal4 && mTemporalScaler4)
    {
        id<MTL4FXTemporalScaler> t = (__bridge id<MTL4FXTemporalScaler>)mTemporalScaler4;
        t.colorTexture = (__bridge id<MTLTexture>)inputs.color;
        t.depthTexture = (__bridge id<MTLTexture>)inputs.depth;
        t.motionTexture = (__bridge id<MTLTexture>)inputs.motion;
        t.outputTexture = (__bridge id<MTLTexture>)inputs.output;
        t.jitterOffsetX = inputs.jitterX;
        t.jitterOffsetY = inputs.jitterY;
        t.motionVectorScaleX = 1.0f;
        t.motionVectorScaleY = 1.0f;
        t.depthReversed = inputs.depthReversed ? YES : NO;
        t.reset = inputs.resetHistory ? YES : NO;
        [t encodeToCommandBuffer:(__bridge id<MTL4CommandBuffer>)commandBuffer];
        return;
    }
    if (!mTemporalScaler)
    {
        return;
    }
    id<MTLFXTemporalScaler> t = (__bridge id<MTLFXTemporalScaler>)mTemporalScaler;
    t.colorTexture = (__bridge id<MTLTexture>)inputs.color;
    t.depthTexture = (__bridge id<MTLTexture>)inputs.depth;
    t.motionTexture = (__bridge id<MTLTexture>)inputs.motion;
    t.outputTexture = (__bridge id<MTLTexture>)inputs.output;
    t.jitterOffsetX = inputs.jitterX;
    t.jitterOffsetY = inputs.jitterY;
    t.motionVectorScaleX = 1.0f;
    t.motionVectorScaleY = 1.0f;
    t.depthReversed = inputs.depthReversed ? YES : NO;
    t.reset = inputs.resetHistory ? YES : NO;
    [t encodeToCommandBuffer:(__bridge id<MTLCommandBuffer>)commandBuffer];
}

bool MetalFxContext::ensureDenoiser(MTL::Device* device,
                                    uint32_t inputWidth,
                                    uint32_t inputHeight,
                                    uint32_t outputWidth,
                                    uint32_t outputHeight)
{
    if (mDenoiser && inputWidth == mDenoiseInputWidth && inputHeight == mDenoiseInputHeight &&
        outputWidth == mDenoiseOutputWidth && outputHeight == mDenoiseOutputHeight)
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

        MTLFXTemporalDenoisedScalerDescriptor* desc = [MTLFXTemporalDenoisedScalerDescriptor new];
        desc.colorTextureFormat = kColorFormat;
        desc.depthTextureFormat = kDepthFormat;
        desc.motionTextureFormat = kMotionFormat;
        desc.diffuseAlbedoTextureFormat = kAlbedoFormat;
        desc.specularAlbedoTextureFormat = kAlbedoFormat;
        desc.normalTextureFormat = kNormalFormat;
        desc.roughnessTextureFormat = kRoughnessFormat;
        desc.outputTextureFormat = kColorFormat;
        // Two inputs aimed at exactly what a path tracer gets wrong: a reflection
        // does not sit on the surface that reflects it, and a motion vector on a
        // mirror describes the mirror rather than the image in it.
        desc.specularHitDistanceTextureEnabled = getenv("STRELKA_NO_SPECDIST") ? NO : YES;
        desc.specularHitDistanceTextureFormat = kSpecularHitDistanceFormat;
        // On, and the aggregate metrics argue against it. They are wrong, and how
        // they are wrong is worth keeping.
        //
        // The mask marks pixels whose history cannot be trusted. The rule that
        // produces it fires when the guides had to be taken past the primary hit,
        // which happens exactly when that hit was too smooth to describe -- water,
        // glass, a mirror. On the pine forest that is 22.6% of the frame, and it
        // is water.
        //
        // Turning it off improves every number the denoise audit reports: swim
        // 0.0793 -> 0.0301, sharpness 0.368 -> 0.441, pixels below a tenth of the
        // truth 0.7% -> 0.4%. It also makes the water surface vanish. The audit
        // averages over the frame, and a fifth of it getting worse in a way that
        // matters is worth less to a mean than four fifths getting slightly
        // better -- so the summary improved while the picture lost an object.
        //
        // STRELKA_NO_REACTIVE=1 turns it off, which is the configuration those
        // numbers describe.
        desc.reactiveMaskTextureEnabled = getenv("STRELKA_NO_REACTIVE") ? NO : YES;
        desc.reactiveMaskTextureFormat = kReactiveFormat;
        desc.inputWidth = inputWidth;
        desc.inputHeight = inputHeight;
        desc.outputWidth = outputWidth;
        desc.outputHeight = outputHeight;
        // Off, and it matters.
        //
        // The colour handed over is linear radiance and the tone curve runs after
        // the scaler, so with auto exposure on there are two normalisations in the
        // loop: MetalFX rescales luminance from its own per-frame estimate, and
        // the tonemapper rescales again by the scene's exposure. The scaler's
        // estimate moves whenever the frame's content does, so the history it
        // blends against was normalised differently from the frame being added --
        // which is the one state a temporal filter cannot converge out of, and it
        // reads as noise that never settles.
        //
        // This is what RRS does, where the same MetalFX denoiser converges well:
        // it turns auto exposure off precisely so the scaler's normalisation does
        // not fight the application's own exposure. STRELKA_MFX_AUTOEXPOSURE=1
        // puts it back for comparison.
        desc.autoExposureEnabled = getenv("STRELKA_MFX_AUTOEXPOSURE") ? YES : NO;
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
        mDenoiser = (void*)denoiser;
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
                               d.specularAlbedoTextureUsage | d.specularHitDistanceTextureUsage |
                               d.reactiveTextureUsage);
}

MTL::TextureUsage MetalFxContext::denoiseOutputUsage() const
{
    if (!mDenoiser) return MTL::TextureUsageShaderRead;
    return (MTL::TextureUsage)((__bridge id<MTLFXTemporalDenoisedScaler>)mDenoiser).outputTextureUsage;
}

void MetalFxContext::encodeDenoise(void* commandBuffer, const DenoiseInputs& inputs)
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
    d.specularHitDistanceTexture = (__bridge id<MTLTexture>)inputs.specularHitDistance;
    d.reactiveMaskTexture = (__bridge id<MTLTexture>)inputs.reactive;
    d.outputTexture = (__bridge id<MTLTexture>)inputs.output;
    d.jitterOffsetX = inputs.jitterX;
    d.jitterOffsetY = inputs.jitterY;
    // Our motion vectors are already in pixels and point from the current frame
    // back to the previous one, which is the sign MetalFX expects.
    d.motionVectorScaleX = 1.0f;
    d.motionVectorScaleY = 1.0f;
    d.depthReversed = inputs.depthReversed ? YES : NO;
    d.shouldResetHistory = inputs.resetHistory ? YES : NO;
    // Reprojection uses the camera directly, not just the motion vectors, which
    // is what lets it tell a moving camera from moving geometry.
    simd_float4x4 w2v, v2c;
    memcpy(&w2v, inputs.worldToView, sizeof(w2v));
    memcpy(&v2c, inputs.viewToClip, sizeof(v2c));
    d.worldToViewMatrix = w2v;
    d.viewToClipMatrix = v2c;

    [d encodeToCommandBuffer:(__bridge id<MTLCommandBuffer>)commandBuffer];
}

void MetalFxContext::release()
{
    if (mTemporalScaler4)
    {
        CFRelease(mTemporalScaler4);
        mTemporalScaler4 = nullptr;
    }
    if (mTemporalScaler)
    {
        CFRelease(mTemporalScaler);
        mTemporalScaler = nullptr;
    }
    mTemporalInputWidth = mTemporalInputHeight = mTemporalOutputWidth = mTemporalOutputHeight = 0;
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
    mDenoiseInputWidth = mDenoiseInputHeight = mDenoiseOutputWidth = mDenoiseOutputHeight = 0;
    mColorFormat = MTL::PixelFormatInvalid;
    mOutputFormat = MTL::PixelFormatInvalid;
    mInputWidth = mInputHeight = mOutputWidth = mOutputHeight = 0;
}

} // namespace oka
