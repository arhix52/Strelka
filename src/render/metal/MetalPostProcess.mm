#include "MetalPostProcess.h"

#include <log.h>
#include <paths.h>

#include <cassert>
#include <cstring>
#include <string>

namespace oka
{
namespace metal
{

MetalPostProcess::~MetalPostProcess()
{
    release();
}

void MetalPostProcess::init(MTL::Device* device, Metal4Context* metal4, HostHooks hooks)
{
    mDevice = device;
    mMetal4 = metal4;
    mHooks = hooks;
}

void MetalPostProcess::markTexturesDirty(bool invalidateReadyIndex)
{
    if (mHooks.metal4ResidencyGeneration)
    {
        *mHooks.metal4ResidencyGeneration = 0;
    }
    if (invalidateReadyIndex && mHooks.readyIndex)
    {
        mHooks.readyIndex->store(-1);
    }
}

void MetalPostProcess::release()
{
    auto safeRelease = [](auto*& p) {
        if (p)
        {
            p->release();
            p = nullptr;
        }
    };
    releaseGuideTextures();
    for (auto*& tex : mDisplayTextures)
        safeRelease(tex);
    for (auto*& tex : mUpscaleTextures)
        safeRelease(tex);
    mMetalFx.release();
    safeRelease(mTonemapperPSO);
    safeRelease(mTonemapperPSO4);
    safeRelease(mTonemapperTexPSO);
    safeRelease(mDenoisedToBufferPSO);
    mDisplayTextureWidth = mDisplayTextureHeight = 0;
    mDisplayTextureUsage = 0;
    mUpscaleTextureWidth = mUpscaleTextureHeight = 0;
}

void MetalPostProcess::releaseGuideTextures()
{
    auto release = [](MTL::Texture*& texture) {
        if (texture)
        {
            texture->release();
            texture = nullptr;
        }
    };
    release(mGuides.color);
    release(mGuides.depth);
    release(mGuides.motion);
    release(mGuides.diffuse);
    release(mGuides.specular);
    release(mGuides.normal);
    release(mGuides.roughness);
    release(mGuides.specularHitDistance);
    release(mGuides.reactive);
    release(mDenoisedTexture);
    mGuideWidth = 0;
    mGuideHeight = 0;
    mGuideOutWidth = 0;
    mGuideOutHeight = 0;
}

void MetalPostProcess::ensureGuideTextures(uint32_t width, uint32_t height, uint32_t outWidth, uint32_t outHeight)
{
    if (width == mGuideWidth && height == mGuideHeight && outWidth == mGuideOutWidth &&
        outHeight == mGuideOutHeight && mGuides.color && mDenoisedTexture)
    {
        return;
    }
    releaseGuideTextures();

    const bool temporalOnly = !mMetalFx.hasDenoiser() && mMetalFx.hasTemporalScaler();
    const MTL::TextureUsage guideUsage =
        MTL::TextureUsageShaderWrite |
        (temporalOnly ? (mMetalFx.temporalDepthUsage() | mMetalFx.temporalMotionUsage())
                      : mMetalFx.denoiseGuideUsage());
    auto make = [&](MTL::PixelFormat fmt, uint32_t w, uint32_t h, MTL::TextureUsage usage) {
        MTL::TextureDescriptor* d = MTL::TextureDescriptor::alloc()->init();
        d->setWidth(w);
        d->setHeight(h);
        d->setPixelFormat(fmt);
        d->setTextureType(MTL::TextureType2D);
        d->setStorageMode(MTL::StorageModePrivate);
        d->setUsage(usage);
        MTL::Texture* t = mDevice->newTexture(d);
        d->release();
        return t;
    };
    mGuides.color = make(MTL::PixelFormatRGBA16Float, width, height,
                         MTL::TextureUsageShaderWrite |
                             (temporalOnly ? mMetalFx.temporalColorUsage() : mMetalFx.denoiseColorUsage()));
    mGuides.depth = make(MTL::PixelFormatR32Float, width, height, guideUsage);
    mGuides.motion = make(MTL::PixelFormatRG16Float, width, height, guideUsage);
    mGuides.diffuse = make(MTL::PixelFormatRGBA16Float, width, height, guideUsage);
    mGuides.specular = make(MTL::PixelFormatRGBA16Float, width, height, guideUsage);
    mGuides.normal = make(MTL::PixelFormatRGBA16Float, width, height, guideUsage);
    mGuides.roughness = make(MTL::PixelFormatR16Float, width, height, guideUsage);
    mGuides.specularHitDistance = make(MTL::PixelFormatR16Float, width, height, guideUsage);
    mGuides.reactive = make(MTL::PixelFormatR8Unorm, width, height, guideUsage);
    mDenoisedTexture = make(MTL::PixelFormatRGBA16Float, outWidth, outHeight,
                            MTL::TextureUsageShaderRead |
                                (temporalOnly ? mMetalFx.temporalOutputUsage()
                                              : mMetalFx.denoiseOutputUsage()));

    mGuideWidth = width;
    mGuideHeight = height;
    mGuideOutWidth = outWidth;
    mGuideOutHeight = outHeight;
    markTexturesDirty(false);
    if (mHooks.resetDenoiseHistory)
    {
        *mHooks.resetDenoiseHistory = true;
    }
}

void MetalPostProcess::ensureUpscaleTextures(uint32_t width, uint32_t height)
{
    if (width == mUpscaleTextureWidth && height == mUpscaleTextureHeight && mUpscaleTextures[0])
    {
        return;
    }
    for (MTL::Texture*& tex : mUpscaleTextures)
    {
        if (tex)
        {
            tex->release();
            tex = nullptr;
        }
    }

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setPixelFormat(MTL::PixelFormatRGBA16Float);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setUsage(MTL::TextureUsageShaderWrite | mMetalFx.requiredColorUsage());
    for (MTL::Texture*& tex : mUpscaleTextures)
    {
        tex = mDevice->newTexture(desc);
    }
    desc->release();

    mUpscaleTextureWidth = width;
    mUpscaleTextureHeight = height;
    markTexturesDirty(true);
}

MTL::Texture* MetalPostProcess::tonemapTarget(bool upscaling) const
{
    const int wi = mHooks.writeIndex ? *mHooks.writeIndex : 0;
    return upscaling ? mUpscaleTextures[wi] : mDisplayTextures[wi];
}

void MetalPostProcess::ensureDisplayTextures(uint32_t width, uint32_t height)
{
    const MTL::TextureUsage usage =
        MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite | mMetalFx.requiredOutputUsage();
    if (width == mDisplayTextureWidth && height == mDisplayTextureHeight && mDisplayTextures[0] &&
        usage == mDisplayTextureUsage)
    {
        return;
    }
    mDisplayTextureUsage = usage;
    for (MTL::Texture*& tex : mDisplayTextures)
    {
        if (tex)
        {
            tex->release();
            tex = nullptr;
        }
    }

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setPixelFormat(MTL::PixelFormatRGBA16Float);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setUsage(usage);
    for (MTL::Texture*& tex : mDisplayTextures)
    {
        tex = mDevice->newTexture(desc);
    }
    desc->release();

    mDisplayTextureWidth = width;
    mDisplayTextureHeight = height;
    markTexturesDirty(true);
}

void MetalPostProcess::buildTonemapperPipeline()
{
    const std::string path = oka::resolveResourcePath("metal/shaders/tonemapper.metallib");
    NS::Error* loadErr = nullptr;
    MTL::Library* pComputeLibrary =
        mDevice->newLibrary(NS::String::string(path.c_str(), NS::UTF8StringEncoding), &loadErr);
    if (!pComputeLibrary)
    {
        STRELKA_FATAL("Failed to load {}: {}", path,
                      loadErr ? loadErr->localizedDescription()->utf8String() : "unknown error");
        return;
    }
    NS::Error* pError = nullptr;
    MTL::Function* pTonemapperFn =
        pComputeLibrary->newFunction(NS::String::string("toneMappingComputeShader", NS::UTF8StringEncoding));
    mTonemapperPSO = mDevice->newComputePipelineState(pTonemapperFn, &pError);
    {
        NS::Error* e2 = nullptr;
        MTL::Function* fn = pComputeLibrary->newFunction(
            NS::String::string("toneMappingTextureShader", NS::UTF8StringEncoding));
        mTonemapperTexPSO = fn ? mDevice->newComputePipelineState(fn, &e2) : nullptr;
        if (fn)
            fn->release();
    }
    {
        NS::Error* e3 = nullptr;
        MTL::Function* fn = pComputeLibrary->newFunction(
            NS::String::string("denoisedTextureToBuffer", NS::UTF8StringEncoding));
        mDenoisedToBufferPSO = fn ? mDevice->newComputePipelineState(fn, &e3) : nullptr;
        if (fn)
            fn->release();
    }
    if (mMetal4 && mMetal4->isValid())
    {
        mTonemapperPSO4 = mMetal4->newComputePipelineState(pComputeLibrary, "toneMappingComputeShader", nullptr);
    }
    if (!mTonemapperPSO)
    {
        STRELKA_FATAL("{}", pError ? pError->localizedDescription()->utf8String() : "unknown error");
        assert(false);
    }

    pTonemapperFn->release();
    pComputeLibrary->release();
}

} // namespace metal
} // namespace oka
