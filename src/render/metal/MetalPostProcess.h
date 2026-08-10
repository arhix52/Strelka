#pragma once

#include "Metal4Context.h"
#include "MetalFxContext.h"

#include <Metal/Metal.hpp>
#include <strelka/render/render.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace oka
{
namespace metal
{

// Tonemap, guide/display/upscale textures, and MetalFxContext usage.
class MetalPostProcess
{
public:
    struct GuideTextures
    {
        MTL::Texture* color = nullptr;
        MTL::Texture* depth = nullptr;
        MTL::Texture* motion = nullptr;
        MTL::Texture* diffuse = nullptr;
        MTL::Texture* specular = nullptr;
        MTL::Texture* normal = nullptr;
        MTL::Texture* roughness = nullptr;
        MTL::Texture* specularHitDistance = nullptr;
        MTL::Texture* reactive = nullptr;
    };

    // Side effects the orchestrator owns (async display index / residency gen /
    // denoise history). PostProcess does not own those members.
    struct HostHooks
    {
        std::atomic<int>* readyIndex = nullptr;
        uint32_t* metal4ResidencyGeneration = nullptr;
        bool* resetDenoiseHistory = nullptr;
        int* writeIndex = nullptr;
    };

    MetalPostProcess() = default;
    ~MetalPostProcess();

    void init(MTL::Device* device, Metal4Context* metal4, HostHooks hooks);
    void release();

    void buildTonemapperPipeline();

    void releaseGuideTextures();
    void ensureGuideTextures(uint32_t width, uint32_t height, uint32_t outWidth, uint32_t outHeight);
    void ensureUpscaleTextures(uint32_t width, uint32_t height);
    void ensureDisplayTextures(uint32_t width, uint32_t height);
    MTL::Texture* tonemapTarget(bool upscaling) const;

    MetalFxContext& metalFx()
    {
        return mMetalFx;
    }
    const MetalFxContext& metalFx() const
    {
        return mMetalFx;
    }

    GuideTextures& guides()
    {
        return mGuides;
    }
    const GuideTextures& guides() const
    {
        return mGuides;
    }
    MTL::Texture* denoisedTexture() const
    {
        return mDenoisedTexture;
    }
    MTL::Texture* displayTexture(int index) const
    {
        return mDisplayTextures[index];
    }
    MTL::Texture* upscaleTexture(int index) const
    {
        return mUpscaleTextures[index];
    }
    MTL::Texture* const* displayTextures() const
    {
        return mDisplayTextures;
    }
    MTL::Texture* const* upscaleTextures() const
    {
        return mUpscaleTextures;
    }

    MTL::ComputePipelineState* tonemapperPSO() const
    {
        return mTonemapperPSO;
    }
    MTL::ComputePipelineState* tonemapperPSO4() const
    {
        return mTonemapperPSO4;
    }
    MTL::ComputePipelineState* tonemapperTexPSO() const
    {
        return mTonemapperTexPSO;
    }
    MTL::ComputePipelineState* denoisedToBufferPSO() const
    {
        return mDenoisedToBufferPSO;
    }

    bool& loggedUpscaleClamp()
    {
        return mLoggedUpscaleClamp;
    }
    bool& loggedMetal4DenoiserGap()
    {
        return mLoggedMetal4DenoiserGap;
    }
    bool& loggedShaderValidationDenoiserGap()
    {
        return mLoggedShaderValidationDenoiserGap;
    }
    bool& prevDenoiseEnabled()
    {
        return mPrevDenoiseEnabled;
    }

private:
    void markTexturesDirty(bool invalidateReadyIndex);

    MTL::Device* mDevice = nullptr;
    Metal4Context* mMetal4 = nullptr;
    HostHooks mHooks;

    MTL::ComputePipelineState* mTonemapperPSO = nullptr;
    MTL::ComputePipelineState* mTonemapperPSO4 = nullptr;
    MTL::ComputePipelineState* mTonemapperTexPSO = nullptr;
    MTL::ComputePipelineState* mDenoisedToBufferPSO = nullptr;

    MTL::Texture* mDisplayTextures[2] = { nullptr, nullptr };
    uint32_t mDisplayTextureWidth = 0;
    uint32_t mDisplayTextureHeight = 0;
    MTL::TextureUsage mDisplayTextureUsage = 0;

    MTL::Texture* mUpscaleTextures[2] = { nullptr, nullptr };
    uint32_t mUpscaleTextureWidth = 0;
    uint32_t mUpscaleTextureHeight = 0;

    MetalFxContext mMetalFx;
    GuideTextures mGuides;
    MTL::Texture* mDenoisedTexture = nullptr;
    uint32_t mGuideWidth = 0;
    uint32_t mGuideHeight = 0;
    uint32_t mGuideOutWidth = 0;
    uint32_t mGuideOutHeight = 0;

    bool mLoggedUpscaleClamp = false;
    bool mLoggedMetal4DenoiserGap = false;
    bool mLoggedShaderValidationDenoiserGap = false;
    bool mPrevDenoiseEnabled = false;
};

} // namespace metal
} // namespace oka
