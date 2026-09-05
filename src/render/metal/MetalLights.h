#pragma once

#include "MetalTextures.h"

#include <Metal/Metal.hpp>
#include <strelka/scene/scene.h>

#include <string>
#include <vector>


namespace oka::metal
{

// Analytic light domain: Scene::Light → UniformLight GPU buffer, plus the packed
// IES candela tables those lights may index and the images projector lights
// throw. Does not own environment / IBL (see MetalEnvironment).
class MetalLights
{
public:
    struct AuditCounts
    {
        uint64_t uploads = 0;
        uint64_t temporalMappingUpdates = 0;
    };
    MetalLights() = default;
    ~MetalLights();

    void init(MTL::Device* device);
    void release();

    /// Rebuild the GPU light table.
    ///
    /// The projector images come in as paths and go out as bindless handles
    /// inside the lights themselves, which is why this takes the texture domain:
    /// the handle cannot be resolved before the image exists, and the shade
    /// kernel has no binding slot left for a table of its own.
    void upload(const std::vector<Scene::Light>& lightDescs,
                const std::vector<Scene::IesProfile>& iesProfiles,
                const std::vector<std::string>& projectorImages,
                MetalTextures& textures);

    MTL::Buffer* buffer() const
    {
        return mLightBuffer;
    }
    MTL::Buffer* previousBuffer() const
    {
        return mPendingTemporalMapping && mPreviousLightBuffer ? mPreviousLightBuffer : mLightBuffer;
    }
    MTL::Buffer* temporalMappingBuffer() const
    {
        return mTemporalMappingBuffer;
    }
    MTL::Buffer* retainedPreviousBuffer() const
    {
        return mPreviousLightBuffer;
    }
    uint32_t previousCount() const
    {
        return mPendingTemporalMapping ? mPreviousLightCount : mLightCount;
    }
    uint64_t previousToCurrentAddress() const
    {
        return mPendingTemporalMapping && mTemporalMappingBuffer ? mTemporalMappingBuffer->gpuAddress() : 0u;
    }
    uint64_t currentToPreviousAddress() const
    {
        return mPendingTemporalMapping && mTemporalMappingBuffer ?
                   mTemporalMappingBuffer->gpuAddress() + mTemporalMappingStride * sizeof(uint32_t) :
                   0u;
    }
    void markFrameEncoded()
    {
        mPendingTemporalMapping = false;
    }
    MTL::Buffer* iesBuffer() const
    {
        return mIesBuffer;
    }
    double totalPower() const
    {
        return mTotalPower;
    }
    uint32_t infiniteLightCount() const
    {
        return mInfiniteLightCount;
    }
    uint64_t infiniteLightIndexAddress() const
    {
        return mLightBuffer && mInfiniteLightCount != 0u ? mLightBuffer->gpuAddress() + mInfiniteLightIndexOffset : 0u;
    }
    AuditCounts auditCounts() const
    {
        return mAuditCounts;
    }
#ifndef NDEBUG
    float proposalCollision() const
    {
        return mProposalCollision;
    }
    float proposalEntropy() const
    {
        return mProposalEntropy;
    }
    bool temporalMappingInjective() const
    {
        return mTemporalMappingInjective;
    }
#endif
    /// For the Metal 4 residency set: an argument table names these by handle,
    /// and a handle whose allocation is not resident is a page fault rather than
    /// a validation message.
    const std::vector<MTL::Texture*>& projectorTextures() const
    {
        return mProjectorTextures;
    }

private:
    void uploadIesProfiles(const std::vector<Scene::IesProfile>& iesProfiles);
    void loadProjectorImages(const std::vector<std::string>& paths, MetalTextures& textures);
    void releaseProjectorTextures();

    MTL::Device* mDevice = nullptr;
    MTL::Buffer* mLightBuffer = nullptr;
    MTL::Buffer* mPreviousLightBuffer = nullptr;
    MTL::Buffer* mTemporalMappingBuffer = nullptr;
    MTL::Buffer* mIesBuffer = nullptr;
    std::vector<Scene::Light> mCpuLights;
    double mTotalPower = 0.0;
    uint32_t mLightCount = 0;
    uint32_t mPreviousLightCount = 0;
    size_t mTemporalMappingStride = 0;
    bool mPendingTemporalMapping = false;
    uint32_t mInfiniteLightCount = 0;
    size_t mInfiniteLightIndexOffset = 0;
    std::vector<MTL::Texture*> mProjectorTextures;
    /// What mProjectorTextures was built from. upload() runs on every light
    /// edit -- dragging an intensity slider is one per frame -- and decoding a
    /// 4K slide each time would make the light unusable to author with.
    std::vector<std::string> mProjectorImagePaths;
    AuditCounts mAuditCounts;
#ifndef NDEBUG
    float mProposalCollision = 0.0f;
    float mProposalEntropy = 0.0f;
    bool mTemporalMappingInjective = true;
#endif
};

} // namespace oka::metal
