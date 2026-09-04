#include "MetalLights.h"

#include "ShaderTypes.h"

#include <host/ies_pack.h>
#include <host/light_selection.h>

#include <strelka/scene/light_desc.h>

#include <log.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

static_assert(sizeof(oka::ies_pack::IesBufferHeader) == sizeof(IesGpuBufferHeader));
static_assert(sizeof(oka::ies_pack::IesProfileHeader) == sizeof(IesGpuProfileHeader));
static_assert(offsetof(oka::ies_pack::IesBufferHeader, floatOffset) == offsetof(IesGpuBufferHeader, floatOffset));
static_assert(offsetof(oka::ies_pack::IesProfileHeader, nHorizontal) == offsetof(IesGpuProfileHeader, nHorizontal));
static_assert(offsetof(oka::ies_pack::IesProfileHeader, anglesOffset) == offsetof(IesGpuProfileHeader, anglesOffset));
static_assert(offsetof(oka::ies_pack::IesProfileHeader, candelaOffset) == offsetof(IesGpuProfileHeader, candelaOffset));
static_assert(offsetof(oka::ies_pack::IesProfileHeader, maxCandela) == offsetof(IesGpuProfileHeader, maxCandela));

namespace oka::metal
{
MetalLights::~MetalLights()
{
    release();
}

void MetalLights::init(MTL::Device* device)
{
    mDevice = device;
    // Shade always binds the IES table; keep a zero-count buffer ready so a
    // frame that runs before the first upload cannot hand the kernel null.
    uploadIesProfiles({});
}

void MetalLights::uploadIesProfiles(const std::vector<Scene::IesProfile>& iesProfiles)
{
    const std::vector<uint8_t> packed = ies_pack::packProfiles(iesProfiles);
    if (!mIesBuffer || mIesBuffer->length() < packed.size())
    {
        if (mIesBuffer)
            mIesBuffer->release();
        mIesBuffer = mDevice->newBuffer(packed.size(), MTL::ResourceStorageModeShared);
    }
    memcpy(mIesBuffer->contents(), packed.data(), packed.size());
}

void MetalLights::release()
{
    if (mLightBuffer)
    {
        mLightBuffer->release();
        mLightBuffer = nullptr;
    }
    mTotalPower = 0.0;
    mInfiniteLightCount = 0;
    mInfiniteLightIndexOffset = 0;
    if (mIesBuffer)
    {
        mIesBuffer->release();
        mIesBuffer = nullptr;
    }
    releaseProjectorTextures();
}

void MetalLights::releaseProjectorTextures()
{
    for (MTL::Texture* texture : mProjectorTextures)
    {
        if (texture)
            texture->release();
    }
    mProjectorTextures.clear();
    mProjectorImagePaths.clear();
}

/// Decode the images projector lights throw, in the order the scene registered
/// them, so that a light's points[0].z indexes this vector.
///
/// LDR slides use an uncompressed sRGB texture; HDR and EXR slides use linear
/// RGBA32F. A slot whose file failed to decode stays null and the shader throws
/// a plain white frame there, which is a visible rectangle rather than a light
/// that quietly stopped working.
void MetalLights::loadProjectorImages(const std::vector<std::string>& paths, MetalTextures& textures)
{
    if (paths == mProjectorImagePaths)
    {
        return;
    }
    releaseProjectorTextures();
    mProjectorImagePaths = paths;
    mProjectorTextures.reserve(paths.size());
    for (const std::string& path : paths)
    {
        mProjectorTextures.push_back(path.empty() ? nullptr : textures.loadProjectorFromFile(path));
        if (const MTL::Texture* texture = mProjectorTextures.back())
        {
            STRELKA_INFO("Loaded projector image: {} ({}x{})", path, texture->width(), texture->height());
        }
    }
}

void MetalLights::upload(const std::vector<Scene::Light>& lightDescs,
                         const std::vector<Scene::IesProfile>& iesProfiles,
                         const std::vector<std::string>& projectorImages,
                         MetalTextures& textures)
{
#ifndef NDEBUG
    ++mAuditCounts.uploads;
#endif
    loadProjectorImages(projectorImages, textures);

    // This backend's UniformLight carries one field the host's Scene::Light does
    // not -- the bindless handle of a projector's image -- so the table is built
    // field for field rather than memcpy'd whole. The shared prefix is still one
    // copy; only the handle is resolved per light, from the slot the scene
    // packed into points[0].z.
    static_assert(offsetof(UniformLight, projectorTexture) == sizeof(Scene::Light),
                  "the host light must be the exact prefix of the GPU light");
    static_assert(sizeof(UniformLight) == sizeof(Scene::Light) + 16,
                  "the GPU light adds a handle and selection data, and nothing else");

    std::vector<double> powers;
    powers.reserve(lightDescs.size());
    for (const Scene::Light& light : lightDescs)
    {
        powers.push_back(analyticLightPower(light));
    }
    const LightSelectionTable selection = buildLightSelectionAlias(powers);
#ifndef NDEBUG
    ++mAuditCounts.temporalMappingUpdates;
#endif
    mTotalPower = selection.totalPower;

    const size_t lightBufferSize = sizeof(UniformLight) * lightDescs.size();
    std::vector<uint32_t> infiniteLightIndices;
    infiniteLightIndices.reserve(lightDescs.size());
    for (uint32_t i = 0; i < lightDescs.size(); ++i)
    {
        if (lightDescs[i].type == LIGHT_TYPE_DISTANT || lightDescs[i].type == LIGHT_TYPE_DOME)
        {
            infiniteLightIndices.push_back(i);
        }
    }
    mInfiniteLightCount = static_cast<uint32_t>(infiniteLightIndices.size());
    mInfiniteLightIndexOffset = lightBufferSize;
    const size_t allocationSize = lightBufferSize + infiniteLightIndices.size() * sizeof(uint32_t);

    if (allocationSize == 0)
    {
        if (mLightBuffer)
        {
            mLightBuffer->release();
            mLightBuffer = nullptr;
        }
    }
    else
    {
        if (!mLightBuffer || mLightBuffer->length() < allocationSize)
        {
            if (mLightBuffer)
                mLightBuffer->release();
            mLightBuffer = mDevice->newBuffer(allocationSize, MTL::ResourceStorageModeShared);
        }
        auto* gpuLights = static_cast<UniformLight*>(mLightBuffer->contents());
        for (size_t i = 0; i < lightDescs.size(); ++i)
        {
            UniformLight& dst = gpuLights[i];
            std::memcpy(&dst, &lightDescs[i], sizeof(Scene::Light));
            dst.projectorTexture = MTL::ResourceID{};
            dst.color.w = selection.entries[i].pdf;
            dst.selectionAliasThreshold = selection.entries[i].aliasThreshold;
            dst.selectionAlias = selection.entries[i].alias;
            if (lightDescs[i].type == LIGHT_TYPE_PROJECTOR)
            {
                const int slot = (int)lightDescs[i].points[0].z;
                if (slot >= 0 && (size_t)slot < mProjectorTextures.size() && mProjectorTextures[slot])
                {
                    dst.projectorTexture = mProjectorTextures[slot]->gpuResourceID();
                }
            }
        }
        if (!infiniteLightIndices.empty())
        {
            std::memcpy(static_cast<uint8_t*>(mLightBuffer->contents()) + mInfiniteLightIndexOffset,
                        infiniteLightIndices.data(), infiniteLightIndices.size() * sizeof(uint32_t));
        }
    }

    const std::vector<uint8_t> packed = ies_pack::packProfiles(iesProfiles);
    if (!mIesBuffer || mIesBuffer->length() < packed.size())
    {
        if (mIesBuffer)
            mIesBuffer->release();
        mIesBuffer = mDevice->newBuffer(packed.size(), MTL::ResourceStorageModeShared);
    }
    memcpy(mIesBuffer->contents(), packed.data(), packed.size());
}

} // namespace oka::metal
