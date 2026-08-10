#pragma once

#include "texture_cache_key.h"

#include <Metal/Metal.hpp>
#include <settings.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace oka
{
namespace metal
{

// Material texture domain: decode → mip/BC → GPU texture + disk cache.
// Does not own IBL or bind into the integrator.
class MetalTextures
{
public:
    MetalTextures() = default;
    ~MetalTextures();

    void init(MTL::Device* device, MTL::CommandQueue* queue, SettingsManager* settings);

    MTL::Texture* loadFromFile(const std::string& absolutePath, bool srgb, TextureKind kind = TextureKind::Color);

    // Deduped load for material build. Tracks ownership in materialTextures().
    MTL::ResourceID loadMaterialTexture(const std::string& absolutePath, bool srgb, TextureKind kind = TextureKind::Color);

    void beginMaterialPass();
    void generateMips();
    void releaseAll();

    const std::vector<MTL::Texture*>& materialTextures() const
    {
        return mMaterialTextures;
    }
    uint32_t cacheHits() const
    {
        return mCacheHits;
    }
    uint32_t cacheMisses() const
    {
        return mCacheMisses;
    }

private:
    std::string cacheKey(const std::string& fileName, bool srgb, TextureKind kind) const;
    MTL::Texture* loadCached(const std::string& cachePath);

    MTL::Device* mDevice = nullptr;
    MTL::CommandQueue* mQueue = nullptr;
    SettingsManager* mSettings = nullptr;
    std::vector<MTL::Texture*> mMaterialTextures;
    std::vector<MTL::Texture*> mTexturesNeedingMips;
    std::unordered_map<std::string, MTL::Texture*> mDedupCache;
    uint32_t mCacheHits = 0;
    uint32_t mCacheMisses = 0;
};

} // namespace metal
} // namespace oka
