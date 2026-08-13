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

    struct Request
    {
        std::string path;
        bool srgb = false;
        TextureKind kind = TextureKind::Color;
    };
    /// Decode, resample, mip and encode every request across all cores, ahead of
    /// the material build that will ask for them one at a time.
    ///
    /// All of that is per-file, pure CPU and by far the longest part of a cold
    /// load -- 53.9 s of the pine forest, against 6.2 s for the encode alone --
    /// and it ran on one core while the rest of the machine sat idle. What it
    /// produces is held until loadFromFile() asks, which then only has the Metal
    /// calls left to make.
    void prewarm(const std::vector<Request>& requests);

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
    /// One texture's pixels, ready to upload. Produced without touching Metal or
    /// any shared state, which is what lets it be produced concurrently.
    struct Payload
    {
        int width = 0;
        int height = 0;
        uint32_t levels = 0;
        uint32_t pixelFormat = 0;
        uint32_t blockBytes = 0;
        std::vector<std::vector<uint8_t>> data;
        bool fromCache = false;
        bool valid = false;
    };
    /// Settings read once, on the calling thread: SettingsManager::getAs is not
    /// const and inserts defaults, so reading it from several threads is a race.
    struct DecodeParams
    {
        uint32_t maxDimension = 0;
        uint32_t downscale = 1;
        bool compress = true;
        bool deviceSupportsBC = false;
    };
    DecodeParams readDecodeParams() const;
    Payload decodeToPayload(const std::string& fileName,
                            bool srgb,
                            TextureKind kind,
                            const std::string& cacheFile,
                            const DecodeParams& params) const;
    static Payload readCachedPayload(const std::string& cachePath);
    MTL::Texture* createFromPayload(const Payload& payload, const std::string& cacheFileToWrite);

    std::string cacheKey(const std::string& fileName, bool srgb, TextureKind kind) const;
    MTL::Texture* loadCached(const std::string& cachePath);

    MTL::Device* mDevice = nullptr;
    MTL::CommandQueue* mQueue = nullptr;
    SettingsManager* mSettings = nullptr;
    std::vector<MTL::Texture*> mMaterialTextures;
    std::vector<MTL::Texture*> mTexturesNeedingMips;
    std::unordered_map<std::string, MTL::Texture*> mDedupCache;
    /// Payloads produced by prewarm(), keyed by cache key, consumed by loadFromFile().
    std::unordered_map<std::string, Payload> mPrewarmed;
    uint32_t mCacheHits = 0;
    uint32_t mCacheMisses = 0;
};

} // namespace metal
} // namespace oka
