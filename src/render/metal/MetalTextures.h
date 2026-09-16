#pragma once

#include <host/texture_asset.h>
#include <host/texture_cache_key.h>

#include <Metal/Metal.hpp>
#include <settings.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace oka::metal
{

using TextureKind = texture::Semantic;

// Material texture domain: decode -> mip/native format -> GPU texture + disk cache.
// Does not own IBL or bind into the integrator.
class MetalTextures
{
public:
    MetalTextures() = default;
    ~MetalTextures();

    void init(MTL::Device* device, SettingsManager* settings);

    MTL::Texture* loadFromFile(const std::string& absolutePath, bool srgb, TextureKind kind = TextureKind::Color);

    /// Load an emitted projector image without material resizing, mipmaps or
    /// block compression. LDR stays encoded in an sRGB UNORM texture; HDR and
    /// EXR retain linear float radiance. The caller owns the returned texture.
    MTL::Texture* loadProjectorFromFile(const std::string& absolutePath);

    struct Request
    {
        std::string path;
        bool srgb = false;
        TextureKind kind = TextureKind::Color;
    };
    /// Resolve and deduplicate the texture plan once for a material pass.
    void beginMaterialPass(const std::vector<Request>& requests);
    /// Decode/cache-read the prepared plan a batch at a time. Returns true when complete.
    bool prewarmStep(double budgetMs);

    // Deduped load for material build. Tracks ownership in materialTextures().
    MTL::ResourceID loadMaterialTexture(const std::string& absolutePath, bool srgb, TextureKind kind = TextureKind::Color);

    void releaseAll();

    size_t prewarmDone() const
    {
        return mPrewarmCursor;
    }
    size_t prewarmTotal() const
    {
        return mPrewarmQueue.size();
    }

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
        uint32_t blockExtent = 0;
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
        bool deviceSupportsAstc = false;
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
    static std::string materialTextureKey(const std::string& fileName, bool srgb, TextureKind kind);
    MTL::Device* mDevice = nullptr;
    SettingsManager* mSettings = nullptr;
    std::vector<MTL::Texture*> mMaterialTextures;
    std::unordered_map<std::string, MTL::Texture*> mDedupCache;
    /// Deduplicated request list and how far through it the fan-out has got.
    std::vector<Request> mPrewarmQueue;
    std::vector<std::string> mPrewarmKeys;
    size_t mPrewarmCursor = 0;
    uint32_t mCacheHits = 0;
    uint32_t mCacheMisses = 0;
};

} // namespace oka::metal
