#include "MetalTextures.h"
#include "texture_compress.h"

#include <log.h>

#include <dispatch/dispatch.h>

#include <thread>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

namespace fs = std::filesystem;


namespace oka::metal
{
namespace
{
struct CachedTextureHeader
{
    char magic[4];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t levels;
    uint32_t pixelFormat;
    uint32_t blockBytes;
    uint32_t reserved;
};
} // namespace

MetalTextures::~MetalTextures()
{
    releaseAll();
}

void MetalTextures::init(MTL::Device* device, MTL::CommandQueue* queue, SettingsManager* settings)
{
    mDevice = device;
    mQueue = queue;
    mSettings = settings;
}

void MetalTextures::beginMaterialPass()
{
    mPrewarmPrepared = false;
    mPrewarmQueue.clear();
    mPrewarmKeys.clear();
    mPrewarmCursor = 0;
    mDedupCache.clear();
}

void MetalTextures::releaseAll()
{
    for (MTL::Texture* t : mMaterialTextures)
    {
        if (t)
            t->release();
    }
    mMaterialTextures.clear();
    mDedupCache.clear();
    mTexturesNeedingMips.clear();
}

std::string MetalTextures::cacheKey(const std::string& fileName, bool srgb, TextureKind kind) const
{
    std::error_code ec;
    const auto size = fs::file_size(fileName, ec);
    const auto stamp = fs::last_write_time(fileName, ec).time_since_epoch().count();
    TextureCacheKeyInputs in;
    in.fileName = fileName;
    in.fileSize = ec ? 0 : (uint64_t)size;
    in.writeTimeCount = ec ? 0 : (int64_t)stamp;
    in.maxDimension = mSettings->getAs<uint32_t>("render/texture/maxDimension");
    in.downscale = mSettings->getAs<uint32_t>("render/texture/downscale");
    in.srgb = srgb;
    in.kind = kind;
    return textureCacheKey(in);
}

MetalTextures::Payload MetalTextures::readCachedPayload(const std::string& cachePath)
{
    Payload payload;
    std::ifstream in(cachePath, std::ios::binary);
    if (!in)
        return payload;
    CachedTextureHeader header{};
    // Binary cache I/O: streaming POD/byte buffers through char* is the idiom.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || std::memcmp(header.magic, "BTEX", 4) != 0 || header.version != kTextureCacheVersion)
        return payload;
    payload.width = (int)header.width;
    payload.height = (int)header.height;
    payload.levels = header.levels;
    payload.pixelFormat = header.pixelFormat;
    payload.blockBytes = header.blockBytes;
    payload.data.reserve(header.levels);
    for (uint32_t l = 0; l < header.levels; ++l)
    {
        uint32_t byteLength = 0;
        in.read(reinterpret_cast<char*>(&byteLength), sizeof(byteLength));
        if (!in || byteLength == 0)
            return Payload{};
        std::vector<uint8_t> level(byteLength);
        in.read(reinterpret_cast<char*>(level.data()), byteLength);
        if (!in)
            return Payload{};
        payload.data.push_back(std::move(level));
    }
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
    payload.fromCache = true;
    payload.valid = true;
    return payload;
}

MTL::Texture* MetalTextures::createFromPayload(const Payload& payload, const std::string& cacheFileToWrite)
{
    if (!payload.valid || payload.data.empty())
        return nullptr;
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(payload.width);
    desc->setHeight(payload.height);
    desc->setMipmapLevelCount(payload.levels);
    desc->setPixelFormat((MTL::PixelFormat)payload.pixelFormat);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead);
    MTL::Texture* texture = mDevice->newTexture(desc);
    desc->release();
    if (!texture)
        return nullptr;
    for (uint32_t l = 0; l < payload.levels && l < payload.data.size(); ++l)
    {
        const uint32_t w = std::max(1, payload.width >> l);
        const uint32_t h = std::max(1, payload.height >> l);
        const size_t rowBytes = payload.blockBytes ? (size_t)((w + 3) / 4) * payload.blockBytes : (size_t)w * 4;
        texture->replaceRegion(MTL::Region::Make3D(0, 0, 0, w, h, 1), l, payload.data[l].data(), rowBytes);
    }
    if (!cacheFileToWrite.empty())
    {
        std::error_code ec;
        fs::create_directories(fs::path(cacheFileToWrite).parent_path(), ec);
        // Written on one thread only: two decoders racing the same .tmp would
        // interleave into a file that reads back as a valid header and rubbish.
        const std::string tmp = cacheFileToWrite + ".tmp";
        std::ofstream out(tmp, std::ios::binary);
        if (out)
        {
            CachedTextureHeader header{};
            std::memcpy(header.magic, "BTEX", 4);
            header.version = kTextureCacheVersion;
            header.width = (uint32_t)payload.width;
            header.height = (uint32_t)payload.height;
            header.levels = payload.levels;
            header.pixelFormat = payload.pixelFormat;
            header.blockBytes = payload.blockBytes;
            // Binary cache I/O: streaming POD/byte buffers through char* is the idiom.
            // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
            out.write(reinterpret_cast<const char*>(&header), sizeof(header));
            for (uint32_t l = 0; l < payload.levels && l < payload.data.size(); ++l)
            {
                const uint32_t byteLength = (uint32_t)payload.data[l].size();
                out.write(reinterpret_cast<const char*>(&byteLength), sizeof(byteLength));
                out.write(reinterpret_cast<const char*>(payload.data[l].data()), byteLength);
            }
            // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
            out.close();
            fs::rename(tmp, cacheFileToWrite, ec);
        }
    }
    return texture;
}

MTL::Texture* MetalTextures::loadCached(const std::string& cachePath)
{
    std::ifstream in(cachePath, std::ios::binary);
    if (!in)
        return nullptr;

    CachedTextureHeader header{};
    // Binary cache I/O: streaming POD/byte buffers through char* is the idiom.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || std::memcmp(header.magic, "BTEX", 4) != 0 || header.version != kTextureCacheVersion)
        return nullptr;

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(header.width);
    desc->setHeight(header.height);
    desc->setMipmapLevelCount(header.levels);
    desc->setPixelFormat((MTL::PixelFormat)header.pixelFormat);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead);
    MTL::Texture* texture = mDevice->newTexture(desc);
    desc->release();
    if (!texture)
        return nullptr;

    std::vector<uint8_t> level;
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    for (uint32_t l = 0; l < header.levels; ++l)
    {
        uint32_t byteLength = 0;
        in.read(reinterpret_cast<char*>(&byteLength), sizeof(byteLength));
        if (!in || byteLength == 0)
        {
            texture->release();
            return nullptr;
        }
        level.resize(byteLength);
        in.read(reinterpret_cast<char*>(level.data()), byteLength);
        if (!in)
        {
            texture->release();
            return nullptr;
        }
        const uint32_t w = std::max(1u, header.width >> l);
        const uint32_t h = std::max(1u, header.height >> l);
        const size_t rowBytes = header.blockBytes ? (size_t)((w + 3) / 4) * header.blockBytes : (size_t)w * 4;
        texture->replaceRegion(MTL::Region::Make3D(0, 0, 0, w, h, 1), l, level.data(), rowBytes);
    }
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
    return texture;
}

MetalTextures::DecodeParams MetalTextures::readDecodeParams() const
{
    DecodeParams params;
    params.maxDimension = mSettings->getAs<uint32_t>("render/texture/maxDimension");
    params.downscale = std::max(1u, mSettings->getAs<uint32_t>("render/texture/downscale"));
    params.compress = mSettings->getAs<bool>("render/texture/compress");
    params.deviceSupportsBC = mDevice->supportsBCTextureCompression();
    return params;
}

// Everything a texture needs that does not involve Metal: read the cache, or
// decode, resample, build the mip chain and encode. No shared state and no
// settings lookups, so any number of these can run at once.
MetalTextures::Payload MetalTextures::decodeToPayload(const std::string& fileName,
                                                      bool srgb,
                                                      TextureKind kind,
                                                      const std::string& cacheFile,
                                                      const DecodeParams& params) const
{
    if (!cacheFile.empty())
    {
        Payload cached = readCachedPayload(cacheFile);
        if (cached.valid)
        {
            return cached;
        }
    }

    int texWidth = 0;
    int texHeight = 0;
    int texChannels = 0;
    stbi_uc* data = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (data == nullptr)
    {
        STRELKA_ERROR("Unable to load texture from file: {}", fileName.c_str());
        return Payload{};
    }

    const uint32_t maxDim = params.maxDimension;
    const uint32_t divisor = std::max(1u, params.downscale);
    // Dimensions are non-negative by construction, so naming the unsigned edge
    // states the conversion once instead of burying it in a comparison.
    const uint32_t srcLongestEdge = (uint32_t)std::max(texWidth, texHeight);
    if ((maxDim > 0 && srcLongestEdge > maxDim) || divisor > 1)
    {
        int dstW = std::max(1, texWidth / (int)divisor);
        int dstH = std::max(1, texHeight / (int)divisor);
        const int maxEdge = (int)std::min<uint32_t>(maxDim, (uint32_t)INT_MAX);
        while (maxEdge > 0 && std::max(dstW, dstH) > maxEdge && dstW > 1 && dstH > 1)
        {
            dstW = std::max(1, dstW / 2);
            dstH = std::max(1, dstH / 2);
        }
        // Freed through stbi_image_free once it takes over `data`, and that is
        // free() under another name, so it has to come from malloc.
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        auto* scaled = (stbi_uc*)malloc((size_t)dstW * dstH * 4);
        const int ok =
            scaled ? (srgb ? stbir_resize_uint8_srgb(data, texWidth, texHeight, 0, scaled, dstW, dstH, 0, 4, 3, 0) :
                             stbir_resize_uint8(data, texWidth, texHeight, 0, scaled, dstW, dstH, 0, 4)) :
                     0;
        if (ok)
        {
            stbi_image_free(data);
            data = scaled;
            texWidth = dstW;
            texHeight = dstH;
        }
        else if (scaled)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
            free(scaled);
        }
    }

    uint32_t levels = 1;
    while ((1u << levels) <= (uint32_t)std::max(texWidth, texHeight))
        ++levels;

    // Keep the decoder's buffer as mip 0. Copying it into a vector just to
    // compress or upload it doubles the full-resolution image in RAM.
    const std::unique_ptr<stbi_uc, void (*)(void*)> base(data, stbi_image_free);
    data = nullptr;
    std::vector<std::vector<uint8_t>> mips;
    mips.reserve(levels > 0 ? levels - 1 : 0);
    const uint8_t* prev = base.get();
    int prevW = texWidth;
    int prevH = texHeight;
    for (uint32_t l = 1; l < levels; ++l)
    {
        const int w = std::max(1, texWidth >> l);
        const int h = std::max(1, texHeight >> l);
        std::vector<uint8_t> next((size_t)w * h * 4);
        const int ok = srgb ? stbir_resize_uint8_srgb(prev, prevW, prevH, 0, next.data(), w, h, 0, 4, 3, 0) :
                              stbir_resize_uint8(prev, prevW, prevH, 0, next.data(), w, h, 0, 4);
        if (!ok)
        {
            levels = l;
            break;
        }
        mips.push_back(std::move(next));
        prev = mips.back().data();
        prevW = w;
        prevH = h;
    }

    // A normal map tagged sRGB is neither compressed nor normalised: BC5 has no
    // sRGB variant, and both the encode and the Z reconstruction need the values
    // linear. It does not happen -- the material build asks for normal maps
    // linear -- but silently dropping the transfer function would be worse than
    // leaving such a texture as it was.
    const bool srgbNormal = kind == TextureKind::Normal && srgb;
    const bool normalMap = kind == TextureKind::Normal && !srgb;
    const bool canCompress = !srgbNormal && params.deviceSupportsBC && params.compress;
    oka::bc::Format bcFormat = oka::bc::Format::BC1;
    if (normalMap)
        bcFormat = oka::bc::Format::BC5;
    else if (canCompress && oka::bc::hasAlpha(base.get(), texWidth, texHeight))
        bcFormat = oka::bc::Format::BC3;

    MTL::PixelFormat format = MTL::PixelFormatRGBA8Unorm;
    if (!canCompress)
        format = srgb ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm;
    else if (bcFormat == oka::bc::Format::BC5)
        format = MTL::PixelFormatBC5_RGUnorm;
    else if (bcFormat == oka::bc::Format::BC3)
        format = srgb ? MTL::PixelFormatBC3_RGBA_sRGB : MTL::PixelFormatBC3_RGBA;
    else
        format = srgb ? MTL::PixelFormatBC1_RGBA_sRGB : MTL::PixelFormatBC1_RGBA;

    // Every mip level, not just the base: the box filter above averages unit
    // vectors, which shortens them, and the shader rebuilds Z on the assumption
    // that they are unit. Done whether or not the texture ends up compressed, so
    // the two paths shade the same.
    if (normalMap)
    {
        oka::bc::normalizeNormalMap(base.get(), texWidth, texHeight);
        for (uint32_t l = 1; l < levels; ++l)
        {
            oka::bc::normalizeNormalMap(mips[l - 1].data(), std::max(1, texWidth >> l), std::max(1, texHeight >> l));
        }
    }

    std::vector<std::vector<uint8_t>> payload;
    payload.reserve(levels);
    if (canCompress)
    {
        payload.push_back(oka::bc::compressImage(base.get(), texWidth, texHeight, bcFormat));
        for (uint32_t l = 1; l < levels; ++l)
        {
            payload.push_back(oka::bc::compressImage(
                mips[l - 1].data(), std::max(1, texWidth >> l), std::max(1, texHeight >> l), bcFormat));
        }
    }
    else
    {
        payload.emplace_back(base.get(), base.get() + (size_t)texWidth * texHeight * 4);
        for (auto& mip : mips)
        {
            payload.push_back(std::move(mip));
        }
    }
    mips.clear();
    mips.shrink_to_fit();

    Payload out;
    out.width = texWidth;
    out.height = texHeight;
    out.levels = levels;
    out.pixelFormat = (uint32_t)format;
    out.blockBytes = canCompress ? (uint32_t)oka::bc::blockBytes(bcFormat) : 0u;
    out.data = std::move(payload);
    out.fromCache = false;
    out.valid = true;
    return out;
}

bool MetalTextures::prewarmStep(const std::vector<Request>& requests, double budgetMs)
{
    if (!mPrewarmPrepared)
    {
        mPrewarmPrepared = true;
        mPrewarmQueue.clear();
        mPrewarmKeys.clear();
        mPrewarmCursor = 0;
        // Deduplicated first: a scene routinely uses one map in several
        // materials, and decoding it once per use would spend the cores undoing
        // the saving the dedup cache exists to make.
        std::unordered_map<std::string, size_t> seen;
        for (const Request& r : requests)
        {
            if (r.path.empty())
                continue;
            std::string key = cacheKey(r.path, r.srgb, r.kind);
            if (seen.count(key) != 0)
                continue;
            seen.emplace(key, mPrewarmQueue.size());
            mPrewarmQueue.push_back(r);
            mPrewarmKeys.push_back(std::move(key));
        }
    }
    if (mPrewarmCursor >= mPrewarmQueue.size())
    {
        return true;
    }

    const fs::path cacheDir = mSettings->getAs<std::string>("render/texture/cachePath");
    const DecodeParams params = readDecodeParams();

    // A batch wide enough to fill the machine, short enough to hand control back
    // between them. One texture can be a second on its own, so the budget is
    // checked per batch rather than relied on to cut one short.
    const size_t batch = std::max<size_t>(1, (size_t)std::thread::hardware_concurrency());
    const auto sliceStart = std::chrono::steady_clock::now();
    // The early return above guarantees at least one pending entry, so the first
    // batch always runs; the condition only gates whether a further batch starts.
    while (mPrewarmCursor < mPrewarmQueue.size() &&
           (budgetMs <= 0.0 ||
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sliceStart).count() <
                budgetMs))
    {
        const size_t begin = mPrewarmCursor;
        const size_t count = std::min(batch, mPrewarmQueue.size() - begin);
        std::vector<Payload> results(count);
        // Written through a pointer: a block captures by const value, and each
        // index is touched by exactly one iteration, so no locking is needed.
        Payload* out = results.data();
        const Request* queue = mPrewarmQueue.data();
        const std::string* keys = mPrewarmKeys.data();
        dispatch_apply(count, DISPATCH_APPLY_AUTO, ^(size_t i) {
          const Request& r = queue[begin + i];
          const std::string cacheFile = cacheDir.empty() ? std::string() : (cacheDir / keys[begin + i]).string();
          out[i] = decodeToPayload(r.path, r.srgb, r.kind, cacheFile, params);
        });
        for (size_t i = 0; i < count; ++i)
        {
            if (results[i].valid)
            {
                mPrewarmed.emplace(mPrewarmKeys[begin + i], std::move(results[i]));
            }
        }
        mPrewarmCursor += count;
    }

    return mPrewarmCursor >= mPrewarmQueue.size();
}

MTL::Texture* MetalTextures::loadFromFile(const std::string& fileName, bool srgb, TextureKind kind)
{
    const fs::path cacheDir = mSettings->getAs<std::string>("render/texture/cachePath");
    const std::string key = cacheKey(fileName, srgb, kind);
    const std::string cacheFile = cacheDir.empty() ? std::string() : (cacheDir / key).string();

    // prewarm() may already have done everything except the Metal calls.
    Payload payload;
    if (auto it = mPrewarmed.find(key); it != mPrewarmed.end())
    {
        payload = std::move(it->second);
        mPrewarmed.erase(it);
    }
    else
    {
        payload = decodeToPayload(fileName, srgb, kind, cacheFile, readDecodeParams());
    }
    if (!payload.valid)
    {
        STRELKA_ERROR("Unable to load texture from file: {}", fileName.c_str());
        return nullptr;
    }
    (payload.fromCache ? mCacheHits : mCacheMisses)++;
    return createFromPayload(payload, payload.fromCache ? std::string() : cacheFile);
}

MTL::ResourceID MetalTextures::loadMaterialTexture(const std::string& absolutePath, bool srgb, TextureKind kind)
{
    if (absolutePath.empty())
        return MTL::ResourceID{};

    const std::string key = absolutePath + (srgb ? "|srgb" : "|linear") + "|" + std::to_string((int)kind);
    if (auto it = mDedupCache.find(key); it != mDedupCache.end())
        return it->second->gpuResourceID();

    MTL::Texture* tex = loadFromFile(absolutePath, srgb, kind);
    if (!tex)
        return MTL::ResourceID{};

    mDedupCache[key] = tex;
    mMaterialTextures.push_back(tex);
    return tex->gpuResourceID();
}

void MetalTextures::generateMips()
{
    if (mTexturesNeedingMips.empty() || !mQueue)
        return;
    MTL::CommandBuffer* cb = mQueue->commandBuffer();
    cb->retain();
    MTL::BlitCommandEncoder* blit = cb->blitCommandEncoder();
    for (const MTL::Texture* t : mTexturesNeedingMips)
        blit->generateMipmaps(t);
    blit->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    cb->release();
    STRELKA_INFO("Generated mipmaps for {} textures", mTexturesNeedingMips.size());
    mTexturesNeedingMips.clear();
}

} // namespace oka::metal

