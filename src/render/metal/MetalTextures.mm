#include "MetalTextures.h"
#include "texture_compress.h"

#include <log.h>

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

namespace fs = std::filesystem;

namespace oka
{
namespace metal
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
        const size_t rowBytes =
            payload.blockBytes ? (size_t)((w + 3) / 4) * payload.blockBytes : (size_t)w * 4;
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
            out.write(reinterpret_cast<const char*>(&header), sizeof(header));
            for (uint32_t l = 0; l < payload.levels && l < payload.data.size(); ++l)
            {
                const uint32_t byteLength = (uint32_t)payload.data[l].size();
                out.write(reinterpret_cast<const char*>(&byteLength), sizeof(byteLength));
                out.write(reinterpret_cast<const char*>(payload.data[l].data()), byteLength);
            }
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
        const size_t rowBytes =
            header.blockBytes ? (size_t)((w + 3) / 4) * header.blockBytes : (size_t)w * 4;
        texture->replaceRegion(MTL::Region::Make3D(0, 0, 0, w, h, 1), l, level.data(), rowBytes);
    }
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
    if ((maxDim > 0 && (uint32_t)std::max(texWidth, texHeight) > maxDim) || divisor > 1)
    {
        int dstW = std::max(1, texWidth / (int)divisor);
        int dstH = std::max(1, texHeight / (int)divisor);
        while (maxDim > 0 && (uint32_t)std::max(dstW, dstH) > maxDim && dstW > 1 && dstH > 1)
        {
            dstW = std::max(1, dstW / 2);
            dstH = std::max(1, dstH / 2);
        }
        auto* scaled = (stbi_uc*)malloc((size_t)dstW * dstH * 4);
        const int ok = scaled ? (srgb ? stbir_resize_uint8_srgb(data, texWidth, texHeight, 0, scaled, dstW,
                                                                dstH, 0, 4, 3, 0)
                                      : stbir_resize_uint8(data, texWidth, texHeight, 0, scaled, dstW, dstH, 0, 4))
                              : 0;
        if (ok)
        {
            stbi_image_free(data);
            data = scaled;
            texWidth = dstW;
            texHeight = dstH;
        }
        else if (scaled)
        {
            free(scaled);
        }
    }

    uint32_t levels = 1;
    while ((1u << levels) <= (uint32_t)std::max(texWidth, texHeight))
        ++levels;

    std::vector<std::vector<uint8_t>> chain;
    chain.reserve(levels);
    chain.emplace_back(data, data + (size_t)texWidth * texHeight * 4);
    stbi_image_free(data);
    for (uint32_t l = 1; l < levels; ++l)
    {
        const int prevW = std::max(1, texWidth >> (l - 1));
        const int prevH = std::max(1, texHeight >> (l - 1));
        const int w = std::max(1, texWidth >> l);
        const int h = std::max(1, texHeight >> l);
        std::vector<uint8_t> next((size_t)w * h * 4);
        const int ok = srgb ? stbir_resize_uint8_srgb(chain[l - 1].data(), prevW, prevH, 0, next.data(), w, h, 0, 4, 3, 0)
                            : stbir_resize_uint8(chain[l - 1].data(), prevW, prevH, 0, next.data(), w, h, 0, 4);
        if (!ok)
        {
            levels = l;
            break;
        }
        chain.push_back(std::move(next));
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
    else if (canCompress && oka::bc::hasAlpha(chain[0].data(), texWidth, texHeight))
        bcFormat = oka::bc::Format::BC3;

    MTL::PixelFormat format;
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
        for (uint32_t l = 0; l < levels; ++l)
        {
            oka::bc::normalizeNormalMap(chain[l].data(), std::max(1, texWidth >> l), std::max(1, texHeight >> l));
        }
    }

    std::vector<std::vector<uint8_t>> payload;
    payload.reserve(levels);
    for (uint32_t l = 0; l < levels; ++l)
    {
        const int w = std::max(1, texWidth >> l);
        const int h = std::max(1, texHeight >> l);
        payload.push_back(canCompress ? oka::bc::compressImage(chain[l].data(), w, h, bcFormat)
                                      : std::move(chain[l]));
    }
    chain.clear();
    chain.shrink_to_fit();

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

void MetalTextures::prewarm(const std::vector<Request>& requests)
{
    if (requests.empty())
    {
        return;
    }
    const fs::path cacheDir = mSettings->getAs<std::string>("render/texture/cachePath");
    const DecodeParams params = readDecodeParams();

    // Deduplicated first: a scene routinely uses one map in several materials,
    // and decoding it once per use would spend the cores undoing the saving the
    // dedup cache exists to make.
    std::vector<Request> unique;
    std::vector<std::string> keys;
    {
        std::unordered_map<std::string, size_t> seen;
        for (const Request& r : requests)
        {
            if (r.path.empty())
                continue;
            std::string key = cacheKey(r.path, r.srgb, r.kind);
            if (seen.count(key) != 0)
                continue;
            seen.emplace(key, unique.size());
            unique.push_back(r);
            keys.push_back(std::move(key));
        }
    }
    if (unique.empty())
    {
        return;
    }

    std::vector<Payload> results(unique.size());
    // Written through a pointer: a block captures by const value, and each index
    // is touched by exactly one iteration, so no locking is needed.
    Payload* out = results.data();
    dispatch_apply(unique.size(), DISPATCH_APPLY_AUTO, ^(size_t i) {
        const Request& r = unique[i];
        const std::string cacheFile = cacheDir.empty() ? std::string() : (cacheDir / keys[i]).string();
        out[i] = decodeToPayload(r.path, r.srgb, r.kind, cacheFile, params);
    });

    for (size_t i = 0; i < unique.size(); ++i)
    {
        if (results[i].valid)
        {
            mPrewarmed.emplace(keys[i], std::move(results[i]));
        }
    }
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

    const std::string key =
        absolutePath + (srgb ? "|srgb" : "|linear") + "|" + std::to_string((int)kind);
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
    for (MTL::Texture* t : mTexturesNeedingMips)
        blit->generateMipmaps(t);
    blit->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    cb->release();
    STRELKA_INFO("Generated mipmaps for {} textures", mTexturesNeedingMips.size());
    mTexturesNeedingMips.clear();
}

} // namespace metal
} // namespace oka
