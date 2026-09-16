#include "MetalTextures.h"
#include <host/projector_transfer.h>
#include <host/texture_compress.h>

#include <ktx.h>
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
#include <tinyexr.h>

namespace fs = std::filesystem;

namespace oka::metal
{
namespace
{
inline constexpr uint32_t kMetalPayloadVersion = 3;

struct CachedTextureHeader
{
    char magic[4];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t levels;
    uint32_t pixelFormat;
    uint32_t blockBytes;
    uint32_t blockExtent;
};

size_t payloadRowBytes(uint32_t width, uint32_t blockExtent, uint32_t blockBytes)
{
    return blockBytes && blockExtent ? static_cast<size_t>((width + blockExtent - 1) / blockExtent) * blockBytes :
                                       static_cast<size_t>(width) * 4;
}

size_t payloadLevelBytes(uint32_t width, uint32_t height, uint32_t blockExtent, uint32_t blockBytes)
{
    if (!blockBytes)
        return static_cast<size_t>(width) * height * 4;
    return payloadRowBytes(width, blockExtent, blockBytes) * ((height + blockExtent - 1) / blockExtent);
}

struct Rgba8LevelView
{
    const uint8_t* data = nullptr;
    size_t size = 0;
};

struct KtxTextureDeleter
{
    void operator()(ktxTexture2* texture) const
    {
        if (texture)
            ktxTexture2_Destroy(texture);
    }
};

bool encodeAstc(const std::vector<Rgba8LevelView>& sourceLevels,
                int width,
                int height,
                texture::NativeFormat format,
                bool srgb,
                std::vector<std::vector<uint8_t>>& encodedLevels,
                std::string& error)
{
    encodedLevels.clear();
    if (sourceLevels.empty() || width <= 0 || height <= 0 ||
        (format != texture::NativeFormat::ASTC4x4 && format != texture::NativeFormat::ASTC6x6))
    {
        error = "invalid ASTC input";
        return false;
    }

    constexpr uint32_t kRgba8Unorm = 37;
    constexpr uint32_t kRgba8Srgb = 43;
    ktxTextureCreateInfo createInfo{};
    createInfo.vkFormat = srgb ? kRgba8Srgb : kRgba8Unorm;
    createInfo.baseWidth = static_cast<uint32_t>(width);
    createInfo.baseHeight = static_cast<uint32_t>(height);
    createInfo.baseDepth = 1;
    createInfo.numDimensions = 2;
    createInfo.numLevels = static_cast<uint32_t>(sourceLevels.size());
    createInfo.numLayers = 1;
    createInfo.numFaces = 1;

    ktxTexture2* rawTexture = nullptr;
    KTX_error_code result = ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &rawTexture);
    const std::unique_ptr<ktxTexture2, KtxTextureDeleter> ktx(rawTexture);
    if (result != KTX_SUCCESS)
    {
        error = ktxErrorString(result);
        return false;
    }

    for (uint32_t level = 0; level < sourceLevels.size(); ++level)
    {
        const size_t expected =
            static_cast<size_t>(std::max(1, width >> level)) * static_cast<size_t>(std::max(1, height >> level)) * 4;
        if (!sourceLevels[level].data || sourceLevels[level].size != expected)
        {
            error = "invalid RGBA8 mip size";
            return false;
        }
        result = ktxTexture_SetImageFromMemory(
            ktxTexture(ktx.get()), level, 0, 0, sourceLevels[level].data, sourceLevels[level].size);
        if (result != KTX_SUCCESS)
        {
            error = ktxErrorString(result);
            return false;
        }
    }

    ktxAstcParams params{};
    params.structSize = sizeof(params);
    params.threadCount = 1;
    params.blockDimension = format == texture::NativeFormat::ASTC4x4 ? KTX_PACK_ASTC_BLOCK_DIMENSION_4x4 :
                                                                       KTX_PACK_ASTC_BLOCK_DIMENSION_6x6;
    params.mode = KTX_PACK_ASTC_ENCODER_MODE_LDR;
    params.qualityLevel = KTX_PACK_ASTC_QUALITY_LEVEL_FAST;
    params.perceptual = srgb;
    result = ktxTexture2_CompressAstcEx(ktx.get(), &params);
    if (result != KTX_SUCCESS)
    {
        error = ktxErrorString(result);
        return false;
    }

    encodedLevels.reserve(sourceLevels.size());
    const uint8_t* data = ktxTexture_GetData(ktxTexture(ktx.get()));
    for (uint32_t level = 0; level < sourceLevels.size(); ++level)
    {
        ktx_size_t offset = 0;
        result = ktxTexture_GetImageOffset(ktxTexture(ktx.get()), level, 0, 0, &offset);
        if (result != KTX_SUCCESS)
        {
            error = ktxErrorString(result);
            encodedLevels.clear();
            return false;
        }
        const size_t bytes = texture::levelBytes(format, std::max(1, width >> level), std::max(1, height >> level));
        encodedLevels.emplace_back(data + offset, data + offset + bytes);
    }
    return true;
}
} // namespace

MetalTextures::~MetalTextures()
{
    releaseAll();
}

void MetalTextures::init(MTL::Device* device, SettingsManager* settings)
{
    mDevice = device;
    mSettings = settings;
}

void MetalTextures::beginMaterialPass(const std::vector<Request>& requests)
{
    mPrewarmQueue.clear();
    mPrewarmKeys.clear();
    mPrewarmCursor = 0;
    mDedupCache.clear();
    mCacheHits = 0;
    mCacheMisses = 0;

    // A scene routinely uses one map in several materials. Resolve metadata and
    // deduplicate once here; subsequent budget slices only advance the queue.
    std::unordered_map<std::string, size_t> seen;
    seen.reserve(requests.size());
    mPrewarmQueue.reserve(requests.size());
    mPrewarmKeys.reserve(requests.size());
    for (const Request& request : requests)
    {
        if (request.path.empty())
            continue;
        const std::string requestKey = materialTextureKey(request.path, request.srgb, request.kind);
        if (!seen.emplace(requestKey, mPrewarmQueue.size()).second)
            continue;
        mPrewarmQueue.push_back(request);
        mPrewarmKeys.push_back(cacheKey(request.path, request.srgb, request.kind));
    }
    STRELKA_INFO("Material textures: {} references, {} unique", requests.size(), mPrewarmQueue.size());
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
}

std::string MetalTextures::cacheKey(const std::string& fileName, bool srgb, TextureKind kind) const
{
    std::error_code sizeError;
    std::error_code stampError;
    const auto size = fs::file_size(fileName, sizeError);
    const auto stamp = fs::last_write_time(fileName, stampError).time_since_epoch().count();
    texture::TextureCacheKeyInputs in;
    in.fileName = fileName;
    in.fileSize = sizeError ? 0 : (uint64_t)size;
    in.writeTimeCount = stampError ? 0 : (int64_t)stamp;
    in.maxDimension = mSettings->getAs<uint32_t>("render/texture/maxDimension");
    in.downscale = mSettings->getAs<uint32_t>("render/texture/downscale");
    in.srgb = srgb;
    in.compressed = mSettings->getAs<bool>("render/texture/compress") && mDevice->supportsFamily(MTL::GPUFamilyApple1);
    in.semantic = kind;
    in.target = texture::TargetProfile::AppleAstc;
    in.encoderVersion = kMetalPayloadVersion;
    return texture::textureCacheKey(in);
}

std::string MetalTextures::materialTextureKey(const std::string& fileName, bool srgb, TextureKind kind)
{
    return fileName + (srgb ? "|srgb" : "|linear") + "|" + std::to_string((int)kind);
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
    if (!in || std::memcmp(header.magic, "BTEX", 4) != 0 || header.version != kMetalPayloadVersion)
        return payload;
    payload.width = (int)header.width;
    payload.height = (int)header.height;
    payload.levels = header.levels;
    payload.pixelFormat = header.pixelFormat;
    payload.blockBytes = header.blockBytes;
    payload.blockExtent = header.blockExtent;
    if (payload.blockBytes && !payload.blockExtent)
        return Payload{};
    payload.data.reserve(header.levels);
    for (uint32_t l = 0; l < header.levels; ++l)
    {
        uint32_t byteLength = 0;
        in.read(reinterpret_cast<char*>(&byteLength), sizeof(byteLength));
        const uint32_t width = std::max(1u, header.width >> l);
        const uint32_t height = std::max(1u, header.height >> l);
        const size_t expected = payloadLevelBytes(width, height, payload.blockExtent, payload.blockBytes);
        if (!in || byteLength == 0 || byteLength != expected)
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
        const size_t rowBytes = payloadRowBytes(w, payload.blockExtent, payload.blockBytes);
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
            header.version = kMetalPayloadVersion;
            header.width = (uint32_t)payload.width;
            header.height = (uint32_t)payload.height;
            header.levels = payload.levels;
            header.pixelFormat = payload.pixelFormat;
            header.blockBytes = payload.blockBytes;
            header.blockExtent = payload.blockExtent;
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

MetalTextures::DecodeParams MetalTextures::readDecodeParams() const
{
    DecodeParams params;
    params.maxDimension = mSettings->getAs<uint32_t>("render/texture/maxDimension");
    params.downscale = std::max(1u, mSettings->getAs<uint32_t>("render/texture/downscale"));
    params.compress = mSettings->getAs<bool>("render/texture/compress");
    params.deviceSupportsAstc = mDevice->supportsFamily(MTL::GPUFamilyApple1);
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

    const texture::Extent extent = texture::resolveExtent(texWidth, texHeight, params.maxDimension, params.downscale);
    if (extent.width != texWidth || extent.height != texHeight)
    {
        // Freed through stbi_image_free once it takes over `data`, and that is
        // free() under another name, so it has to come from malloc.
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        auto* scaled = (stbi_uc*)malloc((size_t)extent.width * extent.height * 4);
        const int ok =
            scaled ?
                (srgb ? stbir_resize_uint8_srgb(
                            data, texWidth, texHeight, 0, scaled, extent.width, extent.height, 0, 4, 3, 0) :
                        stbir_resize_uint8(data, texWidth, texHeight, 0, scaled, extent.width, extent.height, 0, 4)) :
                0;
        if (ok)
        {
            stbi_image_free(data);
            data = scaled;
            texWidth = extent.width;
            texHeight = extent.height;
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

    const bool normalMap = kind == TextureKind::Normal && !srgb;
    if (normalMap)
    {
        bc::normalizeNormalMap(base.get(), texWidth, texHeight);
        for (uint32_t l = 1; l < levels; ++l)
        {
            bc::normalizeNormalMap(mips[l - 1].data(), std::max(1, texWidth >> l), std::max(1, texHeight >> l));
        }
    }

    texture::Recipe recipe;
    recipe.semantic = kind;
    recipe.target = texture::TargetProfile::AppleAstc;
    recipe.compress = params.compress && params.deviceSupportsAstc && !(kind == TextureKind::Normal && srgb);
    const texture::NativeFormat nativeFormat = texture::chooseNativeFormat(recipe);
    bool compressed = texture::formatInfo(nativeFormat).compressed;
    std::vector<std::vector<uint8_t>> payload;
    payload.reserve(levels);
    if (compressed)
    {
        std::vector<Rgba8LevelView> sourceLevels;
        sourceLevels.reserve(levels);
        sourceLevels.push_back({ base.get(), static_cast<size_t>(texWidth) * texHeight * 4 });
        for (uint32_t l = 1; l < levels; ++l)
        {
            sourceLevels.push_back({ mips[l - 1].data(), mips[l - 1].size() });
        }
        std::string error;
        if (!encodeAstc(sourceLevels, texWidth, texHeight, nativeFormat, srgb, payload, error))
        {
            STRELKA_WARNING("ASTC encoding failed for {}: {}; using RGBA8", fileName, error);
            compressed = false;
        }
    }
    if (!compressed)
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
    if (!compressed)
        out.pixelFormat = static_cast<uint32_t>(srgb ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm);
    else if (nativeFormat == texture::NativeFormat::ASTC4x4)
        out.pixelFormat = static_cast<uint32_t>(srgb ? MTL::PixelFormatASTC_4x4_sRGB : MTL::PixelFormatASTC_4x4_LDR);
    else
        out.pixelFormat = static_cast<uint32_t>(srgb ? MTL::PixelFormatASTC_6x6_sRGB : MTL::PixelFormatASTC_6x6_LDR);
    const texture::FormatInfo formatInfo = texture::formatInfo(nativeFormat);
    out.blockExtent = compressed ? formatInfo.blockExtent : 0u;
    out.blockBytes = compressed ? formatInfo.bytesPerBlock : 0u;
    out.data = std::move(payload);
    out.fromCache = false;
    out.valid = true;
    return out;
}

bool MetalTextures::prewarmStep(double budgetMs)
{
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
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sliceStart).count() < budgetMs))
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
            const Payload& payload = results[i];
            if (!payload.valid)
                continue;

            const Request& request = mPrewarmQueue[begin + i];
            const std::string cacheFile =
                cacheDir.empty() ? std::string() : (cacheDir / mPrewarmKeys[begin + i]).string();
            MTL::Texture* texture = createFromPayload(payload, payload.fromCache ? std::string() : cacheFile);
            if (!texture)
                continue;

            (payload.fromCache ? mCacheHits : mCacheMisses)++;
            mDedupCache.emplace(materialTextureKey(request.path, request.srgb, request.kind), texture);
            mMaterialTextures.push_back(texture);
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

    const Payload payload = decodeToPayload(fileName, srgb, kind, cacheFile, readDecodeParams());
    if (!payload.valid)
    {
        STRELKA_ERROR("Unable to load texture from file: {}", fileName.c_str());
        return nullptr;
    }
    (payload.fromCache ? mCacheHits : mCacheMisses)++;
    return createFromPayload(payload, payload.fromCache ? std::string() : cacheFile);
}

MTL::Texture* MetalTextures::loadProjectorFromFile(const std::string& fileName)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    const std::string extension = fs::path(fileName).extension().string();
    const bool isExr = extension == ".exr" || extension == ".EXR";
    const bool isFloat = isExr || stbi_is_hdr(fileName.c_str());
    float* linear = nullptr;
    // stb owns this mutable allocation until stbi_image_free.
    // NOLINTNEXTLINE(misc-const-correctness)
    stbi_uc* encoded = nullptr;

    if (isExr)
    {
        const char* error = nullptr;
        if (LoadEXR(&linear, &width, &height, fileName.c_str(), &error) != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to load EXR projector image: {} ({})", fileName, error ? error : "unknown");
            if (error)
            {
                FreeEXRErrorMessage(error);
            }
        }
    }
    else if (isFloat)
    {
        linear = stbi_loadf(fileName.c_str(), &width, &height, &channels, 4);
    }
    else
    {
        encoded = stbi_load(fileName.c_str(), &width, &height, &channels, 4);
    }
    if ((!linear && !encoded) || width <= 0 || height <= 0)
    {
        STRELKA_ERROR("Unable to load projector image from file: {}", fileName);
        if (isExr)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
            free(linear);
        }
        else
        {
            stbi_image_free(linear ? static_cast<void*>(linear) : static_cast<void*>(encoded));
        }
        return nullptr;
    }

    if (linear)
    {
        projector::sanitizeLinearRgba(linear, static_cast<size_t>(width) * static_cast<size_t>(height));
    }
    MTL::TextureDescriptor* descriptor = MTL::TextureDescriptor::alloc()->init();
    descriptor->setWidth(static_cast<NS::UInteger>(width));
    descriptor->setHeight(static_cast<NS::UInteger>(height));
    descriptor->setMipmapLevelCount(1u);
    descriptor->setPixelFormat(isFloat ? MTL::PixelFormatRGBA32Float : MTL::PixelFormatRGBA8Unorm_sRGB);
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setStorageMode(MTL::StorageModeShared);
    descriptor->setUsage(MTL::TextureUsageShaderRead);
    MTL::Texture* texture = mDevice ? mDevice->newTexture(descriptor) : nullptr;
    descriptor->release();
    if (texture)
    {
        const size_t rowBytes = static_cast<size_t>(width) * (isFloat ? 4u * sizeof(float) : 4u);
        texture->replaceRegion(MTL::Region::Make2D(0, 0, width, height), 0,
                               linear ? static_cast<const void*>(linear) : static_cast<const void*>(encoded), rowBytes);
    }
    if (isExr)
    {
        // LoadEXR allocates through malloc.
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(linear);
    }
    else
    {
        stbi_image_free(linear ? static_cast<void*>(linear) : static_cast<void*>(encoded));
    }
    return texture;
}

MTL::ResourceID MetalTextures::loadMaterialTexture(const std::string& absolutePath, bool srgb, TextureKind kind)
{
    if (absolutePath.empty())
        return MTL::ResourceID{};

    const std::string key = materialTextureKey(absolutePath, srgb, kind);
    if (auto it = mDedupCache.find(key); it != mDedupCache.end())
        return it->second->gpuResourceID();

    MTL::Texture* tex = loadFromFile(absolutePath, srgb, kind);
    if (!tex)
        return MTL::ResourceID{};

    mDedupCache[key] = tex;
    mMaterialTextures.push_back(tex);
    return tex->gpuResourceID();
}

} // namespace oka::metal
