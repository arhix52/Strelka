#pragma once

// The OptiX material-texture pipeline: decode, resample, mip, block compress,
// cache, upload.
//
// What it replaces: a `stbi_load(..., STBI_rgb_alpha)` into an 8-bit RGBA array
// with `cudaReadModeNormalizedFloat` and no transfer function, which meant every
// base-colour map in every scene was shaded with its sRGB *encoding* as though
// it were radiance. That is a ~1.12x error on a mid-grey and it is the whole of
// what `01_srgb_texture` was measuring.
//
// The arithmetic -- extent, level count, format choice, byte counts -- is in
// texture_upload_plan.h with no CUDA in it, so it has unit tests. This header is
// the part that needs a device.
//
// The block encoder and the cache key are shared with the Metal backend rather
// than reimplemented: both are host-side, backend-neutral and already covered by
// tests. Only the cache *payload* differs (CUDA arrays want channel kinds, not
// MTLPixelFormats), which is what the "optix|v<N>|..." prefix `cacheKey()` puts
// in front of the file name separates -- the two backends can share one cache
// directory and will never hash to each other's files.

#include <cuda.h>
#include <cuda_runtime.h>

#include "texture_upload_plan.h"

// Host-side and backend-neutral despite where they live; see the note above.
#include "../metal/texture_cache_key.h"
#include "../metal/texture_compress.h"

// stb_image.h and stb_image_resize.h are deliberately NOT included here. Their
// implementation blocks sit outside their include guards, so a second
// `#include` while STB_*_IMPLEMENTATION is still defined re-expands the whole
// implementation and the translation unit stops compiling. The one unit that
// includes this header (OptixRender.cpp) includes them first, with the
// implementation macros, and that is the contract.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Custom structure representing a CUDA texture.
//
// The unfiltered (point-sampled, border-addressed) object that used to sit
// beside this one is gone. It was created for every texture in the scene,
// tracked, destroyed -- and never bound to anything, because nothing in the
// shading path samples a material texture unfiltered. It was a texture object
// and a header field per texture for nothing.
struct Texture
{
    Texture() = default;

    Texture(cudaTextureObject_t filtered, uint3 dimensions, uint32_t mipLevels)
        : filtered_object(filtered)
        , size(dimensions)
        , levels(mipLevels)
    {
    }

    cudaTextureObject_t filtered_object = 0; // filter mode cudaFilterModeLinear
    uint3 size = make_uint3(0, 0, 0); // size of the texture, level 0
    uint32_t levels = 0; // mip levels resident, 1 when there is no chain
};

namespace oka
{
namespace optix_tex
{

// Bumped whenever the payload this backend writes changes meaning. It rides in
// the cache key, so a bump orphans the old files rather than misreading them.
inline constexpr uint32_t kOptixPayloadVersion = 1;

/// One decoded, resampled, mipped and possibly compressed texture, in the form
/// the CUDA array wants and in the form the cache file holds.
struct Payload
{
    Plan plan;
    std::vector<std::vector<uint8_t>> levels;
    bool valid = false;
    bool fromCache = false;
};

namespace detail
{

struct CachedHeader
{
    char magic[4]; // "OTEX"
    uint32_t payloadVersion;
    uint32_t width;
    uint32_t height;
    uint32_t levels;
    uint32_t format; // optix_tex::Format
    uint32_t flags; // bit 0 srgbTextureFlag, bit 1 srgbBlockFormat
    uint32_t reserved;
};

// The header is written to disk verbatim, so its layout is part of the file
// format. If padding ever appears the version must move with it.
static_assert(sizeof(CachedHeader) == 32, "cache header layout changed; bump kOptixPayloadVersion");

// Kind is cast to oka::metal::TextureKind to build the shared cache key. The
// two enumerations agreeing is what keeps a colour texture from being keyed as
// a normal map.
static_assert((int)Kind::Color == (int)oka::metal::TextureKind::Color);
static_assert((int)Kind::NonColor == (int)oka::metal::TextureKind::NonColor);
static_assert((int)Kind::Normal == (int)oka::metal::TextureKind::Normal);

inline uint32_t packFlags(const Plan& plan)
{
    return (plan.srgbTextureFlag ? 1u : 0u) | (plan.srgbBlockFormat ? 2u : 0u);
}

/// Widen a decoded RGBA buffer of `srcChannels` to the four channels the CUDA
/// array carries. stb already does this for us; kept as a named no-op so the
/// call sites read the same for all three element types.
template <typename T>
inline std::vector<uint8_t> toBytes(const T* src, size_t count)
{
    std::vector<uint8_t> out(count * sizeof(T));
    std::memcpy(out.data(), src, out.size());
    return out;
}

/// sRGB -> linear on 16-bit samples. CUDA's `cudaTextureDesc::sRGB` is defined
/// for 8-bit unorm formats only, so a 16-bit colour map has to be linearised
/// before it is uploaded. 16 bits is enough headroom that doing it here costs
/// nothing visible -- the alternative, quantising to 8 bits so the hardware can
/// do it, would throw away exactly what a 16-bit source was chosen for.
inline void linearizeSrgb16(uint16_t* rgba, size_t texels)
{
    for (size_t i = 0; i < texels; ++i)
    {
        for (int c = 0; c < 3; ++c) // alpha is not encoded
        {
            const float s = (float)rgba[i * 4 + c] / 65535.0f;
            const float lin = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
            rgba[i * 4 + c] = (uint16_t)std::lround(std::clamp(lin, 0.0f, 1.0f) * 65535.0f);
        }
    }
}

inline bool resampleLevel(const Plan& plan,
                          const uint8_t* src,
                          int srcW,
                          int srcH,
                          uint8_t* dst,
                          int dstW,
                          int dstH)
{
    switch (plan.format)
    {
    case Format::RGBA32F:
        return stbir_resize_float((const float*)src, srcW, srcH, 0, (float*)dst, dstW, dstH, 0, 4) != 0;
    case Format::RGBA16:
        return stbir_resize_uint16_generic((const uint16_t*)src, srcW, srcH, 0, (uint16_t*)dst, dstW, dstH, 0, 4,
                                           STBIR_ALPHA_CHANNEL_NONE, 0, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT,
                                           STBIR_COLORSPACE_LINEAR, nullptr) != 0;
    default:
        // 8-bit, compressed or not: the bytes being resampled are still RGBA8.
        return (plan.resampleInSrgb ? stbir_resize_uint8_srgb(src, srcW, srcH, 0, dst, dstW, dstH, 0, 4, 3, 0)
                                    : stbir_resize_uint8(src, srcW, srcH, 0, dst, dstW, dstH, 0, 4)) != 0;
    }
}

inline size_t rawTexelBytes(const Plan& plan)
{
    switch (plan.format)
    {
    case Format::RGBA32F:
        return 16;
    case Format::RGBA16:
        return 8;
    default:
        return 4; // BC formats are encoded from RGBA8
    }
}

inline oka::bc::Format bcFormat(Format f)
{
    switch (f)
    {
    case Format::BC3:
        return oka::bc::Format::BC3;
    case Format::BC5:
        return oka::bc::Format::BC5;
    default:
        return oka::bc::Format::BC1;
    }
}

inline cudaChannelFormatDesc channelDesc(const Plan& plan)
{
    switch (plan.format)
    {
    case Format::RGBA32F:
        return cudaCreateChannelDesc<float4>();
    case Format::RGBA16:
        return cudaCreateChannelDesc<ushort4>();
    case Format::BC1:
        return plan.srgbBlockFormat ?
                   cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1SRGB>() :
                   cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1>();
    case Format::BC3:
        return plan.srgbBlockFormat ?
                   cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3SRGB>() :
                   cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3>();
    case Format::BC5:
        return cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed5>();
    default:
        return cudaCreateChannelDesc<uchar4>();
    }
}

/// Bytes per row of a level, as `cudaMemcpy2DToArray` counts them: whole rows of
/// texels for a linear format, whole rows of 4x4 blocks for a compressed one.
inline size_t levelPitch(const Plan& plan, int width)
{
    const size_t w = (size_t)std::max(1, width);
    if (isCompressed(plan.format))
        return ((w + 3) / 4) * blockBytes(plan.format);
    return w * texelBytes(plan.format);
}

/// Rows a level occupies, as `cudaMemcpy2DToArray` counts them.
inline size_t levelRows(const Plan& plan, int height)
{
    const size_t h = (size_t)std::max(1, height);
    if (isCompressed(plan.format))
        return (h + 3) / 4;
    return h;
}

} // namespace detail

/// The cache file name. Salted so that the Metal backend's payload for the same
/// source file, settings and kind hashes somewhere else: the two encode into
/// different containers and would otherwise collide in a shared cache directory.
inline std::string cacheKey(const std::string& fileName,
                            Kind kind,
                            uint32_t maxDimension,
                            uint32_t downscale,
                            bool blockCompress,
                            bool wantMips)
{
    std::error_code ec;
    namespace fs = std::filesystem;
    const auto size = fs::file_size(fileName, ec);
    const auto stamp = fs::last_write_time(fileName, ec).time_since_epoch().count();

    oka::metal::TextureCacheKeyInputs in;
    // The salt goes in the file name field rather than in a new struct member so
    // that texture_cache_key.h -- which the Metal backend owns and this machine
    // cannot test -- does not have to change to carry a second backend.
    in.fileName = "optix|v" + std::to_string(kOptixPayloadVersion) + (blockCompress ? "|bc" : "|raw") +
                  (wantMips ? "|mips" : "|lod0") + "|" + fileName;
    in.fileSize = ec ? 0 : (uint64_t)size;
    in.writeTimeCount = ec ? 0 : (int64_t)stamp;
    in.maxDimension = maxDimension;
    in.downscale = downscale;
    in.srgb = kind == Kind::Color;
    in.kind = (oka::metal::TextureKind)(int)kind;
    return oka::metal::textureCacheKey(in);
}

inline Payload readCachedPayload(const std::string& cachePath)
{
    Payload payload;
    if (cachePath.empty())
        return payload;
    std::ifstream in(cachePath, std::ios::binary);
    if (!in)
        return payload;
    detail::CachedHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || std::memcmp(header.magic, "OTEX", 4) != 0 || header.payloadVersion != kOptixPayloadVersion)
        return payload;
    payload.plan.extent = Extent{ (int)header.width, (int)header.height };
    payload.plan.levels = header.levels;
    payload.plan.format = (Format)header.format;
    payload.plan.srgbTextureFlag = (header.flags & 1u) != 0;
    payload.plan.srgbBlockFormat = (header.flags & 2u) != 0;
    payload.levels.reserve(header.levels);
    for (uint32_t l = 0; l < header.levels; ++l)
    {
        uint32_t byteLength = 0;
        in.read(reinterpret_cast<char*>(&byteLength), sizeof(byteLength));
        if (!in || byteLength == 0 || byteLength != payload.plan.levelBytes(l))
            return Payload{};
        std::vector<uint8_t> level(byteLength);
        in.read(reinterpret_cast<char*>(level.data()), byteLength);
        if (!in)
            return Payload{};
        payload.levels.push_back(std::move(level));
    }
    payload.fromCache = true;
    payload.valid = true;
    return payload;
}

inline void writeCachedPayload(const Payload& payload, const std::string& cachePath)
{
    if (cachePath.empty() || !payload.valid)
        return;
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(fs::path(cachePath).parent_path(), ec);
    // Through a temporary and a rename: a half-written file that still has a
    // valid header reads back as a texture made of rubbish, which is far worse
    // than a cache miss.
    const std::string tmp = cachePath + ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out)
        return;
    detail::CachedHeader header{};
    std::memcpy(header.magic, "OTEX", 4);
    header.payloadVersion = kOptixPayloadVersion;
    header.width = (uint32_t)payload.plan.extent.width;
    header.height = (uint32_t)payload.plan.extent.height;
    header.levels = payload.plan.levels;
    header.format = (uint32_t)payload.plan.format;
    header.flags = detail::packFlags(payload.plan);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (uint32_t l = 0; l < payload.plan.levels && l < payload.levels.size(); ++l)
    {
        const uint32_t byteLength = (uint32_t)payload.levels[l].size();
        out.write(reinterpret_cast<const char*>(&byteLength), sizeof(byteLength));
        out.write(reinterpret_cast<const char*>(payload.levels[l].data()), byteLength);
    }
    out.close();
    if (!out)
    {
        fs::remove(tmp, ec);
        return;
    }
    fs::rename(tmp, cachePath, ec);
}

/// Decode `fileName` into the exact bytes the CUDA array will hold. No CUDA, no
/// shared state: safe to call from any thread.
inline Payload decodeToPayload(const std::string& fileName, Kind kind, const DecodeSettings& settings)
{
    PlanInputs planIn;
    planIn.maxDimension = settings.maxDimension;
    planIn.downscale = settings.downscale;
    planIn.kind = kind;
    planIn.blockCompress = settings.blockCompress;
    planIn.wantMips = settings.wantMips;

    // What the file is decides the element type. An .hdr or .exr used as a
    // material map keeps its range; a 16-bit PNG keeps its precision. The old
    // loader forced everything through STBI_rgb_alpha and therefore through 8
    // bits, which is a silent clip on the first and a silent quantise on the
    // second.
    planIn.sourceIsFloat = stbi_is_hdr(fileName.c_str()) != 0;
    planIn.sourceIs16Bit = !planIn.sourceIsFloat && stbi_is_16_bit(fileName.c_str()) != 0;

    int srcW = 0;
    int srcH = 0;
    int srcChannels = 0;
    std::vector<uint8_t> base;
    if (planIn.sourceIsFloat)
    {
        float* data = stbi_loadf(fileName.c_str(), &srcW, &srcH, &srcChannels, 4);
        if (data == nullptr)
            return Payload{};
        base = detail::toBytes(data, (size_t)srcW * srcH * 4);
        stbi_image_free(data);
    }
    else if (planIn.sourceIs16Bit)
    {
        stbi_us* data = stbi_load_16(fileName.c_str(), &srcW, &srcH, &srcChannels, 4);
        if (data == nullptr)
            return Payload{};
        if (kind == Kind::Color)
            detail::linearizeSrgb16(data, (size_t)srcW * srcH);
        base = detail::toBytes(data, (size_t)srcW * srcH * 4);
        stbi_image_free(data);
    }
    else
    {
        stbi_uc* data = stbi_load(fileName.c_str(), &srcW, &srcH, &srcChannels, STBI_rgb_alpha);
        if (data == nullptr)
            return Payload{};
        planIn.hasAlpha = oka::bc::hasAlpha(data, srcW, srcH);
        base = detail::toBytes(data, (size_t)srcW * srcH * 4);
        stbi_image_free(data);
    }
    if (srcW <= 0 || srcH <= 0)
        return Payload{};

    planIn.srcWidth = srcW;
    planIn.srcHeight = srcH;
    Plan plan = planTexture(planIn);

    const size_t texelBytesRaw = detail::rawTexelBytes(plan);

    // Resample to the planned extent, if it is not what the file holds.
    if (plan.extent.width != srcW || plan.extent.height != srcH)
    {
        std::vector<uint8_t> scaled((size_t)plan.extent.width * plan.extent.height * texelBytesRaw);
        if (detail::resampleLevel(plan, base.data(), srcW, srcH, scaled.data(), plan.extent.width,
                                  plan.extent.height))
        {
            base = std::move(scaled);
        }
        else
        {
            plan.extent = Extent{ srcW, srcH };
            plan.levels = planIn.wantMips ? mipLevelCount(plan.format, srcW, srcH) : 1u;
        }
    }

    // The uncompressed chain, in the raw element type.
    std::vector<std::vector<uint8_t>> chain;
    chain.reserve(plan.levels);
    chain.push_back(std::move(base));
    for (uint32_t l = 1; l < plan.levels; ++l)
    {
        const int prevW = std::max(1, plan.extent.width >> (l - 1));
        const int prevH = std::max(1, plan.extent.height >> (l - 1));
        const int w = std::max(1, plan.extent.width >> l);
        const int h = std::max(1, plan.extent.height >> l);
        std::vector<uint8_t> next((size_t)w * h * texelBytesRaw);
        if (!detail::resampleLevel(plan, chain[l - 1].data(), prevW, prevH, next.data(), w, h))
        {
            plan.levels = l;
            break;
        }
        chain.push_back(std::move(next));
    }

    // Every level, not only the base: the box filter above averages unit vectors,
    // which shortens them, and the shader rebuilds Z assuming they are unit.
    // Done whether or not the texture ends up compressed, so that turning
    // compression on does not change the shading.
    if (plan.normalizeLevels && plan.format != Format::RGBA32F && plan.format != Format::RGBA16)
    {
        for (uint32_t l = 0; l < plan.levels; ++l)
        {
            oka::bc::normalizeNormalMap(chain[l].data(), std::max(1, plan.extent.width >> l),
                                        std::max(1, plan.extent.height >> l));
        }
    }

    Payload payload;
    payload.plan = plan;
    payload.levels.reserve(plan.levels);
    for (uint32_t l = 0; l < plan.levels; ++l)
    {
        const int w = std::max(1, plan.extent.width >> l);
        const int h = std::max(1, plan.extent.height >> l);
        payload.levels.push_back(isCompressed(plan.format) ?
                                     oka::bc::compressImage(chain[l].data(), w, h, detail::bcFormat(plan.format)) :
                                     std::move(chain[l]));
    }
    payload.valid = true;
    return payload;
}

/// What `createTexture` allocated, so the renderer can free it.
struct TextureResources
{
    cudaArray_t array = nullptr;
    cudaMipmappedArray_t mipmapped = nullptr;
    cudaTextureObject_t object = 0;
};

/// Upload a payload and build its texture object.
///
/// `addressMode` is a parameter rather than a hardcoded `cudaAddressModeWrap`
/// because glTF samplers carry `wrapS`/`wrapT`. The scene loader does not read
/// them yet -- see the hand-off in the report -- so every caller passes wrap
/// today, which is glTF's default; the plumbing is here so that adding it is a
/// loader change and not a renderer change.
inline TextureResources createTexture(const Payload& payload,
                                      cudaTextureAddressMode addressModeU = cudaAddressModeWrap,
                                      cudaTextureAddressMode addressModeV = cudaAddressModeWrap)
{
    TextureResources out;
    if (!payload.valid || payload.levels.empty())
        return out;

    const Plan& plan = payload.plan;
    const cudaChannelFormatDesc channel = detail::channelDesc(plan);

    cudaResourceDesc resDesc{};
    if (plan.levels > 1)
    {
        const cudaExtent extent = make_cudaExtent((size_t)plan.extent.width, (size_t)plan.extent.height, 0);
        if (cudaMallocMipmappedArray(&out.mipmapped, &channel, extent, plan.levels) != cudaSuccess)
            return TextureResources{};
        for (uint32_t l = 0; l < plan.levels && l < payload.levels.size(); ++l)
        {
            cudaArray_t levelArray = nullptr;
            if (cudaGetMipmappedArrayLevel(&levelArray, out.mipmapped, l) != cudaSuccess)
            {
                cudaFreeMipmappedArray(out.mipmapped);
                return TextureResources{};
            }
            const int w = std::max(1, plan.extent.width >> l);
            const int h = std::max(1, plan.extent.height >> l);
            const size_t pitch = detail::levelPitch(plan, w);
            if (cudaMemcpy2DToArray(levelArray, 0, 0, payload.levels[l].data(), pitch, pitch,
                                    detail::levelRows(plan, h), cudaMemcpyHostToDevice) != cudaSuccess)
            {
                cudaFreeMipmappedArray(out.mipmapped);
                return TextureResources{};
            }
        }
        resDesc.resType = cudaResourceTypeMipmappedArray;
        resDesc.res.mipmap.mipmap = out.mipmapped;
    }
    else
    {
        if (cudaMallocArray(&out.array, &channel, (size_t)plan.extent.width, (size_t)plan.extent.height) !=
            cudaSuccess)
            return TextureResources{};
        const size_t pitch = detail::levelPitch(plan, plan.extent.width);
        if (cudaMemcpy2DToArray(out.array, 0, 0, payload.levels[0].data(), pitch, pitch,
                                detail::levelRows(plan, plan.extent.height),
                                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFreeArray(out.array);
            return TextureResources{};
        }
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = out.array;
    }

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = addressModeU;
    texDesc.addressMode[1] = addressModeV;
    texDesc.addressMode[2] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    // A float array is read as it was written; everything else is a normalized
    // integer format and comes back in [0,1].
    texDesc.readMode = plan.format == Format::RGBA32F ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    // The whole point of the exercise: the hardware does the sRGB decode on the
    // three colour channels at fetch, before filtering, which is where it
    // belongs.
    //
    // The flag is set for the block-compressed sRGB channel kinds too, and that
    // is not redundancy: CUDA 13 rejects `cudaCreateTextureObject` with
    // cudaErrorInvalidValue if a *SRGB block format is described by a texture
    // whose sRGB flag is clear -- the two have to agree. Measured, not assumed;
    // the same call succeeds the moment the flag goes on.
    texDesc.sRGB = (plan.srgbTextureFlag || plan.srgbBlockFormat) ? 1 : 0;
    if (plan.levels > 1)
    {
        texDesc.mipmapFilterMode = cudaFilterModeLinear;
        texDesc.maxAnisotropy = 16;
        texDesc.minMipmapLevelClamp = 0.0f;
        texDesc.maxMipmapLevelClamp = (float)(plan.levels - 1);
    }

    if (cudaCreateTextureObject(&out.object, &resDesc, &texDesc, nullptr) != cudaSuccess)
    {
        if (out.mipmapped)
            cudaFreeMipmappedArray(out.mipmapped);
        if (out.array)
            cudaFreeArray(out.array);
        return TextureResources{};
    }
    return out;
}

} // namespace optix_tex
} // namespace oka
