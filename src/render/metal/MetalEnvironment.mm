#include "MetalEnvironment.h"
#include <host/ibl_alias_table.h>

#include "ShaderTypes.h"

#include <log.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <vector>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

namespace fs = std::filesystem;


namespace oka::metal
{
namespace
{
struct FloatImage
{
    float* pixels = nullptr;
    int width = 0;
    int height = 0;
    bool isExr = false;
};

bool loadFloatImage(const std::string& texturePath, const char* kind, FloatImage& image)
{
    const std::string ext = fs::path(texturePath).extension().string();
    const char* error = nullptr;
    int channels = 0;

    image.isExr = ext == ".exr" || ext == ".EXR";
    if (image.isExr)
    {
        if (LoadEXR(&image.pixels, &image.width, &image.height, texturePath.c_str(), &error) != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to load EXR {}: {} ({})", kind, texturePath, error ? error : "unknown");
            if (error)
            {
                FreeEXRErrorMessage(error);
            }
            return false;
        }
        return true;
    }

    image.pixels = stbi_loadf(texturePath.c_str(), &image.width, &image.height, &channels, 4);
    if (!image.pixels)
    {
        STRELKA_ERROR("Failed to load {}: {}", kind, texturePath);
        return false;
    }
    return true;
}

void releaseFloatImage(FloatImage& image)
{
    if (image.isExr)
    {
        // LoadEXR allocates with malloc, so this has to be free.
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(image.pixels);
    }
    else
    {
        stbi_image_free(image.pixels);
    }
    image.pixels = nullptr;
}

MTL::Texture* uploadHalfFloatTexture(MTL::Device* device, const FloatImage& image, float& decodeScale)
{
    constexpr float kHalfMax = 65504.0f;
    const size_t texelCount = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
    float maxChannel = 0.0f;
    for (size_t i = 0; i < texelCount; ++i)
    {
        const float* pixel = image.pixels + 4u * i;
        maxChannel = std::max({ maxChannel, pixel[0], pixel[1], pixel[2] });
    }
    decodeScale = std::max(maxChannel / kHalfMax, 1.0f);

    static_assert(sizeof(_Float16) == 2);
    std::vector<_Float16> halfPixels(texelCount * 4u);
    for (size_t i = 0; i < texelCount; ++i)
    {
        const float* source = image.pixels + 4u * i;
        _Float16* destination = halfPixels.data() + 4u * i;
        destination[0] = static_cast<_Float16>(std::min(source[0] / decodeScale, kHalfMax));
        destination[1] = static_cast<_Float16>(std::min(source[1] / decodeScale, kHalfMax));
        destination[2] = static_cast<_Float16>(std::min(source[2] / decodeScale, kHalfMax));
        destination[3] = static_cast<_Float16>(1.0f);
    }

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    MTL::Texture* texture = nullptr;

    desc->setWidth(image.width);
    desc->setHeight(image.height);
    desc->setPixelFormat(MTL::PixelFormatRGBA16Float);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageShaderRead);
    texture = device->newTexture(desc);
    desc->release();
    texture->replaceRegion(MTL::Region::Make3D(0, 0, 0, image.width, image.height, 1), 0, halfPixels.data(),
                           image.width * sizeof(_Float16) * 4u);
    return texture;
}
} // namespace

MetalEnvironment::~MetalEnvironment()
{
    release();
}

void MetalEnvironment::init(MTL::Device* device, SettingsManager* settings)
{
    mDevice = device;
    mSettings = settings;
}

void MetalEnvironment::release()
{
    clearMap();
    clearBackground();
}

void MetalEnvironment::clearMap()
{
    if (mState.mapTexture)
    {
        mState.mapTexture->release();
        mState.mapTexture = nullptr;
    }
    if (mState.aliasBuffer)
    {
        mState.aliasBuffer->release();
        mState.aliasBuffer = nullptr;
    }
    mState.totalPower = 0.0;
    mState.autoScale = 1.0f;
    mState.mapDecodeScale = 1.0f;
    mState.aliasWidth = 0;
    mState.aliasHeight = 0;
    mState.loaded = false;
}

void MetalEnvironment::clearBackground()
{
    if (mState.backgroundTexture)
    {
        mState.backgroundTexture->release();
        mState.backgroundTexture = nullptr;
    }
    mState.backgroundDecodeScale = 1.0f;
}

void MetalEnvironment::ensurePlaceholderAliasBuffer()
{
    if (mState.aliasBuffer || !mDevice)
        return;
    mState.aliasBuffer = mDevice->newBuffer(sizeof(EnvAliasEntry), MTL::ResourceStorageModeShared);
}

void MetalEnvironment::loadBackground(const std::string& texturePath)
{
    FloatImage image;

    clearBackground();
    if (!loadFloatImage(texturePath, "env background", image))
    {
        return;
    }

    sanitizeEnvironmentPixels(image.pixels, image.width, image.height);
    mState.backgroundTexture = uploadHalfFloatTexture(mDevice, image, mState.backgroundDecodeScale);
    releaseFloatImage(image);
    STRELKA_INFO("Loaded env background: {} ({}x{})", texturePath, image.width, image.height);
}

void MetalEnvironment::loadMap(const std::string& texturePath)
{
    FloatImage image;

    clearMap();
    if (!loadFloatImage(texturePath, "env map", image))
    {
        return;
    }

    STRELKA_INFO("Loaded env map: {} ({}x{})", texturePath, image.width, image.height);
    sanitizeEnvironmentPixels(image.pixels, image.width, image.height);
    mState.mapTexture = uploadHalfFloatTexture(mDevice, image, mState.mapDecodeScale);

    constexpr int kMaxAliasWidth = 2048;
    constexpr int kMaxAliasHeight = 1024;
    const int aliasWidth = std::min(image.width, kMaxAliasWidth);
    const int aliasHeight = std::min(image.height, kMaxAliasHeight);
    const IblAliasTableResult aliasResult =
        buildDownsampledSolidAngleIblAliasTable(image.pixels, image.width, image.height, aliasWidth, aliasHeight);
    static_assert(sizeof(EnvAliasEntry) == sizeof(metal::EnvAliasEntry),
                  "host EnvAliasEntry must match ShaderTypes EnvAliasEntry");
    mState.totalPower = aliasResult.totalPower;
    mState.aliasWidth = static_cast<uint32_t>(aliasWidth);
    mState.aliasHeight = static_cast<uint32_t>(aliasHeight);

    mState.aliasBuffer = mDevice->newBuffer(
        aliasResult.alias.data(), aliasResult.alias.size() * sizeof(EnvAliasEntry), MTL::ResourceStorageModeShared);

    releaseFloatImage(image);

    // Preserve the existing calibration convention. The alias normalizer is
    // now the exact sphere integral, whereas calibration historically used the
    // average centre-Jacobian-weighted luminance.
    const float avgWeightedLum = static_cast<float>(aliasResult.averageWeightedLuminance);
    const bool autoCalibrate = mSettings->getAs<bool>("render/env/autoCalibrate");
    const float kCalibrationTarget = 1000.0f;
    mState.autoScale = (autoCalibrate && avgWeightedLum > 1e-6f) ? kCalibrationTarget / avgWeightedLum : 1.0f;
    mState.loaded = true;

    STRELKA_INFO(
        "Env map alias table built: {}x{} entries for {}x{} map ({:.1f} MB), total power: {:.1f}, avgLum: {:.4f}, "
        "autoScale: {:.1f}",
        aliasWidth, aliasHeight, image.width, image.height,
        aliasResult.alias.size() * sizeof(EnvAliasEntry) / (1024.0 * 1024.0), aliasResult.totalPower, avgWeightedLum,
        mState.autoScale);
}

} // namespace oka::metal
