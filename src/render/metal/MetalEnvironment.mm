#include "MetalEnvironment.h"
#include "ibl_alias_table.h"

#include "ShaderTypes.h"

#include <log.h>

#include <cstdlib>
#include <filesystem>

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

MTL::Texture* uploadFloatTexture(MTL::Device* device, const FloatImage& image)
{
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    MTL::Texture* texture = nullptr;

    desc->setWidth(image.width);
    desc->setHeight(image.height);
    desc->setPixelFormat(MTL::PixelFormatRGBA32Float);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageShaderRead);
    texture = device->newTexture(desc);
    desc->release();
    texture->replaceRegion(
        MTL::Region::Make3D(0, 0, 0, image.width, image.height, 1), 0, image.pixels, image.width * sizeof(float) * 4);
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
    if (mState.mapTexture)
    {
        mState.mapTexture->release();
        mState.mapTexture = nullptr;
    }
    if (mState.backgroundTexture)
    {
        mState.backgroundTexture->release();
        mState.backgroundTexture = nullptr;
    }
    if (mState.aliasBuffer)
    {
        mState.aliasBuffer->release();
        mState.aliasBuffer = nullptr;
    }
    mState.pdfScale = 0.0f;
    mState.autoScale = 1.0f;
    mState.loaded = false;
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

    if (mState.backgroundTexture)
    {
        mState.backgroundTexture->release();
        mState.backgroundTexture = nullptr;
    }
    if (!loadFloatImage(texturePath, "env background", image))
    {
        return;
    }

    mState.backgroundTexture = uploadFloatTexture(mDevice, image);
    releaseFloatImage(image);
    STRELKA_INFO("Loaded env background: {} ({}x{})", texturePath, image.width, image.height);
}

void MetalEnvironment::loadMap(const std::string& texturePath)
{
    FloatImage image;

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
    mState.loaded = false;
    if (!loadFloatImage(texturePath, "env map", image))
    {
        return;
    }

    STRELKA_INFO("Loaded env map: {} ({}x{})", texturePath, image.width, image.height);
    mState.mapTexture = uploadFloatTexture(mDevice, image);

    const auto aliasResult = buildIblAliasTable(image.pixels, image.width, image.height);
    static_assert(sizeof(EnvAliasEntry) == sizeof(metal::EnvAliasEntry),
                  "host EnvAliasEntry must match ShaderTypes EnvAliasEntry");
    mState.pdfScale = aliasResult.envPdfScale;

    mState.aliasBuffer = mDevice->newBuffer(aliasResult.alias.data(),
                                            aliasResult.alias.size() * sizeof(EnvAliasEntry),
                                            MTL::ResourceStorageModeShared);

    releaseFloatImage(image);

    const float avgWeightedLum = (float)(aliasResult.totalPower / (double)(image.width * image.height));
    const bool autoCalibrate = mSettings->getAs<bool>("render/env/autoCalibrate");
    const float kCalibrationTarget = 1000.0f;
    mState.autoScale = (autoCalibrate && avgWeightedLum > 1e-6f) ? kCalibrationTarget / avgWeightedLum : 1.0f;
    mState.loaded = true;

    STRELKA_INFO(
        "Env map alias table built: {} texels ({:.1f} MB), total power: {:.1f}, avgLum: {:.4f}, autoScale: {:.1f}",
        aliasResult.alias.size(), aliasResult.alias.size() * sizeof(EnvAliasEntry) / (1024.0 * 1024.0),
        aliasResult.totalPower, avgWeightedLum, mState.autoScale);
}

} // namespace oka::metal

