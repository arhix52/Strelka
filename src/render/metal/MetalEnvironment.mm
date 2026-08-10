#include "MetalEnvironment.h"
#include "ibl_alias_table.h"

#include "ShaderTypes.h"

#include <log.h>

#include <filesystem>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

namespace fs = std::filesystem;

namespace oka
{
namespace metal
{

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
    mState.aliasBuffer = mDevice->newBuffer(sizeof(EnvAliasEntry), MTL::ResourceStorageModeManaged);
}

void MetalEnvironment::loadBackground(const std::string& texturePath)
{
    if (mState.backgroundTexture)
    {
        mState.backgroundTexture->release();
        mState.backgroundTexture = nullptr;
    }

    int width = 0, height = 0;
    float* pixelData = nullptr;
    bool isExr = false;

    const std::string ext = fs::path(texturePath).extension().string();
    if (ext == ".exr" || ext == ".EXR")
    {
        const char* err = nullptr;
        if (LoadEXR(&pixelData, &width, &height, texturePath.c_str(), &err) != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to load EXR env background: {} ({})", texturePath, err ? err : "unknown");
            if (err)
                FreeEXRErrorMessage(err);
            return;
        }
        isExr = true;
    }
    else
    {
        int channels = 0;
        pixelData = stbi_loadf(texturePath.c_str(), &width, &height, &channels, 4);
        if (!pixelData)
        {
            STRELKA_ERROR("Failed to load env background: {}", texturePath);
            return;
        }
    }

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setPixelFormat(MTL::PixelFormatRGBA32Float);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModeManaged);
    desc->setUsage(MTL::TextureUsageShaderRead);
    mState.backgroundTexture = mDevice->newTexture(desc);
    desc->release();
    mState.backgroundTexture->replaceRegion(MTL::Region::Make3D(0, 0, 0, width, height, 1), 0, pixelData,
                                            width * sizeof(float) * 4);
    if (isExr)
        free(pixelData);
    else
        stbi_image_free(pixelData);

    STRELKA_INFO("Loaded env background: {} ({}x{})", texturePath, width, height);
}

void MetalEnvironment::loadMap(const std::string& texturePath)
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
    mState.loaded = false;

    int width = 0, height = 0;
    float* pixelData = nullptr;
    bool isExr = false;

    const std::string ext = fs::path(texturePath).extension().string();
    if (ext == ".exr" || ext == ".EXR")
    {
        const char* err = nullptr;
        int ret = LoadEXR(&pixelData, &width, &height, texturePath.c_str(), &err);
        if (ret != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to load EXR env map: {} ({})", texturePath, err ? err : "unknown");
            if (err)
                FreeEXRErrorMessage(err);
            return;
        }
        isExr = true;
    }
    else
    {
        int channels = 0;
        pixelData = stbi_loadf(texturePath.c_str(), &width, &height, &channels, 4);
        if (!pixelData)
        {
            STRELKA_ERROR("Failed to load env map: {}", texturePath);
            return;
        }
    }

    STRELKA_INFO("Loaded env map: {} ({}x{})", texturePath, width, height);

    MTL::TextureDescriptor* pTextureDesc = MTL::TextureDescriptor::alloc()->init();
    pTextureDesc->setWidth(width);
    pTextureDesc->setHeight(height);
    pTextureDesc->setPixelFormat(MTL::PixelFormatRGBA32Float);
    pTextureDesc->setTextureType(MTL::TextureType2D);
    pTextureDesc->setStorageMode(MTL::StorageModeManaged);
    pTextureDesc->setUsage(MTL::TextureUsageShaderRead);

    mState.mapTexture = mDevice->newTexture(pTextureDesc);
    pTextureDesc->release();

    const MTL::Region region = MTL::Region::Make3D(0, 0, 0, width, height, 1);
    mState.mapTexture->replaceRegion(region, 0, pixelData, width * sizeof(float) * 4);

    const auto aliasResult = buildIblAliasTable(pixelData, width, height);
    static_assert(sizeof(EnvAliasEntry) == sizeof(metal::EnvAliasEntry),
                  "host EnvAliasEntry must match ShaderTypes EnvAliasEntry");
    mState.pdfScale = aliasResult.envPdfScale;

    mState.aliasBuffer = mDevice->newBuffer(aliasResult.alias.data(),
                                            aliasResult.alias.size() * sizeof(EnvAliasEntry),
                                            MTL::ResourceStorageModeManaged);

    if (isExr)
        free(pixelData);
    else
        stbi_image_free(pixelData);

    const float avgWeightedLum = (float)(aliasResult.totalPower / (double)(width * height));
    const bool autoCalibrate = mSettings->getAs<bool>("render/env/autoCalibrate");
    const float kCalibrationTarget = 1000.0f;
    mState.autoScale = (autoCalibrate && avgWeightedLum > 1e-6f) ? kCalibrationTarget / avgWeightedLum : 1.0f;
    mState.loaded = true;

    STRELKA_INFO(
        "Env map alias table built: {} texels ({:.1f} MB), total power: {:.1f}, avgLum: {:.4f}, autoScale: {:.1f}",
        aliasResult.alias.size(), aliasResult.alias.size() * sizeof(EnvAliasEntry) / (1024.0 * 1024.0),
        aliasResult.totalPower, avgWeightedLum, mState.autoScale);
}

} // namespace metal
} // namespace oka
