#include "MetalEnvironment.h"
#include <host/ibl_alias_table.h>

#include <log.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <numbers>
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

struct EnvRowCosBounds
{
    float north;
    float south;
};
static_assert(sizeof(EnvRowCosBounds) == 8);

struct EnvAliasSelection
{
    uint32_t threshold;
    uint32_t alias;
};
static_assert(sizeof(EnvAliasSelection) == 8);

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
    if (mState.pdfBuffer)
    {
        mState.pdfBuffer->release();
        mState.pdfBuffer = nullptr;
    }
    if (mState.rowCosBoundsBuffer)
    {
        mState.rowCosBoundsBuffer->release();
        mState.rowCosBoundsBuffer = nullptr;
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
    if (!mDevice)
        return;
    if (!mState.aliasBuffer)
    {
        const EnvAliasSelection selection{};
        mState.aliasBuffer = mDevice->newBuffer(&selection, sizeof(selection), MTL::ResourceStorageModeShared);
    }
    if (!mState.pdfBuffer)
    {
        const float pdf = 0.0f;
        mState.pdfBuffer = mDevice->newBuffer(&pdf, sizeof(pdf), MTL::ResourceStorageModeShared);
    }
    if (!mState.rowCosBoundsBuffer)
    {
        const EnvRowCosBounds row{ 1.0f, -1.0f };
        mState.rowCosBoundsBuffer = mDevice->newBuffer(&row, sizeof(row), MTL::ResourceStorageModeShared);
    }
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
    mState.totalPower = aliasResult.totalPower;
    mState.aliasWidth = static_cast<uint32_t>(aliasWidth);
    mState.aliasHeight = static_cast<uint32_t>(aliasHeight);

    std::vector<EnvAliasSelection> selection(aliasResult.alias.size());
    std::vector<float> pdf(aliasResult.alias.size());
    for (size_t i = 0; i < aliasResult.alias.size(); ++i)
    {
        selection[i] = { aliasResult.alias[i].threshold, aliasResult.alias[i].alias };
        pdf[i] = aliasResult.alias[i].solidAnglePdf;
    }
    mState.aliasBuffer = mDevice->newBuffer(
        selection.data(), selection.size() * sizeof(EnvAliasSelection), MTL::ResourceStorageModeShared);
    mState.pdfBuffer = mDevice->newBuffer(pdf.data(), pdf.size() * sizeof(float), MTL::ResourceStorageModeShared);

    std::vector<EnvRowCosBounds> rowCosBounds(static_cast<size_t>(aliasHeight));
    for (int y = 0; y < aliasHeight; ++y)
    {
        const double theta0 = std::numbers::pi * static_cast<double>(y) / static_cast<double>(aliasHeight);
        const double theta1 = std::numbers::pi * static_cast<double>(y + 1) / static_cast<double>(aliasHeight);
        rowCosBounds[static_cast<size_t>(y)] = {
            static_cast<float>(std::cos(theta0)),
            static_cast<float>(std::cos(theta1)),
        };
    }
    mState.rowCosBoundsBuffer = mDevice->newBuffer(
        rowCosBounds.data(), rowCosBounds.size() * sizeof(EnvRowCosBounds), MTL::ResourceStorageModeShared);
    if (!mState.mapTexture || !mState.aliasBuffer || !mState.pdfBuffer || !mState.rowCosBoundsBuffer)
    {
        releaseFloatImage(image);
        STRELKA_ERROR("Failed to allocate Metal environment resources for {}", texturePath);
        clearMap();
        return;
    }
    mState.rowCosBoundsBuffer->setLabel(NS::String::string("environment row cosine bounds", NS::UTF8StringEncoding));
    mState.aliasBuffer->setLabel(NS::String::string("environment alias selection", NS::UTF8StringEncoding));
    mState.pdfBuffer->setLabel(NS::String::string("environment solid-angle PDF", NS::UTF8StringEncoding));

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
        aliasResult.alias.size() * (sizeof(EnvAliasSelection) + sizeof(float)) / (1024.0 * 1024.0),
        aliasResult.totalPower, avgWeightedLum, mState.autoScale);
}

} // namespace oka::metal
