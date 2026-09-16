#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "metal/MetalTextures.h"
#include <Foundation/Foundation.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <vector>
#include <unistd.h>

TEST_CASE("Metal projector loader preserves HDR radiance on an actual device")
{
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    REQUIRE(device != nullptr);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("strelka-projector-" + std::to_string(getpid()) + ".hdr");
    {
        std::ofstream output(path, std::ios::binary);
        REQUIRE(output.good());
        output << "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n";
        const std::array<char, 4> rgbe = { static_cast<char>(128), static_cast<char>(64), static_cast<char>(32),
                                           static_cast<char>(131) };
        output.write(rgbe.data(), static_cast<std::streamsize>(rgbe.size()));
    }

    oka::metal::MetalTextures textures;
    textures.init(device, nullptr);
    MTL::Texture* texture = textures.loadProjectorFromFile(path.string());
    REQUIRE(texture != nullptr);
    CHECK(texture->pixelFormat() == MTL::PixelFormatRGBA32Float);
    CHECK(texture->mipmapLevelCount() == 1u);

    std::array<float, 4> pixel{};
    texture->getBytes(pixel.data(), 4u * sizeof(float), MTL::Region::Make2D(0, 0, 1, 1), 0u);
    CHECK(pixel[0] == doctest::Approx(4.0f));
    CHECK(pixel[1] == doctest::Approx(2.0f));
    CHECK(pixel[2] == doctest::Approx(1.0f));
    CHECK(pixel[3] == doctest::Approx(1.0f));

    texture->release();
    std::error_code error;
    std::filesystem::remove(path, error);
    pool->release();
}

TEST_CASE("Metal material prewarm deduplicates and uploads each request once")
{
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    REQUIRE(device != nullptr);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("strelka-material-" + std::to_string(getpid()) + ".ppm");
    {
        std::ofstream output(path, std::ios::binary);
        REQUIRE(output.good());
        output << "P6\n1 1\n255\n";
        const std::array<char, 3> rgb = { static_cast<char>(64), static_cast<char>(128), static_cast<char>(255) };
        output.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
    }

    oka::SettingsManager settings;
    settings.setAs<uint32_t>("render/texture/maxDimension", 0u);
    settings.setAs<uint32_t>("render/texture/downscale", 1u);
    settings.setAs<bool>("render/texture/compress", false);
    settings.setAs<std::string>("render/texture/cachePath", "");

    oka::metal::MetalTextures textures;
    textures.init(device, &settings);
    const oka::metal::MetalTextures::Request request{ path.string(), true, oka::metal::TextureKind::Color };
    textures.beginMaterialPass(std::vector<oka::metal::MetalTextures::Request>{ request, request, request });
    CHECK(textures.prewarmTotal() == 1u);
    CHECK(textures.prewarmDone() == 0u);
    CHECK(textures.prewarmStep(0.0));
    CHECK(textures.prewarmDone() == 1u);
    REQUIRE(textures.materialTextures().size() == 1u);

    const MTL::ResourceID first = textures.loadMaterialTexture(path.string(), true, oka::metal::TextureKind::Color);
    const MTL::ResourceID second = textures.loadMaterialTexture(path.string(), true, oka::metal::TextureKind::Color);
    CHECK(first._impl != 0u);
    CHECK(second._impl == first._impl);
    CHECK(textures.materialTextures().size() == 1u);
    CHECK(textures.cacheMisses() == 1u);

    textures.releaseAll();
    std::error_code error;
    std::filesystem::remove(path, error);
    pool->release();
}

TEST_CASE("Metal material cache stores and restores native ASTC payloads")
{
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    REQUIRE(device != nullptr);
    if (!device->supportsFamily(MTL::GPUFamilyApple1))
    {
        pool->release();
        return;
    }

    const std::string stem = "strelka-astc-" + std::to_string(getpid());
    const std::filesystem::path path = std::filesystem::temp_directory_path() / (stem + ".ppm");
    const std::filesystem::path cache = std::filesystem::temp_directory_path() / (stem + "-cache");
    {
        std::ofstream output(path, std::ios::binary);
        REQUIRE(output.good());
        output << "P6\n7 7\n255\n";
        for (int i = 0; i < 49; ++i)
        {
            const std::array<char, 3> rgb = { static_cast<char>(i * 5), static_cast<char>(255 - i * 5),
                                              static_cast<char>(64 + i) };
            output.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
        }
    }

    oka::SettingsManager settings;
    settings.setAs<uint32_t>("render/texture/maxDimension", 0u);
    settings.setAs<uint32_t>("render/texture/downscale", 1u);
    settings.setAs<bool>("render/texture/compress", true);
    settings.setAs<std::string>("render/texture/cachePath", cache.string());

    oka::metal::MetalTextures textures;
    textures.init(device, &settings);
    MTL::Texture* first = textures.loadFromFile(path.string(), true, oka::metal::TextureKind::Color);
    REQUIRE(first != nullptr);
    CHECK(first->pixelFormat() == MTL::PixelFormatASTC_6x6_sRGB);
    CHECK(first->mipmapLevelCount() == 3u);
    CHECK(textures.cacheMisses() == 1u);
    first->release();

    MTL::Texture* cached = textures.loadFromFile(path.string(), true, oka::metal::TextureKind::Color);
    REQUIRE(cached != nullptr);
    CHECK(cached->pixelFormat() == MTL::PixelFormatASTC_6x6_sRGB);
    CHECK(cached->mipmapLevelCount() == 3u);
    CHECK(textures.cacheHits() == 1u);
    cached->release();

    MTL::Texture* normal = textures.loadFromFile(path.string(), false, oka::metal::TextureKind::Normal);
    REQUIRE(normal != nullptr);
    CHECK(normal->pixelFormat() == MTL::PixelFormatASTC_4x4_LDR);
    normal->release();

    std::error_code error;
    std::filesystem::remove_all(cache, error);
    std::filesystem::remove(path, error);
    pool->release();
}
