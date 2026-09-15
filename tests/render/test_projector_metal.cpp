#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "metal/MetalTextures.h"
#include <Foundation/Foundation.hpp>

#include <array>
#include <filesystem>
#include <fstream>
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
    textures.init(device, nullptr, nullptr);
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
