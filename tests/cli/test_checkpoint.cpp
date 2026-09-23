#include "../../src/cli/checkpoint.h"

#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

TEST_CASE("checkpoint round trips accumulation and rejects mismatched or corrupt state")
{
    namespace fs = std::filesystem;
    const fs::path path =
        fs::temp_directory_path() /
        ("strelka-checkpoint-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".stc");
    const std::vector<float> pixels{ 0.0f, 0.5f, 2.0f, 1.0f, 1.0f, 0.25f, 9.0f, 1.0f };
    oka::checkpoint::save(path, 2, 1, 17, 0x12345678ull, pixels);

    const auto restored = oka::checkpoint::load(path, 2, 1, 0x12345678ull);
    CHECK(restored.spp == 17);
    CHECK(restored.rgba == pixels);
    CHECK_THROWS_AS(oka::checkpoint::load(path, 1, 2, 0x12345678ull), std::runtime_error);
    CHECK_THROWS_AS(oka::checkpoint::load(path, 2, 1, 0x12345679ull), std::runtime_error);

    {
        std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(sizeof(oka::checkpoint::Header));
        stream.put('X');
    }
    CHECK_THROWS_AS(oka::checkpoint::load(path, 2, 1, 0x12345678ull), std::runtime_error);
    fs::remove(path);

    CHECK_THROWS_AS(oka::checkpoint::save(path, UINT32_MAX, UINT32_MAX, 1, 0, pixels), std::runtime_error);
}
