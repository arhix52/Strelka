#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace oka::checkpoint
{

struct State
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t spp = 0;
    uint64_t signature = 0;
    std::vector<float> rgba;
};

inline uint64_t hashBytes(const void* bytes, size_t count, uint64_t hash = 14695981039346656037ull)
{
    const auto* data = static_cast<const uint8_t*>(bytes);
    for (size_t i = 0; i < count; ++i)
    {
        hash = (hash ^ data[i]) * 1099511628211ull;
    }
    return hash;
}

struct Header
{
    char magic[8];
    uint32_t width;
    uint32_t height;
    uint32_t spp;
    uint32_t reserved;
    uint64_t signature;
    uint64_t pixelHash;
};
static_assert(sizeof(Header) == 40);

inline void save(const std::filesystem::path& path,
                 uint32_t width,
                 uint32_t height,
                 uint32_t spp,
                 uint64_t signature,
                 std::span<const float> rgba)
{
    namespace fs = std::filesystem;
    const uint64_t area = uint64_t(width) * height;
    if (width == 0 || height == 0 || spp == 0 || area > std::numeric_limits<uint64_t>::max() / 4u)
    {
        throw std::runtime_error("invalid checkpoint dimensions or sample count");
    }
    const uint64_t pixelCount = area * 4u;
    if (pixelCount != rgba.size() ||
        pixelCount > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float))
    {
        throw std::runtime_error("invalid checkpoint dimensions or sample count");
    }
    if (path.has_parent_path())
    {
        fs::create_directories(path.parent_path());
    }
    const fs::path temporary = path.string() + ".tmp";
    Header header{};
    std::memcpy(header.magic, "STRLCP01", sizeof(header.magic));
    header.width = width;
    header.height = height;
    header.spp = spp;
    header.signature = signature;
    header.pixelHash = hashBytes(rgba.data(), rgba.size_bytes());
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
        stream.write(reinterpret_cast<const char*>(rgba.data()), static_cast<std::streamsize>(rgba.size_bytes()));
        stream.close();
        if (!stream)
        {
            throw std::runtime_error("failed to write checkpoint: " + temporary.string());
        }
    }
    std::error_code error;
    fs::rename(temporary, path, error);
    if (error)
    {
        fs::remove(temporary);
        throw std::runtime_error("failed to publish checkpoint: " + error.message());
    }
}

inline State load(const std::filesystem::path& path, uint32_t width, uint32_t height, uint64_t signature)
{
    std::ifstream stream(path, std::ios::binary);
    Header header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream || std::memcmp(header.magic, "STRLCP01", sizeof(header.magic)) != 0)
    {
        throw std::runtime_error("invalid checkpoint header: " + path.string());
    }
    if (header.width != width || header.height != height || header.signature != signature || header.spp == 0)
    {
        throw std::runtime_error("checkpoint scene, camera, settings, or dimensions do not match");
    }
    const uint64_t area = uint64_t(width) * height;
    if (area > std::numeric_limits<uint64_t>::max() / 4u ||
        area > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / (4u * sizeof(float)))
    {
        throw std::runtime_error("checkpoint dimensions are too large");
    }
    const uint64_t pixelCount = area * 4u;
    State state{ width, height, header.spp, signature, std::vector<float>(static_cast<size_t>(pixelCount)) };
    stream.read(reinterpret_cast<char*>(state.rgba.data()), static_cast<std::streamsize>(pixelCount * sizeof(float)));
    if (!stream || stream.peek() != std::char_traits<char>::eof() ||
        hashBytes(state.rgba.data(), state.rgba.size() * sizeof(float)) != header.pixelHash)
    {
        throw std::runtime_error("checkpoint pixel data is truncated or corrupt: " + path.string());
    }
    return state;
}

} // namespace oka::checkpoint
