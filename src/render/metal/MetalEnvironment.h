#pragma once

#include <Metal/Metal.hpp>
#include <settings.h>

#include <string>


namespace oka::metal
{

struct EnvMapState
{
    MTL::Texture* mapTexture = nullptr;
    MTL::Texture* backgroundTexture = nullptr;
    MTL::Buffer* aliasBuffer = nullptr;
    double totalPower = 0.0;
    float autoScale = 1.0f;
    float mapDecodeScale = 1.0f;
    float backgroundDecodeScale = 1.0f;
    uint32_t aliasWidth = 0;
    uint32_t aliasHeight = 0;
    bool loaded = false;
};

// Dome / IBL domain: lat-long HDR + importance-sampling alias table.
// Does not own analytic lights or material albedo maps.
class MetalEnvironment
{
public:
    MetalEnvironment() = default;
    ~MetalEnvironment();

    void init(MTL::Device* device, SettingsManager* settings);

    void loadMap(const std::string& absolutePath);
    void loadBackground(const std::string& absolutePath);
    void clearMap();
    void clearBackground();
    void release();

    // One-entry placeholder when no env is loaded (uniforms still bind a buffer).
    void ensurePlaceholderAliasBuffer();

    const EnvMapState& state() const
    {
        return mState;
    }
    EnvMapState& state()
    {
        return mState;
    }

private:
    MTL::Device* mDevice = nullptr;
    SettingsManager* mSettings = nullptr;
    EnvMapState mState;
};

} // namespace oka::metal
