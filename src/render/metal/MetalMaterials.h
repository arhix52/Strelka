#pragma once

#include "MetalTextures.h"

#include <Metal/Metal.hpp>
#include <settings.h>
#include <strelka/scene/scene.h>

#include <loadprogress.h>

#include <cstdint>
#include <string>
#include <vector>

namespace oka
{
namespace metal
{

struct MaterialBuildState;

// GPU material table + scene feature flags that gate wavefront variants
// (cutout / medium / SSS). Texture pixels come from MetalTextures.
class MetalMaterials
{
public:
    MetalMaterials() = default;
    ~MetalMaterials();

    void init(MTL::Device* device, MetalTextures* textures, SettingsManager* settings);
    void release();

    // No budget: one call does the lot (headless / incremental edit path).
    void create(Scene* scene, LoadProgress* progress, const std::string& resourceSearchPath);
    // Resumable: a slice at a time against a millisecond budget. Zero = no limit.
    // Returns true when complete.
    bool step(Scene* scene, LoadProgress* progress, const std::string& resourceSearchPath, double budgetMs);

    bool buildActive() const
    {
        return mBuild != nullptr;
    }

    MTL::Buffer* buffer() const
    {
        return mMaterialBuffer;
    }
    bool hasAlphaMaterials() const
    {
        return mSceneHasAlphaMaterials;
    }
    bool hasBoundedMedium() const
    {
        return mSceneHasBoundedMedium;
    }
    bool hasSubsurfaceMaterials() const
    {
        return mSceneHasSubsurfaceMaterials;
    }
    const std::vector<uint8_t>& isCutout() const
    {
        return mMaterialIsCutout;
    }
    const std::vector<uint32_t>& isMediumBoundary() const
    {
        return mMaterialIsMediumBoundary;
    }

private:
    MTL::Device* mDevice = nullptr;
    MetalTextures* mTextures = nullptr;
    SettingsManager* mSettings = nullptr;

    MTL::Buffer* mMaterialBuffer = nullptr;
    bool mSceneHasAlphaMaterials = false;
    bool mSceneHasBoundedMedium = false;
    bool mSceneHasSubsurfaceMaterials = false;
    std::vector<uint32_t> mMaterialIsMediumBoundary;
    std::vector<uint8_t> mMaterialIsCutout;

    MaterialBuildState* mBuild = nullptr;
};

} // namespace metal
} // namespace oka
