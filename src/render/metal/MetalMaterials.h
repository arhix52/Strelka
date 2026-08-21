#pragma once

#include "MetalTextures.h"

#include <Metal/Metal.hpp>
#include <settings.h>
#include <strelka/scene/scene.h>

#include <loadprogress.h>

#include <cstdint>
#include <string>
#include <vector>

/// The GPU-side material record, shared with the shaders (ShaderTypes.h). Only
/// declared here so the table can be written and patched without this header
/// pulling the shader types in.
struct Material;
/// The OpenPBR argument block (material/openpbr/openpbr_params.h), forward
/// declared for the same reason: only its name is needed here.
struct OpenPBRParams;
/// The bindless map table for one OpenPBR material (ShaderTypes.h).
struct OpenPBRTextures;


namespace oka::metal
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
    /// Writes the whole material table with no textures in it, from the scene
    /// description alone. Cheap, and enough to shade the scene: the shader reads
    /// a material's own factors wherever a texture handle is null. Idempotent.
    void publishParameters(Scene* scene);

    // Resumable: a slice at a time against a millisecond budget. Zero = no limit.
    // Returns true when complete.
    bool step(Scene* scene, LoadProgress* progress, const std::string& resourceSearchPath, double budgetMs);

private:
    /// Writes the whole table. Called once, before any texture is opened.
    void uploadMaterialBuffer(const std::vector<Material>& materials);
    /// Writes the parallel OpenPBR table, or drops it when the scene has none.
    void uploadOpenPBRBuffer(const std::vector<OpenPBRParams>& params);
    /// Allocates the bindless map table, all handles null, one entry per material.
    void allocOpenPBRTextureBuffer(size_t materialCount);
    /// Writes one entry into the live table, for a texture that has just landed.
    void patchMaterial(size_t index, const Material& material);

public:
    bool buildActive() const
    {
        return mBuild != nullptr;
    }

    MTL::Buffer* buffer() const
    {
        return mMaterialBuffer;
    }
    /// The parallel OpenPBR parameter table, or null when no material uses it.
    /// Reached from the shader through Uniforms::openpbrParams rather than a
    /// binding, because the shade stage has no slot left -- but it still has to
    /// be made resident, so the caller needs the buffer itself.
    MTL::Buffer* openpbrBuffer() const
    {
        return mOpenPBRBuffer;
    }
    /// The parallel bindless map table. Null until a material asks for a map.
    MTL::Buffer* openpbrTextureBuffer() const
    {
        return mOpenPBRTexBuffer;
    }
    bool hasOpenPBRMaterials() const
    {
        return mSceneHasOpenPBRMaterials;
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
    MTL::Buffer* mOpenPBRBuffer = nullptr;
    MTL::Buffer* mOpenPBRTexBuffer = nullptr;
    bool mSceneHasOpenPBRMaterials = false;
    bool mSceneHasAlphaMaterials = false;
    bool mSceneHasBoundedMedium = false;
    bool mSceneHasSubsurfaceMaterials = false;
    std::vector<uint32_t> mMaterialIsMediumBoundary;
    std::vector<uint8_t> mMaterialIsCutout;

    MaterialBuildState* mBuild = nullptr;
};

} // namespace oka::metal

