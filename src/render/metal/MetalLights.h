#pragma once

#include "MetalTextures.h"

#include <Metal/Metal.hpp>
#include <strelka/scene/scene.h>

#include <string>
#include <vector>

namespace oka
{
namespace metal
{

// Analytic light domain: Scene::Light → UniformLight GPU buffer, plus the packed
// IES candela tables those lights may index and the images projector lights
// throw. Does not own environment / IBL (see MetalEnvironment).
class MetalLights
{
public:
    MetalLights() = default;
    ~MetalLights();

    void init(MTL::Device* device);
    void release();

    /// Rebuild the GPU light table.
    ///
    /// The projector images come in as paths and go out as bindless handles
    /// inside the lights themselves, which is why this takes the texture domain:
    /// the handle cannot be resolved before the image exists, and the shade
    /// kernel has no binding slot left for a table of its own.
    void upload(const std::vector<Scene::Light>& lightDescs,
                const std::vector<Scene::IesProfile>& iesProfiles,
                const std::vector<std::string>& projectorImages,
                MetalTextures& textures);

    MTL::Buffer* buffer() const
    {
        return mLightBuffer;
    }
    MTL::Buffer* iesBuffer() const
    {
        return mIesBuffer;
    }
    /// For the Metal 4 residency set: an argument table names these by handle,
    /// and a handle whose allocation is not resident is a page fault rather than
    /// a validation message.
    const std::vector<MTL::Texture*>& projectorTextures() const
    {
        return mProjectorTextures;
    }

private:
    void uploadIesProfiles(const std::vector<Scene::IesProfile>& iesProfiles);
    void loadProjectorImages(const std::vector<std::string>& paths, MetalTextures& textures);
    void releaseProjectorTextures();

    MTL::Device* mDevice = nullptr;
    MTL::Buffer* mLightBuffer = nullptr;
    MTL::Buffer* mIesBuffer = nullptr;
    std::vector<MTL::Texture*> mProjectorTextures;
    /// What mProjectorTextures was built from. upload() runs on every light
    /// edit -- dragging an intensity slider is one per frame -- and decoding a
    /// 4K slide each time would make the light unusable to author with.
    std::vector<std::string> mProjectorImagePaths;
};

} // namespace metal
} // namespace oka
