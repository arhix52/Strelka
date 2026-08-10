#pragma once

#include <Metal/Metal.hpp>
#include <strelka/scene/scene.h>

#include <vector>

namespace oka
{
namespace metal
{

// Analytic light domain: Scene::Light → UniformLight GPU buffer.
// Does not own environment / IBL (see MetalEnvironment).
class MetalLights
{
public:
    MetalLights() = default;
    ~MetalLights();

    void init(MTL::Device* device);
    void release();

    void upload(const std::vector<Scene::Light>& lightDescs);

    MTL::Buffer* buffer() const
    {
        return mLightBuffer;
    }

private:
    MTL::Device* mDevice = nullptr;
    MTL::Buffer* mLightBuffer = nullptr;
};

} // namespace metal
} // namespace oka
