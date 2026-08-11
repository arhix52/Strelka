#pragma once

#include <Metal/Metal.hpp>
#include <strelka/scene/scene.h>

#include <vector>

namespace oka
{
namespace metal
{

// Analytic light domain: Scene::Light → UniformLight GPU buffer, plus the packed
// IES candela tables those lights may index. Does not own environment / IBL
// (see MetalEnvironment).
class MetalLights
{
public:
    MetalLights() = default;
    ~MetalLights();

    void init(MTL::Device* device);
    void release();

    void upload(const std::vector<Scene::Light>& lightDescs, const std::vector<Scene::IesProfile>& iesProfiles);

    MTL::Buffer* buffer() const
    {
        return mLightBuffer;
    }
    MTL::Buffer* iesBuffer() const
    {
        return mIesBuffer;
    }

private:
    MTL::Device* mDevice = nullptr;
    MTL::Buffer* mLightBuffer = nullptr;
    MTL::Buffer* mIesBuffer = nullptr;
};

} // namespace metal
} // namespace oka
