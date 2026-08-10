#include "MetalLights.h"

#include "ShaderTypes.h"

#include <cstring>

namespace oka
{
namespace metal
{

MetalLights::~MetalLights()
{
    release();
}

void MetalLights::init(MTL::Device* device)
{
    mDevice = device;
}

void MetalLights::release()
{
    if (mLightBuffer)
    {
        mLightBuffer->release();
        mLightBuffer = nullptr;
    }
}

void MetalLights::upload(const std::vector<Scene::Light>& lightDescs)
{
    static_assert(sizeof(Scene::Light) == sizeof(UniformLight));
    const size_t lightBufferSize = sizeof(Scene::Light) * lightDescs.size();

    if (lightBufferSize == 0)
    {
        if (mLightBuffer)
        {
            mLightBuffer->release();
            mLightBuffer = nullptr;
        }
        return;
    }

    if (!mLightBuffer || mLightBuffer->length() < lightBufferSize)
    {
        if (mLightBuffer)
            mLightBuffer->release();
        mLightBuffer = mDevice->newBuffer(lightBufferSize, MTL::ResourceStorageModeManaged);
    }
    memcpy(mLightBuffer->contents(), lightDescs.data(), lightBufferSize);
    mLightBuffer->didModifyRange(NS::Range::Make(0, lightBufferSize));
}

} // namespace metal
} // namespace oka
