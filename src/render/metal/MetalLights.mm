#include "MetalLights.h"

#include "ShaderTypes.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace oka
{
namespace metal
{
namespace
{

// Always produce a buffer the shade kernel can bind: a zero profile count is a
// valid empty table, and avoids a null pointer when the scene has no IES.
std::vector<uint8_t> packIesProfiles(const std::vector<Scene::IesProfile>& profiles)
{
    std::vector<IesGpuProfileHeader> headers(profiles.size());
    std::vector<float> floats;
    for (size_t i = 0; i < profiles.size(); ++i)
    {
        const Scene::IesProfile& p = profiles[i];
        IesGpuProfileHeader& h = headers[i];
        h.nVertical = (uint32_t)p.verticalAngles.size();
        h.nHorizontal = (uint32_t)p.horizontalAngles.size();
        h.anglesOffset = (uint32_t)floats.size();
        floats.insert(floats.end(), p.verticalAngles.begin(), p.verticalAngles.end());
        floats.insert(floats.end(), p.horizontalAngles.begin(), p.horizontalAngles.end());
        h.candelaOffset = (uint32_t)floats.size();
        // IES files are in candela (lm/sr) and this light's colour is radiant
        // intensity (W/sr), so the table needs a luminous efficacy to divide by.
        // A photometric file does not carry its own spectrum, so a standard one
        // has to be assumed: 177.83 lm/W for D65, which is the figure Cycles
        // uses for the same conversion (cycles/src/util/ies.cpp, where it
        // appears as 4pi/177.83 because a Cycles lamp takes Watts rather than
        // Watts per steradian). Picking the same illuminant is what lets 27_ies
        // compare the angular distribution rather than two guesses at a scale.
        constexpr float kCandelaToRadiantIntensity = 1.0f / 177.83f;
        for (float c : p.candela)
        {
            floats.push_back(c * kCandelaToRadiantIntensity);
        }
        h.maxCandela = p.maxCandela * kCandelaToRadiantIntensity;
        h.pad0 = h.pad1 = h.pad2 = 0.0f;
    }

    IesGpuBufferHeader header{};
    header.profileCount = (uint32_t)profiles.size();
    header.floatOffset =
        (uint32_t)(sizeof(IesGpuBufferHeader) + headers.size() * sizeof(IesGpuProfileHeader));

    std::vector<uint8_t> bytes(header.floatOffset + floats.size() * sizeof(float), 0);
    std::memcpy(bytes.data(), &header, sizeof(header));
    if (!headers.empty())
    {
        std::memcpy(bytes.data() + sizeof(IesGpuBufferHeader), headers.data(),
                    headers.size() * sizeof(IesGpuProfileHeader));
    }
    if (!floats.empty())
    {
        std::memcpy(bytes.data() + header.floatOffset, floats.data(), floats.size() * sizeof(float));
    }
    return bytes;
}

} // namespace

MetalLights::~MetalLights()
{
    release();
}

void MetalLights::init(MTL::Device* device)
{
    mDevice = device;
    // Shade always binds the IES table; keep a zero-count buffer ready so a
    // frame that runs before the first upload cannot hand the kernel null.
    upload({}, {});
}

void MetalLights::release()
{
    if (mLightBuffer)
    {
        mLightBuffer->release();
        mLightBuffer = nullptr;
    }
    if (mIesBuffer)
    {
        mIesBuffer->release();
        mIesBuffer = nullptr;
    }
}

void MetalLights::upload(const std::vector<Scene::Light>& lightDescs,
                         const std::vector<Scene::IesProfile>& iesProfiles)
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
    }
    else
    {
        if (!mLightBuffer || mLightBuffer->length() < lightBufferSize)
        {
            if (mLightBuffer)
                mLightBuffer->release();
            mLightBuffer = mDevice->newBuffer(lightBufferSize, MTL::ResourceStorageModeShared);
        }
        memcpy(mLightBuffer->contents(), lightDescs.data(), lightBufferSize);
    }

    const std::vector<uint8_t> packed = packIesProfiles(iesProfiles);
    if (!mIesBuffer || mIesBuffer->length() < packed.size())
    {
        if (mIesBuffer)
            mIesBuffer->release();
        mIesBuffer = mDevice->newBuffer(packed.size(), MTL::ResourceStorageModeShared);
    }
    memcpy(mIesBuffer->contents(), packed.data(), packed.size());
}

} // namespace metal
} // namespace oka
