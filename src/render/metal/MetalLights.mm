#include "MetalLights.h"

#include "ShaderTypes.h"

#include <host/light_selection.h>

#include <strelka/scene/light_desc.h>

#include <log.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>


namespace oka::metal
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

        for (const float c : p.candela)
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
    uploadIesProfiles({});
}

void MetalLights::uploadIesProfiles(const std::vector<Scene::IesProfile>& iesProfiles)
{
    const std::vector<uint8_t> packed = packIesProfiles(iesProfiles);
    if (!mIesBuffer || mIesBuffer->length() < packed.size())
    {
        if (mIesBuffer)
            mIesBuffer->release();
        mIesBuffer = mDevice->newBuffer(packed.size(), MTL::ResourceStorageModeShared);
    }
    memcpy(mIesBuffer->contents(), packed.data(), packed.size());
}

void MetalLights::release()
{
    if (mLightBuffer)
    {
        mLightBuffer->release();
        mLightBuffer = nullptr;
    }
    mTotalPower = 0.0;
    if (mIesBuffer)
    {
        mIesBuffer->release();
        mIesBuffer = nullptr;
    }
    releaseProjectorTextures();
}

void MetalLights::releaseProjectorTextures()
{
    for (MTL::Texture* texture : mProjectorTextures)
    {
        if (texture)
            texture->release();
    }
    mProjectorTextures.clear();
    mProjectorImagePaths.clear();
}

/// Decode the images projector lights throw, in the order the scene registered
/// them, so that a light's points[0].z indexes this vector.
///
/// Loaded as colour, because a slide is display encoded and its texels are meant
/// to be seen: the sRGB pixel format is what turns them back into the linear
/// radiance the light multiplies. A slot whose file failed to decode stays null
/// and the shader throws a plain white frame there, which is a visible rectangle
/// rather than a light that quietly stopped working.
void MetalLights::loadProjectorImages(const std::vector<std::string>& paths, MetalTextures& textures)
{
    if (paths == mProjectorImagePaths)
    {
        return;
    }
    releaseProjectorTextures();
    mProjectorImagePaths = paths;
    mProjectorTextures.reserve(paths.size());
    for (const std::string& path : paths)
    {
        mProjectorTextures.push_back(path.empty() ? nullptr :
                                                    textures.loadFromFile(path, true, TextureKind::Color));
        if (const MTL::Texture* texture = mProjectorTextures.back())
        {
            STRELKA_INFO("Loaded projector image: {} ({}x{})", path, texture->width(), texture->height());
        }
    }
}

void MetalLights::upload(const std::vector<Scene::Light>& lightDescs,
                         const std::vector<Scene::IesProfile>& iesProfiles,
                         const std::vector<std::string>& projectorImages,
                         MetalTextures& textures)
{
    loadProjectorImages(projectorImages, textures);

    // This backend's UniformLight carries one field the host's Scene::Light does
    // not -- the bindless handle of a projector's image -- so the table is built
    // field for field rather than memcpy'd whole. The shared prefix is still one
    // copy; only the handle is resolved per light, from the slot the scene
    // packed into points[0].z.
    static_assert(offsetof(UniformLight, projectorTexture) == sizeof(Scene::Light),
                  "the host light must be the exact prefix of the GPU light");
    static_assert(sizeof(UniformLight) == sizeof(Scene::Light) + 16,
                  "the GPU light adds a handle and selection data, and nothing else");

    std::vector<double> powers;
    powers.reserve(lightDescs.size());
    for (const Scene::Light& light : lightDescs)
    {
        powers.push_back(analyticLightPower(light));
    }
    const LightSelectionTable selection = buildLightSelectionAlias(powers);
    mTotalPower = selection.totalPower;

    const size_t lightBufferSize = sizeof(UniformLight) * lightDescs.size();

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
        auto* gpuLights = static_cast<UniformLight*>(mLightBuffer->contents());
        for (size_t i = 0; i < lightDescs.size(); ++i)
        {
            UniformLight& dst = gpuLights[i];
            std::memcpy(&dst, &lightDescs[i], sizeof(Scene::Light));
            dst.projectorTexture = MTL::ResourceID{};
            dst.color.w = selection.entries[i].pdf;
            dst.selectionAliasThreshold = selection.entries[i].aliasThreshold;
            dst.selectionAlias = selection.entries[i].alias;
            if (lightDescs[i].type == LIGHT_TYPE_PROJECTOR)
            {
                const int slot = (int)lightDescs[i].points[0].z;
                if (slot >= 0 && (size_t)slot < mProjectorTextures.size() && mProjectorTextures[slot])
                {
                    dst.projectorTexture = mProjectorTextures[slot]->gpuResourceID();
                }
            }
        }
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

} // namespace oka::metal
