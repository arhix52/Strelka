#include "MetalMaterials.h"

#include "ShaderTypes.h"

#include <log.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;

namespace oka
{
namespace metal
{

// Working state of a resumable material build.
//
// These were locals, and they have to survive between calls now that the build
// runs a slice at a time. The texture cache in particular must, or a material
// in the second slice would re-decode a map the first slice already uploaded.
struct MaterialBuildState
{
    std::vector<Material> gpuMaterials;
    // One texture per file, not per slot. A scene routinely uses the same map in
    // several materials -- the pine forest fills 83 slots from 56 files -- and
    // without this each slot decoded and uploaded its own copy, which cost 2.8 GB
    // there. Keyed on path and colour space together, because the same file can
    // legitimately be needed both sRGB-decoded and linear.
    std::unordered_map<std::string, MTL::Texture*> textureCache;
    size_t cursor = 0;
    /// The parameter-only table, written once before any texture is opened.
    bool parametersPublished = false;
};

// Everything about a material that does not come out of a file. Texture
// handles are left null, and the shader reads these factors wherever one is,
// so a table built from this alone already shades the scene correctly -- in
// flat colours, until the maps arrive.
static Material makeMaterialParams(const Scene::MaterialDescription& currMatDesc)
{
    Material material = {};
    const auto& p = currMatDesc.params;
    material.base_color = packed_float3(simd_make_float3(p.base_color.x, p.base_color.y, p.base_color.z));
    material.metallic = p.metallic;
    material.roughness = p.roughness;
    material.ior = p.ior;
    material.specular = p.specular;
    material.subsurface_reference = packed_float3(simd_make_float3(
    p.subsurface_reference.x, p.subsurface_reference.y, p.subsurface_reference.z));
    material.iridescence = p.iridescence;
    material.iridescence_ior = p.iridescence_ior;
    material.iridescence_thickness = p.iridescence_thickness;
    material.specular_color = packed_float3(
    simd_make_float3(p.specular_color.x, p.specular_color.y, p.specular_color.z));
    material.clearcoat_ior = p.clearcoat_ior;
    material.medium_flags = p.medium_flags;
    material.medium_emission = packed_float3(
    simd_make_float3(p.medium_emission.x, p.medium_emission.y, p.medium_emission.z));
    material.subsurface = p.subsurface;
    material.subsurface_anisotropy = p.subsurface_anisotropy;
    material.subsurface_radius = packed_float3(
    simd_make_float3(p.subsurface_radius.x, p.subsurface_radius.y, p.subsurface_radius.z));
    material.sheen = p.sheen;
    material.sheen_roughness = p.sheen_roughness;
    material.sheen_color =
    packed_float3(simd_make_float3(p.sheen_color.x, p.sheen_color.y, p.sheen_color.z));
    material.diffuse_transmission = p.diffuse_transmission;
    material.diffuse_transmission_color = packed_float3(simd_make_float3(
    p.diffuse_transmission_color.x, p.diffuse_transmission_color.y, p.diffuse_transmission_color.z));
    material.transmission = p.transmission;
    material.clearcoat = p.clearcoat;
    material.clearcoat_roughness = p.clearcoat_roughness;
    material.anisotropy = p.anisotropy;
    material.emission = packed_float3(simd_make_float3(p.emission.x, p.emission.y, p.emission.z));
    material.emission_strength = p.emission_strength;
    material.normal_scale = p.normal_scale;
    material.occlusion_strength = p.occlusion_strength;
    material.alpha_cutoff = p.alpha_cutoff;
    material.alpha_mode = p.alpha_mode;
    material.base_color_alpha = p.base_color_alpha;
    material.attenuation_color = packed_float3(
    simd_make_float3(p.attenuation_color.x, p.attenuation_color.y, p.attenuation_color.z));
    material.attenuation_distance = p.attenuation_distance;
    material.uv_offset = simd_make_float2(p.uv_offset_x, p.uv_offset_y);
    material.uv_scale = simd_make_float2(p.uv_scale_x, p.uv_scale_y);
    material.uv_rotation = p.uv_rotation;
    material.material_type = p.material_type;
    material.thin_walled = p.thin_walled;
    material.dielectric_priority = p.dielectric_priority;

    return material;
}

MetalMaterials::~MetalMaterials()
{
    release();
    delete mBuild;
    mBuild = nullptr;
}

void MetalMaterials::init(MTL::Device* device, MetalTextures* textures, SettingsManager* settings)
{
    mDevice = device;
    mTextures = textures;
    mSettings = settings;
}

void MetalMaterials::release()
{
    if (mMaterialBuffer)
    {
        mMaterialBuffer->release();
        mMaterialBuffer = nullptr;
    }
    mSceneHasAlphaMaterials = false;
    mSceneHasBoundedMedium = false;
    mSceneHasSubsurfaceMaterials = false;
    mMaterialIsMediumBoundary.clear();
    mMaterialIsCutout.clear();
}

void MetalMaterials::uploadMaterialBuffer(const std::vector<Material>& materials)
{
    if (mMaterialBuffer)
    {
        mMaterialBuffer->release();
        mMaterialBuffer = nullptr;
    }
    const size_t bytes = sizeof(Material) * materials.size();
    if (bytes == 0)
    {
        return;
    }
    // Shared storage, so a handle arriving later is a write to this pointer
    // rather than a new buffer -- which would mean a new address for the tracer
    // and a residency update in the middle of a frame.
    mMaterialBuffer = mDevice->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (mMaterialBuffer)
    {
        std::memcpy(mMaterialBuffer->contents(), materials.data(), bytes);
    }
}

void MetalMaterials::patchMaterial(size_t index, const Material& material)
{
    if (!mMaterialBuffer || (index + 1) * sizeof(Material) > mMaterialBuffer->length())
    {
        return;
    }
    auto* table = static_cast<Material*>(mMaterialBuffer->contents());
    table[index] = material;
}

void MetalMaterials::publishParameters(Scene* scene)
{
    if (!mBuild)
    {
        mBuild = new MaterialBuildState();
    }
    MaterialBuildState& st = *mBuild;
    if (st.parametersPublished)
    {
        return;
    }
    st.parametersPublished = true;

    const std::vector<Scene::MaterialDescription>& matDescs = scene->getMaterials();
    st.gpuMaterials.clear();
    st.gpuMaterials.reserve(matDescs.size());
    mMaterialIsCutout.clear();
    mMaterialIsMediumBoundary.clear();
    mSceneHasAlphaMaterials = false;
    mSceneHasBoundedMedium = false;
    mSceneHasSubsurfaceMaterials = false;
    for (const Scene::MaterialDescription& desc : matDescs)
    {
        const auto& p = desc.params;
        st.gpuMaterials.push_back(makeMaterialParams(desc));
        if (p.alpha_mode != ALPHA_MODE_OPAQUE)
            mSceneHasAlphaMaterials = true;
        // Both kinds of medium compile into the same free-flight path.
        if (p.subsurface > 0.0f || (p.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u)
            mSceneHasSubsurfaceMaterials = true;
        if ((p.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u)
            mSceneHasBoundedMedium = true;
        mMaterialIsMediumBoundary.push_back((p.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u ? 1u : 0u);
        // Per material, so a BLAS can say which of its geometries actually need
        // the alpha test. The scene-wide flag only answers "is there any cutout
        // anywhere", which in a forest is always yes and drags trunks, rocks and
        // ground into the callback with the needles.
        mMaterialIsCutout.push_back(p.alpha_mode != ALPHA_MODE_OPAQUE ? 1u : 0u);
    }
    uploadMaterialBuffer(st.gpuMaterials);
}

void MetalMaterials::create(Scene* scene, LoadProgress* progress, const std::string& resourceSearchPath)
{
    while (!step(scene, progress, resourceSearchPath, 0.0))
    {
    }
}

bool MetalMaterials::step(Scene* scene, LoadProgress* progress, const std::string& resourceSearchPath, double budgetMs)
{
    using simd::float3;
    const std::vector<Scene::MaterialDescription>& matDescs = scene->getMaterials();
    const fs::path resourcePath = resourceSearchPath;

    if (!mBuild)
    {
        mBuild = new MaterialBuildState();
        mBuild->gpuMaterials.reserve(matDescs.size());
        // Appended to below, so it has to start empty even if a previous build
        // for this renderer got as far as filling some of it.
        mMaterialIsCutout.clear();
        mMaterialIsMediumBoundary.clear();
        mSceneHasAlphaMaterials = false;
        mSceneHasBoundedMedium = false;
        mSceneHasSubsurfaceMaterials = false;
    }
    MaterialBuildState& st = *mBuild;

    if (st.cursor == 0)
    {
        mTextures->beginMaterialPass();
        // Every map the scene will ask for, decoded across all cores before the
        // loop below asks for them one at a time. Each call it makes then has
        // only the Metal calls left to do.
        std::vector<MetalTextures::Request> requests;
        requests.reserve(matDescs.size() * 5);
        auto want = [&](const std::string& path, bool srgb, TextureKind kind) {
            if (!path.empty())
                requests.push_back({ (resourcePath / path).string(), srgb, kind });
        };
        for (const Scene::MaterialDescription& d : matDescs)
        {
            want(d.baseColorTexPath, true, TextureKind::Color);
            want(d.metallicRoughnessTexPath, false, TextureKind::NonColor);
            want(d.normalTexPath, false, TextureKind::Normal);
            want(d.emissionTexPath, true, TextureKind::Color);
            want(d.occlusionTexPath, false, TextureKind::NonColor);
        }
        const auto tPrewarm = std::chrono::steady_clock::now();
        mTextures->prewarm(requests);
        STRELKA_INFO("Textures prewarmed: {} requests in {:.0f} ms",
                     requests.size(),
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tPrewarm).count());
    }
    auto loadTex = [&](const std::string& path, bool srgb,
                       TextureKind kind = TextureKind::Color) -> MTL::ResourceID {
        if (path.empty())
            return MTL::ResourceID{};
        const fs::path fullPath = resourcePath / path;
        return mTextures->loadMaterialTexture(fullPath.string(), srgb, kind);
    };

    // The table is published before a single file is opened.
    //
    // Everything except the texture handles is already known, and the shader
    // falls back to a material's own factors wherever a handle is null, so this
    // table shades the scene correctly from the start -- surfaces in their own
    // base colour, sharpening into their maps as the maps arrive. The geometry
    // no longer waits for the last texture to decode before it can be shown.
    //
    // It also settles the scene-wide flags from the parameters alone. Those
    // gate which wavefront variant is compiled and whether a BLAS needs the
    // alpha test, and they used to be complete only once every map had loaded.
    publishParameters(scene);

    // Checked after every material rather than every so many: one material can
    // pull in five maps, and a map that misses the cache is a decode and an
    // upload -- far more than the clock read that guards it.
    const auto sliceStart = std::chrono::steady_clock::now();
    while (st.cursor < matDescs.size())
    {
        const Scene::MaterialDescription& currMatDesc = matDescs[st.cursor];
        const size_t index = st.cursor;
        ++st.cursor;

        // Only the handles are still missing; the flags and factors were settled
        // when the table was published. Each is written straight into the buffer
        // the tracer is already reading, so a map takes effect on the next
        // published frame without the table being rebuilt.
        Material& material = st.gpuMaterials[index];
        material.baseColorTexture = loadTex(currMatDesc.baseColorTexPath, true);
        material.metallicRoughnessTexture =
            loadTex(currMatDesc.metallicRoughnessTexPath, false, TextureKind::NonColor);
        material.normalTexture = loadTex(currMatDesc.normalTexPath, false, TextureKind::Normal);
        material.emissionTexture = loadTex(currMatDesc.emissionTexPath, true);
        material.occlusionTexture = loadTex(currMatDesc.occlusionTexPath, false, TextureKind::NonColor);
        patchMaterial(index, material);

        if (budgetMs > 0.0 &&
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sliceStart).count() >=
                budgetMs)
        {
            if (progress)
            {
                progress->total.store((uint32_t)matDescs.size(), std::memory_order_relaxed);
                progress->done.store((uint32_t)st.cursor, std::memory_order_relaxed);
            }
            return false;
        }
    }

    mTextures->generateMips();
    {
        // What the texture set costs on the device, since which formats the
        // encoder picked is otherwise only visible in the editor's memory panel.
        size_t bytes = 0;
        for (MTL::Texture* t : mTextures->materialTextures())
        {
            bytes += t ? t->allocatedSize() : 0;
        }
        STRELKA_INFO("Textures: {} from cache, {} built and cached, {:.1f} MB on device",
                     mTextures->cacheHits(), mTextures->cacheMisses(), (double)bytes / (1024.0 * 1024.0));
    }

    // The buffer has been live since the parameters were written, and every
    // handle was patched into it as it arrived, so there is nothing to upload.

    delete mBuild;
    mBuild = nullptr;
    return true;
}

} // namespace metal
} // namespace oka
