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
};

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
        mTextures->beginMaterialPass();
    auto loadTex = [&](const std::string& path, bool srgb,
                       TextureKind kind = TextureKind::Color) -> MTL::ResourceID {
        if (path.empty())
            return MTL::ResourceID{};
        const fs::path fullPath = resourcePath / path;
        return mTextures->loadMaterialTexture(fullPath.string(), srgb, kind);
    };

    // Checked after every material rather than every so many: one material can
    // pull in five maps, and a map that misses the cache is a decode and an
    // upload -- far more than the clock read that guards it.
    const auto sliceStart = std::chrono::steady_clock::now();
    while (st.cursor < matDescs.size())
    {
        const Scene::MaterialDescription& currMatDesc = matDescs[st.cursor];
        ++st.cursor;
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

        material.baseColorTexture = loadTex(currMatDesc.baseColorTexPath, true);
        material.metallicRoughnessTexture =
            loadTex(currMatDesc.metallicRoughnessTexPath, false, TextureKind::NonColor);
        material.normalTexture = loadTex(currMatDesc.normalTexPath, false, TextureKind::Normal);
        material.emissionTexture = loadTex(currMatDesc.emissionTexPath, true);
        material.occlusionTexture = loadTex(currMatDesc.occlusionTexPath, false, TextureKind::NonColor);

        if (p.alpha_mode != ALPHA_MODE_OPAQUE)
            mSceneHasAlphaMaterials = true;
        // Both kinds of medium compile into the same free-flight path.
        if (p.subsurface > 0.0f || (p.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u)
            mSceneHasSubsurfaceMaterials = true;
        if ((p.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u)
            mSceneHasBoundedMedium = true;
        mMaterialIsMediumBoundary.push_back((p.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u ? 1u : 0u);
        // Per material, so a BLAS can say which of its geometries actually need
        // the alpha test. The scene-wide flag below only answers "is there any
        // cutout anywhere", which in a forest is always yes and drags trunks,
        // rocks and ground into the callback with the needles.
        mMaterialIsCutout.push_back(p.alpha_mode != ALPHA_MODE_OPAQUE ? 1u : 0u);
        st.gpuMaterials.push_back(material);

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

    const size_t materialsDataSize = sizeof(Material) * st.gpuMaterials.size();
    if (mMaterialBuffer)
    {
        mMaterialBuffer->release();
        mMaterialBuffer = nullptr;
    }
    if (materialsDataSize > 0)
    {
        mMaterialBuffer = mDevice->newBuffer(materialsDataSize, MTL::ResourceStorageModeShared);
        memcpy(mMaterialBuffer->contents(), st.gpuMaterials.data(), materialsDataSize);
    }

    delete mBuild;
    mBuild = nullptr;
    return true;
}

} // namespace metal
} // namespace oka
