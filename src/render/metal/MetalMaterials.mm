#include "MetalMaterials.h"

#include "ShaderTypes.h"

#include <log.h>
#include <strelka/material/openpbr/openpbr_from_gltf.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;


namespace oka::metal
{

/// Colour space and filtering intent per OpenPBR texture slot.
///
/// Not derivable from the slot number, and getting it wrong is silent: a
/// roughness map read as sRGB is smoothly and plausibly wrong, and a normal map
/// read as colour loses the two-channel encoding the BC5 path depends on.
namespace
{
std::pair<bool, TextureKind> openpbrSlotKind(uint32_t slot)
{
    switch (slot)
    {
    case OPENPBR_TEX_BASE_COLOR:
    case OPENPBR_TEX_SPECULAR_COLOR:
    case OPENPBR_TEX_COAT_COLOR:
    case OPENPBR_TEX_EMISSION_COLOR:
    case OPENPBR_TEX_TRANSMISSION_COLOR:
    case OPENPBR_TEX_SUBSURFACE_COLOR:
    case OPENPBR_TEX_FUZZ_COLOR:
    case OPENPBR_TEX_SUBSURFACE_RADIUS:
        return { true, TextureKind::Color };
    case OPENPBR_TEX_GEOMETRY_NORMAL:
    case OPENPBR_TEX_GEOMETRY_COAT_NORMAL:
        return { false, TextureKind::Normal };
    default:
        // Everything else is a scalar the shader reads linearly.
        return { false, TextureKind::NonColor };
    }
}

/// The slot's guess at the encoding, overridden by whatever the document stated.
///
/// A slot default is a good guess and no more: it is glTF's convention, which
/// the Open Chess Set happens to agree with exactly. Only the document knows
/// that a particular roughness map was authored sRGB-encoded, or that a base
/// colour is already linear. TextureKind is deliberately not overridden -- that
/// says what the data *is*, a normal or a colour or a scalar, which no
/// colorspace attribute changes.
std::pair<bool, TextureKind> openpbrSlotKind(uint32_t slot, TexColorSpace stated)
{
    std::pair<bool, TextureKind> kind = openpbrSlotKind(slot);
    if (stated == TexColorSpace::Linear)
    {
        kind.first = false;
    }
    else if (stated == TexColorSpace::Srgb)
    {
        kind.first = true;
    }
    return kind;
}
} // namespace

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
    /// beginMaterialPass has run and the prewarm queue belongs to this build.
    bool prewarmStarted = false;
};

// Everything about a material that does not come out of a file. Texture
// handles are left null, and the shader reads these factors wherever one is,
// so a table built from this alone already shades the scene correctly -- in
// flat colours, until the maps arrive.
namespace
{
Material makeMaterialParams(const Scene::MaterialDescription& currMatDesc)
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
} // namespace

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
    if (mOpenPBRBuffer)
    {
        mOpenPBRBuffer->release();
        mOpenPBRBuffer = nullptr;
    }
    if (mOpenPBRTexBuffer)
    {
        mOpenPBRTexBuffer->release();
        mOpenPBRTexBuffer = nullptr;
    }
    mSceneHasOpenPBRMaterials = false;
    mSceneHasAlphaMaterials = false;
    mSceneHasBoundedMedium = false;
    mSceneHasSubsurfaceMaterials = false;
    mMaterialIsMediumBoundary.clear();
    mMaterialIsCutout.clear();
}

void MetalMaterials::uploadOpenPBRBuffer(const std::vector<OpenPBRParams>& params)
{
    if (mOpenPBRBuffer)
    {
        mOpenPBRBuffer->release();
        mOpenPBRBuffer = nullptr;
    }
    if (params.empty())
    {
        return;
    }
    // Shared storage for the reason the material table uses it: this address
    // ends up inside Uniforms and inside a residency set, and reallocating it
    // mid-frame would invalidate both.
    const size_t bytes = sizeof(OpenPBRParams) * params.size();
    mOpenPBRBuffer = mDevice->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (mOpenPBRBuffer)
    {
        std::memcpy(mOpenPBRBuffer->contents(), params.data(), bytes);
    }
}

void MetalMaterials::allocOpenPBRTextureBuffer(size_t materialCount)
{
    if (mOpenPBRTexBuffer)
    {
        mOpenPBRTexBuffer->release();
        mOpenPBRTexBuffer = nullptr;
    }
    if (materialCount == 0)
    {
        return;
    }
    // Published empty and filled in as the maps decode, exactly like the material
    // table: a null handle means "use the constant", so the scene shades
    // correctly from the first frame and sharpens into its textures.
    const size_t bytes = sizeof(OpenPBRTextures) * materialCount;
    mOpenPBRTexBuffer = mDevice->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (mOpenPBRTexBuffer)
    {
        std::memset(mOpenPBRTexBuffer->contents(), 0, bytes);
    }
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

    // Which material model the scene shades with. A render setting rather than a
    // scene property on purpose: it makes the two models an A/B on one asset,
    // which is the only external check on the OpenPBR integration that does not
    // involve a second renderer -- Blender has no OpenPBR to compare against.
    // 0 = the glTF model that has always shipped, 1 = OpenPBR.
    const bool openpbrModel = mSettings && mSettings->getAs<uint32_t>("render/material/model") == 1u;

    // A material can arrive as OpenPBR already: the <stem>_openpbr.json sidecar
    // authors the whole parameter block and sets the type at load time. That is a
    // property of the scene, so it holds whatever the render setting says, and
    // the two sources must not fight -- an authored block wins over a translated
    // one, because it is a statement about the surface rather than a best effort
    // at re-spelling a different model.
    bool anyAuthored = false;
    for (const Scene::MaterialDescription& desc : matDescs)
    {
        if (desc.params.material_type == MATERIAL_TYPE_OPENPBR)
        {
            anyAuthored = true;
            break;
        }
    }

    std::vector<OpenPBRParams> openpbrParams;
    if (openpbrModel || anyAuthored)
    {
        openpbrParams.reserve(matDescs.size());
    }

    for (const Scene::MaterialDescription& desc : matDescs)
    {
        const auto& p = desc.params;
        st.gpuMaterials.push_back(makeMaterialParams(desc));
        if (openpbrModel || anyAuthored)
        {
            // The table stays dense so one material id indexes both arrays, even
            // where only some materials are OpenPBR.
            if (p.material_type == MATERIAL_TYPE_OPENPBR)
            {
                openpbrParams.push_back(desc.openpbr);
                mSceneHasOpenPBRMaterials = true;
            }
            else if (openpbrModel && p.material_type != MATERIAL_TYPE_HAIR)
            {
                // Hair keeps its own BSDF: OpenPBR has no fibre model, and a
                // strand shaded as a surface is the defect open-defects.md entry
                // 3 was closed for.
                openpbrParams.push_back(openpbr_from_material_params(p));
                st.gpuMaterials.back().material_type = MATERIAL_TYPE_OPENPBR;
                mSceneHasOpenPBRMaterials = true;
            }
            else
            {
                openpbrParams.push_back(openpbr_make_default_params());
            }

            // The mask is derived rather than authored, so it cannot disagree
            // with the paths beside it: it says which slots this material *names
            // a file for*, and the shading path uses it to skip the whole handle
            // table for a material that names none. Whether the file then
            // decoded is a separate question the null handle answers.
            //
            // After the branches, not inside them. Placed between the first `if`
            // and its `else if`, this took over the chain: a non-OpenPBR material
            // in a scene that also had authored ones then overwrote the previous
            // entry's mask and pushed nothing, which desynchronises a table the
            // shader indexes by material id.
            {
                unsigned int mask = 0u;
                for (uint32_t slot = 0; slot < MAX_OPENPBR_TEXTURES; ++slot)
                {
                    if (!desc.openpbrTexPaths[slot].empty())
                    {
                        mask |= (1u << slot);
                    }
                }
                openpbrParams.back().texture_mask = mask;
                if (mask != 0u)
                {
                    STRELKA_DEBUG("openpbr material '{}': texture slots 0x{:x}, subsurface weight {}", desc.name, mask,
                                  openpbrParams.back().subsurface_weight);
                }
            }

            // Emission is not the BSDF's job in this integrator: the shade
            // kernel reads Material::emission * emission_strength directly, and
            // OpenPBR_PreparedBsdf::emission is never consulted. OpenPBR states
            // the same quantity as a luminance times a tint, so the product has
            // to be mirrored into those two fields -- otherwise an authored
            // emitter renders black and nothing says so.
            //
            // Found by the MaterialX example set: open_pbr_lightbulb, whose only
            // two parameters are emission_luminance 10000 and an orange tint,
            // rendered identical to open_pbr_default, whose emission is zero.
            //
            // Unit note: the spec calls emission_luminance nits, while Strelka's
            // emitters carry radiance in the units the light sidecar uses. The
            // two are passed through 1:1 here because inventing a conversion
            // would be worse than an explicit mismatch; an authored 10000 is
            // therefore 10000 of whatever the scene's lights are measured in.
            if (st.gpuMaterials.back().material_type == MATERIAL_TYPE_OPENPBR)
            {
                const OpenPBRParams& o = openpbrParams.back();
                Material& gm = st.gpuMaterials.back();

                const OpenPBRColor& ec = o.emission_color;
                gm.emission = packed_float3(simd_make_float3(ec.r, ec.g, ec.b));
                gm.emission_strength = o.emission_luminance;

                // --- The interior, in the two places the integrator keeps it ---
                //
                // OpenPBR states one medium; this renderer has two mechanisms for
                // it, and they are not interchangeable:
                //
                //   absorption  Beer-Lambert over a segment, from the IOR stack
                //   scattering  a random walk, entered on diffuse transmission
                //
                // Split along the same line OpenPBR itself does. Transmission
                // depth is pure absorption -- the derived volume's albedo is
                // exactly zero unless transmission_scatter asks otherwise -- and
                // volume.h's glTF reading is sigma_t = -ln(C)/d, which is the
                // same formula Adobe derives. So a transmissive OpenPBR material
                // is handed to the existing absorption path unchanged, and
                // nothing is computed twice.
                //
                // Note this pins the reading: OpenPBR defines the glTF one, so a
                // scene left on render/material/volumeModel = cycles would give
                // its OpenPBR glass a density the specification does not.
                if (o.transmission_depth > 0.0f)
                {
                    gm.attenuation_distance = o.transmission_depth;
                    gm.attenuation_color = packed_float3(
                        simd_make_float3(o.transmission_color.r, o.transmission_color.g, o.transmission_color.b));
                }

                // Scattering is the other half, and only the *trigger* is
                // mirrored: `si.subsurface > 0` is what admits the walk. Its
                // numbers -- extinction, single-scattering albedo, phase
                // anisotropy -- are read from openpbr_interior_volume() in the
                // shader instead, because deriving them here would mean
                // reimplementing the van de Hulst mapping Adobe already has.
                gm.subsurface = o.subsurface_weight;
                gm.thin_walled = o.geometry_thin_walled;

                // And the scene-wide flag, which the loop below derives from the
                // *host* MaterialParams and would therefore miss: an OpenPBR
                // material carries its subsurface weight in its own block, and
                // desc.params.subsurface is whatever the glTF said, usually zero.
                // Without this the kSubsurface variant is never compiled and the
                // walk cannot run at all -- the medium would be entered and then
                // traversed by a kernel that has no free-flight sampling in it.
                //
                // The map matters as much as the constant here: the Open Chess
                // Set leaves subsurface_weight at zero and drives it from
                // king_shared_scattering.jpg, so a scene judged on the constant
                // alone compiles a kernel that cannot walk the medium its own
                // textures ask for.
                const bool weightIsMapped = (o.texture_mask & (1u << OPENPBR_TEX_SUBSURFACE_WEIGHT)) != 0u;
                if ((o.subsurface_weight > 0.0f || weightIsMapped) && o.geometry_thin_walled == 0u)
                {
                    mSceneHasSubsurfaceMaterials = true;
                }
            }
        }
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
    uploadOpenPBRBuffer(openpbrParams);
    allocOpenPBRTextureBuffer(openpbrParams.empty() ? 0 : openpbrParams.size());
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
        if (!st.prewarmStarted)
        {
            st.prewarmStarted = true;
            mTextures->beginMaterialPass();
        }
        // Every map the scene will ask for, decoded across all cores before the
        // loop below asks for them one at a time -- but a batch at a time, so
        // the frame the renderer publishes between slices keeps arriving. Doing
        // the whole set in one call is faster on paper and freezes the window
        // for as long as it takes.
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
            for (uint32_t slot = 0; slot < MAX_OPENPBR_TEXTURES; ++slot)
            {
                const auto kind = openpbrSlotKind(slot, d.openpbrTexColorSpace[slot]);
                want(d.openpbrTexPaths[slot], kind.first, kind.second);
            }
        }
        if (!mTextures->prewarmStep(requests, budgetMs))
        {
            return false;
        }
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

        // The OpenPBR maps go into the parallel table, written straight into the
        // buffer the tracer is already reading -- same contract as the material
        // above, so a map takes effect on the next published frame.
        if (mOpenPBRTexBuffer && (index + 1) * sizeof(OpenPBRTextures) <= mOpenPBRTexBuffer->length())
        {
            auto* table = static_cast<OpenPBRTextures*>(mOpenPBRTexBuffer->contents());
            for (uint32_t slot = 0; slot < MAX_OPENPBR_TEXTURES; ++slot)
            {
                const auto kind = openpbrSlotKind(slot, currMatDesc.openpbrTexColorSpace[slot]);
                table[index].tex[slot] = loadTex(currMatDesc.openpbrTexPaths[slot], kind.first, kind.second);
                // A slot the material named a file for but that produced no
                // handle is the one failure this path can have that the image
                // does not show: the parameter silently keeps its constant.
                if (!currMatDesc.openpbrTexPaths[slot].empty() && table[index].tex[slot]._impl == 0)
                {
                    STRELKA_WARNING("openpbr material '{}': slot {} named '{}' but no texture loaded", currMatDesc.name,
                                    slot, currMatDesc.openpbrTexPaths[slot]);
                }
            }
        }

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
        for (const MTL::Texture* t : mTextures->materialTextures())
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

} // namespace oka::metal

