#include "Material.h"

#include <pxr/base/gf/vec2f.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/usdImaging/usdImaging/tokens.h>

#include <log.h>

PXR_NAMESPACE_OPEN_SCOPE

HdStrelkaMaterial::HdStrelkaMaterial(const SdfPath& id)
    : HdMaterial(id)
{
}

HdStrelkaMaterial::~HdStrelkaMaterial() = default;

HdDirtyBits HdStrelkaMaterial::GetInitialDirtyBitsMask() const
{
    return DirtyBits::AllDirty;
}

// Helper: resolve an SdfAssetPath texture from a UsdPreviewSurface node's connections
static std::string resolveTextureConnection(
    const HdMaterialNetwork& surfaceNetwork,
    const HdMaterialNode& surfaceNode,
    const TfToken& inputName)
{
    // Find connections targeting this input
    for (const auto& rel : surfaceNetwork.relationships)
    {
        if (rel.outputName == inputName && rel.outputId == surfaceNode.path)
        {
            // Found a connection to this input -- look up the source node
            for (const auto& node : surfaceNetwork.nodes)
            {
                if (node.path == rel.inputId)
                {
                    // The source node should have a "file" parameter with the texture path
                    auto it = node.parameters.find(TfToken("file"));
                    if (it != node.parameters.end() && it->second.IsHolding<SdfAssetPath>())
                    {
                        const SdfAssetPath& path = it->second.UncheckedGet<SdfAssetPath>();
                        std::string resolved = path.GetResolvedPath();
                        if (resolved.empty())
                            resolved = path.GetAssetPath();
                        return resolved;
                    }
                }
            }
        }
    }
    return {};
}

void HdStrelkaMaterial::Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits)
{
    TF_UNUSED(renderParam);

    const bool pullMaterial = (*dirtyBits & DirtyBits::DirtyParams) != 0u;

    *dirtyBits = DirtyBits::Clean;

    if (!pullMaterial)
    {
        return;
    }

    const SdfPath& id = GetId();
    const std::string& name = id.GetString();
    STRELKA_INFO("Hydra Material: {}", name.c_str());
    const VtValue& resource = sceneDelegate->GetMaterialResource(id);

    if (!resource.IsHolding<HdMaterialNetworkMap>())
    {
        return;
    }

    auto networkMap = resource.GetWithDefault<HdMaterialNetworkMap>();
    HdMaterialNetwork& surfaceNetwork = networkMap.map[HdMaterialTerminalTokens->surface];

    HdMaterialNode* previewSurfaceNode = nullptr;
    for (auto& node : surfaceNetwork.nodes)
    {
        if (node.identifier == UsdImagingTokens->UsdPreviewSurface)
        {
            previewSurfaceNode = &node;
            break;
        }
    }

    // Initialize description with defaults
    mDescription = {};
    mDescription.name = name;
    mDescription.params.material_type = MATERIAL_TYPE_STANDARD_PBR;
    mDescription.params.base_color = {1.0f, 1.0f, 1.0f};
    mDescription.params.metallic = 0.0f;
    mDescription.params.roughness = 0.5f;
    mDescription.params.ior = 1.5f;
    mDescription.params.specular = 0.5f;
    mDescription.params.specular_color = glm::float3(1.0f);
    mDescription.params.transmission = 0.0f;
    mDescription.params.clearcoat = 0.0f;
    mDescription.params.clearcoat_roughness = 0.01f;
    mDescription.params.anisotropy = 0.0f;
    mDescription.params.emission = {0.0f, 0.0f, 0.0f};
    mDescription.params.emission_strength = 1.0f;
    mDescription.params.normal_scale = 1.0f;
    mDescription.params.occlusion_strength = 1.0f;
    mDescription.params.alpha_cutoff = 0.5f;
    mDescription.params.thin_walled = 0;
    mDescription.params.base_color_tex = -1;
    mDescription.params.metallic_roughness_tex = -1;
    mDescription.params.normal_tex = -1;
    mDescription.params.emission_tex = -1;
    mDescription.params.occlusion_tex = -1;
    mDescription.params.transmission_tex = -1;

    if (!previewSurfaceNode)
    {
        STRELKA_WARNING("Material {} has no UsdPreviewSurface node, using defaults", name.c_str());
        return;
    }

    // Map UsdPreviewSurface parameters to MaterialParams
    const auto& params = previewSurfaceNode->parameters;

    auto getFloat3 = [&](const TfToken& token, float3& out) {
        auto it = params.find(token);
        if (it != params.end() && it->second.IsHolding<GfVec3f>())
        {
            GfVec3f v = it->second.UncheckedGet<GfVec3f>();
            out = {v[0], v[1], v[2]};
        }
    };

    auto getFloat = [&](const TfToken& token, float& out) {
        auto it = params.find(token);
        if (it != params.end() && it->second.IsHolding<float>())
        {
            out = it->second.UncheckedGet<float>();
        }
    };

    auto getInt = [&](const TfToken& token, unsigned int& out) {
        auto it = params.find(token);
        if (it != params.end() && it->second.IsHolding<int>())
        {
            out = static_cast<unsigned int>(it->second.UncheckedGet<int>());
        }
    };

    // UsdPreviewSurface parameters
    getFloat3(TfToken("diffuseColor"), mDescription.params.base_color);
    getFloat3(TfToken("emissiveColor"), mDescription.params.emission);
    getFloat(TfToken("metallic"), mDescription.params.metallic);
    getFloat(TfToken("roughness"), mDescription.params.roughness);
    getFloat(TfToken("ior"), mDescription.params.ior);
    getFloat(TfToken("clearcoat"), mDescription.params.clearcoat);
    getFloat(TfToken("clearcoatRoughness"), mDescription.params.clearcoat_roughness);
    getFloat(TfToken("opacity"), mDescription.params.transmission); // inverted below
    getFloat(TfToken("specular"), mDescription.params.specular);

    // UsdPreviewSurface "opacity" is 1=opaque, but our "transmission" is 0=opaque
    // So: transmission = 1 - opacity
    {
        auto it = params.find(TfToken("opacity"));
        if (it != params.end() && it->second.IsHolding<float>())
        {
            float opacity = it->second.UncheckedGet<float>();
            mDescription.params.transmission = 1.0f - opacity;
        }
    }

    getInt(TfToken("useSpecularWorkflow"), mDescription.params.thin_walled);

    // Resolve texture connections
    mDescription.baseColorTexPath = resolveTextureConnection(
        surfaceNetwork, *previewSurfaceNode, TfToken("diffuseColor"));
    mDescription.normalTexPath = resolveTextureConnection(
        surfaceNetwork, *previewSurfaceNode, TfToken("normal"));
    mDescription.emissionTexPath = resolveTextureConnection(
        surfaceNetwork, *previewSurfaceNode, TfToken("emissiveColor"));
    mDescription.occlusionTexPath = resolveTextureConnection(
        surfaceNetwork, *previewSurfaceNode, TfToken("occlusion"));

    // metallic/roughness are often packed into one texture by UsdPreviewSurface
    mDescription.metallicRoughnessTexPath = resolveTextureConnection(
        surfaceNetwork, *previewSurfaceNode, TfToken("metallic"));

    STRELKA_DEBUG("Material {} mapped: baseColor=({},{},{}), metallic={}, roughness={}, ior={}",
                  name.c_str(),
                  mDescription.params.base_color.x, mDescription.params.base_color.y, mDescription.params.base_color.z,
                  mDescription.params.metallic, mDescription.params.roughness, mDescription.params.ior);
}

PXR_NAMESPACE_CLOSE_SCOPE
