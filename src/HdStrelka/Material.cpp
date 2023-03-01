#include "Material.h"

#include <pxr/base/gf/vec2f.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/usdImaging/usdImaging/tokens.h>

PXR_NAMESPACE_OPEN_SCOPE
// clang-format off
TF_DEFINE_PRIVATE_TOKENS(_tokens,
    (diffuse_color_constant)
);
// clang-format on

HdStrelkaMaterial::HdStrelkaMaterial(const SdfPath& id, const MaterialNetworkTranslator& translator)
    : HdMaterial(id), m_translator(translator)
{
}

HdStrelkaMaterial::~HdStrelkaMaterial()
{
}

HdDirtyBits HdStrelkaMaterial::GetInitialDirtyBitsMask() const
{
    // return DirtyBits::DirtyParams;
    return DirtyBits::AllDirty;
}

void HdStrelkaMaterial::Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits)
{
    TF_UNUSED(renderParam);

    const bool pullMaterial = (*dirtyBits & DirtyBits::DirtyParams);

    *dirtyBits = DirtyBits::Clean;

    if (!pullMaterial)
    {
        return;
    }

    const SdfPath& id = GetId();
    const std::string& name = id.GetString();
    printf("Hydra Material: %s\n", name.c_str());
    const VtValue& resource = sceneDelegate->GetMaterialResource(id);

    if (!resource.IsHolding<HdMaterialNetworkMap>())
    {
        return;
    }

    HdMaterialNetworkMap networkMap = resource.GetWithDefault<HdMaterialNetworkMap>();
    HdMaterialNetwork& surfaceNetwork = networkMap.map[HdMaterialTerminalTokens->surface];

    bool isUsdPreviewSurface = false;
    HdMaterialNode* previewSurfaceNode = nullptr;
    //store material parameters
    for (auto& node : surfaceNetwork.nodes)
    {
        if (node.identifier == UsdImagingTokens->UsdPreviewSurface)
        {
            previewSurfaceNode = &node;
            isUsdPreviewSurface = true;
        }
        for (const std::pair<TfToken, VtValue>& params : node.parameters)
        {
            const std::string name = params.first.GetString();

            const TfType type = params.second.GetType();
            printf("Param name: %s\t%s\n", name.c_str(), params.second.GetTypeName().c_str());

            if (type.IsA<GfVec3f>())
            {
                oka::MaterialManager::Param param;
                param.name = params.first;
                param.type = oka::MaterialManager::Param::Type::eFloat3;
                GfVec3f val = params.second.Get<GfVec3f>();
                param.value.resize(sizeof(val));
                memcpy(param.value.data(), &val, sizeof(val));
                mMaterialParams.push_back(param);
            }
            else if (type.IsA<GfVec4f>())
            {
                oka::MaterialManager::Param param;
                param.name = params.first;
                param.type = oka::MaterialManager::Param::Type::eFloat4;
                GfVec4f val = params.second.Get<GfVec4f>();
                param.value.resize(sizeof(val));
                memcpy(param.value.data(), &val, sizeof(val));
                mMaterialParams.push_back(param);
            }
            else if (type.IsA<float>())
            {
                oka::MaterialManager::Param param;
                param.name = params.first;
                param.type = oka::MaterialManager::Param::Type::eFloat;
                float val = params.second.Get<float>();
                param.value.resize(sizeof(val));
                memcpy(param.value.data(), &val, sizeof(val));
                mMaterialParams.push_back(param);
            }
            else if (type.IsA<int>())
            {
                oka::MaterialManager::Param param;
                param.name = params.first;
                param.type = oka::MaterialManager::Param::Type::eInt;
                int val = params.second.Get<int>();
                param.value.resize(sizeof(val));
                memcpy(param.value.data(), &val, sizeof(val));
                mMaterialParams.push_back(param);
            }
            else if (type.IsA<bool>())
            {
                oka::MaterialManager::Param param;
                param.name = params.first;
                param.type = oka::MaterialManager::Param::Type::eBool;
                bool val = params.second.Get<bool>();
                param.value.resize(sizeof(val));
                memcpy(param.value.data(), &val, sizeof(val));
                mMaterialParams.push_back(param);
            }
            else if (type.IsA<SdfAssetPath>())
            {
                oka::MaterialManager::Param param;
                param.name = params.first;
                param.type = oka::MaterialManager::Param::Type::eTexture;
                const SdfAssetPath val = params.second.Get<SdfAssetPath>();
                printf("path: %s\n", val.GetAssetPath().c_str());
                std::string texPath = val.GetAssetPath();
                if (!texPath.empty())
                {
                    param.value.resize(texPath.size());
                    memcpy(param.value.data(), texPath.data(), texPath.size());
                    mMaterialParams.push_back(param);
                }
            }
            else if (type.IsA<GfVec2f>())
            {
                oka::MaterialManager::Param param;
                param.name = params.first;
                param.type = oka::MaterialManager::Param::Type::eFloat2;
                GfVec2f val = params.second.Get<GfVec2f>();
                param.value.resize(sizeof(val));
                memcpy(param.value.data(), &val, sizeof(val));
                mMaterialParams.push_back(param);
            }
            else
            {
                printf("Unknown parameter type!\n");
            }
        }
    }

    bool isVolume = false;
    const HdMaterialNetwork2 network = HdConvertToHdMaterialNetwork2(networkMap, &isVolume);
    if (isVolume)
    {
        TF_WARN("Volume %s unsupported", id.GetText());
        return;
    }

    if (isUsdPreviewSurface)
    {
        mMaterialXCode = m_translator.ParseNetwork(id, network);
    }
    else
    {
        // MDL
        const bool res = m_translator.ParseMdlNetwork(id, network, mMdlFileUri, mMdlSubIdentifier);
        if (!res)
        {
            TF_RUNTIME_ERROR("Failed to translate material!");
        }
        mIsMdl = true;
    }
}

const std::string& HdStrelkaMaterial::GetStrelkaMaterial() const
{
    return mMaterialXCode;
}

PXR_NAMESPACE_CLOSE_SCOPE
