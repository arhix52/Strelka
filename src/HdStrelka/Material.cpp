#include "Material.h"

#include <pxr/base/gf/vec2f.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/usdImaging/usdImaging/tokens.h>

#include <log.h>

PXR_NAMESPACE_OPEN_SCOPE

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
    STRELKA_INFO("Hydra Material: {}", name.c_str());
    const VtValue& resource = sceneDelegate->GetMaterialResource(id);

    if (!resource.IsHolding<HdMaterialNetworkMap>())
    {
        return;
    }

    HdMaterialNetworkMap networkMap = resource.GetWithDefault<HdMaterialNetworkMap>();
    HdMaterialNetwork& surfaceNetwork = networkMap.map[HdMaterialTerminalTokens->surface];

    bool isUsdPreviewSurface = false;
    HdMaterialNode* previewSurfaceNode = nullptr;
    // store material parameters
    uint32_t nodeIdx = 0;
    for (auto& node : surfaceNetwork.nodes)
    {
        STRELKA_DEBUG("Node #{}: {}", nodeIdx, node.path.GetText());
        if (node.identifier == UsdImagingTokens->UsdPreviewSurface)
        {
            previewSurfaceNode = &node;
            isUsdPreviewSurface = true;
        }
        for (const std::pair<TfToken, VtValue>& params : node.parameters)
        {
            const std::string& name = params.first.GetString();

            const TfType type = params.second.GetType();
            STRELKA_DEBUG("Node name: {}\tParam name: {}\t{}", node.path.GetName(), name.c_str(), params.second.GetTypeName().c_str());

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
                param.name = node.path.GetName() + "_" + std::string(params.first);
                param.type = oka::MaterialManager::Param::Type::eTexture;
                const SdfAssetPath val = params.second.Get<SdfAssetPath>();
                //STRELKA_DEBUG("path: {}", val.GetAssetPath().c_str());
                STRELKA_DEBUG("path: {}", val.GetResolvedPath().c_str());
                // std::string texPath = val.GetAssetPath();
                std::string texPath = val.GetResolvedPath();
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
            else if (type.IsA<TfToken>())
            {
                TfToken val = params.second.Get<TfToken>();
                STRELKA_DEBUG("TfToken: {}", val.GetText());
            }
            else if (type.IsA<std::string>())
            {
                std::string val = params.second.Get<std::string>();
                STRELKA_DEBUG("String: {}", val.c_str());
            }
            else
            {
                STRELKA_ERROR("Unknown parameter type!\n");
            }
        }
        nodeIdx++;
    }

    bool isVolume = false;
    const HdMaterialNetwork2 network = HdConvertToHdMaterialNetwork2(networkMap, &isVolume);
    if (isVolume)
    {
        STRELKA_ERROR("Volume %s unsupported", id.GetText());
        return;
    }

    if (isUsdPreviewSurface)
    {
        mMaterialXCode = m_translator.ParseNetwork(id, network);
        // STRELKA_DEBUG("MaterialX code:\n {}\n", mMaterialXCode.c_str());
    }
    else
    {
        // MDL
        const bool res = m_translator.ParseMdlNetwork(id, network, mMdlFileUri, mMdlSubIdentifier);
        if (!res)
        {
            STRELKA_ERROR("Failed to translate material, replace to default!");
            mMdlFileUri = "default.mdl";
            mMdlSubIdentifier = "default_material";
        }
        mIsMdl = true;
    }
}

const std::string& HdStrelkaMaterial::GetStrelkaMaterial() const
{
    return mMaterialXCode;
}

PXR_NAMESPACE_CLOSE_SCOPE
