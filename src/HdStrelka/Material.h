#pragma once

#include "materialmanager.h"
#include "MaterialNetworkTranslator.h"

#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/sceneDelegate.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdStrelkaMaterial final : public HdMaterial
{
public:
    HF_MALLOC_TAG_NEW("new HdStrelkaMaterial");

    HdStrelkaMaterial(const SdfPath& id,
                   const MaterialNetworkTranslator& translator);

    ~HdStrelkaMaterial() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;

    void Sync(HdSceneDelegate* sceneDelegate,
              HdRenderParam* renderParam,
              HdDirtyBits* dirtyBits) override;

    const std::string& GetStrelkaMaterial() const;
    bool isMdl()
    {
        return mIsMdl;
    }
    std::string getFileUri()
    {
        return mMdlFileUri;
    }
    std::string getSubIdentifier()
    {
        return mMdlSubIdentifier;
    }

    const std::vector<oka::MaterialManager::Param>& getParams() const
    {
        return mMaterialParams;
    }
    
private:
    const MaterialNetworkTranslator& m_translator;
    bool mIsMdl = false;
    std::string mMaterialXCode;
    // MDL related
    std::string mMdlFileUri;
    std::string mMdlSubIdentifier;

    std::vector<oka::MaterialManager::Param> mMaterialParams;
};

PXR_NAMESPACE_CLOSE_SCOPE
