#pragma once

#include <strelka/scene/scene.h>

#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/sceneDelegate.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdStrelkaMaterial final : public HdMaterial
{
public:
    HF_MALLOC_TAG_NEW("new HdStrelkaMaterial");

    HdStrelkaMaterial(const SdfPath& id);

    ~HdStrelkaMaterial() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

    const oka::Scene::MaterialDescription& getDescription() const
    {
        return mDescription;
    }

private:
    oka::Scene::MaterialDescription mDescription;
};

PXR_NAMESPACE_CLOSE_SCOPE
