#pragma once

#include "pxr/pxr.h"
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/sceneDelegate.h>

#include <scene/scene.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdStrelkaLight final : public HdLight
{
public:
    HF_MALLOC_TAG_NEW("new HdStrelkaLight");

    HdStrelkaLight(const SdfPath& id, TfToken const& lightType);

    ~HdStrelkaLight() override;

public:
    void Sync(HdSceneDelegate* delegate,
              HdRenderParam* renderParam,
              HdDirtyBits* dirtyBits) override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;

    oka::Scene::UniformLightDesc getLightDesc();

private:
    TfToken mLightType;
    oka::Scene::UniformLightDesc mLightDesc;
};

PXR_NAMESPACE_CLOSE_SCOPE
