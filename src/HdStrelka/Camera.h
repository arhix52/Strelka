#pragma once

#include <pxr/imaging/hd/camera.h>

#include <scene/scene.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdStrelkaCamera final : public HdCamera
{
public:
    HdStrelkaCamera(const SdfPath& id, oka::Scene& scene);

    ~HdStrelkaCamera() override;

public:
    float GetVFov() const;

    uint32_t GetCameraIndex() const;

public:
    void Sync(HdSceneDelegate* sceneDelegate,
              HdRenderParam* renderParam,
              HdDirtyBits* dirtyBits) override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
    oka::Camera _ConstructStrelkaCamera();

    float m_vfov;
    oka::Scene& mScene;
    uint32_t mCameraIndex = -1;
};

PXR_NAMESPACE_CLOSE_SCOPE
