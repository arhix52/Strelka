#include "Camera.h"

#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/camera.h>

#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/matrix_decompose.hpp>

PXR_NAMESPACE_OPEN_SCOPE

HdStrelkaCamera::HdStrelkaCamera(const SdfPath& id, oka::Scene& scene) : HdCamera(id), mScene(scene), m_vfov(M_PI_2)
{
    const std::string& name = id.GetString();
    oka::Camera okaCamera;
    okaCamera.name = name;
    mCameraIndex = mScene.addCamera(okaCamera);
}

HdStrelkaCamera::~HdStrelkaCamera()
{
}

float HdStrelkaCamera::GetVFov() const
{
    return m_vfov;
}

uint32_t HdStrelkaCamera::GetCameraIndex() const
{
    return mCameraIndex;
}

void HdStrelkaCamera::Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits)
{
    HdDirtyBits dirtyBitsCopy = *dirtyBits;

    HdCamera::Sync(sceneDelegate, renderParam, &dirtyBitsCopy);
    if (*dirtyBits & DirtyBits::DirtyParams)
    {
        // See https://wiki.panotools.org/Field_of_View
        float aperture = _verticalAperture * GfCamera::APERTURE_UNIT;
        float focalLength = _focalLength * GfCamera::FOCAL_LENGTH_UNIT;
        float vfov = 2.0f * std::atanf(aperture / (2.0f * focalLength));

        m_vfov = vfov;
        oka::Camera cam = _ConstructStrelkaCamera();
        mScene.updateCamera(cam, mCameraIndex);
    }

    *dirtyBits = DirtyBits::Clean;
}

HdDirtyBits HdStrelkaCamera::GetInitialDirtyBitsMask() const
{
    return DirtyBits::DirtyParams | DirtyBits::DirtyTransform;
}

oka::Camera HdStrelkaCamera::_ConstructStrelkaCamera()
{
    oka::Camera strelkaCamera;
    GfMatrix4d perspMatrix = ComputeProjectionMatrix();
    GfMatrix4d absInvViewMatrix = GetTransform();
    GfMatrix4d relViewMatrix = absInvViewMatrix; //*m_rootMatrix;
    glm::float4x4 xform;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            xform[i][j] = (float)relViewMatrix[i][j];
        }
    }
    glm::float4x4 persp;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            persp[i][j] = (float)perspMatrix[i][j];
        }
    }
    {
        glm::vec3 scale;
        glm::quat rotation;
        glm::vec3 translation;
        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(xform, scale, rotation, translation, skew, perspective);
        rotation = glm::conjugate(rotation);
        strelkaCamera.position = translation * scale;
        strelkaCamera.mOrientation = rotation;
    }
    strelkaCamera.matrices.perspective = persp;
    strelkaCamera.matrices.invPerspective = glm::inverse(persp);
    strelkaCamera.fov = glm::degrees(GetVFov());

    const std::string& name = GetId().GetString();
    strelkaCamera.name = name;

    return strelkaCamera;
}

PXR_NAMESPACE_CLOSE_SCOPE
