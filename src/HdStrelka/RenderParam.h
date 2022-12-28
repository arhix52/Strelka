#pragma once

#include "pxr/pxr.h"
#include "pxr/imaging/hd/renderDelegate.h"
#include "pxr/imaging/hd/renderThread.h"
#include <scene/scene.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdStrelkaRenderParam final : public HdRenderParam
{
public:
    HdStrelkaRenderParam(oka::Scene* scene, HdRenderThread* renderThread, std::atomic<int>* sceneVersion)
        : mScene(scene), mRenderThread(renderThread), mSceneVersion(sceneVersion)
    {
    }
    virtual ~HdStrelkaRenderParam() = default;

    /// Accessor for the top-level embree scene.
    oka::Scene* AcquireSceneForEdit()
    {
        mRenderThread->StopRender();
        (*mSceneVersion)++;
        return mScene;
    }

private:
    oka::Scene* mScene;
    /// A handle to the global render thread.
    HdRenderThread* mRenderThread;
    /// A version counter for edits to mScene.
    std::atomic<int>* mSceneVersion;
};

PXR_NAMESPACE_CLOSE_SCOPE
