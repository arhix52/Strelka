#pragma once
#include <strelka/render/common.h>
#include <strelka/render/buffer.h>
#include <strelka/scene/scene.h>

namespace oka
{

enum class RenderType : int
{
    eOptiX = 0,
    eMetal,
    eCompute,
};

/**
 * Render interface
 */
class Render
{
public:
    virtual ~Render() = default;

    virtual void init() = 0;
    virtual void render(Buffer* output) = 0;
    virtual Buffer* createBuffer(const BufferDesc& desc) = 0;

    virtual void* getNativeDevicePtr()
    {
        return nullptr;
    }

    virtual void* getNativeCommandQueue()
    {
        return nullptr;
    }

    void setSharedContext(SharedContext* ctx)
    {
        mSharedCtx = ctx;
    }

    SharedContext& getSharedContext()
    {
        return *mSharedCtx;
    }

    void setSettingsManager(SettingsManager* settigns)
    {
        mSettingsManager = settigns;
    }

    SettingsManager* getSettings()
    {
        return mSettingsManager;
    }

    void setScene(Scene* scene)
    {
        mScene = scene;
    }

    Scene* getScene()
    {
        return mScene;
    }

protected:
    SettingsManager* mSettingsManager;
    SharedContext* mSharedCtx = nullptr;
    oka::Scene* mScene = nullptr;
};

class RenderFactory
{
public:
    static Render* createRender(RenderType type);
    static Render* createRender();
};

} // namespace oka
