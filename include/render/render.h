#pragma once
#include "common.h"
#include "buffer.h"
#include <scene/scene.h>

namespace oka
{

/**
 * Render interface
 */
class Render
{
public:
    virtual ~Render(){};

    virtual void init() = 0;
    virtual void render(Buffer* output) = 0;
    virtual Buffer* createBuffer(const BufferDesc& desc) = 0;

    void setSharedContext(SharedContext* ctx)
    {
        mSharedCtx = ctx;
    }

    SharedContext& getSharedContext()
    {
        return *mSharedCtx;
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
    SharedContext* mSharedCtx = nullptr;
    oka::Scene* mScene = nullptr;
};

enum class RenderType: int
{
    eOptiX = 0,
    eMetal,
    eCompute,
};

class RenderFactory
{
public:
    static Render* createRender(const RenderType type); 
};

} // namespace oka