#pragma once
#include <strelka/render/common.h>
#include <strelka/render/buffer.h>
#include <strelka/scene/scene.h>
#include <atomic>

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

    /// Synchronous render: calls render() and waits for GPU completion.
    virtual void renderSync(Buffer* output) { render(output); }

    /// Start a render pass if the GPU is idle. Non-blocking.
    virtual void triggerRenderIfIdle() {}

    /// Return the last completed output buffer, or nullptr if none ready yet.
    virtual Buffer* getReadyBuffer() { return nullptr; }

    /// Last completed render frame time in milliseconds (GPU time).
    double getLastRenderTimeMs() const { return mLastRenderTimeMs.load(std::memory_order_relaxed); }

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
    std::atomic<double> mLastRenderTimeMs{0.0};
};

class RenderFactory
{
public:
    static Render* createRender(RenderType type);
    static Render* createRender();
};

} // namespace oka
