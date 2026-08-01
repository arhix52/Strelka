#pragma once

#include <vector>
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

    /// Tell the renderer that the next frame has no valid temporal predecessor --
    /// a camera cut, a scene change, anything that breaks pixel-to-pixel
    /// correspondence. Smooth camera motion is *not* such an event: motion vectors
    /// exist to carry it, and resetting on it throws the history away exactly when
    /// it is worth most.
    virtual void resetTemporalHistory()
    {
    }
    virtual void render(Buffer* output) = 0;
    virtual Buffer* createBuffer(const BufferDesc& desc) = 0;

    /// Start a render pass if the GPU is idle. Non-blocking.
    virtual void triggerRenderIfIdle() {}

    /// Return the last completed output buffer, or nullptr if none ready yet.
    /// The finished frame as a texture, when the backend can produce one.
    /// Nullptr means the caller should fall back to getReadyBuffer(); OptiX does.
    /// Returned as void* so this header stays free of Metal types.
    /// Read the finished frame back to the CPU as linear RGBA floats -- what the
    /// screen shows, after tonemapping and any post effect. False when the
    /// backend cannot.
    virtual bool readDisplayTexture(std::vector<float>&, uint32_t&, uint32_t&)
    {
        return false;
    }

    virtual void* getReadyTexture()
    {
        return nullptr;
    }

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
