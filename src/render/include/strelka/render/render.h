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

    /// Synchronous render: calls render() and waits for GPU completion.
    virtual void renderSync(Buffer* output)
    {
        render(output);
    }

    /// Start a render pass if the GPU is idle. Non-blocking.
    virtual void triggerRenderIfIdle() {}

    /// True while a submitted frame has not finished on the GPU. The interactive
    /// loop does not need this -- it just draws whatever is ready -- but anything
    /// measuring a frame has to know when that frame is actually there, and
    /// sleeping a guessed interval instead makes the measurement a race.
    virtual bool isRenderBusy() const { return false; }

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

    /// The guide textures handed to the denoiser, in the order the denoiser reads
    /// them. Named rather than indexed by number so a reordering in the renderer
    /// cannot silently change what a check is checking.
    enum class Guide : uint32_t
    {
        Color = 0,
        Depth,
        Motion,
        DiffuseAlbedo,
        SpecularAlbedo,
        Normal,
        Roughness,
        SpecularHitDistance,
        Reactive,
        Denoised,
        Count,
    };

    /// Whether the scene is currently traversed as motion geometry, i.e. whether
    /// the frame being produced actually carries motion blur. Exposed because the
    /// alternative -- inferring it from image sharpness -- is confounded by
    /// accumulation, which removes noise and changes the same gradient measure.
    virtual bool motionGeometryActive()
    {
        return false;
    }

    /// Diagonal of the bounding box of the skinned vertices, read back from the
    /// GPU. Negative when the backend cannot answer or nothing in the scene is
    /// skinned. This exists because a character collapsing to a point is
    /// invisible to every whole-frame metric -- it is small next to its
    /// surroundings, so coverage and mean brightness barely move -- and that is
    /// exactly how a broken skinning submission went unnoticed.
    virtual float skinnedGeometryExtent()
    {
        return -1.0f;
    }

    /// Read one guide back as RGBA floats, unused channels zeroed. This reads the
    /// exact bytes MetalFX consumes, which is the only way to check a guide
    /// without also testing everything downstream of it. False when the backend
    /// has no such texture, or when denoising has never run.
    virtual bool readGuideTexture(Guide, std::vector<float>&, uint32_t&, uint32_t&)
    {
        return false;
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
