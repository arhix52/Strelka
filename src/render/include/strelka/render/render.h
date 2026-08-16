#pragma once

#include <vector>
#include <string>
#include <strelka/render/common.h>
#include <strelka/render/buffer.h>
#include <strelka/scene/scene.h>

#include <loadprogress.h>

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
    struct ReadyFrame
    {
        Buffer* buffer = nullptr;
        void* texture = nullptr;
    };

    virtual ~Render() = default;

    virtual void init() = 0;

    /// Tell the renderer that the next frame has no valid temporal predecessor --
    /// a camera cut, a scene change, anything that breaks pixel-to-pixel
    /// correspondence. Smooth camera motion is *not* such an event: motion vectors
    /// exist to carry it, and resetting on it throws the history away exactly when
    /// it is worth most.
    /// True when a GPU command buffer failed; the image is not trustworthy.
    virtual bool deviceError() const
    {
        return false;
    }

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

    /// Capture one frame into a .gputrace document for Xcode's shader profiler.
    ///
    /// The only place that reports what a shader spends its registers on: the
    /// public API offers `maxTotalThreadsPerThreadgroup` and nothing else, and
    /// the offline `metal` compiler produces AIR, where registers are still
    /// virtual. Requires MTL_CAPTURE_ENABLED=1 in the environment *before* the
    /// device is created -- main() sets it when the flag is given.
    virtual void beginGpuCapture(const std::string&)
    {
    }
    virtual void endGpuCapture()
    {
    }

    /// Start a render pass if the GPU is idle. Non-blocking.
    virtual void triggerRenderIfIdle()
    {
    }

    /// True while a submitted frame has not finished on the GPU. The interactive
    /// loop does not need this -- it just draws whatever is ready -- but anything
    /// measuring a frame has to know when that frame is actually there, and
    /// sleeping a guessed interval instead makes the measurement a race.
    virtual bool isRenderBusy() const
    {
        return false;
    }
    /// Which denoiser this backend actually has.
    ///
    /// Not a detail the renderer needs -- each backend only ever runs its own --
    /// but the editor has one panel for both, and a panel that cannot ask ends up
    /// naming whichever backend was written first. It named MetalFX on OptiX, and
    /// offered MetalFX's free render-scale slider for a model that has exactly
    /// two ratios.
    enum class DenoiserKind : uint32_t
    {
        eNone = 0, ///< this backend denoises nothing
        eMetalFx, ///< MetalFX spatial / temporal scalers
        eOptixAi ///< the OptiX AI denoiser, see optix_denoise_plan.h
    };
    virtual DenoiserKind denoiserKind() const
    {
        return DenoiserKind::eNone;
    }

    /// True when the denoiser the settings asked for could not run and the frame
    /// fell back to something else. What "something else" is differs by backend,
    /// so the message belongs to whoever is drawing it.
    virtual bool denoiserFallbackActive() const
    {
        return false;
    }

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

    virtual Buffer* getReadyBuffer()
    {
        return nullptr;
    }
    virtual ReadyFrame getReadyFrame()
    {
        return { getReadyBuffer(), getReadyTexture() };
    }

    /// Where the memory went, measured rather than estimated.
    ///
    /// Every entry is the size the API reports for the objects themselves, so a
    /// category cannot drift from reality as the code around it changes. The two
    /// totals are independent ground truth -- what the device says it has
    /// allocated, and what the OS says the process occupies -- and the panel
    /// showing this subtracts the categories from them, so anything not accounted
    /// for appears as a slice of its own instead of quietly going missing.
    struct MemoryReport
    {
        struct Entry
        {
            const char* name = nullptr;
            size_t bytes = 0;
        };
        std::vector<Entry> gpu{};
        std::vector<Entry> cpu{};
        size_t deviceAllocated = 0; ///< the backend's own total, 0 if it cannot say
        size_t processFootprint = 0; ///< what the OS charges this process
    };
    virtual bool memoryReport(MemoryReport&) const
    {
        return false;
    }

    /// Where to report the GPU-side scene build, and where to read cancellation
    /// from. The editor hands the same object to the loader, so one bar covers
    /// the parse and the build without the two having to agree on anything.
    void setLoadProgress(LoadProgress* progress)
    {
        mLoadProgress = progress;
    }

    /// True while the scene's GPU resources are still being built. A backend that
    /// builds them in chunks does one chunk per render() call and produces no
    /// image until this goes false; one that builds them up front never returns
    /// true, and the caller's loop is the same either way.
    virtual bool isBuildingScene() const
    {
        return false;
    }

    /// Last completed render frame time in milliseconds (GPU time).
    double getLastRenderTimeMs() const
    {
        return mLastRenderTimeMs.load(std::memory_order_relaxed);
    }

    virtual void* getNativeDevicePtr()
    {
        return nullptr;
    }

    virtual void* getNativeCommandQueue()
    {
        return nullptr;
    }

    /// The event a finished frame signals, and the value it signals with.
    ///
    /// A backend that renders on a different queue than the one displaying the
    /// result has to say so: Metal orders work within a queue and not across
    /// two, so a display that samples the render's texture without waiting reads
    /// whatever was there -- a black frame, or half of one. Null means the render
    /// and the display share a queue and nothing is needed.
    virtual void* getNativeFrameEvent()
    {
        return nullptr;
    }
    virtual uint64_t frameEventValue() const
    {
        return 0;
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
    const SettingsManager* getSettings() const
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
    /// Null unless a caller asked for progress; every use is guarded.
    LoadProgress* mLoadProgress = nullptr;
    SettingsManager* mSettingsManager = nullptr;
    SharedContext* mSharedCtx = nullptr;
    oka::Scene* mScene = nullptr;
    std::atomic<double> mLastRenderTimeMs{ 0.0 };
};

class RenderFactory
{
public:
    static Render* createRender(RenderType type);
    static Render* createRender();
};

} // namespace oka
