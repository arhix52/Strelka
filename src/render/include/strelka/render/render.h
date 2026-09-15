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
        uint64_t frameSerial = 0;
        PresentationMetadata presentation{};
    };

    virtual ~Render() = default;

    virtual void init() = 0;

    virtual bool isReady() const
    {
        return true;
    }

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

    virtual std::string renderWorkAuditJson() const
    {
        return {};
    }

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

    virtual bool isRenderBusy()
    {
        return false;
    }
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

    virtual bool radianceCacheOccupancy(uint32_t& entriesUsed, uint32_t& capacity) const
    {
        (void)entriesUsed;
        (void)capacity;
        return false;
    }

    virtual bool readDisplayTexture(std::vector<float>&, uint32_t&, uint32_t&)
    {
        return false;
    }

    /// Read an SDR display transform for formats such as PNG. HDR backends
    /// override this so highlights are compressed to SDR instead of clipped
    /// after an EDR transform.
    virtual bool readDisplayTextureSdr(std::vector<float>& out, uint32_t& width, uint32_t& height)
    {
        return readDisplayTexture(out, width, height);
    }

    virtual bool readDisplayTextureHdr(std::vector<float>& out, uint32_t& width, uint32_t& height)
    {
        return readDisplayTexture(out, width, height);
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

    virtual bool motionGeometryActive()
    {
        return false;
    }

    virtual float skinnedGeometryExtent()
    {
        return -1.0f;
    }

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

    virtual bool isBuildingScene() const
    {
        return false;
    }

    virtual double pipelineCompileElapsedMs() const
    {
        return -1.0;
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

    /// CUDA identity used by external graphics interop. Non-CUDA backends keep
    /// these defaults so Metal remains independent of CUDA headers and types.
    virtual int activeCudaDeviceOrdinal() const
    {
        return -1;
    }
    virtual void* getNativeCudaStream()
    {
        return nullptr;
    }

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
    static Render* createRender();
};

} // namespace oka
