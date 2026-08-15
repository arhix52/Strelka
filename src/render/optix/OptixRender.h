#pragma once

#include <strelka/render/render.h>

#include <optix.h>

#include "OptixRenderParams.h"

#include <strelka/scene/scene.h>

#include "cuda_checks.h"
#include <strelka/render/common.h>
#include "OptixBuffer.h"
#include "OptixDenoiser.h"
#include "optix_denoise_plan.h"
#include "texture_upload_plan.h"

#include "OptixScenePreparation.h"
#include "gpu_stage_breadcrumb.h"
// Host-side and backend-neutral despite living under metal/: it decides whether
// a half-built scene can be traced at all and how often the partial picture may
// be republished, and neither question has anything Metal in it. Duplicating it
// here would mean two answers to one question, and the copy that has the test
// would be the one that stayed right.
#include <metal/scene_stream.h>

#include <cuda_runtime.h>

#include <atomic>
#include <string>
#include <unordered_map>

struct Texture;

namespace oka
{

struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle ias_handle;
    CUdeviceptr d_instances = 0;
    size_t d_instances_size = 0;

    OptixModuleCompileOptions module_compile_options = {};
    OptixModule ptx_module = 0;
    OptixModule closest_hit_module = 0; // Loaded from OptixRender_closest_hit.cu.optixir
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline pipeline = 0;
    OptixModule m_catromCurveModule = 0;
    /// Built-in intersector for round *linear* curves. A separate module from
    /// the cubic one because the basis is baked into the intersector, so a
    /// scene with both kinds of strand needs both hit groups.
    OptixModule m_linearCurveModule = 0;

    OptixProgramGroup raygen_prog_group = 0;
    OptixProgramGroup radiance_miss_group = 0;
    OptixProgramGroup occlusion_miss_group = 0;
    OptixProgramGroup radiance_default_hit_group = 0;
    OptixProgramGroup radiance_linear_curve_hit_group = 0;
    std::vector<OptixProgramGroup> radiance_hit_groups;
    OptixProgramGroup occlusion_hit_group = 0;
    OptixProgramGroup occlusion_linear_curve_hit_group = 0;
    OptixProgramGroup light_hit_group = 0;
    CUstream stream = 0;
    Params params = {};
    Params prevParams = {};

    std::unique_ptr<OptixBuffer> mParamsBuffer;

    OptixShaderBindingTable sbt = {};
};

class OptiXRender : public Render
{
private:
    struct Mesh
    {
        OptixTraversableHandle gas_handle = 0;
        CUdeviceptr d_gas_output_buffer = 0;
        /// What the structure actually occupies, after compaction if it happened.
        /// Recorded rather than recomputed: the memory report has no other way to
        /// ask a bare CUdeviceptr how big it is, and an estimate made from the
        /// triangle count would be the thing the report exists to avoid.
        size_t gas_bytes = 0;
        ~Mesh()
        {
            CUDA_CHECK(cudaFree((void*)d_gas_output_buffer));
        }
    };

    struct Curve
    {
        OptixTraversableHandle gas_handle = 0;
        CUdeviceptr d_gas_output_buffer = 0;
        /// Which basis this set was built under. The hit group has to match:
        /// fetching a linear segment with the cubic accessor reads four control
        /// points where two were stored.
        bool isLinear = true;
        /// Segments per strand, or 0 when the set's strands differ in length.
        uint32_t segmentsPerStrand = 0;
        size_t gas_bytes = 0;
        ~Curve()
        {
            CUDA_CHECK(cudaFree((void*)d_gas_output_buffer));
        }
    };

    struct Instance
    {
        OptixInstance instance;
    };

    // Per-material data (host-side; GPU data uploaded to shared device buffers)
    struct Material
    {
        MaterialParams params; // Resolved material parameters (with texture indices set)
    };

    struct View
    {
        oka::Camera::Matrices mCamMatrices;
    };

    struct DeviceSkinningPtrs
    {
        sutil::Matrix4x4* d_jointMats = nullptr;
        size_t bytes = 0;
        ~DeviceSkinningPtrs()
        {
            if (d_jointMats)
                cudaFree(d_jointMats);
        }
    };
    DeviceSkinningPtrs mSkinningPtrs;
    std::vector<int> mJointMatOffsets;

    std::vector<oka::Instance> mPrevInstances;

    View mPrevView;

    PathTracerState mState;
    bool mEnableValidation;
    bool mEnableMotionBlur;
    bool mShaderReorderSupported = false;

    /// Set whenever the TLAS is rebuilt from scratch. The SBT is indexed by
    /// instance, so it has to be rebuilt with it; a refit leaves it alone.
    bool mSbtDirty = false;

    /// How many skeletal BLASes may be rebuilt from scratch in one frame, and
    /// where the next frame's round-robin picks up. Bounded so that no single
    /// frame pays for the whole cast; the same shape as Metal's
    /// kMaxBlasRebuildsPerFrame.
    static constexpr size_t kMaxBlasRebuildsPerFrame = 8;
    size_t mNextBlasRebuildIndex = 0;

    /// Instance count the current TLAS was built for. A refit cannot change it,
    /// so a different one is what forces a rebuild.
    size_t mTlasInstanceCount = 0;

    // Previous-frame settings for change detection (replaces static locals in render())
    uint32_t mPrevRectLightSamplingMethod = 0;
    bool mPrevEnableAccumulation = false;
    uint32_t mPrevSspTotal = 0;

    // Device buffers for per-material data (indexed by materialId)
    std::unique_ptr<OptixBuffer> mMaterialParamsBuffer; // MaterialParams[] on device
    uint32_t mMaterialCount = 0;

    void allocJointMatrices();
    std::unique_ptr<Mesh> createMesh(const oka::Mesh& mesh);
    void updateMesh(const oka::Mesh& mesh, int optixMeshesId);
    bool rebuildMesh(const oka::Mesh& mesh, int optixMeshesId);
    std::unique_ptr<Curve> createCurve(const oka::Curve& curve);
    size_t compactAccel(CUdeviceptr& buffer, OptixTraversableHandle& handle, CUdeviceptr result, size_t outputSizeInBytes);

    std::vector<std::unique_ptr<Mesh>> mOptixMeshes;
    std::vector<std::unique_ptr<Curve>> mOptixCurves;

    std::unique_ptr<OptixBuffer> mVertexBuffer;
    std::unique_ptr<OptixBuffer> mPrevVertexBuffer;
    const int NUM_MOTION_KEYS = 2;
    std::unique_ptr<OptixBuffer> mVertexSkinDataBuffer;
    std::unique_ptr<OptixBuffer> mIndexBuffer;
    std::unique_ptr<OptixBuffer> mLightBuffer;
    // TODO: move to raii buffers
    std::unique_ptr<OptixBuffer> mPointsBuffer;
    std::unique_ptr<OptixBuffer> mWidthsBuffer;

    std::vector<std::shared_ptr<OptixBuffer>> mMotionTransformBuffers; // used for motion blur

    std::unique_ptr<OptixBuffer> mTlasBuffer;
    // Bytes the last instance-structure build actually wrote. Not the same as
    // mTlasBuffer->size(): that buffer is reused across scenes and only grows,
    // and a refit has to be told the size of the structure inside it.
    size_t mTlasOutputSize = 0;

    std::unique_ptr<OptixBuffer> mTexturesDataBuffer; // Consolidated GPU texture object array

    // Temporary buffers for GAS building
    // These buffers are reused across multiple GAS builds to reduce allocations
    // They are automatically resized if needed but never shrink
    std::unique_ptr<OptixBuffer> mTempAccelBuffer;        // Temporary buffer for acceleration structure building
    std::unique_ptr<OptixBuffer> mCompactedSizeBuffer;  // Buffer for storing compaction size results
    std::unique_ptr<OptixBuffer> mSegmentIndicesBuffer; // Buffer for curve segment indices

    void createVertexBuffer();
    void createPrevBuffers();
    void createVertexSkinDataBuffer();
    void createIndexBuffer();

    // curve utils
    void createPointsBuffer();
    void createWidthsBuffer();

    void createLightBuffer();

    oka::optix_tex::DecodeSettings textureDecodeSettings() const;
    Texture loadTextureFromFile(const std::string& fileName, oka::optix_tex::Kind kind);
    void loadEnvMap(const std::string& texturePath);
    void loadEnvBackground(const std::string& texturePath);

    void destroyTextures();
    void destroyMaterialTextures();

    std::vector<Material> mMaterials;

    // Texture resource tracking for cleanup. Material textures are tracked
    // apart from the environment map's because createOptixMaterials() now runs
    // again whenever a material changes, and it has to be able to release the
    // set it loaded last time without taking the env map -- whose texture object
    // is already sitting in Params -- with it.
    std::vector<cudaArray_t> mTextureArrays;
    std::vector<cudaMipmappedArray_t> mTextureMipmappedArrays;
    std::vector<cudaTextureObject_t> mTextureObjects;
    uint32_t mTextureCacheHits = 0;
    uint32_t mTextureCacheMisses = 0;
    std::vector<cudaArray_t> mMaterialTextureArrays;
    std::vector<cudaMipmappedArray_t> mMaterialTextureMipmappedArrays;
    std::vector<cudaTextureObject_t> mMaterialTextureObjects;

    // Environment map resources
    std::unique_ptr<OptixBuffer> mEnvAliasBuffer; // Walker/Vose alias table, one entry per texel
    bool mEnvMapLoaded = false;
    float mEnvMapAutoScale = 1.0f; // opt-in HDRI unit reconciliation; 1 unless render/env/autoCalibrate

    void updatePathtracerParams(const uint32_t width, const uint32_t height);

    // --- Denoiser / guides -----------------------------------------------
    OptixDenoiserContext mDenoiser;
    DenoisePlan mDenoisePlan{};
    /// Guide records, one per pixel at render resolution.
    std::unique_ptr<OptixBuffer> mAovBuffer;
    /// The four images the denoiser reads, kept apart from the packed records
    /// because the network wants them in its own formats.
    std::unique_ptr<OptixBuffer> mDenoiseColorBuffer;
    std::unique_ptr<OptixBuffer> mDenoiseAlbedoBuffer;
    std::unique_ptr<OptixBuffer> mDenoiseNormalBuffer;
    std::unique_ptr<OptixBuffer> mDenoiseFlowBuffer;
    /// How far the flow vector at each pixel is to be believed -- the reactive
    /// mask, inverted. OptiX reads this as a single float per pixel.
    std::unique_ptr<OptixBuffer> mDenoiseFlowTrustBuffer;
    /// Radiance at render resolution when that is not the output resolution,
    /// i.e. when the 2x model is upscaling into the caller's buffer.
    std::unique_ptr<OptixBuffer> mRenderImageBuffer;
    /// The buffer the caller last got, so readDisplayTexture() can hand back the
    /// same pixels the screen is showing rather than an intermediate.
    void* mDisplayImage = nullptr;
    uint32_t mDisplayWidth = 0;
    uint32_t mDisplayHeight = 0;
    /// Raised by resetTemporalHistory() and consumed by the next render().
    bool mResetTemporalHistory = true;
    /// True when denoising was asked for and could not be provided.
    bool mDenoiserFallback = false;
    bool mPrevGuidePrimaryHit = false;

    /// Size the guide and denoiser buffers for a plan, reallocating only what
    /// changed.
    void updateGuideBuffers(const DenoisePlan& plan);
    // ---------------------------------------------------------------- timing --
    // A frame is several asynchronous submissions on one stream, so wall clock
    // around render() measures how long it took to *enqueue* them, which on this
    // backend is microseconds however long the GPU then works. Events are the
    // only thing that answers the question the editor's title bar is asking.
    cudaEvent_t mFrameStartEvent = nullptr;
    cudaEvent_t mFrameStopEvent = nullptr;
    /// A pair of events has been recorded and not yet read back.
    bool mFrameTimingPending = false;
    void createTimingEvents();
    /// Reads the recorded pair into mLastRenderTimeMs. `wait` blocks until the
    /// stop event has passed; without it the read is skipped when the frame is
    /// still running, and the previous frame's number stands.
    void collectFrameTiming(bool wait);

    // ----------------------------------------------------------- device error --
    // Latched rather than fatal. An abort inside the renderer takes the editor
    // and the harness down with it and leaves nothing to inspect; a latch lets
    // StrelkaCLI exit non-zero instead of writing a black EXR that reads as a
    // lighting bug, which is the logic it already has and never saw an error to
    // trigger.
    bool mDeviceError = false;
    bool mDeviceErrorReported = false;
    /// Latches and reports. Returns true when `err` was a failure.
    bool latchCudaError(cudaError_t err, const char* what);
    /// Synchronises the frame's work and latches whatever it reports.
    void syncFrameAndLatchErrors();

    // ------------------------------------------------------------ breadcrumbs --
    /// One byte per GpuStage, written by the device in stream order after that
    /// stage's work. Read back only when something failed.
    std::unique_ptr<OptixBuffer> mStageMarkBuffer;
    uint8_t mStageSubmitted[optix::kGpuStageCount] = {};
    void beginFrameBreadcrumbs();
    /// Enqueues stage `stage`'s completion mark behind the work just submitted.
    void markStageSubmitted(optix::GpuStage stage, CUstream stream);
    void reportGpuStageFailure();

    // -------------------------------------------------- nested-dielectric losses --
    /// Three counters the shading path raises when the IOR stack loses a path:
    /// a push onto a full stack, a pop that matched nothing, and a path that
    /// reached the environment still inside a medium. Zeroed before each launch
    /// and read back once per scene, because the numbers are a property of the
    /// asset rather than of the frame. Metal's MetalWavefrontIntegrator reports
    /// the same three; see entry 5 of docs/open-defects.md.
    std::unique_ptr<OptixBuffer> mIorStatsBuffer;
    bool mReportedIorStats = false;
    /// Reads the counters back and warns once, if any of them fired.
    void reportIorStackStats();

    // ------------------------------------------------------------- scene build --
    optix::OptixScenePreparation mScenePrep;
    metal::PublishClock mPublishClock;
    double mBuildStartMs = 0.0;
    bool mReportedFirstPartialFrame = false;
    /// Where the sliced stages left off.
    size_t mBlasMeshCursor = 0;
    size_t mBlasCurveCursor = 0;
    bool mTopLevelBuilt = false;
    size_t mMaterialTextureCursor = 0;
    /// Host mirror of the flat texture-object table, so a slice can rewrite one
    /// material's block without re-reading the device.
    std::vector<cudaTextureObject_t> mHostMaterialTextures;
    std::unordered_map<std::string, cudaTextureObject_t> mTextureCache;

    optix::SceneBuildHooks makeSceneBuildHooks();
    void buildSceneBuffers();
    void buildSceneEnvironment(Buffer* output);
    void publishMaterialParams();
    bool stepStructures(double budgetMs);
    bool stepMaterialTextures(double budgetMs);
    void buildSceneTail(Buffer* output);
    /// A top level with no instances in it. Every ray then misses and reaches
    /// the environment, which is a correct picture of a scene whose geometry has
    /// not arrived yet rather than a broken one.
    void buildEmptyTopLevel();
    bool stepSceneBuild(Buffer* output);
    void finishSceneBuild(Buffer* output);

    // ------------------------------------------------------------ async output --
    // Two buffers so the frame being displayed is never the one being written.
    std::atomic<bool> mRenderBusy{ false };
    std::atomic<int> mReadyIndex{ -1 };
    int mWriteIndex = 0;
    Buffer* mAsyncOutputBuffers[2] = { nullptr, nullptr };

    // ---------------------------------------------------------------- capture --
    bool mCaptureActive = false;

    // --------------------------------------------------------- memory tracking --
    /// Sizes of allocations the report cannot otherwise ask about, recorded where
    /// they are made. Everything with an OptixBuffer behind it is measured from
    /// the object instead.
    size_t mSbtBytes = 0;

public:
    OptiXRender(/* args */);
    ~OptiXRender();

    void init() override;
    void render(Buffer* output_buffer) override;
    void renderSync(Buffer* output) override;
    Buffer* createBuffer(const BufferDesc& desc) override;
    float skinnedGeometryExtent() override;

    void resetTemporalHistory() override
    {
        mResetTemporalHistory = true;
    }

    bool denoiserFallbackActive() const override
    {
        return mDenoiserFallback;
    }

    /// The finished frame, after tonemapping, exactly as the display shows it.
    bool readDisplayTexture(std::vector<float>& out, uint32_t& width, uint32_t& height) override;

    /// One guide, as RGBA floats with unused channels zeroed. These are the
    /// bytes the denoiser consumes, which is the only way to check a guide
    /// without also testing everything downstream of it.
    bool readGuideTexture(Guide guide, std::vector<float>& out, uint32_t& width, uint32_t& height) override;

    bool deviceError() const override
    {
        return mDeviceError;
    }

    bool isBuildingScene() const override
    {
        return mScenePrep.isBuilding();
    }

    bool isRenderBusy() const override
    {
        return mRenderBusy.load(std::memory_order_acquire);
    }

    void triggerRenderIfIdle() override;
    Buffer* getReadyBuffer() override;

    bool memoryReport(MemoryReport& report) const override;

    void beginGpuCapture(const std::string& path) override;
    void endGpuCapture() override;

    void applySkinning();
    void createContext();
    void createBottomLevelAccelerationStructures();
    bool updateBottomLevelAccelerationStructures();
    void createTopLevelAccelerationStructure();
    void updateTopLevelAccelerationStructure();
    void resolveInstanceGeometry(OptixInstance& oi, const oka::Instance& instance) const;

    /// Whether any material in the scene is the boundary of a participating
    /// medium. Gates the second traversal every shadow ray would otherwise take
    /// to accumulate optical depth across those boundaries.
    bool sceneHasBoundedMedium() const;
    void uploadInstancesToDevice(const std::vector<OptixInstance>& optixInstances);
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void createSbt();

};

} // namespace oka
