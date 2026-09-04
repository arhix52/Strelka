#pragma once

#include <strelka/render/render.h>

#include <optix.h>

#include "OptixRenderParams.h"

#include <strelka/scene/scene.h>

#include "cuda_checks.h"
#include "device_ptr.h"
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
#include <host/scene_stream.h>

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <future>
#include <string>
#include <unordered_map>

struct Texture;

namespace oka
{

struct PathTracerState
{
    OptixDeviceContext context = nullptr;

    OptixTraversableHandle ias_handle = 0;
    CUdeviceptr d_instances = 0;
    size_t d_instances_size = 0;

    OptixModuleCompileOptions module_compile_options = {};
    OptixModule ptx_module = nullptr;
    OptixModule closest_hit_module = nullptr; // Loaded from OptixRender_closest_hit.cu.optixir
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline pipeline = nullptr;
    OptixModule m_catromCurveModule = nullptr;
    /// Built-in intersector for round *linear* curves. A separate module from
    /// the cubic one because the basis is baked into the intersector, so a
    /// scene with both kinds of strand needs both hit groups.
    OptixModule m_linearCurveModule = nullptr;

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup radiance_miss_group = nullptr;
    OptixProgramGroup occlusion_miss_group = nullptr;
    OptixProgramGroup radiance_default_hit_group = nullptr;
    OptixProgramGroup radiance_linear_curve_hit_group = nullptr;
    std::vector<OptixProgramGroup> radiance_hit_groups;
    OptixProgramGroup occlusion_hit_group = nullptr;
    OptixProgramGroup occlusion_linear_curve_hit_group = nullptr;
    OptixProgramGroup light_hit_group = nullptr;
    CUstream stream = nullptr;
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
        /// This mesh's opacity micromap array, or 0 when it has none.
        ///
        /// Owned here because the structure references it: an acceleration
        /// structure built with micromaps reads them during traversal, which is
        /// why optixAccelRelocate has an input for relocating them. Freeing it
        /// after the build would leave the GAS pointing at nothing.
        CUdeviceptr d_omm_array = 0;
        size_t omm_bytes = 0;
        ~Mesh()
        {
            CUDA_CHECK(cudaFree(optix::devicePtr<void>(d_gas_output_buffer)));
            if (d_omm_array)
            {
                CUDA_CHECK(cudaFree(optix::devicePtr<void>(d_omm_array)));
            }
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
            CUDA_CHECK(cudaFree(optix::devicePtr<void>(d_gas_output_buffer)));
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
    bool mEnableValidation = false;
    bool mEnableMotionBlur = false;
    bool mShaderReorderSupported = false;

    /// The launch-parameter fields the modules are compiled against as
    /// constants, so the branches they gate are not in the binary at all.
    ///
    /// Each folds a scene-wide branch and its unused code out of the pipeline.
    ///
    /// What is *not* here matters as much. `max_depth` is a slider, and a
    /// recompile is a visible hitch; anything a user drags has to stay a
    /// runtime read.
    struct PipelineSpec
    {
        uint32_t sharcCapacity = 0;
        /// Whether any light is responsive. A constant rather than a runtime
        /// read because it gates a second hash probe on every cached read and a
        /// second set of atomics on every deposit; a scene without a responsive
        /// light must not pay a branch for one.
        uint32_t sharcResponsive = 0;
        uint32_t debug = 0;
        uint32_t estimatorMode = 0;
        uint32_t volumeModel = 0;
        uint32_t misHeuristic = 0;
        uint32_t subsurfaceIterations = 0;
        uint32_t risCandidates = 1;
        uint32_t denoiseDepthMode = 0;
        bool hasBoundedMedium = false;
        bool hasFog = false;
        bool enableMotionBlur = false;
        bool writeAov = false;
        bool writeSplitAov = false;
        bool guidePrimaryHit = false;
        bool hasEnvMap = false;
        bool hasEnvBackground = false;
        bool enableShaderReorder = false;

        bool operator==(const PipelineSpec& other) const = default;
    };
    /// What the currently linked pipeline was compiled for. Compared against the
    /// frame's own values before every launch; a difference is a recompile.
    PipelineSpec mPipelineSpec;
    /// False until the first createModule(), so the initial build is not read as
    /// a respecialisation. Also false while a build is in flight, which is what
    /// stops render() launching against a pipeline that is being replaced.
    bool mPipelineSpecValid = false;

    /// The pipeline build running on a worker thread, if any.
    ///
    /// OptiX compiles the OPTIXIR module on the first launch of every distinct
    /// specialisation, and on a cold module cache that is ten seconds -- measured
    /// on the Cornell box, so it is the module and not the scene. Held on the
    /// main thread it exceeds mutter's five-second check-alive-timeout and the
    /// desktop offers to kill the editor. The scene build carries on stepping
    /// while this runs; only the launch waits.
    std::future<void> mPipelineBuild;
    std::chrono::steady_clock::time_point mPipelineBuildBegin;
    bool mPipelineBuildWasFirst = false;
    /// A synchronous caller wants the frame, not a responsive window. Set around
    /// renderSync() so the build is waited on rather than skipped.
    bool mPipelineBuildBlocking = false;

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

    // OpenPBR, in two arrays parallel to the two above and indexed by the same
    // material id. Both stay null when the scene shades with the glTF model and
    // authors no OpenPBR material, which is what the closest-hit program's null
    // check reads as "this model is not in this launch".
    std::unique_ptr<OptixBuffer> mOpenPBRParamsBuffer; // OpenPBRParams[] on device
    std::unique_ptr<OptixBuffer> mOpenPBRTexturesBuffer; // cudaTextureObject_t[n * MAX_OPENPBR_TEXTURES]
    std::vector<cudaTextureObject_t> mHostOpenPBRTextures;

    void allocJointMatrices();
    std::unique_ptr<Mesh> createMesh(const oka::Mesh& mesh, size_t meshIndex);
    void updateMesh(const oka::Mesh& mesh, int optixMeshesId);
    bool rebuildMesh(const oka::Mesh& mesh, int optixMeshesId);
    std::unique_ptr<Curve> createCurve(const oka::Curve& curve);
    size_t compactAccel(CUdeviceptr& buffer, OptixTraversableHandle& handle, CUdeviceptr result, size_t outputSizeInBytes);

    // --- Opacity micromaps ---------------------------------------------------
    //
    // Off by default. What they buy is traversal that resolves the wholly-opaque
    // and wholly-cut-away parts of an alpha cutout without entering a shader;
    // what they must not do is change what any surviving hit shades. See
    // opacity_micromap_policy.h for the two rules that keep that true.

    /// The base-colour alpha channel of one material, as the *device* texture
    /// holds it: decoded through the same plan the upload used, block
    /// compression included. Alpha read off the source file instead would
    /// describe a texture the renderer does not have.
    struct OmmAlphaImage
    {
        int width = 0;
        int height = 0;
        std::vector<float> alpha;
        /// How far a value here may sit from what the sampler returns.
        float tolerance = 0.0f;
        /// False when the uploaded format is one this cannot read back exactly,
        /// which means no micromap rather than a guessed one.
        bool usable = false;
    };

    /// Which material each mesh is drawn with, or -1 when no instance names it,
    /// or when two instances name different ones. A micromap belongs to the
    /// geometry, so a mesh instanced under two different cutouts cannot have one.
    std::vector<int32_t> mMeshMaterialIds;
    std::unordered_map<int32_t, OmmAlphaImage> mOmmAlphaCache;
    bool mOpacityMicromapsEnabled = false;
    size_t mOmmTotalBytes = 0;

    /// Everything one mesh's micromaps need to survive until the GAS build
    /// reads them. `array` outlives the build; the rest does not.
    struct MeshOpacityMicromap
    {
        CUdeviceptr array = 0;
        size_t arrayBytes = 0;
        CUdeviceptr indices = 0;
        std::vector<OptixOpacityMicromapUsageCount> usage;
        bool valid = false;
    };

    void beginOpacityMicromaps();
    void endOpacityMicromaps();
    void resolveMeshMaterials();
    const OmmAlphaImage* ommAlphaImage(int32_t materialId);
    MeshOpacityMicromap buildMeshOpacityMicromap(const oka::Mesh& mesh, size_t meshIndex);
    void releaseOpacityMicromapScratch(MeshOpacityMicromap& omm);

    std::vector<std::unique_ptr<Mesh>> mOptixMeshes;
    std::vector<std::unique_ptr<Curve>> mOptixCurves;

    std::unique_ptr<OptixBuffer> mVertexBuffer;
    std::unique_ptr<OptixBuffer> mPrevVertexBuffer;
    const int NUM_MOTION_KEYS = 2;
    std::unique_ptr<OptixBuffer> mVertexSkinDataBuffer;
    std::unique_ptr<OptixBuffer> mIndexBuffer;
    std::unique_ptr<OptixBuffer> mLightBuffer;
    std::unique_ptr<OptixBuffer> mEmissiveMeshBuffer;
    std::unique_ptr<OptixBuffer> mEmissiveTriangleBuffer;
    std::unique_ptr<OptixBuffer> mEmissiveInstanceTransformBuffer;
    std::unique_ptr<OptixBuffer> mPrevEmissiveInstanceTransformBuffer;
    double mAnalyticLightPower = 0.0;
    double mEmissiveMeshPower = 0.0;
    /// Packed IES candela tables for every profile the scene loaded, indexed by
    /// each light's points[0].y. Rebuilt with the light buffer.
    std::unique_ptr<OptixBuffer> mIesBuffer;
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
    void createEmissiveMeshLights();
    /// World bounds' diagonal: what the infinite lights' power proxy is scaled by.
    double sceneExtent() const;
    void updateEmitterSelectionProbabilities();
    void createIesBuffer();
    /// Images thrown by projector lights, indexed by each light's points[0].z.
    void createProjectorTextures();
    void destroyProjectorTextures();

    oka::optix_tex::DecodeSettings textureDecodeSettings() const;
    Texture loadTextureFromFile(const std::string& fileName, oka::optix_tex::Kind kind);
    void loadEnvMap(const std::string& texturePath);
    void loadEnvBackground(const std::string& texturePath);
    void updateSceneEnvironment();
    void destroyEnvironmentTextures();

    void destroyTextures();
    void destroyMaterialTextures();

    std::vector<Material> mMaterials;

    // Material textures are tracked apart from the environment because
    // publishMaterialParams() may replace them without replacing the env map.
    std::vector<cudaArray_t> mTextureArrays;
    std::vector<cudaMipmappedArray_t> mTextureMipmappedArrays;
    std::vector<cudaTextureObject_t> mTextureObjects;
    /// A textureless dome's colour, carried to the miss program. See
    /// buildSceneEnvironment for why a constant sky is a miss colour rather than
    /// a sampled light.
    float3 mMissColor = make_float3(0.0f);
    uint32_t mTextureCacheHits = 0;
    uint32_t mTextureCacheMisses = 0;
    std::vector<cudaArray_t> mMaterialTextureArrays;
    std::vector<cudaMipmappedArray_t> mMaterialTextureMipmappedArrays;
    std::vector<cudaTextureObject_t> mMaterialTextureObjects;
    // A third set, apart from both of the above: a projector's slide belongs to
    // the light set, so it survives a material reload, and it is not the
    // environment either.
    std::vector<cudaArray_t> mProjectorTextureArrays;
    std::vector<cudaTextureObject_t> mProjectorTextureObjects;
    std::unique_ptr<OptixBuffer> mProjectorTextureBuffer;

    // Environment map resources
    std::unique_ptr<OptixBuffer> mEnvAliasBuffer; // Walker/Vose alias table, one entry per texel
    bool mEnvMapLoaded = false;
    double mEnvMapPower = 0.0;
    float mEnvMapAutoScale = 1.0f; // opt-in HDRI unit reconciliation; 1 unless render/env/autoCalibrate

    void updatePathtracerParams(const uint32_t width, const uint32_t height);

    // --- Radiance cache ----------------------------------------------------
    //
    // Off by default, and off means `Params::sharcCapacity == 0`, which the
    // device code reads before anything else -- so the default is a
    // byte-for-byte no-op rather than a path that happens to agree.
    std::unique_ptr<OptixBuffer> mSharcBuffer;
    /// One SharcPathState per pixel, allocated only while the cache is on.
    std::unique_ptr<OptixBuffer> mSharcPathBuffer;
    size_t mSharcPathStateCount = 0;
    uint32_t mSharcCapacity = 0;
    /// Whether the table has to be cleared before the next launch.
    ///
    /// Raised when it is allocated, when the scene under it changes, and when a
    /// user asks -- and deliberately *not* when accumulation restarts, which is
    /// what it used to do. Accumulation restarts on every camera movement, so
    /// that rule cleared the whole table several times a second and the cache
    /// never held more than one frame of anything. Entries the camera has
    /// invalidated are now aged out one at a time by the resolve pass instead,
    /// which is what the SDK does and the reason it has a resolve pass at all.
    bool mSharcClearPending = false;
    /// Temporal window and eviction threshold, in frames, read from settings
    /// once per frame and handed to the resolve pass. See sharc_resolve.h.
    uint32_t mSharcAccumFrames = 32;
    uint32_t mSharcStaleFrames = 64;
    /// The window and lifetime of the responsive half, in frames. Short on
    /// purpose: it is the whole of what "responsive" means.
    uint32_t mSharcResponsiveFrames = 4;
    /// One bit per light, set where the light is responsive. Null when none is,
    /// which is also when params.sharcResponsive is 0.
    std::unique_ptr<OptixBuffer> mSharcResponsiveLightBuffer;
    uint32_t mSharcResponsiveLightCount = 0;
    void createSharcResponsiveLightBuffer();
    /// Where the eye was when the previous frame resolved, and whether there was
    /// one. Reprojection needs both cameras to work out which way the grid level
    /// moved under a voxel; on the first frame there is nothing to reproject
    /// from and the probe is skipped rather than fed the origin.
    float mSharcPrevCameraPosition[3] = { 0.0f, 0.0f, 0.0f };
    bool mSharcHasPrevCameraPosition = false;
    /// This frame's eye position, taken in updateSharcParams -- which has the
    /// camera -- and used by resolveSharc, which runs later and does not.
    float mSharcCameraPosition[3] = { 0.0f, 0.0f, 0.0f };
    /// One device word the occupancy kernel counts into, and the last value read
    /// back from it. Allocated only while somebody is looking (the editor panel
    /// asks for it); the readback is a frame behind so that nothing stalls on it.
    std::unique_ptr<OptixBuffer> mSharcOccupancyCounter;
    uint32_t mSharcOccupancyEntries = 0;
    void updateSharcParams(const oka::Camera& camera, uint32_t width, uint32_t height);
    /// Fold this frame's deposits into the cache and age it. Runs on the render
    /// stream straight after the launch, so the writes it reads are the ones
    /// that launch made and the next launch reads what it wrote.
    void resolveSharc();

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
    /// The last completed linear image. readDisplayTexture() copies this into a
    /// scratch allocation and presents the copy, preserving the published data.
    void *mDisplayImage = nullptr;
    uint32_t mDisplayWidth = 0;
    uint32_t mDisplayHeight = 0;
    std::unique_ptr<OptixBuffer> mDisplayReadbackBuffer;
    PresentationMetadata mDisplayPresentation{};
    PresentationMetadata mPendingPresentation{};
    bool readDisplayTextureWithMaxOutput(std::vector<float>& out,
                                         uint32_t& width,
                                         uint32_t& height,
                                         float maxOutput);
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
    /// asset rather than of the frame. Metal reports the same stack overflow,
    /// unmatched exit, and escaped-inside failures.
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
    /// The OpenPBR half of publishMaterialParams(). Called from it, and only from
    /// it -- the two tables have to be decided together or a material id would
    /// index one and not the other.
    void publishOpenPBRParams();
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
    int mCudaDeviceOrdinal = -1;
    Buffer *mAsyncOutputBuffers[2] = { nullptr, nullptr };
    uint64_t mFrameSerials[2] = {};
    PresentationMetadata mFramePresentation[2] = {};
    uint64_t mNextFrameSerial = 1;

    // ---------------------------------------------------------------- capture --
    bool mCaptureActive = false;

    // --------------------------------------------------------- memory tracking --
    /// Sizes of allocations the report cannot otherwise ask about, recorded where
    /// they are made. Everything with an OptixBuffer behind it is measured from
    /// the object instead.
    size_t mSbtBytes = 0;

public:
    OptiXRender(/* args */);
    ~OptiXRender() override;

    void init() override;
    void render(Buffer* output_buffer) override;
    void renderSync(Buffer* output) override;
    Buffer* createBuffer(const BufferDesc& desc) override;
    float skinnedGeometryExtent() override;

    void resetTemporalHistory() override
    {
        mResetTemporalHistory = true;
    }

    DenoiserKind denoiserKind() const override
    {
        return DenoiserKind::eOptixAi;
    }

    bool denoiserFallbackActive() const override
    {
        return mDenoiserFallback;
    }

    bool radianceCacheOccupancy(uint32_t& entriesUsed, uint32_t& capacity) const override
    {
        if (mSharcCapacity == 0 || !mSharcOccupancyCounter)
        {
            return false;
        }
        entriesUsed = mSharcOccupancyEntries;
        capacity = mSharcCapacity;
        return true;
    }

    /// The presented form of the last frame, synthesized without modifying the
    /// scene-linear published buffer.
    bool readDisplayTexture(std::vector<float>& out, uint32_t& width, uint32_t& height) override;
    bool readDisplayTextureSdr(std::vector<float>& out, uint32_t& width, uint32_t& height) override;

    /// One guide, as RGBA floats with unused channels zeroed. These are the
    /// bytes the denoiser consumes, which is the only way to check a guide
    /// without also testing everything downstream of it.
    bool readGuideTexture(Guide guide, std::vector<float>& out, uint32_t& width, uint32_t& height) override;

    bool deviceError() const override
    {
        return mDeviceError;
    }

    double pipelineCompileElapsedMs() const override
    {
        if (!mPipelineBuild.valid())
        {
            return -1.0;
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mPipelineBuildBegin)
            .count();
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
    ReadyFrame getReadyFrame() override;
    int activeCudaDeviceOrdinal() const override
    {
        return mCudaDeviceOrdinal;
    }
    void* getNativeCudaStream() override
    {
        return static_cast<void*>(mState.stream);
    }

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

    /// The specialisation this frame's launch parameters ask for.
    PipelineSpec specFor(const Params& params) const;
    /// Recompile the modules and relink the pipeline when the frame asks for a
    /// specialisation the linked one was not built with. A no-op otherwise,
    /// which is every frame after the first.
    void ensurePipelineSpecialization(const Params& params);
    /// Tear the pipeline, its program groups and its modules down, in that
    /// order. Leaves the handles null so a failed rebuild cannot launch against
    /// a destroyed pipeline.
    void destroyPipeline();

};

} // namespace oka
