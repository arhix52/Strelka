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

#include "gpu_stage_breadcrumb.h"
#include <host/scene_preparation.h>
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
    OptixProgramGroup radiance_openpbr_hit_group = nullptr;
    OptixProgramGroup radiance_openpbr_base_hit_group = nullptr;
    OptixProgramGroup radiance_curve_hit_group = nullptr;
    OptixProgramGroup radiance_linear_curve_hit_group = nullptr;
    OptixProgramGroup occlusion_hit_group = nullptr;
    OptixProgramGroup occlusion_linear_curve_hit_group = nullptr;
    OptixProgramGroup light_hit_group = nullptr;
    OptixProgramGroup light_occlusion_group = nullptr;
    CUstream stream = nullptr;
    Params params = {};
    Params prevParams = {};

    std::unique_ptr<OptixBuffer> mParamsBuffer;

    Params* pinnedParams = nullptr;

    OptixShaderBindingTable sbt = {};
};

class OptiXRender : public Render
{
private:
    struct Mesh
    {
        OptixTraversableHandle gas_handle = 0;
        CUdeviceptr d_gas_output_buffer = 0;
        size_t gas_bytes = 0;
        uint32_t geometry_flags = OPTIX_GEOMETRY_FLAG_NONE;
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

    struct PipelineSpec
    {
        uint32_t sharcCapacity = 0;
        /// Whether the sampler's blue-noise mask is compiled in; see Params.
        uint32_t hasBlueNoise = 0;
        uint32_t sharcResponsive = 0;
        uint32_t debug = 0;
        uint32_t estimatorMode = 0;
        uint32_t volumeModel = 0;
        uint32_t misHeuristic = 0;
        uint32_t subsurfaceIterations = 0;
        uint32_t risCandidates = 1;
        uint32_t denoiseDepthMode = 0;
        bool hasBoundedMedium = false;
        bool hasSubsurface = false;
        /// Scene contents, see Params for what each one gates.
        bool hasCurves = false;
        bool hasCutout = false;
        bool hasOpenPBR = false;
        bool openpbrSheenAndCoat = false;
        bool openpbrDispersion = false;
        bool openpbrTranslucency = false;
        bool openpbrMetallic = false;
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

    std::future<void> mPipelineBuild;
    std::chrono::steady_clock::time_point mPipelineBuildBegin;
    bool mPipelineBuildWasFirst = false;
    /// A synchronous caller wants the frame, not a responsive window. Set around
    /// renderSync() so the build is waited on rather than skipped.
    bool mPipelineBuildBlocking = false;

    /// Set whenever the TLAS is rebuilt from scratch. The SBT is indexed by
    /// instance, so it has to be rebuilt with it; a refit leaves it alone.
    bool mSbtDirty = false;

    static constexpr size_t kMaxBlasRebuildsPerFrame = 8;
    size_t mNextBlasRebuildIndex = 0;

    /// Instance count the current TLAS was built for. A refit cannot change it,
    /// so a different one is what forces a rebuild.
    size_t mTlasInstanceCount = 0;

    // Previous-frame settings for change detection (replaces static locals in render())
    uint32_t mPrevRectLightSamplingMethod = 0;
    /// Last value warned about for render/pt/samplerType, so an unimplemented
    /// one is reported when it is chosen rather than on every frame after.
    uint32_t mReportedSamplerType = 2;
    bool mPrevEnableAccumulation = false;
    uint32_t mPrevSspTotal = 0;

    // Device buffers for per-material data (indexed by materialId)
    std::unique_ptr<OptixBuffer> mMaterialParamsBuffer; // MaterialParams[] on device
    std::unique_ptr<OptixBuffer> mAlphaMaterialParamsBuffer; // traversal-only OptixAlphaMaterialData[]
    uint32_t mMaterialCount = 0;

    std::unique_ptr<OptixBuffer> mOpenPBRParamsBuffer; // OpenPBRParams[] on device
    std::unique_ptr<OptixBuffer> mOpenPBRTexturesBuffer; // cudaTextureObject_t[n * MAX_OPENPBR_TEXTURES]
    std::vector<cudaTextureObject_t> mHostOpenPBRTextures;
    std::vector<uint8_t> mOpenPBRBaseMaterials;

    void allocJointMatrices();
    std::unique_ptr<Mesh> createMesh(const oka::Mesh& mesh, size_t meshIndex);
    void updateMesh(const oka::Mesh& mesh, int optixMeshesId);
    bool rebuildMesh(const oka::Mesh& mesh, int optixMeshesId);
    std::unique_ptr<Curve> createCurve(const oka::Curve& curve);
    size_t compactAccel(CUdeviceptr& buffer, OptixTraversableHandle& handle, CUdeviceptr result, size_t outputSizeInBytes);

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
    /// The analytic lights as custom primitives, so hardware traversal finds
    /// them instead of every ray walking the light table. One structure per
    /// camera visibility, because that is what an instance mask can express.
    struct AnalyticLightAccel
    {
        std::unique_ptr<OptixBuffer> aabbs;
        std::unique_ptr<OptixBuffer> indices;
        CUdeviceptr output = 0;
        OptixTraversableHandle handle = 0;
        uint32_t count = 0;
        ~AnalyticLightAccel()
        {
            CUDA_CHECK(cudaFree(optix::devicePtr<void>(output)));
        }
    };
    /// Where the light structures' instances start in the TLAS, so their SBT
    /// records can be appended to the scene's.
    size_t mAnalyticLightInstanceBase = 0;
    AnalyticLightAccel mVisibleLightAccel;
    AnalyticLightAccel mHiddenLightAccel;
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
    std::unique_ptr<OptixBuffer> mTempAccelBuffer; // Temporary buffer for acceleration structure building
    std::unique_ptr<OptixBuffer> mCompactedSizeBuffer; // Buffer for storing compaction size results
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
    /// Rebuild the analytic lights' custom-primitive structures. Cheap: one AABB
    /// per light, and it runs only when the light table does.
    void createAnalyticLightAccel();
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

    std::unique_ptr<OptixBuffer> mSharcBuffer;
    /// One SharcPathState per pixel, allocated only while the cache is on.
    std::unique_ptr<OptixBuffer> mSharcPathBuffer;
    size_t mSharcPathStateCount = 0;
    uint32_t mSharcCapacity = 0;
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
    void* mDisplayImage = nullptr;
    uint32_t mDisplayWidth = 0;
    uint32_t mDisplayHeight = 0;
    std::unique_ptr<OptixBuffer> mDisplayReadbackBuffer;
    PresentationMetadata mDisplayPresentation{};
    PresentationMetadata mPendingPresentation{};
    bool readDisplayTextureWithMaxOutput(std::vector<float>& out, uint32_t& width, uint32_t& height, float maxOutput);
    /// Raised by resetTemporalHistory() and consumed by the next render().
    bool mResetTemporalHistory = true;
    /// True when denoising was asked for and could not be provided.
    bool mDenoiserFallback = false;
    bool mPrevGuidePrimaryHit = false;

    /// Size the guide and denoiser buffers for a plan, reallocating only what
    /// changed.
    void updateGuideBuffers(const DenoisePlan& plan);
    int mSubmittedIndex = -1;

    cudaEvent_t mFrameStartEvent = nullptr;
    cudaEvent_t mFrameStopEvent = nullptr;
    /// A pair of events has been recorded and not yet read back.
    bool mFrameTimingPending = false;
    void createTimingEvents();
    /// Reads the recorded pair into mLastRenderTimeMs. `wait` blocks until the
    /// stop event has passed; without it the read is skipped when the frame is
    /// still running, and the previous frame's number stands.
    void collectFrameTiming(bool wait);

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

    std::unique_ptr<OptixBuffer> mIorStatsBuffer;
    bool mReportedIorStats = false;
    /// Reads the counters back and warns once, if any of them fired.
    void reportIorStackStats();

    // ------------------------------------------------------------- scene build --
    scene_preparation::ScenePreparation mScenePrep;
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

    scene_preparation::SceneBuildHooks makeSceneBuildHooks();
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
    Buffer* mAsyncOutputBuffers[2] = { nullptr, nullptr };
    uint64_t mFrameSerials[2] = {};
    PresentationMetadata mFramePresentation[2] = {};
    uint64_t mNextFrameSerial = 1;

    // ---------------------------------------------------------------- capture --
    bool mCaptureActive = false;

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
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mPipelineBuildBegin).count();
    }

    bool isBuildingScene() const override
    {
        return mScenePrep.isBuilding();
    }

    bool isRenderBusy() override
    {
        // Poll, do not wait: this is what the audit harnesses spin on after
        // submitting one frame, and the frame they are waiting for is only
        // published when somebody notices it has landed.
        reapSubmittedFrame(false);
        return mRenderBusy.load(std::memory_order_acquire);
    }

    void triggerRenderIfIdle() override;

    void reapSubmittedFrame(bool wait);
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
    bool sceneHasSubsurface() const;
    bool sceneHasCurves() const;
    bool sceneHasCutout() const;
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
