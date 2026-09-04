#pragma once

#include "Metal4Context.h"
#include "wavefront_chunk_plan.h"
#include "MetalEnvironment.h"
#include "MetalTextures.h"
#include "integrator_features.h"
#include <host/integrator_buffer_sizes.h>

#include <strelka/render/buffer.h>
#include <settings.h>

#include <Metal/Metal.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>


namespace oka::metal
{

// One specialised pipeline set for a WavefrontFeatures bit combination.
struct WavefrontVariant
{
    MTL::ComputePipelineState* generate = nullptr;
    MTL::ComputePipelineState* extendMotion = nullptr;
    MTL::ComputePipelineState* extendStatic = nullptr;
    MTL::ComputePipelineState* shade = nullptr;
    MTL::ComputePipelineState* restirSpatial = nullptr;
    MTL::ComputePipelineState* restirFinal = nullptr;
    MTL::ComputePipelineState* miss = nullptr;
    MTL::ComputePipelineState* shadowMotion = nullptr;
    MTL::ComputePipelineState* shadowStatic = nullptr;
    MTL::ComputePipelineState* guideMotion = nullptr;
    MTL::ComputePipelineState* guideStatic = nullptr;
    MTL::IntersectionFunctionTable* extendTableMotion = nullptr;
    MTL::IntersectionFunctionTable* extendTableStatic = nullptr;
    MTL::IntersectionFunctionTable* shadowTableMotion = nullptr;
    MTL::IntersectionFunctionTable* shadowTableStatic = nullptr;
    MTL::IntersectionFunctionTable* guideTableMotion = nullptr;
    MTL::IntersectionFunctionTable* guideTableStatic = nullptr;
};

// Non-owning resources assembled by MetalRender for each encode.
struct IntegratorSceneBindings
{
    MTL::Buffer* instanceBuffer = nullptr;
    MTL::AccelerationStructure* instanceAccelerationStructure = nullptr;
    MTL::AccelerationStructure* volumeAccelerationStructure = nullptr;
    const std::vector<MTL::AccelerationStructure*>* primitiveAccelerationStructures = nullptr;
    MTL::Buffer* materialBuffer = nullptr;
    MTL::Buffer* lightBuffer = nullptr;
    MTL::Buffer* iesBuffer = nullptr;
    MTL::Buffer* geometryEntryBuffer = nullptr;
    MTL::Buffer* vertexBuffer = nullptr;
    MTL::Buffer* prevVertexBuffer = nullptr;
    MTL::Buffer* indexBuffer = nullptr;
    MTL::Buffer* prevFrameVertexBuffer = nullptr;
    MTL::Buffer* prevFrameInstanceBuffer = nullptr;
    /// Stands in for an absent optional buffer. Argument tables persist across
    /// dispatches, so an unbound index keeps whatever the previous encoder left
    /// there, and setAddress(0) is a null pointer the debug layer rejects
    /// outright. Every optional binding points here instead, so a stage that
    /// never reads it still holds an address that is real and resident.
    MTL::Buffer* placeholderBuffer = nullptr;
    MTL::Buffer* curvePointBuffer = nullptr;
    MTL::Buffer* curveSegmentBuffer = nullptr;
    MTL::Buffer* sharcHashBuffer = nullptr;
    MTL::Buffer* sharcAccumulationBuffer = nullptr;
    MTL::Buffer* sharcResolvedBuffer = nullptr;
    MTL::Buffer* sharcStatsBuffer = nullptr;
    MTL::Buffer* accumulationBuffer = nullptr;
    MetalEnvironment* environment = nullptr;
    MetalTextures* textures = nullptr;

    MTL::Texture* guideColor = nullptr;
    MTL::Texture* guideDepth = nullptr;
    MTL::Texture* guideMotion = nullptr;
    MTL::Texture* guideDiffuse = nullptr;
    MTL::Texture* guideSpecular = nullptr;
    MTL::Texture* guideNormal = nullptr;
    MTL::Texture* guideRoughness = nullptr;
    MTL::Texture* guideSpecularHitDistance = nullptr;
    MTL::Texture* guideReactive = nullptr;
    MTL::Texture* guideDenoiseStrength = nullptr;
};

struct IntegratorFrameRequest
{
    MTL::Buffer* uniformBuffer = nullptr;
    Buffer* output = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t sampleCount = 0;
    uint32_t features = 0;
    uint32_t bounceIterations = 0;
    uint32_t traversalBatchThreads = kWavefrontTraversalBatchThreads;
    uint32_t pathCount = 0;
    bool motionBlasBuilt = false;
    bool profileStages = false;
    bool auditRenderWork = false;
    SettingsManager* settings = nullptr;
};

// Path-tracing domain: wavefront queues, variant PSO cache, M3/M4 encode,
// prepare/resolve/AOV PSOs, and stage profiling for the wavefront path.
class MetalWavefrontIntegrator
{
public:
    // One breadcrumb per profiled stage plus the closing mark. A depth-8 scene with
    // alpha and the SSS tail encodes 80 iterations (321 stages), so 256 made a
    // late failure look like "post-integrator" simply because profiling stopped.
    static constexpr uint32_t kMaxStageSamples = 512;

    MetalWavefrontIntegrator() = default;
    ~MetalWavefrontIntegrator();

    void init(MTL::Device* device, Metal4Context* metal4);
    void release();

    void buildPipelines();
    void ensureBuffers(uint32_t width, uint32_t height, uint32_t sharcUpdateDownscale);
    const WavefrontVariant* variantFor(uint32_t features);

    // Returns the encoder to keep using: in profiling mode each stage gets its
    // own, because this hardware can only sample counters at encoder boundaries.
    MTL::ComputeCommandEncoder* encode(MTL::CommandBuffer* cmd,
                                       MTL::ComputeCommandEncoder* enc,
                                       const IntegratorSceneBindings& scene,
                                       const IntegratorFrameRequest& frame);
    void encodeSharcClear(MTL::ComputeCommandEncoder* enc,
                          const IntegratorSceneBindings& scene,
                          const IntegratorFrameRequest& frame,
                          bool clearPersistent);
    void encodeSharcResolve(MTL::ComputeCommandEncoder* enc,
                            const IntegratorSceneBindings& scene,
                            const IntegratorFrameRequest& frame);
    void encodeMetal4(MTL4::ComputeCommandEncoder*& enc,
                      const IntegratorSceneBindings& scene,
                      const IntegratorFrameRequest& frame,
                      const WavefrontChunk& chunk);
    void encodeSharcClearMetal4(MTL4::ComputeCommandEncoder* enc,
                                const IntegratorSceneBindings& scene,
                                const IntegratorFrameRequest& frame,
                                bool clearPersistent);
    void encodeSharcResolveMetal4(MTL4::ComputeCommandEncoder* enc,
                                  const IntegratorSceneBindings& scene,
                                  const IntegratorFrameRequest& frame);
    void resetStageProfilingMetal4();

    void createStageTimestampBuffer();
    void reportStageTimings();
    void reportStageFailureMetal4();
    void reportIorStackStats();
    void reportSharcStats();
    void beginRenderWorkAudit();
    const uint32_t* renderWorkCounters() const;
    uint64_t renderWorkCounterAddress() const;
    const std::map<std::string, uint64_t>& renderWorkDispatches() const
    {
        return mRenderWorkDispatches;
    }

    // Wavefront allocations for Metal 4 residency.
    void addResidentAllocations(const std::function<void(MTL::Allocation*)>& add) const;

    size_t queueBytes() const;

    MTL::Library* library() const
    {
        return mLibrary;
    }
    uint32_t capacity() const
    {
        return mCapacity;
    }
    bool residencyDirty() const
    {
        return mResidencyDirty;
    }
    void clearResidencyDirty()
    {
        mResidencyDirty = false;
    }

    MTL::Buffer* aovBuffer() const
    {
        return mAovBuffer;
    }
    MTL::Buffer* radianceBuffer() const
    {
        return mRadianceBuffer;
    }
    uint64_t guideRayAddress() const
    {
        return mGuideRayBuffer ? mGuideRayBuffer->gpuAddress() : 0ull;
    }
    uint64_t restirReservoirAddress(uint32_t index) const
    {
        return mRestirReservoirBuffer[index & 1u] ? mRestirReservoirBuffer[index & 1u]->gpuAddress() : 0ull;
    }
    uint64_t restirHistoryAddress(uint32_t index) const
    {
        return mRestirSurfaceHistoryBuffer[index & 1u] ? mRestirSurfaceHistoryBuffer[index & 1u]->gpuAddress() : 0ull;
    }
    uint64_t restirShadingPointAddress() const
    {
        return mRestirShadingPointBuffer ? mRestirShadingPointBuffer->gpuAddress() : 0ull;
    }
    MTL::Buffer* iorStatsBuffer() const
    {
        return mIorStatsBuffer;
    }
    MTL::Buffer* sharcStatsBuffer() const
    {
        return mIorStatsBuffer;
    }
    MTL::ComputePipelineState* aovResolvePSO() const
    {
        return mAovResolvePSO;
    }
    MTL::ComputePipelineState* aovResolvePSO4() const
    {
        return mAovResolvePSO4;
    }
    MTL::ComputePipelineState* resolvePSO4() const
    {
        return mResolvePSO4;
    }
    std::vector<uint8_t>& stageKinds()
    {
        return mStageKinds;
    }

private:
    MTL::Device* mDevice = nullptr;
    Metal4Context* mMetal4 = nullptr;

    std::map<uint32_t, WavefrontVariant> mVariants;
    MTL::Library* mLibrary = nullptr;

    MTL::ComputePipelineState* mResolvePSO = nullptr;
    MTL::ComputePipelineState* mPreparePSO = nullptr;
    MTL::ComputePipelineState* mPrepareShadowPSO = nullptr;
    MTL::ComputePipelineState* mPrepareHitMissPSO = nullptr;
    MTL::ComputePipelineState* mAovResolvePSO = nullptr;
    MTL::ComputePipelineState* mSharcClearPSO = nullptr;
    MTL::ComputePipelineState* mSharcResolvePSO = nullptr;

    MTL::ComputePipelineState* mResolvePSO4 = nullptr;
    MTL::ComputePipelineState* mPreparePSO4 = nullptr;
    MTL::ComputePipelineState* mPrepareShadowPSO4 = nullptr;
    MTL::ComputePipelineState* mPrepareHitMissPSO4 = nullptr;
    MTL::ComputePipelineState* mStageBreadcrumbPSO4 = nullptr;
    MTL::ComputePipelineState* mAovResolvePSO4 = nullptr;
    MTL::ComputePipelineState* mSharcClearPSO4 = nullptr;
    MTL::ComputePipelineState* mSharcResolvePSO4 = nullptr;

    MTL::Buffer* mPathStateBuffer = nullptr;
    MTL::Buffer* mMediumPathStateBuffer = nullptr;
    MTL::Buffer* mSharcUpdateStateBuffer = nullptr;
    MTL::Buffer* mPathRayBuffer = nullptr;
    MTL::Buffer* mHitBuffer = nullptr;
    MTL::Buffer* mIorStackBuffer = nullptr;
    MTL::Buffer* mRadianceBuffer = nullptr;
    MTL::Buffer* mGuideRayBuffer = nullptr;
    MTL::Buffer* mPathQueueBuffer[2] = { nullptr, nullptr };
    MTL::Buffer* mControlBuffer = nullptr;
    MTL::Buffer* mTraversalDispatchBuffer = nullptr;
    MTL::Buffer* mShadowRayBuffer = nullptr;
    MTL::Buffer* mHitQueueBuffer = nullptr;
    MTL::Buffer* mMissQueueBuffer = nullptr;
    MTL::Buffer* mAovBuffer = nullptr;
    MTL::Buffer* mRestirReservoirBuffer[2] = { nullptr, nullptr };
    MTL::Buffer* mRestirSurfaceHistoryBuffer[2] = { nullptr, nullptr };
    MTL::Buffer* mRestirShadingPointBuffer = nullptr;

    MTL::CounterSampleBuffer* mStageTimestampBuffer = nullptr;
    MTL::Buffer* mStageStatsBuffer = nullptr;
    MTL::Buffer* mIorStatsBuffer = nullptr;
    MTL::Buffer* mRenderWorkCounterBuffer = nullptr;
    bool mReportedIorStats = false;
    uint64_t mLastSharcActivity = 0;
    std::vector<uint8_t> mStageKinds;
    std::map<std::string, uint64_t> mRenderWorkDispatches;
    uint32_t mCapacity = 0;
    uint32_t mSharcUpdateDownscale = 0;
    bool mResidencyDirty = true;
};

} // namespace oka::metal
