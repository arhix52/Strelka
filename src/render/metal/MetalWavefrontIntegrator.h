#pragma once

#include "Metal4Context.h"
#include "MetalEnvironment.h"
#include "MetalTextures.h"
#include "integrator_features.h"

#include <strelka/render/buffer.h>
#include <settings.h>

#include <Metal/Metal.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <vector>

namespace oka
{
namespace metal
{

// One specialised pipeline set for a WavefrontFeatures bit combination.
struct WavefrontVariant
{
    MTL::ComputePipelineState* generate = nullptr;
    MTL::ComputePipelineState* extendMotion = nullptr;
    MTL::ComputePipelineState* extendStatic = nullptr;
    MTL::ComputePipelineState* shade = nullptr;
    MTL::ComputePipelineState* miss = nullptr;
    MTL::ComputePipelineState* shadowMotion = nullptr;
    MTL::ComputePipelineState* shadowStatic = nullptr;
    // Bound to the shadow dispatch so the alpha test can run inside
    // traversal rather than as a restart loop around it.
    MTL::IntersectionFunctionTable* shadowTableMotion = nullptr;
    MTL::IntersectionFunctionTable* shadowTableStatic = nullptr;
    MTL::ComputePipelineState* sharcDeposit = nullptr;
};

// TODO(phase-N): replace pointer bags with owned domain handles once
// Materials/Geometry/Accel/Lights are extracted. For Phase 2 the integrator
// still binds scene resources MetalRender owns.
struct IntegratorSceneBindings
{
    MTL::Buffer* instanceBuffer = nullptr;
    MTL::AccelerationStructure* instanceAccelerationStructure = nullptr;
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
    MTL::Buffer* curvePointBuffer = nullptr;
    MTL::Buffer* curveSegmentBuffer = nullptr;
    MTL::Buffer* sharcBuffer = nullptr;
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
    bool motionBlasBuilt = false;
    bool profileStages = false;
    SettingsManager* settings = nullptr;
};

// Path-tracing domain: wavefront queues, variant PSO cache, M3/M4 encode,
// prepare/resolve/AOV PSOs, and stage profiling for the wavefront path.
class MetalWavefrontIntegrator
{
public:
    static constexpr uint32_t kMaxStageSamples = 256;

    MetalWavefrontIntegrator() = default;
    ~MetalWavefrontIntegrator();

    void init(MTL::Device* device, Metal4Context* metal4);
    void release();

    void buildPipelines();
    void ensureBuffers(uint32_t width, uint32_t height);
    const WavefrontVariant* variantFor(uint32_t features);

    // Returns the encoder to keep using: in profiling mode each stage gets its
    // own, because this hardware can only sample counters at encoder boundaries.
    MTL::ComputeCommandEncoder* encode(MTL::CommandBuffer* cmd,
                                       MTL::ComputeCommandEncoder* enc,
                                       const IntegratorSceneBindings& scene,
                                       const IntegratorFrameRequest& frame);
    void encodeMetal4(MTL4::CommandBuffer* cmd,
                      MTL4::ComputeCommandEncoder*& enc,
                      const IntegratorSceneBindings& scene,
                      const IntegratorFrameRequest& frame);

    void createStageTimestampBuffer();
    void createStageCounterHeap();
    void reportStageTimings();
    void reportStageTimingsMetal4(double lastRenderTimeMs);
    void reportIorStackStats();

    // Wavefront allocations + shadow tables for Metal 4 residency.
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
    MTL::Buffer* iorStatsBuffer() const
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
    MTL4::CounterHeap* stageCounterHeap() const
    {
        return mStageCounterHeap;
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

    MTL::ComputePipelineState* mResolvePSO4 = nullptr;
    MTL::ComputePipelineState* mPreparePSO4 = nullptr;
    MTL::ComputePipelineState* mPrepareShadowPSO4 = nullptr;
    MTL::ComputePipelineState* mPrepareHitMissPSO4 = nullptr;
    MTL::ComputePipelineState* mAovResolvePSO4 = nullptr;

    MTL::Buffer* mPathStateBuffer = nullptr;
    MTL::Buffer* mPathRayBuffer = nullptr;
    MTL::Buffer* mHitBuffer = nullptr;
    MTL::Buffer* mIorStackBuffer = nullptr;
    MTL::Buffer* mRadianceBuffer = nullptr;
    MTL::Buffer* mGuideRadianceBuffer = nullptr;
    MTL::Buffer* mPathQueueBuffer[2] = { nullptr, nullptr };
    MTL::Buffer* mControlBuffer = nullptr;
    MTL::Buffer* mShadowRayBuffer = nullptr;
    MTL::Buffer* mHitQueueBuffer = nullptr;
    MTL::Buffer* mMissQueueBuffer = nullptr;
    MTL::Buffer* mAovBuffer = nullptr;

    MTL::CounterSampleBuffer* mStageTimestampBuffer = nullptr;
    MTL4::CounterHeap* mStageCounterHeap = nullptr;
    double mGpuTicksToMs = 0.0;
    MTL::Buffer* mStageStatsBuffer = nullptr;
    MTL::Buffer* mIorStatsBuffer = nullptr;
    bool mReportedIorStats = false;
    std::vector<uint8_t> mStageKinds;
    uint32_t mCapacity = 0;
    bool mResidencyDirty = true;
};

} // namespace metal
} // namespace oka
