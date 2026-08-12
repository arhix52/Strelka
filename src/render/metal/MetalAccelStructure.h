#pragma once

#include "MetalGeometry.h"
#include "MetalMaterials.h"
#include "MetalTextures.h"

#include <Metal/Metal.hpp>
#include <settings.h>
#include <strelka/scene/scene.h>

#include <loadprogress.h>

#include <cstdint>
#include <vector>

namespace oka
{
namespace metal
{

struct AsBuildState;

// BLAS/TLAS domain: build, grouping, motion geometry, skeletal refit, instance buffer.
// Reads Geometry mesh records / VB offsets and Materials cutout/medium flags; fills
// GeometryEntry rows via MetalGeometry::geometryEntries() while building BLASes.
class MetalAccelStructure
{
public:
    // One acceleration structure covering N geometries that always move together
    // (in practice: every primitive of one glTF mesh node).
    struct Blas
    {
        MTL::AccelerationStructure* mAs = nullptr;
        // Kept alive for refit. Invariant: it only names buffers, offsets and
        // triangle counts, none of which change while the pose does.
        MTL::PrimitiveAccelerationStructureDescriptor* mDescriptor = nullptr;
        MTL::Buffer* mScratch = nullptr; // persistent, reused every refit/rebuild
        size_t mRefitScratchSize = 0;
        size_t mBuildScratchSize = 0;
        bool mIsSkeletal = false;
        uint32_t mGeometryBase = 0; // first index into GeometryEntry table
    };

    // One emitted TLAS instance. A merged group contributes a single instance,
    // so this no longer maps one-to-one onto Scene::Instance.
    struct EmittedInstance
    {
        uint32_t sceneInstanceId; // representative, supplies the transform
        uint32_t asIndex;
        uint32_t userID;
        uint32_t mask;
    };

    MetalAccelStructure() = default;
    ~MetalAccelStructure();

    void init(MTL::Device* device,
              MTL::CommandQueue* queue,
              MetalGeometry* geometry,
              MetalMaterials* materials,
              MetalTextures* textures);
    void release();

    void setScene(Scene* scene)
    {
        mScene = scene;
    }
    void setSettings(SettingsManager* settings)
    {
        mSettings = settings;
    }
    void setLoadProgress(LoadProgress* progress)
    {
        mLoadProgress = progress;
    }

    void setBuildMotionBlas(bool build)
    {
        mBuildMotionBlas = build;
    }
    bool buildMotionBlas() const
    {
        return mBuildMotionBlas;
    }
    bool motionBlasBuilt() const
    {
        return mMotionBlasBuilt;
    }
    bool needsPrimitiveData() const
    {
        return mNeedsPrimitiveData;
    }
    bool buildActive() const
    {
        return mAsBuild != nullptr;
    }

    /// Build every acceleration structure the scene needs, in one call.
    void create();
    /// Resumable build. Zero budget = no limit. Returns true when complete.
    bool step(double budgetMs);
    /// Rebuild for a different motion setting (drains the queue first).
    void rebuild();

    void updateSkeletalBLAS();
    void updateInstanceTransforms();
    void rebuildTLAS();

    /// Structure builds are committed to the Metal 3 queue, so anything reading
    /// them from that same queue is ordered behind them for free. A consumer on
    /// another queue -- the Metal 4 tracer -- has no such ordering and must wait
    /// on this event for `buildValue()` before it traverses. Without that wait a
    /// per-frame rebuild (the skeletal structures, while an animation plays) is
    /// traversed while it is being written: rays miss the geometry, and once the
    /// structure is resident on the other queue the read faults outright.
    MTL::SharedEvent* buildEvent() const
    {
        return mBuildEvent;
    }
    /// The value the last committed build will signal. Zero until one has been.
    uint64_t buildValue() const
    {
        return mBuildValue;
    }

    MTL::Buffer* instanceBuffer() const
    {
        return mInstanceBuffer;
    }
    MTL::AccelerationStructure* instanceAccelerationStructure() const
    {
        return mInstanceAccelerationStructure;
    }
    const std::vector<MTL::AccelerationStructure*>& primitiveAccelerationStructures() const
    {
        return mPrimitiveAccelerationStructures;
    }
    std::vector<MTL::AccelerationStructure*>& primitiveAccelerationStructures()
    {
        return mPrimitiveAccelerationStructures;
    }
    const std::vector<Blas>& blasList() const
    {
        return mBlasList;
    }
    const std::vector<EmittedInstance>& emittedInstances() const
    {
        return mEmittedInstances;
    }
    size_t tlasInstanceCount() const
    {
        return mTlasInstanceCount;
    }
    MTL::Buffer* tlasScratchBuffer() const
    {
        return mTlasScratchBuffer;
    }

    uint32_t opaqueGeometryCount() const
    {
        return mOpaqueGeometryCount;
    }
    uint32_t cutoutGeometryCount() const
    {
        return mCutoutGeometryCount;
    }

private:
    MTL::AccelerationStructure* createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor);
    void flushAccelerationStructureGroup();
    MTL::AccelerationStructure* createAccelerationStructureNoCompact(MTL::AccelerationStructureDescriptor* descriptor);
    MTL::AccelerationStructureTriangleGeometryDescriptor* createStaticGeometryDescriptor(const oka::Mesh& sceneMesh,
                                                                                         MTL::Buffer* perPrimitiveBuffer,
                                                                                         uint32_t triangleCount);
    MTL::AccelerationStructureMotionTriangleGeometryDescriptor* createMotionGeometryDescriptor(
        const oka::Mesh& sceneMesh, MTL::Buffer* perPrimitiveBuffer, uint32_t triangleCount);
    size_t buildBlas(const std::vector<uint32_t>& sceneInstanceIds, bool skeletal);
    size_t buildCurveBlas(uint32_t sceneInstanceId);
    void ensureScratchBuffer(MTL::Buffer*& buffer, size_t requiredSize);
    /// Tag `commandBuffer` as the newest structure build, for cross-queue waits.
    void signalBuild(MTL::CommandBuffer* commandBuffer);

    MTL::Device* mDevice = nullptr;
    MTL::CommandQueue* mCommandQueue = nullptr;
    MetalGeometry* mGeometry = nullptr;
    MetalMaterials* mMaterials = nullptr;
    MetalTextures* mTextures = nullptr;
    Scene* mScene = nullptr;
    SettingsManager* mSettings = nullptr;
    LoadProgress* mLoadProgress = nullptr;

    std::vector<Blas> mBlasList;
    std::vector<EmittedInstance> mEmittedInstances;
    std::vector<MTL::AccelerationStructure*> mPrimitiveAccelerationStructures;
    MTL::AccelerationStructure* mInstanceAccelerationStructure = nullptr;
    MTL::Buffer* mInstanceBuffer = nullptr;
    MTL::Buffer* mTlasScratchBuffer = nullptr;
    size_t mTlasInstanceCount = 0;

    MTL::CommandBuffer* mAsGroupCommandBuffer = nullptr;
    MTL::AccelerationStructureCommandEncoder* mAsGroupEncoder = nullptr;
    std::vector<MTL::Buffer*> mAsGroupScratch;
    uint32_t mAsGroupPending = 0;

    AsBuildState* mAsBuild = nullptr;

    bool mNeedsPrimitiveData = false;
    bool mMotionBlasBuilt = false;
    bool mBuildMotionBlas = false;
    uint32_t mOpaqueGeometryCount = 0;
    uint32_t mCutoutGeometryCount = 0;

    static constexpr size_t kMaxBlasRebuildsPerFrame = 8;
    size_t mNextBlasRebuildIndex = 0;

    MTL::SharedEvent* mBuildEvent = nullptr;
    uint64_t mBuildValue = 0;
};

} // namespace metal
} // namespace oka
