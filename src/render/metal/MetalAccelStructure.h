#pragma once

#include "AccelBuildPath.h"
#include "MetalGeometry.h"
#include "Metal4Context.h"
#include "MetalMaterials.h"
#include "MetalTextures.h"

#include <Metal/Metal.hpp>
#include <settings.h>
#include <strelka/scene/scene.h>

#include <loadprogress.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace oka
{
namespace metal
{

struct AsBuildState;

// BLAS/TLAS domain: build, grouping, motion geometry, skeletal refit, instance buffer.
// Reads Geometry mesh records / VB offsets and Materials cutout/medium flags; fills
// GeometryEntry rows via MetalGeometry::geometryEntries() while building BLASes.
//
// AS submission is behind AccelBuildPath: Apple9+ shares the Metal 4 tracer queue;
// earlier GPUs build on Metal 3 and sync with SharedEvent. Callers see only
// inlineWithTracer() / encodeInline / submitSide — not an API version.
class MetalAccelStructure
{
public:
    // One acceleration structure covering N geometries that always move together
    // (in practice: every primitive of one glTF mesh node).
    struct Blas
    {
        MTL::AccelerationStructure* mAs = nullptr;
        // Kept alive for refit. Opaque across Metal 3 / Metal 4 descriptor types
        // (both derive from MTL::AccelerationStructureDescriptor).
        MTL::AccelerationStructureDescriptor* mDescriptor = nullptr;
        MTL::Buffer* mScratch = nullptr; // persistent, reused every refit/rebuild
        // Metal 4 motion descriptors read BufferRange arrays from GPU memory.
        std::vector<MTL::Buffer*> mMotionVertexRangeBuffers;
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
              Metal4Context* metal4,
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

    bool inlineWithTracer() const
    {
        return mPath && mPath->inlineWithTracer();
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

    /// Encode dynamic updates into the caller's Metal 4 compute encoder. Only
    /// valid when inlineWithTracer() is true.
    void encodeInline(MTL4::ComputeCommandEncoder* encoder, const AsFrameUpdate& update);
    /// Submit dynamic updates on the Metal 3 AS queue. Only valid when
    /// inlineWithTracer() is false. Signals readyEvent() when done.
    void submitSide(const AsFrameUpdate& update);

    MTL::SharedEvent* readyEvent() const
    {
        return mPath ? mPath->readyEvent() : nullptr;
    }
    uint64_t readyValue() const
    {
        return mPath ? mPath->readyValue() : 0;
    }

    MTL::Buffer* instanceBuffer() const
    {
        return mInstanceBuffer;
    }
    /// Descriptor buffer used by the preceding rendered transform state. When
    /// no transforms changed for the current frame, current is also the honest
    /// previous state and the renderer need not read this buffer.
    MTL::Buffer* previousInstanceBuffer() const
    {
        return mPreviousInstanceBuffer;
    }
    bool instanceTransformsChanged() const
    {
        return mInstanceTransformsChanged;
    }
    void markInstanceTransformsRendered()
    {
        mInstanceTransformsChanged = false;
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
    std::vector<MTL::Buffer*> accelerationStructureAuxiliaryBuffers() const;

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
    size_t buildBlas(const std::vector<uint32_t>& sceneInstanceIds, bool skeletal);
    size_t buildCurveBlas(uint32_t sceneInstanceId);
    void ensureScratchBuffer(MTL::Buffer*& buffer, size_t requiredSize);
    void addDescriptorResidency();
    void writeInstanceTransforms(MTL::Buffer* buffer);
    void releaseRetiredInstanceStructures(uint64_t age);
    void encodeSkeletalUpdates();
    void encodeTlasUpdates();
    MTL::AccelerationStructureUsage tlasUsage() const;

    MTL::Device* mDevice = nullptr;
    MTL::CommandQueue* mCommandQueue = nullptr;
    Metal4Context* mMetal4 = nullptr;
    std::unique_ptr<AccelBuildPath> mPath;
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
    // Reused for every TLAS refit. Path-owned descriptor type (MTL3 or MTL4).
    MTL::AccelerationStructureDescriptor* mTlasDescriptor = nullptr;
    MTL::Buffer* mInstanceBuffer = nullptr;
    // Two are sufficient because MetalRender submits at most one frame at a
    // time: async rendering stays busy until commit feedback (or the Metal 3
    // denoiser completion), and renderSync waits. If that policy changes to
    // multiple frames in flight, this must become one buffer per frame slot.
    MTL::Buffer* mPreviousInstanceBuffer = nullptr;
    bool mInstanceTransformsChanged = false;
    MTL::Buffer* mTlasScratchBuffer = nullptr;
    size_t mTlasInstanceCount = 0;
    // A top level that has to grow is replaced from inside the frame's encoder,
    // where the structure it replaces may still be read by the frames already in
    // flight. Freeing it there is a fault the frame after next; it waits here
    // for as many encodes as there can be frames outstanding instead.
    std::vector<std::pair<MTL::AccelerationStructure*, uint64_t>> mRetiredInstanceStructures;
    uint64_t mTlasEncodeCount = 0;

    AsBuildState* mAsBuild = nullptr;

    bool mNeedsPrimitiveData = false;
    bool mMotionBlasBuilt = false;
    bool mBuildMotionBlas = false;
    uint32_t mOpaqueGeometryCount = 0;
    uint32_t mCutoutGeometryCount = 0;

    // Matches the renderer's frames in flight. Kept here rather than shared,
    // because being wrong on the high side only delays a free.
    static constexpr uint64_t kMaxFramesInFlight = 3;
    static constexpr size_t kMaxBlasRebuildsPerFrame = 8;
    size_t mNextBlasRebuildIndex = 0;
};

} // namespace metal
} // namespace oka
