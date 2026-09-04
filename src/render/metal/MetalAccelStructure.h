#pragma once

#include "AccelBuildPath.h"
#include "MetalGeometry.h"
#include "Metal4Context.h"
#include "MetalMaterials.h"
#include "MetalTextures.h"

#include <host/emissive_mesh_distribution.h>

#include <Metal/Metal.hpp>
#include <settings.h>
#include <strelka/scene/scene.h>

#include <loadprogress.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>


namespace oka::metal
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
        // Geometry order in this emitted BLAS. Needed by emissive-mesh NEE:
        // shared BLAS records describe geometry, while the emitted instance
        // supplies the transform and therefore owns a distinct light source.
        std::vector<uint32_t> geometrySceneInstanceIds;
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
    bool buildActive() const
    {
        return mAsBuild != nullptr;
    }

    bool inlineWithTracer() const
    {
        return mPath && mPath->inlineWithTracer();
    }

    /// A valid top level containing no instances, built before any geometry
    /// exists so the scene can be traced while it loads. Every ray misses it and
    /// reaches the environment, which is the correct picture of a scene whose
    /// objects have not arrived. Replaced by the real one when the build
    /// reaches it; a no-op if a top level already exists.
    void buildEmptyTopLevel();

    /// A top level over the instances built so far, so a scene appears as it
    /// loads instead of arriving whole when the last structure lands. Safe at any
    /// slice boundary: an instance is emitted only once its BLAS exists. Replaced
    /// by the complete one when the build finishes.
    void publishPartialTopLevel();

    /// Resumable build. Zero budget = no limit. Returns true when complete.
    bool step(double budgetMs);
    /// Rebuild for a different motion setting (drains the queue first).
    void rebuild();

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
    /// Triangle-only top level used by bounded-medium random walks. Curve BLAS
    /// are deliberately absent, rather than merely rejected by a ray mask: on
    /// current Metal 4 drivers a handful of deep rays can still spend watchdog-
    /// scale time traversing a mixed TLAS. Scenes without curves reuse the main
    /// top level.
    MTL::AccelerationStructure* volumeAccelerationStructure() const
    {
        if (mVolumeInstanceAccelerationStructure)
        {
            return mVolumeInstanceAccelerationStructure;
        }
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
    MTL::Buffer* emissiveMeshBuffer() const
    {
        return mEmissiveMeshBuffer;
    }
    MTL::Buffer* emissiveTriangleBuffer() const
    {
        return mEmissiveTriangleBuffer;
    }
    uint32_t emissiveMeshCount() const
    {
        return mEmissiveMeshCount;
    }
    double emissiveMeshPower() const
    {
        return mEmissiveMeshPower;
    }
    void rebuildEmissiveMeshLights();

private:
    MTL::AccelerationStructure* createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor);
    void flushAccelerationStructureGroup();
    MTL::AccelerationStructure* createAccelerationStructureNoCompact(MTL::AccelerationStructureDescriptor* descriptor);
    size_t buildBlas(const std::vector<uint32_t>& sceneInstanceIds, bool skeletal);
    /// What buildCurveBlas returns for a set it could not build: an empty one,
    /// or one whose points never got uploaded. Named because the caller has to
    /// test for it, and `(size_t)-1` at both ends said nothing about which end
    /// owned the convention.
    static constexpr size_t kNoCurveBlas = ~size_t{ 0 };
    size_t buildCurveBlas(uint32_t sceneInstanceId);
    size_t buildAnalyticLightBlas(uint32_t intersectionFunctionOffset);
    void ensureScratchBuffer(MTL::Buffer*& buffer, size_t requiredSize);
    void addDescriptorResidency();
    void prepareEmissiveMeshInputs();
    void uploadEmissiveMeshLights();

    /// Residency for an allocation this class owns. Null-safe on both the
    /// allocation and the Metal 4 context, so callers do not repeat either
    /// guard -- and `retireResident` exists so that "out of the set before it is
    /// freed" is a single call rather than a rule to remember. The set does not
    /// retain what it names, and the allocator reuses addresses.
    void makeResident(MTL::Allocation* allocation);
    void retireResident(MTL::Buffer*& buffer);
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
    std::vector<render::EmissiveMeshBuildInput> mSceneEmissiveInputs;
    MTL::Buffer* mEmissiveMeshBuffer = nullptr;
    MTL::Buffer* mEmissiveTriangleBuffer = nullptr;
    MTL::Buffer* mAnalyticLightBoundsBuffer = nullptr;
    uint32_t mEmissiveMeshCount = 0;
    double mEmissiveMeshPower = 0.0;
    std::vector<MTL::AccelerationStructure*> mPrimitiveAccelerationStructures;
    MTL::AccelerationStructure* mInstanceAccelerationStructure = nullptr;
    MTL::AccelerationStructure* mVolumeInstanceAccelerationStructure = nullptr;
    // Reused for every TLAS refit. Path-owned descriptor type (MTL3 or MTL4).
    MTL::AccelerationStructureDescriptor* mTlasDescriptor = nullptr;
    MTL::AccelerationStructureDescriptor* mVolumeTlasDescriptor = nullptr;
    MTL::Buffer* mInstanceBuffer = nullptr;
    // Two are sufficient because MetalRender submits at most one frame at a
    // time: async rendering stays busy until commit feedback (or the Metal 3
    // denoiser completion), and renderSync waits. If that policy changes to
    // multiple frames in flight, this must become one buffer per frame slot.
    MTL::Buffer* mPreviousInstanceBuffer = nullptr;
    bool mInstanceTransformsChanged = false;
    MTL::Buffer* mTlasScratchBuffer = nullptr;
    MTL::Buffer* mVolumeTlasScratchBuffer = nullptr;
    size_t mTlasInstanceCount = 0;
    size_t mVolumeTlasInstanceCount = 0;
    // A top level that has to grow is replaced from inside the frame's encoder,
    // where the structure it replaces may still be read by the frames already in
    // flight. Freeing it there is a fault the frame after next; it waits here
    // for as many encodes as there can be frames outstanding instead.
    std::vector<std::pair<MTL::AccelerationStructure*, uint64_t>> mRetiredInstanceStructures;
    uint64_t mTlasEncodeCount = 0;

    AsBuildState* mAsBuild = nullptr;

    bool mMotionBlasBuilt = false;
    bool mBuildMotionBlas = false;
    uint32_t mOpaqueGeometryCount = 0;
    uint32_t mCutoutGeometryCount = 0;
    // What the bottom-level builds cost this scene. Reset when a build starts.
    double mBlasEncodeMs = 0.0;
    uint32_t mBlasCount = 0;

    // Matches the renderer's frames in flight. Kept here rather than shared,
    // because being wrong on the high side only delays a free.
    static constexpr uint64_t kMaxFramesInFlight = 3;
    static constexpr size_t kMaxBlasRebuildsPerFrame = 8;
    /// How many geometries one bottom level may hold.
    ///
    /// Past a certain width, a structure built with AccelerationStructureUsageRefit
    /// returns no intersections at all on this driver -- it builds without error,
    /// reports a sane size, and every ray misses it. Measured on BrainStem in a
    /// studio set: the character's 59 primitives merge into one bottom level and it
    /// is invisible; split at 58 and it renders, and it renders at 59 with the refit
    /// flag dropped. The flag cannot be dropped -- it is what halves what the builder
    /// allocates, which is the difference between the pine forest fitting in memory
    /// and not -- so the width is capped instead, well short of where it was seen to
    /// fail. Splitting costs one more instance in the top level per 32 primitives.
    static constexpr size_t kMaxGeometriesPerBlas = 32;
    size_t mNextBlasRebuildIndex = 0;
};

} // namespace oka::metal
