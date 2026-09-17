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
#include <span>
#include <utility>
#include <vector>

namespace oka::metal
{

struct AsBuildState;
struct AsBuildGeometry;

class MetalAccelStructure
{
public:
    struct AuditCounts
    {
        uint64_t blasBuilds = 0;
        uint64_t tlasBuilds = 0;
        uint64_t tlasRefits = 0;
    };
    struct Blas
    {
        MTL::AccelerationStructure* mAs = nullptr;
        // Kept alive for refit. Opaque across Metal 3 / Metal 4 descriptor types
        // (both derive from MTL::AccelerationStructureDescriptor).
        MTL::AccelerationStructureDescriptor* mDescriptor = nullptr;
        MTL::Buffer* mScratch = nullptr; // persistent, reused every refit/rebuild
        // One packed object-to-world matrix per geometry in a baked static BLAS.
        // Metal consumes these while building world-space geometry; shaders read
        // the matching matrices from the descriptor-buffer tail instead.
        MTL::Buffer* mGeometryTransformBuffer = nullptr;
        // Metal 4 motion descriptors read BufferRange arrays from GPU memory.
        std::vector<MTL::Buffer*> mMotionVertexRangeBuffers;
        // Driver-reported allocation before optional compaction. Retained only
        // for the scene-build memory summary; mAs->size() is the resident size.
        size_t mBuildSize = 0;
        size_t mRefitScratchSize = 0;
        size_t mBuildScratchSize = 0;
        bool mCompacted = false;
        bool mIsSkeletal = false;
        uint32_t mGeometryBase = 0; // first index into GeometryEntry table
        std::vector<uint32_t> mMeshIds;
    };

    // One emitted TLAS instance. A merged group contributes a single instance,
    // so this no longer maps one-to-one onto Scene::Instance.
    struct EmittedInstance
    {
        uint32_t sceneInstanceId; // representative, supplies the transform
        uint32_t asIndex;
        uint32_t userID;
        uint32_t mask;
        bool identityTransform = false;
        struct Geometry
        {
            uint32_t sceneInstanceId = 0;
            uint32_t firstTriangle = 0;
            uint32_t triangleCount = 0;
        };
        // Geometry order in this emitted BLAS. Needed by emissive-mesh NEE:
        // shared BLAS records describe geometry, while the emitted instance
        // supplies the transform and therefore owns a distinct light source.
        std::vector<Geometry> geometries;
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

    void buildEmptyTopLevel();

    void publishPartialTopLevel();

    /// Resumable build. Zero budget = no limit. Returns true when complete.
    bool step(double budgetMs);
    /// Rebuild for a different motion setting (drains the queue first).
    void rebuild();

    void updateInstanceTransforms();
    void setDirtySkeletalMeshes(std::span<const uint32_t> meshIds);
    void rebuildTLAS();
    bool transformChangesRequireRebuild() const;

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
    MTL::AccelerationStructure* directStaticAccelerationStructure() const
    {
        return mDirectStaticAccelerationStructure;
    }
    uint32_t directStaticGeometryBase() const
    {
        return mDirectStaticGeometryBase;
    }
    uint32_t directStaticInstanceIndex() const
    {
        return mDirectStaticInstanceIndex;
    }
    MTL::AccelerationStructure* volumeAccelerationStructure() const
    {
        if (mVolumeInstanceAccelerationStructure)
        {
            return mVolumeInstanceAccelerationStructure;
        }
        return mInstanceAccelerationStructure;
    }
    MTL::AccelerationStructure* mediumAccelerationStructure() const
    {
        if (mMediumInstanceAccelerationStructure)
        {
            return mMediumInstanceAccelerationStructure;
        }
        return volumeAccelerationStructure();
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
    uint32_t geometryTransformBase() const
    {
        return static_cast<uint32_t>(mEmittedInstances.size());
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
    bool allCutoutGeometrySupportsHardwareAlpha() const
    {
        return mCutoutGeometryCount != 0u && mHardwareAlphaGeometryCount == mCutoutGeometryCount;
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
    AuditCounts auditCounts() const
    {
        return mAuditCounts;
    }

private:
    MTL::AccelerationStructure* createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor);
    void flushAccelerationStructureGroup();
    MTL::AccelerationStructure* createAccelerationStructureNoCompact(MTL::AccelerationStructureDescriptor* descriptor);
    size_t buildBlas(const std::vector<AsBuildGeometry>& geometries, bool skeletal);
    static constexpr size_t kNoCurveBlas = ~size_t{ 0 };
    size_t buildCurveBlas(uint32_t sceneInstanceId);
    size_t buildAnalyticLightBlas(uint32_t intersectionFunctionOffset);
    void ensureScratchBuffer(MTL::Buffer*& buffer, size_t requiredSize);
    void addDescriptorResidency();
    void prepareEmissiveMeshInputs();
    void uploadEmissiveMeshLights();

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
    // Indexed exactly like GeometryEntry. A valid scene-instance id means the
    // geometry was baked into world space and needs that transform for shading.
    std::vector<uint32_t> mGeometryTransformSceneInstances;
    std::vector<uint8_t> mBakedSceneInstances;
    std::vector<uint8_t> mDirtySkeletalMeshes;
    std::vector<render::EmissiveMeshBuildInput> mSceneEmissiveInputs;
    MTL::Buffer* mEmissiveMeshBuffer = nullptr;
    MTL::Buffer* mEmissiveTriangleBuffer = nullptr;
    MTL::Buffer* mAnalyticLightBoundsBuffer = nullptr;
    uint32_t mEmissiveMeshCount = 0;
    double mEmissiveMeshPower = 0.0;
    std::vector<MTL::AccelerationStructure*> mPrimitiveAccelerationStructures;
    // Non-owning view of the sole immutable, world-space triangle BLAS when
    // extend can bypass the TLAS. Ownership remains in mBlasList/the primitive
    // AS vector; the indices preserve the HitRecord ABI used by shade.
    MTL::AccelerationStructure* mDirectStaticAccelerationStructure = nullptr;
    uint32_t mDirectStaticGeometryBase = 0;
    uint32_t mDirectStaticInstanceIndex = 0;
    MTL::AccelerationStructure* mInstanceAccelerationStructure = nullptr;
    MTL::AccelerationStructure* mVolumeInstanceAccelerationStructure = nullptr;
    MTL::AccelerationStructure* mMediumInstanceAccelerationStructure = nullptr;
    // Reused for every TLAS refit. Path-owned descriptor type (MTL3 or MTL4).
    MTL::AccelerationStructureDescriptor* mTlasDescriptor = nullptr;
    MTL::AccelerationStructureDescriptor* mVolumeTlasDescriptor = nullptr;
    MTL::AccelerationStructureDescriptor* mMediumTlasDescriptor = nullptr;
    MTL::Buffer* mInstanceBuffer = nullptr;
    MTL::Buffer* mPreviousInstanceBuffer = nullptr;
    bool mInstanceTransformsChanged = false;
    MTL::Buffer* mTlasScratchBuffer = nullptr;
    MTL::Buffer* mVolumeTlasScratchBuffer = nullptr;
    MTL::Buffer* mMediumTlasScratchBuffer = nullptr;
    size_t mTlasInstanceCount = 0;
    size_t mVolumeTlasInstanceCount = 0;
    size_t mMediumTlasInstanceCount = 0;
    std::vector<std::pair<MTL::AccelerationStructure*, uint64_t>> mRetiredInstanceStructures;
    uint64_t mTlasEncodeCount = 0;

    AsBuildState* mAsBuild = nullptr;

    bool mMotionBlasBuilt = false;
    bool mBuildMotionBlas = false;
    uint32_t mOpaqueGeometryCount = 0;
    uint32_t mCutoutGeometryCount = 0;
    uint32_t mHardwareAlphaGeometryCount = 0;
    uint32_t mPrimitiveSurfaceGeometryCount = 0;
    uint64_t mPrimitiveSurfaceTriangleCount = 0;
    uint32_t mExtendedLimitBlasCount = 0;
    // What the bottom-level builds cost this scene. Reset when a build starts.
    double mBlasEncodeMs = 0.0;
    uint32_t mBlasCount = 0;
    AuditCounts mAuditCounts;

    // Matches the renderer's frames in flight. Kept here rather than shared,
    // because being wrong on the high side only delays a free.
    static constexpr uint64_t kMaxFramesInFlight = 3;
    static constexpr size_t kMaxBlasRebuildsPerFrame = 4;
    static constexpr size_t kMaxGeometriesPerRefitBlas = 32;
    size_t mNextBlasRebuildIndex = 0;
};

} // namespace oka::metal
