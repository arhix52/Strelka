#include "MetalAccelStructure.h"

#include "ShaderTypes.h"

#include <env.h>
#include <log.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

namespace oka
{
namespace metal
{

struct AsBuildState
{
    // Instances that hang off the same node share a transform by construction.
    // The transform is part of the key because one node can carry a million
    // placements -- that is what EXT_mesh_gpu_instancing is -- and keying on the
    // node alone scatters every placement after the first into groups of its own.
    struct GroupKey
    {
        int node;
        int skeletal;
        uint64_t transform;
        bool operator<(const GroupKey& o) const
        {
            if (node != o.node)
                return node < o.node;
            if (skeletal != o.skeletal)
                return skeletal < o.skeletal;
            return transform < o.transform;
        }
    };

    enum class Phase : uint32_t
    {
        Meshes = 0,
        Grouping,
        Blas,
        Curves,
        Lights,
        Finish,
        Done,
    };
    Phase phase = Phase::Meshes;

    // Meshes
    size_t meshCursor = 0;
    size_t primitiveBytes = 0;

    // Grouping
    std::vector<int> instanceNode;
    std::map<GroupKey, size_t> groupOfKey;
    std::vector<std::vector<uint32_t>> groups;
    std::vector<bool> groupSkeletal;
    size_t groupCursor = 0;

    // One BLAS per distinct geometry, shared by every group that holds the same.
    std::map<std::vector<uint64_t>, size_t> blasOfSignature;
    std::vector<size_t> groupBlas;
    size_t blasCursor = 0;
    size_t sharedBlas = 0;
    size_t mergedGeometries = 0;

    // Light instances keep one BLAS per mesh, shared between lights, because
    // their userID must stay the light index.
    std::map<uint32_t, size_t> lightBlasOfMesh;
    size_t lightCursor = 0;

    // Curve instances. One BLAS per curve set, shared between instances that
    // place the same set -- which the sidecar does not yet produce, but a
    // groomed character with two eyebrows would.
    std::map<uint32_t, size_t> curveBlasOfSet;
    size_t curveCursor = 0;

    // Per-phase wall clock, so the split between grouping and building is a
    // measurement rather than an assumption.
    double phaseMs[(size_t)Phase::Done] = {};
};

namespace
{

MTL4::BufferRange bufferRange(MTL::Buffer* buffer, size_t offset = 0)
{
    if (!buffer || offset > buffer->length())
    {
        return { 0, 0 };
    }
    return { buffer->gpuAddress() + offset, buffer->length() - offset };
}

MTL::AccelerationStructureSizes accelerationStructureSizes(
    MTL::Device* device, MTL4::AccelerationStructureDescriptor* descriptor)
{
    // Sizing has no Metal 4 counterpart -- the Metal 4 descriptors derive from
    // the Metal 3 ones precisely so the device can still be asked, so this is an
    // upcast rather than a reinterpretation.
    return device->accelerationStructureSizes(descriptor);
}

} // namespace

bool MetalAccelStructure::beginImmediate(MTL4::CommandBuffer*& commandBuffer,
                                         MTL4::ComputeCommandEncoder*& encoder)
{
    if (!mMetal4 || !mMetal4->isValid())
    {
        STRELKA_ERROR("Metal 4 acceleration-structure build requested without a valid Metal4Context");
        return false;
    }
    mMetal4->commitResidency();
    commandBuffer = mMetal4->beginImmediate();
    encoder = commandBuffer ? commandBuffer->computeCommandEncoder() : nullptr;
    return encoder != nullptr;
}

MTL::AccelerationStructure* MetalAccelStructure::createAccelerationStructure(
    MTL4::AccelerationStructureDescriptor* descriptor)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    // Query for the sizes needed to store and build the acceleration structure.
    const MTL::AccelerationStructureSizes accelSizes = accelerationStructureSizes(mDevice, descriptor);
    // Allocate an acceleration structure large enough for this descriptor. This doesn't actually
    // build the acceleration structure, it just allocates memory.
    MTL::AccelerationStructure* accelerationStructure =
        mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
    if (!accelerationStructure)
    {
        // newAccelerationStructure returns nil when the device cannot find the
        // memory, and nothing downstream notices: the TLAS ends up null, every
        // ray misses, and the result is a black frame that looks like a lighting
        // problem. Say it here instead.
        STRELKA_ERROR("Acceleration structure allocation failed: {:.2f} GB requested, "
                      "{:.2f} GB max buffer. The scene does not fit -- lower "
                      "render/texture/maxDimension or reduce geometry.",
                      accelSizes.accelerationStructureSize / 1e9, mDevice->maxBufferLength() / 1e9);
        pPool->release();
        return nullptr;
    }
    // Allocate scratch space Metal uses to build the acceleration structure.
    // Use MTLResourceStorageModePrivate for best performance because the sample
    // doesn't need access to the buffer's contents.
    MTL::Buffer* scratchBuffer =
        mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
    // Allocate a buffer for Metal to write the compacted accelerated structure's size into.
    MTL::Buffer* compactedSizeBuffer = mDevice->newBuffer(sizeof(uint64_t), MTL::ResourceStorageModeShared);
    mMetal4->addResident(accelerationStructure);
    mMetal4->addResident(scratchBuffer);
    mMetal4->addResident(compactedSizeBuffer);
    addDescriptorResidency();
    MTL4::CommandBuffer* commandBuffer = nullptr;
    MTL4::ComputeCommandEncoder* commandEncoder = nullptr;
    if (!beginImmediate(commandBuffer, commandEncoder))
    {
        mMetal4->removeResident(accelerationStructure);
        mMetal4->removeResident(scratchBuffer);
        mMetal4->removeResident(compactedSizeBuffer);
        accelerationStructure->release();
        scratchBuffer->release();
        compactedSizeBuffer->release();
        pPool->release();
        return nullptr;
    }
    // Schedule the actual acceleration structure build.
    commandEncoder->buildAccelerationStructure(accelerationStructure, descriptor, bufferRange(scratchBuffer));
    // Compute and write the compacted acceleration structure size into the buffer. You
    // need to already have a built accelerated structure because Metal determines the compacted
    // size based on the final size of the acceleration structure. Compacting an acceleration
    // structure can potentially reclaim significant amounts of memory because Metal must
    // create the initial structure using a conservative approach.
    commandEncoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageAccelerationStructure,
                                              MTL4::VisibilityOptionDevice);
    commandEncoder->writeCompactedAccelerationStructureSize(accelerationStructure, bufferRange(compactedSizeBuffer));
    // End encoding and commit the command buffer so the GPU can start building the
    // acceleration structure.
    commandEncoder->endEncoding();
    mMetal4->submitAndWait(commandBuffer);

    // The sample waits for Metal to finish executing the command buffer so that it can
    // read back the compacted size.

    // Note: Don't wait for Metal to finish executing the command buffer if you aren't compacting
    // the acceleration structure because doing so requires CPU/GPU synchronization. You don't have
    // to compact acceleration structures, but it's helpful when creating large static acceleration
    // structures, such as static scene geometry. Avoid compacting acceleration structures that
    // you rebuild every frame because the synchronization cost may be significant.

    const uint64_t compactedSize = *(uint64_t*)compactedSizeBuffer->contents();

    // commandBuffer->release();
    // commandEncoder->release();

    // Allocate a smaller acceleration structure based on the returned size.
    MTL::AccelerationStructure* compactedAccelerationStructure = mDevice->newAccelerationStructure(compactedSize);
    if (!compactedAccelerationStructure)
    {
        STRELKA_ERROR("Compacted acceleration structure allocation failed: {:.2f} GB requested",
                      compactedSize / 1e9);
        mMetal4->removeResident(accelerationStructure);
        mMetal4->removeResident(scratchBuffer);
        mMetal4->removeResident(compactedSizeBuffer);
        accelerationStructure->release();
        scratchBuffer->release();
        compactedSizeBuffer->release();
        pPool->release();
        return nullptr;
    }

    mMetal4->addResident(compactedAccelerationStructure);
    if (!beginImmediate(commandBuffer, commandEncoder))
    {
        mMetal4->removeResident(compactedAccelerationStructure);
        mMetal4->removeResident(accelerationStructure);
        mMetal4->removeResident(scratchBuffer);
        mMetal4->removeResident(compactedSizeBuffer);
        compactedAccelerationStructure->release();
        accelerationStructure->release();
        scratchBuffer->release();
        compactedSizeBuffer->release();
        pPool->release();
        return nullptr;
    }

    // Encode the command to copy and compact the acceleration structure into the
    // smaller acceleration structure.
    commandEncoder->copyAndCompactAccelerationStructure(accelerationStructure, compactedAccelerationStructure);

    // End encoding and commit the command buffer. You don't need to wait for Metal to finish
    // executing this command buffer as long as you synchronize any ray-intersection work
    // to run after this command buffer completes. Dependency tracking covers a
    // consumer on this queue; one on the Metal 4 queue waits on the event instead.
    commandEncoder->endEncoding();
    mMetal4->submitAndWait(commandBuffer);

    // commandEncoder->release();
    // commandBuffer->release();
    mMetal4->removeResident(accelerationStructure);
    mMetal4->removeResident(scratchBuffer);
    mMetal4->removeResident(compactedSizeBuffer);
    mMetal4->commitResidency();
    accelerationStructure->release();
    scratchBuffer->release();
    compactedSizeBuffer->release();

    pPool->release();

    return compactedAccelerationStructure;
}


// Close the current group: end its encoder, commit, and drop the scratch the
// group was using. submitAndWait owns the immediate allocator until every
// reference is complete, so scratch can be released immediately afterwards.
void MetalAccelStructure::flushAccelerationStructureGroup()
{
    if (!mAsGroupCommandBuffer)
    {
        return;
    }
    mAsGroupEncoder->endEncoding();
    mMetal4->submitAndWait(mAsGroupCommandBuffer);
    mAsGroupEncoder = nullptr;
    mAsGroupCommandBuffer = nullptr;
    for (MTL::Buffer* b : mAsGroupScratch)
    {
        mMetal4->removeResident(b);
        b->release();
    }
    mAsGroupScratch.clear();
    for (MTL4::AccelerationStructureDescriptor* descriptor : mAsGroupDescriptors)
    {
        descriptor->release();
    }
    mAsGroupDescriptors.clear();
    mMetal4->commitResidency();
    mAsGroupPending = 0;
}

static double sBlasSizesMs = 0.0, sBlasAllocMs = 0.0, sBlasScratchMs = 0.0, sBlasEncodeMs = 0.0;
static uint32_t sBlasCount = 0;

MTL::AccelerationStructure* MetalAccelStructure::createAccelerationStructureNoCompact(
    MTL4::AccelerationStructureDescriptor* descriptor)
{
    // The usage flags belong to the caller. This used to force Refit on
    // everything it was handed, which silently gave static geometry a tree built
    // to survive a vertex update -- a worse tree to traverse -- for nothing.
    const auto tSizes = std::chrono::steady_clock::now();
    const MTL::AccelerationStructureSizes accelSizes = accelerationStructureSizes(mDevice, descriptor);
    const auto tAlloc = std::chrono::steady_clock::now();
    MTL::AccelerationStructure* accelerationStructure =
        mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
    const auto tScratch = std::chrono::steady_clock::now();
    if (!accelerationStructure)
    {
        // newAccelerationStructure returns nil when the device cannot find the
        // memory, and nothing downstream notices: the TLAS ends up null, every
        // ray misses, and the result is a black frame that looks like a lighting
        // problem. Say it here instead.
        STRELKA_ERROR("Acceleration structure allocation failed: {:.2f} GB requested, "
                      "{:.2f} GB max buffer. The scene does not fit -- lower "
                      "render/texture/maxDimension or reduce geometry.",
                      accelSizes.accelerationStructureSize / 1e9, mDevice->maxBufferLength() / 1e9);
        return nullptr;
    }
    MTL::Buffer* scratchBuffer =
        mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
    const auto tEncode = std::chrono::steady_clock::now();

    // Builds are grouped: STRELKA_AS_GROUP structures share one encoder, each
    // with its own scratch, and the group is committed together. Apple's guidance
    // is that several builds in *one encoder* run in parallel on this hardware,
    // which needs a scratch buffer per build rather than one reused -- two builds
    // sharing scratch is a data race, not a slow path. Group size 1 is the
    // one-per-command-buffer arrangement this replaces.
    static const uint32_t kGroupSize = std::max(1u, envUint("STRELKA_AS_GROUP", 1));

    mMetal4->addResident(accelerationStructure);
    mMetal4->addResident(scratchBuffer);
    addDescriptorResidency();
    if (!mAsGroupCommandBuffer && !beginImmediate(mAsGroupCommandBuffer, mAsGroupEncoder))
    {
        // Both were made resident a few lines up. Freeing them while the
        // residency set still names them leaves it pointing at dead
        // allocations, which is a fault at the next commit rather than here.
        mMetal4->removeResident(accelerationStructure);
        mMetal4->removeResident(scratchBuffer);
        accelerationStructure->release();
        scratchBuffer->release();
        return nullptr;
    }
    mAsGroupEncoder->buildAccelerationStructure(accelerationStructure, descriptor, bufferRange(scratchBuffer));
    // Held until the group is committed: the GPU reads it for the whole build,
    // and dropping the last reference before commit is what a race would look
    // like if Metal did not retain committed resources.
    mAsGroupScratch.push_back(scratchBuffer);
    mAsGroupDescriptors.push_back(descriptor->retain());
    if (++mAsGroupPending >= kGroupSize)
    {
        flushAccelerationStructureGroup();
    }
    {
        using ms = std::chrono::duration<double, std::milli>;
        const auto tEnd = std::chrono::steady_clock::now();
        sBlasSizesMs += ms(tAlloc - tSizes).count();
        sBlasAllocMs += ms(tScratch - tAlloc).count();
        sBlasScratchMs += ms(tEncode - tScratch).count();
        sBlasEncodeMs += ms(tEnd - tEncode).count();
        ++sBlasCount;
    }
    return accelerationStructure;
}

MTL4::AccelerationStructureTriangleGeometryDescriptor* MetalAccelStructure::createStaticGeometryDescriptor(
    const oka::Mesh& sceneMesh, MTL::Buffer* perPrimitiveBuffer, uint32_t triangleCount)
{
    auto* geomDescriptor = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();

    geomDescriptor->setVertexBuffer(
        bufferRange(mGeometry->vertexBuffer(), sceneMesh.mVbOffset * sizeof(Scene::Vertex)));
    geomDescriptor->setVertexFormat(MTL::AttributeFormatFloat3);
    geomDescriptor->setVertexStride(sizeof(Scene::Vertex));
    geomDescriptor->setIndexBuffer(bufferRange(mGeometry->indexBuffer(), sceneMesh.mIndex * sizeof(uint32_t)));
    geomDescriptor->setIndexType(MTL::IndexTypeUInt32);
    geomDescriptor->setTriangleCount(triangleCount);
    if (perPrimitiveBuffer)
    {
        geomDescriptor->setPrimitiveDataBuffer(bufferRange(perPrimitiveBuffer));
        geomDescriptor->setPrimitiveDataElementSize(sizeof(Triangle));
        geomDescriptor->setPrimitiveDataStride(sizeof(Triangle));
    }

    return geomDescriptor;
}

// Build one acceleration structure covering every listed instance's mesh as a
// separate geometry, and record the per-geometry lookup entries.
// Metal 4's flag asking the builder to trade build time and structure size for
// traversal speed. A separate bit from Refit, so it composes with the one this
// renderer already needs for skinning.
//
// On by default because it measured as a free 4.1% on the pine forest: 48.94 ms
// against 46.92 (n=8 each, two mirrored ABBA blocks to cancel thermal drift,
// t=3.14), with the image unchanged and the structures the same 5.46 GB -- the
// concern that a scene this close to the buffer limit could not afford it did
// not materialise. NVIDIA report the same order for PREFER_FAST_TRACE.
//
// The escape hatch is here because that is one scene on one driver, and the
// flag is exactly the kind of hint whose sign can change with either.
static MTL::AccelerationStructureUsage blasExtraUsage()
{
    static const bool disabled = envFlag("STRELKA_NO_PREFER_FAST_INTERSECTION");
    return disabled ? MTL::AccelerationStructureUsageNone
                    : MTL::AccelerationStructureUsagePreferFastIntersection;
}


size_t MetalAccelStructure::buildBlas(const std::vector<uint32_t>& sceneInstanceIds, bool skeletal)
{
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();

    Blas blas;
    blas.mIsSkeletal = skeletal;
    blas.mGeometryBase = (uint32_t)mGeometry->geometryEntries().size();

    std::vector<const NS::Object*> geomDescriptors;
    geomDescriptors.reserve(sceneInstanceIds.size());

    for (const uint32_t instId : sceneInstanceIds)
    {
        const oka::Instance& inst = instances[instId];
        const uint32_t meshId = inst.mMeshId;
        const oka::Mesh& mesh = meshes[meshId];
        MetalGeometry::Mesh* meshData = mGeometry->meshes()[meshId];

        // Opaque unless this geometry's own material has a cutout. Leaving the
        // instance flag alone lets this decide, and traversal then resolves a
        // trunk or a rock without ever leaving the ray tracing unit.
        // STRELKA_ALL_GEOM_OPAQUE=1 forces every geometry opaque -- a diagnostic
        // for whether the alpha test is running at all, not a rendering mode.
        static const bool forceAllOpaque = envFlag("STRELKA_ALL_GEOM_OPAQUE");
        const bool isCutout = !forceAllOpaque && inst.mMaterialId < mMaterials->isCutout().size() &&
                              mMaterials->isCutout()[inst.mMaterialId] != 0;

        MTL4::AccelerationStructureGeometryDescriptor* geom = nullptr;
        if (skeletal && mBuildMotionBlas)
        {
            geom = createMotionGeometryDescriptor(mesh, meshData->mPerPrimitiveBuffer, meshData->mTriangleCount,
                                                  blas.mMotionVertexRangeBuffers);
        }
        else
        {
            geom = createStaticGeometryDescriptor(mesh, meshData->mPerPrimitiveBuffer, meshData->mTriangleCount);
        }
        // STRELKA_NO_SET_OPAQUE=1 leaves the descriptor's own default in place,
        // which is what the code did before this flag existed -- the control for
        // telling "the alpha test changed" apart from "the alpha test started".
        static const bool leaveDefault = envFlag("STRELKA_NO_SET_OPAQUE");
        if (!leaveDefault)
        {
            geom->setOpaque(!isCutout);
        }
        (isCutout ? mCutoutGeometryCount : mOpaqueGeometryCount)++;
        geomDescriptors.push_back(geom);

        // The geometry index within this BLAS is the position in this list, so
        // the material of the instance that contributed it lands in the right slot.
        GeometryEntry entry{};
        entry.vbOffset = mesh.mVbOffset;
        entry.indexOffset = mesh.mIndex;
        entry.materialId = inst.mMaterialId;
        mGeometry->geometryEntries().push_back(entry);
    }

    NS::Array* geomArray = NS::Array::array(geomDescriptors.data(), geomDescriptors.size());
    MTL4::PrimitiveAccelerationStructureDescriptor* primDescriptor =
        MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    primDescriptor->setGeometryDescriptors(geomArray);

    if (skeletal)
    {
        if (mBuildMotionBlas)
        {
            primDescriptor->setMotionKeyframeCount(2);
            primDescriptor->setMotionStartTime(0.0f);
            primDescriptor->setMotionEndTime(1.0f);
            primDescriptor->setMotionStartBorderMode(MTL::MotionBorderModeClamp);
            primDescriptor->setMotionEndBorderMode(MTL::MotionBorderModeClamp);
        }
        primDescriptor->setUsage(MTL::AccelerationStructureUsageRefit | blasExtraUsage());

        const MTL::AccelerationStructureSizes sizes = accelerationStructureSizes(mDevice, primDescriptor);
        blas.mRefitScratchSize = sizes.refitScratchBufferSize;
        blas.mBuildScratchSize = sizes.buildScratchBufferSize;
        ensureScratchBuffer(blas.mScratch, std::max(blas.mBuildScratchSize, blas.mRefitScratchSize));

        blas.mAs = createAccelerationStructureNoCompact(primDescriptor);
        blas.mDescriptor = primDescriptor; // kept for refit, released in the destructor
    }
    else
    {
        // Refit even though nothing here is ever refitted, and no compaction.
        //
        // Both of those are backwards from what the flags suggest, and both are
        // measured. Refit is meant to trade tree quality for the ability to
        // update in place; on this driver it also halves what the builder
        // allocates -- 4.80 GB of structures against 8.80 without, which on a
        // 9.53 GB buffer limit is the difference between fitting and not -- and
        // the result traverses no slower.
        //
        // Compaction is a loss on every axis here. It needs a
        // round trip per structure to read the compacted size back, and with a
        // hundred and thirty of them that was 4.9 seconds of a 13.8 second load.
        // It did not even save memory: 4.80 GB of acceleration structures
        // without it against 4.97 GB with. And the uncompacted trees traverse
        // faster -- 27.9 s against 29.8 for 128 spp of the pine forest.
        primDescriptor->setUsage(MTL::AccelerationStructureUsageRefit | blasExtraUsage());
        blas.mAs = createAccelerationStructureNoCompact(primDescriptor);
        primDescriptor->release();
    }

    for (const NS::Object* g : geomDescriptors)
    {
        ((NS::Object*)g)->release();
    }

    mBlasList.push_back(blas);
    mPrimitiveAccelerationStructures.push_back(blas.mAs);
    return mBlasList.size() - 1;
}

// One acceleration structure per curve set.
//
// Not merged with the triangle path: a curve geometry descriptor names different
// buffers and a different primitive, and the grouping that merges a glTF mesh's
// primitives has nothing to merge here -- the converter already writes one set
// per material.
size_t MetalAccelStructure::buildCurveBlas(uint32_t sceneInstanceId)
{
    const oka::Instance& inst = mScene->getInstances()[sceneInstanceId];
    const uint32_t curveId = inst.mCurveId;
    if (curveId >= mGeometry->curveRanges().size() || mGeometry->curveRanges()[curveId].segmentCount == 0 || !mGeometry->curvePointBuffer())
    {
        return (size_t)-1;
    }
    const MetalGeometry::CurveRange& range = mGeometry->curveRanges()[curveId];
    const oka::Curve& curve = mScene->getCurves()[curveId];

    auto* geom = MTL4::AccelerationStructureCurveGeometryDescriptor::alloc()->init();
    geom->setControlPointBuffer(bufferRange(mGeometry->curvePointBuffer()));
    geom->setControlPointCount(mScene->getCurvesPoint().size());
    geom->setControlPointFormat(MTL::AttributeFormatFloat3);
    geom->setControlPointStride(sizeof(glm::float3));
    geom->setRadiusBuffer(bufferRange(mGeometry->curveRadiusBuffer()));
    geom->setRadiusFormat(MTL::AttributeFormatFloat);
    geom->setRadiusStride(sizeof(float));
    geom->setIndexBuffer(
        bufferRange(mGeometry->curveSegmentBuffer(), range.segmentStart * sizeof(uint32_t)));
    geom->setIndexType(MTL::IndexTypeUInt32);
    geom->setSegmentCount(range.segmentCount);
    geom->setSegmentControlPointCount(range.controlPointsPerSegment);
    geom->setCurveType(MTL::CurveTypeRound);
    geom->setCurveBasis(curve.mType == oka::Curve::Type::eLinear ? MTL::CurveBasisLinear
                                                                 : MTL::CurveBasisBSpline);
    // Spherical caps on a linear basis, so consecutive segments of one strand
    // join without a notch at every control point. A B-spline is already
    // continuous there and takes disks at the two real ends.
    geom->setCurveEndCaps(curve.mType == oka::Curve::Type::eLinear ? MTL::CurveEndCapsSphere
                                                                   : MTL::CurveEndCapsDisk);
    // Hair has no alpha cutout, and leaving it non-opaque would send every curve
    // hit through an intersection function that does not exist for curves.
    geom->setOpaque(true);

    Blas blas;
    blas.mIsSkeletal = false;
    blas.mGeometryBase = (uint32_t)mGeometry->geometryEntries().size();

    GeometryEntry entry{};
    entry.vbOffset = curve.mPointsStart;
    entry.indexOffset = range.segmentStart;
    entry.materialId = inst.mMaterialId;
    entry.flags = GEOM_FLAG_CURVE | (range.segmentsPerStrand & GEOM_CURVE_STRAND_MASK) |
                  (curve.mType == oka::Curve::Type::eLinear ? 0u : GEOM_CURVE_CUBIC);
    mGeometry->geometryEntries().push_back(entry);

    const NS::Object* geoms[] = { geom };
    MTL4::PrimitiveAccelerationStructureDescriptor* primDescriptor =
        MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    primDescriptor->setGeometryDescriptors(NS::Array::array(geoms, 1));
    primDescriptor->setUsage(MTL::AccelerationStructureUsageRefit | blasExtraUsage());
    blas.mAs = createAccelerationStructureNoCompact(primDescriptor);
    primDescriptor->release();
    geom->release();

    mBlasList.push_back(blas);
    mPrimitiveAccelerationStructures.push_back(blas.mAs);
    return mBlasList.size() - 1;
}


// Rebuild the acceleration structures for a different motion setting.
//
// Only reachable from the motion-blur toggle, which is a UI action, so a hitch is
// acceptable — but the structures about to be released may still be referenced by
// the last frame's command buffers, hence the drain.
void MetalAccelStructure::rebuild()
{
    // An empty immediate submission is ordered after every prior Metal 4 frame
    // and submitAndWait does not return until that point has completed.
    MTL4::CommandBuffer* drain = nullptr;
    MTL4::ComputeCommandEncoder* drainEncoder = nullptr;
    if (beginImmediate(drain, drainEncoder))
    {
        drainEncoder->endEncoding();
        mMetal4->submitAndWait(drain);
    }

    auto safeRelease = [](auto*& p) {
        if (p)
        {
            p->release();
            p = nullptr;
        }
    };
    for (Blas& blas : mBlasList)
    {
        mMetal4->removeResident(blas.mScratch);
        safeRelease(blas.mScratch);
        safeRelease(blas.mDescriptor);
        for (MTL::Buffer* buffer : blas.mMotionVertexRangeBuffers)
        {
            mMetal4->removeResident(buffer);
            buffer->release();
        }
        blas.mMotionVertexRangeBuffers.clear();
    }
    mBlasList.clear();
    // The drain above is what makes this safe to free outright.
    releaseRetiredInstanceStructures(0);
    // blas.mAs and the entries here are the same objects; release through one path.
    for (auto*& as : mPrimitiveAccelerationStructures)
    {
        mMetal4->removeResident(as);
        safeRelease(as);
    }
    mPrimitiveAccelerationStructures.clear();
    mMetal4->removeResident(mInstanceAccelerationStructure);
    safeRelease(mInstanceAccelerationStructure);
    safeRelease(mTlasDescriptor);
    mMetal4->removeResident(mInstanceBuffer);
    safeRelease(mInstanceBuffer);
    mMetal4->removeResident(mPreviousInstanceBuffer);
    safeRelease(mPreviousInstanceBuffer);
    mInstanceTransformsChanged = false;
    mGeometry->clearGeometryEntries();
    mMetal4->removeResident(mTlasScratchBuffer);
    safeRelease(mTlasScratchBuffer);
    mMetal4->commitResidency();

    const auto rebuildStart = std::chrono::high_resolution_clock::now();
    create();
    STRELKA_INFO("Acceleration structures rebuilt for motion={} in {:.1f} ms", mBuildMotionBlas,
                 std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - rebuildStart)
                     .count());
}

void MetalAccelStructure::create()
{
    // No budget: one call does the lot, which is what a rebuild driven by an
    // animation and what the headless path both want.
    while (!step(0.0))
    {
    }
}

bool MetalAccelStructure::step(double budgetMs)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    auto finish = [&](bool complete) {
        pPool->release();
        return complete;
    };

    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
    const std::vector<oka::Curve>& curves = mScene->getCurves();
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    if (meshes.empty() && curves.empty())
    {
        delete mAsBuild;
        mAsBuild = nullptr;
        return finish(true);
    }

    using Phase = AsBuildState::Phase;
    if (!mAsBuild)
    {
        mAsBuild = new AsBuildState();
        mOpaqueGeometryCount = 0;
        mCutoutGeometryCount = 0;
    }
    AsBuildState& st = *mAsBuild;

    // The clock is read every so many iterations rather than on each one: for the
    // cheap majority of items -- a group that shares an existing structure --
    // reading it costs more than the work it is measuring.
    constexpr size_t kCheckEvery = 64;
    const auto sliceStart = std::chrono::steady_clock::now();
    auto mark = sliceStart;
    auto chargePhase = [&]() {
        const auto t = std::chrono::steady_clock::now();
        st.phaseMs[(size_t)st.phase] += std::chrono::duration<double, std::milli>(t - mark).count();
        mark = t;
    };
    // Grouping and building are one item per scene instance each, so the two
    // cursors against twice the instance count is close enough for a bar -- the
    // group count is within a percent of the instance count on any scene where
    // this stage is long enough to matter.
    auto report = [&]() {
        if (mLoadProgress)
        {
            mLoadProgress->total.store((uint32_t)(instances.size() * 2), std::memory_order_relaxed);
            mLoadProgress->done.store((uint32_t)(st.groupCursor + st.blasCursor), std::memory_order_relaxed);
        }
    };
    // `force` is for the items that are not cheap: one BLAS build can outlast the
    // whole slice budget on its own, and sampling the clock every kCheckEvery
    // items would then let 511 more of them through behind it.
    auto outOfTime = [&](size_t iteration, bool force = false) {
        if (budgetMs <= 0.0 || (!force && (iteration % kCheckEvery) != 0))
        {
            return false;
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sliceStart).count() >=
               budgetMs;
    };

    if (st.phase == Phase::Meshes)
    {
        // Per-mesh buffers survive a rebuild of the structures that reference them.
        if (mGeometry->meshes().empty())
        {
            // Read once, here: the meshes are built now and the acceleration
            // structures embed whatever they are given, so a later change of tracer
            // cannot retroactively add the data.
            // Nothing reads per-primitive data now: it existed for the megakernel.
            mNeedsPrimitiveData = false;
            while (st.meshCursor < meshes.size())
            {
                mGeometry->createMeshData(mScene, st.meshCursor, mNeedsPrimitiveData);
                st.primitiveBytes += (size_t)(meshes[st.meshCursor].mCount / 3) * sizeof(Triangle);
                ++st.meshCursor;
                if (outOfTime(st.meshCursor))
                {
                    chargePhase();
                    report();
                    return finish(false);
                }
            }
            if (!mNeedsPrimitiveData && st.primitiveBytes > 0)
            {
                STRELKA_INFO("Skipped {:.2f} GB of per-primitive attribute data: the wavefront tracer "
                             "refetches from the vertex buffer",
                             st.primitiveBytes / 1e9);
            }
        }
        mMotionBlasBuilt = mBuildMotionBlas;

        // Before the structures are built, not after. The host arrays are a full
        // duplicate of the GPU buffers -- 2.27 GB here -- and holding them across the
        // acceleration structure build stacks that on top of the largest allocation
        // the renderer makes. Nothing below this point reads them: buildBlas() works
        // from buffer offsets and triangle counts.
        //
        // Off by default because Scene::pick() walks these arrays -- the editor needs
        // them, a headless render does not.
        mGeometry->hostGeometryBytes() = { mScene->getVertices().size() * sizeof(Scene::Vertex),
                               mScene->getIndices().size() * sizeof(uint32_t) };
        if (mSettings->getAs<bool>("scene/releaseHostGeometry") && !mScene->hostGeometryReleased())
        {
            mScene->releaseHostGeometry();
        }

        // --- Group mesh instances that always move together -----------------------
        //
        // glTF splits a mesh into primitives by material, and the loader turns each
        // primitive into its own instance. Left alone that produces one BLAS per
        // primitive, all with the same transform and heavily overlapping bounds, and
        // every ray that has to traverse deeply pays for the overlap. Instances that
        // hang off the same node share a transform by construction, so they can be
        // merged into a single BLAS with one geometry per primitive.
        st.instanceNode.assign(instances.size(), -1);
        const std::vector<Scene::Node>& nodes = mScene->getNodes();
        for (size_t n = 0; n < nodes.size(); ++n)
        {
            for (const uint32_t id : nodes[n].instanceIds)
            {
                if (id < st.instanceNode.size())
                    st.instanceNode[id] = (int)n;
            }
        }

        mGeometry->geometryEntries().clear();
        mEmittedInstances.clear();

        chargePhase();
        st.phase = Phase::Grouping;
    }

    auto hashTransform = [](const glm::mat4& m) {
        uint64_t h = 1469598103934665603ull;
        const auto* raw = reinterpret_cast<const unsigned char*>(&m);
        for (size_t i = 0; i < sizeof(glm::mat4); ++i)
        {
            h ^= raw[i];
            h *= 1099511628211ull;
        }
        return h;
    };

    if (st.phase == Phase::Grouping)
    {
        while (st.groupCursor < instances.size())
        {
            const size_t i = st.groupCursor++;
            const oka::Instance& curr = instances[i];
            // Lights and curves each have a phase of their own. A curve instance
            // reaching here would read meshes[mCurveId] -- the two ids share a
            // union -- and index a mesh that has nothing to do with it.
            if (curr.type == oka::Instance::Type::eMesh)
            {
                const bool skeletal = meshes[curr.mMeshId].isSkeletal;
                const int nodeId = st.instanceNode[i];
                const AsBuildState::GroupKey key{ nodeId >= 0 ? nodeId : -(int)i - 2, skeletal ? 1 : 0,
                                                  hashTransform(curr.transform) };

                auto it = st.groupOfKey.find(key);
                if (it == st.groupOfKey.end())
                {
                    st.groupOfKey[key] = st.groups.size();
                    st.groups.push_back({ (uint32_t)i });
                    st.groupSkeletal.push_back(skeletal);
                }
                // Merging is only valid while the members share a transform.
                // Component equality, not memcmp: glm::mat4 has no unique object
                // representation (padding / signalling NaNs), and tidy flags the
                // byte compare for that reason.
                else if (instances[st.groups[it->second].front()].transform == curr.transform)
                {
                    st.groups[it->second].push_back((uint32_t)i);
                }
                else
                {
                    st.groups.push_back({ (uint32_t)i });
                    st.groupSkeletal.push_back(skeletal);
                }
            }
            if (outOfTime(st.groupCursor))
            {
                chargePhase();
                report();
                return finish(false);
            }
        }
        chargePhase();
        st.phase = Phase::Blas;
        // The key map has done its job and is the largest thing here -- an entry
        // per scene instance. Nothing below reads it.
        st.groupOfKey.clear();
        st.instanceNode.clear();
        st.instanceNode.shrink_to_fit();
    }

    if (st.phase == Phase::Blas)
    {
        // Share one BLAS between every group that holds the same geometry.
        //
        // Scattered scenes are built almost entirely out of repeats: the pine forest
        // places 38 000 instances drawn from 50 distinct objects. A BLAS per instance
        // would be 38 000 structures over the same 50 meshes, which is both the build
        // time and the memory of a scene 700 times larger than the one authored.
        //
        // The signature is (mesh, material) per geometry rather than mesh alone,
        // because the material is baked into the shared geometry entries -- two
        // instances of the same mesh with different materials are not the same BLAS.
        // Skeletal groups are excluded: their vertices are rewritten per frame and
        // the structure refit alongside, so sharing one would mean two instances
        // deforming the same geometry.
        while (st.blasCursor < st.groups.size())
        {
            const size_t g = st.blasCursor++;
            std::vector<uint64_t> signature;
            signature.reserve(st.groups[g].size());
            for (const uint32_t id : st.groups[g])
            {
                signature.push_back(((uint64_t)instances[id].mMeshId << 32) | instances[id].mMaterialId);
            }

            size_t blasIdx;
            auto shared = st.groupSkeletal[g] ? st.blasOfSignature.end() : st.blasOfSignature.find(signature);
            const bool built = shared == st.blasOfSignature.end();
            if (!built)
            {
                blasIdx = shared->second;
                ++st.sharedBlas;
            }
            else
            {
                blasIdx = buildBlas(st.groups[g], st.groupSkeletal[g]);
                st.mergedGeometries += st.groups[g].size();
                if (!st.groupSkeletal[g])
                {
                    st.blasOfSignature.emplace(std::move(signature), blasIdx);
                }
            }

            EmittedInstance emitted{};
            emitted.sceneInstanceId = st.groups[g].front();
            emitted.asIndex = (uint32_t)blasIdx;
            emitted.userID = mBlasList[blasIdx].mGeometryBase;
            // The boundary of a participating medium gets its own mask: shadow
            // rays must pass through it, or a fog gizmo blacks out everything it
            // encloses. Groups are keyed by node, so a gizmo -- one object, one
            // material -- is never merged with anything else, and reading the
            // first member's material is reading the group's.
            const uint32_t groupMaterial = instances[st.groups[g].front()].mMaterialId;
            const bool isMediumBoundary = groupMaterial < mMaterials->isMediumBoundary().size() &&
                                          mMaterials->isMediumBoundary()[groupMaterial] != 0u;
            emitted.mask = isMediumBoundary ? GEOMETRY_MASK_MEDIUM : GEOMETRY_MASK_TRIANGLE;

            mEmittedInstances.push_back(emitted);

            if (outOfTime(st.blasCursor, built))
            {
                // Close whatever group the last build landed in before handing
                // control back, so the scratch buffers it holds are not carried
                // across frames waiting for a group that may be many slices away
                // from filling up.
                flushAccelerationStructureGroup();
                chargePhase();
                report();
                return finish(false);
            }
        }
        chargePhase();
        st.phase = Phase::Curves;
    }

    if (st.phase == Phase::Curves)
    {
        while (st.curveCursor < instances.size())
        {
            const size_t i = st.curveCursor++;
            const oka::Instance& curr = instances[i];
            if (curr.type == oka::Instance::Type::eCurve)
            {
                auto it = st.curveBlasOfSet.find(curr.mCurveId);
                if (it == st.curveBlasOfSet.end())
                {
                    const size_t blasIdx = buildCurveBlas((uint32_t)i);
                    if (blasIdx == (size_t)-1)
                    {
                        continue; // an empty set: warned about in buildCurveBuffers
                    }
                    it = st.curveBlasOfSet.emplace(curr.mCurveId, blasIdx).first;
                }
                EmittedInstance emitted{};
                emitted.sceneInstanceId = (uint32_t)i;
                emitted.asIndex = (uint32_t)it->second;
                emitted.userID = mBlasList[it->second].mGeometryBase;
                emitted.mask = GEOMETRY_MASK_CURVE;
                mEmittedInstances.push_back(emitted);
            }
            if (outOfTime(st.curveCursor, true))
            {
                flushAccelerationStructureGroup();
                chargePhase();
                report();
                return finish(false);
            }
        }
        chargePhase();
        st.phase = Phase::Lights;
    }

    if (st.phase == Phase::Lights)
    {
        while (st.lightCursor < instances.size())
        {
            const size_t i = st.lightCursor++;
            const oka::Instance& curr = instances[i];
            if (curr.type == oka::Instance::Type::eLight)
            {
                auto it = st.lightBlasOfMesh.find(curr.mMeshId);
                if (it == st.lightBlasOfMesh.end())
                {
                    const size_t blasIdx = buildBlas({ (uint32_t)i }, meshes[curr.mMeshId].isSkeletal);
                    it = st.lightBlasOfMesh.emplace(curr.mMeshId, blasIdx).first;
                }
                EmittedInstance emitted{};
                emitted.sceneInstanceId = (uint32_t)i;
                emitted.asIndex = (uint32_t)it->second;
                emitted.userID = curr.mLightId; // lights address the light table, not geometry
                // Point/spot proxies exist for picking and the gizmo; they are not
                // emissive surfaces. Putting them on the light mask would treat their
                // radiant intensity as radiance and blow out the frame.
                //
                // They must not be on the geometry mask either. A point light is sampled
                // at its centre, which sits inside the proxy sphere, so a shadow ray to
                // it necessarily crosses the shell -- and RAY_MASK_SHADOW *is*
                // GEOMETRY_MASK_GEOMETRY, so every next-event connection to a point or
                // spot light was reported occluded and those lights lit nothing at all.
                // Picking runs on the CPU in Scene::pick() and never consults these
                // masks, so a proxy invisible to every ray costs nothing.
                const int lightType =
                    curr.mLightId < mScene->getLightsDesc().size() ? mScene->getLightsDesc()[curr.mLightId].type : -1;
                const bool enabled = curr.mLightId < mScene->getLightsDesc().size() ?
                                         mScene->getLightsDesc()[curr.mLightId].enabled :
                                         true;
                const bool visibleToCamera = curr.mLightId < mScene->getLightsDesc().size() ?
                                                 mScene->getLightsDesc()[curr.mLightId].visibleToCamera :
                                                 true;
                if (!enabled || lightType == LIGHT_TYPE_POINT || lightType == LIGHT_TYPE_SPOT)
                    emitted.mask = 0;
                else
                    emitted.mask = visibleToCamera ? GEOMETRY_MASK_LIGHT : GEOMETRY_MASK_LIGHT_HIDDEN;
                mEmittedInstances.push_back(emitted);
            }
            if (outOfTime(st.lightCursor))
            {
                flushAccelerationStructureGroup();
                chargePhase();
                report();
                return finish(false);
            }
        }
        chargePhase();
        st.phase = Phase::Finish;
        // Not needed past this point and one vector per group, so worth dropping
        // before the top level is built rather than after.
        st.groups.clear();
        st.groups.shrink_to_fit();
        st.groupSkeletal.clear();
        st.blasOfSignature.clear();
    }

    // --- Finish: buffers and the top level ------------------------------------
    //
    // Not sliced. What is left is one pass over the emitted instances and a single
    // top-level build, which measures in the tens of milliseconds even on the
    // largest scene here -- and it cannot be interrupted anyway, since the TLAS
    // descriptor has to see every structure at once.

    // Hand the host copies back once everything that reads them has run: the
    // vertex and index buffers are uploaded, the per-primitive data (if the
    // megakernel wanted any) is built, and the structures are up. On unified
    // memory the GPU-visible buffers and these vectors come out of the same pool,
    // so the duplicate is real, not bookkeeping.
    //
    // Off by default because Scene::pick() walks these arrays -- the editor needs
    // them, a headless render does not.
    const size_t vtxBytes = mGeometry->hostGeometryBytes().first;
    const size_t idxBytes = mGeometry->hostGeometryBytes().second;
    const bool hostFreed = mScene->hostGeometryReleased();

    // Where the memory goes. On a 50 M triangle scene the total runs past what a
    // 16 GB machine holds resident, and the first question is always which part
    // -- so state it rather than leave it to Activity Monitor.
    {
        size_t texBytes = 0;
        for (MTL::Texture* t : mTextures->materialTextures())
        {
            if (!t)
                continue;
            // Every level of a full mip chain adds a third again.
            texBytes += (size_t)t->width() * t->height() * 4 * 4 / 3;
        }
        STRELKA_INFO("Memory: vertices {:.2f} GB, indices {:.2f} GB, textures {:.2f} GB with mips; "
                     "host geometry {}",
                     vtxBytes / 1e9, idxBytes / 1e9, texBytes / 1e9,
                     hostFreed ? "released (picking disabled for this scene)" : "kept (doubles the first two)");
    }

    STRELKA_INFO("Acceleration structures: {} BLAS ({} geometries, {} groups shared one), "
                 "{} TLAS instances (from {} scene instances)",
                 mBlasList.size(), st.mergedGeometries, st.sharedBlas, mEmittedInstances.size(), instances.size());
    STRELKA_INFO("Geometry opacity: {} opaque, {} cutout ({:.1f}% of geometries need the alpha test)",
                 mOpaqueGeometryCount, mCutoutGeometryCount,
                 100.0 * mCutoutGeometryCount /
                     std::max<uint32_t>(1u, mOpaqueGeometryCount + mCutoutGeometryCount));

    // Per-geometry lookup table consumed by the kernel.
    mGeometry->uploadGeometryEntryBuffer();

    mInstanceBuffer = mDevice->newBuffer(
        sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor) *
            std::max<size_t>(mEmittedInstances.size(), 1),
        MTL::ResourceStorageModeShared);
    auto* instanceDescriptors =
        static_cast<MTL::IndirectAccelerationStructureInstanceDescriptor*>(mInstanceBuffer->contents());
    for (size_t d = 0; d < mEmittedInstances.size(); ++d)
    {
        const EmittedInstance& e = mEmittedInstances[d];
        instanceDescriptors[d].accelerationStructureID = mBlasList[e.asIndex].mAs->gpuResourceID();
        // Not marked opaque when the scene has cutouts: the flag makes traversal
        // skip the intersection function, which is what performs the alpha test.
        // The kernels that do not want the test -- extend, and the shadow path of
        // a scene without cutouts -- force opacity on the intersector instead,
        // which overrides this and costs them nothing.
        instanceDescriptors[d].options = mMaterials->hasAlphaMaterials()
                                             ? MTL::AccelerationStructureInstanceOptionNone
                                             : MTL::AccelerationStructureInstanceOptionOpaque;
        instanceDescriptors[d].intersectionFunctionTableOffset = 0;
        instanceDescriptors[d].userID = e.userID;
        instanceDescriptors[d].mask = e.mask;
    }
    writeInstanceTransforms(mInstanceBuffer);
    // Both buffers carry identical immutable descriptor fields. Transform
    // updates can now exchange their roles and overwrite only the new current
    // one; the old current remains the previous rendered pose at zero copy cost.
    mPreviousInstanceBuffer = mDevice->newBuffer(mInstanceBuffer->length(), MTL::ResourceStorageModeShared);
    if (mPreviousInstanceBuffer)
    {
        std::memcpy(mPreviousInstanceBuffer->contents(), mInstanceBuffer->contents(), mInstanceBuffer->length());
    }
    else
    {
        STRELKA_ERROR("Previous instance descriptor buffer allocation failed");
    }
    mInstanceTransformsChanged = false;

    // Every BLAS must be committed before the TLAS that references them is
    // encoded, so close whatever group the last one landed in.
    flushAccelerationStructureGroup();
    mTlasDescriptor = MTL4::InstanceAccelerationStructureDescriptor::alloc()->init();
    MTL4::InstanceAccelerationStructureDescriptor* accelDescriptor = mTlasDescriptor;
    accelDescriptor->setInstanceCount(mEmittedInstances.size());
    accelDescriptor->setInstanceDescriptorBuffer(bufferRange(mInstanceBuffer));
    accelDescriptor->setInstanceDescriptorType(
        MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
    accelDescriptor->setInstanceDescriptorStride(
        sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor));
    accelDescriptor->setInstanceTransformationMatrixLayout(MTL::MatrixLayoutColumnMajor);
    // The top level was built with no usage flags at all, which cost two things.
    // Refit is one of them: rebuildAccelerationStructures refits this structure
    // when an animation moves instances, and refitting one not built to be
    // refittable is undefined -- canRefit checks the instance count and the size
    // and never asked what it was built as. The other is that every ray goes
    // through this structure and a million instances of it, so it wants the same
    // fast-intersection preference the bottom level already gets.
    //
    // STRELKA_TLAS_USAGE selects them for measurement: 1 = Refit, 2 = prefer fast
    // intersection, 3 = both.
    static const uint32_t kTlasUsage = envUint("STRELKA_TLAS_USAGE", 3);
    accelDescriptor->setUsage(
        ((kTlasUsage & 1u) ? MTL::AccelerationStructureUsageRefit : MTL::AccelerationStructureUsageNone) |
        ((kTlasUsage & 2u) ? MTL::AccelerationStructureUsagePreferFastIntersection
                           : MTL::AccelerationStructureUsageNone));

    mInstanceAccelerationStructure = createAccelerationStructure(accelDescriptor);
    if (!mInstanceAccelerationStructure)
    {
        STRELKA_ERROR("Top-level acceleration structure could not be built; every ray will miss "
                      "and the image will be black.");
    }
    {
        size_t asBytes = 0;
        size_t nullAs = 0;
        for (const Blas& b : mBlasList)
        {
            if (b.mAs)
                asBytes += b.mAs->size();
            else
                ++nullAs;
        }
                // The largest few, because a structure that should have been shared and
        // was not is worth several gigabytes and is invisible in the total.
        {
            std::vector<std::pair<size_t, size_t>> bySize;
            bySize.reserve(mBlasList.size());
            for (size_t bi = 0; bi < mBlasList.size(); ++bi)
            {
                if (mBlasList[bi].mAs)
                    bySize.emplace_back(mBlasList[bi].mAs->size(), bi);
            }
            std::sort(bySize.rbegin(), bySize.rend());
            for (size_t k = 0; k < std::min<size_t>(5, bySize.size()); ++k)
            {
                STRELKA_INFO("  BLAS {} : {:.2f} GB, geometry base {}", bySize[k].second,
                             bySize[k].first / 1e9, mBlasList[bySize[k].second].mGeometryBase);
            }
        }
STRELKA_INFO("BLAS build CPU: sizes {:.0f} ms, alloc {:.0f} ms, scratch {:.0f} ms, encode {:.0f} ms ({} structures)",
                 sBlasSizesMs, sBlasAllocMs, sBlasScratchMs, sBlasEncodeMs, sBlasCount);
    STRELKA_INFO("Structures: BLAS {:.2f} GB ({} failed), TLAS {:.3f} GB, device max buffer {:.2f} GB",
                     asBytes / 1e9, nullAs,
                     mInstanceAccelerationStructure ? mInstanceAccelerationStructure->size() / 1e9 : 0.0,
                     mDevice->maxBufferLength() / 1e9);
    }
    mTlasInstanceCount = mEmittedInstances.size();

    chargePhase();
    STRELKA_DEBUG("Acceleration structures CPU: meshes {:.0f} ms, grouping {:.0f} ms, blas {:.0f} ms, "
                  "curves {:.0f} ms, lights {:.0f} ms, finish {:.0f} ms",
                  st.phaseMs[(size_t)Phase::Meshes], st.phaseMs[(size_t)Phase::Grouping],
                  st.phaseMs[(size_t)Phase::Blas], st.phaseMs[(size_t)Phase::Curves],
                  st.phaseMs[(size_t)Phase::Lights], st.phaseMs[(size_t)Phase::Finish]);
    delete mAsBuild;
    mAsBuild = nullptr;
    return finish(true);
}

MTL4::AccelerationStructureMotionTriangleGeometryDescriptor* MetalAccelStructure::createMotionGeometryDescriptor(
    const oka::Mesh& sceneMesh,
    MTL::Buffer* perPrimitiveBuffer,
    uint32_t triangleCount,
    std::vector<MTL::Buffer*>& motionVertexRangeBuffers)
{
    auto* geomDescriptor =
        MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()->init();

    // Metal 4 takes a GPU range containing the keyframe ranges, not an
    // Objective-C array. The descriptor keeps only that address, so the shared
    // buffer must live for as long as a skeletal BLAS can be rebuilt.
    MTL::Buffer* rangesBuffer =
        mDevice->newBuffer(2 * sizeof(MTL4::BufferRange), MTL::ResourceStorageModeShared);
    auto* ranges = static_cast<MTL4::BufferRange*>(rangesBuffer->contents());
    const size_t vertexOffset = sceneMesh.mVbOffset * sizeof(Scene::Vertex);
    ranges[0] = bufferRange(mGeometry->prevVertexBuffer(), vertexOffset);
    ranges[1] = bufferRange(mGeometry->vertexBuffer(), vertexOffset);
    geomDescriptor->setVertexBuffers(bufferRange(rangesBuffer));
    geomDescriptor->setVertexFormat(MTL::AttributeFormatFloat3);
    motionVertexRangeBuffers.push_back(rangesBuffer);
    mMetal4->addResident(rangesBuffer);
    geomDescriptor->setVertexStride(sizeof(Scene::Vertex));

    geomDescriptor->setIndexBuffer(bufferRange(mGeometry->indexBuffer(), sceneMesh.mIndex * sizeof(uint32_t)));
    geomDescriptor->setIndexType(MTL::IndexTypeUInt32);
    geomDescriptor->setTriangleCount(triangleCount);
    if (perPrimitiveBuffer)
    {
        geomDescriptor->setPrimitiveDataBuffer(bufferRange(perPrimitiveBuffer));
        geomDescriptor->setPrimitiveDataElementSize(sizeof(Triangle));
        geomDescriptor->setPrimitiveDataStride(sizeof(Triangle));
    }

    return geomDescriptor;
}

void MetalAccelStructure::ensureScratchBuffer(MTL::Buffer*& buffer, size_t requiredSize)
{
    if (requiredSize == 0)
        requiredSize = 1;
    if (buffer && buffer->length() >= requiredSize)
        return;
    if (buffer)
    {
        mMetal4->removeResident(buffer);
        buffer->release();
    }
    buffer = mDevice->newBuffer(requiredSize, MTL::ResourceStorageModePrivate);
    mMetal4->addResident(buffer);
}

void MetalAccelStructure::updateSkeletalBLAS()
{
    MTL4::CommandBuffer* commandBuffer = nullptr;
    MTL4::ComputeCommandEncoder* commandEncoder = nullptr;
    if (!beginImmediate(commandBuffer, commandEncoder))
    {
        return;
    }
    encodeSkeletalBLAS(commandEncoder);
    commandEncoder->endEncoding();
    mMetal4->submitAndWait(commandBuffer);
}

void MetalAccelStructure::encodeSkeletalBLAS(MTL4::ComputeCommandEncoder* commandEncoder)
{
    if (!commandEncoder)
    {
        return;
    }

    // Skinning may have been encoded earlier on this queue. Metal 4 does no
    // implicit hazard tracking, so make those vertex writes visible to the AS
    // builder before it reads them.
    commandEncoder->barrierAfterEncoderStages(MTL::StageDispatch | MTL::StageBlit,
                                              MTL::StageAccelerationStructure,
                                              MTL4::VisibilityOptionDevice);

    const size_t blasCount = mBlasList.size();
    size_t rebuiltThisFrame = 0;

    for (size_t mi = 0; mi < blasCount; ++mi)
    {
        Blas& blas = mBlasList[mi];
        if (!blas.mIsSkeletal || !blas.mDescriptor)
            continue;

        // The descriptor and its scratch requirements were computed once when the
        // structure was created: they only name buffers, offsets and triangle
        // counts, none of which change while the pose does. Rebuilding them per
        // frame meant a pile of Objective-C allocations plus an
        // accelerationStructureSizes() driver query describing geometry that
        // never changes shape, only contents.
        // The cap only exists so a scene with very many skeletal structures
        // cannot hitch on one frame; anything it skips is refitted and rebuilt
        // on the next.
        const bool rebuild = rebuiltThisFrame < kMaxBlasRebuildsPerFrame && mi >= mNextBlasRebuildIndex;
        if (rebuild)
        {
            // A rebuild needs build scratch, which is the larger of the two.
            ensureScratchBuffer(blas.mScratch, std::max(blas.mBuildScratchSize, blas.mRefitScratchSize));
            commandEncoder->buildAccelerationStructure(blas.mAs, blas.mDescriptor, bufferRange(blas.mScratch));
            ++rebuiltThisFrame;
            mNextBlasRebuildIndex = mi + 1;
        }
        else
        {
            ensureScratchBuffer(blas.mScratch, blas.mRefitScratchSize);
            commandEncoder->refitAccelerationStructure(blas.mAs, blas.mDescriptor, blas.mAs,
                                                        bufferRange(blas.mScratch));
        }
    }

    // Under the cap means everything eligible was covered, so start again from
    // the top rather than from one past the last skeletal structure — which may
    // be well short of blasCount when the scene also has static ones.
    if (rebuiltThisFrame < kMaxBlasRebuildsPerFrame)
    {
        mNextBlasRebuildIndex = 0;
    }

}

void MetalAccelStructure::updateInstanceTransforms()
{
    if (!mInstanceBuffer || !mPreviousInstanceBuffer)
    {
        writeInstanceTransforms(mInstanceBuffer);
        return;
    }

    // The old current buffer is exactly the transform state the preceding frame
    // rendered. Keep it untouched for motion-vector reconstruction and rewrite
    // the other fully initialized descriptor buffer as this frame's current.
    // More than one scene edit can be folded into one rendered frame; only the
    // first update swaps, or the second would turn an intermediate same-frame
    // state into "previous".
    if (!mInstanceTransformsChanged)
    {
        std::swap(mInstanceBuffer, mPreviousInstanceBuffer);
    }
    writeInstanceTransforms(mInstanceBuffer);
    mInstanceTransformsChanged = true;
    if (mTlasDescriptor)
    {
        mTlasDescriptor->setInstanceDescriptorBuffer(bufferRange(mInstanceBuffer));
    }
}

void MetalAccelStructure::writeInstanceTransforms(MTL::Buffer* buffer)
{
    if (!buffer)
    {
        return;
    }
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    auto* instanceDescriptors =
        static_cast<MTL::IndirectAccelerationStructureInstanceDescriptor*>(buffer->contents());

    for (size_t d = 0; d < mEmittedInstances.size(); ++d)
    {
        const Instance& curr = instances[mEmittedInstances[d].sceneInstanceId];
        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                instanceDescriptors[d].transformationMatrix.columns[column][row] = curr.transform[column][row];
            }
        }
    }
}

void MetalAccelStructure::rebuildTLAS()
{
    updateInstanceTransforms();
    MTL4::CommandBuffer* commandBuffer = nullptr;
    MTL4::ComputeCommandEncoder* commandEncoder = nullptr;
    if (!beginImmediate(commandBuffer, commandEncoder))
    {
        return;
    }
    encodeTLAS(commandEncoder);
    commandEncoder->endEncoding();
    mMetal4->submitAndWait(commandBuffer);
}

void MetalAccelStructure::encodeTLAS(MTL4::ComputeCommandEncoder* commandEncoder)
{
    if (!commandEncoder)
    {
        return;
    }

    ++mTlasEncodeCount;
    releaseRetiredInstanceStructures(kMaxFramesInFlight);

    MTL4::InstanceAccelerationStructureDescriptor* accelDescriptor = mTlasDescriptor;
    if (!accelDescriptor)
    {
        return;
    }

    // Only the instance transforms change while an animation plays — the set of
    // instances and the BLAS list are fixed. Refitting in place avoids allocating
    // (and freeing) a whole acceleration structure plus a scratch buffer on every
    // single frame, which is what the previous full rebuild did.
    const MTL::AccelerationStructureSizes sizes = accelerationStructureSizes(mDevice, accelDescriptor);
    const bool canRefit = mInstanceAccelerationStructure != nullptr &&
                          mTlasInstanceCount == mEmittedInstances.size() &&
                          mInstanceAccelerationStructure->size() >= sizes.accelerationStructureSize;

    if (canRefit)
    {
        ensureScratchBuffer(mTlasScratchBuffer, sizes.refitScratchBufferSize);
        // A TLAS reads every BLAS. This also orders it after skeletal builds
        // encoded immediately before this call in the same encoder.
        commandEncoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure,
                                                  MTL::StageAccelerationStructure,
                                                  MTL4::VisibilityOptionDevice);
        commandEncoder->refitAccelerationStructure(
            mInstanceAccelerationStructure, accelDescriptor, mInstanceAccelerationStructure,
            bufferRange(mTlasScratchBuffer));
    }
    else
    {
        if (mInstanceAccelerationStructure)
        {
            mRetiredInstanceStructures.emplace_back(mInstanceAccelerationStructure, mTlasEncodeCount);
            mInstanceAccelerationStructure = nullptr;
        }
        const MTL::AccelerationStructureSizes buildSizes = accelerationStructureSizes(mDevice, accelDescriptor);
        mInstanceAccelerationStructure = mDevice->newAccelerationStructure(buildSizes.accelerationStructureSize);
        ensureScratchBuffer(mTlasScratchBuffer, buildSizes.buildScratchBufferSize);
        mMetal4->addResident(mInstanceAccelerationStructure);
        commandEncoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure,
                                                  MTL::StageAccelerationStructure,
                                                  MTL4::VisibilityOptionDevice);
        commandEncoder->buildAccelerationStructure(mInstanceAccelerationStructure, accelDescriptor,
                                                   bufferRange(mTlasScratchBuffer));
        mTlasInstanceCount = mEmittedInstances.size();
    }

    // Traversal is a dispatch stage in the wavefront kernels.
    commandEncoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageDispatch,
                                              MTL4::VisibilityOptionDevice);
}



MetalAccelStructure::~MetalAccelStructure()
{
    release();
}

void MetalAccelStructure::init(MTL::Device* device,
                               Metal4Context* metal4,
                               MetalGeometry* geometry,
                               MetalMaterials* materials,
                               MetalTextures* textures)
{
    mDevice = device;
    mMetal4 = metal4;
    mGeometry = geometry;
    mMaterials = materials;
    mTextures = textures;
}

void MetalAccelStructure::addDescriptorResidency()
{
    if (!mMetal4)
    {
        return;
    }
    mMetal4->addResident(mGeometry->vertexBuffer());
    mMetal4->addResident(mGeometry->prevVertexBuffer());
    mMetal4->addResident(mGeometry->indexBuffer());
    mMetal4->addResident(mGeometry->curvePointBuffer());
    mMetal4->addResident(mGeometry->curveRadiusBuffer());
    mMetal4->addResident(mGeometry->curveSegmentBuffer());
    mMetal4->addResident(mInstanceBuffer);
    mMetal4->addResident(mPreviousInstanceBuffer);
    for (MetalGeometry::Mesh* mesh : mGeometry->meshes())
    {
        if (mesh)
        {
            mMetal4->addResident(mesh->mPerPrimitiveBuffer);
        }
    }
    for (const Blas& blas : mBlasList)
    {
        mMetal4->addResident(blas.mAs);
        mMetal4->addResident(blas.mScratch);
        for (MTL::Buffer* buffer : blas.mMotionVertexRangeBuffers)
        {
            mMetal4->addResident(buffer);
        }
    }
    mMetal4->addResident(mInstanceAccelerationStructure);
    mMetal4->addResident(mTlasScratchBuffer);
}

void MetalAccelStructure::releaseRetiredInstanceStructures(uint64_t age)
{
    auto expired = [&](const std::pair<MTL::AccelerationStructure*, uint64_t>& retired) {
        return retired.second + age <= mTlasEncodeCount;
    };
    for (const auto& retired : mRetiredInstanceStructures)
    {
        if (!expired(retired))
        {
            continue;
        }
        if (mMetal4)
        {
            mMetal4->removeResident(retired.first);
        }
        retired.first->release();
    }
    mRetiredInstanceStructures.erase(
        std::remove_if(mRetiredInstanceStructures.begin(), mRetiredInstanceStructures.end(), expired),
        mRetiredInstanceStructures.end());
}

std::vector<MTL::Buffer*> MetalAccelStructure::accelerationStructureAuxiliaryBuffers() const
{
    std::vector<MTL::Buffer*> buffers;
    if (mTlasScratchBuffer)
    {
        buffers.push_back(mTlasScratchBuffer);
    }
    for (const Blas& blas : mBlasList)
    {
        if (blas.mScratch)
        {
            buffers.push_back(blas.mScratch);
        }
        buffers.insert(buffers.end(), blas.mMotionVertexRangeBuffers.begin(),
                       blas.mMotionVertexRangeBuffers.end());
    }
    return buffers;
}

void MetalAccelStructure::release()
{
    auto safeRelease = [](auto*& p) {
        if (p)
        {
            p->release();
            p = nullptr;
        }
    };
    auto removeResident = [this](MTL::Allocation* allocation) {
        if (mMetal4)
        {
            mMetal4->removeResident(allocation);
        }
    };
    flushAccelerationStructureGroup();
    for (Blas& blas : mBlasList)
    {
        removeResident(blas.mScratch);
        safeRelease(blas.mScratch);
        safeRelease(blas.mDescriptor);
        for (MTL::Buffer* buffer : blas.mMotionVertexRangeBuffers)
        {
            removeResident(buffer);
            buffer->release();
        }
        blas.mMotionVertexRangeBuffers.clear();
    }
    mBlasList.clear();
    releaseRetiredInstanceStructures(0);
    for (auto*& as : mPrimitiveAccelerationStructures)
    {
        removeResident(as);
        safeRelease(as);
    }
    mPrimitiveAccelerationStructures.clear();
    removeResident(mInstanceAccelerationStructure);
    safeRelease(mInstanceAccelerationStructure);
    safeRelease(mTlasDescriptor);
    removeResident(mInstanceBuffer);
    safeRelease(mInstanceBuffer);
    removeResident(mPreviousInstanceBuffer);
    safeRelease(mPreviousInstanceBuffer);
    mInstanceTransformsChanged = false;
    removeResident(mTlasScratchBuffer);
    safeRelease(mTlasScratchBuffer);
    if (mMetal4)
    {
        mMetal4->commitResidency();
    }
    mEmittedInstances.clear();
    mTlasInstanceCount = 0;
    mOpaqueGeometryCount = 0;
    mCutoutGeometryCount = 0;
    mNextBlasRebuildIndex = 0;
    mNeedsPrimitiveData = false;
    mMotionBlasBuilt = false;
    mBuildMotionBlas = false;
    delete mAsBuild;
    mAsBuild = nullptr;
    mMetal4 = nullptr;
}

} // namespace metal
} // namespace oka
