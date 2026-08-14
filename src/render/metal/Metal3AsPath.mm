#include "AccelBuildPath.h"

#include "ShaderTypes.h"

#include <strelka/scene/scene.h>

#include <env.h>
#include <log.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>

namespace oka
{
namespace metal
{
namespace
{

class Metal3AsPath final : public AccelBuildPath
{
public:
    Metal3AsPath(MTL::Device* device, MTL::CommandQueue* queue) : mDevice(device), mCommandQueue(queue)
    {
        mBuildEvent = device->newSharedEvent();
    }

    ~Metal3AsPath() override
    {
        flushBuildGroup();
        if (mBuildEvent)
        {
            mBuildEvent->release();
            mBuildEvent = nullptr;
        }
    }

    bool inlineWithTracer() const override
    {
        return false;
    }
    MTL::SharedEvent* readyEvent() const override
    {
        return mBuildEvent;
    }

    uint64_t readyValue() const override
    {
        return mBuildValue;
    }

    MTL::AccelerationStructureSizes sizes(MTL::AccelerationStructureDescriptor* descriptor) override
    {
        return mDevice->accelerationStructureSizes(descriptor);
    }

    void addResident(MTL::Allocation*) override
    {
        // Metal 3 AS builds do not use a residency set. The Metal 4 tracer still
        // declares the finished structures resident from MetalAccelStructure.
    }

    void removeResident(MTL::Allocation*) override
    {
    }

    void commitResidency() override
    {
    }

    NS::Object* makeTriangleGeometry(MetalGeometry* geometry,
                                     const oka::Mesh& mesh,
                                     MTL::Buffer* perPrimitiveBuffer,
                                     uint32_t triangleCount) override
    {
        auto* geom = MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
        geom->setVertexBuffer(geometry->vertexBuffer());
        geom->setVertexBufferOffset(mesh.mVbOffset * sizeof(Scene::Vertex));
        geom->setVertexStride(sizeof(Scene::Vertex));
        geom->setIndexBuffer(geometry->indexBuffer());
        geom->setIndexBufferOffset(mesh.mIndex * sizeof(uint32_t));
        geom->setIndexType(MTL::IndexTypeUInt32);
        geom->setTriangleCount(triangleCount);
        if (perPrimitiveBuffer)
        {
            geom->setPrimitiveDataBuffer(perPrimitiveBuffer);
            geom->setPrimitiveDataBufferOffset(0);
            geom->setPrimitiveDataElementSize(sizeof(Triangle));
            geom->setPrimitiveDataStride(sizeof(Triangle));
        }
        return geom;
    }

    NS::Object* makeMotionTriangleGeometry(MTL::Device*,
                                           MetalGeometry* geometry,
                                           const oka::Mesh& mesh,
                                           MTL::Buffer* perPrimitiveBuffer,
                                           uint32_t triangleCount,
                                           std::vector<MTL::Buffer*>& /*motionVertexRangeBuffers*/) override
    {
        auto* geom = MTL::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()->init();
        MTL::MotionKeyframeData* kf0 = MTL::MotionKeyframeData::alloc()->init();
        kf0->setBuffer(geometry->prevVertexBuffer());
        kf0->setOffset(mesh.mVbOffset * sizeof(Scene::Vertex));
        MTL::MotionKeyframeData* kf1 = MTL::MotionKeyframeData::alloc()->init();
        kf1->setBuffer(geometry->vertexBuffer());
        kf1->setOffset(mesh.mVbOffset * sizeof(Scene::Vertex));
        const NS::Object* keyframes[] = { kf0, kf1 };
        geom->setVertexBuffers(NS::Array::array(keyframes, 2UL));
        geom->setVertexStride(sizeof(Scene::Vertex));
        geom->setIndexBuffer(geometry->indexBuffer());
        geom->setIndexBufferOffset(mesh.mIndex * sizeof(uint32_t));
        geom->setIndexType(MTL::IndexTypeUInt32);
        geom->setTriangleCount(triangleCount);
        if (perPrimitiveBuffer)
        {
            geom->setPrimitiveDataBuffer(perPrimitiveBuffer);
            geom->setPrimitiveDataBufferOffset(0);
            geom->setPrimitiveDataElementSize(sizeof(Triangle));
            geom->setPrimitiveDataStride(sizeof(Triangle));
        }
        kf0->release();
        kf1->release();
        return geom;
    }

    NS::Object* makeCurveGeometry(MetalGeometry* geometry,
                                  const oka::Curve& curve,
                                  const MetalGeometry::CurveRange& range,
                                  size_t controlPointCount) override
    {
        auto* geom = MTL::AccelerationStructureCurveGeometryDescriptor::alloc()->init();
        geom->setControlPointBuffer(geometry->curvePointBuffer());
        geom->setControlPointBufferOffset(curve.mPointsStart * sizeof(glm::float3));
        geom->setControlPointCount(controlPointCount);
        geom->setControlPointFormat(MTL::AttributeFormatFloat3);
        geom->setControlPointStride(sizeof(glm::float3));
        geom->setRadiusBuffer(geometry->curveRadiusBuffer());
        // MetalGeometry expands optional widths into point-aligned storage.
        geom->setRadiusBufferOffset(curve.mPointsStart * sizeof(float));
        geom->setRadiusFormat(MTL::AttributeFormatFloat);
        geom->setRadiusStride(sizeof(float));
        geom->setIndexBuffer(geometry->curveSegmentBuffer());
        geom->setIndexBufferOffset(range.segmentStart * sizeof(uint32_t));
        geom->setIndexType(MTL::IndexTypeUInt32);
        geom->setSegmentCount(range.segmentCount);
        geom->setSegmentControlPointCount(range.controlPointsPerSegment);
        geom->setCurveType(MTL::CurveTypeRound);
        geom->setCurveBasis(curve.mType == oka::Curve::Type::eLinear ? MTL::CurveBasisLinear
                                                                     : MTL::CurveBasisBSpline);
        geom->setCurveEndCaps(curve.mType == oka::Curve::Type::eLinear ? MTL::CurveEndCapsSphere
                                                                       : MTL::CurveEndCapsDisk);
        geom->setOpaque(true);
        return geom;
    }

    void setGeometryOpaque(NS::Object* geometryDescriptor, bool opaque) override
    {
        static_cast<MTL::AccelerationStructureGeometryDescriptor*>(geometryDescriptor)->setOpaque(opaque);
    }

    MTL::AccelerationStructureDescriptor* makePrimitiveDescriptor(NS::Array* geometryDescriptors,
                                                                  bool skeletal,
                                                                  bool motionBlur,
                                                                  MTL::AccelerationStructureUsage usage) override
    {
        auto* prim = MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
        prim->setGeometryDescriptors(geometryDescriptors);
        if (skeletal && motionBlur)
        {
            prim->setMotionKeyframeCount(2);
            prim->setMotionStartTime(0.0f);
            prim->setMotionEndTime(1.0f);
            prim->setMotionStartBorderMode(MTL::MotionBorderModeClamp);
            prim->setMotionEndBorderMode(MTL::MotionBorderModeClamp);
        }
        prim->setUsage(usage);
        return prim;
    }

    MTL::AccelerationStructureDescriptor* makeInstanceDescriptor(MTL::Buffer* instanceBuffer,
                                                                 size_t instanceCount,
                                                                 MTL::AccelerationStructureUsage usage) override
    {
        // Indirect descriptors match the Metal 4 tracer / wavefront shader. The
        // instance buffer already stores MTLIndirectAccelerationStructureInstanceDescriptor.
        auto* desc = MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
        desc->setInstanceCount(instanceCount);
        desc->setInstanceDescriptorBuffer(instanceBuffer);
        desc->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
        desc->setInstanceDescriptorStride(sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor));
        desc->setUsage(usage);
        return desc;
    }

    void setInstanceDescriptorBuffer(MTL::AccelerationStructureDescriptor* descriptor,
                                     MTL::Buffer* instanceBuffer) override
    {
        static_cast<MTL::InstanceAccelerationStructureDescriptor*>(descriptor)
            ->setInstanceDescriptorBuffer(instanceBuffer);
    }

    MTL::AccelerationStructure* createCompacted(MTL::AccelerationStructureDescriptor* descriptor) override
    {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        const MTL::AccelerationStructureSizes accelSizes = sizes(descriptor);
        MTL::AccelerationStructure* accelerationStructure =
            mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
        if (!accelerationStructure)
        {
            STRELKA_ERROR("Acceleration structure allocation failed: {:.2f} GB requested, "
                          "{:.2f} GB max buffer. The scene does not fit -- lower "
                          "render/texture/maxDimension or reduce geometry.",
                          accelSizes.accelerationStructureSize / 1e9, mDevice->maxBufferLength() / 1e9);
            pool->release();
            return nullptr;
        }
        MTL::Buffer* scratchBuffer =
            mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
        MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
        MTL::AccelerationStructureCommandEncoder* commandEncoder =
            commandBuffer->accelerationStructureCommandEncoder();
        MTL::Buffer* compactedSizeBuffer = mDevice->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        commandEncoder->buildAccelerationStructure(accelerationStructure, descriptor, scratchBuffer, 0UL);
        commandEncoder->writeCompactedAccelerationStructureSize(accelerationStructure, compactedSizeBuffer, 0UL);
        commandEncoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        if (commandBuffer->status() == MTL::CommandBufferStatusError)
        {
            NS::Error* err = commandBuffer->error();
            STRELKA_ERROR("Acceleration structure build failed on the GPU: {}",
                          err && err->localizedDescription()
                              ? err->localizedDescription()->utf8String()
                              : "unknown error (most likely out of device memory)");
        }
        const uint32_t compactedSize = *(uint32_t*)compactedSizeBuffer->contents();
        MTL::AccelerationStructure* compacted = mDevice->newAccelerationStructure(compactedSize);
        if (!compacted)
        {
            STRELKA_ERROR("Compacted acceleration structure allocation failed: {:.2f} GB requested",
                          compactedSize / 1e9);
            accelerationStructure->release();
            scratchBuffer->release();
            compactedSizeBuffer->release();
            pool->release();
            return nullptr;
        }
        commandBuffer = mCommandQueue->commandBuffer();
        commandEncoder = commandBuffer->accelerationStructureCommandEncoder();
        commandEncoder->copyAndCompactAccelerationStructure(accelerationStructure, compacted);
        commandEncoder->endEncoding();
        signalBuild(commandBuffer);
        commandBuffer->commit();
        accelerationStructure->release();
        scratchBuffer->release();
        compactedSizeBuffer->release();
        pool->release();
        return compacted->retain();
    }

    MTL::AccelerationStructure* createNoCompact(MTL::AccelerationStructureDescriptor* descriptor) override
    {
        const MTL::AccelerationStructureSizes accelSizes = sizes(descriptor);
        MTL::AccelerationStructure* accelerationStructure =
            mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
        if (!accelerationStructure)
        {
            STRELKA_ERROR("Acceleration structure allocation failed: {:.2f} GB requested, "
                          "{:.2f} GB max buffer. The scene does not fit -- lower "
                          "render/texture/maxDimension or reduce geometry.",
                          accelSizes.accelerationStructureSize / 1e9, mDevice->maxBufferLength() / 1e9);
            return nullptr;
        }
        MTL::Buffer* scratchBuffer =
            mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
        static const uint32_t kGroupSize = std::max(1u, envUint("STRELKA_AS_GROUP", 1));
        if (!mAsGroupCommandBuffer)
        {
            mAsGroupCommandBuffer = mCommandQueue->commandBuffer()->retain();
            mAsGroupEncoder = mAsGroupCommandBuffer->accelerationStructureCommandEncoder()->retain();
        }
        mAsGroupEncoder->buildAccelerationStructure(accelerationStructure, descriptor, scratchBuffer, 0UL);
        mAsGroupScratch.push_back(scratchBuffer);
        if (++mAsGroupPending >= kGroupSize)
        {
            flushBuildGroup();
        }
        return accelerationStructure;
    }

    void flushBuildGroup() override
    {
        if (!mAsGroupCommandBuffer)
        {
            return;
        }
        mAsGroupEncoder->endEncoding();
        signalBuild(mAsGroupCommandBuffer);
        mAsGroupCommandBuffer->commit();
        mAsGroupEncoder->release();
        mAsGroupCommandBuffer->release();
        mAsGroupEncoder = nullptr;
        mAsGroupCommandBuffer = nullptr;
        for (MTL::Buffer* b : mAsGroupScratch)
        {
            b->release();
        }
        mAsGroupScratch.clear();
        mAsGroupPending = 0;
    }

    void drain() override
    {
        flushBuildGroup();
        MTL::CommandBuffer* drain = mCommandQueue->commandBuffer();
        drain->commit();
        drain->waitUntilCompleted();
    }

    void beginInline(MTL4::ComputeCommandEncoder*) override
    {
    }

    void barrierAfterSkinning() override
    {
    }

    void build(MTL::AccelerationStructure* as,
               MTL::AccelerationStructureDescriptor* descriptor,
               MTL::Buffer* scratch) override
    {
        assert(mSideEncoder);
        mSideEncoder->buildAccelerationStructure(as, descriptor, scratch, 0UL);
    }

    void refit(MTL::AccelerationStructure* as,
               MTL::AccelerationStructureDescriptor* descriptor,
               MTL::Buffer* scratch) override
    {
        assert(mSideEncoder);
        mSideEncoder->refitAccelerationStructure(as, descriptor, as, scratch, 0UL);
    }

    void barrierBeforeTlas() override
    {
    }

    void barrierAfterTlasBeforeDispatch() override
    {
    }

    void endInline() override
    {
    }

    void submitSide(const AsFrameUpdate& update, const std::function<void()>& body) override
    {
        if (!update.skeletal && !update.tlas)
        {
            return;
        }
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
        if (update.afterSkinning && update.afterSkinningValue != 0)
        {
            commandBuffer->encodeWait(update.afterSkinning, update.afterSkinningValue);
        }
        mSideEncoder = commandBuffer->accelerationStructureCommandEncoder();
        body();
        mSideEncoder->endEncoding();
        mSideEncoder = nullptr;
        signalBuild(commandBuffer);
        commandBuffer->commit();
        pool->release();
    }

private:
    void signalBuild(MTL::CommandBuffer* commandBuffer)
    {
        if (!mBuildEvent)
        {
            mBuildEvent = mDevice->newSharedEvent();
        }
        commandBuffer->encodeSignalEvent(mBuildEvent, ++mBuildValue);
    }

    MTL::Device* mDevice = nullptr;
    MTL::CommandQueue* mCommandQueue = nullptr;
    MTL::SharedEvent* mBuildEvent = nullptr;
    uint64_t mBuildValue = 0;

    MTL::CommandBuffer* mAsGroupCommandBuffer = nullptr;
    MTL::AccelerationStructureCommandEncoder* mAsGroupEncoder = nullptr;
    std::vector<MTL::Buffer*> mAsGroupScratch;
    uint32_t mAsGroupPending = 0;

    MTL::AccelerationStructureCommandEncoder* mSideEncoder = nullptr;
};

} // namespace

std::unique_ptr<AccelBuildPath> createMetal3AsPath(MTL::Device* device, MTL::CommandQueue* queue)
{
    return std::make_unique<Metal3AsPath>(device, queue);
}

} // namespace metal
} // namespace oka
