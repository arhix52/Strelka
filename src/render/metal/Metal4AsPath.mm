#include "AccelBuildPath.h"

#include "ShaderTypes.h"

#include <Metal/MTL4AccelerationStructure.hpp>
#include <Metal/MTL4ComputeCommandEncoder.hpp>
#include <strelka/scene/scene.h>

#include <env.h>
#include <log.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <vector>


namespace oka::metal
{
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

class Metal4AsPath final : public AccelBuildPath
{
public:
    Metal4AsPath(MTL::Device* device, Metal4Context* metal4) : mDevice(device), mMetal4(metal4)
    {
    }

    ~Metal4AsPath() override
    {
        flushBuildGroup();
    }

    bool inlineWithTracer() const override
    {
        return true;
    }
    MTL::AccelerationStructureSizes sizes(MTL::AccelerationStructureDescriptor* descriptor) override
    {
        // Metal 4 descriptors derive from Metal 3 ones so the device query is an
        // upcast. Valid only on hardware that supports Metal 4 ray tracing.
        return mDevice->accelerationStructureSizes(static_cast<MTL4::AccelerationStructureDescriptor*>(descriptor));
    }

    void addResident(MTL::Allocation* allocation) override
    {
        if (mMetal4)
        {
            mMetal4->addResident(allocation);
        }
    }

    void removeResident(MTL::Allocation* allocation) override
    {
        if (mMetal4)
        {
            mMetal4->removeResident(allocation);
        }
    }

    void commitResidency() override
    {
        if (mMetal4)
        {
            mMetal4->commitResidency();
        }
    }

    NS::Object* makeTriangleGeometry(MetalGeometry* geometry,
                                     const oka::Mesh& mesh,
                                     MTL::Buffer* perPrimitiveBuffer,
                                     uint32_t triangleCount) override
    {
        auto* geom = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
        geom->setVertexBuffer(bufferRange(geometry->vertexBuffer(), mesh.mVbOffset * sizeof(Scene::Vertex)));
        geom->setVertexFormat(MTL::AttributeFormatFloat3);
        geom->setVertexStride(sizeof(Scene::Vertex));
        geom->setIndexBuffer(bufferRange(geometry->indexBuffer(), mesh.mIndex * sizeof(uint32_t)));
        geom->setIndexType(MTL::IndexTypeUInt32);
        geom->setTriangleCount(triangleCount);
        if (perPrimitiveBuffer)
        {
            geom->setPrimitiveDataBuffer(bufferRange(perPrimitiveBuffer));
            geom->setPrimitiveDataElementSize(sizeof(Triangle));
            geom->setPrimitiveDataStride(sizeof(Triangle));
        }
        return geom;
    }

    NS::Object* makeMotionTriangleGeometry(MTL::Device* device,
                                           MetalGeometry* geometry,
                                           const oka::Mesh& mesh,
                                           MTL::Buffer* perPrimitiveBuffer,
                                           uint32_t triangleCount,
                                           std::vector<MTL::Buffer*>& motionVertexRangeBuffers) override
    {
        auto* geom = MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()->init();
        MTL::Buffer* rangesBuffer =
            device->newBuffer(2 * sizeof(MTL4::BufferRange), MTL::ResourceStorageModeShared);
        auto* ranges = static_cast<MTL4::BufferRange*>(rangesBuffer->contents());
        const size_t vertexOffset = mesh.mVbOffset * sizeof(Scene::Vertex);
        ranges[0] = bufferRange(geometry->prevVertexBuffer(), vertexOffset);
        ranges[1] = bufferRange(geometry->vertexBuffer(), vertexOffset);
        geom->setVertexBuffers(bufferRange(rangesBuffer));
        geom->setVertexFormat(MTL::AttributeFormatFloat3);
        motionVertexRangeBuffers.push_back(rangesBuffer);
        addResident(rangesBuffer);
        geom->setVertexStride(sizeof(Scene::Vertex));
        geom->setIndexBuffer(bufferRange(geometry->indexBuffer(), mesh.mIndex * sizeof(uint32_t)));
        geom->setIndexType(MTL::IndexTypeUInt32);
        geom->setTriangleCount(triangleCount);
        if (perPrimitiveBuffer)
        {
            geom->setPrimitiveDataBuffer(bufferRange(perPrimitiveBuffer));
            geom->setPrimitiveDataElementSize(sizeof(Triangle));
            geom->setPrimitiveDataStride(sizeof(Triangle));
        }
        return geom;
    }

    NS::Object* makeCurveGeometry(MetalGeometry* geometry,
                                  const oka::Curve& curve,
                                  const MetalGeometry::CurveRange& range,
                                  size_t controlPointCount) override
    {
        auto* geom = MTL4::AccelerationStructureCurveGeometryDescriptor::alloc()->init();
        geom->setControlPointBuffer(
            bufferRange(geometry->curvePointBuffer(), curve.mPointsStart * sizeof(glm::float3)));
        geom->setControlPointCount(controlPointCount);
        geom->setControlPointFormat(MTL::AttributeFormatFloat3);
        geom->setControlPointStride(sizeof(glm::float3));
        geom->setRadiusBuffer(
            bufferRange(geometry->curveRadiusBuffer(), curve.mPointsStart * sizeof(float)));
        geom->setRadiusFormat(MTL::AttributeFormatFloat);
        geom->setRadiusStride(sizeof(float));
        geom->setIndexBuffer(
            bufferRange(geometry->curveSegmentBuffer(), range.segmentStart * sizeof(uint32_t)));
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
        static_cast<MTL4::AccelerationStructureGeometryDescriptor*>(geometryDescriptor)->setOpaque(opaque);
    }

    MTL::AccelerationStructureDescriptor* makePrimitiveDescriptor(NS::Array* geometryDescriptors,
                                                                  bool skeletal,
                                                                  bool motionBlur,
                                                                  MTL::AccelerationStructureUsage usage) override
    {
        auto* prim = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
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
        auto* desc = MTL4::InstanceAccelerationStructureDescriptor::alloc()->init();
        desc->setInstanceCount(instanceCount);
        desc->setInstanceDescriptorBuffer(bufferRange(instanceBuffer));
        desc->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
        desc->setInstanceDescriptorStride(sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor));
        desc->setInstanceTransformationMatrixLayout(MTL::MatrixLayoutColumnMajor);
        desc->setUsage(usage);
        return desc;
    }

    void setInstanceDescriptorBuffer(MTL::AccelerationStructureDescriptor* descriptor,
                                     MTL::Buffer* instanceBuffer) override
    {
        static_cast<MTL4::InstanceAccelerationStructureDescriptor*>(descriptor)
            ->setInstanceDescriptorBuffer(bufferRange(instanceBuffer));
    }

    MTL::AccelerationStructure* createCompacted(MTL::AccelerationStructureDescriptor* descriptor) override
    {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        auto* m4desc = static_cast<MTL4::AccelerationStructureDescriptor*>(descriptor);
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
        MTL::Buffer* compactedSizeBuffer = mDevice->newBuffer(sizeof(uint64_t), MTL::ResourceStorageModeShared);
        addResident(accelerationStructure);
        addResident(scratchBuffer);
        addResident(compactedSizeBuffer);
        commitResidency();

        MTL4::CommandBuffer* commandBuffer = mMetal4->beginImmediate();
        MTL4::ComputeCommandEncoder* commandEncoder = commandBuffer ? commandBuffer->computeCommandEncoder() : nullptr;
        if (!commandEncoder)
        {
            removeResident(accelerationStructure);
            removeResident(scratchBuffer);
            removeResident(compactedSizeBuffer);
            accelerationStructure->release();
            scratchBuffer->release();
            compactedSizeBuffer->release();
            pool->release();
            return nullptr;
        }
        commandEncoder->buildAccelerationStructure(accelerationStructure, m4desc, bufferRange(scratchBuffer));
        commandEncoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageAccelerationStructure,
                                                  MTL4::VisibilityOptionDevice);
        commandEncoder->writeCompactedAccelerationStructureSize(accelerationStructure,
                                                                bufferRange(compactedSizeBuffer));
        commandEncoder->endEncoding();
        mMetal4->submitAndWait(commandBuffer);

        const uint64_t compactedSize = *(uint64_t*)compactedSizeBuffer->contents();
        MTL::AccelerationStructure* compacted = mDevice->newAccelerationStructure(compactedSize);
        if (!compacted)
        {
            STRELKA_ERROR("Compacted acceleration structure allocation failed: {:.2f} GB requested",
                          compactedSize / 1e9);
            removeResident(accelerationStructure);
            removeResident(scratchBuffer);
            removeResident(compactedSizeBuffer);
            accelerationStructure->release();
            scratchBuffer->release();
            compactedSizeBuffer->release();
            pool->release();
            return nullptr;
        }
        addResident(compacted);
        commandBuffer = mMetal4->beginImmediate();
        commandEncoder = commandBuffer ? commandBuffer->computeCommandEncoder() : nullptr;
        if (!commandEncoder)
        {
            removeResident(compacted);
            removeResident(accelerationStructure);
            removeResident(scratchBuffer);
            removeResident(compactedSizeBuffer);
            compacted->release();
            accelerationStructure->release();
            scratchBuffer->release();
            compactedSizeBuffer->release();
            pool->release();
            return nullptr;
        }
        commandEncoder->copyAndCompactAccelerationStructure(accelerationStructure, compacted);
        commandEncoder->endEncoding();
        mMetal4->submitAndWait(commandBuffer);
        removeResident(accelerationStructure);
        removeResident(scratchBuffer);
        removeResident(compactedSizeBuffer);
        commitResidency();
        accelerationStructure->release();
        scratchBuffer->release();
        compactedSizeBuffer->release();
        pool->release();
        return compacted;
    }

    MTL::AccelerationStructure* createNoCompact(MTL::AccelerationStructureDescriptor* descriptor) override
    {
        auto* m4desc = static_cast<MTL4::AccelerationStructureDescriptor*>(descriptor);
        const auto tSizes = std::chrono::steady_clock::now();
        const MTL::AccelerationStructureSizes accelSizes = sizes(descriptor);
        const auto tAlloc = std::chrono::steady_clock::now();
        MTL::AccelerationStructure* accelerationStructure =
            mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
        const auto tScratch = std::chrono::steady_clock::now();
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
        const auto tEncode = std::chrono::steady_clock::now();
        addResident(accelerationStructure);
        addResident(scratchBuffer);

        static const uint32_t kGroupSize = std::max(1u, envUint("STRELKA_AS_GROUP", 1));
        if (!mAsGroupCommandBuffer)
        {
            commitResidency();
            mAsGroupCommandBuffer = mMetal4->beginImmediate();
            mAsGroupEncoder = mAsGroupCommandBuffer ? mAsGroupCommandBuffer->computeCommandEncoder() : nullptr;
            if (!mAsGroupEncoder)
            {
                removeResident(accelerationStructure);
                removeResident(scratchBuffer);
                accelerationStructure->release();
                scratchBuffer->release();
                return nullptr;
            }
        }
        mAsGroupEncoder->buildAccelerationStructure(accelerationStructure, m4desc, bufferRange(scratchBuffer));
        mAsGroupScratch.push_back(scratchBuffer);
        mAsGroupDescriptors.push_back(m4desc->retain());
        if (++mAsGroupPending >= kGroupSize)
        {
            flushBuildGroup();
        }
        (void)tSizes;
        (void)tAlloc;
        (void)tScratch;
        (void)tEncode;
        return accelerationStructure;
    }

    void flushBuildGroup() override
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
            removeResident(b);
            b->release();
        }
        mAsGroupScratch.clear();
        for (MTL4::AccelerationStructureDescriptor* descriptor : mAsGroupDescriptors)
        {
            descriptor->release();
        }
        mAsGroupDescriptors.clear();
        mAsGroupPending = 0;
    }

    void drain() override
    {
        flushBuildGroup();
        MTL4::CommandBuffer* commandBuffer = mMetal4->beginImmediate();
        MTL4::ComputeCommandEncoder* encoder = commandBuffer ? commandBuffer->computeCommandEncoder() : nullptr;
        if (encoder)
        {
            encoder->endEncoding();
            mMetal4->submitAndWait(commandBuffer);
        }
    }

    void beginInline(MTL4::ComputeCommandEncoder* encoder) override
    {
        mInlineEncoder = encoder;
    }

    void barrierAfterSkinning() override
    {
        if (!mInlineEncoder)
        {
            return;
        }
        mInlineEncoder->barrierAfterEncoderStages(MTL::StageDispatch | MTL::StageBlit,
                                                  MTL::StageAccelerationStructure,
                                                  MTL4::VisibilityOptionDevice);
    }

    void build(MTL::AccelerationStructure* as,
               MTL::AccelerationStructureDescriptor* descriptor,
               MTL::Buffer* scratch) override
    {
        assert(mInlineEncoder);
        mInlineEncoder->buildAccelerationStructure(
            as, static_cast<MTL4::AccelerationStructureDescriptor*>(descriptor), bufferRange(scratch));
    }

    void refit(MTL::AccelerationStructure* as,
               MTL::AccelerationStructureDescriptor* descriptor,
               MTL::Buffer* scratch) override
    {
        assert(mInlineEncoder);
        mInlineEncoder->refitAccelerationStructure(
            as, static_cast<MTL4::AccelerationStructureDescriptor*>(descriptor), as, bufferRange(scratch));
    }

    void barrierBeforeTlas() override
    {
        if (!mInlineEncoder)
        {
            return;
        }
        mInlineEncoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure,
                                                  MTL::StageAccelerationStructure,
                                                  MTL4::VisibilityOptionDevice);
    }

    void barrierAfterTlasBeforeDispatch() override
    {
        if (!mInlineEncoder)
        {
            return;
        }
        mInlineEncoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageDispatch,
                                                  MTL4::VisibilityOptionDevice);
    }

    void endInline() override
    {
        mInlineEncoder = nullptr;
    }

    void submitSide(const AsFrameUpdate&, const std::function<void()>&) override
    {
        // Inline path: frame updates are encoded by the caller via beginInline.
    }

private:
    MTL::Device* mDevice = nullptr;
    Metal4Context* mMetal4 = nullptr;
    MTL4::ComputeCommandEncoder* mInlineEncoder = nullptr;

    MTL4::CommandBuffer* mAsGroupCommandBuffer = nullptr;
    MTL4::ComputeCommandEncoder* mAsGroupEncoder = nullptr;
    std::vector<MTL::Buffer*> mAsGroupScratch;
    std::vector<MTL4::AccelerationStructureDescriptor*> mAsGroupDescriptors;
    uint32_t mAsGroupPending = 0;
};

} // namespace

std::unique_ptr<AccelBuildPath> createMetal4AsPath(MTL::Device* device, Metal4Context* metal4)
{
    return std::make_unique<Metal4AsPath>(device, metal4);
}

} // namespace oka::metal

