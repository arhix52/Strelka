#pragma once

#include "Metal4Context.h"
#include "MetalGeometry.h"

#include <Metal/Metal.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

namespace oka
{
struct Mesh;
struct Curve;

namespace metal
{

/// CPU flags for a per-frame acceleration-structure update. Sync fields are used
/// only by the side-queue path (Metal 3 AS builds while skinning/tracing stay on
/// Metal 4): wait for skinning, then signal readyEvent() for the tracer.
struct AsFrameUpdate
{
    bool skeletal = false;
    bool tlas = false;
    MTL::SharedEvent* afterSkinning = nullptr;
    uint64_t afterSkinningValue = 0;
};

/// Submission + descriptor strategy for BLAS/TLAS. MetalAccelStructure owns
/// grouping and instance buffers; this object owns Metal 3 vs Metal 4 API shape.
class AccelBuildPath
{
public:
    virtual ~AccelBuildPath() = default;

    /// True when AS updates share the Metal 4 compute encoder with skinning and
    /// the integrator. False means submitSide() + readyEvent() cross-queue sync.
    virtual bool inlineWithTracer() const = 0;

    virtual MTL::SharedEvent* readyEvent() const
    {
        return nullptr;
    }
    virtual uint64_t readyValue() const
    {
        return 0;
    }
    virtual MTL::AccelerationStructureSizes sizes(MTL::AccelerationStructureDescriptor* descriptor) = 0;

    virtual void addResident(MTL::Allocation* allocation) = 0;
    virtual void removeResident(MTL::Allocation* allocation) = 0;
    virtual void commitResidency() = 0;

    // --- Descriptor factories (return retained objects; caller releases) ---

    virtual NS::Object* makeTriangleGeometry(MetalGeometry* geometry, const oka::Mesh& mesh, uint32_t triangleCount) = 0;

    virtual NS::Object* makeBoundingBoxGeometry(MTL::Buffer* bounds,
                                                size_t offset,
                                                uint32_t intersectionFunctionOffset) = 0;

    virtual NS::Object* makeMotionTriangleGeometry(MTL::Device* device,
                                                   MetalGeometry* geometry,
                                                   const oka::Mesh& mesh,
                                                   uint32_t triangleCount,
                                                   std::vector<MTL::Buffer*>& motionVertexRangeBuffers) = 0;

    virtual NS::Object* makeCurveGeometry(MetalGeometry* geometry,
                                          const oka::Curve& curve,
                                          const MetalGeometry::CurveRange& range,
                                          size_t controlPointCount) = 0;

    virtual void setGeometryOpaque(NS::Object* geometryDescriptor, bool opaque) = 0;

    /// Retained primitive descriptor. `geometryDescriptors` are not retained by
    /// the caller beyond this call; the path retains what the descriptor needs.
    virtual MTL::AccelerationStructureDescriptor* makePrimitiveDescriptor(NS::Array* geometryDescriptors,
                                                                          bool skeletal,
                                                                          bool motionBlur,
                                                                          MTL::AccelerationStructureUsage usage) = 0;

    virtual MTL::AccelerationStructureDescriptor* makeInstanceDescriptor(MTL::Buffer* instanceBuffer,
                                                                         size_t instanceCount,
                                                                         MTL::AccelerationStructureUsage usage) = 0;

    virtual void setInstanceDescriptorBuffer(MTL::AccelerationStructureDescriptor* descriptor,
                                             MTL::Buffer* instanceBuffer) = 0;

    // --- Load-time builds ---

    virtual MTL::AccelerationStructure* createCompacted(MTL::AccelerationStructureDescriptor* descriptor) = 0;
    virtual MTL::AccelerationStructure* createNoCompact(MTL::AccelerationStructureDescriptor* descriptor) = 0;
    virtual void flushBuildGroup() = 0;

    /// Drain in-flight grouped builds (e.g. before tearing structures down).
    virtual void drain() = 0;

    // --- Per-frame encode ---

    /// Metal 4 path: encode into the caller's compute encoder. No-op on Metal 3.
    virtual void beginInline(MTL4::ComputeCommandEncoder* encoder) = 0;
    virtual void barrierAfterSkinning() = 0;
    virtual void build(MTL::AccelerationStructure* as,
                       MTL::AccelerationStructureDescriptor* descriptor,
                       MTL::Buffer* scratch) = 0;
    virtual void refit(MTL::AccelerationStructure* as,
                       MTL::AccelerationStructureDescriptor* descriptor,
                       MTL::Buffer* scratch) = 0;
    virtual void barrierBeforeTlas() = 0;
    virtual void barrierAfterTlasBeforeDispatch() = 0;
    virtual void endInline() = 0;

    /// Metal 3 path: wait afterSkinning, encode body, signal readyEvent. No-op
    /// when inlineWithTracer(). `body` calls build/refit on this path.
    virtual void submitSide(const AsFrameUpdate& update, const std::function<void()>& body) = 0;
};

std::unique_ptr<AccelBuildPath> createMetal4AsPath(MTL::Device* device, Metal4Context* metal4);
std::unique_ptr<AccelBuildPath> createMetal3AsPath(MTL::Device* device, MTL::CommandQueue* queue);

/// Apple9+ with a valid Metal 4 queue: AS builds share the tracer queue.
inline bool supportsMetal4RayTracingAS(MTL::Device* device, Metal4Context* metal4)
{
    return device && metal4 && metal4->isValid() && device->supportsFamily(MTL::GPUFamilyApple9);
}

} // namespace metal
} // namespace oka
