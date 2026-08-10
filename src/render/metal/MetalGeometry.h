#pragma once

#include "ShaderTypes.h" // GeometryEntry

#include <Metal/Metal.hpp>
#include <strelka/scene/scene.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace oka
{
namespace metal
{

// Vertex stream domain: VB/IB, mesh records, curve uploads, GeometryEntry table.
// Does not build acceleration structures (see MetalAccelStructure).
class MetalGeometry
{
public:
    // Per scene mesh (one glTF primitive): just the data the skinning pass and
    // the geometry descriptors need. Acceleration structures live in Accel.
    struct Mesh
    {
        MTL::Buffer* mPerPrimitiveBuffer = nullptr;
        uint32_t mTriangleCount = 0;
        uint32_t mVbOffset = 0;
        bool mIsSkeletal = false;
    };

    /// Per scene curve set: where its segments start in mCurveSegmentBuffer and
    /// how many there are. Filled by buildCurveBuffers, read when its BLAS is
    /// built.
    struct CurveRange
    {
        uint32_t segmentStart = 0;
        uint32_t segmentCount = 0;
        uint32_t controlPointsPerSegment = 2;
        uint32_t segmentsPerStrand = 0;
    };

    MetalGeometry() = default;
    ~MetalGeometry();

    void init(MTL::Device* device);
    void release();

    // Upload VB/IB, optional prev-VB for skeletal scenes, and curve buffers.
    // Does not upload lights or frame uniforms (those are other domains).
    void buildBuffers(Scene* scene);

    void createMeshData(Scene* scene, size_t meshIndex, bool needsPrimitiveData);

    void clearMeshes();
    void clearGeometryEntries();
    void addGeometryEntry(const GeometryEntry& entry);
    void uploadGeometryEntryBuffer();

    MTL::Buffer* vertexBuffer() const
    {
        return mVertexBuffer;
    }
    MTL::Buffer* indexBuffer() const
    {
        return mIndexBuffer;
    }
    MTL::Buffer* prevVertexBuffer() const
    {
        return mPrevVertexBuffer;
    }
    bool ownsPrevVertexBuffer() const
    {
        return mOwnsPrevVertexBuffer;
    }
    // Skinning / motion-blur keyframe writes need a mutable prev VB pointer.
    MTL::Buffer*& prevVertexBufferRef()
    {
        return mPrevVertexBuffer;
    }
    bool& ownsPrevVertexBufferRef()
    {
        return mOwnsPrevVertexBuffer;
    }

    MTL::Buffer* geometryEntryBuffer() const
    {
        return mGeometryEntryBuffer;
    }
    std::vector<GeometryEntry>& geometryEntries()
    {
        return mGeometryEntries;
    }
    const std::vector<GeometryEntry>& geometryEntries() const
    {
        return mGeometryEntries;
    }

    std::vector<Mesh*>& meshes()
    {
        return mMetalMeshes;
    }
    const std::vector<Mesh*>& meshes() const
    {
        return mMetalMeshes;
    }

    MTL::Buffer* curvePointBuffer() const
    {
        return mCurvePointBuffer;
    }
    MTL::Buffer* curveRadiusBuffer() const
    {
        return mCurveRadiusBuffer;
    }
    MTL::Buffer* curveSegmentBuffer() const
    {
        return mCurveSegmentBuffer;
    }
    const std::vector<CurveRange>& curveRanges() const
    {
        return mCurveRanges;
    }
    bool hasCurves() const
    {
        return mSceneHasCurves;
    }

    std::pair<size_t, size_t>& hostGeometryBytes()
    {
        return mHostGeometryBytes;
    }
    const std::pair<size_t, size_t>& hostGeometryBytes() const
    {
        return mHostGeometryBytes;
    }

private:
    void buildCurveBuffers(Scene* scene);

    MTL::Device* mDevice = nullptr;

    MTL::Buffer* mVertexBuffer = nullptr;
    MTL::Buffer* mIndexBuffer = nullptr;
    MTL::Buffer* mPrevVertexBuffer = nullptr;
    bool mOwnsPrevVertexBuffer = false;

    std::vector<Mesh*> mMetalMeshes;
    std::vector<GeometryEntry> mGeometryEntries;
    MTL::Buffer* mGeometryEntryBuffer = nullptr;
    std::pair<size_t, size_t> mHostGeometryBytes{ 0, 0 };

    MTL::Buffer* mCurvePointBuffer = nullptr;
    MTL::Buffer* mCurveRadiusBuffer = nullptr;
    MTL::Buffer* mCurveSegmentBuffer = nullptr;
    std::vector<CurveRange> mCurveRanges;
    bool mSceneHasCurves = false;
};

} // namespace metal
} // namespace oka
