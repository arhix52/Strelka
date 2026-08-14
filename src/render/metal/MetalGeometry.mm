#include "MetalGeometry.h"

#include <log.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace oka
{
namespace metal
{

MetalGeometry::~MetalGeometry()
{
    release();
}

void MetalGeometry::init(MTL::Device* device)
{
    mDevice = device;
}

void MetalGeometry::clearMeshes()
{
    for (Mesh* mesh : mMetalMeshes)
    {
        if (mesh)
        {
            if (mesh->mPerPrimitiveBuffer)
                mesh->mPerPrimitiveBuffer->release();
            delete mesh;
        }
    }
    mMetalMeshes.clear();
}

void MetalGeometry::clearGeometryEntries()
{
    mGeometryEntries.clear();
    if (mGeometryEntryBuffer)
    {
        mGeometryEntryBuffer->release();
        mGeometryEntryBuffer = nullptr;
    }
}

void MetalGeometry::addGeometryEntry(const GeometryEntry& entry)
{
    mGeometryEntries.push_back(entry);
}

void MetalGeometry::uploadGeometryEntryBuffer()
{
    if (mGeometryEntryBuffer)
    {
        mGeometryEntryBuffer->release();
        mGeometryEntryBuffer = nullptr;
    }
    if (mGeometryEntries.empty())
        return;
    mGeometryEntryBuffer = mDevice->newBuffer(mGeometryEntries.size() * sizeof(GeometryEntry),
                                              MTL::ResourceStorageModeShared);
    memcpy(mGeometryEntryBuffer->contents(), mGeometryEntries.data(),
           mGeometryEntries.size() * sizeof(GeometryEntry));
}

void MetalGeometry::release()
{
    clearMeshes();
    clearGeometryEntries();

    auto safeRelease = [](MTL::Buffer*& p) {
        if (p)
        {
            p->release();
            p = nullptr;
        }
    };

    safeRelease(mIndexBuffer);
    if (mOwnsPrevVertexBuffer)
        safeRelease(mPrevVertexBuffer);
    else
        mPrevVertexBuffer = nullptr;
    safeRelease(mVertexBuffer);
    mOwnsPrevVertexBuffer = false;

    safeRelease(mCurvePointBuffer);
    safeRelease(mCurveRadiusBuffer);
    safeRelease(mCurveSegmentBuffer);
    mCurveRanges.clear();
    mSceneHasCurves = false;
    mHostGeometryBytes = { 0, 0 };
}

void MetalGeometry::buildBuffers(Scene* scene)
{
    if (scene->hostGeometryReleased())
    {
        // Rebuilding from arrays that were handed back would quietly replace the
        // buffers with empty ones and render nothing.
        STRELKA_ERROR("buildBuffers() after releaseHostGeometry(); reload the scene instead");
        return;
    }
    const std::vector<Scene::Vertex>& vertices = scene->getVertices();
    const std::vector<uint32_t>& indices = scene->getIndices();

    const size_t vertexDataSize = sizeof(Scene::Vertex) * vertices.size();
    const size_t indexDataSize = sizeof(uint32_t) * indices.size();

    // Shared rather than managed, and this is not a preference.
    //
    // Metal 4 removes the managed storage mode outright -- it exists to keep a
    // separate CPU and GPU copy in step on discrete memory, which is not the
    // architecture Metal 4 targets -- and it removes didModifyRange with it. A
    // managed buffer bound into an argument table by GPU address therefore has
    // no defined behaviour, and the GPU writing one (which is exactly what the
    // skinning kernel does to this buffer) has nowhere to publish the result.
    MTL::Buffer* pVertexBuffer = nullptr;
    if (vertexDataSize > 0)
    {
        pVertexBuffer = mDevice->newBuffer(vertexDataSize, MTL::ResourceStorageModeShared);
        memcpy(pVertexBuffer->contents(), vertices.data(), vertexDataSize);
    }
    MTL::Buffer* pIndexBuffer = nullptr;
    if (indexDataSize > 0)
    {
        pIndexBuffer = mDevice->newBuffer(indexDataSize, MTL::ResourceStorageModeShared);
        memcpy(pIndexBuffer->contents(), indices.data(), indexDataSize);
    }

    // Drop previous geometry buffers before adopting the new ones.
    {
        auto safeRelease = [](MTL::Buffer*& p) {
            if (p)
            {
                p->release();
                p = nullptr;
            }
        };
        safeRelease(mIndexBuffer);
        if (mOwnsPrevVertexBuffer)
            safeRelease(mPrevVertexBuffer);
        safeRelease(mVertexBuffer);
        mOwnsPrevVertexBuffer = false;
        mPrevVertexBuffer = nullptr;
    }

    mVertexBuffer = pVertexBuffer;
    mIndexBuffer = pIndexBuffer;

    // The previous shutter keyframe, for motion blur and for the denoiser's
    // reprojection. It is a second copy of every vertex in the scene -- 1.66 GB
    // on the pine forest -- and it only ever differs from the current one where
    // something deforms. A scene with no skeletal geometry can share the buffer
    // instead of duplicating it, which is the difference between that scene
    // fitting in memory and not.
    bool anySkeletal = false;
    for (const oka::Mesh& mesh : scene->getMeshes())
    {
        anySkeletal = anySkeletal || mesh.isSkeletal;
    }
    if (vertexDataSize > 0 && anySkeletal)
    {
        mPrevVertexBuffer = mDevice->newBuffer(vertexDataSize, MTL::ResourceStorageModeShared);
        memcpy(mPrevVertexBuffer->contents(), vertices.data(), vertexDataSize);
        mOwnsPrevVertexBuffer = true;
    }
    else
    {
        mPrevVertexBuffer = mVertexBuffer;
        mOwnsPrevVertexBuffer = false;
    }

    buildCurveBuffers(scene);
}

void MetalGeometry::buildCurveBuffers(Scene* scene)
{
    const std::vector<oka::Curve>& curves = scene->getCurves();
    if (curves.empty())
    {
        return;
    }
    const std::vector<glm::float3>& points = scene->getCurvesPoint();
    const std::vector<float>& radii = scene->getCurvesWidths();
    const std::vector<uint32_t>& vertexCounts = scene->getCurvesVertexCounts();

    // Curve widths are stored per set and need not share the point stream's
    // offsets. Expand them into point-aligned storage so every descriptor can
    // use the same base offset as its control points.
    std::vector<float> radiusData(points.size(), 0.001f);
    for (const oka::Curve& curve : curves)
    {
        if (curve.mWidthsCount == static_cast<uint32_t>(-1))
        {
            continue;
        }
        const size_t count = std::min<size_t>(curve.mWidthsCount, curve.mPointsCount);
        if (curve.mWidthsStart + count <= radii.size() && curve.mPointsStart + count <= radiusData.size())
        {
            std::copy_n(radii.begin() + curve.mWidthsStart, count,
                        radiusData.begin() + curve.mPointsStart);
        }
    }

    mCurveRanges.assign(curves.size(), CurveRange{});
    std::vector<uint32_t> segments;
    for (size_t c = 0; c < curves.size(); ++c)
    {
        const oka::Curve& curve = curves[c];
        const uint32_t perSegment = (curve.mType == oka::Curve::Type::eLinear) ? 2u : 4u;
        CurveRange& range = mCurveRanges[c];
        range.segmentStart = (uint32_t)segments.size();
        range.controlPointsPerSegment = perSegment;
        range.segmentsPerStrand = curve.mSegmentsPerStrand;

        uint32_t pointCursor = curve.mPointsStart;
        for (uint32_t s = 0; s < curve.mVertexCountsCount; ++s)
        {
            const uint32_t n = vertexCounts[curve.mVertexCountsStart + s];
            // A strand shorter than one segment contributes nothing rather than
            // an index that runs off the end of the point array.
            for (uint32_t seg = 0; seg + perSegment <= n; ++seg)
            {
                // Metal's curve descriptor sees only this set's control-point
                // range, so its indices are relative to that range. The shader
                // adds mPointsStart back when refetching the same points.
                segments.push_back(pointCursor - curve.mPointsStart + seg);
            }
            pointCursor += n;
        }
        range.segmentCount = (uint32_t)segments.size() - range.segmentStart;
    }

    if (segments.empty())
    {
        STRELKA_WARNING("Scene has {} curve set(s) but no segment long enough to build", curves.size());
        return;
    }

    // packed_float3 rather than glm::float3: they are the same 12 bytes here, but
    // the acceleration structure is told the stride explicitly and a mismatch
    // reads every third point as garbage.
    static_assert(sizeof(glm::float3) == 12, "curve control points are uploaded as tight float3");
    mCurvePointBuffer = mDevice->newBuffer(points.size() * sizeof(glm::float3), MTL::ResourceStorageModeShared);
    memcpy(mCurvePointBuffer->contents(), points.data(), points.size() * sizeof(glm::float3));

    mCurveRadiusBuffer = mDevice->newBuffer(points.size() * sizeof(float), MTL::ResourceStorageModeShared);
    memcpy(mCurveRadiusBuffer->contents(), radiusData.data(), points.size() * sizeof(float));

    mCurveSegmentBuffer = mDevice->newBuffer(segments.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    memcpy(mCurveSegmentBuffer->contents(), segments.data(), segments.size() * sizeof(uint32_t));

    mSceneHasCurves = true;
    STRELKA_INFO("Curves: {} set(s), {} control points, {} segments ({:.2f} MB)", curves.size(),
                 points.size(), segments.size(),
                 (points.size() * 16 + segments.size() * 4) / 1e6);
}

void MetalGeometry::createMeshData(Scene* scene, size_t meshIndex, bool needsPrimitiveData)
{
    const oka::Mesh& mesh = scene->getMeshes()[meshIndex];
    auto* result = new Mesh();

    const uint32_t triangleCount = mesh.mCount / 3;
    result->mTriangleCount = triangleCount;
    result->mVbOffset = mesh.mVbOffset;
    result->mIsSkeletal = mesh.isSkeletal;

    const std::vector<Scene::Vertex>& vertices = scene->getVertices();
    const std::vector<uint32_t>& indices = scene->getIndices();

    // Per-primitive data is a second copy of every triangle's attributes, 72
    // bytes each, held both here and inside the acceleration structure Metal
    // builds from it. Only the megakernel reads it: the wavefront tracer cannot,
    // because intersection.primitive_data is addressable solely inside the
    // kernel that ran the intersect, so its shade stage refetches attributes
    // from the vertex buffer instead (see fetchTriangle in wavefront.metal).
    //
    // Skipping it under the wavefront tracer is worth far more than it sounds:
    // on a 50 M triangle forest it is 3.7 GB of host memory, the same again
    // inside the acceleration structures, and the CPU time to fill it.
    if (!needsPrimitiveData)
    {
        mMetalMeshes.push_back(result);
        return;
    }

    std::vector<Triangle> triangleData(triangleCount);
    for (uint32_t i = 0; i < triangleCount; ++i)
    {
        Triangle& curr = triangleData[i];
        const uint32_t i0 = indices[mesh.mIndex + i * 3 + 0];
        const uint32_t i1 = indices[mesh.mIndex + i * 3 + 1];
        const uint32_t i2 = indices[mesh.mIndex + i * 3 + 2];

        curr.positions[0] = packed_float3(simd_make_float3(vertices[mesh.mVbOffset + i0].pos.x,
                                                          vertices[mesh.mVbOffset + i0].pos.y,
                                                          vertices[mesh.mVbOffset + i0].pos.z));
        curr.positions[1] = packed_float3(simd_make_float3(vertices[mesh.mVbOffset + i1].pos.x,
                                                          vertices[mesh.mVbOffset + i1].pos.y,
                                                          vertices[mesh.mVbOffset + i1].pos.z));
        curr.positions[2] = packed_float3(simd_make_float3(vertices[mesh.mVbOffset + i2].pos.x,
                                                          vertices[mesh.mVbOffset + i2].pos.y,
                                                          vertices[mesh.mVbOffset + i2].pos.z));
        curr.normals[0] = vertices[mesh.mVbOffset + i0].normal;
        curr.normals[1] = vertices[mesh.mVbOffset + i1].normal;
        curr.normals[2] = vertices[mesh.mVbOffset + i2].normal;
        curr.tangent[0] = vertices[mesh.mVbOffset + i0].tangent;
        curr.tangent[1] = vertices[mesh.mVbOffset + i1].tangent;
        curr.tangent[2] = vertices[mesh.mVbOffset + i2].tangent;
        curr.uv[0] = vertices[mesh.mVbOffset + i0].uv;
        curr.uv[1] = vertices[mesh.mVbOffset + i1].uv;
        curr.uv[2] = vertices[mesh.mVbOffset + i2].uv;
    }

    result->mPerPrimitiveBuffer =
        mDevice->newBuffer(triangleData.size() * sizeof(Triangle), MTL::ResourceStorageModeShared);
    memcpy(result->mPerPrimitiveBuffer->contents(), triangleData.data(), sizeof(Triangle) * triangleData.size());

    mMetalMeshes.push_back(result);
}

} // namespace metal
} // namespace oka
