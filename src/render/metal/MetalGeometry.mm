#include "MetalGeometry.h"

#include <log.h>
#include <strelka/scene/vertex_packing.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <vector>


namespace oka::metal
{
namespace
{
size_t hostPageSize()
{
    static const size_t page = [] {
        const long p = sysconf(_SC_PAGESIZE);
        return p > 0 ? static_cast<size_t>(p) : 4096u;
    }();
    return page;
}

bool canWrapNoCopy(const void* ptr, size_t bytes, size_t capacityBytes, size_t& paddedBytes)
{
    if (ptr == nullptr || bytes == 0)
    {
        return false;
    }
    const size_t page = hostPageSize();
    if ((std::bit_cast<uintptr_t>(ptr) % page) != 0)
    {
        return false;
    }
    paddedBytes = (bytes + page - 1) & ~(page - 1);
    return paddedBytes <= capacityBytes;
}

MTL::Buffer* makeSharedBuffer(
    MTL::Device* device, const void* data, size_t bytes, size_t capacityBytes, const char* name, bool& wrapped)
{
    wrapped = false;
    if (device == nullptr || data == nullptr || bytes == 0)
    {
        return nullptr;
    }
    size_t padded = 0;
    if (canWrapNoCopy(data, bytes, capacityBytes, padded))
    {
        MTL::Buffer* buffer = device->newBuffer(data, padded, MTL::ResourceStorageModeShared,
                                                ^(void*, NS::UInteger){
                                                });
        if (buffer != nullptr)
        {
            wrapped = true;
            STRELKA_INFO("Metal geometry {}: wrapped {:.2f} GB without copy", name, bytes / 1e9);
            return buffer;
        }
    }
    return device->newBuffer(data, bytes, MTL::ResourceStorageModeShared);
}
} // namespace

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
    mGeometryEntryBuffer =
        mDevice->newBuffer(mGeometryEntries.size() * sizeof(GeometryEntry), MTL::ResourceStorageModeShared);
    memcpy(mGeometryEntryBuffer->contents(), mGeometryEntries.data(), mGeometryEntries.size() * sizeof(GeometryEntry));
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
    safeRelease(mPrimitiveDataBuffer);
    mOwnsPrevVertexBuffer = false;
    mVertexBufferAliased = false;
    mWrappedVertices = false;
    mWrappedIndices = false;
    mAdoptedVertices.clear();
    mAdoptedVertices.shrink_to_fit();
    mAdoptedIndices.clear();
    mAdoptedIndices.shrink_to_fit();

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

    // A no-copy wrap aliases the scene arrays, so the previous buffers have to
    // go before those vectors are replaced on a reload.
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
        safeRelease(mPrimitiveDataBuffer);
        mOwnsPrevVertexBuffer = false;
        mPrevVertexBuffer = nullptr;
    }
    mAdoptedVertices.clear();
    mAdoptedIndices.clear();
    mVertexBufferAliased = false;
    mWrappedVertices = false;
    mWrappedIndices = false;

    const size_t vertexDataSize = sizeof(Scene::Vertex) * vertices.size();
    const size_t indexDataSize = sizeof(uint32_t) * indices.size();

    bool anySkeletal = false;
    for (const oka::Mesh& mesh : scene->getMeshes())
    {
        anySkeletal = anySkeletal || mesh.isSkeletal;
    }

    // Shared rather than managed, and this is not a preference.
    //
    // Metal 4 removes the managed storage mode outright -- it exists to keep a
    // separate CPU and GPU copy in step on discrete memory, which is not the
    // architecture Metal 4 targets -- and it removes didModifyRange with it. A
    // managed buffer bound into an argument table by GPU address therefore has
    // no defined behaviour, and the GPU writing one (which is exactly what the
    // skinning kernel does to this buffer) has nowhere to publish the result.
    //
    // When the pointer is page-aligned and the allocation is large enough, wrap
    // it instead of copying. On the pine forest that is 1.5 GB of vertices plus
    // 0.56 GB of indices that otherwise exist twice in the same unified pool.
    // Skeletal vertex buffers are copied: the skinning kernel rewrites them, and
    // the host array has to keep the bind pose for picking.
    if (vertexDataSize > 0)
    {
        const size_t vertexCapacityBytes = anySkeletal ? 0 : vertices.capacity() * sizeof(Scene::Vertex);
        mVertexBuffer = makeSharedBuffer(
            mDevice, vertices.data(), vertexDataSize, vertexCapacityBytes, "vertices", mWrappedVertices);
    }
    if (indexDataSize > 0)
    {
        mIndexBuffer = makeSharedBuffer(
            mDevice, indices.data(), indexDataSize, indices.capacity() * sizeof(uint32_t), "indices", mWrappedIndices);
    }
    mVertexBufferAliased = mWrappedVertices || mWrappedIndices;

    // The previous shutter keyframe, for motion blur and for the denoiser's
    // reprojection. It is a second copy of every vertex in the scene -- 1.66 GB
    // on the pine forest -- and it only ever differs from the current one where
    // something deforms. A scene with no skeletal geometry can share the buffer
    // instead of duplicating it, which is the difference between that scene
    // fitting in memory and not.
    if (vertexDataSize > 0 && anySkeletal)
    {
        mPrevVertexBuffer = mDevice->newBuffer(vertices.data(), vertexDataSize, MTL::ResourceStorageModeShared);
        mOwnsPrevVertexBuffer = true;
    }
    else
    {
        mPrevVertexBuffer = mVertexBuffer;
        mOwnsPrevVertexBuffer = false;
    }

    // Metal can copy a small application record next to each primitive in the
    // acceleration structure. Build one dense array in the same global
    // triangle order as the index buffer; individual geometry descriptors bind
    // the range that starts at mesh.mIndex / 3.
    std::vector<PrimitiveSurfaceData> primitiveData(indices.size() / 3u);
    for (const oka::Mesh& mesh : scene->getMeshes())
    {
        const uint32_t triangleCount = mesh.mCount / 3u;
        const size_t firstTriangle = mesh.mIndex / 3u;
        for (uint32_t triangle = 0; triangle < triangleCount; ++triangle)
        {
            PrimitiveSurfaceData& dst = primitiveData[firstTriangle + triangle];
            const size_t firstIndex = static_cast<size_t>(mesh.mIndex) + static_cast<size_t>(triangle) * 3u;
            const Scene::Vertex* v[3];
            for (uint32_t k = 0; k < 3u; ++k)
            {
                v[k] = &vertices[static_cast<size_t>(mesh.mVbOffset) + indices[firstIndex + k]];
                dst.normal[k] = v[k]->normal;
            }

            const glm::float3 edge1 = v[1]->pos - v[0]->pos;
            const glm::float3 edge2 = v[2]->pos - v[0]->pos;
            const glm::float3 geometricNormal = glm::cross(edge1, edge2);
            const float objectArea2 = glm::length(geometricNormal);
            dst.geometryNormal = packNormal(objectArea2 > 1e-20f ? geometricNormal / objectArea2 : glm::float3(0, 0, 1));
        }
    }
    if (!primitiveData.empty())
    {
        mPrimitiveDataBuffer = mDevice->newBuffer(
            primitiveData.data(), primitiveData.size() * sizeof(PrimitiveSurfaceData), MTL::ResourceStorageModeShared);
        STRELKA_INFO("Metal primitive surface data: {} triangles, {:.1f} MB", primitiveData.size(),
                     primitiveData.size() * sizeof(PrimitiveSurfaceData) / (1024.0 * 1024.0));
    }

    buildCurveBuffers(scene);
}

void MetalGeometry::adoptAliasedHost(Scene* scene)
{
    if (!mVertexBufferAliased || scene == nullptr || scene->hostGeometryReleased())
    {
        return;
    }
    scene->takeHostGeometry(mAdoptedVertices, mAdoptedIndices);
    if (!mWrappedVertices)
    {
        mAdoptedVertices.clear();
        mAdoptedVertices.shrink_to_fit();
    }
    if (!mWrappedIndices)
    {
        mAdoptedIndices.clear();
        mAdoptedIndices.shrink_to_fit();
    }
    mVertexBufferAliased = false;
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
        if (curve.mWidthsCount == kInvalidIndex)
        {
            continue;
        }
        const size_t count = std::min<size_t>(curve.mWidthsCount, curve.mPointsCount);
        if (curve.mWidthsStart + count <= radii.size() && curve.mPointsStart + count <= radiusData.size())
        {
            std::copy_n(radii.begin() + curve.mWidthsStart, count, radiusData.begin() + curve.mPointsStart);
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
    STRELKA_INFO("Curves: {} set(s), {} control points, {} segments ({:.2f} MB)", curves.size(), points.size(),
                 segments.size(), (points.size() * 16 + segments.size() * 4) / 1e6);
}

void MetalGeometry::createMeshData(Scene* scene, size_t meshIndex)
{
    const oka::Mesh& mesh = scene->getMeshes()[meshIndex];
    auto* result = new Mesh();

    result->mTriangleCount = mesh.mCount / 3;
    const auto* vertices = static_cast<const Scene::Vertex*>(mVertexBuffer->contents());
    const auto* indices = static_cast<const uint32_t*>(mIndexBuffer->contents());
    for (uint32_t i = 0; i < mesh.mCount; ++i)
    {
        if (vertices[mesh.mVbOffset + indices[mesh.mIndex + i]].color != 0xffffffffu)
        {
            result->mHasVertexColor = true;
            break;
        }
    }
    mMetalMeshes.push_back(result);
}

} // namespace oka::metal
