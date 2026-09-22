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
            buffer->setLabel(NS::String::string(name, NS::UTF8StringEncoding));
            wrapped = true;
            STRELKA_INFO("Metal geometry {}: wrapped {:.2f} GB without copy", name, bytes / 1e9);
            return buffer;
        }
    }
    MTL::Buffer* buffer = device->newBuffer(data, bytes, MTL::ResourceStorageModeShared);
    if (buffer)
    {
        buffer->setLabel(NS::String::string(name, NS::UTF8StringEncoding));
    }
    return buffer;
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
    mPrimitiveDataOffsets.clear();
    safeRelease(mPrimitiveAlphaDataBuffer);
    safeRelease(mPrimitiveAlphaDecodeBuffer);
    mPrimitiveAlphaDataOffsets.clear();
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
        mPrimitiveDataOffsets.clear();
        safeRelease(mPrimitiveAlphaDataBuffer);
        safeRelease(mPrimitiveAlphaDecodeBuffer);
        mPrimitiveAlphaDataOffsets.clear();
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

    if (vertexDataSize > 0 && anySkeletal)
    {
        mPrevVertexBuffer = mDevice->newBuffer(vertices.data(), vertexDataSize, MTL::ResourceStorageModeShared);
        if (mPrevVertexBuffer)
        {
            mPrevVertexBuffer->setLabel(NS::String::string("vertices previous", NS::UTF8StringEncoding));
        }
        mOwnsPrevVertexBuffer = true;
    }
    else
    {
        mPrevVertexBuffer = mVertexBuffer;
        mOwnsPrevVertexBuffer = false;
    }

    buildCurveBuffers(scene);
}

void MetalGeometry::buildPrimitiveData(const Scene* scene, std::span<const uint8_t> enabledMeshes)
{
    if (mPrimitiveDataBuffer)
    {
        mPrimitiveDataBuffer->release();
        mPrimitiveDataBuffer = nullptr;
    }
    mPrimitiveDataOffsets.assign(scene ? scene->getMeshes().size() : 0u, kNoPrimitiveDataOffset);
    if (!scene || !mVertexBuffer || !mIndexBuffer || enabledMeshes.size() != scene->getMeshes().size())
    {
        return;
    }

    size_t triangleCount = 0u;
    size_t enabledMeshCount = 0u;
    for (size_t meshIndex = 0; meshIndex < enabledMeshes.size(); ++meshIndex)
    {
        if (enabledMeshes[meshIndex] == 0u)
        {
            continue;
        }
        triangleCount += scene->getMeshes()[meshIndex].mCount / 3u;
        ++enabledMeshCount;
    }
    if (triangleCount == 0u)
    {
        return;
    }
    // Geometry uploads use shared storage on both the copied and the no-copy
    // path, so rebuilds can regenerate this compact source even after the
    // headless loader has released its Scene vectors.
    const auto* vertices = static_cast<const Scene::Vertex*>(mVertexBuffer->contents());
    const auto* indices = static_cast<const uint32_t*>(mIndexBuffer->contents());
    if (!vertices || !indices)
    {
        STRELKA_ERROR("Cannot build compact primitive surface data: geometry buffers are not CPU-visible");
        return;
    }

    std::vector<PrimitiveSurfaceData> primitiveData(triangleCount);
    size_t dstTriangle = 0u;
    for (size_t meshIndex = 0; meshIndex < enabledMeshes.size(); ++meshIndex)
    {
        if (enabledMeshes[meshIndex] == 0u)
        {
            continue;
        }
        const oka::Mesh& mesh = scene->getMeshes()[meshIndex];
        const uint32_t meshTriangleCount = mesh.mCount / 3u;
        mPrimitiveDataOffsets[meshIndex] = dstTriangle * sizeof(PrimitiveSurfaceData);
        for (uint32_t triangle = 0; triangle < meshTriangleCount; ++triangle)
        {
            PrimitiveSurfaceData& dst = primitiveData[dstTriangle++];
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

    mPrimitiveDataBuffer = mDevice->newBuffer(
        primitiveData.data(), primitiveData.size() * sizeof(PrimitiveSurfaceData), MTL::ResourceStorageModeShared);
    if (!mPrimitiveDataBuffer)
    {
        STRELKA_ERROR("Cannot allocate {:.1f} MB of compact primitive surface data",
                      primitiveData.size() * sizeof(PrimitiveSurfaceData) / (1024.0 * 1024.0));
        std::ranges::fill(mPrimitiveDataOffsets, kNoPrimitiveDataOffset);
        return;
    }
    mPrimitiveDataBuffer->setLabel(NS::String::string("primitive surface build data", NS::UTF8StringEncoding));
    STRELKA_INFO("Metal primitive surface data: {} meshes, {} triangles, {:.1f} MB compact source", enabledMeshCount,
                 primitiveData.size(), primitiveData.size() * sizeof(PrimitiveSurfaceData) / (1024.0 * 1024.0));
}

size_t MetalGeometry::primitiveDataOffset(size_t meshIndex, uint32_t firstTriangle) const
{
    if (!mPrimitiveDataBuffer || meshIndex >= mPrimitiveDataOffsets.size() ||
        mPrimitiveDataOffsets[meshIndex] == kNoPrimitiveDataOffset)
    {
        return kNoPrimitiveDataOffset;
    }
    return mPrimitiveDataOffsets[meshIndex] + static_cast<size_t>(firstTriangle) * sizeof(PrimitiveSurfaceData);
}

void MetalGeometry::buildPrimitiveAlphaData(const Scene* scene, std::span<const uint8_t> enabledMeshes)
{
    if (mPrimitiveAlphaDataBuffer)
    {
        mPrimitiveAlphaDataBuffer->release();
        mPrimitiveAlphaDataBuffer = nullptr;
    }
    if (mPrimitiveAlphaDecodeBuffer)
    {
        mPrimitiveAlphaDecodeBuffer->release();
        mPrimitiveAlphaDecodeBuffer = nullptr;
    }
    mPrimitiveAlphaDataOffsets.assign(scene ? scene->getMeshes().size() : 0u, kNoPrimitiveAlphaDataOffset);
    if (!scene || !mVertexBuffer || !mIndexBuffer || enabledMeshes.size() != scene->getMeshes().size())
    {
        return;
    }

    size_t triangleCount = 0u;
    size_t recordCount = 0u;
    size_t enabledMeshCount = 0u;
    for (size_t meshIndex = 0; meshIndex < enabledMeshes.size(); ++meshIndex)
    {
        if (enabledMeshes[meshIndex] != 0u)
        {
            const size_t meshTriangleCount = scene->getMeshes()[meshIndex].mCount / 3u;
            triangleCount += meshTriangleCount;
            recordCount = (recordCount + PRIMITIVE_ALPHA_BLOCK_SIZE - 1u) & ~(PRIMITIVE_ALPHA_BLOCK_SIZE - 1u);
            recordCount += meshTriangleCount;
            ++enabledMeshCount;
        }
    }
    if (triangleCount == 0u)
    {
        return;
    }
    if (recordCount > static_cast<size_t>(GEOM_PRIMITIVE_ALPHA_DATA_INDEX_MASK) + 1u)
    {
        STRELKA_WARNING("Primitive alpha data disabled: {} aligned records exceed the {}-record geometry index",
                        recordCount, static_cast<size_t>(GEOM_PRIMITIVE_ALPHA_DATA_INDEX_MASK) + 1u);
        return;
    }

    const auto* vertices = static_cast<const Scene::Vertex*>(mVertexBuffer->contents());
    const auto* indices = static_cast<const uint32_t*>(mIndexBuffer->contents());
    if (!vertices || !indices)
    {
        STRELKA_ERROR("Cannot build primitive alpha data: geometry buffers are not CPU-visible");
        return;
    }

    std::vector<PrimitiveAlphaData> alphaData(recordCount);
    std::vector<PrimitiveAlphaDecode> alphaDecode((recordCount + PRIMITIVE_ALPHA_BLOCK_SIZE - 1u) >>
                                                  PRIMITIVE_ALPHA_BLOCK_SHIFT);
    size_t dstTriangle = 0u;
    size_t exactTriangleCount = 0u;
    for (size_t meshIndex = 0; meshIndex < enabledMeshes.size(); ++meshIndex)
    {
        if (enabledMeshes[meshIndex] == 0u)
        {
            continue;
        }
        const oka::Mesh& mesh = scene->getMeshes()[meshIndex];
        const uint32_t meshTriangleCount = mesh.mCount / 3u;
        dstTriangle = (dstTriangle + PRIMITIVE_ALPHA_BLOCK_SIZE - 1u) & ~(PRIMITIVE_ALPHA_BLOCK_SIZE - 1u);
        mPrimitiveAlphaDataOffsets[meshIndex] = dstTriangle * sizeof(PrimitiveAlphaData);
        for (uint32_t blockFirst = 0; blockFirst < meshTriangleCount; blockFirst += PRIMITIVE_ALPHA_BLOCK_SIZE)
        {
            const uint32_t blockCount = std::min(PRIMITIVE_ALPHA_BLOCK_SIZE, meshTriangleCount - blockFirst);
            uint32_t minU = 0xffffu;
            uint32_t minV = 0xffffu;
            uint32_t maxU = 0u;
            uint32_t maxV = 0u;
            for (uint32_t localTriangle = 0; localTriangle < blockCount; ++localTriangle)
            {
                const size_t firstIndex =
                    static_cast<size_t>(mesh.mIndex) + static_cast<size_t>(blockFirst + localTriangle) * 3u;
                for (uint32_t k = 0; k < 3u; ++k)
                {
                    const Scene::Vertex& vertex =
                        vertices[static_cast<size_t>(mesh.mVbOffset) + indices[firstIndex + k]];
                    const uint32_t packed = packUV(unpackUV(vertex.uv, vertex.uv1));
                    const uint32_t u = packed & 0xffffu;
                    const uint32_t v = packed >> 16u;
                    minU = std::min(minU, u);
                    minV = std::min(minV, v);
                    maxU = std::max(maxU, u);
                    maxV = std::max(maxV, v);
                }
            }
            const auto quantizationShift = [](uint32_t range) {
                uint32_t shift = 0u;
                while (((range + (1u << shift) - 1u) >> shift) > 1023u)
                {
                    ++shift;
                }
                return shift;
            };
            const uint32_t shiftU = quantizationShift(maxU - minU);
            const uint32_t shiftV = quantizationShift(maxV - minV);
            const uint32_t stepU = 1u << shiftU;
            const uint32_t stepV = 1u << shiftV;
            exactTriangleCount += (shiftU == 0u && shiftV == 0u) ? blockCount : 0u;

            constexpr float kUvScale = 20.0f / 16383.99999f;
            PrimitiveAlphaDecode& decode = alphaDecode[dstTriangle >> PRIMITIVE_ALPHA_BLOCK_SHIFT];
            decode.offsetScale = { static_cast<float>(minU) * kUvScale - 10.0f,
                                   static_cast<float>(minV) * kUvScale - 10.0f,
                                   static_cast<float>(1023u * stepU) * kUvScale,
                                   static_cast<float>(1023u * stepV) * kUvScale };

            for (uint32_t localTriangle = 0; localTriangle < blockCount; ++localTriangle)
            {
                PrimitiveAlphaData& dst = alphaData[dstTriangle++];
                const size_t firstIndex =
                    static_cast<size_t>(mesh.mIndex) + static_cast<size_t>(blockFirst + localTriangle) * 3u;
                uint32_t packedU = 0u;
                uint32_t packedV = 0u;
                for (uint32_t k = 0; k < 3u; ++k)
                {
                    const Scene::Vertex& vertex =
                        vertices[static_cast<size_t>(mesh.mVbOffset) + indices[firstIndex + k]];
                    const uint32_t packed = packUV(unpackUV(vertex.uv, vertex.uv1));
                    const uint32_t u = std::min(((packed & 0xffffu) - minU + stepU / 2u) / stepU, 1023u);
                    const uint32_t v = std::min(((packed >> 16u) - minV + stepV / 2u) / stepV, 1023u);
                    packedU |= u << (10u * k);
                    packedV |= v << (10u * k);
                }
                dst.u = packedU;
                dst.v = packedV;
            }
        }
    }

    mPrimitiveAlphaDataBuffer = mDevice->newBuffer(
        alphaData.data(), alphaData.size() * sizeof(PrimitiveAlphaData), MTL::ResourceStorageModeShared);
    if (!mPrimitiveAlphaDataBuffer)
    {
        STRELKA_ERROR("Cannot allocate {:.1f} MB of primitive alpha data",
                      alphaData.size() * sizeof(PrimitiveAlphaData) / (1024.0 * 1024.0));
        std::ranges::fill(mPrimitiveAlphaDataOffsets, kNoPrimitiveAlphaDataOffset);
        return;
    }
    mPrimitiveAlphaDecodeBuffer = mDevice->newBuffer(
        alphaDecode.data(), alphaDecode.size() * sizeof(PrimitiveAlphaDecode), MTL::ResourceStorageModeShared);
    if (!mPrimitiveAlphaDecodeBuffer)
    {
        STRELKA_ERROR("Cannot allocate {:.1f} MB of primitive alpha decode data",
                      alphaDecode.size() * sizeof(PrimitiveAlphaDecode) / (1024.0 * 1024.0));
        mPrimitiveAlphaDataBuffer->release();
        mPrimitiveAlphaDataBuffer = nullptr;
        std::ranges::fill(mPrimitiveAlphaDataOffsets, kNoPrimitiveAlphaDataOffset);
        return;
    }
    mPrimitiveAlphaDataBuffer->setLabel(NS::String::string("primitive alpha data", NS::UTF8StringEncoding));
    mPrimitiveAlphaDecodeBuffer->setLabel(NS::String::string("primitive alpha decode", NS::UTF8StringEncoding));
    STRELKA_INFO("Metal primitive alpha data: {} meshes, {} triangles, {:.1f} MB, {:.1f}% exact UV blocks",
                 enabledMeshCount, triangleCount,
                 (alphaData.size() * sizeof(PrimitiveAlphaData) + alphaDecode.size() * sizeof(PrimitiveAlphaDecode)) /
                     (1024.0 * 1024.0),
                 100.0 * static_cast<double>(exactTriangleCount) / static_cast<double>(triangleCount));
}

size_t MetalGeometry::primitiveAlphaDataOffset(size_t meshIndex, uint32_t firstTriangle) const
{
    if (!mPrimitiveAlphaDataBuffer || meshIndex >= mPrimitiveAlphaDataOffsets.size() ||
        mPrimitiveAlphaDataOffsets[meshIndex] == kNoPrimitiveAlphaDataOffset)
    {
        return kNoPrimitiveAlphaDataOffset;
    }
    return mPrimitiveAlphaDataOffsets[meshIndex] + static_cast<size_t>(firstTriangle) * sizeof(PrimitiveAlphaData);
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
