#include <strelka/sceneloader/static_blas_partition.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>


namespace oka::sceneloader
{
namespace
{

struct TriangleOrder
{
    uint32_t morton = 0;
    uint32_t triangle = 0;
};

uint32_t spreadTenBits(uint32_t value)
{
    value &= 0x000003ffu;
    value = (value | (value << 16u)) & 0x030000ffu;
    value = (value | (value << 8u)) & 0x0300f00fu;
    value = (value | (value << 4u)) & 0x030c30c3u;
    value = (value | (value << 2u)) & 0x09249249u;
    return value;
}

uint32_t mortonCode(const glm::float3& point, const glm::float3& boundsMin, const glm::float3& boundsMax)
{
    uint32_t coordinate[3] = {};
    for (int axis = 0; axis < 3; ++axis)
    {
        const float extent = boundsMax[axis] - boundsMin[axis];
        float normalised =
            extent > std::numeric_limits<float>::epsilon() ? (point[axis] - boundsMin[axis]) / extent : 0.0f;
        if (!std::isfinite(normalised))
        {
            normalised = 0.0f;
        }
        normalised = std::clamp(normalised, 0.0f, 1.0f);
        coordinate[axis] = static_cast<uint32_t>(normalised * 1023.0f);
    }
    return spreadTenBits(coordinate[0]) | (spreadTenBits(coordinate[1]) << 1u) | (spreadTenBits(coordinate[2]) << 2u);
}

glm::float3 triangleCentroid(std::span<const Scene::Vertex> vertices, std::span<const uint32_t> indices, uint32_t triangle)
{
    glm::float3 centroid(0.0f);
    const size_t first = static_cast<size_t>(triangle) * 3u;
    for (size_t corner = 0; corner < 3; ++corner)
    {
        const uint32_t vertex = indices[first + corner];
        if (vertex < vertices.size())
        {
            centroid += vertices[vertex].pos;
        }
    }
    return centroid / 3.0f;
}

} // namespace

std::vector<Mesh::StaticBlasPartition> partitionStaticTriangles(std::span<const Scene::Vertex> vertices,
                                                                std::span<uint32_t> indices,
                                                                uint32_t maxTriangles)
{
    const uint32_t triangleCount = static_cast<uint32_t>(indices.size() / 3u);
    if (triangleCount == 0 || maxTriangles == 0)
    {
        return {};
    }
    if (triangleCount <= maxTriangles)
    {
        return { { 0, triangleCount } };
    }

    glm::float3 boundsMin(std::numeric_limits<float>::max());
    glm::float3 boundsMax(std::numeric_limits<float>::lowest());
    for (uint32_t triangle = 0; triangle < triangleCount; ++triangle)
    {
        const glm::float3 centroid = triangleCentroid(vertices, indices, triangle);
        boundsMin = glm::min(boundsMin, centroid);
        boundsMax = glm::max(boundsMax, centroid);
    }

    std::vector<TriangleOrder> order(triangleCount);
    for (uint32_t triangle = 0; triangle < triangleCount; ++triangle)
    {
        order[triangle] = { mortonCode(triangleCentroid(vertices, indices, triangle), boundsMin, boundsMax), triangle };
    }
    std::ranges::sort(order, [](const TriangleOrder& a, const TriangleOrder& b) {
        return a.morton != b.morton ? a.morton < b.morton : a.triangle < b.triangle;
    });

    std::vector<uint32_t> sortedIndices(static_cast<size_t>(triangleCount) * 3u);
    for (uint32_t destination = 0; destination < triangleCount; ++destination)
    {
        const size_t source = static_cast<size_t>(order[destination].triangle) * 3u;
        const size_t target = static_cast<size_t>(destination) * 3u;
        std::copy_n(indices.begin() + static_cast<std::ptrdiff_t>(source), 3,
                    sortedIndices.begin() + static_cast<std::ptrdiff_t>(target));
    }
    std::ranges::copy(sortedIndices, indices.begin());

    std::vector<Mesh::StaticBlasPartition> partitions;
    partitions.reserve((triangleCount + maxTriangles - 1u) / maxTriangles);
    for (uint32_t first = 0; first < triangleCount; first += maxTriangles)
    {
        partitions.push_back({ first, std::min(maxTriangles, triangleCount - first) });
    }
    return partitions;
}

} // namespace oka::sceneloader
