#include <doctest/doctest.h>

#include <strelka/sceneloader/static_blas_partition.h>

#include <algorithm>
#include <cstdint>
#include <ranges>
#include <vector>


namespace
{

oka::Scene::Vertex vertex(float x, float y = 0.0f, float z = 0.0f)
{
    oka::Scene::Vertex result{};
    result.pos = glm::float3(x, y, z);
    return result;
}

std::vector<oka::Scene::Vertex> separatedTriangles(const std::vector<float>& centres)
{
    std::vector<oka::Scene::Vertex> vertices;
    vertices.reserve(centres.size() * 3u);
    for (const float centre : centres)
    {
        vertices.push_back(vertex(centre - 0.1f));
        vertices.push_back(vertex(centre + 0.1f));
        vertices.push_back(vertex(centre, 0.1f));
    }
    return vertices;
}

float triangleCentreX(const std::vector<oka::Scene::Vertex>& vertices, const std::vector<uint32_t>& indices, size_t triangle)
{
    const size_t first = triangle * 3u;
    return (vertices[indices[first]].pos.x + vertices[indices[first + 1u]].pos.x + vertices[indices[first + 2u]].pos.x) /
           3.0f;
}

} // namespace

TEST_CASE("small static mesh is not reordered for BLAS partitioning")
{
    const std::vector<oka::Scene::Vertex> vertices = separatedTriangles({ 10.0f, -10.0f });
    std::vector<uint32_t> indices = { 0, 1, 2, 3, 4, 5 };
    const std::vector<uint32_t> before = indices;

    const auto partitions = oka::sceneloader::partitionStaticTriangles(vertices, indices, 2u);

    REQUIRE(partitions.size() == 1u);
    CHECK(partitions[0].firstTriangle == 0u);
    CHECK(partitions[0].triangleCount == 2u);
    CHECK(indices == before);
}

TEST_CASE("large static mesh is spatially ordered into bounded BLAS ranges")
{
    // Source order deliberately alternates between opposite sides of the mesh.
    const std::vector<oka::Scene::Vertex> vertices = separatedTriangles({ 9.0f, -8.0f, 8.0f, -9.0f });
    std::vector<uint32_t> indices(vertices.size());
    for (uint32_t i = 0; i < indices.size(); ++i)
    {
        indices[i] = i;
    }

    const auto partitions = oka::sceneloader::partitionStaticTriangles(vertices, indices, 2u);

    REQUIRE(partitions.size() == 2u);
    CHECK(partitions[0].firstTriangle == 0u);
    CHECK(partitions[0].triangleCount == 2u);
    CHECK(partitions[1].firstTriangle == 2u);
    CHECK(partitions[1].triangleCount == 2u);
    const float leftMax = std::max(triangleCentreX(vertices, indices, 0u), triangleCentreX(vertices, indices, 1u));
    const float rightMin = std::min(triangleCentreX(vertices, indices, 2u), triangleCentreX(vertices, indices, 3u));
    CHECK(leftMax < rightMin);

    for (size_t triangle = 0; triangle < indices.size() / 3u; ++triangle)
    {
        const size_t first = triangle * 3u;
        CHECK(indices[first] / 3u == indices[first + 1u] / 3u);
        CHECK(indices[first] / 3u == indices[first + 2u] / 3u);
    }
    std::ranges::sort(indices);
    for (uint32_t i = 0; i < indices.size(); ++i)
    {
        CHECK(indices[i] == i);
    }
}

TEST_CASE("static BLAS partition tail covers every triangle")
{
    const std::vector<oka::Scene::Vertex> vertices = separatedTriangles({ 6, 1, 5, 2, 4, 3, 0 });
    std::vector<uint32_t> indices(vertices.size());
    for (uint32_t i = 0; i < indices.size(); ++i)
    {
        indices[i] = i;
    }

    const auto partitions = oka::sceneloader::partitionStaticTriangles(vertices, indices, 3u);

    REQUIRE(partitions.size() == 3u);
    CHECK(partitions[0].triangleCount == 3u);
    CHECK(partitions[1].triangleCount == 3u);
    CHECK(partitions[2].firstTriangle == 6u);
    CHECK(partitions[2].triangleCount == 1u);
}
