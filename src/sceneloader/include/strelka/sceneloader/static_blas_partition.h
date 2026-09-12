#pragma once

#include <strelka/scene/scene.h>

#include <cstdint>
#include <span>
#include <vector>


namespace oka::sceneloader
{

/// Spatially order indexed triangles by their object-space centroids and return
/// contiguous ranges no wider than maxTriangles. Indices stay relative to the
/// supplied vertex span. A small mesh is left byte-for-byte unchanged.
std::vector<Mesh::StaticBlasPartition> partitionStaticTriangles(std::span<const Scene::Vertex> vertices,
                                                                std::span<uint32_t> indices,
                                                                uint32_t maxTriangles = Mesh::kMaxStaticBlasTriangles);

} // namespace oka::sceneloader
