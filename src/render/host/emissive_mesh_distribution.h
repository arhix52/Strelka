#pragma once

#include "light_selection.h"

#include <emissive_mesh_light.h>

#include <strelka/scene/scene.h>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

namespace oka::render
{

struct EmissiveMeshBuildInput
{
    uint32_t instanceId = 0;
    uint32_t geometryId = 0;
    uint32_t vertexOffset = 0;
    uint32_t indexOffset = 0;
    uint32_t materialId = 0;
    uint32_t transformIndex = 0;
    std::vector<double> trianglePowers;
};

struct EmissiveMeshDistribution
{
    std::vector<EmissiveMeshLight> meshes;
    std::vector<EmissiveTriangleLight> triangles;
    double totalPower = 0.0;
};

inline double emissiveMaterialLuminance(const Scene::MaterialDescription& material)
{
    glm::dvec3 emission;
    double strength = 0.0;
    if (material.params.material_type == MATERIAL_TYPE_OPENPBR)
    {
        // A MaterialX image input replaces emission_color rather than
        // modulating it. Texture decoding deliberately stays out of proposal
        // construction, so a named map uses a conservative white proxy: dark
        // texels merely add variance, while a zero proxy would lose every
        // positive texel from NEE support.
        if (!material.openpbrTexPaths[OPENPBR_TEX_EMISSION_COLOR].empty())
        {
            emission = glm::dvec3(1.0);
        }
        else
        {
            emission = { material.openpbr.emission_color.r, material.openpbr.emission_color.g,
                         material.openpbr.emission_color.b };
        }
        strength = material.openpbr.emission_luminance;
    }
    else
    {
        // Field-wise, not glm::dvec3(material.params.emission): under the OptiX
        // backend's STRELKA_MATERIAL_CUDA_HOST build, MaterialParams::emission is
        // CUDA's float3, which glm has no converting constructor for. Both
        // spellings expose the same x/y/z, so this works either way.
        emission = { material.params.emission.x, material.params.emission.y, material.params.emission.z };
        strength = material.params.emission_strength;
    }
    if (!(std::isfinite(strength) && strength > 0.0) || !std::isfinite(emission.x) || !std::isfinite(emission.y) ||
        !std::isfinite(emission.z))
    {
        return 0.0;
    }
    // Negative radiance channels are invalid and are clamped by the proposal,
    // not allowed to cancel a positive channel and erase its sampling support.
    const glm::dvec3 positive = glm::max(emission, glm::dvec3(0.0));
    return (0.2126 * positive.r + 0.7152 * positive.g + 0.0722 * positive.b) * strength;
}

inline std::vector<double> emissiveTrianglePowers(std::span<const Scene::Vertex> vertices,
                                                  std::span<const uint32_t> indices,
                                                  const Mesh& mesh,
                                                  const Scene::MaterialDescription& material,
                                                  const glm::mat4& objectToWorld,
                                                  bool preservePotentialMotionSupport = false)
{
    // The luminance decides whether this mesh has any proposal at all, so it is
    // read before the per-triangle vector exists. A forest scene is ~47 M
    // triangles across instances of which none emit: allocating and zeroing a
    // double per triangle for each of them cost 12 s of the 25 s pine load and
    // most of its 42 GB peak RSS, all of it thrown away one call later when the
    // alias table came out with zero power. An empty vector is the same signal
    // -- buildEmissiveMeshDistribution drops a mesh whose table has no power,
    // and it reads triangleCount only after that test.
    const double radiance = emissiveMaterialLuminance(material);
    if (!(radiance > 0.0))
    {
        return {};
    }
    const size_t triangleCount = mesh.mCount / 3u;
    std::vector<double> powers(triangleCount, 0.0);

    for (size_t triangle = 0; triangle < triangleCount; ++triangle)
    {
        const size_t index = size_t(mesh.mIndex) + triangle * 3u;
        if (index + 2u >= indices.size())
        {
            break;
        }
        const size_t i0 = size_t(mesh.mVbOffset) + indices[index + 0u];
        const size_t i1 = size_t(mesh.mVbOffset) + indices[index + 1u];
        const size_t i2 = size_t(mesh.mVbOffset) + indices[index + 2u];
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size())
        {
            continue;
        }
        const glm::float3 p0 = glm::float3(objectToWorld * glm::vec4(vertices[i0].pos, 1.0f));
        const glm::float3 p1 = glm::float3(objectToWorld * glm::vec4(vertices[i1].pos, 1.0f));
        const glm::float3 p2 = glm::float3(objectToWorld * glm::vec4(vertices[i2].pos, 1.0f));
        const float areaPdf = emissiveTriangleAreaPdf(
            make_float3(p0.x, p0.y, p0.z), make_float3(p1.x, p1.y, p1.z), make_float3(p2.x, p2.y, p2.z));
        if (areaPdf > 0.0f)
        {
            // Existing mesh emission is two-sided. This is only a power proxy
            // for selection; the exact sampled density is reconstructed below.
            powers[triangle] = 2.0 * std::numbers::pi_v<double> * radiance / double(areaPdf);
        }
        else if (preservePotentialMotionSupport)
        {
            // The host may only have a bind pose or shutter endpoints while
            // traversal samples a skinned/interpolated pose. A unit-area proxy
            // keeps this triangle reachable; its exact current-time area PDF
            // is reconstructed from device vertices after selection.
            powers[triangle] = 2.0 * std::numbers::pi_v<double> * radiance;
        }
    }
    return powers;
}

inline std::vector<double> emissiveTrianglePowers(const Scene& scene,
                                                  const Mesh& mesh,
                                                  const Scene::MaterialDescription& material,
                                                  const glm::mat4& objectToWorld,
                                                  bool preservePotentialMotionSupport = false)
{
    return emissiveTrianglePowers(
        scene.getVertices(), scene.getIndices(), mesh, material, objectToWorld, preservePotentialMotionSupport);
}

inline EmissiveMeshDistribution buildEmissiveMeshDistribution(const std::vector<EmissiveMeshBuildInput>& inputs)
{
    EmissiveMeshDistribution out;
    std::vector<double> meshPowers;
    meshPowers.reserve(inputs.size());

    for (const EmissiveMeshBuildInput& input : inputs)
    {
        const metal::LightSelectionTable triangleTable = metal::buildLightSelectionAlias(input.trianglePowers);
        if (!(triangleTable.totalPower > 0.0))
        {
            continue;
        }

        EmissiveMeshLight mesh{};
        mesh.instanceId = input.instanceId;
        mesh.geometryId = input.geometryId;
        mesh.triangleOffset = static_cast<uint32_t>(out.triangles.size());
        mesh.triangleCount = static_cast<uint32_t>(input.trianglePowers.size());
        mesh.vertexOffset = input.vertexOffset;
        mesh.indexOffset = input.indexOffset;
        mesh.materialId = input.materialId;
        mesh.transformIndex = input.transformIndex;
        out.meshes.push_back(mesh);
        meshPowers.push_back(triangleTable.totalPower);

        for (const metal::LightSelectionEntry& entry : triangleTable.entries)
        {
            EmissiveTriangleLight triangle{};
            triangle.selectionPdf = entry.pdf;
            triangle.aliasThreshold = entry.aliasThreshold;
            triangle.alias = entry.alias;
            out.triangles.push_back(triangle);
        }
    }

    const metal::LightSelectionTable meshTable = metal::buildLightSelectionAlias(meshPowers);
    out.totalPower = meshTable.totalPower;
    for (size_t i = 0; i < out.meshes.size(); ++i)
    {
        out.meshes[i].selectionPdf = meshTable.entries[i].pdf;
        out.meshes[i].aliasThreshold = meshTable.entries[i].aliasThreshold;
        out.meshes[i].alias = meshTable.entries[i].alias;
    }
    return out;
}

} // namespace oka::render
