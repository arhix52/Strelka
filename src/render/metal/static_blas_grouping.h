#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <glm/glm.hpp>

namespace oka::metal
{

struct StaticBlasItem
{
    uint32_t id = 0;
    uint32_t triangleCount = 0;
    uint32_t mask = 0;
    glm::vec3 centroid{ 0.0f };
};

inline bool shouldBakeStaticMesh(uint32_t useCount, bool skeletal, bool potentiallyAnimated)
{
    return useCount == 1u && !skeletal && !potentiallyAnimated;
}

/// Spatially order unique static geometry and pack it into bottom levels.
/// Masks stay separate because the visibility mask belongs to the TLAS instance,
/// while the triangle limit bounds one driver's temporary/build allocation.
inline std::vector<std::vector<uint32_t>> groupStaticBlasItems(std::span<const StaticBlasItem> items,
                                                               uint64_t maxTriangles)
{
    if (items.empty() || maxTriangles == 0)
    {
        return {};
    }

    auto finite = [](const glm::vec3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    };

    glm::vec3 lo(std::numeric_limits<float>::max());
    glm::vec3 hi(std::numeric_limits<float>::lowest());
    for (const StaticBlasItem& item : items)
    {
        if (finite(item.centroid))
        {
            lo = glm::min(lo, item.centroid);
            hi = glm::max(hi, item.centroid);
        }
    }
    if (!finite(lo) || !finite(hi))
    {
        lo = glm::vec3(0.0f);
        hi = glm::vec3(1.0f);
    }

    auto spreadBits = [](uint32_t value) {
        uint32_t x = value & 0x000003ffu;
        x = (x | (x << 16u)) & 0x030000FFu;
        x = (x | (x << 8u)) & 0x0300F00Fu;
        x = (x | (x << 4u)) & 0x030C30C3u;
        x = (x | (x << 2u)) & 0x09249249u;
        return x;
    };
    auto mortonCode = [&](glm::vec3 centroid) {
        if (!finite(centroid))
        {
            centroid = lo;
        }
        const glm::vec3 extent = hi - lo;
        glm::vec3 normalized(0.0f);
        for (glm::length_t axis = 0; axis < 3; ++axis)
        {
            if (extent[axis] > 0.0f)
            {
                normalized[axis] = glm::clamp((centroid[axis] - lo[axis]) / extent[axis], 0.0f, 1.0f);
            }
        }
        const glm::uvec3 q = glm::uvec3(normalized * 1023.0f + 0.5f);
        return spreadBits(q.x) | (spreadBits(q.y) << 1u) | (spreadBits(q.z) << 2u);
    };

    struct OrderedItem
    {
        uint32_t inputIndex;
        uint32_t mask;
        uint32_t morton;
    };
    std::vector<OrderedItem> ordered;
    ordered.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i)
    {
        if (items[i].triangleCount != 0)
        {
            ordered.push_back({ static_cast<uint32_t>(i), items[i].mask, mortonCode(items[i].centroid) });
        }
    }
    std::ranges::sort(ordered, [](const OrderedItem& a, const OrderedItem& b) {
        if (a.mask != b.mask)
        {
            return a.mask < b.mask;
        }
        return a.morton != b.morton ? a.morton < b.morton : a.inputIndex < b.inputIndex;
    });

    std::vector<std::vector<uint32_t>> groups;
    uint32_t currentMask = 0;
    uint64_t currentTriangles = 0;
    for (const OrderedItem& orderedItem : ordered)
    {
        const StaticBlasItem& item = items[orderedItem.inputIndex];
        const bool needsGroup =
            groups.empty() || item.mask != currentMask || currentTriangles + item.triangleCount > maxTriangles;
        if (needsGroup)
        {
            groups.emplace_back();
            currentMask = item.mask;
            currentTriangles = 0;
        }
        groups.back().push_back(item.id);
        currentTriangles += item.triangleCount;
    }
    return groups;
}

} // namespace oka::metal
