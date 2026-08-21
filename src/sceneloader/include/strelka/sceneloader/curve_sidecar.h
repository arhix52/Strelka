#pragma once

#include <log.h>
#include <strelka/scene/scene.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>


namespace oka::curvesidecar
{

// Curves ride beside the glTF, the way the analytic lights already do.
//
// glTF has no curve primitive and no extension that adds one, and hair is the
// one thing in these scenes that must not be triangulated: a strand is eight
// control points, and the ribbon that would replace it is at least a hundred
// bytes of vertices for a worse silhouette. Both backends already take curves
// natively -- OptiX builds a curve GAS, Metal has a curve geometry descriptor --
// so what was missing was only a way to get them off disk.
//
// Binary rather than JSON because the payload is millions of floats: the kids
// bedroom's two particle systems are 3.6 M control points, which is 60 MB packed
// and roughly 400 MB as text.
//
//   char[8]  "STRKCRV1"
//   uint32   setCount
//   per set:
//     uint32   materialNameLength
//     char[]   materialName        -- matched against Scene::MaterialDescription
//     uint32   basis               -- 0 linear, 1 cubic B-spline
//     uint32   strandCount
//     uint32   pointCount
//     float[16] objectToWorld      -- column-major, as glm stores it
//     uint32[strandCount] controlPointsPerStrand
//     float[3 * pointCount] points
//     float[pointCount] radii
//
// Radii, not diameters: that is what both OptiX's width buffer and Metal's
// radius buffer mean, and converting at every consumer is how the two ended up
// disagreeing about curve type in the first place.
inline constexpr char kMagic[8] = { 'S', 'T', 'R', 'K', 'C', 'R', 'V', '1' };

struct Reader
{
    /// A view over the caller's buffer, deliberately: the sidecar is read once,
    /// in one scope, and copying a multi-megabyte groom to walk it would be the
    /// whole cost of loading it. Non-copyable as a consequence, which is what
    /// cppcoreguidelines-avoid-const-or-ref-data-members is warning about and
    /// what is wanted here.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::vector<char>& data;
    size_t offset = 0;

    bool need(size_t bytes) const
    {
        return offset + bytes <= data.size();
    }
    template <typename T>
    bool pod(T& out)
    {
        if (!need(sizeof(T)))
            return false;
        std::memcpy(&out, data.data() + offset, sizeof(T));
        offset += sizeof(T);
        return true;
    }
    template <typename T>
    bool array(std::vector<T>& out, size_t count)
    {
        if (!need(sizeof(T) * count))
            return false;
        out.resize(count);
        if (count)
        {
            std::memcpy(out.data(), data.data() + offset, sizeof(T) * count);
        }
        offset += sizeof(T) * count;
        return true;
    }
};

inline bool loadCurvesFile(const std::string& path, Scene& scene)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        STRELKA_ERROR("Curve sidecar {} could not be opened", path);
        return false;
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data((size_t)size);
    if (!file.read(data.data(), size))
    {
        STRELKA_ERROR("Curve sidecar {} could not be read", path);
        return false;
    }

    Reader r{ data };
    char magic[8] = {};
    if (!r.pod(magic))
    {
        STRELKA_ERROR("Curve sidecar {} is truncated", path);
        return false;
    }
    if (std::memcmp(magic, kMagic, sizeof(magic)) != 0)
    {
        STRELKA_ERROR("Curve sidecar {} does not start with {}", path, std::string(kMagic, 8));
        return false;
    }

    uint32_t setCount = 0;
    if (!r.pod(setCount))
    {
        STRELKA_ERROR("Curve sidecar {} is truncated at the set count", path);
        return false;
    }

    // Name -> material id, built once. The alternative is a linear scan per set,
    // which is fine at five sets and quadratic if a converter ever writes one set
    // per strand cluster.
    std::unordered_map<std::string, uint32_t> materialOfName;
    const std::vector<Scene::MaterialDescription>& materials = scene.getMaterials();
    for (uint32_t i = 0; i < materials.size(); ++i)
    {
        materialOfName.emplace(materials[i].name, i);
    }

    size_t totalStrands = 0;
    size_t totalPoints = 0;
    uint32_t created = 0;
    for (uint32_t s = 0; s < setCount; ++s)
    {
        uint32_t nameLen = 0;
        if (!r.pod(nameLen) || !r.need(nameLen))
        {
            STRELKA_ERROR("Curve sidecar {}: set {} has no material name", path, s);
            return false;
        }
        const std::string materialName(data.data() + r.offset, nameLen);
        r.offset += nameLen;

        uint32_t basis = 0, strandCount = 0, pointCount = 0;
        float xform[16] = {};
        if (!r.pod(basis) || !r.pod(strandCount) || !r.pod(pointCount) || !r.need(sizeof(xform)))
        {
            STRELKA_ERROR("Curve sidecar {}: set '{}' has a truncated header", path, materialName);
            return false;
        }
        std::memcpy(xform, data.data() + r.offset, sizeof(xform));
        r.offset += sizeof(xform);

        std::vector<uint32_t> vertexCounts;
        std::vector<float> rawPoints;
        std::vector<float> radii;
        if (!r.array(vertexCounts, strandCount) || !r.array(rawPoints, (size_t)pointCount * 3) ||
            !r.array(radii, pointCount))
        {
            STRELKA_ERROR("Curve sidecar {}: set '{}' is truncated in its payload", path, materialName);
            return false;
        }

        // A strand of one control point is a point, not a curve, and a linear set
        // whose counts do not add up to the point array is a converter bug that
        // would otherwise present as a corrupt acceleration structure.
        size_t counted = 0;
        for (const uint32_t c : vertexCounts)
        {
            counted += c;
        }
        if (counted != pointCount)
        {
            STRELKA_ERROR("Curve sidecar {}: set '{}' lists {} control points across {} strands but "
                          "carries {}",
                          path, materialName, counted, strandCount, pointCount);
            return false;
        }

        std::vector<glm::float3> points((size_t)pointCount);
        for (size_t i = 0; i < points.size(); ++i)
        {
            points[i] = glm::float3(rawPoints[i * 3 + 0], rawPoints[i * 3 + 1], rawPoints[i * 3 + 2]);
        }

        const Curve::Type type = basis == 0 ? Curve::Type::eLinear : Curve::Type::eCubic;
        const uint32_t curveId = scene.createCurve(type, vertexCounts, points, radii);

        // Uniform strand resolution is what a particle system produces and what
        // lets the shader recover root-to-tip position from a segment index; a
        // mixed set simply loses the gradient rather than being rejected.
        const uint32_t perStrand = strandCount ? vertexCounts[0] : 0;
        bool uniform = strandCount != 0;
        for (const uint32_t c : vertexCounts)
        {
            uniform = uniform && (c == perStrand);
        }
        const uint32_t controlPointsPerSegment = (type == Curve::Type::eLinear) ? 2u : 4u;
        scene.mCurves[curveId].mSegmentsPerStrand =
            (uniform && perStrand >= controlPointsPerSegment) ?
                (perStrand - controlPointsPerSegment + 1u) :
                0u;

        uint32_t materialId = 0;
        const auto it = materialOfName.find(materialName);
        if (it != materialOfName.end())
        {
            materialId = it->second;
        }
        else
        {
            STRELKA_WARNING("Curve sidecar {}: no material named '{}'; the strands take material 0",
                            path, materialName);
        }

        glm::mat4 transform(1.0f);
        std::memcpy(&transform[0][0], xform, sizeof(xform));
        scene.createInstance(Instance::Type::eCurve, curveId, materialId, transform);

        totalStrands += strandCount;
        totalPoints += pointCount;
        ++created;
    }

    STRELKA_INFO("Loaded {} curve set(s) from {}: {} strands, {} control points", created, path,
                 totalStrands, totalPoints);
    return created != 0;
}

} // namespace oka::curvesidecar

