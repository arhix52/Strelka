#include <strelka/sceneloader/alpha_geometry_analysis.h>
#include <strelka/sceneloader/gltfloader.h>

#include <strelka/scene/vertex_packing.h>

#include <host/texture_compress.h>

#include <cxxopts.hpp>
#include <logmanager.h>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using oka::sceneloader::AlphaGeometryAnalysis;
using oka::sceneloader::AlphaGeometryAnalysisOptions;
using oka::sceneloader::AlphaGeometryFilter;
using oka::sceneloader::AlphaGeometryLevelStats;
using oka::sceneloader::AlphaGeometryMode;
using oka::sceneloader::AlphaUvTriangle;

struct ImageAlpha
{
    std::vector<uint8_t> values;
    uint32_t width = 0u;
    uint32_t height = 0u;
};

struct MaterialWork
{
    std::vector<AlphaUvTriangle> triangles;
    uint64_t uniqueMeshMaterialPairs = 0u;
    uint64_t staticInstances = 0u;
};

struct Aggregate
{
    uint64_t sourceTriangles = 0u;
    double sourceWeight = 0.0;
    std::vector<AlphaGeometryLevelStats> levels;
};

struct TexelCoverage
{
    uint64_t transparent = 0u;
    uint64_t opaque = 0u;
    uint64_t fractional = 0u;
};

struct ContourEdges
{
    uint64_t texelEdges = 0u;
    uint64_t collinearRuns = 0u;
};

struct ContourProxy
{
    ContourEdges exterior;
    ContourEdges opaqueBoundary;
};

void decodeBc4AlphaBlock(const uint8_t block[8], uint8_t out[16])
{
    const int a0 = block[0];
    const int a1 = block[1];
    std::array<int, 8> palette{};
    palette[0] = a0;
    palette[1] = a1;
    if (a0 > a1)
    {
        for (int k = 2; k < 8; ++k)
        {
            palette[k] = ((8 - k) * a0 + (k - 1) * a1) / 7;
        }
    }
    else
    {
        for (int k = 2; k < 6; ++k)
        {
            palette[k] = ((6 - k) * a0 + (k - 1) * a1) / 5;
        }
        palette[6] = 0;
        palette[7] = 255;
    }

    uint64_t indices = 0u;
    for (int byte = 0; byte < 6; ++byte)
    {
        indices |= static_cast<uint64_t>(block[2 + byte]) << (8 * byte);
    }
    for (int pixel = 0; pixel < 16; ++pixel)
    {
        out[pixel] = static_cast<uint8_t>(palette[(indices >> (3 * pixel)) & 7u]);
    }
}

ImageAlpha loadAlpha(const std::filesystem::path& path, bool simulateBc3)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    using StbiPtr = std::unique_ptr<stbi_uc, void (*)(void*)>;
    const StbiPtr rgba(stbi_load(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha), stbi_image_free);
    if (!rgba || width <= 0 || height <= 0)
    {
        throw std::runtime_error("could not decode alpha texture: " + path.string());
    }

    const uint64_t pixelCount = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    if (pixelCount > std::numeric_limits<uint32_t>::max())
    {
        throw std::runtime_error("alpha texture is too large for 32-bit integral tables: " + path.string());
    }

    ImageAlpha image;
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.values.resize(static_cast<size_t>(pixelCount));
    if (!simulateBc3)
    {
        for (size_t pixel = 0u; pixel < image.values.size(); ++pixel)
        {
            image.values[pixel] = rgba.get()[pixel * 4u + 3u];
        }
        return image;
    }

    // Metal uploads alpha-bearing base-colour maps as BC3 by default on the
    // current target. Classify the palette values the GPU will actually sample,
    // rather than the source PNG values that disappear after upload.
    for (int by = 0; by < height; by += 4)
    {
        for (int bx = 0; bx < width; bx += 4)
        {
            uint8_t block[8] = {};
            uint8_t decoded[16] = {};
            oka::bc::compressBlockBC4(rgba.get(), width, height, bx, by, static_cast<size_t>(width) * 4u, 3, block);
            decodeBc4AlphaBlock(block, decoded);
            for (int y = 0; y < 4 && by + y < height; ++y)
            {
                for (int x = 0; x < 4 && bx + x < width; ++x)
                {
                    image.values[static_cast<size_t>(by + y) * width + bx + x] = decoded[y * 4 + x];
                }
            }
        }
    }
    return image;
}

glm::float2 transformUv(glm::float2 uv, const MaterialParams& material)
{
    const float c = std::cos(material.uv_rotation);
    const float s = std::sin(material.uv_rotation);
    return { uv.x * material.uv_scale_x * c - uv.y * material.uv_scale_y * s + material.uv_offset_x,
             uv.x * material.uv_scale_x * s + uv.y * material.uv_scale_y * c + material.uv_offset_y };
}

std::filesystem::path resolveTexturePath(const oka::Scene& scene,
                                         const oka::Scene::MaterialDescription& material,
                                         const std::filesystem::path& textureRoot)
{
    std::filesystem::path stored(material.baseColorTexPath);
    if (stored.is_absolute())
    {
        return stored;
    }
    if (!textureRoot.empty())
    {
        return textureRoot / stored;
    }
    return std::filesystem::path(scene.getSourcePath()).parent_path() / stored;
}

std::map<uint32_t, MaterialWork> collectMaterialWork(const oka::Scene& scene)
{
    const auto& meshes = scene.getMeshes();
    const auto& materials = scene.getMaterials();
    const auto& vertices = scene.getVertices();
    const auto& indices = scene.getIndices();

    std::unordered_map<uint64_t, uint64_t> pairInstances;
    pairInstances.reserve(meshes.size());
    for (const oka::Instance& instance : scene.getInstances())
    {
        if (instance.type != oka::Instance::Type::eMesh || instance.isAnimated || instance.mMeshId >= meshes.size() ||
            instance.mMaterialId >= materials.size() || meshes[instance.mMeshId].isSkeletal ||
            materials[instance.mMaterialId].params.alpha_mode == ALPHA_MODE_OPAQUE)
        {
            continue;
        }
        const uint64_t key = (static_cast<uint64_t>(instance.mMeshId) << 32u) | instance.mMaterialId;
        ++pairInstances[key];
    }

    std::map<uint32_t, MaterialWork> work;
    for (const auto& [key, instanceCount] : pairInstances)
    {
        const uint32_t meshId = static_cast<uint32_t>(key >> 32u);
        const uint32_t materialId = static_cast<uint32_t>(key);
        const oka::Mesh& mesh = meshes[meshId];
        if (mesh.mCount % 3u != 0u || static_cast<uint64_t>(mesh.mIndex) + mesh.mCount > indices.size())
        {
            throw std::runtime_error("invalid triangle range in mesh " + std::to_string(meshId));
        }

        MaterialWork& materialWork = work[materialId];
        ++materialWork.uniqueMeshMaterialPairs;
        materialWork.staticInstances += instanceCount;
        materialWork.triangles.reserve(materialWork.triangles.size() + mesh.mCount / 3u);
        const MaterialParams& material = materials[materialId].params;
        for (uint32_t triangle = 0u; triangle < mesh.mCount / 3u; ++triangle)
        {
            AlphaUvTriangle out;
            for (uint32_t corner = 0u; corner < 3u; ++corner)
            {
                const size_t index = static_cast<size_t>(mesh.mIndex) + static_cast<size_t>(triangle) * 3u + corner;
                const uint32_t local = indices[index];
                if (local >= mesh.mVertexCount || static_cast<uint64_t>(mesh.mVbOffset) + local >= vertices.size())
                {
                    throw std::runtime_error("invalid vertex index in mesh " + std::to_string(meshId));
                }
                out.uv[corner] = transformUv(oka::unpackUV(vertices[mesh.mVbOffset + local].uv), material);
            }
            out.candidateWeight = instanceCount;
            materialWork.triangles.push_back(out);
        }
    }
    return work;
}

void add(Aggregate& aggregate, const AlphaGeometryAnalysis& analysis)
{
    aggregate.sourceTriangles += analysis.sourceTriangles;
    aggregate.sourceWeight += analysis.sourceWeight;
    if (aggregate.levels.empty())
    {
        aggregate.levels.resize(analysis.levels.size());
        for (size_t level = 0u; level < aggregate.levels.size(); ++level)
        {
            aggregate.levels[level].subdivisionLevel = static_cast<uint32_t>(level);
        }
    }
    if (aggregate.levels.size() != analysis.levels.size())
    {
        throw std::runtime_error("incompatible alpha analysis levels");
    }
    for (size_t level = 0u; level < analysis.levels.size(); ++level)
    {
        AlphaGeometryLevelStats& dst = aggregate.levels[level];
        const AlphaGeometryLevelStats& src = analysis.levels[level];
        dst.transparentTriangles += src.transparentTriangles;
        dst.opaqueTriangles += src.opaqueTriangles;
        dst.unknownTriangles += src.unknownTriangles;
        dst.survivingTriangles += src.survivingTriangles;
        dst.subdivisionsAccepted += src.subdivisionsAccepted;
        dst.subdivisionsStoppedMixed += src.subdivisionsStoppedMixed;
        dst.subdivisionsStoppedByBudget += src.subdivisionsStoppedByBudget;
        dst.transparentSourceArea += src.transparentSourceArea;
        dst.opaqueSourceArea += src.opaqueSourceArea;
        dst.unknownSourceArea += src.unknownSourceArea;
        dst.transparentWeightedArea += src.transparentWeightedArea;
        dst.opaqueWeightedArea += src.opaqueWeightedArea;
        dst.unknownWeightedArea += src.unknownWeightedArea;
    }
}

double percent(double value, uint64_t total)
{
    return total == 0u ? 0.0 : 100.0 * value / static_cast<double>(total);
}

TexelCoverage classifyTexels(const ImageAlpha& alpha, const AlphaGeometryAnalysisOptions& options)
{
    TexelCoverage result;
    const float factor = std::clamp(options.alphaFactor, 0.0f, 1.0f);
    const float epsilon = std::clamp(options.epsilon, 0.0f, 0.5f);
    const float cutoff = std::clamp(options.alphaCutoff, 0.0f, 1.0f);
    for (const uint8_t texel : alpha.values)
    {
        const float value = factor * (static_cast<float>(texel) / 255.0f);
        if (options.mode == AlphaGeometryMode::Mask)
        {
            value >= cutoff ? ++result.opaque : ++result.transparent;
        }
        else if (value <= epsilon)
        {
            ++result.transparent;
        }
        else if (value >= 1.0f - epsilon)
        {
            ++result.opaque;
        }
        else
        {
            ++result.fractional;
        }
    }
    return result;
}

uint8_t alphaState(uint8_t texel, const AlphaGeometryAnalysisOptions& options)
{
    const float factor = std::clamp(options.alphaFactor, 0.0f, 1.0f);
    const float value = factor * (static_cast<float>(texel) / 255.0f);
    if (options.mode == AlphaGeometryMode::Mask)
    {
        return value >= std::clamp(options.alphaCutoff, 0.0f, 1.0f) ? 2u : 0u;
    }
    const float epsilon = std::clamp(options.epsilon, 0.0f, 0.5f);
    if (value <= epsilon)
    {
        return 0u;
    }
    return value >= 1.0f - epsilon ? 2u : 1u;
}

ContourEdges countContourEdges(const std::vector<uint8_t>& states, uint32_t width, uint32_t height, uint8_t stateMask)
{
    ContourEdges result;
    const auto selected = [stateMask](uint8_t state) { return (stateMask & (1u << state)) != 0u; };
    const auto different = [&](uint32_t ax, uint32_t ay, uint32_t bx, uint32_t by) {
        const bool a = selected(states[static_cast<size_t>(ay) * width + ax]);
        const bool b = selected(states[static_cast<size_t>(by) * width + bx]);
        return a != b;
    };
    const auto countCircularLine = [&](uint32_t length, const auto& edgeAt) {
        if (length == 0u)
        {
            return;
        }
        uint64_t lineEdges = 0u;
        uint64_t lineRuns = 0u;
        bool previous = edgeAt(length - 1u);
        for (uint32_t i = 0u; i < length; ++i)
        {
            const bool current = edgeAt(i);
            lineEdges += current ? 1u : 0u;
            lineRuns += current && !previous ? 1u : 0u;
            previous = current;
        }
        // A boundary spanning the whole repeated row/column has no false-to-true
        // transition, but is still one closed axis-aligned run.
        lineRuns = lineEdges == length ? 1u : lineRuns;
        result.texelEdges += lineEdges;
        result.collinearRuns += lineRuns;
    };

    // REPEAT addressing makes every scan line circular, including the seam.
    for (uint32_t y = 0u; y < height; ++y)
    {
        const uint32_t nextY = (y + 1u) % height;
        countCircularLine(width, [&](uint32_t x) { return different(x, y, x, nextY); });
    }
    for (uint32_t x = 0u; x < width; ++x)
    {
        const uint32_t nextX = (x + 1u) % width;
        countCircularLine(height, [&](uint32_t y) { return different(x, y, nextX, y); });
    }
    return result;
}

ContourProxy analyzeContour(const ImageAlpha& alpha, const AlphaGeometryAnalysisOptions& options)
{
    std::vector<uint8_t> states(alpha.values.size());
    std::ranges::transform(alpha.values, states.begin(), [&](uint8_t texel) { return alphaState(texel, options); });
    ContourProxy result;
    result.exterior = countContourEdges(states, alpha.width, alpha.height, (1u << 1u) | (1u << 2u));
    result.opaqueBoundary = countContourEdges(states, alpha.width, alpha.height, 1u << 2u);
    return result;
}

void printLevelTable(const Aggregate& aggregate, double blasBytesPerTriangle)
{
    std::cout << "\nAggregate adaptive frontier (source-area percentages are comparable across levels):\n"
              << " L       transparent          opaque         unknown       surviving  growth   unknown-x"
                 "  vertex MiB  index MiB  BLAS MiB\n";
    for (const AlphaGeometryLevelStats& stats : aggregate.levels)
    {
        const double growth = aggregate.sourceTriangles == 0u ? 0.0 :
                                                                static_cast<double>(stats.survivingTriangles) /
                                                                    static_cast<double>(aggregate.sourceTriangles);
        const double reduction = stats.unknownSourceArea > 0.0 ?
                                     static_cast<double>(aggregate.sourceTriangles) / stats.unknownSourceArea :
                                     std::numeric_limits<double>::infinity();
        const double vertexMiB =
            static_cast<double>(stats.survivingTriangles) * 3.0 * sizeof(oka::Scene::Vertex) / (1024.0 * 1024.0);
        const double indexMiB =
            static_cast<double>(stats.survivingTriangles) * 3.0 * sizeof(uint32_t) / (1024.0 * 1024.0);
        const double blasMiB = static_cast<double>(stats.survivingTriangles) * blasBytesPerTriangle / (1024.0 * 1024.0);
        std::cout << std::setw(2) << stats.subdivisionLevel << std::fixed << std::setprecision(2) << std::setw(11)
                  << percent(stats.transparentSourceArea, aggregate.sourceTriangles) << "%" << std::setw(15)
                  << percent(stats.opaqueSourceArea, aggregate.sourceTriangles) << "%" << std::setw(15)
                  << percent(stats.unknownSourceArea, aggregate.sourceTriangles) << "%" << std::setw(16)
                  << stats.survivingTriangles << std::setw(8) << growth << "x" << std::setw(11) << reduction << "x"
                  << std::setw(12) << vertexMiB << std::setw(11) << indexMiB << std::setw(10) << blasMiB << '\n';
    }
    if (!aggregate.levels.empty())
    {
        const AlphaGeometryLevelStats& final = aggregate.levels.back();
        std::cout << "Stops at final level: all-four-unknown=" << final.subdivisionsStoppedMixed
                  << ", growth-budget=" << final.subdivisionsStoppedByBudget
                  << ", accepted subdivisions=" << final.subdivisionsAccepted << '\n';
    }
    std::cout << "\nInstance-weighted traversal proxy:\n"
              << " L       transparent          opaque         unknown   unknown-x\n";
    for (const AlphaGeometryLevelStats& stats : aggregate.levels)
    {
        const double reduction = stats.unknownWeightedArea > 0.0 ? aggregate.sourceWeight / stats.unknownWeightedArea :
                                                                   std::numeric_limits<double>::infinity();
        const auto weightedPercent = [&](double value) {
            return aggregate.sourceWeight > 0.0 ? 100.0 * value / aggregate.sourceWeight : 0.0;
        };
        std::cout << std::setw(2) << stats.subdivisionLevel << std::fixed << std::setprecision(2) << std::setw(11)
                  << weightedPercent(stats.transparentWeightedArea) << "%" << std::setw(15)
                  << weightedPercent(stats.opaqueWeightedArea) << "%" << std::setw(15)
                  << weightedPercent(stats.unknownWeightedArea) << "%" << std::setw(12) << reduction << "x\n";
    }
}

} // namespace

int main(int argc, const char* argv[])
{
    const oka::Logmanager logManager;
    cxxopts::Options options("StrelkaAlphaGeometryAnalyzer", "Analyze adaptive alpha-to-geometry feasibility");
    options.add_options()("s,scene", "Scene file (.gltf/.glb)", cxxopts::value<std::string>()->default_value(""))(
        "max-level", "Maximum barycentric subdivision level", cxxopts::value<uint32_t>()->default_value("3"))(
        "growth-limit", "Maximum surviving triangles / source triangles", cxxopts::value<float>()->default_value("3"))(
        "epsilon", "BLEND zero/one tolerance (0 is unbiased)", cxxopts::value<float>()->default_value("0"))(
        "bc3", "Simulate the renderer's default BC3 alpha upload", cxxopts::value<bool>()->default_value("true"))(
        "filter", "Alpha texture filter: auto, nearest, linear", cxxopts::value<std::string>()->default_value("auto"))(
        "continue-all-mixed", "Diagnostic: subdivide even when all four children remain unknown",
        cxxopts::value<bool>()->default_value("false"))(
        "texture-root", "Override root for relative texture paths", cxxopts::value<std::string>()->default_value(""))(
        "blas-bytes-per-triangle", "Heuristic compacted BLAS bytes per surviving triangle",
        cxxopts::value<double>()->default_value("64"))("h,help", "Print usage");
    options.parse_positional({ "scene" });
    options.positional_help("<scene_path>");

    try
    {
        const cxxopts::ParseResult result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help() << '\n';
            return 0;
        }
        const std::string scenePath = result["scene"].as<std::string>();
        if (scenePath.empty())
        {
            throw std::runtime_error("scene path is required");
        }

        oka::Scene scene;
        oka::GltfLoader loader;
        if (!loader.loadGltf(scenePath, scene))
        {
            throw std::runtime_error("scene load failed: " + scenePath);
        }

        const uint32_t maxLevel = result["max-level"].as<uint32_t>();
        const float growthLimit = result["growth-limit"].as<float>();
        const float epsilon = result["epsilon"].as<float>();
        const bool simulateBc3 = result["bc3"].as<bool>();
        const std::string filterName = result["filter"].as<std::string>();
        if (filterName != "auto" && filterName != "nearest" && filterName != "linear")
        {
            throw std::runtime_error("filter must be 'auto', 'nearest', or 'linear'");
        }
        const bool continueAllMixed = result["continue-all-mixed"].as<bool>();
        const double blasBytesPerTriangle = result["blas-bytes-per-triangle"].as<double>();
        const std::filesystem::path textureRoot(result["texture-root"].as<std::string>());
        if (maxLevel > 12u)
        {
            throw std::runtime_error("max-level must be at most 12");
        }
        if (!(blasBytesPerTriangle >= 0.0) || !std::isfinite(blasBytesPerTriangle))
        {
            throw std::runtime_error("blas-bytes-per-triangle must be finite and non-negative");
        }

        const auto work = collectMaterialWork(scene);
        Aggregate aggregate;
        uint64_t staticInstances = 0u;
        uint64_t uniquePairs = 0u;
        for (const auto& entry : work)
        {
            const MaterialWork& materialWork = entry.second;
            staticInstances += materialWork.staticInstances;
            uniquePairs += materialWork.uniqueMeshMaterialPairs;
        }
        std::cout << "Scene: " << scenePath << '\n'
                  << "Static non-opaque materials: " << work.size() << '\n'
                  << "Static non-opaque instances: " << staticInstances << " from " << uniquePairs
                  << " unique mesh/material pairs\n"
                  << "Sampling model: transformed TEXCOORD_0, REPEAT, "
                  << (filterName == "auto" ? "BLEND nearest / MASK linear" : filterName) << " LOD0, "
                  << (simulateBc3 ? "BC3-decoded alpha" : "source alpha") << '\n'
                  << "BLEND epsilon: " << epsilon << ", growth limit: " << growthLimit
                  << "x, all-four-mixed: " << (continueAllMixed ? "continue (diagnostic)" : "stop") << '\n';

        for (const auto& [materialId, materialWork] : work)
        {
            const auto& material = scene.getMaterials().at(materialId);
            ImageAlpha alpha;
            std::filesystem::path texturePath;
            if (material.baseColorTexPath.empty())
            {
                alpha.values = { 255u };
                alpha.width = 1u;
                alpha.height = 1u;
            }
            else
            {
                texturePath = resolveTexturePath(scene, material, textureRoot);
                alpha = loadAlpha(texturePath, simulateBc3);
            }

            AlphaGeometryAnalysisOptions analysisOptions;
            analysisOptions.mode =
                material.params.alpha_mode == ALPHA_MODE_MASK ? AlphaGeometryMode::Mask : AlphaGeometryMode::Blend;
            analysisOptions.filter =
                filterName == "nearest" || (filterName == "auto" && analysisOptions.mode == AlphaGeometryMode::Blend) ?
                    AlphaGeometryFilter::Nearest :
                    AlphaGeometryFilter::Linear;
            analysisOptions.maxSubdivisionLevel = maxLevel;
            analysisOptions.alphaFactor = material.params.base_color_alpha;
            analysisOptions.alphaCutoff = material.params.alpha_cutoff;
            analysisOptions.epsilon = epsilon;
            analysisOptions.growthLimit = growthLimit;
            analysisOptions.stopWhenAllChildrenUnknown = !continueAllMixed;
            const TexelCoverage texels = classifyTexels(alpha, analysisOptions);
            const ContourProxy contour = analyzeContour(alpha, analysisOptions);
            const uint64_t texelCount = texels.transparent + texels.opaque + texels.fractional;
            const AlphaGeometryAnalysis analysis = oka::sceneloader::analyzeAlphaGeometry(
                materialWork.triangles, alpha.values, alpha.width, alpha.height, analysisOptions);
            add(aggregate, analysis);

            const AlphaGeometryLevelStats& final = analysis.levels.back();
            std::cout << "material " << materialId << " '" << material.name << "': " << analysis.sourceTriangles
                      << " source triangles, " << materialWork.staticInstances << " instances, "
                      << materialWork.uniqueMeshMaterialPairs << " unique mesh/material pairs, final " << std::fixed
                      << std::setprecision(2) << percent(final.transparentSourceArea, analysis.sourceTriangles)
                      << "% transparent / " << percent(final.opaqueSourceArea, analysis.sourceTriangles)
                      << "% opaque / " << percent(final.unknownSourceArea, analysis.sourceTriangles) << "% unknown, "
                      << (analysis.sourceWeight > 0.0 ? 100.0 * final.unknownWeightedArea / analysis.sourceWeight : 0.0)
                      << "% instance-weighted unknown, " << final.survivingTriangles << " surviving";
            if (!texturePath.empty())
            {
                std::cout << ", alpha " << alpha.width << 'x' << alpha.height;
            }
            std::cout << ", texels " << percent(static_cast<double>(texels.transparent), texelCount) << "% zero / "
                      << percent(static_cast<double>(texels.opaque), texelCount) << "% one / "
                      << percent(static_cast<double>(texels.fractional), texelCount) << "% fractional";
            const uint64_t contourRuns = contour.exterior.collinearRuns + contour.opaqueBoundary.collinearRuns;
            const uint64_t contourEdges = contour.exterior.texelEdges + contour.opaqueBoundary.texelEdges;
            const double runsPerSourceTriangle = analysis.sourceTriangles == 0u ?
                                                     0.0 :
                                                     static_cast<double>(contourRuns) *
                                                         static_cast<double>(materialWork.uniqueMeshMaterialPairs) /
                                                         static_cast<double>(analysis.sourceTriangles);
            std::cout << ", contour proxy " << contourEdges << " texel edges / " << contourRuns << " collinear runs ("
                      << runsPerSourceTriangle << " runs/source triangle across pairs)\n";
        }

        std::cout << "\nUnique source triangles: " << aggregate.sourceTriangles << '\n';
        printLevelTable(aggregate, blasBytesPerTriangle);
        std::cout << "Geometry bytes assume unshared 32-byte vertices plus uint32 indices (upper bound).\n"
                  << "BLAS bytes are a heuristic; measure the compacted Metal AS before committing the conversion.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
