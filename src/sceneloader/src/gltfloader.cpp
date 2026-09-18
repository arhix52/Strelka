#include <strelka/sceneloader/gltfloader.h>
#include <strelka/sceneloader/sceneserializer.h>
#include <strelka/sceneloader/curve_sidecar.h>
#include <strelka/sceneloader/lod_filter.h>
#include <strelka/sceneloader/light_json.h>
#include <strelka/sceneloader/material_sidecar.h>
#include <strelka/sceneloader/materialx_loader.h>
#include <strelka/sceneloader/static_blas_partition.h>

#include <strelka/scene/camera.h>
#include <strelka/scene/vertex_packing.h>
#include <strelka/scene/light_desc.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <strelka/scene/transform.h>

#include <cctype>
#include <filesystem>
#if !defined(_WIN32)
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif
#include <utility>
#include <fstream>
#include <limits>
#include <env.h>
#include <hugepages.h>
#include <log.h>

namespace fs = std::filesystem;

#include "nlohmann/json.hpp"
#include <numbers>
using json = nlohmann::json;

namespace oka
{
namespace
{

bool gltfDebugLoggingEnabled()
{
    static const bool enabled = envFlag("STRELKA_GLTF_DEBUG");
    return enabled;
}

// Only the mesh is dropped when a node is filtered. The node still exists and
// its children are still walked, so the hierarchy, cameras and skins are
// unaffected. The rule itself is in lod_filter.h, where it can be tested.
bool lodFilterEnabled()
{
    static const bool enabled = !envFlag("STRELKA_NO_LOD_FILTER");
    return enabled;
}

// Reported once per load: silently dropping geometry is exactly the kind of
// thing that must not be discovered by wondering where an object went.
uint32_t& lodSkipCounter()
{
    static uint32_t skipped = 0;
    return skipped;
}

// packNormal(), packUV(), unpackNormal(), unpackUV() provided by <strelka/scene/vertex_packing.h>
// packTangent uses same format as packNormal (tangents are unit vectors in [-1,1])

void computeTangent(Scene::Vertex* vertices, const uint32_t* indices, size_t indexCount)
{
    const size_t lastIndex = indexCount;
    Scene::Vertex& v0 = vertices[indices[lastIndex - 3]];
    Scene::Vertex& v1 = vertices[indices[lastIndex - 2]];
    Scene::Vertex& v2 = vertices[indices[lastIndex - 1]];

    const glm::float2 uv0 = unpackUV(v0.uv);
    const glm::float2 uv1 = unpackUV(v1.uv);
    const glm::float2 uv2 = unpackUV(v2.uv);

    const glm::float3 deltaPos1 = v1.pos - v0.pos;
    const glm::float3 deltaPos2 = v2.pos - v0.pos;
    const glm::vec2 deltaUV1 = uv1 - uv0;
    const glm::vec2 deltaUV2 = uv2 - uv0;

    glm::vec3 tangent{ 0.0f, 0.0f, 1.0f };
    const float d = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;
    if (abs(d) > 1e-6)
    {
        const float r = 1.0f / d;
        tangent = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r;
    }

    const uint32_t packedTangent = packNormal(tangent);

    v0.tangent = packedTangent;
    v1.tangent = packedTangent;
    v2.tangent = packedTangent;
}

// Maps a glTF (mesh, primitive) onto the oka mesh built for it, so geometry
// referenced by many nodes is parsed and uploaded once.
using MeshCache = std::unordered_map<uint64_t, uint32_t>;
size_t gpuInstanceCount(const tinygltf::Model& model, const tinygltf::Node& node);

struct AnimationUsage
{
    std::vector<uint8_t> potentiallyAnimatedNodes;
    std::vector<uint8_t> deformingSkins;
    std::vector<uint8_t> reusableBindPoseNodes;
};

glm::mat4 gltfNodeTransform(const tinygltf::Node& node)
{
    if (!node.matrix.empty())
    {
        return glm::make_mat4(node.matrix.data());
    }
    const glm::vec3 translation = node.translation.empty() ?
                                      glm::vec3(0.0f) :
                                      glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
    const glm::vec3 scale = node.scale.empty() ? glm::vec3(1.0f) : glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
    const glm::quat rotation = node.rotation.empty() ?
                                   glm::quat(1.0f, 0.0f, 0.0f, 0.0f) :
                                   glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

bool approximatelyIdentity(const glm::mat4& matrix)
{
    for (int column = 0; column < 4; ++column)
    {
        for (int row = 0; row < 4; ++row)
        {
            const float expected = column == row ? 1.0f : 0.0f;
            if (!std::isfinite(matrix[column][row]) || std::abs(matrix[column][row] - expected) > 1e-4f)
            {
                return false;
            }
        }
    }
    return true;
}

AnimationUsage analyzeAnimationUsage(const tinygltf::Model& model)
{
    AnimationUsage usage;
    usage.potentiallyAnimatedNodes.assign(model.nodes.size(), 0);
    std::vector<int> stack;
    for (const tinygltf::Animation& animation : model.animations)
    {
        for (const tinygltf::AnimationChannel& channel : animation.channels)
        {
            if (channel.target_node >= 0 && static_cast<size_t>(channel.target_node) < model.nodes.size())
            {
                stack.push_back(channel.target_node);
            }
        }
    }
    while (!stack.empty())
    {
        const int nodeId = stack.back();
        stack.pop_back();
        if (usage.potentiallyAnimatedNodes[nodeId])
        {
            continue;
        }
        usage.potentiallyAnimatedNodes[nodeId] = 1;
        for (const int child : model.nodes[nodeId].children)
        {
            if (child >= 0 && static_cast<size_t>(child) < model.nodes.size())
            {
                stack.push_back(child);
            }
        }
    }

    usage.deformingSkins.assign(model.skins.size(), 0);
    for (size_t skinId = 0; skinId < model.skins.size(); ++skinId)
    {
        for (const int joint : model.skins[skinId].joints)
        {
            if (joint >= 0 && static_cast<size_t>(joint) < usage.potentiallyAnimatedNodes.size() &&
                usage.potentiallyAnimatedNodes[joint])
            {
                usage.deformingSkins[skinId] = 1;
                break;
            }
        }
    }

    std::vector<int> parents(model.nodes.size(), -1);
    for (size_t nodeId = 0; nodeId < model.nodes.size(); ++nodeId)
    {
        for (const int child : model.nodes[nodeId].children)
        {
            if (child >= 0 && static_cast<size_t>(child) < parents.size())
            {
                parents[child] = static_cast<int>(nodeId);
            }
        }
    }
    std::vector<glm::mat4> globals(model.nodes.size(), glm::mat4(1.0f));
    std::vector<uint8_t> globalReady(model.nodes.size(), 0);
    std::function<glm::mat4(size_t)> globalOf = [&](size_t nodeId) {
        if (globalReady[nodeId])
        {
            return globals[nodeId];
        }
        const glm::mat4 local = gltfNodeTransform(model.nodes[nodeId]);
        const int parent = parents[nodeId];
        globals[nodeId] = parent >= 0 ? globalOf(static_cast<size_t>(parent)) * local : local;
        globalReady[nodeId] = 1;
        return globals[nodeId];
    };
    for (size_t nodeId = 0; nodeId < model.nodes.size(); ++nodeId)
    {
        globalOf(nodeId);
    }

    usage.reusableBindPoseNodes.assign(model.nodes.size(), 0);
    for (size_t nodeId = 0; nodeId < model.nodes.size(); ++nodeId)
    {
        const tinygltf::Node& node = model.nodes[nodeId];
        if (node.skin < 0 || static_cast<size_t>(node.skin) >= model.skins.size() || usage.deformingSkins[node.skin] ||
            usage.potentiallyAnimatedNodes[nodeId])
        {
            continue;
        }
        const tinygltf::Skin& skin = model.skins[node.skin];
        if (skin.inverseBindMatrices < 0 || static_cast<size_t>(skin.inverseBindMatrices) >= model.accessors.size())
        {
            continue;
        }
        const tinygltf::Accessor& accessor = model.accessors[skin.inverseBindMatrices];
        if (accessor.bufferView < 0 || static_cast<size_t>(accessor.bufferView) >= model.bufferViews.size() ||
            accessor.count < skin.joints.size() || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
            accessor.type != TINYGLTF_TYPE_MAT4 || accessor.sparse.isSparse || skin.joints.empty())
        {
            continue;
        }
        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
        if (view.buffer < 0 || static_cast<size_t>(view.buffer) >= model.buffers.size())
        {
            continue;
        }
        const int byteStride = accessor.ByteStride(view);
        const std::vector<unsigned char>& buffer = model.buffers[view.buffer].data;
        if (byteStride < static_cast<int>(sizeof(glm::mat4)) || view.byteOffset > buffer.size() ||
            view.byteLength > buffer.size() - view.byteOffset || accessor.byteOffset > view.byteLength)
        {
            continue;
        }
        const size_t stride = static_cast<size_t>(byteStride);
        const size_t available = view.byteLength - accessor.byteOffset;
        if (available < sizeof(glm::mat4) || skin.joints.size() - 1 > (available - sizeof(glm::mat4)) / stride)
        {
            continue;
        }
        const unsigned char* bytes = buffer.data() + view.byteOffset + accessor.byteOffset;
        const glm::mat4 worldToMesh = glm::inverse(globals[nodeId]);
        bool identity = true;
        for (size_t jointIndex = 0; jointIndex < skin.joints.size(); ++jointIndex)
        {
            const int joint = skin.joints[jointIndex];
            if (joint < 0 || static_cast<size_t>(joint) >= globals.size())
            {
                identity = false;
                break;
            }
            glm::mat4 inverseBind(1.0f);
            std::memcpy(&inverseBind, bytes + jointIndex * stride, sizeof(inverseBind));
            identity &= approximatelyIdentity(worldToMesh * globals[joint] * inverseBind);
        }
        usage.reusableBindPoseNodes[nodeId] = identity;
    }
    return usage;
}

bool nodeHasDeformingSkin(const tinygltf::Node& node, size_t nodeId, const AnimationUsage& usage)
{
    if (node.skin < 0 || static_cast<size_t>(node.skin) >= usage.deformingSkins.size())
    {
        return false;
    }
    return usage.deformingSkins[node.skin] || nodeId >= usage.reusableBindPoseNodes.size() ||
           !usage.reusableBindPoseNodes[nodeId];
}

bool nodeSharesDeformedPose(const tinygltf::Node& node)
{
    if (!node.extras.IsObject() || !node.extras.Has("strelka_shared_pose"))
    {
        return false;
    }
    const tinygltf::Value& marker = node.extras.Get("strelka_shared_pose");
    return marker.IsBool() && marker.Get<bool>();
}

void countModelGeometry(const tinygltf::Model& model,
                        const AnimationUsage& animationUsage,
                        size_t& vertexCount,
                        size_t& indexCount,
                        size_t& skinCount,
                        size_t& meshCount)
{
    vertexCount = 0;
    indexCount = 0;
    skinCount = 0;
    meshCount = 0;
    std::unordered_set<uint64_t> cachedRigidPrimitives;
    std::unordered_set<uint64_t> cachedSharedPosePrimitives;
    for (size_t nodeId = 0; nodeId < model.nodes.size(); ++nodeId)
    {
        const tinygltf::Node& node = model.nodes[nodeId];
        if (node.mesh < 0 || static_cast<size_t>(node.mesh) >= model.meshes.size() ||
            (lodFilterEnabled() && isProxyOrLowerLod(node.name)))
        {
            continue;
        }
        const bool deforming = nodeHasDeformingSkin(node, nodeId, animationUsage);
        const bool sharedPose = deforming && nodeSharesDeformedPose(node);
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); ++primitiveIndex)
        {
            const tinygltf::Primitive& primitive = mesh.primitives[primitiveIndex];
            const auto pos = primitive.attributes.find("POSITION");
            if (pos == primitive.attributes.end())
            {
                continue;
            }
            const uint64_t primitiveKey = (static_cast<uint64_t>(node.mesh) << 32u) | primitiveIndex;
            // EXT_mesh_gpu_instancing copies of one skinned node use the same
            // palette, so they can share the deformed mesh and its BLAS.
            const size_t copies =
                deforming ? (sharedPose ? (cachedSharedPosePrimitives.insert(primitiveKey).second ? 1u : 0u) : 1u) :
                            (cachedRigidPrimitives.insert(primitiveKey).second ? 1u : 0u);
            if (copies == 0)
            {
                continue;
            }
            const size_t n = model.accessors[pos->second].count;
            vertexCount += copies * n;
            if (primitive.indices >= 0)
            {
                indexCount += copies * model.accessors[primitive.indices].count;
            }
            if (deforming && primitive.attributes.count("JOINTS_0") != 0 && primitive.attributes.count("WEIGHTS_0") != 0)
            {
                skinCount += copies * n;
            }
            meshCount += copies;
        }
    }
}

// glTF exposes accessor payloads as byte arrays. Component metadata and stride
// validation above each view establish the typed interpretation used here.
// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
namespace
{

template <typename Fn>
void parallelFor(size_t count, Fn&& body)
{
    constexpr size_t kSerialBelow = size_t{ 64 } * 1024;
    const unsigned hardware = std::thread::hardware_concurrency();
    const size_t threads = count < kSerialBelow ? 1u :
                                                  std::min<size_t>(hardware == 0u ? 1u : hardware,
                                                                   (count + kSerialBelow - 1u) / kSerialBelow);
    if (threads <= 1u)
    {
        body(size_t{ 0 }, count);
        return;
    }

    const size_t chunk = (count + threads - 1u) / threads;
    std::vector<std::thread> workers;
    workers.reserve(threads - 1u);
    for (size_t t = 1u; t < threads; ++t)
    {
        const size_t begin = std::min(count, t * chunk);
        const size_t end = std::min(count, begin + chunk);
        if (begin == end)
        {
            break;
        }
        workers.emplace_back([&body, begin, end]() { body(begin, end); });
    }
    body(size_t{ 0 }, std::min(count, chunk));
    for (std::thread& worker : workers)
    {
        worker.join();
    }
}

} // namespace

void processPrimitive(const tinygltf::Model& model,
                      oka::Scene& scene,
                      const uint32_t parentNodeId,
                      const tinygltf::Primitive& primitive,
                      const glm::float4x4& transform,
                      const float globalScale,
                      MeshCache& meshCache,
                      uint64_t primitiveKey,
                      bool deformingSkin,
                      bool sharedPose)
{
    using namespace std;
    assert(primitive.attributes.find("POSITION") != primitive.attributes.end());

    if (!deformingSkin || sharedPose)
    {
        const auto cached = meshCache.find(primitiveKey);
        if (cached != meshCache.end())
        {
            int cachedMatId = primitive.material;
            if (cachedMatId == -1)
                cachedMatId = 0;
            const uint32_t instId = scene.createInstance(Instance::Type::eMesh, cached->second, cachedMatId, transform);
            scene.mNodes[parentNodeId].instanceIds.push_back(instId);
            return;
        }
    }

    const tinygltf::Accessor& positionAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
    const tinygltf::BufferView& positionView = model.bufferViews[positionAccessor.bufferView];
    const auto* positionData = reinterpret_cast<const float*>(
        &model.buffers[positionView.buffer].data[positionAccessor.byteOffset + positionView.byteOffset]);
    assert(positionData != nullptr);
    const auto vertexCount = static_cast<uint32_t>(positionAccessor.count);
    assert(vertexCount != 0);
    const int byteStride = positionAccessor.ByteStride(positionView);
    assert(byteStride > 0); // -1 means invalid glTF
    const int posStride = byteStride / static_cast<int>(sizeof(float));

    // Normals
    const float* normalsData = nullptr;
    int normalStride = 0;
    if (primitive.attributes.find("NORMAL") != primitive.attributes.end())
    {
        const tinygltf::Accessor& normalAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
        const tinygltf::BufferView& normView = model.bufferViews[normalAccessor.bufferView];
        normalsData = reinterpret_cast<const float*>(
            &(model.buffers[normView.buffer].data[normalAccessor.byteOffset + normView.byteOffset]));
        assert(normalsData != nullptr);
        normalStride = normalAccessor.ByteStride(normView) / static_cast<int>(sizeof(float));
        assert(normalStride > 0);
    }

    // UVs
    const float* texCoord0Data = nullptr;
    int texCoord0Stride = 0;
    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
    {
        const tinygltf::Accessor& uvAccessor = model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
        const tinygltf::BufferView& uvView = model.bufferViews[uvAccessor.bufferView];
        texCoord0Data = reinterpret_cast<const float*>(
            &(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
        texCoord0Stride = uvAccessor.ByteStride(uvView) / static_cast<int>(sizeof(float));
    }

    // Tangents. vec4: xyz is the tangent, w the bitangent handedness. Exporters
    // emit this whenever a normal map is in play (Blender emits it always), and
    // it is per-vertex and smooth, unlike the per-triangle fallback below.
    const float* tangentData = nullptr;
    int tangentStride = 0;
    if (primitive.attributes.find("TANGENT") != primitive.attributes.end())
    {
        const tinygltf::Accessor& tanAccessor = model.accessors[primitive.attributes.find("TANGENT")->second];
        const tinygltf::BufferView& tanView = model.bufferViews[tanAccessor.bufferView];
        tangentData = reinterpret_cast<const float*>(
            &(model.buffers[tanView.buffer].data[tanAccessor.byteOffset + tanView.byteOffset]));
        tangentStride = tanAccessor.ByteStride(tanView) / static_cast<int>(sizeof(float));
        assert(tangentStride > 0);
    }

    // Vertex colours. glTF allows VEC3 or VEC4, as float or as normalised
    // unsigned byte/short, and the values are linear multipliers on base colour.
    const void* colorData = nullptr;
    int colorStride = 0; // in components, not bytes
    int colorComponents = 4;
    int colorComponentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    if (primitive.attributes.find("COLOR_0") != primitive.attributes.end())
    {
        const tinygltf::Accessor& ca = model.accessors[primitive.attributes.find("COLOR_0")->second];
        const tinygltf::BufferView& cv = model.bufferViews[ca.bufferView];
        colorData = reinterpret_cast<const void*>(&model.buffers[cv.buffer].data[ca.byteOffset + cv.byteOffset]);
        colorComponents = ca.type == TINYGLTF_TYPE_VEC3 ? 3 : 4;
        colorComponentType = ca.componentType;
        const int elemSize = colorComponentType == TINYGLTF_COMPONENT_TYPE_FLOAT          ? 4 :
                             colorComponentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 2 :
                                                                                            1;
        colorStride = ca.ByteStride(cv) / elemSize;
        assert(colorStride > 0);
    }

    int matId = primitive.material;
    if (matId == -1)
    {
        matId = 0; // TODO: should be index of default material
    }

    // skinning joints & weights
    const void* jointsData = nullptr;
    int jointsStride = 0;
    const float* weightsData = nullptr;
    int weightsStride = 0;
    bool hasJoints = false;
    std::vector<oka::Scene::vertexSkinData> sb;
    if (deformingSkin && (primitive.attributes.find("JOINTS_0") != primitive.attributes.end()) &&
        (primitive.attributes.find("WEIGHTS_0") != primitive.attributes.end()))
    {
        hasJoints = true;
        const tinygltf::Accessor& jointsAccessor = model.accessors[primitive.attributes.find("JOINTS_0")->second];
        const tinygltf::BufferView& jointsView = model.bufferViews[jointsAccessor.bufferView];
        switch (jointsAccessor.componentType)
        {
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
            jointsData = reinterpret_cast<const void*>(
                &model.buffers[jointsView.buffer].data[jointsAccessor.byteOffset + jointsView.byteOffset]);
            jointsStride = jointsAccessor.ByteStride(jointsView) / static_cast<int>(sizeof(uint32_t));
            assert(jointsData != nullptr);
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
            jointsData = reinterpret_cast<const void*>(
                &model.buffers[jointsView.buffer].data[jointsAccessor.byteOffset + jointsView.byteOffset]);
            jointsStride = jointsAccessor.ByteStride(jointsView) / static_cast<int>(sizeof(uint16_t));
            assert(jointsData != nullptr);
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
            jointsData = reinterpret_cast<const void*>(
                &model.buffers[jointsView.buffer].data[jointsAccessor.byteOffset + jointsView.byteOffset]);
            jointsStride = jointsAccessor.ByteStride(jointsView) / static_cast<int>(sizeof(uint8_t));
            assert(jointsData != nullptr);
            break;
        }
        default:
            STRELKA_WARNING(
                "glTF joint component type {} is not supported; skipping primitive", jointsAccessor.componentType);
            return;
        }
        assert(jointsStride > 0);

        const tinygltf::Accessor& weightsAccessor = model.accessors[primitive.attributes.find("WEIGHTS_0")->second];
        const tinygltf::BufferView& weightsView = model.bufferViews[weightsAccessor.bufferView];
        weightsData = reinterpret_cast<const float*>(
            &model.buffers[weightsView.buffer].data[weightsAccessor.byteOffset + weightsView.byteOffset]);
        assert(weightsData != nullptr);
        weightsStride = weightsAccessor.ByteStride(weightsView) / static_cast<int>(sizeof(float));
        assert(weightsStride > 0);

        sb.reserve(vertexCount);
    }

    auto& vertices = scene.getVertices();
    auto& indicesOut = scene.getIndices();
    const uint32_t vbOffset = static_cast<uint32_t>(vertices.size());
    vertices.resize(size_t(vbOffset) + vertexCount);

    // One vertex, from whatever attributes this primitive turned out to have.
    // The position and normal come back out because the skin below needs them
    // unpacked, and packing is lossy.
    const auto buildVertex = [&](size_t v, glm::float3& outPos, glm::float3& outNorm) -> oka::Scene::Vertex {
        oka::Scene::Vertex vertex{};
        const glm::float3 vPos = glm::make_vec3(&positionData[v * posStride]) * globalScale;
        const glm::float3 vNorm =
            glm::vec3(normalsData ? glm::make_vec3(&normalsData[v * normalStride]) : glm::vec3(0.0f));
        vertex.pos = vPos;
        vertex.normal = packNormal(glm::normalize(vNorm));
        vertex.uv = packUV(texCoord0Data ? glm::make_vec2(&texCoord0Data[v * texCoord0Stride]) : glm::vec3(0.0f));
        if (colorData)
        {
            glm::float4 c(1.0f);
            switch (colorComponentType)
            {
            case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                const float* src = static_cast<const float*>(colorData) + v * colorStride;
                for (int k = 0; k < colorComponents; ++k)
                    c[k] = src[k];
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                const uint8_t* src = static_cast<const uint8_t*>(colorData) + v * colorStride;
                for (int k = 0; k < colorComponents; ++k)
                    c[k] = static_cast<float>(src[k]) / 255.0f;
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                const uint16_t* src = static_cast<const uint16_t*>(colorData) + v * colorStride;
                for (int k = 0; k < colorComponents; ++k)
                    c[k] = static_cast<float>(src[k]) / 65535.0f;
                break;
            }
            default:
                break; // leave white
            }
            vertex.color = packColor(c);
        }
        if (tangentData)
        {
            const float* t = &tangentData[v * tangentStride];
            const glm::float3 tan{ t[0], t[1], t[2] };
            const float lenSq = glm::dot(tan, tan);
            vertex.tangent = packTangent(lenSq > 1e-12f ? tan * glm::inversesqrt(lenSq) : glm::float3(0, 0, 1), t[3]);
        }
        outPos = vPos;
        outNorm = vNorm;
        return vertex;
    };

    if (hasJoints)
    {
        // Sequential: the skin is appended in vertex order, and an unsupported
        // joint component type abandons the primitive from inside the loop.
        for (size_t v = 0; v < vertexCount; ++v)
        {
            glm::float3 vPos{ 0.0f };
            glm::float3 vNorm{ 0.0f };
            vertices[size_t(vbOffset) + v] = buildVertex(v, vPos, vNorm);
            oka::Scene::vertexSkinData skinData{};
            const tinygltf::Accessor& jointsAccessor = model.accessors[primitive.attributes.find("JOINTS_0")->second];
            switch (jointsAccessor.componentType)
            {
            case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
                const uint32_t* jointsDataCasted = static_cast<const uint32_t*>(jointsData);
                skinData.joints = glm::ivec4(static_cast<int>(jointsDataCasted[v * jointsStride + 0]),
                                             static_cast<int>(jointsDataCasted[v * jointsStride + 1]),
                                             static_cast<int>(jointsDataCasted[v * jointsStride + 2]),
                                             static_cast<int>(jointsDataCasted[v * jointsStride + 3]));
                break;
            }
            case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
                const uint16_t* jointsDataCasted = static_cast<const uint16_t*>(jointsData);
                skinData.joints = glm::ivec4(static_cast<int>(jointsDataCasted[v * jointsStride + 0]),
                                             static_cast<int>(jointsDataCasted[v * jointsStride + 1]),
                                             static_cast<int>(jointsDataCasted[v * jointsStride + 2]),
                                             static_cast<int>(jointsDataCasted[v * jointsStride + 3]));
                break;
            }
            case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
                const uint8_t* jointsDataCasted = static_cast<const uint8_t*>(jointsData);
                skinData.joints = glm::ivec4(static_cast<int>(jointsDataCasted[v * jointsStride + 0]),
                                             static_cast<int>(jointsDataCasted[v * jointsStride + 1]),
                                             static_cast<int>(jointsDataCasted[v * jointsStride + 2]),
                                             static_cast<int>(jointsDataCasted[v * jointsStride + 3]));
                break;
            }
            default:
                STRELKA_WARNING(
                    "glTF joint component type {} is not supported; skipping primitive", jointsAccessor.componentType);
                return;
            }
            skinData.weights = glm::make_vec4(&weightsData[v * weightsStride]);
            skinData.pos = vPos;
            skinData.normal = vNorm;
            sb.push_back(skinData);
        }
    }
    else
    {
        parallelFor(vertexCount, [&](size_t begin, size_t end) {
            for (size_t v = begin; v < end; ++v)
            {
                glm::float3 vPos{ 0.0f };
                glm::float3 vNorm{ 0.0f };
                vertices[size_t(vbOffset) + v] = buildVertex(v, vPos, vNorm);
            }
        });
    }
    uint32_t indexCount = 0;
    const uint32_t ibOffset = static_cast<uint32_t>(indicesOut.size());
    const bool hasIndices = (primitive.indices != -1);
    assert(hasIndices); // currently support only this mode
    if (hasIndices)
    {
        const tinygltf::Accessor& accessor = model.accessors[primitive.indices > -1 ? primitive.indices : 0];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

        indexCount = static_cast<uint32_t>(accessor.count);
        assert(indexCount != 0 && (indexCount % 3 == 0));
        const void* dataPtr = &(buffer.data[accessor.byteOffset + bufferView.byteOffset]);

        const size_t indexStart = indicesOut.size();
        indicesOut.resize(indexStart + indexCount);
        uint32_t* dst = indicesOut.data() + indexStart;
        switch (accessor.componentType)
        {
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
            const auto* buf = static_cast<const uint32_t*>(dataPtr);
            std::memcpy(dst, buf, static_cast<size_t>(indexCount) * sizeof(uint32_t));
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
            const auto* buf = static_cast<const uint16_t*>(dataPtr);
            for (uint32_t index = 0; index < indexCount; ++index)
            {
                dst[index] = buf[index];
            }
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
            const auto* buf = static_cast<const uint8_t*>(dataPtr);
            for (uint32_t index = 0; index < indexCount; ++index)
            {
                dst[index] = buf[index];
            }
            break;
        }
        default:
            STRELKA_WARNING("glTF index component type {} is not supported; skipping primitive", accessor.componentType);
            vertices.resize(vbOffset);
            indicesOut.resize(ibOffset);
            return;
        }
        if (!tangentData)
        {
            computeTangent(vertices.data() + vbOffset, indicesOut.data() + ibOffset, indexCount);
        }
    }

    uint32_t sbOffset = 0;
    if (hasJoints)
    {
        auto& skinOut = scene.getVerticesSkinData();
        sbOffset = static_cast<uint32_t>(skinOut.size());
        for (size_t v = 0; v < sb.size(); ++v)
        {
            sb[v].tangent = vertices[vbOffset + v].tangent;
        }
        skinOut.insert(skinOut.end(), sb.begin(), sb.end());
    }

    std::vector<Mesh::StaticBlasPartition> staticBlasPartitions;
    const uint32_t triangleCount = indexCount / 3u;
    if (!hasJoints && triangleCount > Mesh::kMaxStaticBlasTriangles)
    {
        const std::span<const Scene::Vertex> primitiveVertices(vertices.data() + vbOffset, vertexCount);
        const std::span<uint32_t> primitiveIndices(indicesOut.data() + ibOffset, indexCount);
        staticBlasPartitions =
            sceneloader::partitionStaticTriangles(primitiveVertices, primitiveIndices, Mesh::kMaxStaticBlasTriangles);
        STRELKA_INFO("Spatially partitioned static glTF primitive {}:{}: {} triangles into {} BLAS ranges",
                     primitiveKey >> 32u, primitiveKey & 0xffffffffu, triangleCount, staticBlasPartitions.size());
    }

    uint32_t meshId = std::numeric_limits<uint32_t>::max();
    if (hasJoints)
        meshId = scene.createSkeletalMeshFromOffsets(
            vbOffset, vertexCount, ibOffset, indexCount, sbOffset, static_cast<uint32_t>(sb.size()));
    else
        meshId = scene.createMeshFromOffsets(vbOffset, vertexCount, ibOffset, indexCount);
    assert(meshId != std::numeric_limits<uint32_t>::max());
    scene.mMeshes[meshId].mStaticBlasPartitions = std::move(staticBlasPartitions);
    // Independently deforming nodes need separate writable vertex ranges.
    // Rigid and reusable bind-pose nodes can share immutable geometry.
    if (!deformingSkin || sharedPose)
    {
        meshCache.emplace(primitiveKey, meshId);
    }
    const uint32_t instId = scene.createInstance(Instance::Type::eMesh, meshId, matId, transform);
    assert(instId != std::numeric_limits<uint32_t>::max());
    scene.mNodes[parentNodeId].instanceIds.push_back(instId);
}
// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

void processMesh(const tinygltf::Model& model,
                 oka::Scene& scene,
                 const uint32_t parentNodeId,
                 const tinygltf::Mesh& mesh,
                 const glm::float4x4& transform,
                 const float globalScale,
                 MeshCache& meshCache,
                 uint32_t meshIndex,
                 bool deformingSkin,
                 bool sharedPose)
{
    if (gltfDebugLoggingEnabled())
    {
        STRELKA_DEBUG("glTF mesh '{}' has {} primitives", mesh.name, mesh.primitives.size());
    }
    uint64_t primitiveIndex = 0;
    for (const auto& primitive : mesh.primitives)
    {
        processPrimitive(model, scene, parentNodeId, primitive, transform, globalScale, meshCache,
                         ((uint64_t)meshIndex << 32) | primitiveIndex++, deformingSkin, sharedPose);
    }
}

glm::float4x4 getTransform(const tinygltf::Node& node, const float globalScale)
{
    if (node.matrix.empty())
    {
        glm::float3 scale{ 1.0f };
        if (!node.scale.empty())
        {
            scale = glm::float3((float)node.scale[0], (float)node.scale[1], (float)node.scale[2]);
            // check that scale is uniform, otherwise we have to support it in shader
            // assert(scale.x == scale.y && scale.y == scale.z);
        }

        glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        if (!node.rotation.empty())
        {
            rotation = quatFromGltf(
                (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2], (float)node.rotation[3]);
        }

        glm::float3 translation{ 0.0f };
        if (!node.translation.empty())
        {
            translation = glm::float3((float)node.translation[0], (float)node.translation[1], (float)node.translation[2]);
            translation *= globalScale;
        }

        const glm::float4x4 translationMatrix = glm::translate(glm::float4x4(1.0f), translation);
        const glm::float4x4 rotationMatrix{ rotation };
        const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), scale);

        const glm::float4x4 localTransform = translationMatrix * rotationMatrix * scaleMatrix;

        return localTransform;
    }
    else
    {
        glm::float4x4 localTransform = glm::make_mat4(node.matrix.data());
        return localTransform;
    }
}

// Per-instance transforms from EXT_mesh_gpu_instancing, empty when the node has
// none. Floats only: the extension permits normalised integer rotations, which
// nothing here writes, and silently misreading them would be worse than saying so.
void readGpuInstancing(const tinygltf::Model& model, const tinygltf::Node& node, std::vector<glm::float4x4>& out)
{
    const auto ext = node.extensions.find("EXT_mesh_gpu_instancing");
    if (ext == node.extensions.end() || !ext->second.Has("attributes"))
        return;
    const tinygltf::Value& attributes = ext->second.Get("attributes");

    auto findAccessor = [&](const char* name) -> const tinygltf::Accessor* {
        if (!attributes.Has(name))
            return nullptr;
        const int index = attributes.Get(name).GetNumberAsInt();
        if (index < 0 || (size_t)index >= model.accessors.size())
            return nullptr;
        const tinygltf::Accessor& accessor = model.accessors[index];
        if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
        {
            STRELKA_WARNING("EXT_mesh_gpu_instancing {} is not float; ignoring", name);
            return nullptr;
        }
        return &accessor;
    };
    auto bytesOf = [&](const tinygltf::Accessor& accessor) -> const unsigned char* {
        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
        return model.buffers[view.buffer].data.data() + view.byteOffset + accessor.byteOffset;
    };
    auto strideOf = [&](const tinygltf::Accessor& accessor, int components) -> size_t {
        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
        const int stride = accessor.ByteStride(view);
        return stride ? static_cast<size_t>(stride) : sizeof(float) * static_cast<size_t>(components);
    };

    const tinygltf::Accessor* translation = findAccessor("TRANSLATION");
    const tinygltf::Accessor* rotation = findAccessor("ROTATION");
    const tinygltf::Accessor* scale = findAccessor("SCALE");
    const size_t count =
        std::max({ translation ? translation->count : 0, rotation ? rotation->count : 0, scale ? scale->count : 0 });
    if (count == 0)
        return;

    const unsigned char* tBase = translation ? bytesOf(*translation) : nullptr;
    const unsigned char* rBase = rotation ? bytesOf(*rotation) : nullptr;
    const unsigned char* sBase = scale ? bytesOf(*scale) : nullptr;
    const size_t tStride = translation ? strideOf(*translation, 3) : 0;
    const size_t rStride = rotation ? strideOf(*rotation, 4) : 0;
    const size_t sStride = scale ? strideOf(*scale, 3) : 0;

    // Written straight into the output. Three float vectors the size of the
    // forest -- 1.1 million placements -- were a 45 MB staging copy of data
    // that is only ever read once.
    out.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        glm::float3 t(0.0f);
        if (tBase)
        {
            std::memcpy(&t, tBase + i * tStride, sizeof(t));
        }
        glm::quat r(1.0f, 0.0f, 0.0f, 0.0f);
        if (rBase)
        {
            float q[4];
            std::memcpy(q, rBase + i * rStride, sizeof(q));
            // glTF stores a quaternion xyzw; glm::quat takes w first.
            r = glm::quat(q[3], q[0], q[1], q[2]);
        }
        glm::float3 s(1.0f);
        if (sBase)
        {
            std::memcpy(&s, sBase + i * sStride, sizeof(s));
        }
        out[i] = glm::translate(glm::float4x4(1.0f), t) * glm::mat4_cast(r) * glm::scale(glm::float4x4(1.0f), s);
    }
}

size_t gpuInstanceCount(const tinygltf::Model& model, const tinygltf::Node& node)
{
    const auto ext = node.extensions.find("EXT_mesh_gpu_instancing");
    if (ext == node.extensions.end() || !ext->second.Has("attributes"))
        return 0;
    const tinygltf::Value& attributes = ext->second.Get("attributes");
    auto countOf = [&](const char* name) -> size_t {
        if (!attributes.Has(name))
            return 0;
        const int index = attributes.Get(name).GetNumberAsInt();
        if (index < 0 || (size_t)index >= model.accessors.size())
            return 0;
        return model.accessors[index].count;
    };
    return std::max({ countOf("TRANSLATION"), countOf("ROTATION"), countOf("SCALE") });
}

void processNode(const tinygltf::Model& model,
                 oka::Scene& scene,
                 const tinygltf::Node& node,
                 const uint32_t currentNodeId,
                 const glm::float4x4& baseTransform,
                 const float globalScale,
                 MeshCache& meshCache,
                 const std::vector<int>& cameraIndexMap,
                 const AnimationUsage& animationUsage)
{
    if (gltfDebugLoggingEnabled())
    {
        STRELKA_DEBUG("glTF node '{}'", node.name);
    }

    const glm::float4x4 localTransform = getTransform(node, globalScale);
    const glm::float4x4 globalTransform = baseTransform * localTransform;

    if (node.mesh != -1 && lodFilterEnabled() && isProxyOrLowerLod(node.name))
    {
        ++lodSkipCounter();
        if (gltfDebugLoggingEnabled())
        {
            STRELKA_DEBUG("glTF node '{}' skipped: proxy or non-zero LOD", node.name);
        }
    }
    else if (node.mesh != -1) // mesh exist
    {
        scene.mNodes[currentNodeId].type = oka::Scene::Node::NodeType::mesh;
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        const bool deformingSkin = nodeHasDeformingSkin(node, currentNodeId, animationUsage);
        const bool sharedPose = deformingSkin && nodeSharesDeformedPose(node);

        std::vector<glm::float4x4> instanceTransforms;
        readGpuInstancing(model, node, instanceTransforms);
        scene.mNodes[currentNodeId].preserveInstanceOffsets = !instanceTransforms.empty();

        if (!instanceTransforms.empty())
        {
            if (deformingSkin)
            {
                const size_t prototypeBegin = scene.mNodes[currentNodeId].instanceIds.size();
                processMesh(model, scene, currentNodeId, mesh, globalTransform * instanceTransforms.front(),
                            globalScale, meshCache, (uint32_t)node.mesh, true, sharedPose);
                const std::vector<uint32_t> prototypeIds(
                    scene.mNodes[currentNodeId].instanceIds.begin() + static_cast<std::ptrdiff_t>(prototypeBegin),
                    scene.mNodes[currentNodeId].instanceIds.end());
                for (size_t instanceIndex = 1; instanceIndex < instanceTransforms.size(); ++instanceIndex)
                {
                    for (const uint32_t prototypeId : prototypeIds)
                    {
                        const Instance prototype = scene.getInstances()[prototypeId];
                        const uint32_t instanceId =
                            scene.createInstance(Instance::Type::eMesh, prototype.mMeshId, prototype.mMaterialId,
                                                 globalTransform * instanceTransforms[instanceIndex]);
                        scene.mNodes[currentNodeId].instanceIds.push_back(instanceId);
                    }
                }
            }
            else
            {
                for (const glm::float4x4& instance : instanceTransforms)
                {
                    processMesh(model, scene, currentNodeId, mesh, globalTransform * instance, globalScale, meshCache,
                                (uint32_t)node.mesh, false, false);
                }
            }
        }
        else
        {
            processMesh(model, scene, currentNodeId, mesh, globalTransform, globalScale, meshCache, (uint32_t)node.mesh,
                        deformingSkin, sharedPose);
        }

        // skin binding
        if (deformingSkin)
        {
            scene.mNodes[currentNodeId].skin = node.skin;
            scene.mSkines[node.skin].refNodeId = static_cast<int>(currentNodeId);
        }
    }
    else if (node.camera != -1) // camera node
    {
        // Through the map, never by the raw glTF index -- see loadCameras.
        const int cameraId = ((size_t)node.camera < cameraIndexMap.size()) ? cameraIndexMap[node.camera] : -1;
        if (cameraId < 0)
        {
            STRELKA_WARNING("Node '{}' points at glTF camera {}, which was not loaded", node.name, node.camera);
        }
        else
        {
            scene.mNodes[currentNodeId].type = oka::Scene::Node::NodeType::camera;
            scene.mNodes[currentNodeId].camera = cameraId;
            glm::float3 scale;
            glm::quat rotation;
            glm::float3 translation;
            oka::decomposeTrs(globalTransform, translation, rotation, scale);

            rotation = glm::conjugate(rotation);

            oka::Camera& camera = scene.getCamera((uint32_t)cameraId);
            camera.node = static_cast<int>(currentNodeId);
            // decomposeTrs already returns the world translation; multiplying it by
            // the node's scale again moves the camera by however much the hierarchy
            // was scaled. Harmless while every scale is 1, which is why it survived.
            camera.position = translation;
            camera.mOrientation = rotation;
            camera.updateViewMatrix();
            STRELKA_INFO("Camera '{}' (glTF camera {} -> scene camera {}) at [{:.3f} {:.3f} {:.3f}]", node.name,
                         node.camera, cameraId, translation.x, translation.y, translation.z);
        }
    }

    for (const int childIdx : node.children)
    {
        if (scene.mNodes[currentNodeId].type == oka::Scene::Node::NodeType::unknown)
            scene.mNodes[currentNodeId].type = oka::Scene::Node::NodeType::sceneGraph;
        scene.mNodes[childIdx].parent = static_cast<int>(currentNodeId);
        processNode(model, scene, model.nodes[childIdx], childIdx, globalTransform, globalScale, meshCache,
                    cameraIndexMap, animationUsage);
    }
}

std::string extractEmbeddedImage(const tinygltf::Model& model, int imageId, const std::string& modelPath)
{
    const tinygltf::Image& image = model.images[imageId];
    if (image.bufferView < 0 || std::cmp_greater_equal(image.bufferView, model.bufferViews.size()))
        return {};
    const tinygltf::BufferView& view = model.bufferViews[image.bufferView];
    if (view.buffer < 0 || std::cmp_greater_equal(view.buffer, model.buffers.size()))
        return {};
    const std::vector<unsigned char>& data = model.buffers[view.buffer].data;
    if (view.byteOffset + view.byteLength > data.size())
        return {};

    const char* extension = ".bin";
    if (image.mimeType == "image/jpeg")
        extension = ".jpg";
    else if (image.mimeType == "image/png")
        extension = ".png";

    const fs::path source(modelPath);
    const fs::path directory = source.parent_path() / (source.stem().string() + "_embedded");
    // The image name is whatever the exporter felt like writing, so the index
    // carries uniqueness and the name is only there to make the folder readable.
    std::string safeName;
    for (const char c : image.name)
        safeName += (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') ? c : '_';
    const fs::path target = directory / (std::to_string(imageId) + (safeName.empty() ? "" : "_" + safeName) + extension);

    std::error_code ec;
    // Re-extracting on every load would rewrite hundreds of megabytes for
    // nothing; a matching size means this is the same image as last time.
    if (fs::exists(target, ec) && fs::file_size(target, ec) == view.byteLength)
        return target.string();

    fs::create_directories(directory, ec);
    std::ofstream out(target, std::ios::binary);
    if (!out)
    {
        STRELKA_WARNING("glTF image {} is embedded but '{}' could not be written", imageId, target.string());
        return {};
    }
    out.write(
        reinterpret_cast<const char*>(data.data() + view.byteOffset), static_cast<std::streamsize>(view.byteLength));
    return target.string();
}

std::string getTextureUri(const tinygltf::Model& model, int texIndex, const std::string& modelPath)
{
    if (texIndex < 0)
        return {};
    const auto imageId = model.textures[texIndex].source;
    if (imageId < 0 || std::cmp_greater_equal(imageId, model.images.size()))
        return {};
    const std::string& uri = model.images[imageId].uri;
    if (uri.empty())
        return extractEmbeddedImage(model, imageId, modelPath);

    std::string decoded;
    if (tinygltf::URIDecode(uri, &decoded, nullptr))
        return decoded;
    return uri;
}

void readTextureTransform(const tinygltf::Material& material, MaterialParams& p)
{
    p.uv_offset_x = 0.0f;
    p.uv_offset_y = 0.0f;
    p.uv_scale_x = 1.0f;
    p.uv_scale_y = 1.0f;
    p.uv_rotation = 0.0f;

    const tinygltf::ExtensionMap* const slots[] = {
        &material.pbrMetallicRoughness.baseColorTexture.extensions,
        &material.pbrMetallicRoughness.metallicRoughnessTexture.extensions,
        &material.normalTexture.extensions,
        &material.emissiveTexture.extensions,
        &material.occlusionTexture.extensions,
    };
    for (const tinygltf::ExtensionMap* ext : slots)
    {
        const auto it = ext->find("KHR_texture_transform");
        if (it == ext->end() || !it->second.IsObject())
            continue;
        const tinygltf::Value& t = it->second;
        if (t.Has("offset") && t.Get("offset").IsArray() && t.Get("offset").ArrayLen() >= 2)
        {
            p.uv_offset_x = (float)t.Get("offset").Get(0).GetNumberAsDouble();
            p.uv_offset_y = (float)t.Get("offset").Get(1).GetNumberAsDouble();
        }
        if (t.Has("scale") && t.Get("scale").IsArray() && t.Get("scale").ArrayLen() >= 2)
        {
            p.uv_scale_x = (float)t.Get("scale").Get(0).GetNumberAsDouble();
            p.uv_scale_y = (float)t.Get("scale").Get(1).GetNumberAsDouble();
        }
        if (t.Has("rotation"))
            p.uv_rotation = (float)t.Get("rotation").GetNumberAsDouble();
        return;
    }
}

float khrFloat(const tinygltf::Material& material, const char* extension, const char* key, float fallback)
{
    const auto it = material.extensions.find(extension);
    if (it == material.extensions.end() || !it->second.IsObject() || !it->second.Has(key))
        return fallback;
    return (float)it->second.Get(key).GetNumberAsDouble();
}

oka::Scene::MaterialDescription convertToStandardPBR(const tinygltf::Model& model,
                                                     const tinygltf::Material& material,
                                                     const std::string& modelPath)
{
    oka::Scene::MaterialDescription desc{};
    desc.name = material.name.empty() ? "material" : material.name;

    MaterialParams& p = desc.params;
    p.material_type = MATERIAL_TYPE_STANDARD_PBR;

    // STRELKA_materials_hair: Chiang lobe on curve geometry. Absent from glTF,
    // so the converter / feature-test patcher writes it. Without it a strand is
    // a rough dielectric cylinder and the kids-bedroom monster goes dark.
    {
        const auto hit = material.extensions.find("STRELKA_materials_hair");
        if (hit != material.extensions.end())
        {
            p.material_type = MATERIAL_TYPE_HAIR;
            // Radial roughness rides in anisotropy (unused for hair GGX). Coat
            // weight rides in clearcoat -- same mapping hair_chiang_prepare reads.
            if (hit->second.IsObject())
            {
                if (hit->second.Has("radialRoughness"))
                    p.anisotropy = (float)hit->second.Get("radialRoughness").GetNumberAsDouble();
                if (hit->second.Has("coat"))
                    p.clearcoat = (float)hit->second.Get("coat").GetNumberAsDouble();
            }
        }
    }

    // Base color
    const auto& bcf = material.pbrMetallicRoughness.baseColorFactor;
    p.base_color = { (float)bcf[0], (float)bcf[1], (float)bcf[2] };
    p.base_color_alpha = (float)bcf[3];
    p.alpha_mode = material.alphaMode == "MASK"  ? ALPHA_MODE_MASK :
                   material.alphaMode == "BLEND" ? ALPHA_MODE_BLEND :
                                                   ALPHA_MODE_OPAQUE;

    // Metallic / roughness
    p.roughness = (float)material.pbrMetallicRoughness.roughnessFactor;
    p.metallic = (float)material.pbrMetallicRoughness.metallicFactor;

    p.ior = khrFloat(material, "KHR_materials_ior", "ior", 1.5f);
    p.specular = 0.5f * khrFloat(material, "KHR_materials_specular", "specularFactor", 1.0f);
    // specularColorFactor, white when the extension is absent or silent about it.
    p.specular_color = glm::float3(1.0f);
    {
        const auto sp = material.extensions.find("KHR_materials_specular");
        if (sp != material.extensions.end() && sp->second.IsObject() && sp->second.Has("specularColorFactor"))
        {
            const tinygltf::Value& c = sp->second.Get("specularColorFactor");
            if (c.IsArray() && c.ArrayLen() >= 3)
            {
                p.specular_color = { (float)c.Get(0).GetNumberAsDouble(), (float)c.Get(1).GetNumberAsDouble(),
                                     (float)c.Get(2).GetNumberAsDouble() };
            }
        }
    }
    p.transmission = khrFloat(material, "KHR_materials_transmission", "transmissionFactor", 0.0f);

    p.diffuse_transmission = khrFloat(material, "KHR_materials_diffuse_transmission", "diffuseTransmissionFactor", 0.0f);
    p.diffuse_transmission_color = glm::float3(1.0f);
    {
        const auto ext = material.extensions.find("KHR_materials_diffuse_transmission");
        if (ext != material.extensions.end() && ext->second.Has("diffuseTransmissionColorFactor"))
        {
            const auto& c = ext->second.Get("diffuseTransmissionColorFactor");
            if (c.IsArray() && c.ArrayLen() >= 3)
            {
                p.diffuse_transmission_color =
                    glm::float3((float)c.Get(0).GetNumberAsDouble(), (float)c.Get(1).GetNumberAsDouble(),
                                (float)c.Get(2).GetNumberAsDouble());
            }
        }
    }
    p.sheen = 0.0f;
    p.sheen_roughness = khrFloat(material, "KHR_materials_sheen", "sheenRoughnessFactor", 0.0f);
    p.sheen_color = glm::float3(1.0f);
    {
        const auto sit = material.extensions.find("KHR_materials_sheen");
        if (sit != material.extensions.end() && sit->second.IsObject() && sit->second.Has("sheenColorFactor"))
        {
            const tinygltf::Value& c = sit->second.Get("sheenColorFactor");
            if (c.IsArray() && c.ArrayLen() >= 3)
            {
                const glm::float3 sheenColor((float)c.Get(0).GetNumberAsDouble(), (float)c.Get(1).GetNumberAsDouble(),
                                             (float)c.Get(2).GetNumberAsDouble());
                p.sheen = std::max({ sheenColor.x, sheenColor.y, sheenColor.z });
                if (p.sheen > 0.0f)
                    p.sheen_color = sheenColor / p.sheen;
            }
        }
    }

    p.subsurface = 0.0f;
    p.subsurface_radius = glm::float3(0.0f);
    p.subsurface_anisotropy = 0.0f;
    {
        const auto sit = material.extensions.find("STRELKA_materials_subsurface");
        if (sit != material.extensions.end() && sit->second.IsObject())
        {
            const tinygltf::Value& ext = sit->second;
            if (ext.Has("scatterRadius"))
            {
                const tinygltf::Value& r = ext.Get("scatterRadius");
                if (r.IsArray() && r.ArrayLen() >= 3)
                {
                    p.subsurface_radius = { (float)r.Get(0).GetNumberAsDouble(), (float)r.Get(1).GetNumberAsDouble(),
                                            (float)r.Get(2).GetNumberAsDouble() };
                }
            }
            p.subsurface_anisotropy = khrFloat(material, "STRELKA_materials_subsurface", "anisotropy", 0.0f);
            // The albedo scatterColor was derived from. Absent means "the same
            // colour", i.e. a flat material, and the ratio the walk takes is 1.
            p.subsurface_reference = glm::float3(0.0f);
            if (ext.Has("scatterReference"))
            {
                const tinygltf::Value& r = ext.Get("scatterReference");
                if (r.IsArray() && r.ArrayLen() >= 3)
                {
                    p.subsurface_reference = { (float)r.Get(0).GetNumberAsDouble(), (float)r.Get(1).GetNumberAsDouble(),
                                               (float)r.Get(2).GetNumberAsDouble() };
                }
            }
            p.subsurface = khrFloat(material, "STRELKA_materials_subsurface", "subsurfaceFactor", 1.0f);

            // A zero mean free path is an infinitely dense medium, i.e. a walk
            // that never terminates. Treat it as "no medium" rather than as a
            // hang.
            if (p.subsurface_radius.x <= 0.0f && p.subsurface_radius.y <= 0.0f && p.subsurface_radius.z <= 0.0f)
            {
                p.subsurface = 0.0f;
            }

            if (p.subsurface > 0.0f)
            {
                p.diffuse_transmission = p.subsurface;
                const auto cit = ext.Has("scatterColor") ? &ext.Get("scatterColor") : nullptr;
                if (cit && cit->IsArray() && cit->ArrayLen() >= 3)
                {
                    p.diffuse_transmission_color = { (float)cit->Get(0).GetNumberAsDouble(),
                                                     (float)cit->Get(1).GetNumberAsDouble(),
                                                     (float)cit->Get(2).GetNumberAsDouble() };
                }
            }
        }
    }

    p.medium_flags = 0u;
    p.medium_emission = glm::float3(0.0f);
    {
        const auto mit = material.extensions.find("STRELKA_materials_medium");
        if (mit != material.extensions.end() && mit->second.IsObject())
        {
            const tinygltf::Value& ext = mit->second;
            const float density = khrFloat(material, "STRELKA_materials_medium", "density", 1.0f);
            if (density > 0.0f)
            {
                p.medium_flags |= MEDIUM_FLAG_BOUNDARY;
                p.subsurface_radius = glm::float3(1.0f / density);
                p.subsurface_anisotropy = khrFloat(material, "STRELKA_materials_medium", "anisotropy", 0.0f);
                p.diffuse_transmission_color = glm::float3(1.0f);
                if (ext.Has("scatterColor"))
                {
                    const tinygltf::Value& c = ext.Get("scatterColor");
                    if (c.IsArray() && c.ArrayLen() >= 3)
                    {
                        p.diffuse_transmission_color = { (float)c.Get(0).GetNumberAsDouble(),
                                                         (float)c.Get(1).GetNumberAsDouble(),
                                                         (float)c.Get(2).GetNumberAsDouble() };
                    }
                }
                if (ext.Has("emissionColor"))
                {
                    const tinygltf::Value& c = ext.Get("emissionColor");
                    if (c.IsArray() && c.ArrayLen() >= 3)
                    {
                        p.medium_emission = { (float)c.Get(0).GetNumberAsDouble(), (float)c.Get(1).GetNumberAsDouble(),
                                              (float)c.Get(2).GetNumberAsDouble() };
                    }
                }
            }
        }
    }

    // KHR_materials_iridescence. With no thickness texture the spec says to use
    // iridescenceThicknessMaximum, so that is the only thickness read here; an
    // exporter that wants a specific film writes the same value to both bounds.
    p.iridescence = khrFloat(material, "KHR_materials_iridescence", "iridescenceFactor", 0.0f);
    p.iridescence_ior = khrFloat(material, "KHR_materials_iridescence", "iridescenceIor", 1.3f);
    p.iridescence_thickness = khrFloat(material, "KHR_materials_iridescence", "iridescenceThicknessMaximum", 400.0f);

    p.clearcoat = khrFloat(material, "KHR_materials_clearcoat", "clearcoatFactor", 0.0f);
    p.clearcoat_roughness = khrFloat(material, "KHR_materials_clearcoat", "clearcoatRoughnessFactor", 0.0f);
    p.clearcoat_ior = 1.5f;
    {
        const auto cit = material.extensions.find("KHR_materials_clearcoat");
        if (cit != material.extensions.end() && cit->second.Has("clearcoatIor"))
            p.clearcoat_ior = (float)cit->second.Get("clearcoatIor").GetNumberAsDouble();
    }
    p.anisotropy = khrFloat(material, "KHR_materials_anisotropy", "anisotropyStrength", 0.0f);
    p.anisotropy_rotation = khrFloat(material, "KHR_materials_anisotropy", "anisotropyRotation", 0.0f);

    // KHR_materials_volume. attenuationDistance defaults to +infinity, i.e. no
    // absorption; 0 is the encoding used downstream for "none".
    p.attenuation_distance = khrFloat(material, "KHR_materials_volume", "attenuationDistance", 0.0f);
    p.attenuation_color = { 1.0f, 1.0f, 1.0f };
    {
        const auto vit = material.extensions.find("KHR_materials_volume");
        if (vit != material.extensions.end() && vit->second.IsObject() && vit->second.Has("attenuationColor"))
        {
            const tinygltf::Value& c = vit->second.Get("attenuationColor");
            if (c.IsArray() && c.ArrayLen() >= 3)
            {
                p.attenuation_color = { (float)c.Get(0).GetNumberAsDouble(), (float)c.Get(1).GetNumberAsDouble(),
                                        (float)c.Get(2).GetNumberAsDouble() };
            }
        }
    }

    const auto& emf = material.emissiveFactor;
    p.emission = { (float)emf[0], (float)emf[1], (float)emf[2] };
    p.emission_strength = khrFloat(material, "KHR_materials_emissive_strength", "emissiveStrength", 1.0f);

    // Normal / occlusion / alpha
    p.normal_scale = (float)material.normalTexture.scale;
    p.occlusion_strength = (float)material.occlusionTexture.strength;
    p.alpha_cutoff = (float)material.alphaCutoff;

    // Texture indices default to -1 (no texture); renderer assigns real indices
    p.base_color_tex = -1;
    p.metallic_roughness_tex = -1;
    p.normal_tex = -1;
    p.emission_tex = -1;
    p.occlusion_tex = -1;
    p.transmission_tex = -1;
    p.thin_walled = 0;
    {
        const auto vit = material.extensions.find("KHR_materials_volume");
        if (vit != material.extensions.end() && vit->second.IsObject())
        {
            const float thickness = khrFloat(material, "KHR_materials_volume", "thicknessFactor", 0.0f);
            p.thin_walled = (thickness <= 0.0f) ? 1u : 0u;
        }
    }
    // A transmissive surface is a dielectric volume and needs a priority so the
    // nested-dielectric IOR stack can order it; an opaque one must stay at 0.
    p.dielectric_priority = p.transmission > 0.0f ? 10u : 0u;

    readTextureTransform(material, p);

    // Store texture file paths for the renderer to load
    desc.baseColorTexPath = getTextureUri(model, material.pbrMetallicRoughness.baseColorTexture.index, modelPath);
    desc.metallicRoughnessTexPath =
        getTextureUri(model, material.pbrMetallicRoughness.metallicRoughnessTexture.index, modelPath);
    desc.normalTexPath = getTextureUri(model, material.normalTexture.index, modelPath);
    desc.emissionTexPath = getTextureUri(model, material.emissiveTexture.index, modelPath);
    desc.occlusionTexPath = getTextureUri(model, material.occlusionTexture.index, modelPath);

    return desc;
}

void loadMaterials(const tinygltf::Model& model, oka::Scene& scene)
{
    for (const tinygltf::Material& material : model.materials)
    {
        scene.addMaterial(convertToStandardPBR(model, material, scene.getSourcePath()));
    }
}

void loadCameras(const tinygltf::Model& model, oka::Scene& scene, std::vector<int>& gltfToScene)
{
    gltfToScene.assign(model.cameras.size(), -1);

    for (size_t i = 0; i < model.cameras.size(); ++i)
    {
        const auto& cameraGltf = model.cameras[i];
        oka::Camera camera;
        camera.name = cameraGltf.name;

        if (cameraGltf.type == "perspective")
        {
            camera.projection = oka::Camera::ProjectionType::perspective;
            camera.fov = static_cast<float>(cameraGltf.perspective.yfov) * (180.0f / std::numbers::pi_v<float>);
            camera.authoredAspect = (float)cameraGltf.perspective.aspectRatio;
            camera.znear = static_cast<float>(cameraGltf.perspective.znear);
            camera.zfar = static_cast<float>(cameraGltf.perspective.zfar);
        }
        else if (cameraGltf.type == "orthographic")
        {
            // glTF xmag/ymag are half-extents, so they go straight across; the
            // authored aspect follows from them and needs no separate field.
            camera.projection = oka::Camera::ProjectionType::orthographic;
            camera.xmag = (float)cameraGltf.orthographic.xmag;
            camera.ymag = (float)cameraGltf.orthographic.ymag;
            camera.authoredAspect = (camera.ymag > 0.0f) ? (camera.xmag / camera.ymag) : 0.0f;
            camera.znear = static_cast<float>(cameraGltf.orthographic.znear);
            camera.zfar = static_cast<float>(cameraGltf.orthographic.zfar);
        }
        else
        {
            STRELKA_WARNING("glTF camera {} '{}': unknown type '{}', skipped", i, cameraGltf.name, cameraGltf.type);
            continue;
        }

        gltfToScene[i] = (int)scene.getCameraCount();
        scene.addCamera(camera);
    }
    // No default camera added here — the editor creates its own "Main" camera
    // with proper scene-fit positioning in EditorApp::prepare().
}

void loadAnimation(const tinygltf::Model& model, oka::Scene& scene)
{
    std::vector<oka::Scene::Animation> animations;
    scene.blasUpdateCount = 0;
    scene.tlasUpdateCount = 0;

    using namespace std;
    for (const tinygltf::Animation& animation : model.animations)
    {
        oka::Scene::Animation anim{};

        anim.name = animation.name;
        if (anim.name.empty())
        {
            anim.name = "noname animation";
        }
        if (gltfDebugLoggingEnabled())
        {
            STRELKA_DEBUG("glTF animation '{}'", anim.name);
        }

        for (const tinygltf::AnimationSampler& sampler : animation.samplers)
        {
            oka::Scene::AnimationSampler samp{};
            {
                if (sampler.interpolation == "STEP")
                    samp.interpolation = oka::Scene::AnimationSampler::InterpolationType::STEP;
                if (sampler.interpolation == "CUBICSPLINE")
                    samp.interpolation = oka::Scene::AnimationSampler::InterpolationType::CUBICSPLINE;
            }
            {
                const tinygltf::Accessor& accessor = model.accessors[sampler.input];
                const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                const void* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                const auto* buf = static_cast<const float*>(dataPtr);

                for (size_t index = 0; index < accessor.count; index++)
                {
                    samp.inputs.push_back(buf[index]);
                }

                for (auto input : samp.inputs)
                {
                    if (input < anim.start)
                    {
                        anim.start = input;
                        anim.current = input;
                    };
                    if (input > anim.end)
                    {
                        anim.end = input;
                    }
                }
            }
            {
                const tinygltf::Accessor& accessor = model.accessors[sampler.output];
                const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                const void* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                switch (accessor.type)
                {
                case TINYGLTF_TYPE_VEC3: {
                    const auto* buf = static_cast<const glm::vec3*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++)
                    {
                        samp.outputsVec4.emplace_back(buf[index], 0.0f);
                    }
                    break;
                }
                case TINYGLTF_TYPE_VEC4: {
                    const auto* buf = static_cast<const glm::vec4*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++)
                    {
                        samp.outputsVec4.push_back(buf[index]);
                    }
                    break;
                }
                default: {
                    STRELKA_WARNING("glTF animation '{}' uses unsupported output accessor type {}; skipping output",
                                    animation.name, accessor.type);
                    break;
                }
                }
                anim.samplers.push_back(samp);
            }
        }
        for (const tinygltf::AnimationChannel& channel : animation.channels)
        {
            oka::Scene::AnimationChannel chan{};
            if (channel.target_path == "rotation")
            {
                chan.path = oka::Scene::AnimationChannel::PathType::ROTATION;
            }
            if (channel.target_path == "translation")
            {
                chan.path = oka::Scene::AnimationChannel::PathType::TRANSLATION;
            }
            if (channel.target_path == "scale")
            {
                chan.path = oka::Scene::AnimationChannel::PathType::SCALE;
            }
            if (channel.target_path == "weights")
            {
                STRELKA_WARNING(
                    "glTF animation '{}' uses an unsupported weights channel; skipping channel", animation.name);
                continue;
            }
            chan.samplerIndex = channel.sampler;
            chan.node = channel.target_node;
            if (chan.node < 0)
            {
                STRELKA_WARNING(
                    "glTF animation '{}' channel has invalid node {}; skipping channel", animation.name, chan.node);
                continue;
            }

            anim.channels.push_back(chan);
        }
        animations.push_back(anim);
    }
    scene.mAnimations = animations;
}

void loadNodes(const tinygltf::Model& model, oka::Scene& scene, const float globalScale = 1.0f)
{
    scene.mNodes.reserve(model.nodes.size());
    for (const auto& node : model.nodes)
    {
        oka::Scene::Node n{};
        n.name = node.name;
        n.children = node.children;

        glm::float3 scale{ 1.0f };
        if (!node.scale.empty())
        {
            scale = glm::float3((float)node.scale[0], (float)node.scale[1], (float)node.scale[2]);
            // check that scale is uniform, otherwise we have to support it in shader
            // assert(scale.x == scale.y && scale.y == scale.z);
        }
        n.scale = scale;

        // glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        glm::quat rotation = glm::quat_cast(glm::float4x4(1.0f));
        if (!node.rotation.empty())
        {
            rotation = quatFromGltf(
                (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2], (float)node.rotation[3]);
        }
        n.rotation = rotation;

        glm::float3 translation{ 0.0f };
        if (!node.translation.empty())
        {
            translation = glm::float3((float)node.translation[0], (float)node.translation[1], (float)node.translation[2]);
            translation *= globalScale;
        }
        n.translation = translation;
        scene.mNodes.push_back(n);
    }
}

void loadSkeletalData(const tinygltf::Model& model, oka::Scene& scene, const float /*globalScale*/ = 1.0f)
{
    // glTF stores matrices in byte-backed accessor payloads; the accessor type
    // and stride validate this float view.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    for (const auto& skin : model.skins)
    {
        oka::Scene::Skin s{};
        s.name = skin.name;
        s.skeletonId = skin.skeleton;
        s.joints = skin.joints;

        const tinygltf::Accessor& matrixAccessor = model.accessors[skin.inverseBindMatrices];
        const tinygltf::BufferView& bufferView = model.bufferViews[matrixAccessor.bufferView];
        const auto* matrixData = reinterpret_cast<const float*>(
            &model.buffers[bufferView.buffer].data[matrixAccessor.byteOffset + bufferView.byteOffset]);
        assert(matrixData != nullptr);
        const auto matrixCount = static_cast<uint32_t>(matrixAccessor.count);
        assert(matrixCount != 0);
        const int matStride = matrixAccessor.ByteStride(bufferView) / static_cast<int>(sizeof(float));
        assert(matStride > 0);

        for (const int jointid : s.joints)
        {
            scene.mNodes[jointid].type = oka::Scene::Node::NodeType::skeleton;
            // s.inverseBindMatrices.push_back(glm::inverse(scene.calculateNodeGlobalTransform(jointid)));
        }
        for (size_t m = 0; m < matrixCount; ++m)
        {
            const glm::mat4 inverseBindMatrix = glm::make_mat4(&matrixData[m * matStride]);
            s.inverseBindMatrices.push_back(inverseBindMatrix);
        }

        scene.mSkines.push_back(s);
    }
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
}

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
bool loadCurvesFromSidecar(const std::string& modelPath, oka::Scene& scene)
{
    const std::string stem = modelPath.substr(0, modelPath.rfind('.'));
    const std::string curvePath = stem + "_curves.bin";
    if (!fs::exists(curvePath))
    {
        return false;
    }
    STRELKA_INFO("Found curve file: {}", curvePath);
    return curvesidecar::loadCurvesFile(curvePath, scene);
}

namespace
{

void loadOpenPBRMaterialsFromSidecar(const std::string& modelPath, oka::Scene& scene)
{
    const std::string stem = modelPath.substr(0, modelPath.rfind('.'));
    const std::string path = oka::materialsidecar::findMaterialSidecar(stem);
    if (path.empty())
    {
        return;
    }
    oka::materialsidecar::loadMaterialsJson(scene, path);
}

void loadMaterialXFromSidecar(const std::string& modelPath, oka::Scene& scene)
{
    const std::string stem = modelPath.substr(0, modelPath.rfind('.'));
    const std::string path = stem + ".mtlx";
    if (!fs::exists(path))
    {
        return;
    }
    oka::mtlx::applyMaterialXDocument(scene, path);
}

bool loadLightsFromJson(const std::string& modelPath, oka::Scene& scene)
{
    // First try exact match: <modelname>_light.json
    const std::string fileName = modelPath.substr(0, modelPath.rfind('.')); // w/o extension
    std::string jsonPath = fileName + "_light" + ".json";

    // If not found, scan directory for any *_light.json file
    if (!fs::exists(jsonPath))
    {
        const fs::path dir = fs::path(modelPath).parent_path();
        jsonPath.clear();
        for (const auto& entry : fs::directory_iterator(dir))
        {
            if (entry.is_regular_file())
            {
                const std::string name = entry.path().filename().string();
                if (name.size() > 11 && name.substr(name.size() - 11) == "_light.json")
                {
                    jsonPath = entry.path().string();
                    break;
                }
            }
        }
    }

    if (jsonPath.empty() || !fs::exists(jsonPath))
        return false;

    STRELKA_INFO("Found light file: {}", jsonPath);
    return loadLightsJson(scene, jsonPath);
}

static constexpr float kLumensPerWatt = 683.0f;

// KHR_lights_punctual: lights live in root extensions and are referenced from
// nodes. Intensity is candela for point/spot and lux for directional.
bool loadPunctualLights(const tinygltf::Model& model, oka::Scene& scene)
{
    const auto rootIt = model.extensions.find("KHR_lights_punctual");
    if (rootIt == model.extensions.end() || !rootIt->second.IsObject())
        return false;
    const tinygltf::Value& rootExt = rootIt->second;
    if (!rootExt.Has("lights") || !rootExt.Get("lights").IsArray())
        return false;
    const tinygltf::Value& lightsArr = rootExt.Get("lights");
    if (lightsArr.ArrayLen() == 0)
        return false;

    auto readVec3 = [](const tinygltf::Value& v, glm::float3 fallback) {
        if (!v.IsArray() || v.ArrayLen() < 3)
            return fallback;
        return glm::float3((float)v.Get(0).GetNumberAsDouble(), (float)v.Get(1).GetNumberAsDouble(),
                           (float)v.Get(2).GetNumberAsDouble());
    };

    uint32_t created = 0;
    for (size_t nodeIdx = 0; nodeIdx < model.nodes.size(); ++nodeIdx)
    {
        const tinygltf::Node& node = model.nodes[nodeIdx];
        const auto lightIt = node.extensions.find("KHR_lights_punctual");
        if (lightIt == node.extensions.end() || !lightIt->second.Has("light"))
            continue;
        const int lightIndex = lightIt->second.Get("light").GetNumberAsInt();
        if (lightIndex < 0 || (size_t)lightIndex >= lightsArr.ArrayLen())
            continue;
        const tinygltf::Value& L = lightsArr.Get(lightIndex);
        if (!L.IsObject())
            continue;

        Scene::UniformLightDesc desc{};
        desc.enabled = true;
        desc.name = L.Has("name") ? L.Get("name").Get<std::string>() : node.name;
        desc.color = L.Has("color") ? readVec3(L.Get("color"), glm::float3(1.0f)) : glm::float3(1.0f);
        desc.intensity = (L.Has("intensity") ? (float)L.Get("intensity").GetNumberAsDouble() : 1.0f) / kLumensPerWatt;
        desc.range = L.Has("range") ? (float)L.Get("range").GetNumberAsDouble() : 0.0f;

        const std::string type = L.Has("type") ? L.Get("type").Get<std::string>() : "point";
        if (type == "directional")
        {
            desc.type = LIGHT_TYPE_DISTANT;
            desc.intensityUnit = LIGHT_UNIT_IRRADIANCE;
            // glTF has no sun angular size; use a small disk so soft shadows work.
            desc.halfAngle = 0.53f * 0.5f * (std::numbers::pi_v<float> / 180.0f);
        }
        else if (type == "spot")
        {
            desc.type = LIGHT_TYPE_SPOT;
            desc.intensityUnit = LIGHT_UNIT_INTENSITY;
            float inner = 0.0f;
            float outer = std::numbers::pi_v<float> / 4.0f;
            if (L.Has("spot") && L.Get("spot").IsObject())
            {
                const tinygltf::Value& spot = L.Get("spot");
                if (spot.Has("innerConeAngle"))
                    inner = (float)spot.Get("innerConeAngle").GetNumberAsDouble();
                if (spot.Has("outerConeAngle"))
                    outer = (float)spot.Get("outerConeAngle").GetNumberAsDouble();
            }
            desc.innerConeAngle = inner;
            desc.outerConeAngle = outer;
        }
        else
        {
            desc.type = LIGHT_TYPE_POINT;
            desc.intensityUnit = LIGHT_UNIT_INTENSITY;
        }

        // Node world transform → position + orientation. Prefer the already
        // computed scene global transform when the node was ingested.
        if (nodeIdx < scene.getGlobalTransforms().size())
        {
            glm::float3 T, S;
            glm::quat R;
            decomposeTrs(scene.getGlobalTransforms()[nodeIdx], T, R, S);
            desc.position = T;
            desc.orientation = glm::degrees(glm::eulerAngles(R));
        }
        else
        {
            desc.useXform = true;
            desc.xform = getTransform(node, 1.0f);
        }

        scene.createLight(desc);
        ++created;
    }

    if (created > 0)
        STRELKA_INFO("Loaded {} KHR_lights_punctual light(s)", created);
    return created > 0;
}

void loadCamerasFromJson(const std::string& modelPath, oka::Scene& scene)
{
    const std::string fileName = modelPath.substr(0, modelPath.rfind('.'));
    std::string jsonPath = fileName + "_camera.json";

    if (!fs::exists(jsonPath))
    {
        const fs::path dir = fs::path(modelPath).parent_path();
        jsonPath.clear();
        for (const auto& entry : fs::directory_iterator(dir))
        {
            if (entry.is_regular_file())
            {
                const std::string name = entry.path().filename().string();
                if (name.size() > 12 && name.substr(name.size() - 12) == "_camera.json")
                {
                    jsonPath = entry.path().string();
                    break;
                }
            }
        }
    }

    if (jsonPath.empty() || !fs::exists(jsonPath))
        return;

    STRELKA_INFO("Found camera file: {}", jsonPath);
    std::ifstream i(jsonPath);
    json root;
    i >> root;

    if (!root.contains("cameras"))
        return;

    for (const auto& cam : root["cameras"])
    {
        if (!cam.contains("name"))
            continue;

        const std::string name = cam["name"].get<std::string>();
        const uint32_t idx = scene.findCameraByName(name);
        if (idx == kInvalidIndex)
        {
            STRELKA_WARNING("Camera JSON: no matching camera '{}' in scene", name);
            continue;
        }

        oka::Camera& camera = scene.getCamera(idx);

        if (cam.contains("focal_length_mm"))
            camera.focalLengthMm = cam["focal_length_mm"].get<float>();
        if (cam.contains("sensor_width"))
            camera.sensorWidth = cam["sensor_width"].get<float>();
        if (cam.contains("sensor_height"))
            camera.sensorHeight = cam["sensor_height"].get<float>();
        if (cam.contains("shift_x"))
            camera.shiftX = cam["shift_x"].get<float>();
        if (cam.contains("shift_y"))
            camera.shiftY = cam["shift_y"].get<float>();

        if (cam.contains("dof"))
        {
            const auto& dof = cam["dof"];
            if (dof.contains("enabled"))
                camera.useDof = dof["enabled"].get<bool>();
            if (dof.contains("focus_distance"))
                camera.focalDistance = dof["focus_distance"].get<float>();
            if (dof.contains("fstop"))
                camera.fStopDof = dof["fstop"].get<float>();
            if (dof.contains("blades"))
                camera.apertureBlades = dof["blades"].get<int>();
            if (dof.contains("blade_rotation"))
                camera.bladeRotation = dof["blade_rotation"].get<float>();
            if (dof.contains("anamorphic_ratio"))
                camera.anamorphicRatio = dof["anamorphic_ratio"].get<float>();
        }

        STRELKA_INFO("Camera JSON: applied properties to '{}'", name);
    }
}

} // namespace

namespace
{

bool readWholeFileMapped(std::vector<unsigned char>* out, std::string* err, const std::string& path, void* userData)
{
#if defined(_WIN32)
    return tinygltf::ReadWholeFile(out, err, path, userData);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
    {
        return tinygltf::ReadWholeFile(out, err, path, userData);
    }
    struct stat status = {};
    if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0)
    {
        ::close(fd);
        return tinygltf::ReadWholeFile(out, err, path, userData);
    }
    const size_t size = static_cast<size_t>(status.st_size);
    void* mapped = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED)
    {
        return tinygltf::ReadWholeFile(out, err, path, userData);
    }
    // The copy below is one forward pass, which is exactly what these two say.
    ::madvise(mapped, size, MADV_SEQUENTIAL);
    ::madvise(mapped, size, MADV_WILLNEED);
    const unsigned char* bytes = static_cast<const unsigned char*>(mapped);
    // Sized before the copy so the destination can be asked for huge pages: a
    // 2.8 GB buffer is 700 000 faults at 4 KiB a page and 1 400 at 2 MiB.
    out->reserve(size);
    adviseHugePages(out->data(), size);
    out->assign(bytes, bytes + size);
    ::munmap(mapped, size);
    return true;
#endif
}

} // namespace

bool GltfLoader::loadGltf(const std::string& modelPath, oka::Scene& scene)
{
    if (modelPath.empty())
    {
        return false;
    }
    const fs::path exportLock = fs::path(modelPath).parent_path() / ".wow2strelka-exporting";
    if (fs::exists(exportLock))
    {
        STRELKA_WARNING("Scene export is still in progress: {}", exportLock.string());
        return false;
    }

    const auto loadStarted = std::chrono::steady_clock::now();
    scene.setSourcePath(modelPath);

    using namespace std;
    tinygltf::Model model;
    tinygltf::TinyGLTF gltf_ctx;
    gltf_ctx.SetMaxExternalFileSize(std::numeric_limits<size_t>::max());
    // Everything but the read stays on tinygltf's own callbacks; SetFsCallbacks
    // refuses a partially filled set.
    tinygltf::FsCallbacks fsCallbacks;
    fsCallbacks.FileExists = &tinygltf::FileExists;
    fsCallbacks.ExpandFilePath = &tinygltf::ExpandFilePath;
    fsCallbacks.ReadWholeFile = &readWholeFileMapped;
    fsCallbacks.WriteWholeFile = &tinygltf::WriteWholeFile;
    fsCallbacks.GetFileSizeInBytes = &tinygltf::GetFileSizeInBytes;
    fsCallbacks.user_data = nullptr;
    std::string fsErr;
    if (!gltf_ctx.SetFsCallbacks(fsCallbacks, &fsErr))
    {
        STRELKA_WARNING("Falling back to tinygltf's own file reader: {}", fsErr);
    }
    gltf_ctx.SetImageLoader([](tinygltf::Image*, const int, std::string*, std::string*, int, int, const unsigned char*,
                               int, void*) { return true; },
                            nullptr);
    std::string err;
    std::string warn;
    bool res = false;
    const std::string ext = fs::path(modelPath).extension().string();
    if (mProgress)
    {
        mProgress->beginStage(LoadProgress::Stage::Reading, 0);
    }
    if (ext == ".glb")
    {
        res = gltf_ctx.LoadBinaryFromFile(&model, &err, &warn, modelPath.c_str());
    }
    else
    {
        res = gltf_ctx.LoadASCIIFromFile(&model, &err, &warn, modelPath.c_str());
    }
    if (!warn.empty())
    {
        STRELKA_WARNING("glTF warning: {}", warn);
    }
    if (!res)
    {
        STRELKA_ERROR("Unable to load file: {}{}", modelPath, err.empty() ? "" : " — " + err);
        return res;
    }
    const auto gltfDecoded = std::chrono::steady_clock::now();

    int sceneId = model.defaultScene < 0 ? 0 : model.defaultScene;
    if (model.scenes.size() > 1)
    {
        // Only the default scene is instantiated. A file with several is
        // ambiguous by construction, and silently drawing a fraction of it looks
        // like missing geometry rather than a choice.
        STRELKA_WARNING("glTF has {} scenes; loading only '{}' (index {}). The rest are ignored.", model.scenes.size(),
                        model.scenes[sceneId].name.empty() ? "<unnamed>" : model.scenes[sceneId].name, sceneId);
    }

    struct Phase
    {
        const char* name;
        std::function<void()> run;
    };

    bool hadJsonLights = false;
    const float globalScale = 1.0f;
    // Lives for the whole graph walk: two nodes anywhere in the scene that point
    // at the same glTF mesh share the geometry built for the first of them.
    MeshCache meshCache;
    // glTF camera index -> scene camera index; filled by loadCameras, read by the
    // graph walk, so it has to outlive both phases.
    std::vector<int> cameraIndexMap;
    const AnimationUsage animationUsage = analyzeAnimationUsage(model);
    size_t deformingSkinNodes = 0;
    size_t reusableSkinNodes = 0;
    for (size_t nodeId = 0; nodeId < model.nodes.size(); ++nodeId)
    {
        const tinygltf::Node& node = model.nodes[nodeId];
        if (node.skin < 0)
        {
            continue;
        }
        if (nodeHasDeformingSkin(node, nodeId, animationUsage))
        {
            ++deformingSkinNodes;
        }
        else if (nodeId < animationUsage.reusableBindPoseNodes.size() && animationUsage.reusableBindPoseNodes[nodeId])
        {
            ++reusableSkinNodes;
        }
    }
    if (deformingSkinNodes != 0 || reusableSkinNodes != 0)
    {
        STRELKA_INFO(
            "Skinning plan: {} deforming node(s), {} shared bind-pose node(s)", deformingSkinNodes, reusableSkinNodes);
    }

    size_t expectedVertices = 0;
    size_t expectedIndices = 0;
    size_t expectedSkin = 0;
    size_t expectedMeshes = 0;
    countModelGeometry(model, animationUsage, expectedVertices, expectedIndices, expectedSkin, expectedMeshes);
    scene.reserveGeometry(expectedVertices, expectedIndices, expectedSkin);
    {
        scene.mMeshes.reserve(expectedMeshes);
        size_t expectedInstances = 0;
        for (const tinygltf::Node& node : model.nodes)
        {
            if (node.mesh < 0)
            {
                continue;
            }
            const size_t prims = model.meshes[node.mesh].primitives.size();
            const size_t instanced = gpuInstanceCount(model, node);
            expectedInstances += (instanced == 0 ? 1 : instanced) * prims;
        }
        scene.getInstances().reserve(expectedInstances);
    }

    std::vector<Phase> phases;
    phases.push_back({ "materials", [&] { loadMaterials(model, scene); } });
    phases.push_back({ "openpbr materials", [&] { loadOpenPBRMaterialsFromSidecar(modelPath, scene); } });
    // After the materials, which the sets are matched against by name.
    phases.push_back({ "curves", [&] { loadCurvesFromSidecar(modelPath, scene); } });
    phases.push_back({ "lights", [&] { hadJsonLights = loadLightsFromJson(modelPath, scene); } });
    phases.push_back({ "cameras", [&] {
                          loadCameras(model, scene, cameraIndexMap);
                          loadCamerasFromJson(modelPath, scene);
                      } });
    phases.push_back({ "nodes", [&] {
                          loadNodes(model, scene, globalScale);
                          lodSkipCounter() = 0;
                      } });
    phases.push_back({ "skins", [&] { loadSkeletalData(model, scene, globalScale); } });
    for (size_t i = 0; i < model.scenes[sceneId].nodes.size(); ++i)
    {
        phases.push_back({ "geometry", [&, i] {
                              const int rootNodeIdx = model.scenes[sceneId].nodes[i];
                              processNode(model, scene, model.nodes[rootNodeIdx], rootNodeIdx, glm::float4x4(1.0f),
                                          globalScale, meshCache, cameraIndexMap, animationUsage);
                          } });
    }
    phases.push_back({ "geometry", [&] {
                          if (lodSkipCounter() != 0)
                          {
                              STRELKA_INFO(
                                  "Skipped {} proxy / non-zero-LOD nodes; set STRELKA_NO_LOD_FILTER=1 "
                                  "to draw every level at once",
                                  lodSkipCounter());
                          }
                      } });
    // Punctual lights need node world transforms, so they land after the graph.
    phases.push_back({ "punctual lights", [&] {
                          if (!hadJsonLights && !loadPunctualLights(model, scene))
                          {
                              STRELKA_WARNING("No light in scene, adding default distant light");
                              oka::Scene::UniformLightDesc lightDesc{};
                              lightDesc.useXform = false;
                              lightDesc.position = glm::float3(0.0f, 0.0f, 0.0f);
                              lightDesc.orientation = glm::float3(-45.0f, 15.0f, 0.0f);
                              lightDesc.type = LIGHT_TYPE_DISTANT;
                              lightDesc.halfAngle = 10.0f * 0.5f * (std::numbers::pi_v<float> / 180.0f);
                              lightDesc.intensity = 100000;
                              lightDesc.color = glm::float3(1.0);
                              scene.createLight(lightDesc);
                          }
                      } });
    phases.push_back({ "materialx", [&] { loadMaterialXFromSidecar(modelPath, scene); } });
    phases.push_back({ "animation", [&] { loadAnimation(model, scene); } });

    if (mProgress)
    {
        mProgress->beginStage(LoadProgress::Stage::Parsing, (uint32_t)phases.size());
    }
    for (const Phase& phase : phases)
    {
        // Checked between phases rather than inside them: a cancelled load should
        // stop within one phase's worth of work, and threading a flag through the
        // node recursion would put a check on every node for no extra benefit.
        if (mProgress && mProgress->isCancelled())
        {
            STRELKA_INFO("Scene load cancelled during '{}'", phase.name);
            return false;
        }
        phase.run();
        if (mProgress)
        {
            mProgress->step();
        }
    }

    {
        size_t released = 0;
        for (tinygltf::Buffer& buffer : model.buffers)
        {
            released += buffer.data.size();
            buffer.data.clear();
            buffer.data.shrink_to_fit();
        }
        if (released > 0)
        {
            STRELKA_INFO("Released {:.2f} GB of glTF buffer data after loading scene data", released / 1e9);
        }
    }

    STRELKA_INFO("Scene host geometry: {} vertices ({:.2f} GB), {} indices ({:.2f} GB), {} meshes, {} instances",
                 scene.getVertices().size(), scene.getVertices().size() * sizeof(oka::Scene::Vertex) / 1e9,
                 scene.getIndices().size(), scene.getIndices().size() * sizeof(uint32_t) / 1e9, scene.mMeshes.size(),
                 scene.getInstances().size());
    const auto loadFinished = std::chrono::steady_clock::now();
    using Milliseconds = std::chrono::duration<double, std::milli>;
    STRELKA_INFO("Scene host load: glTF decode {:.0f} ms, conversion/sidecars {:.0f} ms, total {:.0f} ms",
                 Milliseconds(gltfDecoded - loadStarted).count(), Milliseconds(loadFinished - gltfDecoded).count(),
                 Milliseconds(loadFinished - loadStarted).count());

    return res;
}
} // namespace oka
