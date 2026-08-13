#include <strelka/sceneloader/gltfloader.h>
#include <strelka/sceneloader/sceneserializer.h>
#include <strelka/sceneloader/curve_sidecar.h>
#include <strelka/sceneloader/light_json.h>

#include <strelka/scene/camera.h>
#include <strelka/scene/vertex_packing.h>
#include <strelka/scene/light_desc.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/compatibility.hpp>

#include <strelka/scene/transform.h>

#include <iostream>
#include <limits>
#include <env.h>
#include <log.h>

namespace fs = std::filesystem;

#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace oka
{

bool gltfDebugLoggingEnabled()
{
    static const bool enabled = envFlag("STRELKA_GLTF_DEBUG");
    return enabled;
}

// Detail levels and proxy stand-ins, which glTF has no standard way to express.
//
// MSFT_lod would say which of a set of alternatives to draw; scenes exported
// from Blender carry no such thing, and encode the choice in node names instead
// -- cover_01_lod0 beside cover_01_lod1, rock_proxy beside the detailed rock. A
// loader that takes every node at face value draws them all, stacked in the same
// space. That is not merely wasted work: the alternatives are near-coincident,
// so which surface a ray reaches first is decided by numerical accident, and a
// proxy is a flat untextured hull. On the pine forest, two rock_proxy nodes
// (baseColorFactor 0.8, no texture) sit inside 82 textured rocks, and where the
// hull wins the picture shows a bright grey patch where the rocks should be --
// intermittently, because anything that perturbs the acceleration structure
// flips which one is in front.
//
// Only the mesh is dropped. The node still exists and its children are still
// walked, so the hierarchy, cameras and skins are unaffected.
bool lodFilterEnabled()
{
    static const bool enabled = !envFlag("STRELKA_NO_LOD_FILTER");
    return enabled;
}

// Reported once per load: silently dropping geometry is exactly the kind of
// thing that must not be discovered by wondering where an object went.
namespace
{
uint32_t& lodSkipCounter()
{
    static uint32_t skipped = 0;
    return skipped;
}
} // namespace

bool isProxyOrLowerLod(const std::string& name)
{
    std::string lower(name.size(), '\0');
    std::transform(name.begin(), name.end(), lower.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });

    if (lower.find("proxy") != std::string::npos)
    {
        return true;
    }
    // "lod0" / "lod_0" is the level to keep; anything above it is an alternative
    // to the same geometry. A trailing "lod" with no digit names nothing in
    // particular and is left alone.
    for (size_t pos = lower.find("lod"); pos != std::string::npos; pos = lower.find("lod", pos + 3))
    {
        size_t d = pos + 3;
        if (d < lower.size() && lower[d] == '_')
        {
            ++d;
        }
        if (d >= lower.size() || std::isdigit((unsigned char)lower[d]) == 0)
        {
            continue;
        }
        size_t end = d;
        while (end < lower.size() && std::isdigit((unsigned char)lower[end]) != 0)
        {
            ++end;
        }
        if (lower.substr(d, end - d).find_first_not_of('0') != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

// packNormal(), packUV(), unpackNormal(), unpackUV() provided by <strelka/scene/vertex_packing.h>
// packTangent uses same format as packNormal (tangents are unit vectors in [-1,1])

void computeTangent(std::vector<Scene::Vertex>& vertices,
                                 const std::vector<uint32_t>& indices)
{
    const size_t lastIndex = indices.size();
    Scene::Vertex& v0 = vertices[indices[lastIndex - 3]];
    Scene::Vertex& v1 = vertices[indices[lastIndex - 2]];
    Scene::Vertex& v2 = vertices[indices[lastIndex - 1]];

    glm::float2 uv0 = unpackUV(v0.uv);
    glm::float2 uv1 = unpackUV(v1.uv);
    glm::float2 uv2 = unpackUV(v2.uv);

    glm::float3 deltaPos1 = v1.pos - v0.pos;
    glm::float3 deltaPos2 = v2.pos - v0.pos;
    glm::vec2 deltaUV1 = uv1 - uv0;
    glm::vec2 deltaUV2 = uv2 - uv0;

    glm::vec3 tangent{ 0.0f, 0.0f, 1.0f };
    const float d = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;
    if (abs(d) > 1e-6)
    {
        float r = 1.0f / d;
        tangent = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r;
    }

    glm::uint32_t packedTangent = packNormal(tangent);

    v0.tangent = packedTangent;
    v1.tangent = packedTangent;
    v2.tangent = packedTangent;
}

// Maps a glTF (mesh, primitive) onto the oka mesh built for it, so geometry
// referenced by many nodes is parsed and uploaded once.
using MeshCache = std::unordered_map<uint64_t, uint32_t>;

void processPrimitive(const tinygltf::Model& model, oka::Scene& scene, const uint32_t parentNodeId, const tinygltf::Primitive& primitive, const glm::float4x4& transform, const float globalScale, MeshCache& meshCache, uint64_t primitiveKey)
{
    using namespace std;
    assert(primitive.attributes.find("POSITION") != primitive.attributes.end());

    // A glTF mesh referenced by more than one node is one mesh, not one per
    // node. The pine forest scatters 34 539 placements over 24 objects; built
    // per node that is 34 539 copies of vertices that are bit-identical, and
    //34 539 acceleration structures over them -- the instancing the scene is
    // made of buys nothing at all.
    //
    // Safe because the vertices here are in object space: only globalScale is
    // folded in, the node transform goes to the instance. Checked before the
    // accessors are read, so a hit skips the parse as well as the upload.
    {
        const auto cached = meshCache.find(primitiveKey);
        if (cached != meshCache.end())
        {
            int cachedMatId = primitive.material;
            if (cachedMatId == -1)
                cachedMatId = 0;
            const uint32_t instId =
                scene.createInstance(Instance::Type::eMesh, cached->second, cachedMatId, transform);
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
    int posStride = byteStride / sizeof(float);

    // Normals
    const float* normalsData = nullptr;
    int normalStride = 0;
    if (primitive.attributes.find("NORMAL") != primitive.attributes.end())
    {
        const tinygltf::Accessor& normalAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
        const tinygltf::BufferView& normView = model.bufferViews[normalAccessor.bufferView];
        normalsData = reinterpret_cast<const float*>(&(model.buffers[normView.buffer].data[normalAccessor.byteOffset + normView.byteOffset]));
        assert(normalsData != nullptr);
        normalStride = normalAccessor.ByteStride(normView) / sizeof(float);
        assert(normalStride > 0);
    }

    // UVs
    const float* texCoord0Data = nullptr;
    int texCoord0Stride = 0;
    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
    {
        const tinygltf::Accessor& uvAccessor = model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
        const tinygltf::BufferView& uvView = model.bufferViews[uvAccessor.bufferView];
        texCoord0Data = reinterpret_cast<const float*>(&(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
        texCoord0Stride = uvAccessor.ByteStride(uvView) / sizeof(float);
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
        tangentStride = tanAccessor.ByteStride(tanView) / sizeof(float);
        assert(tangentStride > 0);
    }

    // Vertex colours. glTF allows VEC3 or VEC4, as float or as normalised
    // unsigned byte/short, and the values are linear multipliers on base colour.
    const void* colorData = nullptr;
    int colorStride = 0;          // in components, not bytes
    int colorComponents = 4;
    int colorComponentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    if (primitive.attributes.find("COLOR_0") != primitive.attributes.end())
    {
        const tinygltf::Accessor& ca = model.accessors[primitive.attributes.find("COLOR_0")->second];
        const tinygltf::BufferView& cv = model.bufferViews[ca.bufferView];
        colorData = reinterpret_cast<const void*>(&model.buffers[cv.buffer].data[ca.byteOffset + cv.byteOffset]);
        colorComponents = ca.type == TINYGLTF_TYPE_VEC3 ? 3 : 4;
        colorComponentType = ca.componentType;
        const int elemSize = colorComponentType == TINYGLTF_COMPONENT_TYPE_FLOAT          ? 4
                             : colorComponentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 2
                                                                                            : 1;
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
    if ( (primitive.attributes.find("JOINTS_0") != primitive.attributes.end()) && (primitive.attributes.find("WEIGHTS_0") != primitive.attributes.end()) )
    {
        hasJoints = true;
        const tinygltf::Accessor& jointsAccessor = model.accessors[primitive.attributes.find("JOINTS_0")->second];
        const tinygltf::BufferView& jointsView = model.bufferViews[jointsAccessor.bufferView];
        switch (jointsAccessor.componentType)
        {
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
            jointsData = reinterpret_cast<const void*>(&model.buffers[jointsView.buffer].data[jointsAccessor.byteOffset + jointsView.byteOffset]);
            jointsStride = jointsAccessor.ByteStride(jointsView) / sizeof(uint32_t);
            assert(jointsData != nullptr);
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
            jointsData = reinterpret_cast<const void*>(&model.buffers[jointsView.buffer].data[jointsAccessor.byteOffset + jointsView.byteOffset]);
            jointsStride = jointsAccessor.ByteStride(jointsView) / sizeof(uint16_t);
            assert(jointsData != nullptr);
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
            jointsData = reinterpret_cast<const void*>(&model.buffers[jointsView.buffer].data[jointsAccessor.byteOffset + jointsView.byteOffset]);
            jointsStride = jointsAccessor.ByteStride(jointsView) / sizeof(uint8_t);
            assert(jointsData != nullptr);
            break;
        }
        default:
            std::cerr << "Joint component type " << jointsAccessor.componentType << " not supported" << std::endl;
            return;
        }
        assert(jointsStride > 0);

        const tinygltf::Accessor& weightsAccessor = model.accessors[primitive.attributes.find("WEIGHTS_0")->second];
        const tinygltf::BufferView& weightsView = model.bufferViews[weightsAccessor.bufferView];
        weightsData = reinterpret_cast<const float*>(&model.buffers[weightsView.buffer].data[weightsAccessor.byteOffset + weightsView.byteOffset]);
        assert(weightsData != nullptr);
        weightsStride = weightsAccessor.ByteStride(weightsView) / sizeof(float);
        assert(weightsStride > 0);

        sb.reserve(vertexCount);
    }

    glm::float3 sum = glm::float3(0.0f, 0.0f, 0.0f);
    std::vector<oka::Scene::Vertex> vertices;
    vertices.reserve(vertexCount);
    for (uint32_t v = 0; v < vertexCount; ++v)
    {
        oka::Scene::Vertex vertex{};
        glm::float3 vPos = glm::make_vec3(&positionData[v * posStride]) * globalScale;
        glm::float3 vNorm = glm::vec3(normalsData ? glm::make_vec3(&normalsData[v * normalStride]) : glm::vec3(0.0f));
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
                    c[k] = src[k] / 255.0f;
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                const uint16_t* src = static_cast<const uint16_t*>(colorData) + v * colorStride;
                for (int k = 0; k < colorComponents; ++k)
                    c[k] = src[k] / 65535.0f;
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
        vertices.push_back(vertex);
        sum += vertex.pos;

        if (hasJoints)
        {
            oka::Scene::vertexSkinData skinData{};
            const tinygltf::Accessor& jointsAccessor = model.accessors[primitive.attributes.find("JOINTS_0")->second];
            switch (jointsAccessor.componentType)
            {
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
                    const uint32_t* jointsDataCasted = static_cast<const uint32_t*>(jointsData);
                    skinData.joints = glm::ivec4(
                        static_cast<int>(jointsDataCasted[v * jointsStride + 0]),
                        static_cast<int>(jointsDataCasted[v * jointsStride + 1]),
                        static_cast<int>(jointsDataCasted[v * jointsStride + 2]),
                        static_cast<int>(jointsDataCasted[v * jointsStride + 3])
                    );
                    break;
                }
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
                    const uint16_t* jointsDataCasted = static_cast<const uint16_t*>(jointsData);
                    skinData.joints = glm::ivec4(
                        static_cast<int>(jointsDataCasted[v * jointsStride + 0]),
                        static_cast<int>(jointsDataCasted[v * jointsStride + 1]),
                        static_cast<int>(jointsDataCasted[v * jointsStride + 2]),
                        static_cast<int>(jointsDataCasted[v * jointsStride + 3])
                    );
                    break;
                }
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
                    const uint8_t* jointsDataCasted = static_cast<const uint8_t*>(jointsData);
                    skinData.joints = glm::ivec4(
                        static_cast<int>(jointsDataCasted[v * jointsStride + 0]),
                        static_cast<int>(jointsDataCasted[v * jointsStride + 1]),
                        static_cast<int>(jointsDataCasted[v * jointsStride + 2]),
                        static_cast<int>(jointsDataCasted[v * jointsStride + 3])
                    );
                    break;
                }
                default:
                    std::cerr << "Joint component type " << jointsAccessor.componentType << " not supported!" << std::endl;
                    return;
            }
            skinData.weights = glm::make_vec4(&weightsData[v * weightsStride]);
            skinData.pos = vPos;
            skinData.normal = vNorm;
            sb.push_back(skinData);
        }
    }
    const glm::float3 massCenter = sum / (float)vertexCount;

    uint32_t indexCount = 0;
    std::vector<uint32_t> indices;
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

        indices.reserve(indexCount);
        switch (accessor.componentType)
        {
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
            const auto* buf = static_cast<const uint32_t*>(dataPtr);
            for (size_t index = 0; index < indexCount; index++)
            {
                indices.push_back(buf[index]);
            }
            if (!tangentData)
                computeTangent(vertices, indices);
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
            const auto* buf = static_cast<const uint16_t*>(dataPtr);
            for (size_t index = 0; index < indexCount; index++)
            {
                indices.push_back(buf[index]);
            }
            if (!tangentData)
                computeTangent(vertices, indices);
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
            const auto* buf = static_cast<const uint8_t*>(dataPtr);
            for (size_t index = 0; index < indexCount; index++)
            {
                indices.push_back(buf[index]);
            }
            if (!tangentData)
                computeTangent(vertices, indices);
            break;
        }
        default:
            std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
            return;
        }
    }

    // Copy computed tangent from vertices into skin data (tangent is computed after vertex loop)
    if (hasJoints)
    {
        for (size_t v = 0; v < sb.size(); ++v)
        {
            sb[v].tangent = vertices[v].tangent;
        }
    }

    uint32_t meshId = -1;
    if (hasJoints)
        meshId = scene.createSkeletalMesh(vertices, indices, sb);
    else
        meshId = scene.createMesh(vertices, indices);
    assert(meshId != -1);
    // Skinned meshes are deliberately never cached: their vertices are rewritten
    // per frame from their own skin, so two nodes sharing one would deform the
    // same geometry twice.
    if (!hasJoints)
        meshCache.emplace(primitiveKey, meshId);
    uint32_t instId = scene.createInstance(Instance::Type::eMesh, meshId, matId, transform);
    assert(instId != -1);
    scene.mNodes[parentNodeId].instanceIds.push_back(instId);
}

void processMesh(const tinygltf::Model& model, oka::Scene& scene, const uint32_t parentNodeId, const tinygltf::Mesh& mesh, const glm::float4x4& transform, const float globalScale, MeshCache& meshCache, uint32_t meshIndex)
{
    if (gltfDebugLoggingEnabled())
    {
        STRELKA_DEBUG("glTF mesh '{}' has {} primitives", mesh.name, mesh.primitives.size());
    }
    uint64_t primitiveIndex = 0;
    for (const auto& primitive : mesh.primitives)
    {
        processPrimitive(model, scene, parentNodeId, primitive, transform, globalScale, meshCache,
                         ((uint64_t)meshIndex << 32) | primitiveIndex++);
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
            const float floatRotation[4] = {
                (float)node.rotation[3],
                (float)node.rotation[0],
                (float)node.rotation[1],
                (float)node.rotation[2],
            };
            rotation = glm::make_quat(floatRotation);
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
void readGpuInstancing(const tinygltf::Model& model, const tinygltf::Node& node,
                       std::vector<glm::float4x4>& out)
{
    const auto ext = node.extensions.find("EXT_mesh_gpu_instancing");
    if (ext == node.extensions.end() || !ext->second.Has("attributes"))
        return;
    const tinygltf::Value& attributes = ext->second.Get("attributes");

    auto readAttribute = [&](const char* name, int components,
                             std::vector<float>& values) -> size_t {
        if (!attributes.Has(name))
            return 0;
        const int index = attributes.Get(name).GetNumberAsInt();
        if (index < 0 || (size_t)index >= model.accessors.size())
            return 0;
        const tinygltf::Accessor& accessor = model.accessors[index];
        if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
        {
            STRELKA_WARNING("EXT_mesh_gpu_instancing {} is not float; ignoring", name);
            return 0;
        }
        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
        const unsigned char* base =
            model.buffers[view.buffer].data.data() + view.byteOffset + accessor.byteOffset;
        const size_t stride = accessor.ByteStride(view) ? accessor.ByteStride(view)
                                                        : sizeof(float) * components;
        values.resize(accessor.count * components);
        for (size_t i = 0; i < accessor.count; ++i)
        {
            std::memcpy(&values[i * components], base + i * stride, sizeof(float) * components);
        }
        return accessor.count;
    };

    std::vector<float> translation, rotation, scale;
    size_t count = readAttribute("TRANSLATION", 3, translation);
    count = std::max(count, readAttribute("ROTATION", 4, rotation));
    count = std::max(count, readAttribute("SCALE", 3, scale));
    if (count == 0)
        return;

    out.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        const glm::float3 t = translation.empty()
                                  ? glm::float3(0.0f)
                                  : glm::float3(translation[i * 3], translation[i * 3 + 1],
                                                translation[i * 3 + 2]);
        // glTF stores a quaternion xyzw; glm::quat takes w first.
        const glm::quat r = rotation.empty()
                                ? glm::quat(1.0f, 0.0f, 0.0f, 0.0f)
                                : glm::quat(rotation[i * 4 + 3], rotation[i * 4], rotation[i * 4 + 1],
                                            rotation[i * 4 + 2]);
        const glm::float3 s = scale.empty()
                                  ? glm::float3(1.0f)
                                  : glm::float3(scale[i * 3], scale[i * 3 + 1], scale[i * 3 + 2]);
        out[i] = glm::translate(glm::float4x4(1.0f), t) * glm::mat4_cast(r) *
                 glm::scale(glm::float4x4(1.0f), s);
    }
}

void processNode(const tinygltf::Model& model, oka::Scene& scene, const tinygltf::Node& node,
                 const uint32_t currentNodeId, const glm::float4x4& baseTransform, const float globalScale,
                 MeshCache& meshCache, const std::vector<int>& cameraIndexMap)
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

        // EXT_mesh_gpu_instancing: the node's mesh is drawn once per entry in
        // the TRANSLATION/ROTATION/SCALE accessors, and the node itself is not
        // drawn on its own.
        //
        // A scattered scene is almost entirely this. Written as one node per
        // placement -- the obvious first version -- the pine forest's 2.2 M
        // placements came to 763 MB of JSON and a third of the load time was
        // spent reading it back as text.
        std::vector<glm::float4x4> instanceTransforms;
        readGpuInstancing(model, node, instanceTransforms);

        if (!instanceTransforms.empty())
        {
            for (const glm::float4x4& instance : instanceTransforms)
            {
                processMesh(model, scene, currentNodeId, mesh, globalTransform * instance, globalScale,
                            meshCache, (uint32_t)node.mesh);
            }
        }
        else
        {
            processMesh(model, scene, currentNodeId, mesh, globalTransform, globalScale, meshCache, (uint32_t)node.mesh);
        }

        //skin binding
        if (node.skin != -1)
        {
            scene.mNodes[currentNodeId].skin = node.skin;
            scene.mSkines[node.skin].refNodeId = currentNodeId;
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
            camera.node = currentNodeId;
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

    for (int childIdx : node.children)
    {
        if (scene.mNodes[currentNodeId].type == oka::Scene::Node::NodeType::unknown)
            scene.mNodes[currentNodeId].type = oka::Scene::Node::NodeType::sceneGraph;
        scene.mNodes[childIdx].parent = currentNodeId;
        processNode(model, scene, model.nodes[childIdx], childIdx, globalTransform, globalScale, meshCache,
                    cameraIndexMap);
    }
}

std::string getTextureUri(const tinygltf::Model& model, int texIndex)
{
    if (texIndex < 0)
        return {};
    const auto imageId = model.textures[texIndex].source;
    const std::string& uri = model.images[imageId].uri;

    // A glTF URI is percent-encoded, and a filename is not: an exporter that
    // writes "Material #449.png" stores "Material%20%23449.png", which opens
    // nothing. Spaces in texture names are common enough in DCC exports that
    // this shows up as a single missing texture rather than as an obvious fault.
    std::string decoded;
    if (tinygltf::URIDecode(uri, &decoded, nullptr))
        return decoded;
    return uri;
}

// Read one scalar out of a KHR_materials_* extension, falling back to the
// spec default when the extension or the key is absent.
//
// Blender writes the whole Principled BSDF through these: ior, specular,
// transmission, anisotropy and emissive strength all leave as extensions rather
// than as core glTF fields. Every one of them used to be hardcoded below, which
// is why a scene could round-trip through glTF carrying the right numbers and
// still render with none of them.
// KHR_texture_transform lives on the texture *slot*, not the material, so it has
// to be dug out of whichever slot carries one. Blender drives every slot of a
// material from a single Mapping node, so taking the first is not a compromise
// in practice -- and without it a tiled texture authored at scale 0.1 renders
// ten times too large.
static void readTextureTransform(const tinygltf::Material& material, MaterialParams& p)
{
    p.uv_offset_x = 0.0f;
    p.uv_offset_y = 0.0f;
    p.uv_scale_x = 1.0f;
    p.uv_scale_y = 1.0f;
    p.uv_rotation = 0.0f;

    const tinygltf::ExtensionMap* slots[] = {
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

static float khrFloat(const tinygltf::Material& material,
                      const char* extension,
                      const char* key,
                      float fallback)
{
    const auto it = material.extensions.find(extension);
    if (it == material.extensions.end() || !it->second.IsObject() || !it->second.Has(key))
        return fallback;
    return (float)it->second.Get(key).GetNumberAsDouble();
}

oka::Scene::MaterialDescription convertToStandardPBR(const tinygltf::Model& model, const tinygltf::Material& material)
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
    p.base_color = {(float)bcf[0], (float)bcf[1], (float)bcf[2]};
    p.base_color_alpha = (float)bcf[3];
    p.alpha_mode = material.alphaMode == "MASK"    ? ALPHA_MODE_MASK
                   : material.alphaMode == "BLEND" ? ALPHA_MODE_BLEND
                                                   : ALPHA_MODE_OPAQUE;

    // Metallic / roughness
    p.roughness = (float)material.pbrMetallicRoughness.roughnessFactor;
    p.metallic  = (float)material.pbrMetallicRoughness.metallicFactor;

    // IOR / specular / transmission / anisotropy, from the KHR extensions.
    // specularFactor is a 0..1 multiplier on the dielectric F0, and glTF's
    // default of 1.0 corresponds to Strelka's specular 0.5 -- so halve it, or a
    // material Blender exported with specularFactor 0 still gets an F0 = 0.04
    // lobe it was never meant to have.
    p.ior = khrFloat(material, "KHR_materials_ior", "ior", 1.5f);
    p.specular = 0.5f * khrFloat(material, "KHR_materials_specular", "specularFactor", 1.0f);
    // specularColorFactor, white when the extension is absent or silent about it.
    p.specular_color = glm::float3(1.0f);
    {
        const auto sp = material.extensions.find("KHR_materials_specular");
        if (sp != material.extensions.end() && sp->second.IsObject() &&
            sp->second.Has("specularColorFactor"))
        {
            const tinygltf::Value& c = sp->second.Get("specularColorFactor");
            if (c.IsArray() && c.ArrayLen() >= 3)
            {
                p.specular_color = { (float)c.Get(0).GetNumberAsDouble(),
                                     (float)c.Get(1).GetNumberAsDouble(),
                                     (float)c.Get(2).GetNumberAsDouble() };
            }
        }
    }
    p.transmission = khrFloat(material, "KHR_materials_transmission", "transmissionFactor", 0.0f);

    // KHR_materials_diffuse_transmission. Foliage: light enters the leaf and
    // leaves diffusely on the far side. Blender writes this through a Translucent
    // BSDF, which the glTF exporter cannot express, so export_scene.py injects
    // the extension after the fact -- see flatten_materials.py.
    p.diffuse_transmission =
        khrFloat(material, "KHR_materials_diffuse_transmission", "diffuseTransmissionFactor", 0.0f);
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
    // KHR_materials_sheen. The extension carries a colour and a roughness and no
    // separate weight, so the weight is the colour's peak channel and the colour
    // is normalised by it -- that keeps a dim grey sheen dim rather than turning
    // it into a full-strength grey layer, and leaves sheen at 0 (the lobe off)
    // when the extension is absent.
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
                const glm::float3 sheenColor((float)c.Get(0).GetNumberAsDouble(),
                                             (float)c.Get(1).GetNumberAsDouble(),
                                             (float)c.Get(2).GetNumberAsDouble());
                p.sheen = std::max(sheenColor.x, std::max(sheenColor.y, sheenColor.z));
                if (p.sheen > 0.0f)
                    p.sheen_color = sheenColor / p.sheen;
            }
        }
    }

    // STRELKA_materials_subsurface. Not a ratified extension: glTF has nothing
    // for subsurface, and the alternative was to fold it into
    // KHR_materials_diffuse_transmission, which describes a leaf -- light out the
    // far side immediately -- and not a random walk through wax.
    //
    // scatterColor is the *single-scattering* albedo, not the diffuse albedo a
    // DCC shows in its colour picker. The two are related by an inversion that
    // needs a fit, and stating which one this field is beats implementing the
    // fit badly. Values near 1 are what marble, soap and skin want.
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
                    p.subsurface_radius = { (float)r.Get(0).GetNumberAsDouble(),
                                            (float)r.Get(1).GetNumberAsDouble(),
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
                    p.subsurface_reference = { (float)r.Get(0).GetNumberAsDouble(),
                                               (float)r.Get(1).GetNumberAsDouble(),
                                               (float)r.Get(2).GetNumberAsDouble() };
                }
            }
            p.subsurface = khrFloat(material, "STRELKA_materials_subsurface", "subsurfaceFactor", 1.0f);

            // A zero mean free path is an infinitely dense medium, i.e. a walk
            // that never terminates. Treat it as "no medium" rather than as a
            // hang.
            if (p.subsurface_radius.x <= 0.0f && p.subsurface_radius.y <= 0.0f &&
                p.subsurface_radius.z <= 0.0f)
            {
                p.subsurface = 0.0f;
            }

            if (p.subsurface > 0.0f)
            {
                // The medium is entered through the diffuse transmission lobe:
                // everything that would have scattered diffusely goes in instead
                // and comes back out of the walk. scatterColor rides on
                // diffuse_transmission_color, which is what the walk uses as its
                // single-scattering albedo.
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

    // STRELKA_materials_medium: a participating medium bounded by the geometry
    // carrying this material. V-Ray's EnvironmentFog with a gizmo.
    //
    // The medium's interior parameters ride on the subsurface fields -- a fog
    // volume and a block of wax differ in where light enters, not in what happens
    // once it is inside -- so only the emission and the boundary flag are read
    // here. `density` is the extinction, i.e. the reciprocal of the mean free
    // path, because that is the number a DCC's fog gizmo exposes.
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
                p.subsurface_anisotropy =
                    khrFloat(material, "STRELKA_materials_medium", "anisotropy", 0.0f);
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
                        p.medium_emission = { (float)c.Get(0).GetNumberAsDouble(),
                                              (float)c.Get(1).GetNumberAsDouble(),
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
    p.iridescence_thickness =
        khrFloat(material, "KHR_materials_iridescence", "iridescenceThicknessMaximum", 400.0f);

    p.clearcoat = khrFloat(material, "KHR_materials_clearcoat", "clearcoatFactor", 0.0f);
    p.clearcoat_roughness =
        khrFloat(material, "KHR_materials_clearcoat", "clearcoatRoughnessFactor", 0.0f);
    // Not in KHR_materials_clearcoat, which fixes the coat at a clear lacquer.
    // Blender writes it into extras, and the ceramics in the bathroom scene are
    // authored at 2.0 -- an F0 of 0.111 against the extension's 0.04, which is
    // most of the difference between glazed and painted.
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
        if (vit != material.extensions.end() && vit->second.IsObject() &&
            vit->second.Has("attenuationColor"))
        {
            const tinygltf::Value& c = vit->second.Get("attenuationColor");
            if (c.IsArray() && c.ArrayLen() >= 3)
            {
                p.attenuation_color = { (float)c.Get(0).GetNumberAsDouble(),
                                        (float)c.Get(1).GetNumberAsDouble(),
                                        (float)c.Get(2).GetNumberAsDouble() };
            }
        }
    }

    // Emission. emissiveFactor is clamped to [0,1] by the spec, so anything
    // brighter than 1 leaves in KHR_materials_emissive_strength -- which is
    // exactly the field this used to overwrite with a presence flag, collapsing
    // every emitter in every Blender export to 1x.
    const auto& emf = material.emissiveFactor;
    p.emission = {(float)emf[0], (float)emf[1], (float)emf[2]};
    p.emission_strength =
        khrFloat(material, "KHR_materials_emissive_strength", "emissiveStrength", 1.0f);

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
    // Thin-walled: a surface with no interior, so light passes straight through
    // instead of refracting twice. A soap bubble, not a marble.
    //
    // glTF says a transmissive material is thin-walled unless KHR_materials_volume
    // gives it a non-zero thickness, and taken literally that would be right. It
    // is not followed here, because Blender only writes that extension for a
    // specific node setup ("glTF Material Output" with a Thickness socket): under
    // the literal reading every ordinary glass export becomes a bubble. Absence
    // is read as solid, and thin-walledness has to be stated.
    //
    // What this fixes: the bubbles floating on the bath water rendered as dark
    // specks, because a solid sphere of IOR 1.6 refracts into itself and the path
    // dies before it gets out.
    p.thin_walled = 0;
    {
        const auto vit = material.extensions.find("KHR_materials_volume");
        if (vit != material.extensions.end() && vit->second.IsObject())
        {
            const float thickness =
                khrFloat(material, "KHR_materials_volume", "thicknessFactor", 0.0f);
            p.thin_walled = (thickness <= 0.0f) ? 1u : 0u;
        }
    }
    // A transmissive surface is a dielectric volume and needs a priority so the
    // nested-dielectric IOR stack can order it; an opaque one must stay at 0.
    p.dielectric_priority = p.transmission > 0.0f ? 10u : 0u;

    readTextureTransform(material, p);

    // Store texture file paths for the renderer to load
    desc.baseColorTexPath = getTextureUri(model, material.pbrMetallicRoughness.baseColorTexture.index);
    desc.metallicRoughnessTexPath = getTextureUri(model, material.pbrMetallicRoughness.metallicRoughnessTexture.index);
    desc.normalTexPath = getTextureUri(model, material.normalTexture.index);
    desc.emissionTexPath = getTextureUri(model, material.emissiveTexture.index);
    desc.occlusionTexPath = getTextureUri(model, material.occlusionTexture.index);

    return desc;
}

void loadMaterials(const tinygltf::Model& model, oka::Scene& scene)
{
    for (const tinygltf::Material& material : model.materials)
    {
        // alphaMode describes opacity, not material type. Routing MASK and BLEND
        // to the dielectric converter turned every cutout and every blended
        // surface into rough glass -- and, worse, silently dropped all five
        // texture paths, because convertToDielectric never assigns them. Glass
        // arrives through KHR_materials_transmission instead, which
        // convertToStandardPBR now parses into a transmission lobe.
        scene.addMaterial(convertToStandardPBR(model, material));
    }
}

// Fills `gltfToScene` with one entry per glTF camera: the index of the camera it
// became in the scene, or -1 if it was not loaded.
//
// The map is the point. Camera *nodes* address cameras by glTF index, so if this
// function ever appends fewer cameras than the file declares, every later index
// is off by the number skipped and the last one addresses past the end of the
// vector. That went unnoticed for a while because it does not crash: it corrupts
// a projection matrix, which then compares unequal to itself in MetalRender's
// "did the camera move" test and resets the accumulator on every single frame.
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
            camera.fov = cameraGltf.perspective.yfov * (180.0f / 3.1415926f);
            camera.authoredAspect = (float)cameraGltf.perspective.aspectRatio;
            camera.znear = cameraGltf.perspective.znear;
            camera.zfar = cameraGltf.perspective.zfar;
        }
        else if (cameraGltf.type == "orthographic")
        {
            // glTF xmag/ymag are half-extents, so they go straight across; the
            // authored aspect follows from them and needs no separate field.
            camera.projection = oka::Camera::ProjectionType::orthographic;
            camera.xmag = (float)cameraGltf.orthographic.xmag;
            camera.ymag = (float)cameraGltf.orthographic.ymag;
            camera.authoredAspect = (camera.ymag > 0.0f) ? (camera.xmag / camera.ymag) : 0.0f;
            camera.znear = cameraGltf.orthographic.znear;
            camera.zfar = cameraGltf.orthographic.zfar;
        }
        else
        {
            STRELKA_WARNING("glTF camera {} '{}': unknown type '{}', skipped", i, cameraGltf.name,
                            cameraGltf.type);
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
                    std::cout << "unknown type" << std::endl;
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
                std::cout << "weights not yet supported, skipping channel" << std::endl;
                continue;
            }
            chan.samplerIndex = channel.sampler;
            chan.node = channel.target_node;
            if (chan.node < 0)
            {
                std::cout << "node id < 0, skipping channel" << std::endl;
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

        //glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        glm::quat rotation = glm::quat_cast(glm::float4x4(1.0f));
        if (!node.rotation.empty())
        {
            const float floatRotation[4] = {
                (float)node.rotation[3],
                (float)node.rotation[0],
                (float)node.rotation[1],
                (float)node.rotation[2],
            };
            rotation = glm::make_quat(floatRotation);
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

void loadSkeletalData(const tinygltf::Model& model, oka::Scene& scene, const float globalScale = 1.0f)
{
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
        const int matStride = matrixAccessor.ByteStride(bufferView) / sizeof(float);;
        assert(matStride > 0);

        for (const int jointid : s.joints)
        {
            scene.mNodes[jointid].type = oka::Scene::Node::NodeType::skeleton;
            //s.inverseBindMatrices.push_back(glm::inverse(scene.calculateNodeGlobalTransform(jointid)));
        }
        for (uint32_t m = 0; m < matrixCount; ++m)
        {
            glm::mat4 inverseBindMatrix = glm::make_mat4(&matrixData[m * matStride]);
            s.inverseBindMatrices.push_back(inverseBindMatrix);
        }

        scene.mSkines.push_back(s);
    }
}


// Curves ride in a binary sidecar; see curve_sidecar.h for the format and for
// why glTF cannot carry them. Named after the model rather than scanned for,
// unlike the light sidecar: a directory holding two converted scenes would
// otherwise give one of them the other's hair.
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

bool loadLightsFromJson(const std::string& modelPath, oka::Scene& scene)
{
    // First try exact match: <modelname>_light.json
    std::string fileName = modelPath.substr(0, modelPath.rfind('.')); // w/o extension
    std::string jsonPath = fileName + "_light" + ".json";

    // If not found, scan directory for any *_light.json file
    if (!fs::exists(jsonPath))
    {
        fs::path dir = fs::path(modelPath).parent_path();
        jsonPath.clear();
        for (const auto& entry : fs::directory_iterator(dir))
        {
            if (entry.is_regular_file())
            {
                std::string name = entry.path().filename().string();
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

// Peak luminous efficacy, lm/W. KHR_lights_punctual intensity is photometric
// (candela for point/spot, lux for directional) while everything downstream of
// UniformLightDesc is radiometric, so the two are exactly this factor apart.
// Blender's exporter multiplies watts by the same constant on the way out, so
// dividing here round-trips: 27175.7 cd / 683 = 39.79 W/sr = 500 W / 4pi.
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
        if (lightIndex < 0 || lightIndex >= (int)lightsArr.ArrayLen())
            continue;
        const tinygltf::Value& L = lightsArr.Get(lightIndex);
        if (!L.IsObject())
            continue;

        Scene::UniformLightDesc desc{};
        desc.enabled = true;
        desc.name = L.Has("name") ? L.Get("name").Get<std::string>() : node.name;
        desc.color = L.Has("color") ? readVec3(L.Get("color"), glm::float3(1.0f)) : glm::float3(1.0f);
        desc.intensity =
            (L.Has("intensity") ? (float)L.Get("intensity").GetNumberAsDouble() : 1.0f) / kLumensPerWatt;
        desc.range = L.Has("range") ? (float)L.Get("range").GetNumberAsDouble() : 0.0f;

        const std::string type = L.Has("type") ? L.Get("type").Get<std::string>() : "point";
        if (type == "directional")
        {
            desc.type = LIGHT_TYPE_DISTANT;
            desc.intensityUnit = LIGHT_UNIT_IRRADIANCE;
            // glTF has no sun angular size; use a small disk so soft shadows work.
            desc.halfAngle = 0.53f * 0.5f * (float(M_PI) / 180.0f);
        }
        else if (type == "spot")
        {
            desc.type = LIGHT_TYPE_SPOT;
            desc.intensityUnit = LIGHT_UNIT_INTENSITY;
            float inner = 0.0f;
            float outer = float(M_PI) / 4.0f;
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
    std::string fileName = modelPath.substr(0, modelPath.rfind('.'));
    std::string jsonPath = fileName + "_camera.json";

    if (!fs::exists(jsonPath))
    {
        fs::path dir = fs::path(modelPath).parent_path();
        jsonPath.clear();
        for (const auto& entry : fs::directory_iterator(dir))
        {
            if (entry.is_regular_file())
            {
                std::string name = entry.path().filename().string();
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

        std::string name = cam["name"].get<std::string>();
        uint32_t idx = scene.findCameraByName(name);
        if (idx == (uint32_t)-1)
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

bool GltfLoader::loadGltf(const std::string& modelPath, oka::Scene& scene)
{
    if (modelPath.empty())
    {
        return false;
    }

    scene.setSourcePath(modelPath);

    using namespace std;
    tinygltf::Model model;
    tinygltf::TinyGLTF gltf_ctx;
    // tinygltf refuses external buffers over INT32_MAX by default. A production
    // scene passes that easily -- a 50 M triangle forest lands at 2.8 GB of
    // positions, normals, indices and uvs -- and the refusal reads as a plain
    // load failure with nothing to act on.
    gltf_ctx.SetMaxExternalFileSize(std::numeric_limits<size_t>::max());
    // Do not let tinygltf decode the images.
    //
    // The only thing this loader ever reads out of model.images is the uri --
    // the renderer opens the file itself, mips it, compresses it and keeps its
    // own cache of the result. Every pixel tinygltf decodes here is thrown
    // away, and it decodes them one at a time on the calling thread while
    // parsing. On the pine forest that was 9017 of the 14680 samples in a CPU
    // profile of the load: 61% of it, for nothing.
    //
    // An image embedded in a buffer view rather than referenced by uri would
    // lose its pixels this way -- but such an image has no file to open either,
    // so it was never supported.
    gltf_ctx.SetImageLoader(
        [](tinygltf::Image*, const int, std::string*, std::string*, int, int,
           const unsigned char*, int, void*) { return true; },
        nullptr);
    std::string err;
    std::string warn;
    bool res = false;
    const std::string ext = fs::path(modelPath).extension().string();
    // One indivisible read with no way in for a callback, and a third of the load
    // on a scene the size of the pine forest. It gets a stage of its own so the
    // UI can name what it is waiting on instead of showing a bar that does not
    // move for a second.
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

    int sceneId = model.defaultScene < 0 ? 0 : model.defaultScene;
    if (model.scenes.size() > 1)
    {
        // Only the default scene is instantiated. A file with several is
        // ambiguous by construction, and silently drawing a fraction of it looks
        // like missing geometry rather than a choice.
        STRELKA_WARNING("glTF has {} scenes; loading only '{}' (index {}). The rest are ignored.",
                        model.scenes.size(),
                        model.scenes[sceneId].name.empty() ? "<unnamed>" : model.scenes[sceneId].name,
                        sceneId);
    }

    // The load expressed as a list of phases rather than as a sequence of calls.
    //
    // Progress is then a property of the list: the loop reports it, so there is
    // no per-phase call to forget and no separately maintained total that has to
    // agree with how many of those calls there are. Adding a phase and not
    // listing it here means it does not run at all, which is a loud failure --
    // unlike a bar that quietly stops short of the end.
    //
    // The graph walk contributes one phase per root node because it is most of
    // the parse, and a bar that sits still through it is indistinguishable from
    // one that has hung.
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

    std::vector<Phase> phases;
    phases.push_back({ "materials", [&] { loadMaterials(model, scene); } });
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
                              processNode(model, scene, model.nodes[rootNodeIdx], rootNodeIdx,
                                          glm::float4x4(1.0f), globalScale, meshCache, cameraIndexMap);
                          } });
    }
    phases.push_back({ "geometry", [&] {
                          if (lodSkipCounter() != 0)
                          {
                              STRELKA_INFO("Skipped {} proxy / non-zero-LOD nodes; set STRELKA_NO_LOD_FILTER=1 "
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
                              lightDesc.halfAngle = 10.0f * 0.5f * (float(M_PI) / 180.0f);
                              lightDesc.intensity = 100000;
                              lightDesc.color = glm::float3(1.0);
                              scene.createLight(lightDesc);
                          }
                      } });
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

    // Geometry, skins, and animations have copied all binary data into the
    // scene's own arrays. Holding the source buffers while the renderer builds
    // acceleration structures can make otherwise valid large scenes run out of
    // memory.
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
            STRELKA_INFO("Released {:.2f} GB of glTF buffer data after loading scene data",
                         released / 1e9);
        }
    }

    return res;
}
} // namespace oka
