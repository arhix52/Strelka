#include <strelka/sceneloader/gltfloader.h>
#include <strelka/sceneloader/sceneserializer.h>
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
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/compatibility.hpp>

#include <strelka/scene/transform.h>

#include <iostream>
#include <log.h>

namespace fs = std::filesystem;

#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace oka
{

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

void processPrimitive(const tinygltf::Model& model, oka::Scene& scene, const uint32_t parentNodeId, const tinygltf::Primitive& primitive, const glm::float4x4& transform, const float globalScale)
{
    using namespace std;
    assert(primitive.attributes.find("POSITION") != primitive.attributes.end());

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
    uint32_t instId = scene.createInstance(Instance::Type::eMesh, meshId, matId, transform);
    assert(instId != -1);
    scene.mNodes[parentNodeId].instanceIds.push_back(instId);
}

void processMesh(const tinygltf::Model& model, oka::Scene& scene, const uint32_t parentNodeId, const tinygltf::Mesh& mesh, const glm::float4x4& transform, const float globalScale)
{
    using namespace std;
    cout << "Mesh name: " << mesh.name << endl;
    cout << "Primitive count: " << mesh.primitives.size() << endl;
    for (const auto& primitive : mesh.primitives)
    {
        processPrimitive(model, scene, parentNodeId, primitive, transform, globalScale);
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

void processNode(const tinygltf::Model& model, oka::Scene& scene, const tinygltf::Node& node, const uint32_t currentNodeId, const glm::float4x4& baseTransform, const float globalScale)
{
    using namespace std;
    cout << "Node name: " << node.name << endl;

    const glm::float4x4 localTransform = getTransform(node, globalScale);
    const glm::float4x4 globalTransform = baseTransform * localTransform;

    if (node.mesh != -1) // mesh exist
    {
        scene.mNodes[currentNodeId].type = oka::Scene::Node::NodeType::mesh;
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        processMesh(model, scene, currentNodeId, mesh, globalTransform, globalScale);

        //skin binding
        if (node.skin != -1)
        {
            scene.mNodes[currentNodeId].skin = node.skin;
            scene.mSkines[node.skin].refNodeId = currentNodeId;
        }
    }
    else if (node.camera != -1) // camera node
    {
        scene.mNodes[currentNodeId].type = oka::Scene::Node::NodeType::camera;
        scene.mNodes[currentNodeId].camera = node.camera;
        glm::float3 scale;
        glm::quat rotation;
        glm::float3 translation;
        oka::decomposeTrs(globalTransform, translation, rotation, scale);

        rotation = glm::conjugate(rotation);

        scene.getCamera(node.camera).node = currentNodeId;
        scene.getCamera(node.camera).position = translation * scale;
        scene.getCamera(node.camera).mOrientation = rotation;
        scene.getCamera(node.camera).updateViewMatrix();
    }

    for (int childIdx : node.children)
    {
        if (scene.mNodes[currentNodeId].type == oka::Scene::Node::NodeType::unknown)
            scene.mNodes[currentNodeId].type = oka::Scene::Node::NodeType::sceneGraph;
        scene.mNodes[childIdx].parent = currentNodeId;
        processNode(model, scene, model.nodes[childIdx], childIdx, globalTransform, globalScale);
    }
}

std::string getTextureUri(const tinygltf::Model& model, int texIndex)
{
    if (texIndex < 0)
        return {};
    const auto imageId = model.textures[texIndex].source;
    return model.images[imageId].uri;
}

// Read one scalar out of a KHR_materials_* extension, falling back to the
// spec default when the extension or the key is absent.
//
// Blender writes the whole Principled BSDF through these: ior, specular,
// transmission, anisotropy and emissive strength all leave as extensions rather
// than as core glTF fields. Every one of them used to be hardcoded below, which
// is why a scene could round-trip through glTF carrying the right numbers and
// still render with none of them.
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
    p.specular_tint = 0.0f;
    p.transmission = khrFloat(material, "KHR_materials_transmission", "transmissionFactor", 0.0f);
    p.clearcoat = khrFloat(material, "KHR_materials_clearcoat", "clearcoatFactor", 0.0f);
    p.clearcoat_roughness =
        khrFloat(material, "KHR_materials_clearcoat", "clearcoatRoughnessFactor", 0.0f);
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
    p.thin_walled = 0;
    // A transmissive surface is a dielectric volume and needs a priority so the
    // nested-dielectric IOR stack can order it; an opaque one must stay at 0.
    p.dielectric_priority = p.transmission > 0.0f ? 10u : 0u;

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

void loadCameras(const tinygltf::Model& model, oka::Scene& scene)
{
    for (const auto& cameraGltf : model.cameras)
    {
        if (strcmp(cameraGltf.type.c_str(), "perspective") == 0)
        {
            oka::Camera camera;
            camera.fov = cameraGltf.perspective.yfov * (180.0f / 3.1415926f);
            camera.znear = cameraGltf.perspective.znear;
            camera.zfar = cameraGltf.perspective.zfar;
            camera.name = cameraGltf.name;
            scene.addCamera(camera);
        }
        else
        {
            // not supported
        }
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
        cout << "Animation name: " << anim.name << endl;

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
    std::string err;
    std::string warn;
    bool res = false;
    const std::string ext = fs::path(modelPath).extension().string();
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

    int sceneId = model.defaultScene;

    loadMaterials(model, scene);
    const bool hadJsonLights = loadLightsFromJson(modelPath, scene);

    loadCameras(model, scene);
    loadCamerasFromJson(modelPath, scene);

    const float globalScale = 1.0f;
    loadNodes(model, scene, globalScale);

    loadSkeletalData(model, scene, globalScale);

    for (int i = 0; i < model.scenes[sceneId].nodes.size(); ++i)
    {
        const int rootNodeIdx = model.scenes[sceneId].nodes[i];
        processNode(model, scene, model.nodes[rootNodeIdx], rootNodeIdx, glm::float4x4(1.0f), globalScale);
    }

    // Punctual lights need node world transforms, so they land after the graph.
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

    loadAnimation(model, scene);

    return res;
}
} // namespace oka
