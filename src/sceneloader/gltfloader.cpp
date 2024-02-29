#include "gltfloader.h"

#include "camera.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYOBJLOADER_IMPLEMENTATION
// #include <tiny_obj_loader.h>
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <iostream>

namespace fs = std::filesystem;

#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace oka
{

//  valid range of coordinates [-10; 10]
uint32_t packUV(const glm::float2& uv)
{
    int32_t packed = (uint32_t)((uv.x + 10.0f) / 20.0f * 16383.99999f);
    packed += (uint32_t)((uv.y + 10.0f) / 20.0f * 16383.99999f) << 16;
    return packed;
}

//  valid range of coordinates [-1; 1]
uint32_t packNormal(const glm::float3& normal)
{
    uint32_t packed = (uint32_t)((normal.x + 1.0f) / 2.0f * 511.99999f);
    packed += (uint32_t)((normal.y + 1.0f) / 2.0f * 511.99999f) << 10;
    packed += (uint32_t)((normal.z + 1.0f) / 2.0f * 511.99999f) << 20;
    return packed;
}

//  valid range of coordinates [-10; 10]
uint32_t packTangent(const glm::float3& tangent)
{
    uint32_t packed = (uint32_t)((tangent.x + 10.0f) / 20.0f * 511.99999f);
    packed += (uint32_t)((tangent.y + 10.0f) / 20.0f * 511.99999f) << 10;
    packed += (uint32_t)((tangent.z + 10.0f) / 20.0f * 511.99999f) << 20;
    return packed;
}

glm::float2 unpackUV(uint32_t val)
{
    glm::float2 uv;
    uv.y = ((val & 0xffff0000) >> 16) / 16383.99999f * 10.0f - 5.0f;
    uv.x = (val & 0x0000ffff) / 16383.99999f * 10.0f - 5.0f;

    return uv;
}

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

    glm::uint32_t packedTangent = packTangent(tangent);

    v0.tangent = packedTangent;
    v1.tangent = packedTangent;
    v2.tangent = packedTangent;
}

void processPrimitive(const tinygltf::Model& model, oka::Scene& scene, const tinygltf::Primitive& primitive, const glm::float4x4& transform, const float globalScale)
{
    using namespace std;
    assert(primitive.attributes.find("POSITION") != primitive.attributes.end());

    const tinygltf::Accessor& positionAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
    const tinygltf::BufferView& positionView = model.bufferViews[positionAccessor.bufferView];
    const float* positionData = reinterpret_cast<const float*>(&model.buffers[positionView.buffer].data[positionAccessor.byteOffset + positionView.byteOffset]);
    assert(positionData != nullptr);
    const uint32_t vertexCount = static_cast<uint32_t>(positionAccessor.count);
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

    int matId = primitive.material;
    if (matId == -1)
    {
        matId = 0; // TODO: should be index of default material
    }

    glm::float3 sum = glm::float3(0.0f, 0.0f, 0.0f);
    std::vector<oka::Scene::Vertex> vertices;
    vertices.reserve(vertexCount);
    for (uint32_t v = 0; v < vertexCount; ++v)
    {
        oka::Scene::Vertex vertex{};
        vertex.pos = glm::make_vec3(&positionData[v * posStride]) * globalScale;
        vertex.normal = packNormal(glm::normalize(glm::vec3(normalsData ? glm::make_vec3(&normalsData[v * normalStride]) : glm::vec3(0.0f))));
        vertex.uv = packUV(texCoord0Data ? glm::make_vec2(&texCoord0Data[v * texCoord0Stride]) : glm::vec3(0.0f));
        vertices.push_back(vertex);
        sum += vertex.pos;
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
            const uint32_t* buf = static_cast<const uint32_t*>(dataPtr);
            for (size_t index = 0; index < indexCount; index++)
            {
                indices.push_back(buf[index]);
            }
            computeTangent(vertices, indices);
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
            const uint16_t* buf = static_cast<const uint16_t*>(dataPtr);
            for (size_t index = 0; index < indexCount; index++)
            {
                indices.push_back(buf[index]);
            }
            computeTangent(vertices, indices);
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
            const uint8_t* buf = static_cast<const uint8_t*>(dataPtr);
            for (size_t index = 0; index < indexCount; index++)
            {
                indices.push_back(buf[index]);
            }
            computeTangent(vertices, indices);
            break;
        }
        default:
            std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
            return;
        }
    }

    uint32_t meshId = scene.createMesh(vertices, indices);
    assert(meshId != -1);
    uint32_t instId = scene.createInstance(Instance::Type::eMesh, meshId, matId, transform);
    assert(instId != -1);
}

void processMesh(const tinygltf::Model& model, oka::Scene& scene, const tinygltf::Mesh& mesh, const glm::float4x4& transform, const float globalScale)
{
    using namespace std;
    cout << "Mesh name: " << mesh.name << endl;
    cout << "Primitive count: " << mesh.primitives.size() << endl;
    for (size_t i = 0; i < mesh.primitives.size(); ++i)
    {
        processPrimitive(model, scene, mesh.primitives[i], transform, globalScale);
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
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        processMesh(model, scene, mesh, globalTransform, globalScale);
    }
    else if (node.camera != -1) // camera node
    {
        glm::vec3 scale;
        glm::quat rotation;
        glm::vec3 translation;
        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(globalTransform, scale, rotation, translation, skew, perspective);

        rotation = glm::conjugate(rotation);

        scene.getCamera(node.camera).node = currentNodeId;
        scene.getCamera(node.camera).position = translation * scale;
        scene.getCamera(node.camera).mOrientation = rotation;
        scene.getCamera(node.camera).updateViewMatrix();
    }

    for (int i = 0; i < node.children.size(); ++i)
    {
        scene.mNodes[node.children[i]].parent = currentNodeId;
        processNode(model, scene, model.nodes[node.children[i]], node.children[i], globalTransform, globalScale);
    }
}

// VkSamplerAddressMode getVkWrapMode(int32_t wrapMode)
// {
//     switch (wrapMode)
//     {
//     case 10497:
//         return VK_SAMPLER_ADDRESS_MODE_REPEAT;
//     case 33071:
//         return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
//     case 33648:
//         return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
//     default:
//         return VK_SAMPLER_ADDRESS_MODE_REPEAT;
//     }
// }

// VkFilter getVkFilterMode(int32_t filterMode)
// {
//     switch (filterMode)
//     {
//     case 9728:
//         return VK_FILTER_NEAREST;
//     case 9729:
//         return VK_FILTER_LINEAR;
//     case 9984:
//         return VK_FILTER_NEAREST;
//     case 9985:
//         return VK_FILTER_NEAREST;
//     case 9986:
//         return VK_FILTER_LINEAR;
//     case 9987:
//         return VK_FILTER_LINEAR;
//     default:
//         return VK_FILTER_LINEAR;
//     }
// }
std::unordered_map<uint32_t, uint32_t> texIdToModelSamp{};
std::unordered_map<uint32_t, uint32_t> modelSampIdToLoadedSampId{};

// void findTextureSamplers(const tinygltf::Model& model, oka::Scene& scene, oka::TextureManager& textureManager)
// {
//     oka::TextureManager::TextureSamplerDesc currentSamplerDesc{};
//     uint32_t samplerNumber = 0;

//     for (const tinygltf::Sampler& sampler : model.samplers)
//     {
//         currentSamplerDesc = { getVkFilterMode(sampler.minFilter), getVkFilterMode(sampler.magFilter), getVkWrapMode(sampler.wrapS), getVkWrapMode(sampler.wrapT) };
//         if (textureManager.sampDescToId.count(currentSamplerDesc) == 0)
//         {
//             if (textureManager.sampDescToId.size() < 15)
//             {
//                 textureManager.createTextureSampler(currentSamplerDesc);
//             }
//             else
//             {
//                 std::cerr << "Samplers size limit exceeded" << std::endl;
//             }
//         }
//         modelSampIdToLoadedSampId[samplerNumber] = textureManager.sampDescToId.find(currentSamplerDesc)->second;
//         ++samplerNumber;
//     }
// }

// void loadTextures(const tinygltf::Model& model, oka::Scene& scene, oka::TextureManager& textureManager)
// {
//     texIdToModelSamp[-1] = 0;
//     for (const tinygltf::Texture& tex : model.textures)
//     {
//         const tinygltf::Image& image = model.images[tex.source];
//         // TODO: create sampler for tex

//         if (image.component == 3)
//         {
//             // unsupported
//             return;
//         }
//         else if (image.component == 4)
//         {
//             // supported
//         }
//         else
//         {
//             // error
//         }

//         const void* data = image.image.data();
//         uint32_t width = image.width;
//         uint32_t height = image.height;

//         const std::string name = image.uri;

//         int texId = textureManager.loadTextureGltf(data, width, height, name);
//         assert(texId != -1);

//         texIdToModelSamp[texId] = modelSampIdToLoadedSampId.find(tex.sampler)->second;
//     }
// }

void loadMaterials(const tinygltf::Model& model, oka::Scene& scene)
{
    for (const tinygltf::Material& material : model.materials)
    {
        // oka::Scene::MaterialDescription currMaterial{};
        // currMaterial.specular = glm::float4(1.0f);
        // currMaterial.diffuse = glm::float4(material.pbrMetallicRoughness.baseColorFactor[0],
        //                                    material.pbrMetallicRoughness.baseColorFactor[1],
        //                                    material.pbrMetallicRoughness.baseColorFactor[2],
        //                                    material.pbrMetallicRoughness.baseColorFactor[3]);
        // currMaterial.texNormalId = material.normalTexture.index;
        // currMaterial.sampNormalId = texIdToModelSamp.find(currMaterial.texNormalId)->second;

        // currMaterial.baseColorFactor = glm::float4(material.pbrMetallicRoughness.baseColorFactor[0],
        //                                            material.pbrMetallicRoughness.baseColorFactor[1],
        //                                            material.pbrMetallicRoughness.baseColorFactor[2],
        //                                            material.pbrMetallicRoughness.baseColorFactor[3]);

        // currMaterial.texBaseColor = material.pbrMetallicRoughness.baseColorTexture.index;
        // currMaterial.sampBaseId = texIdToModelSamp.find(currMaterial.texBaseColor)->second;

        // currMaterial.roughnessFactor = (float)material.pbrMetallicRoughness.roughnessFactor;
        // currMaterial.metallicFactor = (float)material.pbrMetallicRoughness.metallicFactor;

        // currMaterial.texMetallicRoughness = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
        // currMaterial.sampMetallicRoughness = texIdToModelSamp.find(currMaterial.texMetallicRoughness)->second;

        // currMaterial.emissiveFactor = glm::float3(material.emissiveFactor[0],
        //                                           material.emissiveFactor[1],
        //                                           material.emissiveFactor[2]);
        // currMaterial.texEmissive = material.emissiveTexture.index;
        // currMaterial.sampEmissiveId = texIdToModelSamp.find(currMaterial.texEmissive)->second;

        // currMaterial.texOcclusion = material.occlusionTexture.index;
        // currMaterial.sampOcclusionId = texIdToModelSamp.find(currMaterial.texOcclusion)->second;

        // currMaterial.d = (float)material.pbrMetallicRoughness.baseColorFactor[3];

        // currMaterial.illum = material.alphaMode == "OPAQUE" ? 2 : 1;
        // currMaterial.isLight = 0;

        const std::string& fileUri = "OmniPBR.mdl";
        const std::string& name = "OmniPBR";
        oka::Scene::MaterialDescription materialDesc;
        materialDesc.file = fileUri;
        materialDesc.name = name;
        materialDesc.type = oka::Scene::MaterialDescription::Type::eMdl;
        // materialDesc.color = glm::float3(material.pbrMetallicRoughness.baseColorFactor[0], material.pbrMetallicRoughness.baseColorFactor[1], material.pbrMetallicRoughness.baseColorFactor[2]);
        materialDesc.color = glm::float3(1.0f);
        materialDesc.hasColor = true;
        oka::MaterialManager::Param colorParam = {};
        colorParam.name = "diffuse_color_constant";
        colorParam.type = oka::MaterialManager::Param::Type::eFloat3;
        colorParam.value.resize(sizeof(float) * 3);
        memcpy(colorParam.value.data(), glm::value_ptr(materialDesc.color), sizeof(float) * 3);
        materialDesc.params.push_back(colorParam);

        auto texId = material.pbrMetallicRoughness.baseColorTexture.index;
        if (texId > 0)
        {
            auto imageId = model.textures[texId].source;
            auto textureUri = model.images[imageId].uri;

            oka::MaterialManager::Param diffuseTexture{};
            diffuseTexture.name = "diffuse_texture";
            diffuseTexture.type = oka::MaterialManager::Param::Type::eTexture;
            diffuseTexture.value.resize(textureUri.size());
            memcpy(diffuseTexture.value.data(), textureUri.data(), textureUri.size());
            materialDesc.params.push_back(diffuseTexture);
        }

        scene.addMaterial(materialDesc);

    }
}

void loadCameras(const tinygltf::Model& model, oka::Scene& scene)
{
    for (uint32_t i = 0; i < model.cameras.size(); ++i)
    {
        const tinygltf::Camera& cameraGltf = model.cameras[i];
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
    if (scene.getCameraCount() == 0)
    {
        // add default camera
        Camera camera;
        camera.updateViewMatrix();
        scene.addCamera(camera);
    }
}

void loadAnimation(const tinygltf::Model& model, oka::Scene& scene)
{
    std::vector<oka::Scene::Animation> animations;

    using namespace std;
    for (const tinygltf::Animation& animation : model.animations)
    {
        oka::Scene::Animation anim{};
        cout << "Animation name: " << animation.name << endl;
        for (const tinygltf::AnimationSampler& sampler : animation.samplers)
        {
            oka::Scene::AnimationSampler samp{};
            {
                const tinygltf::Accessor& accessor = model.accessors[sampler.input];
                const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                const void* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                const float* buf = static_cast<const float*>(dataPtr);

                for (size_t index = 0; index < accessor.count; index++)
                {
                    samp.inputs.push_back(buf[index]);
                }

                for (auto input : samp.inputs)
                {
                    if (input < anim.start)
                    {
                        anim.start = input;
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
                    const glm::vec3* buf = static_cast<const glm::vec3*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++)
                    {
                        samp.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
                    }
                    break;
                }
                case TINYGLTF_TYPE_VEC4: {
                    const glm::vec4* buf = static_cast<const glm::vec4*>(dataPtr);
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
                std::cout << "skipping channel" << std::endl;
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

oka::Scene::UniformLightDesc parseFromJson(json light, uint32_t j)
{
    oka::Scene::UniformLightDesc desc;

    desc.position = glm::float3(
        light["lights"][j]["position"][0], light["lights"][j]["position"][1], light["lights"][j]["position"][2]);
    desc.orientation = glm::float3(light["lights"][j]["orientation"][0], light["lights"][j]["orientation"][1],
                                   light["lights"][j]["orientation"][2]);
    desc.width = float(light["lights"][j]["width"]);
    desc.height = light["lights"][j]["height"];
    
    desc.color =
        glm::float3(light["lights"][j]["color"][0], light["lights"][j]["color"][1], light["lights"][j]["color"][2]);
    desc.intensity = float(light["lights"][j]["intensity"]);

    desc.useXform = 0;
    desc.type = 0;
    return desc;
}

void loadFromJson(const std::string& modelPath, oka::Scene& scene)
{
    std::string fileName = modelPath.substr(0, modelPath.rfind('.')); // w/o extension
    std::string jsonPath = fileName + "_light" + ".json";
    if (fs::exists(jsonPath))
    {
        std::ifstream i(jsonPath);
        json light;
        i >> light;

        for (uint32_t j = 0; j < light["lights"].size(); ++j)
        {
            Scene::UniformLightDesc desc = parseFromJson(light, j);
            scene.createLight(desc);
        }
    }
}


bool GltfLoader::loadGltf(const std::string& modelPath, oka::Scene& scene)
{
    if (modelPath.empty())
    {
        return false;
    }

    using namespace std;
    tinygltf::Model model;
    tinygltf::TinyGLTF gltf_ctx;
    std::string err;
    std::string warn;
    bool res = gltf_ctx.LoadASCIIFromFile(&model, &err, &warn, modelPath.c_str());
    if (!res)
    {
        cerr << "Unable to load file: " << modelPath << endl;
        return res;
    }
    for (int i = 0; i < model.scenes.size(); ++i)
    {
        cout << "Scene: " << model.scenes[i].name << endl;
    }

    int sceneId = model.defaultScene;

    // findTextureSamplers(model, scene);
    // loadTextures(model, scene, *mTexManager);
    loadMaterials(model, scene);
    loadFromJson(modelPath, scene);

    // oka::Scene::UniformLightDesc lightDesc {};
    
    // lightDesc.xform = glm::mat4(1.0f);
    // lightDesc.type = 3; // distant light
    // lightDesc.halfAngle = 10.0f * 0.5f * (M_PI / 180.0f);
    // lightDesc.intensity = 15000;
    // lightDesc.color = glm::float3(1.0);
    // scene.createLight(lightDesc);

    loadCameras(model, scene);

    const float globalScale = 1.0f;
    loadNodes(model, scene, globalScale);

    for (int i = 0; i < model.scenes[sceneId].nodes.size(); ++i)
    {
        const int rootNodeIdx = model.scenes[sceneId].nodes[i];
        processNode(model, scene, model.nodes[rootNodeIdx], rootNodeIdx, glm::float4x4(1.0f), globalScale);
    }

    loadAnimation(model, scene);

    return res;
}
} // namespace oka
