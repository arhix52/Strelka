#include "gltfloader.h"

#include "camera.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <iostream>
#include <log/log.h>

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
    auto packed = (uint32_t)((normal.x + 1.0f) / 2.0f * 511.99999f);
    packed += (uint32_t)((normal.y + 1.0f) / 2.0f * 511.99999f) << 10;
    packed += (uint32_t)((normal.z + 1.0f) / 2.0f * 511.99999f) << 20;
    return packed;
}

//  valid range of coordinates [-10; 10]
uint32_t packTangent(const glm::float3& tangent)
{
    auto packed = (uint32_t)((tangent.x + 10.0f) / 20.0f * 511.99999f);
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
            computeTangent(vertices, indices);
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
            const auto* buf = static_cast<const uint16_t*>(dataPtr);
            for (size_t index = 0; index < indexCount; index++)
            {
                indices.push_back(buf[index]);
            }
            computeTangent(vertices, indices);
            break;
        }
        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
            const auto* buf = static_cast<const uint8_t*>(dataPtr);
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

    uint32_t meshId = -1;
    if (hasJoints)
        meshId = scene.createMesh(vertices, indices, sb);
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

    for (int childIdx : node.children)
    {
        if (scene.mNodes[currentNodeId].type == oka::Scene::Node::NodeType::unknown)
            scene.mNodes[currentNodeId].type = oka::Scene::Node::NodeType::sceneGraph;
        scene.mNodes[childIdx].parent = currentNodeId;
        processNode(model, scene, model.nodes[childIdx], childIdx, globalTransform, globalScale);
    }
}

oka::Scene::MaterialDescription convertToOmniPBR(const tinygltf::Model& model, const tinygltf::Material& material)
{
    const std::string& fileUri = "OmniPBR.mdl";
    const std::string& name = "OmniPBR";
    oka::Scene::MaterialDescription materialDesc{};
    materialDesc.file = fileUri;
    materialDesc.name = name;
    materialDesc.type = oka::Scene::MaterialDescription::Type::eMdl;
    materialDesc.color =
        glm::float3(material.pbrMetallicRoughness.baseColorFactor[0], material.pbrMetallicRoughness.baseColorFactor[1],
                    material.pbrMetallicRoughness.baseColorFactor[2]);
    materialDesc.hasColor = true;
    oka::MaterialManager::Param colorParam = {};
    colorParam.name = "diffuse_color_constant";
    colorParam.type = oka::MaterialManager::Param::Type::eFloat3;
    colorParam.value.resize(sizeof(float) * 3);
    memcpy(colorParam.value.data(), glm::value_ptr(materialDesc.color), sizeof(float) * 3);
    materialDesc.params.push_back(colorParam);

    auto addFloat = [&](float value, const char* materialParamName) {
        oka::MaterialManager::Param param{};
        param.name = materialParamName;
        param.type = oka::MaterialManager::Param::Type::eFloat;
        param.value.resize(sizeof(float));
        *((float*)param.value.data()) = value;
        materialDesc.params.push_back(param);
    };

    addFloat((float)material.pbrMetallicRoughness.roughnessFactor, "reflection_roughness_constant");
    addFloat((float)material.pbrMetallicRoughness.metallicFactor, "metallic_constant");

    auto addTexture = [&](int texId, const char* materialParamName) {
        const auto imageId = model.textures[texId].source;
        const auto textureUri = model.images[imageId].uri;
        oka::MaterialManager::Param paramTexture{};
        paramTexture.name = materialParamName;
        paramTexture.type = oka::MaterialManager::Param::Type::eTexture;
        paramTexture.value.resize(textureUri.size());
        memcpy(paramTexture.value.data(), textureUri.data(), textureUri.size());
        materialDesc.params.push_back(paramTexture);
    };
    auto texId = material.pbrMetallicRoughness.baseColorTexture.index;
    if (texId >= 0)
    {
        addTexture(texId, "diffuse_texture");
    }
    auto normalTexId = material.normalTexture.index;
    if (normalTexId >= 0)
    {
        addTexture(normalTexId, "normalmap_texture");
    }
    return materialDesc;
}

oka::Scene::MaterialDescription convertToOmniGlass(const tinygltf::Model& model, const tinygltf::Material& material)
{
    oka::Scene::MaterialDescription materialDesc{};

    materialDesc.file = "OmniGlass.mdl";
    materialDesc.name = "OmniGlass";
    materialDesc.type = oka::Scene::MaterialDescription::Type::eMdl;
    oka::MaterialManager::Param param{};
    param.name = "enable_opacity";
    param.type = oka::MaterialManager::Param::Type::eBool;
    param.value.resize(sizeof(bool));
    *((bool*)param.value.data()) = true;
    materialDesc.params.push_back(param);
    
    // materialDesc.color =
    //     glm::float3(material.pbrMetallicRoughness.baseColorFactor[0], material.pbrMetallicRoughness.baseColorFactor[1],
    //                 material.pbrMetallicRoughness.baseColorFactor[2]);

    // oka::MaterialManager::Param colorParam = {};
    // colorParam.name = "glass_color";
    // colorParam.type = oka::MaterialManager::Param::Type::eFloat3;
    // colorParam.value.resize(sizeof(float) * 3);
    // memcpy(colorParam.value.data(), glm::value_ptr(materialDesc.color), sizeof(float) * 3);
    // materialDesc.params.push_back(colorParam);
    
    auto addBool = [&](bool value, const char* materialParamName) {
        oka::MaterialManager::Param param{};
        param.name = materialParamName;
        param.type = oka::MaterialManager::Param::Type::eBool;
        param.value.resize(sizeof(float));
        *((bool*)param.value.data()) = value;
        materialDesc.params.push_back(param);
    };

    addBool(false, "thin_walled");

    auto addFloat = [&](float value, const char* materialParamName) {
        oka::MaterialManager::Param param{};
        param.name = materialParamName;
        param.type = oka::MaterialManager::Param::Type::eFloat;
        param.value.resize(sizeof(float));
        *((float*)param.value.data()) = value;
        materialDesc.params.push_back(param);
    };
    
    addFloat((float)material.pbrMetallicRoughness.roughnessFactor, "frosting_roughness");
    
    return materialDesc;
}

void loadMaterials(const tinygltf::Model& model, oka::Scene& scene)
{
    for (const tinygltf::Material& material : model.materials)
    {
        if (material.alphaMode == "OPAQUE") 
        {
            scene.addMaterial(convertToOmniPBR(model, material));
        }
        else 
        {
            scene.addMaterial(convertToOmniGlass(model, material));
        }
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

void loadSkines(const tinygltf::Model& model, oka::Scene& scene, const float globalScale = 1.0f)
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


oka::Scene::UniformLightDesc parseFromJson(const json& light)
{
    oka::Scene::UniformLightDesc desc{};

    const auto position = light["position"];
    desc.position = glm::float3(position[0], position[1], position[2]);
    const auto orientation = light["orientation"];
    desc.orientation = glm::float3(orientation[0], orientation[1], orientation[2]);
    desc.width = float(light["width"]);
    desc.height = light["height"];
    const auto color = light["color"];
    desc.color = glm::float3(color[0], color[1], color[2]);
    desc.intensity = float(light["intensity"]);

    desc.useXform = false;
    desc.type = 0;
    return desc;
}

bool loadLightsFromJson(const std::string& modelPath, oka::Scene& scene)
{
    std::string fileName = modelPath.substr(0, modelPath.rfind('.')); // w/o extension
    std::string jsonPath = fileName + "_light" + ".json";
    if (fs::exists(jsonPath))
    {
        STRELKA_INFO("Found light file, loading lights from it");
        std::ifstream i(jsonPath);
        json light;
        i >> light;

        for (const auto& light : light["lights"])
        {
            Scene::UniformLightDesc desc = parseFromJson(light);
            scene.createLight(desc);
        }
        return true;
    }
    return false;
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
        STRELKA_ERROR("Unable to load file: {}", modelPath);
        return res;
    }

    int sceneId = model.defaultScene;

    loadMaterials(model, scene);
    if (loadLightsFromJson(modelPath, scene) == false)
    {
        STRELKA_WARNING("No light in scene, adding default distant light");
        oka::Scene::UniformLightDesc lightDesc {};
        // lightDesc.xform = glm::mat4(1.0f);
        // lightDesc.useXform = true;
        lightDesc.useXform = false;
        lightDesc.position = glm::float3(0.0f, 0.0f, 0.0f);
        lightDesc.orientation = glm::float3(-45.0f, 15.0f, 0.0f);
        lightDesc.type = 3; // distant light
        lightDesc.halfAngle = 10.0f * 0.5f * (M_PI / 180.0f);
        lightDesc.intensity = 100000;
        lightDesc.color = glm::float3(1.0);
        scene.createLight(lightDesc);
    }

    loadCameras(model, scene);

    const float globalScale = 1.0f;
    loadNodes(model, scene, globalScale);

    loadSkines(model, scene, globalScale);

    for (int i = 0; i < model.scenes[sceneId].nodes.size(); ++i)
    {
        const int rootNodeIdx = model.scenes[sceneId].nodes[i];
        processNode(model, scene, model.nodes[rootNodeIdx], rootNodeIdx, glm::float4x4(1.0f), globalScale);
    }

    loadAnimation(model, scene);

    return res;
}
} // namespace oka
