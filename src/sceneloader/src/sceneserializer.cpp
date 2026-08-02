#include <strelka/sceneloader/sceneserializer.h>
#include <strelka/sceneloader/light_json.h>
#include <strelka/scene/vertex_packing.h>
#include <strelka/scene/light_desc.h>

#include "tiny_gltf.h"
#include "nlohmann/json.hpp"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <log.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace oka
{
namespace
{

std::string lightJsonPathFromScenePath(const std::string& gltfOrJsonPath)
{
    if (gltfOrJsonPath.size() >= 11 &&
        gltfOrJsonPath.compare(gltfOrJsonPath.size() - 11, 11, "_light.json") == 0)
    {
        return gltfOrJsonPath;
    }
    const std::string stem = gltfOrJsonPath.substr(0, gltfOrJsonPath.rfind('.'));
    return stem + "_light.json";
}

} // namespace

bool saveLightsJson(const Scene& scene, const std::string& gltfOrJsonPath)
{
    const std::string jsonPath = lightJsonPathFromScenePath(gltfOrJsonPath);
    json root;
    root["lights"] = json::array();

    const auto& descs = scene.getLightsDesc();
    for (const Scene::UniformLightDesc& desc : descs)
        root["lights"].push_back(lightjson::toJson(desc));

    if (const auto& env = scene.getEnvLight(); env.has_value())
    {
        json envJ;
        envJ["texture"] = env->texturePath;
        envJ["intensity"] = env->intensity;
        envJ["color"] = { env->color.x, env->color.y, env->color.z };
        envJ["rotation"] = env->rotationY;
        root["environment"] = envJ;
    }

    std::ofstream out(jsonPath);
    if (!out)
    {
        STRELKA_ERROR("Failed to write light JSON: {}", jsonPath);
        return false;
    }
    out << root.dump(2);
    STRELKA_INFO("Wrote light file: {}", jsonPath);
    return true;
}

bool loadLightsJson(Scene& scene, const std::string& lightJsonPath)
{
    if (!fs::exists(lightJsonPath))
    {
        STRELKA_ERROR("Light JSON not found: {}", lightJsonPath);
        return false;
    }
    std::ifstream i(lightJsonPath);
    json root;
    i >> root;
    if (!root.contains("lights"))
        return false;

    const std::string searchDir = fs::path(lightJsonPath).parent_path().string();
    for (const auto& light : root["lights"])
    {
        Scene::UniformLightDesc desc = lightjson::parseDesc(light, searchDir);
        lightjson::resolveIes(scene, desc, searchDir);
        scene.createLight(desc);
    }

    if (root.contains("environment"))
    {
        const auto& env = root["environment"];
        Scene::EnvLightDesc envDesc{};
        if (env.contains("texture"))
            envDesc.texturePath = env["texture"].get<std::string>();
        if (env.contains("intensity"))
            envDesc.intensity = env["intensity"].get<float>();
        if (env.contains("color"))
        {
            const auto& c = env["color"];
            envDesc.color = glm::float3(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
        }
        if (env.contains("rotation"))
            envDesc.rotationY = env["rotation"].get<float>();
        scene.setEnvLight(envDesc);
    }
    return true;
}

bool saveGltf(const Scene& scene, const std::string& outputPath)
{
    tinygltf::Model model;
    model.asset.version = "2.0";
    model.asset.generator = "Strelka";

    tinygltf::Buffer buffer;
    auto appendBytes = [&](const void* data, size_t size) -> size_t {
        const size_t offset = buffer.data.size();
        const size_t aligned = (size + 3) & ~size_t(3);
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        buffer.data.insert(buffer.data.end(), bytes, bytes + size);
        while (buffer.data.size() < offset + aligned)
            buffer.data.push_back(0);
        return offset;
    };

    // Materials
    for (const auto& matDesc : scene.getMaterials())
    {
        tinygltf::Material mat;
        mat.name = matDesc.name;
        mat.pbrMetallicRoughness.baseColorFactor = {
            matDesc.params.base_color.x, matDesc.params.base_color.y, matDesc.params.base_color.z, 1.0
        };
        mat.pbrMetallicRoughness.metallicFactor = matDesc.params.metallic;
        mat.pbrMetallicRoughness.roughnessFactor = matDesc.params.roughness;
        mat.emissiveFactor = { matDesc.params.emission.x, matDesc.params.emission.y, matDesc.params.emission.z };
        model.materials.push_back(mat);
    }
    if (model.materials.empty())
    {
        tinygltf::Material mat;
        mat.name = "default";
        model.materials.push_back(mat);
    }

    // One glTF mesh per Scene::Mesh
    const auto& vertices = scene.getVertices();
    const auto& indices = scene.getIndices();
    const auto& meshes = scene.getMeshes();

    for (size_t meshId = 0; meshId < meshes.size(); ++meshId)
    {
        const Mesh& mesh = meshes[meshId];
        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> uvs;
        positions.reserve(mesh.mVertexCount * 3);
        normals.reserve(mesh.mVertexCount * 3);
        uvs.reserve(mesh.mVertexCount * 2);
        for (uint32_t i = 0; i < mesh.mVertexCount; ++i)
        {
            const Scene::Vertex& v = vertices[mesh.mVbOffset + i];
            positions.push_back(v.pos.x);
            positions.push_back(v.pos.y);
            positions.push_back(v.pos.z);
            const glm::float3 n = unpackNormal(v.normal);
            normals.push_back(n.x);
            normals.push_back(n.y);
            normals.push_back(n.z);
            const glm::float2 uv = unpackUV(v.uv);
            uvs.push_back(uv.x);
            uvs.push_back(uv.y);
        }

        std::vector<uint32_t> idx(indices.begin() + mesh.mIndex, indices.begin() + mesh.mIndex + mesh.mCount);

        const size_t posOffset = appendBytes(positions.data(), positions.size() * sizeof(float));
        const size_t nrmOffset = appendBytes(normals.data(), normals.size() * sizeof(float));
        const size_t uvOffset = appendBytes(uvs.data(), uvs.size() * sizeof(float));
        const size_t idxOffset = appendBytes(idx.data(), idx.size() * sizeof(uint32_t));

        auto makeView = [&](size_t offset, size_t byteLength, int target) {
            tinygltf::BufferView view;
            view.buffer = 0;
            view.byteOffset = offset;
            view.byteLength = byteLength;
            view.target = target;
            model.bufferViews.push_back(view);
            return int(model.bufferViews.size() - 1);
        };

        const int posView = makeView(posOffset, positions.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);
        const int nrmView = makeView(nrmOffset, normals.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);
        const int uvView = makeView(uvOffset, uvs.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);
        const int idxView = makeView(idxOffset, idx.size() * sizeof(uint32_t), TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER);

        tinygltf::Accessor posAcc;
        posAcc.bufferView = posView;
        posAcc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        posAcc.count = mesh.mVertexCount;
        posAcc.type = TINYGLTF_TYPE_VEC3;
        posAcc.maxValues = { 1, 1, 1 };
        posAcc.minValues = { -1, -1, -1 };
        // Compute bounds
        posAcc.minValues = { 1e30, 1e30, 1e30 };
        posAcc.maxValues = { -1e30, -1e30, -1e30 };
        for (uint32_t i = 0; i < mesh.mVertexCount; ++i)
        {
            const auto& p = vertices[mesh.mVbOffset + i].pos;
            posAcc.minValues[0] = std::min(posAcc.minValues[0], (double)p.x);
            posAcc.minValues[1] = std::min(posAcc.minValues[1], (double)p.y);
            posAcc.minValues[2] = std::min(posAcc.minValues[2], (double)p.z);
            posAcc.maxValues[0] = std::max(posAcc.maxValues[0], (double)p.x);
            posAcc.maxValues[1] = std::max(posAcc.maxValues[1], (double)p.y);
            posAcc.maxValues[2] = std::max(posAcc.maxValues[2], (double)p.z);
        }
        model.accessors.push_back(posAcc);
        const int posAccIdx = int(model.accessors.size() - 1);

        tinygltf::Accessor nrmAcc;
        nrmAcc.bufferView = nrmView;
        nrmAcc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        nrmAcc.count = mesh.mVertexCount;
        nrmAcc.type = TINYGLTF_TYPE_VEC3;
        model.accessors.push_back(nrmAcc);
        const int nrmAccIdx = int(model.accessors.size() - 1);

        tinygltf::Accessor uvAcc;
        uvAcc.bufferView = uvView;
        uvAcc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        uvAcc.count = mesh.mVertexCount;
        uvAcc.type = TINYGLTF_TYPE_VEC2;
        model.accessors.push_back(uvAcc);
        const int uvAccIdx = int(model.accessors.size() - 1);

        tinygltf::Accessor idxAcc;
        idxAcc.bufferView = idxView;
        idxAcc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
        idxAcc.count = mesh.mCount;
        idxAcc.type = TINYGLTF_TYPE_SCALAR;
        model.accessors.push_back(idxAcc);
        const int idxAccIdx = int(model.accessors.size() - 1);

        tinygltf::Primitive prim;
        prim.attributes["POSITION"] = posAccIdx;
        prim.attributes["NORMAL"] = nrmAccIdx;
        prim.attributes["TEXCOORD_0"] = uvAccIdx;
        prim.indices = idxAccIdx;
        prim.material = 0;
        prim.mode = TINYGLTF_MODE_TRIANGLES;

        // Prefer material from first instance referencing this mesh
        for (const auto& inst : scene.getInstances())
        {
            if (inst.type == Instance::Type::eMesh && inst.mMeshId == meshId)
            {
                if (inst.mMaterialId < model.materials.size())
                    prim.material = int(inst.mMaterialId);
                break;
            }
        }

        tinygltf::Mesh gltfMesh;
        gltfMesh.name = "mesh_" + std::to_string(meshId);
        gltfMesh.primitives.push_back(prim);
        model.meshes.push_back(gltfMesh);
    }

    model.buffers.push_back(std::move(buffer));

    // Nodes — skip light-only instances; export mesh nodes with local TRS
    const auto& nodes = scene.getNodes();
    model.nodes.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        const Scene::Node& n = nodes[i];
        tinygltf::Node& gn = model.nodes[i];
        gn.name = n.name;
        gn.translation = { n.translation.x, n.translation.y, n.translation.z };
        gn.scale = { n.scale.x, n.scale.y, n.scale.z };
        gn.rotation = { n.rotation.x, n.rotation.y, n.rotation.z, n.rotation.w };
        gn.children = n.children;
        if (n.type == Scene::Node::NodeType::mesh && !n.instanceIds.empty())
        {
            const uint32_t instId = n.instanceIds.front();
            if (instId < scene.getInstances().size())
            {
                const auto& inst = scene.getInstances()[instId];
                if (inst.type == Instance::Type::eMesh && inst.mMeshId < model.meshes.size())
                    gn.mesh = int(inst.mMeshId);
            }
        }
    }

    tinygltf::Scene gltfScene;
    gltfScene.name = "Scene";
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        if (nodes[i].parent == -1)
            gltfScene.nodes.push_back(int(i));
    }
    // Fallback: one node per mesh if hierarchy empty
    if (gltfScene.nodes.empty())
    {
        for (size_t meshId = 0; meshId < model.meshes.size(); ++meshId)
        {
            tinygltf::Node gn;
            gn.name = "mesh_inst_" + std::to_string(meshId);
            gn.mesh = int(meshId);
            // Find instance transform
            for (const auto& inst : scene.getInstances())
            {
                if (inst.type == Instance::Type::eMesh && inst.mMeshId == meshId)
                {
                    const glm::mat4& m = inst.transform;
                    gn.matrix = {
                        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3],
                        m[2][0], m[2][1], m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3]
                    };
                    break;
                }
            }
            model.nodes.push_back(gn);
            gltfScene.nodes.push_back(int(model.nodes.size() - 1));
        }
    }

    model.scenes.push_back(gltfScene);
    model.defaultScene = 0;

    tinygltf::TinyGLTF writer;
    const bool isBinary = fs::path(outputPath).extension() == ".glb";
    const bool ok = writer.WriteGltfSceneToFile(&model, outputPath, true, true, true, isBinary);
    if (!ok)
    {
        STRELKA_ERROR("Failed to write glTF: {}", outputPath);
        return false;
    }
    STRELKA_INFO("Wrote glTF: {}", outputPath);
    return true;
}

} // namespace oka
