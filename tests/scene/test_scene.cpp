#include <doctest/doctest.h>

#include <strelka/scene/scene.h>
#include <light_types.h>

#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

using namespace oka;

static uint32_t addUnitTriangle(Scene& scene, const glm::mat4& xform, uint32_t matId = 0)
{
    if (scene.getMaterials().empty())
    {
        Scene::MaterialDescription mat{};
        mat.name = "default";
        mat.params.base_color = glm::float3(0.8f);
        scene.addMaterial(mat);
        matId = 0;
    }

    std::vector<Scene::Vertex> vb(3);
    vb[0].pos = glm::float3(-1, -1, 0);
    vb[1].pos = glm::float3(1, -1, 0);
    vb[2].pos = glm::float3(0, 1, 0);
    std::vector<uint32_t> ib = { 0, 1, 2 };
    const uint32_t meshId = scene.createMesh(vb, ib);
    return scene.createInstance(Instance::Type::eMesh, meshId, matId, xform);
}

TEST_CASE("ChangeBits lifecycle: setLight marks Lights and consume clears")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_DISTANT;
    desc.intensity = 10.0f;
    desc.color = glm::float3(1.0f);
    desc.orientation = glm::float3(-45.0f, 0.0f, 0.0f);
    desc.halfAngle = 0.01f;
    const uint32_t id = scene.createLight(desc);
    // createLight dirties; clear to isolate setLight
    scene.consumeChanges();
    CHECK(scene.peekChanges() == ChangeBits::None);

    desc.intensity = 42.0f;
    scene.setLight(id, desc);
    CHECK(any(scene.peekChanges() & ChangeBits::Lights));
    const ChangeBits bits = scene.consumeChanges();
    CHECK(any(bits & ChangeBits::Lights));
    CHECK(scene.peekChanges() == ChangeBits::None);
    CHECK(scene.consumeChanges() == ChangeBits::None);
    CHECK(scene.getLightsDesc()[id].intensity == doctest::Approx(42.0f));
    CHECK(scene.getLights()[id].color.x == doctest::Approx(42.0f));
}

TEST_CASE("setLight keeps desc and GPU light in sync for rect")
{
    Scene scene;
    Scene::UniformLightDesc desc{};
    desc.type = LIGHT_TYPE_RECT;
    desc.position = glm::float3(1, 2, 3);
    desc.orientation = glm::float3(0, 0, 0);
    desc.width = 2.0f;
    desc.height = 1.0f;
    desc.color = glm::float3(0.5f, 0.25f, 0.1f);
    desc.intensity = 8.0f;
    const uint32_t id = scene.createLight(desc);

    desc.position = glm::float3(4, 5, 6);
    desc.intensity = 16.0f;
    scene.setLight(id, desc);

    CHECK(scene.getLightsDesc()[id].position.x == doctest::Approx(4.0f));
    CHECK(scene.getLightsDesc()[id].intensity == doctest::Approx(16.0f));
    CHECK(scene.getLights()[id].type == LIGHT_TYPE_RECT);
    CHECK(scene.getLights()[id].color.x == doctest::Approx(0.5f * 16.0f));

    const uint32_t instId = scene.getLightInstanceId(id);
    REQUIRE(instId != (uint32_t)-1);
    const glm::float3 t = glm::float3(scene.getInstances()[instId].transform[3]);
    CHECK(t.x == doctest::Approx(4.0f));
    CHECK(t.y == doctest::Approx(5.0f));
    CHECK(t.z == doctest::Approx(6.0f));
}

TEST_CASE("createLight distant vs rect bake differently")
{
    Scene scene;
    Scene::UniformLightDesc distant{};
    distant.type = LIGHT_TYPE_DISTANT;
    distant.halfAngle = 0.1f;
    distant.intensity = 1.0f;
    distant.color = glm::float3(1.0f);
    distant.orientation = glm::float3(-90.0f, 0.0f, 0.0f);
    const uint32_t dId = scene.createLight(distant);

    Scene::UniformLightDesc rect{};
    rect.type = LIGHT_TYPE_RECT;
    rect.width = 1.0f;
    rect.height = 1.0f;
    rect.intensity = 1.0f;
    rect.color = glm::float3(1.0f);
    rect.position = glm::float3(0.0f);
    const uint32_t rId = scene.createLight(rect);

    CHECK(scene.getLights()[dId].type == LIGHT_TYPE_DISTANT);
    CHECK(scene.getLights()[rId].type == LIGHT_TYPE_RECT);
    CHECK(scene.getLights()[dId].halfAngle == doctest::Approx(0.1f));
}

TEST_CASE("Parent translate updates child instance world transform")
{
    Scene scene;
    Scene::MaterialDescription mat{};
    scene.addMaterial(mat);

    Scene::Node parent{};
    parent.name = "parent";
    parent.translation = glm::float3(0.0f);
    parent.scale = glm::float3(1.0f);
    parent.rotation = glm::quat(1, 0, 0, 0);
    parent.type = Scene::Node::NodeType::sceneGraph;
    scene.mNodes.push_back(parent);
    const uint32_t parentId = 0;

    Scene::Node child{};
    child.name = "child";
    child.parent = (int)parentId;
    child.translation = glm::float3(0, 0, 2);
    child.scale = glm::float3(1.0f);
    child.rotation = glm::quat(1, 0, 0, 0);
    child.type = Scene::Node::NodeType::mesh;
    scene.mNodes.push_back(child);
    const uint32_t childId = 1;
    scene.mNodes[parentId].children.push_back((int)childId);

    const uint32_t instId = addUnitTriangle(scene, glm::translate(glm::mat4(1.0f), glm::float3(0, 0, 2)));
    scene.mNodes[childId].instanceIds.push_back(instId);

    scene.setNodeLocalTransform(parentId, glm::float3(10, 0, 0), glm::quat(1, 0, 0, 0), glm::float3(1.0f));

    const glm::mat4& world = scene.getGlobalTransforms()[childId];
    const glm::float3 childWorldPos = glm::float3(world[3]);
    CHECK(childWorldPos.x == doctest::Approx(10.0f).epsilon(1e-4));
    CHECK(childWorldPos.z == doctest::Approx(2.0f).epsilon(1e-4));

    const glm::float3 instPos = glm::float3(scene.getInstances()[instId].transform[3]);
    CHECK(instPos.x == doctest::Approx(childWorldPos.x).epsilon(1e-4));
    CHECK(instPos.z == doctest::Approx(childWorldPos.z).epsilon(1e-4));
    CHECK(any(scene.peekChanges() & ChangeBits::Transforms));
}

TEST_CASE("Deep hierarchy leaf world is product of locals")
{
    Scene scene;
    Scene::MaterialDescription mat{};
    scene.addMaterial(mat);

    auto addNode = [&](int parent, const glm::float3& t) -> uint32_t {
        Scene::Node n{};
        n.parent = parent;
        n.translation = t;
        n.scale = glm::float3(1.0f);
        n.rotation = glm::quat(1, 0, 0, 0);
        n.type = Scene::Node::NodeType::sceneGraph;
        const uint32_t id = (uint32_t)scene.mNodes.size();
        scene.mNodes.push_back(n);
        if (parent >= 0)
            scene.mNodes[parent].children.push_back((int)id);
        return id;
    };

    const uint32_t n0 = addNode(-1, glm::float3(1, 0, 0));
    const uint32_t n1 = addNode((int)n0, glm::float3(0, 2, 0));
    const uint32_t n2 = addNode((int)n1, glm::float3(0, 0, 3));
    scene.mNodes[n2].type = Scene::Node::NodeType::mesh;

    const uint32_t instId = addUnitTriangle(scene, glm::mat4(1.0f));
    scene.mNodes[n2].instanceIds.push_back(instId);

    scene.setNodeLocalTransform(n0, glm::float3(1, 0, 0), glm::quat(1, 0, 0, 0), glm::float3(1.0f));

    const glm::float3 leaf = glm::float3(scene.getGlobalTransforms()[n2][3]);
    CHECK(leaf.x == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(leaf.y == doctest::Approx(2.0f).epsilon(1e-4));
    CHECK(leaf.z == doctest::Approx(3.0f).epsilon(1e-4));
}

TEST_CASE("setMaterial updates params and marks Materials")
{
    Scene scene;
    Scene::MaterialDescription mat{};
    mat.params.roughness = 0.5f;
    const uint32_t id = scene.addMaterial(mat);
    scene.consumeChanges();

    mat.params.roughness = 0.2f;
    mat.params.metallic = 1.0f;
    scene.setMaterial(id, mat);
    CHECK(scene.getMaterials()[id].params.roughness == doctest::Approx(0.2f));
    CHECK(scene.getMaterials()[id].params.metallic == doctest::Approx(1.0f));
    CHECK(any(scene.peekChanges() & ChangeBits::Materials));
}

TEST_CASE("CPU pick hits unit triangle and misses offset ray")
{
    Scene scene;
    const uint32_t instId = addUnitTriangle(scene, glm::mat4(1.0f));

    // Attach a node for selection mapping
    Scene::Node node{};
    node.type = Scene::Node::NodeType::mesh;
    node.instanceIds.push_back(instId);
    node.translation = glm::float3(0);
    node.scale = glm::float3(1);
    node.rotation = glm::quat(1, 0, 0, 0);
    scene.mNodes.push_back(node);

    const Scene::PickHit hit = scene.pick(glm::float3(0, 0, 5), glm::float3(0, 0, -1));
    CHECK(hit.hit);
    CHECK(hit.instanceId == instId);
    CHECK(hit.nodeId == 0);

    const Scene::PickHit miss = scene.pick(glm::float3(10, 10, 5), glm::float3(0, 0, -1));
    CHECK_FALSE(miss.hit);
}

TEST_CASE("CPU pick prefers closer of two overlapping instances")
{
    Scene scene;
    const uint32_t nearId = addUnitTriangle(scene, glm::translate(glm::mat4(1.0f), glm::float3(0, 0, 1)));
    const uint32_t farId = addUnitTriangle(scene, glm::translate(glm::mat4(1.0f), glm::float3(0, 0, -1)));

    const Scene::PickHit hit = scene.pick(glm::float3(0, 0, 5), glm::float3(0, 0, -1));
    REQUIRE(hit.hit);
    CHECK(hit.instanceId == nearId);
    CHECK(hit.instanceId != farId);
}

TEST_CASE("createMeshFromOffsets records appended geometry without copying")
{
    Scene scene;
    scene.reserveGeometry(6, 6);
    auto& vertices = scene.getVertices();
    auto& indices = scene.getIndices();
    const uint32_t vbOffset = static_cast<uint32_t>(vertices.size());
    Scene::Vertex v{};
    v.pos = glm::float3(0, 0, 0);
    vertices.push_back(v);
    v.pos = glm::float3(1, 0, 0);
    vertices.push_back(v);
    v.pos = glm::float3(0, 1, 0);
    vertices.push_back(v);
    const uint32_t ibOffset = static_cast<uint32_t>(indices.size());
    indices.push_back(0);
    indices.push_back(1);
    indices.push_back(2);

    const uint32_t meshId = scene.createMeshFromOffsets(vbOffset, 3, ibOffset, 3);
    REQUIRE(meshId == 0);
    CHECK(scene.getMeshes()[meshId].mVbOffset == 0);
    CHECK(scene.getMeshes()[meshId].mVertexCount == 3);
    CHECK(scene.getMeshes()[meshId].mIndex == 0);
    CHECK(scene.getMeshes()[meshId].mCount == 3);
    CHECK(scene.getVertices().size() == 3);
    CHECK(scene.getVertices().capacity() >= 6);
}

TEST_CASE("takeHostGeometry moves arrays and blocks picking")
{
    Scene scene;
    addUnitTriangle(scene, glm::mat4(1.0f));
    REQUIRE_FALSE(scene.getVertices().empty());

    std::vector<Scene::Vertex> vertices;
    std::vector<uint32_t> indices;
    scene.takeHostGeometry(vertices, indices);
    CHECK(scene.hostGeometryReleased());
    CHECK(scene.getVertices().empty());
    CHECK(scene.getIndices().empty());
    CHECK(vertices.size() == 3);
    CHECK(indices.size() == 3);

    const Scene::PickHit hit = scene.pick(glm::float3(0, 0, 5), glm::float3(0, 0, -1));
    CHECK_FALSE(hit.hit);
}
