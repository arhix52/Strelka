#include <doctest/doctest.h>

#include <strelka/scene/scene.h>
#include <strelka/sceneloader/gltfloader.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>

using namespace oka;

namespace
{

// One quad in the XY plane, every vertex fully weighted to a single joint, so the
// posed geometry is the rest quad moved by that joint's world transform. Node 0
// is the joint, node 1 owns the mesh instance and references the skin.
struct SkinnedQuad
{
    Scene scene;
    uint32_t jointNodeId = 0;
    uint32_t meshNodeId = 1;
    uint32_t instId = (uint32_t)-1;
    glm::float3 restMin{ -1.0f, -1.0f, 0.0f };
    glm::float3 restMax{ 1.0f, 1.0f, 0.0f };
};

std::unique_ptr<SkinnedQuad> makeSkinnedQuad(const glm::float3& jointTranslation)
{
    auto fixture = std::make_unique<SkinnedQuad>();
    Scene& scene = fixture->scene;

    Scene::MaterialDescription mat{};
    mat.name = "default";
    const uint32_t matId = scene.addMaterial(mat);

    const glm::float3 positions[4] = { { -1, -1, 0 }, { 1, -1, 0 }, { 1, 1, 0 }, { -1, 1, 0 } };
    std::vector<Scene::Vertex> vb(4);
    std::vector<Scene::vertexSkinData> sb(4);
    for (int i = 0; i < 4; ++i)
    {
        vb[i].pos = positions[i];
        sb[i].pos = positions[i];
        sb[i].normal = glm::float3(0, 0, 1);
        sb[i].joints = glm::ivec4(0);
        sb[i].weights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const std::vector<uint32_t> ib = { 0, 1, 2, 0, 2, 3 };
    const uint32_t meshId = scene.createSkeletalMesh(vb, ib, sb);
    fixture->instId = scene.createInstance(Instance::Type::eMesh, meshId, matId, glm::mat4(1.0f));

    Scene::Node joint{};
    joint.name = "joint";
    joint.type = Scene::Node::NodeType::skeleton;
    joint.translation = jointTranslation;
    joint.scale = glm::float3(1.0f);
    joint.rotation = glm::quat(1, 0, 0, 0);
    scene.mNodes.push_back(joint);

    Scene::Node meshNode{};
    meshNode.name = "skinnedMesh";
    meshNode.type = Scene::Node::NodeType::mesh;
    meshNode.translation = glm::float3(0.0f);
    meshNode.scale = glm::float3(1.0f);
    meshNode.rotation = glm::quat(1, 0, 0, 0);
    meshNode.skin = 0;
    meshNode.instanceIds.push_back(fixture->instId);
    scene.mNodes.push_back(meshNode);

    Scene::Skin skin{};
    skin.name = "skin";
    skin.joints = { (int)fixture->jointNodeId };
    skin.inverseBindMatrices = { glm::mat4(1.0f) };
    skin.refNodeId = (int)fixture->meshNodeId;
    scene.mSkines.push_back(skin);

    return fixture;
}

} // namespace

TEST_CASE("Skinned bounds follow the joint, not the rest pose")
{
    auto fixture = makeSkinnedQuad(glm::float3(5.0f, 0.0f, 0.0f));
    Scene& scene = fixture->scene;

    glm::float3 bbMin(0.0f);
    glm::float3 bbMax(0.0f);
    REQUIRE(scene.computeInstanceBounds(fixture->instId, bbMin, bbMax));

    CHECK(bbMin.x == doctest::Approx(4.0f).epsilon(1e-4));
    CHECK(bbMax.x == doctest::Approx(6.0f).epsilon(1e-4));
    CHECK(bbMin.y == doctest::Approx(-1.0f).epsilon(1e-4));
    CHECK(bbMax.y == doctest::Approx(1.0f).epsilon(1e-4));

    // The rest-pose box would have been centred on the origin.
    CHECK(bbMin.x > fixture->restMax.x);
}

TEST_CASE("Skinned bounds track joint animation")
{
    auto fixture = makeSkinnedQuad(glm::float3(0.0f));
    Scene& scene = fixture->scene;

    glm::float3 bbMin(0.0f);
    glm::float3 bbMax(0.0f);
    REQUIRE(scene.computeInstanceBounds(fixture->instId, bbMin, bbMax));
    CHECK(bbMin.x == doctest::Approx(-1.0f).epsilon(1e-4));

    scene.setNodeLocalTransform(fixture->jointNodeId, glm::float3(0.0f, 3.0f, 0.0f),
                                glm::quat(1, 0, 0, 0), glm::float3(1.0f));

    REQUIRE(scene.computeInstanceBounds(fixture->instId, bbMin, bbMax));
    CHECK(bbMin.y == doctest::Approx(2.0f).epsilon(1e-4));
    CHECK(bbMax.y == doctest::Approx(4.0f).epsilon(1e-4));
    CHECK(bbMin.x == doctest::Approx(-1.0f).epsilon(1e-4));
}

TEST_CASE("Skinned bounds include the joint scale and rotation")
{
    auto fixture = makeSkinnedQuad(glm::float3(0.0f));
    Scene& scene = fixture->scene;

    scene.setNodeLocalTransform(fixture->jointNodeId, glm::float3(0.0f), glm::quat(1, 0, 0, 0),
                                glm::float3(2.0f, 1.0f, 1.0f));

    glm::float3 bbMin(0.0f);
    glm::float3 bbMax(0.0f);
    REQUIRE(scene.computeInstanceBounds(fixture->instId, bbMin, bbMax));
    CHECK(bbMin.x == doctest::Approx(-2.0f).epsilon(1e-4));
    CHECK(bbMax.x == doctest::Approx(2.0f).epsilon(1e-4));
    CHECK(bbMax.y == doctest::Approx(1.0f).epsilon(1e-4));

    // 90 degrees about X turns the XY quad into an XZ one.
    const glm::quat rotX = glm::angleAxis(glm::radians(90.0f), glm::float3(1, 0, 0));
    scene.setNodeLocalTransform(fixture->jointNodeId, glm::float3(0.0f), rotX, glm::float3(1.0f));
    REQUIRE(scene.computeInstanceBounds(fixture->instId, bbMin, bbMax));
    CHECK(bbMax.y == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(bbMin.y == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(bbMax.z == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(bbMin.z == doctest::Approx(-1.0f).epsilon(1e-4));
}

TEST_CASE("Bounds of a skinned instance leave the rest-pose vertex buffer alone")
{
    auto fixture = makeSkinnedQuad(glm::float3(5.0f, 0.0f, 0.0f));
    Scene& scene = fixture->scene;

    const std::vector<Scene::Vertex> before = scene.getVertices();
    glm::float3 bbMin(0.0f);
    glm::float3 bbMax(0.0f);
    REQUIRE(scene.computeInstanceBounds(fixture->instId, bbMin, bbMax));
    scene.pick(glm::float3(5, 0, 5), glm::float3(0, 0, -1));

    // Skinning runs on the GPU from these rest positions; posing them CPU side
    // would corrupt every later frame.
    const std::vector<Scene::Vertex>& after = scene.getVertices();
    REQUIRE(after.size() == before.size());
    for (size_t i = 0; i < after.size(); ++i)
    {
        CHECK(after[i].pos.x == doctest::Approx(before[i].pos.x));
        CHECK(after[i].pos.y == doctest::Approx(before[i].pos.y));
        CHECK(after[i].pos.z == doctest::Approx(before[i].pos.z));
    }
}

TEST_CASE("Picking a skinned mesh follows the posed geometry")
{
    auto fixture = makeSkinnedQuad(glm::float3(5.0f, 0.0f, 0.0f));
    Scene& scene = fixture->scene;

    const Scene::PickHit posed = scene.pick(glm::float3(5, 0, 5), glm::float3(0, 0, -1));
    CHECK(posed.hit);
    CHECK(posed.instanceId == fixture->instId);
    CHECK(posed.nodeId == fixture->meshNodeId);

    // Where the quad sits only if skinning is ignored.
    const Scene::PickHit rest = scene.pick(glm::float3(0, 0, 5), glm::float3(0, 0, -1));
    CHECK_FALSE(rest.hit);
}

TEST_CASE("Skinning is composed with the instance transform")
{
    auto fixture = makeSkinnedQuad(glm::float3(5.0f, 0.0f, 0.0f));
    Scene& scene = fixture->scene;

    // The renderer applies the instance transform on top of the skinned
    // positions, so a CPU pick has to do the same.
    scene.updateInstanceTransform(fixture->instId, glm::translate(glm::mat4(1.0f), glm::float3(0, 10, 0)));

    CHECK(scene.pick(glm::float3(5, 10, 5), glm::float3(0, 0, -1)).hit);
    CHECK_FALSE(scene.pick(glm::float3(5, 0, 5), glm::float3(0, 0, -1)).hit);
}

// Same quad, but the joint hangs off a parent and carries a real inverse bind
// matrix, which is what an exported character looks like. Both are places where a
// wrong convention or a stale world-transform cache shows up as a box that drifts
// away from the geometry.
namespace
{

struct BoundSkin
{
    Scene scene;
    uint32_t rootNodeId = 0;
    uint32_t jointNodeId = 1;
    uint32_t meshNodeId = 2;
    uint32_t instId = (uint32_t)-1;
};

std::unique_ptr<BoundSkin> makeBoundSkin()
{
    auto fixture = std::make_unique<BoundSkin>();
    Scene& scene = fixture->scene;

    const Scene::MaterialDescription mat{};
    const uint32_t matId = scene.addMaterial(mat);

    const glm::float3 positions[4] = { { -1, -1, 0 }, { 1, -1, 0 }, { 1, 1, 0 }, { -1, 1, 0 } };
    std::vector<Scene::Vertex> vb(4);
    std::vector<Scene::vertexSkinData> sb(4);
    for (int i = 0; i < 4; ++i)
    {
        vb[i].pos = positions[i];
        sb[i].pos = positions[i];
        sb[i].normal = glm::float3(0, 0, 1);
        sb[i].joints = glm::ivec4(0);
        sb[i].weights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const std::vector<uint32_t> ib = { 0, 1, 2, 0, 2, 3 };
    const uint32_t meshId = scene.createSkeletalMesh(vb, ib, sb);
    fixture->instId = scene.createInstance(Instance::Type::eMesh, meshId, matId, glm::mat4(1.0f));

    auto addNode = [&](int parent, const glm::float3& t, Scene::Node::NodeType type) -> uint32_t {
        Scene::Node n{};
        n.parent = parent;
        n.translation = t;
        n.scale = glm::float3(1.0f);
        n.rotation = glm::quat(1, 0, 0, 0);
        n.type = type;
        const uint32_t id = (uint32_t)scene.mNodes.size();
        scene.mNodes.push_back(n);
        if (parent >= 0)
        {
            scene.mNodes[parent].children.push_back((int)id);
        }
        return id;
    };

    fixture->rootNodeId = addNode(-1, glm::float3(1, 0, 0), Scene::Node::NodeType::sceneGraph);
    fixture->jointNodeId =
        addNode((int)fixture->rootNodeId, glm::float3(0, 2, 0), Scene::Node::NodeType::skeleton);
    fixture->meshNodeId = addNode(-1, glm::float3(0.0f), Scene::Node::NodeType::mesh);
    scene.mNodes[fixture->meshNodeId].skin = 0;
    scene.mNodes[fixture->meshNodeId].instanceIds.push_back(fixture->instId);

    Scene::Skin skin{};
    skin.joints = { (int)fixture->jointNodeId };
    // Bind pose of the joint is its current world transform, so the palette must
    // come out as identity until something moves.
    skin.inverseBindMatrices = { glm::inverse(glm::translate(glm::mat4(1.0f), glm::float3(1, 2, 0))) };
    skin.refNodeId = (int)fixture->meshNodeId;
    scene.mSkines.push_back(skin);

    return fixture;
}

} // namespace

TEST_CASE("At bind pose the skinned bounds equal the rest bounds")
{
    auto fixture = makeBoundSkin();
    glm::float3 bbMin(0.0f);
    glm::float3 bbMax(0.0f);
    REQUIRE(fixture->scene.computeInstanceBounds(fixture->instId, bbMin, bbMax));
    CHECK(bbMin.x == doctest::Approx(-1.0f).epsilon(1e-4));
    CHECK(bbMax.x == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(bbMin.y == doctest::Approx(-1.0f).epsilon(1e-4));
    CHECK(bbMax.y == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(bbMin.z == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(bbMax.z == doctest::Approx(0.0f).epsilon(1e-4));
}

TEST_CASE("Moving an ancestor of the joint moves the skinned bounds")
{
    auto fixture = makeBoundSkin();
    Scene& scene = fixture->scene;

    scene.setNodeLocalTransform(fixture->rootNodeId, glm::float3(1, 0, 5), glm::quat(1, 0, 0, 0),
                                glm::float3(1.0f));

    glm::float3 bbMin(0.0f);
    glm::float3 bbMax(0.0f);
    REQUIRE(scene.computeInstanceBounds(fixture->instId, bbMin, bbMax));
    CHECK(bbMin.z == doctest::Approx(5.0f).epsilon(1e-4));
    CHECK(bbMax.z == doctest::Approx(5.0f).epsilon(1e-4));
    CHECK(bbMin.x == doctest::Approx(-1.0f).epsilon(1e-4));
    CHECK(bbMax.y == doctest::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("Rotating the joint swings the bounds around the bind position")
{
    auto fixture = makeBoundSkin();
    Scene& scene = fixture->scene;

    // 90 degrees about Z, so a rest corner p maps to bind + Rz(p - bind) with
    // bind = (1, 2, 0).
    const glm::quat rotZ = glm::angleAxis(glm::radians(90.0f), glm::float3(0, 0, 1));
    scene.setNodeLocalTransform(fixture->jointNodeId, glm::float3(0, 2, 0), rotZ, glm::float3(1.0f));

    glm::float3 bbMin(0.0f);
    glm::float3 bbMax(0.0f);
    REQUIRE(scene.computeInstanceBounds(fixture->instId, bbMin, bbMax));
    CHECK(bbMin.x == doctest::Approx(2.0f).epsilon(1e-4));
    CHECK(bbMax.x == doctest::Approx(4.0f).epsilon(1e-4));
    CHECK(bbMin.y == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(bbMax.y == doctest::Approx(2.0f).epsilon(1e-4));
}

// Synthetic fixtures pin down the maths; this one checks the same code against a
// real exported character. The validation dataset ships BrainStem for that; an
// explicit STRELKA_SKINNED_GLTF still wins when set.
TEST_CASE("Skinned bounds on a real asset stay glued to the animated pose")
{
    std::string assetPath;
    // Single-threaded test, and nothing writes the environment while it reads.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (const char* env = std::getenv("STRELKA_SKINNED_GLTF"))
    {
        assetPath = env;
    }
    else
    {
        assetPath = std::string(STRELKA_TEST_ASSETS_DIR) + "/brainstem/BrainStem.glb";
    }
    if (!std::filesystem::exists(assetPath))
    {
        FAIL("skinned regression asset missing: " << assetPath);
    }

    Scene scene;
    oka::GltfLoader loader;
    REQUIRE(loader.loadGltf(assetPath, scene));

    int skinnedNodeId = -1;
    for (uint32_t n = 0; n < scene.getNodes().size(); ++n)
    {
        if (scene.getNodes()[n].skin >= 0 && !scene.getNodes()[n].instanceIds.empty())
        {
            skinnedNodeId = (int)n;
            break;
        }
    }
    REQUIRE_MESSAGE(skinnedNodeId >= 0, "asset has no skinned mesh instance");
    const uint32_t instId = scene.getNodes()[skinnedNodeId].instanceIds[0];

    glm::float3 restMin(0.0f);
    glm::float3 restMax(0.0f);
    REQUIRE(scene.computeInstanceBounds(instId, restMin, restMax));

    // An asset usually splits its animations per subject (BrainStem animates the
    // camera in one and the skeleton in another), so all of them have to run to
    // reach the pose the renderer draws.
    REQUIRE_FALSE(scene.getAnimations().empty());
    for (uint32_t a = 0; a < scene.mAnimations.size(); ++a)
    {
        Scene::Animation& anim = scene.mAnimations[a];
        anim.current = 0.5f * (anim.start + anim.end);
        scene.applyAnimation(a);
    }

    glm::float3 posedMin(0.0f);
    glm::float3 posedMax(0.0f);
    REQUIRE(scene.computeInstanceBounds(instId, posedMin, posedMax));

    // Bounds must be finite and enclose something.
    CHECK(std::isfinite(posedMin.x));
    CHECK(std::isfinite(posedMax.x));
    CHECK(posedMax.x > posedMin.x);
    CHECK(posedMax.y > posedMin.y);

    // Every posed vertex has to be inside the reported box, and the box has to be
    // tight: it is the extent of those very vertices.
    const std::vector<glm::mat4> palette = scene.buildJointPalette(instId);
    REQUIRE_FALSE(palette.empty());
    const Mesh& mesh = scene.getMeshes()[scene.getInstances()[instId].mMeshId];
    glm::float3 tightMin(std::numeric_limits<float>::max());
    glm::float3 tightMax(std::numeric_limits<float>::lowest());
    bool posedDiffersFromRest = false;
    for (uint32_t i = 0; i < mesh.mVertexCount; ++i)
    {
        const glm::float3 p = scene.posedVertexPosition(mesh, i, palette);
        CHECK(p.x >= posedMin.x - 1e-3f);
        CHECK(p.x <= posedMax.x + 1e-3f);
        tightMin = glm::min(tightMin, p);
        tightMax = glm::max(tightMax, p);
        posedDiffersFromRest |= glm::length(p - scene.getVertices()[mesh.mVbOffset + i].pos) > 1e-4f;
    }
    CHECK(tightMin.x == doctest::Approx(posedMin.x));
    CHECK(tightMax.z == doctest::Approx(posedMax.z));
    // If this fails the overlay is drawing the rest pose, which is the bug this
    // whole path exists to avoid.
    CHECK(posedDiffersFromRest);

    // The palette must agree with the world transforms derived from scratch, so a
    // stale cache cannot silently park the box next to the character.
    const uint32_t skinId = (uint32_t)scene.getNodes()[skinnedNodeId].skin;
    const Scene::Skin& skin = scene.mSkines[skinId];
    for (size_t j = 0; j < skin.joints.size(); ++j)
    {
        const glm::mat4 reference =
            scene.calculateNodeGlobalTransform((uint32_t)skin.joints[j]) * skin.inverseBindMatrices[j];
        for (int c = 0; c < 4; ++c)
        {
            for (int r = 0; r < 4; ++r)
            {
                CHECK(palette[j][c][r] == doctest::Approx(reference[c][r]).epsilon(1e-4));
            }
        }
    }
}

TEST_CASE("Rigid mesh bounds are the plain vertex extent")
{
    Scene scene;
    const Scene::MaterialDescription mat{};
    const uint32_t matId = scene.addMaterial(mat);

    std::vector<Scene::Vertex> vb(3);
    vb[0].pos = glm::float3(-1, -1, 0);
    vb[1].pos = glm::float3(3, -1, 0);
    vb[2].pos = glm::float3(0, 2, 4);
    const std::vector<uint32_t> ib = { 0, 1, 2 };
    const uint32_t meshId = scene.createMesh(vb, ib);
    // A non-identity instance transform must not leak into the bounds: they are
    // expressed in the space that transform maps to world.
    const uint32_t instId = scene.createInstance(Instance::Type::eMesh, meshId, matId,
                                                 glm::translate(glm::mat4(1.0f), glm::float3(100, 0, 0)));

    glm::float3 bbMin(0.0f);
    glm::float3 bbMax(0.0f);
    REQUIRE(scene.computeInstanceBounds(instId, bbMin, bbMax));
    CHECK(bbMin.x == doctest::Approx(-1.0f));
    CHECK(bbMax.x == doctest::Approx(3.0f));
    CHECK(bbMin.y == doctest::Approx(-1.0f));
    CHECK(bbMax.y == doctest::Approx(2.0f));
    CHECK(bbMin.z == doctest::Approx(0.0f));
    CHECK(bbMax.z == doctest::Approx(4.0f));
}
