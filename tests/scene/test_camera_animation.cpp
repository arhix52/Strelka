#include <doctest/doctest.h>

#include <strelka/scene/scene.h>

#include <memory>

using namespace oka;

namespace
{

// A glTF camera as exporters lay it out: an animated rig node with the camera
// node hanging off it, so the camera pose only exists as a side effect of the
// node hierarchy being evaluated.
std::unique_ptr<Scene> makeAnimatedCameraScene()
{
    auto scene = std::make_unique<Scene>();

    Camera cam;
    cam.name = "Camera";
    cam.node = 1;
    scene->addCamera(cam);

    Scene::Node rig{};
    rig.name = "Rig";
    rig.type = Scene::Node::NodeType::sceneGraph;
    rig.children = { 1 };
    scene->mNodes.push_back(rig);

    Scene::Node camNode{};
    camNode.name = "CameraOrientation";
    camNode.type = Scene::Node::NodeType::camera;
    camNode.camera = 0;
    camNode.parent = 0;
    scene->mNodes.push_back(camNode);

    Scene::AnimationSampler sampler{};
    sampler.interpolation = Scene::AnimationSampler::InterpolationType::LINEAR;
    sampler.inputs = { 0.0f, 1.0f };
    sampler.outputsVec4 = { glm::float4(0.0f, 0.0f, 0.0f, 0.0f), glm::float4(10.0f, 0.0f, 0.0f, 0.0f) };

    Scene::AnimationChannel channel{};
    channel.path = Scene::AnimationChannel::PathType::TRANSLATION;
    channel.node = 0;
    channel.samplerIndex = 0;

    Scene::Animation animation{};
    animation.samplers.push_back(sampler);
    animation.channels.push_back(channel);
    animation.start = 0.0f;
    animation.end = 1.0f;
    animation.current = 0.0f;
    scene->mAnimations.push_back(animation);

    return scene;
}

} // namespace

TEST_CASE("a default node is an identity transform")
{
    // Everything downstream of a node reads its world transform, and the
    // hierarchy multiplies a garbage local into every descendant.
    Scene::Node node{};
    CHECK(node.translation == glm::float3(0.0f));
    CHECK(node.scale == glm::float3(1.0f));
    CHECK(node.rotation == glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
}

TEST_CASE("animation poses a camera through its node")
{
    auto scene = makeAnimatedCameraScene();

    scene->mAnimations[0].current = 0.0f;
    scene->applyAnimation(0);
    CHECK(scene->getCamera(0).position.x == doctest::Approx(0.0f));

    scene->mAnimations[0].current = 0.5f;
    scene->applyAnimation(0);
    CHECK(scene->getCamera(0).position.x == doctest::Approx(5.0f));

    scene->mAnimations[0].current = 1.0f;
    scene->applyAnimation(0);
    CHECK(scene->getCamera(0).position.x == doctest::Approx(10.0f));

    // The view matrix has to follow, not just the stored position: it is what
    // the renderer draws with and what the viewport projects the overlay with.
    const glm::float4x4 viewAtEnd = scene->getCamera(0).matrices.view;
    CHECK(glm::float3(viewAtEnd * glm::float4(10.0f, 0.0f, 0.0f, 1.0f)).x == doctest::Approx(0.0f));
}

TEST_CASE("manual control keeps animation off a camera")
{
    auto scene = makeAnimatedCameraScene();

    // The user takes the camera over somewhere along the clip.
    scene->mAnimations[0].current = 0.5f;
    scene->applyAnimation(0);

    Camera& cam = scene->getCamera(0);
    cam.manualControl = true;
    cam.position = glm::float3(-3.0f, 2.0f, 1.0f);
    cam.updateViewMatrix();
    const glm::float4x4 userView = cam.matrices.view;

    // Playback continues -- the pose is the user's, not the clip's. Both have to
    // hold: a stale position would show up in the gizmo and picking ray, a stale
    // view matrix in the rendered frame.
    for (float t : { 0.25f, 0.75f, 1.0f })
    {
        scene->mAnimations[0].current = t;
        scene->applyAnimation(0);
        CHECK(scene->getCamera(0).position.x == doctest::Approx(-3.0f));
        CHECK(scene->getCamera(0).position.y == doctest::Approx(2.0f));
        CHECK(scene->getCamera(0).matrices.view == userView);
    }

    // Re-attaching hands the camera back.
    scene->getCamera(0).manualControl = false;
    scene->mAnimations[0].current = 1.0f;
    scene->applyAnimation(0);
    CHECK(scene->getCamera(0).position.x == doctest::Approx(10.0f));
}

TEST_CASE("manual control is per camera")
{
    auto scene = makeAnimatedCameraScene();

    Camera second;
    second.name = "Second";
    second.node = 3;
    scene->addCamera(second);

    Scene::Node secondRig{};
    secondRig.name = "SecondRig";
    secondRig.type = Scene::Node::NodeType::sceneGraph;
    secondRig.children = { 3 };
    scene->mNodes.push_back(secondRig);

    Scene::Node secondCamNode{};
    secondCamNode.name = "SecondCameraOrientation";
    secondCamNode.type = Scene::Node::NodeType::camera;
    secondCamNode.camera = 1;
    secondCamNode.parent = 2;
    scene->mNodes.push_back(secondCamNode);

    Scene::AnimationChannel channel{};
    channel.path = Scene::AnimationChannel::PathType::TRANSLATION;
    channel.node = 2;
    channel.samplerIndex = 0;
    scene->mAnimations[0].channels.push_back(channel);

    scene->getCamera(0).manualControl = true;
    scene->getCamera(0).position = glm::float3(0.0f);
    scene->mAnimations[0].current = 1.0f;
    scene->applyAnimation(0);

    CHECK(scene->getCamera(0).position.x == doctest::Approx(0.0f));
    CHECK(scene->getCamera(1).position.x == doctest::Approx(10.0f));
}
