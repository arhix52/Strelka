#include <strelka/scene/scene.h>

#include <algorithm>
#include <chrono>
#include <strelka/scene/vertex_packing.h>
#include <strelka/scene/light_desc.h>
#include <analytic_light.h>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/norm.hpp>

#include <strelka/scene/transform.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <unordered_map>

#include <log.h>
#include <numbers>

namespace fs = std::filesystem;

namespace oka
{

namespace
{

glm::float3 transformedAreaLightNormal(const glm::float4x4& transform)
{
    const glm::float3 axisX(transform * glm::float4(1.0f, 0.0f, 0.0f, 0.0f));
    const glm::float3 axisY(transform * glm::float4(0.0f, 1.0f, 0.0f, 0.0f));
    const glm::float3 axisZ(transform * glm::float4(0.0f, 0.0f, 1.0f, 0.0f));
    return transformAffineNormal(axisX, axisY, axisZ, glm::float3(0.0f, 0.0f, -1.0f));
}

glm::float3 transformedDirectionOrZero(const glm::float4x4& transform, const glm::float3& direction)
{
    const glm::float3 transformed(transform * glm::float4(direction, 0.0f));
    return normalizeFiniteVectorOrZero(transformed);
}

} // namespace

uint32_t Scene::acquireMeshSlot(Mesh*& mesh)
{
    uint32_t meshId = kInvalidIndex;
    if (mDelMesh.empty())
    {
        meshId = static_cast<uint32_t>(mMeshes.size());
        mMeshes.push_back({});
        mesh = &mMeshes.back();
    }
    else
    {
        meshId = mDelMesh.top();
        mDelMesh.pop();
        mesh = &mMeshes[meshId];
    }
    return meshId;
}

void Scene::reserveGeometry(size_t vertexCount, size_t indexCount, size_t skinCount)
{
    // 16 KiB is the Apple Silicon page and a multiple of 4 KiB, so rounding
    // capacity to it makes a Metal no-copy wrap legal on either.
    constexpr size_t kPage = 16384;
    auto pageRound = [](size_t count, size_t elemSize) -> size_t {
        if (count == 0)
        {
            return 0;
        }
        const size_t bytes = count * elemSize;
        const size_t padded = (bytes + kPage - 1) & ~(kPage - 1);
        return padded / elemSize;
    };
    mVertices.reserve(pageRound(vertexCount, sizeof(Vertex)));
    mIndices.reserve(pageRound(indexCount, sizeof(uint32_t)));
    if (skinCount != 0)
    {
        mVerticesSkinData.reserve(skinCount);
    }
}

uint32_t Scene::createMesh(const std::vector<Vertex>& vb, const std::vector<uint32_t>& ib)
{
    const std::scoped_lock lock(mMeshMutex);

    Mesh* mesh = nullptr;
    const uint32_t meshId = acquireMeshSlot(mesh);

    mesh->mIndex = static_cast<uint32_t>(mIndices.size());
    mesh->mCount = static_cast<uint32_t>(ib.size());
    mesh->mVbOffset = static_cast<uint32_t>(mVertices.size());
    mesh->mVertexCount = static_cast<uint32_t>(vb.size());

    mIndices.insert(mIndices.end(), ib.begin(), ib.end());
    mVertices.insert(mVertices.end(), vb.begin(), vb.end());
    return meshId;
}

uint32_t Scene::createSkeletalMesh(const std::vector<Vertex>& vb,
                                   const std::vector<uint32_t>& ib,
                                   const std::vector<oka::Scene::vertexSkinData>& sb)
{
    const std::scoped_lock lock(mMeshMutex);

    Mesh* mesh = nullptr;
    const uint32_t meshId = acquireMeshSlot(mesh);

    mesh->mIndex = static_cast<uint32_t>(mIndices.size());
    mesh->mCount = static_cast<uint32_t>(ib.size());
    mesh->mVbOffset = static_cast<uint32_t>(mVertices.size());
    mesh->mVertexCount = static_cast<uint32_t>(vb.size());
    mesh->mSbOffset = static_cast<uint32_t>(mVerticesSkinData.size());
    mesh->isSkeletal = true;

    mIndices.insert(mIndices.end(), ib.begin(), ib.end());
    mVertices.insert(mVertices.end(), vb.begin(), vb.end());
    mVerticesSkinData.insert(mVerticesSkinData.end(), sb.begin(), sb.end());
    return meshId;
}

uint32_t Scene::createMeshFromOffsets(uint32_t vbOffset, uint32_t vertexCount, uint32_t ibOffset, uint32_t indexCount)
{
    const std::scoped_lock lock(mMeshMutex);

    Mesh* mesh = nullptr;
    const uint32_t meshId = acquireMeshSlot(mesh);
    mesh->mIndex = ibOffset;
    mesh->mCount = indexCount;
    mesh->mVbOffset = vbOffset;
    mesh->mVertexCount = vertexCount;
    return meshId;
}

uint32_t Scene::createSkeletalMeshFromOffsets(uint32_t vbOffset,
                                              uint32_t vertexCount,
                                              uint32_t ibOffset,
                                              uint32_t indexCount,
                                              uint32_t sbOffset,
                                              uint32_t /*skinCount*/)
{
    const std::scoped_lock lock(mMeshMutex);

    Mesh* mesh = nullptr;
    const uint32_t meshId = acquireMeshSlot(mesh);
    mesh->mIndex = ibOffset;
    mesh->mCount = indexCount;
    mesh->mVbOffset = vbOffset;
    mesh->mVertexCount = vertexCount;
    mesh->mSbOffset = sbOffset;
    mesh->isSkeletal = true;
    return meshId;
}

uint32_t Scene::createInstance(const Instance::Type type,
                               const uint32_t geomId,
                               const uint32_t materialId,
                               const glm::mat4& transform,
                               const uint32_t lightId)
{
    const std::scoped_lock lock(mInstanceMutex);

    Instance* inst = nullptr;
    uint32_t instId = -1;
    if (mDelInstances.empty())
    {
        instId = mInstances.size(); // add instance to storage
        mInstances.push_back({});
        inst = &mInstances.back();
    }
    else
    {
        instId = mDelInstances.top(); // get index from stack
        mDelInstances.pop(); // del taken index from stack
        inst = &mInstances[instId];
    }
    inst->type = type;
    if (inst->type == Instance::Type::eMesh || inst->type == Instance::Type::eLight)
    {
        inst->mMeshId = geomId;
    }
    else if (inst->type == Instance::Type::eCurve)
    {
        inst->mCurveId = geomId;
    }
    inst->mMaterialId = materialId;
    inst->transform = transform;
    inst->mLightId = lightId;

    return instId;
}

uint32_t Scene::addMaterial(const MaterialDescription& material)
{
    // TODO: fix here
    const uint32_t res = mMaterialsDescs.size();
    mMaterialsDescs.push_back(material);
    return res;
}

std::string Scene::getSceneFileName()
{
    const fs::path p(modelPath);
    return p.filename().string();
};

std::string Scene::getSceneDir()
{
    const fs::path p(modelPath);
    return p.parent_path().string();
}

glm::quat Scene::makeQuatFromFloat4(const glm::float4& value)
{
    return quatFromGltf(value[0], value[1], value[2], value[3]);
}

glm::float4 Scene::makeFloat4FromQuat(const glm::quat& q)
{
    return { q.x, q.y, q.z, q.w };
}

// packNormal() provided by <strelka/scene/vertex_packing.h>

glm::float4 Scene::interpolate(const AnimationSampler& sampler,
                               const AnimationChannel::PathType targetProperty,
                               const float time)
{
    const bool isCubic = (sampler.interpolation == AnimationSampler::InterpolationType::CUBICSPLINE);
    // For CUBICSPLINE, outputsVec4 stores triplets: [inTangent, value, outTangent] per keyframe.
    // For LINEAR/STEP, outputsVec4 stores one value per keyframe.
    const int stride = isCubic ? 3 : 1;
    const int valueOffset = isCubic ? 1 : 0;

    const int n = (int)sampler.inputs.size();
    if (n == 0)
        return glm::float4(0.0f);

    // Clamp to range
    if (time <= sampler.inputs[0])
        return sampler.outputsVec4[valueOffset];
    if (time >= sampler.inputs[n - 1])
        return sampler.outputsVec4[(n - 1) * stride + valueOffset];

    // Find bracket: inputs[prevIdx] <= time < inputs[nextIdx].
    // inputs is sorted by construction, so binary search it — the previous linear
    // scan cost O(keyframes) per channel per frame (BrainStem has channels with
    // 838 keys, evaluated 116 times per pass and twice per frame).
    const auto upper = std::ranges::upper_bound(sampler.inputs, time);
    int nextIdx = (int)std::distance(sampler.inputs.begin(), upper);
    nextIdx = std::clamp(nextIdx, 1, n - 1);
    const int prevIdx = nextIdx - 1;

    const float previousTime = sampler.inputs[prevIdx];
    const float nextTime = sampler.inputs[nextIdx];

    // Exact match — return value directly
    if (std::abs(time - previousTime) < 1e-7f)
        return sampler.outputsVec4[prevIdx * stride + valueOffset];

    glm::float4 result;

    switch (sampler.interpolation)
    {
    case AnimationSampler::InterpolationType::STEP:
        result = sampler.outputsVec4[prevIdx * stride + valueOffset];
        break;

    case AnimationSampler::InterpolationType::CUBICSPLINE: {
        // glTF cubic spline: Hermite interpolation
        // outputsVec4 layout per keyframe: [inTangent, value, outTangent]
        const float deltaTime = nextTime - previousTime;
        const float t = (time - previousTime) / deltaTime;
        const float t2 = t * t;
        const float t3 = t2 * t;

        const glm::float4 p0 = sampler.outputsVec4[prevIdx * 3 + 1]; // value at prev
        const glm::float4 m0 = sampler.outputsVec4[prevIdx * 3 + 2] * deltaTime; // out-tangent at prev
        const glm::float4 p1 = sampler.outputsVec4[nextIdx * 3 + 1]; // value at next
        const glm::float4 m1 = sampler.outputsVec4[nextIdx * 3 + 0] * deltaTime; // in-tangent at next

        result = (2.0f * t3 - 3.0f * t2 + 1.0f) * p0 + (t3 - 2.0f * t2 + t) * m0 + (-2.0f * t3 + 3.0f * t2) * p1 +
                 (t3 - t2) * m1;

        if (targetProperty == AnimationChannel::PathType::ROTATION)
            result = makeFloat4FromQuat(glm::normalize(makeQuatFromFloat4(result)));
        break;
    }

    default: // LINEAR
    {
        const float interpolationValue = (time - previousTime) / (nextTime - previousTime);
        const glm::float4 prevVal = sampler.outputsVec4[prevIdx];
        const glm::float4 nextVal = sampler.outputsVec4[nextIdx];
        if (targetProperty != AnimationChannel::PathType::ROTATION)
            result = glm::mix(prevVal, nextVal, interpolationValue);
        else
            result = makeFloat4FromQuat(
                glm::slerp(makeQuatFromFloat4(prevVal), makeQuatFromFloat4(nextVal), interpolationValue));
        break;
    }
    }
    return result;
}

void Scene::buildNodeOrder()
{
    mNodeOrder.clear();
    mNodeOrder.reserve(mNodes.size());

    // Breadth-first from every root guarantees a parent is emitted before any of
    // its children, which is all the top-down transform pass needs.
    std::vector<int> queue;
    queue.reserve(mNodes.size());
    for (size_t i = 0; i < mNodes.size(); ++i)
    {
        if (mNodes[i].parent == -1)
            queue.push_back((int)i);
    }
    for (size_t head = 0; head < queue.size(); ++head)
    {
        const int nodeId = queue[head];
        mNodeOrder.push_back(nodeId);
        for (const int childId : mNodes[nodeId].children)
        {
            if (childId >= 0 && (size_t)childId < mNodes.size())
                queue.push_back(childId);
        }
    }

    // A malformed hierarchy (cycle or orphan) would leave nodes unvisited; append
    // them so their transforms are at least computed from their own local TRS.
    if (mNodeOrder.size() != mNodes.size())
    {
        std::vector<uint8_t> seen(mNodes.size(), 0);
        for (const int nodeId : mNodeOrder)
            seen[nodeId] = 1;
        for (size_t i = 0; i < mNodes.size(); ++i)
        {
            if (!seen[i])
                mNodeOrder.push_back((int)i);
        }
    }
}

void Scene::refreshGlobalTransforms()
{
    mGlobalTransforms.resize(mNodes.size());
    for (const int nodeId : mNodeOrder)
    {
        const glm::mat4 local = calculateNodeLocalTransform(nodeId);
        const int parent = mNodes[nodeId].parent;
        mGlobalTransforms[nodeId] = (parent == -1) ? local : mGlobalTransforms[parent] * local;
    }
}

void Scene::ensureGlobalTransforms()
{
    if (mNodeOrder.size() != mNodes.size())
    {
        buildNodeOrder();
        mNodeDirty.assign(mNodes.size(), 0);
        refreshGlobalTransforms();
    }
}

bool Scene::applyNodeSideEffects(const uint32_t nodeId)
{
    // Preserve the legacy traversal contract: mesh and camera nodes
    // consume the update and do not propagate it to their children.
    switch (mNodes[nodeId].type)
    {
    case Node::NodeType::mesh:
        for (const auto instId : mNodes[nodeId].instanceIds)
        {
            Instance& inst = mInstances[instId];
            inst.transform = mGlobalTransforms[nodeId];
            inst.isAnimated = true;
            mDirtyInstances.insert(instId);
            ++mTransformGeneration;
        }
        return false;

    case Node::NodeType::camera:
        if (mNodes[nodeId].camera >= 0 && (size_t)mNodes[nodeId].camera < mCameras.size() &&
            !mCameras[mNodes[nodeId].camera].manualControl)
        {
            glm::float3 scale;
            glm::quat rotation;
            glm::float3 translation;
            decomposeTrs(mGlobalTransforms[nodeId], translation, rotation, scale);
            rotation = glm::conjugate(rotation);

            Camera& cam = mCameras[mNodes[nodeId].camera];
            cam.position = translation * scale;
            cam.mOrientation = rotation;
            cam.updateViewMatrix();
        }
        return false;

    case Node::NodeType::skeleton:
    default:
        break;
    }

    bool skeletonUpdated = (mNodes[nodeId].type == Node::NodeType::skeleton);
    for (const auto childId : mNodes[nodeId].children)
    {
        skeletonUpdated |= applyNodeSideEffects(childId);
    }
    return skeletonUpdated;
}

bool Scene::applyAnimation(const uint32_t animId)
{
    ensureGlobalTransforms();

    // Two phases instead of one update per channel. The legacy path recomputed each visited
    // node's world transform by walking back up to the root — so a scene with C
    // channels cost O(C * subtree * depth) matrix builds every frame, re-deriving
    // the same ancestors again and again. BrainStem has 116 channels over a
    // 30-node graph and paid that twice per frame (motion blur is a two-pass
    // evaluation), which is what made playback stutter on a 34k-triangle scene.
    //
    // Phase 1 only writes local TRS; phase 2 derives every world transform in a
    // single parent-before-child sweep.
    auto& animation = mAnimations[animId];
    std::ranges::fill(mNodeDirty, 0);

    for (size_t i = 0; i < animation.channels.size(); ++i)
    {
        const uint32_t nodeId = animation.channels[i].node;
        if (nodeId >= mNodes.size())
            continue;

        const AnimationChannel::PathType targetProperty = animation.channels[i].path;
        const glm::float4 value =
            interpolate(animation.samplers[animation.channels[i].samplerIndex], targetProperty, animation.current);

        switch (targetProperty)
        {
        case AnimationChannel::PathType::TRANSLATION:
            mNodes[nodeId].translation = glm::float3(value);
            break;
        case AnimationChannel::PathType::SCALE:
            mNodes[nodeId].scale = glm::float3(value);
            break;
        case AnimationChannel::PathType::ROTATION:
            mNodes[nodeId].rotation = makeQuatFromFloat4(value);
            break;
        default:
            continue;
        }
        mNodeDirty[nodeId] = 1;
    }

    // Every world transform is recomputed, so the cache stays valid for skinning
    // even where the side-effect traversal below stops early.
    refreshGlobalTransforms();

    bool blasChanged = false;
    for (size_t nodeId = 0; nodeId < mNodes.size(); ++nodeId)
    {
        if (mNodeDirty[nodeId])
            blasChanged |= applyNodeSideEffects((uint32_t)nodeId);
    }
    // Playback already returns whether the accel structure needs a rebuild; do
    // not also raise ChangeBits::Transforms or the renderer would rebuild TLAS
    // twice a frame (once from handleSceneChanges, once from the anim path).
    return blasChanged;
}

void Scene::computeJointMatrices(std::vector<glm::mat4>* jointMatrices, const size_t jointCount, const uint32_t skinId)
{
    ensureGlobalTransforms();

    auto& skin = mSkines[skinId];
    jointMatrices->reserve(jointMatrices->size() + jointCount);
    for (size_t i = 0; i < jointCount; ++i)
    {
        // Read the cached world transform instead of re-walking to the root for
        // each joint: applyAnimation() already refreshed the whole table.
        jointMatrices->push_back(mGlobalTransforms[skin.joints[i]] * skin.inverseBindMatrices[i]);
    }
}

glm::mat4 Scene::calculateNodeLocalTransform(const uint32_t nodeId)
{
    const glm::float4x4 translationMatrix = glm::translate(glm::float4x4(1.0f), mNodes[nodeId].translation);
    const glm::float4x4 rotationMatrix{ mNodes[nodeId].rotation };
    const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), mNodes[nodeId].scale);
    return translationMatrix * rotationMatrix * scaleMatrix;
}

glm::mat4 Scene::calculateNodeGlobalTransform(const uint32_t nodeId)
{
    const int parentId = mNodes[nodeId].parent;
    if (parentId == -1)
    {
        return calculateNodeLocalTransform(nodeId);
    }
    else
    {
        return calculateNodeGlobalTransform(parentId) * calculateNodeLocalTransform(nodeId);
    }
}

uint32_t Scene::createRectLightMesh()
{
    if (mRectLightMeshId != kInvalidIndex)
    {
        return mRectLightMeshId;
    }

    std::vector<Scene::Vertex> vb;
    Scene::Vertex v1, v2, v3, v4;
    v1.pos = glm::float4(0.5f, 0.5f, 0.0f, 1.0f); // top right 0
    v2.pos = glm::float4(-0.5f, 0.5f, 0.0f, 1.0f); // top left 1
    v3.pos = glm::float4(-0.5f, -0.5f, 0.0f, 1.0f); // bottom left 2
    v4.pos = glm::float4(0.5f, -0.5f, 0.0f, 1.0f); // bottom right 3
    const glm::float3 normal = glm::float3(0.f, 0.f, 1.f);
    v1.normal = v2.normal = v3.normal = v4.normal = packNormal(normal);
    const std::vector<uint32_t> ib = { 0, 1, 2, 2, 3, 0 };
    vb.push_back(v1);
    vb.push_back(v2);
    vb.push_back(v3);
    vb.push_back(v4);

    const uint32_t meshId = createMesh(vb, ib);
    assert(meshId != std::numeric_limits<uint32_t>::max());

    return meshId;
}

uint32_t Scene::createSphereLightMesh()
{
    if (mSphereLightMeshId != kInvalidIndex)
    {
        return mSphereLightMeshId;
    }

    std::vector<Scene::Vertex> vertices;
    std::vector<uint32_t> indices;
    const int segments = 16;
    const int rings = 16;
    const float radius = 1.0f;
    // Generate vertices and normals
    for (int i = 0; i <= rings; ++i)
    {
        const float theta = static_cast<float>(i) * std::numbers::pi_v<float> / static_cast<float>(rings);
        const float sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);

        for (int j = 0; j <= segments; ++j)
        {
            const float phi = static_cast<float>(j) * 2.0f * std::numbers::pi_v<float> / static_cast<float>(segments);
            const float sinPhi = std::sin(phi);
            const float cosPhi = std::cos(phi);

            const float x = cosPhi * sinTheta;
            const float y = cosTheta;
            const float z = sinPhi * sinTheta;

            const glm::float3 pos = { radius * x, radius * y, radius * z };
            const glm::float3 normal = { x, y, z };

            vertices.push_back(Scene::Vertex{ pos, 0, packNormal(normal), 0 });
        }
    }
    // Generate indices
    for (int i = 0; i < rings; ++i)
    {
        for (int j = 0; j < segments; ++j)
        {
            int p0 = i * (segments + 1) + j;
            int p1 = p0 + 1;
            int p2 = (i + 1) * (segments + 1) + j;
            int p3 = p2 + 1;

            indices.push_back(p0);
            indices.push_back(p1);
            indices.push_back(p2);

            indices.push_back(p2);
            indices.push_back(p1);
            indices.push_back(p3);
        }
    }
    const uint32_t meshId = createMesh(vertices, indices);
    assert(meshId != std::numeric_limits<uint32_t>::max());

    return meshId;
}

uint32_t Scene::createDiscLightMesh()
{
    if (mDiskLightMeshId != kInvalidIndex)
    {
        return mDiskLightMeshId;
    }

    std::vector<Scene::Vertex> vertices;
    std::vector<uint32_t> indices;

    Scene::Vertex v1, v2;
    v1.pos = glm::float4(0.f, 0.f, 0.f, 1.f);
    v2.pos = glm::float4(1.0f, 0.f, 0.f, 1.f);

    const glm::float3 normal = glm::float3(0.f, 0.f, 1.f);
    v1.normal = v2.normal = packNormal(normal);

    vertices.push_back(v1); // central point
    vertices.push_back(v2); // first point

    const float diskRadius = 1.0f; // param
    const float step = 2.0f * std::numbers::pi / 16;
    float angle = 0;
    for (int i = 0; i < 16; ++i)
    {
        indices.push_back(0); // each triangle have central point
        indices.push_back(vertices.size() - 1); // prev vertex

        angle += step;
        const float x = std::cos(angle) * diskRadius;
        const float y = std::sin(angle) * diskRadius;

        Scene::Vertex v;
        v.pos = glm::float4(x, y, 0.0f, 1.0f);
        v.normal = packNormal(normal);
        vertices.push_back(v);

        indices.push_back(vertices.size() - 1); // added vertex
    }

    const uint32_t meshId = createMesh(vertices, indices);
    assert(meshId != std::numeric_limits<uint32_t>::max());

    return meshId;
}

uint32_t Scene::createLight(const UniformLightDesc& desc)
{
    const auto lightId = (uint32_t)mLights.size();
    const Light l{};
    mLights.push_back(l);
    mLightDesc.push_back(desc);

    updateLight(lightId, desc);

    // Lazy init light mesh. Distant lights have none; point/spot get a small
    // proxy so the outliner and the gizmo still have something to select.
    glm::float4x4 scaleMatrix = glm::float4x4(1.0f);
    uint32_t currentLightMeshId = 0;
    if (desc.type == LIGHT_TYPE_RECT)
    {
        mRectLightMeshId = createRectLightMesh();
        currentLightMeshId = mRectLightMeshId;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.width, desc.height, 1.0f));
    }
    else if (desc.type == LIGHT_TYPE_DISC)
    {
        mDiskLightMeshId = createDiscLightMesh();
        currentLightMeshId = mDiskLightMeshId;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.radius, desc.radius, desc.radius));
    }
    else if (desc.type == LIGHT_TYPE_SPHERE || desc.type == LIGHT_TYPE_POINT)
    {
        mSphereLightMeshId = createSphereLightMesh();
        currentLightMeshId = mSphereLightMeshId;
        const float r = desc.radius > 1e-4f ? desc.radius : 0.05f;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(r));
    }
    else if (desc.type == LIGHT_TYPE_SPOT || desc.type == LIGHT_TYPE_PROJECTOR)
    {
        mDiskLightMeshId = createDiscLightMesh();
        currentLightMeshId = mDiskLightMeshId;
        const float r = desc.radius > 1e-4f ? desc.radius : 0.05f;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(r));
    }
    else
    {
        return lightId;
    }

    const glm::float4x4 transform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);
    const uint32_t instId = createInstance(
        Instance::Type::eLight, currentLightMeshId, std::numeric_limits<uint32_t>::max(), transform, lightId);
    assert(instId != std::numeric_limits<uint32_t>::max());

    mLightIdToInstanceId[lightId] = instId;

    return lightId;
}

void Scene::updateLight(const uint32_t lightId, const UniformLightDesc& desc)
{
    // transform to GPU light
    if (desc.type == LIGHT_TYPE_RECT)
    {
        const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.width, desc.height, 1.0f));
        const glm::float4x4 localTransform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);

        mLights[lightId].points[0] = localTransform * glm::float4(0.5f, 0.5f, 0.0f, 1.0f);
        mLights[lightId].points[1] = localTransform * glm::float4(-0.5f, 0.5f, 0.0f, 1.0f);
        mLights[lightId].points[2] = localTransform * glm::float4(-0.5f, -0.5f, 0.0f, 1.0f);
        mLights[lightId].points[3] = localTransform * glm::float4(0.5f, -0.5f, 0.0f, 1.0f);
        mLights[lightId].normal = glm::float4(transformedAreaLightNormal(localTransform), 0.0f);

        mLights[lightId].type = LIGHT_TYPE_RECT;
        mLights[lightId].halfAngle = 0.0f;
        mLights[lightId].pad0 = 0.0f;
        // Controlled-falloff cutoff distance for area lights, read by
        // areaFalloff(); 0 (the default range) leaves the light unbounded.
        mLights[lightId].pad1 = desc.range;
    }
    else if (desc.type == LIGHT_TYPE_DISC)
    {
        const glm::float4x4 scaleMatrix =
            glm::scale(glm::float4x4(1.0f), glm::float3(desc.radius, desc.radius, desc.radius));
        const glm::float4x4 localTransform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);

        mLights[lightId].points[0] = glm::float4(desc.radius, 0.f, 0.f, 0.f);
        mLights[lightId].points[1] = localTransform * glm::float4(0.f, 0.f, 0.f, 1.f);
        mLights[lightId].points[2] = localTransform * glm::float4(1.f, 0.f, 0.f, 0.f);
        mLights[lightId].points[3] = localTransform * glm::float4(0.f, 1.f, 0.f, 0.f);

        mLights[lightId].normal = glm::float4(transformedAreaLightNormal(localTransform), 0.0f);
        mLights[lightId].type = LIGHT_TYPE_DISC;
        mLights[lightId].halfAngle = 0.0f;
        mLights[lightId].pad0 = 0.0f;
        // Controlled-falloff cutoff distance for area lights, read by
        // areaFalloff(); 0 (the default range) leaves the light unbounded.
        mLights[lightId].pad1 = desc.range;
    }
    else if (desc.type == LIGHT_TYPE_SPHERE)
    {
        const glm::float4x4 scaleMatrix =
            glm::scale(glm::float4x4(1.0f), glm::float3(desc.radius, desc.radius, desc.radius));
        const glm::float4x4 localTransform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);

        // Centre plus all three transformed unit-sphere axes. This is the full
        // affine ellipsoid used by both sampling and renderer intersection.
        mLights[lightId].points[0] = localTransform * glm::float4(1.f, 0.f, 0.f, 0.f);
        mLights[lightId].points[1] = localTransform * glm::float4(0.f, 0.f, 0.f, 1.f);
        mLights[lightId].points[2] = localTransform * glm::float4(0.f, 1.f, 0.f, 0.f);
        mLights[lightId].points[3] = localTransform * glm::float4(0.f, 0.f, 1.f, 0.f);

        mLights[lightId].type = LIGHT_TYPE_SPHERE;
        mLights[lightId].halfAngle = 0.0f;
        mLights[lightId].pad0 = 0.0f;
        mLights[lightId].pad1 = 0.0f;
    }
    else if (lightTypeIsPunctual(desc.type))
    {
        const glm::float4x4 localTransform = desc.useXform ? desc.xform : getTransform(desc);
        // The three lamps share one packing -- see lightIsPunctual() in
        // light_pdf.h. points[0] is (soft radius, IES profile, projector image,
        // frame aspect); a light that has no use for a slot carries -1 or 0 in
        // it rather than a stale value, because the shader decides what a light
        // does from these numbers and not from its type alone.
        //
        // A projector's angular shape is its image, so it never also carries an
        // IES profile: -1 goes in that slot even when the desc still remembers a
        // file from before the type was switched.
        const bool isProjector = desc.type == LIGHT_TYPE_PROJECTOR;
        mLights[lightId].points[0] =
            glm::float4(desc.radius, isProjector ? -1.0f : (float)desc.iesProfile,
                        isProjector ? (float)desc.projectorImage : -1.0f, isProjector ? desc.projectorAspect : 0.0f);
        mLights[lightId].points[1] = localTransform * glm::float4(0.f, 0.f, 0.f, 1.f);
        // Local axes so an IES profile or a projected image can be evaluated in
        // light space.
        mLights[lightId].points[2] = localTransform * glm::float4(1.f, 0.f, 0.f, 0.f);
        mLights[lightId].points[3] = localTransform * glm::float4(0.f, 1.f, 0.f, 0.f);
        mLights[lightId].normal = glm::float4(transformedDirectionOrZero(localTransform, glm::float3(0.0f, 0.0f, -1.0f)),
                                             0.0f);
        mLights[lightId].type = desc.type;
        // Spot: the outer cone. Projector: half of the horizontal field of view,
        // which is the same quantity in the same field -- both are the angle at
        // which the light stops -- so the two need no separate slot.
        mLights[lightId].halfAngle = (desc.type == LIGHT_TYPE_SPOT || isProjector) ? desc.outerConeAngle : 0.0f;
        mLights[lightId].pad0 = isProjector                  ? desc.projectorEdgeSoftness :
                                desc.type == LIGHT_TYPE_SPOT ? desc.innerConeAngle :
                                                               desc.radius;
        mLights[lightId].pad1 = desc.range;
    }
    else if (desc.type == LIGHT_TYPE_DISTANT)
    {
        mLights[lightId].type = LIGHT_TYPE_DISTANT;
        mLights[lightId].halfAngle =
            std::isfinite(desc.halfAngle) ? std::clamp(desc.halfAngle, 0.0f, std::numbers::pi_v<float>) : 0.0f;
        mLights[lightId].pad0 = 0.0f;
        mLights[lightId].pad1 = 0.0f;
        const glm::float4x4 localTransform = desc.useXform ? desc.xform : getTransform(desc);
        mLights[lightId].normal = glm::float4(transformedDirectionOrZero(localTransform, glm::float3(0.0f, 0.0f, -1.0f)),
                                             0.0f);
    }
    else if (desc.type == LIGHT_TYPE_DOME)
    {
        mLights[lightId].type = LIGHT_TYPE_DOME;
        mLights[lightId].halfAngle = 0.0f;
        mLights[lightId].pad0 = 0.0f;
        mLights[lightId].pad1 = 0.0f;
    }

    const glm::float3 radiometric =
        desc.enabled ?
            bakeLightRadiometric(desc.type, desc.intensityUnit, desc.color, desc.intensity, desc.width, desc.height,
                                 desc.radius, desc.halfAngle, desc.outerConeAngle, desc.projectorAspect) :
            glm::float3(0.0f);
    mLights[lightId].color = glm::float4(radiometric, 1.0f);
    // Disc and sphere intersections are evaluated against their smooth
    // analytic geometry in every renderer. Keep visibility beside the packed
    // axes so the manual intersection path has the same camera-hidden semantics
    // the former proxy instance mask had.
    mLights[lightId].normal.w = desc.enabled ? float(STRELKA_ANALYTIC_LIGHT_SECONDARY_BIT |
                                                     (desc.visibleToCamera ? STRELKA_ANALYTIC_LIGHT_CAMERA_BIT : 0u)) :
                                               0.0f;
    markChanged(ChangeBits::Lights);
}

int32_t Scene::addIesProfile(IesProfile profile)
{
    // Reuse an already-loaded path so toggling the same file in the UI does not
    // grow the table without bound.
    for (size_t i = 0; i < mIesProfiles.size(); ++i)
    {
        if (mIesProfiles[i].path == profile.path)
        {
            return (int32_t)i;
        }
    }
    mIesProfiles.push_back(std::move(profile));
    return (int32_t)mIesProfiles.size() - 1;
}

int32_t Scene::addProjectorImage(const std::string& path)
{
    // Same dedup as addIesProfile, and for the same reason: picking the same
    // file twice in the UI must not grow the GPU texture table, and two
    // projectors showing one slide should be one upload.
    for (size_t i = 0; i < mProjectorImages.size(); ++i)
    {
        if (mProjectorImages[i] == path)
        {
            return (int32_t)i;
        }
    }
    mProjectorImages.push_back(path);
    return (int32_t)mProjectorImages.size() - 1;
}

void Scene::setLight(const uint32_t lightId, const UniformLightDesc& desc)
{
    assert(lightId < mLightDesc.size());
    mLightDesc[lightId] = desc;
    updateLight(lightId, desc);

    glm::float4x4 scaleMatrix = glm::float4x4(1.0f);
    uint32_t desiredMeshId = kInvalidIndex;
    if (desc.type == LIGHT_TYPE_RECT)
    {
        if (mRectLightMeshId == kInvalidIndex)
        {
            mRectLightMeshId = createRectLightMesh();
        }
        desiredMeshId = mRectLightMeshId;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.width, desc.height, 1.0f));
    }
    else if (desc.type == LIGHT_TYPE_DISC || desc.type == LIGHT_TYPE_SPOT || desc.type == LIGHT_TYPE_PROJECTOR)
    {
        if (mDiskLightMeshId == kInvalidIndex)
        {
            mDiskLightMeshId = createDiscLightMesh();
        }
        desiredMeshId = mDiskLightMeshId;
        const float radius = desc.type == LIGHT_TYPE_DISC ? desc.radius : (desc.radius > 1e-4f ? desc.radius : 0.05f);
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(radius));
    }
    else if (desc.type == LIGHT_TYPE_SPHERE || desc.type == LIGHT_TYPE_POINT)
    {
        if (mSphereLightMeshId == kInvalidIndex)
        {
            mSphereLightMeshId = createSphereLightMesh();
        }
        desiredMeshId = mSphereLightMeshId;
        const float radius = desc.type == LIGHT_TYPE_SPHERE ? desc.radius : (desc.radius > 1e-4f ? desc.radius : 0.05f);
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(radius));
    }

    auto it = mLightIdToInstanceId.find(lightId);
    if (desiredMeshId != kInvalidIndex)
    {
        const glm::float4x4 transform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);
        if (it == mLightIdToInstanceId.end())
        {
            const uint32_t instanceId =
                createInstance(Instance::Type::eLight, desiredMeshId, kInvalidIndex, transform, lightId);
            mLightIdToInstanceId[lightId] = instanceId;
            markChanged(ChangeBits::Geometry);
        }
        else
        {
            Instance& instance = mInstances[it->second];
            if (instance.mMeshId != desiredMeshId)
            {
                instance.mMeshId = desiredMeshId;
                markChanged(ChangeBits::Geometry);
            }
            updateInstanceTransform(it->second, transform);
        }
    }
    markChanged(ChangeBits::Lights | ChangeBits::Transforms);
}

void Scene::setNodeLocalTransform(const uint32_t nodeId,
                                  const glm::float3& translation,
                                  const glm::quat& rotation,
                                  const glm::float3& scale)
{
    assert(nodeId < mNodes.size());
    mNodes[nodeId].translation = translation;
    mNodes[nodeId].rotation = rotation;
    mNodes[nodeId].scale = scale;

    ensureGlobalTransforms();
    mNodeDirty.assign(mNodes.size(), 0);
    mNodeDirty[nodeId] = 1;
    // Mark entire subtree dirty so side effects propagate
    std::vector<uint32_t> stack;
    stack.push_back(nodeId);
    while (!stack.empty())
    {
        const uint32_t id = stack.back();
        stack.pop_back();
        mNodeDirty[id] = 1;
        for (const int child : mNodes[id].children)
        {
            if (child >= 0 && (size_t)child < mNodes.size())
                stack.push_back((uint32_t)child);
        }
    }
    refreshGlobalTransforms();
    for (size_t i = 0; i < mNodes.size(); ++i)
    {
        if (mNodeDirty[i])
            applyNodeSideEffects((uint32_t)i);
    }
    markChanged(ChangeBits::Transforms);
}

void Scene::setMaterial(const uint32_t id, const MaterialDescription& desc)
{
    assert(id < mMaterialsDescs.size());
    mMaterialsDescs[id] = desc;
    markChanged(ChangeBits::Materials);
}

namespace
{
bool intersectTriangle(const glm::float3& orig,
                       const glm::float3& dir,
                       const glm::float3& v0,
                       const glm::float3& v1,
                       const glm::float3& v2,
                       float& tOut)
{
    const glm::float3 e1 = v1 - v0;
    const glm::float3 e2 = v2 - v0;
    const glm::float3 pvec = glm::cross(dir, e2);
    const float det = glm::dot(e1, pvec);
    if (std::fabs(det) < 1e-8f)
        return false;
    const float invDet = 1.0f / det;
    const glm::float3 tvec = orig - v0;
    // Barycentric bounds are tested with a tolerance: a ray through a point on an
    // edge shared by two triangles would otherwise be rejected by both, so
    // clicking along an interior edge of a mesh selects whatever is behind it.
    // Counting such a hit twice is harmless, the closest one wins either way.
    constexpr float kEdgeTolerance = 1e-6f;
    const float u = glm::dot(tvec, pvec) * invDet;
    if (u < -kEdgeTolerance || u > 1.0f + kEdgeTolerance)
        return false;
    const glm::float3 qvec = glm::cross(tvec, e1);
    const float v = glm::dot(dir, qvec) * invDet;
    if (v < -kEdgeTolerance || u + v > 1.0f + kEdgeTolerance)
        return false;
    const float t = glm::dot(e2, qvec) * invDet;
    if (t < 1e-5f)
        return false;
    tOut = t;
    return true;
}
} // namespace

int Scene::findInstanceNodeId(const uint32_t instId) const
{
    for (uint32_t n = 0; n < mNodes.size(); ++n)
    {
        const std::vector<uint32_t>& ids = mNodes[n].instanceIds;
        if (std::ranges::find(ids, instId) != ids.end())
        {
            return (int)n;
        }
    }
    return -1;
}

std::vector<glm::mat4> Scene::buildJointPalette(const uint32_t instId)
{
    if (instId >= mInstances.size())
    {
        return {};
    }
    const Instance& inst = mInstances[instId];
    if (inst.mMeshId >= mMeshes.size() || !mMeshes[inst.mMeshId].isSkeletal)
    {
        return {};
    }
    const int nodeId = findInstanceNodeId(instId);
    if (nodeId < 0 || mNodes[nodeId].skin < 0 || (size_t)mNodes[nodeId].skin >= mSkines.size())
    {
        return {};
    }
    const uint32_t skinId = (uint32_t)mNodes[nodeId].skin;
    std::vector<glm::mat4> palette;
    computeJointMatrices(&palette, mSkines[skinId].joints.size(), skinId);
    return palette;
}

bool Scene::vertexSkinMatrix(const Mesh& mesh,
                             const uint32_t vertexIndex,
                             const std::vector<glm::mat4>& jointPalette,
                             glm::mat4& outMat) const
{
    if (!mesh.isSkeletal || jointPalette.empty())
    {
        return false;
    }
    const size_t skinIndex = (size_t)mesh.mSbOffset + vertexIndex;
    if (skinIndex >= mVerticesSkinData.size())
    {
        return false;
    }

    const vertexSkinData& skinData = mVerticesSkinData[skinIndex];
    glm::mat4 skinMat(0.0f);
    for (int j = 0; j < 4; ++j)
    {
        const int joint = skinData.joints[j];
        if (skinData.weights[j] == 0.0f || joint < 0 || (size_t)joint >= jointPalette.size())
        {
            continue;
        }
        skinMat += skinData.weights[j] * jointPalette[joint];
    }
    // The bottom right element accumulates the weights, so a zero there means the
    // vertex is bound to no joint and its stored position already is final.
    if (skinMat[3][3] < 1e-6f)
    {
        return false;
    }
    outMat = skinMat;
    return true;
}

glm::float3 Scene::posedVertexPosition(const Mesh& mesh,
                                       const uint32_t vertexIndex,
                                       const std::vector<glm::mat4>& jointPalette) const
{
    glm::mat4 skinMat(0.0f);
    if (!vertexSkinMatrix(mesh, vertexIndex, jointPalette, skinMat))
    {
        return mVertices[mesh.mVbOffset + vertexIndex].pos;
    }
    return { skinMat * glm::float4(mVerticesSkinData[mesh.mSbOffset + vertexIndex].pos, 1.0f) };
}

bool Scene::meshBounds(const uint32_t meshId, glm::float3& outMin, glm::float3& outMax)
{
    if (mHostGeometryReleased || meshId >= mMeshes.size())
    {
        return false;
    }
    const Mesh& mesh = mMeshes[meshId];
    if (mesh.mVertexCount == 0 || mesh.isSkeletal)
    {
        return false;
    }
    if (mMeshBounds.size() != mMeshes.size())
    {
        mMeshBounds.resize(mMeshes.size());
    }
    MeshBounds& cached = mMeshBounds[meshId];
    if (!cached.valid)
    {
        // No palette: a mesh that is not skeletal has no pose, so the rest
        // position is the only position it has.
        const std::vector<glm::mat4> noPalette;
        cached.min = glm::float3(std::numeric_limits<float>::max());
        cached.max = glm::float3(std::numeric_limits<float>::lowest());
        for (uint32_t i = 0; i < mesh.mVertexCount; ++i)
        {
            const glm::float3 p = posedVertexPosition(mesh, i, noPalette);
            cached.min = glm::min(cached.min, p);
            cached.max = glm::max(cached.max, p);
        }
        cached.valid = true;
    }
    outMin = cached.min;
    outMax = cached.max;
    return true;
}

bool Scene::computeInstanceBounds(const uint32_t instId, glm::float3& outMin, glm::float3& outMax)
{
    if (instId >= mInstances.size())
    {
        return false;
    }
    const Instance& inst = mInstances[instId];
    if (inst.mMeshId >= mMeshes.size())
    {
        return false;
    }
    const Mesh& mesh = mMeshes[inst.mMeshId];
    if (mesh.mVertexCount == 0)
    {
        return false;
    }
    // The common case: bounds that do not depend on the instance, answered from
    // the cache instead of by walking the mesh again.
    if (!mesh.isSkeletal)
    {
        return meshBounds(inst.mMeshId, outMin, outMax);
    }

    const std::vector<glm::mat4> palette = buildJointPalette(instId);
    outMin = glm::float3(std::numeric_limits<float>::max());
    outMax = glm::float3(std::numeric_limits<float>::lowest());
    for (uint32_t i = 0; i < mesh.mVertexCount; ++i)
    {
        const glm::float3 p = posedVertexPosition(mesh, i, palette);
        outMin = glm::min(outMin, p);
        outMax = glm::max(outMax, p);
    }
    return true;
}

void Scene::ensureInstanceWorldBounds()
{
    if (mInstanceWorldBounds.size() == mInstances.size() && mInstanceBoundsGeneration == mTransformGeneration)
    {
        return;
    }
    // Built from the eight transformed corners of the mesh box, which is
    // conservative -- looser than the oriented box, never tighter, so it cannot
    // reject something the triangles would have hit.
    mInstanceWorldBounds.assign(mInstances.size(), MeshBounds{});
    for (uint32_t instId = 0; instId < mInstances.size(); ++instId)
    {
        const Instance& inst = mInstances[instId];
        glm::float3 lo(0.0f), hi(0.0f);
        if (!meshBounds(inst.mMeshId, lo, hi))
        {
            continue;
        }
        MeshBounds& wb = mInstanceWorldBounds[instId];
        wb.min = glm::float3(std::numeric_limits<float>::max());
        wb.max = glm::float3(std::numeric_limits<float>::lowest());
        for (int c = 0; c < 8; ++c)
        {
            const glm::float3 corner((c & 1) ? hi.x : lo.x, (c & 2) ? hi.y : lo.y, (c & 4) ? hi.z : lo.z);
            const glm::float3 w = glm::float3(inst.transform * glm::float4(corner, 1.0f));
            wb.min = glm::min(wb.min, w);
            wb.max = glm::max(wb.max, w);
        }
        wb.valid = true;
    }
    mInstanceBoundsGeneration = mTransformGeneration;
}

bool Scene::worldBounds(glm::float3& outMin, glm::float3& outMax)
{
    ensureInstanceWorldBounds();
    bool any = false;
    glm::float3 lo(std::numeric_limits<float>::max());
    glm::float3 hi(std::numeric_limits<float>::lowest());
    for (const MeshBounds& wb : mInstanceWorldBounds)
    {
        if (!wb.valid)
        {
            continue;
        }
        lo = glm::min(lo, wb.min);
        hi = glm::max(hi, wb.max);
        any = true;
    }
    if (!any)
    {
        return false;
    }
    outMin = lo;
    outMax = hi;
    return true;
}

Scene::PickHit Scene::pick(const glm::float3& origin, const glm::float3& direction)
{
    // The arrays this walks can have been handed back to the OS; see
    // releaseHostGeometry(). Returning a miss is the honest answer -- the
    // alternative is reading a freed vector.
    if (mHostGeometryReleased)
    {
        return {};
    }

    const auto pickStart = std::chrono::steady_clock::now();
    size_t traversed = 0, trianglesTested = 0;

    PickHit best;
    best.hit = false;
    best.distance = std::numeric_limits<float>::max();

    const glm::float3 dir = glm::normalize(direction);

    // Slab test against a box. This is the whole reason picking is usable on a
    // scattered scene: without it every one of 1.1 million instances has all of
    // its triangles tested, and with it all but a handful stop at twelve
    // compares.
    auto missesBounds = [](const glm::float3& o, const glm::float3& d, const glm::float3& bbMin,
                           const glm::float3& bbMax, float maxT, float* tEnter = nullptr) {
        float tMin = 0.0f;
        float tMax = maxT;
        for (int a = 0; a < 3; ++a)
        {
            // A component of exactly zero would make this a nan rather than an
            // infinity, and nan compares false against everything, which would
            // let the box through instead of rejecting it.
            const float inv = 1.0f / (d[a] != 0.0f ? d[a] : 1e-20f);
            float t0 = (bbMin[a] - o[a]) * inv;
            float t1 = (bbMax[a] - o[a]) * inv;
            if (t0 > t1)
                std::swap(t0, t1);
            tMin = std::max(tMin, t0);
            tMax = std::min(tMax, t1);
            if (tMax < tMin)
                return true;
        }
        if (tEnter)
        {
            *tEnter = tMin;
        }
        return false;
    };

    // World boxes first, so the common rejection costs no matrix work at all.
    // Built from the eight transformed corners of the mesh box, which is
    // conservative -- looser than the oriented box, never tighter, so it cannot
    // reject something the triangles would have hit.
    ensureInstanceWorldBounds();

    // Candidates first, nearest box first.
    //
    // The box test throws out all but a handful, but that handful can still be
    // tens of millions of triangles -- a ground plane and a canopy are one mesh
    // each here. Testing them in instance order means the ray may walk the
    // furthest one before it has any hit distance to prune with; in entry-point
    // order the first hit usually makes every remaining candidate a single
    // compare.
    struct Candidate
    {
        uint32_t instId;
        float tEnter;
    };
    std::vector<Candidate> candidates;
    for (uint32_t instId = 0; instId < mInstances.size(); ++instId)
    {
        const Instance& inst = mInstances[instId];
        if (inst.type != Instance::Type::eMesh && inst.type != Instance::Type::eLight)
            continue;
        if (inst.mMeshId >= mMeshes.size())
            continue;

        const MeshBounds& wb = mInstanceWorldBounds[instId];
        float tEnter = 0.0f;
        if (wb.valid && missesBounds(origin, dir, wb.min, wb.max, std::numeric_limits<float>::max(), &tEnter))
        {
            continue;
        }
        candidates.push_back({ instId, tEnter });
    }
    std::ranges::sort(candidates, [](const Candidate& a, const Candidate& b) { return a.tEnter < b.tEnter; });

    for (const Candidate& candidate : candidates)
    {
        // Everything left starts beyond the closest hit found so far, and the
        // list is sorted, so nothing after this can win either.
        if (candidate.tEnter >= best.distance)
        {
            break;
        }
        const uint32_t instId = candidate.instId;
        const Instance& inst = mInstances[instId];
        ++traversed;

        if (inst.type == Instance::Type::eLight && inst.mLightId < mLights.size())
        {
            const Light& light = mLights[inst.mLightId];
            AnalyticLightIntersection analyticHit{};
            if (light.type == LIGHT_TYPE_DISC)
            {
                analyticHit = intersectAnalyticDisc(origin, dir, 1e-5f, best.distance, glm::float3(light.points[1]),
                                                    glm::float3(light.points[2]), glm::float3(light.points[3]),
                                                    glm::float3(light.normal));
            }
            else if (light.type == LIGHT_TYPE_SPHERE)
            {
                analyticHit = intersectAnalyticEllipsoid(origin, dir, 1e-5f, best.distance,
                                                         glm::float3(light.points[1]), glm::float3(light.points[0]),
                                                         glm::float3(light.points[2]), glm::float3(light.points[3]));
            }
            if (light.type == LIGHT_TYPE_DISC || light.type == LIGHT_TYPE_SPHERE)
            {
                if (analyticHit.hit)
                {
                    best.hit = true;
                    best.distance = analyticHit.distance;
                    best.position = analyticHit.point;
                    best.instanceId = instId;
                    best.lightId = inst.mLightId;
                }
                continue;
            }
        }

        const Mesh& mesh = mMeshes[inst.mMeshId];
        const glm::mat4& xform = inst.transform;
        const glm::mat4 invXform = glm::inverse(xform);

        const glm::float3 localOrig = glm::float3(invXform * glm::float4(origin, 1.0f));
        const glm::float3 localDir = glm::normalize(glm::float3(invXform * glm::float4(dir, 0.0f)));

        // Skinning happens in the same space the instance transform maps to
        // world, so only the vertex positions have to be posed: the ray stays in
        // instance space. Only skeletal meshes have a pose to build; for the rest
        // this was an allocation per instance for an empty vector.
        const std::vector<glm::mat4> palette = mesh.isSkeletal ? buildJointPalette(instId) : std::vector<glm::mat4>();

        trianglesTested += mesh.mCount / 3;
        for (uint32_t i = 0; i + 2 < mesh.mCount; i += 3)
        {
            const uint32_t i0 = mIndices[mesh.mIndex + i];
            const uint32_t i1 = mIndices[mesh.mIndex + i + 1];
            const uint32_t i2 = mIndices[mesh.mIndex + i + 2];
            const glm::float3 v0 = posedVertexPosition(mesh, i0, palette);
            const glm::float3 v1 = posedVertexPosition(mesh, i1, palette);
            const glm::float3 v2 = posedVertexPosition(mesh, i2, palette);

            float tLocal = 0.0f;
            if (!intersectTriangle(localOrig, localDir, v0, v1, v2, tLocal))
                continue;

            const glm::float3 localHit = localOrig + localDir * tLocal;
            const glm::float3 worldHit = glm::float3(xform * glm::float4(localHit, 1.0f));
            const float tWorld = glm::length(worldHit - origin);
            if (tWorld >= best.distance)
                continue;

            best.hit = true;
            best.distance = tWorld;
            best.position = worldHit;
            best.instanceId = instId;
            best.lightId = inst.mLightId;
        }
    }

    // The owning node, resolved for the one instance that won rather than by
    // building a reverse map of every instance in the scene on every click.
    best.nodeId = kInvalidIndex;
    if (best.hit)
    {
        for (uint32_t n = 0; n < mNodes.size() && best.nodeId == kInvalidIndex; ++n)
        {
            for (const uint32_t instId : mNodes[n].instanceIds)
            {
                if (instId == best.instanceId)
                {
                    best.nodeId = n;
                    break;
                }
            }
        }
    }
    STRELKA_DEBUG("Pick: {} instances, {} boxes hit, {} traversed, {} triangles, {:.1f} ms ({})", mInstances.size(),
                  candidates.size(), traversed, trianglesTested,
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pickStart).count(),
                  best.hit ? "hit" : "miss");
    return best;
}

void Scene::removeInstance(const uint32_t instId)
{
    mDelInstances.push(instId); // marked as removed
}

void Scene::removeMesh(const uint32_t meshId)
{
    mDelMesh.push(meshId); // marked as removed
}

void Scene::removeMaterial(const uint32_t materialId)
{
    mDelMaterial.push(materialId); // marked as removed
}

void Scene::updateInstanceTransform(uint32_t instId, glm::float4x4 newTransform)
{
    Instance& inst = mInstances[instId];
    inst.transform = newTransform;
    mDirtyInstances.insert(instId);
    ++mTransformGeneration;
    markChanged(ChangeBits::Transforms);
}

uint32_t Scene::createCurve(const Curve::Type type,
                            const std::vector<uint32_t>& vertexCounts,
                            const std::vector<glm::float3>& points,
                            const std::vector<float>& widths)
{
    Curve c = {};
    c.mType = type;
    c.mPointsStart = mCurvePoints.size();
    c.mPointsCount = points.size();
    mCurvePoints.insert(mCurvePoints.end(), points.begin(), points.end());
    c.mVertexCountsStart = mCurveVertexCounts.size();
    c.mVertexCountsCount = vertexCounts.size();
    mCurveVertexCounts.insert(mCurveVertexCounts.end(), vertexCounts.begin(), vertexCounts.end());
    if (!widths.empty())
    {
        c.mWidthsCount = widths.size();
        c.mWidthsStart = mCurveWidths.size();
        mCurveWidths.insert(mCurveWidths.end(), widths.begin(), widths.end());
    }
    else
    {
        c.mWidthsCount = -1;
        c.mWidthsStart = -1;
    }
    const uint32_t res = mCurves.size();
    mCurves.push_back(c);
    return res;
}


} // namespace oka
