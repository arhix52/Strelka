#include <strelka/scene/scene.h>
#include <strelka/scene/vertex_packing.h>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <algorithm>
#include <filesystem>

#include <log.h>

namespace fs = std::filesystem;

namespace oka
{

uint32_t Scene::createMesh(const std::vector<Vertex>& vb, const std::vector<uint32_t>& ib)
{
    std::scoped_lock lock(mMeshMutex);

    Mesh* mesh = nullptr;
    uint32_t meshId = -1;
    if (mDelMesh.empty())
    {
        meshId = mMeshes.size(); // add mesh to storage
        mMeshes.push_back({});
        mesh = &mMeshes.back();
    }
    else
    {
        meshId = mDelMesh.top(); // get index from stack
        mDelMesh.pop(); // del taken index from stack
        mesh = &mMeshes[meshId];
    }

    mesh->mIndex = mIndices.size(); // Index of 1st index in index buffer
    mesh->mCount = ib.size(); // amount of indices in mesh

    mesh->mVbOffset = mVertices.size();
    mesh->mVertexCount = vb.size();

    // const uint32_t ibOffset = mVertices.size(); // adjust indices for global index buffer
    // for (int i = 0; i < ib.size(); ++i)
    // {
    //     mIndices.push_back(ibOffset + ib[i]);
    // }
    mIndices.insert(mIndices.end(), ib.begin(), ib.end());
    mVertices.insert(mVertices.end(), vb.begin(), vb.end()); // copy vertices
    return meshId;
}

uint32_t Scene::createSkeletalMesh(const std::vector<Vertex>& vb, const std::vector<uint32_t>& ib, const std::vector<oka::Scene::vertexSkinData>& sb)
{
    std::scoped_lock lock(mMeshMutex);

    Mesh* mesh = nullptr;
    uint32_t meshId = -1;
    if (mDelMesh.empty())
    {
        meshId = mMeshes.size(); // add mesh to storage
        mMeshes.push_back({});
        mesh = &mMeshes.back();
    }
    else
    {
        meshId = mDelMesh.top(); // get index from stack
        mDelMesh.pop(); // del taken index from stack
        mesh = &mMeshes[meshId];
    }

    mesh->mIndex = mIndices.size(); // Index of 1st index in index buffer
    mesh->mCount = ib.size(); // amount of indices in mesh

    mesh->mVbOffset = mVertices.size();
    mesh->mVertexCount = vb.size();

    mesh->mSbOffset = mVerticesSkinData.size();
    mesh->isSkeletal = true;

    // const uint32_t ibOffset = mVertices.size(); // adjust indices for global index buffer
    // for (int i = 0; i < ib.size(); ++i)
    // {
    //     mIndices.push_back(ibOffset + ib[i]);
    // }
    mIndices.insert(mIndices.end(), ib.begin(), ib.end());
    mVertices.insert(mVertices.end(), vb.begin(), vb.end()); // copy vertices
    mVerticesSkinData.insert(mVerticesSkinData.end(), sb.begin(), sb.end());
    return meshId;
}

uint32_t Scene::createInstance(const Instance::Type type,
                               const uint32_t geomId,
                               const uint32_t materialId,
                               const glm::mat4& transform,
                               const uint32_t lightId)
{
    std::scoped_lock lock(mInstanceMutex);

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

    mOpaqueInstances.push_back(instId);

    return instId;
}

uint32_t Scene::addMaterial(const MaterialDescription& material)
{
    // TODO: fix here
    uint32_t res = mMaterialsDescs.size();
    mMaterialsDescs.push_back(material);
    return res;
}

std::string Scene::getSceneFileName()
{
    fs::path p(modelPath);
    return p.filename().string();
};

std::string Scene::getSceneDir()
{
    fs::path p(modelPath);
    return p.parent_path().string();
}

glm::quat Scene::makeQuatFromFloat4(const glm::float4 &value)
{
    const float floatRotation[4] = {
                value[3],
                value[0],
                value[1],
                value[2],
            };
    return glm::make_quat(floatRotation);
}

glm::float4 Scene::makeFloat4FromQuat(const glm::quat &q)
{
    return glm::float4(q.x, q.y, q.z, q.w);
}

// packNormal() provided by <strelka/scene/vertex_packing.h>

glm::float4 Scene::interpolate(const AnimationSampler &sampler, const AnimationChannel::PathType targetProperty, const float time)
{
    const bool isCubic = (sampler.interpolation == AnimationSampler::InterpolationType::CUBICSPLINE);
    // For CUBICSPLINE, outputsVec4 stores triplets: [inTangent, value, outTangent] per keyframe.
    // For LINEAR/STEP, outputsVec4 stores one value per keyframe.
    const int stride = isCubic ? 3 : 1;
    const int valueOffset = isCubic ? 1 : 0;

    const int n = (int)sampler.inputs.size();
    if (n == 0) return glm::float4(0.0f);

    // Clamp to range
    if (time <= sampler.inputs[0])
        return sampler.outputsVec4[valueOffset];
    if (time >= sampler.inputs[n - 1])
        return sampler.outputsVec4[(n - 1) * stride + valueOffset];

    // Find bracket: inputs[prevIdx] <= time < inputs[nextIdx].
    // inputs is sorted by construction, so binary search it — the previous linear
    // scan cost O(keyframes) per channel per frame (BrainStem has channels with
    // 838 keys, evaluated 116 times per pass and twice per frame).
    const auto upper = std::upper_bound(sampler.inputs.begin(), sampler.inputs.end(), time);
    int nextIdx = (int)std::distance(sampler.inputs.begin(), upper);
    nextIdx = std::clamp(nextIdx, 1, n - 1);
    const int prevIdx = nextIdx - 1;

    float previousTime = sampler.inputs[prevIdx];
    float nextTime = sampler.inputs[nextIdx];

    // Exact match — return value directly
    if (std::abs(time - previousTime) < 1e-7f)
        return sampler.outputsVec4[prevIdx * stride + valueOffset];

    glm::float4 result;

    switch (sampler.interpolation)
    {
    case AnimationSampler::InterpolationType::STEP:
        result = sampler.outputsVec4[prevIdx * stride + valueOffset];
        break;

    case AnimationSampler::InterpolationType::CUBICSPLINE:
    {
        // glTF cubic spline: Hermite interpolation
        // outputsVec4 layout per keyframe: [inTangent, value, outTangent]
        float deltaTime = nextTime - previousTime;
        float t = (time - previousTime) / deltaTime;
        float t2 = t * t;
        float t3 = t2 * t;

        glm::float4 p0 = sampler.outputsVec4[prevIdx * 3 + 1]; // value at prev
        glm::float4 m0 = sampler.outputsVec4[prevIdx * 3 + 2] * deltaTime; // out-tangent at prev
        glm::float4 p1 = sampler.outputsVec4[nextIdx * 3 + 1]; // value at next
        glm::float4 m1 = sampler.outputsVec4[nextIdx * 3 + 0] * deltaTime; // in-tangent at next

        result = (2.0f * t3 - 3.0f * t2 + 1.0f) * p0
               + (t3 - 2.0f * t2 + t) * m0
               + (-2.0f * t3 + 3.0f * t2) * p1
               + (t3 - t2) * m1;

        if (targetProperty == AnimationChannel::PathType::ROTATION)
            result = makeFloat4FromQuat(glm::normalize(makeQuatFromFloat4(result)));
        break;
    }

    default: // LINEAR
    {
        float interpolationValue = (time - previousTime) / (nextTime - previousTime);
        glm::float4 prevVal = sampler.outputsVec4[prevIdx];
        glm::float4 nextVal = sampler.outputsVec4[nextIdx];
        if (targetProperty != AnimationChannel::PathType::ROTATION)
            result = glm::lerp(prevVal, nextVal, interpolationValue);
        else
            result = makeFloat4FromQuat(glm::slerp(makeQuatFromFloat4(prevVal), makeQuatFromFloat4(nextVal), interpolationValue));
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
            if (childId >= 0 && childId < (int)mNodes.size())
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
        mGlobalTransforms[nodeId] =
            (parent == -1) ? local : mGlobalTransforms[parent] * local;
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
    // Mirrors the traversal the old updateNode() performed: mesh and camera nodes
    // consume the update and do not propagate it to their children.
    switch (mNodes[nodeId].type)
    {
    case Node::NodeType::mesh:
        for (const auto instId : mNodes[nodeId].instanceIds)
        {
            Instance& inst = mInstances[instId];
            inst.transform = mGlobalTransforms[nodeId];
            inst.isAnimated = true;
        }
        return false;

    case Node::NodeType::camera:
        if (mNodes[nodeId].camera >= 0 && mNodes[nodeId].camera < (int)mCameras.size())
        {
            glm::vec3 scale;
            glm::quat rotation;
            glm::vec3 translation;
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(mGlobalTransforms[nodeId], scale, rotation, translation, skew, perspective);
            rotation = glm::conjugate(rotation);

            Camera& cam = mCameras[mNodes[nodeId].camera];
            cam.position = translation * scale;
            cam.mOrientation = rotation;
            cam.updateViewMatrix();
        }
        return false;

    case Node::NodeType::skeleton:
        break;

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

    // Two phases instead of one update per channel. The old code called
    // updateNode() for every channel, and updateNode() recomputed each visited
    // node's world transform by walking back up to the root — so a scene with C
    // channels cost O(C * subtree * depth) matrix builds every frame, re-deriving
    // the same ancestors again and again. BrainStem has 116 channels over a
    // 30-node graph and paid that twice per frame (motion blur is a two-pass
    // evaluation), which is what made playback stutter on a 34k-triangle scene.
    //
    // Phase 1 only writes local TRS; phase 2 derives every world transform in a
    // single parent-before-child sweep.
    auto& animation = mAnimations[animId];
    std::fill(mNodeDirty.begin(), mNodeDirty.end(), 0);

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
    return blasChanged;
}

void Scene::applySkinning()
{
    for (auto& node: mNodes)
    {
        if (node.skin != -1 && node.type == Node::NodeType::mesh)
        {
            auto jointCount = mSkines[node.skin].joints.size();
            std::vector<glm::mat4> jointMat;
            computeJointMatrices(&jointMat, jointCount, node.skin);
            for (const auto instId: node.instanceIds) {
                auto &mesh = mMeshes[mInstances[instId].mMeshId];
                int vbOffset = mesh.mVbOffset;
                int sbOffset = mesh.mSbOffset;
                for (int iv = 0; iv < mesh.mVertexCount; ++iv)
                {
                    glm::vec4 v_weight = mVerticesSkinData[sbOffset + iv].weights;
                    glm::u16vec4 v_joint = mVerticesSkinData[sbOffset + iv].joints;
                    glm::mat4 skinMat = v_weight[0] * jointMat[v_joint[0]]
                                      + v_weight[1] * jointMat[v_joint[1]]
                                      + v_weight[2] * jointMat[v_joint[2]]
                                      + v_weight[3] * jointMat[v_joint[3]];
                    mVertices[vbOffset + iv].pos = skinMat * glm::vec4(mVerticesSkinData[sbOffset + iv].pos, 1.0);
                    mVertices[vbOffset + iv].normal = packNormal(glm::normalize(glm::vec3(glm::mat3(skinMat) * glm::vec4(mVerticesSkinData[sbOffset + iv].normal, 1.0))));
                }
            }
        }
    }
}

void Scene::computeJointMatrices(std::vector<glm::mat4> *jointMatrices, int jointCount, const uint32_t skinId)
{
    ensureGlobalTransforms();

    auto &skin = mSkines[skinId];
    jointMatrices->reserve(jointMatrices->size() + jointCount);
    for (int i = 0; i < jointCount; ++i)
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
    int parentId = mNodes[nodeId].parent;
    if (parentId == -1) {
        return calculateNodeLocalTransform(nodeId);
    }
    else {
        return calculateNodeGlobalTransform(parentId) * calculateNodeLocalTransform(nodeId);
    }
}

bool Scene::animateNode(const uint32_t nodeId, AnimationChannel::PathType targetProperty, const glm::float3 newValue)
{
    switch (targetProperty)
    {
    case AnimationChannel::PathType::TRANSLATION:
        mNodes[nodeId].translation = newValue;
        return updateNode(nodeId);
        break;

    case AnimationChannel::PathType::SCALE:
        mNodes[nodeId].scale = newValue;
        return updateNode(nodeId);
        break;
    
    case AnimationChannel::PathType::ROTATION:
        STRELKA_DEBUG("Invalid value to animate ROTATION, use 2nd definition");
        break;

    default:
        break;
    }
    return false;
}

bool Scene::animateNode(const uint32_t nodeId, AnimationChannel::PathType targetProperty, const glm::quat newValue) 
{
    switch (targetProperty)
    {
    case AnimationChannel::PathType::TRANSLATION:
        STRELKA_DEBUG("Invalid value to animate TRANSLATION, use 1st definition");

    case AnimationChannel::PathType::SCALE:
        STRELKA_DEBUG("Invalid value to animate SCALE, use 1st definition");
        break;
    
    case AnimationChannel::PathType::ROTATION:
        mNodes[nodeId].rotation = newValue;
        return updateNode(nodeId);
        break;

    default:
        break;
    }
    return false;
}

bool Scene::updateNode(const uint32_t nodeId)
{
    bool skeletonNodesUpdated = false;
    const glm::float4x4 globalTransform = calculateNodeGlobalTransform(nodeId);

    // if this node is mesh node - updating Instances
    // if this node is skeleton node - need to rebuild blas
    switch (mNodes[nodeId].type)
    {
        case Node::NodeType::mesh:
            for (const auto instId: mNodes[nodeId].instanceIds) {
                Instance& inst = mInstances[instId];
                inst.transform = globalTransform;
                inst.isAnimated = true;
            }
            return false;
            break;

        case Node::NodeType::camera:
            if (mNodes[nodeId].camera >= 0 && mNodes[nodeId].camera < (int)mCameras.size())
            {
                glm::vec3 scale;
                glm::quat rotation;
                glm::vec3 translation;
                glm::vec3 skew;
                glm::vec4 perspective;
                glm::decompose(globalTransform, scale, rotation, translation, skew, perspective);
                rotation = glm::conjugate(rotation);

                Camera& cam = mCameras[mNodes[nodeId].camera];
                cam.position = translation * scale;
                cam.mOrientation = rotation;
                cam.updateViewMatrix();
            }
            return false;

        case Node::NodeType::skeleton:
            skeletonNodesUpdated = true;
            break;

        default:
            break;
    }

    for (const auto childId: mNodes[nodeId].children) 
    {
        skeletonNodesUpdated |= updateNode(childId);
    }

    return skeletonNodesUpdated;
}

uint32_t Scene::createRectLightMesh()
{
    if (mRectLightMeshId != -1)
    {
        return mRectLightMeshId;
    }

    std::vector<Scene::Vertex> vb;
    Scene::Vertex v1, v2, v3, v4;
    v1.pos = glm::float4(0.5f, 0.5f, 0.0f, 1.0f); // top right 0
    v2.pos = glm::float4(-0.5f, 0.5f, 0.0f, 1.0f); // top left 1
    v3.pos = glm::float4(-0.5f, -0.5f, 0.0f, 1.0f); // bottom left 2
    v4.pos = glm::float4(0.5f, -0.5f, 0.0f, 1.0f); // bottom right 3
    glm::float3 normal = glm::float3(0.f, 0.f, 1.f);
    v1.normal = v2.normal = v3.normal = v4.normal = packNormal(normal);
    std::vector<uint32_t> ib = { 0, 1, 2, 2, 3, 0 };
    vb.push_back(v1);
    vb.push_back(v2);
    vb.push_back(v3);
    vb.push_back(v4);

    uint32_t meshId = createMesh(vb, ib);
    assert(meshId != -1);

    return meshId;
}

uint32_t Scene::createSphereLightMesh()
{
    if (mSphereLightMeshId != -1)
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
        float theta = static_cast<float>(i) * static_cast<float>(M_PI) / static_cast<float>(rings);
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int j = 0; j <= segments; ++j)
        {
            float phi = static_cast<float>(j) * 2.0f * static_cast<float>(M_PI) / static_cast<float>(segments);
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            float x = cosPhi * sinTheta;
            float y = cosTheta;
            float z = sinPhi * sinTheta;

            glm::float3 pos = { radius * x, radius * y, radius * z };
            glm::float3 normal = { x, y, z };

            vertices.push_back(Scene::Vertex{ pos, 0, packNormal(normal) });
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
    assert(meshId != -1);

    return meshId;
}

uint32_t Scene::createDiscLightMesh()
{
    if (mDiskLightMeshId != -1)
    {
        return mDiskLightMeshId;
    }

    std::vector<Scene::Vertex> vertices;
    std::vector<uint32_t> indices;

    Scene::Vertex v1, v2;
    v1.pos = glm::float4(0.f, 0.f, 0.f, 1.f);
    v2.pos = glm::float4(1.0f, 0.f, 0.f, 1.f);

    glm::float3 normal = glm::float3(0.f, 0.f, 1.f);
    v1.normal = v2.normal = packNormal(normal);

    vertices.push_back(v1); // central point
    vertices.push_back(v2); // first point

    const float diskRadius = 1.0f; // param
    const float step = 2.0f * M_PI / 16;
    float angle = 0;
    for (int i = 0; i < 16; ++i)
    {
        indices.push_back(0); // each triangle have central point
        indices.push_back(vertices.size() - 1); // prev vertex

        angle += step;
        const float x = cos(angle) * diskRadius;
        const float y = sin(angle) * diskRadius;

        Scene::Vertex v;
        v.pos = glm::float4(x, y, 0.0f, 1.0f);
        v.normal = packNormal(normal);
        vertices.push_back(v);

        indices.push_back(vertices.size() - 1); // added vertex
    }

    uint32_t meshId = createMesh(vertices, indices);
    assert(meshId != -1);

    return meshId;
}

void Scene::updateAnimation(const float time)
{
    if (mAnimations.empty())
    {
        return;
    }
    auto& animation = mAnimations[0];
    for (auto& channel : animation.channels)
    {
        assert(channel.node < mNodes.size());
        auto& sampler = animation.samplers[channel.samplerIndex];
        if (sampler.inputs.size() > sampler.outputsVec4.size())
        {
            continue;
        }
        for (size_t i = 0; i < sampler.inputs.size() - 1; i++)
        {
            if ((time >= sampler.inputs[i]) && (time <= sampler.inputs[i + 1]))
            {
                float u = std::max(0.0f, time - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
                if (u <= 1.0f)
                {
                    switch (channel.path)
                    {
                    case AnimationChannel::PathType::TRANSLATION: {
                        glm::vec4 trans = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                        mNodes[channel.node].translation = glm::float3(trans);
                        break;
                    }
                    case AnimationChannel::PathType::SCALE: {
                        glm::vec4 scale = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                        mNodes[channel.node].scale = glm::float3(scale);
                        break;
                    }
                    case AnimationChannel::PathType::ROTATION: {
                        float floatRotation[4] = { (float)sampler.outputsVec4[i][3], (float)sampler.outputsVec4[i][0],
                                                   (float)sampler.outputsVec4[i][1], (float)sampler.outputsVec4[i][2] };
                        float floatRotation1[4] = { (float)sampler.outputsVec4[i + 1][3],
                                                    (float)sampler.outputsVec4[i + 1][0],
                                                    (float)sampler.outputsVec4[i + 1][1],
                                                    (float)sampler.outputsVec4[i + 1][2] };
                        glm::quat q1 = glm::make_quat(floatRotation);
                        glm::quat q2 = glm::make_quat(floatRotation1);
                        mNodes[channel.node].rotation = glm::normalize(glm::slerp(q1, q2, u));
                        break;
                    }
                    }
                }
            }
        }
    }
    mCameras[0].matrices.view = getTransform(mCameras[0].node);
}

uint32_t Scene::createLight(const UniformLightDesc& desc)
{
    auto lightId = (uint32_t)mLights.size();
    Light l;
    mLights.push_back(l);
    mLightDesc.push_back(desc);

    updateLight(lightId, desc);

    // TODO: only for rect light
    // Lazy init light mesh
    glm::float4x4 scaleMatrix = glm::float4x4(0.f);
    uint32_t currentLightMeshId = 0;
    if (desc.type == 0)
    {
        mRectLightMeshId = createRectLightMesh();
        currentLightMeshId = mRectLightMeshId;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.width, desc.height, 1.0f));
    }
    else if (desc.type == 1)
    {
        mDiskLightMeshId = createDiscLightMesh();
        currentLightMeshId = mDiskLightMeshId;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.radius, desc.radius, desc.radius));
    }
    else if (desc.type == 2)
    {
        mSphereLightMeshId = createSphereLightMesh();
        currentLightMeshId = mSphereLightMeshId;
        scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.radius, desc.radius, desc.radius));
    }
    else if (desc.type == 3)
    {
        // distant light has no mesh so skip
        return lightId;
    }

    const glm::float4x4 transform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);
    uint32_t instId = createInstance(Instance::Type::eLight, currentLightMeshId, (uint32_t)-1, transform, lightId);
    assert(instId != -1);

    mLightIdToInstanceId[lightId] = instId;

    return lightId;
}

void Scene::updateLight(const uint32_t lightId, const UniformLightDesc& desc)
{
    const float intensityPerPoint = desc.intensity; // light intensity
    // transform to GPU light
    // Rect Light
    if (desc.type == 0)
    {
        const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(desc.width, desc.height, 1.0f));
        const glm::float4x4 localTransform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);

        mLights[lightId].points[0] = localTransform * glm::float4(0.5f, 0.5f, 0.0f, 1.0f);
        mLights[lightId].points[1] = localTransform * glm::float4(-0.5f, 0.5f, 0.0f, 1.0f);
        mLights[lightId].points[2] = localTransform * glm::float4(-0.5f, -0.5f, 0.0f, 1.0f);
        mLights[lightId].points[3] = localTransform * glm::float4(0.5f, -0.5f, 0.0f, 1.0f);

        mLights[lightId].type = LIGHT_TYPE_RECT;
    }
    else if (desc.type == LIGHT_TYPE_DISC)
    {
        // Disk Light
        const glm::float4x4 scaleMatrix =
            glm::scale(glm::float4x4(1.0f), glm::float3(desc.radius, desc.radius, desc.radius));
        const glm::float4x4 localTransform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);

        mLights[lightId].points[0] = glm::float4(desc.radius, 0.f, 0.f, 0.f); // save radius
        mLights[lightId].points[1] = localTransform * glm::float4(0.f, 0.f, 0.f, 1.f); // save O
        mLights[lightId].points[2] = localTransform * glm::float4(1.f, 0.f, 0.f, 0.f); // OXws
        mLights[lightId].points[3] = localTransform * glm::float4(0.f, 1.f, 0.f, 0.f); // OYws

        glm::float4 normal = localTransform * glm::float4(0, 0, 1.f, 0.0f);
        mLights[lightId].normal = normal;
        mLights[lightId].type = LIGHT_TYPE_DISC;
    }
    else if (desc.type == LIGHT_TYPE_SPHERE)
    {
        // Sphere Light
        const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(1.0f, 1.0f, 1.0f));
        const glm::float4x4 localTransform = desc.useXform ? scaleMatrix * desc.xform : getTransform(desc);

        mLights[lightId].points[0] = glm::float4(desc.radius, 0.f, 0.f, 0.f); // save radius
        mLights[lightId].points[1] = localTransform * glm::float4(0.f, 0.f, 0.f, 1.f); // save O

        mLights[lightId].type = LIGHT_TYPE_SPHERE;
    }
    else if (desc.type == LIGHT_TYPE_DISTANT)
    {
        // distant light https://openusd.org/release/api/class_usd_lux_distant_light.html
        mLights[lightId].type = LIGHT_TYPE_DISTANT;
        mLights[lightId].halfAngle = desc.halfAngle;
        const glm::float4x4 scaleMatrix = glm::float4x4(1.0f);
        const glm::float4x4 localTransform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);
        mLights[lightId].normal = glm::normalize(localTransform * glm::float4(0.0f, 0.0f, -1.0f, 0.0f)); // -Z
    }

    mLights[lightId].color = glm::float4(desc.color, 1.0f) * intensityPerPoint;
    mDirty = DirtyFlag::eLights;
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

std::vector<uint32_t>& Scene::getOpaqueInstancesToRender(const glm::float3& camPos)
{
    return mOpaqueInstances;
}

std::vector<uint32_t>& Scene::getTransparentInstancesToRender(const glm::float3& camPos)
{
    return mTransparentInstances;
}

std::set<uint32_t> Scene::getDirtyInstances()
{
    return this->mDirtyInstances;
}

void Scene::updateInstanceTransform(uint32_t instId, glm::float4x4 newTransform)
{
    Instance& inst = mInstances[instId];
    inst.transform = newTransform;
    mDirtyInstances.insert(instId);
}

uint32_t Scene::createCurve(const Curve::Type type,
                            const std::vector<uint32_t>& vertexCounts,
                            const std::vector<glm::float3>& points,
                            const std::vector<float>& widths)
{
    Curve c = {};
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
    uint32_t res = mCurves.size();
    mCurves.push_back(c);
    return res;
}


} // namespace oka
