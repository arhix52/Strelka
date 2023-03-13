#include "scene.h"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <map>
#include <utility>

namespace fs = std::filesystem;

namespace oka
{
using Lookup = std::map<std::pair<uint32_t, uint32_t>, uint32_t>;
using IndexedMesh = std::pair<std::vector<Scene::Vertex>, std::vector<uint32_t>>;

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
    if (inst->type == Instance::Type::eMesh)
    {
        inst->mMeshId = geomId;
    }
    else
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
};

//  valid range of coordinates [-1; 1]
uint32_t packNormals(const glm::float3& normal)
{
    uint32_t packed = (uint32_t)((normal.x + 1.0f) / 2.0f * 511.99999f);
    packed += (uint32_t)((normal.y + 1.0f) / 2.0f * 511.99999f) << 10;
    packed += (uint32_t)((normal.z + 1.0f) / 2.0f * 511.99999f) << 20;
    return packed;
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
    v1.normal = v2.normal = v3.normal = v4.normal = packNormals(normal);
    std::vector<uint32_t> ib = { 0, 1, 2, 2, 3, 0 };
    vb.push_back(v1);
    vb.push_back(v2);
    vb.push_back(v3);
    vb.push_back(v4);

    uint32_t meshId = createMesh(vb, ib);
    assert(meshId != -1);

    return meshId;
}

uint32_t vertexForEdge(Lookup& lookup, std::vector<Scene::Vertex>& vertices, uint32_t first, uint32_t second)
{
    Lookup::key_type key(first, second);
    if (key.first > key.second)
    {
        std::swap(key.first, key.second);
    }

    auto inserted = lookup.insert({ key, vertices.size() });
    if (inserted.second)
    {
        auto& edge0 = vertices[first].pos;
        auto& edge1 = vertices[second].pos;
        auto point = normalize(glm::float3{ (edge0.x + edge1.x) / 2, (edge0.y + edge1.y) / 2, (edge0.z + edge1.z) / 2 });
        Scene::Vertex v;
        v.pos = point;
        vertices.push_back(v);
    }

    return inserted.first->second;
}

std::vector<uint32_t> subdivide(std::vector<Scene::Vertex>& vertices, std::vector<uint32_t>& indices)
{
    Lookup lookup;
    std::vector<uint32_t> result;

    for (uint32_t i = 0; i < indices.size(); i += 3)
    {
        std::array<uint32_t, 3> mid;
        for (int edge = 0; edge < 3; ++edge)
        {
            mid[edge] = vertexForEdge(lookup, vertices, indices[i + edge], indices[(i + (edge + 1) % 3)]);
        }

        result.push_back(indices[i]);
        result.push_back(mid[0]);
        result.push_back(mid[2]);

        result.push_back(indices[i + 1]);
        result.push_back(mid[1]);
        result.push_back(mid[0]);

        result.push_back(indices[i + 2]);
        result.push_back(mid[2]);
        result.push_back(mid[1]);

        result.push_back(mid[0]);
        result.push_back(mid[1]);
        result.push_back(mid[2]);
    }

    return result;
}

IndexedMesh subdivideIcosphere(int subdivisions, std::vector<Scene::Vertex>& _vertices, std::vector<uint32_t>& _indices)
{
    std::vector<Scene::Vertex> vertices = _vertices;
    std::vector<uint32_t> triangles = _indices;

    for (int i = 0; i < subdivisions; ++i)
    {
        triangles = subdivide(vertices, triangles);
    }

    return { vertices, triangles };
}

uint32_t Scene::createSphereLightMesh()
{
    if (mSphereLightMeshId != -1)
    {
        return mSphereLightMeshId;
    }

    std::vector<Scene::Vertex> vertices(12); // 12 vertices
    std::vector<uint32_t> indices;

    const float PI = acos(-1);
    const float H_ANGLE = PI / 180 * 72; // 72 degree = 360 / 5
    const float V_ANGLE = atanf(1.0f / 2); // elevation = 26.565 degree

    int i0 = 0, i1, i2, i3 = 11; // indices
    float z, xy; // coords
    float hAngle1 = -PI / 2 - H_ANGLE / 2; // start from -126 deg at 2nd row
    float hAngle2 = -PI / 2; // start from -90 deg at 3rd row

    // the first top vertex (0, 0, r)
    float radius = 1.0f;
    Scene::Vertex v0, v1, v2;
    v0.pos = { 0, 0, radius };

    vertices[i0] = v0;

    // 10 vertices at 2nd and 3rd rows
    for (int i = 1; i <= 4; ++i)
    {
        i1 = i; // 2nd row
        i2 = i + 5; // 3d row

        z = radius * sinf(V_ANGLE); // elevaton
        xy = radius * cosf(V_ANGLE);

        v1.pos = { xy * cosf(hAngle1), xy * sinf(hAngle1), z };
        v2.pos = { xy * cosf(hAngle2), xy * sinf(hAngle2), -z };

        vertices[i1] = v1;
        vertices[i2] = v2;

        // 1st row
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i1 + 1);

        // 2nd row
        indices.push_back(i1);
        indices.push_back(i2);
        indices.push_back(i1 + 1);

        indices.push_back(i2);
        indices.push_back(i1 + 6);
        indices.push_back(i1 + 1);

        // 3d row
        indices.push_back(i2);
        indices.push_back(i3);
        indices.push_back(i1 + 6);

        // next horizontal angles
        hAngle1 += H_ANGLE;
        hAngle2 += H_ANGLE;
    }

    i1 = 5; // 2nd row
    i2 = 10; // 3d row

    z = radius * sinf(V_ANGLE); // elevaton
    xy = radius * cosf(V_ANGLE);

    v1.pos = { xy * cosf(hAngle1), xy * sinf(hAngle1), z };
    v2.pos = { xy * cosf(hAngle2), xy * sinf(hAngle2), -z };

    vertices[i1] = v1;
    vertices[i2] = v2;

    // 1st row
    indices.push_back(i0);
    indices.push_back(i1);
    indices.push_back(1);

    // 2nd row
    indices.push_back(i1);
    indices.push_back(i2);
    indices.push_back(1);

    indices.push_back(i2);
    indices.push_back(2);
    indices.push_back(1);

    // 3d row
    indices.push_back(i2);
    indices.push_back(i3);
    indices.push_back(3);

    // the last bottom vertex (0, 0, -r)
    v1.pos = { 0, 0, -radius };
    vertices[i3] = v1;

    IndexedMesh im = subdivideIcosphere(3, vertices, indices);

    uint32_t meshId = createMesh(im.first, im.second);
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
    v1.normal = v2.normal = packNormals(normal);

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
        v.normal = packNormals(normal);
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
    uint32_t lightId = (uint32_t)mLights.size();
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

    const glm::float4x4 transform = desc.useXform ? desc.xform * scaleMatrix : getTransform(desc);
    uint32_t instId = createInstance(Instance::Type::eLight, currentLightMeshId, (uint32_t)-1, transform, lightId);
    assert(instId != -1);

    mLightIdToInstanceId[lightId] = instId;

    return lightId;
}

void Scene::updateLight(const uint32_t lightId, const UniformLightDesc& desc)
{
    float intensityPerPoint = desc.intensity; // light intensity
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

        mLights[lightId].type = 0;
    }
    else if (desc.type == 1)
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
        mLights[lightId].type = 1;
    }
    else if (desc.type == 2)
    {
        // Sphere Light
        const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), glm::float3(1.0f, 1.0f, 1.0f));
        const glm::float4x4 localTransform = desc.useXform ? scaleMatrix * desc.xform : getTransform(desc);

        mLights[lightId].points[0] = glm::float4(desc.radius, 0.f, 0.f, 0.f); // save radius
        mLights[lightId].points[1] = localTransform * glm::float4(0.f, 0.f, 0.f, 1.f); // save O

        mLights[lightId].type = 2;
    }

    mLights[lightId].color = glm::float4(desc.color, 1.0f) * intensityPerPoint;
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

bool Scene::getFrMod()
{
    return this->FrMod;
}

void Scene::updateInstanceTransform(uint32_t instId, glm::float4x4 newTransform)
{
    Instance& inst = mInstances[instId];
    inst.transform = newTransform;
    mDirtyInstances.insert(instId);
}

void Scene::beginFrame()
{
    FrMod = true;
    mDirtyInstances.clear();
}

void Scene::endFrame()
{
    FrMod = false;
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
