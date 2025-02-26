#pragma once

#include "camera.h"
// #include "materials.h"
// #undef float4
// #undef float3

#include <cstdint>
#include <mutex>
#include <set>
#include <stack>
#include <unordered_map>
#include <vector>
#include <mutex>

#include <materialmanager/materialmanager.h>

namespace oka
{
struct Mesh
{
    uint32_t mIndex; // Index of 1st index in index buffer
    uint32_t mCount; // amount of indices in mesh
    uint32_t mVbOffset; // start in vb
    uint32_t mVertexCount; // number of vertices in mesh
    uint32_t mSbOffset; // start in sb
};

struct Curve
{
    enum class Type : uint8_t
    {
        eLinear,
        eCubic,
    };
    uint32_t mVertexCountsStart;
    uint32_t mVertexCountsCount;
    uint32_t mPointsStart;
    uint32_t mPointsCount;
    uint32_t mWidthsStart;
    uint32_t mWidthsCount;
};

struct Instance
{
    glm::mat4 transform;
    enum class Type : uint8_t
    {
        eMesh,
        eLight,
        eCurve
    } type;
    union
    {
        uint32_t mMeshId;
        uint32_t mCurveId;
    };
    uint32_t mMaterialId = 0;
    uint32_t mLightId = (uint32_t)-1;
};

enum class DirtyFlag: uint32_t {
    eNone,
    eLights,
};

class Scene
{
public:
    struct MaterialDescription
    {
        enum class Type
        {
            eMdl,
            eMaterialX
        } type;
        std::string code;
        std::string file;
        std::string name;
        bool hasColor = false;
        glm::float3 color;
        std::vector<MaterialManager::Param> params;
    };

    struct Vertex
    {
        glm::float3 pos;
        uint32_t tangent;

        uint32_t normal;
        uint32_t uv;
        float pad0;
        float pad1;
    };

    struct vertexSkinData //vertex skin data
    {
        glm::ivec4 joints{0};
        glm::vec4 weights{0.0};
        glm::float3 pos;
        glm::float3 normal;
    };
    std::vector<vertexSkinData> mVertexSkinData;

    struct Node
    {
        enum class NodeType
        {
            unknown,
            sceneGraph,
            mesh,
            camera,
            skeleton
        };
        NodeType type = NodeType::unknown;
        std::string name;
        glm::float3 translation; //local translation
        glm::float3 scale; //local scale
        glm::quat rotation; //local rotation
        int parent = -1;
        std::vector<int> children;
        std::vector<uint32_t> instanceIds;
        int skin = -1;
    };
    std::vector<Node> mNodes;

    struct Skin
    {
        std::string name;
        int skeletonId = -1;
        std::vector<int> joints;
        std::vector<glm::float4x4> inverseBindMatrices;

        int refNodeId = -1;
    };
    std::vector<Skin> mSkines;

    enum class AnimationState : uint32_t
    {
        eStop,
        ePlay,
        eScroll,
    };
    AnimationState mAnimState = AnimationState::eStop;
    struct AnimationSampler
    {
        enum class InterpolationType
        {
            LINEAR,
            STEP,
            CUBICSPLINE
        };
        InterpolationType interpolation;
        std::vector<float> inputs;
        std::vector<glm::float4> outputsVec4;
    };

    struct AnimationChannel
    {
        enum class PathType
        {
            TRANSLATION,
            ROTATION,
            SCALE
        };
        PathType path;
        int node;
        uint32_t samplerIndex;
    };

    struct Animation
    {
        std::string name;
        std::vector<AnimationSampler> samplers;
        std::vector<AnimationChannel> channels;
        float start = std::numeric_limits<float>::max();
        float end = std::numeric_limits<float>::min();
        float current;
    };
    std::vector<Animation> mAnimations;

    // GPU side structure
    struct Light
    {
        glm::float4 points[4];
        glm::float4 color = glm::float4(1.0f);
        glm::float4 normal;
        int type;
        float halfAngle;
        float pad0;
        float pad1;
    };

    // CPU side structure
    struct UniformLightDesc
    {
        int32_t type;
        glm::float4x4 xform{ 1.0 };
        glm::float3 position; // world position
        glm::float3 orientation; // euler angles in degrees
        bool useXform;

        // OX - axis of light or normal
        glm::float3 color;
        float intensity;

        // rectangle light
        float width; // OY
        float height; // OZ

        // disc/sphere light
        float radius;
        // distant light
        float halfAngle; 
    };

    std::vector<UniformLightDesc> mLightDesc;
    enum class DebugView : uint32_t
    {
        eNone = 0,
        eNormals = 1,
        eShadows = 2,
        eLTC = 3,
        eMotion = 4,
        eCustomDebug = 5,
        ePTDebug = 11
    };

    DebugView mDebugViewSettings = DebugView::eNone;

    bool transparentMode = true;
    bool opaqueMode = true;

    glm::float4 mLightPosition{ 10.0, 10.0, 10.0, 1.0 };

    std::mutex mMeshMutex;
    std::mutex mInstanceMutex;

    std::vector<Vertex> mVertices;
    std::vector<uint32_t> mIndices;

    std::vector<glm::float3> mCurvePoints;
    std::vector<float> mCurveWidths;
    std::vector<uint32_t> mCurveVertexCounts;

    std::string modelPath;
    std::string getSceneFileName();
    std::string getSceneDir();

    std::vector<Mesh> mMeshes;
    std::vector<Curve> mCurves;
    std::vector<Instance> mInstances;
    std::vector<Light> mLights;

    std::vector<uint32_t> mTransparentInstances;
    std::vector<uint32_t> mOpaqueInstances;

    Scene() = default;

    ~Scene() = default;

    std::unordered_map<uint32_t, uint32_t> mLightIdToInstanceId{};

    std::unordered_map<int32_t, std::string> mTexIdToTexName{};

    std::vector<Vertex>& getVertices()
    {
        return mVertices;
    }

    std::vector<uint32_t>& getIndices()
    {
        return mIndices;
    }

    std::vector<MaterialDescription>& getMaterials()
    {
        return mMaterialsDescs;
    }

    std::vector<Light>& getLights()
    {
        return mLights;
    }

    std::vector<UniformLightDesc>& getLightsDesc()
    {
        return mLightDesc;
    }

    std::vector<Animation>& getAnimations()
    {
        return mAnimations;
    }

    glm::quat makeQuatFromFloat4 (const glm::float4 &value);
    glm::float4 makeFloat4FromQuat(const glm::quat &q);
    glm::float4 interpolate(const AnimationSampler &sampler, const AnimationChannel::PathType targetProperty, const float time);
    bool applyAnimation(const uint32_t animId);
    void applySkinning();
    void computeJointMatrices(std::vector<glm::mat4> *jointMatrices, int jointCount, const uint32_t skinId);
    const std::vector<Node>& getNodes() const
    {
        return mNodes;
    }

    glm::mat4 calculateNodeLocalTransform(const uint32_t nodeId);
    glm::mat4 calculateNodeGlobalTransform(const uint32_t nodeId);
    bool animateNode(const uint32_t nodeId, AnimationChannel::PathType targetProperty, const glm::float3 newValue);
    bool animateNode(const uint32_t nodeId, AnimationChannel::PathType targetProperty, const glm::quat newValue);
    bool updateNode(const uint32_t nodeId);

    uint32_t findCameraByName(const std::string& name)
    {
        std::scoped_lock lock(mCameraMutex);
        if (mNameToCamera.find(name) != mNameToCamera.end())
        {
            return mNameToCamera[name];
        }
        return (uint32_t)-1;
    }

    uint32_t addCamera(Camera& camera)
    {
        std::scoped_lock lock(mCameraMutex);
        mCameras.push_back(camera);
        // store camera index
        mNameToCamera[camera.name] = (uint32_t)mCameras.size() - 1;
        return (uint32_t)mCameras.size() - 1;
    }

    void updateCamera(Camera& camera, uint32_t index)
    {
        assert(index < mCameras.size());
        std::scoped_lock lock(mCameraMutex);
        mCameras[index] = camera;
    }

    Camera& getCamera(uint32_t index)
    {
        assert(index < mCameras.size());
        std::scoped_lock lock(mCameraMutex);
        return mCameras[index];
    }

    const std::vector<Camera>& getCameras()
    {
        std::scoped_lock lock(mCameraMutex);
        return mCameras;
    }

    size_t getCameraCount()
    {
        std::scoped_lock lock(mCameraMutex);
        return mCameras.size();
    }

    const std::vector<Instance>& getInstances() const
    {
        return mInstances;
    }

    const std::vector<Mesh>& getMeshes() const
    {
        return mMeshes;
    }

    const std::vector<Curve>& getCurves() const
    {
        return mCurves;
    }

    const std::vector<glm::float3>& getCurvesPoint() const
    {
        return mCurvePoints;
    }

    const std::vector<float>& getCurvesWidths() const
    {
        return mCurveWidths;
    }

    const std::vector<uint32_t>& getCurvesVertexCounts() const
    {
        return mCurveVertexCounts;
    }

    void updateCamerasParams(int width, int height)
    {
        for (Camera& camera : mCameras)
        {
            camera.updateAspectRatio((float)width / height);
        }
    }

    glm::float4x4 getTransform(const Scene::UniformLightDesc& desc)
    {
        const glm::float4x4 translationMatrix = glm::translate(glm::float4x4(1.0f), desc.position);
        glm::quat rotation = glm::quat(glm::radians(desc.orientation)); // to quaternion
        const glm::float4x4 rotationMatrix{ rotation };
        glm::float3 scale = { desc.width, desc.height, 1.0f };
        const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), scale);

        const glm::float4x4 localTransform = translationMatrix * rotationMatrix * scaleMatrix;

        return localTransform;
    }

    glm::float4x4 getTransformFromRoot(int nodeIdx)
    {
        std::stack<glm::float4x4> xforms;
        while (nodeIdx != -1)
        {
            const Node& n = mNodes[nodeIdx];
            glm::float4x4 xform = glm::translate(glm::float4x4(1.0f), n.translation) * glm::float4x4(n.rotation) *
                                  glm::scale(glm::float4x4(1.0f), n.scale);
            xforms.push(xform);
            nodeIdx = n.parent;
        }
        glm::float4x4 xform = glm::float4x4(1.0);
        while (!xforms.empty())
        {
            xform = xform * xforms.top();
            xforms.pop();
        }
        return xform;
    }

    glm::float4x4 getTransform(int nodeIdx)
    {
        glm::float4x4 xform = glm::float4x4(1.0);
        while (nodeIdx != -1)
        {
            const Node& n = mNodes[nodeIdx];
            xform = glm::translate(glm::float4x4(1.0f), n.translation) * glm::float4x4(n.rotation) *
                    glm::scale(glm::float4x4(1.0f), n.scale) * xform;
            nodeIdx = n.parent;
        }
        return xform;
    }


    glm::float4x4 getCameraTransform(int nodeIdx)
    {
        int child = mNodes[nodeIdx].children[0];
        return getTransform(child);
    }

    void updateAnimation(const float dt);

    void updateLight(uint32_t lightId, const UniformLightDesc& desc);
    /// <summary>
    /// Create Mesh geometry
    /// </summary>
    /// <param name="vb">Vertices</param>
    /// <param name="ib">Indices</param>
    /// <returns>Mesh id in scene</returns>
    uint32_t createMesh(const std::vector<Vertex>& vb, const std::vector<uint32_t>& ib);
    uint32_t createMesh(const std::vector<Vertex>& vb, const std::vector<uint32_t>& ib, const std::vector<oka::Scene::vertexSkinData>& sb);
    /// <summary>
    /// Creates Instance
    /// </summary>
    /// <param name="meshId">valid mesh id</param>
    /// <param name="materialId">valid material id</param>
    /// <param name="transform">transform</param>
    /// <returns>Instance id in scene</returns>
    uint32_t createInstance(const Instance::Type type,
                            const uint32_t geomId,
                            const uint32_t materialId,
                            const glm::mat4& transform,
                            const uint32_t lightId = (uint32_t)-1);

    uint32_t addMaterial(const MaterialDescription& material);

    uint32_t createCurve(const Curve::Type type,
                         const std::vector<uint32_t>& vertexCounts,
                         const std::vector<glm::float3>& points,
                         const std::vector<float>& widths);

    uint32_t createLight(const UniformLightDesc& desc);
    /// <summary>
    /// Removes instance/mesh/material
    /// </summary>
    /// <param name="meshId">valid mesh id</param>
    /// <param name="materialId">valid material id</param>
    /// <param name="instId">valid instance id</param>
    /// <returns>Nothing</returns>
    void removeInstance(uint32_t instId);
    void removeMesh(uint32_t meshId);
    void removeMaterial(uint32_t materialId);

    std::vector<uint32_t>& getOpaqueInstancesToRender(const glm::float3& camPos);

    std::vector<uint32_t>& getTransparentInstancesToRender(const glm::float3& camPos);

    DirtyFlag getDirtyState()
    {
        return mDirty;
    }

    void clearDirtyState()
    {
        mDirty = DirtyFlag::eNone;
    }

    /// <summary>
    /// Get set of DirtyInstances
    /// </summary>
    /// <returns>Set of instances</returns>
    std::set<uint32_t> getDirtyInstances();

    /// <summary>
    /// Updates Instance matrix(transform)
    /// </summary>
    /// <param name="instId">valid instance id</param>
    /// <param name="newTransform">new transformation matrix</param>
    /// <returns>Nothing</returns>
    void updateInstanceTransform(uint32_t instId, glm::float4x4 newTransform);
    /// <summary>
    /// Changes status of scene and cleans up mDirty* sets
    /// </summary>
    /// <returns>Nothing</returns>
    void beginFrame();
    /// <summary>
    /// Changes status of scene
    /// </summary>
    /// <returns>Nothing</returns>
    void endFrame();

private:
    std::vector<Camera> mCameras;
    std::unordered_map<std::string, uint32_t> mNameToCamera;
    std::mutex mCameraMutex;

    std::stack<uint32_t> mDelInstances;
    std::stack<uint32_t> mDelMesh;
    std::stack<uint32_t> mDelMaterial;

    std::vector<MaterialDescription> mMaterialsDescs;

    uint32_t createRectLightMesh();
    uint32_t createDiscLightMesh();
    uint32_t createSphereLightMesh();

    DirtyFlag mDirty;

    std::set<uint32_t> mDirtyInstances;

    uint32_t mRectLightMeshId = (uint32_t)-1;
    uint32_t mDiskLightMeshId = (uint32_t)-1;
    uint32_t mSphereLightMeshId = (uint32_t)-1;
};
} // namespace oka
