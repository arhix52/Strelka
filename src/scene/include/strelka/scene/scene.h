#pragma once

#include "camera.h"
#include <strelka/material/material_params.h>
#include <light_types.h>

#include <cstdint>
#include <mutex>
#include <optional>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace oka
{

struct Mesh
{
    uint32_t mIndex; // Index of 1st index in index buffer
    uint32_t mCount; // amount of indices in mesh
    uint32_t mVbOffset; // start in vb
    uint32_t mVertexCount; // number of vertices in mesh
    uint32_t mSbOffset; // start in sb
    bool isSkeletal = false;
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
    bool isAnimated = false;
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

enum class ChangeBits : uint32_t
{
    None = 0,
    Transforms = 1u << 0,
    Lights = 1u << 1,
    Materials = 1u << 2,
    Env = 1u << 3,
    Geometry = 1u << 4,
};

inline ChangeBits operator|(ChangeBits a, ChangeBits b)
{
    return static_cast<ChangeBits>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline ChangeBits operator&(ChangeBits a, ChangeBits b)
{
    return static_cast<ChangeBits>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline ChangeBits& operator|=(ChangeBits& a, ChangeBits b)
{
    a = a | b;
    return a;
}

inline bool any(ChangeBits bits)
{
    return static_cast<uint32_t>(bits) != 0;
}

class Scene
{
public:
    struct MaterialDescription
    {
        std::string name;
        MaterialParams params = {};  // GPU-ready PBR material parameters

        // Texture file paths (resolved by renderer into GPU texture objects)
        std::string baseColorTexPath;
        std::string metallicRoughnessTexPath;
        std::string normalTexPath;
        std::string emissionTexPath;
        std::string occlusionTexPath;
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
        float pad0;
        glm::float3 normal;
        uint32_t tangent{0}; // rest-pose packed tangent for skinning
    };
    std::vector<vertexSkinData> mVerticesSkinData;

    struct Node
    {
        enum class NodeType : uint8_t
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
        int camera = -1;
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
    int blasUpdateCount;
    int tlasUpdateCount;

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

    std::vector<vertexSkinData>& getVerticesSkinData()
    {
        return mVerticesSkinData;
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

    const std::vector<Light>& getLights() const
    {
        return mLights;
    }

    std::vector<UniformLightDesc>& getLightsDesc()
    {
        return mLightDesc;
    }

    const std::vector<UniformLightDesc>& getLightsDesc() const
    {
        return mLightDesc;
    }

    const std::vector<MaterialDescription>& getMaterials() const
    {
        return mMaterialsDescs;
    }

    const std::vector<Vertex>& getVertices() const
    {
        return mVertices;
    }

    const std::vector<uint32_t>& getIndices() const
    {
        return mIndices;
    }

    const std::vector<Instance>& getInstances() const
    {
        return mInstances;
    }

    std::vector<Instance>& getInstances()
    {
        return mInstances;
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

    /// World transform of every node, refreshed in one top-down pass.
    ///
    /// Recomputing a node's world transform by walking up to the root (as
    /// calculateNodeGlobalTransform does) is O(depth) *per node*, and driving it
    /// from every animation channel independently re-walked the same subtrees
    /// over and over. Caching turns per-frame animation into O(nodes).
    const std::vector<glm::mat4>& getGlobalTransforms()
    {
        ensureGlobalTransforms();
        return mGlobalTransforms;
    }

private:
    std::vector<glm::mat4> mGlobalTransforms;
    std::vector<uint8_t> mNodeDirty;
    std::vector<int> mNodeOrder; // parents always precede their children

    void buildNodeOrder();
    void ensureGlobalTransforms();
    void refreshGlobalTransforms();
    /// Apply the side effects (instance transforms, camera poses) of a changed
    /// node subtree and report whether a skeleton node was touched.
    bool applyNodeSideEffects(const uint32_t nodeId);

public:

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

    struct EnvLightDesc
    {
        std::string texturePath;
        float intensity = 1.0f;
        glm::float3 color = glm::float3(1.0f);
        float rotationY = 0.0f;
    };

    void setEnvLight(const EnvLightDesc& desc)
    {
        mEnvLight = desc;
        markChanged(ChangeBits::Env);
    }
    const std::optional<EnvLightDesc>& getEnvLight() const { return mEnvLight; }

    void setSourcePath(const std::string& path) { modelPath = path; }
    const std::string& getSourcePath() const { return modelPath; }

    /// Unified light edit: updates desc, baked GPU light, and proxy instance.
    void setLight(uint32_t lightId, const UniformLightDesc& desc);
    /// Legacy: bakes GPU light from desc without writing mLightDesc (prefer setLight).
    void updateLight(uint32_t lightId, const UniformLightDesc& desc);

    uint32_t getLightInstanceId(uint32_t lightId) const
    {
        auto it = mLightIdToInstanceId.find(lightId);
        return it != mLightIdToInstanceId.end() ? it->second : (uint32_t)-1;
    }

    /// Authoritative node local TRS edit; refreshes derived instance transforms.
    void setNodeLocalTransform(uint32_t nodeId,
                               const glm::float3& translation,
                               const glm::quat& rotation,
                               const glm::float3& scale);

    void setMaterial(uint32_t id, const MaterialDescription& desc);
    /// <summary>
    /// Create Mesh geometry
    /// </summary>
    /// <param name="vb">Vertices</param>
    /// <param name="ib">Indices</param>
    /// <returns>Mesh id in scene</returns>
    uint32_t createMesh(const std::vector<Vertex>& vb, const std::vector<uint32_t>& ib);
    uint32_t createSkeletalMesh(const std::vector<Vertex>& vb, const std::vector<uint32_t>& ib, const std::vector<oka::Scene::vertexSkinData>& sb);
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

    ChangeBits peekChanges() const
    {
        return mChanges;
    }

    ChangeBits consumeChanges()
    {
        const ChangeBits bits = mChanges;
        mChanges = ChangeBits::None;
        mDirtyInstances.clear();
        return bits;
    }

    void markChanged(ChangeBits bits)
    {
        mChanges |= bits;
    }

    /// Legacy wrappers — prefer peekChanges / consumeChanges.
    ChangeBits getDirtyState()
    {
        return mChanges;
    }

    void clearDirtyState()
    {
        mChanges = ChangeBits::None;
        mDirtyInstances.clear();
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

    struct PickHit
    {
        bool hit = false;
        uint32_t instanceId = (uint32_t)-1;
        uint32_t nodeId = (uint32_t)-1;
        uint32_t lightId = (uint32_t)-1;
        float distance = 0.0f;
        glm::float3 position{ 0.0f };
    };

    /// CPU raycast against mesh instances (and light proxies). Closest hit wins.
    PickHit pick(const glm::float3& origin, const glm::float3& direction);

    /// Axis aligned bounds of an instance in the space its transform maps to
    /// world, with the current skinning pose applied.
    ///
    /// Skinning runs on the GPU and its result never comes back, so the CPU
    /// vertex buffer of a skeletal mesh keeps holding the rest pose. Anything
    /// CPU side that needs the posed geometry has to re-evaluate it from the
    /// joint palette, which is what this does.
    bool computeInstanceBounds(uint32_t instId, glm::float3& outMin, glm::float3& outMax);

    /// Joint matrices driving the instance this frame, empty when it is rigid.
    std::vector<glm::mat4> buildJointPalette(uint32_t instId);

    /// Position of a mesh vertex after skinning, in the space the instance
    /// transform maps to world. Pass the palette from buildJointPalette().
    glm::float3 posedVertexPosition(const Mesh& mesh,
                                    uint32_t vertexIndex,
                                    const std::vector<glm::mat4>& jointPalette) const;

    /// Node owning the instance, -1 when the instance is not attached to one.
    int findInstanceNodeId(uint32_t instId) const;

private:
    std::vector<Camera> mCameras;
    std::unordered_map<std::string, uint32_t> mNameToCamera;
    std::mutex mCameraMutex;

    std::stack<uint32_t> mDelInstances;
    std::stack<uint32_t> mDelMesh;
    std::stack<uint32_t> mDelMaterial;

    std::vector<MaterialDescription> mMaterialsDescs;

    /// Weight blended skinning matrix of a mesh vertex. Returns false when the
    /// vertex carries no skinning, in which case its stored position is final.
    bool vertexSkinMatrix(const Mesh& mesh,
                          uint32_t vertexIndex,
                          const std::vector<glm::mat4>& jointPalette,
                          glm::mat4& outMat) const;

    uint32_t createRectLightMesh();
    uint32_t createDiscLightMesh();
    uint32_t createSphereLightMesh();

    ChangeBits mChanges = ChangeBits::None;

    std::optional<EnvLightDesc> mEnvLight;

    std::set<uint32_t> mDirtyInstances;

    uint32_t mRectLightMeshId = (uint32_t)-1;
    uint32_t mDiskLightMeshId = (uint32_t)-1;
    uint32_t mSphereLightMeshId = (uint32_t)-1;
};
} // namespace oka
