#pragma once

#include "camera.h"
#include <strelka/material/material_params.h>
#include <light_types.h>

#include <cstdint>
#include <cmath>
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
    // The basis the control points are meant to be read under. It used to be a
    // parameter of createCurve that nothing stored, so every consumer had to
    // assume one -- and the two backends assumed different ones.
    Type mType = Type::eLinear;
    /// Segments per strand when every strand in the set has the same count, else
    /// 0. Hair from a particle system always does, and knowing it lets a shader
    /// recover where along a strand a hit landed from the segment index alone --
    /// which is the whole of what a root-to-tip gradient needs, for no memory.
    uint32_t mSegmentsPerStrand = 0;
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
    // ChangeBits is a bitmask; combined values deliberately need not name an
    // enumerator.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
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

    // 32 bytes, and it must stay 32: the Metal shaders hardcode this stride and
    // read attributes by byte offset (search for vtxStride). The two trailing
    // words were padding; uv1 and color fit in them exactly, so a second UV set
    // and vertex colours cost nothing and change no offset that already exists.
    //
    // color defaults to opaque white rather than zero: an unset vertex colour is
    // a multiplier of 1, and a zeroed one would render the surface black.
    struct Vertex
    {
        glm::float3 pos{ 0.0f };
        uint32_t tangent = 0;

        uint32_t normal = 0;
        uint32_t uv = 0;
        uint32_t uv1 = 0;                  // byte 24, packUV format
        uint32_t color = 0xFFFFFFFFu;      // byte 28, packed RGBA8, linear
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
        // Identity, not left to whatever the allocation held: a node the loader
        // did not fill in every field of still has to describe a usable
        // transform, and a garbage 3x3 propagates to every descendant.
        glm::float3 translation{ 0.0f }; //local translation
        glm::float3 scale{ 1.0f }; //local scale
        glm::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f }; //local rotation
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
    int blasUpdateCount = 0;
    int tlasUpdateCount = 0;

    // GPU side structure
    // Uploaded to the GPU verbatim, so every field is initialized: a light type
    // that does not write one (a rect light never sets normal) otherwise ships
    // whatever the allocation held, and a huge or NaN value sitting in a buffer
    // the shader may start reading is a trap that costs a day to find.
    struct Light
    {
        glm::float4 points[4]{};
        // Area / distant: radiance. Point / spot: radiant intensity (shader / r²).
        glm::float4 color = glm::float4(1.0f);
        glm::float4 normal{ 0.0f };
        int type = -1;
        // Distant: angular half-width (rad). Spot: outer cone angle (rad).
        float halfAngle = 0.0f;
        // Spot: inner cone angle (rad). Point: soft radius (0 = sharp).
        float pad0 = 0.0f;
        // Attenuation range in world units; 0 = infinite (KHR_lights_punctual).
        float pad1 = 0.0f;
    };

    // CPU side structure
    struct UniformLightDesc
    {
        int32_t type = LIGHT_TYPE_RECT;
        glm::float4x4 xform{ 1.0 };
        glm::float3 position{ 0.0f }; // world position
        glm::float3 orientation{ 0.0f }; // euler angles in degrees
        bool useXform = false;
        bool enabled = true;
        /// Whether camera rays may hit the light's own geometry.
        ///
        /// False is a light that lights the scene and appears in reflections but
        /// is not in frame -- a softbox just outside the crop, which is what
        /// V-Ray's "invisible" flag means. Distinct from `enabled`, which turns
        /// the light off entirely.
        bool visibleToCamera = true;
        /// Whether this light's contribution is cached separately, in the
        /// radiance cache's short-window "responsive" entries.
        ///
        /// For a light that changes fast enough that the cache's ordinary
        /// temporal window lags visibly behind it -- a torch being swung, a lamp
        /// switched on, anything animated. The cache then tracks this light's
        /// contribution over a few frames while the rest of the signal keeps
        /// averaging over dozens, which is what the two windows are for. Costs a
        /// second entry per voxel and a second deposit per path, so it is off
        /// unless a light asks for it. Ignored entirely when the cache is off.
        bool responsive = false;
        std::string name;

        glm::float3 color{ 1.0f };
        float intensity = 1.0f;
        // How `intensity` is interpreted; see LightIntensityUnit. Default keeps
        // the historical colour×intensity = radiance (or legacy intensity) path.
        int32_t intensityUnit = LIGHT_UNIT_RADIANCE;

        // rectangle light
        float width = 1.0f;
        float height = 1.0f;

        // disc / sphere / soft point
        float radius = 0.0f;
        // distant: half-angle in radians. spot: unused here (see cone angles).
        float halfAngle = 0.0f;

        // Spot cone, radians. Defaults match KHR_lights_punctual (π/4 outer,
        // 0 inner = hard edge). Emission along local -Z, like every other light.
        float innerConeAngle = 0.0f;
        float outerConeAngle = float(M_PI) / 4.0f;

        // KHR range; 0 = infinite. Applied as a smooth window for point/spot.
        float range = 0.0f;

        // Optional IES profile path (resolved relative to the scene). Empty =
        // isotropic. Indexed into Scene::mIesProfiles at bake time.
        std::string iesPath;
        int32_t iesProfile = -1;
    };

    // Candela table from an IESNA LM-63 file. Sampled on the GPU by (θ, φ).
    struct IesProfile
    {
        std::string path;
        std::vector<float> verticalAngles; // degrees
        std::vector<float> horizontalAngles; // degrees
        std::vector<float> candela; // row-major: v + h * nVertical
        float maxCandela = 0.0f;
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
    // Filled on first use, indexed by mesh. Empty extents mean "not computed
    // yet"; a mesh with no vertices never gets an entry to begin with.
    struct MeshBounds
    {
        glm::float3 min{ 0.0f };
        glm::float3 max{ 0.0f };
        bool valid = false;
    };
    std::vector<MeshBounds> mMeshBounds;

    // Conservative world-space box per instance, for picking.
    //
    // Rebuilt when the transforms have moved since it was made, which is what the
    // generation counter tracks -- mDirtyInstances cannot be used for this, since
    // the renderer consumes and clears it every frame. Without the cache a pick
    // pays a 4x4 inverse per instance, and at 1.1 million instances that alone is
    // most of a second of latency on a click.
    std::vector<MeshBounds> mInstanceWorldBounds;
    uint64_t mTransformGeneration = 1;
    uint64_t mInstanceBoundsGeneration = 0;
    std::vector<Curve> mCurves;
    std::vector<Instance> mInstances;
    std::vector<Light> mLights;
    std::vector<IesProfile> mIesProfiles;

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

    /// Drop the CPU-side vertex and index arrays once a backend has uploaded
    /// them. On a 50 M triangle scene that is 2.3 GB held for the life of the
    /// process, duplicating what already sits in GPU-visible buffers -- on
    /// unified memory both halves are the same pool, and the machine starts
    /// swapping.
    ///
    /// Not free of consequence, which is why it is opt-in: Scene::pick() walks
    /// these arrays, so the editor keeps them and headless rendering does not.
    /// Anything that reads them afterwards must handle them being empty.
    void releaseHostGeometry()
    {
        std::vector<Vertex>().swap(mVertices);
        std::vector<uint32_t>().swap(mIndices);
        mHostGeometryReleased = true;
    }

    bool hostGeometryReleased() const
    {
        return mHostGeometryReleased;
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
    void computeJointMatrices(std::vector<glm::mat4>* jointMatrices, size_t jointCount, uint32_t skinId);
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
        const std::scoped_lock lock(mCameraMutex);
        if (mNameToCamera.find(name) != mNameToCamera.end())
        {
            return mNameToCamera[name];
        }
        return (uint32_t)-1;
    }

    uint32_t addCamera(Camera& camera)
    {
        const std::scoped_lock lock(mCameraMutex);
        mCameras.push_back(camera);
        // store camera index
        mNameToCamera[camera.name] = (uint32_t)mCameras.size() - 1;
        return (uint32_t)mCameras.size() - 1;
    }

    void updateCamera(Camera& camera, uint32_t index)
    {
        assert(index < mCameras.size());
        const std::scoped_lock lock(mCameraMutex);
        mCameras[index] = camera;
    }

    Camera& getCamera(uint32_t index)
    {
        assert(index < mCameras.size());
        const std::scoped_lock lock(mCameraMutex);
        return mCameras[index];
    }

    const std::vector<Camera>& getCameras()
    {
        const std::scoped_lock lock(mCameraMutex);
        return mCameras;
    }

    size_t getCameraCount()
    {
        const std::scoped_lock lock(mCameraMutex);
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
            camera.updateAspectRatio(static_cast<float>(width) / static_cast<float>(height));
        }
    }

    glm::float4x4 getTransform(const Scene::UniformLightDesc& desc)
    {
        const glm::float4x4 translationMatrix = glm::translate(glm::float4x4(1.0f), desc.position);
        const glm::quat rotation = glm::quat(glm::radians(desc.orientation)); // to quaternion
        const glm::float4x4 rotationMatrix{ rotation };
        // The shape's own size, not always the rectangle's. A disc or sphere light
        // carries a radius and no width, so scaling every light by
        // (width, height, 1) collapsed its transform: the in-plane axes came out
        // zero and the light's mesh was squashed flat, which left a light that
        // illuminated nothing and could not be seen either.
        glm::float3 scale{ 1.0f };
        if (desc.type == LIGHT_TYPE_RECT)
        {
            scale = glm::float3(desc.width, desc.height, 1.0f);
        }
        else if (desc.type == LIGHT_TYPE_DISC || desc.type == LIGHT_TYPE_SPHERE ||
                 desc.type == LIGHT_TYPE_POINT || desc.type == LIGHT_TYPE_SPOT)
        {
            // Point/spot use radius as a viewport proxy (and soft size); a zero
            // radius still needs a unit scale so the orientation is not lost.
            const float r = desc.radius > 0.0f ? desc.radius : 1.0f;
            scale = glm::float3(r);
        }
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
            const glm::float4x4 xform = glm::translate(glm::float4x4(1.0f), n.translation) *
                                        glm::float4x4(n.rotation) *
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
        const int child = mNodes[nodeIdx].children[0];
        return getTransform(child);
    }

    void updateAnimation(const float dt);

    struct EnvLightDesc
    {
        std::string texturePath;
        float intensity = 1.0f;
        glm::float3 color = glm::float3(1.0f);
        float rotationY = 0.0f;
        // What camera rays see, when that differs from what lights the scene.
        // Production worlds routinely branch on Light Path: the pine forest
        // shows an 8k HDRI backdrop at strength 0.2 to the camera and a
        // procedural sky at 0.7 to everything else. Baking one environment out
        // of that has to pick a side, and either choice is wrong in the frame.
        // Empty means camera rays see the lighting environment, as before.
        std::string backgroundTexturePath;
        float backgroundIntensity = 1.0f;
    };

    // Homogeneous atmospheric scattering below `height`. A slab, not a bounded
    // volume -- see fog.h for why that is the shape offered.
    /// Camera exposure, in the photographic terms the tonemapper already takes.
    ///
    /// glTF cameras carry a projection and nothing else -- no ISO, no aperture,
    /// no shutter -- so a scene cannot say how bright it is meant to look, and a
    /// renderer defaulting to a daylight exposure renders a world authored in
    /// normalised units as black. Whoever builds the scene knows which it is, so
    /// this travels in the light sidecar alongside the lights it belongs with.
    struct ExposureDesc
    {
        float filmIso = 100.0f;
        float fStop = 1.0f;
        float shutterSpeed = 1.0f;
        float cm2Factor = 1.0f;
    };
    void setExposure(const ExposureDesc& desc)
    {
        mExposure = desc;
    }
    const std::optional<ExposureDesc>& getExposure() const { return mExposure; }

    struct AtmosphereDesc
    {
        glm::float3 color = glm::float3(1.0f); // single-scattering albedo
        float density = 0.0f;                  // extinction, per world unit
        float anisotropy = 0.0f;               // Henyey-Greenstein g
        float height = 0.0f;                   // world y above which there is none
    };

    void setAtmosphere(const AtmosphereDesc& desc)
    {
        mAtmosphere = desc;
        markChanged(ChangeBits::Env);
    }
    const std::optional<AtmosphereDesc>& getAtmosphere() const { return mAtmosphere; }

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
    int32_t addIesProfile(IesProfile profile);
    const std::vector<IesProfile>& getIesProfiles() const
    {
        return mIesProfiles;
    }

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

    /// Local-space bounds of a mesh, computed once and kept.
    ///
    /// A mesh's extent is a property of the mesh, not of the instance drawing it,
    /// and a scattered scene has orders of magnitude more instances than meshes:
    /// the pine forest places 1.1 million of them over 316 geometries. Walking
    /// the vertices per instance is what made selecting anything there take
    /// seconds. Skeletal meshes are excluded -- their extent depends on the pose,
    /// so they keep the per-instance path.
    bool meshBounds(uint32_t meshId, glm::float3& outMin, glm::float3& outMax);

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

    // Set by releaseHostGeometry(). Anything that needs the arrays back has to
    // reload the scene.
    bool mHostGeometryReleased = false;

    std::optional<EnvLightDesc> mEnvLight;
    std::optional<AtmosphereDesc> mAtmosphere;
    std::optional<ExposureDesc> mExposure;

    std::set<uint32_t> mDirtyInstances;

    uint32_t mRectLightMeshId = (uint32_t)-1;
    uint32_t mDiskLightMeshId = (uint32_t)-1;
    uint32_t mSphereLightMeshId = (uint32_t)-1;
};
} // namespace oka
