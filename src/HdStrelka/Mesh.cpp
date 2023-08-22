#include "Mesh.h"

#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/hd/smoothNormals.h>
#include <pxr/imaging/hd/vertexAdjacency.h>

#include <log.h>

#include <tracy/Tracy.hpp>

PXR_NAMESPACE_OPEN_SCOPE

// clang-format off
TF_DEFINE_PRIVATE_TOKENS(_tokens,
    (st)
);
// clang-format on

HdStrelkaMesh::HdStrelkaMesh(const SdfPath& id, oka::Scene* scene)
    : HdMesh(id), mPrototypeTransform(1.0), mColor(0.0, 0.0, 0.0), mHasColor(false), mScene(scene)
{
}

HdStrelkaMesh::~HdStrelkaMesh()
{
}

void HdStrelkaMesh::Sync(HdSceneDelegate* sceneDelegate,
                         HdRenderParam* renderParam,
                         HdDirtyBits* dirtyBits,
                         const TfToken& reprToken)
{
    ZoneScoped;
    TF_UNUSED(renderParam);
    TF_UNUSED(reprToken);

    HD_TRACE_FUNCTION();
    HF_MALLOC_TAG_FUNCTION();

    HdRenderIndex& renderIndex = sceneDelegate->GetRenderIndex();

    if ((*dirtyBits & HdChangeTracker::DirtyInstancer) | (*dirtyBits & HdChangeTracker::DirtyInstanceIndex))
    {
        HdDirtyBits dirtyBitsCopy = *dirtyBits;
        _UpdateInstancer(sceneDelegate, &dirtyBitsCopy);
        const SdfPath& instancerId = GetInstancerId();
        HdInstancer::_SyncInstancerAndParents(renderIndex, instancerId);
    }

    const SdfPath& id = GetId();
    mName = id.GetText();

    if (*dirtyBits & HdChangeTracker::DirtyMaterialId)
    {
        const SdfPath& materialId = sceneDelegate->GetMaterialId(id);
        SetMaterialId(materialId);
    }

    if (*dirtyBits & HdChangeTracker::DirtyTransform)
    {
        mPrototypeTransform = sceneDelegate->GetTransform(id);
    }

    const bool updateGeometry = (*dirtyBits & HdChangeTracker::DirtyPoints) |
                                (*dirtyBits & HdChangeTracker::DirtyNormals) |
                                (*dirtyBits & HdChangeTracker::DirtyTopology);

    *dirtyBits = HdChangeTracker::Clean;

    if (!updateGeometry)
    {
        return;
    }

    mFaces.clear();
    mPoints.clear();
    mNormals.clear();

    _UpdateGeometry(sceneDelegate);
}

//  valid range of coordinates [-1; 1]
static uint32_t packNormal(const glm::float3& normal)
{
    uint32_t packed = (uint32_t)((normal.x + 1.0f) / 2.0f * 511.99999f);
    packed += (uint32_t)((normal.y + 1.0f) / 2.0f * 511.99999f) << 10;
    packed += (uint32_t)((normal.z + 1.0f) / 2.0f * 511.99999f) << 20;
    return packed;
}

void HdStrelkaMesh::_ConvertMesh()
{
    ZoneScoped;
    const std::vector<GfVec3f>& meshPoints = GetPoints();
    const std::vector<GfVec3f>& meshNormals = GetNormals();
    const std::vector<GfVec3i>& meshFaces = GetFaces();
    TF_VERIFY(meshPoints.size() == meshNormals.size());
    const size_t vertexCount = meshPoints.size();

    std::vector<oka::Scene::Vertex> vertices(vertexCount);
    std::vector<uint32_t> indices(meshFaces.size() * 3);

    for (size_t j = 0; j < meshFaces.size(); ++j)
    {
        const GfVec3i& vertexIndices = meshFaces[j];
        indices[j * 3 + 0] = vertexIndices[0];
        indices[j * 3 + 1] = vertexIndices[1];
        indices[j * 3 + 2] = vertexIndices[2];
    }
    for (size_t j = 0; j < vertexCount; ++j)
    {
        const GfVec3f& point = meshPoints[j];
        const GfVec3f& normal = meshNormals[j];

        oka::Scene::Vertex& vertex = vertices[j];
        vertex.pos[0] = point[0];
        vertex.pos[1] = point[1];
        vertex.pos[2] = point[2];

        const glm::float3 glmNormal = glm::float3(normal[0], normal[1], normal[2]);
        vertex.normal = packNormal(glmNormal);
    }

    mStrelkaMeshId = mScene->createMesh(vertices, indices);
    assert(mStrelkaMeshId != -1);
}

void HdStrelkaMesh::_UpdateGeometry(HdSceneDelegate* sceneDelegate)
{
    ZoneScoped;
    const HdMeshTopology& topology = GetMeshTopology(sceneDelegate);
    const SdfPath& id = GetId();
    const HdMeshUtil meshUtil(&topology, id);

    VtVec3iArray indices;
    VtIntArray primitiveParams;
    meshUtil.ComputeTriangleIndices(&indices, &primitiveParams);

    VtVec3fArray points;
    VtVec3fArray normals;
    VtVec2fArray uvs;
    bool indexedNormals;
    bool indexedUVs;
    _PullPrimvars(sceneDelegate, points, normals, uvs, indexedNormals, indexedUVs, mColor, mHasColor);
    const bool hasUVs = !uvs.empty();
    for (int i = 0; i < indices.size(); i++)
    {
        GfVec3i newFaceIndices(i * 3 + 0, i * 3 + 1, i * 3 + 2);
        mFaces.push_back(newFaceIndices);

        const GfVec3i& faceIndices = indices[i];
        mPoints.push_back(points[faceIndices[0]]);
        mPoints.push_back(points[faceIndices[1]]);
        mPoints.push_back(points[faceIndices[2]]);
        auto computeTangent = [](const GfVec3f& normal) {
            GfVec3f c1 = GfCross(normal, GfVec3f(1.0f, 0.0f, 0.0f));
            GfVec3f c2 = GfCross(normal, GfVec3f(0.0f, 1.0f, 0.0f));
            GfVec3f tangent;
            if (c1.GetLengthSq() > c2.GetLengthSq())
            {
                tangent = c1;
            }
            else
            {
                tangent = c2;
            }
            GfNormalize(&tangent);
            return tangent;
        };

        mNormals.push_back(normals[indexedNormals ? faceIndices[0] : newFaceIndices[0]]);
        mTangents.push_back(computeTangent(normals[indexedNormals ? faceIndices[0] : newFaceIndices[0]]));
        mNormals.push_back(normals[indexedNormals ? faceIndices[1] : newFaceIndices[1]]);
        mTangents.push_back(computeTangent(normals[indexedNormals ? faceIndices[1] : newFaceIndices[1]]));
        mNormals.push_back(normals[indexedNormals ? faceIndices[2] : newFaceIndices[2]]);
        mTangents.push_back(computeTangent(normals[indexedNormals ? faceIndices[2] : newFaceIndices[2]]));

        if (hasUVs)
        {
            mUvs.push_back(uvs[indexedUVs ? faceIndices[0] : newFaceIndices[0]]);
            mUvs.push_back(uvs[indexedUVs ? faceIndices[1] : newFaceIndices[1]]);
            mUvs.push_back(uvs[indexedUVs ? faceIndices[2] : newFaceIndices[2]]);
        }
    }
}

bool HdStrelkaMesh::_FindPrimvar(HdSceneDelegate* sceneDelegate,
                                 const TfToken& primvarName,
                                 HdInterpolation& interpolation) const
{
    ZoneScoped;
    const HdInterpolation interpolations[] = {
        HdInterpolation::HdInterpolationVertex,   HdInterpolation::HdInterpolationFaceVarying,
        HdInterpolation::HdInterpolationConstant, HdInterpolation::HdInterpolationUniform,
        HdInterpolation::HdInterpolationVarying,  HdInterpolation::HdInterpolationInstance
    };
    for (const HdInterpolation& currInteroplation : interpolations)
    {
        const auto& primvarDescs = GetPrimvarDescriptors(sceneDelegate, currInteroplation);
        for (const HdPrimvarDescriptor& primvar : primvarDescs)
        {
            if (primvar.name == primvarName)
            {
                interpolation = currInteroplation;
                return true;
            }
        }
    }
    return false;
}

void HdStrelkaMesh::_PullPrimvars(HdSceneDelegate* sceneDelegate,
                                  VtVec3fArray& points,
                                  VtVec3fArray& normals,
                                  VtVec2fArray& uvs,
                                  bool& indexedNormals,
                                  bool& indexedUVs,
                                  GfVec3f& color,
                                  bool& hasColor) const
{
    ZoneScoped;
    const SdfPath& id = GetId();
    // Handle points.
    HdInterpolation pointInterpolation;
    const bool foundPoints = _FindPrimvar(sceneDelegate, HdTokens->points, pointInterpolation);

    if (!foundPoints)
    {
        TF_RUNTIME_ERROR("Points primvar not found!");
        return;
    }
    else if (pointInterpolation != HdInterpolation::HdInterpolationVertex)
    {
        TF_RUNTIME_ERROR("Points primvar is not vertex-interpolated!");
        return;
    }

    const VtValue boxedPoints = sceneDelegate->Get(id, HdTokens->points);
    points = boxedPoints.Get<VtVec3fArray>();

    // Handle color.
    HdInterpolation colorInterpolation;
    const bool foundColor = _FindPrimvar(sceneDelegate, HdTokens->displayColor, colorInterpolation);

    if (foundColor && colorInterpolation == HdInterpolation::HdInterpolationConstant)
    {
        const VtValue boxedColors = sceneDelegate->Get(id, HdTokens->displayColor);
        const VtVec3fArray& colors = boxedColors.Get<VtVec3fArray>();
        color = colors[0];
        hasColor = true;
    }

    const HdMeshTopology topology = GetMeshTopology(sceneDelegate);

    // Handle normals.
    HdInterpolation normalInterpolation;
    const bool foundNormals = _FindPrimvar(sceneDelegate, HdTokens->normals, normalInterpolation);

    if (foundNormals && normalInterpolation == HdInterpolation::HdInterpolationVertex)
    {
        const VtValue boxedNormals = sceneDelegate->Get(id, HdTokens->normals);
        normals = boxedNormals.Get<VtVec3fArray>();
        indexedNormals = true;
    }
    if (foundNormals && normalInterpolation == HdInterpolation::HdInterpolationFaceVarying)
    {
        const VtValue boxedFvNormals = sceneDelegate->Get(id, HdTokens->normals);
        const VtVec3fArray& fvNormals = boxedFvNormals.Get<VtVec3fArray>();

        const HdMeshUtil meshUtil(&topology, id);
        VtValue boxedTriangulatedNormals;
        if (!meshUtil.ComputeTriangulatedFaceVaryingPrimvar(
                fvNormals.cdata(), fvNormals.size(), HdTypeFloatVec3, &boxedTriangulatedNormals))
        {
            TF_CODING_ERROR("Unable to triangulate face-varying normals of %s", id.GetText());
        }

        normals = boxedTriangulatedNormals.Get<VtVec3fArray>();
        indexedNormals = false;
    }
    else
    {
        Hd_VertexAdjacency adjacency;
        adjacency.BuildAdjacencyTable(&topology);
        normals = Hd_SmoothNormals::ComputeSmoothNormals(&adjacency, points.size(), points.cdata());
        indexedNormals = true;
    }
    // Handle texture coords
    HdInterpolation textureCoordInterpolation;
    const bool foundTextureCoord = _FindPrimvar(sceneDelegate, _tokens->st, textureCoordInterpolation);
    if (foundTextureCoord && textureCoordInterpolation == HdInterpolationVertex)
    {
        uvs = sceneDelegate->Get(id, _tokens->st).Get<VtVec2fArray>();
        indexedUVs = true;
    }
    if (foundTextureCoord && textureCoordInterpolation == HdInterpolation::HdInterpolationFaceVarying)
    {
        const VtValue boxedUVs = sceneDelegate->Get(id, _tokens->st);
        const VtVec2fArray& fvUVs = boxedUVs.Get<VtVec2fArray>();

        const HdMeshUtil meshUtil(&topology, id);
        VtValue boxedTriangulatedUVS;
        if (!meshUtil.ComputeTriangulatedFaceVaryingPrimvar(
                fvUVs.cdata(), fvUVs.size(), HdTypeFloatVec2, &boxedTriangulatedUVS))
        {
            TF_CODING_ERROR("Unable to triangulate face-varying UVs of %s", id.GetText());
        }
        uvs = boxedTriangulatedUVS.Get<VtVec2fArray>();
        indexedUVs = false;
    }
}


const TfTokenVector& HdStrelkaMesh::GetBuiltinPrimvarNames() const
{
    return BUILTIN_PRIMVAR_NAMES;
}

const std::vector<GfVec3f>& HdStrelkaMesh::GetPoints() const
{
    return mPoints;
}

const std::vector<GfVec3f>& HdStrelkaMesh::GetNormals() const
{
    return mNormals;
}

const std::vector<GfVec3f>& HdStrelkaMesh::GetTangents() const
{
    return mTangents;
}

const std::vector<GfVec3i>& HdStrelkaMesh::GetFaces() const
{
    return mFaces;
}

const std::vector<GfVec2f>& HdStrelkaMesh::GetUVs() const
{
    return mUvs;
}

const GfMatrix4d& HdStrelkaMesh::GetPrototypeTransform() const
{
    return mPrototypeTransform;
}

const GfVec3f& HdStrelkaMesh::GetColor() const
{
    return mColor;
}

bool HdStrelkaMesh::HasColor() const
{
    return mHasColor;
}

const char* HdStrelkaMesh::getName() const
{
    return mName.c_str();
}

HdDirtyBits HdStrelkaMesh::GetInitialDirtyBitsMask() const
{
    return HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyTopology |
           HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex | HdChangeTracker::DirtyTransform |
           HdChangeTracker::DirtyMaterialId | HdChangeTracker::DirtyPrimvar;
}

HdDirtyBits HdStrelkaMesh::_PropagateDirtyBits(HdDirtyBits bits) const
{
    return bits;
}

void HdStrelkaMesh::_InitRepr(const TfToken& reprName, HdDirtyBits* dirtyBits)
{
    TF_UNUSED(reprName);
    TF_UNUSED(dirtyBits);
}

PXR_NAMESPACE_CLOSE_SCOPE
