#include "Mesh.h"

#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/hd/smoothNormals.h>
#include <pxr/imaging/hd/vertexAdjacency.h>

PXR_NAMESPACE_OPEN_SCOPE

// clang-format off
TF_DEFINE_PRIVATE_TOKENS(_tokens,
    (st)
);
// clang-format on

HdStrelkaMesh::HdStrelkaMesh(const SdfPath& id, oka::Scene* scene)
    : HdMesh(id), m_prototypeTransform(1.0), m_color(0.0, 0.0, 0.0), m_hasColor(false), mScene(scene)
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
    const char* meshName = id.GetText();
    mName = meshName;
    printf("Mesh: %s\n", meshName);

    if (*dirtyBits & HdChangeTracker::DirtyMaterialId)
    {
        const SdfPath& materialId = sceneDelegate->GetMaterialId(id);
        SetMaterialId(materialId);
    }

    if (*dirtyBits & HdChangeTracker::DirtyTransform)
    {
        m_prototypeTransform = sceneDelegate->GetTransform(id);
    }

    bool updateGeometry = (*dirtyBits & HdChangeTracker::DirtyPoints) | (*dirtyBits & HdChangeTracker::DirtyNormals) |
                          (*dirtyBits & HdChangeTracker::DirtyTopology);

    *dirtyBits = HdChangeTracker::Clean;

    if (!updateGeometry)
    {
        return;
    }

    m_faces.clear();
    m_points.clear();
    m_normals.clear();

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
    GfMatrix4d transform = m_prototypeTransform; // need to add instancer support
    GfMatrix4d normalMatrix = transform.GetInverse().GetTranspose();

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
    glm::float3 sum = glm::float3(0.0f, 0.0f, 0.0f);
    for (size_t j = 0; j < vertexCount; ++j)
    {
        const GfVec3f& point = meshPoints[j];
        const GfVec3f& normal = meshNormals[j];

        oka::Scene::Vertex& vertex = vertices[j];
        vertex.pos[0] = point[0];
        vertex.pos[1] = point[1];
        vertex.pos[2] = point[2];

        glm::float3 glmNormal = glm::float3(normal[0], normal[1], normal[2]);
        vertex.setNormal(packNormal(glmNormal));
        sum += glm::float3(vertex.pos);
    }
    const glm::float3 massCenter = sum / (float)vertexCount;

    glm::float4x4 glmTransform;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            glmTransform[i][j] = (float)transform[i][j];
        }
    }

    uint32_t meshId = mScene->createMesh(vertices, indices);
    assert(meshId != -1);
    // uint32_t instId = mScene->createInstance(meshId, materialIndex, glmTransform, massCenter);
    // assert(instId != -1);
}

void HdStrelkaMesh::_UpdateGeometry(HdSceneDelegate* sceneDelegate)
{
    const HdMeshTopology& topology = GetMeshTopology(sceneDelegate);
    const SdfPath& id = GetId();
    HdMeshUtil meshUtil(&topology, id);

    VtVec3iArray indices;
    VtIntArray primitiveParams;
    meshUtil.ComputeTriangleIndices(&indices, &primitiveParams);

    VtVec3fArray points;
    VtVec3fArray normals;
    VtVec2fArray uvs;
    bool indexedNormals;
    bool indexedUVs;
    _PullPrimvars(sceneDelegate, points, normals, uvs, indexedNormals, indexedUVs, m_color, m_hasColor);
    const bool hasUVs = !uvs.empty();
    for (int i = 0; i < indices.size(); i++)
    {
        GfVec3i newFaceIndices(i * 3 + 0, i * 3 + 1, i * 3 + 2);
        m_faces.push_back(newFaceIndices);

        const GfVec3i& faceIndices = indices[i];
        m_points.push_back(points[faceIndices[0]]);
        m_points.push_back(points[faceIndices[1]]);
        m_points.push_back(points[faceIndices[2]]);
        m_normals.push_back(normals[indexedNormals ? faceIndices[0] : newFaceIndices[0]]);
        m_normals.push_back(normals[indexedNormals ? faceIndices[1] : newFaceIndices[1]]);
        m_normals.push_back(normals[indexedNormals ? faceIndices[2] : newFaceIndices[2]]);
        if (hasUVs)
        {
            m_uvs.push_back(uvs[indexedUVs ? faceIndices[0] : newFaceIndices[0]]);
            m_uvs.push_back(uvs[indexedUVs ? faceIndices[1] : newFaceIndices[1]]);
            m_uvs.push_back(uvs[indexedUVs ? faceIndices[2] : newFaceIndices[2]]);
        }
    }
}

bool HdStrelkaMesh::_FindPrimvar(HdSceneDelegate* sceneDelegate, TfToken primvarName, HdInterpolation& interpolation) const
{
    HdInterpolation interpolations[] = {
        HdInterpolation::HdInterpolationVertex,   HdInterpolation::HdInterpolationFaceVarying,
        HdInterpolation::HdInterpolationConstant, HdInterpolation::HdInterpolationUniform,
        HdInterpolation::HdInterpolationVarying,  HdInterpolation::HdInterpolationInstance
    };
    for (HdInterpolation i : interpolations)
    {
        const auto& primvarDescs = GetPrimvarDescriptors(sceneDelegate, i);
        for (const HdPrimvarDescriptor& primvar : primvarDescs)
        {
            if (primvar.name == primvarName)
            {
                interpolation = i;

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
    const SdfPath& id = GetId();
    // Handle points.
    HdInterpolation pointInterpolation;
    bool foundPoints = _FindPrimvar(sceneDelegate, HdTokens->points, pointInterpolation);

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

    VtValue boxedPoints = sceneDelegate->Get(id, HdTokens->points);
    points = boxedPoints.Get<VtVec3fArray>();

    // Handle color.
    HdInterpolation colorInterpolation;
    bool foundColor = _FindPrimvar(sceneDelegate, HdTokens->displayColor, colorInterpolation);

    if (foundColor && colorInterpolation == HdInterpolation::HdInterpolationConstant)
    {
        VtValue boxedColors = sceneDelegate->Get(id, HdTokens->displayColor);
        const VtVec3fArray& colors = boxedColors.Get<VtVec3fArray>();
        color = colors[0];
        hasColor = true;
    }

    HdMeshTopology topology = GetMeshTopology(sceneDelegate);

    // Handle normals.
    HdInterpolation normalInterpolation;
    bool foundNormals = _FindPrimvar(sceneDelegate, HdTokens->normals, normalInterpolation);

    if (foundNormals && normalInterpolation == HdInterpolation::HdInterpolationVertex)
    {
        VtValue boxedNormals = sceneDelegate->Get(id, HdTokens->normals);
        normals = boxedNormals.Get<VtVec3fArray>();
        indexedNormals = true;
    }
    if (foundNormals && normalInterpolation == HdInterpolation::HdInterpolationFaceVarying)
    {
        VtValue boxedFvNormals = sceneDelegate->Get(id, HdTokens->normals);
        const VtVec3fArray& fvNormals = boxedFvNormals.Get<VtVec3fArray>();

        HdMeshUtil meshUtil(&topology, id);
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
    bool foundTextureCoord = _FindPrimvar(sceneDelegate, _tokens->st, textureCoordInterpolation);
    if (foundTextureCoord && textureCoordInterpolation == HdInterpolationVertex)
    {
        uvs = sceneDelegate->Get(id, _tokens->st).Get<VtVec2fArray>();
        indexedUVs = true;
    }
    if (foundTextureCoord && textureCoordInterpolation == HdInterpolation::HdInterpolationFaceVarying)
    {
        VtValue boxedUVs = sceneDelegate->Get(id, _tokens->st);
        const VtVec2fArray& fvUVs = boxedUVs.Get<VtVec2fArray>();

        HdMeshUtil meshUtil(&topology, id);
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

const TfTokenVector BUILTIN_PRIMVAR_NAMES = { HdTokens->points, HdTokens->normals };

const TfTokenVector& HdStrelkaMesh::GetBuiltinPrimvarNames() const
{
    return BUILTIN_PRIMVAR_NAMES;
}

const std::vector<GfVec3f>& HdStrelkaMesh::GetPoints() const
{
    return m_points;
}

const std::vector<GfVec3f>& HdStrelkaMesh::GetNormals() const
{
    return m_normals;
}

const std::vector<GfVec3i>& HdStrelkaMesh::GetFaces() const
{
    return m_faces;
}

const std::vector<GfVec2f>& HdStrelkaMesh::GetUVs() const
{
    return m_uvs;
}

const GfMatrix4d& HdStrelkaMesh::GetPrototypeTransform() const
{
    return m_prototypeTransform;
}

const GfVec3f& HdStrelkaMesh::GetColor() const
{
    return m_color;
}

bool HdStrelkaMesh::HasColor() const
{
    return m_hasColor;
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
