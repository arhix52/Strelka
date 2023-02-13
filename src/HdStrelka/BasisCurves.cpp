#include "BasisCurves.h"

PXR_NAMESPACE_OPEN_SCOPE
void HdStrelkaBasisCurves::Sync(HdSceneDelegate* sceneDelegate,
                                HdRenderParam* renderParam,
                                HdDirtyBits* dirtyBits,
                                const TfToken& reprToken)
{
    TF_UNUSED(renderParam);
    TF_UNUSED(reprToken);

    HdRenderIndex& renderIndex = sceneDelegate->GetRenderIndex();

    const SdfPath& id = GetId();
    mName = id.GetText();
    printf("Curve Name: %s\n", mName.c_str());

    if (*dirtyBits & HdChangeTracker::DirtyMaterialId)
    {
        const SdfPath& materialId = sceneDelegate->GetMaterialId(id);
        SetMaterialId(materialId);
    }

    if (*dirtyBits & HdChangeTracker::DirtyTopology)
    {
        mTopology = sceneDelegate->GetBasisCurvesTopology(id);
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

    // m_faces.clear();
    mPoints.clear();
    mNormals.clear();

    _UpdateGeometry(sceneDelegate);
}

bool HdStrelkaBasisCurves::_FindPrimvar(HdSceneDelegate* sceneDelegate,
                                        TfToken primvarName,
                                        HdInterpolation& interpolation) const
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

void HdStrelkaBasisCurves::_PullPrimvars(HdSceneDelegate* sceneDelegate,
                                         VtVec3fArray& points,
                                         VtVec3fArray& normals,
                                         VtFloatArray& widths,
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

    HdBasisCurvesTopology topology = GetBasisCurvesTopology(sceneDelegate);

    VtIntArray curveVertexCounts = topology.GetCurveVertexCounts();

    // Handle normals.
    HdInterpolation normalInterpolation;
    bool foundNormals = _FindPrimvar(sceneDelegate, HdTokens->normals, normalInterpolation);

    if (foundNormals && normalInterpolation == HdInterpolation::HdInterpolationVarying)
    {
        VtValue boxedNormals = sceneDelegate->Get(id, HdTokens->normals);
        normals = boxedNormals.Get<VtVec3fArray>();
        indexedNormals = true;
    }

    // Handle width.
    HdInterpolation widthInterpolation;
    bool foundWidth = _FindPrimvar(sceneDelegate, HdTokens->widths, widthInterpolation);

    if (foundWidth)
    {
        VtValue boxedWidths = sceneDelegate->Get(id, HdTokens->widths);
        widths = boxedWidths.Get<VtFloatArray>();
    }
}

void HdStrelkaBasisCurves::_UpdateGeometry(HdSceneDelegate* sceneDelegate)
{
    const HdBasisCurvesTopology& topology = mTopology;
    const SdfPath& id = GetId();

    // Get USD Curve Metadata
    mVertexCounts = topology.GetCurveVertexCounts();
    TfToken curveType = topology.GetCurveType();
    TfToken curveBasis = topology.GetCurveBasis();
    TfToken curveWrap = topology.GetCurveWrap();

    size_t num_curves = mVertexCounts.size();
    size_t num_keys = 0;

    bool indexedNormals;
    bool indexedUVs;
    bool hasColor = true;
    _PullPrimvars(sceneDelegate, mPoints, mNormals, mWidths, indexedNormals, indexedUVs, mColor, hasColor);
    _ConvertCurve();
}

HdStrelkaBasisCurves::HdStrelkaBasisCurves(const SdfPath& id, oka::Scene* scene) : HdBasisCurves(id), mScene(scene)
{
}

HdStrelkaBasisCurves::~HdStrelkaBasisCurves()
{
}

HdDirtyBits HdStrelkaBasisCurves::GetInitialDirtyBitsMask() const
{
    return HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyTopology |
           HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex | HdChangeTracker::DirtyTransform |
           HdChangeTracker::DirtyMaterialId | HdChangeTracker::DirtyPrimvar;
}

HdDirtyBits HdStrelkaBasisCurves::_PropagateDirtyBits(HdDirtyBits bits) const
{
    return bits;
}

void HdStrelkaBasisCurves::_InitRepr(const TfToken& reprName, HdDirtyBits* dirtyBits)
{
    TF_UNUSED(reprName);
    TF_UNUSED(dirtyBits);
}

void HdStrelkaBasisCurves::_ConvertCurve()
{
    // calculate phantom points
    // https://raytracing-docs.nvidia.com/optix7/guide/index.html#curves#differences-between-curves-spheres-and-triangles
    glm::float3 p1 = glm::float3(mPoints[0][0], mPoints[0][1], mPoints[0][2]);
    glm::float3 p2 = glm::float3(mPoints[1][0], mPoints[1][1], mPoints[1][2]);
    glm::float3 p0 = p1 + (p1 - p2);
    mCurvePoints.push_back(p0);
    for (const GfVec3f& p : mPoints)
    {
        mCurvePoints.push_back(glm::float3(p[0], p[1], p[2]));
    }
    int n = mPoints.size() - 1;
    glm::float3 pn = glm::float3(mPoints[n][0], mPoints[n][1], mPoints[n][2]);
    glm::float3 pn1 = glm::float3(mPoints[n - 1][0], mPoints[n - 1][1], mPoints[n - 1][2]);
    glm::float3 pnn = pn + (pn - pn1);
    mCurvePoints.push_back(pnn);

    mCurveWidths.push_back(mWidths[0] * 0.5);

    assert((mWidths.size() == mPoints.size()) || (mWidths.size() == 1));

    if (mWidths.size() == 1)
    {
        for (int i = 0; i < mPoints.size(); ++i)
        {
            mCurveWidths.push_back(mWidths[0] * 0.5);
        }
    }
    else
    {
        for (const float w : mWidths)
        {
            mCurveWidths.push_back(w * 0.5f);
        }
    }

    mCurveWidths.push_back(mCurveWidths.back());

    for (const int i : mVertexCounts)
    {
        mCurveVertexCounts.push_back(i);
    }
}
const std::vector<glm::float3>& HdStrelkaBasisCurves::GetPoints() const
{
    return mCurvePoints;
}
const std::vector<float>& HdStrelkaBasisCurves::GetWidths() const
{
    return mCurveWidths;
}
const std::vector<uint32_t>& HdStrelkaBasisCurves::GetVertexCounts() const
{
    return mCurveVertexCounts;
}
const GfMatrix4d& HdStrelkaBasisCurves::GetPrototypeTransform() const
{
    return m_prototypeTransform;
}
PXR_NAMESPACE_CLOSE_SCOPE
