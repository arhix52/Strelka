#pragma once

#include <pxr/pxr.h>
#include <pxr/imaging/hd/mesh.h>

#include <scene/scene.h>

#include <pxr/base/gf/vec2f.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdStrelkaMesh final : public HdMesh
{
public:
    HF_MALLOC_TAG_NEW("new HdStrelkaMesh");

    HdStrelkaMesh(const SdfPath& id, oka::Scene* scene);

    ~HdStrelkaMesh() override;

    void Sync(HdSceneDelegate* delegate,
              HdRenderParam* renderParam,
              HdDirtyBits* dirtyBits,
              const TfToken& reprToken) override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;

    const TfTokenVector& GetBuiltinPrimvarNames() const override;

    const std::vector<GfVec3f>& GetPoints() const;
    const std::vector<GfVec3f>& GetNormals() const;
    const std::vector<GfVec3i>& GetFaces() const;
    const std::vector<GfVec2f>& GetUVs() const;
    const GfMatrix4d& GetPrototypeTransform() const;

    const GfVec3f& GetColor() const;

    bool HasColor() const;

    const char* getName() const;

protected:
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override;

    void _InitRepr(const TfToken& reprName, HdDirtyBits* dirtyBits) override;

private:
    void _ConvertMesh();

    void _UpdateGeometry(HdSceneDelegate* sceneDelegate);

    bool _FindPrimvar(HdSceneDelegate* sceneDelegate, const TfToken& primvarName, HdInterpolation& interpolation) const;

    void _PullPrimvars(HdSceneDelegate* sceneDelegate,
                       VtVec3fArray& points,
                       VtVec3fArray& normals,
                       VtVec2fArray& uvs,
                       bool& indexedNormals,
                       bool& indexedUVs,
                       GfVec3f& color,
                       bool& hasColor) const;

    const TfTokenVector BUILTIN_PRIMVAR_NAMES = { HdTokens->points, HdTokens->normals };

    GfMatrix4d mPrototypeTransform;
    std::vector<GfVec3f> mPoints;
    std::vector<GfVec3f> mNormals;
    std::vector<GfVec2f> mUvs;
    std::vector<GfVec3i> mFaces;
    GfVec3f mColor;
    bool mHasColor;
    oka::Scene* mScene;
    std::string mName;
    uint32_t mStrelkaMeshId;
};

PXR_NAMESPACE_CLOSE_SCOPE
