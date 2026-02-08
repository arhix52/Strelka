//    Copyright (C) 2021 Pablo Delgado Krämer
//
//        This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//                                         (at your option) any later version.
//
//                                         This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program. If not, see <https://www.gnu.org/licenses/>.

#include "Instancer.h"

#include <pxr/base/gf/quatd.h>
#include <pxr/imaging/hd/sceneDelegate.h>

PXR_NAMESPACE_OPEN_SCOPE

HdStrelkaInstancer::HdStrelkaInstancer(HdSceneDelegate* delegate,
                                       const SdfPath& id)
    : HdInstancer(delegate, id)
{
}

HdStrelkaInstancer::~HdStrelkaInstancer()
{
}

void HdStrelkaInstancer::Sync(HdSceneDelegate* sceneDelegate,
                              HdRenderParam* renderParam,
                              HdDirtyBits* dirtyBits)
{
    TF_UNUSED(renderParam);

    _UpdateInstancer(sceneDelegate, dirtyBits);

    const SdfPath& id = GetId();

    if (!HdChangeTracker::IsAnyPrimvarDirty(*dirtyBits, id))
    {
        return;
    }

    const HdPrimvarDescriptorVector& primvars = sceneDelegate->GetPrimvarDescriptors(id, HdInterpolation::HdInterpolationInstance);

    for (const HdPrimvarDescriptor& primvar : primvars)
    {
        TfToken primName = primvar.name;

        if (primName != HdInstancerTokens->instanceTranslations &&
            primName != HdInstancerTokens->instanceRotations &&
            primName != HdInstancerTokens->instanceScales &&
            primName != HdInstancerTokens->instanceTransforms)
        {
            continue;
        }

        if (!HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, primName))
        {
            continue;
        }

        VtValue value = sceneDelegate->Get(id, primName);

        m_primvarMap[primName] = value;
    }
}

VtMatrix4dArray HdStrelkaInstancer::ComputeInstanceTransforms(const SdfPath& prototypeId)
{
    HdSceneDelegate* sceneDelegate = GetDelegate();

    const SdfPath& id = GetId();

    // Calculate instance transforms for this instancer.
    VtValue boxedTranslates = m_primvarMap[HdInstancerTokens->instanceTranslations];
    VtValue boxedRotates = m_primvarMap[HdInstancerTokens->instanceRotations];
    VtValue boxedScales = m_primvarMap[HdInstancerTokens->instanceScales];
    VtValue boxedInstanceTransforms = m_primvarMap[HdInstancerTokens->instanceTransforms];

    VtVec3fArray translates;
    if (boxedTranslates.IsHolding<VtVec3fArray>())
    {
        translates = boxedTranslates.UncheckedGet<VtVec3fArray>();
    }
    else if (!boxedTranslates.IsEmpty())
    {
        TF_CODING_WARNING("Instancer translate values are not of type Vec3f!");
    }

    VtVec4fArray rotates;
    if (boxedRotates.IsHolding<VtVec4fArray>())
    {
        rotates = boxedRotates.Get<VtVec4fArray>();
    }
    else if (!boxedRotates.IsEmpty())
    {
        TF_CODING_WARNING("Instancer rotate values are not of type Vec3f!");
    }

    VtVec3fArray scales;
    if (boxedScales.IsHolding<VtVec3fArray>())
    {
        scales = boxedScales.Get<VtVec3fArray>();
    }
    else if (!boxedScales.IsEmpty())
    {
        TF_CODING_WARNING("Instancer scale values are not of type Vec3f!");
    }

    VtMatrix4dArray instanceTransforms;
    if (boxedInstanceTransforms.IsHolding<VtMatrix4dArray>())
    {
        instanceTransforms = boxedInstanceTransforms.Get<VtMatrix4dArray>();
    }

    GfMatrix4d instancerTransform = sceneDelegate->GetInstancerTransform(id);

    const VtIntArray& instanceIndices = sceneDelegate->GetInstanceIndices(id, prototypeId);

    VtMatrix4dArray transforms;
    transforms.resize(instanceIndices.size());

    for (size_t i = 0; i < instanceIndices.size(); i++)
    {
        int instanceIndex = instanceIndices[i];

        GfMatrix4d mat = instancerTransform;

        GfMatrix4d temp;

        if (i < translates.size())
        {
            auto trans = GfVec3d(translates[instanceIndex]);
            temp.SetTranslate(trans);
            mat = temp * mat;
        }
        if (i < rotates.size())
        {
            GfVec4f rot = rotates[instanceIndex];
            temp.SetRotate(GfQuatd(rot[0], rot[1], rot[2], rot[3]));
            mat = temp * mat;
        }
        if (i < scales.size())
        {
            auto scale = GfVec3d(scales[instanceIndex]);
            temp.SetScale(scale);
            mat = temp * mat;
        }
        if (i < instanceTransforms.size())
        {
            temp = instanceTransforms[instanceIndex];
            mat = temp * mat;
        }

        transforms[i] = mat;
    }

    // Calculate instance transforms for all instancer instances.
    const SdfPath& parentId = GetParentId();

    if (parentId.IsEmpty())
    {
        return transforms;
    }

    const HdRenderIndex& renderIndex = sceneDelegate->GetRenderIndex();
    HdInstancer* boxedParentInstancer = renderIndex.GetInstancer(parentId);
    HdStrelkaInstancer* parentInstancer = dynamic_cast<HdStrelkaInstancer*>(boxedParentInstancer);

    VtMatrix4dArray parentTransforms = parentInstancer->ComputeInstanceTransforms(id);

    VtMatrix4dArray transformProducts;
    transformProducts.resize(parentTransforms.size() * transforms.size());

    for (size_t i = 0; i < parentTransforms.size(); i++)
    {
        for (size_t j = 0; j < transforms.size(); j++)
        {
            size_t index = i * transforms.size() + j;

            transformProducts[index] = transforms[j] * parentTransforms[i];
        }
    }

    return transformProducts;
}

PXR_NAMESPACE_CLOSE_SCOPE
