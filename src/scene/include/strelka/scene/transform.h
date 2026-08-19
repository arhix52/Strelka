#pragma once

#include <strelka/scene/glm_wrapper.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cmath>

namespace oka
{

/// Split an affine transform into the translation, rotation and scale of a glTF
/// node.
///
/// glm::decompose() is the obvious choice and cannot be used here: it reports
/// failure when the determinant of the upper 3x3 falls below epsilon<float>()
/// (1.2e-7) and leaves every output untouched, which for a uniformly scaled node
/// means anything under about 0.005 -- an asset authored in centimetres and
/// scaled down by 0.003 is enough. Callers that trusted the outputs got whatever
/// the stack held, and the node's transform became NaN on the first edit.
///
/// Shear is not represented: a glTF node cannot express it, and neither can a
/// gizmo edit. A sheared matrix comes back as the closest rotation.
inline void decomposeTrs(const glm::float4x4& matrix, glm::float3& translation, glm::quat& rotation, glm::float3& scale)
{
    translation = glm::float3(matrix[3]);

    glm::float3 basis[3] = { glm::float3(matrix[0]), glm::float3(matrix[1]), glm::float3(matrix[2]) };
    scale = glm::float3(glm::length(basis[0]), glm::length(basis[1]), glm::length(basis[2]));

    // A mirrored basis is not a rotation. glTF and the scale gizmo both express
    // the flip as a negative scale factor, so fold it into x and let the division
    // below turn the basis right-handed again.
    if (glm::dot(glm::cross(basis[0], basis[1]), basis[2]) < 0.0f)
    {
        scale.x = -scale.x;
    }

    // An axis scaled to nothing carries no direction to recover. Substituting the
    // identity axis keeps the result finite and orthonormal, which is the only
    // thing the caller can still use.
    const glm::float3 identityBasis[3] = { glm::float3(1.0f, 0.0f, 0.0f), glm::float3(0.0f, 1.0f, 0.0f),
                                           glm::float3(0.0f, 0.0f, 1.0f) };
    for (int axis = 0; axis < 3; ++axis)
    {
        basis[axis] = (std::fabs(scale[axis]) > 0.0f) ? basis[axis] / scale[axis] : identityBasis[axis];
    }

    rotation = glm::quat_cast(glm::float3x3(basis[0], basis[1], basis[2]));
}

/// The transform a node with this translation, rotation and scale describes.
inline glm::float4x4 composeTrs(const glm::float3& translation, const glm::quat& rotation, const glm::float3& scale)
{
    return glm::translate(glm::float4x4(1.0f), translation) * glm::float4x4(rotation) *
           glm::scale(glm::float4x4(1.0f), scale);
}

/// glTF stores rotation as [x, y, z, w]. GLM's value constructor is (w, x, y, z)
/// regardless of how the components sit in memory, so this does not depend on
/// GLM_FORCE_QUAT_DATA_WXYZ / XYZW.
inline glm::quat quatFromGltf(float x, float y, float z, float w)
{
    return glm::quat(w, x, y, z);
}

} // namespace oka
