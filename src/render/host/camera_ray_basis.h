#pragma once

#include <strelka/scene/glm_wrapper.hpp>

namespace oka::metal
{

struct PerspectiveCameraRayBasis
{
    glm::float3 right;
    glm::float3 up;
    glm::float3 forward;
};

// Expands the two matrix-vector products used to unproject a perspective ray.
// With clip = (ndc.x, ndc.y, 1, 1), the view-space direction is
// clipToView[0] * x + clipToView[1] * y + clipToView[2] + clipToView[3].
inline PerspectiveCameraRayBasis perspectiveCameraRayBasis(const glm::float4x4& viewToWorld,
                                                           const glm::float4x4& clipToView)
{
    const glm::float3 right = glm::float3(viewToWorld * glm::float4(glm::float3(clipToView[0]), 0.0f));
    const glm::float3 up = glm::float3(viewToWorld * glm::float4(glm::float3(clipToView[1]), 0.0f));
    const glm::float3 viewForward = glm::float3(clipToView[2] + clipToView[3]);
    const glm::float3 forward = glm::float3(viewToWorld * glm::float4(viewForward, 0.0f));
    return { right, up, forward };
}

} // namespace oka::metal
