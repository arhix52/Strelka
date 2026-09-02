#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <strelka/scene/camera.h>
#include <strelka/scene/glm_wrapper.hpp>


namespace oka::editor_overlay
{

// Path-traced camera rays start at the eye/film and do not use the authored
// raster near plane. Clip overlays only far enough in front of the eye to avoid
// projecting through its w=0 singularity.
inline constexpr float kEyePlaneDistance = 1e-4f;

/// Build the projection ImGuizmo needs for a path-traced target.
///
/// ImGuizmo rejects a perspective target when its projected reverse-Z depth is
/// below 0.001. That includes valid path-traced geometry before an authored near
/// plane (the bathroom camera uses 10 m). Worse, that early return leaks
/// ImGuizmo's draw-list clip rect and hides every viewport overlay drawn
/// afterwards. Keep the current screen-space projection, but fit a private depth
/// range around the gizmo origin.
inline bool prepareGizmoCamera(Camera& camera, const glm::float3& worldTarget)
{
    if (camera.projection == Camera::ProjectionType::orthographic)
    {
        return true;
    }

    const glm::float4 viewTarget = camera.matrices.view * glm::float4(worldTarget, 1.0f);
    const float depth = -viewTarget.z;
    if (!std::isfinite(depth) || depth <= kEyePlaneDistance || depth > std::numeric_limits<float>::max() * 0.25f)
    {
        return false;
    }

    // Recover the already aspect-adapted vertical FOV from the live projection,
    // so changing only its depth terms cannot move the gizmo on screen.
    const float projectionX = std::fabs(camera.matrices.perspective[0][0]);
    const float projectionY = std::fabs(camera.matrices.perspective[1][1]);
    if (!(projectionX > 0.0f) || !(projectionY > 0.0f) || !std::isfinite(projectionX) || !std::isfinite(projectionY))
    {
        return false;
    }
    const float aspect = projectionY / projectionX;
    const float verticalFov = glm::degrees(2.0f * std::atan(1.0f / projectionY));
    const float nearPlane = std::max(kEyePlaneDistance, depth * 0.05f);
    const float farPlane = std::max(nearPlane * 2.0f, depth * 4.0f);
    camera.setPerspective(verticalFov, aspect, nearPlane, farPlane);
    return true;
}

/// Trim a segment to the part of it the camera can see, given the view-space
/// depth of both endpoints, and report whether anything is left.
///
/// The test is on view depth rather than on clip w, which is what the selection
/// box used to do. For a perspective frame the two are the same thing (w is the
/// distance down the view axis), but an orthographic clip w is always 1: every
/// endpoint read as visible, so a box behind the camera projected through the
/// origin and drew over the frame as if it were in front of it.
///
/// Clip space is an affine function of the view position for both projections, so
/// the crossing found by interpolating depth is the same point in clip space.
inline bool trimSegmentToNearPlane(
    glm::float4& clipA, glm::float4& clipB, const float viewZa, const float viewZb, const float nearZ)
{
    // View space looks down -Z: visible depth is at most -nearZ.
    const float limit = -std::fabs(nearZ);
    const bool aFront = viewZa <= limit;
    const bool bFront = viewZb <= limit;
    if (!aFront && !bFront)
    {
        return false;
    }
    if (aFront && bFront)
    {
        return true;
    }
    const float denom = viewZb - viewZa;
    if (std::fabs(denom) < 1e-12f)
    {
        return false;
    }
    const float t = (limit - viewZa) / denom;
    const glm::float4 crossing = clipA + (clipB - clipA) * t;
    if (aFront)
    {
        clipB = crossing;
    }
    else
    {
        clipA = crossing;
    }
    return true;
}

/// Clip space to pixels within a rect, y down. Fails instead of returning a
/// non-finite point: ImGui takes vertices at face value, and one NaN in a draw
/// list is enough to lose everything drawn after it.
inline bool clipToScreen(const glm::float4& clip,
                         const glm::float2& rectMin,
                         const glm::float2& rectSize,
                         glm::float2& outPixels)
{
    if (std::fabs(clip.w) < 1e-12f)
    {
        return false;
    }
    const float ndcX = clip.x / clip.w;
    const float ndcY = clip.y / clip.w;
    if (!std::isfinite(ndcX) || !std::isfinite(ndcY))
    {
        return false;
    }
    outPixels = glm::float2(rectMin.x + (ndcX * 0.5f + 0.5f) * rectSize.x, rectMin.y + (0.5f - ndcY * 0.5f) * rectSize.y);
    return std::isfinite(outPixels.x) && std::isfinite(outPixels.y);
}

} // namespace oka::editor_overlay
