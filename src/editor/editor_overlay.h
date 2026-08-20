#pragma once

#include <cmath>

#include <strelka/scene/glm_wrapper.hpp>


namespace oka::editor_overlay
{

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

