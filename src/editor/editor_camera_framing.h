#pragma once

#include <strelka/scene/camera.h>

#include <algorithm>
#include <cmath>
#include <limits>


namespace oka::editor_camera_framing
{

/// World AABB of an oriented box: transform the eight local corners and take the
/// axis-aligned envelope. The selection overlay stores the same local box +
/// transform; framing needs the world envelope so the film can be sized against
/// the camera axes rather than against the object's.
inline void worldAabbFromLocalBox(const glm::float3& localMin,
                                  const glm::float3& localMax,
                                  const glm::mat4& localToWorld,
                                  glm::float3& worldMin,
                                  glm::float3& worldMax)
{
    worldMin = glm::float3(std::numeric_limits<float>::max());
    worldMax = glm::float3(std::numeric_limits<float>::lowest());
    for (int i = 0; i < 8; ++i)
    {
        const glm::float3 corner((i & 1) ? localMax.x : localMin.x, (i & 2) ? localMax.y : localMin.y,
                                 (i & 4) ? localMax.z : localMin.z);
        const glm::float3 world = glm::float3(localToWorld * glm::float4(corner, 1.0f));
        worldMin = glm::min(worldMin, world);
        worldMax = glm::max(worldMax, world);
    }
}

/// Half-extents of a world AABB projected onto the camera's right / up / front.
/// Front is the view direction; the depth half-extent is how far the box reaches
/// toward and away from the film along that axis.
struct ProjectedExtents
{
    float halfWidth = 0.0f;
    float halfHeight = 0.0f;
    float halfDepth = 0.0f;
};

inline ProjectedExtents projectAabbOntoCamera(const glm::float3& worldMin,
                                              const glm::float3& worldMax,
                                              const glm::float3& center,
                                              const glm::float3& right,
                                              const glm::float3& up,
                                              const glm::float3& front)
{
    ProjectedExtents e;
    for (int i = 0; i < 8; ++i)
    {
        const glm::float3 corner((i & 1) ? worldMax.x : worldMin.x, (i & 2) ? worldMax.y : worldMin.y,
                                 (i & 4) ? worldMax.z : worldMin.z);
        const glm::float3 d = corner - center;
        e.halfWidth = std::max(e.halfWidth, std::abs(glm::dot(d, right)));
        e.halfHeight = std::max(e.halfHeight, std::abs(glm::dot(d, up)));
        e.halfDepth = std::max(e.halfDepth, std::abs(glm::dot(d, front)));
    }
    return e;
}

/// Distance from the AABB centre at which a perspective camera with the given
/// vertical FOV (degrees, already aspect-adapted) and aspect fits the projected
/// half-extents. Padding > 1 leaves a margin around the box.
inline float perspectiveFitDistance(float halfWidth,
                                    float halfHeight,
                                    float halfDepth,
                                    float verticalFovDegrees,
                                    float aspect,
                                    float padding = 1.1f)
{
    constexpr float kMinHalf = 1e-4f;
    halfWidth = std::max(halfWidth, kMinHalf);
    halfHeight = std::max(halfHeight, kMinHalf);
    halfDepth = std::max(halfDepth, 0.0f);

    const float halfV = glm::radians(verticalFovDegrees) * 0.5f;
    const float tanV = std::tan(halfV);
    // Horizontal half-angle from the same film: tan(h) = tan(v) * aspect.
    const float tanH = tanV * std::max(aspect, 1e-4f);
    if (!(tanV > 0.0f) || !(tanH > 0.0f))
    {
        return halfDepth + 1.0f;
    }

    const float dist = std::max(halfWidth / tanH, halfHeight / tanV) * padding;
    // The nearest corner must stay in front of the film; otherwise the box is
    // straddling the near plane and the frame that "fits" still clips it.
    return std::max(dist, halfDepth * padding + kMinHalf);
}

/// Orthographic half-extents that cover the projected box after magForAspect has
/// adapted them to `aspect`. Setting raw xmag/ymag to the projected halves is not
/// enough: a viewport wider than the object would then shrink the vertical film
/// and crop the top and bottom.
inline void orthographicFitExtents(float halfWidth,
                                   float halfHeight,
                                   float aspect,
                                   float& outXmag,
                                   float& outYmag,
                                   float padding = 1.1f)
{
    constexpr float kMinHalf = 1e-4f;
    const float needW = std::max(halfWidth, kMinHalf) * padding;
    const float needH = std::max(halfHeight, kMinHalf) * padding;
    aspect = std::max(aspect, 1e-4f);

    if (aspect >= 1.0f)
    {
        // Landscape path of magForAspect holds xmag and derives height as xmag/aspect.
        outXmag = std::max(needW, needH * aspect);
        outYmag = outXmag / aspect;
    }
    else
    {
        // Portrait path holds ymag and derives width as ymag*aspect.
        outYmag = std::max(needH, needW / aspect);
        outXmag = outYmag * aspect;
    }
}

/// Place `cam` so the world AABB fills the frame. Orientation is kept: framing
/// is a move (and, for orthographic, a film resize), not an orbit.
///
/// Perspective: dolly along the current view axis to the fit distance.
/// Orthographic: set xmag/ymag to the fit extents and put the film just in front
/// of the box -- sliding along the view axis alone would not change the image.
inline void frameCamera(Camera& cam,
                        const glm::float3& worldMin,
                        const glm::float3& worldMax,
                        float aspect,
                        float padding = 1.1f)
{
    const glm::float3 center = 0.5f * (worldMin + worldMax);
    const glm::float3 right = cam.getRight();
    const glm::float3 up = cam.getUp();
    const glm::float3 front = cam.getFront();
    const ProjectedExtents e = projectAabbOntoCamera(worldMin, worldMax, center, right, up, front);

    if (cam.projection == Camera::ProjectionType::orthographic)
    {
        float xmag = cam.xmag;
        float ymag = cam.ymag;
        orthographicFitExtents(e.halfWidth, e.halfHeight, aspect, xmag, ymag, padding);
        cam.setOrthographic(xmag, ymag, cam.znear, cam.zfar);
        // Film through the near face of the box, not through its centre: an
        // orthographic ray starts on the film, so a film that bisects the AABB
        // would leave the back half unlit and the front half starting inside.
        constexpr float kMinPush = 1e-3f;
        const float push = std::max(e.halfDepth * padding, kMinPush);
        cam.position = center - front * push;
    }
    else
    {
        const float fov = cam.fovForAspect(aspect);
        const float distance = perspectiveFitDistance(e.halfWidth, e.halfHeight, e.halfDepth, fov, aspect, padding);
        cam.position = center - front * distance;
    }
    cam.updateViewMatrix();
}

} // namespace oka::editor_camera_framing

