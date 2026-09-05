#pragma once

// The box traversal culls an analytic light by.
//
// The OptiX backend gives every analytic emitter an AABB in a custom-primitive
// structure, and the intersection program decides the rest. That makes the box
// the only thing standing between a ray and a light it should have hit: a box
// that is too small does not slow anything down, it silently drops the emitter
// from the frame, and only for the rays that graze it.
//
// So it is built from the same packed points the exact intersector reads, it is
// loose on purpose -- the component sum of the axes rather than their true
// extent -- and it lives here, out of the .cpp, so that a test can hold it
// against the exact intersector rather than against its own arithmetic.
// tests/render/test_analytic_light_bounds.cpp is that test.

#include <light_types.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

namespace oka::optix_lights
{

struct Aabb
{
    glm::vec3 lo{ 0.0f };
    glm::vec3 hi{ 0.0f };
};

/// `points` is UniformLight::points, verbatim: what each entry means depends on
/// the type, exactly as intersectAnalyticLightSurfaceUnchecked() reads it.
inline Aabb analyticLightAabb(int lightType, const glm::vec4* points)
{
    const float radius = points[0].x;
    glm::vec3 center(0.0f);
    glm::vec3 extent(0.0f);
    if (lightType == LIGHT_TYPE_RECT)
    {
        // Four corners, and the fourth is implied by the other three: the exact
        // test spans point0 by point1 - point0 and point3 - point0.
        const glm::vec3 corner(points[0]);
        const glm::vec3 opposite = glm::vec3(points[1]) + glm::vec3(points[3]) - corner;
        const glm::vec3 lo = glm::min(glm::min(corner, glm::vec3(points[1])),
                                        glm::min(glm::vec3(points[3]), opposite));
        const glm::vec3 hi = glm::max(glm::max(corner, glm::vec3(points[1])),
                                        glm::max(glm::vec3(points[3]), opposite));
        center = 0.5f * (lo + hi);
        extent = 0.5f * (hi - lo);
    }
    else if (lightType == LIGHT_TYPE_DISC)
    {
        center = glm::vec3(points[1]);
        extent = glm::abs(glm::vec3(points[2])) + glm::abs(glm::vec3(points[3]));
    }
    else if (lightType == LIGHT_TYPE_SPHERE)
    {
        // Sum of the axis components bounds every affine image of the unit
        // sphere, shear and mirror included -- the same argument the ellipsoid
        // sampler makes, one axis at a time.
        center = glm::vec3(points[1]);
        extent = glm::abs(glm::vec3(points[0])) + glm::abs(glm::vec3(points[2])) +
                 glm::abs(glm::vec3(points[3]));
    }
    else
    {
        // A soft point, spot or projector: the sphere the radius describes.
        center = glm::vec3(points[1]);
        extent = glm::vec3(radius);
    }

    // A flat light -- and a rectangle is always flat -- gives a box a ray can
    // slip through edge-on. The epsilon is relative to where the light is, so
    // it survives a scene authored in kilometres.
    const float slack =
        1e-4f * std::max(1.0f, std::max(std::fabs(center.x), std::max(std::fabs(center.y), std::fabs(center.z))));
    extent += glm::vec3(slack);

    Aabb box;
    box.lo = center - extent;
    box.hi = center + extent;
    return box;
}

} // namespace oka::optix_lights
