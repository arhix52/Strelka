// The box has to contain the surface the intersection program will find.
//
// On the OptiX backend an analytic emitter is a custom primitive: hardware
// traversal culls it by its AABB and only then runs the intersection program.
// That inverts the old failure mode. A light-table walk that skipped a light
// cost a wrong pixel; a box that is one epsilon too small costs a light that is
// simply not there for the rays that graze it -- no error, no warning, and the
// only symptom is a slightly darker frame.
//
// This is the check that could not be written against the pre-test it replaced.
// That one -- analyticLightBoundsMayIntersect(), a ball around each light in
// front of both scans -- had a test of exactly this shape over random spheres,
// which passed, and it still disagreed with the exact intersector on 333 of
// kids_room's 921 600 pixels. So this one does not test the box against its own
// arithmetic: it takes the rays the exact intersector says are hits and demands
// the box contain the point it returned, for every light type the backend
// builds a box for.

#include <host/analytic_light_bounds.h>

#include <analytic_light.h>

#include <doctest/doctest.h>

namespace
{

// The linear congruential generator the other analytic-light tests use, so a
// failure here is reproducible from the seed alone.
struct Rng
{
    uint32_t s = 0x2545f491u;
    float next()
    {
        s = s * 1664525u + 1013904223u;
        return float(s >> 8) * 0x1p-24f;
    }
    float sym(float k) { return (2.0f * next() - 1.0f) * k; }
};

bool contains(const oka::optix_lights::Aabb& box, const float3& p)
{
    return p.x >= box.lo.x && p.x <= box.hi.x && p.y >= box.lo.y && p.y <= box.hi.y && p.z >= box.lo.z &&
           p.z <= box.hi.z;
}

} // namespace

TEST_CASE("the analytic light AABB contains every hit the intersector reports")
{
    Rng rng;
    const int types[] = { LIGHT_TYPE_RECT, LIGHT_TYPE_DISC, LIGHT_TYPE_SPHERE, LIGHT_TYPE_SPOT };
    uint32_t hits[4] = { 0u, 0u, 0u, 0u };

    for (int t = 0; t < 4; ++t)
    {
        const int type = types[t];
        for (int i = 0; i < 20000; ++i)
        {
            const glm::vec3 centre(rng.sym(3.0f), rng.sym(3.0f), rng.sym(3.0f));
            const float scale = 0.05f + 2.0f * rng.next();
            // Sheared and mirrored axes, not merely rotated ones: the disc and
            // the ellipsoid are affine images of the unit circle and sphere,
            // and the scene packs whatever the glTF transform gives them.
            const glm::vec3 axisX(scale * rng.sym(1.0f), scale * rng.sym(0.4f), scale * rng.sym(0.4f));
            const glm::vec3 axisY(scale * rng.sym(0.4f), scale * rng.sym(1.0f), scale * rng.sym(0.4f));
            const glm::vec3 axisZ(scale * rng.sym(0.4f), scale * rng.sym(0.4f), scale * rng.sym(1.0f));

            glm::vec4 points[4] = {};
            float3 emissionNormal = make_float3(0.0f, 0.0f, 1.0f);
            if (type == LIGHT_TYPE_RECT)
            {
                // points[] as the scene packs a rectangle: four corners, and
                // the exact test spans the first by the second and the fourth.
                const glm::vec3 corner = centre - 0.5f * (axisX + axisY);
                points[0] = glm::vec4(corner, 0.0f);
                points[1] = glm::vec4(corner + axisX, 0.0f);
                points[2] = glm::vec4(corner + axisX + axisY, 0.0f);
                points[3] = glm::vec4(corner + axisY, 0.0f);
                const glm::vec3 n = glm::normalize(glm::cross(axisX, axisY));
                emissionNormal = make_float3(n.x, n.y, n.z);
            }
            else if (type == LIGHT_TYPE_DISC)
            {
                // Centre, then the two axes of the ellipse.
                points[1] = glm::vec4(centre, 0.0f);
                points[2] = glm::vec4(axisX, 0.0f);
                points[3] = glm::vec4(axisY, 0.0f);
                const glm::vec3 n = glm::normalize(glm::cross(axisX, axisY));
                emissionNormal = make_float3(n.x, n.y, n.z);
            }
            else if (type == LIGHT_TYPE_SPHERE)
            {
                // Axis, centre, axis, axis -- the order the ellipsoid reads.
                points[0] = glm::vec4(axisX, 0.0f);
                points[1] = glm::vec4(centre, 0.0f);
                points[2] = glm::vec4(axisY, 0.0f);
                points[3] = glm::vec4(axisZ, 0.0f);
            }
            else
            {
                // A soft spot: a sphere of `radius`, which rides in points[0].x
                // beside the cone angles the intersector does not use.
                points[0] = glm::vec4(scale, rng.sym(1.0f), rng.sym(1.0f), 0.0f);
                points[1] = glm::vec4(centre, 0.0f);
                points[2] = glm::vec4(axisX, 0.0f);
                points[3] = glm::vec4(axisY, 0.0f);
            }

            const float3 origin = make_float3(rng.sym(6.0f), rng.sym(6.0f), rng.sym(6.0f));
            // Half the rays aimed near the light: a test whose rays all miss
            // would pass against a box of zero size.
            const float3 aim = make_float3(centre.x + rng.sym(2.0f), centre.y + rng.sym(2.0f), centre.z + rng.sym(2.0f));
            float3 direction = (rng.next() < 0.5f) ?
                                   make_float3(aim.x - origin.x, aim.y - origin.y, aim.z - origin.z) :
                                   make_float3(rng.sym(1.0f), rng.sym(1.0f), rng.sym(1.0f));
            if (!(dot(direction, direction) > 0.0f))
            {
                continue;
            }
            direction = normalize(direction);

            const float3 p0 = make_float3(points[0].x, points[0].y, points[0].z);
            const float3 p1 = make_float3(points[1].x, points[1].y, points[1].z);
            const float3 p2 = make_float3(points[2].x, points[2].y, points[2].z);
            const float3 p3 = make_float3(points[3].x, points[3].y, points[3].z);
            const AnalyticLightIntersection hit = intersectAnalyticLightSurfaceUnchecked(
                type, p0, p1, p2, p3, emissionNormal, origin, direction, 0.0f, 40.0f);
            if (!hit.hit)
            {
                continue;
            }
            ++hits[t];
            CAPTURE(type);
            CAPTURE(i);
            REQUIRE(contains(oka::optix_lights::analyticLightAabb(type, points), hit.point));
        }
    }

    // Every type has to have produced hits, or its box was never tested.
    for (const uint32_t count : hits)
    {
        CHECK(count > 200u);
    }
}
