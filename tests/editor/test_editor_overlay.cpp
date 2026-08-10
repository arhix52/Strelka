#include <doctest/doctest.h>

#include "editor_overlay.h"

#include <strelka/scene/camera.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <vector>

using namespace oka;

namespace
{

constexpr float kRectW = 1024.0f;
constexpr float kRectH = 768.0f;
const glm::float2 kRectMin(100.0f, 50.0f);
const glm::float2 kRectSize(kRectW, kRectH);

Camera makePerspective()
{
    Camera cam;
    cam.position = glm::float3(0.0f, 0.0f, 5.0f);
    cam.mOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    cam.setPerspective(45.0f, kRectW / kRectH, 0.1f, 1000.0f);
    cam.updateViewMatrix();
    cam.updateAspectRatio(kRectW / kRectH);
    return cam;
}

Camera makeOrthographic(float xmag, float ymag)
{
    Camera cam;
    cam.position = glm::float3(0.0f, 0.0f, 5.0f);
    cam.mOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    cam.setOrthographic(xmag, ymag, 0.1f, 1000.0f);
    cam.updateViewMatrix();
    cam.updateAspectRatio(kRectW / kRectH);
    return cam;
}

// The loop the selection box draws with, minus ImGui: the pixel endpoints of every
// edge that survived the near plane.
std::vector<glm::float2> boxEdgePixels(const Camera& cam,
                                       const glm::float3& bbMin,
                                       const glm::float3& bbMax,
                                       const glm::mat4& worldFromLocal = glm::mat4(1.0f))
{
    const glm::mat4 viewFromLocal = cam.matrices.view * worldFromLocal;
    const glm::mat4 clipFromLocal = cam.matrices.perspective * viewFromLocal;

    glm::float4 clip[8];
    float viewZ[8];
    for (int i = 0; i < 8; ++i)
    {
        const glm::float4 local(
            (i & 1) ? bbMax.x : bbMin.x, (i & 2) ? bbMax.y : bbMin.y, (i & 4) ? bbMax.z : bbMin.z, 1.0f);
        clip[i] = clipFromLocal * local;
        viewZ[i] = (viewFromLocal * local).z;
    }

    static const int edges[12][2] = { { 0, 1 }, { 1, 3 }, { 3, 2 }, { 2, 0 }, { 4, 5 }, { 5, 7 },
                                      { 7, 6 }, { 6, 4 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } };
    std::vector<glm::float2> pixels;
    for (const auto& e : edges)
    {
        glm::float4 a = clip[e[0]];
        glm::float4 b = clip[e[1]];
        if (!editor_overlay::trimSegmentToNearPlane(a, b, viewZ[e[0]], viewZ[e[1]], cam.znear))
        {
            continue;
        }
        glm::float2 pa(0.0f), pb(0.0f);
        if (!editor_overlay::clipToScreen(a, kRectMin, kRectSize, pa) ||
            !editor_overlay::clipToScreen(b, kRectMin, kRectSize, pb))
        {
            continue;
        }
        pixels.push_back(pa);
        pixels.push_back(pb);
    }
    return pixels;
}

bool insideRect(const glm::float2& p)
{
    return p.x >= kRectMin.x - 1.0f && p.x <= kRectMin.x + kRectW + 1.0f && p.y >= kRectMin.y - 1.0f &&
           p.y <= kRectMin.y + kRectH + 1.0f;
}

// Distance between a world point and the pick ray fired at the pixel that point
// projected to. Zero means the box is drawn around exactly what a click there
// would select.
float pickMissDistance(const Camera& cam, const glm::float3& worldPoint)
{
    const glm::float4 clip = cam.matrices.perspective * cam.matrices.view * glm::float4(worldPoint, 1.0f);
    glm::float2 pixels(0.0f);
    REQUIRE(editor_overlay::clipToScreen(clip, kRectMin, kRectSize, pixels));

    const glm::float2 uv((pixels.x - kRectMin.x) / kRectW, (pixels.y - kRectMin.y) / kRectH);
    glm::float3 origin(0.0f), dir(0.0f);
    generatePickRay(cam, uv, origin, dir);

    const glm::float3 closest = origin + dir * glm::dot(worldPoint - origin, dir);
    return glm::length(closest - worldPoint);
}

} // namespace

TEST_CASE("Selection box projects inside the viewport for a framed object")
{
    const glm::float3 bbMin(-0.5f, -0.5f, -0.5f);
    const glm::float3 bbMax(0.5f, 0.5f, 0.5f);

    SUBCASE("perspective")
    {
        const std::vector<glm::float2> pixels = boxEdgePixels(makePerspective(), bbMin, bbMax);
        REQUIRE(pixels.size() == 24);
        for (const glm::float2& p : pixels)
        {
            CHECK(insideRect(p));
        }
    }

    SUBCASE("orthographic")
    {
        const std::vector<glm::float2> pixels = boxEdgePixels(makeOrthographic(1.5f, 1.5f), bbMin, bbMax);
        REQUIRE(pixels.size() == 24);
        for (const glm::float2& p : pixels)
        {
            CHECK(insideRect(p));
        }
    }
}

// An orthographic clip w is 1 everywhere, so the w test the box used to trim with
// accepted every endpoint: geometry the camera had already passed drew a box over
// the frame, in the mirrored position, as though it were still in view.
TEST_CASE("Selection box behind the camera draws nothing")
{
    // Camera sits at z = 5 looking down -Z; this box is well behind it.
    const glm::float3 bbMin(-0.5f, -0.5f, 8.0f);
    const glm::float3 bbMax(0.5f, 0.5f, 9.0f);

    SUBCASE("perspective")
    {
        CHECK(boxEdgePixels(makePerspective(), bbMin, bbMax).empty());
    }

    SUBCASE("orthographic")
    {
        CHECK(boxEdgePixels(makeOrthographic(1.5f, 1.5f), bbMin, bbMax).empty());
    }
}

TEST_CASE("Selection box straddling the near plane keeps its visible half")
{
    // Spans from in front of the camera to behind it.
    const glm::float3 bbMin(-0.5f, -0.5f, 0.0f);
    const glm::float3 bbMax(0.5f, 0.5f, 10.0f);

    SUBCASE("perspective")
    {
        const std::vector<glm::float2> pixels = boxEdgePixels(makePerspective(), bbMin, bbMax);
        CHECK(pixels.size() > 0);
        for (const glm::float2& p : pixels)
        {
            CHECK(std::isfinite(p.x));
            CHECK(std::isfinite(p.y));
        }
    }

    SUBCASE("orthographic")
    {
        const std::vector<glm::float2> pixels = boxEdgePixels(makeOrthographic(1.5f, 1.5f), bbMin, bbMax);
        CHECK(pixels.size() > 0);
        for (const glm::float2& p : pixels)
        {
            CHECK(std::isfinite(p.x));
            CHECK(std::isfinite(p.y));
        }
    }
}

// The box and the pick have to agree on where a point is, or the highlight lands
// somewhere other than what the click selected.
TEST_CASE("Projected pixel and pick ray agree, both projections")
{
    const glm::float3 points[] = { glm::float3(0.0f, 0.0f, 0.0f), glm::float3(0.6f, 0.4f, -0.3f),
                                   glm::float3(-1.0f, -0.7f, 1.0f) };

    SUBCASE("perspective")
    {
        const Camera cam = makePerspective();
        for (const glm::float3& p : points)
        {
            CHECK(pickMissDistance(cam, p) == doctest::Approx(0.0f).epsilon(1e-3));
        }
    }

    SUBCASE("orthographic")
    {
        // Authored square, rendered 4:3: magForAspect reframes, and the box has to
        // follow the same reframe the pick ray does.
        const Camera cam = makeOrthographic(2.0f, 2.0f);
        for (const glm::float3& p : points)
        {
            CHECK(pickMissDistance(cam, p) == doctest::Approx(0.0f).epsilon(1e-3));
        }
    }
}

TEST_CASE("Degenerate clip points are rejected instead of drawn as NaN")
{
    glm::float2 pixels(0.0f);
    CHECK_FALSE(editor_overlay::clipToScreen(glm::float4(1.0f, 1.0f, 1.0f, 0.0f), kRectMin, kRectSize, pixels));

    const float inf = std::numeric_limits<float>::infinity();
    CHECK_FALSE(editor_overlay::clipToScreen(glm::float4(inf, 0.0f, 0.0f, inf), kRectMin, kRectSize, pixels));
    CHECK_FALSE(editor_overlay::clipToScreen(glm::float4(std::nanf(""), 0.0f, 0.0f, 1.0f), kRectMin, kRectSize, pixels));
}
