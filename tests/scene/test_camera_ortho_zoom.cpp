#include <doctest/doctest.h>

#include <strelka/scene/camera.h>

using namespace oka;

namespace
{

Camera orthoCamera()
{
    Camera cam;
    // The framing the Isometric Kids Bedroom is authored with, which is where the
    // dead forward key was noticed.
    cam.setOrthographic(0.45f, 0.45f, 0.001f, 1000.0f);
    cam.updateViewMatrix();
    return cam;
}

void holdForward(Camera& cam, float seconds)
{
    cam.keys = {};
    cam.keys.forward = true;
    cam.update(seconds);
    cam.keys = {};
}

void holdBack(Camera& cam, float seconds)
{
    cam.keys = {};
    cam.keys.back = true;
    cam.update(seconds);
    cam.keys = {};
}

} // namespace

// The bug this pins: W and S on an orthographic camera translated it along the
// view direction, which is the one direction a parallel projection cannot see. The
// camera moved, the pose changed, nothing about the image did, and it read as dead
// input rather than as a projection doing what a projection does.
TEST_CASE("Forward and back zoom an orthographic camera instead of moving it")
{
    Camera cam = orthoCamera();
    const glm::float3 startPos = cam.position;
    const float startMag = cam.xmag;

    holdForward(cam, 0.1f);
    CHECK(cam.xmag < startMag);
    CHECK(cam.position == startPos);

    const float zoomedIn = cam.xmag;
    holdBack(cam, 0.1f);
    CHECK(cam.xmag > zoomedIn);
    CHECK(cam.position == startPos);
    // Multiplicative, so the same held time undoes itself exactly and one notch is
    // the same proportion of the frame however far in the user already is.
    CHECK(cam.xmag == doctest::Approx(startMag).epsilon(1e-5));

    // Both axes take one factor: a zoom that drifted the extents apart would
    // silently restretch the frame, and magForAspect would then reframe against an
    // aspect the camera was never authored with.
    CHECK(cam.xmag / cam.ymag == doctest::Approx(1.0f).epsilon(1e-5));
}

TEST_CASE("Forward still moves a perspective camera")
{
    Camera cam;
    cam.setPerspective(45.0f, 1.0f, 0.1f, 100.0f);
    cam.updateViewMatrix();
    const glm::float3 startPos = cam.position;
    const float startMag = cam.xmag;

    holdForward(cam, 0.1f);
    CHECK(cam.position != startPos);
    CHECK(cam.xmag == startMag);

    // And the extents are not a control on this camera type at all, so a caller
    // that reaches for them is a no-op rather than a camera that quietly reframes.
    cam.zoomOrthographic(0.5f);
    CHECK(cam.xmag == startMag);
}

// A field changing is not the same as the frame changing. What an orthographic
// camera sees is the film rectangle, which reaches both the shader (as
// orthoHalfWidth/Height) and picking through updateAspectRatio -- so the zoom is
// only real if it comes out the other end of that.
TEST_CASE("Orthographic zoom changes the film extents rays are built from")
{
    Camera cam = orthoCamera();
    const float aspect = 16.0f / 9.0f;
    cam.updateAspectRatio(aspect);

    glm::float3 originBefore, dirBefore;
    generatePickRay(cam, glm::float2(1.0f, 0.5f), originBefore, dirBefore);

    cam.zoomOrthographic(0.5f);
    cam.updateAspectRatio(aspect);

    glm::float3 originAfter, dirAfter;
    generatePickRay(cam, glm::float2(1.0f, 0.5f), originAfter, dirAfter);

    // Half the extent, so the ray through the right edge starts half as far off the
    // camera axis: the frame covers half the world it did.
    const float offsetBefore = glm::length(originBefore - cam.position);
    const float offsetAfter = glm::length(originAfter - cam.position);
    CHECK(offsetAfter == doctest::Approx(offsetBefore * 0.5f).epsilon(1e-4));
    // Parallel rays stay parallel; zoom is not a change of direction.
    CHECK(glm::length(dirAfter - dirBefore) == doctest::Approx(0.0f).epsilon(1e-5));
}

TEST_CASE("Orthographic zoom cannot collapse or invert the frame")
{
    Camera cam = orthoCamera();

    // A key held far longer than anyone means to. Extents must stay positive: a
    // zero extent is a frame that cannot be zoomed back out of, and a negative one
    // is a mirrored image that looks like broken geometry, not like a camera.
    for (int i = 0; i < 2000; ++i)
    {
        holdForward(cam, 0.1f);
    }
    CHECK(cam.xmag > 0.0f);
    CHECK(cam.ymag > 0.0f);
    CHECK(std::isfinite(cam.xmag));

    for (int i = 0; i < 4000; ++i)
    {
        holdBack(cam, 0.1f);
    }
    CHECK(std::isfinite(cam.xmag));
    CHECK(cam.xmag / cam.ymag == doctest::Approx(1.0f).epsilon(1e-5));

    // Nonsense factors are rejected rather than propagated into the projection.
    const float mag = cam.xmag;
    cam.zoomOrthographic(0.0f);
    cam.zoomOrthographic(-2.0f);
    CHECK(cam.xmag == mag);
}
