#include <doctest/doctest.h>

#include <shading/fibre_geometry.h>

#include <cmath>

// ---------------------------------------------------------------------------
// Where a ray that scattered through a strand comes out.
//
// The transport side of this cannot be a unit test -- its instrument is the
// isolated-strand depth ladder in tools/feature_tests/strand_probe.py, which had
// to go flat past two bounces and instead climbed 1.00 -> 1.29. What *can* be
// pinned here is the geometry underneath it: the chord across a cylinder, and
// the two degenerate cases the closest-hit program relies on falling back to an
// ordinary surface offset.
// ---------------------------------------------------------------------------

namespace
{

// A strand along +Z of radius r, hit on its +X side.
FibreExit crossing(float3 dir, float radius = 0.5f)
{
    const float3 tangent = make_float3(0.0f, 0.0f, 1.0f);
    const float3 normal = make_float3(1.0f, 0.0f, 0.0f);
    const float3 position = make_float3(radius, 0.0f, 0.0f);
    return fibre_exit(position, tangent, normal, radius, dir);
}

bool near(float a, float b, float eps = 1e-4f)
{
    return std::fabs(a - b) < eps;
}

} // namespace

TEST_CASE("a ray straight across the strand exits on the far side")
{
    // Dead through the axis: the chord is the diameter, and the exit normal is
    // the opposite of the entry normal.
    const FibreExit e = crossing(make_float3(-1.0f, 0.0f, 0.0f));
    CHECK(e.crossed);
    CHECK(near(e.position.x, -0.5f));
    CHECK(near(e.position.y, 0.0f));
    CHECK(near(e.normal.x, -1.0f));
}

TEST_CASE("the exit point stays on the cylinder for an oblique crossing")
{
    // Any direction that enters the fibre has to leave it on the surface, which
    // is the property the offset at the exit depends on: the distance from the
    // axis must come back out as the radius.
    for (float t : { 0.2f, 0.5f, 1.0f, 2.0f, 5.0f })
    {
        const float3 dir = safe_normalize(make_float3(-1.0f, t, 0.3f));
        const FibreExit e = crossing(dir);
        CAPTURE(t);
        REQUIRE(e.crossed);
        // Radial distance from the axis (the axis runs along Z through 0).
        const float r = std::sqrt(e.position.x * e.position.x + e.position.y * e.position.y);
        CHECK(near(r, 0.5f, 1e-3f));
        // And the outward normal there is the radial direction at the exit.
        CHECK(near(e.normal.x, e.position.x / r, 1e-3f));
        CHECK(near(e.normal.y, e.position.y / r, 1e-3f));
    }
}

TEST_CASE("a ray leaving on the side it arrived from does not move")
{
    // Reflection rather than transmission: the chord is negative, and the caller
    // wants an ordinary surface offset at the hit it already has.
    const FibreExit e = crossing(make_float3(1.0f, 0.0f, 0.0f));
    CHECK_FALSE(e.crossed);
    CHECK(near(e.position.x, 0.5f));
    CHECK(near(e.normal.x, 1.0f));
}

TEST_CASE("a ray along the strand has no far wall")
{
    // Straight down the axis there is no crossing to charge for, and the chord
    // would divide by a zero-length projection on the way to saying so.
    const FibreExit e = crossing(make_float3(0.0f, 0.0f, 1.0f));
    CHECK_FALSE(e.crossed);
    CHECK(near(e.position.x, 0.5f));
}

TEST_CASE("a radius of zero is not a fibre")
{
    // A curve hit whose radius could not be recovered must degrade to the
    // surface behaviour rather than to a NaN origin.
    const FibreExit e = crossing(make_float3(-1.0f, 0.0f, 0.0f), 0.0f);
    CHECK_FALSE(e.crossed);
    CHECK(std::isfinite(e.position.x));
    CHECK(std::isfinite(e.normal.x));
}

TEST_CASE("the chord scales with the radius")
{
    // -2r(n.u) with n and u opposed is 2r, so twice the strand is twice the
    // distance travelled inside it. This is what makes absorption over the
    // crossing track the exported diameter -- the half-radius bug in
    // docs/open-defects.md doubled exactly this.
    const FibreExit thin = crossing(make_float3(-1.0f, 0.0f, 0.0f), 0.25f);
    const FibreExit thick = crossing(make_float3(-1.0f, 0.0f, 0.0f), 0.5f);
    REQUIRE(thin.crossed);
    REQUIRE(thick.crossed);
    const float thinChord = 0.25f - thin.position.x;
    const float thickChord = 0.5f - thick.position.x;
    CHECK(near(thickChord, 2.0f * thinChord, 1e-3f));
}
