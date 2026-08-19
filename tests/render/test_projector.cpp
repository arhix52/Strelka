#include <doctest/doctest.h>

// The frustum maths both backends compile into their shaders. Scalars in, place
// in the image out -- see the note at the top of projector.h for why the texture
// fetch is not here and cannot be.
#include <projector.h>

// The host mirror of projectorSolidAngle(), which bakes a projector's Watts.
// The last case in this file is what keeps the copy honest.
#include <strelka/scene/glm_wrapper.hpp>
#include <strelka/scene/light_desc.h>

#include <cmath>

static constexpr float kPi = 3.14159265358979323846f;

// A 90 degree horizontal field on a square frame: the pyramid whose apex angle
// makes six of it fill the sphere. Used by several cases below.
static constexpr float kHalfFov90 = 0.25f * kPi;

TEST_CASE("the beam axis lands in the middle of the frame")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 1.0f);
    const ProjectorSample s = projectorProject(0.0f, 0.0f, 1.0f, tx, ty, 0.0f);

    CHECK(s.inside);
    CHECK(s.u == doctest::Approx(0.5f));
    CHECK(s.v == doctest::Approx(0.5f));
    CHECK(s.falloff == doctest::Approx(1.0f));
}

TEST_CASE("the frame is upright: +X is to the right and +Y is the top row")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 1.0f);

    // Half way to the right edge at unit depth.
    const ProjectorSample right = projectorProject(0.5f * tx, 0.0f, 1.0f, tx, ty, 0.0f);
    CHECK(right.inside);
    CHECK(right.u == doctest::Approx(0.75f));
    CHECK(right.v == doctest::Approx(0.5f));

    // The light's up axis has to reach the *top* of the image, which is v = 0:
    // a decoder hands back the top row first. Get this backwards and the image
    // is not obviously wrong -- it is a projector mounted upside down, which is
    // a thing people do on purpose.
    const ProjectorSample up = projectorProject(0.0f, 0.5f * ty, 1.0f, tx, ty, 0.0f);
    CHECK(up.inside);
    CHECK(up.v == doctest::Approx(0.25f));
}

TEST_CASE("a direction outside the rectangle throws nothing")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 1.0f);

    // Just past the corner in x, still inside in y.
    const ProjectorSample out = projectorProject(1.01f * tx, 0.0f, 1.0f, tx, ty, 0.0f);
    CHECK_FALSE(out.inside);
    CHECK(out.falloff == doctest::Approx(0.0f));

    // Exactly on the border is still the image: the sampler clamps there, and a
    // strict test would leave a one-texel gap at every edge.
    const ProjectorSample edge = projectorProject(tx, 0.0f, 1.0f, tx, ty, 0.0f);
    CHECK(edge.inside);
    CHECK(edge.u == doctest::Approx(1.0f));
}

TEST_CASE("nothing is thrown backwards out of the lens")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 1.0f);

    // Behind the projector. Not a smooth term that happens to reach zero -- the
    // perspective divide would fold this direction onto a perfectly plausible
    // place in the image, and the light would throw a mirrored copy of the frame
    // out of its own back.
    CHECK_FALSE(projectorProject(0.0f, 0.0f, -1.0f, tx, ty, 0.0f).inside);
    CHECK_FALSE(projectorProject(0.1f, 0.1f, -1.0f, tx, ty, 0.0f).inside);
    // Exactly sideways, where the divide is a division by zero.
    CHECK_FALSE(projectorProject(1.0f, 0.0f, 0.0f, tx, ty, 0.0f).inside);
}

TEST_CASE("the aspect ratio shapes the frame, and 16:9 is wider than it is tall")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 16.0f / 9.0f);
    CHECK(ty == doctest::Approx(tx * 9.0f / 16.0f));

    // A direction that clears the vertical edge of a square frame is outside a
    // 16:9 one at the same field of view.
    const float y = 0.9f * tx;
    CHECK(projectorProject(0.0f, y, 1.0f, tx, tx, 0.0f).inside);
    CHECK_FALSE(projectorProject(0.0f, y, 1.0f, tx, ty, 0.0f).inside);
}

TEST_CASE("the frame grows linearly with throw distance")
{
    // What makes a projector a projector: the image on a wall twice as far away
    // is twice as wide, and the same texel is at the same place in it.
    const float tx = projectorTanHalfX(0.3f);
    const float ty = projectorTanHalfY(tx, 16.0f / 9.0f);

    const ProjectorSample near = projectorProject(0.4f * tx, 0.2f * ty, 1.0f, tx, ty, 0.0f);
    const ProjectorSample far = projectorProject(0.8f * tx, 0.4f * ty, 2.0f, tx, ty, 0.0f);
    CHECK(near.inside);
    CHECK(far.inside);
    CHECK(far.u == doctest::Approx(near.u));
    CHECK(far.v == doctest::Approx(near.v));
}

TEST_CASE("edge softness fades the border and leaves the middle alone")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 1.0f);
    const float softness = 0.25f;

    // Well inside the feathered band: untouched.
    CHECK(projectorProject(0.5f * tx, 0.0f, 1.0f, tx, ty, softness).falloff == doctest::Approx(1.0f));
    // At the border: gone.
    CHECK(projectorProject(tx, 0.0f, 1.0f, tx, ty, softness).falloff == doctest::Approx(0.0f));
    // Half way through the band: the smoothstep's midpoint.
    CHECK(projectorProject(0.875f * tx, 0.0f, 1.0f, tx, ty, softness).falloff == doctest::Approx(0.5f));

    // Zero softness is a crisp edge, which is what a focused projector has.
    CHECK(projectorProject(0.999f * tx, 0.0f, 1.0f, tx, ty, 0.0f).falloff == doctest::Approx(1.0f));
}

TEST_CASE("six square pyramids of 90 degrees fill the sphere")
{
    // The one solid angle with an answer that can be checked without trusting
    // the formula: the six faces of a cube seen from its centre partition 4pi.
    const float tx = projectorTanHalfX(kHalfFov90);
    const float omega = projectorSolidAngle(tx, projectorTanHalfY(tx, 1.0f));
    CHECK(6.0f * omega == doctest::Approx(4.0f * kPi).epsilon(1e-5));
}

TEST_CASE("a narrow pyramid approaches the product of its angular extents")
{
    // Small angles: Omega -> 4 tan(a) tan(b), the area of the frame at unit
    // depth. A cone's solid angle is not this number, which is the whole reason
    // projectorSolidAngle() exists.
    const float tx = projectorTanHalfX(0.02f);
    const float ty = projectorTanHalfY(tx, 16.0f / 9.0f);
    CHECK(projectorSolidAngle(tx, ty) == doctest::Approx(4.0f * tx * ty).epsilon(1e-4));

    const float cone = 4.0f * kPi * std::sin(0.01f) * std::sin(0.01f);
    CHECK(cone > 1.35f * projectorSolidAngle(tx, ty));
}

TEST_CASE("the host's solid angle is the shader's, to the last bit")
{
    // oka::projectorSolidAngleFromFov() divides a projector's Watts in
    // scene/light_desc.h and projectorSolidAngle() is what the shader spreads
    // the image over. They are two copies of one number -- the scene header
    // cannot include the shader one, see the comment on the host copy -- so a
    // drift between them scales every projector in the scene and nothing else
    // would catch it.
    const float aspects[] = { 1.0f, 4.0f / 3.0f, 16.0f / 9.0f, 2.39f, 0.75f };
    const float halfFovs[] = { 0.01f, 0.1f, 0.3f, kHalfFov90, 1.5f };
    for (const float aspect : aspects)
    {
        for (const float halfFov : halfFovs)
        {
            const float tx = projectorTanHalfX(halfFov);
            const float shader = projectorSolidAngle(tx, projectorTanHalfY(tx, aspect));
            CHECK(oka::projectorSolidAngleFromFov(halfFov, aspect) == doctest::Approx(shader));
        }
    }
}

TEST_CASE("a degenerate field of view does not produce infinity or zero area")
{
    // 90 degrees is where the tangent blows up and the pyramid stops being one.
    // Clamped rather than guarded at every call site, so the bake divides by
    // something finite whatever a sidecar or a slider hands over.
    const float wide = projectorTanHalfX(0.5f * kPi);
    CHECK(std::isfinite(wide));
    CHECK(std::isfinite(projectorSolidAngle(wide, projectorTanHalfY(wide, 1.0f))));
    CHECK(projectorSolidAngle(wide, projectorTanHalfY(wide, 1.0f)) > 0.0f);

    const float narrow = projectorTanHalfX(0.0f);
    CHECK(narrow > 0.0f);
    CHECK(projectorSolidAngle(narrow, projectorTanHalfY(narrow, 1.0f)) > 0.0f);
}
