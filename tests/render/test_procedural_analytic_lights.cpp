#include <doctest/doctest.h>

#include <analytic_light.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>

TEST_CASE("procedural sphere returns the near root")
{
    const auto hit = intersectCanonicalSphere(make_float3(0.0f, 0.0f, -3.0f), make_float3(0.0f, 0.0f, 1.0f), 0.0f, 10.0f);
    REQUIRE(hit.hit);
    CHECK(hit.distance == doctest::Approx(2.0f));
}

TEST_CASE("procedural sphere returns the far root from inside")
{
    const auto hit = intersectCanonicalSphere(make_float3(0.0f), make_float3(2.0f, 0.0f, 0.0f), 0.0f, 10.0f);
    REQUIRE(hit.hit);
    CHECK(hit.distance == doctest::Approx(0.5f));
}

TEST_CASE("procedural disc keeps its analytic boundary")
{
    CHECK(intersectCanonicalDisc(make_float3(0.999f, 0.0f, 2.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f, 3.0f).hit);
    CHECK_FALSE(intersectCanonicalDisc(make_float3(1.001f, 0.0f, 2.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f, 3.0f).hit);
}

TEST_CASE("non-uniform sphere sampling matches ellipsoid intersection")
{
    const float3 center = make_float3(1.0f, 2.0f, 3.0f);
    const float3 axisX = make_float3(2.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.0f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 0.5f);
    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 1.0f, 0.0f);
    const AnalyticLightIntersection hit = intersectAnalyticEllipsoid(
        sample.point + 2.0f * sample.normal, -sample.normal, 0.0f, 3.0f, center, axisX, axisY, axisZ);
    REQUIRE(hit.hit);
    CHECK(length(hit.point - sample.point) < 1e-5f);
    CHECK(dot(hit.normal, sample.normal) > 0.99999f);
}

TEST_CASE("non-uniform mirrored disc sampling matches ellipse intersection")
{
    const float3 center = make_float3(-1.0f, 1.0f, 2.0f);
    const float3 axisX = make_float3(-2.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 0.5f, 0.0f);
    const float3 normal =
        transformAffineNormal(axisX, axisY, make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f));
    const AnalyticLightSample sample = sampleAnalyticDisc(center, axisX, axisY, normal, 0.81f, 0.0f);
    const AnalyticLightIntersection hit =
        intersectAnalyticDisc(sample.point - 2.0f * normal, normal, 0.0f, 3.0f, center, axisX, axisY, normal);
    REQUIRE(hit.hit);
    CHECK(length(hit.point - sample.point) < 1e-5f);
    CHECK(hit.normal == normal);
}

TEST_CASE("canonical samples intersect themselves and report exact Jacobian PDFs")
{
    const float3 objectPoint = sampleCanonicalSphere(0.31f, 0.73f);
    const auto canonicalHit = intersectCanonicalSphere(3.0f * objectPoint, -2.0f * objectPoint, 0.0f, 3.0f);
    REQUIRE(canonicalHit.hit);
    CHECK(canonicalHit.distance == doctest::Approx(1.0f).epsilon(1e-5));

    const float3 axisX = make_float3(2.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 3.0f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 4.0f);
    const AnalyticLightSample sphere = sampleAnalyticEllipsoid(make_float3(0.0f), axisX, axisY, axisZ, 0.31f, 0.73f);
    const float jacobian = 24.0f * length(make_float3(objectPoint.x / 2.0f, objectPoint.y / 3.0f, objectPoint.z / 4.0f));
    CHECK(sphere.areaPdf == doctest::Approx(1.0f / (4.0f * M_PI_F * jacobian)).epsilon(2e-5));

    const AnalyticLightSample disc =
        sampleAnalyticDisc(make_float3(0.0f), axisX, axisY, make_float3(0.0f, 0.0f, -1.0f), 0.42f, 0.19f);
    CHECK(disc.areaPdf == doctest::Approx(1.0f / (6.0f * M_PI_F)).epsilon(1e-6));
}

TEST_CASE("procedural sphere preserves near and far shadow self-occlusion")
{
    const float3 center = make_float3(0.0f, 0.0f, 3.0f);
    const float3 axisX = make_float3(1.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.0f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
    const float3 origin = make_float3(0.0f);
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);
    CHECK_FALSE(
        intersectAnalyticEllipsoid(origin, direction, 0.0f, std::nextafter(2.0f, 0.0f), center, axisX, axisY, axisZ).hit);
    CHECK(intersectAnalyticEllipsoid(origin, direction, 0.0f, std::nextafter(4.0f, 0.0f), center, axisX, axisY, axisZ).hit);

    const std::filesystem::path root = std::filesystem::path(STRELKA_TEST_ASSETS_DIR).parent_path().parent_path();
    std::ifstream source(root / "src/render/metal/MetalAccelStructure.mm");
    REQUIRE(source.good());
    const std::string text((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
    CHECK(text.find("emitted.userID = curr.mLightId") != std::string::npos);
}


TEST_CASE("the quick affine factorisation agrees with the exact one it skips")
{
    // scaledAffineBasis() equilibrates the axes by powers of two before it
    // forms the determinant and the adjugate. That exists for one reason: the
    // determinant is cubic in the axis scale, so a light with axes of 1e13
    // overflows a float without it, and half a dozen tests in
    // test_light_pdf.cpp and test_lights.cpp are exactly those lights. It costs
    // twelve frexp and twelve ldexp per call, and every call that matters for a
    // frame is on a light authored in the range a scene in metres uses, where
    // nothing can overflow and the factorisation is pure overhead -- 7% of
    // kids_room, measured by removing it.
    //
    // So there are two paths now, and the risk is that they disagree: the quick
    // one is not a different formula, it is the same formula on unscaled axes,
    // and what says so is that the solve inverts the axes it was given either
    // way. This drives axis scales across the window's boundary at 1e10 and
    // 1e-10, so a change to either bound has to keep both sides inverting.
    struct Rng
    {
        uint32_t s = 0x9e3779b9u;
        float next()
        {
            s = s * 1664525u + 1013904223u;
            return float(s >> 8) * 0x1p-24f;
        }
        float sym(float k) { return (2.0f * next() - 1.0f) * k; }
    } rng;

    uint32_t quick = 0;
    uint32_t exact = 0;
    for (int i = 0; i < 4000; ++i)
    {
        // Scales from 1e-12 to 1e12: either side of the window, and across it.
        const float scale = std::pow(10.0f, rng.sym(12.0f));
        const float3 axisX = make_float3(scale * (0.4f + rng.next()), scale * rng.sym(0.3f), scale * rng.sym(0.3f));
        const float3 axisY = make_float3(scale * rng.sym(0.3f), scale * (0.4f + rng.next()), scale * rng.sym(0.3f));
        const float3 axisZ = make_float3(scale * rng.sym(0.3f), scale * rng.sym(0.3f), scale * (0.4f + rng.next()));

        const ScaledAffineBasis basis = scaledAffineBasis(axisX, axisY, axisZ);
        if (!basis.valid)
        {
            continue;
        }
        // Which path ran is visible in the basis: the quick one leaves the axes
        // alone and every exponent at zero.
        const bool tookQuickPath = basis.worldExponentX == 0 && basis.worldExponentY == 0 &&
                                   basis.worldExponentZ == 0 && basis.objectExponentX == 0 &&
                                   basis.objectExponentY == 0 && basis.objectExponentZ == 0;
        tookQuickPath ? ++quick : ++exact;

        // A point with known object coordinates, recovered through the solve.
        const float3 expected = make_float3(0.3f, -0.6f, 0.45f);
        const float3 worldOffset = expected.x * axisX + expected.y * axisY + expected.z * axisZ;
        float3 recovered;
        REQUIRE(solveAffineCoordinates(axisX, axisY, axisZ, worldOffset, recovered));
        CAPTURE(scale);
        CAPTURE(tookQuickPath);
        // Tight on purpose: a compensated solve of a well-conditioned basis
        // lands within a few ULP, so a loose bound here would accept a basis
        // that is merely close to the axes it claims to invert.
        CHECK(recovered.x == doctest::Approx(expected.x).epsilon(1e-6f));
        CHECK(recovered.y == doctest::Approx(expected.y).epsilon(1e-6f));
        CHECK(recovered.z == doctest::Approx(expected.z).epsilon(1e-6f));

        // The cofactors the basis carries are the adjugate of the basis it
        // carries, whichever path filled them in. Both rows: cofactorX does not
        // involve basis.x, so on its own it would not notice a wrong one.
        const float3 cofactorX = accurateCross(basis.y, basis.z);
        const float3 cofactorZ = accurateCross(basis.x, basis.y);
        CHECK(basis.cofactorX.x == doctest::Approx(cofactorX.x).epsilon(1e-6f));
        CHECK(basis.cofactorX.y == doctest::Approx(cofactorX.y).epsilon(1e-6f));
        CHECK(basis.cofactorX.z == doctest::Approx(cofactorX.z).epsilon(1e-6f));
        CHECK(basis.cofactorZ.x == doctest::Approx(cofactorZ.x).epsilon(1e-6f));
        CHECK(basis.cofactorZ.y == doctest::Approx(cofactorZ.y).epsilon(1e-6f));
        CHECK(basis.cofactorZ.z == doctest::Approx(cofactorZ.z).epsilon(1e-6f));
    }
    // Both paths have to have run, or this measured one of them twice.
    CHECK(quick > 500u);
    CHECK(exact > 500u);
}
