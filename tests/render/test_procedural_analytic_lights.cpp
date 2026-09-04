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
