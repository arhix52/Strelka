#include <doctest/doctest.h>

#include <host/light_selection.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

using oka::metal::analyticLightPower;
using oka::metal::buildLightSelectionCdf;

TEST_CASE("light selection CDF follows power and keeps a uniform floor")
{
    const auto table = buildLightSelectionCdf({ 1.0, 3.0, 0.0 });

    REQUIRE(table.entries.size() == 3);
    CHECK(table.totalPower == doctest::Approx(4.0));
    CHECK(table.entries[0].pdf < table.entries[1].pdf);
    CHECK(table.entries[2].pdf > 0.0f);
    CHECK(table.entries[0].cdf == doctest::Approx(table.entries[0].pdf));
    CHECK(table.entries[1].cdf == doctest::Approx(table.entries[0].pdf + table.entries[1].pdf));
    CHECK(table.entries.back().cdf == doctest::Approx(1.0f));
}

TEST_CASE("invalid or black light powers fall back to uniform selection")
{
    const auto table = buildLightSelectionCdf(
        { 0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() });

    REQUIRE(table.entries.size() == 4);
    CHECK(table.totalPower == doctest::Approx(0.0));
    for (const auto& entry : table.entries)
    {
        CHECK(entry.pdf == doctest::Approx(0.25f));
    }
    CHECK(table.entries.back().cdf == doctest::Approx(1.0f));
}

TEST_CASE("analytic light power accounts for emitting measure")
{
    oka::Scene::Light rect{};
    rect.type = LIGHT_TYPE_RECT;
    rect.color = glm::float4(1.0f);
    rect.points[0] = glm::float4(1.0f, 0.5f, 0.0f, 1.0f);
    rect.points[1] = glm::float4(-1.0f, 0.5f, 0.0f, 1.0f);
    rect.points[3] = glm::float4(1.0f, -0.5f, 0.0f, 1.0f);

    oka::Scene::Light point{};
    point.type = LIGHT_TYPE_POINT;
    point.color = glm::float4(1.0f);

    CHECK(analyticLightPower(rect) == doctest::Approx(2.0 * std::numbers::pi));
    CHECK(analyticLightPower(point) == doctest::Approx(4.0 * std::numbers::pi));
}

TEST_CASE("analytic light power uses transformed smooth area")
{
    oka::Scene::Light disc{};
    disc.type = LIGHT_TYPE_DISC;
    disc.color = glm::float4(1.0f);
    disc.points[2] = glm::float4(-1.0f, 0.0f, 0.0f, 0.0f);
    disc.points[3] = glm::float4(0.0f, 1.5f, 0.0f, 0.0f);
    const double discArea = analyticDiscArea(float3(disc.points[2]), float3(disc.points[3]));
    CHECK(analyticLightPower(disc) == doctest::Approx(std::numbers::pi * discArea).epsilon(1e-6));

    oka::Scene::Light sphere{};
    sphere.type = LIGHT_TYPE_SPHERE;
    sphere.color = glm::float4(1.0f);
    sphere.points[0] = glm::float4(-1.0f, 0.0f, 0.0f, 0.0f);
    sphere.points[1] = glm::float4(0.3f, -0.2f, 0.5f, 1.0f);
    sphere.points[2] = glm::float4(0.0f, 1.5f, 0.0f, 0.0f);
    sphere.points[3] = glm::float4(0.25f, 0.0f, 2.0f, 0.0f);
    const double sphereArea =
        analyticEllipsoidSurfaceArea(float3(sphere.points[0]), float3(sphere.points[2]), float3(sphere.points[3]));
    CHECK(analyticLightPower(sphere) == doctest::Approx(std::numbers::pi * sphereArea).epsilon(1e-6));
}
