#include <doctest/doctest.h>

#include <host/light_selection.h>
#include <light_alias_sampling.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

using oka::metal::analyticLightPower;
using oka::metal::binaryPowerProbability;
using oka::metal::buildLightSelectionAlias;

TEST_CASE("light selection alias table follows power and excludes zero bins")
{
    const auto table = buildLightSelectionAlias({ 1.0, 3.0, 0.0 });

    REQUIRE(table.entries.size() == 3);
    CHECK(table.totalPower == doctest::Approx(4.0));
    CHECK(table.entries[0].pdf < table.entries[1].pdf);
    CHECK(table.entries[2].pdf == 0.0f);
    CHECK(table.entries[0].pdf == doctest::Approx(0.25f));
    CHECK(table.entries[1].pdf == doctest::Approx(0.75f));
    for (const auto& entry : table.entries)
    {
        CHECK(entry.aliasProbability >= 0.0f);
        CHECK(entry.aliasProbability <= 1.0f);
        CHECK(entry.alias < table.entries.size());
    }
}

TEST_CASE("uniform and sub-float positive light weights remain reachable")
{
    const auto uniform = buildLightSelectionAlias(std::vector<double>(257, 1.0));
    REQUIRE(uniform.entries.size() == 257);
    for (size_t i = 0; i < uniform.entries.size(); ++i)
    {
        const auto& entry = uniform.entries[i];
        CHECK(entry.aliasProbability == 1.0f);
        CHECK(entry.alias == i);
        CHECK(entry.pdf == doctest::Approx(1.0f / 257.0f).epsilon(1e-6));
    }

    const auto extreme = buildLightSelectionAlias({ 1.0, std::numeric_limits<double>::denorm_min() });
    REQUIRE(extreme.entries.size() == 2);
    CHECK(extreme.entries[0].pdf > 0.0f);
    CHECK(extreme.entries[1].pdf > 0.0f);
    CHECK(lightAliasSelect(2u, 1u, 0x1p-23f, extreme.entries[1].aliasProbability, extreme.entries[1].alias) == 1u);
    CHECK(double(extreme.entries[0].pdf) + double(extreme.entries[1].pdf) == doctest::Approx(1.0).epsilon(1e-7));
}

TEST_CASE("invalid or black light powers have empty selection support")
{
    const auto table = buildLightSelectionAlias(
        { 0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() });

    REQUIRE(table.entries.size() == 4);
    CHECK(table.totalPower == doctest::Approx(0.0));
    for (const auto& entry : table.entries)
    {
        CHECK(entry.pdf == 0.0f);
        CHECK(entry.alias == std::numeric_limits<uint32_t>::max());
        CHECK(entry.aliasProbability == 0.0f);
    }
    CHECK(lightAliasSelect(4u, 0u, 0.5f, table.entries[0].aliasProbability, table.entries[0].alias) == 4u);
    CHECK(binaryPowerProbability(0.0, 3.0) == 0.0f);
    CHECK(binaryPowerProbability(3.0, 0.0) == 1.0f);
    CHECK(binaryPowerProbability(1.0, 3.0) == doctest::Approx(0.25f));
    CHECK(binaryPowerProbability(std::numeric_limits<double>::denorm_min(), 1.0) > 0.0f);
    CHECK(binaryPowerProbability(1.0, std::numeric_limits<double>::denorm_min()) < 1.0f);
}

TEST_CASE("million-light selection preserves positive support and excludes zero weights")
{
    constexpr size_t count = size_t{ 1 } << 20u;
    std::vector<double> powers(count, 1.0);
    powers[0] = 1.0e12;
    powers[count / 3] = 0.0;
    powers[count / 2] = std::numeric_limits<double>::quiet_NaN();

    const auto table = buildLightSelectionAlias(powers);
    REQUIRE(table.entries.size() == count);

    std::vector<double> represented(count, 0.0);
    const double bucketMass = 1.0 / double(count);
    for (size_t bucket = 0; bucket < count; ++bucket)
    {
        const auto& entry = table.entries[bucket];
        REQUIRE(entry.alias < count);
        represented[bucket] += bucketMass * double(entry.aliasProbability);
        represented[entry.alias] += bucketMass * (1.0 - double(entry.aliasProbability));
    }
    double pmfSum = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        const auto& entry = table.entries[i];
        CAPTURE(i);
        if (powers[i] > 0.0 && std::isfinite(powers[i]))
        {
            CHECK(represented[i] > 0.0);
            CHECK(entry.pdf > 0.0f);
        }
        else
        {
            CHECK(represented[i] == 0.0);
            CHECK(entry.pdf == 0.0f);
        }
        CHECK(double(entry.pdf) == doctest::Approx(represented[i]).epsilon(2e-7));
        pmfSum += entry.pdf;
    }
    CHECK(pmfSum == doctest::Approx(1.0).epsilon(2e-6));

    // The old flat-float mutation loses roughly one fifth of its positive tail
    // even when the host accumulates in double.
    constexpr double oldUniformMix = 0.05;
    const float dominant = float((1.0 - oldUniformMix) + oldUniformMix / double(count));
    const float floor = float(oldUniformMix / double(count));
    double oldCdf = 0.0;
    float oldPrevious = 0.0f;
    size_t oldCollapsed = 0;
    for (size_t i = 0; i < count; ++i)
    {
        oldCdf += i == 0 ? dominant : floor;
        const float endpoint = i + 1 == count ? 1.0f : float(oldCdf);
        oldCollapsed += endpoint == oldPrevious ? 1u : 0u;
        oldPrevious = endpoint;
    }
    CHECK(oldCollapsed > count / 10u);
}

TEST_CASE("light alias empirical frequencies match its represented PMF")
{
    const auto table = buildLightSelectionAlias({ 1.0, 3.0, 6.0, 0.0 });
    REQUIRE(table.entries.size() == 4);
    constexpr uint32_t sampleCount = 1u << 20u;
    uint32_t counts[4]{};
    auto hash = [](uint32_t value) {
        value ^= value >> 16u;
        value *= 0x7feb352du;
        value ^= value >> 15u;
        value *= 0x846ca68bu;
        return value ^ (value >> 16u);
    };
    for (uint32_t sample = 0; sample < sampleCount; ++sample)
    {
        const float bucketU = (float(sample) + 0.5f) / float(sampleCount);
        const float aliasU = (float(hash(sample) & 0x00ffffffu) + 0.5f) * (1.0f / 16777216.0f);
        const uint32_t bucket = lightAliasBucket(uint32_t(table.entries.size()), bucketU);
        const auto& entry = table.entries[bucket];
        const uint32_t selected =
            lightAliasSelect(uint32_t(table.entries.size()), bucket, aliasU, entry.aliasProbability, entry.alias);
        REQUIRE(selected < table.entries.size());
        ++counts[selected];
    }
    double chiSquare = 0.0;
    for (size_t i = 0; i < table.entries.size(); ++i)
    {
        const double expected = double(sampleCount) * table.entries[i].pdf;
        if (expected > 0.0)
        {
            const double error = double(counts[i]) - expected;
            chiSquare += error * error / expected;
        }
        else
        {
            CHECK(counts[i] == 0u);
        }
    }
    CHECK(chiSquare < 16.0);
    CHECK(lightAliasBucket(4u, 1.0f) == 3u);
    CHECK(lightAliasBucket(4u, std::numeric_limits<float>::quiet_NaN()) == 0u);
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
