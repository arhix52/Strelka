#include <doctest/doctest.h>

#include <host/light_selection.h>
#include <host/emissive_mesh_distribution.h>
#include <emissive_mesh_light.h>
#include <light_alias_sampling.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <ranges>
#include <vector>

using oka::metal::analyticLightPower;
using oka::metal::binaryPowerProbability;
using oka::metal::buildLightSelectionAlias;
using oka::metal::emitterSelectionProbabilities;
using oka::metal::environmentLightPower;

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
        CHECK(entry.aliasThreshold == 0u);
        CHECK(entry.alias == i);
        CHECK(entry.pdf == doctest::Approx(1.0f / 257.0f).epsilon(1e-6));
    }

    const auto extreme = buildLightSelectionAlias({ 1.0, std::numeric_limits<double>::denorm_min() });
    REQUIRE(extreme.entries.size() == 2);
    CHECK(extreme.entries[0].pdf > 0.0f);
    CHECK(extreme.entries[1].pdf > 0.0f);
    CHECK(lightAliasSelect(2u, 1u, 0u, extreme.entries[1].aliasThreshold, extreme.entries[1].alias) == 1u);
    CHECK(double(extreme.entries[0].pdf) + double(extreme.entries[1].pdf) == doctest::Approx(1.0).epsilon(1e-7));
}

TEST_CASE("categorical bucket draw reaches every bucket beyond the float mantissa")
{
    constexpr uint32_t randomValues = 1u << 23u;
    constexpr uint32_t bucketCount = randomValues + 1u;
    std::vector<bool> oldReached(bucketCount, false);
    for (uint32_t raw = 0u; raw < randomValues; ++raw)
    {
        const float u = static_cast<float>(raw) / static_cast<float>(randomValues);
        oldReached[static_cast<uint32_t>(u * static_cast<float>(bucketCount))] = true;
    }
    CHECK(std::ranges::count(oldReached, true) == randomValues);

    std::vector<bool> reached(bucketCount, false);
    constexpr uint64_t wordCount = uint64_t{ 1 } << 32u;
    for (uint32_t bucket = 0u; bucket < bucketCount; ++bucket)
    {
        const uint64_t numerator = uint64_t(bucket) * wordCount;
        const uint32_t firstWord =
            static_cast<uint32_t>(numerator == 0u ? 0u : 1u + (numerator - 1u) / bucketCount);
        reached[lightAliasBucket(bucketCount, firstWord)] = true;
    }
    CHECK(std::ranges::count(reached, true) == bucketCount);
}

TEST_CASE("integer alias and class thresholds equal their strict-comparison masses")
{
    const auto table = buildLightSelectionAlias({ 3.0, 7.0 });
    REQUIRE(table.entries.size() == 2u);
    constexpr double states = 4294967296.0;
    for (uint32_t bucket = 0u; bucket < table.entries.size(); ++bucket)
    {
        const auto& entry = table.entries[bucket];
        if (entry.alias != bucket && entry.aliasThreshold > 0u)
        {
            CHECK(lightAliasSelect(2u, bucket, entry.aliasThreshold - 1u, entry.aliasThreshold, entry.alias) == bucket);
            CHECK(lightAliasSelect(2u, bucket, entry.aliasThreshold, entry.aliasThreshold, entry.alias) == entry.alias);
            CHECK(double(entry.aliasThreshold) / states > 0.0);
        }
    }

    const float classProbability = binaryPowerProbability(3.0, 7.0);
    const uint32_t classThreshold = discreteProbabilityThreshold(classProbability);
    CHECK(discreteThresholdProbability(classThreshold) == classProbability);
    CHECK(discreteBernoulli(classThreshold - 1u, classProbability));
    CHECK_FALSE(discreteBernoulli(classThreshold, classProbability));

    // Mutation: a real-valued float threshold is not, in general, the mass of
    // a strict comparison on the old 23-bit lattice.
    constexpr float oldThreshold = 0.3f;
    const double oldMass = std::ceil(double(oldThreshold) * double(1u << 23u)) / double(1u << 23u);
    CHECK(oldMass != doctest::Approx(double(oldThreshold)).epsilon(1e-12));
}

TEST_CASE("finite light powers normalize without overflowing their alias table")
{
    const double largest = std::numeric_limits<double>::max();
    const auto table = buildLightSelectionAlias({ largest, largest, 1.0 });

    REQUIRE(table.entries.size() == 3);
    CHECK(std::isfinite(table.totalPower));
    double sum = 0.0;
    for (const auto& entry : table.entries)
    {
        CHECK(std::isfinite(entry.pdf));
        CHECK(entry.alias < table.entries.size());
        CHECK(entry.pdf > 0.0f);
        sum += entry.pdf;
    }
    CHECK(sum == doctest::Approx(1.0).epsilon(1e-7));
    CHECK(table.entries[0].pdf == doctest::Approx(0.5f).epsilon(1e-6));
    CHECK(table.entries[1].pdf == doctest::Approx(0.5f).epsilon(1e-6));
    CHECK(table.entries[2].pdf > 0.0f);
    CHECK(table.entries[2].pdf < 1e-6f);
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
        CHECK(entry.aliasThreshold == 0u);
    }
    CHECK(lightAliasSelect(4u, 0u, 0u, table.entries[0].aliasThreshold, table.entries[0].alias) == 4u);
    CHECK(binaryPowerProbability(0.0, 3.0) == 0.0f);
    CHECK(binaryPowerProbability(3.0, 0.0) == 1.0f);
    CHECK(binaryPowerProbability(1.0, 3.0) == doctest::Approx(0.25f));
    CHECK(binaryPowerProbability(std::numeric_limits<double>::denorm_min(), 1.0) > 0.0f);
    CHECK(binaryPowerProbability(1.0, std::numeric_limits<double>::denorm_min()) < 1.0f);
}

TEST_CASE("a positive sharp distant light retains discrete selection support")
{
    oka::Scene::Light light;
    light.type = LIGHT_TYPE_DISTANT;
    light.color = glm::float4(2.0f, 1.0f, 0.5f, 1.0f);
    light.normal = glm::float4(0.0f, 0.0f, -1.0f, 0.0f);
    light.halfAngle = 0.0f;

    // Its conditional distribution is a Dirac mass, so its continuous
    // solid-angle PDF is zero. That must not also erase its outer discrete
    // selection mass when another emitter competes with it.
    CHECK(analyticLightPower(light) > 0.0);

    light.color = glm::float4(1.0f, -1000.0f, 0.0f, 1.0f);
    CHECK(analyticLightPower(light) > 0.0);
}

TEST_CASE("shared emitter hierarchy yields one complete represented marginal PMF")
{
    const auto analytic = buildLightSelectionAlias({ 1.0, 3.0 });
    const auto classes = emitterSelectionProbabilities(true, 9.0, true, analytic.totalPower, true, 2.0);

    const double environment = classes.environment;
    const double analytic0 = classes.local * classes.analyticGivenLocal * analytic.entries[0].pdf;
    const double analytic1 = classes.local * classes.analyticGivenLocal * analytic.entries[1].pdf;
    const double mesh = classes.local * classes.meshGivenLocal;

    CHECK(environment == doctest::Approx(0.6));
    CHECK(analytic0 == doctest::Approx(1.0 / 15.0));
    CHECK(analytic1 == doctest::Approx(0.2));
    CHECK(mesh == doctest::Approx(2.0 / 15.0));
    CHECK(environment + analytic0 + analytic1 + mesh == doctest::Approx(1.0).epsilon(1e-7));

    // The old OptiX mutation selected environment/local 50:50 and analytic
    // identities uniformly. It describes a different estimator distribution.
    const double oldEnvironment = 0.5;
    const double oldAnalytic0 = 0.5 * (4.0 / 6.0) * 0.5;
    CHECK(oldEnvironment != doctest::Approx(environment));
    CHECK(oldAnalytic0 != doctest::Approx(analytic0));
}

TEST_CASE("environment power proxy is shared and finite at scene-boundary cases")
{
    const double pi = std::numbers::pi_v<double>;
    CHECK(environmentLightPower(4.0 * pi, 2.0, 2.0, 0.5) == doctest::Approx(4.0 * pi * pi));
    CHECK(environmentLightPower(4.0 * pi, std::numeric_limits<double>::infinity(), 1.0, 1.0) ==
          doctest::Approx(4.0 * pi * pi));
    CHECK(environmentLightPower(4.0 * pi, 2.0, -1.0, 1.0) == 0.0);
    const double extreme = environmentLightPower(1e300, 1e300, 1e300, 1e300);
    CHECK(std::isfinite(extreme));
    CHECK(extreme > 0.0);

    const auto classes = emitterSelectionProbabilities(true, extreme, true, extreme, true, extreme);
    CHECK(classes.environment > 0.0f);
    CHECK(classes.local > 0.0f);
    CHECK(classes.meshGivenLocal > 0.0f);
    CHECK(classes.analyticGivenLocal > 0.0f);
    CHECK(classes.environment + classes.local == doctest::Approx(1.0f));
    CHECK(classes.meshGivenLocal + classes.analyticGivenLocal == doctest::Approx(1.0f));
}

TEST_CASE("a rare emitter class retains entropy for every conditional light bucket")
{
    constexpr float localProbability = 0x1p-22f;
    constexpr uint32_t lightCount = 8u;
    std::array<bool, lightCount> reached{};
    for (uint32_t raw = 0u; raw < 1u << 23u; ++raw)
    {
        const float u = static_cast<float>(raw) * 0x1p-23f;
        if (u >= localProbability)
        {
            break;
        }
        reached[static_cast<uint32_t>((u / localProbability) * static_cast<float>(lightCount))] = true;
    }
    CHECK_FALSE(std::ranges::all_of(reached, [](bool value) { return value; }));

    reached.fill(false);
    for (uint32_t bucket = 0u; bucket < lightCount; ++bucket)
    {
        const uint32_t independent = bucket * (std::numeric_limits<uint32_t>::max() / lightCount) +
                                     std::numeric_limits<uint32_t>::max() / (2u * lightCount);
        reached[lightAliasBucket(lightCount, independent)] = true;
    }
    CHECK(std::ranges::all_of(reached, [](bool value) { return value; }));
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
    constexpr double integerStateCount = 4294967296.0;
    for (size_t bucket = 0; bucket < count; ++bucket)
    {
        const auto& entry = table.entries[bucket];
        REQUIRE(entry.alias < count);
        const double bucketMass = double(discreteBucketStateCount(uint32_t(count), uint32_t(bucket))) /
                                  integerStateCount;
        const double own = entry.alias == bucket ? 1.0 : double(entry.aliasThreshold) / integerStateCount;
        represented[bucket] += bucketMass * own;
        represented[entry.alias] += bucketMass * (1.0 - own);
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
        const uint32_t bucketWord = sample << 12u;
        const uint32_t aliasWord = hash(sample);
        const uint32_t bucket = lightAliasBucket(uint32_t(table.entries.size()), bucketWord);
        const auto& entry = table.entries[bucket];
        const uint32_t selected =
            lightAliasSelect(uint32_t(table.entries.size()), bucket, aliasWord, entry.aliasThreshold, entry.alias);
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
    CHECK(lightAliasBucket(4u, std::numeric_limits<uint32_t>::max()) == 3u);
}

TEST_CASE("analytic light power accounts for emitting measure")
{
    oka::Scene::Light rect{};
    rect.type = LIGHT_TYPE_RECT;
    rect.color = glm::float4(1.0f);
    rect.normal = glm::float4(0.0f, 0.0f, -1.0f, 0.0f);
    rect.points[0] = glm::float4(1.0f, 0.5f, 0.0f, 1.0f);
    rect.points[1] = glm::float4(-1.0f, 0.5f, 0.0f, 1.0f);
    rect.points[3] = glm::float4(1.0f, -0.5f, 0.0f, 1.0f);

    oka::Scene::Light point{};
    point.type = LIGHT_TYPE_POINT;
    point.color = glm::float4(1.0f);
    point.points[0].y = -1.0f;

    CHECK(analyticLightPower(rect) == doctest::Approx(2.0 * std::numbers::pi));
    CHECK(analyticLightPower(point) == doctest::Approx(4.0 * std::numbers::pi));
}

TEST_CASE("invalid directional frames have zero selection power")
{
    oka::Scene::Light light{};
    light.color = glm::float4(1.0f);
    light.normal = glm::float4(0.0f);
    light.halfAngle = 0.1f;

    for (const int type : { LIGHT_TYPE_SPOT, LIGHT_TYPE_PROJECTOR, LIGHT_TYPE_DISTANT })
    {
        light.type = type;
        CHECK(analyticLightPower(light) == 0.0);
    }

    light.type = LIGHT_TYPE_POINT;
    light.points[0].y = 0.0f; // active IES profile needs the missing frame
    CHECK(analyticLightPower(light) == 0.0);
    light.points[0].y = -1.0f; // an isotropic point has no directional frame
    CHECK(analyticLightPower(light) > 0.0);
}

TEST_CASE("analytic light power uses transformed smooth area")
{
    oka::Scene::Light disc{};
    disc.type = LIGHT_TYPE_DISC;
    disc.color = glm::float4(1.0f);
    disc.normal = glm::float4(0.0f, 0.0f, -1.0f, 0.0f);
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

    disc.normal = glm::float4(0.0f);
    sphere.points[3] = glm::float4(0.0f);
    CHECK(analyticLightPower(disc) == 0.0);
    CHECK(analyticLightPower(sphere) == 0.0);

    sphere.points[0] = glm::float4(1e20f, 0.0f, 0.0f, 0.0f);
    sphere.points[2] = glm::float4(0.0f, 1e20f, 0.0f, 0.0f);
    sphere.points[3] = glm::float4(0.0f, 0.0f, 1e20f, 0.0f);
    CHECK(analyticLightPower(sphere) == 0.0);

    sphere.points[0] = glm::float4(1e13f, 0.0f, 0.0f, 0.0f);
    sphere.points[2] = glm::float4(0.0f, 1e13f, 0.0f, 0.0f);
    sphere.points[3] = glm::float4(0.0f, 0.0f, 1e13f, 0.0f);
    CHECK(analyticLightPower(sphere) > 0.0);
    CHECK(std::isfinite(analyticLightPower(sphere)));

    const float cosine = std::cos(0.6f);
    const float sine = std::sin(0.6f);
    sphere.points[0] = glm::float4(16.0f * cosine, 16.0f * sine, 0.0f, 0.0f);
    sphere.points[2] = glm::float4(-sine, cosine, 0.0f, 0.0f);
    sphere.points[3] = glm::float4(0.0f, 0.0f, 1.0f, 0.0f);
    CHECK(analyticLightPower(sphere) > 0.0);

    sphere.points[0] = glm::float4(1500.0f * cosine, 1500.0f * sine, 0.0f, 0.0f);
    CHECK(analyticLightPower(sphere) > 0.0);

    sphere.points[0] = glm::float4(7.1825589e-8f, 5.9514921e-8f, -2.69455853e-8f, 0.0f);
    sphere.points[2] = glm::float4(-8.47644524e-11f, 9.36010366e-11f, -1.92090233e-11f, 0.0f);
    sphere.points[3] = glm::float4(9.36574361e-5f, 2.48845143e-4f, 7.99277448e-4f, 0.0f);
    CHECK(analyticLightPower(sphere) == 0.0);

    disc.points[1] = glm::float4(std::numeric_limits<float>::max(), 0.0f, 0.0f, 1.0f);
    disc.points[2] = glm::float4(1e32f, 0.0f, 0.0f, 0.0f);
    disc.points[3] = glm::float4(0.0f, 1e-10f, 0.0f, 0.0f);
    disc.normal = glm::float4(0.0f, 0.0f, 1.0f, 0.0f);
    CHECK(analyticLightPower(disc) == 0.0);

    sphere.points[1] = disc.points[1];
    sphere.points[0] = disc.points[2];
    sphere.points[2] = disc.points[3];
    sphere.points[3] = glm::float4(0.0f, 0.0f, 1e-10f, 0.0f);
    CHECK(analyticLightPower(sphere) == 0.0);
}

TEST_CASE("emissive mesh hierarchy preserves mesh and triangle PMFs")
{
    std::vector<oka::render::EmissiveMeshBuildInput> inputs(2);
    inputs[0].instanceId = 4;
    inputs[0].geometryId = 1;
    inputs[0].trianglePowers = { 1.0, 3.0, 0.0 };
    inputs[1].instanceId = 9;
    inputs[1].geometryId = 0;
    inputs[1].trianglePowers = { 6.0 };

    const auto distribution = oka::render::buildEmissiveMeshDistribution(inputs);
    REQUIRE(distribution.meshes.size() == 2);
    REQUIRE(distribution.triangles.size() == 4);
    CHECK(distribution.totalPower == doctest::Approx(10.0));
    CHECK(distribution.meshes[0].selectionPdf == doctest::Approx(0.4));
    CHECK(distribution.meshes[1].selectionPdf == doctest::Approx(0.6));
    CHECK(distribution.triangles[0].selectionPdf == doctest::Approx(0.25));
    CHECK(distribution.triangles[1].selectionPdf == doctest::Approx(0.75));
    CHECK(distribution.triangles[2].selectionPdf == 0.0f);
    CHECK(distribution.triangles[3].selectionPdf == 1.0f);

    // The complete marginal triangle masses are 0.1, 0.3, 0, 0.6.
    CHECK(distribution.meshes[0].selectionPdf * distribution.triangles[0].selectionPdf == doctest::Approx(0.1));
    CHECK(distribution.meshes[0].selectionPdf * distribution.triangles[1].selectionPdf == doctest::Approx(0.3));
    CHECK(distribution.meshes[1].selectionPdf * distribution.triangles[3].selectionPdf == doctest::Approx(0.6));

    std::array<uint32_t, 4> counts{};
    uint32_t state = 0x8f3a12cdu;
    auto randomWord = [&state]() {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return state;
    };
    constexpr uint32_t draws = 1u << 20u;
    for (uint32_t draw = 0u; draw < draws; ++draw)
    {
        const uint32_t meshBucket =
            lightAliasBucket(static_cast<uint32_t>(distribution.meshes.size()), randomWord());
        const EmissiveMeshLight& meshEntry = distribution.meshes[meshBucket];
        const uint32_t meshId = lightAliasSelect(static_cast<uint32_t>(distribution.meshes.size()), meshBucket,
                                                 randomWord(), meshEntry.aliasThreshold, meshEntry.alias);
        REQUIRE(meshId < distribution.meshes.size());
        const EmissiveMeshLight& mesh = distribution.meshes[meshId];
        const uint32_t triangleBucket = lightAliasBucket(mesh.triangleCount, randomWord());
        const EmissiveTriangleLight& triangleEntry = distribution.triangles[mesh.triangleOffset + triangleBucket];
        const uint32_t triangleId = lightAliasSelect(
            mesh.triangleCount, triangleBucket, randomWord(), triangleEntry.aliasThreshold, triangleEntry.alias);
        REQUIRE(triangleId < mesh.triangleCount);
        ++counts[mesh.triangleOffset + triangleId];
    }
    CHECK(double(counts[0]) / draws == doctest::Approx(0.1).epsilon(0.01));
    CHECK(double(counts[1]) / draws == doctest::Approx(0.3).epsilon(0.01));
    CHECK(counts[2] == 0u);
    CHECK(double(counts[3]) / draws == doctest::Approx(0.6).epsilon(0.01));
}

TEST_CASE("emissive mesh production input retains every positive radiance channel")
{
    oka::Scene scene;
    std::vector<oka::Scene::Vertex> vertices(3);
    vertices[0].pos = { 0.0f, 0.0f, 0.0f };
    vertices[1].pos = { 1.0f, 0.0f, 0.0f };
    vertices[2].pos = { 0.0f, 1.0f, 0.0f };
    const uint32_t meshId = scene.createMesh(vertices, { 0u, 1u, 2u });

    oka::Scene::MaterialDescription material;
    // Invalid negative channels must not cancel the valid red emitter and
    // erase it from the discrete proposal support.
    material.params.emission = { 1.0f, -10.0f, 0.0f };
    material.params.emission_strength = 2.0f;
    const uint32_t materialId = scene.addMaterial(material);
    const glm::mat4 transform = glm::scale(glm::mat4(1.0f), glm::vec3(-2.0f, 3.0f, 0.5f));
    scene.createInstance(oka::Instance::Type::eMesh, meshId, materialId, transform);

    const auto powers = oka::render::emissiveTrianglePowers(
        scene, scene.getMeshes()[meshId], scene.getMaterials()[materialId], transform);
    REQUIRE(powers.size() == 1u);
    // World area is 3, two-sided projected power is 2*pi*L*A, and only the
    // positive red channel participates in the selection proxy.
    const double expected = 2.0 * std::numbers::pi * (0.2126 * 2.0) * 3.0;
    CHECK(powers[0] == doctest::Approx(expected).epsilon(1e-6));
    CHECK(powers[0] > 0.0);
}

TEST_CASE("emissive triangle sample and hit PDF agree under affine transform")
{
    const float3 object0 = make_float3(-1.0f, -0.5f, 0.0f);
    const float3 object1 = make_float3(1.0f, -0.5f, 0.0f);
    const float3 object2 = make_float3(-1.0f, 0.5f, 0.0f);
    // Non-uniform, sheared and mirrored affine image plus translation.
    auto transform = [](float3 p) {
        return make_float3(-2.0f * p.x + 0.4f * p.y + 0.3f, 0.25f * p.y - 0.2f, 0.3f * p.x + 1.5f);
    };
    const float3 p0 = transform(object0);
    const float3 p1 = transform(object1);
    const float3 p2 = transform(object2);
    const float2 uv0 = make_float2(0.0f, 0.0f);
    const float2 uv1 = make_float2(1.0f, 0.0f);
    const float2 uv2 = make_float2(0.0f, 1.0f);
    const float3 shadingPoint = make_float3(0.1f, -0.4f, -0.7f);

    for (uint32_t i = 0; i < 4096; ++i)
    {
        const float u0 = (float(i) + 0.5f) / 4096.0f;
        const float u1 = (float((i * 2654435761u) & 4095u) + 0.5f) / 4096.0f;
        const EmissiveTriangleSample sample = sampleEmissiveTriangle(p0, p1, p2, uv0, uv1, uv2, u0, u1);
        REQUIRE(sample.valid);

        const glm::dvec3 d0 = glm::dvec3(p0.x, p0.y, p0.z);
        const glm::dvec3 d1 = glm::dvec3(p1.x, p1.y, p1.z);
        const glm::dvec3 d2 = glm::dvec3(p2.x, p2.y, p2.z);
        const glm::dvec3 e1 = d1 - d0;
        const glm::dvec3 e2 = d2 - d0;
        const double area = 0.5 * std::sqrt(glm::dot(glm::cross(e1, e2), glm::cross(e1, e2)));
        REQUIRE(area > 0.0);
        CHECK(sample.areaPdf == doctest::Approx(1.0 / area).epsilon(2e-6));

        const glm::dvec3 q(sample.point.x, sample.point.y, sample.point.z);
        const glm::dvec3 s(shadingPoint.x, shadingPoint.y, shadingPoint.z);
        const glm::dvec3 delta = q - s;
        const double distanceSquared = glm::dot(delta, delta);
        const glm::dvec3 normal = glm::normalize(glm::cross(e1, e2));
        const double cosine = std::abs(glm::dot(normal, -delta / std::sqrt(distanceSquared)));
        const double oraclePdf = distanceSquared / (area * cosine);
        CHECK(emissiveTriangleSolidAnglePdf(sample.areaPdf, shadingPoint, sample.point, sample.normal) ==
              doctest::Approx(oraclePdf).epsilon(4e-6));
        CHECK(emissiveMeshMarginalSolidAnglePdf(0.8f, 0.25f, 0.4f, 0.75f, sample.areaPdf, shadingPoint, sample.point,
                                                sample.normal) == doctest::Approx(oraclePdf * 0.06).epsilon(4e-6));
        CHECK(std::isfinite(sample.areaPdf));
    }

    CHECK(emissiveMeshMarginalSolidAnglePdf(
              0.0f, 1.0f, 1.0f, 1.0f, 1.0f, shadingPoint, p0, make_float3(0.0f, 0.0f, 1.0f)) == 0.0f);
    const float extremePdf =
        emissiveMeshMarginalSolidAnglePdf(1.0f, 1.0f, 1.0f, 1.0f, std::numeric_limits<float>::max(), make_float3(0.0f),
                                          make_float3(0.0f, 0.0f, 2.0f), make_float3(0.0f, 0.0f, 1.0f));
    CHECK(std::isfinite(extremePdf));
    CHECK(extremePdf > 0.0f);
}

TEST_CASE("emissive triangle interpolation stays finite at float range")
{
    const float maxFinite = std::numeric_limits<float>::max();
    const float3 p0 = make_float3(maxFinite, 0.0f, 0.0f);
    const float3 p1 = make_float3(maxFinite, 1.0f, 0.0f);
    const float3 p2 = make_float3(maxFinite, 0.0f, 1.0f);
    const float2 uv = make_float2(maxFinite, maxFinite);
    const float u0 = 0.000610351562f;
    const float u1 = 0.00891113281f;
    const EmissiveTriangleSample sample = sampleEmissiveTriangle(p0, p1, p2, uv, uv, uv, u0, u1);
    REQUIRE(sample.valid);
    CHECK(sample.point.x == maxFinite);
    CHECK(sample.uv == uv);
    CHECK(sample.areaPdf == 2.0f);

    const float root = std::sqrt(u0);
    const float b0 = 1.0f - root;
    const float b1 = root * (1.0f - u1);
    const float b2 = root - b1;
    CHECK_FALSE(std::isfinite((b0 * p0 + b1 * p1 + b2 * p2).x));
}

TEST_CASE("emissive triangle measure survives overflowing endpoint differences")
{
    const float maxFinite = std::numeric_limits<float>::max();
    const float3 p0 = make_float3(-maxFinite, 0.0f, 0.0f);
    const float3 p1 = make_float3(maxFinite, 0.0f, 0.0f);
    const float3 p2 = make_float3(-maxFinite, std::numeric_limits<float>::min(), 0.0f);
    const float2 uv = make_float2(0.0f, 0.0f);
    const EmissiveTriangleSample sample = sampleEmissiveTriangle(p0, p1, p2, uv, uv, uv, 0.25f, 0.0f);
    REQUIRE(sample.valid);
    CHECK(sample.point.x == 0.0f);
    CHECK(sample.normal == make_float3(0.0f, 0.0f, 1.0f));

    const long double twiceArea =
        (2.0L * static_cast<long double>(maxFinite)) * static_cast<long double>(p2.y);
    CHECK(sample.areaPdf == doctest::Approx(static_cast<double>(2.0L / twiceArea)).epsilon(2e-6));
    CHECK_FALSE(std::isfinite((p1 - p0).x));

    oka::Scene scene;
    std::vector<oka::Scene::Vertex> vertices(3);
    vertices[0].pos = glm::float3(p0);
    vertices[1].pos = glm::float3(p1);
    vertices[2].pos = glm::float3(p2);
    const uint32_t meshId = scene.createMesh(vertices, { 0u, 1u, 2u });
    oka::Scene::MaterialDescription material;
    material.params.emission = glm::float3(1.0f);
    material.params.emission_strength = 1.0f;
    const auto powers = oka::render::emissiveTrianglePowers(
        scene, scene.getMeshes()[meshId], material, glm::mat4(1.0f));
    REQUIRE(powers.size() == 1u);
    CHECK(powers[0] > 0.0);
    CHECK(std::isfinite(powers[0]));
}

TEST_CASE("emissive triangle measure retains exponent-separated endpoint terms")
{
    const EmissiveTriangleMeasure normalDensity = emissiveTriangleMeasure(
        make_float3(2.69389246e33f, 3.64638875e17f, 8.27110457e16f),
        make_float3(-7.85297065e19f, -23.1444607f, 8.62724393e-39f),
        make_float3(4.73868002e17f, 1.35268463e-35f, 2.89990249e-18f));
    CHECK(normalDensity.areaPdf == doctest::Approx(6.784540066654614e-38).epsilon(2e-6));
    CHECK(normalDensity.areaPdf >= std::numeric_limits<float>::min());
    CHECK(dot(normalDensity.normal, normalDensity.normal) == doctest::Approx(1.0f).epsilon(2e-6));

    const EmissiveTriangleMeasure underflowDensity = emissiveTriangleMeasure(
        make_float3(-1.57713117e36f, 5.03585699e-21f, 2.16896687e28f),
        make_float3(-6.44862957e-5f, -3.8660869e-31f, -3.72892914e-16f),
        make_float3(1.50318351e20f, 1.41720677e-20f, -99582920.0f));
    CHECK(underflowDensity.areaPdf == 0.0f);
    CHECK(underflowDensity.normal == make_float3(0.0f));

    const EmissiveTriangleMeasure cancelledDensity = emissiveTriangleMeasure(
        make_float3(-2.96706332e25f, 3.29073524e-17f, 8.87590965e35f),
        make_float3(1.26941101e-26f, 4.70448121e-33f, 1.92276515e-23f),
        make_float3(-1.86062789e-6f, 6.83772451e-36f, -41.1517296f));
    CHECK(cancelledDensity.areaPdf == doctest::Approx(1.2101428097e-30).epsilon(2e-6));
    CHECK(dot(cancelledDensity.normal, cancelledDensity.normal) == doctest::Approx(1.0f).epsilon(2e-6));
}

TEST_CASE("emissive mesh power excludes unrepresentable area densities")
{
    oka::Scene scene;
    std::vector<oka::Scene::Vertex> vertices(3);
    vertices[0].pos = { 0.0f, 0.0f, 0.0f };
    vertices[1].pos = { 1e-20f, 0.0f, 0.0f };
    vertices[2].pos = { 0.0f, 1e-20f, 0.0f };
    const uint32_t meshId = scene.createMesh(vertices, { 0u, 1u, 2u });
    oka::Scene::MaterialDescription material;
    material.params.emission = glm::float3(1.0f);
    material.params.emission_strength = 1.0f;

    CHECK(emissiveTriangleAreaPdf(float3(vertices[0].pos), float3(vertices[1].pos), float3(vertices[2].pos)) == 0.0f);
    const auto powers = oka::render::emissiveTrianglePowers(
        scene, scene.getMeshes()[meshId], material, glm::mat4(1.0f));
    REQUIRE(powers.size() == 1u);
    CHECK(powers[0] == 0.0);

    // Mutation: double area alone is positive although 1 / area exceeds the
    // conditional float representation used by both GPU backends.
    const glm::dvec3 e1(vertices[1].pos);
    const glm::dvec3 e2(vertices[2].pos);
    CHECK(0.5 * glm::length(glm::cross(e1, e2)) > 0.0);
}

TEST_CASE("large finite emissive triangles retain area-measure support")
{
    const float3 p0 = make_float3(0.0f);
    const float3 p1 = make_float3(1e19f, 0.0f, 0.0f);
    const float3 p2 = make_float3(0.0f, 1e19f, 0.0f);
    const float2 uv = make_float2(0.0f, 0.0f);
    const EmissiveTriangleSample sample = sampleEmissiveTriangle(p0, p1, p2, uv, uv, uv, 0.25f, 0.5f);
    REQUIRE(sample.valid);
    CHECK(sample.areaPdf > 0.0f);
    CHECK(std::isfinite(sample.areaPdf));
    CHECK(sample.normal == make_float3(0.0f, 0.0f, 1.0f));
    const float pdf =
        emissiveTriangleSolidAnglePdf(sample.areaPdf, make_float3(0.0f, 0.0f, 1.0f), sample.point, sample.normal);
    CHECK(pdf > 0.0f);
    CHECK(std::isfinite(pdf));

    // Mutation: the old length(cross) squares a finite 1e38 vector and
    // overflows before the reciprocal area density can be formed.
    CHECK_FALSE(std::isfinite(length(cross(p1 - p0, p2 - p0))));
    CHECK(inverseFiniteCrossLength(p1 - p0, p2 - p0) > 0.0f);

    const EmissiveTriangleSample subnormal = sampleEmissiveTriangle(
        p0, make_float3(2e19f, 0.0f, 0.0f), make_float3(0.0f, 2e19f, 0.0f), uv, uv, uv, 0.25f, 0.5f);
    CHECK_FALSE(subnormal.valid);

    float distance = 0.0f;
    const float3 direction = finiteDirectionAndDistance(make_float3(1e20f, 1e14f, 0.0f), distance);
    CHECK(distance > 0.0f);
    CHECK(std::isfinite(distance));
    CHECK(dot(direction, direction) == doctest::Approx(1.0f));
    CHECK_FALSE(std::isfinite(length(make_float3(1e20f, 1e14f, 0.0f))));
}

TEST_CASE("emissive triangle PDF clamps rounded unit cosines")
{
    const float3 normal = make_float3(-0.408859611f, 0.248610795f, -0.878081203f);
    const float roundedCosine = dot(normal, normal);
    REQUIRE(roundedCosine > 1.0f);
    const float3 pointOnLight = -normal * 2e19f;
    const float pdf = emissiveTriangleSolidAnglePdf(1.0f, make_float3(0.0f), pointOnLight, normal);
    CHECK(pdf == std::numeric_limits<float>::max());
    CHECK(std::isfinite(pdf));
}

TEST_CASE("emissive mesh visibility ends before the traversed emitter")
{
    constexpr float offset = 1.0f / 65536.0f;
    const float3 source = make_float3(0.0f, 0.0f, offset);
    const float3 target = make_float3(0.0f, 0.0f, 1.0f - offset);
    const EmissiveVisibilitySegment segment = emissiveVisibilitySegment(source, target);
    REQUIRE(segment.valid);
    CHECK(segment.direction.x == 0.0f);
    CHECK(segment.direction.y == 0.0f);
    CHECK(segment.direction.z == 1.0f);
    CHECK(segment.maxDistance == doctest::Approx(1.0f - 2.0f * offset));

    // The old distance-minus-1e-5 segment starts at the offset source but is
    // measured from the unoffset shading point. It therefore crosses z=1 and
    // lets the sampled emitter report itself as an occluder.
    const float emitterIntersection = 1.0f - offset;
    const float oldMaxDistance = 1.0f - 1e-5f;
    CHECK(oldMaxDistance > emitterIntersection);
    CHECK(segment.maxDistance < emitterIntersection);
}

TEST_CASE("emissive triangle NEE recovers a positive one-bounce integral")
{
    const float3 p0 = make_float3(-0.5f, -0.5f, 1.0f);
    const float3 p1 = make_float3(0.5f, -0.5f, 1.0f);
    const float3 p2 = make_float3(-0.5f, 0.5f, 1.0f);
    const float2 uv = make_float2(0.0f, 0.0f);
    const float3 shadingPoint = make_float3(0.0f);
    constexpr uint32_t sampleCount = 1u << 18u;
    double estimate = 0.0;
    for (uint32_t i = 0; i < sampleCount; ++i)
    {
        const float u0 = (float(i) + 0.5f) / float(sampleCount);
        uint32_t reversed = i;
        reversed = ((reversed & 0x55555555u) << 1u) | ((reversed >> 1u) & 0x55555555u);
        reversed = ((reversed & 0x33333333u) << 2u) | ((reversed >> 2u) & 0x33333333u);
        reversed = ((reversed & 0x0f0f0f0fu) << 4u) | ((reversed >> 4u) & 0x0f0f0f0fu);
        reversed = (reversed << 24u) | ((reversed & 0xff00u) << 8u) | ((reversed >> 8u) & 0xff00u) | (reversed >> 24u);
        reversed >>= 8u;
        const float u1 = (float(reversed) + 0.5f) * (1.0f / 16777216.0f);
        const EmissiveTriangleSample sample = sampleEmissiveTriangle(p0, p1, p2, uv, uv, uv, u0, u1);
        REQUIRE(sample.valid);
        const float3 delta = sample.point - shadingPoint;
        const float3 wi = delta / length(delta);
        const float pdf = emissiveTriangleSolidAnglePdf(sample.areaPdf, shadingPoint, sample.point, sample.normal);
        REQUIRE(pdf > 0.0f);
        estimate += double(fmaxf(wi.z, 0.0f)) * M_1_PI_F / double(pdf);
    }
    estimate /= double(sampleCount);

    // Independent midpoint quadrature in barycentric coordinates.
    double reference = 0.0;
    constexpr uint32_t rows = 1024;
    for (uint32_t y = 0; y < rows; ++y)
    {
        for (uint32_t x = 0; x <= y; ++x)
        {
            const double b1 = (double(x) + 0.5) / double(rows);
            const double b2 = (double(y - x) + 0.5) / double(rows);
            if (b1 + b2 >= 1.0)
                continue;
            const double px = -0.5 + b1;
            const double py = -0.5 + b2;
            const double r2 = px * px + py * py + 1.0;
            reference += 1.0 / (std::numbers::pi * r2 * r2);
        }
    }
    reference /= double(rows * rows);

    CHECK(estimate > 0.0);
    CHECK(estimate == doctest::Approx(reference).epsilon(3e-3));
    const double legacyOmittedNee = 0.0;
    CHECK(std::abs(legacyOmittedNee - reference) > 0.05);
}
