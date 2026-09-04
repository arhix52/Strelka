#include <doctest/doctest.h>

#include <restir_reservoir.h>

#include <array>
#include <cmath>
#include <random>

TEST_CASE("one ReSTIR candidate reduces to NEE")
{
    RestirReservoirState r{};
    CHECK(restirReservoirUpdate(r, 4.0f, 2.0f, 1u, 0.7f));
    CHECK(restirReservoirNormalization(r) == doctest::Approx(2.0f));
    CHECK(2.0f * restirReservoirNormalization(r) == doctest::Approx(4.0f));
}

TEST_CASE("ReSTIR weighted replacement follows candidate weights")
{
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> uniform;
    uint32_t second = 0;
    constexpr uint32_t trials = 100000;
    for (uint32_t i = 0; i < trials; ++i)
    {
        RestirReservoirState r{};
        restirReservoirUpdate(r, 1.0f, 1.0f, 1u, uniform(rng));
        second += restirReservoirUpdate(r, 3.0f, 3.0f, 1u, uniform(rng)) ? 1u : 0u;
    }
    CHECK(float(second) / float(trials) == doctest::Approx(0.75f).epsilon(0.01));
}

TEST_CASE("equal ReSTIR candidates have no material order dependence")
{
    std::mt19937 rng(91);
    std::uniform_real_distribution<float> uniform;
    std::array<uint32_t, 4> selected{};
    constexpr uint32_t trials = 80000;
    for (uint32_t i = 0; i < trials; ++i)
    {
        RestirReservoirState r{};
        uint32_t survivor = 0;
        for (uint32_t candidate = 0; candidate < selected.size(); ++candidate)
        {
            if (restirReservoirUpdate(r, 1.0f, 1.0f, 1u, uniform(rng)))
            {
                survivor = candidate;
            }
        }
        ++selected[survivor];
    }
    for (const uint32_t count : selected)
    {
        CHECK(float(count) / float(trials) == doctest::Approx(0.25f).epsilon(0.02));
    }
}

TEST_CASE("invalid ReSTIR reservoir normalizes to zero")
{
    RestirReservoirState r{};
    CHECK_FALSE(restirReservoirUpdate(r, 0.0f, 0.0f, 1u, 0.0f));
    CHECK(r.M == 1u);
    CHECK(restirReservoirNormalization(r) == 0.0f);
    CHECK(std::isfinite(restirReservoirNormalization(r)));
}

TEST_CASE("ReSTIR reservoir merge accounts for source M and current target")
{
    RestirReservoirState destination{};
    CHECK(restirReservoirUpdate(destination, 2.0f, 2.0f, 1u, 0.0f));
    const RestirReservoirState source{ 12.0f, 3.0f, 4u, RESTIR_RESERVOIR_VALID };
    const float mergeWeight = restirReservoirMergeWeight(source, 6.0f);
    CHECK(mergeWeight == doctest::Approx(24.0f));
    CHECK(restirReservoirUpdate(destination, mergeWeight, 6.0f, source.M, 0.0f));
    CHECK(destination.M == 5u);
    CHECK(destination.weightSum == doctest::Approx(26.0f));
    CHECK(restirReservoirNormalization(destination) == doctest::Approx(26.0f / 30.0f));
}

TEST_CASE("ReSTIR history is reweighted with the current target")
{
    const RestirReservoirState source{ 8.0f, 2.0f, 4u, RESTIR_RESERVOIR_VALID };
    CHECK(restirReservoirMergeWeight(source, 6.0f) == doctest::Approx(24.0f));
    CHECK(restirReservoirMergeWeight(source, source.target) == doctest::Approx(source.weightSum));
}

TEST_CASE("ReSTIR history rejects disocclusion and incompatible surfaces")
{
    CHECK(restirSurfaceCompatible(2.0f, 2.1f, 0.95f, 7u, 7u, true));
    CHECK_FALSE(restirSurfaceCompatible(2.0f, 3.0f, 0.95f, 7u, 7u, true));
    CHECK_FALSE(restirSurfaceCompatible(2.0f, 2.1f, 0.5f, 7u, 7u, true));
    CHECK_FALSE(restirSurfaceCompatible(2.0f, 2.1f, 0.95f, 7u, 8u, true));
    CHECK_FALSE(restirSurfaceCompatible(2.0f, 2.1f, 0.95f, 7u, 7u, false));
}
