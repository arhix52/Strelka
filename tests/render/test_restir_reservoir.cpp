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

TEST_CASE("initial RIS remains the NEE estimator")
{
    RestirReservoirState r{};
    CHECK(restirReservoirUpdate(r, 4.0f, 2.0f, 1u, 0.0f));
    CHECK_FALSE(restirReservoirUpdate(r, 12.0f, 6.0f, 1u, 1.0f));
    CHECK(r.target * restirReservoirNormalization(r) == doctest::Approx((4.0f + 12.0f) / 2.0f));
}

TEST_CASE("basic normalization merges different source targets")
{
    RestirReservoirState merged{ 28.0f, 1.0f, 8u, RESTIR_RESERVOIR_VALID };
    const float sourceTargetSum = 4u * 4.0f + 4u * 2.0f;
    restirReservoirApplyBasicNormalization(merged, 2.0f, sourceTargetSum);
    CHECK(restirReservoirNormalization(merged) == doctest::Approx(28.0f * 2.0f / sourceTargetSum));
}

TEST_CASE("basic normalization excludes a zero-target source")
{
    RestirReservoirState merged{ 12.0f, 3.0f, 6u, RESTIR_RESERVOIR_VALID };
    const float sourceTargetSum = 2u * 3.0f + 4u * 0.0f;
    restirReservoirApplyBasicNormalization(merged, 3.0f, sourceTargetSum);
    CHECK(restirReservoirNormalization(merged) == doctest::Approx(2.0f));
}

TEST_CASE("temporal basic normalization uses previous-frame source target")
{
    RestirReservoirState selectedPrevious{ 15.0f, 2.0f, 5u, RESTIR_RESERVOIR_VALID };
    const float currentTarget = 2.0f;
    const float previousTarget = 0.5f;
    const float sourceTargetSum = currentTarget + 4u * previousTarget;
    restirReservoirApplyBasicNormalization(selectedPrevious, previousTarget, sourceTargetSum);
    CHECK(restirReservoirNormalization(selectedPrevious) ==
          doctest::Approx(15.0f * previousTarget / (currentTarget * sourceTargetSum)));
    CHECK(restirReservoirNormalization(selectedPrevious) !=
          doctest::Approx(15.0f * currentTarget / (currentTarget * sourceTargetSum)));
}

TEST_CASE("basic correction removes support shift retained by off")
{
    // RTXDI DI spatial BASIC:
    // https://github.com/NVIDIA-RTX/RTXDI-Library/blob/main/Include/Rtxdi/DI/SpatialResampling.hlsli#L242-L296
    // Two M=2 sources have disjoint target support over two unit-contribution samples.
    RestirReservoirState merged{};
    CHECK(restirReservoirUpdate(merged, 2.0f, 1.0f, 2u, 0.0f));
    CHECK_FALSE(restirReservoirUpdate(merged, 2.0f, 1.0f, 2u, 1.0f));
    const float off = restirReservoirNormalization(merged);
    RestirReservoirState basic = merged;
    restirReservoirApplyBasicNormalization(basic, 1.0f, 2.0f);
    CHECK(off == doctest::Approx(1.0f));
    CHECK(restirReservoirNormalization(basic) == doctest::Approx(2.0f));
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

TEST_CASE("ReSTIR M limit preserves reservoir normalization")
{
    RestirReservoirState reservoir{ 400.0f, 2.0f, 200u, RESTIR_RESERVOIR_VALID };
    const float before = restirReservoirNormalization(reservoir);
    restirReservoirLimitM(reservoir, 50u);
    CHECK(reservoir.M == 50u);
    CHECK(reservoir.weightSum == doctest::Approx(100.0f));
    CHECK(restirReservoirNormalization(reservoir) == doctest::Approx(before));
}

TEST_CASE("ReSTIR temporal history M clamp is bias-mode independent")
{
    const RestirReservoirState expanded{ 1024.0f, 2.0f, 512u, RESTIR_RESERVOIR_VALID };
    for (const unsigned int biasMode : { 0u, 1u, 2u })
    {
        RestirReservoirState history = expanded;
        restirReservoirLimitHistoryM(history, 2u, 8u);
        CAPTURE(biasMode);
        CHECK(history.M == 16u);
        CHECK(restirReservoirNormalization(history) == doctest::Approx(restirReservoirNormalization(expanded)));
    }
}

TEST_CASE("temporal ReSTIR advances candidate sequence across accumulation resets")
{
    CHECK(restirSampleSequenceBase(false, true, false, 17u, 1u, 0u) == 0u);
    CHECK(restirSampleSequenceBase(false, true, true, 17u, 1u, 0u) == 17u);
    CHECK(restirSampleSequenceBase(false, true, true, 18u, 2u, 0u) == 36u);
}

TEST_CASE("initial visibility discard keeps reservoir population")
{
    RestirReservoirState reservoir{ 8.0f, 2.0f, 7u, RESTIR_RESERVOIR_VALID | 3u };
    restirReservoirDiscardSample(reservoir);
    CHECK(reservoir.M == 7u);
    CHECK((reservoir.ageAndFlags & RESTIR_RESERVOIR_VALID) == 0u);
    CHECK(restirReservoirNormalization(reservoir) == 0.0f);
}

TEST_CASE("ray-traced normalization removes an occluded source from BASIC support")
{
    // RTXDI SpatialResampling.hlsli, BASIC/RAY_TRACED normalization:
    // https://github.com/NVIDIA-RTX/RTXDI-Library/blob/main/Include/Rtxdi/DI/SpatialResampling.hlsli
    const float target[2][2] = { { 1.0f, 4.0f }, { 1.0f, 2.0f } };
    const float visibility[2][2] = { { 1.0f, 1.0f }, { 0.0f, 1.0f } };
    const RestirReservoirState merged{ 2.0f, 1.0f, 2u, RESTIR_RESERVOIR_VALID };
    const float off = restirReservoirNormalization(merged);
    RestirReservoirState basic = merged;
    restirReservoirApplyBasicNormalization(basic, target[0][0], target[0][0] + target[1][0]);
    RestirReservoirState rayTraced = merged;
    restirReservoirApplyBasicNormalization(
        rayTraced, target[0][0], target[0][0] * visibility[0][0] + target[1][0] * visibility[1][0]);
    CHECK(off == doctest::Approx(1.0f));
    CHECK(restirReservoirNormalization(basic) == doctest::Approx(1.0f));
    CHECK(restirReservoirNormalization(rayTraced) == doctest::Approx(2.0f));
    CHECK(target[0][1] * visibility[0][1] + target[1][1] * visibility[1][1] == doctest::Approx(6.0f));
}

TEST_CASE("compatible invalid ReSTIR source still contributes M")
{
    RestirReservoirState merged{ 4.0f, 2.0f, 1u, RESTIR_RESERVOIR_VALID };
    const RestirReservoirState invalidSource{ 0.0f, 0.0f, 4u, 0u };
    CHECK_FALSE(
        restirReservoirUpdate(merged, restirReservoirMergeWeight(invalidSource, 2.0f), 0.0f, invalidSource.M, 0.5f));
    CHECK(merged.M == 5u);
    restirReservoirApplyBasicNormalization(merged, 2.0f, 1u * 2.0f + 4u * 3.0f);
    CHECK(restirReservoirNormalization(merged) == doctest::Approx(4.0f / 14.0f));
}

TEST_CASE("ReSTIR sample key preserves category and stable ID")
{
    RestirLightSample sample{};
    sample.typeAndLightId = restirSampleKey(RESTIR_SAMPLE_EMISSIVE_TRIANGLE, 1234567u);
    CHECK(restirSampleType(sample) == RESTIR_SAMPLE_EMISSIVE_TRIANGLE);
    CHECK(restirSampleLightId(sample) == 1234567u);
    CHECK(sizeof(RestirLightSample) == 16);
}
