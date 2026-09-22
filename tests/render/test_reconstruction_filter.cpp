#include <doctest/doctest.h>

#include <reconstruction_filter.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

TEST_CASE("Tent sampling covers a symmetric two-pixel width with unit weights")
{
    CHECK(sampleTent(0.0f).offset == doctest::Approx(-1.0f));
    CHECK(sampleTent(0.5f).offset == doctest::Approx(0.0f));
    CHECK(sampleTent(1.0f).offset == doctest::Approx(1.0f));

    constexpr int sampleCount = 1 << 16;
    double mean = 0.0;
    double secondMoment = 0.0;
    for (int i = 0; i < sampleCount; ++i)
    {
        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(sampleCount);
        const ReconstructionFilterSample sample = sampleTent(u);
        mean += sample.offset;
        secondMoment += sample.offset * sample.offset;
        CHECK(sample.weight == doctest::Approx(1.0f));
    }
    CHECK(mean / sampleCount == doctest::Approx(0.0).epsilon(1.0e-5));
    CHECK(secondMoment / sampleCount == doctest::Approx(1.0 / 6.0).epsilon(1.0e-4));
}

TEST_CASE("Mitchell absolute CDF covers its full two-pixel support")
{
    CHECK(mitchellAbsoluteHalfIntegral(0.0f) == doctest::Approx(0.0f));
    CHECK(mitchellAbsoluteHalfIntegral(2.0f) == doctest::Approx(367.0f / 686.0f));
}

TEST_CASE("Mitchell absolute sampling is symmetric and reconstructs a constant")
{
    constexpr int sampleCount = 1 << 16;
    double weightSum = 0.0;
    float maximumOffset = 0.0f;
    for (int i = 0; i < sampleCount; ++i)
    {
        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(sampleCount);
        const ReconstructionFilterSample sample = sampleMitchell(u);
        maximumOffset = std::max(maximumOffset, std::abs(sample.offset));
        weightSum += sample.weight;
    }
    CHECK(maximumOffset <= 2.0f);
    CHECK(weightSum / static_cast<double>(sampleCount) == doctest::Approx(1.0).epsilon(2.0e-3));

    const ReconstructionFilterSample left = sampleMitchell(0.173f);
    const ReconstructionFilterSample right = sampleMitchell(0.673f);
    CHECK(left.offset == doctest::Approx(-right.offset).epsilon(2.0e-3));
    CHECK(left.weight == doctest::Approx(right.weight));
}

TEST_CASE("Lanczos 2 absolute sampling is normalized and symmetric")
{
    CHECK(lanczos2Kernel(0.0f) == doctest::Approx(1.0f));
    CHECK(lanczos2Kernel(1.0f) == doctest::Approx(0.0f).epsilon(1.0e-6));
    CHECK(lanczos2Kernel(2.0f) == doctest::Approx(0.0f));

    constexpr int sampleCount = 1 << 18;
    double weightSum = 0.0;
    double weightedSecondMoment = 0.0;
    double weightedFourthMoment = 0.0;
    float maximumOffset = 0.0f;
    for (int i = 0; i < sampleCount; ++i)
    {
        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(sampleCount);
        const ReconstructionFilterSample sample = sampleLanczos2(u);
        maximumOffset = std::max(maximumOffset, std::abs(sample.offset));
        weightSum += sample.weight;
        const double x2 = static_cast<double>(sample.offset) * sample.offset;
        weightedSecondMoment += sample.weight * x2;
        weightedFourthMoment += sample.weight * x2 * x2;
    }
    CHECK(maximumOffset <= 2.0f);
    CHECK(weightSum / static_cast<double>(sampleCount) == doctest::Approx(1.0).epsilon(2.0e-5));
    CHECK(weightedSecondMoment / static_cast<double>(sampleCount) == doctest::Approx(0.0).epsilon(1.0e-4));
    CHECK(weightedFourthMoment / static_cast<double>(sampleCount) == doctest::Approx(-0.289179).epsilon(3.0e-3));

    const ReconstructionFilterSample left = sampleLanczos2(0.173f);
    const ReconstructionFilterSample right = sampleLanczos2(0.673f);
    CHECK(left.offset == doctest::Approx(-right.offset).epsilon(1.0e-6));
    CHECK(left.weight == doctest::Approx(right.weight));
}

TEST_CASE("Signed reconstruction is invariant to samples per launch")
{
    constexpr uint32_t sampleCount = 257u;
    auto render = [](uint32_t batchSize, auto sampleFilter) {
        float accumulated = 0.0f;
        uint32_t accumulatedSamples = 0u;
        for (uint32_t begin = 0u; begin < sampleCount; begin += batchSize)
        {
            const uint32_t count = std::min(batchSize, sampleCount - begin);
            float encoded = 0.0f;
            float currentWeight = 1.0f;
            for (uint32_t local = 0u; local < count; ++local)
            {
                const uint32_t i = begin + local;
                const float ux = (float((i * 73u) % sampleCount) + 0.5f) / float(sampleCount);
                const float uy = (float((i * 151u) % sampleCount) + 0.5f) / float(sampleCount);
                const ReconstructionFilterSample sx = sampleFilter(ux);
                const ReconstructionFilterSample sy = sampleFilter(uy);
                const float nextWeight = sx.weight * sy.weight;
                if (local != 0u)
                {
                    encoded *= reconstructionBatchScale(currentWeight, nextWeight);
                }
                currentWeight = nextWeight;
                encoded += 0.4f + 0.2f * std::sin(sx.offset * 1.7f) + 0.1f * std::cos(sy.offset * 2.3f);
            }
            const float batchMean = encoded * currentWeight / float(count);
            const float a = float(count) / float(accumulatedSamples + count);
            accumulated += (batchMean - accumulated) * a;
            accumulatedSamples += count;
        }
        return accumulated;
    };

    for (const uint32_t batchSize : { 16u, 64u })
    {
        CHECK(render(batchSize, sampleMitchell) == doctest::Approx(render(1u, sampleMitchell)).epsilon(2.0e-5));
        CHECK(render(batchSize, sampleLanczos2) == doctest::Approx(render(1u, sampleLanczos2)).epsilon(2.0e-5));
    }
}
