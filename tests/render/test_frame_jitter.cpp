#include <doctest/doctest.h>

#include "sampling_math.h"
#include <reconstruction_filter.h>
#include <temporal_reconstruction.h>

#include <cmath>

using oka::metal::frameJitter;
using oka::metal::haltonAt;
using oka::metal::metalFxMipBias;

TEST_CASE("haltonAt base 2 first samples")
{
    CHECK(haltonAt(1, 2) == doctest::Approx(0.5f));
    CHECK(haltonAt(2, 2) == doctest::Approx(0.25f));
    CHECK(haltonAt(3, 2) == doctest::Approx(0.75f));
}

TEST_CASE("frameJitter wraps phase and centres on pixel")
{
    float x = 0.0f;
    float y = 0.0f;
    frameJitter(0, 8, x, y);
    CHECK(x == doctest::Approx(haltonAt(1, 2) - 0.5f));
    CHECK(y == doctest::Approx(haltonAt(1, 3) - 0.5f));

    float x2 = 0.0f;
    float y2 = 0.0f;
    frameJitter(8, 8, x2, y2);
    CHECK(x2 == doctest::Approx(x));
    CHECK(y2 == doctest::Approx(y));

    float x3 = 0.0f;
    float y3 = 0.0f;
    frameJitter(1, 8, x3, y3);
    CHECK(x3 == doctest::Approx(haltonAt(2, 2) - 0.5f));
    CHECK(y3 == doctest::Approx(haltonAt(2, 3) - 0.5f));
}

TEST_CASE("frameJitter phaseCount 0 treated as 1")
{
    float x = 0.0f;
    float y = 0.0f;
    frameJitter(5, 0, x, y);
    CHECK(x == doctest::Approx(haltonAt(1, 2) - 0.5f));
    CHECK(y == doctest::Approx(haltonAt(1, 3) - 0.5f));
}

TEST_CASE("camera jitter and MetalFX offset use opposite signs")
{
    const float jitter = 0.125f;
    const float cameraSample = strelkaCameraSampleCoordinate(jitter);
    CHECK(cameraSample == doctest::Approx(0.625f));
    CHECK((cameraSample - 0.5f) + strelkaMetalFxJitterOffset(jitter) == doctest::Approx(0.0f));
}

TEST_CASE("static projection has zero dejittered screen motion")
{
    const float jitter = -0.1875f;
    const float pixelCenter = 42.5f;
    const float projectedPixel = pixelCenter + jitter;
    CHECK(strelkaScreenMotionAxis(projectedPixel, pixelCenter, jitter) == doctest::Approx(0.0f));
}

TEST_CASE("MetalFX mip bias follows spatial and temporal reconstruction rules")
{
    CHECK(metalFxMipBias(960, 1920, false) == doctest::Approx(-1.0f));
    CHECK(metalFxMipBias(960, 1920, true) == doctest::Approx(-2.0f));
    CHECK(metalFxMipBias(1920, 1920, true) == doctest::Approx(-1.0f));
}

TEST_CASE("reconstruction filters preserve constant radiance")
{
    constexpr int sampleCount = 20000;
    for (unsigned int filter = RECONSTRUCTION_FILTER_BOX; filter <= RECONSTRUCTION_FILTER_BLACKMAN_HARRIS; ++filter)
    {
        float meanWeight = 0.0f;
        for (int i = 0; i < sampleCount; ++i)
            meanWeight += reconstructionFilterSample(filter, (i + 0.5f) / sampleCount).weight;
        CHECK(meanWeight / sampleCount == doctest::Approx(1.0f).epsilon(2.0e-4f));
    }

    const auto box = reconstructionFilterSample(RECONSTRUCTION_FILTER_BOX, 0.25f);
    CHECK(box.offset == doctest::Approx(-0.25f));
    CHECK(box.weight == doctest::Approx(1.0f));
    CHECK(reconstructionFilterSample(RECONSTRUCTION_FILTER_BLACKMAN_HARRIS, 0.25f).offset !=
          doctest::Approx(box.offset));
    CHECK(reconstructionFilterSample(RECONSTRUCTION_FILTER_MITCHELL, 0.499f).weight < 0.0f);
}

TEST_CASE("reconstruction filters gather across neighboring pixel cells")
{
    constexpr int sampleCount = 200000;
    const auto stepEdge = [](unsigned int filter, float pixelCenter)
    {
        float value = 0.0f;
        for (int i = 0; i < sampleCount; ++i)
        {
            const auto sample = reconstructionFilterSample(filter, (i + 0.5f) / sampleCount);
            value += sample.weight * (pixelCenter + sample.offset >= 0.0f ? 1.0f : 0.0f);
        }
        return value / sampleCount;
    };

    CHECK(stepEdge(RECONSTRUCTION_FILTER_BOX, -1.0f) == doctest::Approx(0.0f));
    CHECK(stepEdge(RECONSTRUCTION_FILTER_BOX, 0.0f) == doctest::Approx(0.5f));
    CHECK(stepEdge(RECONSTRUCTION_FILTER_BOX, 1.0f) == doctest::Approx(1.0f));
    CHECK(stepEdge(RECONSTRUCTION_FILTER_MITCHELL, -1.0f) < -0.01f);
    CHECK(stepEdge(RECONSTRUCTION_FILTER_MITCHELL, 1.0f) > 1.01f);
    CHECK(stepEdge(RECONSTRUCTION_FILTER_BLACKMAN_HARRIS, -1.0f) > 0.03f);
    CHECK(stepEdge(RECONSTRUCTION_FILTER_BLACKMAN_HARRIS, 1.0f) < 0.97f);
}
