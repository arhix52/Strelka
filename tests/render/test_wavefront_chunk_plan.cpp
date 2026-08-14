#include <doctest/doctest.h>

#include "wavefront_chunk_plan.h"

using oka::metal::makeMetal4WavefrontChunkPlan;
using oka::metal::makeWavefrontChunkGroups;
using oka::metal::makeWavefrontChunkPlan;
using oka::metal::wavefrontChunkIterations;
using oka::metal::WavefrontChunkPhase;

TEST_CASE("wavefront chunk plan preserves sample and bounce order")
{
    const auto chunks = makeWavefrontChunkPlan(2, 10, 4);

    REQUIRE(chunks.size() == 6);
    CHECK(chunks[0].sampleIndex == 0);
    CHECK(chunks[0].bounceBegin == 0);
    CHECK(chunks[0].bounceEnd == 4);
    CHECK(chunks[0].generate);
    CHECK_FALSE(chunks[0].resolve);
    CHECK(chunks[2].bounceBegin == 8);
    CHECK(chunks[2].bounceEnd == 10);
    CHECK(chunks[3].sampleIndex == 1);
    CHECK(chunks[3].generate);
    CHECK(chunks[5].resolve);
}

TEST_CASE("wavefront chunk plan still generates and resolves a zero-bounce sample")
{
    const auto chunks = makeWavefrontChunkPlan(1, 0, 4);

    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0].generate);
    CHECK(chunks[0].resolve);
}

TEST_CASE("long wavefront paths use short early and wider tail chunks")
{
    const auto chunks = makeWavefrontChunkPlan(1, 80, 8);

    REQUIRE(chunks.size() == 20);
    CHECK(chunks[0].bounceBegin == 0);
    CHECK(chunks[0].bounceEnd == 1);
    CHECK(chunks[15].bounceEnd == 16);
    CHECK(chunks[16].bounceEnd == 32);
    CHECK(chunks[19].bounceEnd == 80);

    const auto groups = makeWavefrontChunkGroups(chunks);
    REQUIRE(groups.size() == chunks.size());
    CHECK(groups[0].begin == 0);
    CHECK(groups[0].end == 1);
    CHECK(groups[15].end == 16);
    CHECK(groups[16].begin == 16);
    CHECK(groups[16].end == 17);
    CHECK(groups.back().begin == chunks.size() - 1);
    CHECK(groups.back().end == chunks.size());
}

TEST_CASE("wavefront chunk size follows pixel workload")
{
    CHECK(wavefrontChunkIterations(960, 540) == 16);
    CHECK(wavefrontChunkIterations(1920, 1080) == 5);
    CHECK(wavefrontChunkIterations(1440, 810) == 8);
    CHECK(wavefrontChunkIterations(4112, 2516) == 1);
    CHECK(wavefrontChunkIterations(0, 1080) == 1);
}

TEST_CASE("Metal 4 plan isolates early bounces and splits curve extend")
{
    const auto logical = makeWavefrontChunkPlan(1, 20, 5);
    const auto chunks = makeMetal4WavefrontChunkPlan(logical, 8, 4, 8, true);

    // Each of the first eight bounces becomes two bounded extend workloads and
    // one finish workload. The remaining logical chunks stay intact.
    REQUIRE(chunks.size() == 33);
    CHECK(chunks[0].phase == WavefrontChunkPhase::Extend);
    CHECK(chunks[0].generate);
    CHECK(chunks[0].traversalBatchBegin == 0);
    CHECK(chunks[0].traversalBatchEnd == 4);
    CHECK(chunks[1].phase == WavefrontChunkPhase::Extend);
    CHECK_FALSE(chunks[1].generate);
    CHECK(chunks[1].traversalBatchBegin == 4);
    CHECK(chunks[1].traversalBatchEnd == 8);
    CHECK(chunks[2].phase == WavefrontChunkPhase::Finish);
    CHECK(chunks[2].bounceBegin == 0);
    CHECK(chunks[2].bounceEnd == 1);
    CHECK(chunks.back().resolve);
}

TEST_CASE("Metal 4 plan isolates triangle bounces without splitting their stages")
{
    const auto logical = makeWavefrontChunkPlan(1, 10, 5);
    const auto chunks = makeMetal4WavefrontChunkPlan(logical, 8, 4, 8, false);

    REQUIRE(chunks.size() == 9);
    for (size_t i = 0; i < 8; ++i)
    {
        CHECK(chunks[i].phase == WavefrontChunkPhase::Complete);
        CHECK(chunks[i].bounceBegin == i);
        CHECK(chunks[i].bounceEnd == i + 1);
    }
    CHECK(chunks.back().bounceBegin == 8);
    CHECK(chunks.back().bounceEnd == 10);
    CHECK(chunks.back().resolve);
}

TEST_CASE("Metal 4 plan leaves small traversal workloads unchanged")
{
    const auto logical = makeWavefrontChunkPlan(2, 10, 4);
    const auto chunks = makeMetal4WavefrontChunkPlan(logical, 4, 4, 8, true);
    CHECK(chunks == logical);
}
