#include <doctest/doctest.h>

#include "sharc_grid.h"
#include "sharc_grid_size.h"
#include <sharc_query_eligibility.h>

#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <vector>
#include <numbers>

using namespace oka::sharc;

TEST_CASE("Metal SHARC voxel size follows its requested perspective footprint")
{
    constexpr float fov = 0.0449924f;
    constexpr uint32_t height = 540u;
    constexpr float requestedPixels = 4.0f;
    constexpr int32_t levelBias = 16;
    const float pixelAngle = 2.0f * std::tan(fov * 0.5f) / static_cast<float>(height);
    const float baseSize = oka::metal::sharcBaseSizeForPerspective(fov, height, requestedPixels);
    CHECK(oka::metal::sharcBaseSizeForPerspective(fov, height * 2u, requestedPixels) == doctest::Approx(baseSize * 0.5f));

    for (const float distance : { 0.05f, 0.25f, 0.75f, 1.0f, 3.0f, 10.0f, 100.0f })
    {
        const float voxelSize = oka::metal::sharcVoxelSizeForDistance(distance, baseSize, levelBias);
        const float projectedPixels = voxelSize / (pixelAngle * distance);
        CHECK(projectedPixels <= requestedPixels * 1.0001f);
        CHECK(projectedPixels > requestedPixels * 0.5f * 0.9999f);
    }
}

TEST_CASE("SHARC only reads receivers represented by its angular bandwidth")
{
    constexpr float minimumRoughness = 0.4f;

    CHECK(sharcReceiverCacheEligible(1.0f, 0.0f, false, minimumRoughness));
    CHECK(sharcReceiverCacheEligible(minimumRoughness, 0.0f, false, minimumRoughness));
    CHECK_FALSE(sharcReceiverCacheEligible(0.399f, 0.0f, false, minimumRoughness));

    // Even rough glass is not addressable by position + normal alone: the
    // answer depends on side and medium state, neither of which is in the key.
    CHECK_FALSE(sharcReceiverCacheEligible(1.0f, 0.001f, false, minimumRoughness));
    CHECK_FALSE(sharcReceiverCacheEligible(1.0f, 0.0f, true, minimumRoughness));
}

TEST_CASE("SHARC receiver roughness follows the narrowest active layer")
{
    float roughness = 1.0f;
    roughness = sharcReceiverLobeRoughness(roughness, 0.6f, 1.0f);
    roughness = sharcReceiverLobeRoughness(roughness, 0.2f, 0.0f);
    CHECK(roughness == doctest::Approx(0.6f));

    roughness = sharcReceiverLobeRoughness(roughness, 0.3f, 0.25f);
    CHECK(roughness == doctest::Approx(0.3f));
}

TEST_CASE("a voxel subtends the same angle wherever it is")
{
    // The whole reason the size follows the distance: one setting has to mean
    // the same thing in a room and in a forest. baseSize is the world size of a
    // pixel at unit distance times the pixels a voxel should span, so
    // size/distance is that product to within the power-of-two quantisation.
    const float baseSize = 4.0f * (2.0f * std::tan(0.3926991f) / 512.0f); // 45 deg, 512 px, 4 px voxels
    // Integer induction: a float loop counter both drifts and trips
    // cert-flp30-c / clang-analyzer-security.FloatLoopCounter.
    for (int step = 0; step < 27; ++step)
    {
        const float distance = 0.5f * std::pow(1.3f, (float)step);
        const Voxel v = voxelForDistance(distance, baseSize);
        const float angle = v.size / distance;
        // Quantising to powers of two can only halve or hold the ideal, never
        // more, so the subtended angle stays inside one octave of the target.
        CHECK(angle <= baseSize * 1.0001f);
        CHECK(angle > baseSize * 0.5f * 0.9999f);
    }
}

TEST_CASE("the level is the exponent of the size, exactly")
{
    for (int step = 0; step < 26; ++step)
    {
        const float distance = 0.01f * std::pow(1.7f, (float)step);
        const Voxel v = voxelForDistance(distance, 0.01f);
        CHECK(v.size == std::exp2((float)v.level));
        // Powers of two, so a point near a level boundary lands in one voxel or
        // the other rather than smearing across both.
        CHECK(std::exp2(std::floor(std::log2(v.size))) == v.size);
    }
}

TEST_CASE("a degenerate distance still yields a usable voxel")
{
    // A shading point at the eye, or a base size of zero from a camera nobody
    // configured. Neither may produce a zero size: the coordinate divides by it.
    for (const float baseSize : { 0.0f, 1e-30f, 1.0f })
    {
        const Voxel v = voxelForDistance(0.0f, baseSize);
        CHECK(v.size > 0.0f);
        CHECK(std::isfinite(v.size));
        CHECK(std::isfinite(voxelCoordinate(1.0f, v.size) * 1.0f));
    }
}

TEST_CASE("voxel coordinates tile the axis without a gap or an overlap at zero")
{
    // floor, not truncation. Truncating puts [-1, 1) into one cell twice as wide
    // as every other, which is a seam through the origin of every scene.
    const float size = 0.5f;
    CHECK(voxelCoordinate(-0.6f, size) == -2);
    CHECK(voxelCoordinate(-0.5f, size) == -1);
    CHECK(voxelCoordinate(-0.4f, size) == -1);
    CHECK(voxelCoordinate(-0.1f, size) == -1);
    CHECK(voxelCoordinate(0.0f, size) == 0);
    CHECK(voxelCoordinate(0.4f, size) == 0);
    CHECK(voxelCoordinate(0.5f, size) == 1);
}

TEST_CASE("invalid and out-of-range voxel coordinates have defined sentinels")
{
    const float infinity = std::numeric_limits<float>::infinity();
    CHECK(voxelCoordinate(std::nanf(""), 1.0f) == 0);
    CHECK(voxelCoordinate(0.0f, 0.0f) == 0);
    CHECK(voxelCoordinate(infinity, 1.0f) == std::numeric_limits<int32_t>::max());
    CHECK(voxelCoordinate(-infinity, 1.0f) == std::numeric_limits<int32_t>::min());
    CHECK(voxelCoordinate(1.0f, 0.0f) == std::numeric_limits<int32_t>::max());
    CHECK(voxelCoordinate(-1.0f, 0.0f) == std::numeric_limits<int32_t>::min());
}

TEST_CASE("the normal bucket separates the six faces and nothing else")
{
    CHECK(normalBucket(1.0f, 0.0f, 0.0f) == 0u);
    CHECK(normalBucket(-1.0f, 0.0f, 0.0f) == 1u);
    CHECK(normalBucket(0.0f, 1.0f, 0.0f) == 2u);
    CHECK(normalBucket(0.0f, -1.0f, 0.0f) == 3u);
    CHECK(normalBucket(0.0f, 0.0f, 1.0f) == 4u);
    CHECK(normalBucket(0.0f, 0.0f, -1.0f) == 5u);

    // A floor and the ceiling below it must not share radiance, which is the
    // whole reason the normal is in the key.
    CHECK(normalBucket(0.05f, 0.99f, -0.02f) != normalBucket(0.05f, -0.99f, -0.02f));
    // A smooth surface must not be split into stripes by it: two normals a few
    // degrees apart about the same axis stay together.
    CHECK(normalBucket(0.0f, 1.0f, 0.0f) == normalBucket(0.2f, 0.97f, 0.1f));

    for (int i = 0; i < 6; ++i)
    {
        CHECK(normalBucket(i == 0 ? 1.f : 0.f, i == 2 ? 1.f : 0.f, i == 4 ? 1.f : 0.f) < 6u);
    }
}

TEST_CASE("a key round-trips every field it carries")
{
    // The key stopped being a checksum so that two things could read it back:
    // reprojection, which needs the voxel's position and level, and responsive
    // lighting, which needs the flag. Both are silent if a field does not
    // survive the packing -- reprojection simply finds nothing.
    for (int32_t level : { -255, -20, -1, 0, 1, 7, 255 })
        for (uint32_t bucket = 0; bucket < 6u; ++bucket)
            for (bool responsive : { false, true })
            {
                const uint64_t key = voxelKey(-37, 1024, -65536, level, bucket, responsive);
                CHECK(unpackCoordinate(key, 0u) == -37);
                CHECK(unpackCoordinate(key, 1u) == 1024);
                CHECK(unpackCoordinate(key, 2u) == -65536);
                CHECK(unpackLevel(key) == level);
                CHECK(unpackBucket(key) == bucket);
                CHECK(isResponsiveKey(key) == responsive);
            }
}

TEST_CASE("neighbouring voxels do not collide, and the hash spreads")
{
    // A key that repeats across a small neighbourhood is a wall bleeding into
    // the room behind it. 8x8x8 of them, all six normal buckets. Exact now
    // rather than probable: the key carries the coordinates instead of a mix
    // of them, so two different voxels cannot agree by accident.
    std::set<uint64_t> keys;
    std::set<uint32_t> slots;
    size_t total = 0;
    for (int32_t x = 0; x < 8; ++x)
        for (int32_t y = 0; y < 8; ++y)
            for (int32_t z = 0; z < 8; ++z)
                for (uint32_t b = 0; b < 6; ++b)
                {
                    const uint64_t key = voxelKey(x, y, z, 3, b, false);
                    keys.insert(key);
                    slots.insert(keyHash(key));
                    ++total;
                }
    CHECK(keys.size() == total);
    // And the slot hash still spreads them: a key that is exact but hashes into
    // a handful of slots would fill eight probes and drop the rest.
    CHECK(slots.size() > total * 9 / 10);
}

TEST_CASE("the same voxel at two levels is two voxels")
{
    // The level is in the key because a coarse voxel and a fine one with the
    // same integer coordinate are different regions of the world.
    CHECK(voxelKey(3, 4, 5, 2, 0, false) != voxelKey(3, 4, 5, 3, 0, false));
    CHECK(voxelKey(3, 4, 5, -2, 0, false) != voxelKey(3, 4, 5, 2, 0, false));
}

TEST_CASE("the responsive half of a voxel is a different voxel to the table")
{
    // Same place, same normal, same level -- and it has to land in its own slot,
    // because the two hold different halves of the signal and a reader adds
    // them. If they collided, one would overwrite the other and the sum would be
    // whichever won.
    const uint64_t key = voxelKey(11, -3, 40, 2, 4, false);
    const uint64_t responsive = responsiveKey(key);
    CHECK(responsive != key);
    CHECK(isResponsiveKey(responsive));
    CHECK(!isResponsiveKey(key));
    CHECK(keyHash(responsive) != keyHash(key));
    // And everything else about it is unchanged, so reprojection and eviction
    // treat it as the same place.
    CHECK(unpackCoordinate(responsive, 0u) == unpackCoordinate(key, 0u));
    CHECK(unpackLevel(responsive) == unpackLevel(key));
    CHECK(unpackBucket(responsive) == unpackBucket(key));
    // Idempotent: marking an already-responsive key must not toggle it back.
    CHECK(responsiveKey(responsive) == responsive);
}

TEST_CASE("a key is never zero, because zero means empty")
{
    // Zero is the empty slot, so a voxel that packed to zero would be invisible
    // to every probe that walked past it. The level field is biased to make that
    // unreachable: it is at least 1 for any level the clamp admits.
    for (int32_t level = -300; level <= 300; ++level)
    {
        CHECK(voxelKey(0, 0, 0, level, 0u, false) != 0ull);
    }
    for (uint32_t i = 0; i < 50000u; ++i)
    {
        const int32_t x = (int32_t)hash(i) % 4096;
        const int32_t y = (int32_t)hash(i + 1u) % 4096;
        const int32_t z = (int32_t)hash(i + 2u) % 4096;
        CHECK(voxelKey(x, y, z, (int32_t)(i % 40u) - 20, i % 6u, false) != 0ull);
    }
}

TEST_CASE("reprojection looks coarser when the eye came closer, finer when it went away")
{
    // The level follows distance to the eye, so moving the eye re-quantises a
    // world that has not moved. This is the lookup that finds where the data
    // went. Which direction it looks is the whole of its correctness: looking
    // the wrong way finds an unrelated voxel and blends it in.
    const uint64_t key = voxelKey(8, 8, 8, 3, 2, false);

    // Eye now near the voxel, previously far: this entry is the finer one, and
    // the history is a level coarser.
    const uint64_t closer = adjacentLevelKey(key, 64.0f, 64.0f, 64.0f, 4000.0f, 4000.0f, 4000.0f);
    CHECK(unpackLevel(closer) == 4);
    CHECK(unpackCoordinate(closer, 0u) == 4);

    // And the other way.
    const uint64_t further = adjacentLevelKey(key, 4000.0f, 4000.0f, 4000.0f, 64.0f, 64.0f, 64.0f);
    CHECK(unpackLevel(further) == 2);
    CHECK(unpackCoordinate(further, 0u) == 16);
}

TEST_CASE("reprojection compares large camera distances without integer overflow")
{
    const uint64_t key = voxelKey(8, 8, 8, 0, 0, false);
    const uint64_t closer = adjacentLevelKey(key, 100000.0f, 100000.0f, 100000.0f, 200000.0f, 200000.0f, 200000.0f);
    const uint64_t further = adjacentLevelKey(key, 200000.0f, 200000.0f, 200000.0f, 100000.0f, 100000.0f, 100000.0f);

    CHECK(unpackLevel(closer) == 1);
    CHECK(unpackCoordinate(closer, 0u) == 4);
    CHECK(unpackLevel(further) == -1);
    CHECK(unpackCoordinate(further, 0u) == 16);
}

TEST_CASE("reprojection keeps the surface it is about")
{
    // The normal bucket and the responsive flag are properties of the surface
    // and the signal, not of the resolution it is stored at. Dropping either
    // would blend a floor into the ceiling below it, or the steady half of a
    // voxel into the responsive half.
    for (uint32_t bucket = 0; bucket < 6u; ++bucket)
        for (bool responsive : { false, true })
        {
            const uint64_t key = voxelKey(5, -9, 2, 1, bucket, responsive);
            const uint64_t adjacent = adjacentLevelKey(key, 1.0f, 1.0f, 1.0f, 900.0f, 900.0f, 900.0f);
            CHECK(unpackBucket(adjacent) == bucket);
            CHECK(isResponsiveKey(adjacent) == responsive);
        }
}

TEST_CASE("reprojection halves a negative coordinate downwards")
{
    // floor, not truncation, and the reason is the same one that made
    // voxelCoordinate use floor: truncating folds the two cells either side of
    // the origin into one, which puts a seam through the middle of every scene
    // -- and here it would put it there only while the camera is moving.
    const uint64_t key = voxelKey(-3, -1, -7, 3, 0, false);
    const uint64_t coarser = adjacentLevelKey(key, 0.0f, 0.0f, 0.0f, 5000.0f, 5000.0f, 5000.0f);
    CHECK(unpackLevel(coarser) == 4);
    CHECK(unpackCoordinate(coarser, 0u) == -2);
    CHECK(unpackCoordinate(coarser, 1u) == -1);
    CHECK(unpackCoordinate(coarser, 2u) == -4);
}

TEST_CASE("reprojection cannot walk off the end of the level field")
{
    // A wrapped level is a key pointing at an unrelated region of the world,
    // which is the one outcome worse than not reprojecting at all. The bound is
    // a rail rather than a working limit -- voxelForDistance floors the
    // footprint at 1e-4, so real levels run about [-14, 14] against a field that
    // holds [-255, 255] -- and the point of the sweep is that nothing gets
    // anywhere near it whatever the two cameras are.
    for (const int32_t level : { kLevelMin, kLevelMin + 1, -14, 0, 14, kLevelMax - 1, kLevelMax })
        for (const float near : { 0.0f, 1.0f, 1e4f })
            for (const float far : { 0.0f, 2.0f, 1e5f })
            {
                const uint64_t key = voxelKey(1, 1, 1, level, 0, false);
                const int32_t moved = unpackLevel(adjacentLevelKey(key, near, near, near, far, far, far));
                CHECK(moved >= kLevelMin);
                CHECK(moved <= kLevelMax);
                // And it moves by at most one level, in either direction: the
                // blend is only meaningful against an immediate neighbour.
                CHECK(std::abs(moved - level) <= 1);
            }
}

TEST_CASE("an adjacent level blends by the samples behind each")
{
    Resolved own;
    own.r = own.g = own.b = 1.0f;
    own.sampleNum = 10.0f;
    Resolved adjacent;
    adjacent.r = adjacent.g = adjacent.b = 5.0f;
    adjacent.sampleNum = 30.0f;

    const Resolved blended = blendAdjacentLevel(own, adjacent);
    // (1*10 + 5*30) / 40 = 4
    CHECK(blended.r == doctest::Approx(4.0f).epsilon(0.001));
    // The samples come with it, or the blended entry stays under the threshold a
    // path has to clear before it may read -- and the reprojection buys nothing.
    CHECK(blended.sampleNum == doctest::Approx(40.0f).epsilon(0.001));
}

TEST_CASE("blending against an empty neighbour changes nothing")
{
    // The common case by far: most voxels have no counterpart one level away.
    Resolved own;
    own.r = 2.0f;
    own.g = 3.0f;
    own.b = 4.0f;
    own.sampleNum = 7.0f;
    const Resolved blended = blendAdjacentLevel(own, Resolved{});
    CHECK(blended.r == doctest::Approx(own.r));
    CHECK(blended.sampleNum == doctest::Approx(own.sampleNum));
}

TEST_CASE("the probe run visits eight distinct slots and stays in the table")
{
    const uint32_t capacity = 1u << 16;
    for (const uint32_t seed : { 0u, 1u, 12345u, 0xFFFFFFFFu, capacity - 1u })
    {
        std::set<uint32_t> visited;
        for (uint32_t p = 0; p < kProbeCount; ++p)
        {
            const uint32_t slot = probeSlot(seed, capacity, p);
            CHECK(slot < capacity);
            visited.insert(slot);
        }
        CHECK(visited.size() == kProbeCount);
    }
}

TEST_CASE("the fixed point round trips within half its own quantum")
{
    for (const float v : { 0.0f, 0.001f, 0.5f, 1.0f, 7.25f, 100.0f, 255.9f })
    {
        const uint32_t e = encode(v);
        const float back = decode(e, 1u);
        CHECK(std::fabs(back - v) <= 0.5f / kScale + 1e-6f);
    }
}

TEST_CASE("the quantisation error is zero-mean, because a one-sided one is a bias")
{
    // The whole reason encode() rounds rather than truncates. Truncation loses
    // half a quantum on every deposit in the same direction; a quantum is
    // 1/kScale, so against an outgoing radiance of a couple of units that is a
    // quarter of a percent off the converged image, every time. Measured on
    // 00_calibration and 02_basecolor, which both read 0.997 against the same
    // render with the cache off until this was rounding.
    double error = 0.0;
    int samples = 0;
    for (int i = 0; i < 20000; ++i)
    {
        // Deliberately not on the quantisation lattice: values that land exactly
        // on it have no error either way and would hide the effect.
        const float v = 0.017f + (float)i * 0.000713f;
        error += (double)decode(encode(v), 1u) - (double)v;
        ++samples;
    }
    const double meanError = error / (double)samples;
    // Under a quantum of a hundredth of the value the cache typically holds.
    CHECK(std::fabs(meanError) < 0.1 / (double)kScale);
}

TEST_CASE("the fixed point refuses what would poison a voxel")
{
    // A firefly read back by every path through the voxel turns one bright pixel
    // into a bright region, so it is clamped rather than stored.
    CHECK(decode(encode(1e9f), 1u) == doctest::Approx(kClamp).epsilon(1e-4));
    // Negative and NaN both come from an estimator that went wrong upstream; a
    // negative sum would wrap the unsigned accumulator.
    CHECK(encode(-1.0f) == 0u);
    CHECK(encode(std::nanf("")) == 0u);
    CHECK(encode(-std::numeric_limits<float>::infinity()) == 0u);
    CHECK(decode(encode(std::numeric_limits<float>::infinity()), 1u) == doctest::Approx(kClamp).epsilon(1e-4));
}

TEST_CASE("an empty slot reads as black rather than as a division by zero")
{
    CHECK(decode(0u, 0u) == 0.0f);
    CHECK(decode(12345u, 0u) == 0.0f);
}

TEST_CASE("the accumulator cannot be made to wrap")
{
    // kMaxCount deposits of the largest value the clamp allows must still fit in
    // 32 bits, or a long render reads a full voxel back as a nearly black one.
    const double largest = (double)kMaxCount * (double)encode(kClamp);
    CHECK(largest <= 4294967295.0);
    // And it has to be a count a render can actually reach, not one so small the
    // cache stops working in the first second.
    CHECK(kMaxCount > 100000u);
}

TEST_CASE("the mean of a slot is the mean of what went into it")
{
    // What the cache stores is an average, and the estimator is only unbiased if
    // reading it back gives that average.
    const std::vector<float> samples = { 0.5f, 1.5f, 2.0f, 4.25f, 0.0f, 3.75f, 1.0f };
    uint32_t sum = 0;
    double reference = 0.0;
    for (const float s : samples)
    {
        sum += encode(s);
        reference += s;
    }
    reference /= (double)samples.size();
    CHECK(decode(sum, (uint32_t)samples.size()) == doctest::Approx((float)reference).epsilon(0.01));
}

TEST_CASE("one path in eight records, and which one does not depend on the pixel")
{
    // The share has to be a share: too few and the cache freezes at whatever the
    // first paths through a voxel found, too many and there is nothing left to
    // read it.
    size_t updates = 0;
    size_t total = 0;
    for (uint32_t pixel = 0; pixel < 512u; ++pixel)
    {
        for (uint32_t sample = 0; sample < 64u; ++sample)
        {
            updates += isUpdatePath(pixel, sample) ? 1u : 0u;
            ++total;
        }
    }
    const double share = (double)updates / (double)total;
    CHECK(share == doctest::Approx(1.0 / (double)kUpdateShare).epsilon(0.06));

    // And no pixel may be an update path on every sample, or that pixel never
    // reads the cache at all.
    for (uint32_t pixel = 0; pixel < 64u; ++pixel)
    {
        size_t perPixel = 0;
        for (uint32_t sample = 0; sample < 128u; ++sample)
        {
            perPixel += isUpdatePath(pixel, sample) ? 1u : 0u;
        }
        CHECK(perPixel > 0u);
        CHECK(perPixel < 128u);
    }
}

TEST_CASE("the thresholds are the ones Metal measured, and in the right direction")
{
    // These are not taste. Each one is a number Metal's port arrived at by
    // measuring what the other value cost, and the two backends have to agree or
    // their caches are not comparable.
    CHECK(kScale == 64.0f);
    CHECK(kClamp == 256.0f);
    CHECK(kProbeCount == 8u);
    CHECK(kUpdateShare == 8u);
    CHECK(kMinRecordThroughput == doctest::Approx(0.05f));
    CHECK(kThroughputFloor == doctest::Approx(0.02f));
    // A clamp below the radiance an interior legitimately produces is a
    // systematic darkening, which is worse than the noise it replaces.
    CHECK(kClamp > kScale);
    // The floor has to be below the threshold, or nothing that gets recorded is
    // ever divided by its own throughput.
    CHECK(kThroughputFloor < kMinRecordThroughput);
}

TEST_CASE("the capacity floor keeps a mistyped setting from becoming one slot")
{
    CHECK(kMinCapacity == (1u << 16));
    // Power of two, because the probe sequence masks rather than divides.
    CHECK((kMinCapacity & (kMinCapacity - 1u)) == 0u);
}

// ---------------------------------------------------------------------------
// Resolve: what a voxel keeps between frames
// ---------------------------------------------------------------------------
//
// These cover the half of the cache that decides whether it is a cache at all.
// Every rule here fails silently on a GPU and looks like something else in the
// image: a window that does not normalise reads as a cache that ignores the
// lights, a staleness rule that never fires reads as a table that is full, and
// one that fires too eagerly reads as a cache that is simply slow.

TEST_CASE("binary16 round-trips the values a voxel actually holds")
{
    // Radiance, and sample counts up to a few thousand. Relative error of
    // binary16 is 2^-11, so anything inside half a percent is the format doing
    // its job rather than the code doing it wrong.
    for (const float v : { 0.001f, 0.5f, 1.0f, std::numbers::pi_v<float>, 42.0f, 255.0f, 1000.0f, 30000.0f })
    {
        const float back = unpackHalf(packHalf(v));
        CHECK(back == doctest::Approx(v).epsilon(0.001));
    }
}

TEST_CASE("binary16 encoding admits nothing that would poison a voxel")
{
    // A negative or a NaN reaching the resolved half is read back by every path
    // that passes through the voxel, so both are refused at the door rather than
    // encoded and discovered later.
    CHECK(unpackHalf(packHalf(-1.0f)) == 0.0f);
    CHECK(unpackHalf(packHalf(-0.0f)) == 0.0f);
    CHECK(unpackHalf(packHalf(std::nanf(""))) == 0.0f);
    CHECK(unpackHalf(packHalf(0.0f)) == 0.0f);

    // And an overflow is clamped to the largest finite value rather than
    // becoming an infinity, which would spread through the blend below.
    const float huge = unpackHalf(packHalf(1e30f));
    CHECK(std::isfinite(huge));
    CHECK(huge == doctest::Approx(kHalfMax).epsilon(0.001));
    CHECK(std::isfinite(unpackHalf(packHalf(std::numeric_limits<float>::infinity()))));
}

TEST_CASE("rounding to binary16 is unbiased")
{
    // The same argument that made `encode` round rather than truncate: a bias of
    // half a quantum, applied to every voxel every frame in the same direction,
    // is a bias in the image and not noise in it.
    // Integer induction, because accumulating the step in a float both drifts
    // and trips cert-flp30-c.
    const int steps = 5000;
    double sum = 0.0;
    for (int step = 0; step < steps; ++step)
    {
        const float v = 0.5f + (float)step * 0.0007f;
        sum += (double)unpackHalf(packHalf(v)) - (double)v;
    }
    const double meanError = sum / steps;
    CHECK(std::fabs(meanError) < 1e-5);
}

TEST_CASE("an entry's two packed words survive a round trip")
{
    Resolved value;
    value.r = 1.25f;
    value.g = 0.5f;
    value.b = 12.0f;
    value.sampleNum = 640.0f;

    uint32_t lo = 0u, hi = 0u;
    packResolved(value, lo, hi);
    const Resolved back = unpackResolved(lo, hi);

    CHECK(back.r == doctest::Approx(value.r).epsilon(0.001));
    CHECK(back.g == doctest::Approx(value.g).epsilon(0.001));
    CHECK(back.b == doctest::Approx(value.b).epsilon(0.001));
    CHECK(back.sampleNum == doctest::Approx(value.sampleNum).epsilon(0.001));

    // Channels must not bleed into each other: r and g share a word, and so do
    // b and the sample count.
    uint32_t soloLo = 0u, soloHi = 0u;
    Resolved solo;
    solo.r = 7.0f;
    packResolved(solo, soloLo, soloHi);
    const Resolved soloBack = unpackResolved(soloLo, soloHi);
    CHECK(soloBack.r == doctest::Approx(7.0f).epsilon(0.001));
    CHECK(soloBack.g == 0.0f);
    CHECK(soloBack.b == 0.0f);
    CHECK(soloBack.sampleNum == 0.0f);
}

TEST_CASE("the two frame counters share a word without touching")
{
    uint32_t accumFrames = 0u, staleFrames = 0u;
    unpackFrameData(packFrameData(31u, 7u), accumFrames, staleFrames);
    CHECK(accumFrames == 31u);
    CHECK(staleFrames == 7u);

    // Saturating rather than wrapping. A wrapped accumulation count reads as a
    // brand-new entry, which would restart the window on a voxel that has been
    // averaging for an hour.
    unpackFrameData(packFrameData(100000u, 100000u), accumFrames, staleFrames);
    CHECK(accumFrames == kFrameNumMask);
    CHECK(staleFrames == kFrameNumMask);
}

namespace
{
/// One frame of deposits: `count` samples, each of `radiance`, into a voxel.
ResolveInput depositFrame(const ResolveOutput& previous, float radiance, uint32_t count)
{
    ResolveInput input;
    input.accum[0] = encode(radiance) * count;
    input.accum[1] = encode(radiance) * count;
    input.accum[2] = encode(radiance) * count;
    input.accumCount = count;
    input.resolvedLo = previous.resolvedLo;
    input.resolvedHi = previous.resolvedHi;
    input.frameData = previous.frameData;
    return input;
}

/// A frame in which nobody visited the voxel.
ResolveInput idleFrame(const ResolveOutput& previous)
{
    ResolveInput input;
    input.resolvedLo = previous.resolvedLo;
    input.resolvedHi = previous.resolvedHi;
    input.frameData = previous.frameData;
    return input;
}
} // namespace

TEST_CASE("a voxel's first frame resolves to the mean of what it was given")
{
    const ResolveOutput fresh;
    const ResolveOutput out = resolveEntry(depositFrame(fresh, 2.5f, 16u), 32u, 64u);

    CHECK(!out.evict);
    const Resolved resolved = unpackResolved(out.resolvedLo, out.resolvedHi);
    CHECK(resolved.r == doctest::Approx(2.5f).epsilon(0.01));
    CHECK(resolved.sampleNum == doctest::Approx(16.0f).epsilon(0.01));

    uint32_t accumFrames = 0u, staleFrames = 0u;
    unpackFrameData(out.frameData, accumFrames, staleFrames);
    CHECK(accumFrames == 1u);
    CHECK(staleFrames == 0u);
}

TEST_CASE("a steady voxel stays where it is")
{
    // The cache's own fixed point. If repeated identical deposits drift, every
    // number below is measuring drift instead of what it claims to.
    ResolveOutput state;
    for (int frame = 0; frame < 200; ++frame)
    {
        state = resolveEntry(depositFrame(state, 3.0f, 8u), 32u, 64u);
        REQUIRE(!state.evict);
    }
    const Resolved resolved = unpackResolved(state.resolvedLo, state.resolvedHi);
    CHECK(resolved.r == doctest::Approx(3.0f).epsilon(0.01));
    CHECK(std::isfinite(resolved.sampleNum));
}

TEST_CASE("the window bounds the weight of the history")
{
    // Without the normalisation the accumulated sample count grows without
    // bound, and a voxel that has been averaging for a thousand frames can no
    // longer notice that somebody turned a light on. The count has to settle.
    const uint32_t window = 8u;
    const uint32_t perFrame = 10u;
    ResolveOutput state;
    for (int frame = 0; frame < 500; ++frame)
    {
        state = resolveEntry(depositFrame(state, 1.0f, perFrame), window, 64u);
    }
    const Resolved resolved = unpackResolved(state.resolvedLo, state.resolvedHi);
    // The fixed point of n -> n * window / (window + 1) + perFrame, which is
    // perFrame * (window + 1) and not perFrame * window -- the frame being
    // folded in is one the window has not yet charged for. Worth pinning: the
    // difference is one window's worth of weight, which is exactly how much a
    // step change in the lighting lags.
    CHECK(resolved.sampleNum == doctest::Approx((float)(perFrame * (window + 1u))).epsilon(0.05));

    uint32_t accumFrames = 0u, staleFrames = 0u;
    unpackFrameData(state.frameData, accumFrames, staleFrames);
    CHECK(accumFrames == window);
}

TEST_CASE("a voxel follows the lights, and does it inside its window")
{
    // The whole reason the window exists. A cache that averages for ever is a
    // photograph of the first frame.
    const uint32_t window = 16u;
    ResolveOutput state;
    for (int frame = 0; frame < 200; ++frame)
    {
        state = resolveEntry(depositFrame(state, 1.0f, 8u), window, 512u);
    }
    CHECK(unpackResolved(state.resolvedLo, state.resolvedHi).r == doctest::Approx(1.0f).epsilon(0.02));

    // Somebody turns the light up. Within a few window lengths the voxel should
    // have followed it essentially all the way.
    for (int frame = 0; frame < 4 * (int)window; ++frame)
    {
        state = resolveEntry(depositFrame(state, 5.0f, 8u), window, 512u);
    }
    CHECK(unpackResolved(state.resolvedLo, state.resolvedHi).r == doctest::Approx(5.0f).epsilon(0.05));
}

TEST_CASE("a larger window is slower to follow than a smaller one")
{
    // The knob has to mean something, and this is the direction it means. It is
    // the trade the editor's tooltip promises: quieter, and slower to react.
    auto settleAfter = [](uint32_t window, int frames) {
        ResolveOutput state;
        for (int i = 0; i < 200; ++i)
        {
            state = resolveEntry(depositFrame(state, 1.0f, 8u), window, 512u);
        }
        for (int i = 0; i < frames; ++i)
        {
            state = resolveEntry(depositFrame(state, 5.0f, 8u), window, 512u);
        }
        return unpackResolved(state.resolvedLo, state.resolvedHi).r;
    };
    CHECK(settleAfter(4u, 8) > settleAfter(64u, 8));
}

TEST_CASE("an unvisited voxel keeps its answer, then gives back its slot")
{
    // Both halves matter. A voxel that is off screen for a moment must still be
    // the answer when a path reaches it again -- that is what survives a camera
    // movement, and what the host used to throw away by clearing the table. But
    // it cannot be kept for ever, or the table fills with places the camera has
    // left and never frees a slot.
    const uint32_t staleMax = 16u;
    ResolveOutput state = resolveEntry(depositFrame(ResolveOutput{}, 4.0f, 32u), 32u, staleMax);
    REQUIRE(!state.evict);

    for (uint32_t frame = 1; frame < staleMax; ++frame)
    {
        state = resolveEntry(idleFrame(state), 32u, staleMax);
        REQUIRE(!state.evict);
        // Still answering, with the value it had.
        CHECK(unpackResolved(state.resolvedLo, state.resolvedHi).r == doctest::Approx(4.0f).epsilon(0.01));
    }

    state = resolveEntry(idleFrame(state), 32u, staleMax);
    CHECK(state.evict);
}

TEST_CASE("eviction cannot be made hair-trigger from a setting")
{
    // The SDK's own warning: evicting too eagerly costs more in re-insertion
    // than the slots are worth, so the threshold is clamped no matter what the
    // caller asks for. A config that says 0 must not turn the table over every
    // frame.
    ResolveOutput state = resolveEntry(depositFrame(ResolveOutput{}, 1.0f, 4u), 32u, 0u);
    REQUIRE(!state.evict);
    for (uint32_t frame = 1; frame < kStaleFrameNumMin; ++frame)
    {
        state = resolveEntry(idleFrame(state), 32u, 0u);
        CHECK(!state.evict);
    }
}

TEST_CASE("a voxel nobody has ever deposited into is still evicted")
{
    // Insertion and deposit are separate: a path may take a slot and then fail
    // the throughput test, or be a reading path that never records. Those slots
    // have to come back, or a moving camera leaks the table one probe run at a
    // time.
    ResolveOutput state;
    bool evicted = false;
    for (uint32_t frame = 0; frame < kStaleFrameNumMin + 1u; ++frame)
    {
        state = resolveEntry(idleFrame(state), 32u, 0u);
        evicted = evicted || state.evict;
    }
    CHECK(evicted);
}

TEST_CASE("the eligibility test refuses a segment that never left its voxel")
{
    // The hit is inside the voxel the ray departed from, so the average the
    // cache would return includes the very point being shaded. No lobe is wide
    // enough to make that acceptable.
    CHECK(!mayReadCache(/*segmentLength=*/0.5f, /*launchRoughness=*/1.0f, /*voxelSize=*/1.0f));
    CHECK(!mayReadCache(1.7f, 1.0f, 1.0f)); // just under the diagonal
    CHECK(mayReadCache(1.8f, 1.0f, 1.0f)); // just over it
}

TEST_CASE("the eligibility test refuses a lobe that still carries an image")
{
    // A mirror, and anything near one. This is the condition that keeps a
    // reflection sharp, and it asks about the surface the segment *left*.
    const float voxel = 0.1f;
    const float segment = 5.0f; // long enough that the diagonal test passes
    CHECK(!mayReadCache(segment, 0.0f, voxel));
    CHECK(!mayReadCache(segment, 0.01f, voxel));
    // ... and admits one that has spread wider than a voxel by the time it
    // arrived, which is the whole point: a wide lobe is already an average.
    CHECK(mayReadCache(segment, 0.5f, voxel));
    CHECK(mayReadCache(segment, 1.0f, voxel));
}

TEST_CASE("the eligibility test is monotone in every argument")
{
    // The three knobs have to push in the directions the reasoning claims, or a
    // scene tuned on one of them moves the wrong way.
    const float voxel = 0.2f;
    // Rougher is more eligible.
    bool seenFalse = false, seenTrue = false;
    for (int step = 0; step <= 50; ++step)
    {
        const float r = (float)step * 0.02f;
        const bool ok = mayReadCache(4.0f, r, voxel);
        if (ok)
            seenTrue = true;
        else
            CHECK(!seenTrue); // never goes back to ineligible once eligible
        seenFalse = seenFalse || !ok;
    }
    CHECK(seenFalse);
    CHECK(seenTrue);
    // Longer segments are more eligible, larger voxels less.
    CHECK(mayReadCache(8.0f, 0.4f, voxel));
    CHECK(!mayReadCache(8.0f, 0.4f, voxel * 100.0f));
}

TEST_CASE("a degenerate voxel or segment cannot make the test say yes by accident")
{
    CHECK(!mayReadCache(0.0f, 1.0f, 1.0f));
    CHECK(!mayReadCache(-1.0f, 1.0f, 1.0f));
    CHECK(!mayReadCache(std::nanf(""), 1.0f, 1.0f));
    // A perfectly smooth surface is refused at every distance, which is the
    // mirror case and the one that must never slip through.
    for (int step = 0; step < 11; ++step)
    {
        const float d = 0.01f * std::pow(3.0f, (float)step);
        CHECK(!mayReadCache(d, 0.0f, 0.001f));
    }
}
