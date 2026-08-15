#include <doctest/doctest.h>

#include "sharc_grid.h"

#include <cmath>
#include <map>
#include <set>
#include <vector>

using namespace oka::sharc;

TEST_CASE("a voxel subtends the same angle wherever it is")
{
    // The whole reason the size follows the distance: one setting has to mean
    // the same thing in a room and in a forest. baseSize is the world size of a
    // pixel at unit distance times the pixels a voxel should span, so
    // size/distance is that product to within the power-of-two quantisation.
    const float baseSize = 4.0f * (2.0f * std::tan(0.3926991f) / 512.0f); // 45 deg, 512 px, 4 px voxels
    for (float distance = 0.5f; distance < 500.0f; distance *= 1.3f)
    {
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
    for (float distance = 0.01f; distance < 10000.0f; distance *= 1.7f)
    {
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
    for (float baseSize : { 0.0f, 1e-30f, 1.0f })
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

TEST_CASE("neighbouring voxels do not collide, and the hash spreads")
{
    // A key that repeats across a small neighbourhood is a wall bleeding into
    // the room behind it. 8x8x8 of them, all six normal buckets.
    std::set<uint32_t> keys;
    size_t total = 0;
    for (int32_t x = 0; x < 8; ++x)
        for (int32_t y = 0; y < 8; ++y)
            for (int32_t z = 0; z < 8; ++z)
                for (uint32_t b = 0; b < 6; ++b)
                {
                    keys.insert(voxelKey(voxelHash(x, y, z, 3, b)));
                    ++total;
                }
    CHECK(keys.size() == total);
}

TEST_CASE("the same voxel at two levels is two voxels")
{
    // The level is in the key because a coarse voxel and a fine one with the
    // same integer coordinate are different regions of the world.
    CHECK(voxelHash(3, 4, 5, 2, 0) != voxelHash(3, 4, 5, 3, 0));
    CHECK(voxelHash(3, 4, 5, -2, 0) != voxelHash(3, 4, 5, 2, 0));
}

TEST_CASE("a key is never zero, because zero means empty")
{
    for (uint32_t i = 0; i < 200000u; ++i)
    {
        CHECK(voxelKey(hash(i)) != 0u);
    }
}

TEST_CASE("the probe run visits eight distinct slots and stays in the table")
{
    const uint32_t capacity = 1u << 16;
    for (uint32_t seed : { 0u, 1u, 12345u, 0xFFFFFFFFu, capacity - 1u })
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
    for (float v : { 0.0f, 0.001f, 0.5f, 1.0f, 7.25f, 100.0f, 255.9f })
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
    for (float s : samples)
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
    CHECK(kMinRoughness == doctest::Approx(0.3f));
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
