#include <doctest/doctest.h>

#include "sampling_math.h"

#include <cmath>

using oka::metal::frameJitter;
using oka::metal::haltonAt;

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
