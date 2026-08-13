#include <doctest/doctest.h>

#include "scene_stream.h"

using oka::metal::canTracePartial;
using oka::metal::PublishClock;
using oka::metal::StreamReadiness;

// Two decisions govern what a half-loaded scene looks like: whether a frame can
// be traced at all, and how often the picture is allowed to change while assets
// keep arriving. Both fail quietly -- one as a black screen, the other as a
// picture that never converges -- so they are pinned here rather than tuned by
// watching a load.

TEST_CASE("a frame needs somewhere to accumulate and something to intersect")
{
    StreamReadiness r;
    CHECK_FALSE(canTracePartial(r));

    // Geometry alone is not enough: there is nowhere to put the result.
    r.hasTopLevel = true;
    CHECK_FALSE(canTracePartial(r));

    // An empty top level is deliberately sufficient. Every ray misses and
    // reaches the environment, which is the correct picture of a scene whose
    // geometry has not arrived -- not a broken one.
    r.hasOutputTargets = true;
    CHECK(canTracePartial(r));
}

TEST_CASE("the environment is not required to start tracing")
{
    // A scene with no environment at all still traces; its rays simply reach
    // nothing. Requiring hasEnvironment would hold such a scene black for its
    // whole build.
    StreamReadiness r;
    r.hasOutputTargets = true;
    r.hasTopLevel = true;
    r.hasEnvironment = false;
    CHECK(canTracePartial(r));
}

TEST_CASE("publishing waits for something to have arrived")
{
    PublishClock clock;
    clock.reset(1000.0);

    // No arrivals: however long has passed, republishing an unchanged scene
    // only throws away the samples accumulated since the last one.
    CHECK_FALSE(clock.shouldPublish(1000.0, 500.0, false));
    CHECK_FALSE(clock.shouldPublish(9999.0, 500.0, false));
    CHECK_FALSE(clock.shouldPublish(9999.0, 500.0, true));

    clock.noteArrivals();
    CHECK(clock.pendingArrivals() == 1);
    CHECK_FALSE(clock.shouldPublish(1100.0, 500.0, false)); // too soon
    CHECK(clock.shouldPublish(1500.0, 500.0, false));       // interval reached
}

TEST_CASE("a finished build is published without waiting for the interval")
{
    PublishClock clock;
    clock.reset(1000.0);
    clock.noteArrivals(3);

    CHECK_FALSE(clock.shouldPublish(1001.0, 500.0, false));
    // The completed scene is the one state that must not sit behind a timer.
    CHECK(clock.shouldPublish(1001.0, 500.0, true));
}

TEST_CASE("the first arrival is published immediately")
{
    // Before any interval has been established there is nothing on screen, so
    // waiting buys nothing and costs the whole first interval of black.
    PublishClock clock;
    clock.noteArrivals();
    CHECK(clock.shouldPublish(0.0, 500.0, false));
}

TEST_CASE("publishing clears what was pending and restarts the interval")
{
    PublishClock clock;
    clock.reset(1000.0);
    clock.noteArrivals(5);
    REQUIRE(clock.shouldPublish(1600.0, 500.0, false));

    clock.notePublished(1600.0);
    CHECK(clock.pendingArrivals() == 0);
    CHECK_FALSE(clock.shouldPublish(1700.0, 500.0, false));

    clock.noteArrivals();
    CHECK_FALSE(clock.shouldPublish(1700.0, 500.0, false)); // interval restarted at 1600
    CHECK(clock.shouldPublish(2100.0, 500.0, false));
}

TEST_CASE("a clock that goes backwards does not wedge publishing shut")
{
    // Sleep, or a different time source between calls. Comparing an elapsed
    // time that has gone negative against the interval would hold the picture
    // frozen until the clock caught up.
    PublishClock clock;
    clock.reset(10000.0);
    clock.noteArrivals();
    CHECK(clock.shouldPublish(1.0, 500.0, false));
}

TEST_CASE("a zero or negative interval publishes on every arrival")
{
    PublishClock clock;
    clock.reset(1000.0);
    clock.noteArrivals();
    CHECK(clock.shouldPublish(1000.0, 0.0, false));
    CHECK(clock.shouldPublish(1000.0, -1.0, false));

    // Still nothing to publish when nothing arrived, whatever the interval.
    clock.notePublished(1000.0);
    CHECK_FALSE(clock.shouldPublish(1000.0, 0.0, false));
}

TEST_CASE("arrivals accumulate across slices that do not publish")
{
    PublishClock clock;
    clock.reset(1000.0);
    for (int i = 0; i < 4; ++i)
    {
        clock.noteArrivals(2);
        CHECK_FALSE(clock.shouldPublish(1000.0 + i, 500.0, false));
    }
    CHECK(clock.pendingArrivals() == 8);
    CHECK(clock.shouldPublish(1500.0, 500.0, false));
}
