#pragma once

#include <cstdint>

namespace oka::metal
{

struct StreamReadiness
{
    /// Somewhere to accumulate into, sized to the current output.
    bool hasOutputTargets = false;
    /// The environment map is resident, or the scene has none and never will.
    bool hasEnvironment = false;
    /// A traversable top level exists. It may contain no instances at all.
    bool hasTopLevel = false;
    /// Every stage has run; nothing further will arrive.
    bool buildComplete = false;
};

inline bool canTracePartial(const StreamReadiness& readiness)
{
    return readiness.hasOutputTargets && readiness.hasTopLevel;
}

class PublishClock
{
public:
    /// Starts the interval. Until this is called the first arrival publishes
    /// immediately, which is what shows the environment as soon as it lands.
    void reset(double nowMs)
    {
        mLastPublishMs = nowMs;
        mPending = 0;
        mStarted = true;
    }

    void noteArrivals(uint32_t count = 1)
    {
        mPending += count;
    }

    uint32_t pendingArrivals() const
    {
        return mPending;
    }

    bool shouldPublish(double nowMs, double intervalMs, bool buildComplete) const
    {
        if (mPending == 0)
        {
            return false;
        }
        if (buildComplete || !mStarted || intervalMs <= 0.0)
        {
            return true;
        }
        // A clock that has gone backwards -- a different time source, a machine
        // that slept -- must not be able to wedge this shut until it catches up.
        const double elapsed = nowMs - mLastPublishMs;
        return elapsed < 0.0 || elapsed >= intervalMs;
    }

    void notePublished(double nowMs)
    {
        mLastPublishMs = nowMs;
        mPending = 0;
        mStarted = true;
    }

private:
    double mLastPublishMs = 0.0;
    uint32_t mPending = 0;
    bool mStarted = false;
};

} // namespace oka::metal

