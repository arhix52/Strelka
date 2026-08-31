#pragma once

#include <cstdint>


namespace oka::metal
{

/// Which parts of the scene are on the GPU, and therefore showable, right now.
///
/// A scene build runs in slices, and until it finishes the renderer used to
/// show nothing at all -- fourteen seconds of black on the pine forest. Every
/// stage of that build produces something that is already correct; these flags
/// say which of them have.
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

/// Whether a frame can be traced from what has arrived so far.
///
/// An *empty* top level is enough, and that is the point rather than an
/// oversight: with no instances every ray reaches the environment, which is not
/// a broken picture but a correct picture of a scene whose geometry has not
/// loaded yet -- the sky, and the light it casts. Geometry then appears in it
/// rather than replacing it.
///
/// The top level is still required. Tracing against nothing at all is not the
/// same thing: it means handing the intersector a null structure.
inline bool canTracePartial(const StreamReadiness& readiness)
{
    return readiness.hasOutputTargets && readiness.hasTopLevel;
}

/// Paces how often a partially loaded scene is republished.
///
/// Publishing is not free: every new snapshot invalidates the accumulated
/// image and the denoiser's history, so a renderer that republished on each
/// arriving asset would restart convergence thousands of times and finish the
/// load with a noisier picture than if it had shown nothing. Batching trades
/// latency for that, and the trade only pays when something actually arrived --
/// republishing an unchanged scene costs a reset and buys nothing.
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

    /// True when the snapshot should be published now.
    ///
    /// `buildComplete` overrides the interval: the finished scene is the one
    /// state that must never be left waiting behind a timer.
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

