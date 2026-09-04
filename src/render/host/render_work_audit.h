#pragma once

#include <cstdint>

namespace oka::metal
{

struct RenderWorkInvariantSample
{
    uint64_t primaryRays = 0;
    uint64_t extendRays = 0;
    uint64_t firstBounceNee = 0;
    uint64_t candidateQueries = 0;
    uint64_t reuseQueries = 0;
    uint64_t eligibleHits = 0;
    uint64_t finalVisibilityRays = 0;
    uint64_t blasBuilds = 0;
    uint64_t tlasBuilds = 0;
    uint64_t tlasRefits = 0;
    uint64_t manualAnalyticLightTests = 0;
    uint64_t restirSpatialDispatches = 0;
    uint64_t restirFinalDispatches = 0;
    uint64_t guideActiveItems = 0;
    uint64_t guideDispatches = 0;
    uint64_t frames = 0;
    uint64_t maxTlasRefitsPerFrame = 0;
    uint64_t maxLightUploadsPerFrame = 0;
    uint64_t maxTemporalMappingsPerFrame = 0;
    uint64_t primaryDispatches = 0;
};

struct CommandBufferAuditSample
{
    uint64_t creations = 0;
    uint64_t encoderCreations = 0;
    uint64_t dispatches = 0;
    uint64_t endEncodings = 0;
    uint64_t commits = 0;
    uint64_t waits = 0;
    uint64_t readbacks = 0;
};

inline bool guidesShareSurfaceTraversal(const RenderWorkInvariantSample& off, const RenderWorkInvariantSample& on)
{
    return off.primaryRays == on.primaryRays && off.extendRays == on.extendRays;
}

inline bool restirExcludesFirstBounceNee(bool enabled, const RenderWorkInvariantSample& sample)
{
    return !enabled || sample.firstBounceNee == 0u;
}

inline bool restirReuseDoesNotTrace(const RenderWorkInvariantSample& sample)
{
    return sample.candidateQueries == 0u && sample.reuseQueries == 0u;
}

inline bool finalVisibilityIsBounded(const RenderWorkInvariantSample& sample)
{
    return sample.finalVisibilityRays <= sample.eligibleHits;
}

inline bool staticAccelerationStructuresAreStable(const RenderWorkInvariantSample& sample)
{
    return sample.blasBuilds == 0u && sample.tlasBuilds == 0u && sample.tlasRefits == 0u;
}

inline bool analyticLightWorkIsCountIndependent(const RenderWorkInvariantSample& one,
                                                const RenderWorkInvariantSample& many)
{
    return one.manualAnalyticLightTests == many.manualAnalyticLightTests;
}

inline bool restirDispatchesMatch(bool enabled,
                                  bool spatialEnabled,
                                  uint32_t neighbors,
                                  const RenderWorkInvariantSample& sample)
{
    const uint64_t expectedFinal = enabled ? 1u : 0u;
    const uint64_t expectedSpatial = enabled && spatialEnabled && neighbors != 0u ? 1u : 0u;
    return sample.restirFinalDispatches == expectedFinal && sample.restirSpatialDispatches == expectedSpatial;
}

inline bool guideDispatchesMatchActiveQueue(const RenderWorkInvariantSample& sample)
{
    return sample.guideDispatches == (sample.guideActiveItems != 0u ? 1u : 0u);
}

inline bool commandBufferCommitsOnce(const CommandBufferAuditSample& sample)
{
    return sample.creations == 1u && sample.commits == 1u;
}

inline bool movingLightFrameWorkIsBounded(const RenderWorkInvariantSample& sample)
{
    return sample.maxTlasRefitsPerFrame <= 1u && sample.maxLightUploadsPerFrame <= 1u &&
           sample.maxTemporalMappingsPerFrame <= 1u && sample.primaryDispatches == sample.frames;
}

} // namespace oka::metal
