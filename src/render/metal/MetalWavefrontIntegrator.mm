#include "MetalWavefrontIntegrator.h"

#include "MetalBuffer.h"
#include <host/integrator_buffer_sizes.h>
#include "wavefront_stage_diagnostic.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include <log.h>
#include <env.h>
#include <paths.h>

#include <simd/simd.h>
#include <mach/mach_time.h>

#include "ShaderTypes.h"
#include <strelka/material/ior_stack.h>

using namespace oka;
using namespace oka::metal;

MetalWavefrontIntegrator::~MetalWavefrontIntegrator()
{
    release();
}

void MetalWavefrontIntegrator::init(MTL::Device* device, Metal4Context* metal4)
{
    mDevice = device;
    mMetal4 = metal4;
}

void MetalWavefrontIntegrator::release()
{
    auto safeRelease = [](auto*& p) {
        if (p)
        {
            p->release();
            p = nullptr;
        }
    };

    safeRelease(mResolvePSO);
    safeRelease(mPreparePSO);
    safeRelease(mPrepareShadowPSO);
    safeRelease(mClassifySssPSO);
    safeRelease(mPrepareSssPSO);
    safeRelease(mPathStateBuffer);
    safeRelease(mMediumPathStateBuffer);
    safeRelease(mSharcUpdateStateBuffer);
    safeRelease(mPathRayBuffer);
    safeRelease(mHitBuffer);
    safeRelease(mIorStackBuffer);
    safeRelease(mRadianceBuffer);
    safeRelease(mGuideRayBuffer);
    safeRelease(mSurfaceGeometryBuffer);
    safeRelease(mBaseLightConnectionBuffer);
    safeRelease(mGuideQueueBuffer);
    safeRelease(mPathQueueBuffer[0]);
    safeRelease(mPathQueueBuffer[1]);
    safeRelease(mSssQueueBuffer);
    safeRelease(mSssControlBuffer);
    safeRelease(mControlBuffer);
    safeRelease(mTraversalDispatchBuffer);
    safeRelease(mShadowRayBuffer);
    safeRelease(mHitQueueBuffer);
    safeRelease(mAovBuffer);
    safeRelease(mRestirReservoirBuffer[0]);
    safeRelease(mRestirReservoirBuffer[1]);
    safeRelease(mRestirSurfaceHistoryBuffer[0]);
    safeRelease(mRestirSurfaceHistoryBuffer[1]);
    safeRelease(mRestirSurfaceDataBuffer[0]);
    safeRelease(mRestirSurfaceDataBuffer[1]);
    safeRelease(mMissQueueBuffer);
    for (auto& kv : mVariants)
    {
        safeRelease(kv.second.generate);
        safeRelease(kv.second.extendMotion);
        safeRelease(kv.second.extendStatic);
        safeRelease(kv.second.sssWalkMotion);
        safeRelease(kv.second.sssWalkStatic);
        safeRelease(kv.second.connectBase);
        safeRelease(kv.second.shadeBase);
        safeRelease(kv.second.shadeLayer);
        safeRelease(kv.second.shadeTranslucent);
        safeRelease(kv.second.shade);
        safeRelease(kv.second.restirSpatialFinal);
        safeRelease(kv.second.miss);
        safeRelease(kv.second.shadowMotion);
        safeRelease(kv.second.shadowStatic);
        safeRelease(kv.second.guideMotion);
        safeRelease(kv.second.guideStatic);
        safeRelease(kv.second.extendTableMotion);
        safeRelease(kv.second.extendTableStatic);
        safeRelease(kv.second.shadowTableMotion);
        safeRelease(kv.second.shadowTableStatic);
        safeRelease(kv.second.guideTableMotion);
        safeRelease(kv.second.guideTableStatic);
        safeRelease(kv.second.restirShadeDiagnosticTable);
        safeRelease(kv.second.restirSpatialDiagnosticTable);
    }
    mVariants.clear();
    safeRelease(mLibrary);
    safeRelease(mPrepareHitMissPSO);
    safeRelease(mResolvePSO4);
    safeRelease(mPreparePSO4);
    safeRelease(mPrepareShadowPSO4);
    safeRelease(mPrepareHitMissPSO4);
    safeRelease(mClassifySssPSO4);
    safeRelease(mPrepareSssPSO4);
    safeRelease(mStageBreadcrumbPSO4);
    safeRelease(mAovResolvePSO);
    safeRelease(mAovResolvePSO4);
    safeRelease(mSharcClearPSO);
    safeRelease(mSharcResolvePSO);
    safeRelease(mSharcClearPSO4);
    safeRelease(mSharcResolvePSO4);
    safeRelease(mStageTimestampBuffer);
    safeRelease(mStageTimestampHeap4);
    safeRelease(mStageStatsBuffer);
    safeRelease(mIorStatsBuffer);
    safeRelease(mRenderWorkCounterBuffer);
    mCapacity = 0;
    mSharcUpdateDownscale = 0;
    mSplitBaseNeeAllocated = false;
    mResidencyDirty = true;
    mReportedIorStats = false;
    mLastSharcActivity = 0;
    mStageKinds.clear();
    mStageBounces.clear();
    mRenderWorkDispatches.clear();
}

void MetalWavefrontIntegrator::addResidentAllocations(const std::function<void(MTL::Allocation*)>& add) const
{
    add(mIorStatsBuffer);
    add(mRenderWorkCounterBuffer);
    add(mPathStateBuffer);
    add(mMediumPathStateBuffer);
    add(mSharcUpdateStateBuffer);
    add(mPathRayBuffer);
    add(mHitBuffer);
    add(mIorStackBuffer);
    add(mRadianceBuffer);
    add(mGuideRayBuffer);
    add(mSurfaceGeometryBuffer);
    add(mBaseLightConnectionBuffer);
    add(mGuideQueueBuffer);
    add(mPathQueueBuffer[0]);
    add(mPathQueueBuffer[1]);
    add(mSssQueueBuffer);
    add(mSssControlBuffer);
    add(mHitQueueBuffer);
    add(mMissQueueBuffer);
    add(mShadowRayBuffer);
    add(mAovBuffer);
    add(mRestirReservoirBuffer[0]);
    add(mRestirReservoirBuffer[1]);
    add(mRestirSurfaceHistoryBuffer[0]);
    add(mRestirSurfaceHistoryBuffer[1]);
    add(mRestirSurfaceDataBuffer[0]);
    add(mRestirSurfaceDataBuffer[1]);
    add(mControlBuffer);
    add(mTraversalDispatchBuffer);
    add(mStageStatsBuffer);
    for (const auto& entry : mVariants)
    {
        add(entry.second.extendTableMotion);
        add(entry.second.extendTableStatic);
        add(entry.second.shadowTableMotion);
        add(entry.second.shadowTableStatic);
        add(entry.second.guideTableMotion);
        add(entry.second.guideTableStatic);
        add(entry.second.restirShadeDiagnosticTable);
        add(entry.second.restirSpatialDiagnosticTable);
    }
}

size_t MetalWavefrontIntegrator::queueBytes() const
{
    auto bufBytes = [](MTL::Buffer* b) { return b ? b->length() : 0; };
    return bufBytes(mPathStateBuffer) + bufBytes(mMediumPathStateBuffer) + bufBytes(mSharcUpdateStateBuffer) +
           bufBytes(mPathRayBuffer) + bufBytes(mHitBuffer) + bufBytes(mIorStackBuffer) + bufBytes(mRadianceBuffer) +
           bufBytes(mGuideRayBuffer) + bufBytes(mSurfaceGeometryBuffer) + bufBytes(mBaseLightConnectionBuffer) +
           bufBytes(mGuideQueueBuffer) + bufBytes(mPathQueueBuffer[0]) + bufBytes(mPathQueueBuffer[1]) +
           bufBytes(mSssQueueBuffer) + bufBytes(mSssControlBuffer) + bufBytes(mControlBuffer) +
           bufBytes(mTraversalDispatchBuffer) + bufBytes(mShadowRayBuffer) + bufBytes(mHitQueueBuffer) +
           bufBytes(mMissQueueBuffer) + bufBytes(mAovBuffer) + bufBytes(mStageStatsBuffer) +
           bufBytes(mRestirReservoirBuffer[0]) + bufBytes(mRestirReservoirBuffer[1]) +
           bufBytes(mRestirSurfaceHistoryBuffer[0]) + bufBytes(mRestirSurfaceHistoryBuffer[1]) +
           bufBytes(mRestirSurfaceDataBuffer[0]) + bufBytes(mRestirSurfaceDataBuffer[1]) +
           bufBytes(mRenderWorkCounterBuffer);
}

size_t MetalWavefrontIntegrator::restirBytes() const
{
    auto bufBytes = [](MTL::Buffer* b) { return b ? b->length() : 0; };
    return bufBytes(mRestirReservoirBuffer[0]) + bufBytes(mRestirReservoirBuffer[1]) +
           bufBytes(mRestirSurfaceHistoryBuffer[0]) + bufBytes(mRestirSurfaceHistoryBuffer[1]) +
           bufBytes(mRestirSurfaceDataBuffer[0]) + bufBytes(mRestirSurfaceDataBuffer[1]);
}

// Two timestamps per stage (encoder start and end), so the counter buffer holds
// 2 * kMaxStageSamples entries.
namespace
{
enum StageKind : uint8_t
{
    kStageGenerate = 0,
    kStagePrepare,
    kStageExtend,
    kStageSssWalk,
    kStageConnect,
    kStageShade,
    kStageShadeBase,
    kStageShadeLayer,
    kStageShadeTranslucent,
    kStageShadeTail,
    kStageRestirSpatial,
    kStagePrepareShadow,
    kStageShadow,
    kStageMiss,
    kStageGuide,
    kStageResolve,
    kStageSort,
    kStageCount
};
const char* const kStageNames[kStageCount] = { "generate",
                                               "prepare",
                                               "extend",
                                               "sssWalk",
                                               "connect",
                                               "shade",
                                               "shadeBase",
                                               "shadeLayer",
                                               "shadeTranslucent",
                                               "shadeTail",
                                               "restirSpatialFinal",
                                               "prepShadow",
                                               "shadow",
                                               "miss",
                                               "guide",
                                               "resolve",
                                               "sort" };
} // namespace

// A timestamp counter buffer, if the device can sample at dispatch boundaries.
// Apple Silicon can; the check exists because the API does not promise it.
void MetalWavefrontIntegrator::createStageTimestampBuffer()
{
    if (mStageTimestampBuffer)
    {
        return;
    }
    // M1/M2 sample only at encoder boundaries, not at dispatch boundaries, which
    // is why profiling mode gives every stage its own encoder rather than
    // stamping around each dispatch.
    if (!mDevice->supportsCounterSampling(MTL::CounterSamplingPointAtStageBoundary))
    {
        STRELKA_WARNING("stage profiling unavailable: no counter sampling at encoder boundaries");
        return;
    }
    const MTL::CounterSet* timestampSet = nullptr;
    const NS::Array* sets = mDevice->counterSets();
    for (NS::UInteger i = 0; sets && i < sets->count(); ++i)
    {
        // metal-cpp exposes this framework-owned collection as NS::Object.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        const MTL::CounterSet* set = static_cast<MTL::CounterSet*>(sets->object(i));
        if (set->name()->isEqualToString(MTL::CommonCounterSetTimestamp))
        {
            timestampSet = set;
            break;
        }
    }
    if (!timestampSet)
    {
        return;
    }

    MTL::CounterSampleBufferDescriptor* desc = MTL::CounterSampleBufferDescriptor::alloc()->init();
    desc->setCounterSet(timestampSet);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setSampleCount(static_cast<NS::UInteger>(2) * kMaxStageSamples); // start and end per stage
    NS::Error* err = nullptr;
    mStageTimestampBuffer = mDevice->newCounterSampleBuffer(desc, &err);
    desc->release();
    if (!mStageTimestampBuffer)
    {
        STRELKA_WARNING(
            "stage profiling unavailable: {}", err ? err->localizedDescription()->utf8String() : "unknown error");
    }
}

void MetalWavefrontIntegrator::createStageTimestampHeapMetal4()
{
    if (mStageTimestampHeap4)
    {
        return;
    }

    MTL4::CounterHeapDescriptor* desc = MTL4::CounterHeapDescriptor::alloc()->init();
    desc->setType(MTL4::CounterHeapTypeTimestamp);
    desc->setCount(static_cast<NS::UInteger>(2) * kMaxStageSamples);
    NS::Error* err = nullptr;
    mStageTimestampHeap4 = mDevice->newCounterHeap(desc, &err);
    desc->release();
    if (!mStageTimestampHeap4)
    {
        STRELKA_WARNING("Metal 4 stage profiling unavailable: {}",
                        err && err->localizedDescription() ? err->localizedDescription()->utf8String() : "unknown error");
    }
}

void MetalWavefrontIntegrator::reportStageFailureMetal4()
{
    if (!mStageStatsBuffer || mStageKinds.empty())
    {
        STRELKA_ERROR("Metal 4 stage diagnosis unavailable; reproduce with STRELKA_STAGES=1 STRELKA_STAGE_BREADCRUMBS=1");
        return;
    }

    const auto* stats = static_cast<const uint32_t*>(mStageStatsBuffer->contents());
    const uint32_t enteredStage = stats[kWavefrontStageBreadcrumbOffset];
    const WavefrontStageFailure failure = inferWavefrontStageFailure(enteredStage, mStageKinds.size());

    const char* lastCompleted =
        failure.lastCompletedStage >= 0 ? kStageNames[mStageKinds[failure.lastCompletedStage]] : "none";
    const char* suspected = failure.postIntegrator ? "post-integrator" : kStageNames[mStageKinds[failure.suspectedStage]];
    STRELKA_ERROR("Metal 4 stage diagnosis: last_completed={} suspected={} breadcrumb={}/{}", lastCompleted, suspected,
                  enteredStage, mStageKinds.size());
    if (!failure.postIntegrator && failure.suspectedStage >= 0 && mStageStatsBuffer)
    {
        const size_t suspectedIndex = static_cast<size_t>(failure.suspectedStage);
        int32_t bounce =
            failure.suspectedStage >= 0 && suspectedIndex < mStageBounces.size() ? mStageBounces[suspectedIndex] : -1;
        if (mStageBounces.empty())
        {
            for (int32_t i = 0; i <= failure.suspectedStage; ++i)
            {
                if (mStageKinds[i] == kStageGenerate)
                {
                    bounce = -1;
                }
                else if (mStageKinds[i] == kStageExtend)
                {
                    ++bounce;
                }
            }
        }
        // -1 means no extend stage ran since the last generate, so there is no
        // bounce to report. Converted once, past that test, and the slot index
        // is what the rest of the block wants anyway.
        const uint32_t bounceIndex = static_cast<uint32_t>(bounce);
        if (bounce >= 0 && bounceIndex < kWavefrontStageDiagnosticBounces)
        {
            const uint32_t base = kWavefrontStageDiagnosticBase + bounceIndex * kWavefrontStageDiagnosticStride;
            const uint32_t active = stats[base];
            STRELKA_ERROR("Metal 4 stage diagnosis: bounce={} active_paths={}", bounce, active);
            const uint32_t lanes = std::min(active, kWavefrontStageDiagnosticLanes);
            for (uint32_t lane = 0; lane < lanes; ++lane)
            {
                const uint32_t* d = stats + static_cast<size_t>(base) + 2u +
                                    static_cast<size_t>(lane) * kWavefrontStageDiagnosticLaneUints;
                const uint32_t medium = d[1];
                STRELKA_ERROR(
                    "Metal 4 pending ray {}: tid={} medium=0x{:x} depth_flags=0x{:x} "
                    "origin=({:.6g},{:.6g},{:.6g}) direction=({:.6g},{:.6g},{:.6g}) "
                    "dir_len2={:.6g} throughput_max={:.6g}",
                    lane, d[0], medium, d[2], std::bit_cast<float>(d[3]), std::bit_cast<float>(d[4]),
                    std::bit_cast<float>(d[5]), std::bit_cast<float>(d[6]), std::bit_cast<float>(d[7]),
                    std::bit_cast<float>(d[8]), std::bit_cast<float>(d[9]), std::bit_cast<float>(d[10]));
            }
        }
    }
}

void MetalWavefrontIntegrator::reportStageTimings()
{
    if (!mStageTimestampBuffer || mStageKinds.empty())
    {
        return;
    }
    const NS::UInteger n = 2 * mStageKinds.size();
    const NS::Data* data = mStageTimestampBuffer->resolveCounterRange(NS::Range::Make(0, n));
    if (!data)
    {
        return;
    }
    const auto* ts = static_cast<const MTL::CounterResultTimestamp*>(data->bytes());
    std::vector<uint64_t> timestamps(n);
    for (NS::UInteger i = 0; i < n; ++i)
    {
        timestamps[i] = ts[i].timestamp;
    }
    reportStageTimestampValues(timestamps.data(), timestamps.size(), 1.0);
}

void MetalWavefrontIntegrator::reportStageTimingsMetal4()
{
    if (!mStageTimestampHeap4 || mStageKinds.empty())
    {
        return;
    }
    const NS::UInteger n = 2 * mStageKinds.size();
    const NS::Data* data = mStageTimestampHeap4->resolveCounterRange(NS::Range::Make(0, n));
    if (!data || data->length() < n * sizeof(MTL4::TimestampHeapEntry))
    {
        return;
    }
    const auto* entries = static_cast<const MTL4::TimestampHeapEntry*>(data->bytes());
    std::vector<uint64_t> timestamps(n);
    for (NS::UInteger i = 0; i < n; ++i)
    {
        timestamps[i] = entries[i].timestamp;
    }
    MTL::Timestamp cpuEnd = 0;
    MTL::Timestamp gpuEnd = 0;
    mDevice->sampleTimestamps(&cpuEnd, &gpuEnd);
    if (cpuEnd <= mStageCpuTimestampStart || gpuEnd <= mStageGpuTimestampStart)
    {
        return;
    }
    mach_timebase_info_data_t timebase{};
    mach_timebase_info(&timebase);
    const double cpuNanoseconds = static_cast<double>(cpuEnd - mStageCpuTimestampStart) * timebase.numer / timebase.denom;
    const double nanosecondsPerGpuTick = cpuNanoseconds / static_cast<double>(gpuEnd - mStageGpuTimestampStart);
    reportStageTimestampValues(timestamps.data(), timestamps.size(), nanosecondsPerGpuTick);
}

void MetalWavefrontIntegrator::reportStageTimestampValues(const uint64_t* timestamps,
                                                          size_t count,
                                                          double nanosecondsPerTick)
{
    if (!timestamps || count < 2 * mStageKinds.size())
    {
        return;
    }

    double totals[kStageCount] = {};
    uint32_t counts[kStageCount] = {};
    double primaryExtendMs = 0.0;
    // Per-bounce durations of the three traversal-heavy stages. The cost of a
    // bounce says more than the total does: bounce 0 is a coherent primary pass
    // and the later ones are not, which is what decides whether sorting rays is
    // worth anything.
    std::string perBounce[kStageCount];
    for (NS::UInteger i = 0; i < mStageKinds.size(); ++i)
    {
        const uint64_t a = timestamps[2 * i];
        const uint64_t b = timestamps[2 * i + 1];
        if (a == MTL::CounterErrorValue || b == MTL::CounterErrorValue || b <= a)
        {
            continue;
        }
        const uint8_t kind = mStageKinds[i];
        const double ms = static_cast<double>(b - a) * nanosecondsPerTick / 1e6;
        totals[kind] += ms;
        ++counts[kind];
        if (kind == kStageExtend && i < mStageBounces.size() && mStageBounces[i] == 0)
        {
            primaryExtendMs += ms;
        }
        if (kind == kStageExtend || kind == kStageSssWalk || kind == kStageConnect || kind == kStageShade ||
            kind == kStageShadeBase || kind == kStageShadeLayer || kind == kStageShadeTranslucent ||
            kind == kStageShadeTail || kind == kStageShadow)
        {
            perBounce[kind] += fmt::format("{:.2f} ", ms);
        }
    }

    double sum = 0.0;
    for (const double total : totals)
    {
        sum += total;
    }
    std::string line;
    for (uint32_t k = 0; k < kStageCount; ++k)
    {
        if (counts[k] == 0)
        {
            continue;
        }
        line += fmt::format("{} {:.2f}ms({:.0f}%, n={})  ", kStageNames[k], totals[k],
                            sum > 0.0 ? 100.0 * totals[k] / sum : 0.0, counts[k]);
    }
    STRELKA_INFO("STAGES total {:.2f}ms  {}", sum, line);
    if (totals[kStageExtend] > 0.0)
    {
        STRELKA_INFO("STAGES primary extend {:.2f}ms ({:.1f}% of extend)", primaryExtendMs,
                     100.0 * primaryExtendMs / totals[kStageExtend]);
    }
    STRELKA_INFO(
        "STAGES per dispatch: extend [{}] sss [{}] connect [{}] base [{}] layer [{}] translucent [{}] tail [{}] "
        "shade [{}] shadow [{}]",
        perBounce[kStageExtend], perBounce[kStageSssWalk], perBounce[kStageConnect], perBounce[kStageShadeBase],
        perBounce[kStageShadeLayer], perBounce[kStageShadeTranslucent], perBounce[kStageShadeTail],
        perBounce[kStageShade], perBounce[kStageShadow]);

    // The legacy path copies queue counters into this fixed layout. Metal 4
    // uses the same buffer for breadcrumbs and per-lane hang diagnostics.
    if (mStageStatsBuffer && mStageTimestampBuffer)
    {
        const uint32_t* stats = static_cast<const uint32_t*>(mStageStatsBuffer->contents());
        std::string paths, shadows;
        for (uint32_t i = 0; i < counts[kStageExtend]; ++i)
        {
            paths += fmt::format("{:.0f}k ", stats[32 + i] / 1000.0);
            shadows += fmt::format("{:.0f}k ", stats[64 + i] / 1000.0);
        }
        STRELKA_INFO("STAGES rays per bounce: paths [{}] shadow [{}]", paths, shadows);
    }
    else if (mStageStatsBuffer && mStageTimestampHeap4)
    {
        const uint32_t* stats = static_cast<const uint32_t*>(mStageStatsBuffer->contents());
        std::string sssOccupancy;
        for (uint32_t bounce = 0; bounce < kWavefrontStageDiagnosticBounces; ++bounce)
        {
            const int32_t bounceIndex = static_cast<int32_t>(bounce);
            const bool measured = std::ranges::any_of(
                mStageBounces, [bounceIndex](int32_t stageBounce) { return stageBounce == bounceIndex; });
            if (!measured)
            {
                continue;
            }
            const uint32_t base = kWavefrontStageDiagnosticBase + bounce * kWavefrontStageDiagnosticStride;
            const uint32_t active = stats[base];
            const uint32_t sss = stats[base + kWavefrontStageDiagnosticSssCountOffset];
            sssOccupancy += fmt::format(
                "{}/{}({:.1f}%) ", sss, active, active != 0u ? 100.0 * static_cast<double>(sss) / active : 0.0);
        }
        STRELKA_INFO("STAGES SSS paths/active per bounce: [{}]", sssOccupancy);
    }
}

// Report nested-dielectric loss once per scene; generate clears the per-sample counters.
// Overflow needs a deeper stack, while unmatched or escaped paths indicate broken volume boundaries.
void MetalWavefrontIntegrator::reportIorStackStats()
{
    if (mReportedIorStats || !mIorStatsBuffer)
    {
        return;
    }
    const uint32_t* stats = static_cast<const uint32_t*>(mIorStatsBuffer->contents());
    const uint32_t overflow = stats[IOR_STAT_OVERFLOW];
    const uint32_t unmatched = stats[IOR_STAT_UNMATCHED];
    const uint32_t escaped = stats[IOR_STAT_ESCAPED_INSIDE];
    if (overflow == 0 && unmatched == 0 && escaped == 0)
    {
        return;
    }
    mReportedIorStats = true;
    STRELKA_WARNING(
        "Nested dielectrics lost paths, per sample: {} push(es) onto a full stack of {}, "
        "{} pop(s) that matched nothing, {} path(s) that reached the environment still "
        "inside a medium. The first wants a deeper stack; the other two are a mesh with a "
        "hole in it, seen from each side -- and the third is the one no exit event can "
        "catch, because the ray left through the hole. Each of them carries the wrong "
        "medium, and therefore the wrong absorption, for the rest of its life.",
        overflow, int{ IOR_STACK_SIZE }, unmatched, escaped);
}

void MetalWavefrontIntegrator::reportSharcStats()
{
    if (!mIorStatsBuffer)
    {
        return;
    }
    const uint32_t* stats = static_cast<const uint32_t*>(mIorStatsBuffer->contents()) + IOR_STAT_COUNT;
    const uint32_t attempts = stats[SHARC_STAT_QUERY_ATTEMPT];
    uint64_t activity = 0;
    for (uint32_t i = 0; i < SHARC_STAT_COUNT; ++i)
    {
        activity += stats[i];
    }
    if (activity == 0u || activity == mLastSharcActivity)
    {
        return;
    }
    mLastSharcActivity = activity;
    const uint32_t hits = stats[SHARC_STAT_QUERY_HIT];
    const auto occupiedBits = [](uint32_t value) -> uint32_t {
        return value == 0u ? 0u : 32u - static_cast<uint32_t>(std::countl_zero(value));
    };
    STRELKA_INFO(
        "SHARC stats: insertions={} failed={} collisions={} queries={} hits={} hit_rate={:.1f}% "
        "evictions={} segment_rejects={} footprint_rejects={} receiver_rejects={} accumulation_clamps={} "
        "nonfinite_rejects={} radiance_bits={} sample_bits={}",
        stats[SHARC_STAT_INSERTION], stats[SHARC_STAT_INSERTION_FAILURE], stats[SHARC_STAT_COLLISION], attempts, hits,
        attempts ? 100.0 * static_cast<double>(hits) / attempts : 0.0, stats[SHARC_STAT_EVICTION],
        stats[SHARC_STAT_SEGMENT_REJECT], stats[SHARC_STAT_FOOTPRINT_REJECT], stats[SHARC_STAT_RECEIVER_REJECT],
        stats[SHARC_STAT_ACCUMULATION_CLAMP], stats[SHARC_STAT_NONFINITE_REJECT],
        occupiedBits(stats[SHARC_STAT_MAX_RADIANCE_FIXED]), occupiedBits(stats[SHARC_STAT_MAX_SAMPLE_COUNT]));
}

void MetalWavefrontIntegrator::resetStageProfilingMetal4()
{
    mStageKinds.clear();
    mStageBounces.clear();
    if (mStageTimestampHeap4)
    {
        mStageTimestampHeap4->invalidateCounterRange(NS::Range::Make(0, static_cast<NS::UInteger>(2) * kMaxStageSamples));
        mDevice->sampleTimestamps(&mStageCpuTimestampStart, &mStageGpuTimestampStart);
    }
    if (mStageStatsBuffer)
    {
        auto* stats = static_cast<uint32_t*>(mStageStatsBuffer->contents());
        stats[kWavefrontStageBreadcrumbOffset] = kWavefrontStageNotStarted;
    }
}

void MetalWavefrontIntegrator::beginRenderWorkAudit()
{
    constexpr size_t diagnosticBytes =
        RESTIR_DIAGNOSTIC_PIXEL_COUNT *
        (sizeof(uint32_t) + sizeof(RestirDiagnosticRecord) + sizeof(RestirCandidateAuditRecord));
    constexpr size_t lightIdBytes = RESTIR_AUDIT_LIGHT_ID_WORDS * sizeof(uint32_t);
    if (!mRenderWorkCounterBuffer)
    {
        mRenderWorkCounterBuffer = mDevice->newBuffer(
            WORK_COUNTER_COUNT * sizeof(uint32_t) + lightIdBytes + diagnosticBytes, MTL::ResourceStorageModeShared);
        mResidencyDirty = true;
    }
    if (mRenderWorkCounterBuffer)
    {
        memset(mRenderWorkCounterBuffer->contents(), 0, mRenderWorkCounterBuffer->length());
        auto* words = static_cast<uint32_t*>(mRenderWorkCounterBuffer->contents());
        auto* pixels = words + WORK_COUNTER_COUNT + RESTIR_AUDIT_LIGHT_ID_WORDS;
        std::fill_n(pixels, RESTIR_DIAGNOSTIC_PIXEL_COUNT, std::numeric_limits<uint32_t>::max());
        // NOLINTNEXTLINE(concurrency-mt-unsafe) -- audit setup precedes worker threads.
        if (const char* value = std::getenv("STRELKA_RESTIR_DIAGNOSTIC_PIXELS"))
        {
            for (uint32_t i = 0; i < RESTIR_DIAGNOSTIC_PIXEL_COUNT && *value != '\0'; ++i)
            {
                char* end = nullptr;
                const unsigned long pixel = std::strtoul(value, &end, 10);
                if (end == value || pixel > std::numeric_limits<uint32_t>::max())
                {
                    break;
                }
                pixels[i] = static_cast<uint32_t>(pixel);
                value = *end == ',' ? end + 1 : end;
            }
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto* records = reinterpret_cast<RestirDiagnosticRecord*>(pixels + RESTIR_DIAGNOSTIC_PIXEL_COUNT);
        for (uint32_t i = 0; i < RESTIR_DIAGNOSTIC_PIXEL_COUNT; ++i)
        {
            records[i].pixelIndex = pixels[i];
        }
        auto* candidateRecords = reinterpret_cast<RestirCandidateAuditRecord*>(records + RESTIR_DIAGNOSTIC_PIXEL_COUNT);
        for (uint32_t i = 0; i < RESTIR_DIAGNOSTIC_PIXEL_COUNT; ++i)
        {
            candidateRecords[i].pixelIndex = pixels[i];
            candidateRecords[i].frameIndex = std::numeric_limits<uint32_t>::max();
        }
    }
    mRenderWorkDispatches.clear();
}

const RestirCandidateAuditRecord* MetalWavefrontIntegrator::restirCandidateAuditRecords() const
{
    const RestirDiagnosticRecord* records = restirDiagnosticRecords();
    return records ? reinterpret_cast<const RestirCandidateAuditRecord*>(records + RESTIR_DIAGNOSTIC_PIXEL_COUNT) :
                     nullptr;
}

const uint32_t* MetalWavefrontIntegrator::renderWorkCounters() const
{
    return mRenderWorkCounterBuffer ? static_cast<const uint32_t*>(mRenderWorkCounterBuffer->contents()) : nullptr;
}

const RestirDiagnosticRecord* MetalWavefrontIntegrator::restirDiagnosticRecords() const
{
    if (!mRenderWorkCounterBuffer)
    {
        return nullptr;
    }
    const auto* words = static_cast<const uint32_t*>(mRenderWorkCounterBuffer->contents());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<const RestirDiagnosticRecord*>(words + WORK_COUNTER_COUNT + RESTIR_AUDIT_LIGHT_ID_WORDS +
                                                           RESTIR_DIAGNOSTIC_PIXEL_COUNT);
}

#ifndef NDEBUG
const uint32_t* MetalWavefrontIntegrator::restirAuditLightIdWords() const
{
    const auto* words = renderWorkCounters();
    return words ? words + WORK_COUNTER_COUNT : nullptr;
}
#endif

uint64_t MetalWavefrontIntegrator::renderWorkCounterAddress() const
{
    return mRenderWorkCounterBuffer ? mRenderWorkCounterBuffer->gpuAddress() : 0ull;
}

void MetalWavefrontIntegrator::encodeMetal4(MTL4::ComputeCommandEncoder*& enc,
                                            const IntegratorSceneBindings& scene,
                                            const IntegratorFrameRequest& frame,
                                            const WavefrontChunk& chunk)
{
    const uint32_t features = frame.features;
    const uint32_t width = frame.width;
    const uint32_t height = frame.height;
    const uint32_t sampleCount = frame.sampleCount;
    MTL::Buffer* uniformBuffer = frame.uniformBuffer;
    Buffer* output = frame.output;

    const WavefrontVariant* variant = variantFor(features | WavefrontFeatures::kMetal4);
    if (!variant)
    {
        return;
    }
    const uint32_t pixels = frame.pathCount != 0u ? frame.pathCount : width * height;
    MTL::Buffer* outputBuffer = ((MetalBuffer*)output)->getNativePtr();
    const auto* uniforms = static_cast<const Uniforms*>(uniformBuffer->contents());
    const bool restirDiagnostic = (features & WavefrontFeatures::kRestirRayTracedDiagnostic) != 0u;

    MTL4::ArgumentTable* table = mMetal4->argumentTable();
    ConstantRing& ring = mMetal4->constants();
    enc->setArgumentTable(table);

    const uint32_t kThreadsPerGroup = 64;
    const MTL::Size tg = MTL::Size(kThreadsPerGroup, 1, 1);
    const MTL::Size fullGrid = MTL::Size((pixels + kThreadsPerGroup - 1) / kThreadsPerGroup, 1, 1);
    const bool sharcUpdatePass = (features & WavefrontFeatures::kSharcUpdate) != 0u;
    const uint32_t generateScale = sharcUpdatePass && uniforms ? std::max(uniforms->sharcUpdateDownscale, 1u) : 1u;
    const uint32_t generateWidth = (width + generateScale - 1u) / generateScale;
    const uint32_t generateHeight = (height + generateScale - 1u) / generateScale;
    const MTL::Size generateTg = MTL::Size(32, 8, 1);
    const MTL::Size generateGrid = MTL::Size((generateWidth + generateTg.width - 1) / generateTg.width,
                                             (generateHeight + generateTg.height - 1) / generateTg.height, 1);
    const MTL::GPUAddress control = mControlBuffer->gpuAddress();
    const MTL::GPUAddress traversalDispatches = mTraversalDispatchBuffer->gpuAddress();
    const NS::UInteger kShadowCounterOffset = 6 * sizeof(uint32_t);
    const NS::UInteger kHitArgsOffset = 13 * sizeof(uint32_t);
    const NS::UInteger kHitCounterOffset = 11 * sizeof(uint32_t);
    const NS::UInteger kShadeBaseArgsOffset = 80 * sizeof(uint32_t);
    const NS::UInteger kShadeLayerArgsOffset = 84 * sizeof(uint32_t);
    const NS::UInteger kShadeTranslucentArgsOffset = 88 * sizeof(uint32_t);
    const NS::UInteger kShadeTailArgsOffset = 92 * sizeof(uint32_t);
    const NS::UInteger kMissArgsOffset = 18 * sizeof(uint32_t);
    const NS::UInteger kMissCounterOffset = 16 * sizeof(uint32_t);
    const NS::UInteger kGuideArgsOffset = 25 * sizeof(uint32_t);
    const NS::UInteger kRestirArgsOffset = 29 * sizeof(uint32_t);
    const uint32_t traversalBatchThreads = frame.traversalBatchThreads;
    const uint32_t traversalBatchCount = wavefrontTraversalBatchCount(pixels, traversalBatchThreads);
    const uint32_t shadowBatchThreads =
        (features & WavefrontFeatures::kCurves) != 0 ? kWavefrontTraversalBatchThreads : traversalBatchThreads;
    const uint32_t shadowBatchCount = wavefrontTraversalBatchCount(pixels, shadowBatchThreads);
    const MTL::GPUAddress traversalBatchThreadsAddress = ring.push(traversalBatchThreads);
    const MTL::GPUAddress traversalBatchCountAddress = ring.push(traversalBatchCount);

    const bool useMotion = frame.motionBlasBuilt || variant->extendStatic == nullptr ||
                           frame.settings->getAs<uint32_t>("render/pt/staticTraversal") == 0;
    // A terminal camera hit never samples a BSDF and therefore cannot enter an
    // SSS medium. Keep the SSS-specialised shade variant, but do not scan the
    // full camera queue or encode the empty SSS walk for a depth-1 capture.
    const bool fusedSss = uniforms && uniforms->maxDepth > 1u && (features & WavefrontFeatures::kSubsurface) != 0u &&
                          (features & WavefrontFeatures::kSharcUpdate) == 0u && variant->sssWalkMotion &&
                          variant->sssWalkStatic;

    // Every dispatch here reads what the one before it wrote. Metal 4 does not
    // work that out, so say it: dispatch-to-dispatch, visible device-wide.
    auto barrier = [&]() {
        enc->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    };
    auto traversalBatchBarrier = [&]() {
        // This is an execution boundary only. Intermediate traversal batches
        // append through atomics and consume none of each other's ordinary
        // writes; the device-wide visibility barrier after the final batch
        // publishes their combined queues to the next stage.
        enc->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionNone);
    };
    auto bind = [&](MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index) {
        // The table is sized once, in Metal4Context, and every stage shares it.
        // Binding past its end is silent without the debug layer, so say it here
        // too: this is the file that keeps growing new bindings.
        assert(index < kMetal4BufferBindCount);
        if (!buffer)
        {
            buffer = scene.placeholderBuffer;
            offset = 0;
        }
        table->setAddress(buffer ? buffer->gpuAddress() + offset : 0, index);
    };

    const uint32_t s = chunk.sampleIndex;
    const uint32_t accumulatedSample = uniforms ? uniforms->subframeIndex + s : s;
    const uint32_t profileStart = envUint("STRELKA_STAGE_PROFILE_START", 0u);
    const uint32_t profileCount = envUint("STRELKA_STAGE_PROFILE_COUNT", 4u);
    const bool profileThisSample =
        frame.profileStages && accumulatedSample >= profileStart && accumulatedSample - profileStart < profileCount;

    // Breadcrumbs identify a failed stage; precise timestamps measure
    // successful runs. Limit the default window because hundreds of precise
    // boundaries on every sample of a long SSS render eventually trip the GPU
    // watchdog. STRELKA_STAGE_PROFILE_START/COUNT select another window.
    const bool diagnose = profileThisSample && envUint("STRELKA_STAGE_BREADCRUMBS", 0u) != 0u && mStageBreadcrumbPSO4 &&
                          mStageStatsBuffer;
    // Keep the two modes mutually exclusive. A breadcrumb is a one-thread
    // dispatch plus a barrier; combining hundreds of them with precise
    // timestamp boundaries can trip the GPU watchdog in long SSS frames.
    const bool timeStages = profileThisSample && !diagnose && mStageTimestampHeap4;
    auto shouldTimeStage = [](uint8_t kind) {
        // Precise timestamps force prior GPU work to finish and can split an
        // encoder. Keep them around the costly stages; timing every one-thread
        // prepare dispatch made repeated SSS profiles trip the watchdog.
        return kind != kStagePrepare && kind != kStagePrepareShadow;
    };
    auto writeBreadcrumb = [&](uint32_t stageIndex) {
        enc->setComputePipelineState(mStageBreadcrumbPSO4);
        bind(mStageStatsBuffer, kWavefrontStageBreadcrumbOffset * sizeof(uint32_t), 0);
        table->setAddress(ring.push(stageIndex), 1);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
        barrier();
    };
    constexpr uint32_t kInvalidStageSample = std::numeric_limits<uint32_t>::max();
    auto beginStage = [&](uint8_t kind, uint32_t bounce = std::numeric_limits<uint32_t>::max()) -> uint32_t {
        if ((!diagnose && !timeStages) || mStageKinds.size() >= kMaxStageSamples)
        {
            return kInvalidStageSample;
        }
        const uint32_t stageIndex = static_cast<uint32_t>(mStageKinds.size());
        mStageKinds.push_back(kind);
        mStageBounces.push_back(
            bounce <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ? static_cast<int32_t>(bounce) : -1);
        if (diagnose)
        {
            writeBreadcrumb(stageIndex);
        }
        if (timeStages && shouldTimeStage(kind))
        {
            enc->writeTimestamp(
                MTL4::TimestampGranularityPrecise, mStageTimestampHeap4, static_cast<NS::UInteger>(2) * stageIndex);
        }
        return stageIndex;
    };
    auto endStage = [&](uint32_t stageIndex) {
        if (timeStages && stageIndex != kInvalidStageSample && shouldTimeStage(mStageKinds[stageIndex]))
        {
            enc->writeTimestamp(MTL4::TimestampGranularityPrecise, mStageTimestampHeap4,
                                static_cast<NS::UInteger>(2) * stageIndex + 1u);
        }
    };
    auto auditDispatch = [&](const char* label, uint64_t count = 1u) {
        if (frame.auditRenderWork)
        {
            mRenderWorkDispatches[label] += count;
        }
    };

    const MTL::GPUAddress sampleIdx = ring.push(s);
    // A few queue counters and four diagnostic lanes are cheap enough to
    // collect with stage profiling. Breadcrumb dispatches remain opt-in.
    const MTL::GPUAddress stageCountersEnabled = ring.push(profileThisSample ? 1u : 0u);
    const MTL::GPUAddress diagnosticsMediumEnabled =
        ring.push((features & WavefrontFeatures::kSubsurface) != 0u ? 1u : 0u);
    if (chunk.generate)
    {
        auditDispatch("wavefrontGenerate");
        const uint32_t stage = beginStage(kStageGenerate);
        enc->setComputePipelineState(variant->generate);
        bind(uniformBuffer, 0, 0);
        bind(mPathStateBuffer, 0, 1);
        bind(mRadianceBuffer, 0, 2);
        bind(mIorStackBuffer, 0, 3);
        table->setAddress(sampleIdx, 4);
        bind(mPathQueueBuffer[0], 0, 5);
        bind(mControlBuffer, 0, 6);
        bind(mAovBuffer, 0, 7);
        bind(mPathRayBuffer, 0, 8);
        bind(mIorStatsBuffer, 0, 9);
        bind(mSharcUpdateStateBuffer, 0, 10);
        bind(mMediumPathStateBuffer, 0, 11);
        enc->dispatchThreadgroups(generateGrid, generateTg);
        barrier();
        endStage(stage);
    }

    for (uint32_t bounce = chunk.bounceBegin; bounce < chunk.bounceEnd; ++bounce)
    {
        const uint32_t src = bounce & 1u;
        const uint32_t dst = src ^ 1u;
        const MTL::GPUAddress srcIdx = ring.push(src);
        const MTL::GPUAddress groupSize = ring.push(kThreadsPerGroup);
        const MTL::GPUAddress bounceIdx = ring.push(bounce);
        const bool encodeExtend = chunk.phase != WavefrontChunkPhase::Finish;
        const bool finishBounce = chunk.phase != WavefrontChunkPhase::Extend;
        const bool beginExtend = chunk.phase == WavefrontChunkPhase::Complete || chunk.traversalBatchBegin == 0u;

        if (encodeExtend)
        {
            // Prepare owns clearing the append counters, so only the first
            // partial extend workload may run it. Later workloads append another
            // disjoint range of the source queue into the same hit/miss queues.
            if (beginExtend)
            {
                auditDispatch("wavefrontPrepare");
                const uint32_t stage = beginStage(kStagePrepare, bounce);
                enc->setComputePipelineState(mPreparePSO4);
                bind(mControlBuffer, 0, 0);
                table->setAddress(srcIdx, 1);
                table->setAddress(groupSize, 2);
                table->setAddress(bounceIdx, 3);
                bind(mStageStatsBuffer, 0, 4);
                bind(mPathStateBuffer, 0, 5);
                bind(mPathRayBuffer, 0, 6);
                bind(mPathQueueBuffer[src], 0, 7);
                table->setAddress(stageCountersEnabled, 8);
                bind(mTraversalDispatchBuffer, 0, 9);
                table->setAddress(traversalBatchThreadsAddress, 10);
                table->setAddress(traversalBatchCountAddress, 11);
                bind(mMediumPathStateBuffer, 0, 12);
                table->setAddress(diagnosticsMediumEnabled, 13);
                bind(mSssControlBuffer, 0, 14);
                bind(mHitQueueBuffer, 0, 15);
                enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
                barrier();
                endStage(stage);
            }

            const uint32_t extendStage = beginStage(kStageExtend, bounce);
            enc->setComputePipelineState(useMotion ? variant->extendMotion : variant->extendStatic);
            bind(uniformBuffer, 0, 0);
            bind(scene.instanceBuffer, 0, 1);
            table->setResource(scene.instanceAccelerationStructure->gpuResourceID(), 2);
            bind(mPathRayBuffer, 0, 3);
            bind(mHitBuffer, 0, 4);
            table->setAddress(sampleIdx, 5);
            bind(mPathQueueBuffer[src], 0, 6);
            bind(mControlBuffer, 0, 7);
            bind(mHitQueueBuffer, 0, 8);
            bind(mControlBuffer, kHitCounterOffset, 9);
            bind(mMissQueueBuffer, 0, 10);
            bind(mControlBuffer, kMissCounterOffset, 11);
            bind(mPathStateBuffer, 0, 12);
            bind(scene.materialBuffer, 0, 13);
            // Only the camera bounce is denied the hidden lights.
            const uint32_t extendMask =
                (bounce == 0) ? uniforms->primaryRayMask : (uniforms->primaryRayMask | GEOMETRY_MASK_LIGHT_HIDDEN);
            table->setAddress(ring.push(extendMask), 14);
            table->setResource(scene.volumeAccelerationStructure->gpuResourceID(), 15);
            bind(mMediumPathStateBuffer, 0, 17);
            bind(scene.geometryEntryBuffer, 0, 18);
            table->setResource(
                (useMotion ? variant->extendTableMotion : variant->extendTableStatic)->gpuResourceID(), 19);
            bind(scene.vertexBuffer, 0, 20);
            bind(scene.prevVertexBuffer, 0, 21);
            bind(scene.indexBuffer, 0, 22);
            const uint32_t batchBegin = chunk.phase == WavefrontChunkPhase::Complete ? 0u : chunk.traversalBatchBegin;
            const uint32_t batchEnd = chunk.phase == WavefrontChunkPhase::Complete ?
                                          traversalBatchCount :
                                          std::min(chunk.traversalBatchEnd, traversalBatchCount);
            for (uint32_t batch = batchBegin; batch < batchEnd; ++batch)
            {
                if (fusedSss)
                {
                    auditDispatch("wavefrontClassifySss");
                    enc->setComputePipelineState(mClassifySssPSO4);
                    bind(uniformBuffer, 0, 0);
                    bind(mPathQueueBuffer[src], 0, 1);
                    bind(mControlBuffer, 0, 2);
                    bind(mMediumPathStateBuffer, 0, 3);
                    bind(scene.materialBuffer, 0, 4);
                    bind(mSssQueueBuffer, 0, 5);
                    bind(mSssControlBuffer, 0, 6);
                    table->setAddress(ring.push(batch * traversalBatchThreads), 7);
                    enc->dispatchThreadgroups(
                        traversalDispatches + static_cast<MTL::GPUAddress>(batch) * 3u * sizeof(uint32_t), tg);
                    traversalBatchBarrier();
                    // All Metal 4 kernels share one argument table. Classification
                    // overwrote its first slots, so restore extend's bindings.
                    enc->setComputePipelineState(useMotion ? variant->extendMotion : variant->extendStatic);
                    bind(uniformBuffer, 0, 0);
                    bind(scene.instanceBuffer, 0, 1);
                    table->setResource(scene.instanceAccelerationStructure->gpuResourceID(), 2);
                    bind(mPathRayBuffer, 0, 3);
                    bind(mHitBuffer, 0, 4);
                    table->setAddress(sampleIdx, 5);
                    bind(mPathQueueBuffer[src], 0, 6);
                    bind(mControlBuffer, 0, 7);
                    bind(mHitQueueBuffer, 0, 8);
                    bind(mControlBuffer, kHitCounterOffset, 9);
                    bind(mMissQueueBuffer, 0, 10);
                    bind(mControlBuffer, kMissCounterOffset, 11);
                    bind(mPathStateBuffer, 0, 12);
                    bind(scene.materialBuffer, 0, 13);
                    table->setAddress(ring.push(extendMask), 14);
                    table->setResource(scene.volumeAccelerationStructure->gpuResourceID(), 15);
                    bind(mMediumPathStateBuffer, 0, 17);
                    bind(scene.geometryEntryBuffer, 0, 18);
                    table->setResource(
                        (useMotion ? variant->extendTableMotion : variant->extendTableStatic)->gpuResourceID(), 19);
                    bind(scene.vertexBuffer, 0, 20);
                    bind(scene.prevVertexBuffer, 0, 21);
                    bind(scene.indexBuffer, 0, 22);
                }
                auditDispatch(useMotion ? "wavefrontExtend" : "wavefrontExtendStatic");
                table->setAddress(ring.push(batch * traversalBatchThreads), 16);
                enc->dispatchThreadgroups(
                    traversalDispatches + static_cast<MTL::GPUAddress>(batch) * 3u * sizeof(uint32_t), tg);
                if (batch + 1 < batchEnd)
                {
                    traversalBatchBarrier();
                }
            }
            barrier();
            endStage(extendStage);
        }

        if (fusedSss && finishBounce)
        {
            const uint32_t stage = beginStage(kStageSssWalk, bounce);
            auditDispatch("wavefrontPrepareSss");
            enc->setComputePipelineState(mPrepareSssPSO4);
            bind(mSssControlBuffer, 0, 0);
            table->setAddress(groupSize, 1);
            bind(mStageStatsBuffer, 0, 2);
            table->setAddress(bounceIdx, 3);
            table->setAddress(stageCountersEnabled, 4);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            barrier();

            auditDispatch(useMotion ? "wavefrontSssWalk" : "wavefrontSssWalkStatic");
            enc->setComputePipelineState(useMotion ? variant->sssWalkMotion : variant->sssWalkStatic);
            bind(uniformBuffer, 0, 0);
            bind(scene.instanceBuffer, 0, 1);
            table->setResource(scene.volumeAccelerationStructure->gpuResourceID(), 2);
            bind(mPathRayBuffer, 0, 3);
            bind(mPathStateBuffer, 0, 4);
            bind(mMediumPathStateBuffer, 0, 5);
            bind(scene.materialBuffer, 0, 6);
            table->setAddress(sampleIdx, 7);
            bind(mSssQueueBuffer, 0, 8);
            bind(mSssControlBuffer, 0, 9);
            bind(mHitBuffer, 0, 10);
            bind(mHitQueueBuffer, 0, 11);
            bind(mControlBuffer, kHitCounterOffset, 12);
            bind(mMissQueueBuffer, 0, 13);
            bind(mControlBuffer, kMissCounterOffset, 14);
            bind(mPathQueueBuffer[dst], 0, 15);
            bind(mControlBuffer, dst * sizeof(uint32_t), 16);
            bind(mControlBuffer, 0, 17);
            const uint32_t extendMask =
                (bounce == 0) ? uniforms->primaryRayMask : (uniforms->primaryRayMask | GEOMETRY_MASK_LIGHT_HIDDEN);
            table->setAddress(ring.push(extendMask), 18);
            bind(scene.geometryEntryBuffer, 0, 19);
            enc->dispatchThreadgroups(mSssControlBuffer->gpuAddress() + 2u * sizeof(uint32_t), tg);
            barrier();
            endStage(stage);
        }

        if (finishBounce)
        {
            auditDispatch("wavefrontPrepareHitMiss");
            const uint32_t prepareStage = beginStage(kStagePrepare, bounce);
            enc->setComputePipelineState(mPrepareHitMissPSO4);
            bind(mControlBuffer, 0, 0);
            table->setAddress(groupSize, 1);
            bind(mHitQueueBuffer, 0, 2);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            barrier();
            endStage(prepareStage);

            const uint32_t missStage = beginStage(kStageMiss, bounce);
            auditDispatch("wavefrontMiss");
            enc->setComputePipelineState(variant->miss);
            bind(uniformBuffer, 0, 0);
            bind(mPathStateBuffer, 0, 1);
            bind(mPathRayBuffer, 0, 2);
            bind(mRadianceBuffer, 0, 3);
            bind(mMissQueueBuffer, 0, 4);
            bind(mControlBuffer, 0, 5);
            bind(mAovBuffer, 0, 6);
            table->setAddress(sampleIdx, 7);
            bind(mIorStackBuffer, 0, 8);
            bind(mIorStatsBuffer, 0, 9);
            bind(mSharcUpdateStateBuffer, 0, 10);
            bind(scene.sharcAccumulationBuffer, 0, 11);
            bind(scene.lightBuffer, 0, 12);
            bind(scene.environment ? scene.environment->state().aliasBuffer : nullptr, 0, 13);
            if (scene.environment && scene.environment->state().mapTexture)
            {
                table->setTexture(scene.environment->state().mapTexture->gpuResourceID(), 0);
                table->setTexture(
                    (scene.environment->state().backgroundTexture ? scene.environment->state().backgroundTexture :
                                                                    scene.environment->state().mapTexture)
                        ->gpuResourceID(),
                    1);
            }
            enc->dispatchThreadgroups(control + kMissArgsOffset, tg);
            const bool sharcUpdate = (features & WavefrontFeatures::kSharcUpdate) != 0u;

            // Miss and shade touch disjoint queue entries, so production does
            // not need a barrier between them. Diagnosis does: seeing the shade
            // breadcrumb must prove that miss completed, not merely overlapped.
            if (diagnose || (uniforms->restirDIEnabled != 0u && bounce == 0u && !sharcUpdate))
            {
                barrier();
            }
            endStage(missStage);

            if (variant->connectBase)
            {
                auditDispatch("wavefrontConnectBase");
                const uint32_t connectStage = beginStage(kStageConnect, bounce);
                enc->setComputePipelineState(variant->connectBase);
                bind(uniformBuffer, 0, 0);
                bind(scene.instanceBuffer, 0, 1);
                bind(scene.iesBuffer, 0, 2);
                bind(scene.lightBuffer, 0, 3);
                bind(scene.materialBuffer, 0, 4);
                bind(mPathStateBuffer, 0, 5);
                bind(mPathRayBuffer, 0, 6);
                bind(mHitBuffer, 0, 7);
                table->setAddress(sampleIdx, 8);
                bind(mHitQueueBuffer, 0, 9);
                bind(mControlBuffer, 0, 10);
                bind(scene.environment ? scene.environment->state().aliasBuffer : nullptr, 0, 11);
                bind(scene.vertexBuffer, 0, 12);
                bind(scene.prevVertexBuffer, 0, 13);
                bind(scene.indexBuffer, 0, 14);
                if (scene.environment && scene.environment->state().mapTexture)
                {
                    table->setTexture(scene.environment->state().mapTexture->gpuResourceID(), 0);
                }
                enc->dispatchThreadgroups(control + kShadeBaseArgsOffset, tg);
                barrier();
                endStage(connectStage);
            }
            auditDispatch("wavefrontShade");
            auto bindShadeResources = [&]() {
                bind(uniformBuffer, 0, 0);
                bind(scene.instanceBuffer, 0, 1);
                bind(scene.iesBuffer, 0, 2);
                bind(scene.lightBuffer, 0, 3);
                bind(scene.materialBuffer, 0, 4);
                bind(mPathStateBuffer, 0, 5);
                bind(mHitBuffer, 0, 6);
                bind(mRadianceBuffer, 0, 7);
                bind(mIorStackBuffer, 0, 8);
                bind(scene.geometryEntryBuffer, 0, 9);
                bind(scene.environment ? scene.environment->state().aliasBuffer : nullptr, 0, 10);
                bind(scene.vertexBuffer, 0, 11);
                bind(scene.prevVertexBuffer, 0, 12);
                bind(scene.indexBuffer, 0, 13);
                table->setAddress(sampleIdx, 14);
                bind(mHitQueueBuffer, 0, 15);
                bind(mPathQueueBuffer[dst], 0, 16);
                bind(mControlBuffer, dst * sizeof(uint32_t), 17);
                bind(mControlBuffer, 0, 18);
                bind(mShadowRayBuffer, 0, 19);
                bind(mControlBuffer, kShadowCounterOffset, 20);
                bind(mPathRayBuffer, 0, 21);
                bind(mAovBuffer, 0, 22);
                bind(sharcUpdate ? scene.sharcAccumulationBuffer :
                                   (scene.prevFrameVertexBuffer ? scene.prevFrameVertexBuffer : scene.vertexBuffer),
                     0, 23);
                bind(sharcUpdate ? scene.sharcResolvedBuffer :
                                   (scene.prevFrameInstanceBuffer ? scene.prevFrameInstanceBuffer : scene.instanceBuffer),
                     0, 24);
                table->setResource(scene.instanceAccelerationStructure->gpuResourceID(), 25);
                table->setResource(restirDiagnostic ? variant->restirShadeDiagnosticTable->gpuResourceID() :
                                                      variant->extendTableStatic->gpuResourceID(),
                                   26);
                bind(scene.curveSegmentBuffer, 0, 27);
                bind(mIorStatsBuffer, 0, 28);
                bind(sharcUpdate ? mSharcUpdateStateBuffer : scene.sharcResolvedBuffer, 0, 29);
                bind(mMediumPathStateBuffer, 0, 30);
                if (scene.environment && scene.environment->state().mapTexture)
                {
                    table->setTexture(scene.environment->state().mapTexture->gpuResourceID(), 0);
                }
            };
            bindShadeResources();
            auto dispatchShadeStage = [&](uint8_t kind, MTL::ComputePipelineState* pso, NS::UInteger argsOffset) {
                const uint32_t stage = beginStage(kind, bounce);
                if (diagnose)
                {
                    // The breadcrumb kernel shares and overwrites the argument table.
                    bindShadeResources();
                }
                enc->setComputePipelineState(pso);
                enc->dispatchThreadgroups(control + argsOffset, tg);
                endStage(stage);
            };
            if (variant->shadeBase)
            {
                dispatchShadeStage(kStageShadeBase, variant->shadeBase, kShadeBaseArgsOffset);
                dispatchShadeStage(kStageShadeLayer, variant->shadeLayer, kShadeLayerArgsOffset);
                dispatchShadeStage(kStageShadeTranslucent, variant->shadeTranslucent, kShadeTranslucentArgsOffset);
                dispatchShadeStage(kStageShadeTail, variant->shade, kShadeTailArgsOffset);
            }
            else
            {
                dispatchShadeStage(kStageShade, variant->shade, kHitArgsOffset);
            }
            barrier();

            if (uniforms->restirDIEnabled != 0u && bounce == 0u && !sharcUpdate)
            {
                const uint32_t stage = beginStage(kStageRestirSpatial, bounce);
                auditDispatch("wavefrontRestirSpatialFinal");
                enc->setComputePipelineState(variant->restirSpatialFinal);
                bind(uniformBuffer, 0, 0);
                bind(scene.instanceBuffer, 0, 1);
                bind(scene.iesBuffer, 0, 2);
                bind(scene.lightBuffer, 0, 3);
                bind(scene.materialBuffer, 0, 4);
                bind(scene.environment ? scene.environment->state().aliasBuffer : nullptr, 0, 5);
                bind(scene.vertexBuffer, 0, 6);
                bind(scene.prevVertexBuffer, 0, 7);
                bind(scene.indexBuffer, 0, 8);
                bind(mMissQueueBuffer, 0, 9);
                bind(mShadowRayBuffer, 0, 10);
                bind(mControlBuffer, kShadowCounterOffset, 11);
                bind(mRadianceBuffer, 0, 12);
                bind(mHitBuffer, 0, 13);
                bind(mControlBuffer, 0, 14);
                bind(scene.geometryEntryBuffer, 0, 15);
                bind(scene.curvePointBuffer, 0, 16);
                bind(scene.curveSegmentBuffer, 0, 17);
                if (restirDiagnostic)
                {
                    table->setResource(scene.instanceAccelerationStructure->gpuResourceID(), 18);
                    table->setResource(variant->restirSpatialDiagnosticTable->gpuResourceID(), 19);
                }
                else
                {
                    table->setResource(scene.instanceAccelerationStructure->gpuResourceID(), 18);
                    table->setResource(variant->extendTableStatic->gpuResourceID(), 19);
                }
                if (scene.environment && scene.environment->state().mapTexture)
                {
                    table->setTexture(scene.environment->state().mapTexture->gpuResourceID(), 0);
                }
                enc->dispatchThreadgroups(control + kRestirArgsOffset, tg);
                barrier();
                endStage(stage);
            }

            auditDispatch("wavefrontPrepareShadow");
            const uint32_t prepareShadowStage = beginStage(kStagePrepareShadow, bounce);
            enc->setComputePipelineState(mPrepareShadowPSO4);
            bind(mControlBuffer, 0, 0);
            table->setAddress(groupSize, 1);
            table->setAddress(bounceIdx, 2);
            bind(mTraversalDispatchBuffer, 0, 3);
            table->setAddress(ring.push(shadowBatchThreads), 4);
            table->setAddress(ring.push(shadowBatchCount), 5);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            barrier();
            endStage(prepareShadowStage);

            const uint32_t shadowStage = beginStage(kStageShadow, bounce);
            enc->setComputePipelineState(useMotion ? variant->shadowMotion : variant->shadowStatic);
            bind(uniformBuffer, 0, 0);
            table->setResource(scene.instanceAccelerationStructure->gpuResourceID(), 1);
            bind(mShadowRayBuffer, 0, 2);
            bind(mRadianceBuffer, 0, 3);
            bind(mControlBuffer, 0, 4);
            table->setAddress(sampleIdx, 5);
            bind(scene.instanceBuffer, 0, 6);
            bind(scene.materialBuffer, 0, 7);
            bind(scene.geometryEntryBuffer, 0, 8);
            bind(scene.vertexBuffer, 0, 9);
            bind(scene.indexBuffer, 0, 10);
            bind(scene.lightBuffer, 0, 11);
            table->setResource(
                (useMotion ? variant->shadowTableMotion : variant->shadowTableStatic)->gpuResourceID(), 15);
            table->setAddress(bounceIdx, 16);
            for (uint32_t batch = 0; batch < shadowBatchCount; ++batch)
            {
                auditDispatch(useMotion ? "wavefrontShadow" : "wavefrontShadowStatic");
                table->setAddress(ring.push(batch * shadowBatchThreads), 12);
                bind(mSharcUpdateStateBuffer, 0, 13);
                bind(scene.sharcAccumulationBuffer, 0, 14);
                enc->dispatchThreadgroups(
                    traversalDispatches + static_cast<MTL::GPUAddress>(batch) * 3u * sizeof(uint32_t), tg);
                if (batch + 1 < shadowBatchCount)
                {
                    traversalBatchBarrier();
                }
            }
            barrier();
            endStage(shadowStage);
        }
    }

    if (!chunk.resolve)
    {
        return;
    }

    if (uniforms->writeAov && variant->guideMotion && variant->guideStatic)
    {
        const uint32_t stage = beginStage(kStageGuide);
        enc->setComputePipelineState(useMotion ? variant->guideMotion : variant->guideStatic);
        bind(uniformBuffer, 0, 0);
        bind(scene.instanceBuffer, 0, 1);
        table->setResource(scene.instanceAccelerationStructure->gpuResourceID(), 2);
        bind(mAovBuffer, 0, 4);
        bind(scene.materialBuffer, 0, 5);
        bind(scene.geometryEntryBuffer, 0, 6);
        bind(scene.vertexBuffer, 0, 7);
        bind(scene.prevVertexBuffer, 0, 8);
        bind(scene.indexBuffer, 0, 9);
        bind(scene.curvePointBuffer, 0, 10);
        bind(scene.curveSegmentBuffer, 0, 11);
        bind(scene.lightBuffer, 0, 12);
        bind(mControlBuffer, 0, 14);
        table->setResource((useMotion ? variant->guideTableMotion : variant->guideTableStatic)->gpuResourceID(), 13);
        enc->dispatchThreadgroups(control + kGuideArgsOffset, tg);
        barrier();
        endStage(stage);
    }

    const uint32_t resolveStage = beginStage(kStageResolve);
    auditDispatch("wavefrontResolve");
    enc->setComputePipelineState(mResolvePSO4);
    bind(uniformBuffer, 0, 0);
    bind(mRadianceBuffer, 0, 1);
    bind(outputBuffer, 0, 2);
    bind(scene.accumulationBuffer, 0, 3);
    table->setAddress(ring.push(sampleCount), 4);
    bind(mAovBuffer, 0, 5);
    bind(scene.sharcHashBuffer, 0, 6);
    bind(scene.sharcResolvedBuffer, 0, 7);
    enc->dispatchThreadgroups(fullGrid, tg);

    // Guide resolve. The guides exist only when something downstream reads them,
    // which is either MetalFX mode, so their presence is the condition -- the
    // encoder has no other way to know which one the frame selected.
    if (scene.guideColor && mAovResolvePSO4)
    {
        auditDispatch("wavefrontResolveAov");
        barrier();
        enc->setComputePipelineState(mAovResolvePSO4);
        bind(uniformBuffer, 0, 0);
        bind(mAovBuffer, 0, 1);
        bind(mRadianceBuffer, 0, 2);
        table->setAddress(ring.push(sampleCount), 3);
        table->setTexture(scene.guideColor->gpuResourceID(), 0);
        table->setTexture(scene.guideDepth->gpuResourceID(), 1);
        table->setTexture(scene.guideMotion->gpuResourceID(), 2);
        table->setTexture(scene.guideDiffuse->gpuResourceID(), 3);
        table->setTexture(scene.guideSpecular->gpuResourceID(), 4);
        table->setTexture(scene.guideNormal->gpuResourceID(), 5);
        table->setTexture(scene.guideRoughness->gpuResourceID(), 6);
        table->setTexture(scene.guideSpecularHitDistance->gpuResourceID(), 7);
        table->setTexture(scene.guideReactive->gpuResourceID(), 8);
        table->setTexture(scene.guideDenoiseStrength->gpuResourceID(), 9);
        enc->dispatchThreadgroups(MTL::Size((width + 7) / 8, (height + 7) / 8, 1), MTL::Size(8, 8, 1));
    }
    endStage(resolveStage);

    if (diagnose)
    {
        // stageCount is the post-integrator sentinel understood by the CPU.
        barrier();
        writeBreadcrumb(static_cast<uint32_t>(mStageKinds.size()));
    }
}

void MetalWavefrontIntegrator::encodeSharcClear(MTL::ComputeCommandEncoder* enc,
                                                const IntegratorSceneBindings& scene,
                                                const IntegratorFrameRequest& frame,
                                                bool clearPersistent)
{
    if (!enc || !mSharcClearPSO || !scene.sharcHashBuffer || !scene.sharcAccumulationBuffer ||
        !scene.sharcResolvedBuffer || !scene.sharcStatsBuffer)
    {
        return;
    }
    const auto* uniforms = static_cast<const Uniforms*>(frame.uniformBuffer->contents());
    const uint32_t clear = clearPersistent ? 1u : 0u;
    enc->setComputePipelineState(mSharcClearPSO);
    enc->setBuffer(frame.uniformBuffer, 0, 0);
    enc->setBuffer(scene.sharcHashBuffer, 0, 1);
    enc->setBuffer(scene.sharcAccumulationBuffer, 0, 2);
    enc->setBuffer(scene.sharcResolvedBuffer, 0, 3);
    enc->setBuffer(scene.sharcStatsBuffer, IOR_STAT_COUNT * sizeof(uint32_t), 4);
    enc->setBytes(&clear, sizeof(clear), 5);
    enc->dispatchThreads(MTL::Size(uniforms->sharcCapacity, 1, 1), MTL::Size(256, 1, 1));
    enc->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalWavefrontIntegrator::encodeSharcResolve(MTL::ComputeCommandEncoder* enc,
                                                  const IntegratorSceneBindings& scene,
                                                  const IntegratorFrameRequest& frame)
{
    if (!enc || !mSharcResolvePSO || !scene.sharcHashBuffer || !scene.sharcAccumulationBuffer ||
        !scene.sharcResolvedBuffer || !scene.sharcStatsBuffer)
    {
        return;
    }
    const auto* uniforms = static_cast<const Uniforms*>(frame.uniformBuffer->contents());
    enc->setComputePipelineState(mSharcResolvePSO);
    enc->setBuffer(frame.uniformBuffer, 0, 0);
    enc->setBuffer(scene.sharcHashBuffer, 0, 1);
    enc->setBuffer(scene.sharcAccumulationBuffer, 0, 2);
    enc->setBuffer(scene.sharcResolvedBuffer, 0, 3);
    enc->setBuffer(scene.sharcStatsBuffer, IOR_STAT_COUNT * sizeof(uint32_t), 4);
    enc->dispatchThreads(MTL::Size(uniforms->sharcCapacity, 1, 1), MTL::Size(256, 1, 1));
    enc->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalWavefrontIntegrator::encodeSharcClearMetal4(MTL4::ComputeCommandEncoder* enc,
                                                      const IntegratorSceneBindings& scene,
                                                      const IntegratorFrameRequest& frame,
                                                      bool clearPersistent)
{
    if (!enc || !mSharcClearPSO4 || !scene.sharcHashBuffer || !scene.sharcAccumulationBuffer ||
        !scene.sharcResolvedBuffer || !scene.sharcStatsBuffer)
    {
        return;
    }
    const auto* uniforms = static_cast<const Uniforms*>(frame.uniformBuffer->contents());
    MTL4::ArgumentTable* table = mMetal4->argumentTable();
    table->setAddress(frame.uniformBuffer->gpuAddress(), 0);
    table->setAddress(scene.sharcHashBuffer->gpuAddress(), 1);
    table->setAddress(scene.sharcAccumulationBuffer->gpuAddress(), 2);
    table->setAddress(scene.sharcResolvedBuffer->gpuAddress(), 3);
    table->setAddress(scene.sharcStatsBuffer->gpuAddress() + IOR_STAT_COUNT * sizeof(uint32_t), 4);
    table->setAddress(mMetal4->constants().push(clearPersistent ? 1u : 0u), 5);
    enc->setArgumentTable(table);
    enc->setComputePipelineState(mSharcClearPSO4);
    const uint32_t groups = (uniforms->sharcCapacity + 255u) / 256u;
    enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(256, 1, 1));
    enc->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}

void MetalWavefrontIntegrator::encodeSharcResolveMetal4(MTL4::ComputeCommandEncoder* enc,
                                                        const IntegratorSceneBindings& scene,
                                                        const IntegratorFrameRequest& frame)
{
    if (!enc || !mSharcResolvePSO4 || !scene.sharcHashBuffer || !scene.sharcAccumulationBuffer ||
        !scene.sharcResolvedBuffer || !scene.sharcStatsBuffer)
    {
        return;
    }
    const auto* uniforms = static_cast<const Uniforms*>(frame.uniformBuffer->contents());
    MTL4::ArgumentTable* table = mMetal4->argumentTable();
    table->setAddress(frame.uniformBuffer->gpuAddress(), 0);
    table->setAddress(scene.sharcHashBuffer->gpuAddress(), 1);
    table->setAddress(scene.sharcAccumulationBuffer->gpuAddress(), 2);
    table->setAddress(scene.sharcResolvedBuffer->gpuAddress(), 3);
    table->setAddress(scene.sharcStatsBuffer->gpuAddress() + IOR_STAT_COUNT * sizeof(uint32_t), 4);
    enc->setArgumentTable(table);
    enc->setComputePipelineState(mSharcResolvePSO4);
    const uint32_t groups = (uniforms->sharcCapacity + 255u) / 256u;
    enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(256, 1, 1));
    enc->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}

MTL::ComputeCommandEncoder* MetalWavefrontIntegrator::encode(MTL::CommandBuffer* pCmd,
                                                             MTL::ComputeCommandEncoder* enc,
                                                             const IntegratorSceneBindings& scene,
                                                             const IntegratorFrameRequest& frame)
{
    const uint32_t features = frame.features;
    const uint32_t width = frame.width;
    const uint32_t height = frame.height;
    const uint32_t sampleCount = frame.sampleCount;
    MTL::Buffer* uniformBuffer = frame.uniformBuffer;
    Buffer* output = frame.output;

    const uint32_t pixels = frame.pathCount != 0u ? frame.pathCount : width * height;
    const WavefrontVariant* variant = variantFor(features);
    const uint32_t bounceIterations = frame.bounceIterations;
    const MTL::Buffer* outputBuffer = ((MetalBuffer*)output)->getNativePtr();
    const auto* uniforms = static_cast<const Uniforms*>(uniformBuffer->contents());
    const bool sharcUpdatePass = (features & WavefrontFeatures::kSharcUpdate) != 0u;
    const bool restirDiagnostic = (features & WavefrontFeatures::kRestirRayTracedDiagnostic) != 0u;
    // Textures are reached through resource IDs inside the Material struct, so
    // Metal cannot infer their use from the bindings and every encoder has to be
    // told about them again.
    auto declareResidency = [&](MTL::ComputeCommandEncoder* e) {
        if (scene.textures && !scene.textures->materialTextures().empty())
        {
            // Metal's batched residency API accepts Resource pointers, while
            // metal-cpp exposes the same Objective-C objects as Texture pointers.
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            e->useResources(reinterpret_cast<const MTL::Resource* const*>(scene.textures->materialTextures().data()),
                            scene.textures->materialTextures().size(), MTL::ResourceUsageRead);
        }
        if (scene.primitiveAccelerationStructures && !scene.primitiveAccelerationStructures->empty())
        {
            // See the Texture batch above; these are also Resource objects in Metal.
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            e->useResources(reinterpret_cast<const MTL::Resource* const*>(scene.primitiveAccelerationStructures->data()),
                            scene.primitiveAccelerationStructures->size(), MTL::ResourceUsageRead);
        }
        if (scene.instanceAccelerationStructure)
        {
            e->useResource(scene.instanceAccelerationStructure, MTL::ResourceUsageRead);
        }
        if (scene.volumeAccelerationStructure && scene.volumeAccelerationStructure != scene.instanceAccelerationStructure)
        {
            e->useResource(scene.volumeAccelerationStructure, MTL::ResourceUsageRead);
        }
        if (scene.environment && scene.environment->state().mapTexture)
        {
            e->useResource(scene.environment->state().mapTexture, MTL::ResourceUsageRead);
            if (scene.environment->state().backgroundTexture)
            {
                e->useResource(scene.environment->state().backgroundTexture, MTL::ResourceUsageRead);
            }
        }
        e->useResource(((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageWrite);
        // wavefrontShade reaches this through Uniforms::guideRays because all
        // explicit buffer slots are occupied.
        e->useResource(mGuideRayBuffer, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        e->useResource(mSurfaceGeometryBuffer, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        e->useResource(mRestirReservoirBuffer[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        e->useResource(mRestirReservoirBuffer[1], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        e->useResource(mRestirSurfaceHistoryBuffer[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        e->useResource(mRestirSurfaceHistoryBuffer[1], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        e->useResource(mRestirSurfaceDataBuffer[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        if (mRestirSurfaceDataBuffer[1])
            e->useResource(mRestirSurfaceDataBuffer[1], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        if (scene.previousLightBuffer)
            e->useResource(scene.previousLightBuffer, MTL::ResourceUsageRead);
        if (scene.lightTemporalMappingBuffer)
            e->useResource(scene.lightTemporalMappingBuffer, MTL::ResourceUsageRead);
    };
    declareResidency(enc);

    const MTL::Size grid = MTL::Size(pixels, 1, 1);
    const uint32_t kThreadsPerGroup = 64;
    const MTL::Size tg = MTL::Size(kThreadsPerGroup, 1, 1);
    const uint32_t generateScale = sharcUpdatePass && uniforms ? std::max(uniforms->sharcUpdateDownscale, 1u) : 1u;
    const MTL::Size generateGrid =
        MTL::Size((width + generateScale - 1u) / generateScale, (height + generateScale - 1u) / generateScale, 1);
    const MTL::Size generateTg = MTL::Size(32, 8, 1);
    // Byte offset of the indirect dispatch arguments inside the control buffer.
    const NS::UInteger kDispatchArgsOffset = 2 * sizeof(uint32_t);
    const NS::UInteger kShadowArgsOffset = 8 * sizeof(uint32_t);
    const NS::UInteger kShadowCounterOffset = 6 * sizeof(uint32_t);
    const NS::UInteger kHitArgsOffset = 13 * sizeof(uint32_t);
    const NS::UInteger kHitCounterOffset = 11 * sizeof(uint32_t);
    const NS::UInteger kShadeBaseArgsOffset = 80 * sizeof(uint32_t);
    const NS::UInteger kShadeLayerArgsOffset = 84 * sizeof(uint32_t);
    const NS::UInteger kShadeTranslucentArgsOffset = 88 * sizeof(uint32_t);
    const NS::UInteger kShadeTailArgsOffset = 92 * sizeof(uint32_t);
    const NS::UInteger kMissArgsOffset = 18 * sizeof(uint32_t);
    const NS::UInteger kMissCounterOffset = 16 * sizeof(uint32_t);
    const NS::UInteger kGuideArgsOffset = 25 * sizeof(uint32_t);
    const NS::UInteger kRestirArgsOffset = 29 * sizeof(uint32_t);
    const uint32_t traversalBatchThreads = kWavefrontTraversalBatchThreads;
    const uint32_t traversalBatchCount = wavefrontTraversalBatchCount(pixels);
    const uint32_t traversalQueueOffset = 0;
    // Nothing in the scene deforms -> traverse it as a static structure. Every ray
    // was otherwise paying for motion-BVH traversal it could not use.
    const bool useMotion = frame.motionBlasBuilt || !variant || variant->extendStatic == nullptr ||
                           frame.settings->getAs<uint32_t>("render/pt/staticTraversal") == 0;
    if (!variant)
    {
        return enc;
    }
    // See the Metal 4 path above: depth 1 cannot produce an SSS continuation,
    // even when alpha pass-through headroom keeps the host iteration loop alive.
    const bool fusedSss = uniforms && uniforms->maxDepth > 1u && (features & WavefrontFeatures::kSubsurface) != 0u &&
                          !sharcUpdatePass && variant->sssWalkMotion && variant->sssWalkStatic;

    // Profiling gives each stage its own encoder, because this hardware samples
    // counters only at encoder boundaries. That costs encoder overhead, so it is
    // a measurement mode and not something to leave on.
    mStageKinds.clear();
    mStageBounces.clear();
    const bool profile = frame.profileStages && mStageTimestampBuffer != nullptr;
    const uint32_t diagnosticsEnabled = profile ? 1u : 0u;
    const uint32_t diagnosticsMediumEnabled = (features & WavefrontFeatures::kSubsurface) != 0u ? 1u : 0u;
    auto stamp = [&](uint8_t kind) {
        if (!profile || mStageKinds.size() >= kMaxStageSamples)
        {
            return;
        }
        enc->endEncoding();
        const MTL::ComputePassDescriptor* desc = MTL::ComputePassDescriptor::computePassDescriptor();
        MTL::ComputePassSampleBufferAttachmentDescriptor* att = desc->sampleBufferAttachments()->object(0);
        att->setSampleBuffer(mStageTimestampBuffer);
        att->setStartOfEncoderSampleIndex(2 * mStageKinds.size());
        att->setEndOfEncoderSampleIndex(2 * mStageKinds.size() + 1);
        enc = pCmd->computeCommandEncoder(desc);
        // Named so that a capture can attribute to a stage. Instruments reports
        // register spills against an encoder id and nothing else, and an
        // unlabelled trace says only that *something* spilled.
        enc->setLabel(NS::String::string(kStageNames[kind], NS::UTF8StringEncoding));
        declareResidency(enc);
        mStageKinds.push_back(kind);
    };

    for (uint32_t s = 0; s < sampleCount; ++s)
    {
        stamp(kStageGenerate);
        enc->setComputePipelineState(variant->generate);
        enc->setBuffer(uniformBuffer, 0, 0);
        enc->setBuffer(mPathStateBuffer, 0, 1);
        enc->setBuffer(mRadianceBuffer, 0, 2);
        enc->setBuffer(mIorStackBuffer, 0, 3);
        enc->setBytes(&s, sizeof(uint32_t), 4);
        enc->setBuffer(mPathQueueBuffer[0], 0, 5);
        enc->setBuffer(mControlBuffer, 0, 6);
        enc->setBuffer(mAovBuffer, 0, 7);
        enc->setBuffer(mPathRayBuffer, 0, 8);
        enc->setBuffer(mIorStatsBuffer, 0, 9);
        enc->setBuffer(mSharcUpdateStateBuffer, 0, 10);
        enc->setBuffer(mMediumPathStateBuffer, 0, 11);
        enc->dispatchThreads(generateGrid, generateTg);

        for (uint32_t bounce = 0; bounce < bounceIterations; ++bounce)
        {
            const uint32_t src = bounce & 1u;
            const uint32_t dst = src ^ 1u;

            // Publish this bounce's live count and clear the destination's, then
            // dispatch both stages indirectly from it. Nothing crosses to the CPU.
            stamp(kStagePrepare);
            enc->setComputePipelineState(mPreparePSO);
            enc->setBuffer(mControlBuffer, 0, 0);
            enc->setBytes(&src, sizeof(uint32_t), 1);
            enc->setBytes(&kThreadsPerGroup, sizeof(uint32_t), 2);
            enc->setBytes(&bounce, sizeof(uint32_t), 3);
            enc->setBuffer(mStageStatsBuffer, 0, 4);
            enc->setBuffer(mPathStateBuffer, 0, 5);
            enc->setBuffer(mPathRayBuffer, 0, 6);
            enc->setBuffer(mPathQueueBuffer[src], 0, 7);
            enc->setBytes(&diagnosticsEnabled, sizeof(diagnosticsEnabled), 8);
            enc->setBuffer(mTraversalDispatchBuffer, 0, 9);
            enc->setBytes(&traversalBatchThreads, sizeof(traversalBatchThreads), 10);
            enc->setBytes(&traversalBatchCount, sizeof(traversalBatchCount), 11);
            enc->setBuffer(mMediumPathStateBuffer, 0, 12);
            enc->setBytes(&diagnosticsMediumEnabled, sizeof(diagnosticsMediumEnabled), 13);
            enc->setBuffer(mSssControlBuffer, 0, 14);
            enc->setBuffer(mHitQueueBuffer, 0, 15);
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

            // Sort the queue this bounce is about to traverse. Bounce 0 is the
            // camera and already perfectly coherent, so it is skipped -- sorting
            // it is pure cost.

            stamp(kStageExtend);
            if (fusedSss)
            {
                enc->setComputePipelineState(mClassifySssPSO);
                enc->setBuffer(uniformBuffer, 0, 0);
                enc->setBuffer(mPathQueueBuffer[src], 0, 1);
                enc->setBuffer(mControlBuffer, 0, 2);
                enc->setBuffer(mMediumPathStateBuffer, 0, 3);
                enc->setBuffer(scene.materialBuffer, 0, 4);
                enc->setBuffer(mSssQueueBuffer, 0, 5);
                enc->setBuffer(mSssControlBuffer, 0, 6);
                enc->setBytes(&traversalQueueOffset, sizeof(traversalQueueOffset), 7);
                enc->dispatchThreadgroups(mControlBuffer, kDispatchArgsOffset, tg);
                enc->memoryBarrier(MTL::BarrierScopeBuffers);
            }
            enc->pushDebugGroup(NS::String::string("extend", NS::UTF8StringEncoding));
            enc->setComputePipelineState(useMotion ? variant->extendMotion : variant->extendStatic);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(scene.instanceBuffer, 0, 1);
            enc->setAccelerationStructure(scene.instanceAccelerationStructure, 2);
            enc->setBuffer(mPathRayBuffer, 0, 3);
            enc->setBuffer(mHitBuffer, 0, 4);
            enc->setBytes(&s, sizeof(uint32_t), 5);
            enc->setBuffer(mPathQueueBuffer[src], 0, 6);
            enc->setBuffer(mControlBuffer, 0, 7);
            enc->setBuffer(mHitQueueBuffer, 0, 8);
            enc->setBuffer(mControlBuffer, kHitCounterOffset, 9);
            enc->setBuffer(mMissQueueBuffer, 0, 10);
            enc->setBuffer(mControlBuffer, kMissCounterOffset, 11);
            enc->setBuffer(mPathStateBuffer, 0, 12);
            enc->setBuffer(scene.materialBuffer, 0, 13);
            // Only the camera bounce is denied the hidden lights.
            const uint32_t extendMask =
                (bounce == 0) ? uniforms->primaryRayMask : (uniforms->primaryRayMask | GEOMETRY_MASK_LIGHT_HIDDEN);
            enc->setBytes(&extendMask, sizeof(uint32_t), 14);
            enc->setAccelerationStructure(scene.volumeAccelerationStructure, 15);
            enc->setBytes(&traversalQueueOffset, sizeof(traversalQueueOffset), 16);
            enc->setBuffer(mMediumPathStateBuffer, 0, 17);
            enc->setBuffer(scene.geometryEntryBuffer, 0, 18);
            MTL::IntersectionFunctionTable* extendTable =
                useMotion ? variant->extendTableMotion : variant->extendTableStatic;
            enc->setIntersectionFunctionTable(extendTable, 19);
            enc->useResource(extendTable, MTL::ResourceUsageRead);
            enc->setBuffer(scene.vertexBuffer, 0, 20);
            enc->setBuffer(scene.prevVertexBuffer, 0, 21);
            enc->setBuffer(scene.indexBuffer, 0, 22);
            enc->dispatchThreadgroups(mControlBuffer, kDispatchArgsOffset, tg);
            enc->popDebugGroup();

            if (fusedSss)
            {
                enc->memoryBarrier(MTL::BarrierScopeBuffers);
                enc->setComputePipelineState(mPrepareSssPSO);
                enc->setBuffer(mSssControlBuffer, 0, 0);
                enc->setBytes(&kThreadsPerGroup, sizeof(kThreadsPerGroup), 1);
                enc->setBuffer(mStageStatsBuffer, 0, 2);
                enc->setBytes(&bounce, sizeof(bounce), 3);
                enc->setBytes(&diagnosticsEnabled, sizeof(diagnosticsEnabled), 4);
                enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
                enc->memoryBarrier(MTL::BarrierScopeBuffers);

                enc->setComputePipelineState(useMotion ? variant->sssWalkMotion : variant->sssWalkStatic);
                enc->setBuffer(uniformBuffer, 0, 0);
                enc->setBuffer(scene.instanceBuffer, 0, 1);
                enc->setAccelerationStructure(scene.volumeAccelerationStructure, 2);
                enc->setBuffer(mPathRayBuffer, 0, 3);
                enc->setBuffer(mPathStateBuffer, 0, 4);
                enc->setBuffer(mMediumPathStateBuffer, 0, 5);
                enc->setBuffer(scene.materialBuffer, 0, 6);
                enc->setBytes(&s, sizeof(uint32_t), 7);
                enc->setBuffer(mSssQueueBuffer, 0, 8);
                enc->setBuffer(mSssControlBuffer, 0, 9);
                enc->setBuffer(mHitBuffer, 0, 10);
                enc->setBuffer(mHitQueueBuffer, 0, 11);
                enc->setBuffer(mControlBuffer, kHitCounterOffset, 12);
                enc->setBuffer(mMissQueueBuffer, 0, 13);
                enc->setBuffer(mControlBuffer, kMissCounterOffset, 14);
                enc->setBuffer(mPathQueueBuffer[dst], 0, 15);
                enc->setBuffer(mControlBuffer, dst * sizeof(uint32_t), 16);
                enc->setBuffer(mControlBuffer, 0, 17);
                enc->setBytes(&extendMask, sizeof(uint32_t), 18);
                enc->setBuffer(scene.geometryEntryBuffer, 0, 19);
                enc->dispatchThreadgroups(mSssControlBuffer, 2u * sizeof(uint32_t), tg);
                enc->memoryBarrier(MTL::BarrierScopeBuffers);
            }

            enc->setComputePipelineState(mPrepareHitMissPSO);
            enc->setBuffer(mControlBuffer, 0, 0);
            enc->setBytes(&kThreadsPerGroup, sizeof(uint32_t), 1);
            enc->setBuffer(mHitQueueBuffer, 0, 2);
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

            stamp(kStageMiss);
            enc->setComputePipelineState(variant->miss);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(mPathStateBuffer, 0, 1);
            enc->setBuffer(mPathRayBuffer, 0, 2);
            enc->setBuffer(mRadianceBuffer, 0, 3);
            enc->setBuffer(mMissQueueBuffer, 0, 4);
            enc->setBuffer(mControlBuffer, 0, 5);
            enc->setBuffer(mAovBuffer, 0, 6);
            enc->setBytes(&s, sizeof(uint32_t), 7);
            enc->setBuffer(mIorStackBuffer, 0, 8);
            enc->setBuffer(mIorStatsBuffer, 0, 9);
            enc->setBuffer(mSharcUpdateStateBuffer, 0, 10);
            enc->setBuffer(scene.sharcAccumulationBuffer, 0, 11);
            enc->setBuffer(scene.lightBuffer, 0, 12);
            enc->setBuffer(scene.environment ? scene.environment->state().aliasBuffer : nullptr, 0, 13);
            if (scene.environment && scene.environment->state().mapTexture)
            {
                enc->setTexture(scene.environment->state().mapTexture, 0);
                enc->setTexture(scene.environment->state().backgroundTexture ?
                                    scene.environment->state().backgroundTexture :
                                    scene.environment->state().mapTexture,
                                1);
            }
            enc->dispatchThreadgroups(mControlBuffer, kMissArgsOffset, tg);
            const bool sharcUpdate = (features & WavefrontFeatures::kSharcUpdate) != 0u;
            if (uniforms->restirDIEnabled != 0u && bounce == 0u && !sharcUpdate)
            {
                enc->memoryBarrier(MTL::BarrierScopeBuffers);
            }

            stamp(kStageShade);
            if (variant->connectBase)
            {
                enc->pushDebugGroup(NS::String::string("connect base", NS::UTF8StringEncoding));
                enc->setComputePipelineState(variant->connectBase);
                enc->setBuffer(uniformBuffer, 0, 0);
                enc->setBuffer(scene.instanceBuffer, 0, 1);
                enc->setBuffer(scene.iesBuffer, 0, 2);
                enc->setBuffer(scene.lightBuffer, 0, 3);
                enc->setBuffer(scene.materialBuffer, 0, 4);
                enc->setBuffer(mPathStateBuffer, 0, 5);
                enc->setBuffer(mPathRayBuffer, 0, 6);
                enc->setBuffer(mHitBuffer, 0, 7);
                enc->setBytes(&s, sizeof(uint32_t), 8);
                enc->setBuffer(mHitQueueBuffer, 0, 9);
                enc->setBuffer(mControlBuffer, 0, 10);
                enc->setBuffer(scene.environment ? scene.environment->state().aliasBuffer : nullptr, 0, 11);
                enc->setBuffer(scene.vertexBuffer, 0, 12);
                enc->setBuffer(scene.prevVertexBuffer, 0, 13);
                enc->setBuffer(scene.indexBuffer, 0, 14);
                if (scene.environment && scene.environment->state().mapTexture)
                {
                    enc->setTexture(scene.environment->state().mapTexture, 0);
                }
                enc->dispatchThreadgroups(mControlBuffer, kShadeBaseArgsOffset, tg);
                enc->memoryBarrier(MTL::BarrierScopeBuffers);
                enc->popDebugGroup();
            }
            enc->pushDebugGroup(NS::String::string("shade", NS::UTF8StringEncoding));
            enc->setComputePipelineState(variant->shade);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(scene.instanceBuffer, 0, 1);
            enc->setBuffer(scene.iesBuffer, 0, 2);
            enc->setBuffer(scene.lightBuffer, 0, 3);
            enc->setBuffer(scene.materialBuffer, 0, 4);
            enc->setBuffer(mPathStateBuffer, 0, 5);
            enc->setBuffer(mHitBuffer, 0, 6);
            enc->setBuffer(mRadianceBuffer, 0, 7);
            enc->setBuffer(mIorStackBuffer, 0, 8);
            enc->setBuffer(scene.geometryEntryBuffer, 0, 9);
            enc->setBuffer(scene.environment ? scene.environment->state().aliasBuffer : nullptr, 0, 10);
            enc->setBuffer(scene.vertexBuffer, 0, 11);
            enc->setBuffer(scene.prevVertexBuffer, 0, 12);
            enc->setBuffer(scene.indexBuffer, 0, 13);
            enc->setBytes(&s, sizeof(uint32_t), 14);
            enc->setBuffer(mHitQueueBuffer, 0, 15);
            enc->setBuffer(mPathQueueBuffer[dst], 0, 16);
            enc->setBuffer(mControlBuffer, dst * sizeof(uint32_t), 17);
            enc->setBuffer(mControlBuffer, 0, 18);
            enc->setBuffer(mShadowRayBuffer, 0, 19);
            enc->setBuffer(mControlBuffer, kShadowCounterOffset, 20);
            enc->setBuffer(mPathRayBuffer, 0, 21);
            enc->setBuffer(mAovBuffer, 0, 22);
            // With nothing deforming, the current vertex buffer already is the
            // previous pose, so it is bound directly rather than copied.
            enc->setBuffer(sharcUpdate ? scene.sharcAccumulationBuffer :
                                         (scene.prevFrameVertexBuffer ? scene.prevFrameVertexBuffer : scene.vertexBuffer),
                           0, 23);
            enc->setBuffer(sharcUpdate ?
                               scene.sharcResolvedBuffer :
                               (scene.prevFrameInstanceBuffer ? scene.prevFrameInstanceBuffer : scene.instanceBuffer),
                           0, 24);
            if (scene.environment && scene.environment->state().mapTexture)
            {
                enc->setTexture(scene.environment->state().mapTexture, 0);
            }
            if (restirDiagnostic)
            {
                enc->setAccelerationStructure(scene.instanceAccelerationStructure, 25);
                enc->setIntersectionFunctionTable(variant->restirShadeDiagnosticTable, 26);
                enc->useResource(variant->restirShadeDiagnosticTable, MTL::ResourceUsageRead);
            }
            enc->setBuffer(scene.curveSegmentBuffer ? scene.curveSegmentBuffer : scene.placeholderBuffer, 0, 27);
            enc->setBuffer(mIorStatsBuffer, 0, 28);
            enc->setBuffer(sharcUpdate ? mSharcUpdateStateBuffer : scene.sharcResolvedBuffer, 0, 29);
            enc->setBuffer(mMediumPathStateBuffer, 0, 30);
            if (variant->shadeBase)
            {
                enc->setComputePipelineState(variant->shadeBase);
                enc->dispatchThreadgroups(mControlBuffer, kShadeBaseArgsOffset, tg);
                enc->setComputePipelineState(variant->shadeLayer);
                enc->dispatchThreadgroups(mControlBuffer, kShadeLayerArgsOffset, tg);
                enc->setComputePipelineState(variant->shadeTranslucent);
                enc->dispatchThreadgroups(mControlBuffer, kShadeTranslucentArgsOffset, tg);
                enc->setComputePipelineState(variant->shade);
                enc->dispatchThreadgroups(mControlBuffer, kShadeTailArgsOffset, tg);
            }
            else
            {
                enc->dispatchThreadgroups(mControlBuffer, kHitArgsOffset, tg);
            }
            enc->popDebugGroup();

            if (uniforms->restirDIEnabled != 0u && bounce == 0u && !sharcUpdate)
            {
                enc->memoryBarrier(MTL::BarrierScopeBuffers);
                stamp(kStageRestirSpatial);
                enc->setComputePipelineState(variant->restirSpatialFinal);
                enc->setBuffer(uniformBuffer, 0, 0);
                enc->setBuffer(scene.instanceBuffer, 0, 1);
                enc->setBuffer(scene.iesBuffer, 0, 2);
                enc->setBuffer(scene.lightBuffer, 0, 3);
                enc->setBuffer(scene.materialBuffer, 0, 4);
                enc->setBuffer(scene.environment ? scene.environment->state().aliasBuffer : nullptr, 0, 5);
                enc->setBuffer(scene.vertexBuffer, 0, 6);
                enc->setBuffer(scene.prevVertexBuffer, 0, 7);
                enc->setBuffer(scene.indexBuffer, 0, 8);
                enc->setBuffer(mMissQueueBuffer, 0, 9);
                enc->setBuffer(mShadowRayBuffer, 0, 10);
                enc->setBuffer(mControlBuffer, kShadowCounterOffset, 11);
                enc->setBuffer(mRadianceBuffer, 0, 12);
                enc->setBuffer(mHitBuffer, 0, 13);
                enc->setBuffer(mControlBuffer, 0, 14);
                enc->setBuffer(scene.geometryEntryBuffer, 0, 15);
                enc->setBuffer(scene.curvePointBuffer ? scene.curvePointBuffer : scene.placeholderBuffer, 0, 16);
                enc->setBuffer(scene.curveSegmentBuffer ? scene.curveSegmentBuffer : scene.placeholderBuffer, 0, 17);
                if (restirDiagnostic)
                {
                    enc->setAccelerationStructure(scene.instanceAccelerationStructure, 18);
                    enc->setIntersectionFunctionTable(variant->restirSpatialDiagnosticTable, 19);
                    enc->useResource(variant->restirSpatialDiagnosticTable, MTL::ResourceUsageRead);
                }
                if (scene.environment && scene.environment->state().mapTexture)
                {
                    enc->setTexture(scene.environment->state().mapTexture, 0);
                }
                enc->dispatchThreadgroups(mControlBuffer, kRestirArgsOffset, tg);
                enc->memoryBarrier(MTL::BarrierScopeBuffers);
            }

            // Resolve this bounce's direct light before the next bounce contributes emission.
            stamp(kStagePrepareShadow);
            enc->setComputePipelineState(mPrepareShadowPSO);
            enc->setBuffer(mControlBuffer, 0, 0);
            enc->setBytes(&kThreadsPerGroup, sizeof(uint32_t), 1);
            enc->setBytes(&bounce, sizeof(uint32_t), 2);
            enc->setBuffer(mTraversalDispatchBuffer, 0, 3);
            enc->setBytes(&traversalBatchThreads, sizeof(traversalBatchThreads), 4);
            enc->setBytes(&traversalBatchCount, sizeof(traversalBatchCount), 5);
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

            stamp(kStageShadow);
            enc->setComputePipelineState(useMotion ? variant->shadowMotion : variant->shadowStatic);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setAccelerationStructure(scene.instanceAccelerationStructure, 1);
            enc->setBuffer(mShadowRayBuffer, 0, 2);
            enc->setBuffer(mRadianceBuffer, 0, 3);
            enc->setBuffer(mControlBuffer, 0, 4);
            enc->setBytes(&s, sizeof(uint32_t), 5);
            // Cutout shadows need to resolve the material and its uv at each hit.
            enc->setBuffer(scene.instanceBuffer, 0, 6);
            enc->setBuffer(scene.materialBuffer, 0, 7);
            enc->setBuffer(scene.geometryEntryBuffer, 0, 8);
            enc->setBuffer(scene.vertexBuffer, 0, 9);
            enc->setBuffer(scene.indexBuffer, 0, 10);
            enc->setBuffer(scene.lightBuffer, 0, 11);
            enc->setBytes(&traversalQueueOffset, sizeof(traversalQueueOffset), 12);
            enc->setBuffer(mSharcUpdateStateBuffer, 0, 13);
            enc->setBuffer(scene.sharcAccumulationBuffer, 0, 14);
            MTL::IntersectionFunctionTable* shadowTable =
                useMotion ? variant->shadowTableMotion : variant->shadowTableStatic;
            enc->setIntersectionFunctionTable(shadowTable, 15);
            enc->setBytes(&bounce, sizeof(uint32_t), 16);
            enc->useResource(shadowTable, MTL::ResourceUsageRead);
            enc->dispatchThreadgroups(mControlBuffer, kShadowArgsOffset, tg);
        }
    }

    if (sharcUpdatePass)
    {
        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        return enc;
    }

    if (uniforms->writeAov && variant->guideMotion && variant->guideStatic)
    {
        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        stamp(kStageGuide);
        enc->setComputePipelineState(useMotion ? variant->guideMotion : variant->guideStatic);
        enc->setBuffer(uniformBuffer, 0, 0);
        enc->setBuffer(scene.instanceBuffer, 0, 1);
        enc->setAccelerationStructure(scene.instanceAccelerationStructure, 2);
        enc->setBuffer(mAovBuffer, 0, 4);
        enc->setBuffer(scene.materialBuffer, 0, 5);
        enc->setBuffer(scene.geometryEntryBuffer, 0, 6);
        enc->setBuffer(scene.vertexBuffer, 0, 7);
        enc->setBuffer(scene.prevVertexBuffer, 0, 8);
        enc->setBuffer(scene.indexBuffer, 0, 9);
        enc->setBuffer(scene.curvePointBuffer ? scene.curvePointBuffer : scene.placeholderBuffer, 0, 10);
        enc->setBuffer(scene.curveSegmentBuffer ? scene.curveSegmentBuffer : scene.placeholderBuffer, 0, 11);
        enc->setBuffer(scene.lightBuffer, 0, 12);
        enc->setBuffer(mControlBuffer, 0, 14);
        MTL::IntersectionFunctionTable* guideTable = useMotion ? variant->guideTableMotion : variant->guideTableStatic;
        enc->setIntersectionFunctionTable(guideTable, 13);
        enc->useResource(guideTable, MTL::ResourceUsageRead);
        enc->dispatchThreadgroups(mControlBuffer, kGuideArgsOffset, tg);
        enc->memoryBarrier(MTL::BarrierScopeBuffers);
    }

    // Resolve this launch's radiance into the persistent accumulation buffer.
    stamp(kStageResolve);
    enc->setComputePipelineState(mResolvePSO);
    enc->setBuffer(uniformBuffer, 0, 0);
    enc->setBuffer(mRadianceBuffer, 0, 1);
    enc->setBuffer(outputBuffer, 0, 2);
    enc->setBuffer(scene.accumulationBuffer, 0, 3);
    enc->setBytes(&sampleCount, sizeof(uint32_t), 4);
    enc->setBuffer(mAovBuffer, 0, 5);
    enc->setBuffer(scene.sharcHashBuffer, 0, 6);
    enc->setBuffer(scene.sharcResolvedBuffer, 0, 7);
    enc->dispatchThreads(grid, tg);

    if (profile && mStageStatsBuffer)
    {
        enc->endEncoding();
        MTL::BlitCommandEncoder* blit = pCmd->blitCommandEncoder();
        blit->copyFromBuffer(mControlBuffer, 0, mStageStatsBuffer, 0, mControlBuffer->length());
        blit->endEncoding();
        enc = pCmd->computeCommandEncoder();
        enc->setLabel(NS::String::string("stats readback", NS::UTF8StringEncoding));
        declareResidency(enc);
    }
    return enc;
}

// Build (or return) the pipeline set specialised for one combination of scene
// features. Compiling seven kernels takes a few milliseconds, which is fine
// because the key only changes when a setting or the scene does — never per
// frame.
const WavefrontVariant* MetalWavefrontIntegrator::variantFor(uint32_t features)
{
    const auto it = mVariants.find(features);
    if (it != mVariants.end())
    {
        return &it->second;
    }
    if (!mLibrary)
    {
        return nullptr;
    }

    MTL::FunctionConstantValues* values = MTL::FunctionConstantValues::alloc()->init();
    const bool envMap = (features & WavefrontFeatures::kEnvMap) != 0;
    const bool lights = (features & WavefrontFeatures::kLights) != 0;
    const bool motionBlur = (features & WavefrontFeatures::kMotionBlur) != 0;
    const bool dof = (features & WavefrontFeatures::kDof) != 0;
    const bool debug = (features & WavefrontFeatures::kDebug) != 0;
    const bool alpha = (features & WavefrontFeatures::kAlpha) != 0;
    const bool fog = (features & WavefrontFeatures::kFog) != 0;
    values->setConstantValue(&envMap, MTL::DataTypeBool, (NS::UInteger)0);
    values->setConstantValue(&lights, MTL::DataTypeBool, (NS::UInteger)1);
    values->setConstantValue(&motionBlur, MTL::DataTypeBool, (NS::UInteger)2);
    values->setConstantValue(&dof, MTL::DataTypeBool, (NS::UInteger)3);
    values->setConstantValue(&debug, MTL::DataTypeBool, (NS::UInteger)4);
    values->setConstantValue(&alpha, MTL::DataTypeBool, (NS::UInteger)5);
    values->setConstantValue(&fog, MTL::DataTypeBool, (NS::UInteger)6);
    const bool sharc = (features & WavefrontFeatures::kSharc) != 0;
    values->setConstantValue(&sharc, MTL::DataTypeBool, (NS::UInteger)7);
    const bool subsurface = (features & WavefrontFeatures::kSubsurface) != 0;
    values->setConstantValue(&subsurface, MTL::DataTypeBool, (NS::UInteger)8);

    // A pipeline built the Metal 3 way cannot be used with an argument table, so
    // the two paths need separate pipelines and the mode is part of the cache key.
    const bool useMetal4 = (features & WavefrontFeatures::kMetal4) != 0;
    // Curves change the intersector's *type*, which no function constant can do,
    // so this picks a different entry point out of the same library. `shade`
    // takes a constant as well: it has no intersector, only the branch that
    // rebuilds a curve hit's geometry, and that one is worth compiling out.
    const bool curves = (features & WavefrontFeatures::kCurves) != 0;
    values->setConstantValue(&curves, MTL::DataTypeBool, (NS::UInteger)9);
    const bool sharcUpdate = (features & WavefrontFeatures::kSharcUpdate) != 0;
    values->setConstantValue(&sharcUpdate, MTL::DataTypeBool, (NS::UInteger)10);
    const bool openpbr = (features & WavefrontFeatures::kOpenPBR) != 0;
    values->setConstantValue(&openpbr, MTL::DataTypeBool, (NS::UInteger)11);
    const bool auditRenderWork = (features & WavefrontFeatures::kRenderWorkAudit) != 0;
    values->setConstantValue(&auditRenderWork, MTL::DataTypeBool, (NS::UInteger)12);
    const bool restirRayTracedDiagnostic = (features & WavefrontFeatures::kRestirRayTracedDiagnostic) != 0;
    values->setConstantValue(&restirRayTracedDiagnostic, MTL::DataTypeBool, (NS::UInteger)13);
    const bool restir = (features & WavefrontFeatures::kRestir) != 0;
    values->setConstantValue(&restir, MTL::DataTypeBool, (NS::UInteger)14);
    const bool risOne = (features & WavefrontFeatures::kRisOne) != 0;
    values->setConstantValue(&risOne, MTL::DataTypeBool, (NS::UInteger)15);
    const bool aov = (features & WavefrontFeatures::kAov) != 0;
    values->setConstantValue(&aov, MTL::DataTypeBool, (NS::UInteger)16);
    const bool allOpenPBR = (features & WavefrontFeatures::kAllOpenPBR) != 0;
    values->setConstantValue(&allOpenPBR, MTL::DataTypeBool, (NS::UInteger)17);
    const uint32_t samplerType = (features & WavefrontFeatures::kSamplerMask) >> WavefrontFeatures::kSamplerShift;
    values->setConstantValue(&samplerType, MTL::DataTypeUInt, (NS::UInteger)18);
    const bool allNativeOpenPBR = (features & WavefrontFeatures::kAllNativeOpenPBR) != 0;
    values->setConstantValue(&allNativeOpenPBR, MTL::DataTypeBool, (NS::UInteger)19);
    const bool openpbrFeatureEnabled = true;
    values->setConstantValue(&openpbrFeatureEnabled, MTL::DataTypeBool, (NS::UInteger)20);
    values->setConstantValue(&openpbrFeatureEnabled, MTL::DataTypeBool, (NS::UInteger)21);
    values->setConstantValue(&openpbrFeatureEnabled, MTL::DataTypeBool, (NS::UInteger)22);
    values->setConstantValue(&openpbrFeatureEnabled, MTL::DataTypeBool, (NS::UInteger)23);
    const uint32_t shadeProbe = envUint("STRELKA_SHADE_PROBE", 0u);
    values->setConstantValue(&shadeProbe, MTL::DataTypeUInt, (NS::UInteger)24);
    const bool emissiveMeshLights = (features & WavefrontFeatures::kEmissiveMeshLights) != 0;
    values->setConstantValue(&emissiveMeshLights, MTL::DataTypeBool, (NS::UInteger)25);
    const bool allAnalyticLightsRect = (features & WavefrontFeatures::kAllAnalyticLightsRect) != 0;
    values->setConstantValue(&allAnalyticLightsRect, MTL::DataTypeBool, (NS::UInteger)26);
    const bool uniformRectLightSampling = (features & WavefrontFeatures::kUniformRectLightSampling) != 0;
    values->setConstantValue(&uniformRectLightSampling, MTL::DataTypeBool, (NS::UInteger)27);
    const bool splitBaseNee = (features & WavefrontFeatures::kSplitBaseNee) != 0;
    values->setConstantValue(&splitBaseNee, MTL::DataTypeBool, (NS::UInteger)28);
    auto entry = [&](const char* base) -> std::string {
        return curves ? std::string(base) + "Curve" : std::string(base);
    };
    NS::Error* err = nullptr;
    auto make = [&](const char* name, const char* label = nullptr) -> MTL::ComputePipelineState* {
        if (useMetal4)
        {
            return mMetal4->newComputePipelineState(mLibrary, name, values, label);
        }
        MTL::Function* fn = mLibrary->newFunction(NS::String::string(name, NS::UTF8StringEncoding), values, &err);
        if (!fn)
        {
            STRELKA_FATAL("wavefront: specialising {} -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
            return nullptr;
        }
        auto* descriptor = MTL::ComputePipelineDescriptor::alloc()->init();
        descriptor->setComputeFunction(fn);
        if (label)
        {
            descriptor->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
        }
        MTL::ComputePipelineState* pso =
            mDevice->newComputePipelineState(descriptor, MTL::PipelineOptionNone, nullptr, &err);
        descriptor->release();
        if (!pso)
        {
            STRELKA_FATAL("wavefront: {} -> {}", name, err ? err->localizedDescription()->utf8String() : "unknown error");
        }
        fn->release();
        return pso;
    };

    auto makeTraversal = [&](const std::string& name, const char* intersectionFamily, bool motion,
                             MTL::IntersectionFunctionTable*& table) -> MTL::ComputePipelineState* {
        const std::string suffix = std::string(motion ? "Motion" : "") + (curves ? "Curve" : "");
        const std::string sphereName = std::string("analyticSphereIntersection") + intersectionFamily + suffix;
        const std::string discName = std::string("analyticDiscIntersection") + intersectionFamily + suffix;
        MTL::Function* sphere = nullptr;
        MTL::Function* disc = nullptr;
        MTL::ComputePipelineState* pso = nullptr;
        if (useMetal4)
        {
            pso = mMetal4->newComputePipelineStateLinked(
                mLibrary, name.c_str(), sphereName.c_str(), discName.c_str(), values);
        }
        else
        {
            sphere = mLibrary->newFunction(NS::String::string(sphereName.c_str(), NS::UTF8StringEncoding), values, &err);
            disc = mLibrary->newFunction(NS::String::string(discName.c_str(), NS::UTF8StringEncoding), values, &err);
            MTL::Function* kernel =
                mLibrary->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding), values, &err);
            if (!sphere || !disc || !kernel)
            {
                STRELKA_FATAL("wavefront: loading procedural traversal functions for {}", name);
            }
            else
            {
                const NS::Object* functions[] = { sphere, disc };
                auto* linked = MTL::LinkedFunctions::alloc()->init();
                linked->setFunctions(NS::Array::array(functions, 2));
                auto* descriptor = MTL::ComputePipelineDescriptor::alloc()->init();
                descriptor->setComputeFunction(kernel);
                descriptor->setLinkedFunctions(linked);
                pso = mDevice->newComputePipelineState(descriptor, MTL::PipelineOptionNone, nullptr, &err);
                descriptor->release();
                linked->release();
            }
            if (kernel)
                kernel->release();
        }
        if (!pso)
        {
            STRELKA_FATAL("wavefront: procedural traversal pipeline {} -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
        }
        else
        {
            auto* descriptor = MTL::IntersectionFunctionTableDescriptor::alloc()->init();
            descriptor->setFunctionCount(ANALYTIC_INTERSECTION_FUNCTION_COUNT);
            table = pso->newIntersectionFunctionTable(descriptor);
            descriptor->release();
            const MTL::FunctionHandle* sphereHandle =
                sphere ? pso->functionHandle(sphere) :
                         pso->functionHandle(NS::String::string(sphereName.c_str(), NS::UTF8StringEncoding));
            const MTL::FunctionHandle* discHandle =
                disc ? pso->functionHandle(disc) :
                       pso->functionHandle(NS::String::string(discName.c_str(), NS::UTF8StringEncoding));
            if (!table || !sphereHandle || !discHandle)
            {
                STRELKA_FATAL("wavefront: procedural intersection table {}", name);
            }
            else
            {
                table->setFunction(sphereHandle, ANALYTIC_INTERSECTION_SPHERE);
                table->setFunction(discHandle, ANALYTIC_INTERSECTION_DISC);
            }
        }
        if (sphere)
            sphere->release();
        if (disc)
            disc->release();
        return pso;
    };

    WavefrontVariant v;
    v.generate = make("wavefrontGenerate");
    v.extendMotion = makeTraversal(entry("wavefrontExtend"), "Extend", true, v.extendTableMotion);
    v.extendStatic = makeTraversal(entry("wavefrontExtendStatic"), "Extend", false, v.extendTableStatic);
    if (subsurface && !sharcUpdate)
    {
        v.sssWalkMotion = make("wavefrontSssWalk");
        v.sssWalkStatic = make("wavefrontSssWalkStatic");
    }
    if (restirRayTracedDiagnostic)
    {
        v.shade = makeTraversal("wavefrontShade", "RestirShadeDiagnostic", false, v.restirShadeDiagnosticTable);
    }
    else if (openpbr)
    {
        v.shade = make("wavefrontShadeTail", "wavefrontShadeTail");
        // Plain one-candidate NEE can propose a light before material shading.
        // ReSTIR retains its coupled candidate/evaluation loop in shade.
        if (splitBaseNee)
        {
            v.connectBase = make("wavefrontConnectBase", "wavefrontConnectBase");
        }
        const bool disabled = false;
        values->setConstantValue(&disabled, MTL::DataTypeBool, (NS::UInteger)20);
        values->setConstantValue(&disabled, MTL::DataTypeBool, (NS::UInteger)23);
        // Bucket 2 contains unlayered water/glass/SSS only. Dispersion is
        // excluded by the host classifier, so compile it out together with
        // coat/fuzz and metallic rather than carrying Tail's mixed lobe stack.
        values->setConstantValue(&disabled, MTL::DataTypeBool, (NS::UInteger)21);
        v.shadeTranslucent = make("wavefrontShadeTranslucent", "wavefrontShadeTranslucent");
        values->setConstantValue(&openpbrFeatureEnabled, MTL::DataTypeBool, (NS::UInteger)20);
        values->setConstantValue(&openpbrFeatureEnabled, MTL::DataTypeBool, (NS::UInteger)21);
        values->setConstantValue(&openpbrFeatureEnabled, MTL::DataTypeBool, (NS::UInteger)23);
        values->setConstantValue(&disabled, MTL::DataTypeBool, (NS::UInteger)21);
        values->setConstantValue(&disabled, MTL::DataTypeBool, (NS::UInteger)22);
        v.shadeLayer = make("wavefrontShadeLayer", "wavefrontShadeLayer");
        values->setConstantValue(&disabled, MTL::DataTypeBool, (NS::UInteger)20);
        v.shadeBase = make("wavefrontShadeBase", "wavefrontShadeBase");
        values->setConstantValue(&openpbrFeatureEnabled, MTL::DataTypeBool, (NS::UInteger)20);
        values->setConstantValue(&openpbrFeatureEnabled, MTL::DataTypeBool, (NS::UInteger)21);
        values->setConstantValue(&openpbrFeatureEnabled, MTL::DataTypeBool, (NS::UInteger)22);
    }
    else
    {
        v.shade = make("wavefrontShade");
    }
    v.restirSpatialFinal = restirRayTracedDiagnostic ?
                               makeTraversal("wavefrontRestirSpatialFinal", "RestirSpatialDiagnostic", false,
                                             v.restirSpatialDiagnosticTable) :
                               make("wavefrontRestirSpatialFinal");
    v.miss = make("wavefrontMiss");
    v.shadowMotion = makeTraversal(entry("wavefrontShadow"), "Shadow", true, v.shadowTableMotion);
    v.shadowStatic = makeTraversal(entry("wavefrontShadowStatic"), "Shadow", false, v.shadowTableStatic);
    v.guideMotion = makeTraversal(entry("wavefrontGuide"), "Guide", true, v.guideTableMotion);
    v.guideStatic = makeTraversal(entry("wavefrontGuideStatic"), "Guide", false, v.guideTableStatic);

    values->release();
    mResidencyDirty = true;

    if (!v.shade)
    {
        return nullptr;
    }
    STRELKA_INFO(
        "wavefront variant env={} lights={} motion={} dof={} debug={} alpha={} fog={} sss={} sharc={} "
        "curves={} sharcUpdate={} openpbr={} allOpenpbr={} allNativeOpenpbr={} risOne={} aov={} sampler={} metal4={}",
        envMap, lights, motionBlur, dof, debug, alpha, fog, subsurface, sharc, curves, sharcUpdate, openpbr, allOpenPBR,
        allNativeOpenPBR, risOne, aov, samplerType, useMetal4);
    // maxTotalThreadsPerThreadgroup is Metal's available proxy for per-pipeline register pressure.
    auto tgLimit = [](MTL::ComputePipelineState* p) -> uint32_t {
        return p ? (uint32_t)p->maxTotalThreadsPerThreadgroup() : 0u;
    };
    STRELKA_INFO(
        "  maxThreadsPerTG: generate {} extend {} (motion {}) sssWalk {} (motion {}) connect base {} shade base {} layer {} "
        "translucent {} tail {} shadow {} (motion {}) guide {} (motion {}) miss {}",
        tgLimit(v.generate), tgLimit(v.extendStatic), tgLimit(v.extendMotion), tgLimit(v.sssWalkStatic),
        tgLimit(v.sssWalkMotion), tgLimit(v.connectBase), tgLimit(v.shadeBase), tgLimit(v.shadeLayer),
        tgLimit(v.shadeTranslucent), tgLimit(v.shade), tgLimit(v.shadowStatic), tgLimit(v.shadowMotion),
        tgLimit(v.guideStatic), tgLimit(v.guideMotion), tgLimit(v.miss));
    return &mVariants.emplace(features, v).first->second;
}

void MetalWavefrontIntegrator::buildPipelines()
{
    const std::string path = oka::resolveResourcePath("metal/shaders/wavefront.metallib");
    NS::Error* loadErr = nullptr;
    MTL::Library* lib = mDevice->newLibrary(NS::String::string(path.c_str(), NS::UTF8StringEncoding), &loadErr);
    if (!lib)
    {
        STRELKA_FATAL(
            "Failed to load {}: {}", path, loadErr ? loadErr->localizedDescription()->utf8String() : "unknown error");
        return;
    }
    mLibrary = lib->retain();
    NS::Error* err = nullptr;
    // Only the kernels that reference no function constants are built here. The
    // rest are specialised per scene by variantFor(), and Metal refuses
    // to build a pipeline from an unspecialised function that declares any.
    auto make = [&](const char* name) -> MTL::ComputePipelineState* {
        MTL::Function* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
        if (!fn)
        {
            STRELKA_FATAL("wavefront: missing function {}", name);
            return nullptr;
        }
        MTL::ComputePipelineState* pso = mDevice->newComputePipelineState(fn, &err);
        if (!pso)
        {
            STRELKA_FATAL("wavefront: {} -> {}", name, err ? err->localizedDescription()->utf8String() : "unknown error");
        }
        fn->release();
        return pso;
    };
    mResolvePSO = make("wavefrontResolve");
    mPreparePSO = make("wavefrontPrepare");
    mPrepareShadowPSO = make("wavefrontPrepareShadow");
    mPrepareHitMissPSO = make("wavefrontPrepareHitMiss");
    mClassifySssPSO = make("wavefrontClassifySss");
    mPrepareSssPSO = make("wavefrontPrepareSss");
    mAovResolvePSO = make("wavefrontAovResolve");
    mSharcClearPSO = make("sharcClear");
    mSharcResolvePSO = make("sharcResolve");
    if (mMetal4 && mMetal4->isValid())
    {
        // The same four stages again, built by the other compiler: a pipeline is
        // tied to the binding model it was compiled for.
        mResolvePSO4 = mMetal4->newComputePipelineState(lib, "wavefrontResolve", nullptr);
        mPreparePSO4 = mMetal4->newComputePipelineState(lib, "wavefrontPrepare", nullptr);
        mPrepareShadowPSO4 = mMetal4->newComputePipelineState(lib, "wavefrontPrepareShadow", nullptr);
        mPrepareHitMissPSO4 = mMetal4->newComputePipelineState(lib, "wavefrontPrepareHitMiss", nullptr);
        mClassifySssPSO4 = mMetal4->newComputePipelineState(lib, "wavefrontClassifySss", nullptr);
        mPrepareSssPSO4 = mMetal4->newComputePipelineState(lib, "wavefrontPrepareSss", nullptr);
        mStageBreadcrumbPSO4 = mMetal4->newComputePipelineState(lib, "wavefrontStageBreadcrumb", nullptr);
        // The guide resolve too: without it the Metal 4 path cannot feed either
        // MetalFX mode, both of which read depth and motion from these textures.
        mAovResolvePSO4 = mMetal4->newComputePipelineState(lib, "wavefrontAovResolve", nullptr);
        mSharcClearPSO4 = mMetal4->newComputePipelineState(lib, "sharcClear", nullptr);
        mSharcResolvePSO4 = mMetal4->newComputePipelineState(lib, "sharcResolve", nullptr);
    }
    lib->release();
}

void MetalWavefrontIntegrator::ensureBuffers(uint32_t width,
                                             uint32_t height,
                                             uint32_t sharcUpdateDownscale,
                                             bool restirEnabled,
                                             bool restirBasic,
                                             bool splitBaseNee)
{
    const uint32_t pixels = width * height;
    sharcUpdateDownscale = std::max(sharcUpdateDownscale, 1u);
    if (pixels == mCapacity && sharcUpdateDownscale == mSharcUpdateDownscale && restirEnabled == mRestirAllocated &&
        restirBasic == mRestirBasicAllocated && splitBaseNee == mSplitBaseNeeAllocated && mPathStateBuffer)
    {
        return;
    }
    // Out of the residency set before it is freed, for the same reason the
    // acceleration structures do it: the set does not retain what it names, so a
    // released allocation leaves a dangling entry, and the allocator readily
    // hands the same address back for the replacement below. addAllocation then
    // sees an address the set already holds and the new buffer is never made
    // resident. `makeResourcesResidentForMetal4` reconciles by pointer
    // set-difference, so a reused address is invisible to it and cannot repair
    // this. Every buffer released here is named by addResidentAllocations().
    auto release = [this](MTL::Buffer*& b) {
        if (b)
        {
            if (mMetal4)
            {
                mMetal4->removeResident(b);
            }
            b->release();
            b = nullptr;
        }
    };
    release(mPathStateBuffer);
    release(mMediumPathStateBuffer);
    release(mSharcUpdateStateBuffer);
    release(mPathRayBuffer);
    release(mHitBuffer);
    release(mIorStackBuffer);
    release(mRadianceBuffer);
    release(mGuideRayBuffer);
    release(mSurfaceGeometryBuffer);
    release(mBaseLightConnectionBuffer);
    release(mGuideQueueBuffer);
    release(mPathQueueBuffer[0]);
    release(mPathQueueBuffer[1]);
    release(mSssQueueBuffer);
    release(mSssControlBuffer);
    release(mControlBuffer);
    release(mTraversalDispatchBuffer);
    release(mShadowRayBuffer);
    release(mStageStatsBuffer);
    release(mHitQueueBuffer);
    release(mAovBuffer);
    release(mMissQueueBuffer);
    release(mRestirReservoirBuffer[0]);
    release(mRestirReservoirBuffer[1]);
    release(mRestirSurfaceHistoryBuffer[0]);
    release(mRestirSurfaceHistoryBuffer[1]);
    release(mRestirSurfaceDataBuffer[0]);
    release(mRestirSurfaceDataBuffer[1]);

    metal::WavefrontElementSizes sz;
    sz.pathState = sizeof(PathState);
    sz.mediumPathState = sizeof(MediumPathState);
    sz.sharcUpdateState = sizeof(SharcUpdateState);
    sz.pathRay = sizeof(PathRay);
    sz.hitRecord = sizeof(HitRecord);
    sz.iorStack = sizeof(IorStack);
    sz.radiance = sizeof(simd::float4);
    sz.guideRay = sizeof(GuideRay);
    sz.surfaceGeometry = sizeof(SurfaceGeometryPayload);
    sz.baseLightConnection = sizeof(BaseLightConnectionPayload);
    sz.shadowRay = sizeof(ShadowRay);
    sz.aovSample = sizeof(AovSample);
    sz.restirReservoir = sizeof(RestirReservoir);
    sz.restirSurfaceHistory = sizeof(RestirSurfaceHistory);
    sz.restirShadingPoint = sizeof(RestirShadingPoint);
    sz.restirTargetSurface = sizeof(RestirTargetSurface);
    const metal::WavefrontBufferLayout layout =
        metal::wavefrontBufferLayout(width, height, sz, sharcUpdateDownscale, restirEnabled);

    // Private storage: these never leave the GPU.
    mPathStateBuffer = mDevice->newBuffer(layout.pathStateBytes, MTL::ResourceStorageModePrivate);
    mMediumPathStateBuffer = mDevice->newBuffer(layout.mediumPathStateBytes, MTL::ResourceStorageModePrivate);
    mSharcUpdateStateBuffer = mDevice->newBuffer(layout.sharcUpdateStateBytes, MTL::ResourceStorageModePrivate);
    mPathRayBuffer = mDevice->newBuffer(layout.pathRayBytes, MTL::ResourceStorageModePrivate);
    mHitBuffer = mDevice->newBuffer(layout.hitBytes, MTL::ResourceStorageModePrivate);
    mIorStackBuffer = mDevice->newBuffer(layout.iorStackBytes, MTL::ResourceStorageModePrivate);
    mRadianceBuffer = mDevice->newBuffer(layout.radianceBytes, MTL::ResourceStorageModePrivate);
    mGuideRayBuffer = mDevice->newBuffer(layout.guideRayBytes, MTL::ResourceStorageModePrivate);
    mSurfaceGeometryBuffer = mDevice->newBuffer(layout.surfaceGeometryBytes, MTL::ResourceStorageModePrivate);
    if (splitBaseNee)
    {
        mBaseLightConnectionBuffer = mDevice->newBuffer(layout.baseLightConnectionBytes, MTL::ResourceStorageModePrivate);
    }
    mGuideQueueBuffer = mDevice->newBuffer(layout.guideQueueBytes, MTL::ResourceStorageModePrivate);
    mPathQueueBuffer[0] = mDevice->newBuffer(layout.pathQueueBytes, MTL::ResourceStorageModePrivate);
    mPathQueueBuffer[1] = mDevice->newBuffer(layout.pathQueueBytes, MTL::ResourceStorageModePrivate);
    mSssQueueBuffer = mDevice->newBuffer(layout.pathQueueBytes, MTL::ResourceStorageModePrivate);
    mSssControlBuffer = mDevice->newBuffer(5u * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    // Queue counters, active counts, and two sets of indirect dispatch arguments.
    mControlBuffer = mDevice->newBuffer(layout.controlBytes, MTL::ResourceStorageModePrivate);
    // Reused within a bounce: prepare fills it for extend, then prepareShadow
    // overwrites it after extend has completed.
    mTraversalDispatchBuffer = mDevice->newBuffer(layout.traversalDispatchBytes, MTL::ResourceStorageModePrivate);
    // At most one deferred connection per path per bounce.
    mShadowRayBuffer = mDevice->newBuffer(layout.shadowRayBytes, MTL::ResourceStorageModePrivate);
    mStageStatsBuffer = mDevice->newBuffer(layout.stageStatsBytes, MTL::ResourceStorageModeShared);
    // Shared, so it needs no blit to read: the Metal 4 path encodes none, and
    // two words are not worth an encoder either way.
    if (!mIorStatsBuffer)
    {
        mIorStatsBuffer =
            mDevice->newBuffer((IOR_STAT_COUNT + SHARC_STAT_COUNT) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        memset(mIorStatsBuffer->contents(), 0, mIorStatsBuffer->length());
    }
    mAovBuffer = mDevice->newBuffer(layout.aovBytes, MTL::ResourceStorageModePrivate);
    mHitQueueBuffer =
        mDevice->newBuffer(4u * layout.hitQueueBytes + 4u * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    mMissQueueBuffer = mDevice->newBuffer(layout.missQueueBytes, MTL::ResourceStorageModePrivate);
    if (restirEnabled)
    {
        for (uint32_t i = 0; i < 2; ++i)
        {
            mRestirReservoirBuffer[i] = mDevice->newBuffer(layout.restirReservoirBytes, MTL::ResourceStorageModePrivate);
            mRestirSurfaceHistoryBuffer[i] =
                mDevice->newBuffer(layout.restirSurfaceHistoryBytes, MTL::ResourceStorageModePrivate);
        }
        const size_t surfaceBytes = restirBasic ? layout.restirTargetSurfaceBytes : layout.restirShadingPointBytes;
        mRestirSurfaceDataBuffer[0] = mDevice->newBuffer(surfaceBytes, MTL::ResourceStorageModePrivate);
        if (restirBasic)
        {
            mRestirSurfaceDataBuffer[1] = mDevice->newBuffer(surfaceBytes, MTL::ResourceStorageModePrivate);
        }
    }

    mCapacity = pixels;
    mSharcUpdateDownscale = sharcUpdateDownscale;
    mRestirAllocated = restirEnabled;
    mRestirBasicAllocated = restirBasic;
    mSplitBaseNeeAllocated = splitBaseNee;

    STRELKA_INFO("wavefront buffers for {}x{}: {:.1f} MB total", width, height, queueBytes() / (1024.0 * 1024.0));
}
