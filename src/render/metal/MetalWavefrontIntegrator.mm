#include "MetalWavefrontIntegrator.h"

#include "MetalBuffer.h"
#include "integrator_buffer_sizes.h"

#include <algorithm>
#include <cstring>
#include <string>

#include <log.h>
#include <paths.h>

#include <simd/simd.h>

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
    safeRelease(mPathStateBuffer);
    safeRelease(mPathRayBuffer);
    safeRelease(mHitBuffer);
    safeRelease(mIorStackBuffer);
    safeRelease(mRadianceBuffer);
    safeRelease(mGuideRadianceBuffer);
    safeRelease(mPathQueueBuffer[0]);
    safeRelease(mPathQueueBuffer[1]);
    safeRelease(mControlBuffer);
    safeRelease(mShadowRayBuffer);
    safeRelease(mHitQueueBuffer);
    safeRelease(mAovBuffer);
    safeRelease(mMissQueueBuffer);
    for (auto& kv : mVariants)
    {
        safeRelease(kv.second.generate);
        safeRelease(kv.second.extendMotion);
        safeRelease(kv.second.extendStatic);
        safeRelease(kv.second.shade);
        safeRelease(kv.second.miss);
        safeRelease(kv.second.shadowMotion);
        safeRelease(kv.second.shadowStatic);
        safeRelease(kv.second.shadowTableMotion);
        safeRelease(kv.second.shadowTableStatic);
        safeRelease(kv.second.sharcDeposit);
    }
    mVariants.clear();
    safeRelease(mLibrary);
    safeRelease(mPrepareHitMissPSO);
    safeRelease(mResolvePSO4);
    safeRelease(mPreparePSO4);
    safeRelease(mPrepareShadowPSO4);
    safeRelease(mPrepareHitMissPSO4);
    safeRelease(mAovResolvePSO);
    safeRelease(mAovResolvePSO4);
    safeRelease(mStageTimestampBuffer);
    safeRelease(mStageStatsBuffer);
    safeRelease(mIorStatsBuffer);
    safeRelease(mStageCounterHeap);
    mCapacity = 0;
    mResidencyDirty = true;
    mReportedIorStats = false;
    mGpuTicksToMs = 0.0;
    mStageKinds.clear();
}

void MetalWavefrontIntegrator::addResidentAllocations(const std::function<void(MTL::Allocation*)>& add) const
{
    add(mIorStatsBuffer);
    add(mPathStateBuffer);
    add(mPathRayBuffer);
    add(mHitBuffer);
    add(mIorStackBuffer);
    add(mRadianceBuffer);
    add(mGuideRadianceBuffer);
    add(mPathQueueBuffer[0]);
    add(mPathQueueBuffer[1]);
    add(mHitQueueBuffer);
    add(mMissQueueBuffer);
    add(mShadowRayBuffer);
    add(mAovBuffer);
    add(mControlBuffer);
    for (const auto& entry : mVariants)
    {
        add(entry.second.shadowTableMotion);
        add(entry.second.shadowTableStatic);
    }
}

size_t MetalWavefrontIntegrator::queueBytes() const
{
    auto bufBytes = [](MTL::Buffer* b) { return b ? b->length() : 0; };
    return bufBytes(mPathStateBuffer) + bufBytes(mPathRayBuffer) + bufBytes(mHitBuffer) +
           bufBytes(mIorStackBuffer) + bufBytes(mRadianceBuffer) + bufBytes(mGuideRadianceBuffer) +
           bufBytes(mPathQueueBuffer[0]) + bufBytes(mPathQueueBuffer[1]) + bufBytes(mControlBuffer) +
           bufBytes(mShadowRayBuffer) + bufBytes(mHitQueueBuffer) + bufBytes(mMissQueueBuffer) +
           bufBytes(mAovBuffer) + bufBytes(mStageStatsBuffer);
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
    kStageShade,
    kStagePrepareShadow,
    kStageShadow,
    kStageMiss,
    kStageResolve,
    kStageSort,
    kStageCount
};
const char* const kStageNames[kStageCount] = { "generate",   "prepare", "extend", "shade",
                                               "prepShadow", "shadow",  "miss",   "resolve",
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
    MTL::CounterSet* timestampSet = nullptr;
    NS::Array* sets = mDevice->counterSets();
    for (NS::UInteger i = 0; sets && i < sets->count(); ++i)
    {
        MTL::CounterSet* set = static_cast<MTL::CounterSet*>(sets->object(i));
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
        STRELKA_WARNING("stage profiling unavailable: {}",
                        err ? err->localizedDescription()->utf8String() : "unknown error");
    }
}

// Resolve the timestamps and print the per-stage breakdown. GPU timestamps are
// in nanoseconds on Apple Silicon; a sample can come back as MTLCounterErrorValue
// when the GPU dropped it, and those gaps are skipped rather than counted as
// enormous durations.
void MetalWavefrontIntegrator::createStageCounterHeap()
{
    if (mStageCounterHeap || !mMetal4 || !mMetal4->isValid())
    {
        return;
    }
    MTL4::CounterHeapDescriptor* d = MTL4::CounterHeapDescriptor::alloc()->init();
    d->setType(MTL4::CounterHeapTypeTimestamp);
    d->setCount(static_cast<NS::UInteger>(kMaxStageSamples) * 2);
    NS::Error* err = nullptr;
    mStageCounterHeap = mDevice->newCounterHeap(d, &err);
    d->release();
    if (!mStageCounterHeap)
    {
        STRELKA_WARNING("Metal 4 counter heap unavailable: {}",
                        err && err->localizedDescription() ? err->localizedDescription()->utf8String()
                                                           : "unknown error");
    }
}

// Same report as the Metal 3 path, from timestamps written inline rather than at
// encoder boundaries. Stage kinds are recorded in encode order, one pair each.
void MetalWavefrontIntegrator::reportStageTimingsMetal4(double lastRenderTimeMs)
{
    if (!mStageCounterHeap || mStageKinds.empty())
    {
        return;
    }
    const NS::UInteger n = mStageKinds.size() + 1;
    NS::Data* data = mStageCounterHeap->resolveCounterRange(NS::Range::Make(0, n));
    if (!data)
    {
        return;
    }
    const MTL4::TimestampHeapEntry* ts = static_cast<const MTL4::TimestampHeapEntry*>(data->bytes());

    // Anchor the tick scale once, against the frame this heap just timed.
    //
    // The obvious calibration -- MTLDevice::sampleTimestamps, which pairs a CPU
    // clock with a GPU one -- does not apply: it reports the GPU side in
    // nanoseconds, while the counter heap ticks at its own rate (about 24 MHz on
    // this part, so ~42 ns each). Calibrating against it returned a factor of
    // exactly 1 ns/tick and a report forty times too small. The command buffer's
    // measured GPU time is the one number known to be in milliseconds and to
    // cover the same work, so the marks are scaled to it.
    //
    // Everything outside the marks -- the resolve and the tonemapper, together
    // well under a millisecond -- is folded into the factor, which biases it by
    // under a percent and only once.
    // A mark that was never written reads back as zero, and a run that wrote
    // fewer marks than the last one leaves the tail stale. Anchoring on either
    // gives a span orders of magnitude too large and a scale that rounds every
    // stage to nothing -- which is how this first presented.
    bool marksValid = ts[0].timestamp != 0;
    for (NS::UInteger i = 0; marksValid && i < mStageKinds.size(); ++i)
    {
        marksValid = ts[i + 1].timestamp >= ts[i].timestamp && ts[i + 1].timestamp != 0;
    }
    if (!marksValid)
    {
        return;
    }

    const uint64_t spanTicks = ts[mStageKinds.size()].timestamp - ts[0].timestamp;
    if (mGpuTicksToMs == 0.0 && spanTicks > 0)
    {
        const double frameMs = lastRenderTimeMs;
        if (frameMs > 0.0)
        {
            mGpuTicksToMs = frameMs / double(spanTicks);
            STRELKA_INFO("GPU timestamp scale anchored: 1 tick = {:.2f} ns ({} ticks over {:.2f} ms)",
                         mGpuTicksToMs * 1e6, spanTicks, frameMs);
        }
    }
    if (mGpuTicksToMs == 0.0)
    {
        return;
    }

    double totals[kStageCount] = {};
    uint32_t counts[kStageCount] = {};
    std::string perBounce[kStageCount];
    for (NS::UInteger i = 0; i < mStageKinds.size(); ++i)
    {
        const uint64_t a = ts[i].timestamp;
        const uint64_t b = ts[i + 1].timestamp;
        if (a == 0 || b <= a)
        {
            continue;
        }
        const uint8_t kind = mStageKinds[i];
        const double ms = double(b - a) * mGpuTicksToMs;
        totals[kind] += ms;
        ++counts[kind];
        if (kind == kStageExtend || kind == kStageShade || kind == kStageShadow)
        {
            perBounce[kind] += fmt::format("{:.2f} ", ms);
        }
    }
    double sum = 0.0;
    for (uint32_t k = 0; k < kStageCount; ++k)
    {
        sum += totals[k];
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
    // Diagnostic: the span between the first and last mark against the command
    // buffer's own GPU time. If the span matches and the gaps do not, the marks
    // are misplaced; if neither matches, the clock is not the one being assumed.
    STRELKA_INFO("STAGES total {:.2f}ms  {}", sum, line);
    STRELKA_INFO("STAGES per bounce: extend [{}] shade [{}] shadow [{}]", perBounce[kStageExtend],
                 perBounce[kStageShade], perBounce[kStageShadow]);
}

void MetalWavefrontIntegrator::reportStageTimings()
{
    if (!mStageTimestampBuffer || mStageKinds.empty())
    {
        return;
    }
    const NS::UInteger n = 2 * mStageKinds.size();
    NS::Data* data = mStageTimestampBuffer->resolveCounterRange(NS::Range::Make(0, n));
    if (!data)
    {
        return;
    }
    const MTL::CounterResultTimestamp* ts = static_cast<const MTL::CounterResultTimestamp*>(data->bytes());

    double totals[kStageCount] = {};
    uint32_t counts[kStageCount] = {};
    // Per-bounce durations of the three traversal-heavy stages. The cost of a
    // bounce says more than the total does: bounce 0 is a coherent primary pass
    // and the later ones are not, which is what decides whether sorting rays is
    // worth anything.
    std::string perBounce[kStageCount];
    for (NS::UInteger i = 0; i < mStageKinds.size(); ++i)
    {
        const MTL::CounterResultTimestamp& a = ts[2 * i];
        const MTL::CounterResultTimestamp& b = ts[2 * i + 1];
        if (a.timestamp == MTL::CounterErrorValue || b.timestamp == MTL::CounterErrorValue ||
            b.timestamp <= a.timestamp)
        {
            continue;
        }
        const uint8_t kind = mStageKinds[i];
        const double ms = static_cast<double>(b.timestamp - a.timestamp) / 1e6; // ns -> ms
        totals[kind] += ms;
        ++counts[kind];
        if (kind == kStageExtend || kind == kStageShade || kind == kStageShadow)
        {
            perBounce[kind] += fmt::format("{:.2f} ", ms);
        }
    }

    double sum = 0.0;
    for (uint32_t k = 0; k < kStageCount; ++k)
    {
        sum += totals[k];
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
    STRELKA_INFO("STAGES per bounce: extend [{}] shade [{}] shadow [{}]", perBounce[kStageExtend],
                 perBounce[kStageShade], perBounce[kStageShadow]);

    if (mStageStatsBuffer)
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
}

// What the nested-dielectric stack lost, once per scene.
//
// Once, not once per frame: this is a fact about the geometry and the materials,
// not about this frame, and a warning that fires sixty times a second is a
// warning nobody reads. The counters are per *sample*, because `generate` clears
// them and generate runs per sample -- so the number is a rate, and comparable
// between a 4 spp preview and a 4096 spp render.
//
// The two failures mean different things and are worth telling apart. An
// overflow is four nested dielectrics, which is a scene that wants a deeper
// stack. An unmatched pop is a ray leaving something it never entered, which is
// almost always a mesh with a hole in it -- and that one cannot be fixed in the
// renderer at all, because the ray left through the hole without crossing a
// surface. See docs/open-defects.md entry 1.
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
        overflow, IOR_STACK_SIZE, unmatched, escaped);
}

void MetalWavefrontIntegrator::encodeMetal4(MTL4::CommandBuffer* cmd, MTL4::ComputeCommandEncoder*& enc,
                                            const IntegratorSceneBindings& scene,
                                            const IntegratorFrameRequest& frame)
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
    const uint32_t pixels = width * height;
    const uint32_t bounceIterations = frame.bounceIterations;
    MTL::Buffer* outputBuffer = ((MetalBuffer*)output)->getNativePtr();
    const auto* uniforms = reinterpret_cast<const Uniforms*>(uniformBuffer->contents());
    const uint32_t dispatchSampleCount = sampleCount + (uniforms->canonicalGuideSample ? 1u : 0u);

    MTL4::ArgumentTable* table = mMetal4->argumentTable();
    ConstantRing& ring = mMetal4->constants();
    enc->setArgumentTable(table);

    const uint32_t kThreadsPerGroup = 64;
    const MTL::Size tg = MTL::Size(kThreadsPerGroup, 1, 1);
    const MTL::Size fullGrid = MTL::Size((pixels + kThreadsPerGroup - 1) / kThreadsPerGroup, 1, 1);
    const MTL::GPUAddress control = mControlBuffer->gpuAddress();
    const NS::UInteger kDispatchArgsOffset = 2 * sizeof(uint32_t);
    const NS::UInteger kShadowArgsOffset = 8 * sizeof(uint32_t);
    const NS::UInteger kShadowCounterOffset = 6 * sizeof(uint32_t);
    const NS::UInteger kHitArgsOffset = 13 * sizeof(uint32_t);
    const NS::UInteger kHitCounterOffset = 11 * sizeof(uint32_t);
    const NS::UInteger kMissArgsOffset = 18 * sizeof(uint32_t);
    const NS::UInteger kMissCounterOffset = 16 * sizeof(uint32_t);

    const bool useMotion = frame.motionBlasBuilt || variant->extendStatic == nullptr ||
                           frame.settings->getAs<uint32_t>("render/pt/staticTraversal") == 0;

    // One timestamp per stage boundary. Each mark is written in the *same*
    // encoder as the work it closes, immediately after a queue-scoped barrier
    // that waits for that work -- not at the start of the next encoder, which is
    // where the first working version put it and which attributed each stage's
    // wait to its successor (extend and shadow swapped places against Metal 3).
    //
    // The barrier has to be queue-scoped: barrierAfterEncoderStages, which every
    // stage below uses, orders only within one encoder, so a split without it
    // left stages reading indirect arguments the previous one had not finished
    // writing and the frame collapsed to a tenth of its work.
    //
    // A stage's duration is the gap to the next mark, so there is one more mark
    // than stage. This reshapes the frame exactly as the Metal 3 path does -- a
    // measurement mode, not something to leave on.
    const bool profile = frame.profileStages && mStageCounterHeap != nullptr;
    uint32_t markIndex = 0;
    auto writeMark = [&]() {
        enc->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        enc->writeTimestamp(MTL4::TimestampGranularityPrecise, mStageCounterHeap, markIndex++);
        enc->endEncoding();
        enc = cmd->computeCommandEncoder();
        enc->setArgumentTable(table);
    };
    auto mark = [&](uint8_t kind) {
        if (!profile || mStageKinds.size() + 1 >= kMaxStageSamples)
        {
            return;
        }
        // The opening mark rides in the encoder that carries the first stage's
        // work. On its own, in an encoder holding nothing else, it never lands:
        // the timestamp reads back zero and the whole run is discarded. Nothing
        // precedes it, so it needs no barrier either.
        if (markIndex == 0)
        {
            enc->writeTimestamp(MTL4::TimestampGranularityPrecise, mStageCounterHeap, markIndex++);
        }
        mStageKinds.push_back(kind);
    };
    auto closeStage = [&]() {
        if (!profile || mStageKinds.empty())
        {
            return;
        }
        writeMark();
    };

    // Every dispatch here reads what the one before it wrote. Metal 4 does not
    // work that out, so say it: dispatch-to-dispatch, visible device-wide.
    auto barrier = [&]() {
        enc->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    };
    auto bind = [&](MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index) {
        table->setAddress(buffer ? buffer->gpuAddress() + offset : 0, index);
    };

    for (uint32_t s = 0; s < dispatchSampleCount; ++s)
    {
        const MTL::GPUAddress sampleIdx = ring.push(s);
        MTL::Buffer* sampleRadiance =
            (uniforms->canonicalGuideSample && s == 0u) ? mGuideRadianceBuffer : mRadianceBuffer;

        mark(kStageGenerate);
        enc->setComputePipelineState(variant->generate);
        bind(uniformBuffer, 0, 0);
        bind(mPathStateBuffer, 0, 1);
        bind(sampleRadiance, 0, 2);
        bind(mIorStackBuffer, 0, 3);
        table->setAddress(sampleIdx, 4);
        bind(mPathQueueBuffer[0], 0, 5);
        bind(mControlBuffer, 0, 6);
        bind(mAovBuffer, 0, 7);
        bind(mPathRayBuffer, 0, 8);
        bind(mIorStatsBuffer, 0, 9);
        enc->dispatchThreadgroups(fullGrid, tg);
        closeStage();
        barrier();

        for (uint32_t bounce = 0; bounce < bounceIterations; ++bounce)
        {
            const uint32_t src = bounce & 1u;
            const uint32_t dst = src ^ 1u;
            const MTL::GPUAddress srcIdx = ring.push(src);
            const MTL::GPUAddress groupSize = ring.push(kThreadsPerGroup);
            const MTL::GPUAddress bounceIdx = ring.push(bounce);

            enc->setComputePipelineState(mPreparePSO4);
            bind(mControlBuffer, 0, 0);
            table->setAddress(srcIdx, 1);
            table->setAddress(groupSize, 2);
            table->setAddress(bounceIdx, 3);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            barrier();

            mark(kStageExtend);
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
                (bounce == 0) ? uniforms->primaryRayMask
                              : (uniforms->primaryRayMask | GEOMETRY_MASK_LIGHT_HIDDEN);
            table->setAddress(ring.push(extendMask), 14);
            enc->dispatchThreadgroups(control + kDispatchArgsOffset, tg);
            closeStage();
            barrier();

            enc->setComputePipelineState(mPrepareHitMissPSO4);
            bind(mControlBuffer, 0, 0);
            table->setAddress(groupSize, 1);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            barrier();

            mark(kStageMiss);
            enc->setComputePipelineState(variant->miss);
            bind(uniformBuffer, 0, 0);
            bind(mPathStateBuffer, 0, 1);
            bind(mPathRayBuffer, 0, 2);
            bind(sampleRadiance, 0, 3);
            bind(mMissQueueBuffer, 0, 4);
            bind(mControlBuffer, 0, 5);
            bind(mAovBuffer, 0, 6);
            table->setAddress(sampleIdx, 7);
            bind(mIorStackBuffer, 0, 8);
            bind(mIorStatsBuffer, 0, 9);
            if (scene.environment && scene.environment->state().mapTexture)
            {
                table->setTexture(scene.environment->state().mapTexture->gpuResourceID(), 0);
                table->setTexture((scene.environment->state().backgroundTexture ? scene.environment->state().backgroundTexture : scene.environment->state().mapTexture)
                                      ->gpuResourceID(),
                                  1);
            }
            enc->dispatchThreadgroups(control + kMissArgsOffset, tg);
            closeStage();

            mark(kStageShade);
            enc->setComputePipelineState(variant->shade);
            bind(uniformBuffer, 0, 0);
            bind(scene.instanceBuffer, 0, 1);
            bind(scene.iesBuffer, 0, 2);
            bind(scene.lightBuffer, 0, 3);
            bind(scene.materialBuffer, 0, 4);
            bind(mPathStateBuffer, 0, 5);
            bind(mHitBuffer, 0, 6);
            bind(sampleRadiance, 0, 7);
            bind(mIorStackBuffer, 0, 8);
            bind(scene.geometryEntryBuffer, 0, 9);
            bind(scene.environment->state().aliasBuffer, 0, 10);
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
            bind(scene.prevFrameVertexBuffer ? scene.prevFrameVertexBuffer : scene.vertexBuffer, 0, 23);
            bind(scene.prevFrameInstanceBuffer ? scene.prevFrameInstanceBuffer : scene.instanceBuffer, 0, 24);
            // See the Metal 3 path: a curve hit is rebuilt from these, not carried.
            if (scene.curvePointBuffer)
            {
                bind(scene.curvePointBuffer, 0, 26);
                bind(scene.curveSegmentBuffer, 0, 27);
            }
            bind(mIorStatsBuffer, 0, 28);
            enc->dispatchThreadgroups(control + kHitArgsOffset, tg);
            closeStage();
            barrier();

            enc->setComputePipelineState(mPrepareShadowPSO4);
            bind(mControlBuffer, 0, 0);
            table->setAddress(groupSize, 1);
            table->setAddress(bounceIdx, 2);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            barrier();

            mark(kStageShadow);
            enc->setComputePipelineState(useMotion ? variant->shadowMotion : variant->shadowStatic);
            bind(uniformBuffer, 0, 0);
            table->setResource(scene.instanceAccelerationStructure->gpuResourceID(), 1);
            bind(mShadowRayBuffer, 0, 2);
            bind(sampleRadiance, 0, 3);
            bind(mControlBuffer, 0, 4);
            table->setAddress(sampleIdx, 5);
            bind(scene.instanceBuffer, 0, 6);
            bind(scene.materialBuffer, 0, 7);
            bind(scene.geometryEntryBuffer, 0, 8);
            bind(scene.vertexBuffer, 0, 9);
            bind(scene.indexBuffer, 0, 10);
            // Cutout shadows: without this the shadow kernel calls into a table
            // that was never bound, and a canopy blocks light outright instead
            // of letting the alpha test decide.
            MTL::IntersectionFunctionTable* shadowTable =
                useMotion ? variant->shadowTableMotion : variant->shadowTableStatic;
            if (shadowTable)
            {
                shadowTable->setBuffer(scene.instanceBuffer, 0, 0);
                shadowTable->setBuffer(scene.materialBuffer, 0, 1);
                shadowTable->setBuffer(scene.geometryEntryBuffer, 0, 2);
                shadowTable->setBuffer(scene.vertexBuffer, 0, 3);
                shadowTable->setBuffer(scene.indexBuffer, 0, 4);
                table->setResource(shadowTable->gpuResourceID(), 11);
            }
            enc->dispatchThreadgroups(control + kShadowArgsOffset, tg);
            closeStage();
            barrier();
        }
    }

    enc->setComputePipelineState(mResolvePSO4);
    bind(uniformBuffer, 0, 0);
    bind(mRadianceBuffer, 0, 1);
    bind(outputBuffer, 0, 2);
    bind(scene.accumulationBuffer, 0, 3);
    table->setAddress(ring.push(sampleCount), 4);
    bind(mAovBuffer, 0, 5);
    enc->dispatchThreadgroups(fullGrid, tg);

    // Guide resolve. The guides exist only when something downstream reads them,
    // which is either MetalFX mode, so their presence is the condition -- the
    // encoder has no other way to know which one the frame selected.
    if (scene.guideColor && mAovResolvePSO4)
    {
        barrier();
        enc->setComputePipelineState(mAovResolvePSO4);
        bind(uniformBuffer, 0, 0);
        bind(mAovBuffer, 0, 1);
        bind(mRadianceBuffer, 0, 2);
        table->setAddress(ring.push(sampleCount), 3);
        bind(scene.accumulationBuffer, 0, 4);
        table->setTexture(scene.guideColor->gpuResourceID(), 0);
        table->setTexture(scene.guideDepth->gpuResourceID(), 1);
        table->setTexture(scene.guideMotion->gpuResourceID(), 2);
        table->setTexture(scene.guideDiffuse->gpuResourceID(), 3);
        table->setTexture(scene.guideSpecular->gpuResourceID(), 4);
        table->setTexture(scene.guideNormal->gpuResourceID(), 5);
        table->setTexture(scene.guideRoughness->gpuResourceID(), 6);
        table->setTexture(scene.guideSpecularHitDistance->gpuResourceID(), 7);
        table->setTexture(scene.guideReactive->gpuResourceID(), 8);
        enc->dispatchThreadgroups(MTL::Size((width + 7) / 8, (height + 7) / 8, 1), MTL::Size(8, 8, 1));
    }
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

    const uint32_t pixels = width * height;
    const WavefrontVariant* variant = variantFor(features);
    const uint32_t bounceIterations = frame.bounceIterations;
    MTL::Buffer* outputBuffer = ((MetalBuffer*)output)->getNativePtr();
    const auto* uniforms = reinterpret_cast<const Uniforms*>(uniformBuffer->contents());
    const uint32_t dispatchSampleCount = sampleCount + (uniforms->canonicalGuideSample ? 1u : 0u);

    // Textures are reached through resource IDs inside the Material struct, so
    // Metal cannot infer their use from the bindings and every encoder has to be
    // told about them again.
    auto declareResidency = [&](MTL::ComputeCommandEncoder* e) {
        if (scene.textures && !scene.textures->materialTextures().empty())
        {
            e->useResources(reinterpret_cast<const MTL::Resource* const*>(scene.textures->materialTextures().data()),
                            scene.textures->materialTextures().size(), MTL::ResourceUsageRead);
        }
        if (scene.primitiveAccelerationStructures && !scene.primitiveAccelerationStructures->empty())
        {
            e->useResources(reinterpret_cast<const MTL::Resource* const*>(scene.primitiveAccelerationStructures->data()),
                            scene.primitiveAccelerationStructures->size(), MTL::ResourceUsageRead);
        }
        if (scene.instanceAccelerationStructure)
        {
            e->useResource(scene.instanceAccelerationStructure, MTL::ResourceUsageRead);
        }
        if (scene.environment && scene.environment->state().mapTexture)
        {
            e->useResource(scene.environment->state().mapTexture, MTL::ResourceUsageRead);
        }
        e->useResource(((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageWrite);
    };
    declareResidency(enc);

    const MTL::Size grid = MTL::Size(pixels, 1, 1);
    const uint32_t kThreadsPerGroup = 64;
    const MTL::Size tg = MTL::Size(kThreadsPerGroup, 1, 1);
    // Byte offset of the indirect dispatch arguments inside the control buffer.
    const NS::UInteger kDispatchArgsOffset = 2 * sizeof(uint32_t);
    const NS::UInteger kShadowArgsOffset = 8 * sizeof(uint32_t);
    const NS::UInteger kShadowCounterOffset = 6 * sizeof(uint32_t);
    const NS::UInteger kHitArgsOffset = 13 * sizeof(uint32_t);
    const NS::UInteger kHitCounterOffset = 11 * sizeof(uint32_t);
    const NS::UInteger kMissArgsOffset = 18 * sizeof(uint32_t);
    const NS::UInteger kMissCounterOffset = 16 * sizeof(uint32_t);
    // Nothing in the scene deforms -> traverse it as a static structure. Every ray
    // was otherwise paying for motion-BVH traversal it could not use.
    const bool useMotion = frame.motionBlasBuilt || !variant || variant->extendStatic == nullptr ||
                          frame.settings->getAs<uint32_t>("render/pt/staticTraversal") == 0;
    if (!variant)
    {
        return enc;
    }


    // Profiling gives each stage its own encoder, because this hardware samples
    // counters only at encoder boundaries. That costs encoder overhead, so it is
    // a measurement mode and not something to leave on.
    mStageKinds.clear();
    const bool profile = frame.profileStages && mStageTimestampBuffer != nullptr;
    auto stamp = [&](uint8_t kind) {
        if (!profile || mStageKinds.size() >= kMaxStageSamples)
        {
            return;
        }
        enc->endEncoding();
        MTL::ComputePassDescriptor* desc = MTL::ComputePassDescriptor::computePassDescriptor();
        MTL::ComputePassSampleBufferAttachmentDescriptor* att =
            desc->sampleBufferAttachments()->object(0);
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

    for (uint32_t s = 0; s < dispatchSampleCount; ++s)
    {
        MTL::Buffer* sampleRadiance =
            (uniforms->canonicalGuideSample && s == 0u) ? mGuideRadianceBuffer : mRadianceBuffer;
        stamp(kStageGenerate);
        enc->setComputePipelineState(variant->generate);
        enc->setBuffer(uniformBuffer, 0, 0);
        enc->setBuffer(mPathStateBuffer, 0, 1);
        enc->setBuffer(sampleRadiance, 0, 2);
        enc->setBuffer(mIorStackBuffer, 0, 3);
        enc->setBytes(&s, sizeof(uint32_t), 4);
        enc->setBuffer(mPathQueueBuffer[0], 0, 5);
        enc->setBuffer(mControlBuffer, 0, 6);
        enc->setBuffer(mAovBuffer, 0, 7);
        enc->setBuffer(mPathRayBuffer, 0, 8);
        enc->setBuffer(mIorStatsBuffer, 0, 9);
        enc->dispatchThreads(grid, tg);

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
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

            // Sort the queue this bounce is about to traverse. Bounce 0 is the
            // camera and already perfectly coherent, so it is skipped -- sorting
            // it is pure cost.


            stamp(kStageExtend);
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
                (bounce == 0) ? uniforms->primaryRayMask
                              : (uniforms->primaryRayMask | GEOMETRY_MASK_LIGHT_HIDDEN);
            enc->setBytes(&extendMask, sizeof(uint32_t), 14);
            enc->dispatchThreadgroups(mControlBuffer, kDispatchArgsOffset, tg);
            enc->popDebugGroup();

            enc->setComputePipelineState(mPrepareHitMissPSO);
            enc->setBuffer(mControlBuffer, 0, 0);
            enc->setBytes(&kThreadsPerGroup, sizeof(uint32_t), 1);
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

            stamp(kStageMiss);
            enc->setComputePipelineState(variant->miss);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(mPathStateBuffer, 0, 1);
            enc->setBuffer(mPathRayBuffer, 0, 2);
            enc->setBuffer(sampleRadiance, 0, 3);
            enc->setBuffer(mMissQueueBuffer, 0, 4);
            enc->setBuffer(mControlBuffer, 0, 5);
            enc->setBuffer(mAovBuffer, 0, 6);
            enc->setBytes(&s, sizeof(uint32_t), 7);
            enc->setBuffer(mIorStackBuffer, 0, 8);
            enc->setBuffer(mIorStatsBuffer, 0, 9);
            if (scene.environment && scene.environment->state().mapTexture)
            {
                enc->setTexture(scene.environment->state().mapTexture, 0);
                enc->setTexture(scene.environment->state().backgroundTexture ? scene.environment->state().backgroundTexture : scene.environment->state().mapTexture, 1);
            }
            enc->dispatchThreadgroups(mControlBuffer, kMissArgsOffset, tg);

            stamp(kStageShade);
            enc->pushDebugGroup(NS::String::string("shade", NS::UTF8StringEncoding));
            enc->setComputePipelineState(variant->shade);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(scene.instanceBuffer, 0, 1);
            enc->setBuffer(scene.iesBuffer, 0, 2);
            enc->setBuffer(scene.lightBuffer, 0, 3);
            enc->setBuffer(scene.materialBuffer, 0, 4);
            enc->setBuffer(mPathStateBuffer, 0, 5);
            enc->setBuffer(mHitBuffer, 0, 6);
            enc->setBuffer(sampleRadiance, 0, 7);
            enc->setBuffer(mIorStackBuffer, 0, 8);
            enc->setBuffer(scene.geometryEntryBuffer, 0, 9);
            enc->setBuffer(scene.environment->state().aliasBuffer, 0, 10);
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
            enc->setBuffer(scene.prevFrameVertexBuffer ? scene.prevFrameVertexBuffer : scene.vertexBuffer, 0, 23);
            enc->setBuffer(scene.prevFrameInstanceBuffer ? scene.prevFrameInstanceBuffer : scene.instanceBuffer, 0, 24);
            if (scene.environment && scene.environment->state().mapTexture)
            {
                enc->setTexture(scene.environment->state().mapTexture, 0);
            }
            if (scene.sharcBuffer)
            {
                enc->setBuffer(scene.sharcBuffer, 0, 25);
            }
            // A curve hit carries a segment index and a parameter along it, and
            // nothing else: the position, the tangent and the radius all come
            // back out of these three buffers, the same way a triangle hit is
            // refetched from the vertex buffer.
            if (scene.curvePointBuffer)
            {
                enc->setBuffer(scene.curvePointBuffer, 0, 26);
                enc->setBuffer(scene.curveSegmentBuffer, 0, 27);
            }
            enc->setBuffer(mIorStatsBuffer, 0, 28);
            enc->dispatchThreadgroups(mControlBuffer, kHitArgsOffset, tg);
            enc->popDebugGroup();

            // Deferred occlusion. It has to run before the next bounce's shade,
            // so that this bounce's direct lighting lands in the accumulator
            // ahead of the next bounce's emission -- the same order the
            // megakernel adds them in.
            stamp(kStagePrepareShadow);
            enc->setComputePipelineState(mPrepareShadowPSO);
            enc->setBuffer(mControlBuffer, 0, 0);
            enc->setBytes(&kThreadsPerGroup, sizeof(uint32_t), 1);
            enc->setBytes(&bounce, sizeof(uint32_t), 2);
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

            stamp(kStageShadow);
            enc->setComputePipelineState(useMotion ? variant->shadowMotion : variant->shadowStatic);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setAccelerationStructure(scene.instanceAccelerationStructure, 1);
            enc->setBuffer(mShadowRayBuffer, 0, 2);
            enc->setBuffer(sampleRadiance, 0, 3);
            enc->setBuffer(mControlBuffer, 0, 4);
            enc->setBytes(&s, sizeof(uint32_t), 5);
            // Cutout shadows need to resolve the material and its uv at each hit.
            enc->setBuffer(scene.instanceBuffer, 0, 6);
            enc->setBuffer(scene.materialBuffer, 0, 7);
            enc->setBuffer(scene.geometryEntryBuffer, 0, 8);
            enc->setBuffer(scene.vertexBuffer, 0, 9);
            enc->setBuffer(scene.indexBuffer, 0, 10);
            // The intersection function reads the same tables the kernel does,
            // through the function table's own binding points.
            MTL::IntersectionFunctionTable* shadowTable =
                useMotion ? variant->shadowTableMotion : variant->shadowTableStatic;
            if (shadowTable)
            {
                shadowTable->setBuffer(scene.instanceBuffer, 0, 0);
                shadowTable->setBuffer(scene.materialBuffer, 0, 1);
                shadowTable->setBuffer(scene.geometryEntryBuffer, 0, 2);
                shadowTable->setBuffer(scene.vertexBuffer, 0, 3);
                shadowTable->setBuffer(scene.indexBuffer, 0, 4);
                enc->setIntersectionFunctionTable(shadowTable, 11);
                enc->useResource(shadowTable, MTL::ResourceUsageRead);
            }
            enc->dispatchThreadgroups(mControlBuffer, kShadowArgsOffset, tg);

        }

        // Every path that passed through a cache voxel deposits what it gathered
        // after it -- here, at the end of the sample, because by now the sample's
        // deferred shadow rays have landed in the accumulator too.
        if (scene.sharcBuffer && variant->sharcDeposit)
        {
            enc->setComputePipelineState(variant->sharcDeposit);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(mPathStateBuffer, 0, 1);
            enc->setBuffer(sampleRadiance, 0, 2);
            enc->setBuffer(scene.sharcBuffer, 0, 3);
            const MTL::Size depositTg = MTL::Size::Make(
                std::min<NS::UInteger>(variant->sharcDeposit->maxTotalThreadsPerThreadgroup(), 256u),
                1, 1);
            enc->dispatchThreads(MTL::Size::Make(static_cast<NS::UInteger>(width) * height, 1, 1), depositTg);
        }
    }

    // Fold the accumulated radiance into the output exactly as the megakernel's
    // tail does, reusing the resolve kernel in wavefront.metal.
    stamp(kStageResolve);
    enc->setComputePipelineState(mResolvePSO);
    enc->setBuffer(uniformBuffer, 0, 0);
    enc->setBuffer(mRadianceBuffer, 0, 1);
    enc->setBuffer(outputBuffer, 0, 2);
    enc->setBuffer(scene.accumulationBuffer, 0, 3);
    enc->setBytes(&sampleCount, sizeof(uint32_t), 4);
    enc->setBuffer(mAovBuffer, 0, 5);
    enc->dispatchThreads(grid, tg);

    if (profile && mStageStatsBuffer)
    {
        enc->endEncoding();
        MTL::BlitCommandEncoder* blit = pCmd->blitCommandEncoder();
        blit->copyFromBuffer(mControlBuffer, 0, mStageStatsBuffer, 0, mStageStatsBuffer->length());
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
    auto entry = [&](const char* base) -> std::string {
        return curves ? std::string(base) + "Curve" : std::string(base);
    };
    NS::Error* err = nullptr;
    auto make = [&](const char* name) -> MTL::ComputePipelineState* {
        if (useMetal4)
        {
            return mMetal4->newComputePipelineState(mLibrary, name, values);
        }
        MTL::Function* fn = mLibrary->newFunction(
            NS::String::string(name, NS::UTF8StringEncoding), values, &err);
        if (!fn)
        {
            STRELKA_FATAL("wavefront: specialising {} -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
            return nullptr;
        }
        MTL::ComputePipelineState* pso = mDevice->newComputePipelineState(fn, &err);
        if (!pso)
        {
            STRELKA_FATAL("wavefront: {} -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
        }
        fn->release();
        return pso;
    };

    // The alpha-shadow intersection function has to be linked into the pipelines
    // that call it, and the pipeline then hands out a table to bind it through.
    MTL::Function* anyHitFn = nullptr;
    MTL::LinkedFunctions* linked = nullptr;
    // The table's tags must match the intersector's, so a curve-capable shadow
    // pipeline links a curve-tagged copy of the same test. Same body; the tag
    // list is the whole difference.
    const std::string anyHitName = entry("shadowAlphaAnyHit");
    if (alpha && !useMetal4)
    {
        anyHitFn = mLibrary->newFunction(
            NS::String::string(anyHitName.c_str(), NS::UTF8StringEncoding), values, &err);
        if (anyHitFn)
        {
            const NS::Object* fns[] = { anyHitFn };
            linked = MTL::LinkedFunctions::alloc()->init();
            linked->setFunctions(NS::Array::array(fns, 1));
        }
        else
        {
            STRELKA_ERROR("wavefront: specialising {} -> {}", anyHitName,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
        }
    }
    auto makeLinked = [&](const char* name) -> MTL::ComputePipelineState* {
        if (useMetal4)
        {
            // Metal 4 states linking through descriptors, so it never builds the
            // MTL::Function above and cannot go through the branch below.
            return alpha ? mMetal4->newComputePipelineStateLinked(mLibrary, name,
                                                                 anyHitName.c_str(), values)
                         : make(name);
        }
        if (!linked)
        {
            return make(name);
        }
        MTL::Function* fn = mLibrary->newFunction(
            NS::String::string(name, NS::UTF8StringEncoding), values, &err);
        if (!fn)
        {
            STRELKA_FATAL("wavefront: specialising {} -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
            return nullptr;
        }
        MTL::ComputePipelineDescriptor* desc = MTL::ComputePipelineDescriptor::alloc()->init();
        desc->setComputeFunction(fn);
        desc->setLinkedFunctions(linked);
        MTL::ComputePipelineState* pso =
            mDevice->newComputePipelineState(desc, MTL::PipelineOptionNone, nullptr, &err);
        if (!pso)
        {
            STRELKA_FATAL("wavefront: {} with linked functions -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
        }
        desc->release();
        fn->release();
        return pso;
    };

    WavefrontVariant v;
    v.generate = make("wavefrontGenerate");
    v.extendMotion = make(entry("wavefrontExtend").c_str());
    v.extendStatic = make(entry("wavefrontExtendStatic").c_str());
    v.shade = make("wavefrontShade");
    v.miss = make("wavefrontMiss");
    v.sharcDeposit = make("wavefrontSharcDeposit");
    v.shadowMotion = makeLinked(entry("wavefrontShadow").c_str());
    v.shadowStatic = makeLinked(entry("wavefrontShadowStatic").c_str());

    // One table per pipeline: it is created from the pipeline that will bind it,
    // and the two shadow pipelines are different pipelines.
    if ((linked && anyHitFn) || (useMetal4 && alpha))
    {
        auto makeTable = [&](MTL::ComputePipelineState* pso) -> MTL::IntersectionFunctionTable* {
            if (!pso)
                return nullptr;
            MTL::IntersectionFunctionTableDescriptor* d =
                MTL::IntersectionFunctionTableDescriptor::alloc()->init();
            d->setFunctionCount(1);
            MTL::IntersectionFunctionTable* table = pso->newIntersectionFunctionTable(d);
            d->release();
            if (!table)
                return nullptr;
            // By name on the Metal 4 path: linking there is stated with
            // descriptors, so there is no MTL::Function to ask for a handle.
            MTL::FunctionHandle* handle =
                anyHitFn ? pso->functionHandle(anyHitFn)
                         : pso->functionHandle(
                               NS::String::string(anyHitName.c_str(), NS::UTF8StringEncoding));
            if (!handle)
            {
                STRELKA_ERROR("wavefront: no function handle for {}", anyHitName);
                table->release();
                return nullptr;
            }
            table->setFunction(handle, 0);
            return table;
        };
        v.shadowTableMotion = makeTable(v.shadowMotion);
        v.shadowTableStatic = makeTable(v.shadowStatic);
        mResidencyDirty = true;
    }
    if (anyHitFn)
        anyHitFn->release();
    if (linked)
        linked->release();
    values->release();

    if (!v.shade)
    {
        return nullptr;
    }
    STRELKA_INFO(
        "wavefront variant env={} lights={} motion={} dof={} debug={} alpha={} fog={} sss={} sharc={} "
        "curves={} metal4={}",
        envMap, lights, motionBlur, dof, debug, alpha, fog, subsurface, sharc, curves, useMetal4);
    // Every pipeline's threadgroup limit, not just two of them.
    //
    // This is the only figure the public API gives on register pressure -- the
    // driver's own answer to how many threads fit -- and against the 1024 a
    // register-light kernel reaches it reads directly: 640 is roughly 1.6x the
    // registers per thread, 384 is 2.7x. There is no breakdown of what they
    // hold anywhere in Metal; the way to find that is to remove something and
    // watch this number, and having all of them at once makes each rebuild
    // answer for the whole renderer rather than for one kernel.
    auto tgLimit = [](MTL::ComputePipelineState* p) -> uint32_t {
        return p ? (uint32_t)p->maxTotalThreadsPerThreadgroup() : 0u;
    };
    STRELKA_INFO("  maxThreadsPerTG: generate {} extend {} (motion {}) shade {} shadow {} (motion {}) "
                 "miss {} sharcDeposit {}",
                 tgLimit(v.generate), tgLimit(v.extendStatic), tgLimit(v.extendMotion),
                 tgLimit(v.shade), tgLimit(v.shadowStatic), tgLimit(v.shadowMotion),
                 tgLimit(v.miss), tgLimit(v.sharcDeposit));
    return &mVariants.emplace(features, v).first->second;
}

void MetalWavefrontIntegrator::buildPipelines()
{
    const std::string path = oka::resolveResourcePath("metal/shaders/wavefront.metallib");
    NS::Error* loadErr = nullptr;
    MTL::Library* lib =
        mDevice->newLibrary(NS::String::string(path.c_str(), NS::UTF8StringEncoding), &loadErr);
    if (!lib)
    {
        STRELKA_FATAL("Failed to load {}: {}", path,
                      loadErr ? loadErr->localizedDescription()->utf8String() : "unknown error");
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
            STRELKA_FATAL("wavefront: {} -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
        }
        fn->release();
        return pso;
    };
    mResolvePSO = make("wavefrontResolve");
    mPreparePSO = make("wavefrontPrepare");
    mPrepareShadowPSO = make("wavefrontPrepareShadow");
    mPrepareHitMissPSO = make("wavefrontPrepareHitMiss");
    mAovResolvePSO = make("wavefrontAovResolve");
    if (mMetal4 && mMetal4->isValid())
    {
        // The same four stages again, built by the other compiler: a pipeline is
        // tied to the binding model it was compiled for.
        mResolvePSO4 = mMetal4->newComputePipelineState(lib, "wavefrontResolve", nullptr);
        mPreparePSO4 = mMetal4->newComputePipelineState(lib, "wavefrontPrepare", nullptr);
        mPrepareShadowPSO4 = mMetal4->newComputePipelineState(lib, "wavefrontPrepareShadow", nullptr);
        mPrepareHitMissPSO4 = mMetal4->newComputePipelineState(lib, "wavefrontPrepareHitMiss", nullptr);
        // The guide resolve too: without it the Metal 4 path cannot feed either
        // MetalFX mode, both of which read depth and motion from these textures.
        mAovResolvePSO4 = mMetal4->newComputePipelineState(lib, "wavefrontAovResolve", nullptr);
    }
    lib->release();
}

void MetalWavefrontIntegrator::ensureBuffers(uint32_t width, uint32_t height)
{
    const uint32_t pixels = width * height;
    if (pixels == mCapacity && mPathStateBuffer)
    {
        return;
    }
    auto release = [](MTL::Buffer*& b) { if (b) { b->release(); b = nullptr; } };
    release(mPathStateBuffer);
    release(mPathRayBuffer);
    release(mHitBuffer);
    release(mIorStackBuffer);
    release(mRadianceBuffer);
    release(mGuideRadianceBuffer);
    release(mPathQueueBuffer[0]);
    release(mPathQueueBuffer[1]);
    release(mControlBuffer);
    release(mShadowRayBuffer);
    release(mStageStatsBuffer);
    release(mHitQueueBuffer);
    release(mAovBuffer);
    release(mMissQueueBuffer);

    metal::WavefrontElementSizes sz;
    sz.pathState = sizeof(PathState);
    sz.pathRay = sizeof(PathRay);
    sz.hitRecord = sizeof(HitRecord);
    sz.iorStack = sizeof(IorStack);
    sz.radiance = sizeof(simd::float4);
    sz.shadowRay = sizeof(ShadowRay);
    sz.aovSample = sizeof(AovSample);
    const metal::WavefrontBufferLayout layout = metal::wavefrontBufferLayout(width, height, sz);

    // Private storage: these never leave the GPU.
    mPathStateBuffer = mDevice->newBuffer(layout.pathStateBytes, MTL::ResourceStorageModePrivate);
    mPathRayBuffer = mDevice->newBuffer(layout.pathRayBytes, MTL::ResourceStorageModePrivate);
    mHitBuffer = mDevice->newBuffer(layout.hitBytes, MTL::ResourceStorageModePrivate);
    mIorStackBuffer = mDevice->newBuffer(layout.iorStackBytes, MTL::ResourceStorageModePrivate);
    mRadianceBuffer = mDevice->newBuffer(layout.radianceBytes, MTL::ResourceStorageModePrivate);
    mGuideRadianceBuffer = mDevice->newBuffer(layout.guideRadianceBytes, MTL::ResourceStorageModePrivate);
    mPathQueueBuffer[0] = mDevice->newBuffer(layout.pathQueueBytes, MTL::ResourceStorageModePrivate);
    mPathQueueBuffer[1] = mDevice->newBuffer(layout.pathQueueBytes, MTL::ResourceStorageModePrivate);
    // Queue counters, active counts, and two sets of indirect dispatch arguments.
    mControlBuffer = mDevice->newBuffer(layout.controlBytes, MTL::ResourceStorageModePrivate);
    // At most one deferred connection per path per bounce.
    mShadowRayBuffer = mDevice->newBuffer(layout.shadowRayBytes, MTL::ResourceStorageModePrivate);
    mStageStatsBuffer = mDevice->newBuffer(layout.stageStatsBytes, MTL::ResourceStorageModeShared);
    // Shared, so it needs no blit to read: the Metal 4 path encodes none, and
    // two words are not worth an encoder either way.
    if (!mIorStatsBuffer)
    {
        mIorStatsBuffer =
            mDevice->newBuffer(IOR_STAT_COUNT * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        memset(mIorStatsBuffer->contents(), 0, mIorStatsBuffer->length());
    }
    mAovBuffer = mDevice->newBuffer(layout.aovBytes, MTL::ResourceStorageModePrivate);
    mHitQueueBuffer = mDevice->newBuffer(layout.hitQueueBytes, MTL::ResourceStorageModePrivate);
    mMissQueueBuffer = mDevice->newBuffer(layout.missQueueBytes, MTL::ResourceStorageModePrivate);

    mCapacity = pixels;

    STRELKA_INFO("wavefront buffers for {}x{}: {:.1f} MB total", width, height,
                 (pixels * (sizeof(PathState) + sizeof(HitRecord) + sizeof(IorStack) + sizeof(simd::float4)))
                     / (1024.0 * 1024.0));
}
