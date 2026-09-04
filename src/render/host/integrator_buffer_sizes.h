#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>


namespace oka::metal
{

// Byte sizes of one wavefront pixel's worth of path buffers. Callers pass
// sizeof(PathState) etc. from ShaderTypes.h so this header stays Metal-free;
// MetalRender static_asserts the production sizes match.
struct WavefrontElementSizes
{
    size_t pathState = 0;
    size_t mediumPathState = 0;
    size_t sharcUpdateState = 0;
    size_t pathRay = 0;
    size_t hitRecord = 0;
    size_t iorStack = 0;
    size_t radiance = 0; // float4
    size_t guideRay = 0;
    size_t shadowRay = 0;
    size_t aovSample = 0;
    size_t restirReservoir = 0;
    size_t restirSurfaceHistory = 0;
};

struct WavefrontBufferLayout
{
    size_t pathStateBytes = 0;
    size_t mediumPathStateBytes = 0;
    size_t sharcUpdateStateBytes = 0;
    size_t pathRayBytes = 0;
    size_t hitBytes = 0;
    size_t iorStackBytes = 0;
    size_t radianceBytes = 0;
    size_t guideRayBytes = 0;
    size_t pathQueueBytes = 0; // one ping-pong queue
    size_t controlBytes = 0;
    size_t traversalDispatchBytes = 0;
    size_t shadowRayBytes = 0;
    size_t stageStatsBytes = 0;
    size_t aovBytes = 0;
    size_t hitQueueBytes = 0;
    size_t missQueueBytes = 0;
    size_t restirReservoirBytes = 0; // one of two history buffers
    size_t restirSurfaceHistoryBytes = 0; // one of two history buffers
    uint32_t pixels = 0;
    uint32_t sharcUpdatePaths = 0;
};

inline constexpr uint32_t kWavefrontControlUints = 96;
// A large hardware traversal dispatch can be non-preemptible long enough to
// trip the Metal watchdog. Prepare writes one indirect argument triplet per
// batch, so inactive tail batches remain true zero-work dispatches.
inline constexpr uint32_t kWavefrontTraversalBatchThreads = 256 * 1024;
// Triangle traversal is preemptible enough on Apple silicon to amortise the
// launch with a larger batch. Curves retain the conservative size above: those
// were the workloads that originally hit the watchdog.
inline constexpr uint32_t kWavefrontTriangleTraversalBatchThreads = 512 * 1024;
// Diagnostic subdivision may reduce a traversal dispatch to this size. Keeping
// the indirect-argument buffer large enough costs less than 7 KB at 1080p and
// lets a reproducer narrow a hang without reallocating the wavefront queues.
inline constexpr uint32_t kWavefrontMinDiagnosticTraversalBatchThreads = 4 * 1024;
// Curve traversal needs a lower non-preemptible hardware-dispatch ceiling than
// triangle traversal. 256K curve dispatches repeatedly hung on kids_room's
// 7.5M-segment AS; 128K dispatches completed 280 consecutive full frames. Two
// dispatches per scheduler group retain launch efficiency while keeping a 2x
// safety margin at the hardware boundary.
inline constexpr uint32_t kWavefrontCurveTraversalBatchThreads = 128 * 1024;
inline constexpr uint32_t kWavefrontCurveTraversalBatchesPerGroup = 2;
// A Metal 4 command buffer may contain this many traversal dispatches before it
// is retired. Four batches cap one curve-extend scheduler workload at roughly a
// megapath; the remaining batches append into the same hit/miss queues from the
// following command buffer.
inline constexpr uint32_t kWavefrontTraversalBatchesPerCommandBuffer = 4;

inline constexpr uint32_t wavefrontTraversalBatchCount(uint32_t pixels,
                                                       uint32_t batchThreads = kWavefrontTraversalBatchThreads)
{
    return std::max(1u, (pixels + batchThreads - 1u) / batchThreads);
}
// Metal 4 fault diagnosis writes the stage it is about to enter here. Keep it
// outside the Metal 3 control-buffer snapshot at the start of stageStats.
inline constexpr uint32_t kWavefrontStageBreadcrumbOffset = kWavefrontControlUints;
inline constexpr uint32_t kWavefrontStageDiagnosticBase = kWavefrontStageBreadcrumbOffset + 1;
inline constexpr uint32_t kWavefrontStageDiagnosticBounces = 96;
inline constexpr uint32_t kWavefrontStageDiagnosticLanes = 4;
inline constexpr uint32_t kWavefrontStageDiagnosticLaneUints = 11;
inline constexpr uint32_t kWavefrontStageDiagnosticStride =
    1 + kWavefrontStageDiagnosticLanes * kWavefrontStageDiagnosticLaneUints;
inline constexpr uint32_t kWavefrontStageStatsUints =
    kWavefrontStageDiagnosticBase + kWavefrontStageDiagnosticBounces * kWavefrontStageDiagnosticStride;

inline WavefrontBufferLayout wavefrontBufferLayout(uint32_t width,
                                                   uint32_t height,
                                                   const WavefrontElementSizes& sz,
                                                   uint32_t sharcUpdateDownscale = 1u)
{
    const uint32_t pixels = width * height;
    WavefrontBufferLayout layout;
    layout.pixels = pixels;
    layout.pathStateBytes = (size_t)pixels * sz.pathState;
    layout.mediumPathStateBytes = (size_t)pixels * sz.mediumPathState;
    const uint32_t updateScale = std::max(sharcUpdateDownscale, 1u);
    layout.sharcUpdatePaths = ((width + updateScale - 1u) / updateScale) * ((height + updateScale - 1u) / updateScale);
    layout.sharcUpdateStateBytes = (size_t)layout.sharcUpdatePaths * sz.sharcUpdateState;
    layout.pathRayBytes = (size_t)pixels * sz.pathRay;
    layout.hitBytes = (size_t)pixels * sz.hitRecord;
    layout.iorStackBytes = (size_t)pixels * sz.iorStack;
    layout.radianceBytes = (size_t)pixels * sz.radiance;
    layout.guideRayBytes = (size_t)pixels * sz.guideRay;
    layout.pathQueueBytes = (size_t)pixels * sizeof(uint32_t);
    layout.controlBytes = (size_t)kWavefrontControlUints * sizeof(uint32_t);
    layout.traversalDispatchBytes =
        (size_t)wavefrontTraversalBatchCount(pixels, kWavefrontMinDiagnosticTraversalBatchThreads) * 3 * sizeof(uint32_t);
    layout.shadowRayBytes = (size_t)pixels * sz.shadowRay;
    // The first control-sized block is the Metal 3 profiling snapshot. Fault
    // breadcrumbs follow it so that copying the snapshot cannot overwrite them.
    layout.stageStatsBytes = (size_t)kWavefrontStageStatsUints * sizeof(uint32_t);
    layout.aovBytes = (size_t)pixels * sz.aovSample;
    layout.hitQueueBytes = (size_t)pixels * sizeof(uint32_t);
    layout.missQueueBytes = (size_t)pixels * sizeof(uint32_t);
    layout.restirReservoirBytes = (size_t)pixels * sz.restirReservoir;
    layout.restirSurfaceHistoryBytes = (size_t)pixels * sz.restirSurfaceHistory;
    return layout;
}

} // namespace oka::metal
