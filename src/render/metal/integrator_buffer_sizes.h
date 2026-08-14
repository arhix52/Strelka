#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace oka
{
namespace metal
{

// Byte sizes of one wavefront pixel's worth of path buffers. Callers pass
// sizeof(PathState) etc. from ShaderTypes.h so this header stays Metal-free;
// MetalRender static_asserts the production sizes match.
struct WavefrontElementSizes
{
    size_t pathState = 0;
    size_t pathRay = 0;
    size_t hitRecord = 0;
    size_t iorStack = 0;
    size_t radiance = 0; // float4
    size_t shadowRay = 0;
    size_t aovSample = 0;
};

struct WavefrontBufferLayout
{
    size_t pathStateBytes = 0;
    size_t pathRayBytes = 0;
    size_t hitBytes = 0;
    size_t iorStackBytes = 0;
    size_t radianceBytes = 0;
    size_t guideRadianceBytes = 0;
    size_t pathQueueBytes = 0; // one ping-pong queue
    size_t controlBytes = 0;
    size_t traversalDispatchBytes = 0;
    size_t shadowRayBytes = 0;
    size_t stageStatsBytes = 0;
    size_t aovBytes = 0;
    size_t hitQueueBytes = 0;
    size_t missQueueBytes = 0;
    uint32_t pixels = 0;
};

inline constexpr uint32_t kWavefrontControlUints = 96;
// A large hardware traversal dispatch can be non-preemptible long enough to
// trip the Metal watchdog. Prepare writes one indirect argument triplet per
// batch, so inactive tail batches remain true zero-work dispatches.
inline constexpr uint32_t kWavefrontTraversalBatchThreads = 256 * 1024;
// A Metal 4 command buffer may contain this many traversal dispatches before it
// is retired. Four batches cap one curve-extend scheduler workload at roughly a
// megapath; the remaining batches append into the same hit/miss queues from the
// following command buffer.
inline constexpr uint32_t kWavefrontTraversalBatchesPerCommandBuffer = 4;

inline constexpr uint32_t wavefrontTraversalBatchCount(uint32_t pixels)
{
    return std::max(1u, (pixels + kWavefrontTraversalBatchThreads - 1u) /
                            kWavefrontTraversalBatchThreads);
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

inline WavefrontBufferLayout wavefrontBufferLayout(uint32_t width, uint32_t height, const WavefrontElementSizes& sz)
{
    const uint32_t pixels = width * height;
    WavefrontBufferLayout layout;
    layout.pixels = pixels;
    layout.pathStateBytes = (size_t)pixels * sz.pathState;
    layout.pathRayBytes = (size_t)pixels * sz.pathRay;
    layout.hitBytes = (size_t)pixels * sz.hitRecord;
    layout.iorStackBytes = (size_t)pixels * sz.iorStack;
    layout.radianceBytes = (size_t)pixels * sz.radiance;
    layout.guideRadianceBytes = (size_t)pixels * sz.radiance;
    layout.pathQueueBytes = (size_t)pixels * sizeof(uint32_t);
    layout.controlBytes = (size_t)kWavefrontControlUints * sizeof(uint32_t);
    layout.traversalDispatchBytes =
        (size_t)wavefrontTraversalBatchCount(pixels) * 3 * sizeof(uint32_t);
    layout.shadowRayBytes = (size_t)pixels * sz.shadowRay;
    // The first control-sized block is the Metal 3 profiling snapshot. Fault
    // breadcrumbs follow it so that copying the snapshot cannot overwrite them.
    layout.stageStatsBytes = (size_t)kWavefrontStageStatsUints * sizeof(uint32_t);
    layout.aovBytes = (size_t)pixels * sz.aovSample;
    layout.hitQueueBytes = (size_t)pixels * sizeof(uint32_t);
    layout.missQueueBytes = (size_t)pixels * sizeof(uint32_t);
    return layout;
}

} // namespace metal
} // namespace oka
