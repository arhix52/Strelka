#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace oka
{
namespace metal
{

enum class WavefrontChunkPhase : uint8_t
{
    Complete,
    Extend,
    Finish
};

struct WavefrontChunk
{
    uint32_t sampleIndex = 0;
    uint32_t bounceBegin = 0;
    uint32_t bounceEnd = 0;
    bool generate = false;
    bool resolve = false;
    WavefrontChunkPhase phase = WavefrontChunkPhase::Complete;
    uint32_t traversalBatchBegin = 0;
    uint32_t traversalBatchEnd = 0;

    bool operator==(const WavefrontChunk&) const = default;
};

struct WavefrontChunkGroup
{
    size_t begin = 0;
    size_t end = 0;
};

inline uint32_t wavefrontChunkIterations(uint32_t width, uint32_t height)
{
    // Keep each command buffer near ten million pixel-iterations. Curve traversal
    // can be much more expensive than triangle traversal, while late wavefront
    // queues usually shrink enough that the estimate overstates their work.
    constexpr uint64_t kTargetPixelIterations = 10ull * 1024ull * 1024ull;
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (pixels == 0)
    {
        return 1;
    }
    return std::clamp(static_cast<uint32_t>(kTargetPixelIterations / pixels), 1u, 16u);
}

inline std::vector<WavefrontChunk> makeWavefrontChunkPlan(uint32_t sampleCount,
                                                          uint32_t bounceIterations,
                                                          uint32_t iterationsPerChunk)
{
    std::vector<WavefrontChunk> chunks;
    if (sampleCount == 0)
    {
        return chunks;
    }

    const uint32_t chunkSize = std::max(iterationsPerChunk, 1u);
    for (uint32_t sample = 0; sample < sampleCount; ++sample)
    {
        if (bounceIterations == 0)
        {
            chunks.push_back({ sample, 0, 0, true, false });
            continue;
        }
        uint32_t begin = 0;
        while (begin < bounceIterations)
        {
            const bool longPath = bounceIterations > 16;
            const bool earlyPath = longPath && begin < 16;
            const uint32_t currentChunkSize = earlyPath ? 1u : (longPath ? std::max(chunkSize, 16u) : chunkSize);
            const uint32_t phaseEnd = earlyPath ? 16u : bounceIterations;
            const uint32_t end = begin + std::min(currentChunkSize, std::min(phaseEnd - begin, bounceIterations - begin));
            chunks.push_back({ sample, begin, end, begin == 0, false });
            begin = end;
        }
    }
    chunks.back().resolve = true;
    return chunks;
}

inline std::vector<WavefrontChunkGroup> makeWavefrontChunkGroups(const std::vector<WavefrontChunk>& chunks)
{
    std::vector<WavefrontChunkGroup> groups;
    groups.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i)
    {
        // A Metal 4 commit of several command buffers is one scheduler workload.
        // Grouping the four cheap-looking SSS tail buffers back together undid
        // the watchdog protection provided by chunking: their combined GPU
        // interval reached 120--150 ms and the device killed whichever tiny
        // dispatch happened to be current. Keep the submission boundary aligned
        // with the command-buffer boundary.
        groups.push_back({ i, i + 1 });
    }
    return groups;
}

// Turn the throughput-oriented bounce plan into watchdog-sized Metal 4
// scheduler workloads. At high resolutions even one curve extend can run long
// enough to be killed, despite being made of several indirect dispatches: a
// dispatch boundary is not a command-buffer retirement boundary.
//
// Early bounces are isolated for every geometry type. Curve traversal is split
// further, before the hit queues are consumed, so each Extend chunk appends a
// bounded range of the same queue and Finish performs miss/shade/shadow once all
// of those ranges have completed.
inline std::vector<WavefrontChunk> makeMetal4WavefrontChunkPlan(const std::vector<WavefrontChunk>& chunks,
                                                                uint32_t traversalBatchCount,
                                                                uint32_t maxTraversalBatchesPerChunk,
                                                                uint32_t isolatedBounces,
                                                                bool splitTraversal)
{
    const uint32_t batchLimit = std::max(maxTraversalBatchesPerChunk, 1u);
    if (traversalBatchCount <= batchLimit || isolatedBounces == 0u)
    {
        return chunks;
    }

    std::vector<WavefrontChunk> result;
    auto append = [&](const WavefrontChunk& chunk, bool allowTraversalSplit) {
        const bool oneBounce = chunk.bounceEnd == chunk.bounceBegin + 1u;
        if (!splitTraversal || !allowTraversalSplit || !oneBounce)
        {
            result.push_back(chunk);
            return;
        }

        bool first = true;
        for (uint32_t batch = 0; batch < traversalBatchCount; batch += batchLimit)
        {
            WavefrontChunk extend = chunk;
            extend.generate = chunk.generate && first;
            extend.resolve = false;
            extend.phase = WavefrontChunkPhase::Extend;
            extend.traversalBatchBegin = batch;
            extend.traversalBatchEnd = std::min(batch + batchLimit, traversalBatchCount);
            result.push_back(extend);
            first = false;
        }

        WavefrontChunk finish = chunk;
        finish.generate = false;
        finish.phase = WavefrontChunkPhase::Finish;
        result.push_back(finish);
    };

    for (const WavefrontChunk& chunk : chunks)
    {
        if (chunk.bounceBegin == chunk.bounceEnd)
        {
            append(chunk, false);
            continue;
        }

        uint32_t begin = chunk.bounceBegin;
        bool first = true;
        while (begin < chunk.bounceEnd)
        {
            const bool isolate = begin < isolatedBounces;
            const uint32_t end = isolate ? begin + 1u : chunk.bounceEnd;
            WavefrontChunk part = chunk;
            part.bounceBegin = begin;
            part.bounceEnd = end;
            part.generate = chunk.generate && first;
            part.resolve = chunk.resolve && end == chunk.bounceEnd;
            append(part, isolate);
            first = false;
            begin = end;
        }
    }
    return result;
}

inline const char* wavefrontChunkPhaseName(WavefrontChunkPhase phase)
{
    switch (phase)
    {
    case WavefrontChunkPhase::Complete:
        return "complete";
    case WavefrontChunkPhase::Extend:
        return "extend";
    case WavefrontChunkPhase::Finish:
        return "finish";
    }
    return "unknown";
}

} // namespace metal
} // namespace oka
