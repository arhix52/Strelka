#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace oka
{
namespace metal
{

struct WavefrontStageFailure
{
    size_t validMarks = 0;
    int32_t lastCompletedStage = -1;
    int32_t suspectedStage = -1;
    bool postIntegrator = false;
};

inline WavefrontStageFailure inferWavefrontStageFailure(const std::vector<uint64_t>& timestamps,
                                                        size_t stageCount)
{
    WavefrontStageFailure result;
    uint64_t previous = 0;
    while (result.validMarks < timestamps.size())
    {
        const uint64_t timestamp = timestamps[result.validMarks];
        if (timestamp == 0 || (result.validMarks > 0 && timestamp < previous))
        {
            break;
        }
        previous = timestamp;
        ++result.validMarks;
    }

    if (result.validMarks >= 2)
    {
        result.lastCompletedStage = static_cast<int32_t>(result.validMarks - 2);
    }
    if (result.validMarks > stageCount)
    {
        result.postIntegrator = true;
    }
    else if (result.validMarks > 0)
    {
        result.suspectedStage = static_cast<int32_t>(result.validMarks - 1);
    }
    return result;
}

} // namespace metal
} // namespace oka
