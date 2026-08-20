#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>


namespace oka::metal
{

struct WavefrontStageFailure
{
    int32_t lastCompletedStage = -1;
    int32_t suspectedStage = -1;
    bool postIntegrator = false;
};

inline constexpr uint32_t kWavefrontStageNotStarted = std::numeric_limits<uint32_t>::max();

// The GPU writes a stage index immediately before entering that stage. A value
// equal to stageCount is the closing breadcrumb written after the integrator.
inline WavefrontStageFailure inferWavefrontStageFailure(uint32_t enteredStage, size_t stageCount)
{
    WavefrontStageFailure result;
    if (stageCount == 0)
    {
        result.postIntegrator = enteredStage != kWavefrontStageNotStarted;
        return result;
    }
    if (enteredStage == kWavefrontStageNotStarted)
    {
        result.suspectedStage = 0;
        return result;
    }
    if (enteredStage >= stageCount)
    {
        result.lastCompletedStage = static_cast<int32_t>(stageCount - 1);
        result.postIntegrator = true;
    }
    else
    {
        result.lastCompletedStage = static_cast<int32_t>(enteredStage) - 1;
        result.suspectedStage = static_cast<int32_t>(enteredStage);
    }
    return result;
}

} // namespace oka::metal

