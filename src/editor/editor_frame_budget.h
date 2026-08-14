#pragma once

#include "../render/metal/render_resolution.h"

#include <array>
#include <cstdint>

namespace oka
{
namespace editor_frame_budget
{

inline constexpr double kInteractiveBudgetMs = 500.0;

struct RenderSettingsSnapshot
{
    uint32_t outputWidth = 1;
    uint32_t outputHeight = 1;
    bool enableUpscale = false;
    float upscaleFactor = 1.0f;
};

struct FrameSample
{
    double gpuTimeMs = 0.0;
    uint64_t pathTracePixels = 0;
};

struct Assessment
{
    double predictedGpuTimeMs = 0.0;
    uint32_t pathTraceWidth = 0;
    uint32_t pathTraceHeight = 0;
    bool hasPrediction = false;
    bool exceedsBudget = false;
};

inline uint64_t pathTracePixels(const RenderSettingsSnapshot& settings)
{
    const render_resolution::Resolution resolution =
        render_resolution::resolve(settings.outputWidth, settings.outputHeight, settings.enableUpscale,
                                   settings.upscaleFactor);
    return static_cast<uint64_t>(resolution.pathTraceWidth) * resolution.pathTraceHeight;
}

inline FrameSample sampleFrom(double gpuTimeMs, const RenderSettingsSnapshot& settings)
{
    return { gpuTimeMs, pathTracePixels(settings) };
}

inline double predictGpuTimeMs(const FrameSample& sample, uint64_t proposedPixels)
{
    if (sample.gpuTimeMs <= 0.0 || sample.pathTracePixels == 0 || proposedPixels == 0)
    {
        return 0.0;
    }
    return sample.gpuTimeMs * static_cast<double>(proposedPixels) / static_cast<double>(sample.pathTracePixels);
}

inline Assessment assess(const FrameSample& sample,
                         const RenderSettingsSnapshot& proposed,
                         double budgetMs = kInteractiveBudgetMs)
{
    Assessment result;
    const render_resolution::Resolution resolution =
        render_resolution::resolve(proposed.outputWidth, proposed.outputHeight, proposed.enableUpscale,
                                   proposed.upscaleFactor);
    result.pathTraceWidth = resolution.pathTraceWidth;
    result.pathTraceHeight = resolution.pathTraceHeight;
    result.predictedGpuTimeMs =
        predictGpuTimeMs(sample, static_cast<uint64_t>(resolution.pathTraceWidth) * resolution.pathTraceHeight);
    result.hasPrediction = result.predictedGpuTimeMs > 0.0;
    result.exceedsBudget = result.hasPrediction && result.predictedGpuTimeMs > budgetMs;
    return result;
}

inline float recommendedScale(const FrameSample& sample,
                              uint32_t outputWidth,
                              uint32_t outputHeight,
                              double budgetMs = kInteractiveBudgetMs)
{
    constexpr std::array<float, 3> kScales = { 0.75f, 0.5f, 0.25f };
    for (const float scale : kScales)
    {
        const RenderSettingsSnapshot proposed{ outputWidth, outputHeight, true, scale };
        if (!assess(sample, proposed, budgetMs).exceedsBudget)
        {
            return scale;
        }
    }
    return kScales.back();
}

} // namespace editor_frame_budget
} // namespace oka
