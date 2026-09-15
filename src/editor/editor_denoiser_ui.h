#pragma once

#include <strelka/render/render.h>

#include <host/render_resolution.h>
#include "../render/optix/optix_denoise_plan.h"

#include <algorithm>
#include <cstdint>

namespace oka::editor_denoiser
{

struct Mode
{
    const char* label = "Off";
    bool denoise = false; ///< render/pt/denoise
    bool upscale = false; ///< render/pt/enableUpscale
};

/// Everything the panel needs in order to draw one backend's denoiser controls.
struct Ui
{
    /// Combo label -- the product name, so the panel says which denoiser this is.
    const char* title = "Denoiser";
    const Mode* modes = nullptr;
    int modeCount = 0;
    /// Whether the render scale is the user's to pick.
    bool freeRenderScale = false;
    /// The fraction a fixed-ratio backend renders at when a mode asks to
    /// upscale. Read only when `freeRenderScale` is false.
    float fixedRenderScale = 1.0f;
    bool temporalToggle = false;
    /// Metal's path-traced playback blur. It is a guide-production choice inside
    /// the wavefront integrator (MetalFrameUniforms.mm) and has no OptiX reader.
    bool playbackMotionBlurToggle = false;
    /// The denoiser consumes this launch's color rather than the accumulated PT
    /// mean. Accumulation then controls only when new denoiser frames stop.
    bool perFrameDenoiseInput = false;
    /// What `Render::denoiserFallbackActive()` means on this backend.
    const char* fallbackMessage = "Denoiser unavailable; showing the raw image";
    /// Shown under the mode combo when the mode is not Off. Says the thing about
    /// this denoiser a user cannot infer from the control itself.
    const char* modeHint = nullptr;
};

/// MetalFX: two scalers, either ratio, and the render scale decides whether the
/// scaler does any scaling at all. Kept exactly as the panel had it, since this
/// is the backend the naming was written for.
inline constexpr Mode kMetalFxModes[] = {
    { "Off", false, false },
    { "Spatial upscale", false, true },
    { "Temporal denoise", true, true },
};

/// OptiX: upscaling implies denoising, because there is no path through the
/// plan that scales without also running the network.
inline constexpr Mode kOptixModes[] = {
    { "Off", false, false },
    { "Denoise", true, false },
    { "Denoise + 2x upscale", true, true },
};

inline Ui uiFor(Render::DenoiserKind kind)
{
    Ui ui;
    switch (kind)
    {
    case Render::DenoiserKind::eMetalFx:
        ui.title = "MetalFX";
        ui.modes = kMetalFxModes;
        ui.modeCount = static_cast<int>(std::size(kMetalFxModes));
        ui.freeRenderScale = true;
        ui.playbackMotionBlurToggle = true;
        ui.perFrameDenoiseInput = true;
        // MetalFX refuses a temporal scaler below its supported ratio and the
        // renderer drops to the spatial one rather than showing nothing.
        ui.fallbackMessage = "Temporal denoiser ratio unsupported; using spatial upscale";
        break;
    case Render::DenoiserKind::eOptixAi:
        ui.title = "OptiX denoiser";
        ui.modes = kOptixModes;
        ui.modeCount = static_cast<int>(std::size(kOptixModes));
        ui.fixedRenderScale = 0.5f;
        ui.temporalToggle = true;
        // A denoiser that failed to configure leaves the path-traced image in
        // place -- point-sampled up to size when the plan was an upscaling one.
        ui.fallbackMessage = "Denoiser could not run; showing the path-traced image";
        ui.modeHint = "Upscaling is 2x only, and needs even output dimensions";
        break;
    case Render::DenoiserKind::eNone:
        break;
    }
    return ui;
}

inline bool hasDenoiser(const Ui& ui)
{
    return ui.modeCount > 0;
}

/// Clamped rather than asserted: a mode index outliving the list it indexed is
/// exactly how the old combo came to show one thing while the renderer ran
/// another.
inline Mode modeAt(const Ui& ui, int index)
{
    if (ui.modeCount <= 0)
    {
        return Mode{};
    }
    return ui.modes[std::clamp(index, 0, ui.modeCount - 1)];
}

inline bool usesPerFrameDenoiseInput(const Ui& ui, int modeIndex)
{
    return ui.perFrameDenoiseInput && modeAt(ui, modeIndex).denoise;
}

inline int modeIndexFromSettings(const Ui& ui, bool denoise, bool upscale)
{
    for (int i = 0; i < ui.modeCount; ++i)
    {
        if (ui.modes[i].denoise == denoise && ui.modes[i].upscale == upscale)
        {
            return i;
        }
    }
    for (int i = 0; i < ui.modeCount; ++i)
    {
        if (ui.modes[i].denoise >= denoise && ui.modes[i].upscale >= upscale)
        {
            return i;
        }
    }
    return 0;
}

/// What to write to `render/pt/enableUpscale` for this mode.
inline bool shouldUpscale(const Ui& ui, int modeIndex, float requestedScale)
{
    const Mode mode = modeAt(ui, modeIndex);
    if (!mode.upscale)
    {
        return false;
    }
    // At 1:1 a free-scale backend has nothing to upscale, and asking it to
    // anyway is how "Spatial upscale" came to mean "off" without saying so.
    return ui.freeRenderScale ? requestedScale < 1.0f : true;
}

inline bool settingsMatchMode(const Ui& ui, int modeIndex, bool denoise, bool upscale, float requestedScale)
{
    return modeAt(ui, modeIndex).denoise == denoise && shouldUpscale(ui, modeIndex, requestedScale) == upscale;
}

/// The scale the path tracer will actually run at, as a fraction of the output.
inline float appliedScale(const Ui& ui, int modeIndex, float requestedScale)
{
    if (!shouldUpscale(ui, modeIndex, requestedScale))
    {
        return 1.0f;
    }
    return ui.freeRenderScale ? requestedScale : ui.fixedRenderScale;
}

struct Resolution
{
    uint32_t pathTraceWidth = 1;
    uint32_t pathTraceHeight = 1;
    uint32_t outputWidth = 1;
    uint32_t outputHeight = 1;
    bool upscaling = false;
};

inline Resolution resolution(const Ui& ui, int modeIndex, float requestedScale, uint32_t width, uint32_t height)
{
    Resolution out;
    out.outputWidth = std::max(1u, width);
    out.outputHeight = std::max(1u, height);
    out.pathTraceWidth = out.outputWidth;
    out.pathTraceHeight = out.outputHeight;

    const bool upscale = shouldUpscale(ui, modeIndex, requestedScale);
    if (ui.freeRenderScale)
    {
        const render_resolution::Resolution resolved =
            render_resolution::resolve(out.outputWidth, out.outputHeight, upscale, requestedScale);
        out.pathTraceWidth = resolved.pathTraceWidth;
        out.pathTraceHeight = resolved.pathTraceHeight;
        out.upscaling = resolved.upscaling;
        return out;
    }

    const Mode mode = modeAt(ui, modeIndex);
    // The debug view is passed as None: it changes which guides the plan asks
    // for, never the resolution it launches at.
    const DenoisePlan plan = denoisePlan(mode.denoise, upscale, 0u, 0u, out.outputWidth, out.outputHeight);
    out.pathTraceWidth = plan.renderWidth;
    out.pathTraceHeight = plan.renderHeight;
    out.upscaling = plan.upscale;
    return out;
}

} // namespace oka::editor_denoiser

